//! Asynchronous NVMe weight reader via Linux `io_uring` + `O_DIRECT`.
//!
//! Why `O_DIRECT`:
//! bypasses the page cache so weight pages are DMA'd straight into our
//! mlock'd prefetch arenas. Under a fixed RAM budget this avoids
//! double-buffering in the kernel page cache (file cache + our arena).
//!
//! Why `io_uring`:
//! submission / completion separation lets the CPU compute on Buffer A while
//! the NVMe controller fills Buffer B — true compute/I/O overlap without a
//! blocking `read(2)` on the hot path.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use io_uring::{opcode, types, IoUring};

use super::error::{IoError, Result};
use super::prefetch::{align_up, is_aligned, PrefetchBuffer, DIRECT_ALIGN};

/// Default SQ/CQ depth — one in-flight layer read is enough for double buffering;
/// keep a small ring so setup stays cheap.
const RING_ENTRIES: u32 = 8;

/// Tracks a single outstanding direct read into a [`PrefetchBuffer`].
struct InFlight {
    user_data: u64,
    /// Expected transfer length (already aligned up for O_DIRECT).
    expected: usize,
    /// Buffer slot that will receive the data (0 or 1).
    slot: usize,
}

/// `io_uring`-backed reader for LLM weight blobs opened with `O_DIRECT`.
pub struct AsyncNvmeReader {
    /// Kept alive for the lifetime of the reader (FD registered implicitly via raw fd).
    _file: File,
    fd: types::Fd,
    ring: IoUring,
    in_flight: Option<InFlight>,
    /// Monotonic user_data counter for correlating CQEs.
    next_user_data: u64,
}

impl AsyncNvmeReader {
    /// Open `path` with `O_DIRECT` and create a private `io_uring` instance.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            // O_DIRECT: DMA into user buffers; skips kernel page cache.
            // Requires aligned offset, length, and buffer address (see DIRECT_ALIGN).
            .custom_flags(libc::O_DIRECT)
            .open(path_ref)
            .map_err(|e| IoError::Open(path_ref.display().to_string(), e))?;

        let fd = types::Fd(file.as_raw_fd());
        let ring = IoUring::new(RING_ENTRIES).map_err(IoError::UringSetup)?;

        Ok(Self {
            _file: file,
            fd,
            ring,
            in_flight: None,
            next_user_data: 1,
        })
    }

    /// Submit an asynchronous read of `len` bytes at file `offset` into `buf`.
    ///
    /// Length and offset are rounded up / validated for `O_DIRECT`. The actual
    /// DMA length may be slightly larger than `len` (zero-padded tail inside the
    /// buffer); [`wait_completion`] reports the logical `len` as `valid_len`.
    ///
    /// Only one read may be in flight at a time (sufficient for ping-pong).
    pub fn submit_read(
        &mut self,
        buf: &mut PrefetchBuffer,
        slot: usize,
        offset: u64,
        len: usize,
    ) -> Result<()> {
        if self.in_flight.is_some() {
            return Err(IoError::Busy(
                self.in_flight.as_ref().map(|f| f.user_data).unwrap_or(0),
            ));
        }
        if len > buf.capacity() {
            return Err(IoError::BufferTooSmall(len, buf.capacity()));
        }
        if offset % DIRECT_ALIGN as u64 != 0 {
            return Err(IoError::Alignment(offset as usize, DIRECT_ALIGN));
        }

        let transfer = align_up(len, DIRECT_ALIGN);
        if transfer > buf.capacity() {
            return Err(IoError::BufferTooSmall(transfer, buf.capacity()));
        }

        let ptr = buf.as_mut_ptr();
        if !is_aligned(ptr, DIRECT_ALIGN) {
            return Err(IoError::Alignment(ptr as usize, DIRECT_ALIGN));
        }

        let user_data = self.next_user_data;
        self.next_user_data = self.next_user_data.wrapping_add(1);

        // opcode::Read — fixed-buffer style single-shot read into our arena.
        let sqe = opcode::Read::new(self.fd, ptr, transfer as u32)
            .offset(offset)
            .build()
            .user_data(user_data);

        // SAFETY: `ptr` points into `buf`'s live mmap for `transfer` bytes.
        // The caller must keep `buf` (and this reader) alive until
        // `wait_completion` returns. We enforce single-flight to make that
        // contract easy to uphold in the ping-pong loop.
        unsafe {
            self.ring.submission().push(&sqe).map_err(|_entry| {
                IoError::UringSubmit(std::io::Error::other(
                    "io_uring submission queue full",
                ))
            })?;
        }

        // Kick the NVMe submission without waiting — overlap with CPU work.
        self.ring.submit().map_err(IoError::UringSubmit)?;

        self.in_flight = Some(InFlight {
            user_data,
            expected: transfer,
            slot,
        });

        // Logical payload length (may be < transfer due to align_up).
        buf.set_valid_len(len);
        Ok(())
    }

    /// Block until the outstanding read completes; returns `(slot, logical_len)`.
    pub fn wait_completion(&mut self) -> Result<(usize, usize)> {
        let flight = self.in_flight.take().ok_or(IoError::NoInFlight)?;

        // Wait for at least one CQE.
        self.ring
            .submit_and_wait(1)
            .map_err(IoError::UringSubmit)?;

        let cqe = self
            .ring
            .completion()
            .next()
            .ok_or(IoError::MissingCompletion)?;

        let res = cqe.result();
        let ud = cqe.user_data();
        // Consume CQE so the CQ slot is reused.
        // (Drop of cqe is enough with io-uring 0.7 owned Cqe.)

        if ud != flight.user_data {
            return Err(IoError::Io(std::io::Error::other(format!(
                "CQE user_data mismatch: got {ud}, expected {}",
                flight.user_data
            ))));
        }

        if res < 0 {
            return Err(IoError::ReadFailed(
                res,
                std::io::Error::from_raw_os_error(-res),
            ));
        }

        let got = res as usize;
        if got < flight.expected {
            // O_DIRECT short reads can happen at EOF; treat as error for weight files.
            return Err(IoError::ShortRead {
                expected: flight.expected,
                got,
            });
        }

        // valid_len was set to logical `len` in submit_read; return slot + that hint
        // via expected's logical portion — we stored transfer in expected.
        // Re-derive: caller already set valid_len on the buffer; return slot and
        // transfer for diagnostics.
        Ok((flight.slot, got))
    }

    /// Returns `true` if a read is outstanding.
    #[inline]
    pub fn has_in_flight(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Slot index of the in-flight read, if any.
    #[inline]
    #[must_use]
    #[allow(dead_code)] // public introspection API
    pub fn in_flight_slot(&self) -> Option<usize> {
        self.in_flight.as_ref().map(|f| f.slot)
    }
}
