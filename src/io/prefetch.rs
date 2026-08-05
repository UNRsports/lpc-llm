//! Double-buffer (ping-pong) prefetch arena backed by anonymous `mmap`.
//!
//! Why this design:
//! - **mmap (anonymous)**: obtains page-aligned virtual memory suitable for `O_DIRECT`
//!   DMA (device/driver typically require 512/4096-byte address & length alignment).
//! - **mlock (optional)**: when `RLIMIT_MEMLOCK` already covers the arenas, pages are
//!   pinned so reclaim cannot swap them mid-inference. Default path never requires
//!   `ulimit` / CAP_IPC_LOCK — unlocked arenas are correct for `O_DIRECT`.

use memmap2::{MmapMut, MmapOptions};

use super::error::{IoError, Result};

/// Soft-raise `RLIMIT_MEMLOCK` to its hard cap when that helps (no user action).
/// Never prints hints about `ulimit` — unlocked arenas are the normal fallback.
pub fn try_raise_memlock_limit() {
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) != 0 {
            return;
        }
        if lim.rlim_cur < lim.rlim_max {
            lim.rlim_cur = lim.rlim_max;
            let _ = libc::setrlimit(libc::RLIMIT_MEMLOCK, &lim);
        }
    }
}

/// Current effective memlock budget in bytes (`None` = unlimited).
fn memlock_budget_bytes() -> Option<u64> {
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) != 0 {
            return Some(0);
        }
        if lim.rlim_cur == libc::RLIM_INFINITY {
            None
        } else {
            Some(lim.rlim_cur as u64)
        }
    }
}

/// True when the process may `mlock` `need_bytes` without raising privileges.
fn memlock_covers(need_bytes: usize) -> bool {
    match memlock_budget_bytes() {
        None => true,
        Some(budget) => budget >= need_bytes as u64,
    }
}

/// Logical block size commonly required by `O_DIRECT` on NVMe (4 KiB pages).
pub const DIRECT_ALIGN: usize = 4096;

/// One half of the double buffer — a contiguous, page-aligned region.
pub struct PrefetchBuffer {
    /// Anonymous mapping; length is always a multiple of [`DIRECT_ALIGN`].
    map: MmapMut,
    /// Bytes of valid weight data currently residing in this buffer.
    valid_len: usize,
    /// Whether [`libc::mlock`] was successfully applied (so Drop can munlock).
    locked: bool,
}

impl PrefetchBuffer {
    fn allocate(capacity: usize, lock: bool) -> Result<Self> {
        if capacity == 0 || capacity % DIRECT_ALIGN != 0 {
            return Err(IoError::Alignment(capacity, DIRECT_ALIGN));
        }

        // mmap returns a page-aligned address (≥ 4 KiB on x86_64), satisfying
        // O_DIRECT buffer-address alignment without posix_memalign.
        let mut map = MmapOptions::new()
            .len(capacity)
            .map_anon()
            .map_err(|e| IoError::Mmap(capacity, e))?;

        let mut locked = false;
        if lock {
            // SAFETY: `map` owns a valid anonymous mapping of `capacity` bytes.
            let rc = unsafe { libc::mlock(map.as_ptr().cast(), map.len()) };
            if rc != 0 {
                return Err(IoError::Mlock(capacity, std::io::Error::last_os_error()));
            }
            locked = true;
        }

        // First-touch: fault pages in now so the critical path avoids major faults
        // even without mlock.
        map.fill(0);

        Ok(Self {
            map,
            valid_len: 0,
            locked,
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn valid_len(&self) -> usize {
        self.valid_len
    }

    #[inline]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    pub(crate) fn set_valid_len(&mut self, len: usize) {
        debug_assert!(len <= self.map.len());
        self.valid_len = len;
    }

    /// Raw mutable pointer for `io_uring` `Read` SQE (must stay alive for the op).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.map.as_mut_ptr()
    }

    /// CPU-side view of the currently loaded weight slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.map[..self.valid_len]
    }

    /// Full backing region (for DMA destination); length == capacity.
    #[inline]
    #[allow(dead_code)] // used by PrefetchBufferManager::clear
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.map[..]
    }
}

impl Drop for PrefetchBuffer {
    fn drop(&mut self) {
        if !self.locked {
            return;
        }
        // SAFETY: region was successfully mlock'd in `allocate`; munlock is the
        // matching cleanup. Ignoring errors on drop (process exit will unlock).
        unsafe {
            libc::munlock(self.map.as_ptr().cast(), self.map.len());
        }
    }
}

/// Manages the two arenas used by the ping-pong prefetch pipeline.
///
/// Layout (example with 2 GiB slots → 4 GiB total pinned):
/// ```text
///   ┌──────────── Buffer A ────────────┐┌──────────── Buffer B ────────────┐
///   │  mmap + mlock (DIRECT_ALIGN)     ││  mmap + mlock (DIRECT_ALIGN)     │
///   └──────────────────────────────────┘└──────────────────────────────────┘
/// ```
pub struct PrefetchBufferManager {
    slots: [PrefetchBuffer; 2],
}

impl PrefetchBufferManager {
    /// Allocate arenas for hybrid DMA. Pins with `mlock` only when the process
    /// already has enough `RLIMIT_MEMLOCK`; otherwise uses unlocked mmap.
    /// No `ulimit` / CAP_IPC_LOCK required for correctness.
    pub fn new(slot_bytes: usize) -> Result<Self> {
        Self::new_auto(slot_bytes)
    }

    /// Opportunistic lock: pin if budget covers 2× slots, else unlocked.
    pub fn new_auto(slot_bytes: usize) -> Result<Self> {
        try_raise_memlock_limit();
        let need = slot_bytes.saturating_mul(2);
        if memlock_covers(need) {
            match Self::with_lock(slot_bytes, true) {
                Ok(v) => return Ok(v),
                Err(_) => { /* fall through to unlocked */ }
            }
        }
        Self::with_lock(slot_bytes, false)
    }

    /// Force unlocked arenas (tests / demos).
    pub fn new_unlocked(slot_bytes: usize) -> Result<Self> {
        Self::with_lock(slot_bytes, false)
    }

    /// Force `mlock` (fails if limit is too small). Prefer [`Self::new`].
    #[allow(dead_code)]
    pub fn new_locked(slot_bytes: usize) -> Result<Self> {
        Self::with_lock(slot_bytes, true)
    }

    fn with_lock(slot_bytes: usize, lock: bool) -> Result<Self> {
        let a = PrefetchBuffer::allocate(slot_bytes, lock)?;
        let b = PrefetchBuffer::allocate(slot_bytes, lock)?;
        Ok(Self { slots: [a, b] })
    }

    #[inline]
    #[allow(dead_code)] // public API
    pub fn slot_capacity(&self) -> usize {
        self.slots[0].capacity()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Result<&PrefetchBuffer> {
        self.slots.get(index).ok_or(IoError::BadSlot(index))
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Result<&mut PrefetchBuffer> {
        self.slots.get_mut(index).ok_or(IoError::BadSlot(index))
    }

    /// Zero a slot (e.g. before submitting a fresh layer read).
    #[allow(dead_code)] // public API
    pub fn clear(&mut self, index: usize) -> Result<()> {
        let buf = self.get_mut(index)?;
        buf.as_mut_bytes().fill(0);
        buf.set_valid_len(0);
        Ok(())
    }

    /// Total arena bytes across both slots.
    pub fn total_pinned_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.capacity()).sum()
    }

    pub fn both_locked(&self) -> bool {
        self.slots.iter().all(|s| s.is_locked())
    }
}

/// N-slot ring for MoE expert DMA (Top-K streaming). Extends the classic 2×
/// ping-pong when more than two experts are active per token.
pub struct PrefetchRing {
    slots: Vec<PrefetchBuffer>,
}

impl PrefetchRing {
    /// Opportunistic lock (same policy as [`PrefetchBufferManager::new`]).
    pub fn new(slot_bytes: usize, n_slots: usize) -> Result<Self> {
        Self::new_auto(slot_bytes, n_slots)
    }

    pub fn new_auto(slot_bytes: usize, n_slots: usize) -> Result<Self> {
        try_raise_memlock_limit();
        let n = n_slots.max(2);
        let need = slot_bytes.saturating_mul(n);
        if memlock_covers(need) {
            match Self::with_lock(slot_bytes, n_slots, true) {
                Ok(v) => return Ok(v),
                Err(_) => {}
            }
        }
        Self::with_lock(slot_bytes, n_slots, false)
    }

    pub fn new_unlocked(slot_bytes: usize, n_slots: usize) -> Result<Self> {
        Self::with_lock(slot_bytes, n_slots, false)
    }

    #[allow(dead_code)]
    pub fn new_locked(slot_bytes: usize, n_slots: usize) -> Result<Self> {
        Self::with_lock(slot_bytes, n_slots, true)
    }

    fn with_lock(slot_bytes: usize, n_slots: usize, lock: bool) -> Result<Self> {
        let n = n_slots.max(2);
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(PrefetchBuffer::allocate(slot_bytes, lock)?);
        }
        Ok(Self { slots })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn slot_capacity(&self) -> usize {
        self.slots.first().map(|s| s.capacity()).unwrap_or(0)
    }

    #[inline]
    pub fn get(&self, index: usize) -> Result<&PrefetchBuffer> {
        self.slots.get(index).ok_or(IoError::BadSlot(index))
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Result<&mut PrefetchBuffer> {
        self.slots.get_mut(index).ok_or(IoError::BadSlot(index))
    }

    #[allow(dead_code)]
    pub fn total_pinned_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.capacity()).sum()
    }

    pub fn all_locked(&self) -> bool {
        !self.slots.is_empty() && self.slots.iter().all(|s| s.is_locked())
    }
}

/// Round `n` up to the next multiple of `align` (align must be power of two).
#[inline]
pub fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

/// True if `ptr` is aligned to `align` bytes.
#[inline]
pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
    (ptr as usize) & (align - 1) == 0
}
