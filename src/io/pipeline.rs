//! Ping-pong (double-buffer) compute / I/O pipeline.
//!
//! ```text
//!   time ──────────────────────────────────────────────────────────►
//!
//!   CPU:   [ compute L0 on A ][ compute L1 on B ][ compute L2 on A ] …
//!   NVMe:  [ prefetch L1→B  ][ prefetch L2→A  ][ prefetch L3→B  ] …
//! ```
//!
//! While the CPU consumes weights in the *compute* slot, `io_uring` fills the
//! *prefetch* slot with the next layer. Slots swap each iteration.

use std::time::{Duration, Instant};

use super::error::Result;
use super::nvme::AsyncNvmeReader;
use super::prefetch::{align_up, PrefetchBufferManager, DIRECT_ALIGN};

/// Description of one contiguous weight region (e.g. a transformer layer).
#[derive(Debug, Clone, Copy)]
pub struct LayerExtent {
    pub index: usize,
    /// File offset — must be a multiple of [`DIRECT_ALIGN`].
    pub offset: u64,
    /// Logical payload size (internally aligned up for O_DIRECT).
    pub len: usize,
}

/// Timing breakdown for one pipeline step (useful for verifying I/O/compute overlap).
#[derive(Debug, Clone, Default)]
pub struct StepStats {
    pub layer: usize,
    pub compute_slot: usize,
    pub compute_us: u64,
    pub wait_prefetch_us: u64,
    pub submit_next_us: u64,
    /// Checksum over the compute buffer (proves data was really loaded).
    pub checksum: u64,
}

/// Dummy “matmul” stand-in: burn `compute_delay` while touching the weight bytes
/// so the compiler cannot DCE the slice, and produce a cheap checksum.
pub fn dummy_compute(weights: &[u8], compute_delay: Duration) -> u64 {
    let start = Instant::now();
    let mut acc = 0u64;
    // Strided touch — enough to exercise the RAM path without dominating I/O.
    for (i, chunk) in weights.chunks(4096).enumerate() {
        if let Some(&b) = chunk.first() {
            acc = acc.wrapping_add((b as u64).wrapping_mul(i as u64 + 1));
        }
    }
    let elapsed = start.elapsed();
    if elapsed < compute_delay {
        std::thread::sleep(compute_delay - elapsed);
    }
    acc
}

/// Drive the double-buffer pipeline over `layers`.
///
/// Protocol:
/// 1. Prefetch layer 0 into slot 0; wait.
/// 2. For each layer `i`:
///    - Submit prefetch of layer `i+1` into the other slot (if any).
///    - Compute on slot `i % 2` (overlaps with the in-flight NVMe read).
///    - Wait for that prefetch to complete before the next iteration.
pub fn run_pipeline(
    buffers: &mut PrefetchBufferManager,
    reader: &mut AsyncNvmeReader,
    layers: &[LayerExtent],
    compute_delay: Duration,
) -> Result<Vec<StepStats>> {
    if layers.is_empty() {
        return Ok(Vec::new());
    }

    // --- Bootstrap: load layer 0 into slot 0 ---
    {
        let layer = &layers[0];
        let buf = buffers.get_mut(0)?;
        reader.submit_read(buf, 0, layer.offset, layer.len)?;
        let (_slot, _n) = reader.wait_completion()?;
    }

    let mut stats = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let compute_slot = i % 2;
        let prefetch_slot = 1 - compute_slot;

        // Submit next-layer prefetch *before* compute so NVMe and CPU overlap.
        let submit_us = if let Some(next) = layers.get(i + 1) {
            let t0 = Instant::now();
            let buf = buffers.get_mut(prefetch_slot)?;
            reader.submit_read(buf, prefetch_slot, next.offset, next.len)?;
            t0.elapsed().as_micros() as u64
        } else {
            0
        };

        // CPU work on the slot that already holds layer i.
        let t_compute = Instant::now();
        let checksum = {
            let buf = buffers.get(compute_slot)?;
            debug_assert_eq!(buf.valid_len(), layer.len);
            dummy_compute(buf.as_slice(), compute_delay)
        };
        let compute_us = t_compute.elapsed().as_micros() as u64;

        // Synchronize the outstanding prefetch before the next iteration
        // reuses / swaps roles.
        let wait_us = if reader.has_in_flight() {
            let t0 = Instant::now();
            let (slot, _n) = reader.wait_completion()?;
            debug_assert_eq!(slot, prefetch_slot);
            t0.elapsed().as_micros() as u64
        } else {
            0
        };

        stats.push(StepStats {
            layer: layer.index,
            compute_slot,
            compute_us,
            wait_prefetch_us: wait_us,
            submit_next_us: submit_us,
            checksum,
        });
    }

    Ok(stats)
}

/// Build evenly spaced layer extents for a synthetic weight file of `file_size`
/// bytes split into `num_layers` chunks (each chunk aligned for O_DIRECT).
pub fn synthesize_layers(file_size: u64, num_layers: usize) -> Vec<LayerExtent> {
    assert!(num_layers > 0);
    if file_size == 0 {
        return Vec::new();
    }

    let raw = (file_size / num_layers as u64).max(DIRECT_ALIGN as u64) as usize;
    let chunk = align_up(raw, DIRECT_ALIGN);

    (0..num_layers)
        .filter_map(|i| {
            let offset = (i as u64).checked_mul(chunk as u64)?;
            if offset >= file_size {
                return None;
            }
            let remaining = (file_size - offset) as usize;
            let len = align_up(remaining.min(chunk), DIRECT_ALIGN);
            // Do not read past EOF; shrink to an aligned size that still fits.
            let len = if offset + len as u64 > file_size {
                let fit = remaining - (remaining % DIRECT_ALIGN);
                if fit == 0 {
                    return None;
                }
                fit
            } else {
                len
            };
            Some(LayerExtent {
                index: i,
                offset,
                len,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_aligned_layers() {
        let layers = synthesize_layers(16 * DIRECT_ALIGN as u64, 4);
        assert_eq!(layers.len(), 4);
        for l in &layers {
            assert_eq!(l.offset % DIRECT_ALIGN as u64, 0);
            assert_eq!(l.len % DIRECT_ALIGN, 0);
            assert!(l.len > 0);
        }
    }
}
