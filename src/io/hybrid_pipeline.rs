//! Ping-pong prefetch driven by a real [`GgufLayerMap`].

use std::time::{Duration, Instant};

use super::error::Result;
use super::gguf_map::GgufLayerMap;
use super::nvme::AsyncNvmeReader;
use super::pipeline::{dummy_compute, StepStats};
use super::prefetch::PrefetchBufferManager;

/// Run the double-buffer pipeline over GGUF layer DMA plans.
///
/// `on_layer(layer_index, dma_bytes)` runs while the next layer is in flight.
pub fn run_gguf_prefetch_pipeline<F>(
    map: &GgufLayerMap,
    buffers: &mut PrefetchBufferManager,
    reader: &mut AsyncNvmeReader,
    compute_delay: Duration,
    mut on_layer: Option<F>,
) -> Result<Vec<StepStats>>
where
    F: FnMut(usize, &[u8]) -> Result<u64>,
{
    let layers = &map.layers;
    if layers.is_empty() {
        return Ok(Vec::new());
    }

    {
        let layer = &layers[0];
        let buf = buffers.get_mut(0)?;
        reader.submit_read(buf, 0, layer.read_offset, layer.read_len)?;
        let _ = reader.wait_completion()?;
    }

    let mut stats = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let compute_slot = i % 2;
        let prefetch_slot = 1 - compute_slot;

        let submit_us = if let Some(next) = layers.get(i + 1) {
            let t0 = Instant::now();
            let buf = buffers.get_mut(prefetch_slot)?;
            reader.submit_read(buf, prefetch_slot, next.read_offset, next.read_len)?;
            t0.elapsed().as_micros() as u64
        } else {
            0
        };

        let t_compute = Instant::now();
        let checksum = {
            let buf = buffers.get(compute_slot)?;
            let dma = buf.as_slice();
            if let Some(ref mut cb) = on_layer {
                cb(layer.index, dma)?
            } else {
                dummy_compute(dma, compute_delay)
            }
        };
        let compute_us = t_compute.elapsed().as_micros() as u64;

        let wait_us = if reader.has_in_flight() {
            let t0 = Instant::now();
            let (slot, _) = reader.wait_completion()?;
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
