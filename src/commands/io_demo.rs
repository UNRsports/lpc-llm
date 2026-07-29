//! Existing NVMe double-buffer prefetch demo, exposed as `lpc-llm io`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;

use crate::error::{AppError, Result};
use crate::io::prefetch::DIRECT_ALIGN;
use crate::io::{
    align_up, run_pipeline, synthesize_layers, AsyncNvmeReader, PrefetchBufferManager,
};

#[derive(Debug, Args)]
pub struct IoArgs {
    /// Weight blob path
    #[arg(long, default_value = "demo_weights.bin")]
    pub weights: PathBuf,

    /// Create a synthetic weight file if missing
    #[arg(long)]
    pub create_demo: bool,

    /// Per-buffer size in MiB (use 2048 for 2 GiB × 2)
    #[arg(long, default_value_t = 2)]
    pub slot_mib: usize,

    /// Number of synthetic layers
    #[arg(long, default_value_t = 8)]
    pub layers: usize,

    /// Dummy compute duration per layer in ms
    #[arg(long, default_value_t = 5)]
    pub compute_ms: u64,

    /// Skip mlock (demo only)
    #[arg(long)]
    pub no_mlock: bool,
}

fn create_demo_weights(path: &Path, layers: usize, layer_bytes: usize) -> std::io::Result<()> {
    let layer_bytes = align_up(layer_bytes, DIRECT_ALIGN);
    let total = layer_bytes * layers;
    eprintln!(
        "creating demo weights at {} ({} layers × {} bytes = {} MiB)",
        path.display(),
        layers,
        layer_bytes,
        total / (1024 * 1024)
    );

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let mut chunk = vec![0u8; layer_bytes.min(1024 * 1024)];
    for layer in 0..layers {
        let pattern = (0xA0u8).wrapping_add(layer as u8);
        let mut remaining = layer_bytes;
        while remaining > 0 {
            let n = remaining.min(chunk.len());
            chunk[..n].fill(pattern);
            if remaining == layer_bytes && n >= 8 {
                chunk[..8].copy_from_slice(&(layer as u64).to_le_bytes());
            }
            file.write_all(&chunk[..n])?;
            remaining -= n;
        }
    }
    file.sync_all()?;
    Ok(())
}

pub fn run(args: IoArgs) -> Result<()> {
    if args.slot_mib == 0 {
        return Err(AppError::msg("--slot-mib must be ≥ 1"));
    }
    if args.layers == 0 {
        return Err(AppError::msg("--layers must be ≥ 1"));
    }

    let slot_bytes = args.slot_mib * 1024 * 1024;
    if slot_bytes % DIRECT_ALIGN != 0 {
        return Err(AppError::msg(format!(
            "slot size {slot_bytes} is not {DIRECT_ALIGN}-byte aligned"
        )));
    }

    let layer_bytes = slot_bytes;
    if args.create_demo || !args.weights.exists() {
        if !args.weights.exists() && !args.create_demo {
            eprintln!(
                "note: {} missing — creating demo file (pass --create-demo to silence)",
                args.weights.display()
            );
        }
        create_demo_weights(&args.weights, args.layers, layer_bytes)?;
    }

    let file_len = File::open(&args.weights)?.metadata()?.len();
    eprintln!(
        "weights: {} ({} bytes)",
        args.weights.display(),
        file_len
    );

    eprintln!(
        "allocating double buffer: 2 × {} MiB = {} MiB pinned arena",
        args.slot_mib,
        args.slot_mib * 2
    );

    let mut buffers = if args.no_mlock {
        PrefetchBufferManager::new_unlocked(slot_bytes).map_err(|e| AppError::msg(e.to_string()))?
    } else {
        match PrefetchBufferManager::new(slot_bytes) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("warning: {e}");
                eprintln!(
                    "falling back to unlocked buffers — raise `ulimit -l` for production \
                     (or pass --no-mlock explicitly)"
                );
                PrefetchBufferManager::new_unlocked(slot_bytes)
                    .map_err(|e| AppError::msg(e.to_string()))?
            }
        }
    };

    eprintln!(
        "arena ready: {} bytes total, align={}, mlock={}",
        buffers.total_pinned_bytes(),
        DIRECT_ALIGN,
        buffers.both_locked()
    );

    let mut reader =
        AsyncNvmeReader::open(&args.weights).map_err(|e| AppError::msg(e.to_string()))?;
    let layers = synthesize_layers(file_len.min((layer_bytes * args.layers) as u64), args.layers);
    if layers.is_empty() {
        return Err(AppError::msg("no layers to process"));
    }
    let layers: Vec<_> = layers
        .into_iter()
        .map(|mut l| {
            l.len = l.len.min(slot_bytes);
            l
        })
        .collect();

    eprintln!(
        "pipeline: {} layers, compute≈{} ms/layer (I/O overlaps with compute)",
        layers.len(),
        args.compute_ms
    );

    let t0 = std::time::Instant::now();
    let stats = run_pipeline(
        &mut buffers,
        &mut reader,
        &layers,
        Duration::from_millis(args.compute_ms),
    )
    .map_err(|e| AppError::msg(e.to_string()))?;
    let wall = t0.elapsed();

    println!();
    println!(
        "{:<6} {:>5} {:>12} {:>14} {:>12} {:>18}",
        "layer", "slot", "compute_us", "wait_pref_us", "submit_us", "checksum"
    );
    println!("{}", "-".repeat(72));
    for s in &stats {
        println!(
            "{:<6} {:>5} {:>12} {:>14} {:>12} {:>18}",
            s.layer,
            s.compute_slot,
            s.compute_us,
            s.wait_prefetch_us,
            s.submit_next_us,
            s.checksum
        );
    }
    println!();
    println!(
        "wall: {:.3}s | layers: {} | note: low wait_pref_us while compute_us≈{}000 means good overlap",
        wall.as_secs_f64(),
        stats.len(),
        args.compute_ms
    );

    Ok(())
}
