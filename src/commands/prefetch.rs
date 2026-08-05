//! `lpc-llm prefetch` — pack layers into engine cache and benchmark io_uring ping-pong.

use std::time::Duration;

use console::style;
use dialoguer::Confirm;

use crate::catalog;
use crate::error::{AppError, Result};
use crate::io::{
    ensure_packed, run_gguf_prefetch_pipeline, AsyncNvmeReader, GgufLayerMap,
    PrefetchBufferManager,
};
use crate::pull;
use crate::store::LocalStore;

pub fn run(name: &str, auto_pull: bool) -> Result<()> {
    let store = LocalStore::open()?;
    let entry = catalog::find(name).ok_or_else(|| AppError::UnknownModel(name.into()))?;

    let installed = match store.resolve(&entry)? {
        Some(m) => m,
        None => {
            if !auto_pull {
                let ok = Confirm::new()
                    .with_prompt(format!(
                        "`{name}` is not installed. Pull it now ({})?",
                        entry.approx_size
                    ))
                    .default(true)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                if !ok {
                    return Err(AppError::NotInstalled(name.into()));
                }
            }
            pull::pull_model(&store, &entry)?
        }
    };

    eprintln!(
        "{} mapping GGUF layers in {}",
        style("·").cyan(),
        installed.model_path.display()
    );
    let mut map =
        GgufLayerMap::open(&installed.model_path).map_err(|e| AppError::msg(e.to_string()))?;

    let sparse_before = map.layers.iter().filter(|l| l.sparse).count();
    let dense_before = map.layers.iter().filter(|l| !l.sparse).count();
    println!(
        "raw GGUF: path={} arch={} blocks={} layers={} dense={} sparse={} \
         tensor_data_offset=0x{:x} max_layer={} KiB",
        map.path.display(),
        map.architecture,
        map.block_count,
        map.layers.len(),
        dense_before,
        sparse_before,
        map.tensor_data_offset,
        map.max_layer_bytes / 1024
    );

    let pack_cache = store.pack_cache_dir(&entry.name);
    let packed = ensure_packed(&installed.model_path, &map, &pack_cache)
        .map_err(|e| AppError::msg(e.to_string()))?;
    map.layers = packed.layers.clone();
    map.max_layer_bytes = packed.max_layer_bytes;

    let slot = packed.recommended_slot_bytes();
    println!(
        "pack: {}  layers={}  dense={}  max_layer={} KiB  slot={} MiB",
        packed.pack_path.display(),
        map.layers.len(),
        map.layers.iter().filter(|l| !l.sparse).count(),
        map.max_layer_bytes / 1024,
        slot / (1024 * 1024)
    );
    println!(
        "hot tensors: {}",
        map.hot
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(l0) = map.layers.first() {
        println!(
            "layer0: pack_offset=0x{:x} len={} KiB tensors={}",
            l0.read_offset,
            l0.read_len / 1024,
            l0.tensors.len()
        );
    }

    let mut buffers =
        PrefetchBufferManager::new(slot).map_err(|e| AppError::msg(e.to_string()))?;
    let mut reader =
        AsyncNvmeReader::open(&packed.pack_path).map_err(|e| AppError::msg(e.to_string()))?;

    eprintln!(
        "{} running io_uring ping-pong over packed {} layers …",
        style("↓").cyan(),
        map.layers.len()
    );

    let stats = run_gguf_prefetch_pipeline(
        &map,
        &mut buffers,
        &mut reader,
        Duration::from_millis(0),
        None::<fn(usize, &[u8]) -> crate::io::Result<u64>>,
    )
    .map_err(|e| AppError::msg(e.to_string()))?;

    println!();
    println!(
        "{:<6} {:>5} {:>12} {:>14} {:>12}",
        "layer", "slot", "compute_us", "wait_pref_us", "submit_us"
    );
    println!("{}", "-".repeat(56));
    let mut wait_sum = 0u64;
    let mut compute_sum = 0u64;
    for s in &stats {
        wait_sum += s.wait_prefetch_us;
        compute_sum += s.compute_us;
        println!(
            "{:<6} {:>5} {:>12} {:>14} {:>12}",
            s.layer, s.compute_slot, s.compute_us, s.wait_prefetch_us, s.submit_next_us
        );
    }
    println!();
    println!(
        "total compute_us={}  total wait_pref_us={}  (low wait ⇒ good overlap)",
        compute_sum, wait_sum
    );
    Ok(())
}
