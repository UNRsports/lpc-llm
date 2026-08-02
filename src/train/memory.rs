//! Memory-aware training knobs (`--ram-mib`, activation budget).

use super::tiny::TinyConfig;
use crate::error::{AppError, Result};

/// Estimate peak training RAM (params + AdamW moments + activations), MiB.
pub fn estimate_train_mib(cfg: &TinyConfig, max_seq: usize, grad_checkpoint: bool) -> f64 {
    let params = cfg.param_count() as f64;
    // params f32 + AdamW m/v ≈ 3× param bytes
    let optim_bytes = params * 4.0 * 3.0;
    // Activations: rough B=1, layers × seq × emb (×2 without checkpoint)
    let act_factor = if grad_checkpoint { 2.0 } else { 4.0 };
    let act_bytes = act_factor * cfg.n_layers as f64 * max_seq as f64 * cfg.n_embd as f64 * 4.0;
    // Attention scores [H,T,T]
    let attn_bytes = cfg.n_heads as f64 * max_seq as f64 * max_seq as f64 * 4.0;
    (optim_bytes + act_bytes + attn_bytes) / (1024.0 * 1024.0)
}

/// Clamp `max_seq` so the estimate fits in `ram_mib` (minimum 8).
pub fn clamp_max_seq(
    cfg: &TinyConfig,
    requested: usize,
    ram_mib: usize,
    grad_checkpoint: bool,
) -> Result<usize> {
    if ram_mib == 0 {
        return Err(AppError::msg("--ram-mib must be > 0"));
    }
    let mut seq = requested.max(2);
    while seq > 8 {
        let need = estimate_train_mib(cfg, seq, grad_checkpoint);
        if need <= ram_mib as f64 {
            break;
        }
        seq = (seq * 3 / 4).max(8);
    }
    let need = estimate_train_mib(cfg, seq, grad_checkpoint);
    if need > ram_mib as f64 * 1.25 {
        return Err(AppError::msg(format!(
            "training needs ~{need:.0} MiB but --ram-mib={ram_mib}; \
             shrink --n-embd/--layers/--max-seq or raise --ram-mib"
        )));
    }
    Ok(seq)
}

pub fn log_memory_plan(cfg: &TinyConfig, max_seq: usize, ram_mib: usize, grad_checkpoint: bool) {
    let need = estimate_train_mib(cfg, max_seq, grad_checkpoint);
    eprintln!(
        "  memory plan: ~{need:.1} MiB est / {ram_mib} MiB budget  params≈{}  grad_checkpoint={grad_checkpoint}  max_seq={max_seq}",
        cfg.param_count()
    );
}
