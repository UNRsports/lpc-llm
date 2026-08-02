//! From-scratch tiny Transformer training loop.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW};
use console::style;

use super::checkpoint::save_checkpoint;
use super::data::{load_training_texts, tokenize_chunks};
use super::gguf_export::{export_gguf, register_gguf_model};
use super::memory::{clamp_max_seq, log_memory_plan};
use super::tiny::{TinyConfig, TinyModel};
use super::tokenizer_tiny::build_char_tokenizer;
use crate::error::{AppError, Result};
use crate::store::LocalStore;

#[derive(Debug, Clone)]
pub struct ScratchConfig {
    pub name: String,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub grad_checkpoint: bool,
    pub n_embd: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_ff: usize,
    pub seed: u64,
    /// When true, export GGUF and register under `name` for `lpc-llm run`.
    pub register: bool,
}

impl Default for ScratchConfig {
    fn default() -> Self {
        Self {
            name: "tiny:demo".into(),
            steps: 64,
            lr: 3e-3,
            max_seq: 64,
            ram_mib: 1024,
            grad_checkpoint: true,
            n_embd: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            n_ff: 512,
            seed: 42,
            register: true,
        }
    }
}

/// Train a tiny model from scratch; write checkpoint (+ optional GGUF register).
pub fn train_scratch(
    store: &LocalStore,
    from: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    cfg: ScratchConfig,
) -> Result<PathBuf> {
    let _ = cfg.seed;
    let texts = load_training_texts(from)?;
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let tok_path = out_dir.join(super::checkpoint::TOKENIZER_FILE);
    let (tokenizer, vocab_size) = build_char_tokenizer(&texts, &tok_path)?;

    let mut tiny_cfg = TinyConfig {
        vocab_size,
        n_embd: cfg.n_embd,
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        n_ff: cfg.n_ff,
        max_seq: cfg.max_seq,
        tie_embeddings: true,
        ..TinyConfig::default()
    };
    tiny_cfg.validate()?;
    let max_seq = clamp_max_seq(
        &tiny_cfg,
        cfg.max_seq,
        cfg.ram_mib,
        cfg.grad_checkpoint,
    )?;
    tiny_cfg.max_seq = max_seq;

    let chunks = tokenize_chunks(&tokenizer, &texts, max_seq)?;
    let device = Device::Cpu;
    let model = TinyModel::new(tiny_cfg.clone(), &device)?;

    eprintln!(
        "{} Phase 5 scratch train: out={} layers={} embd={} vocab={} steps={}",
        style("▸").cyan(),
        style(&cfg.name).bold(),
        tiny_cfg.n_layers,
        tiny_cfg.n_embd,
        tiny_cfg.vocab_size,
        cfg.steps
    );
    log_memory_plan(&tiny_cfg, max_seq, cfg.ram_mib, cfg.grad_checkpoint);

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(model.all_vars(), params)?;

    let t0 = Instant::now();
    let mut last_loss = 0f32;
    for step in 0..cfg.steps {
        let step_t0 = Instant::now();
        let chunk = &chunks[step % chunks.len()];
        let tokens = Tensor::new(chunk.as_slice(), &device)?;
        let logits = model.forward(&tokens, cfg.grad_checkpoint)?; // [1,T,V]
        let (_b, t_dim, vocab) = logits.dims3()?;
        let t = chunk.len().min(t_dim);
        if t < 2 {
            return Err(AppError::msg("training chunk too short"));
        }
        let logits = logits.narrow(1, 0, t - 1)?.reshape((t - 1, vocab))?;
        let targets = Tensor::new(&chunk[1..t], &device)?;
        let loss_t = loss::cross_entropy(&logits, &targets)?;
        last_loss = loss_t.to_vec0::<f32>()?;
        opt.backward_step(&loss_t)?;

        let done = step + 1;
        if done == 1 || done == cfg.steps || done % 8 == 0 {
            eprintln!(
                "  step {:>4}/{}  loss={:.4}  step={:.2}s  elapsed={:.0}s",
                done,
                cfg.steps,
                last_loss,
                step_t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let ckpt = save_checkpoint(out_dir, &cfg.name, &model, cfg.steps, Some(last_loss), Some(&tok_path))?;
    eprintln!(
        "{} checkpoint → {} (final loss={:.4})",
        style("✓").green(),
        ckpt.display(),
        last_loss
    );

    if cfg.register {
        let gguf = out_dir.join("model.gguf");
        export_gguf(&model, &gguf)?;
        register_gguf_model(store, &cfg.name, &gguf, &tok_path)?;
    }
    Ok(ckpt)
}
