//! Local full fine-tune SFT for tiny checkpoints (Phase 5).
//! Catalog LoRA SFT remains `lpc-llm adapter create` (Phase 4).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW};
use console::style;
use tokenizers::Tokenizer;

use super::checkpoint::{load_checkpoint, save_checkpoint, TOKENIZER_FILE};
use super::data::{load_training_texts, tokenize_chunks};
use super::gguf_export::{export_gguf, register_gguf_model};
use super::memory::{clamp_max_seq, log_memory_plan};
use crate::error::{AppError, Result};
use crate::store::LocalStore;

#[derive(Debug, Clone)]
pub struct SftConfig {
    pub name: String,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub grad_checkpoint: bool,
    pub register: bool,
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            name: "tiny:sft".into(),
            steps: 32,
            lr: 1e-3,
            max_seq: 64,
            ram_mib: 1024,
            grad_checkpoint: true,
            register: true,
        }
    }
}

/// Full-parameter SFT continuing from a tiny checkpoint directory.
pub fn train_sft_full(
    store: &LocalStore,
    ckpt_dir: impl AsRef<Path>,
    from: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    cfg: SftConfig,
) -> Result<PathBuf> {
    let ckpt_dir = ckpt_dir.as_ref();
    let tok_path = ckpt_dir.join(TOKENIZER_FILE);
    if !tok_path.is_file() {
        return Err(AppError::msg(format!(
            "checkpoint missing tokenizer at {}",
            tok_path.display()
        )));
    }
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
        AppError::msg(format!("load tokenizer: {e}"))
    })?;
    let texts = load_training_texts(from)?;
    let device = Device::Cpu;
    let mut model = load_checkpoint(ckpt_dir, &device)?;

    let max_seq = clamp_max_seq(
        &model.cfg,
        cfg.max_seq.min(model.cfg.max_seq),
        cfg.ram_mib,
        cfg.grad_checkpoint,
    )?;
    model.cfg.max_seq = model.cfg.max_seq.max(max_seq);
    let chunks = tokenize_chunks(&tokenizer, &texts, max_seq)?;

    eprintln!(
        "{} Phase 5 full SFT: base={} out={} steps={} chunks={}",
        style("▸").cyan(),
        ckpt_dir.display(),
        style(&cfg.name).bold(),
        cfg.steps,
        chunks.len()
    );
    log_memory_plan(&model.cfg, max_seq, cfg.ram_mib, cfg.grad_checkpoint);

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: 0.01,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(model.all_vars(), params)?;
    let t0 = Instant::now();
    let mut last_loss = 0f32;
    for step in 0..cfg.steps {
        let chunk = &chunks[step % chunks.len()];
        let tokens = Tensor::new(chunk.as_slice(), &device)?;
        let logits = model.forward(&tokens, cfg.grad_checkpoint)?;
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
                "  step {:>4}/{}  loss={:.4}  elapsed={:.0}s",
                done,
                cfg.steps,
                last_loss,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    let out_dir = out_dir.as_ref();
    let path = save_checkpoint(
        out_dir,
        &cfg.name,
        &model,
        cfg.steps,
        Some(last_loss),
        Some(&tok_path),
    )?;
    if cfg.register {
        let gguf = out_dir.join("model.gguf");
        export_gguf(&model, &gguf)?;
        register_gguf_model(store, &cfg.name, &gguf, &out_dir.join(TOKENIZER_FILE))?;
    }
    Ok(path)
}
