//! Minimal DPO (Direct Preference Optimization) for tiny checkpoints.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::{ops, AdamW, Optimizer, ParamsAdamW};
use console::style;
use tokenizers::Tokenizer;

use super::checkpoint::{load_checkpoint, save_checkpoint, TOKENIZER_FILE};
use super::data::load_preference_pairs;
use super::gguf_export::{export_gguf, register_gguf_model};
use super::memory::{clamp_max_seq, log_memory_plan};
use super::tiny::TinyModel;
use crate::error::{AppError, Result};
use crate::store::LocalStore;

#[derive(Debug, Clone)]
pub struct DpoConfig {
    pub name: String,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub beta: f64,
    pub grad_checkpoint: bool,
    pub register: bool,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            name: "tiny:dpo".into(),
            steps: 32,
            lr: 5e-4,
            max_seq: 64,
            ram_mib: 1024,
            beta: 0.1,
            grad_checkpoint: true,
            register: true,
        }
    }
}

/// DPO on preference JSONL against a frozen reference copy of the base checkpoint.
pub fn train_dpo(
    store: &LocalStore,
    ckpt_dir: impl AsRef<Path>,
    from: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    cfg: DpoConfig,
) -> Result<PathBuf> {
    let ckpt_dir = ckpt_dir.as_ref();
    let tok_path = ckpt_dir.join(TOKENIZER_FILE);
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
        AppError::msg(format!("load tokenizer: {e}"))
    })?;
    let pairs = load_preference_pairs(from)?;
    let device = Device::Cpu;

    let policy = load_checkpoint(ckpt_dir, &device)?;
    let reference = load_checkpoint(ckpt_dir, &device)?; // frozen (never optimized)

    let max_seq = clamp_max_seq(
        &policy.cfg,
        cfg.max_seq.min(policy.cfg.max_seq),
        cfg.ram_mib,
        cfg.grad_checkpoint,
    )?;

    eprintln!(
        "{} Phase 5 DPO: base={} out={} pairs={} steps={} beta={}",
        style("▸").cyan(),
        ckpt_dir.display(),
        style(&cfg.name).bold(),
        pairs.len(),
        cfg.steps,
        cfg.beta
    );
    log_memory_plan(&policy.cfg, max_seq, cfg.ram_mib, cfg.grad_checkpoint);

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: 0.0,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(policy.all_vars(), params)?;
    let t0 = Instant::now();
    let mut last_loss = 0f32;

    for step in 0..cfg.steps {
        let pair = &pairs[step % pairs.len()];
        let chosen = encode_pair(&tokenizer, &pair.prompt, &pair.chosen, max_seq)?;
        let rejected = encode_pair(&tokenizer, &pair.prompt, &pair.rejected, max_seq)?;

        let pi_w = seq_logprob(&policy, &chosen.tokens, chosen.prompt_len, cfg.grad_checkpoint)?;
        let pi_l = seq_logprob(&policy, &rejected.tokens, rejected.prompt_len, cfg.grad_checkpoint)?;
        // Reference is detached: use no backward through it.
        let ref_w = seq_logprob(&reference, &chosen.tokens, chosen.prompt_len, false)?
            .detach();
        let ref_l = seq_logprob(&reference, &rejected.tokens, rejected.prompt_len, false)?
            .detach();

        // L = -log σ(β * ((πw - πref_w) - (πl - πref_l)))
        let delta = ((&pi_w - &ref_w)? - (&pi_l - &ref_l)?)?;
        let beta = Tensor::new(cfg.beta as f32, &device)?;
        let logits = (delta * beta)?;
        // -log σ(x)
        let loss_t = ops::sigmoid(&logits)?.log()?.neg()?;
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
        &policy,
        cfg.steps,
        Some(last_loss),
        Some(&tok_path),
    )?;
    if cfg.register {
        let gguf = out_dir.join("model.gguf");
        export_gguf(&policy, &gguf)?;
        register_gguf_model(store, &cfg.name, &gguf, &out_dir.join(TOKENIZER_FILE))?;
    }
    Ok(path)
}

struct Encoded {
    tokens: Vec<u32>,
    prompt_len: usize,
}

fn encode_pair(
    tokenizer: &Tokenizer,
    prompt: &str,
    completion: &str,
    max_seq: usize,
) -> Result<Encoded> {
    let full = format!("{prompt}{completion}");
    let enc = tokenizer
        .encode(full.as_str(), true)
        .map_err(|e| AppError::msg(format!("tokenize: {e}")))?;
    let prompt_enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| AppError::msg(format!("tokenize prompt: {e}")))?;
    let mut tokens = enc.get_ids().to_vec();
    if tokens.len() > max_seq {
        tokens.truncate(max_seq);
    }
    if tokens.len() < 2 {
        return Err(AppError::msg("preference sample too short after tokenize"));
    }
    let prompt_len = prompt_enc.get_ids().len().min(tokens.len().saturating_sub(1));
    Ok(Encoded { tokens, prompt_len })
}

/// Mean log-prob of completion tokens (positions `prompt_len..`).
fn seq_logprob(
    model: &TinyModel,
    tokens: &[u32],
    prompt_len: usize,
    grad_checkpoint: bool,
) -> Result<Tensor> {
    let device = model.device();
    let input = Tensor::new(tokens, device)?;
    let logits = model.forward(&input, grad_checkpoint)?; // [1,T,V]
    let (_b, t_dim, vocab) = logits.dims3()?;
    let t = tokens.len().min(t_dim);
    if t < 2 {
        return Err(AppError::msg("seq too short for logprob"));
    }
    let start = prompt_len.min(t - 1).max(1);
    // Predict tokens[1..t] from positions 0..t-1; score completion part.
    let logits = logits.narrow(1, 0, t - 1)?.reshape((t - 1, vocab))?;
    let log_probs = ops::log_softmax(&logits, candle_core::D::Minus1)?;
    let targets = Tensor::new(&tokens[1..t], device)?;
    let gathered = log_probs.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let n = t - 1;
    let comp_start = (start - 1).min(n);
    if comp_start >= n {
        return gathered.mean_all().map_err(Into::into);
    }
    gathered
        .narrow(0, comp_start, n - comp_start)?
        .mean_all()
        .map_err(Into::into)
}
