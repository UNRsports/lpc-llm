//! Phase 5 CLI: `lpc-llm train scratch|sft|dpo|export`.

use std::path::PathBuf;

use console::style;

use crate::config::resolve_train_from;
use crate::error::{AppError, Result};
use crate::store::LocalStore;
use crate::train::checkpoint::CONFIG_FILE;
use crate::train::{
    export_and_register, export_checkpoint_dir, train_dpo, train_scratch, train_sft_full, DpoConfig,
    ScratchConfig, SftConfig,
};

pub struct ScratchOpts {
    pub from: String,
    pub out: String,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub grad_checkpoint: bool,
    pub n_embd: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_ff: usize,
    pub no_register: bool,
}

pub struct SftOpts {
    pub ckpt: String,
    pub from: String,
    pub out: String,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub grad_checkpoint: bool,
    pub no_register: bool,
}

pub struct DpoOpts {
    pub ckpt: String,
    pub from: String,
    pub out: String,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub beta: f64,
    pub grad_checkpoint: bool,
    pub no_register: bool,
}

pub struct ExportOpts {
    pub ckpt: String,
    pub out_gguf: Option<String>,
    pub name: Option<String>,
    pub register: bool,
}

pub fn scratch(opts: ScratchOpts) -> Result<()> {
    let store = LocalStore::open()?;
    let from = resolve_train_from(&opts.from, store.train_dir())?;
    let out_dir = store
        .cache_dir()
        .join("train")
        .join(opts.out.replace([':', '/'], "_"));
    train_scratch(
        &store,
        &from,
        &out_dir,
        ScratchConfig {
            name: opts.out,
            steps: opts.steps,
            lr: opts.lr,
            max_seq: opts.max_seq,
            ram_mib: opts.ram_mib,
            grad_checkpoint: opts.grad_checkpoint,
            n_embd: opts.n_embd,
            n_layers: opts.n_layers,
            n_heads: opts.n_heads,
            n_kv_heads: opts.n_heads,
            n_ff: opts.n_ff,
            register: !opts.no_register,
            ..ScratchConfig::default()
        },
    )?;
    Ok(())
}

pub fn sft(opts: SftOpts) -> Result<()> {
    let store = LocalStore::open()?;
    let from = resolve_train_from(&opts.from, store.train_dir())?;
    let ckpt = resolve_ckpt(&store, &opts.ckpt)?;
    let out_dir = store
        .cache_dir()
        .join("train")
        .join(opts.out.replace([':', '/'], "_"));
    train_sft_full(
        &store,
        &ckpt,
        &from,
        &out_dir,
        SftConfig {
            name: opts.out,
            steps: opts.steps,
            lr: opts.lr,
            max_seq: opts.max_seq,
            ram_mib: opts.ram_mib,
            grad_checkpoint: opts.grad_checkpoint,
            register: !opts.no_register,
        },
    )?;
    Ok(())
}

pub fn dpo(opts: DpoOpts) -> Result<()> {
    let store = LocalStore::open()?;
    let from = resolve_train_from(&opts.from, store.train_dir())?;
    let ckpt = resolve_ckpt(&store, &opts.ckpt)?;
    let out_dir = store
        .cache_dir()
        .join("train")
        .join(opts.out.replace([':', '/'], "_"));
    train_dpo(
        &store,
        &ckpt,
        &from,
        &out_dir,
        DpoConfig {
            name: opts.out,
            steps: opts.steps,
            lr: opts.lr,
            max_seq: opts.max_seq,
            ram_mib: opts.ram_mib,
            beta: opts.beta,
            grad_checkpoint: opts.grad_checkpoint,
            register: !opts.no_register,
        },
    )?;
    Ok(())
}

pub fn export(opts: ExportOpts) -> Result<()> {
    let store = LocalStore::open()?;
    let ckpt = resolve_ckpt(&store, &opts.ckpt)?;
    if opts.register {
        let name = opts
            .name
            .ok_or_else(|| AppError::msg("--name is required with --register"))?;
        export_and_register(&store, &ckpt, &name)?;
    } else {
        let out = opts
            .out_gguf
            .ok_or_else(|| AppError::msg("provide --out <model.gguf> or --register --name …"))?;
        export_checkpoint_dir(&ckpt, &out)?;
        eprintln!(
            "{} exported {} (not registered — pass --register --name)",
            style("✓").green(),
            out
        );
    }
    Ok(())
}

fn resolve_ckpt(store: &LocalStore, spec: &str) -> Result<PathBuf> {
    let p = PathBuf::from(spec);
    if p.join(CONFIG_FILE).is_file() {
        return Ok(p);
    }
    let under_cache = store
        .cache_dir()
        .join("train")
        .join(spec.replace([':', '/'], "_"));
    if under_cache.join(CONFIG_FILE).is_file() {
        return Ok(under_cache);
    }
    let jobs = store.cache_dir().join("jobs");
    if jobs.is_dir() {
        for ent in std::fs::read_dir(&jobs)? {
            let ent = ent?;
            let cand = ent
                .path()
                .join("ckpts")
                .join(spec.replace([':', '/'], "_"));
            if cand.join(CONFIG_FILE).is_file() {
                return Ok(cand);
            }
        }
    }
    Err(AppError::msg(format!(
        "checkpoint `{spec}` not found (tried path and cache/train/…)"
    )))
}
