//! Adapter management CLI (`list` / `create` / `install-demo`).

use console::style;
use dialoguer::Confirm;

use crate::adapter::{train_adapter, write_demo_adapter, TrainConfig};
use crate::catalog;
use crate::config::resolve_train_from;
use crate::error::{AppError, Result};
use crate::pull;
use crate::store::{InstalledAdapter, LocalStore};

pub fn list() -> Result<()> {
    let store = LocalStore::open()?;
    let adapters = store.list_adapters()?;
    if adapters.is_empty() {
        println!(
            "{}",
            style("(no adapters — place one under adapters/<name>/ or run `adapter install-demo`)")
                .dim()
        );
        return Ok(());
    }
    println!(
        "{:<20} {:<16} {}",
        style("NAME").bold(),
        style("BASE").bold(),
        style("PATH").bold()
    );
    for a in adapters {
        println!(
            "{:<20} {:<16} {}",
            a.name,
            a.base_model,
            a.path.display()
        );
    }
    Ok(())
}

pub struct CreateOpts {
    pub from: String,
    pub out: String,
    pub base: String,
    pub rank: usize,
    pub alpha: f64,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    pub last_layers: usize,
    pub pull: bool,
}

pub fn create(opts: CreateOpts) -> Result<()> {
    if opts.out.trim().is_empty() {
        return Err(AppError::msg("--out adapter name must be non-empty"));
    }

    let entry = catalog::find(&opts.base).ok_or_else(|| AppError::UnknownModel(opts.base.clone()))?;
    let store = LocalStore::open()?;
    let from = resolve_train_from(&opts.from, store.train_dir())?;
    let installed = match store.resolve(&entry)? {
        Some(m) => m,
        None => {
            if !opts.pull {
                let ok = Confirm::new()
                    .with_prompt(format!(
                        "base model `{}` is not installed. Pull it now ({})?",
                        opts.base, entry.approx_size
                    ))
                    .default(true)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                if !ok {
                    return Err(AppError::NotInstalled(opts.base));
                }
            }
            pull::pull_model(&store, &entry)?
        }
    };

    let out_dir = store.adapter_path(&opts.out);
    if out_dir.exists() {
        eprintln!(
            "{} overwriting existing adapter dir {}",
            style("!").yellow(),
            out_dir.display()
        );
    }

    let cfg = TrainConfig {
        name: opts.out.clone(),
        base_model: opts.base.clone(),
        rank: opts.rank,
        alpha: opts.alpha,
        steps: opts.steps,
        lr: opts.lr,
        max_seq: opts.max_seq,
        ram_mib: opts.ram_mib,
        last_layers: opts.last_layers,
        seed: 42,
    };

    let pack_cache = store.pack_cache_dir(&entry.name);
    let path = train_adapter(
        &installed.model_path,
        &installed.tokenizer_path,
        pack_cache,
        &from,
        &out_dir,
        cfg,
    )?;

    store.record_adapter(InstalledAdapter {
        name: opts.out.clone(),
        path: path.clone(),
        base_model: opts.base.clone(),
        recorded_at_unix: crate::store::now_unix(),
    })?;

    println!(
        "{} adapter ready — try: lpc-llm run {} --adapter {}",
        style("✓").green(),
        opts.base,
        opts.out
    );
    Ok(())
}

/// Install a zero-filled demo adapter shaped for a known base (integration fixture).
pub fn install_demo(
    name: String,
    base: String,
    layers: usize,
    emb_dim: usize,
    rank: usize,
) -> Result<()> {
    let store = LocalStore::open()?;
    let dir = store.adapter_path(&name);
    write_demo_adapter(
        &dir,
        &name,
        &base,
        layers,
        emb_dim,
        rank,
        (rank * 2) as f64,
        false,
    )?;
    store.record_adapter(InstalledAdapter {
        name: name.clone(),
        path: dir.clone(),
        base_model: base.clone(),
        recorded_at_unix: crate::store::now_unix(),
    })?;
    println!(
        "{} wrote zero demo adapter `{}` for base `{}` → {}",
        style("✓").green(),
        style(&name).bold(),
        base,
        dir.display()
    );
    Ok(())
}
