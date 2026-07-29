use console::style;
use dialoguer::{Confirm, Select};

use crate::catalog;
use crate::error::{AppError, Result};
use crate::hybrid::HybridConfig;
use crate::infer::ChatSession;
use crate::pull;
use crate::store::LocalStore;

pub struct RunOpts {
    pub name: Option<String>,
    pub auto_pull: bool,
    pub hybrid: bool,
    pub hot_layers: Option<usize>,
    pub ram_mib: usize,
    pub burst: usize,
}

pub fn run(opts: RunOpts) -> Result<()> {
    let store = LocalStore::open()?;

    let name = match opts.name {
        Some(n) => n,
        None => select_model_name(&store)?,
    };

    let entry = catalog::find(&name).ok_or_else(|| AppError::UnknownModel(name.clone()))?;

    let use_hybrid = opts.hybrid || entry.name.starts_with("gemma");
    let cfg = HybridConfig {
        ram_budget_mib: opts.ram_mib,
        hot_layers: opts.hot_layers,
        first_burst_tokens: opts.burst,
    };

    // Prefer durable blobs via resolve (no re-download after engine upgrade).
    let installed = match store.resolve(&entry)? {
        Some(m) => m,
        None => {
            if !opts.auto_pull {
                let ok = Confirm::new()
                    .with_prompt(format!(
                        "`{name}` is not installed. Pull it now ({})?",
                        entry.approx_size
                    ))
                    .default(true)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                if !ok {
                    return Err(AppError::NotInstalled(name));
                }
            } else {
                eprintln!(
                    "{} auto-pulling {} …",
                    style("↓").cyan(),
                    style(&name).bold()
                );
            }
            pull::pull_model(&store, &entry)?
        }
    };

    let pack_cache = store.pack_cache_dir(&entry.name);
    let mut session =
        ChatSession::load_with_config(&installed, entry, use_hybrid, cfg, &pack_cache)?;
    session.run_repl()?;
    Ok(())
}

fn select_model_name(store: &LocalStore) -> Result<String> {
    let catalog = catalog::catalog();
    let installed = store.list_installed()?;
    let installed_set: std::collections::HashSet<_> =
        installed.iter().map(|m| m.name.clone()).collect();

    let labels: Vec<String> = catalog
        .iter()
        .map(|e| {
            let tag = if installed_set.contains(&e.name) {
                "local"
            } else {
                "pull"
            };
            format!("{:<16} [{tag}]  {} ({})", e.name, e.display, e.approx_size)
        })
        .collect();

    let idx = Select::new()
        .with_prompt("Select a model to run")
        .items(&labels)
        .default(0)
        .interact()
        .map_err(|e| AppError::msg(e.to_string()))?;

    Ok(catalog[idx].name.clone())
}
