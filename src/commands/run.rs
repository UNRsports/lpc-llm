use console::style;
use dialoguer::{Confirm, Select};

use crate::adapter::AdapterSet;
use crate::agent;
use crate::catalog;
use crate::error::{AppError, Result};
use crate::hybrid::HybridConfig;
use crate::infer::ChatSession;
use crate::pull;
use crate::store::LocalStore;
use candle_core::Device;

pub struct RunOpts {
    pub name: Option<String>,
    pub auto_pull: bool,
    pub hybrid: bool,
    pub hot_layers: Option<usize>,
    pub ram_mib: usize,
    pub burst: usize,
    pub adapter: Option<String>,
    pub agent: bool,
    pub agent_model: String,
}

pub fn run(opts: RunOpts) -> Result<()> {
    let store = LocalStore::open()?;

    let name = match opts.name {
        Some(n) => n,
        None => select_model_name(&store)?,
    };

    let (entry, installed) = resolve_run_model(&store, &name, opts.auto_pull)?;

    if opts.agent {
        let router_mib = agent::router_ram_hint_mib(&opts.agent_model);
        if router_mib > opts.ram_mib {
            return Err(AppError::msg(format!(
                "--agent router `{}` needs ~{router_mib} MiB but --ram-mib is {}",
                opts.agent_model, opts.ram_mib
            )));
        }
        let router_entry = catalog::find(&opts.agent_model).ok_or_else(|| {
            AppError::msg(format!("unknown agent model `{}`", opts.agent_model))
        })?;
        if store.resolve(&router_entry)?.is_none() {
            if !opts.auto_pull {
                let ok = Confirm::new()
                    .with_prompt(format!(
                        "agent router `{}` is not installed. Pull it now ({})?",
                        opts.agent_model, router_entry.approx_size
                    ))
                    .default(true)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                if !ok {
                    return Err(AppError::NotInstalled(opts.agent_model.clone()));
                }
            }
            pull::pull_model(&store, &router_entry)?;
        }
    }

    let pack_cache = store.pack_cache_dir(&entry.name);
    let cfg = HybridConfig {
        ram_budget_mib: opts.ram_mib,
        hot_layers: opts.hot_layers,
        first_burst_tokens: opts.burst,
        adapter_resident_bytes: 0, // filled after adapter resolve
    };

    if opts.agent {
        // Time-share: read first user turn → router (exclusive) → drop → main.
        return ChatSession::run_agent_repl(
            &store,
            &installed,
            entry,
            cfg,
            &pack_cache,
            opts.hybrid,
            opts.adapter,
            opts.agent_model,
        );
    }

    let use_hybrid =
        opts.hybrid || opts.adapter.is_some() || entry.name.starts_with("gemma");

    let (adapter_set, cfg) = resolve_adapter(&store, &entry.name, opts.adapter.as_deref(), cfg)?;

    let mut session = ChatSession::load_with_config(
        &installed,
        entry,
        use_hybrid,
        cfg,
        &pack_cache,
        adapter_set,
    )?;
    session.run_repl()?;
    Ok(())
}

pub(crate) fn resolve_adapter(
    store: &LocalStore,
    base_model: &str,
    adapter_name: Option<&str>,
    mut cfg: HybridConfig,
) -> Result<(Option<AdapterSet>, HybridConfig)> {
    let Some(name) = adapter_name else {
        return Ok((None, cfg));
    };
    let installed = store.resolve_adapter(name)?.ok_or_else(|| {
        AppError::msg(format!(
            "adapter `{name}` not found — run `lpc-llm adapter list` \
             or place files under adapters/{name}/"
        ))
    })?;
    if installed.base_model != base_model {
        return Err(AppError::msg(format!(
            "adapter `{}` is for base `{}`, but run target is `{base_model}`",
            name, installed.base_model
        )));
    }
    let set = AdapterSet::load(&installed.path, &Device::Cpu)?;
    if set.base_model() != base_model {
        return Err(AppError::msg(format!(
            "adapter file base `{}` mismatches run target `{base_model}`",
            set.base_model()
        )));
    }
    eprintln!(
        "{} adapter `{}` ({:.1} MiB)",
        style("·").cyan(),
        style(set.name()).bold(),
        set.resident_bytes as f64 / (1024.0 * 1024.0)
    );
    cfg.adapter_resident_bytes = set.resident_bytes;
    Ok((Some(set), cfg))
}

fn resolve_run_model(
    store: &LocalStore,
    name: &str,
    auto_pull: bool,
) -> Result<(catalog::ModelEntry, crate::store::InstalledModel)> {
    if let Some(entry) = catalog::find(name) {
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
                        return Err(AppError::NotInstalled(name.to_string()));
                    }
                } else {
                    eprintln!(
                        "{} auto-pulling {} …",
                        style("↓").cyan(),
                        style(name).bold()
                    );
                }
                pull::pull_model(store, &entry)?
            }
        };
        return Ok((entry, installed));
    }

    // Locally trained / imported models live only in the manifest.
    let installed = store.resolve_name(name)?.ok_or_else(|| {
        AppError::UnknownModel(name.to_string())
    })?;
    let entry = catalog::entry_for_local(
        &installed.name,
        &installed.gguf_file,
        &installed.tokenizer_repo,
    );
    Ok((entry, installed))
}

fn select_model_name(store: &LocalStore) -> Result<String> {
    let catalog = catalog::catalog();
    let installed = store.list_installed()?;
    let installed_set: std::collections::HashSet<_> =
        installed.iter().map(|m| m.name.clone()).collect();

    let mut labels: Vec<String> = catalog
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
    let mut names: Vec<String> = catalog.iter().map(|e| e.name.clone()).collect();

    for m in &installed {
        if catalog::find(&m.name).is_some() {
            continue;
        }
        labels.push(format!(
            "{:<16} [local]  trained/imported",
            m.name
        ));
        names.push(m.name.clone());
    }

    let idx = Select::new()
        .with_prompt("Select a model to run")
        .items(&labels)
        .default(0)
        .interact()
        .map_err(|e| AppError::msg(e.to_string()))?;

    Ok(names[idx].clone())
}
