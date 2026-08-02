use console::style;
use dialoguer::{Confirm, Select};

use crate::adapter::AdapterSet;
use crate::agent;
use crate::catalog;
use crate::error::{AppError, Result};
use crate::hybrid::HybridConfig;
use crate::infer::{ChatSession, SessionExtras};
use crate::knowledge::KnowledgeStore;
use crate::project_map::{resolve_map_dir, ProjectMapReader};
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
    /// Disable automatic `adapters/user_profile/` attach.
    pub no_user_profile: bool,
    /// Path or hash for project-map overview context.
    pub project_map: Option<String>,
    /// Inject retrieved knowledge into prompts.
    pub knowledge: bool,
}

pub fn run(opts: RunOpts) -> Result<()> {
    let store = LocalStore::open()?;

    let name = match opts.name.as_deref() {
        Some(n) => n.to_string(),
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

    let extras = build_extras(&store, &opts)?;

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
            opts.no_user_profile,
            extras,
        );
    }

    let adapter_name = resolve_adapter_name(
        &store,
        &entry.name,
        opts.adapter.as_deref(),
        None,
        opts.no_user_profile,
    )?;

    let use_hybrid = opts.hybrid
        || adapter_name.is_some()
        || opts.project_map.is_some()
        || entry.name.starts_with("gemma");

    let (adapter_set, cfg) =
        resolve_adapter(&store, &entry.name, adapter_name.as_deref(), cfg)?;

    let mut session = ChatSession::load_with_config(
        &installed,
        entry,
        use_hybrid,
        cfg,
        &pack_cache,
        adapter_set,
        extras,
    )?;
    session.run_repl()?;
    Ok(())
}

fn build_extras(store: &LocalStore, opts: &RunOpts) -> Result<SessionExtras> {
    let knowledge = if opts.knowledge {
        Some(KnowledgeStore::open(store)?)
    } else {
        // Still open for gap-triggered background jobs; injection optional.
        Some(KnowledgeStore::open(store)?)
    };
    let project_map = if let Some(ref spec) = opts.project_map {
        let (_hash, dir) = resolve_map_dir(store, spec)?;
        eprintln!(
            "{} project-map {}",
            style("·").cyan(),
            style(dir.display()).bold()
        );
        Some(ProjectMapReader::open(&dir)?)
    } else {
        None
    };
    Ok(SessionExtras {
        knowledge,
        inject_knowledge: opts.knowledge,
        project_map,
        log_turns: true,
        model_name: String::new(), // filled in load_with_config
    })
}

/// Priority: explicit `--adapter` > agent choice > `user_profile` (if present) > none.
pub(crate) fn resolve_adapter_name(
    store: &LocalStore,
    base_model: &str,
    explicit: Option<&str>,
    agent_choice: Option<&str>,
    no_user_profile: bool,
) -> Result<Option<String>> {
    if let Some(name) = explicit {
        return Ok(Some(name.to_string()));
    }
    if let Some(name) = agent_choice {
        return Ok(Some(name.to_string()));
    }
    if no_user_profile {
        return Ok(None);
    }
    // Auto-attach user_profile when present and base matches.
    if let Some(a) = store.resolve_adapter("user_profile")? {
        if a.base_model == base_model {
            eprintln!(
                "{} auto-attaching adapter `user_profile`",
                style("·").cyan()
            );
            return Ok(Some("user_profile".into()));
        }
        eprintln!(
            "{} user_profile base `{}` ≠ run target `{base_model}` — skip",
            style("·").dim(),
            a.base_model
        );
    }
    Ok(None)
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
