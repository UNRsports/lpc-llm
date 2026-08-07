use console::style;
use dialoguer::Select;

use crate::catalog;
use crate::commands;
use crate::error::{AppError, Result};
use crate::i18n::Locale;
use crate::store::LocalStore;

pub fn run() -> Result<()> {
    let loc = Locale::load();
    println!(
        "{} {}",
        style("lpc-llm").bold().cyan(),
        style(loc.t("menu.tagline")).dim()
    );

    let actions = vec![
        loc.t("menu.act_run"),
        loc.t("menu.act_list"),
        loc.t("menu.act_pull"),
        loc.t("menu.act_show"),
        loc.t("menu.act_rm"),
        loc.t("menu.act_exit"),
    ];

    loop {
        let action = Select::new()
            .with_prompt(loc.t("menu.prompt"))
            .items(&actions)
            .default(0)
            .interact()
            .map_err(|e| AppError::msg(e.to_string()))?;

        match action {
            0 => {
                commands::cmd_run(commands::run::RunOpts {
                    name: None,
                    auto_pull: false,
                    hybrid: false,
                    hot_layers: None,
                    ram_mib: 4096,
                    max_tokens: 2048,
                    adapter: None,
                    agent: false,
                    agent_model: "smollm2:360m".into(),
                    no_user_profile: false,
                    project_map: None,
                    knowledge: false,
                    device: None,
                })?;
                return Ok(());
            }
            1 => commands::cmd_list(false)?,
            2 => {
                let name = pick_catalog_name(loc.t("menu.pick_pull"))?;
                commands::cmd_pull(&name)?;
            }
            3 => {
                let store = LocalStore::open()?;
                let installed = store.list_installed()?;
                if installed.is_empty() {
                    println!("{}", style(loc.t("menu.rm_empty")).yellow());
                    println!("{}", style(loc.t("list.none_hint")).dim());
                    continue;
                }
                let labels: Vec<_> = installed.iter().map(|m| m.name.as_str()).collect();
                let idx = Select::new()
                    .with_prompt(loc.t("menu.pick_show"))
                    .items(&labels)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                commands::cmd_show(labels[idx])?;
            }
            4 => {
                let store = LocalStore::open()?;
                let installed = store.list_installed()?;
                if installed.is_empty() {
                    println!("{}", style(loc.t("menu.rm_empty")).yellow());
                    continue;
                }
                let labels: Vec<_> = installed.iter().map(|m| m.name.as_str()).collect();
                let idx = Select::new()
                    .with_prompt(loc.t("menu.rm_pick"))
                    .items(&labels)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                let name = labels[idx].to_string();
                let modes = [
                    loc.t("menu.rm_mode_soft"),
                    loc.t("menu.rm_mode_purge"),
                    loc.t("menu.rm_mode_cache"),
                ];
                let mode = Select::new()
                    .with_prompt(loc.t("menu.rm_mode"))
                    .items(&modes)
                    .default(0)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                let (purge, cache_only) = match mode {
                    1 => (true, false),
                    2 => (false, true),
                    _ => (false, false),
                };
                commands::cmd_rm(commands::rm::RmOpts {
                    name,
                    purge,
                    cache_only,
                    with_adapters: false,
                    yes: false,
                })?;
            }
            _ => return Ok(()),
        }
    }
}

fn pick_catalog_name(prompt: &str) -> Result<String> {
    let catalog = catalog::catalog();
    let labels: Vec<String> = catalog
        .iter()
        .map(|e| format!("{:<16}  {} ({})", e.name, e.display, e.approx_size))
        .collect();
    let idx = Select::new()
        .with_prompt(prompt)
        .items(&labels)
        .default(0)
        .interact()
        .map_err(|e| AppError::msg(e.to_string()))?;
    Ok(catalog[idx].name.clone())
}
