use console::style;
use dialoguer::Select;

use crate::catalog;
use crate::commands;
use crate::error::{AppError, Result};
use crate::store::LocalStore;

pub fn run() -> Result<()> {
    println!(
        "{} {}",
        style("lpc-llm").bold().cyan(),
        style("— local LLM runner (pure Rust / Candle)").dim()
    );

    let actions = vec![
        "Run a model (chat)",
        "List models",
        "Pull a model",
        "Show model info",
        "Remove local registry entry",
        "Exit",
    ];

    loop {
        let action = Select::new()
            .with_prompt("What do you want to do?")
            .items(&actions)
            .default(0)
            .interact()
            .map_err(|e| AppError::msg(e.to_string()))?;

        match action {
            0 => {
                commands::cmd_run(None, false, false, None, 4096, 24, None)?;
                return Ok(());
            }
            1 => commands::cmd_list()?,
            2 => {
                let name = pick_catalog_name("Select a model to pull")?;
                commands::cmd_pull(&name)?;
            }
            3 => {
                let name = pick_catalog_name("Select a model to show")?;
                commands::cmd_show(&name)?;
            }
            4 => {
                let store = LocalStore::open()?;
                let installed = store.list_installed()?;
                if installed.is_empty() {
                    println!("{}", style("No locally registered models.").yellow());
                    continue;
                }
                let labels: Vec<_> = installed.iter().map(|m| m.name.as_str()).collect();
                let idx = Select::new()
                    .with_prompt("Select a model to remove from registry")
                    .items(&labels)
                    .interact()
                    .map_err(|e| AppError::msg(e.to_string()))?;
                commands::cmd_rm(labels[idx])?;
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
