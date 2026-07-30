//! Adapter management CLI (`list` / stub `create`).

use console::style;

use crate::adapter::write_demo_adapter;
use crate::error::{AppError, Result};
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

pub fn create(_from: Option<String>, _out: Option<String>, _base: Option<String>) -> Result<()> {
    Err(AppError::msg(
        "`adapter create` is not implemented yet (Phase 4). \
         Use `lpc-llm adapter install-demo --base <model>` for a zero LoRA fixture, \
         or drop `adapter.json` + `weights.bin` under ~/.local/share/lpc-llm/adapters/<name>/.",
    ))
}

/// Install a zero-filled demo adapter shaped for a known base (integration fixture).
pub fn install_demo(name: String, base: String, layers: usize, emb_dim: usize, rank: usize) -> Result<()> {
    let store = LocalStore::open()?;
    let dir = store.adapter_path(&name);
    write_demo_adapter(&dir, &name, &base, layers, emb_dim, rank, (rank * 2) as f64, false)?;
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
