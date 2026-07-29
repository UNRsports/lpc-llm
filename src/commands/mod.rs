pub mod io_demo;
pub mod list;
pub mod menu;
pub mod prefetch;
pub mod rm;
pub mod run;
pub mod show;

use crate::catalog;
use crate::error::{AppError, Result};
use crate::pull;
use crate::store::LocalStore;

pub fn cmd_pull(name: &str) -> Result<()> {
    let entry = catalog::find(name).ok_or_else(|| AppError::UnknownModel(name.into()))?;
    let store = LocalStore::open()?;
    pull::pull_model(&store, &entry)?;
    Ok(())
}

pub fn cmd_list() -> Result<()> {
    list::run()
}

pub fn cmd_show(name: &str) -> Result<()> {
    show::run(name)
}

pub fn cmd_rm(name: &str) -> Result<()> {
    rm::run(name)
}

pub fn cmd_run(
    name: Option<String>,
    auto_pull: bool,
    hybrid: bool,
    hot_layers: Option<usize>,
    ram_mib: usize,
    burst: usize,
) -> Result<()> {
    run::run(run::RunOpts {
        name,
        auto_pull,
        hybrid,
        hot_layers,
        ram_mib,
        burst,
    })
}

pub fn cmd_prefetch(name: &str, auto_pull: bool) -> Result<()> {
    prefetch::run(name, auto_pull)
}

pub fn cmd_menu() -> Result<()> {
    menu::run()
}
