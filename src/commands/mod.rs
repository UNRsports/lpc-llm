pub mod adapter;
pub mod config_cmd;
pub mod io_demo;
pub mod job;
pub mod list;
pub mod menu;
pub mod prefetch;
pub mod rm;
pub mod run;
pub mod show;
pub mod train;

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
    adapter: Option<String>,
    agent: bool,
    agent_model: String,
) -> Result<()> {
    run::run(run::RunOpts {
        name,
        auto_pull,
        hybrid,
        hot_layers,
        ram_mib,
        burst,
        adapter,
        agent,
        agent_model,
    })
}

pub fn cmd_prefetch(name: &str, auto_pull: bool) -> Result<()> {
    prefetch::run(name, auto_pull)
}

pub fn cmd_menu() -> Result<()> {
    menu::run()
}

pub fn cmd_adapter_list() -> Result<()> {
    adapter::list()
}

pub fn cmd_adapter_create(opts: adapter::CreateOpts) -> Result<()> {
    adapter::create(opts)
}

pub fn cmd_adapter_install_demo(
    name: String,
    base: String,
    layers: usize,
    emb_dim: usize,
    rank: usize,
) -> Result<()> {
    adapter::install_demo(name, base, layers, emb_dim, rank)
}

pub fn cmd_train_scratch(opts: train::ScratchOpts) -> Result<()> {
    train::scratch(opts)
}

pub fn cmd_train_sft(opts: train::SftOpts) -> Result<()> {
    train::sft(opts)
}

pub fn cmd_train_dpo(opts: train::DpoOpts) -> Result<()> {
    train::dpo(opts)
}

pub fn cmd_train_export(opts: train::ExportOpts) -> Result<()> {
    train::export(opts)
}

pub fn cmd_job_init(template: String, out: String) -> Result<()> {
    job::init(template, out)
}

pub fn cmd_job_run(config: String, local: bool) -> Result<()> {
    job::run(config, local)
}

pub fn cmd_job_status(name: String) -> Result<()> {
    job::status(name)
}

pub fn cmd_job_import(gguf: String, tokenizer: String, name: String) -> Result<()> {
    job::import(gguf, tokenizer, name)
}

pub fn cmd_job_convert(from_dir: String, name: String, backend: String) -> Result<()> {
    job::convert(from_dir, name, backend)
}

pub fn cmd_config_show() -> Result<()> {
    config_cmd::show()
}

pub fn cmd_config_init(force: bool) -> Result<()> {
    config_cmd::init(force)
}

pub fn cmd_config_get(key: &str) -> Result<()> {
    config_cmd::get(key)
}

pub fn cmd_config_example() -> Result<()> {
    config_cmd::print_example()
}
