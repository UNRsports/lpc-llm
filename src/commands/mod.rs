pub mod adapter;
pub mod config_cmd;
pub mod io_demo;
pub mod job;
pub mod list;
pub mod menu;
pub mod prefetch;
pub mod project_map_cmd;
pub mod rm;
pub mod run;
pub mod search;
pub mod setup;
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

pub fn cmd_list(all: bool) -> Result<()> {
    list::run(all)
}

pub fn cmd_show(name: &str) -> Result<()> {
    show::run(name)
}

pub fn cmd_rm(opts: rm::RmOpts) -> Result<()> {
    rm::run(opts)
}

pub fn cmd_run(opts: run::RunOpts) -> Result<()> {
    run::run(opts)
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

pub fn cmd_adapter_auto_train(opts: crate::user_adapt::AutoTrainOpts) -> Result<()> {
    adapter::auto_train(opts)
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

pub fn cmd_config_init(force: bool, interactive: bool) -> Result<()> {
    config_cmd::init(force, interactive)
}

pub fn cmd_setup() -> Result<()> {
    setup::run()
}

pub fn cmd_config_get(key: &str) -> Result<()> {
    config_cmd::get(key)
}

pub fn cmd_config_example() -> Result<()> {
    config_cmd::print_example()
}

pub fn cmd_search(query: &str) -> Result<()> {
    search::search(query)
}

pub fn cmd_knowledge_list() -> Result<()> {
    search::knowledge_list()
}

pub fn cmd_knowledge_purge() -> Result<()> {
    search::knowledge_purge()
}

pub fn cmd_project_map_build(path: String) -> Result<()> {
    project_map_cmd::build(path)
}

pub fn cmd_project_map_status(path_or_hash: String) -> Result<()> {
    project_map_cmd::status(path_or_hash)
}

pub fn cmd_project_map_rebuild(path: String) -> Result<()> {
    project_map_cmd::rebuild(path)
}
