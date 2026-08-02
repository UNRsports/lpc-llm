//! Phase 6 CLI: `lpc-llm job init|run|status|import|convert`.

use crate::error::Result;
use crate::job::{
    convert_checkpoint_to_gguf, import_gguf, job_status, run_job, write_template,
};
use crate::store::LocalStore;

pub fn init(template: String, out: String) -> Result<()> {
    write_template(&template, &out)
}

pub fn run(config: String, local: bool) -> Result<()> {
    let store = LocalStore::open()?;
    run_job(&store, &config, local)
}

pub fn status(name: String) -> Result<()> {
    let store = LocalStore::open()?;
    job_status(&store, &name)
}

pub fn import(gguf: String, tokenizer: String, name: String) -> Result<()> {
    let store = LocalStore::open()?;
    import_gguf(&store, gguf, tokenizer, &name)
}

pub fn convert(from_dir: String, name: String, backend: String) -> Result<()> {
    let store = LocalStore::open()?;
    convert_checkpoint_to_gguf(&store, from_dir, &name, &backend)?;
    Ok(())
}
