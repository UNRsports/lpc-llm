use console::style;

use crate::error::{AppError, Result};
use crate::store::LocalStore;

pub fn run(name: &str) -> Result<()> {
    let store = LocalStore::open()?;
    if !store.is_installed(name)? {
        return Err(AppError::NotInstalled(name.into()));
    }
    store.remove(name)?;
    println!(
        "{} removed `{}` from registry (model blobs kept under {})",
        style("✓").green(),
        name,
        store.blobs_dir().display()
    );
    Ok(())
}
