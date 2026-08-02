//! CLI: `lpc-llm project-map build|status|rebuild`.

use std::path::PathBuf;

use console::style;

use crate::error::{AppError, Result};
use crate::project_map::{build_project_map, load_status, rebuild_project_map};
use crate::store::LocalStore;

pub fn build(path: String) -> Result<()> {
    let store = LocalStore::open()?;
    let root = PathBuf::from(&path);
    if !root.exists() {
        return Err(AppError::msg(format!("path not found: {path}")));
    }
    eprintln!("{} indexing {} …", style("·").cyan(), root.display());
    let st = build_project_map(&store, &root)?;
    print_status(&st);
    Ok(())
}

pub fn rebuild(path: String) -> Result<()> {
    let store = LocalStore::open()?;
    let root = PathBuf::from(&path);
    if !root.exists() {
        return Err(AppError::msg(format!("path not found: {path}")));
    }
    eprintln!("{} rebuilding {} …", style("·").cyan(), root.display());
    let st = rebuild_project_map(&store, &root)?;
    print_status(&st);
    Ok(())
}

pub fn status(path_or_hash: String) -> Result<()> {
    let store = LocalStore::open()?;
    let st = load_status(&store, &path_or_hash)?;
    print_status(&st);
    Ok(())
}

fn print_status(st: &crate::project_map::MapStatus) {
    println!("{} project-map", style("✓").green());
    println!("  hash:     {}", st.hash);
    println!("  source:   {}", st.source_path);
    println!("  dir:      {}", st.dir.display());
    println!("  nodes:    {}", st.node_count);
    println!("  edges:    {}", st.edge_count);
    println!("  map.bin:  {} bytes", st.map_bin_bytes);
    println!("  built_at: {}", st.built_at_unix);
}
