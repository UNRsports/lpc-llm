use console::style;

use crate::catalog;
use crate::error::Result;
use crate::store::LocalStore;

pub fn run() -> Result<()> {
    let store = LocalStore::open()?;
    let installed = store.list_installed()?;
    let installed_names: std::collections::HashSet<_> =
        installed.iter().map(|m| m.name.as_str()).collect();

    println!(
        "{:<16} {:<8} {:<10} {:<10} {}",
        style("NAME").bold(),
        style("STATUS").bold(),
        style("SIZE").bold(),
        style("RAM").bold(),
        style("DESCRIPTION").bold()
    );

    for entry in catalog::catalog() {
        let status = if installed_names.contains(entry.name.as_str()) {
            style("local").green().to_string()
        } else {
            style("available").dim().to_string()
        };
        println!(
            "{:<16} {:<8} {:<10} {:<10} {}",
            entry.name, status, entry.approx_size, entry.min_ram_hint, entry.display
        );
    }

    if !installed.is_empty() {
        println!();
        println!("{}", style("Model module (durable blobs):").dim());
        for m in &installed {
            println!("  {} → {}", m.name, m.model_path.display());
        }
        println!(
            "{}",
            style(format!(
                "Data root: {}  (blobs + engine cache under here)",
                store.root().display()
            ))
            .dim()
        );
        println!(
            "{}",
            style(format!(
                "Engine cache (regenerable): {}",
                store.cache_dir().display()
            ))
            .dim()
        );
    }

    Ok(())
}
