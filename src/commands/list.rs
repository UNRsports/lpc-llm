use console::style;

use crate::catalog;
use crate::error::Result;
use crate::i18n::Locale;
use crate::store::LocalStore;

/// When `all` is false (default), only installed / local models are listed.
/// When true, print the full catalog with `local` / `available` status.
pub fn run(all: bool) -> Result<()> {
    let loc = Locale::load();
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

    if all {
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
        // Custom / trained models not in the static catalog.
        for m in &installed {
            if catalog::find(&m.name).is_some() {
                continue;
            }
            println!(
                "{:<16} {:<8} {:<10} {:<10} {}",
                m.name,
                style("local").green(),
                "—",
                "—",
                loc.t("list.custom_desc")
            );
        }
        if installed.is_empty() {
            println!();
            println!("{}", style(loc.t("list.none_local")).yellow());
        }
    } else if installed.is_empty() {
        println!("{}", style(loc.t("list.none_local")).yellow());
        println!("{}", style(loc.t("list.none_hint")).dim());
    } else {
        for m in &installed {
            if let Some(entry) = catalog::find(&m.name) {
                println!(
                    "{:<16} {:<8} {:<10} {:<10} {}",
                    entry.name,
                    style("local").green(),
                    entry.approx_size,
                    entry.min_ram_hint,
                    entry.display
                );
            } else {
                println!(
                    "{:<16} {:<8} {:<10} {:<10} {}",
                    m.name,
                    style("local").green(),
                    "—",
                    "—",
                    loc.t("list.custom_desc")
                );
            }
        }
        println!();
        println!("{}", style(loc.t("list.catalog_hint")).dim());
    }

    if !installed.is_empty() {
        println!();
        println!("{}", style(loc.t("list.blobs_hdr")).dim());
        for m in &installed {
            println!("  {} → {}", m.name, m.model_path.display());
        }
        println!(
            "{}",
            style(loc.tf(
                "list.data_root",
                &[("dir", &store.root().display().to_string())]
            ))
            .dim()
        );
        println!(
            "{}",
            style(loc.tf(
                "list.cache_root",
                &[("dir", &store.cache_dir().display().to_string())]
            ))
            .dim()
        );
    }

    Ok(())
}
