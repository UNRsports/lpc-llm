use console::style;
use dialoguer::Confirm;

use crate::error::{AppError, Result};
use crate::i18n::Locale;
use crate::store::{format_bytes, LocalStore};

pub struct RmOpts {
    pub name: String,
    /// Delete durable blobs + pack cache + registry.
    pub purge: bool,
    /// Delete pack cache only (blobs and registry kept).
    pub cache_only: bool,
    /// With `--purge`, also remove LoRA adapters bound to this base model.
    pub with_adapters: bool,
    /// Skip confirmation for destructive modes.
    pub yes: bool,
}

pub fn run(opts: RmOpts) -> Result<()> {
    let loc = Locale::load();

    if opts.with_adapters && !opts.purge {
        return Err(AppError::msg(loc.t("rm.err_with_adapters")));
    }
    if opts.purge && opts.cache_only {
        return Err(AppError::msg(loc.t("rm.err_both_flags")));
    }

    let store = LocalStore::open()?;
    if !store.is_installed(&opts.name)? {
        return Err(AppError::NotInstalled(opts.name));
    }

    if opts.cache_only {
        return run_cache_only(&store, &opts, &loc);
    }
    if opts.purge {
        return run_purge(&store, &opts, &loc);
    }
    run_soft(&store, &opts, &loc)
}

fn run_soft(store: &LocalStore, opts: &RmOpts, loc: &Locale) -> Result<()> {
    store.remove(&opts.name)?;
    let dir = store.blobs_dir().display().to_string();
    println!(
        "{} {}",
        style("✓").green(),
        loc.tf("rm.soft_ok", &[("name", &opts.name), ("dir", &dir)])
    );
    println!("{}", style(loc.t("rm.soft_tip")).dim());
    Ok(())
}

fn run_cache_only(store: &LocalStore, opts: &RmOpts, loc: &Locale) -> Result<()> {
    let dir = store.pack_cache_model_dir(&opts.name);
    let dir_s = dir.display().to_string();
    if !opts.yes {
        let ok = Confirm::new()
            .with_prompt(loc.tf(
                "rm.cache_confirm",
                &[("name", &opts.name), ("dir", &dir_s)],
            ))
            .default(false)
            .interact()
            .map_err(|e| AppError::msg(e.to_string()))?;
        if !ok {
            println!("{}", style(loc.t("rm.cancelled")).yellow());
            return Ok(());
        }
    }
    let report = store.wipe_pack_cache(&opts.name)?;
    let bytes = format_bytes(report.bytes_freed);
    println!(
        "{} {}",
        style("✓").green(),
        loc.tf(
            "rm.cache_ok",
            &[("name", &opts.name), ("bytes", &bytes)]
        )
    );
    for p in &report.paths {
        println!("  - {}", p.display());
    }
    if report.paths.is_empty() {
        println!("{}", style(loc.t("rm.nothing")).dim());
    }
    Ok(())
}

fn run_purge(store: &LocalStore, opts: &RmOpts, loc: &Locale) -> Result<()> {
    if !opts.yes {
        let root = store.root().display().to_string();
        let mut prompt = loc.tf(
            "rm.purge_confirm",
            &[("name", &opts.name), ("dir", &root)],
        );
        if opts.with_adapters {
            let adapters = store.adapters_for_base(&opts.name)?;
            if !adapters.is_empty() {
                let list = adapters
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let n = adapters.len().to_string();
                prompt.push_str(&loc.tf(
                    "rm.purge_adapters",
                    &[("n", &n), ("list", &list)],
                ));
            } else {
                prompt.push_str(loc.t("rm.purge_adapters_none"));
            }
        }
        let ok = Confirm::new()
            .with_prompt(prompt)
            .default(false)
            .interact()
            .map_err(|e| AppError::msg(e.to_string()))?;
        if !ok {
            println!("{}", style(loc.t("rm.cancelled")).yellow());
            return Ok(());
        }
    }

    let report = store.purge_model(&opts.name, opts.with_adapters)?;
    let bytes = format_bytes(report.bytes_freed);
    println!(
        "{} {}",
        style("✓").green(),
        loc.tf(
            "rm.purge_ok",
            &[("name", &report.name), ("bytes", &bytes)]
        )
    );
    if report.registry_removed {
        println!("{}", loc.t("rm.registry_removed"));
    }
    for p in &report.blob_paths {
        let path = p.display().to_string();
        println!("{}", loc.tf("rm.blob_line", &[("path", &path)]));
    }
    for p in &report.cache_paths {
        let path = p.display().to_string();
        println!("{}", loc.tf("rm.cache_line", &[("path", &path)]));
    }
    for name in &report.adapters_removed {
        println!("{}", loc.tf("rm.adapter_line", &[("name", name)]));
    }
    for p in &report.skipped_shared {
        let path = p.display().to_string();
        println!(
            "{} {}",
            style("·").yellow(),
            loc.tf("rm.skipped_shared", &[("path", &path)])
        );
    }
    Ok(())
}
