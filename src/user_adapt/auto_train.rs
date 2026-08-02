//! Idle-time delta LoRA training → `adapters/user_profile/`.

use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use console::style;

use crate::adapter::{train_adapter, TrainConfig};
use crate::catalog;
use crate::error::{AppError, Result};
use crate::pull;
use crate::store::{InstalledAdapter, LocalStore};
use crate::user_adapt::idle::wait_until_idle;
use crate::user_adapt::log::build_training_corpus;

const USER_PROFILE: &str = "user_profile";
const DEFAULT_MIN_SAMPLES: usize = 8;
const DEFAULT_MAX_TRAIN_SECS: u64 = 600;
const DEFAULT_IDLE_SECS: u64 = 120;

pub struct AutoTrainOpts {
    pub base: String,
    pub once: bool,
    pub daemon: bool,
    pub min_samples: usize,
    pub ram_mib: usize,
    pub steps: usize,
    pub rank: usize,
    pub alpha: f64,
    pub max_seq: usize,
    pub last_layers: usize,
    pub idle_secs: u64,
    pub max_train_secs: u64,
    pub pull: bool,
}

impl Default for AutoTrainOpts {
    fn default() -> Self {
        Self {
            base: String::new(),
            once: true,
            daemon: false,
            min_samples: DEFAULT_MIN_SAMPLES,
            ram_mib: 4096,
            steps: 32,
            rank: 4,
            alpha: 8.0,
            max_seq: 128,
            last_layers: 4,
            idle_secs: DEFAULT_IDLE_SECS,
            max_train_secs: DEFAULT_MAX_TRAIN_SECS,
            pull: false,
        }
    }
}

/// Run auto-train once or as a simple idle daemon loop.
pub fn run_auto_train(opts: AutoTrainOpts) -> Result<()> {
    if opts.daemon && opts.once {
        // clap may set both; daemon wins for loop behavior.
    }
    if opts.base.trim().is_empty() {
        return Err(AppError::msg(
            "adapter auto-train requires --base <catalog-model>",
        ));
    }

    if opts.daemon {
        eprintln!(
            "{} auto-train daemon (idle≥{}s, min_samples={}, base={})",
            style("·").cyan(),
            opts.idle_secs,
            opts.min_samples,
            opts.base
        );
        loop {
            wait_until_idle(opts.idle_secs, Duration::from_secs(3600))?;
            match run_one_cycle(&opts) {
                Ok(Some(path)) => {
                    eprintln!(
                        "{} updated {}",
                        style("✓").green(),
                        path.display()
                    );
                }
                Ok(None) => {
                    eprintln!("{}", style("(auto-train skipped — guards)").dim());
                }
                Err(e) => {
                    eprintln!("{} auto-train failed: {e}", style("!").yellow());
                }
            }
            // Cool-down between cycles.
            thread::sleep(Duration::from_secs(opts.idle_secs.max(60)));
        }
    } else {
        if opts.idle_secs > 0 {
            eprintln!(
                "{} waiting for idle ≥ {}s …",
                style("·").cyan(),
                opts.idle_secs
            );
            wait_until_idle(opts.idle_secs, Duration::from_secs(opts.idle_secs + 300))?;
        }
        match run_one_cycle(&opts)? {
            Some(path) => {
                println!(
                    "{} user_profile adapter → {}",
                    style("✓").green(),
                    path.display()
                );
            }
            None => {
                println!("{}", style("(auto-train skipped — guards)").dim());
            }
        }
    }
    Ok(())
}

fn run_one_cycle(opts: &AutoTrainOpts) -> Result<Option<PathBuf>> {
    let store = LocalStore::open()?;
    let corpus_path = store.user_logs_dir().join("auto_train_corpus.txt");
    let n = match build_training_corpus(&store, &corpus_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("{} {e}", style("·").dim());
            return Ok(None);
        }
    };
    if n < opts.min_samples {
        eprintln!(
            "{} samples={n} < min_samples={} — skip",
            style("·").dim(),
            opts.min_samples
        );
        return Ok(None);
    }

    let entry = catalog::find(&opts.base)
        .ok_or_else(|| AppError::UnknownModel(opts.base.clone()))?;
    let installed = match store.resolve(&entry)? {
        Some(m) => m,
        None => {
            if !opts.pull {
                return Err(AppError::NotInstalled(opts.base.clone()));
            }
            pull::pull_model(&store, &entry)?
        }
    };

    let out_dir = store.user_profile_adapter_path();
    let backup_dir = store.adapters_dir().join("user_profile.bak");
    // Snapshot existing adapter for rollback.
    if out_dir.join("adapter.json").exists() {
        let _ = fs::remove_dir_all(&backup_dir);
        copy_dir_recursive(&out_dir, &backup_dir)?;
    }

    let cfg = TrainConfig {
        name: USER_PROFILE.to_string(),
        base_model: opts.base.clone(),
        rank: opts.rank,
        alpha: opts.alpha,
        steps: opts.steps,
        lr: 1e-3,
        max_seq: opts.max_seq,
        ram_mib: opts.ram_mib,
        last_layers: opts.last_layers,
        seed: 7,
    };

    let pack_cache = store.pack_cache_dir(&entry.name);
    let started = Instant::now();
    let train_result = train_adapter(
        &installed.model_path,
        &installed.tokenizer_path,
        pack_cache,
        &corpus_path,
        &out_dir,
        cfg,
    );
    if started.elapsed() > Duration::from_secs(opts.max_train_secs) {
        eprintln!(
            "{} training exceeded max_train_secs={} — keeping result if ok",
            style("!").yellow(),
            opts.max_train_secs
        );
    }

    match train_result {
        Ok(path) => {
            store.record_adapter(InstalledAdapter {
                name: USER_PROFILE.to_string(),
                path: path.clone(),
                base_model: opts.base.clone(),
                recorded_at_unix: crate::store::now_unix(),
            })?;
            let _ = fs::remove_dir_all(&backup_dir);
            Ok(Some(path))
        }
        Err(e) => {
            // Rollback
            if backup_dir.join("adapter.json").exists() {
                let _ = fs::remove_dir_all(&out_dir);
                if let Err(rb) = copy_dir_recursive(&backup_dir, &out_dir) {
                    eprintln!("{} rollback copy failed: {rb}", style("!").red());
                } else {
                    eprintln!("{} rolled back user_profile to previous weights", style("·").yellow());
                }
            }
            Err(e)
        }
    }
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}
