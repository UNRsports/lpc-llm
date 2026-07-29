//! Model-module downloads: fetch into durable `blobs/` only when missing.
//!
//! Re-running `pull` / engine upgrades must **reuse** existing Gemma (and other)
//! weights — never re-download a non-empty blob.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::catalog::ModelEntry;
use crate::error::{AppError, Result};
use crate::store::{now_unix, InstalledModel, LocalStore};

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .expect("spinner template")
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn hf_resolve_url(repo_id: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/main/{filename}")
}

fn curl_download(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("part");
    if tmp.exists() {
        let _ = fs::remove_file(&tmp);
    }

    let status = if Command::new("curl").arg("--version").output().is_ok() {
        let mut cmd = Command::new("curl");
        cmd.args(["-fL", "--retry", "3", "--retry-delay", "2", "-o"]);
        cmd.arg(&tmp);
        cmd.arg("--progress-bar");
        if let Ok(token) = std::env::var("HF_TOKEN") {
            cmd.arg("-H");
            cmd.arg(format!("Authorization: Bearer {token}"));
        }
        cmd.arg(url);
        cmd.status()
            .map_err(|e| AppError::msg(format!("failed to spawn curl: {e}")))?
    } else if Command::new("wget").arg("--version").output().is_ok() {
        let mut cmd = Command::new("wget");
        cmd.args(["-O"]);
        cmd.arg(&tmp);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            cmd.arg("--header");
            cmd.arg(format!("Authorization: Bearer {token}"));
        }
        cmd.arg(url);
        cmd.status()
            .map_err(|e| AppError::msg(format!("failed to spawn wget: {e}")))?
    } else {
        return Err(AppError::msg(
            "neither `curl` nor `wget` found — install one to download models",
        ));
    };

    if !status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(AppError::msg(format!(
            "download failed for {url} (exit {status}). \
             Gated models need HF_TOKEN; accept the model license on Hugging Face first."
        )));
    }

    fs::rename(&tmp, dest)?;
    Ok(())
}

enum BlobState {
    Cached(PathBuf),
    Downloaded(PathBuf),
}

fn ensure_blob(store: &LocalStore, repo_id: &str, filename: &str) -> Result<BlobState> {
    let dest = store.blob_path(repo_id, filename);
    if dest.exists() && dest.metadata()?.len() > 0 {
        return Ok(BlobState::Cached(dest));
    }
    let url = hf_resolve_url(repo_id, filename);
    curl_download(&url, &dest)?;
    Ok(BlobState::Downloaded(dest))
}

/// Ensure model + tokenizer blobs exist and register them.
/// Existing blobs are always reused (no second download).
pub fn pull_model(store: &LocalStore, entry: &ModelEntry) -> Result<InstalledModel> {
    // Fast path: both blobs already on disk (typical after engine upgrade).
    if let Some(existing) = store.resolve(entry)? {
        eprintln!(
            "{} {} already in model module — reusing blobs (no download)",
            style("·").cyan(),
            style(&entry.name).bold()
        );
        eprintln!("  model     {}", existing.model_path.display());
        eprintln!("  tokenizer {}", existing.tokenizer_path.display());
        return Ok(existing);
    }

    eprintln!(
        "{} pulling {} ({}) from {}",
        style("↓").cyan(),
        style(&entry.name).bold(),
        entry.approx_size,
        entry.hf_repo
    );

    let pb = spinner(&format!("model {}", entry.gguf_file));
    let model_state = ensure_blob(store, &entry.hf_repo, &entry.gguf_file)?;
    pb.finish_and_clear();
    match &model_state {
        BlobState::Cached(p) => {
            eprintln!("{} model  {} (cached)", style("·").cyan(), p.display())
        }
        BlobState::Downloaded(p) => {
            eprintln!("{} model  {}", style("✓").green(), p.display())
        }
    }

    let pb = spinner(&format!("tokenizer {}", entry.tokenizer_repo));
    let tok_state = ensure_blob(store, &entry.tokenizer_repo, "tokenizer.json")?;
    pb.finish_and_clear();
    match &tok_state {
        BlobState::Cached(p) => {
            eprintln!("{} tokenizer {} (cached)", style("·").cyan(), p.display())
        }
        BlobState::Downloaded(p) => {
            eprintln!("{} tokenizer {}", style("✓").green(), p.display())
        }
    }

    let model_path = match model_state {
        BlobState::Cached(p) | BlobState::Downloaded(p) => p,
    };
    let tokenizer_path = match tok_state {
        BlobState::Cached(p) | BlobState::Downloaded(p) => p,
    };

    let installed = InstalledModel {
        name: entry.name.clone(),
        model_path,
        tokenizer_repo: entry.tokenizer_repo.clone(),
        tokenizer_path,
        hf_repo: entry.hf_repo.clone(),
        gguf_file: entry.gguf_file.clone(),
        pulled_at_unix: now_unix(),
    };
    store.record(installed.clone())?;
    eprintln!(
        "{} {} ready  (blobs: {}  engine-cache: {})",
        style("✓").green().bold(),
        style(&entry.name).bold(),
        store.blobs_dir().display(),
        store.cache_dir().display()
    );
    Ok(installed)
}
