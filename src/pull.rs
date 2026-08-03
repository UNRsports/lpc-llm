//! Model-module downloads: fetch into durable `blobs/` only when missing.
//!
//! Re-running `pull` / engine upgrades must **reuse** existing Gemma (and other)
//! weights — never re-download a non-empty blob.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

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

fn format_bytes(n: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let x = n as f64;
    if x >= GIB {
        format!("{:.2} GiB", x / GIB)
    } else {
        format!("{:.1} MiB", x / MIB)
    }
}

/// Best-effort Content-Length via HTTP HEAD (follows redirects like the download).
fn probe_content_length(url: &str) -> Option<u64> {
    let mut cmd = Command::new("curl");
    cmd.args(["-fsSL", "-I"]);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        cmd.arg("-H");
        cmd.arg(format!("Authorization: Bearer {token}"));
    }
    cmd.arg(url);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let headers = String::from_utf8_lossy(&out.stdout);
    headers
        .lines()
        .filter_map(|line| {
            let lower = line.to_ascii_lowercase();
            lower
                .strip_prefix("content-length:")
                .and_then(|v| v.trim().parse::<u64>().ok())
        })
        .last()
}

fn curl_download(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    // `foo.gguf` → `foo.part` (resume-friendly; do not delete existing partials).
    let tmp = dest.with_extension("part");
    let already = tmp.metadata().map(|m| m.len()).unwrap_or(0);
    let total_hint = probe_content_length(url);

    if already > 0 {
        eprintln!(
            "  resuming {} from {}",
            dest.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("download"),
            format_bytes(already)
        );
    }

    if Command::new("curl").arg("--version").output().is_ok() {
        let mut cmd = Command::new("curl");
        // `-C -` resumes; `-sS` silent body, errors on stderr; progress via our poller
        // (curl `--progress-bar` uses `\r`, which background terminal captures miss).
        cmd.args(["-fL", "--retry", "3", "--retry-delay", "2", "-C", "-", "-o"]);
        cmd.arg(&tmp);
        cmd.arg("-sS");
        if let Ok(token) = std::env::var("HF_TOKEN") {
            cmd.arg("-H");
            cmd.arg(format!("Authorization: Bearer {token}"));
        }
        cmd.arg(url);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| AppError::msg(format!("failed to spawn curl: {e}")))?;

        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        let tmp_watch = tmp.clone();
        let started = Instant::now();
        let monitor = thread::spawn(move || {
            let mut last = already;
            while !done_flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(2));
                let now = fs::metadata(&tmp_watch).map(|m| m.len()).unwrap_or(last);
                let elapsed = started.elapsed().as_secs_f64().max(0.001);
                let speed = (now.saturating_sub(already)) as f64 / elapsed;
                let speed_s = if speed >= 1024.0 * 1024.0 {
                    format!("{:.1} MiB/s", speed / (1024.0 * 1024.0))
                } else {
                    format!("{:.0} KiB/s", speed / 1024.0)
                };
                let pct = total_hint
                    .filter(|&t| t > 0)
                    .map(|t| 100.0 * (now as f64) / (t as f64))
                    .unwrap_or(-1.0);
                if pct >= 0.0 {
                    eprintln!(
                        "  download: {} / {} ({:.1}%)  {}",
                        format_bytes(now),
                        format_bytes(total_hint.unwrap_or(0)),
                        pct,
                        speed_s
                    );
                } else {
                    eprintln!("  download: {}  {}", format_bytes(now), speed_s);
                }
                last = now;
            }
        });

        let stderr = child.stderr.take();
        let status = child
            .wait()
            .map_err(|e| AppError::msg(format!("curl wait: {e}")))?;
        done.store(true, Ordering::Relaxed);
        let _ = monitor.join();

        if let Some(err) = stderr {
            let mut buf = String::new();
            for line in BufReader::new(err).lines().flatten() {
                if !line.trim().is_empty() {
                    buf.push_str(&line);
                    buf.push('\n');
                }
            }
            if !buf.is_empty() && !status.success() {
                eprintln!("{}", style(buf.trim_end()).yellow());
            }
        }

        if !status.success() {
            // Keep `.part` for resume on the next attempt.
            return Err(AppError::msg(format!(
                "download failed for {url} (exit {status}). \
                 Partial kept at {}. \
                 Gated models need HF_TOKEN; accept the model license on Hugging Face first.",
                tmp.display()
            )));
        }
    } else if Command::new("wget").arg("--version").output().is_ok() {
        let mut cmd = Command::new("wget");
        cmd.args(["-c", "-O"]);
        cmd.arg(&tmp);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            cmd.arg("--header");
            cmd.arg(format!("Authorization: Bearer {token}"));
        }
        cmd.arg(url);
        let status = cmd
            .status()
            .map_err(|e| AppError::msg(format!("failed to spawn wget: {e}")))?;
        if !status.success() {
            return Err(AppError::msg(format!(
                "download failed for {url} (exit {status}). \
                 Gated models need HF_TOKEN; accept the model license on Hugging Face first."
            )));
        }
    } else {
        return Err(AppError::msg(
            "neither `curl` nor `wget` found — install one to download models",
        ));
    }

    let final_len = tmp.metadata()?.len();
    if final_len == 0 {
        let _ = fs::remove_file(&tmp);
        return Err(AppError::msg(format!("download produced empty file for {url}")));
    }
    if let Some(total) = total_hint {
        if total > 0 && final_len < total {
            return Err(AppError::msg(format!(
                "download incomplete for {url}: got {} of {} — re-run pull to resume",
                format_bytes(final_len),
                format_bytes(total)
            )));
        }
    }

    fs::rename(&tmp, dest)?;
    eprintln!("  download: {} complete", format_bytes(final_len));
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
