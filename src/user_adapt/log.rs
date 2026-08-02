//! Conversation / correction logs under `cache/user_logs/` (private, rotated).

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};
use crate::store::{now_unix, LocalStore};
use crate::user_adapt::features::extract_style_features;

const LOG_FILE: &str = "turns.jsonl";
const MAX_LOG_BYTES: u64 = 8 * 1024 * 1024; // 8 MiB soft cap
const MAX_FIELD_CHARS: usize = 8_192;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLogEntry {
    pub ts_unix: u64,
    pub model: String,
    pub user: String,
    pub assistant: String,
    /// Optional: user correction / preferred rewrite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction: Option<String>,
    #[serde(default)]
    pub style_notes: Vec<String>,
}

/// Append one turn to the rotating JSONL log.
pub fn append_turn(
    store: &LocalStore,
    model: &str,
    user: &str,
    assistant: &str,
    correction: Option<&str>,
) -> Result<()> {
    let dir = store.user_logs_dir();
    fs::create_dir_all(&dir)?;
    rotate_logs(&dir)?;

    let user = truncate_field(user);
    let assistant = truncate_field(assistant);
    let correction = correction.map(truncate_field);
    let style_notes = extract_style_features(&user, correction.as_deref());

    let entry = UserLogEntry {
        ts_unix: now_unix(),
        model: model.to_string(),
        user,
        assistant,
        correction,
        style_notes,
    };
    let line = serde_json::to_string(&entry)?;
    let path = dir.join(LOG_FILE);
    let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

/// Rotate / truncate oversized logs (keep newest half by file rewrite).
pub fn rotate_logs(dir: &Path) -> Result<()> {
    let path = dir.join(LOG_FILE);
    if !path.exists() {
        return Ok(());
    }
    let meta = fs::metadata(&path)?;
    if meta.len() <= MAX_LOG_BYTES {
        return Ok(());
    }
    let text = fs::read_to_string(&path)?;
    let lines: Vec<&str> = text.lines().collect();
    let keep = lines.len() / 2;
    let kept = lines[lines.len().saturating_sub(keep)..].join("\n");
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, format!("{kept}\n"))?;
    fs::rename(tmp, path)?;
    Ok(())
}

/// Build a plain-text / JSONL corpus for Phase 4 `train_adapter` from logs.
pub fn build_training_corpus(store: &LocalStore, out_path: impl AsRef<Path>) -> Result<usize> {
    let dir = store.user_logs_dir();
    let path = dir.join(LOG_FILE);
    if !path.exists() {
        return Err(AppError::msg(
            "no user logs yet — chat with `lpc-llm run` first (logs under cache/user_logs/)",
        ));
    }
    let text = fs::read_to_string(&path)?;
    let mut samples = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: UserLogEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        // Prefer corrections; otherwise user+assistant as SFT-style text.
        if let Some(corr) = entry.correction {
            samples.push(format!("User: {}\nAssistant: {}", entry.user, corr));
        } else if !entry.user.is_empty() && !entry.assistant.is_empty() {
            samples.push(format!(
                "User: {}\nAssistant: {}",
                entry.user, entry.assistant
            ));
        }
        // Style notes as soft prompts.
        if !entry.style_notes.is_empty() {
            samples.push(format!(
                "Style preferences: {}\n",
                entry.style_notes.join("; ")
            ));
        }
    }
    if samples.is_empty() {
        return Err(AppError::msg("user logs contain no usable training samples"));
    }
    let out_path = out_path.as_ref();
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::new();
    for s in &samples {
        body.push_str(s);
        if !s.ends_with('\n') {
            body.push('\n');
        }
        body.push('\n');
    }
    fs::write(out_path, body)?;
    Ok(samples.len())
}

fn truncate_field(s: &str) -> String {
    if s.chars().count() <= MAX_FIELD_CHARS {
        s.to_string()
    } else {
        s.chars().take(MAX_FIELD_CHARS).collect()
    }
}

#[allow(dead_code)]
pub fn log_path(store: &LocalStore) -> PathBuf {
    store.user_logs_dir().join(LOG_FILE)
}
