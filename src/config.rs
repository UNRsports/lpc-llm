//! Path / install layout from `config_lpcllm`.
//!
//! Load order (later wins):
//! 1. Built-in defaults (XDG user data + user bin)
//! 2. System file `/etc/lpc-llm/config_lpcllm` (shared binary hints only when present)
//! 3. User file `$XDG_CONFIG_HOME/lpc-llm/config_lpcllm`
//! 4. Explicit `$LPC_LLM_CONFIG` (replaces 2–3 when set; still layered on defaults)
//! 5. Env overrides: `LPC_LLM_DATA_DIR`, `LPC_LLM_TRAIN_DIR`, `LPC_LLM_BIN_DIR`
//!
//! Privacy rule: private corpora and runtime state live under the user data tree
//! (or an explicit `train_dir`). The git repo must not hold private datasets.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};

/// File name used under XDG config and `/etc/lpc-llm/`.
pub const CONFIG_FILE_NAME: &str = "config_lpcllm";

/// System-wide config directory (binary install hints; no user corpora).
pub const SYSTEM_CONFIG_DIR: &str = "/etc/lpc-llm";

/// Default system binary directory when `install.mode = "system"`.
pub const DEFAULT_SYSTEM_BIN_DIR: &str = "/usr/local/bin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InstallMode {
    /// Per-user PATH entry (default: `~/.local/bin`).
    #[default]
    User,
    /// Shared machine binary only (default: `/usr/local/bin`). Data stays per-user.
    System,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathsSection {
    /// Root for blobs / adapters / cache / manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
    /// Private training corpora (default: `<data_dir>/train`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_dir: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstallSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<InstallMode>,
    /// Directory that should contain the `lpc-llm` binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_dir: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub paths: PathsSection,
    #[serde(default)]
    pub install: InstallSection,
}

/// Resolved runtime configuration (absolute paths).
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub train_dir: PathBuf,
    pub bin_dir: PathBuf,
    pub install_mode: InstallMode,
    /// Config files that contributed settings (for `config show`).
    pub loaded_from: Vec<PathBuf>,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let mut file = ConfigFile::default();
        let mut loaded_from = Vec::new();

        if let Ok(explicit) = env::var("LPC_LLM_CONFIG") {
            let path = expand_path(&explicit);
            if path.is_file() {
                merge_file(&mut file, &path)?;
                loaded_from.push(path);
            } else {
                return Err(AppError::msg(format!(
                    "LPC_LLM_CONFIG points to missing file: {}",
                    path.display()
                )));
            }
        } else {
            let system_path = PathBuf::from(SYSTEM_CONFIG_DIR).join(CONFIG_FILE_NAME);
            if system_path.is_file() {
                merge_file(&mut file, &system_path)?;
                loaded_from.push(system_path);
            }
            let user_path = user_config_path()?;
            if user_path.is_file() {
                merge_file(&mut file, &user_path)?;
                loaded_from.push(user_path);
            }
        }

        resolve(file, loaded_from)
    }

    /// Default TOML text for `config init` / example file.
    pub fn default_toml() -> String {
        format!(
            r#"# config_lpcllm — lpc-llm path and install layout
#
# Privacy:
#   - Private training corpora and runtime state belong under paths below
#     (user home / XDG). Do not put private data in the git repository.
#   - Repo `data/train/` and `examples/` are for public/dev samples only.
#   - Shared install copies the binary only; each user keeps their own data_dir.
#
# Search order: /etc/lpc-llm/config_lpcllm → ~/.config/lpc-llm/config_lpcllm
#               → $LPC_LLM_CONFIG (exclusive overlay on defaults when set)
# Env overrides: LPC_LLM_DATA_DIR, LPC_LLM_TRAIN_DIR, LPC_LLM_BIN_DIR

[paths]
# User data root (blobs, adapters, cache, manifest).
# Default: $XDG_DATA_HOME/lpc-llm  (~/.local/share/lpc-llm)
# data_dir = "~/.local/share/lpc-llm"

# Private training corpora (.txt / .jsonl).
# Default: <data_dir>/train
# train_dir = "~/.local/share/lpc-llm/train"
# Alternative:
# train_dir = "~/Documents/lpc-llm/train"

[install]
# user   → per-user binary (default bin_dir ~/.local/bin)
# system → shared binary only (default bin_dir /usr/local/bin); data stays per-user
mode = "user"
# bin_dir = "~/.local/bin"
# For system-wide binary install:
# mode = "system"
# bin_dir = "{DEFAULT_SYSTEM_BIN_DIR}"
"#
        )
    }

    pub fn write_user_default(force: bool) -> Result<PathBuf> {
        let path = user_config_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if path.exists() && !force {
            return Err(AppError::msg(format!(
                "config already exists: {} (pass --force to overwrite)",
                path.display()
            )));
        }
        fs::write(&path, Self::default_toml())?;
        Ok(path)
    }
}

fn resolve(file: ConfigFile, loaded_from: Vec<PathBuf>) -> Result<AppConfig> {
    let default_data = default_data_dir()?;
    let data_dir = env::var("LPC_LLM_DATA_DIR")
        .ok()
        .map(|s| expand_path(&s))
        .or_else(|| file.paths.data_dir.as_ref().map(|s| expand_path(s)))
        .unwrap_or(default_data);

    let train_dir = env::var("LPC_LLM_TRAIN_DIR")
        .ok()
        .map(|s| expand_path(&s))
        .or_else(|| file.paths.train_dir.as_ref().map(|s| expand_path(s)))
        .unwrap_or_else(|| data_dir.join("train"));

    let install_mode = file.install.mode.unwrap_or_default();
    let default_bin = match install_mode {
        InstallMode::User => default_user_bin_dir()?,
        InstallMode::System => PathBuf::from(DEFAULT_SYSTEM_BIN_DIR),
    };
    let bin_dir = env::var("LPC_LLM_BIN_DIR")
        .ok()
        .map(|s| expand_path(&s))
        .or_else(|| file.install.bin_dir.as_ref().map(|s| expand_path(s)))
        .unwrap_or(default_bin);

    Ok(AppConfig {
        data_dir,
        train_dir,
        bin_dir,
        install_mode,
        loaded_from,
    })
}

fn merge_file(into: &mut ConfigFile, path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let overlay: ConfigFile = toml::from_str(&text).map_err(|e| {
        AppError::msg(format!(
            "invalid config {}: {e}",
            path.display()
        ))
    })?;
    if overlay.paths.data_dir.is_some() {
        into.paths.data_dir = overlay.paths.data_dir;
    }
    if overlay.paths.train_dir.is_some() {
        into.paths.train_dir = overlay.paths.train_dir;
    }
    if overlay.install.mode.is_some() {
        into.install.mode = overlay.install.mode;
    }
    if overlay.install.bin_dir.is_some() {
        into.install.bin_dir = overlay.install.bin_dir;
    }
    Ok(())
}

pub fn user_config_path() -> Result<PathBuf> {
    let base = dirs::config_dir().ok_or_else(|| {
        AppError::msg("could not resolve XDG config directory (~/.config)")
    })?;
    Ok(base.join("lpc-llm").join(CONFIG_FILE_NAME))
}

pub fn system_config_path() -> PathBuf {
    PathBuf::from(SYSTEM_CONFIG_DIR).join(CONFIG_FILE_NAME)
}

pub fn default_data_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir().ok_or_else(|| {
        AppError::msg("could not resolve XDG data directory (~/.local/share)")
    })?;
    Ok(base.join("lpc-llm"))
}

pub fn default_user_bin_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| AppError::msg("could not resolve home directory"))?;
    Ok(home.join(".local").join("bin"))
}

/// Expand a leading `~/` (or bare `~`) using the user home directory.
pub fn expand_path(raw: &str) -> PathBuf {
    let raw = raw.trim();
    if raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(raw)
}

/// Resolve a `--from` training path.
///
/// Order: expanded path as given (cwd-relative or absolute) if it is a file;
/// otherwise `<train_dir>/<spec>` when that is a file.
pub fn resolve_train_from(spec: &str, train_dir: &Path) -> Result<PathBuf> {
    if spec.trim().is_empty() {
        return Err(AppError::msg("--from path must be non-empty"));
    }
    let expanded = expand_path(spec);
    if expanded.is_file() {
        return Ok(canonicalize_or_abs(&expanded));
    }
    if !expanded.is_absolute() {
        let under_train = train_dir.join(&expanded);
        if under_train.is_file() {
            return Ok(canonicalize_or_abs(&under_train));
        }
    }
    Err(AppError::msg(format!(
        "--from file not found: {spec}\n  looked for: {}\n  and under train_dir: {}\n  (private corpora belong in train_dir; see `lpc-llm config show`)",
        expanded.display(),
        train_dir.join(spec).display()
    )))
}

fn canonicalize_or_abs(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_home_prefix() {
        let p = expand_path("~/foo/bar");
        assert!(p.ends_with("foo/bar"));
        assert!(!p.to_string_lossy().contains('~'));
    }

    #[test]
    fn default_train_under_data() {
        let file = ConfigFile::default();
        let cfg = resolve(file, Vec::new()).expect("resolve");
        assert_eq!(cfg.train_dir, cfg.data_dir.join("train"));
    }
}
