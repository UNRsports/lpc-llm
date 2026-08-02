//! Path / install / UI / runtime layout from `config_lpcllm`.
//!
//! Load order (later wins):
//! 1. Built-in defaults (XDG user data + user bin)
//! 2. System file `/etc/lpc-llm/config_lpcllm` (shared binary hints only when present)
//! 3. User file `$XDG_CONFIG_HOME/lpc-llm/config_lpcllm`
//! 4. Explicit `$LPC_LLM_CONFIG` (replaces 2–3 when set; still layered on defaults)
//! 5. Env overrides: `LPC_LLM_DATA_DIR`, `LPC_LLM_TRAIN_DIR`, `LPC_LLM_BIN_DIR`,
//!    `LPC_LLM_LANGUAGE`, `LPC_LLM_DEVICE`
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

/// UI language for interactive prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum UiLanguage {
    #[default]
    En,
    Ja,
}

impl UiLanguage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ja => "ja",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "en" | "english" => Some(Self::En),
            "ja" | "jp" | "japanese" | "日本語" => Some(Self::Ja),
            _ => None,
        }
    }
}

/// Preferred compute backend (Phase 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevicePref {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Vulkan,
}

impl ComputeDevicePref {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "vulkan" | "vk" => Some(Self::Vulkan),
            _ => None,
        }
    }
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
pub struct UiSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<UiLanguage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeSection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<ComputeDevicePref>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub paths: PathsSection,
    #[serde(default)]
    pub install: InstallSection,
    #[serde(default)]
    pub ui: UiSection,
    #[serde(default)]
    pub runtime: RuntimeSection,
}

/// Resolved runtime configuration (absolute paths).
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub train_dir: PathBuf,
    pub bin_dir: PathBuf,
    pub install_mode: InstallMode,
    pub ui_language: UiLanguage,
    pub compute_device: ComputeDevicePref,
    /// True when `[runtime].device` was set in a config file (before env).
    pub runtime_device_configured: bool,
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
            r#"# config_lpcllm — lpc-llm path, UI, and compute layout
#
# Privacy:
#   - Private training corpora and runtime state belong under paths below
#     (user home / XDG). Do not put private data in the git repository.
#   - Repo `data/train/` and `examples/` are for public/dev samples only.
#   - Shared install copies the binary only; each user keeps their own data_dir.
#
# Search order: /etc/lpc-llm/config_lpcllm → ~/.config/lpc-llm/config_lpcllm
#               → $LPC_LLM_CONFIG (exclusive overlay on defaults when set)
# Env overrides: LPC_LLM_DATA_DIR, LPC_LLM_TRAIN_DIR, LPC_LLM_BIN_DIR,
#                LPC_LLM_LANGUAGE, LPC_LLM_DEVICE
# Interactive:   lpc-llm setup   /   lpc-llm config init --interactive

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

[ui]
# Interactive UI language: "en" | "ja"
# language = "en"

[runtime]
# Compute backend: "auto" | "cpu" | "cuda" | "vulkan"
# auto → Vulkan if available, else CUDA (feature-built), else CPU
# device = "auto"
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

    /// Merge `patch` into the existing user config (or defaults) and write it back.
    pub fn save_user_merged(patch: &ConfigFile) -> Result<PathBuf> {
        let path = user_config_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = if path.is_file() {
            let text = fs::read_to_string(&path)?;
            toml::from_str(&text).map_err(|e| {
                AppError::msg(format!("invalid config {}: {e}", path.display()))
            })?
        } else {
            ConfigFile::default()
        };
        merge_overlay(&mut file, patch);
        let text = toml::to_string_pretty(&file).map_err(|e| {
            AppError::msg(format!("serialize config: {e}"))
        })?;
        let header = "# Written by lpc-llm (setup / config). Edit freely.\n\
                      # See `lpc-llm config example` for documented defaults.\n\n";
        fs::write(&path, format!("{header}{text}"))?;
        Ok(path)
    }

    /// True when the user should be offered first-run setup.
    pub fn needs_setup() -> Result<bool> {
        let user = user_config_path()?;
        if !user.is_file() {
            return Ok(true);
        }
        let text = fs::read_to_string(&user)?;
        let file: ConfigFile = toml::from_str(&text).map_err(|e| {
            AppError::msg(format!("invalid config {}: {e}", user.display()))
        })?;
        Ok(file.runtime.device.is_none())
    }
}

fn resolve(file: ConfigFile, loaded_from: Vec<PathBuf>) -> Result<AppConfig> {
    let runtime_device_configured = file.runtime.device.is_some();
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

    let ui_language = env::var("LPC_LLM_LANGUAGE")
        .ok()
        .and_then(|s| UiLanguage::parse(&s))
        .or(file.ui.language)
        .unwrap_or_else(detect_locale_from_env);

    let compute_device = env::var("LPC_LLM_DEVICE")
        .ok()
        .and_then(|s| ComputeDevicePref::parse(&s))
        .or(file.runtime.device)
        .unwrap_or(ComputeDevicePref::Auto);

    Ok(AppConfig {
        data_dir,
        train_dir,
        bin_dir,
        install_mode,
        ui_language,
        compute_device,
        runtime_device_configured,
        loaded_from,
    })
}

fn detect_locale_from_env() -> UiLanguage {
    for key in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(v) = env::var(key) {
            let lower = v.to_ascii_lowercase();
            if lower.starts_with("ja") {
                return UiLanguage::Ja;
            }
        }
    }
    UiLanguage::En
}

fn merge_file(into: &mut ConfigFile, path: &Path) -> Result<()> {
    let text = fs::read_to_string(path)?;
    let overlay: ConfigFile = toml::from_str(&text).map_err(|e| {
        AppError::msg(format!(
            "invalid config {}: {e}",
            path.display()
        ))
    })?;
    merge_overlay(into, &overlay);
    Ok(())
}

fn merge_overlay(into: &mut ConfigFile, overlay: &ConfigFile) {
    if overlay.paths.data_dir.is_some() {
        into.paths.data_dir = overlay.paths.data_dir.clone();
    }
    if overlay.paths.train_dir.is_some() {
        into.paths.train_dir = overlay.paths.train_dir.clone();
    }
    if overlay.install.mode.is_some() {
        into.install.mode = overlay.install.mode;
    }
    if overlay.install.bin_dir.is_some() {
        into.install.bin_dir = overlay.install.bin_dir.clone();
    }
    if overlay.ui.language.is_some() {
        into.ui.language = overlay.ui.language;
    }
    if overlay.runtime.device.is_some() {
        into.runtime.device = overlay.runtime.device;
    }
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

    #[test]
    fn parse_device_pref() {
        assert_eq!(ComputeDevicePref::parse("vulkan"), Some(ComputeDevicePref::Vulkan));
        assert_eq!(UiLanguage::parse("ja"), Some(UiLanguage::Ja));
    }
}
