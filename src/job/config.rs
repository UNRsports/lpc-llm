//! Declarative training / conversion job configs (JSON).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};

pub const JOB_FORMAT: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub version: u32,
    pub name: String,
    #[serde(default)]
    pub work_dir: Option<PathBuf>,
    #[serde(default)]
    pub remote: Option<RemoteSpec>,
    pub stages: Vec<JobStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteSpec {
    /// Shell command to launch an external/distributed job (env: LPC_LLM_JOB_*).
    pub launch: String,
    /// Glob or path hint where artifacts appear after remote completion.
    #[serde(default)]
    pub artifact_glob: Option<String>,
    /// Optional poll command that exits 0 when the remote job is done.
    #[serde(default)]
    pub status_cmd: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobStage {
    /// Tiny from-scratch (Phase 5).
    Scratch {
        from: PathBuf,
        out: String,
        #[serde(default = "default_steps")]
        steps: usize,
        #[serde(default = "default_embd")]
        n_embd: usize,
        #[serde(default = "default_layers")]
        n_layers: usize,
        #[serde(default = "default_ram")]
        ram_mib: usize,
    },
    /// Full SFT on a tiny checkpoint.
    Sft {
        base_ckpt: PathBuf,
        from: PathBuf,
        out: String,
        #[serde(default = "default_steps")]
        steps: usize,
        #[serde(default = "default_ram")]
        ram_mib: usize,
    },
    /// Preference opt (DPO).
    Dpo {
        base_ckpt: PathBuf,
        from: PathBuf,
        out: String,
        #[serde(default = "default_steps")]
        steps: usize,
        #[serde(default = "default_beta")]
        beta: f64,
        #[serde(default = "default_ram")]
        ram_mib: usize,
    },
    /// LoRA SFT via Phase 4 trainer (catalog base).
    LoraSft {
        base: String,
        from: PathBuf,
        out: String,
        #[serde(default = "default_steps")]
        steps: usize,
        #[serde(default = "default_rank")]
        rank: usize,
        #[serde(default = "default_ram")]
        ram_mib: usize,
    },
    /// Export checkpoint → GGUF and register.
    ExportGguf {
        ckpt: PathBuf,
        name: String,
    },
    /// Import an existing GGUF (+ tokenizer) into blobs/manifest.
    ImportGguf {
        gguf: PathBuf,
        tokenizer: PathBuf,
        name: String,
    },
    /// Large/external conversion bridge (see `convert`).
    Convert {
        from_dir: PathBuf,
        name: String,
        #[serde(default = "default_backend")]
        backend: String,
    },
    /// RLHF stage marker / stub (PPO etc. — external accelerator).
    RlhfStage {
        kind: String,
        #[serde(default)]
        base: Option<String>,
        #[serde(default)]
        from: Option<PathBuf>,
        #[serde(default)]
        out: Option<String>,
        #[serde(default)]
        note: Option<String>,
    },
}

fn default_steps() -> usize {
    32
}
fn default_embd() -> usize {
    128
}
fn default_layers() -> usize {
    2
}
fn default_ram() -> usize {
    1024
}
fn default_beta() -> f64 {
    0.1
}
fn default_rank() -> usize {
    8
}
fn default_backend() -> String {
    "builtin".into()
}

impl JobConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref()).map_err(|e| {
            AppError::msg(format!("read job config {}: {e}", path.as_ref().display()))
        })?;
        let cfg: Self = serde_json::from_str(&text)?;
        if cfg.version != JOB_FORMAT {
            return Err(AppError::msg(format!(
                "unsupported job version {} (expected {JOB_FORMAT})",
                cfg.version
            )));
        }
        if cfg.stages.is_empty() {
            return Err(AppError::msg("job has no stages"));
        }
        Ok(cfg)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(p) = path.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}
