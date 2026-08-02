//! RLHF pipeline stage definitions (SFT → preference → PPO stub).

use serde::{Deserialize, Serialize};

use super::config::{JobConfig, JobStage, JOB_FORMAT};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RlhfStageKind {
    Sft,
    RewardOrPreference,
    Ppo,
    Eval,
    Emit,
}

impl RlhfStageKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::RewardOrPreference => "reward_or_preference",
            Self::Ppo => "ppo",
            Self::Eval => "eval",
            Self::Emit => "emit",
        }
    }
}

/// Default full RLHF stage list as a declarative job (local DPO + remote PPO stub).
pub fn default_rlhf_pipeline(name: &str, corpus: &str, prefs: &str) -> JobConfig {
    JobConfig {
        version: JOB_FORMAT,
        name: name.to_string(),
        work_dir: Some(std::path::PathBuf::from("cache/jobs").join(name.replace([':', '/'], "_"))),
        remote: None,
        stages: vec![
            JobStage::Scratch {
                from: corpus.into(),
                out: format!("{name}-base"),
                steps: 64,
                n_embd: 128,
                n_layers: 2,
                ram_mib: 1024,
            },
            JobStage::Sft {
                base_ckpt: std::path::PathBuf::from("PLACEHOLDER"), // filled by runner relative paths
                from: corpus.into(),
                out: format!("{name}-sft"),
                steps: 32,
                ram_mib: 1024,
            },
            JobStage::RlhfStage {
                kind: RlhfStageKind::RewardOrPreference.as_str().into(),
                base: Some(format!("{name}-sft")),
                from: Some(prefs.into()),
                out: Some(format!("{name}-pref")),
                note: Some("local DPO stands in for reward-model training".into()),
            },
            JobStage::Dpo {
                base_ckpt: std::path::PathBuf::from("PLACEHOLDER"),
                from: prefs.into(),
                out: format!("{name}-dpo"),
                steps: 32,
                beta: 0.1,
                ram_mib: 1024,
            },
            JobStage::RlhfStage {
                kind: RlhfStageKind::Ppo.as_str().into(),
                base: Some(format!("{name}-dpo")),
                from: None,
                out: Some(format!("{name}-ppo")),
                note: Some(
                    "PPO / full RLHF requires external accelerator; \
                     use job remote.launch or LPC_LLM_CONVERT_CMD / cluster scripts"
                        .into(),
                ),
            },
            JobStage::RlhfStage {
                kind: RlhfStageKind::Eval.as_str().into(),
                base: Some(format!("{name}-dpo")),
                from: None,
                out: None,
                note: Some("run `lpc-llm run <name>` smoke + optional regression".into()),
            },
            JobStage::ExportGguf {
                ckpt: std::path::PathBuf::from("PLACEHOLDER"),
                name: format!("{name}:rlhf"),
            },
            JobStage::RlhfStage {
                kind: RlhfStageKind::Emit.as_str().into(),
                base: Some(format!("{name}:rlhf")),
                from: None,
                out: Some(format!("{name}:rlhf")),
                note: Some("weights land in blobs/; LoRA deltas may go to adapters/".into()),
            },
        ],
    }
}
