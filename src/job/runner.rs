//! Execute declarative jobs locally or via remote launch bridge.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use console::style;
use serde::{Deserialize, Serialize};

use super::config::{JobConfig, JobStage, RemoteSpec, JOB_FORMAT};
use super::convert::{convert_checkpoint_to_gguf, import_gguf};
use super::rlhf::default_rlhf_pipeline;
use crate::adapter::{train_adapter, TrainConfig};
use crate::catalog;
use crate::error::{AppError, Result};
use crate::pull;
use crate::store::{now_unix, LocalStore};
use crate::train::{
    export_and_register, train_dpo, train_scratch, train_sft_full, DpoConfig, ScratchConfig,
    SftConfig,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusFile {
    pub name: String,
    pub state: String,
    pub stage_index: usize,
    pub stage_total: usize,
    pub updated_at_unix: u64,
    #[serde(default)]
    pub message: Option<String>,
}

pub fn write_template(kind: &str, out: impl AsRef<Path>) -> Result<()> {
    let cfg = match kind {
        "scratch" => JobConfig {
            version: JOB_FORMAT,
            name: "scratch-demo".into(),
            work_dir: Some(PathBuf::from("cache/jobs/scratch-demo")),
            remote: None,
            stages: vec![JobStage::Scratch {
                from: PathBuf::from("examples/train-sample.txt"),
                out: "tiny:demo".into(),
                steps: 32,
                n_embd: 128,
                n_layers: 2,
                ram_mib: 1024,
            }],
        },
        "sft" => JobConfig {
            version: JOB_FORMAT,
            name: "sft-demo".into(),
            work_dir: Some(PathBuf::from("cache/jobs/sft-demo")),
            remote: None,
            stages: vec![
                JobStage::Scratch {
                    from: PathBuf::from("examples/train-sample.txt"),
                    out: "tiny:base".into(),
                    steps: 32,
                    n_embd: 128,
                    n_layers: 2,
                    ram_mib: 1024,
                },
                JobStage::Sft {
                    base_ckpt: PathBuf::from("@ckpt:tiny:base"),
                    from: PathBuf::from("examples/train-sample.txt"),
                    out: "tiny:sft".into(),
                    steps: 16,
                    ram_mib: 1024,
                },
            ],
        },
        "rlhf" => {
            let mut cfg = default_rlhf_pipeline(
                "rlhf-demo",
                "examples/train-sample.txt",
                "examples/pref-sample.jsonl",
            );
            // Resolve stage paths to @ckpt references for the runner.
            cfg.stages = vec![
                JobStage::Scratch {
                    from: PathBuf::from("examples/train-sample.txt"),
                    out: "rlhf-demo-base".into(),
                    steps: 32,
                    n_embd: 128,
                    n_layers: 2,
                    ram_mib: 1024,
                },
                JobStage::Sft {
                    base_ckpt: PathBuf::from("@ckpt:rlhf-demo-base"),
                    from: PathBuf::from("examples/train-sample.txt"),
                    out: "rlhf-demo-sft".into(),
                    steps: 16,
                    ram_mib: 1024,
                },
                JobStage::Dpo {
                    base_ckpt: PathBuf::from("@ckpt:rlhf-demo-sft"),
                    from: PathBuf::from("examples/pref-sample.jsonl"),
                    out: "rlhf-demo-dpo".into(),
                    steps: 16,
                    beta: 0.1,
                    ram_mib: 1024,
                },
                JobStage::RlhfStage {
                    kind: "ppo".into(),
                    base: Some("rlhf-demo-dpo".into()),
                    from: None,
                    out: Some("rlhf-demo-ppo".into()),
                    note: Some(
                        "stub: launch remote PPO via job.remote or skip for local DPO endpoint"
                            .into(),
                    ),
                },
                JobStage::ExportGguf {
                    ckpt: PathBuf::from("@ckpt:rlhf-demo-dpo"),
                    name: "rlhf-demo:dpo".into(),
                },
            ];
            cfg
        },
        "remote" => JobConfig {
            version: JOB_FORMAT,
            name: "remote-bridge".into(),
            work_dir: Some(PathBuf::from("cache/jobs/remote-bridge")),
            remote: Some(RemoteSpec {
                launch: "echo \"remote train placeholder; replace with ssh/sbatch\"".into(),
                artifact_glob: Some("$LPC_LLM_JOB_WORK/artifacts/*.gguf".into()),
                status_cmd: Some("true".into()),
            }),
            stages: vec![JobStage::ImportGguf {
                gguf: PathBuf::from("@remote:model.gguf"),
                tokenizer: PathBuf::from("@remote:tokenizer.json"),
                name: "remote:import".into(),
            }],
        },
        other => {
            return Err(AppError::msg(format!(
                "unknown job template `{other}` (scratch|sft|rlhf|remote)"
            )));
        }
    };
    cfg.save(out.as_ref())?;
    eprintln!(
        "{} wrote job template `{kind}` → {}",
        style("✓").green(),
        out.as_ref().display()
    );
    Ok(())
}

pub fn run_job(store: &LocalStore, config_path: impl AsRef<Path>, local_only: bool) -> Result<()> {
    let config_path = config_path.as_ref();
    let cfg = JobConfig::load(config_path)?;
    let work = resolve_work_dir(store, &cfg)?;
    fs::create_dir_all(&work)?;
    write_status(
        &work,
        &cfg.name,
        "running",
        0,
        cfg.stages.len(),
        Some("starting"),
    )?;

    if let Some(remote) = &cfg.remote {
        if !local_only {
            launch_remote(remote, &cfg, &work)?;
        } else {
            eprintln!(
                "{} --local: skipping remote.launch",
                style("·").dim()
            );
        }
    }

    for (i, stage) in cfg.stages.iter().enumerate() {
        write_status(
            &work,
            &cfg.name,
            "running",
            i,
            cfg.stages.len(),
            Some(&stage_label(stage)),
        )?;
        eprintln!(
            "{} job `{}` stage {}/{}: {}",
            style("▸").cyan(),
            cfg.name,
            i + 1,
            cfg.stages.len(),
            stage_label(stage)
        );
        run_stage(store, &work, stage)?;
    }

    write_status(
        &work,
        &cfg.name,
        "done",
        cfg.stages.len(),
        cfg.stages.len(),
        Some("completed"),
    )?;
    eprintln!(
        "{} job `{}` completed ({})",
        style("✓").green(),
        style(&cfg.name).bold(),
        work.display()
    );
    Ok(())
}

pub fn job_status(store: &LocalStore, name_or_path: &str) -> Result<()> {
    let path = if Path::new(name_or_path).is_file() {
        PathBuf::from(name_or_path)
    } else {
        store
            .cache_dir()
            .join("jobs")
            .join(name_or_path.replace([':', '/'], "_"))
            .join("status.json")
    };
    if !path.is_file() {
        return Err(AppError::msg(format!(
            "no job status at {}",
            path.display()
        )));
    }
    let st: JobStatusFile = serde_json::from_str(&fs::read_to_string(&path)?)?;
    println!(
        "job={} state={} stage={}/{} updated={} {}",
        st.name,
        st.state,
        st.stage_index,
        st.stage_total,
        st.updated_at_unix,
        st.message.unwrap_or_default()
    );
    Ok(())
}

fn resolve_work_dir(store: &LocalStore, cfg: &JobConfig) -> Result<PathBuf> {
    match &cfg.work_dir {
        Some(p) if p.is_absolute() => Ok(p.clone()),
        Some(p) => Ok(store.root().join(p)),
        None => Ok(store.cache_dir().join("jobs").join(cfg.name.replace([':', '/'], "_"))),
    }
}

fn ckpt_dir(work: &Path, name: &str) -> PathBuf {
    work.join("ckpts").join(name.replace([':', '/'], "_"))
}

fn resolve_ckpt_ref(work: &Path, path: &Path) -> Result<PathBuf> {
    let s = path.to_string_lossy();
    if let Some(name) = s.strip_prefix("@ckpt:") {
        let dir = ckpt_dir(work, name);
        if !dir.join(crate::train::checkpoint::CONFIG_FILE).is_file() {
            return Err(AppError::msg(format!(
                "checkpoint ref `{s}` not found at {}",
                dir.display()
            )));
        }
        return Ok(dir);
    }
    Ok(path.to_path_buf())
}

fn run_stage(store: &LocalStore, work: &Path, stage: &JobStage) -> Result<()> {
    match stage {
        JobStage::Scratch {
            from,
            out,
            steps,
            n_embd,
            n_layers,
            ram_mib,
        } => {
            let out_dir = ckpt_dir(work, out);
            train_scratch(
                store,
                from,
                &out_dir,
                ScratchConfig {
                    name: out.clone(),
                    steps: *steps,
                    n_embd: *n_embd,
                    n_layers: *n_layers,
                    n_heads: 4,
                    n_kv_heads: 4,
                    n_ff: n_embd * 4,
                    ram_mib: *ram_mib,
                    register: true,
                    ..ScratchConfig::default()
                },
            )?;
        }
        JobStage::Sft {
            base_ckpt,
            from,
            out,
            steps,
            ram_mib,
        } => {
            let base = resolve_ckpt_ref(work, base_ckpt)?;
            let out_dir = ckpt_dir(work, out);
            train_sft_full(
                store,
                &base,
                from,
                &out_dir,
                SftConfig {
                    name: out.clone(),
                    steps: *steps,
                    ram_mib: *ram_mib,
                    register: true,
                    ..SftConfig::default()
                },
            )?;
        }
        JobStage::Dpo {
            base_ckpt,
            from,
            out,
            steps,
            beta,
            ram_mib,
        } => {
            let base = resolve_ckpt_ref(work, base_ckpt)?;
            let out_dir = ckpt_dir(work, out);
            train_dpo(
                store,
                &base,
                from,
                &out_dir,
                DpoConfig {
                    name: out.clone(),
                    steps: *steps,
                    beta: *beta,
                    ram_mib: *ram_mib,
                    register: true,
                    ..DpoConfig::default()
                },
            )?;
        }
        JobStage::LoraSft {
            base,
            from,
            out,
            steps,
            rank,
            ram_mib,
        } => {
            let entry = catalog::find(base).ok_or_else(|| AppError::UnknownModel(base.clone()))?;
            let installed = match store.resolve(&entry)? {
                Some(m) => m,
                None => pull::pull_model(store, &entry)?,
            };
            let out_dir = store.adapter_path(out);
            train_adapter(
                &installed.model_path,
                &installed.tokenizer_path,
                store.pack_cache_dir(base),
                from,
                &out_dir,
                TrainConfig {
                    name: out.clone(),
                    base_model: base.clone(),
                    rank: *rank,
                    steps: *steps,
                    ram_mib: *ram_mib,
                    ..TrainConfig::default()
                },
            )?;
            store.record_adapter(crate::store::InstalledAdapter {
                name: out.clone(),
                path: out_dir,
                base_model: base.clone(),
                recorded_at_unix: now_unix(),
            })?;
        }
        JobStage::ExportGguf { ckpt, name } => {
            let dir = resolve_ckpt_ref(work, ckpt)?;
            export_and_register(store, &dir, name)?;
        }
        JobStage::ImportGguf {
            gguf,
            tokenizer,
            name,
        } => {
            let g = resolve_maybe_remote(work, gguf)?;
            let t = resolve_maybe_remote(work, tokenizer)?;
            import_gguf(store, g, t, name)?;
        }
        JobStage::Convert {
            from_dir,
            name,
            backend,
        } => {
            convert_checkpoint_to_gguf(store, from_dir, name, backend)?;
        }
        JobStage::RlhfStage {
            kind,
            note,
            ..
        } => {
            eprintln!(
                "  {} RLHF stage `{kind}` (marker){}",
                style("·").dim(),
                note.as_ref()
                    .map(|n| format!(" — {n}"))
                    .unwrap_or_default()
            );
            if kind == "ppo" {
                eprintln!(
                    "  {} PPO not executed locally; use remote.launch / external accelerator",
                    style("!").yellow()
                );
            }
        }
    }
    Ok(())
}

fn resolve_maybe_remote(work: &Path, path: &Path) -> Result<PathBuf> {
    let s = path.to_string_lossy();
    if let Some(rel) = s.strip_prefix("@remote:") {
        let p = work.join("artifacts").join(rel);
        if !p.is_file() {
            return Err(AppError::msg(format!(
                "remote artifact missing: {} (place files under {}/artifacts/)",
                p.display(),
                work.display()
            )));
        }
        return Ok(p);
    }
    Ok(path.to_path_buf())
}

fn launch_remote(remote: &RemoteSpec, cfg: &JobConfig, work: &Path) -> Result<()> {
    eprintln!(
        "{} remote launch: {}",
        style("↗").magenta(),
        remote.launch
    );
    let status = Command::new("sh")
        .arg("-c")
        .arg(&remote.launch)
        .env("LPC_LLM_JOB_NAME", &cfg.name)
        .env("LPC_LLM_JOB_WORK", work.as_os_str())
        .status()
        .map_err(|e| AppError::msg(format!("remote launch failed: {e}")))?;
    if !status.success() {
        return Err(AppError::msg(format!(
            "remote launch exited with {status}"
        )));
    }
    if let Some(cmd) = &remote.status_cmd {
        let st = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .env("LPC_LLM_JOB_NAME", &cfg.name)
            .env("LPC_LLM_JOB_WORK", work.as_os_str())
            .status()
            .map_err(|e| AppError::msg(format!("remote status_cmd failed: {e}")))?;
        if !st.success() {
            return Err(AppError::msg(format!(
                "remote status_cmd exited with {st}"
            )));
        }
    }
    if let Some(glob_hint) = &remote.artifact_glob {
        eprintln!(
            "  {} artifact hint: {glob_hint}",
            style("·").dim()
        );
    }
    Ok(())
}

fn write_status(
    work: &Path,
    name: &str,
    state: &str,
    stage_index: usize,
    stage_total: usize,
    message: Option<&str>,
) -> Result<()> {
    let st = JobStatusFile {
        name: name.to_string(),
        state: state.into(),
        stage_index,
        stage_total,
        updated_at_unix: now_unix(),
        message: message.map(ToOwned::to_owned),
    };
    fs::write(work.join("status.json"), serde_json::to_string_pretty(&st)?)?;
    Ok(())
}

fn stage_label(stage: &JobStage) -> String {
    match stage {
        JobStage::Scratch { out, .. } => format!("scratch → {out}"),
        JobStage::Sft { out, .. } => format!("sft → {out}"),
        JobStage::Dpo { out, .. } => format!("dpo → {out}"),
        JobStage::LoraSft { out, .. } => format!("lora_sft → {out}"),
        JobStage::ExportGguf { name, .. } => format!("export_gguf → {name}"),
        JobStage::ImportGguf { name, .. } => format!("import_gguf → {name}"),
        JobStage::Convert { name, backend, .. } => format!("convert/{backend} → {name}"),
        JobStage::RlhfStage { kind, .. } => format!("rlhf_stage:{kind}"),
    }
}
