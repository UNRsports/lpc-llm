//! Checkpoint / TinyModel → F16 GGUF + LocalStore registration.

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device};
use console::style;

use super::checkpoint::{self, CheckpointMeta, TOKENIZER_FILE};
use super::tiny::{TinyConfig, TinyModel};
use crate::error::{AppError, Result};
use crate::store::{now_unix, InstalledModel, LocalStore};

/// Export a loaded tiny model to GGUF (F16 tensors, llama architecture).
pub fn export_gguf(model: &TinyModel, out_path: impl AsRef<Path>) -> Result<PathBuf> {
    let out_path = out_path.as_ref();
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let cfg = &model.cfg;
    let named = model.named_tensors()?;

    let mut qtensors: Vec<(String, QTensor)> = Vec::new();
    let mut order = Vec::new();
    order.push("token_embd.weight".to_string());
    for i in 0..cfg.n_layers {
        for suffix in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            order.push(format!("blk.{i}.{suffix}"));
        }
    }
    order.push("output_norm.weight".to_string());
    if !cfg.tie_embeddings {
        order.push("output.weight".to_string());
    }

    for name in &order {
        let t = named.get(name).ok_or_else(|| {
            AppError::msg(format!("missing tensor `{name}` for GGUF export"))
        })?;
        let t = t.to_dtype(DType::F32)?;
        let qt = QTensor::quantize(&t, GgmlDType::F16)
            .map_err(|e| AppError::msg(format!("quantize {name}: {e}")))?;
        qtensors.push((name.clone(), qt));
    }

    let metadata = llama_metadata(cfg);
    let meta_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = qtensors
        .iter()
        .map(|(n, t)| (n.as_str(), t))
        .collect();

    let mut file = fs::File::create(out_path)?;
    gguf_file::write(&mut file, &meta_refs, &tensor_refs)
        .map_err(|e| AppError::msg(format!("GGUF write: {e}")))?;

    eprintln!(
        "{} wrote GGUF {} ({:.2} MiB, arch=llama, f16)",
        style("✓").green(),
        out_path.display(),
        out_path.metadata()?.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(out_path.to_path_buf())
}

pub fn export_checkpoint_dir(
    ckpt_dir: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
) -> Result<PathBuf> {
    let device = Device::Cpu;
    let model = checkpoint::load_checkpoint(ckpt_dir.as_ref(), &device)?;
    export_gguf(&model, out_path)
}

/// Copy GGUF + tokenizer into `blobs/` and record in manifest for `lpc-llm run`.
pub fn register_gguf_model(
    store: &LocalStore,
    name: &str,
    gguf_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
) -> Result<InstalledModel> {
    let gguf_path = gguf_path.as_ref();
    let tokenizer_path = tokenizer_path.as_ref();
    if !gguf_path.is_file() {
        return Err(AppError::msg(format!(
            "GGUF not found: {}",
            gguf_path.display()
        )));
    }
    if !tokenizer_path.is_file() {
        return Err(AppError::msg(format!(
            "tokenizer not found: {}",
            tokenizer_path.display()
        )));
    }

    let safe = name.replace([':', '/'], "_");
    let repo = format!("local/{safe}");
    let gguf_file = format!("{safe}.gguf");
    let dest_gguf = store.blob_path(&repo, &gguf_file);
    let dest_tok = store.blob_path(&repo, "tokenizer.json");
    if let Some(p) = dest_gguf.parent() {
        fs::create_dir_all(p)?;
    }
    fs::copy(gguf_path, &dest_gguf)?;
    fs::copy(tokenizer_path, &dest_tok)?;

    let installed = InstalledModel {
        name: name.to_string(),
        model_path: dest_gguf,
        tokenizer_repo: repo.clone(),
        tokenizer_path: dest_tok,
        hf_repo: repo,
        gguf_file,
        pulled_at_unix: now_unix(),
    };
    store.record(installed.clone())?;
    eprintln!(
        "{} registered model `{}` → {}",
        style("✓").green(),
        style(name).bold(),
        installed.model_path.display()
    );
    Ok(installed)
}

/// Export checkpoint to GGUF under blobs and register.
pub fn export_and_register(
    store: &LocalStore,
    ckpt_dir: impl AsRef<Path>,
    name: &str,
) -> Result<InstalledModel> {
    let ckpt_dir = ckpt_dir.as_ref();
    let meta: CheckpointMeta =
        serde_json::from_str(&fs::read_to_string(ckpt_dir.join(checkpoint::CONFIG_FILE))?)?;
    let tok = ckpt_dir.join(TOKENIZER_FILE);
    if !tok.is_file() {
        return Err(AppError::msg(format!(
            "checkpoint missing {TOKENIZER_FILE}"
        )));
    }
    let safe = name.replace([':', '/'], "_");
    let staging = store
        .cache_dir()
        .join("train")
        .join(&safe)
        .join("export");
    fs::create_dir_all(&staging)?;
    let gguf = staging.join(format!("{safe}.gguf"));
    let _ = meta;
    export_checkpoint_dir(ckpt_dir, &gguf)?;
    register_gguf_model(store, name, &gguf, &tok)
}

fn llama_metadata(cfg: &TinyConfig) -> Vec<(String, gguf_file::Value)> {
    use gguf_file::Value;
    vec![
        (
            "general.architecture".into(),
            Value::String("llama".into()),
        ),
        (
            "general.name".into(),
            Value::String("lpc-llm-tiny".into()),
        ),
        (
            "general.file_type".into(),
            Value::U32(1), // mostly F16
        ),
        ("llama.block_count".into(), Value::U32(cfg.n_layers as u32)),
        (
            "llama.context_length".into(),
            Value::U32(cfg.max_seq as u32),
        ),
        (
            "llama.embedding_length".into(),
            Value::U32(cfg.n_embd as u32),
        ),
        (
            "llama.feed_forward_length".into(),
            Value::U32(cfg.n_ff as u32),
        ),
        (
            "llama.attention.head_count".into(),
            Value::U32(cfg.n_heads as u32),
        ),
        (
            "llama.attention.head_count_kv".into(),
            Value::U32(cfg.n_kv_heads as u32),
        ),
        (
            "llama.rope.dimension_count".into(),
            Value::U32(cfg.head_dim() as u32),
        ),
        (
            "llama.rope.freq_base".into(),
            Value::F32(cfg.rope_theta),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(cfg.rms_norm_eps as f32),
        ),
        ("llama.vocab_size".into(), Value::U32(cfg.vocab_size as u32)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::tiny::TinyConfig;

    #[test]
    fn export_tiny_gguf_readable() {
        let device = Device::Cpu;
        let cfg = TinyConfig {
            vocab_size: 32,
            n_embd: 32,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 4,
            n_ff: 64,
            max_seq: 16,
            tie_embeddings: true,
            ..TinyConfig::default()
        };
        let model = TinyModel::new(cfg, &device).unwrap();
        let dir = std::env::temp_dir().join(format!("lpc-gguf-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        export_gguf(&model, &path).unwrap();

        let mut f = fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut f).unwrap();
        assert_eq!(
            content
                .metadata
                .get("general.architecture")
                .unwrap()
                .to_string()
                .unwrap(),
            "llama"
        );
        assert!(content.tensor_infos.contains_key("token_embd.weight"));
        // Engine path should accept the file.
        let compute = crate::device::ComputeContext::from_pref(crate::config::ComputeDevicePref::Cpu)
            .unwrap();
        let eng = crate::engine::Engine::load(&path, compute).unwrap();
        assert_eq!(eng.architecture(), "llama");
        let _ = fs::remove_dir_all(&dir);
    }
}
