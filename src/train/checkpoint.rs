//! Intermediate training checkpoint (config + f32 weights + optional tokenizer).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use super::tiny::{TinyConfig, TinyModel};
use crate::error::{AppError, Result};

pub const CONFIG_FILE: &str = "config.json";
pub const WEIGHTS_FILE: &str = "weights.bin";
pub const INDEX_FILE: &str = "tensors.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub format_version: u32,
    pub name: String,
    pub config: TinyConfig,
    pub step: usize,
    pub loss: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorIndex {
    tensors: BTreeMap<String, TensorLoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorLoc {
    offset: u64,
    shape: Vec<usize>,
    dtype: String,
}

pub fn save_checkpoint(
    dir: impl AsRef<Path>,
    name: &str,
    model: &TinyModel,
    step: usize,
    loss: Option<f32>,
    tokenizer_src: Option<&Path>,
) -> Result<PathBuf> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let meta = CheckpointMeta {
        format_version: 1,
        name: name.to_string(),
        config: model.cfg.clone(),
        step,
        loss,
    };
    fs::write(
        dir.join(CONFIG_FILE),
        serde_json::to_string_pretty(&meta)?,
    )?;

    let named = model.named_tensors()?;
    let mut blob = Vec::new();
    let mut tensors = BTreeMap::new();
    for (name, t) in named {
        let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let offset = blob.len() as u64;
        for v in &flat {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        tensors.insert(
            name,
            TensorLoc {
                offset,
                shape: t.dims().to_vec(),
                dtype: "f32".into(),
            },
        );
    }
    fs::write(
        dir.join(INDEX_FILE),
        serde_json::to_string_pretty(&TensorIndex { tensors })?,
    )?;
    fs::write(dir.join(WEIGHTS_FILE), &blob)?;

    if let Some(src) = tokenizer_src {
        let dst = dir.join(TOKENIZER_FILE);
        if src != dst.as_path() {
            fs::copy(src, &dst)?;
        }
    }
    Ok(dir.to_path_buf())
}

pub fn load_checkpoint(dir: impl AsRef<Path>, device: &Device) -> Result<TinyModel> {
    let dir = dir.as_ref();
    let meta: CheckpointMeta = serde_json::from_str(&fs::read_to_string(dir.join(CONFIG_FILE))?)?;
    meta.config.validate()?;
    let index: TensorIndex = serde_json::from_str(&fs::read_to_string(dir.join(INDEX_FILE))?)?;
    let blob = fs::read(dir.join(WEIGHTS_FILE))?;

    let mut tensors = BTreeMap::new();
    for (name, loc) in &index.tensors {
        let n: usize = loc.shape.iter().product();
        let byte_len = n.saturating_mul(4);
        let start = loc.offset as usize;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| AppError::msg("tensor offset overflow"))?;
        if end > blob.len() {
            return Err(AppError::msg(format!("tensor `{name}` out of weights.bin")));
        }
        let mut vals = Vec::with_capacity(n);
        for chunk in blob[start..end].chunks_exact(4) {
            vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let t = Tensor::from_vec(vals, loc.shape.as_slice(), device)?;
        // VarMap keys match VarBuilder paths (`token_embd.weight`, `blk.0.attn_q.weight`, …).
        tensors.insert(name.clone(), t);
    }

    let varmap = VarMap::new();
    {
        let mut data = varmap
            .data()
            .lock()
            .map_err(|_| AppError::msg("VarMap lock poisoned"))?;
        for (path, t) in tensors {
            data.insert(path, candle_core::Var::from_tensor(&t)?);
        }
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    TinyModel::from_vb(meta.config, varmap, vb, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_roundtrip() {
        let device = Device::Cpu;
        let cfg = TinyConfig {
            vocab_size: 16,
            n_embd: 16,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            n_ff: 32,
            max_seq: 8,
            ..TinyConfig::default()
        };
        let model = TinyModel::new(cfg, &device).unwrap();
        let dir = std::env::temp_dir().join(format!("lpc-ckpt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        save_checkpoint(&dir, "t", &model, 1, Some(1.23), None).unwrap();
        let loaded = load_checkpoint(&dir, &device).unwrap();
        assert_eq!(loaded.cfg.n_embd, 16);
        let _ = fs::remove_dir_all(&dir);
    }
}
