//! On-disk adapter layout: `adapter.json` + `weights.bin` (FP16 LE).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use candle_core::Device;
use serde::{Deserialize, Serialize};

use super::lora::{
    delta_from_f16_bytes, f32_to_f16_bits, LayerLora, LoraModuleName,
};
use crate::error::{AppError, Result};

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterFileMeta {
    pub format_version: u32,
    pub name: String,
    pub base_model: String,
    pub rank: usize,
    pub alpha: f64,
    pub dtype: String,
    pub targets: Vec<String>,
    pub layers: Vec<AdapterLayerMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterLayerMeta {
    pub index: usize,
    pub modules: BTreeMap<String, ModuleWeightsMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleWeightsMeta {
    pub a_offset: u64,
    pub a_shape: Vec<usize>,
    pub b_offset: u64,
    pub b_shape: Vec<usize>,
}

/// Loaded adapter ready to bind onto [`crate::hybrid::HybridEngine`].
#[derive(Debug, Clone)]
#[allow(dead_code)] // path / helpers kept for CLI and future bind checks
pub struct AdapterSet {
    pub meta: AdapterFileMeta,
    pub path: PathBuf,
    /// One entry per transformer layer index (may be empty for sparse adapters).
    pub layers: Vec<LayerLora>,
    /// Resident bytes (A/B tensors roughly).
    pub resident_bytes: usize,
}

impl AdapterSet {
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let meta_path = dir.join("adapter.json");
        let weights_path = dir.join("weights.bin");
        let text = fs::read_to_string(&meta_path).map_err(|e| {
            AppError::msg(format!("adapter meta {}: {e}", meta_path.display()))
        })?;
        let meta: AdapterFileMeta = serde_json::from_str(&text)?;
        if meta.format_version != FORMAT_VERSION {
            return Err(AppError::msg(format!(
                "unsupported adapter format_version {} (want {FORMAT_VERSION})",
                meta.format_version
            )));
        }
        if meta.dtype != "f16" {
            return Err(AppError::msg(format!(
                "unsupported adapter dtype `{}` (want f16)",
                meta.dtype
            )));
        }
        if meta.rank == 0 {
            return Err(AppError::msg("adapter rank must be > 0"));
        }

        let mut weights = Vec::new();
        File::open(&weights_path)
            .and_then(|mut f| f.read_to_end(&mut weights))
            .map_err(|e| {
                AppError::msg(format!("adapter weights {}: {e}", weights_path.display()))
            })?;

        let scale = meta.alpha / meta.rank as f64;
        let max_layer = meta
            .layers
            .iter()
            .map(|l| l.index)
            .max()
            .map(|i| i + 1)
            .unwrap_or(0);
        let mut layers = vec![LayerLora::default(); max_layer];
        let mut resident_bytes = 0usize;

        for layer_meta in &meta.layers {
            if layer_meta.index >= layers.len() {
                layers.resize_with(layer_meta.index + 1, LayerLora::default);
            }
            for (mod_name, wm) in &layer_meta.modules {
                let Some(kind) = LoraModuleName::parse(mod_name) else {
                    return Err(AppError::msg(format!(
                        "unknown LoRA module `{mod_name}` in adapter {}",
                        meta.name
                    )));
                };
                let a_n: usize = wm.a_shape.iter().product();
                let b_n: usize = wm.b_shape.iter().product();
                let a_bytes = a_n * 2;
                let b_bytes = b_n * 2;
                let a_off = wm.a_offset as usize;
                let b_off = wm.b_offset as usize;
                if a_off + a_bytes > weights.len() || b_off + b_bytes > weights.len() {
                    return Err(AppError::msg(format!(
                        "adapter {} layer {} module {mod_name}: weight slice out of range",
                        meta.name, layer_meta.index
                    )));
                }
                let delta = delta_from_f16_bytes(
                    &weights[a_off..a_off + a_bytes],
                    &wm.a_shape,
                    &weights[b_off..b_off + b_bytes],
                    &wm.b_shape,
                    scale,
                    device,
                )?;
                resident_bytes = resident_bytes
                    .saturating_add(a_n.saturating_mul(4))
                    .saturating_add(b_n.saturating_mul(4));
                layers[layer_meta.index].set(kind, delta);
            }
        }

        Ok(Self {
            meta,
            path: dir.to_path_buf(),
            layers,
            resident_bytes,
        })
    }

    #[allow(dead_code)]
    pub fn layer(&self, index: usize) -> Option<&LayerLora> {
        self.layers.get(index)
    }

    pub fn name(&self) -> &str {
        &self.meta.name
    }

    pub fn base_model(&self) -> &str {
        &self.meta.base_model
    }
}

/// Write a zero-filled (or tiny-noise) demo adapter for integration tests.
///
/// Targets `attn_q`, `attn_v`, `attn_output` with square `emb_dim` projections.
pub fn write_demo_adapter(
    dir: impl AsRef<Path>,
    name: &str,
    base_model: &str,
    n_layers: usize,
    emb_dim: usize,
    rank: usize,
    alpha: f64,
    // When true, fill with tiny random-ish values instead of zeros.
    tiny_noise: bool,
) -> Result<PathBuf> {
    if rank == 0 || emb_dim == 0 || n_layers == 0 {
        return Err(AppError::msg(
            "write_demo_adapter: n_layers, emb_dim, rank must be > 0",
        ));
    }
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let targets = vec![
        "attn_q".to_string(),
        "attn_v".to_string(),
        "attn_output".to_string(),
    ];
    let mut blob: Vec<u8> = Vec::new();
    let mut layers = Vec::with_capacity(n_layers);
    let mut seed = 0xC0FFEEu64;

    for index in 0..n_layers {
        let mut modules = BTreeMap::new();
        for t in &targets {
            let a_shape = vec![rank, emb_dim];
            let b_shape = vec![emb_dim, rank];
            let a_offset = blob.len() as u64;
            append_f16_matrix(&mut blob, rank * emb_dim, tiny_noise, &mut seed);
            let b_offset = blob.len() as u64;
            // B starts at 0 so a zero A or zero B keeps Δ ≈ 0 when not using noise.
            append_f16_matrix(&mut blob, emb_dim * rank, tiny_noise, &mut seed);
            modules.insert(
                t.clone(),
                ModuleWeightsMeta {
                    a_offset,
                    a_shape,
                    b_offset,
                    b_shape,
                },
            );
        }
        layers.push(AdapterLayerMeta { index, modules });
    }

    let meta = AdapterFileMeta {
        format_version: FORMAT_VERSION,
        name: name.to_string(),
        base_model: base_model.to_string(),
        rank,
        alpha,
        dtype: "f16".into(),
        targets,
        layers,
    };

    let meta_path = dir.join("adapter.json");
    let weights_path = dir.join("weights.bin");
    fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta)?,
    )?;
    let mut f = File::create(&weights_path)?;
    f.write_all(&blob)?;
    Ok(dir.to_path_buf())
}

fn append_f16_matrix(out: &mut Vec<u8>, n: usize, tiny_noise: bool, seed: &mut u64) {
    for _ in 0..n {
        let v = if tiny_noise {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((*seed >> 33) as u32) as f32 / (u32::MAX as f32);
            (u - 0.5) * 1e-4
        } else {
            0.0
        };
        let bits = f32_to_f16_bits(v);
        out.push((bits & 0xff) as u8);
        out.push((bits >> 8) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn roundtrip_demo_adapter_zero() {
        let dir = std::env::temp_dir().join(format!(
            "lpc-llm-adapter-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        write_demo_adapter(&dir, "demo", "gemma2:2b", 2, 64, 4, 8.0, false).unwrap();
        let set = AdapterSet::load(&dir, &Device::Cpu).unwrap();
        assert_eq!(set.meta.name, "demo");
        assert_eq!(set.layers.len(), 2);
        assert!(set.layers[0].q.is_some());
        assert!(set.layers[0].v.is_some());
        assert!(set.layers[0].o.is_some());
        // Zero A/B → forward ≈ 0
        let x = candle_core::Tensor::randn(0f32, 1.0, (1, 3, 64), &Device::Cpu).unwrap();
        let d = set.layers[0].q.as_ref().unwrap().forward(&x).unwrap();
        let flat = d.flatten_all().unwrap();
        let vals = flat.to_vec1::<f32>().unwrap();
        let max = vals.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(max < 1e-5, "zero adapter delta too large: {max}");
        let _ = fs::remove_dir_all(&dir);
    }
}
