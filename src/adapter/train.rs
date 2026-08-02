//! Phase 4: LoRA SFT prototype (`lpc-llm adapter create`).
//!
//! Frozen quantized base (HybridEngine) + trainable side-path LoRA.
//! Gradients reach LoRA via a differentiable lm_head (`forward_via_f16`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarMap};
use console::style;
use tokenizers::Tokenizer;

use super::format::{
    write_adapter, AdapterFileMeta, AdapterLayerMeta, ModuleWeightsMeta, FORMAT_VERSION,
};
use super::lora::{f32_to_f16_bits, LayerLora, LoraDelta, LoraModuleName};
use crate::error::{AppError, Result};
use crate::hybrid::{HybridConfig, HybridEngine};

/// CLI / trainer knobs for `adapter create`.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub name: String,
    pub base_model: String,
    pub rank: usize,
    pub alpha: f64,
    pub steps: usize,
    pub lr: f64,
    pub max_seq: usize,
    pub ram_mib: usize,
    /// When > 0, only the last N transformer layers get LoRA.
    pub last_layers: usize,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            name: "adapter".into(),
            base_model: "smollm2:360m".into(),
            rank: 8,
            alpha: 16.0,
            steps: 64,
            lr: 1e-3,
            max_seq: 128,
            ram_mib: 4096,
            last_layers: 0,
            seed: 42,
        }
    }
}

const DEFAULT_TARGETS: &[LoraModuleName] = &[
    LoraModuleName::AttnQ,
    LoraModuleName::AttnV,
    LoraModuleName::AttnOutput,
];

fn module_weight_suffix(m: LoraModuleName) -> &'static str {
    match m {
        LoraModuleName::AttnQ => "attn_q.weight",
        LoraModuleName::AttnK => "attn_k.weight",
        LoraModuleName::AttnV => "attn_v.weight",
        LoraModuleName::AttnOutput => "attn_output.weight",
        LoraModuleName::FfnGate => "ffn_gate.weight",
        LoraModuleName::FfnUp => "ffn_up.weight",
        LoraModuleName::FfnDown => "ffn_down.weight",
    }
}

/// Load training texts from a plain-text or JSONL file.
pub fn load_training_texts(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path).map_err(|e| {
        AppError::msg(format!("read training file {}: {e}", path.display()))
    })?;
    if raw.trim().is_empty() {
        return Err(AppError::msg("training file is empty"));
    }

    let is_jsonl = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"));

    if is_jsonl {
        let mut out = Vec::new();
        for (i, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                AppError::msg(format!("jsonl line {}: {e}", i + 1))
            })?;
            let text = v
                .get("text")
                .and_then(|t| t.as_str())
                .or_else(|| v.as_str())
                .ok_or_else(|| {
                    AppError::msg(format!(
                        "jsonl line {}: expected string or object with `text`",
                        i + 1
                    ))
                })?;
            if !text.trim().is_empty() {
                out.push(text.to_string());
            }
        }
        if out.is_empty() {
            return Err(AppError::msg("jsonl file has no usable `text` rows"));
        }
        return Ok(out);
    }

    // Plain text: non-empty lines as samples; if only one line, use whole file.
    let lines: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if lines.is_empty() {
        return Err(AppError::msg("training file has no non-empty lines"));
    }
    if lines.len() == 1 && raw.lines().count() > 1 {
        // Mostly blank lines — fall back to full trimmed body.
        return Ok(vec![raw.trim().to_string()]);
    }
    Ok(lines)
}

fn tokenize_chunks(
    tokenizer: &Tokenizer,
    texts: &[String],
    max_seq: usize,
) -> Result<Vec<Vec<u32>>> {
    if max_seq < 2 {
        return Err(AppError::msg("--max-seq must be >= 2"));
    }
    let mut chunks = Vec::new();
    for text in texts {
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| AppError::msg(format!("tokenize: {e}")))?;
        let ids = encoding.get_ids();
        if ids.len() < 2 {
            continue;
        }
        let mut start = 0;
        while start < ids.len() {
            let end = (start + max_seq).min(ids.len());
            if end - start >= 2 {
                chunks.push(ids[start..end].to_vec());
            }
            if end >= ids.len() {
                break;
            }
            // Overlap a little so long docs aren't chopped harshly.
            start = end.saturating_sub(max_seq / 4).max(start + 1);
        }
    }
    if chunks.is_empty() {
        return Err(AppError::msg(
            "no training chunks with >= 2 tokens (check --from / tokenizer)",
        ));
    }
    Ok(chunks)
}

struct TrainableLora {
    varmap: VarMap,
    layers: Vec<LayerLora>,
    /// Parallel to `layers`: which modules were created (for save).
    module_shapes: Vec<BTreeMap<&'static str, (usize, usize)>>,
    layer_indices: Vec<usize>,
    targets: Vec<&'static str>,
    rank: usize,
    alpha: f64,
}

impl TrainableLora {
    fn init(
        engine: &HybridEngine,
        cfg: &TrainConfig,
        device: &Device,
    ) -> Result<Self> {
        if cfg.rank == 0 {
            return Err(AppError::msg("--rank must be > 0"));
        }
        let n = engine.n_layers();
        let start = if cfg.last_layers == 0 {
            0
        } else {
            n.saturating_sub(cfg.last_layers)
        };
        let layer_indices: Vec<usize> = (start..n).collect();
        if layer_indices.is_empty() {
            return Err(AppError::msg("no layers selected for LoRA training"));
        }

        let varmap = VarMap::new();
        let scale = cfg.alpha / cfg.rank as f64;
        let mut layers = vec![LayerLora::default(); n];
        let mut module_shapes = vec![BTreeMap::new(); n];
        let mut targets = Vec::new();

        for &li in &layer_indices {
            for &mod_name in DEFAULT_TARGETS {
                let suffix = module_weight_suffix(mod_name);
                let (out_f, in_f) = match engine.projection_dims(li, suffix) {
                    Ok(d) => d,
                    Err(_) => continue, // skip missing (e.g. MoE without dense FFN targets)
                };
                let tname = mod_name.as_str();
                if !targets.iter().any(|t| *t == tname) {
                    targets.push(tname);
                }
                let a_path = format!("layers.{li}.{tname}.A");
                let b_path = format!("layers.{li}.{tname}.B");
                // LoRA init: A ~ N(0, 0.01), B = 0 → Δ ≈ 0 at step 0.
                let a = varmap.get(
                    (cfg.rank, in_f),
                    &a_path,
                    candle_nn::Init::Randn {
                        mean: 0.0,
                        stdev: 0.01,
                    },
                    DType::F32,
                    device,
                )?;
                let b = varmap.get(
                    (out_f, cfg.rank),
                    &b_path,
                    candle_nn::Init::Const(0.),
                    DType::F32,
                    device,
                )?;
                layers[li].set(
                    mod_name,
                    LoraDelta {
                        a,
                        b,
                        scale,
                    },
                );
                module_shapes[li].insert(tname, (out_f, in_f));
            }
        }

        if targets.is_empty() {
            return Err(AppError::msg(
                "could not place any LoRA modules (missing attn_q/v/output weights?)",
            ));
        }

        Ok(Self {
            varmap,
            layers,
            module_shapes,
            layer_indices,
            targets,
            rank: cfg.rank,
            alpha: cfg.alpha,
        })
    }

    fn save(&self, dir: &Path, name: &str, base_model: &str) -> Result<PathBuf> {
        let data = self
            .varmap
            .data()
            .lock()
            .map_err(|_| AppError::msg("VarMap lock poisoned"))?;
        let mut blob: Vec<u8> = Vec::new();
        let mut layers_meta = Vec::new();

        for &li in &self.layer_indices {
            let shapes = &self.module_shapes[li];
            if shapes.is_empty() {
                continue;
            }
            let mut modules = BTreeMap::new();
            for (tname, &(out_f, in_f)) in shapes {
                let a_path = format!("layers.{li}.{tname}.A");
                let b_path = format!("layers.{li}.{tname}.B");
                let a = data
                    .get(&a_path)
                    .ok_or_else(|| AppError::msg(format!("missing var {a_path}")))?
                    .as_tensor();
                let b = data
                    .get(&b_path)
                    .ok_or_else(|| AppError::msg(format!("missing var {b_path}")))?
                    .as_tensor();
                let a_offset = blob.len() as u64;
                append_tensor_f16(&mut blob, a)?;
                let b_offset = blob.len() as u64;
                append_tensor_f16(&mut blob, b)?;
                modules.insert(
                    (*tname).to_string(),
                    ModuleWeightsMeta {
                        a_offset,
                        a_shape: vec![self.rank, in_f],
                        b_offset,
                        b_shape: vec![out_f, self.rank],
                    },
                );
            }
            if !modules.is_empty() {
                layers_meta.push(AdapterLayerMeta {
                    index: li,
                    modules,
                });
            }
        }

        let meta = AdapterFileMeta {
            format_version: FORMAT_VERSION,
            name: name.to_string(),
            base_model: base_model.to_string(),
            rank: self.rank,
            alpha: self.alpha,
            dtype: "f16".into(),
            targets: self.targets.iter().map(|s| (*s).to_string()).collect(),
            layers: layers_meta,
        };
        drop(data);
        write_adapter(dir, &meta, &blob)
    }
}

fn append_tensor_f16(out: &mut Vec<u8>, t: &Tensor) -> Result<()> {
    let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    out.reserve(flat.len().saturating_mul(2));
    for v in flat {
        let bits = f32_to_f16_bits(v);
        out.push((bits & 0xff) as u8);
        out.push((bits >> 8) as u8);
    }
    Ok(())
}

/// Run LoRA SFT and write `adapters/<name>/`.
pub fn train_adapter(
    model_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
    pack_cache: impl AsRef<Path>,
    texts_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    cfg: TrainConfig,
) -> Result<PathBuf> {
    let texts = load_training_texts(texts_path)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| {
        AppError::msg(format!(
            "load tokenizer {}: {e}",
            tokenizer_path.as_ref().display()
        ))
    })?;
    let chunks = tokenize_chunks(&tokenizer, &texts, cfg.max_seq)?;

    eprintln!(
        "{} Phase 4 LoRA train: base={} out={} rank={} alpha={} steps={} max_seq={} samples={} chunks={}",
        style("▸").cyan(),
        style(&cfg.base_model).bold(),
        style(&cfg.name).bold(),
        cfg.rank,
        cfg.alpha,
        cfg.steps,
        cfg.max_seq,
        texts.len(),
        chunks.len()
    );

    // Pin all layers for training (clamped to n_layers inside choose_hot_layers).
    let hcfg = HybridConfig {
        ram_budget_mib: cfg.ram_mib,
        hot_layers: Some(usize::MAX),
        first_burst_tokens: 1,
        adapter_resident_bytes: 0,
    };

    let compute = crate::device::ComputeContext::from_pref(crate::config::ComputeDevicePref::Cpu)?;
    let mut engine = HybridEngine::load_with_config(model_path, hcfg, pack_cache, None, compute)?;
    let device = engine.device().clone();
    let trainable = TrainableLora::init(&engine, &cfg, &device)?;
    let n_params: usize = trainable
        .varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum();
    eprintln!(
        "  trainable LoRA params: {} ({:.2} MiB f32) layers={:?}",
        n_params,
        n_params as f64 * 4.0 / (1024.0 * 1024.0),
        trainable.layer_indices
    );

    // Tensors share storage with VarMap; AdamW updates are visible on the next forward.
    engine.set_lora_layers(trainable.layers.clone());

    let params = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: 0.0,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(trainable.varmap.all_vars(), params)?;

    let t0 = Instant::now();
    let mut last_loss = 0f32;
    eprintln!(
        "  training… (CPU LoRA SFT; each step can take tens of seconds — progress every step)"
    );
    for step in 0..cfg.steps {
        let step_t0 = Instant::now();
        let chunk = &chunks[step % chunks.len()];
        let logits = engine.forward_train(chunk)?; // [1, T, V]
        let (_b, t_dim, vocab) = logits.dims3()?;
        let t = chunk.len().min(t_dim);
        if t < 2 {
            return Err(AppError::msg("training chunk too short after forward"));
        }
        let logits = logits.narrow(1, 0, t - 1)?.reshape((t - 1, vocab))?;
        let targets = Tensor::new(&chunk[1..t], &device)?;
        let loss_t = loss::cross_entropy(&logits, &targets)?;
        last_loss = loss_t.to_vec0::<f32>()?;
        opt.backward_step(&loss_t)?;

        let done = step + 1;
        let step_s = step_t0.elapsed().as_secs_f64();
        let elapsed = t0.elapsed().as_secs_f64();
        let eta = if done > 0 {
            let avg = elapsed / done as f64;
            avg * (cfg.steps - done) as f64
        } else {
            0.0
        };
        eprintln!(
            "  step {:>4}/{}  loss={:.4}  step={:.1}s  elapsed={:.0}s  eta={:.0}s",
            done, cfg.steps, last_loss, step_s, elapsed, eta
        );
    }

    let out_dir = out_dir.as_ref();
    let path = trainable.save(out_dir, &cfg.name, &cfg.base_model)?;
    let weight_bytes = fs::metadata(path.join("weights.bin"))?.len();
    eprintln!(
        "{} wrote adapter `{}` ({:.2} MiB weights, final loss={:.4}) → {}",
        style("✓").green(),
        style(&cfg.name).bold(),
        weight_bytes as f64 / (1024.0 * 1024.0),
        last_loss,
        path.display()
    );
    let _ = cfg.seed; // reserved for future deterministic init
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_plain_and_jsonl() {
        let dir = std::env::temp_dir().join(format!("lpc-llm-train-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let plain = dir.join("a.txt");
        fs::write(&plain, "hello world\nsecond line\n").unwrap();
        let texts = load_training_texts(&plain).unwrap();
        assert_eq!(texts.len(), 2);

        let jsonl = dir.join("b.jsonl");
        let mut f = fs::File::create(&jsonl).unwrap();
        writeln!(f, "{{\"text\":\"alpha\"}}").unwrap();
        writeln!(f, "{{\"text\":\"beta\"}}").unwrap();
        let texts = load_training_texts(&jsonl).unwrap();
        assert_eq!(texts, vec!["alpha".to_string(), "beta".to_string()]);

        let _ = fs::remove_dir_all(&dir);
    }
}
