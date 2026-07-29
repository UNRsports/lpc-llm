//! Hybrid streaming inference tuned for 16 GiB hosts:
//! 1. Pack GGUF layers into a dense sidecar → one DMA / layer
//! 2. Keep the first N layers resident (hot ratio)
//! 3. Double-buffer the rest via io_uring while computing
//! 4. Track I/O vs compute to keep overlap healthy

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use candle_core::quantized::{ggml_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{ops, Embedding};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use crate::error::{AppError, Result};
use crate::io::gguf_map::{qtensor_from_loc, GgufLayerMap, LayerDmaPlan, TensorLoc};
use crate::io::nvme::AsyncNvmeReader;
use crate::io::pack::ensure_packed;
use crate::io::prefetch::PrefetchBufferManager;

const MAX_SEQ_LEN: usize = 4096;

/// Tunables for hybrid memory / SSD balance.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Soft RAM budget for weights (hot layers + 2 prefetch slots), MiB.
    pub ram_budget_mib: usize,
    /// Force hot layer count; `None` = derive from budget.
    pub hot_layers: Option<usize>,
    /// Tokens to emit ASAP on the first turn (body of “思考の小分け”).
    pub first_burst_tokens: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            ram_budget_mib: 4096,
            hot_layers: None,
            first_burst_tokens: 24,
        }
    }
}

struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    /// Gemma2 GGUF uses GeLU; llama-family uses SiLU.
    use_gelu: bool,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let lhs = if self.use_gelu {
            gate.gelu()?
        } else {
            candle_nn::ops::silu(&gate)?
        };
        let rhs = self.up.forward(xs)?;
        let mid = (lhs * rhs)?;
        self.down.forward(&mid)
    }
}

/// RMSNorm. GGUF Gemma weights are already converted to full scale `(1+δ)`
/// by the HF→GGUF exporter, so we always multiply by `w` (never `1+w` again).
struct Norm {
    weight: Tensor,
    eps: f64,
}

impl Norm {
    fn from_qtensor(q: QTensor, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: q.dequantize(&q.device())?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(ops::rms_norm(xs, &self.weight, self.eps as f32)?)
    }
}

struct LayerLive {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    attn_norm: Norm,
    ffn_norm: Norm,
    post_attention_norm: Option<Norm>,
    post_ffw_norm: Option<Norm>,
    mlp: Mlp,
}

/// Streaming llama-family model: packed DMA + hot resident layers.
pub struct HybridEngine {
    map: GgufLayerMap,
    /// Dense DMA plans pointing into the pack file (not the raw GGUF).
    packed_layers: Vec<LayerDmaPlan>,
    device: Device,
    embeddings: Embedding,
    output_norm: Norm,
    output: QMatMul,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Vec<Option<(Tensor, Tensor)>>,
    masks: HashMap<usize, Tensor>,
    buffers: PrefetchBufferManager,
    reader: AsyncNvmeReader,
    /// First `hot_count` layers kept in RAM.
    resident: Vec<Option<LayerLive>>,
    hot_count: usize,
    config: HybridConfig,
    /// Rolling average wait / compute micros (chunk-size feedback).
    avg_wait_us: f64,
    avg_compute_us: f64,
}

impl HybridEngine {
    pub fn load_with_config(
        path: impl AsRef<std::path::Path>,
        config: HybridConfig,
        pack_cache: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let pack_cache = pack_cache.as_ref();
        let mut map = GgufLayerMap::open(path).map_err(|e| AppError::msg(e.to_string()))?;
        let device = Device::Cpu;

        if map.embedding_length == 0 || map.head_count == 0 {
            return Err(AppError::msg(format!(
                "incomplete GGUF metadata for {} (arch={})",
                path.display(),
                map.architecture
            )));
        }

        // --- (大) pack rearrange (engine cache; GGUF in blobs/ untouched) ---
        let packed =
            ensure_packed(path, &map, pack_cache).map_err(|e| AppError::msg(e.to_string()))?;
        map.layers = packed.layers.clone();
        map.max_layer_bytes = packed.max_layer_bytes;

        let slot = packed.recommended_slot_bytes();
        let hot_count = choose_hot_layers(
            map.layers.len(),
            packed.max_layer_bytes,
            config.ram_budget_mib,
            config.hot_layers,
        );

        let buffers = match PrefetchBufferManager::new(slot) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("warning: mlock failed ({e}); using unlocked arenas");
                PrefetchBufferManager::new_unlocked(slot)
                    .map_err(|e| AppError::msg(e.to_string()))?
            }
        };
        let reader =
            AsyncNvmeReader::open(&packed.pack_path).map_err(|e| AppError::msg(e.to_string()))?;

        let mut file = File::open(path)?;
        let tok_loc = find_hot(&map.hot, "token_embd.weight")?;
        let norm_loc = find_hot(&map.hot, "output_norm.weight")?;
        let out_loc = map.hot.iter().find(|t| t.name == "output.weight").cloned();

        let tok_q = read_tensor_from_file(&mut file, tok_loc, &device)?;
        let emb_dim = map.embedding_length;
        let embeddings = Embedding::new(tok_q.dequantize(&device)?, emb_dim);
        let output_norm = Norm::from_qtensor(
            read_tensor_from_file(&mut file, norm_loc, &device)?,
            map.rms_norm_eps,
        )?;
        let output = match &out_loc {
            Some(loc) => QMatMul::from_qtensor(read_tensor_from_file(&mut file, loc, &device)?)?,
            None => QMatMul::from_qtensor(tok_q)?,
        };

        let (cos, sin) =
            precompute_rope(map.head_dim.max(map.rope_dim), map.rope_freq_base, &device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)?;
        let n_layers = packed.layers.len();

        // --- (中〜大) pin hot layers ---
        let mut resident = Vec::with_capacity(n_layers);
        resident.resize_with(n_layers, || None);
        if hot_count > 0 {
            eprintln!("pinning hot layers 0..{hot_count} into RAM …");
            let mut pack_file = File::open(&packed.pack_path)?;
            for i in 0..hot_count {
                let plan = &packed.layers[i];
                let mut buf = vec![0u8; plan.read_len];
                pack_file.seek(SeekFrom::Start(plan.read_offset))?;
                // Pack is 4K-aligned; buffered read is fine for one-time pin.
                let n = plan.payload_bytes.min(plan.read_len);
                pack_file.read_exact(&mut buf[..n])?;
                // Zero pad remainder already in vec.
                let layer = materialize_layer(plan, &buf, &map, &device)?;
                resident[i] = Some(layer);
            }
        }

        eprintln!(
            "hybrid: arch={} layers={} hot={} stream={} slot={} MiB softcap={:?}/{:?} pack={}",
            map.architecture,
            n_layers,
            hot_count,
            n_layers.saturating_sub(hot_count),
            slot / (1024 * 1024),
            map.attn_logit_softcapping,
            map.final_logit_softcapping,
            packed.pack_path.display()
        );
        if let Some(sw) = map.sliding_window {
            eprintln!("hybrid: sliding_window={sw} (short prompts use full causal mask)");
        }

        Ok(Self {
            map,
            packed_layers: packed.layers,
            device,
            embeddings,
            output_norm,
            output,
            cos,
            sin,
            neg_inf,
            kv_cache: vec![None; n_layers],
            masks: HashMap::new(),
            buffers,
            reader,
            resident,
            hot_count,
            config,
            avg_wait_us: 0.0,
            avg_compute_us: 0.0,
        })
    }

    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    pub fn architecture(&self) -> &str {
        &self.map.architecture
    }

    pub fn device_name(&self) -> &str {
        "CPU+pack+io_uring"
    }

    pub fn reset_state(&mut self) {
        for slot in &mut self.kv_cache {
            *slot = None;
        }
        self.masks.clear();
    }

    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| AppError::msg(format!("tokenize: {e}")))?;
        let mut tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Err(AppError::msg("empty prompt"));
        }

        // Warm first streamed layer during prefill setup (TTFT).
        self.warm_first_stream_layer()?;

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), None);
        let eos = eos_ids(tokenizer);

        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = prepare_logits(self.forward(&input, 0)?)?;

        let mut generated = String::new();
        for _ in 0..max_tokens {
            let next = logits_processor.sample(&logits)?;
            tokens.push(next);
            let piece = tokenizer
                .decode(&[next], true)
                .map_err(|e| AppError::msg(format!("decode: {e}")))?;
            on_token(&piece)?;
            generated.push_str(&piece);
            if eos.contains(&next) {
                break;
            }
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = prepare_logits(self.forward(&input, tokens.len() - 1)?)?;
        }
        Ok(generated)
    }

    fn warm_first_stream_layer(&mut self) -> Result<()> {
        let i = self.hot_count;
        if i >= self.packed_layers.len() {
            return Ok(());
        }
        if self.reader.has_in_flight() {
            return Ok(());
        }
        let plan = &self.packed_layers[i];
        let buf = self.buffers.get_mut(0)?;
        self.reader
            .submit_read(buf, 0, plan.read_offset, plan.read_len)?;
        self.reader.wait_completion()?;
        Ok(())
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len)?)
        };

        let mut xs = self.embeddings.forward(x)?;
        // Gemma/Gemma2: scale embeddings by √hidden.
        if self.map.is_gemma_family() {
            xs = (xs * (self.map.embedding_length as f64).sqrt())?;
        }
        let n = self.packed_layers.len();

        // Bootstrap first streamed layer into slot 0 (double-buffer start).
        if self.hot_count < n {
            let plan = &self.packed_layers[self.hot_count];
            let buf = self.buffers.get_mut(0)?;
            self.reader
                .submit_read(buf, 0, plan.read_offset, plan.read_len)?;
            self.reader.wait_completion()?;
        }

        for i in 0..n {
            if i < self.hot_count {
                // Resident path — no I/O.
                // Safety: temporarily take layer, run, put back.
                let layer = self.resident[i]
                    .take()
                    .ok_or_else(|| AppError::msg(format!("hot layer {i} missing")))?;
                xs = self.forward_one_layer(i, &layer, &xs, mask.as_ref(), index_pos)?;
                self.resident[i] = Some(layer);
                continue;
            }

            let stream_idx = i - self.hot_count;
            let compute_slot = stream_idx % 2;
            let prefetch_slot = 1 - compute_slot;
            let plan = self.packed_layers[i].clone();

            // Prefetch next streamed layer.
            if let Some(next) = self.packed_layers.get(i + 1) {
                if i + 1 >= self.hot_count {
                    let t0 = Instant::now();
                    let buf = self.buffers.get_mut(prefetch_slot)?;
                    self.reader.submit_read(
                        buf,
                        prefetch_slot,
                        next.read_offset,
                        next.read_len,
                    )?;
                    let submit_us = t0.elapsed().as_micros() as f64;
                    let _ = submit_us;
                }
            }

            let t_compute = Instant::now();
            let layer = {
                let buf = self.buffers.get(compute_slot)?;
                materialize_layer(&plan, buf.as_slice(), &self.map, &self.device)?
            };
            xs = self.forward_one_layer(i, &layer, &xs, mask.as_ref(), index_pos)?;
            let compute_us = t_compute.elapsed().as_micros() as f64;

            let wait_us = if self.reader.has_in_flight() {
                let t0 = Instant::now();
                self.reader.wait_completion()?;
                t0.elapsed().as_micros() as f64
            } else {
                0.0
            };

            // --- (小) chunk / overlap feedback: EMA of wait vs compute ---
            const A: f64 = 0.2;
            self.avg_compute_us = (1.0 - A) * self.avg_compute_us + A * compute_us;
            self.avg_wait_us = (1.0 - A) * self.avg_wait_us + A * wait_us;
        }

        let xs = self.output_norm.forward(&xs)?;
        let xs = xs.i((.., seq_len - 1, ..))?;
        let logits = self.output.forward(&xs)?;
        match self.map.final_logit_softcapping {
            Some(sc) if sc > 0.0 => Ok(((logits / sc)?.tanh()? * sc)?),
            _ => Ok(logits),
        }
    }

    fn forward_one_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerLive,
        xs: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let x = layer.attn_norm.forward(xs)?;
        let attn = self.forward_attn(layer_idx, layer, &x, mask, index_pos)?;
        let attn = match &layer.post_attention_norm {
            Some(n) => n.forward(&attn)?,
            None => attn,
        };
        let x = (attn + residual)?;

        let residual = &x;
        let x = layer.ffn_norm.forward(&x)?;
        let x = layer.mlp.forward(&x)?;
        let x = match &layer.post_ffw_norm {
            Some(n) => n.forward(&x)?,
            None => x,
        };
        Ok((x + residual)?)
    }

    fn forward_attn(
        &mut self,
        layer_idx: usize,
        layer: &LayerLive,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        let n_head = self.map.head_count;
        let n_kv = self.map.head_count_kv;
        let head_dim = self.map.head_dim;

        let q = layer.wq.forward(x)?;
        let k = layer.wk.forward(x)?;
        let v = layer.wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, n_kv, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, n_kv, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = apply_rope(
            &q,
            &self.cos,
            &self.sin,
            index_pos,
            self.map.is_gemma_family(),
        )?;
        let k = apply_rope(
            &k,
            &self.cos,
            &self.sin,
            index_pos,
            self.map.is_gemma_family(),
        )?;

        let (k, v) = match &self.kv_cache[layer_idx] {
            None => (k, v),
            Some((_kc, _vc)) if index_pos == 0 => (k, v),
            Some((kc, vc)) => {
                let k = Tensor::cat(&[kc, &k], 2)?;
                let v = Tensor::cat(&[vc, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache[layer_idx] = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, n_head / n_kv.max(1))?;
        let v = repeat_kv(v, n_head / n_kv.max(1))?;

        let mut att = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
        if let Some(sc) = self.map.attn_logit_softcapping {
            att = ((att / sc)?.tanh()? * sc)?;
        }
        let att = match mask {
            None => att,
            Some(m) => {
                let m = m.broadcast_as(att.shape())?;
                let on_true = self.neg_inf.broadcast_as(att.shape())?;
                m.where_cond(&on_true, &att)?
            }
        };
        let att = ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, n_head * head_dim])?;
        Ok(layer.wo.forward(&y)?)
    }

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(m) = self.masks.get(&t) {
            return Ok(m.clone());
        }
        let mask: Vec<_> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }

    /// I/O wait vs compute ratio — >1 means still I/O bound.
    pub fn io_compute_ratio(&self) -> f64 {
        if self.avg_compute_us < 1.0 {
            return 0.0;
        }
        self.avg_wait_us / self.avg_compute_us
    }
}

fn choose_hot_layers(
    n_layers: usize,
    layer_bytes: usize,
    budget_mib: usize,
    override_hot: Option<usize>,
) -> usize {
    if let Some(h) = override_hot {
        return h.min(n_layers);
    }
    if n_layers == 0 || layer_bytes == 0 {
        return 0;
    }
    let budget = budget_mib.saturating_mul(1024 * 1024);
    // Reserve two prefetch slots + ~512 MiB headroom (KV / runtime).
    let reserve = layer_bytes.saturating_mul(2).saturating_add(512 * 1024 * 1024);
    let hot_budget = budget.saturating_sub(reserve);
    let by_ram = hot_budget / layer_bytes;
    by_ram.min(n_layers.saturating_mul(1) / 2).min(8)
}

fn try_tensor<'a>(plan: &'a LayerDmaPlan, suffix: &str) -> Option<&'a TensorLoc> {
    let name = format!("blk.{}.{}", plan.index, suffix);
    plan.tensors.iter().find(|t| t.name == name)
}

fn require_tensor<'a>(plan: &'a LayerDmaPlan, suffix: &str) -> Result<&'a TensorLoc> {
    try_tensor(plan, suffix).ok_or_else(|| {
        AppError::msg(format!(
            "missing tensor blk.{}.{}",
            plan.index, suffix
        ))
    })
}

fn build_layer_live(
    map: &GgufLayerMap,
    plan: &LayerDmaPlan,
    dma: &[u8],
    device: &Device,
) -> Result<LayerLive> {
    let gemma = map.is_gemma_family();
    let q = |suffix: &str| -> Result<QTensor> {
        let loc = require_tensor(plan, suffix)?;
        Ok(qtensor_from_loc(loc, dma, device)?)
    };
    let opt_norm = |suffix: &str| -> Result<Option<Norm>> {
        match try_tensor(plan, suffix) {
            Some(loc) => Ok(Some(Norm::from_qtensor(
                qtensor_from_loc(loc, dma, device)?,
                map.rms_norm_eps,
            )?)),
            None => Ok(None),
        }
    };
    Ok(LayerLive {
        wq: QMatMul::from_qtensor(q("attn_q.weight")?)?,
        wk: QMatMul::from_qtensor(q("attn_k.weight")?)?,
        wv: QMatMul::from_qtensor(q("attn_v.weight")?)?,
        wo: QMatMul::from_qtensor(q("attn_output.weight")?)?,
        attn_norm: Norm::from_qtensor(q("attn_norm.weight")?, map.rms_norm_eps)?,
        ffn_norm: Norm::from_qtensor(q("ffn_norm.weight")?, map.rms_norm_eps)?,
        post_attention_norm: opt_norm("post_attention_norm.weight")?,
        post_ffw_norm: opt_norm("post_ffw_norm.weight")?,
        mlp: Mlp {
            gate: QMatMul::from_qtensor(q("ffn_gate.weight")?)?,
            down: QMatMul::from_qtensor(q("ffn_down.weight")?)?,
            up: QMatMul::from_qtensor(q("ffn_up.weight")?)?,
            // llama.cpp Gemma2 uses ggml_gelu (tanh approx); candle `gelu` matches.
            use_gelu: gemma,
        },
    })
}

fn materialize_layer(
    plan: &LayerDmaPlan,
    dma: &[u8],
    map: &GgufLayerMap,
    device: &Device,
) -> Result<LayerLive> {
    build_layer_live(map, plan, dma, device)
}

fn find_hot<'a>(hot: &'a [TensorLoc], name: &str) -> Result<&'a TensorLoc> {
    hot.iter()
        .find(|t| t.name == name)
        .ok_or_else(|| AppError::msg(format!("missing hot tensor {name}")))
}

fn read_tensor_from_file(
    file: &mut File,
    loc: &TensorLoc,
    device: &Device,
) -> Result<QTensor> {
    let mut raw = vec![0u8; loc.size_bytes];
    file.seek(SeekFrom::Start(loc.abs_offset))?;
    file.read_exact(&mut raw)?;
    Ok(ggml_file::qtensor_from_ggml(
        loc.dtype,
        &raw,
        loc.shape.clone(),
        device,
    )?)
}

fn precompute_rope(head_dim: usize, freq_base: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?;
    let idx_theta = idx.matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    neox: bool,
) -> Result<Tensor> {
    let (_b, _h, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?;
    let sin = sin.narrow(0, index_pos, seq_len)?;
    let x = x.contiguous()?;
    if neox {
        // Gemma / Gemma2: rotate-half (Neox) RoPE.
        Ok(candle_nn::rotary_emb::rope(&x, &cos, &sin)?)
    } else {
        // Llama-family GGUF: interleaved pairs.
        Ok(candle_nn::rotary_emb::rope_i(&x, &cos, &sin)?)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep <= 1 {
        return Ok(x);
    }
    let (b, n_kv, s, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))?)
}

fn eos_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    const CANDIDATES: &[&str] = &[
        "<|endoftext|>",
        "</s>",
        "<|eot_id|>",
        "<|end|>",
        "<end_of_turn>",
        "<|im_end|>",
    ];
    let mut ids = Vec::new();
    for c in CANDIDATES {
        if let Some(id) = tokenizer.token_to_id(c) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn prepare_logits(logits: Tensor) -> Result<Tensor> {
    let mut logits = logits.squeeze(0)?;
    if logits.dims().len() > 1 {
        let last = logits.dim(0)? - 1;
        logits = logits.get(last)?;
    }
    Ok(logits.clamp(-100.0, 100.0)?)
}
