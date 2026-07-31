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

use crate::adapter::{qmatmul_with_lora, AdapterSet, LayerLora};
use crate::error::{AppError, Result};
use crate::io::gguf_map::{qtensor_from_loc, GgufLayerMap, LayerDmaPlan, TensorLoc};
use crate::io::moe::{ExpertDmaPlan, MoeInfo};
use crate::io::nvme::AsyncNvmeReader;
use crate::io::pack::{ensure_experts_packed, ensure_packed, PackedExperts};
use crate::io::prefetch::{PrefetchBufferManager, PrefetchRing};

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
    /// Extra resident bytes reserved for a bound LoRA adapter (deducted from hot budget).
    pub adapter_resident_bytes: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            ram_budget_mib: 4096,
            hot_layers: None,
            first_burst_tokens: 24,
            adapter_resident_bytes: 0,
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
    fn forward(&self, xs: &Tensor, lora: Option<&LayerLora>) -> Result<Tensor> {
        let gate = qmatmul_with_lora(&self.gate, lora.and_then(|l| l.gate.as_ref()), xs)?;
        let lhs = if self.use_gelu {
            gate.gelu()?
        } else {
            candle_nn::ops::silu(&gate)?
        };
        let rhs = qmatmul_with_lora(&self.up, lora.and_then(|l| l.up.as_ref()), xs)?;
        let mid = (lhs * rhs)?;
        qmatmul_with_lora(&self.down, lora.and_then(|l| l.down.as_ref()), &mid)
    }
}

/// Dense MLP or MoE block (router resident; experts DMA'd on demand).
enum FeedForward {
    Dense(Mlp),
    MoE {
        router: QMatMul,
        n_expert_used: usize,
        use_gelu: bool,
    },
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
    ff: FeedForward,
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
    /// MoE expert pack + DMA ring (absent on dense models).
    moe_runtime: Option<MoeRuntime>,
    /// First `hot_count` layers kept in RAM.
    resident: Vec<Option<LayerLive>>,
    hot_count: usize,
    config: HybridConfig,
    /// Per-layer LoRA slots (empty when no adapter bound).
    lora: Vec<LayerLora>,
    /// Optional Top-K expert affinity hints from the Phase 3 agent.
    expert_prefetch_hints: Vec<usize>,
    /// Rolling average wait / compute micros (chunk-size feedback).
    avg_wait_us: f64,
    avg_compute_us: f64,
}

/// Expert streaming state: packed plans + dedicated io_uring reader + ring.
struct MoeRuntime {
    /// Soft link / introspection (kept for API completeness).
    #[allow(dead_code)]
    info: MoeInfo,
    packed: PackedExperts,
    reader: AsyncNvmeReader,
    ring: PrefetchRing,
}

impl HybridEngine {
    pub fn load_with_config(
        path: impl AsRef<std::path::Path>,
        config: HybridConfig,
        pack_cache: impl AsRef<std::path::Path>,
        adapter: Option<AdapterSet>,
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

        let packed_experts = ensure_experts_packed(path, &map, pack_cache)
            .map_err(|e| AppError::msg(e.to_string()))?;
        if let Some(ref pe) = packed_experts {
            map.experts = pe.experts.clone();
            map.max_expert_bytes = pe.max_expert_bytes;
            map.moe = Some(pe.moe.clone());
        }

        let slot = packed.recommended_slot_bytes();
        let hot_count = choose_hot_layers(
            map.layers.len(),
            packed.max_layer_bytes,
            config.ram_budget_mib,
            config.hot_layers,
            config.adapter_resident_bytes,
            packed_experts
                .as_ref()
                .map(|p| p.recommended_slot_bytes().saturating_mul(p.moe.expert_used_count.max(2)))
                .unwrap_or(0),
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

        let moe_runtime = if let Some(pe) = packed_experts {
            let n_slots = pe.moe.expert_used_count.max(2);
            let expert_slot = pe.recommended_slot_bytes();
            let ring = match PrefetchRing::new(expert_slot, n_slots) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("warning: expert mlock failed ({e}); using unlocked ring");
                    PrefetchRing::new_unlocked(expert_slot, n_slots)
                        .map_err(|e| AppError::msg(e.to_string()))?
                }
            };
            let expert_reader =
                AsyncNvmeReader::open(&pe.pack_path).map_err(|e| AppError::msg(e.to_string()))?;
            eprintln!(
                "MoE: family={:?} experts={} top-k={} expert_slot={} KiB ring={}",
                pe.moe.family,
                pe.moe.expert_count,
                pe.moe.expert_used_count,
                expert_slot / 1024,
                n_slots
            );
            Some(MoeRuntime {
                info: pe.moe.clone(),
                packed: pe,
                reader: expert_reader,
                ring,
            })
        } else {
            None
        };

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

        let mut lora = vec![LayerLora::default(); n_layers];
        if let Some(set) = adapter {
            eprintln!(
                "binding adapter `{}` ({:.1} MiB resident) …",
                set.name(),
                set.resident_bytes as f64 / (1024.0 * 1024.0)
            );
            for (i, layer) in set.layers.into_iter().enumerate() {
                if i < lora.len() {
                    lora[i] = layer;
                }
            }
        }

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
            moe_runtime,
            resident,
            hot_count,
            config,
            lora,
            expert_prefetch_hints: Vec::new(),
            avg_wait_us: 0.0,
            avg_compute_us: 0.0,
        })
    }

    /// Agent / caller may set preferred expert IDs for the next forward (prefetch hints).
    pub fn set_expert_prefetch_hints(&mut self, hints: Vec<usize>) {
        self.expert_prefetch_hints = hints;
    }

    #[allow(dead_code)]
    pub fn moe_info(&self) -> Option<&MoeInfo> {
        self.moe_runtime.as_ref().map(|m| &m.info)
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
        let x = {
            let lora = self.lora.get(layer_idx);
            match &layer.ff {
                FeedForward::Dense(mlp) => mlp.forward(&x, lora)?,
                FeedForward::MoE {
                    router,
                    n_expert_used,
                    use_gelu,
                } => self.forward_moe(
                    layer_idx,
                    router,
                    *n_expert_used,
                    *use_gelu,
                    &x,
                )?,
            }
        };
        let x = match &layer.post_ffw_norm {
            Some(n) => n.forward(&x)?,
            None => x,
        };
        Ok((x + residual)?)
    }

    /// Gating → Top-K → expert DMA from `experts.pack` → weighted combine.
    fn forward_moe(
        &mut self,
        layer_idx: usize,
        router: &QMatMul,
        n_expert_used: usize,
        use_gelu: bool,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let router_logits = router.forward(&xs_flat)?;
        let routing = ops::softmax_last_dim(&router_logits)?;
        let routing_vec = routing.to_vec2::<f32>()?;

        // Apply optional agent affinity: boost hinted experts slightly.
        let hints = &self.expert_prefetch_hints;

        let mut top_x: Vec<Vec<u32>> = vec![Vec::new(); routing_vec.first().map(|r| r.len()).unwrap_or(0)];
        let mut selected_rws: Vec<Vec<f32>> = vec![Vec::new(); top_x.len()];

        for (row_idx, rw) in routing_vec.iter().enumerate() {
            let mut dst: Vec<u32> = (0..rw.len() as u32).collect();
            dst.sort_by(|&i, &j| {
                let mut wi = rw[i as usize];
                let mut wj = rw[j as usize];
                if hints.contains(&(i as usize)) {
                    wi *= 1.05;
                }
                if hints.contains(&(j as usize)) {
                    wj *= 1.05;
                }
                wj.total_cmp(&wi)
            });
            let mut sum = 0f32;
            for &expert_idx in dst.iter().take(n_expert_used) {
                sum += rw[expert_idx as usize];
            }
            let norm = if sum > 0.0 { sum } else { 1.0 };
            for &expert_idx in dst.iter().take(n_expert_used) {
                let expert_idx = expert_idx as usize;
                top_x[expert_idx].push(row_idx as u32);
                selected_rws[expert_idx].push(rw[expert_idx] / norm);
            }
        }

        let mut ys = xs_flat.zeros_like()?;

        // Collect experts that have tokens, prefetch via ring while computing.
        let active: Vec<usize> = top_x
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v.is_empty() { None } else { Some(i) })
            .collect();

        for (ai, &expert_id) in active.iter().enumerate() {
            let slot = ai % self
                .moe_runtime
                .as_ref()
                .map(|m| m.ring.len())
                .unwrap_or(2);

            // Prefetch next active expert into the other ring slot.
            if let Some(&next_id) = active.get(ai + 1) {
                let next_slot = (ai + 1)
                    % self
                        .moe_runtime
                        .as_ref()
                        .map(|m| m.ring.len())
                        .unwrap_or(2);
                self.dma_expert(layer_idx, next_id, next_slot)?;
            }

            let mlp = self.load_expert_mlp(layer_idx, expert_id, slot, use_gelu)?;
            let row_ids = &top_x[expert_id];
            if row_ids.is_empty() {
                continue;
            }
            let index = Tensor::new(row_ids.as_slice(), &self.device)?;
            let indexed = xs_flat.index_select(&index, 0)?;
            let out = mlp.forward(&indexed, None)?;
            let rw = Tensor::new(selected_rws[expert_id].as_slice(), &self.device)?
                .reshape((row_ids.len(), 1))?;
            let weighted = out.broadcast_mul(&rw)?;
            ys = ys.index_add(&index, &weighted, 0)?;
        }

        Ok(ys.reshape((b_size, seq_len, hidden_dim))?)
    }

    fn dma_expert(&mut self, layer_idx: usize, expert_id: usize, slot: usize) -> Result<()> {
        let rt = self
            .moe_runtime
            .as_mut()
            .ok_or_else(|| AppError::msg("MoE runtime missing"))?;
        let plan = rt
            .packed
            .plan(layer_idx, expert_id)
            .ok_or_else(|| {
                AppError::msg(format!(
                    "missing expert plan layer={layer_idx} expert={expert_id}"
                ))
            })?
            .clone();
        if rt.reader.has_in_flight() {
            rt.reader.wait_completion()?;
        }
        let buf = rt.ring.get_mut(slot)?;
        rt.reader
            .submit_read(buf, slot, plan.read_offset, plan.read_len)?;
        rt.reader.wait_completion()?;
        Ok(())
    }

    fn load_expert_mlp(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
        slot: usize,
        use_gelu: bool,
    ) -> Result<Mlp> {
        // Ensure this expert is in the ring slot.
        self.dma_expert(layer_idx, expert_id, slot)?;
        let rt = self
            .moe_runtime
            .as_ref()
            .ok_or_else(|| AppError::msg("MoE runtime missing"))?;
        let plan = rt.packed.plan(layer_idx, expert_id).ok_or_else(|| {
            AppError::msg(format!(
                "missing expert plan layer={layer_idx} expert={expert_id}"
            ))
        })?;
        let dma = rt.ring.get(slot)?.as_slice();
        materialize_expert_mlp(plan, dma, &self.device, use_gelu)
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

        let (q, k, v) = {
            let lora = self.lora.get(layer_idx);
            let q = qmatmul_with_lora(&layer.wq, lora.and_then(|l| l.q.as_ref()), x)?;
            let k = qmatmul_with_lora(&layer.wk, lora.and_then(|l| l.k.as_ref()), x)?;
            let v = qmatmul_with_lora(&layer.wv, lora.and_then(|l| l.v.as_ref()), x)?;
            (q, k, v)
        };

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
        let lora = self.lora.get(layer_idx);
        qmatmul_with_lora(&layer.wo, lora.and_then(|l| l.o.as_ref()), &y)
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
    adapter_resident_bytes: usize,
    expert_ring_bytes: usize,
) -> usize {
    if let Some(h) = override_hot {
        return h.min(n_layers);
    }
    if n_layers == 0 || layer_bytes == 0 {
        return 0;
    }
    let budget = budget_mib.saturating_mul(1024 * 1024);
    // Reserve two prefetch slots + ~512 MiB headroom (KV / runtime) + adapter + MoE ring.
    let reserve = layer_bytes
        .saturating_mul(2)
        .saturating_add(512 * 1024 * 1024)
        .saturating_add(adapter_resident_bytes)
        .saturating_add(expert_ring_bytes);
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

    let ff = if try_tensor(plan, "ffn_gate.weight").is_some() {
        FeedForward::Dense(Mlp {
            gate: QMatMul::from_qtensor(q("ffn_gate.weight")?)?,
            down: QMatMul::from_qtensor(q("ffn_down.weight")?)?,
            up: QMatMul::from_qtensor(q("ffn_up.weight")?)?,
            use_gelu: gemma,
        })
    } else if try_tensor(plan, "ffn_gate_inp.weight").is_some() {
        let n_used = map
            .moe
            .as_ref()
            .map(|m| m.expert_used_count)
            .unwrap_or(2);
        FeedForward::MoE {
            router: QMatMul::from_qtensor(q("ffn_gate_inp.weight")?)?,
            n_expert_used: n_used,
            use_gelu: gemma,
        }
    } else {
        return Err(AppError::msg(format!(
            "layer {} has neither dense FFN nor MoE router (ffn_gate / ffn_gate_inp)",
            plan.index
        )));
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
        ff,
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

fn materialize_expert_mlp(
    plan: &ExpertDmaPlan,
    dma: &[u8],
    device: &Device,
    use_gelu: bool,
) -> Result<Mlp> {
    let find = |role: &str| -> Result<&TensorLoc> {
        let suffix = format!("ffn_{role}.{}.weight", plan.expert_id);
        plan.tensors
            .iter()
            .find(|t| t.name.ends_with(&suffix) || t.name.contains(&format!("ffn_{role}.")))
            .or_else(|| {
                // Packed names are canonical `blk.L.ffn_ROLE.E.weight`.
                let full = format!("blk.{}.ffn_{}.{}.weight", plan.layer_index, role, plan.expert_id);
                plan.tensors.iter().find(|t| t.name == full)
            })
            .ok_or_else(|| {
                AppError::msg(format!(
                    "expert L{} E{} missing ffn_{role}",
                    plan.layer_index, plan.expert_id
                ))
            })
    };
    let gate = QMatMul::from_qtensor(qtensor_from_loc(find("gate")?, dma, device)?)?;
    let up = QMatMul::from_qtensor(qtensor_from_loc(find("up")?, dma, device)?)?;
    let down = QMatMul::from_qtensor(qtensor_from_loc(find("down")?, dma, device)?)?;
    Ok(Mlp {
        gate,
        up,
        down,
        use_gelu,
    })
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
