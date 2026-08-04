//! Hybrid streaming inference tuned for 16 GiB hosts:
//! 1. Pack GGUF layers into a dense sidecar → one DMA / layer
//! 2. Keep the first N layers resident (hot ratio)
//! 3. Double-buffer the rest via io_uring while computing
//! 4. Track I/O vs compute to keep overlap healthy

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

use candle_core::quantized::{ggml_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{ops, Embedding};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use crate::adapter::{AdapterSet, LayerLora, LoraDelta};
use crate::device::ComputeContext;
use crate::error::{AppError, Result};
use crate::io::gguf_map::{qtensor_from_loc, GgufLayerMap, LayerDmaPlan, TensorLoc};
use crate::io::moe::{ExpertDmaPlan, MoeInfo};
use crate::io::nvme::AsyncNvmeReader;
use crate::io::pack::{ensure_experts_packed, ensure_packed, PackedExperts};
use crate::io::prefetch::{PrefetchBufferManager, PrefetchRing};
use crate::progress;

const MAX_SEQ_LEN: usize = 4096;

/// Tunables for hybrid memory / SSD balance.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Soft RAM budget for weights (hot layers + 2 prefetch slots), MiB.
    pub ram_budget_mib: usize,
    /// Force hot layer count; `None` = derive from budget.
    pub hot_layers: Option<usize>,
    /// Retained for config compatibility; REPL generation uses `--max-tokens`.
    #[allow(dead_code)]
    pub first_burst_tokens: usize,
    /// Extra resident bytes reserved for a bound LoRA adapter (deducted from hot budget).
    pub adapter_resident_bytes: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            ram_budget_mib: 4096,
            hot_layers: None,
            first_burst_tokens: 0,
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
    fn forward(
        &self,
        compute: &ComputeContext,
        xs: &Tensor,
        lora: Option<&LayerLora>,
    ) -> Result<Tensor> {
        let gate = qmm(compute, &self.gate, lora.and_then(|l| l.gate.as_ref()), xs)?;
        let lhs = if self.use_gelu {
            gate.gelu()?
        } else {
            candle_nn::ops::silu(&gate)?
        };
        let rhs = qmm(compute, &self.up, lora.and_then(|l| l.up.as_ref()), xs)?;
        let mid = (lhs * rhs)?;
        qmm(compute, &self.down, lora.and_then(|l| l.down.as_ref()), &mid)
    }

    fn warm_q4k(&self, compute: &ComputeContext) {
        compute.warm_q4k(&self.gate);
        compute.warm_q4k(&self.up);
        compute.warm_q4k(&self.down);
    }

    fn clone_handles(&self) -> Self {
        Self {
            gate: self.gate.clone(),
            up: self.up.clone(),
            down: self.down.clone(),
            use_gelu: self.use_gelu,
        }
    }
}

/// RAM LRU of materialized MoE experts — avoids re-reading `experts.pack` every token.
struct ExpertLru {
    map: HashMap<(usize, usize), Mlp>,
    order: VecDeque<(usize, usize)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl ExpertLru {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: (usize, usize)) -> Option<Mlp> {
        if !self.map.contains_key(&key) {
            self.misses = self.misses.saturating_add(1);
            return None;
        }
        self.hits = self.hits.saturating_add(1);
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
        self.map.get(&key).map(|m| m.clone_handles())
    }

    fn insert(&mut self, key: (usize, usize), mlp: Mlp) {
        if self.map.contains_key(&key) {
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
            self.map.insert(key, mlp);
            return;
        }
        while self.map.len() >= self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            } else {
                break;
            }
        }
        self.order.push_back(key);
        self.map.insert(key, mlp);
    }
}

fn qmm(
    compute: &ComputeContext,
    w: &QMatMul,
    lora: Option<&LoraDelta>,
    x: &Tensor,
) -> Result<Tensor> {
    let y = compute.qmatmul(w, x)?;
    match lora {
        None => Ok(y),
        Some(d) => Ok((y + d.forward(x)?)?),
    }
}

/// Dense MLP or MoE block (router resident; experts DMA'd on demand).
enum FeedForward {
    Dense(Mlp),
    MoE {
        router: QMatMul,
        /// Gemma 4: `ffn_gate_inp.scale` (elementwise before router).
        router_scale: Option<Tensor>,
        /// Gemma 4 shared expert (dense `ffn_gate/up/down`); absent on Mixtral/Qwen.
        shared: Option<Mlp>,
        /// Gemma 4: `pre_ffw_norm_2` before routed experts.
        pre_ffw_norm_2: Option<Norm>,
        /// Gemma 4: `post_ffw_norm_1` after shared expert.
        post_ffw_norm_1: Option<Norm>,
        /// Gemma 4: `post_ffw_norm_2` after routed experts.
        post_ffw_norm_2: Option<Norm>,
        /// Per-expert down scales (`ffn_down_exps.scale`), length = n_expert.
        expert_down_scales: Option<Vec<f32>>,
        n_expert_used: usize,
        use_gelu: bool,
        /// Special Gemma 4 router: rms_norm(attn_out)/sqrt(n) * scale → logits.
        gemma4_router: bool,
    },
}

/// RMSNorm. GGUF Gemma weights are already converted to full scale `(1+δ)`
/// by the HF→GGUF exporter, so we always multiply by `w` (never `1+w` again).
#[derive(Clone)]
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
        // Candle `rms_norm` only accepts F32 (and matching weight dtype).
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.clone()
        };
        let w = if self.weight.dtype() != DType::F32 {
            self.weight.to_dtype(DType::F32)?
        } else {
            self.weight.clone()
        };
        Ok(ops::rms_norm(&xs, &w, self.eps as f32)?)
    }
}

struct LayerLive {
    wq: QMatMul,
    wk: QMatMul,
    /// Absent on some Gemma 4 global layers (`attention_k_eq_v`).
    wv: Option<QMatMul>,
    wo: QMatMul,
    attn_norm: Norm,
    ffn_norm: Norm,
    post_attention_norm: Option<Norm>,
    post_ffw_norm: Option<Norm>,
    /// Gemma 3/4 QK-Norm (per-head dim).
    attn_q_norm: Option<Norm>,
    attn_k_norm: Option<Norm>,
    /// Gemma 4 `layer_output_scale`.
    layer_output_scale: Option<Tensor>,
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
    /// Gemma 3 local-attention RoPE (period-local layers); falls back to `cos`/`sin`.
    cos_local: Option<Tensor>,
    sin_local: Option<Tensor>,
    neg_inf: Tensor,
    kv_cache: Vec<Option<(Tensor, Tensor)>>,
    /// Masks keyed by `(seq_len, window)` where `window == 0` means full causal.
    masks: HashMap<(usize, usize), Tensor>,
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
    /// Phase 9 compute backend (CPU / CUDA / Vulkan QMatMul offload).
    compute: ComputeContext,
    device_label: String,
    /// When true, `forward_hidden` prints per-layer prefill progress on stderr.
    report_prefill: bool,
    /// MoE expert MLP LRU (RAM); survives KV resets across chat turns.
    expert_cache: ExpertLru,
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
        compute: ComputeContext,
    ) -> Result<Self> {
        let path = path.as_ref();
        let pack_cache = pack_cache.as_ref();
        let mut map = GgufLayerMap::open(path).map_err(|e| AppError::msg(e.to_string()))?;
        let device = compute.device().clone();
        let device_label = format!("{}+pack+io_uring", compute.label());
        let phases: u32 = if map.moe.is_some() { 5 } else { 4 };
        eprintln!("compute backend: {}", compute.label());

        if map.embedding_length == 0 || map.head_count == 0 {
            return Err(AppError::msg(format!(
                "incomplete GGUF metadata for {} (arch={})",
                path.display(),
                map.architecture
            )));
        }

        // --- (大) pack rearrange (engine cache; GGUF in blobs/ untouched) ---
        progress::phase(1, phases, "ensuring layer / expert packs …");
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

        // Estimate always-resident bytes (embeddings dequantized below; use Q size ×2 as f16 floor).
        let emb_q_bytes = map
            .hot
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| t.size_bytes)
            .unwrap_or(0);
        // f16 dequant of large emb ≈ 2× Q8 / roughly vocab*dim*2; use max(q, rough f16).
        let emb_f16_est = map.embedding_length.saturating_mul(262_144).saturating_mul(2);
        let always_resident_est = emb_q_bytes
            .saturating_mul(2)
            .max(if map.is_gemma4() {
                emb_f16_est
            } else {
                emb_q_bytes
            })
            .saturating_add(64 * 1024 * 1024); // norms / lm_head / KV headroom beyond the 512 MiB reserve

        let hot_count = choose_hot_layers(
            map.layers.len(),
            packed.max_layer_bytes,
            config.ram_budget_mib,
            config.hot_layers,
            config.adapter_resident_bytes,
            packed_experts
                .as_ref()
                .map(|p| {
                    p.recommended_slot_bytes()
                        .saturating_mul(p.moe.expert_used_count.max(2))
                })
                .unwrap_or(0),
            always_resident_est,
        );

        progress::phase(2, phases, "allocating prefetch arenas / MoE ring …");
        let slot = packed.recommended_slot_bytes();
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

        progress::phase(
            3,
            phases,
            &format!(
                "loading embeddings ({:.0} MiB quantized → dequant) …",
                tok_loc.size_bytes as f64 / (1024.0 * 1024.0)
            ),
        );
        let tok_q = read_tensor_from_file(&mut file, tok_loc, &device)?;
        let emb_dim = map.embedding_length;
        // Large-vocab MoE (Gemma 4): keep embeddings in f16 to stay under --ram-mib.
        progress::note("dequantizing token embeddings …");
        let emb_tensor = tok_q.dequantize(&device)?;
        let emb_tensor = if map.is_gemma4() || emb_tensor.elem_count() > 200_000_000 {
            emb_tensor.to_dtype(DType::F16)?
        } else {
            emb_tensor
        };
        let emb_resident = emb_tensor.elem_count()
            * match emb_tensor.dtype() {
                DType::F16 | DType::BF16 => 2,
                _ => 4,
            };
        let embeddings = Embedding::new(emb_tensor, emb_dim);
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
        let (cos_local, sin_local) = if map.is_gemma3() || map.is_gemma4() {
            let local_base = map.rope_freq_base_local.unwrap_or(10_000.0);
            let local_dim = map.head_dim_local.unwrap_or(map.head_dim);
            let (c, s) = precompute_rope(local_dim, local_base, &device)?;
            (Some(c), Some(s))
        } else {
            (None, None)
        };
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)?;
        let n_layers = packed.layers.len();

        if map.has_vision_tensors() {
            eprintln!(
                "warning: GGUF includes vision/projector tensors — Phase 12 text-only; \
                 image input is not implemented (language weights still run)"
            );
        }
        if map.is_gemma3() || map.is_gemma4() {
            let n_local = map.layer_is_sliding.iter().filter(|&&s| s).count();
            let n_global = n_layers.saturating_sub(n_local);
            let label = if map.is_gemma4() { "gemma4" } else { "gemma3" };
            eprintln!(
                "{label}: sliding_window={:?} local_layers={n_local} global_layers={n_global} \
                 rope_global={} rope_local={} head_dim={}/{:?}",
                map.sliding_window,
                map.rope_freq_base,
                map.rope_freq_base_local.unwrap_or(10_000.0),
                map.head_dim,
                map.head_dim_local
            );
            if map.is_gemma4() {
                if let Some(ref moe) = map.moe {
                    eprintln!(
                        "gemma4 MoE: experts={} top-k={} shared={} layout={:?} \
                         target resident ≤ --ram-mib (experts on NVMe)",
                        moe.expert_count,
                        moe.expert_used_count,
                        moe.has_shared_expert,
                        moe.layout
                    );
                }
                eprintln!(
                    "hint: gemma4:26b-a4b — prefer `--hybrid --ram-mib 16384`; \
                     ctx capped at {MAX_SEQ_LEN} for this build (full 256K later)"
                );
            } else if n_layers >= 40 {
                eprintln!(
                    "hint: gemma3 large — prefer `--ram-mib 16384`+ (or `--hot-layers N`) so \
                     fewer layers stream from NVMe; ctx capped at {MAX_SEQ_LEN} for this build"
                );
            }
        }

        let emb_mib = emb_resident as f64 / (1024.0 * 1024.0);
        eprintln!(
            "hybrid resident estimate: emb≈{emb_mib:.0} MiB + hot_layers={hot_count} \
             + 2×slot + MoE ring (budget {} MiB)",
            config.ram_budget_mib
        );

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
            progress::phase(
                4,
                phases,
                &format!("pinning hot layers 0..{hot_count} into RAM …"),
            );
            let mut pack_file = File::open(&packed.pack_path)?;
            let mut pin_prog = progress::Counter::start("pinned layer", hot_count);
            for i in 0..hot_count {
                let plan = &packed.layers[i];
                let mut buf = vec![0u8; plan.read_len];
                pack_file.seek(SeekFrom::Start(plan.read_offset))?;
                // Pack is 4K-aligned; buffered read is fine for one-time pin.
                let n = plan.payload_bytes.min(plan.read_len);
                pack_file.read_exact(&mut buf[..n])?;
                // Zero pad remainder already in vec.
                let layer = materialize_layer(plan, &buf, &map, &device)?;
                warm_layer_q4k(&compute, &layer);
                resident[i] = Some(layer);
                pin_prog.tick();
            }
        } else {
            progress::phase(4, phases, "no hot layers to pin (all streamed)");
        }

        let expert_slot_bytes = moe_runtime
            .as_ref()
            .map(|m| m.packed.recommended_slot_bytes())
            .unwrap_or(4 * 1024 * 1024);
        let expert_cache_cap = choose_expert_cache_cap(
            config.ram_budget_mib,
            always_resident_est,
            hot_count,
            packed.max_layer_bytes,
            expert_slot_bytes,
            moe_runtime.is_some(),
        );
        if moe_runtime.is_some() {
            progress::note(&format!(
                "MoE expert RAM cache capacity={expert_cache_cap} (~{:.0} MiB); \
                 first turn warms cache, later turns reuse",
                expert_cache_cap as f64 * expert_slot_bytes as f64 / (1024.0 * 1024.0)
            ));
        }

        let stream_count = n_layers.saturating_sub(hot_count);
        progress::phase(
            phases,
            phases,
            &format!(
                "ready — arch={} layers={} hot={} stream={}",
                map.architecture, n_layers, hot_count, stream_count
            ),
        );
        eprintln!(
            "hybrid: arch={} layers={} hot={} stream={} slot={} MiB softcap={:?}/{:?} pack={}",
            map.architecture,
            n_layers,
            hot_count,
            stream_count,
            slot / (1024 * 1024),
            map.attn_logit_softcapping,
            map.final_logit_softcapping,
            packed.pack_path.display()
        );
        if stream_count > 0 {
            eprintln!(
                "hint: {stream_count} layers stream from pack each forward — \
                 for lower latency try `--hot-layers {n_layers}` or raise `--ram-mib` \
                 (and `ulimit -l` if mlock failed)"
            );
        }
        if let Some(sw) = map.sliding_window {
            if map.is_gemma3() || map.is_gemma4() {
                eprintln!(
                    "hybrid: sliding_window={sw} (Gemma local layers; global layers use full causal)"
                );
            } else {
                eprintln!("hybrid: sliding_window={sw} (short prompts use full causal mask)");
            }
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
            cos_local,
            sin_local,
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
            compute,
            device_label,
            report_prefill: false,
            expert_cache: ExpertLru::new(expert_cache_cap),
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

    #[allow(dead_code)]
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    pub fn architecture(&self) -> &str {
        &self.map.architecture
    }

    pub fn device_name(&self) -> &str {
        &self.device_label
    }

    pub fn reset_state(&mut self) {
        for slot in &mut self.kv_cache {
            *slot = None;
        }
        self.masks.clear();
    }

    pub fn n_layers(&self) -> usize {
        self.packed_layers.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Replace per-layer LoRA slots (used by `adapter create` training).
    pub fn set_lora_layers(&mut self, layers: Vec<LayerLora>) {
        let n = self.packed_layers.len();
        let mut lora = layers;
        if lora.len() < n {
            lora.resize_with(n, LayerLora::default);
        } else if lora.len() > n {
            lora.truncate(n);
        }
        self.lora = lora;
    }

    /// `(out_features, in_features)` for a dense projection weight in `blk.{i}.*`.
    pub fn projection_dims(
        &self,
        layer_idx: usize,
        weight_suffix: &str,
    ) -> Result<(usize, usize)> {
        let plan = self.packed_layers.get(layer_idx).ok_or_else(|| {
            AppError::msg(format!("layer {layer_idx} out of range"))
        })?;
        let loc = require_tensor(plan, weight_suffix)?;
        match loc.shape.as_slice() {
            [out_f, in_f] => Ok((*out_f, *in_f)),
            other => Err(AppError::msg(format!(
                "unexpected shape for blk.{layer_idx}.{weight_suffix}: {other:?}"
            ))),
        }
    }

    pub fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        mut on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<crate::engine::GenerateOutcome> {
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

        let n_layers = self.packed_layers.len();
        progress::phase(
            1,
            2,
            &format!(
                "prefill {} prompt tokens × {} layers{}",
                tokens.len(),
                n_layers,
                if self.moe_runtime.is_some() {
                    " (MoE Top-K DMA; first token can take minutes)"
                } else {
                    ""
                }
            ),
        );
        self.report_prefill = true;
        let t_prefill = Instant::now();
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = prepare_logits(self.forward(&input, 0)?)?;
        self.report_prefill = false;
        let prefill_s = t_prefill.elapsed().as_secs_f64();
        progress::phase(
            2,
            2,
            &format!(
                "prefill done in {prefill_s:.1}s — generating up to {max_tokens} tokens \
                 (expert cache {}/{} hits={}/misses={})",
                self.expert_cache.map.len(),
                self.expert_cache.capacity,
                self.expert_cache.hits,
                self.expert_cache.misses
            ),
        );
        if cfg!(debug_assertions) && prefill_s > 30.0 {
            eprintln!(
                "hint: debug build is slow — use `cargo build --release` and \
                 `./target/release/lpc-llm run …` for conversation-speed decode"
            );
        }

        let mut generated = String::new();
        let mut tokens_generated = 0usize;
        let mut hit_eos = false;
        for _ in 0..max_tokens {
            let next = logits_processor.sample(&logits)?;
            tokens.push(next);
            tokens_generated += 1;
            let piece = tokenizer
                .decode(&[next], true)
                .map_err(|e| AppError::msg(format!("decode: {e}")))?;
            on_token(&piece)?;
            generated.push_str(&piece);
            if eos.contains(&next) {
                hit_eos = true;
                break;
            }
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = prepare_logits(self.forward(&input, tokens.len() - 1)?)?;
        }
        Ok(crate::engine::GenerateOutcome {
            text: generated,
            hit_eos,
            tokens_generated,
        })
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
        let xs = self.forward_hidden(x, index_pos)?;
        let xs = xs.i((.., seq_len - 1, ..))?;
        let logits = self.compute.qmatmul(&self.output, &xs)?;
        self.apply_final_softcap(logits)
    }

    /// Full-sequence logits for LoRA SFT (`adapter create`).
    ///
    /// Uses a dequantized / f16 matmul lm_head so gradients reach the LoRA
    /// side-path (quantized `QMatMul` is `no_bwd`).
    pub fn forward_train(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(AppError::msg("forward_train: empty token sequence"));
        }
        self.reset_state();
        let x = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let xs = self.forward_hidden(&x, 0)?;
        let logits = self.output.forward_via_f16(&xs)?;
        self.apply_final_softcap(logits)
    }

    fn apply_final_softcap(&self, logits: Tensor) -> Result<Tensor> {
        match self.map.final_logit_softcapping {
            Some(sc) if sc > 0.0 => Ok(((logits / sc)?.tanh()? * sc)?),
            _ => Ok(logits),
        }
    }

    /// Hidden states after `output_norm`, shape `[B, T, C]`.
    fn forward_hidden(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = x.dims2()?;

        let mut xs = self.embeddings.forward(x)?;
        // Embeddings may be stored as F16 to save RAM; Candle rms_norm / most
        // matmuls expect F32 activations.
        if xs.dtype() != DType::F32 {
            xs = xs.to_dtype(DType::F32)?;
        }
        // Gemma/Gemma2/Gemma4: scale embeddings by √hidden.
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
            // Per-layer mask: Gemma3 local layers use sliding-window+causal.
            let mask = if seq_len == 1 {
                None
            } else {
                Some(self.mask_for_layer(i, seq_len)?)
            };

            if i < self.hot_count {
                // Resident path — no I/O.
                // Safety: temporarily take layer, run, put back.
                let layer = self.resident[i]
                    .take()
                    .ok_or_else(|| AppError::msg(format!("hot layer {i} missing")))?;
                xs = self.forward_one_layer(i, &layer, &xs, mask.as_ref(), index_pos)?;
                self.resident[i] = Some(layer);
                if self.report_prefill {
                    eprint!("\r  prefill layer {}/{} …", i + 1, n);
                    let _ = std::io::stderr().flush();
                }
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
            if self.report_prefill {
                eprint!("\r  prefill layer {}/{} …", i + 1, n);
                let _ = std::io::stderr().flush();
            }
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

        if self.report_prefill {
            eprintln!();
        }

        self.output_norm.forward(&xs)
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
        let x = {
            let lora = self.lora.get(layer_idx);
            match &layer.ff {
                FeedForward::Dense(mlp) => {
                    let x = layer.ffn_norm.forward(residual)?;
                    mlp.forward(&self.compute, &x, lora)?
                }
                FeedForward::MoE {
                    gemma4_router: true,
                    ..
                } => self.forward_moe_gemma4(layer_idx, layer, residual)?,
                FeedForward::MoE {
                    router,
                    n_expert_used,
                    use_gelu,
                    gemma4_router: false,
                    ..
                } => {
                    let x = layer.ffn_norm.forward(residual)?;
                    self.forward_moe(layer_idx, router, *n_expert_used, *use_gelu, &x)?
                }
            }
        };
        let x = match &layer.post_ffw_norm {
            Some(n) => n.forward(&x)?,
            None => x,
        };
        let mut out = (x + residual)?;
        if let Some(scale) = &layer.layer_output_scale {
            out = out.broadcast_mul(scale)?;
        }
        Ok(out)
    }

    /// Gemma 4 MoE: shared dense expert ∥ Top-K routed experts (llama.cpp gemma4 graph).
    fn forward_moe_gemma4(
        &mut self,
        layer_idx: usize,
        layer: &LayerLive,
        attn_out: &Tensor,
    ) -> Result<Tensor> {
        let (
            shared,
            router,
            router_scale,
            pre_ffw_norm_2,
            post_ffw_norm_1,
            post_ffw_norm_2,
            down_scales,
            n_expert_used,
            use_gelu,
        ) = match &layer.ff {
            FeedForward::MoE {
                router,
                router_scale,
                shared: Some(shared),
                pre_ffw_norm_2,
                post_ffw_norm_1,
                post_ffw_norm_2,
                expert_down_scales,
                n_expert_used,
                use_gelu,
                gemma4_router: true,
            } => (
                // Clone MLP handles (QMatMul is Arc-backed / cheap).
                Mlp {
                    gate: shared.gate.clone(),
                    up: shared.up.clone(),
                    down: shared.down.clone(),
                    use_gelu: shared.use_gelu,
                },
                router.clone(),
                router_scale.clone(),
                pre_ffw_norm_2.clone(),
                post_ffw_norm_1.clone(),
                post_ffw_norm_2.clone(),
                expert_down_scales.clone(),
                *n_expert_used,
                *use_gelu,
            ),
            _ => {
                return Err(AppError::msg(
                    "forward_moe_gemma4: not a Gemma4 MoE layer with shared expert",
                ))
            }
        };
        let ffn_norm = Norm {
            weight: layer.ffn_norm.weight.clone(),
            eps: layer.ffn_norm.eps,
        };

        // Shared expert path.
        let mut cur_mlp = ffn_norm.forward(attn_out)?;
        cur_mlp = shared.forward(&self.compute, &cur_mlp, None)?;
        if let Some(n) = &post_ffw_norm_1 {
            cur_mlp = n.forward(&cur_mlp)?;
        }

        // Routed experts path.
        let mut cur_moe = match &pre_ffw_norm_2 {
            Some(n) => n.forward(attn_out)?,
            None => attn_out.clone(),
        };

        let (b_size, seq_len, hidden_dim) = attn_out.dims3()?;
        let flat = attn_out.reshape(((), hidden_dim))?;
        let ones = Tensor::ones(hidden_dim, DType::F32, &self.device)?;
        let mut tmp = ops::rms_norm(&flat, &ones, self.map.rms_norm_eps as f32)?;
        let scale = 1.0f64 / (hidden_dim as f64).sqrt();
        tmp = (tmp * scale)?;
        if let Some(rs) = &router_scale {
            tmp = tmp.broadcast_mul(rs)?;
        }
        let router_logits = self.compute.qmatmul(&router, &tmp)?;
        let routing = ops::softmax_last_dim(&router_logits)?;
        let routing_vec = routing.to_vec2::<f32>()?;

        let n_expert = routing_vec.first().map(|r| r.len()).unwrap_or(0);
        let mut top_x: Vec<Vec<u32>> = vec![Vec::new(); n_expert];
        let mut selected_rws: Vec<Vec<f32>> = vec![Vec::new(); n_expert];
        let hints = self.expert_prefetch_hints.clone();

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

        let xs_flat = cur_moe.reshape(((), hidden_dim))?;
        let mut ys = xs_flat.zeros_like()?;
        let active: Vec<usize> = top_x
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v.is_empty() { None } else { Some(i) })
            .collect();

        for (ai, &expert_id) in active.iter().enumerate() {
            let slot = ai
                % self
                    .moe_runtime
                    .as_ref()
                    .map(|m| m.ring.len())
                    .unwrap_or(2);
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
            let mut out = mlp.forward(&self.compute, &indexed, None)?;
            if let Some(ref scales) = down_scales {
                if let Some(&s) = scales.get(expert_id) {
                    out = (out * f64::from(s))?;
                }
            }
            let rw = Tensor::new(selected_rws[expert_id].as_slice(), &self.device)?
                .reshape((row_ids.len(), 1))?;
            let weighted = out.broadcast_mul(&rw)?;
            ys = ys.index_add(&index, &weighted, 0)?;
        }

        cur_moe = ys.reshape((b_size, seq_len, hidden_dim))?;
        if let Some(n) = &post_ffw_norm_2 {
            cur_moe = n.forward(&cur_moe)?;
        }

        Ok((cur_mlp + cur_moe)?)
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
        let router_logits = self.compute.qmatmul(router, &xs_flat)?;
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
            let out = mlp.forward(&self.compute, &indexed, None)?;
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
        let key = (layer_idx, expert_id);
        if let Some(cached) = self.expert_cache.get(key) {
            return Ok(cached);
        }
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
        let mlp = materialize_expert_mlp(plan, dma, &self.device, use_gelu)?;
        mlp.warm_q4k(&self.compute);
        self.expert_cache.insert(key, mlp.clone_handles());
        Ok(mlp)
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
        let n_kv = self.map.head_count_kv_for_layer(layer_idx);
        let head_dim = self.map.head_dim_for_layer(layer_idx);
        let sliding = self.map.layer_sliding(layer_idx);
        let window = self.map.sliding_window.unwrap_or(0);

        let (q, k, v) = {
            let lora = self.lora.get(layer_idx);
            let q = qmm(&self.compute, &layer.wq, lora.and_then(|l| l.q.as_ref()), x)?;
            let k = qmm(&self.compute, &layer.wk, lora.and_then(|l| l.k.as_ref()), x)?;
            let v = match &layer.wv {
                Some(wv) => qmm(&self.compute, wv, lora.and_then(|l| l.v.as_ref()), x)?,
                None => k.clone(), // Gemma 4 attention_k_eq_v
            };
            (q, k, v)
        };

        let mut q = q
            .reshape((b_sz, seq_len, n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, seq_len, n_kv, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, n_kv, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Gemma 3/4: QK-Norm before RoPE (replaces Gemma 2 attn softcap).
        if let Some(n) = &layer.attn_q_norm {
            q = n.forward(&q)?;
        }
        if let Some(n) = &layer.attn_k_norm {
            k = n.forward(&k)?;
        }
        // Gemma 4: V also gets RMSNorm (unparameterized) when present as separate proj.
        if self.map.is_gemma4() && layer.wv.is_some() {
            let v_f32 = if v.dtype() != DType::F32 {
                v.to_dtype(DType::F32)?
            } else {
                v
            };
            let ones = Tensor::ones(head_dim, DType::F32, &self.device)?;
            v = ops::rms_norm(&v_f32, &ones, self.map.rms_norm_eps as f32)?;
        }

        let (rope_cos, rope_sin) = if sliding {
            (
                self.cos_local.as_ref().unwrap_or(&self.cos),
                self.sin_local.as_ref().unwrap_or(&self.sin),
            )
        } else {
            (&self.cos, &self.sin)
        };
        let rope_dims = if sliding {
            None
        } else {
            self.map.rope_dim_global
        };
        let q = apply_rope_maybe_partial(
            &q,
            rope_cos,
            rope_sin,
            index_pos,
            self.map.is_gemma_family(),
            rope_dims,
        )?;
        let k = apply_rope_maybe_partial(
            &k,
            rope_cos,
            rope_sin,
            index_pos,
            self.map.is_gemma_family(),
            rope_dims,
        )?;

        let (mut k, mut v) = match &self.kv_cache[layer_idx] {
            None => (k, v),
            Some((_kc, _vc)) if index_pos == 0 => (k, v),
            Some((kc, vc)) => {
                let k = Tensor::cat(&[kc, &k], 2)?;
                let v = Tensor::cat(&[vc, &v], 2)?;
                (k, v)
            }
        };

        // Gemma 3 local layers: keep only the last `window` KV positions.
        if sliding && window > 0 {
            let kv_len = k.dim(2)?;
            if kv_len > window {
                let start = kv_len - window;
                k = k.narrow(2, start, window)?;
                v = v.narrow(2, start, window)?;
            }
        }
        self.kv_cache[layer_idx] = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, n_head / n_kv.max(1))?;
        let v = repeat_kv(v, n_head / n_kv.max(1))?;

        let scale = self
            .map
            .attention_scale
            .unwrap_or(1.0 / (head_dim as f64).sqrt());
        let mut att = (q.matmul(&k.t()?)? * scale)?;
        // Gemma 2 softcap only when present (Gemma 3/4 use QK-Norm instead).
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
        qmm(&self.compute, &layer.wo, lora.and_then(|l| l.o.as_ref()), &y)
    }

    fn mask_for_layer(&mut self, layer_idx: usize, t: usize) -> Result<Tensor> {
        let window = if self.map.layer_sliding(layer_idx) {
            self.map.sliding_window.unwrap_or(0)
        } else {
            0
        };
        self.mask(t, window)
    }

    /// Causal mask; when `window > 0`, also mask keys outside the sliding window.
    /// Mask value `1` means "masked out" (filled with -inf).
    fn mask(&mut self, t: usize, window: usize) -> Result<Tensor> {
        let key = (t, window);
        if let Some(m) = self.masks.get(&key) {
            return Ok(m.clone());
        }
        let mask: Vec<_> = (0..t)
            .flat_map(|i| {
                (0..t).map(move |j| {
                    let future = j > i;
                    let out_of_window = window > 0 && i >= window && j + window <= i;
                    u8::from(future || out_of_window)
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
        self.masks.insert(key, mask.clone());
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
    always_resident_bytes: usize,
) -> usize {
    if let Some(h) = override_hot {
        return h.min(n_layers);
    }
    if n_layers == 0 || layer_bytes == 0 {
        return 0;
    }
    let budget = budget_mib.saturating_mul(1024 * 1024);
    // Reserve two prefetch slots + ~512 MiB headroom (KV / runtime) + adapter + MoE ring
    // + always-resident emb/lm_head.
    let reserve = layer_bytes
        .saturating_mul(2)
        .saturating_add(512 * 1024 * 1024)
        .saturating_add(adapter_resident_bytes)
        .saturating_add(expert_ring_bytes)
        .saturating_add(always_resident_bytes);
    let hot_budget = budget.saturating_sub(reserve);
    let by_ram = hot_budget / layer_bytes;
    // Keep as many layers resident as the RAM budget allows. The old hard
    // cap of 8 forced NVMe streaming on small models (e.g. gemma2:2b) and
    // dominated latency even when --ram-mib had headroom.
    by_ram.min(n_layers)
}

/// How many MoE experts to keep materialized in RAM under `--ram-mib`.
fn choose_expert_cache_cap(
    budget_mib: usize,
    always_resident_bytes: usize,
    hot_count: usize,
    layer_bytes: usize,
    expert_slot_bytes: usize,
    has_moe: bool,
) -> usize {
    if !has_moe || expert_slot_bytes == 0 {
        return 1;
    }
    let budget = budget_mib.saturating_mul(1024 * 1024);
    let pinned = always_resident_bytes
        .saturating_add(hot_count.saturating_mul(layer_bytes))
        .saturating_add(512 * 1024 * 1024)
        .saturating_add(expert_slot_bytes.saturating_mul(8));
    let spare = budget.saturating_sub(pinned);
    // Leave ~2 GiB OS/KV headroom inside the spare.
    let for_cache = spare.saturating_sub(2 * 1024 * 1024 * 1024);
    let cap = for_cache / expert_slot_bytes;
    cap.clamp(64, 2048)
}

fn warm_layer_q4k(compute: &ComputeContext, layer: &LayerLive) {
    compute.warm_q4k(&layer.wq);
    compute.warm_q4k(&layer.wk);
    if let Some(ref wv) = layer.wv {
        compute.warm_q4k(wv);
    }
    compute.warm_q4k(&layer.wo);
    match &layer.ff {
        FeedForward::Dense(mlp) => mlp.warm_q4k(compute),
        FeedForward::MoE {
            router, shared, ..
        } => {
            compute.warm_q4k(router);
            if let Some(s) = shared {
                s.warm_q4k(compute);
            }
        }
    }
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
    let opt_q = |suffix: &str| -> Result<Option<QTensor>> {
        match try_tensor(plan, suffix) {
            Some(loc) => Ok(Some(qtensor_from_loc(loc, dma, device)?)),
            None => Ok(None),
        }
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

    let ff = if try_tensor(plan, "ffn_gate_inp.weight").is_some() {
        let n_used = map
            .moe
            .as_ref()
            .map(|m| m.expert_used_count)
            .unwrap_or(2);
        let gemma4 = map.is_gemma4()
            || map
                .moe
                .as_ref()
                .map(|m| m.family == crate::io::moe::MoeFamily::Gemma4)
                .unwrap_or(false);
        let shared = if try_tensor(plan, "ffn_gate.weight").is_some() {
            Some(Mlp {
                gate: QMatMul::from_qtensor(q("ffn_gate.weight")?)?,
                down: QMatMul::from_qtensor(q("ffn_down.weight")?)?,
                up: QMatMul::from_qtensor(q("ffn_up.weight")?)?,
                use_gelu: gemma,
            })
        } else {
            None
        };
        let router_scale = match try_tensor(plan, "ffn_gate_inp.scale") {
            Some(loc) => {
                let t = qtensor_from_loc(loc, dma, device)?.dequantize(device)?;
                Some(t)
            }
            None => None,
        };
        let expert_down_scales = match try_tensor(plan, "ffn_down_exps.scale") {
            Some(loc) => {
                let t = qtensor_from_loc(loc, dma, device)?.dequantize(device)?;
                Some(t.flatten_all()?.to_vec1::<f32>()?)
            }
            None => None,
        };
        FeedForward::MoE {
            router: QMatMul::from_qtensor(q("ffn_gate_inp.weight")?)?,
            router_scale,
            shared,
            pre_ffw_norm_2: opt_norm("pre_ffw_norm_2.weight")?,
            post_ffw_norm_1: opt_norm("post_ffw_norm_1.weight")?,
            post_ffw_norm_2: opt_norm("post_ffw_norm_2.weight")?,
            expert_down_scales,
            n_expert_used: n_used,
            use_gelu: gemma,
            gemma4_router: gemma4,
        }
    } else if try_tensor(plan, "ffn_gate.weight").is_some() {
        FeedForward::Dense(Mlp {
            gate: QMatMul::from_qtensor(q("ffn_gate.weight")?)?,
            down: QMatMul::from_qtensor(q("ffn_down.weight")?)?,
            up: QMatMul::from_qtensor(q("ffn_up.weight")?)?,
            use_gelu: gemma,
        })
    } else {
        return Err(AppError::msg(format!(
            "layer {} has neither dense FFN nor MoE router (ffn_gate / ffn_gate_inp)",
            plan.index
        )));
    };

    let wv = match opt_q("attn_v.weight")? {
        Some(t) => Some(QMatMul::from_qtensor(t)?),
        None => None,
    };
    let layer_output_scale = match try_tensor(plan, "layer_output_scale.weight") {
        Some(loc) => {
            let t = qtensor_from_loc(loc, dma, device)?.dequantize(device)?;
            Some(t)
        }
        None => None,
    };

    Ok(LayerLive {
        wq: QMatMul::from_qtensor(q("attn_q.weight")?)?,
        wk: QMatMul::from_qtensor(q("attn_k.weight")?)?,
        wv,
        wo: QMatMul::from_qtensor(q("attn_output.weight")?)?,
        attn_norm: Norm::from_qtensor(q("attn_norm.weight")?, map.rms_norm_eps)?,
        ffn_norm: Norm::from_qtensor(q("ffn_norm.weight")?, map.rms_norm_eps)?,
        post_attention_norm: opt_norm("post_attention_norm.weight")?,
        post_ffw_norm: opt_norm("post_ffw_norm.weight")?,
        attn_q_norm: opt_norm("attn_q_norm.weight")?,
        attn_k_norm: opt_norm("attn_k_norm.weight")?,
        layer_output_scale,
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

#[allow(dead_code)]
fn apply_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    neox: bool,
) -> Result<Tensor> {
    apply_rope_maybe_partial(x, cos, sin, index_pos, neox, None)
}

fn apply_rope_maybe_partial(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    index_pos: usize,
    neox: bool,
    partial_dims: Option<usize>,
) -> Result<Tensor> {
    let (_b, _h, seq_len, head_dim) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?.contiguous()?;
    let sin = sin.narrow(0, index_pos, seq_len)?.contiguous()?;
    let x = x.contiguous()?;
    let n_rot = partial_dims.unwrap_or(head_dim).min(head_dim);
    if n_rot == 0 {
        return Ok(x);
    }
    if n_rot == head_dim {
        // Gemma / Gemma2: rotate-half (Neox) RoPE; llama-family: interleaved.
        if neox {
            Ok(candle_nn::rotary_emb::rope(&x, &cos, &sin)?)
        } else {
            Ok(candle_nn::rotary_emb::rope_i(&x, &cos, &sin)?)
        }
    } else {
        // Partial RoPE (Gemma 4 global): rotate first `n_rot` dims only.
        let x_rot = x.narrow(3, 0, n_rot)?.contiguous()?;
        let x_pass = x.narrow(3, n_rot, head_dim - n_rot)?;
        let n_freq = n_rot / 2;
        let cos_use = cos
            .narrow(1, 0, n_freq.min(cos.dim(1)?))?
            .contiguous()?;
        let sin_use = sin
            .narrow(1, 0, n_freq.min(sin.dim(1)?))?
            .contiguous()?;
        let rotated = if neox {
            candle_nn::rotary_emb::rope(&x_rot, &cos_use, &sin_use)?
        } else {
            candle_nn::rotary_emb::rope_i(&x_rot, &cos_use, &sin_use)?
        };
        Ok(Tensor::cat(&[&rotated, &x_pass], 3)?)
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
        "<turn|>",
        "<eos>",
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
