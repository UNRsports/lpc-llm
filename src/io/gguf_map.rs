//! GGUF → O_DIRECT-aligned per-layer DMA plans.
//!
//! Transformer blocks (`blk.N.*`) are coalesced into one contiguous file span
//! per layer, then expanded to 4 KiB boundaries so [`crate::io::AsyncNvmeReader`]
//! can DMA the whole block into a prefetch arena. Always-hot tensors
//! (embeddings, norms, lm_head) stay outside the ping-pong path.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use candle_core::quantized::gguf_file::{self, TensorInfo};
use candle_core::quantized::GgmlDType;
use candle_core::Device;

use super::error::{IoError, Result};
use super::moe::{
    build_expert_plans_from_locs, classify_block_suffix, fused_role, is_fused_expert_suffix,
    is_per_expert_suffix, slice_fused_expert, slice_fused_expert_trailing, split_block_name,
    split_gate_up_loc, BlockTensorKind, ExpertDmaPlan, MoeFamily, MoeInfo, MoeLayout,
};
use super::prefetch::{align_up, DIRECT_ALIGN};

/// One tensor's location inside a layer DMA window (or hot region).
#[derive(Debug, Clone)]
pub struct TensorLoc {
    pub name: String,
    /// Absolute file offset of the first payload byte.
    pub abs_offset: u64,
    pub size_bytes: usize,
    pub dtype: GgmlDType,
    pub shape: Vec<usize>,
    /// Byte offset from the start of the DMA buffer (`read_offset`).
    pub rel_offset: usize,
}

/// O_DIRECT read window covering all tensors of one transformer block.
#[derive(Debug, Clone)]
pub struct LayerDmaPlan {
    pub index: usize,
    /// Aligned file offset passed to `io_uring` Read (valid when `!sparse`).
    pub read_offset: u64,
    /// Aligned transfer length (valid when `!sparse`).
    pub read_len: usize,
    pub tensors: Vec<TensorLoc>,
    /// Sum of raw tensor sizes (informational).
    pub payload_bytes: usize,
    /// When true, layer tensors are scattered on disk — load via seek+read
    /// instead of one coalesced DMA (avoids multi-hundred-MiB windows).
    pub sparse: bool,
}

/// Full GGUF layout plan for hybrid prefetch.
#[derive(Debug, Clone)]
pub struct GgufLayerMap {
    pub path: std::path::PathBuf,
    pub architecture: String,
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_freq_base: f32,
    /// Attention-score softcap (Gemma2); `None` = disabled.
    pub attn_logit_softcapping: Option<f64>,
    /// Final logit softcap (Gemma2); `None` = disabled.
    pub final_logit_softcapping: Option<f64>,
    pub sliding_window: Option<usize>,
    /// Gemma 3/4: period of local/global interleave (e.g. 6 ⇒ 5 local + 1 global).
    /// `0` means unused (no pattern / explicit per-layer vector only).
    #[allow(dead_code)]
    pub sliding_window_pattern: usize,
    /// Per-layer: `true` = local sliding-window attention (Gemma 3/4).
    pub layer_is_sliding: Vec<bool>,
    /// Local-attention RoPE base (Gemma 3/4 SWA); `None` ⇒ reuse `rope_freq_base`.
    pub rope_freq_base_local: Option<f32>,
    /// Gemma 4 SWA head dim (`attention.key_length_swa`); `None` ⇒ use `head_dim`.
    pub head_dim_local: Option<usize>,
    /// Gemma 4: per-layer KV head counts (global layers often use fewer).
    pub head_count_kv_per_layer: Vec<usize>,
    /// Gemma 4: attention score scale (`1.0` = no `1/sqrt(d)`). `None` ⇒ classic scale.
    pub attention_scale: Option<f64>,
    /// Gemma 4 global RoPE: rotate only this many dims (`partial_rotary`); `None` ⇒ full head.
    pub rope_dim_global: Option<usize>,
    pub tensor_data_offset: u64,
    pub layers: Vec<LayerDmaPlan>,
    pub hot: Vec<TensorLoc>,
    /// Max `read_len` across layers — drives prefetch slot sizing.
    pub max_layer_bytes: usize,
    /// Present when the GGUF carries MoE expert weights.
    pub moe: Option<MoeInfo>,
    /// Per-(layer, expert) DMA plans (empty when `moe` is `None`).
    pub experts: Vec<ExpertDmaPlan>,
    /// Max expert window — drives the expert prefetch ring slot size.
    pub max_expert_bytes: usize,
}

impl GgufLayerMap {
    pub fn is_gemma_family(&self) -> bool {
        matches!(
            self.architecture.as_str(),
            "gemma" | "gemma2" | "gemma3" | "gemma4"
        )
    }

    pub fn is_gemma3(&self) -> bool {
        self.architecture.eq_ignore_ascii_case("gemma3")
    }

    pub fn is_gemma4(&self) -> bool {
        self.architecture.eq_ignore_ascii_case("gemma4")
    }

    /// Effective attention head dim for a layer (SWA vs global on Gemma 4).
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.layer_sliding(layer_idx) {
            self.head_dim_local.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }

    /// Effective KV head count for a layer.
    pub fn head_count_kv_for_layer(&self, layer_idx: usize) -> usize {
        self.head_count_kv_per_layer
            .get(layer_idx)
            .copied()
            .unwrap_or(self.head_count_kv)
    }

    /// True when the GGUF also carries vision / multimodal projector weights.
    pub fn has_vision_tensors(&self) -> bool {
        let is_vision = |name: &str| {
            let n = name.to_ascii_lowercase();
            n.starts_with("v.")
                || n.starts_with("mm.")
                || n.starts_with("vision")
                || n.contains("multi_modal")
                || n.contains("mm_projector")
        };
        self.hot.iter().any(|t| is_vision(&t.name))
            || self
                .layers
                .iter()
                .any(|l| l.tensors.iter().any(|t| is_vision(&t.name)))
    }

    pub fn layer_sliding(&self, layer_idx: usize) -> bool {
        self.layer_is_sliding
            .get(layer_idx)
            .copied()
            .unwrap_or(false)
    }

    #[allow(dead_code)]
    pub fn is_moe(&self) -> bool {
        self.moe
            .as_ref()
            .map(|m| m.expert_count > 1)
            .unwrap_or(false)
    }

    #[allow(dead_code)]
    pub fn expert_plan(&self, layer: usize, expert: usize) -> Option<&ExpertDmaPlan> {
        self.experts
            .iter()
            .find(|e| e.layer_index == layer && e.expert_id == expert)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| {
            IoError::Open(path.display().to_string(), e)
        })?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| IoError::Io(std::io::Error::other(format!("GGUF parse: {e}"))))?;

        let architecture = md_string(&content, "general.architecture")
            .unwrap_or_else(|| "llama".into());

        let block_count = md_u32_arch(&content, &architecture, "block_count")
            .or_else(|| md_u32(&content, "llama.block_count"))
            .ok_or_else(|| {
                IoError::Io(std::io::Error::other(
                    "GGUF missing block_count metadata",
                ))
            })? as usize;

        let embedding_length = md_u32_arch(&content, &architecture, "embedding_length")
            .or_else(|| md_u32(&content, "llama.embedding_length"))
            .unwrap_or(0) as usize;

        let head_count = md_u32_arch(&content, &architecture, "attention.head_count")
            .or_else(|| md_u32(&content, "llama.attention.head_count"))
            .unwrap_or(0) as usize;

        let head_count_kv = md_u32_arch(&content, &architecture, "attention.head_count_kv")
            .or_else(|| md_u32(&content, "llama.attention.head_count_kv"))
            .unwrap_or(head_count as u32) as usize;

        let rope_dim = md_u32_arch(&content, &architecture, "rope.dimension_count")
            .or_else(|| md_u32(&content, "llama.rope.dimension_count"))
            .or_else(|| md_u32_arch(&content, &architecture, "attention.key_length"))
            .unwrap_or(if head_count > 0 {
                (embedding_length / head_count) as u32
            } else {
                0
            }) as usize;

        // Gemma2 keeps head_dim in attention.key_length (often ≠ emb/heads).
        let head_dim = md_u32_arch(&content, &architecture, "attention.key_length")
            .or_else(|| md_u32(&content, "llama.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(if head_count > 0 {
                embedding_length / head_count
            } else {
                rope_dim
            });

        let rms_norm_eps = md_f32_arch(
            &content,
            &architecture,
            "attention.layer_norm_rms_epsilon",
        )
        .or_else(|| md_f32(&content, "llama.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5) as f64;

        let rope_freq_base = md_f32_arch(&content, &architecture, "rope.freq_base")
            .or_else(|| md_f32(&content, "llama.rope.freq_base"))
            .unwrap_or(10_000.0);

        let attn_logit_softcapping = md_f32_arch(&content, &architecture, "attn_logit_softcapping")
            .map(|v| v as f64)
            .filter(|v| *v > 0.0);
        let final_logit_softcapping = md_f32_arch(&content, &architecture, "final_logit_softcapping")
            .map(|v| v as f64)
            .filter(|v| *v > 0.0);
        let sliding_window = md_u32_arch(&content, &architecture, "attention.sliding_window")
            .map(|v| v as usize)
            .filter(|v| *v > 0);

        let rope_freq_base_local = md_f32_arch(&content, &architecture, "rope.freq_base_swa")
            .or_else(|| md_f32_arch(&content, &architecture, "rope.local.freq_base"))
            .filter(|v| *v > 0.0);

        let head_dim_local = md_u32_arch(&content, &architecture, "attention.key_length_swa")
            .map(|v| v as usize)
            .filter(|v| *v > 0);

        let attention_scale = if architecture.eq_ignore_ascii_case("gemma4") {
            // llama.cpp: Gemma4 uses self.scaling = 1.0 (no classic 1/sqrt(d)).
            Some(1.0f64)
        } else {
            None
        };

        // Global layers: partial rotary (HF partial_rotary_factor=0.25 → 512*0.25=128).
        let rope_dim_global = if architecture.eq_ignore_ascii_case("gemma4") {
            let full = md_u32_arch(&content, &architecture, "rope.dimension_count")
                .map(|v| v as usize)
                .unwrap_or(head_dim);
            Some((full / 4).max(1))
        } else {
            None
        };

        let sliding_window_pattern = md_u32_arch(
            &content,
            &architecture,
            "attention.sliding_window_pattern",
        )
        .map(|v| v as usize)
        .unwrap_or(0);

        let layer_is_sliding = build_layer_is_sliding(
            &architecture,
            block_count,
            sliding_window_pattern,
            md_bool_array_arch(&content, &architecture, "attention.sliding_window_pattern"),
        );

        let head_count_kv_per_layer = md_u32_array_arch(
            &content,
            &architecture,
            "attention.head_count_kv",
        )
        .map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>())
        .filter(|v| v.len() == block_count)
        .unwrap_or_else(|| vec![head_count_kv; block_count]);

        let tensor_data_offset = content.tensor_data_offset;

        // MoE metadata (may also be inferred from tensor names below).
        let meta_expert_count = md_u32_arch(&content, &architecture, "expert_count")
            .or_else(|| md_u32(&content, "llama.expert_count"))
            .unwrap_or(0) as usize;
        let meta_expert_used = md_u32_arch(&content, &architecture, "expert_used_count")
            .or_else(|| md_u32(&content, "llama.expert_used_count"))
            .unwrap_or(0) as usize;

        // Group blk.N.* tensors into core vs expert.
        let mut by_layer_core: BTreeMap<usize, Vec<(String, TensorInfo)>> = BTreeMap::new();
        let mut by_layer_expert: BTreeMap<(usize, usize), Vec<(String, TensorInfo)>> =
            BTreeMap::new();
        let mut fused_by_layer: BTreeMap<usize, Vec<(String, TensorInfo)>> = BTreeMap::new();
        let mut fused_slices: BTreeMap<(usize, usize), Vec<TensorLoc>> = BTreeMap::new();
        let mut hot_infos: Vec<(String, TensorInfo)> = Vec::new();
        let mut saw_per_expert = false;
        let mut saw_fused = false;
        let mut saw_gate_up = false;
        let mut max_expert_id = 0usize;
        let mut saw_shared_dense_ffn = false;

        for (name, info) in &content.tensor_infos {
            if let Some((layer_idx, suffix)) = split_block_name(name) {
                if suffix == "ffn_gate.weight" {
                    saw_shared_dense_ffn = true;
                }
                match classify_block_suffix(suffix) {
                    BlockTensorKind::Expert => {
                        if suffix == "ffn_gate_up_exps.weight" {
                            saw_gate_up = true;
                            saw_fused = true;
                            fused_by_layer
                                .entry(layer_idx)
                                .or_default()
                                .push((name.clone(), clone_tensor_info(info)));
                        } else if is_fused_expert_suffix(suffix) {
                            saw_fused = true;
                            fused_by_layer
                                .entry(layer_idx)
                                .or_default()
                                .push((name.clone(), clone_tensor_info(info)));
                        } else if let Some(eid) = is_per_expert_suffix(suffix) {
                            saw_per_expert = true;
                            max_expert_id = max_expert_id.max(eid);
                            by_layer_expert
                                .entry((layer_idx, eid))
                                .or_default()
                                .push((name.clone(), clone_tensor_info(info)));
                        }
                    }
                    BlockTensorKind::Router | BlockTensorKind::Core => {
                        by_layer_core
                            .entry(layer_idx)
                            .or_default()
                            .push((name.clone(), clone_tensor_info(info)));
                    }
                }
            } else if is_hot_tensor(name) {
                hot_infos.push((name.clone(), clone_tensor_info(info)));
            }
        }

        if by_layer_core.is_empty() && by_layer_expert.is_empty() && fused_by_layer.is_empty() {
            return Err(IoError::Io(std::io::Error::other(
                "no blk.N.* tensors found in GGUF",
            )));
        }

        let moe = if saw_per_expert || saw_fused || meta_expert_count > 1 {
            let family = MoeFamily::from_architecture(&architecture);
            let layout = if saw_gate_up {
                MoeLayout::FusedGateUpTrailing
            } else if saw_fused && !saw_per_expert {
                MoeLayout::FusedExps
            } else {
                MoeLayout::PerExpert
            };
            let expert_count = if meta_expert_count > 1 {
                // Prefer GGUF metadata. Never `max` with a tensor dim that may be
                // `expert_feed_forward_length` (e.g. 704) — that blew Gemma 4 up to
                // 704 "experts" per layer.
                meta_expert_count
            } else {
                max_expert_id.saturating_add(1).max(if saw_fused {
                    fused_by_layer
                        .values()
                        .next()
                        .and_then(|v| v.first())
                        .map(|(_, info)| {
                            let dims = info.shape.dims();
                            if layout == MoeLayout::FusedGateUpTrailing {
                                // Candle reverses GGUF dims; trailing expert becomes dims[0].
                                dims.first()
                                    .copied()
                                    .filter(|&d| d > 1)
                                    .or_else(|| dims.last().copied())
                                    .unwrap_or(0)
                            } else {
                                dims.first().copied().unwrap_or(0)
                            }
                        })
                        .unwrap_or(0)
                } else {
                    0
                })
            };
            let expert_used = if meta_expert_used > 0 {
                meta_expert_used
            } else if family == MoeFamily::Gemma4 {
                8.min(expert_count.max(1))
            } else {
                2.min(expert_count.max(1))
            };
            let has_shared = family == MoeFamily::Gemma4
                || (saw_shared_dense_ffn && (saw_fused || saw_per_expert));
            if expert_count > 1 {
                Some(MoeInfo {
                    layout,
                    expert_count,
                    expert_used_count: expert_used.min(expert_count),
                    family,
                    has_shared_expert: has_shared,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Expand fused experts into per-expert logical locs before planning.
        if let Some(ref info) = moe {
            if info.layout == MoeLayout::FusedExps || info.layout == MoeLayout::FusedGateUpTrailing {
                for (layer_idx, fused_tensors) in &fused_by_layer {
                    for (name, tinfo) in fused_tensors {
                        let size = tensor_nbytes(tinfo)?;
                        let abs = tensor_data_offset + tinfo.offset;
                        let suffix = split_block_name(name)
                            .map(|(_, s)| s)
                            .unwrap_or("");
                        let role = fused_role(suffix).unwrap_or("gate");
                        let fused_loc = TensorLoc {
                            name: name.clone(),
                            abs_offset: abs,
                            size_bytes: size,
                            dtype: tinfo.ggml_dtype,
                            shape: tinfo.shape.dims().to_vec(),
                            rel_offset: 0,
                        };
                        for eid in 0..info.expert_count {
                            if role == "gate_up" {
                                let Some(gate_up) = slice_fused_expert_trailing(
                                    &fused_loc,
                                    eid,
                                    info.expert_count,
                                    "gate_up",
                                    *layer_idx,
                                ) else {
                                    continue;
                                };
                                if let Some((gate, up)) =
                                    split_gate_up_loc(&gate_up, *layer_idx, eid)
                                {
                                    fused_slices
                                        .entry((*layer_idx, eid))
                                        .or_default()
                                        .push(gate);
                                    fused_slices
                                        .entry((*layer_idx, eid))
                                        .or_default()
                                        .push(up);
                                }
                            } else if info.layout == MoeLayout::FusedGateUpTrailing {
                                if let Some(slice) = slice_fused_expert_trailing(
                                    &fused_loc,
                                    eid,
                                    info.expert_count,
                                    role,
                                    *layer_idx,
                                ) {
                                    fused_slices
                                        .entry((*layer_idx, eid))
                                        .or_default()
                                        .push(slice);
                                }
                            } else if let Some(slice) = slice_fused_expert(
                                &fused_loc,
                                eid,
                                info.expert_count,
                                role,
                                *layer_idx,
                            ) {
                                fused_slices
                                    .entry((*layer_idx, eid))
                                    .or_default()
                                    .push(slice);
                            }
                        }
                    }
                }
            }
        }

        let mut layers = Vec::with_capacity(by_layer_core.len());
        let mut max_layer_bytes = 0usize;

        for (index, mut tensors) in by_layer_core {
            tensors.sort_by_key(|(_, info)| info.offset);
            let plan = build_layer_plan(index, tensor_data_offset, &tensors)?;
            if !plan.sparse {
                max_layer_bytes = max_layer_bytes.max(plan.read_len);
            } else {
                max_layer_bytes = max_layer_bytes.max(align_up(plan.payload_bytes, DIRECT_ALIGN));
            }
            layers.push(plan);
        }
        layers.sort_by_key(|l| l.index);

        let mut experts = Vec::new();
        let mut max_expert_bytes = 0usize;
        if moe.is_some() {
            // Per-expert tensors from GGUF names.
            for ((layer_idx, expert_id), tensors) in &by_layer_expert {
                let mut locs = Vec::with_capacity(tensors.len());
                for (name, info) in tensors {
                    let size = tensor_nbytes(info)?;
                    locs.push(TensorLoc {
                        name: name.clone(),
                        abs_offset: tensor_data_offset + info.offset,
                        size_bytes: size,
                        dtype: info.ggml_dtype,
                        shape: info.shape.dims().to_vec(),
                        rel_offset: 0,
                    });
                }
                let plan = build_expert_plans_from_locs(*layer_idx, *expert_id, locs);
                max_expert_bytes = max_expert_bytes.max(plan.read_len.max(plan.payload_bytes));
                experts.push(plan);
            }
            // Fused → sliced views.
            for ((layer_idx, expert_id), locs) in fused_slices {
                let plan = build_expert_plans_from_locs(layer_idx, expert_id, locs);
                max_expert_bytes = max_expert_bytes.max(plan.read_len.max(plan.payload_bytes));
                experts.push(plan);
            }
            experts.sort_by_key(|e| (e.layer_index, e.expert_id));
        }

        // Hot tensors: absolute locs with rel_offset = 0 (loaded individually).
        let mut hot = Vec::with_capacity(hot_infos.len());
        for (name, info) in hot_infos {
            let size = tensor_nbytes(&info)?;
            let abs = tensor_data_offset + info.offset;
            hot.push(TensorLoc {
                name,
                abs_offset: abs,
                size_bytes: size,
                dtype: info.ggml_dtype,
                shape: info.shape.dims().to_vec(),
                rel_offset: 0,
            });
        }

        Ok(Self {
            path: path.to_path_buf(),
            architecture,
            block_count,
            embedding_length,
            head_count,
            head_count_kv,
            head_dim,
            rope_dim,
            rms_norm_eps,
            rope_freq_base,
            attn_logit_softcapping,
            final_logit_softcapping,
            sliding_window,
            sliding_window_pattern,
            layer_is_sliding,
            rope_freq_base_local,
            head_dim_local,
            head_count_kv_per_layer,
            attention_scale,
            rope_dim_global,
            tensor_data_offset,
            layers,
            hot,
            max_layer_bytes,
            moe,
            experts,
            max_expert_bytes,
        })
    }
}

fn is_hot_tensor(name: &str) -> bool {
    matches!(
        name,
        "token_embd.weight"
            | "output_norm.weight"
            | "output.weight"
            | "rope_freqs.weight"
            | "token_types.weight"
    ) || name.starts_with("rope_")
}

fn clone_tensor_info(info: &TensorInfo) -> TensorInfo {
    TensorInfo {
        ggml_dtype: info.ggml_dtype,
        shape: info.shape.clone(),
        offset: info.offset,
    }
}

fn tensor_nbytes(info: &TensorInfo) -> Result<usize> {
    let elems = info.shape.elem_count();
    let bs = info.ggml_dtype.block_size();
    if elems % bs != 0 {
        return Err(IoError::Io(std::io::Error::other(format!(
            "tensor elems {elems} not divisible by block size {bs}"
        ))));
    }
    Ok(elems / bs * info.ggml_dtype.type_size())
}

fn build_layer_plan(
    index: usize,
    tensor_data_offset: u64,
    tensors: &[(String, TensorInfo)],
) -> Result<LayerDmaPlan> {
    let mut payload = 0usize;
    let mut min_abs = u64::MAX;
    let mut max_abs_end = 0u64;
    let mut locs = Vec::with_capacity(tensors.len());

    for (name, info) in tensors {
        let size = tensor_nbytes(info)?;
        let abs = tensor_data_offset + info.offset;
        let end = abs + size as u64;
        min_abs = min_abs.min(abs);
        max_abs_end = max_abs_end.max(end);
        payload += size;
        locs.push(TensorLoc {
            name: name.clone(),
            abs_offset: abs,
            size_bytes: size,
            dtype: info.ggml_dtype,
            shape: info.shape.dims().to_vec(),
            rel_offset: 0, // filled below for dense DMA
        });
    }

    let read_offset = min_abs & !(DIRECT_ALIGN as u64 - 1);
    let read_end = align_up(max_abs_end as usize, DIRECT_ALIGN) as u64;
    let read_len = (read_end - read_offset) as usize;

    // Treat as sparse when the coalesced window is much larger than the payload
    // (interleaved layout) or exceeds 96 MiB — keeps prefetch arenas practical.
    const MAX_DENSE_BYTES: usize = 96 * 1024 * 1024;
    let sparse = read_len > MAX_DENSE_BYTES
        || (payload > 0 && read_len as u64 > payload as u64 * 2 + DIRECT_ALIGN as u64);

    if sparse {
        // Pack logically for scratch sizing; rel_offset unused for file loads.
        let mut cursor = 0usize;
        for loc in &mut locs {
            loc.rel_offset = cursor;
            cursor += loc.size_bytes;
        }
        return Ok(LayerDmaPlan {
            index,
            read_offset: 0,
            read_len: align_up(payload, DIRECT_ALIGN),
            tensors: locs,
            payload_bytes: payload,
            sparse: true,
        });
    }

    for loc in &mut locs {
        loc.rel_offset = (loc.abs_offset - read_offset) as usize;
    }

    Ok(LayerDmaPlan {
        index,
        read_offset,
        read_len,
        tensors: locs,
        payload_bytes: payload,
        sparse: false,
    })
}

fn md_string(ct: &gguf_file::Content, key: &str) -> Option<String> {
    ct.metadata
        .get(key)
        .and_then(|v| v.to_string().ok())
        .map(ToOwned::to_owned)
}

fn md_u32(ct: &gguf_file::Content, key: &str) -> Option<u32> {
    let v = ct.metadata.get(key)?;
    v.to_u32()
        .ok()
        .or_else(|| v.to_i32().ok().and_then(|i| if i >= 0 { Some(i as u32) } else { None }))
        .or_else(|| v.to_u64().ok().and_then(|u| u32::try_from(u).ok()))
}

fn md_f32(ct: &gguf_file::Content, key: &str) -> Option<f32> {
    ct.metadata.get(key).and_then(|v| v.to_f32().ok())
}

fn md_u32_arch(ct: &gguf_file::Content, arch: &str, suffix: &str) -> Option<u32> {
    md_u32(ct, &format!("{arch}.{suffix}"))
}

fn md_f32_arch(ct: &gguf_file::Content, arch: &str, suffix: &str) -> Option<f32> {
    md_f32(ct, &format!("{arch}.{suffix}"))
}

/// Bool array metadata (Gemma 3/4 may store per-layer sliding flags this way).
fn md_bool_array_arch(
    ct: &gguf_file::Content,
    arch: &str,
    suffix: &str,
) -> Option<Vec<bool>> {
    let key = format!("{arch}.{suffix}");
    let v = ct.metadata.get(&key)?;
    match v {
        gguf_file::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    gguf_file::Value::Bool(b) => out.push(*b),
                    gguf_file::Value::U8(x) => out.push(*x != 0),
                    gguf_file::Value::U32(x) => out.push(*x != 0),
                    gguf_file::Value::I32(x) => out.push(*x != 0),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn md_u32_array_arch(
    ct: &gguf_file::Content,
    arch: &str,
    suffix: &str,
) -> Option<Vec<u32>> {
    let key = format!("{arch}.{suffix}");
    let v = ct.metadata.get(&key)?;
    match v {
        gguf_file::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let n = item
                    .to_u32()
                    .ok()
                    .or_else(|| {
                        item.to_i32()
                            .ok()
                            .and_then(|i| if i >= 0 { Some(i as u32) } else { None })
                    })
                    .or_else(|| item.to_u64().ok().and_then(|u| u32::try_from(u).ok()))?;
                out.push(n);
            }
            Some(out)
        }
        _ => None,
    }
}

/// Build per-layer sliding flags for Gemma 3/4 (5 local : 1 global by default).
pub fn build_layer_is_sliding(
    architecture: &str,
    n_layers: usize,
    pattern_period: usize,
    pattern_bools: Option<Vec<bool>>,
) -> Vec<bool> {
    if let Some(flags) = pattern_bools {
        if flags.len() == n_layers {
            return flags;
        }
    }
    let arch = architecture.to_ascii_lowercase();
    if arch != "gemma3" && arch != "gemma4" {
        return vec![false; n_layers];
    }
    // Default: 5 local + 1 global (period 6), matching llama.cpp / Gemma reports.
    let period = if pattern_period >= 2 {
        pattern_period
    } else {
        6
    };
    (0..n_layers)
        .map(|il| il % period < period.saturating_sub(1))
        .collect()
}

/// Load a [`TensorLoc`] from a DMA buffer into a [`candle_core::quantized::QTensor`].
pub fn qtensor_from_loc(
    loc: &TensorLoc,
    dma: &[u8],
    device: &Device,
) -> std::result::Result<candle_core::quantized::QTensor, candle_core::Error> {
    let end = loc.rel_offset + loc.size_bytes;
    if end > dma.len() {
        candle_core::bail!(
            "tensor {} extends past DMA buffer ({}+{} > {})",
            loc.name,
            loc.rel_offset,
            loc.size_bytes,
            dma.len()
        );
    }
    let raw = &dma[loc.rel_offset..end];
    candle_core::quantized::ggml_file::qtensor_from_ggml(
        loc.dtype,
        raw,
        loc.shape.clone(),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::build_layer_is_sliding;

    #[test]
    fn align_math() {
        assert_eq!(0u64 & !(4096u64 - 1), 0);
        assert_eq!(4097u64 & !(4096u64 - 1), 4096);
    }

    #[test]
    fn gemma3_default_sliding_pattern_5_local_1_global() {
        let flags = build_layer_is_sliding("gemma3", 12, 0, None);
        assert_eq!(flags.len(), 12);
        // period 6: indices 0..4 local, 5 global, 6..10 local, 11 global
        assert!(flags[0] && flags[4] && !flags[5]);
        assert!(flags[6] && flags[10] && !flags[11]);
        assert_eq!(flags.iter().filter(|&&s| s).count(), 10);
        assert_eq!(flags.iter().filter(|&&s| !s).count(), 2);
    }

    #[test]
    fn gemma3_explicit_period() {
        let flags = build_layer_is_sliding("gemma3", 4, 4, None);
        assert_eq!(flags, vec![true, true, true, false]);
    }

    #[test]
    fn gemma3_bool_array_wins() {
        let flags = build_layer_is_sliding(
            "gemma3",
            3,
            6,
            Some(vec![true, false, true]),
        );
        assert_eq!(flags, vec![true, false, true]);
    }

    #[test]
    fn non_gemma3_no_sliding_flags() {
        let flags = build_layer_is_sliding("gemma2", 8, 6, None);
        assert!(flags.iter().all(|&s| !s));
    }

    #[test]
    fn gemma4_default_sliding_pattern() {
        let flags = build_layer_is_sliding("gemma4", 6, 0, None);
        assert_eq!(flags, vec![true, true, true, true, true, false]);
    }
}
