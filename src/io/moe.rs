//! MoE expert taxonomy + DMA plans for hybrid streaming.
//!
//! Two on-disk GGUF layouts are recognized:
//! - **PerExpert** (Mixtral / llama.cpp): `blk.N.ffn_{gate,up,down}.E.weight`
//! - **FusedExps** (Qwen-MoE etc.): `blk.N.ffn_{gate,up,down}_exps.weight` `[E, …]`
//!
//! Router (`ffn_gate_inp`) stays with the dense layer core in `layers.pack`.
//! Experts are rearranged into `experts.pack` for on-demand Top-K DMA.

use serde::{Deserialize, Serialize};

use super::gguf_map::TensorLoc;
use super::prefetch::{align_up, DIRECT_ALIGN};

/// How expert weights are stored in the source GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeLayout {
    /// One set of gate/up/down tensors per expert index.
    PerExpert,
    /// Fused `[n_expert, …]` tensors (`*_exps.weight`).
    FusedExps,
}

/// MoE metadata extracted from GGUF (or inferred from tensor names).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeInfo {
    pub layout: MoeLayout,
    pub expert_count: usize,
    pub expert_used_count: usize,
    /// Architecture family hint for gating quirks.
    pub family: MoeFamily,
}

/// Coarse arch fork used by the hybrid MoE forward path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeFamily {
    Mixtral,
    QwenMoe,
    DeepSeek,
    Unknown,
}

impl MoeFamily {
    pub fn from_architecture(arch: &str) -> Self {
        let a = arch.to_ascii_lowercase();
        if a.contains("mixtral") || a == "llama" {
            // llama.expert_count > 0 ⇒ Mixtral-style in practice
            Self::Mixtral
        } else if a.contains("qwen") {
            Self::QwenMoe
        } else if a.contains("deepseek") {
            Self::DeepSeek
        } else {
            Self::Unknown
        }
    }
}

/// One expert's O_DIRECT DMA window inside `experts.pack` (or GGUF before pack).
#[derive(Debug, Clone)]
pub struct ExpertDmaPlan {
    pub layer_index: usize,
    pub expert_id: usize,
    pub read_offset: u64,
    pub read_len: usize,
    pub tensors: Vec<TensorLoc>,
    pub payload_bytes: usize,
}

impl ExpertDmaPlan {
    #[allow(dead_code)]
    pub fn key(&self) -> (usize, usize) {
        (self.layer_index, self.expert_id)
    }
}

/// Classify a `blk.N.*` suffix (after `blk.N.`) as expert weight / router / core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTensorKind {
    /// Dense attention / norms / dense FFN / etc.
    Core,
    /// Gating network (`ffn_gate_inp`) — resident with the layer core.
    Router,
    /// Per-expert or fused expert payload.
    Expert,
}

/// Parse `blk.{layer}.{rest}` → `(layer, rest)`.
pub fn split_block_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let layer: usize = idx.parse().ok()?;
    Some((layer, suffix))
}

pub fn classify_block_suffix(suffix: &str) -> BlockTensorKind {
    if suffix == "ffn_gate_inp.weight" || suffix.starts_with("ffn_gate_inp.") {
        return BlockTensorKind::Router;
    }
    if is_fused_expert_suffix(suffix) || is_per_expert_suffix(suffix).is_some() {
        return BlockTensorKind::Expert;
    }
    BlockTensorKind::Core
}

pub fn is_fused_expert_suffix(suffix: &str) -> bool {
    matches!(
        suffix,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight"
    )
}

/// `ffn_gate.3.weight` → Some(3); dense `ffn_gate.weight` → None.
pub fn is_per_expert_suffix(suffix: &str) -> Option<usize> {
    for prefix in ["ffn_gate.", "ffn_up.", "ffn_down."] {
        if let Some(rest) = suffix.strip_prefix(prefix) {
            let (idx, rem) = rest.split_once('.')?;
            if rem == "weight" {
                return idx.parse().ok();
            }
        }
    }
    None
}

pub fn fused_role(suffix: &str) -> Option<&'static str> {
    match suffix {
        "ffn_gate_exps.weight" => Some("gate"),
        "ffn_up_exps.weight" => Some("up"),
        "ffn_down_exps.weight" => Some("down"),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn per_expert_role(suffix: &str) -> Option<&'static str> {
    if suffix.starts_with("ffn_gate.") {
        Some("gate")
    } else if suffix.starts_with("ffn_up.") {
        Some("up")
    } else if suffix.starts_with("ffn_down.") {
        Some("down")
    } else {
        None
    }
}

/// Build dense DMA windows for each (layer, expert) from scattered GGUF locs.
pub fn build_expert_plans_from_locs(
    layer_index: usize,
    expert_id: usize,
    mut tensors: Vec<TensorLoc>,
) -> ExpertDmaPlan {
    tensors.sort_by_key(|t| t.abs_offset);
    let payload: usize = tensors.iter().map(|t| t.size_bytes).sum();
    let mut min_abs = u64::MAX;
    let mut max_end = 0u64;
    for t in &tensors {
        min_abs = min_abs.min(t.abs_offset);
        max_end = max_end.max(t.abs_offset + t.size_bytes as u64);
    }
    if tensors.is_empty() {
        return ExpertDmaPlan {
            layer_index,
            expert_id,
            read_offset: 0,
            read_len: 0,
            tensors,
            payload_bytes: 0,
        };
    }

    // Prefer contiguous packing later; for raw GGUF we may be sparse — pack
    // path rewrites offsets. Here we still expose a coalesced window when dense.
    let read_offset = min_abs & !(DIRECT_ALIGN as u64 - 1);
    let read_end = align_up(max_end as usize, DIRECT_ALIGN) as u64;
    let read_len = (read_end - read_offset) as usize;
    let dense = read_len <= payload.saturating_mul(2).saturating_add(DIRECT_ALIGN)
        && read_len <= 64 * 1024 * 1024;

    if dense {
        for t in &mut tensors {
            t.rel_offset = (t.abs_offset - read_offset) as usize;
        }
        ExpertDmaPlan {
            layer_index,
            expert_id,
            read_offset,
            read_len,
            tensors,
            payload_bytes: payload,
        }
    } else {
        // Logical layout for pack scratch sizing.
        let mut cursor = 0usize;
        for t in &mut tensors {
            t.rel_offset = cursor;
            cursor += t.size_bytes;
        }
        ExpertDmaPlan {
            layer_index,
            expert_id,
            read_offset: 0,
            read_len: align_up(payload, DIRECT_ALIGN),
            tensors,
            payload_bytes: payload,
        }
    }
}

/// Slice a fused `[n_expert, …]` tensor into per-expert [`TensorLoc`] views.
pub fn slice_fused_expert(
    fused: &TensorLoc,
    expert_id: usize,
    n_expert: usize,
    role: &str,
    layer_index: usize,
) -> Option<TensorLoc> {
    if n_expert == 0 || expert_id >= n_expert {
        return None;
    }
    if fused.size_bytes % n_expert != 0 {
        return None;
    }
    let per = fused.size_bytes / n_expert;
    let mut shape = fused.shape.clone();
    if !shape.is_empty() && shape[0] == n_expert {
        shape[0] = 1;
        // Collapse leading singleton for QMatMul friendliness → drop expert dim.
        shape.remove(0);
    }
    Some(TensorLoc {
        name: format!("blk.{layer_index}.ffn_{role}.{expert_id}.weight"),
        abs_offset: fused.abs_offset + (expert_id * per) as u64,
        size_bytes: per,
        dtype: fused.dtype,
        shape,
        rel_offset: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_mixtral_names() {
        assert_eq!(
            classify_block_suffix("ffn_gate_inp.weight"),
            BlockTensorKind::Router
        );
        assert_eq!(
            classify_block_suffix("ffn_gate.3.weight"),
            BlockTensorKind::Expert
        );
        assert_eq!(is_per_expert_suffix("ffn_up.7.weight"), Some(7));
        assert_eq!(is_per_expert_suffix("ffn_gate.weight"), None);
        assert_eq!(
            classify_block_suffix("attn_q.weight"),
            BlockTensorKind::Core
        );
    }

    #[test]
    fn classify_fused_names() {
        assert!(is_fused_expert_suffix("ffn_gate_exps.weight"));
        assert_eq!(
            classify_block_suffix("ffn_down_exps.weight"),
            BlockTensorKind::Expert
        );
    }

    #[test]
    fn slice_fused_bytes() {
        use candle_core::quantized::GgmlDType;
        let fused = TensorLoc {
            name: "blk.0.ffn_gate_exps.weight".into(),
            abs_offset: 1000,
            size_bytes: 800,
            dtype: GgmlDType::F16,
            shape: vec![8, 10, 5],
            rel_offset: 0,
        };
        let e3 = slice_fused_expert(&fused, 3, 8, "gate", 0).unwrap();
        assert_eq!(e3.abs_offset, 1000 + 3 * 100);
        assert_eq!(e3.size_bytes, 100);
        assert_eq!(e3.shape, vec![10, 5]);
        assert_eq!(e3.name, "blk.0.ffn_gate.3.weight");
    }
}
