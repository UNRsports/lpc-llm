//! MoE expert taxonomy + DMA plans for hybrid streaming.
//!
//! On-disk GGUF layouts recognized:
//! - **PerExpert** (Mixtral / llama.cpp): `blk.N.ffn_{gate,up,down}.E.weight`
//! - **FusedExps** (Qwen-MoE etc.): `blk.N.ffn_{gate,up,down}_exps.weight` `[E, …]`
//! - **FusedGateUp** (Gemma 4): `ffn_gate_up_exps` `[…, E]` + `ffn_down_exps` `[…, E]`
//!   with a dense shared expert (`ffn_gate/up/down`) resident in `layers.pack`.
//!
//! Router (`ffn_gate_inp`) stays with the dense layer core in `layers.pack`.
//! Routed experts are rearranged into `experts.pack` for on-demand Top-K DMA.

use serde::{Deserialize, Serialize};

use super::gguf_map::TensorLoc;
use super::prefetch::{align_up, DIRECT_ALIGN};

/// How expert weights are stored in the source GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeLayout {
    /// One set of gate/up/down tensors per expert index.
    PerExpert,
    /// Fused `[n_expert, …]` tensors (`*_exps.weight`), expert dim leading.
    FusedExps,
    /// Gemma 4: fused `ffn_gate_up_exps` + `ffn_down_exps`, expert dim trailing.
    FusedGateUpTrailing,
}

/// MoE metadata extracted from GGUF (or inferred from tensor names).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeInfo {
    pub layout: MoeLayout,
    pub expert_count: usize,
    pub expert_used_count: usize,
    /// Architecture family hint for gating quirks.
    pub family: MoeFamily,
    /// Dense shared-expert FFN present alongside routed experts (Gemma 4).
    #[serde(default)]
    pub has_shared_expert: bool,
}

/// Coarse arch fork used by the hybrid MoE forward path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeFamily {
    Mixtral,
    QwenMoe,
    DeepSeek,
    Gemma4,
    Unknown,
}

impl MoeFamily {
    pub fn from_architecture(arch: &str) -> Self {
        let a = arch.to_ascii_lowercase();
        if a.contains("gemma4") || a == "gemma4" {
            Self::Gemma4
        } else if a.contains("mixtral") || a == "llama" {
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
    // Expert *scales* stay with the layer core (tiny; needed at routing time).
    if suffix.ends_with("_exps.scale") || suffix.contains("_exps.scale") {
        return BlockTensorKind::Core;
    }
    if is_fused_expert_suffix(suffix) || is_per_expert_suffix(suffix).is_some() {
        return BlockTensorKind::Expert;
    }
    BlockTensorKind::Core
}

pub fn is_fused_expert_suffix(suffix: &str) -> bool {
    matches!(
        suffix,
        "ffn_gate_exps.weight"
            | "ffn_up_exps.weight"
            | "ffn_down_exps.weight"
            | "ffn_gate_up_exps.weight"
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
        "ffn_gate_up_exps.weight" => Some("gate_up"),
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

/// Slice a trailing-expert fused tensor (Gemma 4 `*_exps`).
///
/// GGUF stores expert as the last `ne` (outermost). Candle reverses dims, so the
/// same tensor appears as `[n_expert, …]` in [`TensorLoc::shape`]. Accept either
/// end so shape drops the expert axis while bytes stay `size / n_expert` chunks.
pub fn slice_fused_expert_trailing(
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
    if shape.last().copied() == Some(n_expert) {
        shape.pop();
    } else if shape.first().copied() == Some(n_expert) {
        shape.remove(0);
    } else {
        return None;
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

/// Split one expert's fused `gate_up` into gate + up halves (equal byte sizes).
///
/// Candle-reversed Gemma 4 stores `[2*n_ff, n_embd]` (prefer splitting dim 0).
/// Unreversed layouts use `[n_embd, 2*n_ff]` (split dim 1 when dim 0 is odd).
pub fn split_gate_up_loc(
    gate_up: &TensorLoc,
    layer_index: usize,
    expert_id: usize,
) -> Option<(TensorLoc, TensorLoc)> {
    if gate_up.size_bytes % 2 != 0 {
        return None;
    }
    let half = gate_up.size_bytes / 2;
    let mut gate_shape = gate_up.shape.clone();
    let mut up_shape = gate_up.shape.clone();
    if gate_shape.len() >= 2 {
        if gate_shape[0] % 2 == 0 {
            let ff2 = gate_shape[0];
            gate_shape[0] = ff2 / 2;
            up_shape[0] = ff2 / 2;
        } else if gate_shape[1] % 2 == 0 {
            let ff2 = gate_shape[1];
            gate_shape[1] = ff2 / 2;
            up_shape[1] = ff2 / 2;
        } else {
            return None;
        }
    }
    let gate = TensorLoc {
        name: format!("blk.{layer_index}.ffn_gate.{expert_id}.weight"),
        abs_offset: gate_up.abs_offset,
        size_bytes: half,
        dtype: gate_up.dtype,
        shape: gate_shape,
        rel_offset: 0,
    };
    let up = TensorLoc {
        name: format!("blk.{layer_index}.ffn_up.{expert_id}.weight"),
        abs_offset: gate_up.abs_offset + half as u64,
        size_bytes: half,
        dtype: gate_up.dtype,
        shape: up_shape,
        rel_offset: 0,
    };
    Some((gate, up))
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
        assert_eq!(
            classify_block_suffix("ffn_down_exps.scale"),
            BlockTensorKind::Core
        );
    }

    #[test]
    fn classify_fused_names() {
        assert!(is_fused_expert_suffix("ffn_gate_exps.weight"));
        assert!(is_fused_expert_suffix("ffn_gate_up_exps.weight"));
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

    #[test]
    fn slice_trailing_and_split_gate_up() {
        use candle_core::quantized::GgmlDType;
        // Expert last in this synthetic shape (unreversed); after pop → [20, 10].
        let fused = TensorLoc {
            name: "blk.0.ffn_gate_up_exps.weight".into(),
            abs_offset: 0,
            size_bytes: 128 * 200,
            dtype: GgmlDType::F16,
            shape: vec![20, 10, 128],
            rel_offset: 0,
        };
        let e2 = slice_fused_expert_trailing(&fused, 2, 128, "gate_up", 0).unwrap();
        assert_eq!(e2.abs_offset, 2 * 200);
        assert_eq!(e2.size_bytes, 200);
        assert_eq!(e2.shape, vec![20, 10]);
        let (g, u) = split_gate_up_loc(&e2, 0, 2).unwrap();
        assert_eq!(g.size_bytes, 100);
        assert_eq!(u.size_bytes, 100);
        assert_eq!(g.shape, vec![10, 10]);
        assert_eq!(u.abs_offset, g.abs_offset + 100);
    }

    #[test]
    fn gemma4_candle_reversed_gate_up_slice() {
        use candle_core::quantized::GgmlDType;
        // GGUF `[2816, 1408, 128]` → Candle reverses → `[128, 1408, 2816]`.
        let fused = TensorLoc {
            name: "blk.0.ffn_gate_up_exps.weight".into(),
            abs_offset: 1000,
            size_bytes: 128 * 1408 * 2816 * 2,
            dtype: GgmlDType::F16,
            shape: vec![128, 1408, 2816],
            rel_offset: 0,
        };
        let e0 = slice_fused_expert_trailing(&fused, 0, 128, "gate_up", 0).unwrap();
        assert_eq!(e0.shape, vec![1408, 2816]);
        assert_eq!(e0.size_bytes, 1408 * 2816 * 2);
        let (g, u) = split_gate_up_loc(&e0, 0, 0).unwrap();
        assert_eq!(g.shape, vec![704, 2816]);
        assert_eq!(u.shape, vec![704, 2816]);
        assert_eq!(g.size_bytes, u.size_bytes);
        assert_eq!(u.abs_offset, g.abs_offset + g.size_bytes as u64);
    }

    #[test]
    fn gemma4_family_detect() {
        assert_eq!(MoeFamily::from_architecture("gemma4"), MoeFamily::Gemma4);
        assert_eq!(MoeFamily::from_architecture("gemma3"), MoeFamily::Unknown);
    }
}
