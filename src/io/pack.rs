//! Pack scattered GGUF `blk.N.*` tensors into a contiguous, O_DIRECT-friendly
//! sidecar so every layer is one DMA window (no sparse seek storms).
//!
//! When the GGUF is MoE, expert weights go to a separate `experts.pack` so the
//! hybrid path can DMA Top-K experts without pulling the whole layer FFN.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::{IoError, Result};
use super::gguf_map::{GgufLayerMap, LayerDmaPlan, TensorLoc};
use super::moe::{ExpertDmaPlan, MoeInfo};
use super::prefetch::{align_up, DIRECT_ALIGN};
use crate::progress;

const PACK_VERSION: u32 = 3;
const EXPERT_PACK_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackMeta {
    #[serde(default)]
    version: u32,
    source_gguf: String,
    source_size: u64,
    source_mtime_unix: u64,
    layer_count: usize,
    max_layer_bytes: usize,
    layers: Vec<PackLayerMeta>,
    /// Soft link to MoE sidecar (informational).
    #[serde(default)]
    experts_pack: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackLayerMeta {
    index: usize,
    offset: u64,
    len: usize,
    payload_bytes: usize,
    tensors: Vec<PackTensorMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackTensorMeta {
    name: String,
    rel_offset: usize,
    size_bytes: usize,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertPackMeta {
    version: u32,
    source_gguf: String,
    source_size: u64,
    source_mtime_unix: u64,
    moe: MoeInfo,
    max_expert_bytes: usize,
    expert_count_total: usize,
    experts: Vec<PackExpertMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackExpertMeta {
    layer_index: usize,
    expert_id: usize,
    offset: u64,
    len: usize,
    payload_bytes: usize,
    tensors: Vec<PackTensorMeta>,
}

/// Result of ensuring a packed layer blob exists beside the GGUF.
#[derive(Debug, Clone)]
pub struct PackedWeights {
    pub pack_path: PathBuf,
    pub layers: Vec<LayerDmaPlan>,
    pub max_layer_bytes: usize,
}

impl PackedWeights {
    pub fn recommended_slot_bytes(&self) -> usize {
        let mib = 1024 * 1024;
        align_up(self.max_layer_bytes.max(mib), mib)
    }
}

/// Packed MoE experts (`experts.pack` + index table).
#[derive(Debug, Clone)]
pub struct PackedExperts {
    pub pack_path: PathBuf,
    pub moe: MoeInfo,
    pub experts: Vec<ExpertDmaPlan>,
    pub max_expert_bytes: usize,
}

impl PackedExperts {
    pub fn recommended_slot_bytes(&self) -> usize {
        let kib = 64 * 1024;
        align_up(self.max_expert_bytes.max(kib), DIRECT_ALIGN)
    }

    pub fn plan(&self, layer: usize, expert: usize) -> Option<&ExpertDmaPlan> {
        self.experts
            .iter()
            .find(|e| e.layer_index == layer && e.expert_id == expert)
    }
}

/// Build or reuse a packed layer blob under `cache_dir` (engine module).
///
/// The GGUF under `blobs/` is never modified. Packs are keyed by engine version
/// via the caller-supplied `cache_dir` so upgrades regenerate layout without
/// re-downloading weights.
pub fn ensure_packed(gguf: &Path, map: &GgufLayerMap, cache_dir: &Path) -> Result<PackedWeights> {
    fs::create_dir_all(cache_dir)?;
    let pack_path = cache_dir.join("layers.pack");
    let meta_path = cache_dir.join("layers.pack.json");
    let src_meta = source_fingerprint(gguf)?;

    // Drop interrupted builds left beside the GGUF by older layouts.
    let legacy_partial = {
        let mut s = gguf.as_os_str().to_os_string();
        s.push(".layers.pack.partial");
        PathBuf::from(s)
    };
    if legacy_partial.exists() {
        let _ = fs::remove_file(&legacy_partial);
    }

    if pack_path.exists() && meta_path.exists() {
        if let Ok(meta) = load_meta(&meta_path) {
            if meta.version == PACK_VERSION
                && meta.source_size == src_meta.0
                && meta.source_mtime_unix == src_meta.1
                && meta.layer_count == map.layers.len()
            {
                progress::note(&format!(
                    "layers.pack cache hit ({} layers)",
                    map.layers.len()
                ));
                let layers = plans_from_meta(&meta, map)?;
                return Ok(PackedWeights {
                    pack_path,
                    max_layer_bytes: meta.max_layer_bytes,
                    layers,
                });
            }
        }
    }

    progress::note(&format!(
        "building layers.pack ({} layers, one-time; GGUF untouched)",
        map.layers.len()
    ));
    build_pack(gguf, map, &pack_path, &meta_path, src_meta)?;
    let meta = load_meta(&meta_path)?;
    let layers = plans_from_meta(&meta, map)?;
    Ok(PackedWeights {
        pack_path,
        max_layer_bytes: meta.max_layer_bytes,
        layers,
    })
}

/// Build or reuse `experts.pack` when `map.moe` is set.
pub fn ensure_experts_packed(
    gguf: &Path,
    map: &GgufLayerMap,
    cache_dir: &Path,
) -> Result<Option<PackedExperts>> {
    let Some(moe) = map.moe.clone() else {
        return Ok(None);
    };
    if map.experts.is_empty() {
        return Ok(None);
    }

    fs::create_dir_all(cache_dir)?;
    let pack_path = cache_dir.join("experts.pack");
    let meta_path = cache_dir.join("experts.pack.json");
    let src_meta = source_fingerprint(gguf)?;

    if pack_path.exists() && meta_path.exists() {
        if let Ok(meta) = load_expert_meta(&meta_path) {
            if meta.version == EXPERT_PACK_VERSION
                && meta.source_size == src_meta.0
                && meta.source_mtime_unix == src_meta.1
                && meta.expert_count_total == map.experts.len()
                && meta.moe.expert_count == moe.expert_count
            {
                progress::note(&format!(
                    "experts.pack cache hit ({} experts, top-k={})",
                    map.experts.len(),
                    moe.expert_used_count
                ));
                let experts = expert_plans_from_meta(&meta, map)?;
                return Ok(Some(PackedExperts {
                    pack_path,
                    moe: meta.moe,
                    max_expert_bytes: meta.max_expert_bytes,
                    experts,
                }));
            }
        }
    }

    progress::note(&format!(
        "building experts.pack ({} experts, one-time; may take several minutes)",
        map.experts.len()
    ));
    build_expert_pack(gguf, map, &moe, &pack_path, &meta_path, src_meta)?;
    let meta = load_expert_meta(&meta_path)?;
    let experts = expert_plans_from_meta(&meta, map)?;
    Ok(Some(PackedExperts {
        pack_path,
        moe: meta.moe,
        max_expert_bytes: meta.max_expert_bytes,
        experts,
    }))
}

fn source_fingerprint(gguf: &Path) -> Result<(u64, u64)> {
    let md = fs::metadata(gguf).map_err(|e| IoError::Open(gguf.display().to_string(), e))?;
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok((md.len(), mtime))
}

fn load_meta(path: &Path) -> Result<PackMeta> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| IoError::Io(std::io::Error::other(e)))
}

fn load_expert_meta(path: &Path) -> Result<ExpertPackMeta> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| IoError::Io(std::io::Error::other(e)))
}

fn plans_from_meta(meta: &PackMeta, map: &GgufLayerMap) -> Result<Vec<LayerDmaPlan>> {
    let mut out = Vec::with_capacity(meta.layers.len());
    for (pl, src) in meta.layers.iter().zip(map.layers.iter()) {
        let mut tensors = Vec::with_capacity(pl.tensors.len());
        for (pt, st) in pl.tensors.iter().zip(src.tensors.iter()) {
            let st = src
                .tensors
                .iter()
                .find(|t| t.name == pt.name)
                .unwrap_or(st);
            tensors.push(TensorLoc {
                name: pt.name.clone(),
                abs_offset: pl.offset + pt.rel_offset as u64,
                size_bytes: pt.size_bytes,
                dtype: st.dtype,
                shape: pt.shape.clone(),
                rel_offset: pt.rel_offset,
            });
        }
        out.push(LayerDmaPlan {
            index: pl.index,
            read_offset: pl.offset,
            read_len: pl.len,
            tensors,
            payload_bytes: pl.payload_bytes,
            sparse: false,
        });
    }
    Ok(out)
}

fn expert_plans_from_meta(meta: &ExpertPackMeta, map: &GgufLayerMap) -> Result<Vec<ExpertDmaPlan>> {
    let mut out = Vec::with_capacity(meta.experts.len());
    for pe in &meta.experts {
        let src = map
            .experts
            .iter()
            .find(|e| e.layer_index == pe.layer_index && e.expert_id == pe.expert_id);
        let mut tensors = Vec::with_capacity(pe.tensors.len());
        for pt in &pe.tensors {
            let dtype = src
                .and_then(|s| s.tensors.iter().find(|t| t.name == pt.name))
                .map(|t| t.dtype)
                .or_else(|| src.and_then(|s| s.tensors.first().map(|t| t.dtype)))
                .ok_or_else(|| {
                    IoError::Io(std::io::Error::other(format!(
                        "expert dtype missing for {}",
                        pt.name
                    )))
                })?;
            tensors.push(TensorLoc {
                name: pt.name.clone(),
                abs_offset: pe.offset + pt.rel_offset as u64,
                size_bytes: pt.size_bytes,
                dtype,
                shape: pt.shape.clone(),
                rel_offset: pt.rel_offset,
            });
        }
        out.push(ExpertDmaPlan {
            layer_index: pe.layer_index,
            expert_id: pe.expert_id,
            read_offset: pe.offset,
            read_len: pe.len,
            tensors,
            payload_bytes: pe.payload_bytes,
        });
    }
    Ok(out)
}

fn build_pack(
    gguf: &Path,
    map: &GgufLayerMap,
    pack_path: &Path,
    meta_path: &Path,
    src_meta: (u64, u64),
) -> Result<()> {
    let mut src = File::open(gguf).map_err(|e| IoError::Open(gguf.display().to_string(), e))?;
    let tmp = pack_path.with_extension("pack.partial");
    let mut dst = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;

    let mut cursor = 0u64;
    let mut max_layer = 0usize;
    let mut layers_meta = Vec::with_capacity(map.layers.len());
    let mut scratch = Vec::new();
    let mut prog = progress::Counter::start("packed layer", map.layers.len());

    for layer in map.layers.iter() {
        let pad = (DIRECT_ALIGN as u64 - (cursor % DIRECT_ALIGN as u64)) % DIRECT_ALIGN as u64;
        if pad > 0 {
            dst.write_all(&vec![0u8; pad as usize])?;
            cursor += pad;
        }
        let layer_start = cursor;

        let mut tensors = layer.tensors.clone();
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        let mut tensors_meta = Vec::with_capacity(tensors.len());
        let mut rel = 0usize;

        for t in &tensors {
            scratch.resize(t.size_bytes, 0);
            src.seek(SeekFrom::Start(t.abs_offset))?;
            src.read_exact(&mut scratch)?;
            dst.write_all(&scratch)?;
            tensors_meta.push(PackTensorMeta {
                name: t.name.clone(),
                rel_offset: rel,
                size_bytes: t.size_bytes,
                dtype: format!("{:?}", t.dtype),
                shape: t.shape.clone(),
            });
            rel += t.size_bytes;
            cursor += t.size_bytes as u64;
        }

        let payload = rel;
        let chunk = chunk_align_for(payload);
        let aligned_len = align_up(payload, chunk);
        if aligned_len > payload {
            dst.write_all(&vec![0u8; aligned_len - payload])?;
            cursor += (aligned_len - payload) as u64;
        }
        max_layer = max_layer.max(aligned_len);

        layers_meta.push(PackLayerMeta {
            index: layer.index,
            offset: layer_start,
            len: aligned_len,
            payload_bytes: payload,
            tensors: tensors_meta,
        });
        prog.tick();
    }

    dst.sync_all()?;
    drop(dst);
    fs::rename(&tmp, pack_path)?;

    let experts_pack = map.moe.as_ref().map(|_| {
        cache_dir_of(pack_path)
            .join("experts.pack")
            .display()
            .to_string()
    });

    let meta = PackMeta {
        version: PACK_VERSION,
        source_gguf: gguf.display().to_string(),
        source_size: src_meta.0,
        source_mtime_unix: src_meta.1,
        layer_count: map.layers.len(),
        max_layer_bytes: max_layer,
        layers: layers_meta,
        experts_pack,
    };
    fs::write(
        meta_path,
        serde_json::to_string_pretty(&meta).map_err(|e| IoError::Io(std::io::Error::other(e)))?,
    )?;

    eprintln!(
        "pack ready: {} (max layer {} KiB)",
        pack_path.display(),
        max_layer / 1024
    );
    Ok(())
}

fn build_expert_pack(
    gguf: &Path,
    map: &GgufLayerMap,
    moe: &MoeInfo,
    pack_path: &Path,
    meta_path: &Path,
    src_meta: (u64, u64),
) -> Result<()> {
    let mut src = File::open(gguf).map_err(|e| IoError::Open(gguf.display().to_string(), e))?;
    let tmp = pack_path.with_extension("pack.partial");
    let mut dst = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;

    let mut cursor = 0u64;
    let mut max_expert = 0usize;
    let mut experts_meta = Vec::with_capacity(map.experts.len());
    let mut scratch = Vec::new();
    let mut prog = progress::Counter::start("packed expert", map.experts.len());

    for expert in map.experts.iter() {
        let pad = (DIRECT_ALIGN as u64 - (cursor % DIRECT_ALIGN as u64)) % DIRECT_ALIGN as u64;
        if pad > 0 {
            dst.write_all(&vec![0u8; pad as usize])?;
            cursor += pad;
        }
        let start = cursor;

        let mut tensors = expert.tensors.clone();
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        let mut tensors_meta = Vec::with_capacity(tensors.len());
        let mut rel = 0usize;
        for t in &tensors {
            scratch.resize(t.size_bytes, 0);
            src.seek(SeekFrom::Start(t.abs_offset))?;
            src.read_exact(&mut scratch)?;
            dst.write_all(&scratch)?;
            tensors_meta.push(PackTensorMeta {
                name: t.name.clone(),
                rel_offset: rel,
                size_bytes: t.size_bytes,
                dtype: format!("{:?}", t.dtype),
                shape: t.shape.clone(),
            });
            rel += t.size_bytes;
            cursor += t.size_bytes as u64;
        }

        let payload = rel;
        let aligned_len = align_up(payload, DIRECT_ALIGN);
        if aligned_len > payload {
            dst.write_all(&vec![0u8; aligned_len - payload])?;
            cursor += (aligned_len - payload) as u64;
        }
        max_expert = max_expert.max(aligned_len);

        experts_meta.push(PackExpertMeta {
            layer_index: expert.layer_index,
            expert_id: expert.expert_id,
            offset: start,
            len: aligned_len,
            payload_bytes: payload,
            tensors: tensors_meta,
        });
        prog.tick();
    }

    dst.sync_all()?;
    drop(dst);
    fs::rename(&tmp, pack_path)?;

    let meta = ExpertPackMeta {
        version: EXPERT_PACK_VERSION,
        source_gguf: gguf.display().to_string(),
        source_size: src_meta.0,
        source_mtime_unix: src_meta.1,
        moe: moe.clone(),
        max_expert_bytes: max_expert,
        expert_count_total: map.experts.len(),
        experts: experts_meta,
    };
    fs::write(
        meta_path,
        serde_json::to_string_pretty(&meta).map_err(|e| IoError::Io(std::io::Error::other(e)))?,
    )?;

    eprintln!(
        "experts pack ready: {} (max expert {} KiB, top-k={})",
        pack_path.display(),
        max_expert / 1024,
        moe.expert_used_count
    );
    Ok(())
}

fn cache_dir_of(pack_path: &Path) -> PathBuf {
    pack_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Micro chunk size for layer DMA windows (must be ≥ [`DIRECT_ALIGN`] and a multiple).
fn chunk_align_for(payload: usize) -> usize {
    const K64: usize = 64 * 1024;
    const K256: usize = 256 * 1024;
    if payload >= K256 {
        K256
    } else if payload >= K64 {
        K64
    } else {
        DIRECT_ALIGN
    }
}
