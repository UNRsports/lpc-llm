//! Build / rebuild on-disk project maps (`map.bin` + `map.json`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AppError, Result};
use crate::io::prefetch::{align_up, DIRECT_ALIGN};
use crate::project_map::embed::{embed_to_i16, hash_embed, EMBED_DIM};
use crate::project_map::extract::{extract_tree, CallEdge, FileExtract, SymbolKind};
use crate::store::{now_unix, LocalStore};

pub const MAP_FORMAT_VERSION: u32 = 1;
/// Fixed logical payload per node before padding (must fit in one aligned record).
pub const NODE_PAYLOAD_MAX: usize = 3072;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapMeta {
    pub format_version: u32,
    pub source_path: String,
    pub source_hash: String,
    pub built_at_unix: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub record_align: usize,
    pub nodes: Vec<NodeMeta>,
    pub edges: Vec<EdgeMeta>,
    /// Relative path → mtime for incremental updates.
    pub files: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    pub id: u32,
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
    pub offset: u64,
    pub len: usize,
    pub payload_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMeta {
    pub from: u32,
    pub to: u32,
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct MapStatus {
    pub hash: String,
    pub dir: PathBuf,
    pub source_path: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub built_at_unix: u64,
    pub map_bin_bytes: u64,
}

/// Stable hash of a canonicalized absolute path (directory identity).
pub fn project_hash(path: &Path) -> Result<String> {
    let canon = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let s = canon.to_string_lossy();
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    Ok(hex_short(&digest, 16))
}

/// Build (or incremental-update) a project map under `cache/projects/<hash>/`.
pub fn build_project_map(store: &LocalStore, project_path: &Path) -> Result<MapStatus> {
    let root = fs::canonicalize(project_path).map_err(|e| {
        AppError::msg(format!(
            "cannot resolve project path {}: {e}",
            project_path.display()
        ))
    })?;
    let hash = project_hash(&root)?;
    let dir = store.project_map_dir(&hash);
    fs::create_dir_all(&dir)?;

    let meta_path = dir.join("map.json");
    let bin_path = dir.join("map.bin");

    let extracts = extract_tree(&root)?;
    let prev = if meta_path.exists() {
        load_meta(&meta_path).ok()
    } else {
        None
    };

    let (extracts, incremental) = if let Some(ref prev) = prev {
        if prev.source_path == root.to_string_lossy() {
            let changed = filter_changed(&extracts, prev);
            if changed.len() < extracts.len() {
                // For correctness with call edges across files, full rebuild is safer
                // when many files change; if few change, still rebuild full graph from
                // all extracts (cheap heuristics) but keep path.
                (extracts, true)
            } else {
                (extracts, false)
            }
        } else {
            (extracts, false)
        }
    } else {
        (extracts, false)
    };
    let _ = incremental;

    let (meta, blob) = materialize_map(&root, &hash, &extracts)?;
    write_aligned_bin(&bin_path, &blob)?;
    save_meta(&meta_path, &meta)?;

    let map_bin_bytes = fs::metadata(&bin_path)?.len();
    Ok(MapStatus {
        hash,
        dir,
        source_path: meta.source_path,
        node_count: meta.node_count,
        edge_count: meta.edge_count,
        built_at_unix: meta.built_at_unix,
        map_bin_bytes,
    })
}

/// Force a clean rebuild (delete previous artifacts first).
pub fn rebuild_project_map(store: &LocalStore, project_path: &Path) -> Result<MapStatus> {
    let root = fs::canonicalize(project_path).unwrap_or_else(|_| project_path.to_path_buf());
    let hash = project_hash(&root)?;
    let dir = store.project_map_dir(&hash);
    if dir.exists() {
        fs::remove_dir_all(&dir)?;
    }
    build_project_map(store, &root)
}

pub fn load_status(store: &LocalStore, path_or_hash: &str) -> Result<MapStatus> {
    let (hash, dir) = resolve_map_dir(store, path_or_hash)?;
    let meta = load_meta(&dir.join("map.json"))?;
    let map_bin_bytes = fs::metadata(dir.join("map.bin"))
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(MapStatus {
        hash,
        dir,
        source_path: meta.source_path,
        node_count: meta.node_count,
        edge_count: meta.edge_count,
        built_at_unix: meta.built_at_unix,
        map_bin_bytes,
    })
}

pub fn resolve_map_dir(store: &LocalStore, path_or_hash: &str) -> Result<(String, PathBuf)> {
    let p = Path::new(path_or_hash);
    if p.is_dir() {
        let hash = project_hash(p)?;
        let dir = store.project_map_dir(&hash);
        if !dir.join("map.json").exists() {
            return Err(AppError::msg(format!(
                "no project-map for {} — run `lpc-llm project-map build {}`",
                p.display(),
                p.display()
            )));
        }
        return Ok((hash, dir));
    }
    // Treat as hash / directory name under cache/projects/
    let dir = store.project_map_dir(path_or_hash);
    if dir.join("map.json").exists() {
        return Ok((path_or_hash.to_string(), dir));
    }
    Err(AppError::msg(format!(
        "project-map `{path_or_hash}` not found under {}",
        store.projects_dir().display()
    )))
}

pub fn load_meta(path: &Path) -> Result<MapMeta> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn save_meta(path: &Path, meta: &MapMeta) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(meta)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn filter_changed<'a>(extracts: &'a [FileExtract], prev: &MapMeta) -> Vec<&'a FileExtract> {
    extracts
        .iter()
        .filter(|e| {
            let key = e.path.to_string_lossy().to_string();
            match prev.files.get(&key) {
                Some(mt) => *mt != e.mtime_unix,
                None => true,
            }
        })
        .collect()
}

fn materialize_map(
    root: &Path,
    hash: &str,
    extracts: &[FileExtract],
) -> Result<(MapMeta, Vec<u8>)> {
    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    let mut nodes_out: Vec<NodeMeta> = Vec::new();
    let mut blob: Vec<u8> = Vec::new();
    let mut files = BTreeMap::new();

    // Collect unique symbols by qualified key file::name
    let mut sym_list = Vec::new();
    for fe in extracts {
        files.insert(fe.path.to_string_lossy().to_string(), fe.mtime_unix);
        for s in &fe.symbols {
            let key = format!("{}::{}", s.file.display(), s.name);
            if name_to_id.contains_key(&key) {
                continue;
            }
            let id = nodes_out.len() as u32;
            name_to_id.insert(key, id);
            // Also index bare name → first id for call resolution.
            name_to_id.entry(s.name.clone()).or_insert(id);
            sym_list.push(s.clone());
            nodes_out.push(NodeMeta {
                id,
                name: s.name.clone(),
                kind: kind_str(s.kind).to_string(),
                file: s.file.to_string_lossy().to_string(),
                line: s.line,
                offset: 0,
                len: 0,
                payload_len: 0,
            });
        }
    }

    for (i, s) in sym_list.iter().enumerate() {
        let emb = embed_to_i16(&hash_embed(&format!(
            "{} {} {}",
            s.name, s.signature, s.preview
        )));
        let payload = encode_node_payload(s, &emb)?;
        let payload_len = payload.len();
        let rec_len = align_up(payload_len.max(1), DIRECT_ALIGN);
        let offset = blob.len() as u64;
        blob.extend_from_slice(&payload);
        if rec_len > payload_len {
            blob.resize(blob.len() + (rec_len - payload_len), 0);
        }
        nodes_out[i].offset = offset;
        nodes_out[i].len = rec_len;
        nodes_out[i].payload_len = payload_len;
    }

    let mut edges = Vec::new();
    let mut edge_set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for fe in extracts {
        for CallEdge { from, to, .. } in &fe.calls {
            let Some(&fid) = name_to_id.get(from) else {
                continue;
            };
            let Some(&tid) = name_to_id.get(to) else {
                continue;
            };
            if fid == tid {
                continue;
            }
            if edge_set.insert((fid, tid)) {
                edges.push(EdgeMeta {
                    from: fid,
                    to: tid,
                    kind: "call".into(),
                });
            }
        }
    }

    // Soft cap on edges for huge repos.
    if edges.len() > 50_000 {
        edges.truncate(50_000);
    }

    let meta = MapMeta {
        format_version: MAP_FORMAT_VERSION,
        source_path: root.to_string_lossy().to_string(),
        source_hash: hash.to_string(),
        built_at_unix: now_unix(),
        node_count: nodes_out.len(),
        edge_count: edges.len(),
        record_align: DIRECT_ALIGN,
        nodes: nodes_out,
        edges,
        files,
    };
    Ok((meta, blob))
}

/// On-disk node record: magic + lengths + embedding + utf8 fields.
fn encode_node_payload(s: &crate::project_map::extract::Symbol, emb: &[i16; EMBED_DIM]) -> Result<Vec<u8>> {
    let name = s.name.as_bytes();
    let sig = s.signature.as_bytes();
    let prev = s.preview.as_bytes();
    let file = s.file.to_string_lossy();
    let file_b = file.as_bytes();
    let need = 4 + 2 + 2 + 2 + 2 + 4 + EMBED_DIM * 2 + name.len() + sig.len() + prev.len() + file_b.len();
    if need > NODE_PAYLOAD_MAX {
        // Truncate preview/signature to fit.
    }
    let mut buf = Vec::with_capacity(need.min(NODE_PAYLOAD_MAX));
    buf.extend_from_slice(b"PMND");
    let name = name;
    let mut sig = sig;
    let mut prev = prev;
    let mut file_b = file_b;
    // Shrink until under max.
    while 4 + 8 + 4 + EMBED_DIM * 2 + name.len() + sig.len() + prev.len() + file_b.len() > NODE_PAYLOAD_MAX
    {
        if prev.len() > 32 {
            prev = &prev[..prev.len() / 2];
        } else if sig.len() > 32 {
            sig = &sig[..sig.len() / 2];
        } else if file_b.len() > 32 {
            file_b = &file_b[..file_b.len() / 2];
        } else {
            break;
        }
    }
    buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(sig.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(prev.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(file_b.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(s.line as u32).to_le_bytes());
    for x in emb {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    buf.extend_from_slice(name);
    buf.extend_from_slice(sig);
    buf.extend_from_slice(prev);
    buf.extend_from_slice(file_b);
    Ok(buf)
}

pub fn decode_node_payload(data: &[u8]) -> Result<DecodedNode> {
    if data.len() < 4 + 8 + 4 + EMBED_DIM * 2 {
        return Err(AppError::msg("project-map node record too short"));
    }
    if &data[0..4] != b"PMND" {
        return Err(AppError::msg("project-map node magic mismatch"));
    }
    let name_len = u16::from_le_bytes([data[4], data[5]]) as usize;
    let sig_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let prev_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let file_len = u16::from_le_bytes([data[10], data[11]]) as usize;
    let line = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let mut emb = [0i16; EMBED_DIM];
    let mut off = 16;
    for i in 0..EMBED_DIM {
        emb[i] = i16::from_le_bytes([data[off], data[off + 1]]);
        off += 2;
    }
    let end = off + name_len + sig_len + prev_len + file_len;
    if data.len() < end {
        return Err(AppError::msg("project-map node payload truncated"));
    }
    let name = String::from_utf8_lossy(&data[off..off + name_len]).into_owned();
    off += name_len;
    let signature = String::from_utf8_lossy(&data[off..off + sig_len]).into_owned();
    off += sig_len;
    let preview = String::from_utf8_lossy(&data[off..off + prev_len]).into_owned();
    off += prev_len;
    let file = String::from_utf8_lossy(&data[off..off + file_len]).into_owned();
    Ok(DecodedNode {
        name,
        signature,
        preview,
        file,
        line,
        embed: emb,
    })
}

#[derive(Debug, Clone)]
pub struct DecodedNode {
    pub name: String,
    pub signature: String,
    pub preview: String,
    pub file: String,
    pub line: usize,
    pub embed: [i16; EMBED_DIM],
}

fn write_aligned_bin(path: &Path, blob: &[u8]) -> Result<()> {
    let tmp = path.with_extension("bin.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(blob)?;
        // Pad file to DIRECT_ALIGN for O_DIRECT readers that read past last record.
        let pad = align_up(blob.len(), DIRECT_ALIGN).saturating_sub(blob.len());
        if pad > 0 {
            f.write_all(&vec![0u8; pad])?;
        }
        f.sync_all()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn kind_str(k: SymbolKind) -> &'static str {
    match k {
        SymbolKind::Function => "function",
        SymbolKind::Method => "method",
        SymbolKind::Struct => "struct",
        SymbolKind::Class => "class",
        SymbolKind::Trait => "trait",
        SymbolKind::Type => "type",
        SymbolKind::Module => "module",
        SymbolKind::Other => "other",
    }
}

fn hex_short(bytes: &[u8], n_bytes: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(n_bytes * 2);
    for &b in bytes.iter().take(n_bytes) {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}
