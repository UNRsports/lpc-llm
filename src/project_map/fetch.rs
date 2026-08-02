//! `io_uring` / `O_DIRECT` on-demand node reads from `map.bin` (buffered fallback).

use std::path::{Path, PathBuf};

use crate::error::{AppError, Result};
use crate::io::nvme::AsyncNvmeReader;
use crate::io::prefetch::{align_up, PrefetchRing, DIRECT_ALIGN};
use crate::project_map::build::{decode_node_payload, load_meta, DecodedNode, MapMeta, NodeMeta};

enum Backend {
    Direct {
        reader: AsyncNvmeReader,
        ring: PrefetchRing,
    },
    Buffered,
}

pub struct ProjectMapReader {
    pub dir: PathBuf,
    pub meta: MapMeta,
    backend: Backend,
}

impl ProjectMapReader {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let meta = load_meta(&dir.join("map.json"))?;
        if meta.format_version != crate::project_map::build::MAP_FORMAT_VERSION {
            return Err(AppError::msg(format!(
                "unsupported project-map format_version {}",
                meta.format_version
            )));
        }
        let bin = dir.join("map.bin");
        if !bin.exists() {
            return Err(AppError::msg(format!("missing {}", bin.display())));
        }

        let backend = match AsyncNvmeReader::open(&bin) {
            Ok(reader) => {
                let slot = align_up(
                    meta.nodes
                        .iter()
                        .map(|n| n.len)
                        .max()
                        .unwrap_or(DIRECT_ALIGN)
                        .max(DIRECT_ALIGN),
                    DIRECT_ALIGN,
                );
                let ring = PrefetchRing::new_unlocked(slot, 8).map_err(AppError::from)?;
                Backend::Direct { reader, ring }
            }
            Err(e) => {
                eprintln!(
                    "project-map: O_DIRECT unavailable ({e}); using buffered reads"
                );
                Backend::Buffered
            }
        };

        Ok(Self { dir, meta, backend })
    }

    pub fn node_meta(&self, id: u32) -> Option<&NodeMeta> {
        self.meta.nodes.iter().find(|n| n.id == id)
    }
}

/// Prefetch selected node records (io_uring when available) and decode them.
pub fn fetch_nodes(reader: &mut ProjectMapReader, ids: &[u32]) -> Result<Vec<DecodedNode>> {
    match &mut reader.backend {
        Backend::Direct { reader: nvme, ring } => fetch_direct(nvme, ring, &reader.meta, ids),
        Backend::Buffered => fetch_nodes_buffered(&reader.dir, &reader.meta, ids),
    }
}

fn fetch_direct(
    nvme: &mut AsyncNvmeReader,
    ring: &mut PrefetchRing,
    meta: &MapMeta,
    ids: &[u32],
) -> Result<Vec<DecodedNode>> {
    let mut out = Vec::with_capacity(ids.len());
    let n_slots = ring.len();
    for (i, id) in ids.iter().enumerate() {
        let Some(node) = meta.nodes.iter().find(|n| n.id == *id).cloned() else {
            continue;
        };
        if node.offset % DIRECT_ALIGN as u64 != 0 {
            return Err(AppError::msg(format!(
                "node {} offset {} not {}-aligned",
                node.id, node.offset, DIRECT_ALIGN
            )));
        }
        let slot = i % n_slots;
        let transfer = align_up(node.len.max(1), DIRECT_ALIGN);
        {
            let buf = ring.get_mut(slot).map_err(AppError::from)?;
            nvme.submit_read(buf, slot, node.offset, transfer)
                .map_err(AppError::from)?;
        }
        let (done_slot, valid) = nvme.wait_completion().map_err(AppError::from)?;
        let buf = ring.get(done_slot).map_err(AppError::from)?;
        let slice = buf.as_slice();
        let take = node.payload_len.min(valid).min(slice.len());
        let decoded = decode_node_payload(&slice[..take])?;
        out.push(decoded);
    }
    Ok(out)
}

fn fetch_nodes_buffered(dir: &Path, meta: &MapMeta, ids: &[u32]) -> Result<Vec<DecodedNode>> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    let mut f = File::open(dir.join("map.bin"))?;
    let mut out = Vec::new();
    for id in ids {
        let Some(n) = meta.nodes.iter().find(|x| x.id == *id) else {
            continue;
        };
        f.seek(SeekFrom::Start(n.offset))?;
        let mut buf = vec![0u8; n.payload_len];
        f.read_exact(&mut buf)?;
        out.push(decode_node_payload(&buf)?);
    }
    Ok(out)
}
