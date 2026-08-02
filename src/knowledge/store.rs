//! Persistent knowledge chunks under `cache/knowledge/`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{AppError, Result};
use crate::knowledge::backend::SearchHit;
use crate::store::{now_unix, LocalStore};

const INDEX_FILE: &str = "index.json";
const MAX_CHUNK_CHARS: usize = 4_096;
const MAX_CHUNKS: usize = 2_048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeChunk {
    pub id: String,
    pub query: String,
    pub title: String,
    pub url: String,
    pub text: String,
    pub tags: Vec<String>,
    pub fetched_at_unix: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct KnowledgeIndex {
    chunks: Vec<KnowledgeChunk>,
}

pub struct KnowledgeStore {
    dir: PathBuf,
}

impl KnowledgeStore {
    pub fn open(store: &LocalStore) -> Result<Self> {
        let dir = store.knowledge_dir();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn open_path(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join(INDEX_FILE)
    }

    fn load_index(&self) -> Result<KnowledgeIndex> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(KnowledgeIndex::default());
        }
        let text = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&text)?)
    }

    fn save_index(&self, index: &KnowledgeIndex) -> Result<()> {
        let path = self.index_path();
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(index)?)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<KnowledgeChunk>> {
        Ok(self.load_index()?.chunks)
    }

    pub fn purge(&self) -> Result<usize> {
        let mut index = self.load_index()?;
        let n = index.chunks.len();
        index.chunks.clear();
        self.save_index(&index)?;
        // Remove loose chunk files if any.
        if self.dir.exists() {
            for entry in fs::read_dir(&self.dir)? {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name == INDEX_FILE {
                    continue;
                }
                if name.ends_with(".txt") || name.ends_with(".json") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        Ok(n)
    }

    /// Persist search hits as chunks keyed by content hash.
    pub fn ingest_hits(&self, query: &str, hits: &[SearchHit], tags: &[&str]) -> Result<usize> {
        if hits.is_empty() {
            return Ok(0);
        }
        let mut index = self.load_index()?;
        let mut added = 0usize;
        let now = now_unix();
        for hit in hits {
            let text = truncate_chars(&hit.snippet, MAX_CHUNK_CHARS);
            if text.trim().is_empty() {
                continue;
            }
            let id = chunk_id(query, &hit.url, &text);
            if index.chunks.iter().any(|c| c.id == id) {
                continue;
            }
            let chunk = KnowledgeChunk {
                id: id.clone(),
                query: query.to_string(),
                title: truncate_chars(&hit.title, 200),
                url: hit.url.clone(),
                text,
                tags: tags.iter().map(|t| (*t).to_string()).collect(),
                fetched_at_unix: now,
            };
            // Sidecar body for inspection / backup.
            let body_path = self.dir.join(format!("{id}.txt"));
            fs::write(&body_path, &chunk.text)?;
            index.chunks.push(chunk);
            added += 1;
        }
        // Rotation: keep newest MAX_CHUNKS.
        if index.chunks.len() > MAX_CHUNKS {
            let drop_n = index.chunks.len() - MAX_CHUNKS;
            let removed: Vec<_> = index.chunks.drain(0..drop_n).collect();
            for c in removed {
                let _ = fs::remove_file(self.dir.join(format!("{}.txt", c.id)));
            }
        }
        self.save_index(&index)?;
        Ok(added)
    }

    /// Rank chunks by simple term overlap with `query` (token budget aware).
    pub fn retrieve(&self, query: &str, max_chunks: usize, max_chars: usize) -> Result<Vec<KnowledgeChunk>> {
        let index = self.load_index()?;
        if index.chunks.is_empty() || max_chunks == 0 || max_chars == 0 {
            return Ok(Vec::new());
        }
        let terms = tokenize(query);
        let mut scored: Vec<(usize, &KnowledgeChunk)> = index
            .chunks
            .iter()
            .map(|c| {
                let mut score = overlap_score(&terms, &c.text) + overlap_score(&terms, &c.title);
                score += overlap_score(&terms, &c.query) * 2;
                for tag in &c.tags {
                    score += overlap_score(&terms, tag);
                }
                (score, c)
            })
            .filter(|(s, _)| *s > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        let mut out = Vec::new();
        let mut used = 0usize;
        for (_, c) in scored.into_iter().take(max_chunks.saturating_mul(2)) {
            let need = c.text.len() + c.title.len() + 32;
            if used + need > max_chars && !out.is_empty() {
                break;
            }
            if out.len() >= max_chunks {
                break;
            }
            used += need.min(max_chars.saturating_sub(used));
            out.push(c.clone());
        }
        Ok(out)
    }
}

fn chunk_id(query: &str, url: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query.as_bytes());
    hasher.update(b"|");
    hasher.update(url.as_bytes());
    hasher.update(b"|");
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    hex_short(&digest)
}

fn hex_short(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(16);
    for &b in bytes.iter().take(8) {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect()
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn overlap_score(terms: &[String], text: &str) -> usize {
    if terms.is_empty() {
        return 0;
    }
    let lower = text.to_ascii_lowercase();
    let mut score = 0usize;
    for t in terms {
        if lower.contains(t.as_str()) {
            score += 1;
        }
    }
    score
}

/// Validate a user-supplied tag list (length / charset).
#[allow(dead_code)]
pub fn validate_tag(tag: &str) -> Result<()> {
    let t = tag.trim();
    if t.is_empty() || t.len() > 64 {
        return Err(AppError::msg("tag must be 1..=64 chars"));
    }
    if !t
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '/')
    {
        return Err(AppError::msg(
            "tag may contain only ASCII alphanumerics, '-', '_', '/'",
        ));
    }
    Ok(())
}
