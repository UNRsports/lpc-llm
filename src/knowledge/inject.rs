//! RAG-style knowledge injection into the user turn (token/char budget aware).

use crate::error::Result;
use crate::knowledge::store::{KnowledgeChunk, KnowledgeStore};

/// Options for prompt-side knowledge synthesis.
#[derive(Debug, Clone)]
pub struct KnowledgeInjectOpts {
    pub max_chunks: usize,
    /// Soft character budget for injected context (approx. token proxy).
    pub max_chars: usize,
}

impl Default for KnowledgeInjectOpts {
    fn default() -> Self {
        Self {
            max_chunks: 3,
            max_chars: 1_500,
        }
    }
}

/// Retrieve related chunks and prepend a compact knowledge block to `user`.
pub fn inject_knowledge(
    store: &KnowledgeStore,
    user: &str,
    opts: &KnowledgeInjectOpts,
) -> Result<(String, Vec<KnowledgeChunk>)> {
    let chunks = store.retrieve(user, opts.max_chunks, opts.max_chars)?;
    if chunks.is_empty() {
        return Ok((user.to_string(), chunks));
    }
    let block = synthesize_block(&chunks, opts.max_chars);
    let mut enriched = String::with_capacity(block.len() + user.len() + 32);
    enriched.push_str("[Retrieved knowledge]\n");
    enriched.push_str(&block);
    enriched.push_str("\n\n[User]\n");
    enriched.push_str(user);
    Ok((enriched, chunks))
}

fn synthesize_block(chunks: &[KnowledgeChunk], max_chars: usize) -> String {
    let mut out = String::new();
    for (i, c) in chunks.iter().enumerate() {
        let header = if c.url.is_empty() {
            format!("{}. {}", i + 1, c.title)
        } else {
            format!("{}. {} ({})", i + 1, c.title, c.url)
        };
        let piece = format!("{header}\n{}\n", c.text.trim());
        if out.len() + piece.len() > max_chars && !out.is_empty() {
            break;
        }
        if out.len() + piece.len() > max_chars {
            let remain = max_chars.saturating_sub(out.len());
            out.push_str(&truncate_bytes_safe(&piece, remain));
            break;
        }
        out.push_str(&piece);
    }
    out
}

fn truncate_bytes_safe(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}
