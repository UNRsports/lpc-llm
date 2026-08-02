//! Shared training corpus loaders (text SFT + preference pairs).

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{AppError, Result};

/// Re-export Phase 4 loader for Phase 5 SFT / scratch.
pub use crate::adapter::load_training_texts;

#[derive(Debug, Clone)]
pub struct PreferencePair {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
}

#[derive(Debug, Deserialize)]
struct PrefRow {
    prompt: String,
    chosen: String,
    rejected: String,
}

/// Load DPO/ORPO preference rows from JSONL:
/// `{"prompt":"...","chosen":"...","rejected":"..."}`.
pub fn load_preference_pairs(path: impl AsRef<Path>) -> Result<Vec<PreferencePair>> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path).map_err(|e| {
        AppError::msg(format!("read preference file {}: {e}", path.display()))
    })?;
    if raw.trim().is_empty() {
        return Err(AppError::msg("preference file is empty"));
    }
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: PrefRow = serde_json::from_str(line).map_err(|e| {
            AppError::msg(format!(
                "preference jsonl line {}: {e} (need prompt/chosen/rejected)",
                i + 1
            ))
        })?;
        if row.prompt.trim().is_empty()
            || row.chosen.trim().is_empty()
            || row.rejected.trim().is_empty()
        {
            return Err(AppError::msg(format!(
                "preference jsonl line {}: empty prompt/chosen/rejected",
                i + 1
            )));
        }
        out.push(PreferencePair {
            prompt: row.prompt,
            chosen: row.chosen,
            rejected: row.rejected,
        });
    }
    if out.is_empty() {
        return Err(AppError::msg("preference file has no usable rows"));
    }
    Ok(out)
}

pub fn tokenize_chunks(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[String],
    max_seq: usize,
) -> Result<Vec<Vec<u32>>> {
    if max_seq < 2 {
        return Err(AppError::msg("--max-seq must be >= 2"));
    }
    let mut chunks = Vec::new();
    for text in texts {
        let encoding = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| AppError::msg(format!("tokenize: {e}")))?;
        let ids = encoding.get_ids();
        if ids.len() < 2 {
            continue;
        }
        let mut start = 0;
        while start < ids.len() {
            let end = (start + max_seq).min(ids.len());
            if end - start >= 2 {
                chunks.push(ids[start..end].to_vec());
            }
            if end >= ids.len() {
                break;
            }
            start = end.saturating_sub(max_seq / 4).max(start + 1);
        }
    }
    if chunks.is_empty() {
        return Err(AppError::msg(
            "no training chunks with >= 2 tokens (check --from / tokenizer)",
        ));
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_prefs() {
        let dir = std::env::temp_dir().join(format!("lpc-pref-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.jsonl");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"prompt":"hi","chosen":"yes","rejected":"no"}}"#
        )
        .unwrap();
        let rows = load_preference_pairs(&path).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].chosen, "yes");
        let _ = fs::remove_dir_all(&dir);
    }
}
