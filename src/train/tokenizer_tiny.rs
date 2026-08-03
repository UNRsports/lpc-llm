//! Minimal character-level tokenizer for tiny from-scratch models.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::error::{AppError, Result};

pub const UNK: &str = "<unk>";
pub const PAD: &str = "<pad>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";

/// Build a char-level WordLevel tokenizer covering specials + chars in `texts`
/// (plus printable ASCII), save to `path`, return (tokenizer, vocab_size).
pub fn build_char_tokenizer(texts: &[String], path: impl AsRef<Path>) -> Result<(Tokenizer, usize)> {
    let mut chars: BTreeSet<char> = BTreeSet::new();
    for c in 0x20u8..=0x7eu8 {
        chars.insert(c as char);
    }
    chars.insert('\n');
    chars.insert('\t');
    for text in texts {
        for c in text.chars() {
            chars.insert(c);
        }
    }

    let mut vocab: BTreeMap<String, u32> = BTreeMap::new();
    vocab.insert(UNK.into(), 0);
    vocab.insert(PAD.into(), 1);
    vocab.insert(BOS.into(), 2);
    vocab.insert(EOS.into(), 3);
    let mut id = 4u32;
    for c in chars {
        let s = c.to_string();
        if vocab.contains_key(&s) {
            continue;
        }
        vocab.insert(s, id);
        id += 1;
    }
    let vocab_size = vocab.len();

    let model = WordLevel::builder()
        .unk_token(UNK.into())
        .vocab(vocab.into_iter().collect())
        .build()
        .map_err(|e| AppError::msg(format!("WordLevel tokenizer: {e}")))?;

    let mut tokenizer = Tokenizer::new(model);
    let split = Split::new(
        SplitPattern::Regex(r"(?s).".into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| AppError::msg(format!("char split pre-tokenizer: {e}")))?;
    tokenizer.with_pre_tokenizer(Some(split));

    tokenizer.add_special_tokens(&[
        AddedToken::from(UNK, true),
        AddedToken::from(PAD, true),
        AddedToken::from(BOS, true),
        AddedToken::from(EOS, true),
    ]);

    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    tokenizer
        .save(path, false)
        .map_err(|e| AppError::msg(format!("save tokenizer {}: {e}", path.display())))?;

    // Reload to ensure on-disk form round-trips with tokenizers crate.
    let tokenizer = Tokenizer::from_file(path).map_err(|e| {
        AppError::msg(format!("reload tokenizer {}: {e}", path.display()))
    })?;
    Ok((tokenizer, vocab_size))
}

#[allow(dead_code)]
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let path = path.as_ref();
    Tokenizer::from_file(path).map_err(|e| {
        AppError::msg(format!("load tokenizer {}: {e}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_tokenizer_roundtrip_file() {
        let dir = std::env::temp_dir().join(format!("lpc-tok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        let texts = vec!["hello\n世界".into()];
        let (tok, n) = build_char_tokenizer(&texts, &path).unwrap();
        assert!(n > 4);
        let enc = tok.encode("hi", true).unwrap();
        assert!(!enc.get_ids().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
