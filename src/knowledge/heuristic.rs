//! Heuristics for “knowledge gap” during chat.

/// Why a search was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnowledgeGap {
    ExplicitSearch,
    UnknownEntity,
    LowConfidenceCue,
}

/// Detect whether the user turn suggests missing external knowledge.
pub fn needs_knowledge(user: &str) -> Option<(KnowledgeGap, String)> {
    let trimmed = user.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some(q) = strip_explicit_search(trimmed) {
        return Some((KnowledgeGap::ExplicitSearch, q));
    }

    let lower = trimmed.to_ascii_lowercase();
    const LOW_CONF: &[&str] = &[
        "i don't know",
        "i do not know",
        "わからない",
        "分からない",
        "知らない",
        "調べて",
        "look up",
        "search for",
        "what is the latest",
        "最新の",
    ];
    for cue in LOW_CONF {
        if lower.contains(cue) {
            let q = extract_topic(trimmed);
            return Some((KnowledgeGap::LowConfidenceCue, q));
        }
    }

    // Capitalized multi-word proper-noun-ish spans (ASCII) as unknown entities.
    if let Some(entity) = find_unknown_entity(trimmed) {
        return Some((KnowledgeGap::UnknownEntity, entity));
    }

    None
}

fn strip_explicit_search(s: &str) -> Option<String> {
    let lower = s.to_ascii_lowercase();
    const PREFIXES: &[&str] = &[
        "/search ",
        "search:",
        "search ",
        "調べて:",
        "調べて：",
        "検索:",
        "検索：",
        "検索して ",
    ];
    for p in PREFIXES {
        if lower.starts_with(p) || s.starts_with(p) {
            let rest = s.get(p.len()..).unwrap_or("").trim();
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}

fn extract_topic(s: &str) -> String {
    let mut t = s.to_string();
    for cue in [
        "調べて",
        "search for",
        "look up",
        "what is the latest",
        "最新の",
    ] {
        if let Some(idx) = t.to_ascii_lowercase().find(cue) {
            t = t[idx + cue.len()..].trim().to_string();
            break;
        }
    }
    truncate_query(&t)
}

fn find_unknown_entity(s: &str) -> Option<String> {
    // Skip short / command-like turns.
    if s.len() < 12 || s.starts_with('/') {
        return None;
    }
    let mut best: Option<String> = None;
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if clean.len() >= 3
            && clean.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            && clean.chars().skip(1).any(|c| c.is_ascii_lowercase())
        {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&clean);
        } else if !cur.is_empty() {
            if cur.contains(' ') || cur.len() >= 6 {
                best = Some(cur.clone());
            }
            cur.clear();
        }
    }
    if !cur.is_empty() && (cur.contains(' ') || cur.len() >= 6) {
        best = Some(cur);
    }
    best.map(|b| truncate_query(&b))
}

fn truncate_query(s: &str) -> String {
    let t = s.trim();
    if t.chars().count() <= 160 {
        t.to_string()
    } else {
        t.chars().take(160).collect()
    }
}
