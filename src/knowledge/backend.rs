//! Pluggable search backends (DuckDuckGo Instant Answer / Custom HTTP API).

use serde::Deserialize;
use serde_json::Value;

use crate::error::{AppError, Result};

/// Selected search backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBackendKind {
    /// DuckDuckGo Instant Answer JSON API (default).
    DuckDuckGo,
    /// `GET $LPC_LLM_SEARCH_URL?q=<query>` expecting JSON hits.
    CustomHttp,
}

impl SearchBackendKind {
    /// Resolve from `LPC_LLM_SEARCH_BACKEND` (`duckduckgo` | `custom`).
    pub fn from_env() -> Self {
        match std::env::var("LPC_LLM_SEARCH_BACKEND")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "custom" | "http" | "custom_http" => Self::CustomHttp,
            _ => Self::DuckDuckGo,
        }
    }
}

/// One search result hit.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Run a search against the configured backend.
pub fn search_query(query: &str) -> Result<Vec<SearchHit>> {
    let q = query.trim();
    if q.is_empty() {
        return Err(AppError::msg("search query must be non-empty"));
    }
    if q.len() > 512 {
        return Err(AppError::msg("search query too long (max 512 chars)"));
    }
    match SearchBackendKind::from_env() {
        SearchBackendKind::DuckDuckGo => search_duckduckgo(q),
        SearchBackendKind::CustomHttp => search_custom_http(q),
    }
}

fn search_duckduckgo(query: &str) -> Result<Vec<SearchHit>> {
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding_encode(query)
    );
    let body = http_get_text(&url)?;
    let parsed: DdgResponse = serde_json::from_str(&body).unwrap_or(DdgResponse::default());
    let mut hits = Vec::new();

    if !parsed.abstract_text.trim().is_empty() {
        hits.push(SearchHit {
            title: if parsed.heading.is_empty() {
                query.to_string()
            } else {
                parsed.heading
            },
            url: parsed.abstract_url,
            snippet: parsed.abstract_text,
        });
    }

    for topic in parsed.related_topics {
        collect_related(&topic, &mut hits, 8);
        if hits.len() >= 8 {
            break;
        }
    }

    if hits.is_empty() && !parsed.answer.trim().is_empty() {
        hits.push(SearchHit {
            title: query.to_string(),
            url: String::new(),
            snippet: parsed.answer,
        });
    }

    Ok(hits)
}

fn collect_related(value: &Value, out: &mut Vec<SearchHit>, limit: usize) {
    if out.len() >= limit {
        return;
    }
    match value {
        Value::Object(map) => {
            if let Some(Value::String(text)) = map.get("Text") {
                let url = map
                    .get("FirstURL")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let snippet = text.trim().to_string();
                if snippet.is_empty() {
                    return;
                }
                let title = snippet
                    .split(" - ")
                    .next()
                    .unwrap_or(&snippet)
                    .chars()
                    .take(80)
                    .collect();
                out.push(SearchHit {
                    title,
                    url,
                    snippet,
                });
            }
            if let Some(Value::Array(topics)) = map.get("Topics") {
                for t in topics {
                    collect_related(t, out, limit);
                    if out.len() >= limit {
                        return;
                    }
                }
            }
        }
        Value::Array(arr) => {
            for t in arr {
                collect_related(t, out, limit);
                if out.len() >= limit {
                    return;
                }
            }
        }
        _ => {}
    }
}

fn search_custom_http(query: &str) -> Result<Vec<SearchHit>> {
    let base = std::env::var("LPC_LLM_SEARCH_URL").map_err(|_| {
        AppError::msg(
            "LPC_LLM_SEARCH_BACKEND=custom requires LPC_LLM_SEARCH_URL \
             (GET ?q=… returning JSON array of {title,url,snippet})",
        )
    })?;
    if base.trim().is_empty() {
        return Err(AppError::msg("LPC_LLM_SEARCH_URL is empty"));
    }
    let sep = if base.contains('?') { '&' } else { '?' };
    let url = format!("{base}{sep}q={}", urlencoding_encode(query));
    let body = http_get_text(&url)?;
    let value: Value = serde_json::from_str(&body)?;
    parse_custom_hits(&value)
}

fn parse_custom_hits(value: &Value) -> Result<Vec<SearchHit>> {
    let arr = match value {
        Value::Array(a) => a,
        Value::Object(m) => m
            .get("results")
            .or_else(|| m.get("hits"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| AppError::msg("custom search JSON must be an array or {results|hits}"))?,
        _ => {
            return Err(AppError::msg(
                "custom search JSON must be an array or object with results/hits",
            ));
        }
    };
    let mut hits = Vec::new();
    for item in arr.iter().take(16) {
        let title = item
            .get("title")
            .or_else(|| item.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("result")
            .to_string();
        let url = item
            .get("url")
            .or_else(|| item.get("href"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let snippet = item
            .get("snippet")
            .or_else(|| item.get("text"))
            .or_else(|| item.get("body"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if snippet.trim().is_empty() && title.trim().is_empty() {
            continue;
        }
        hits.push(SearchHit {
            title,
            url,
            snippet,
        });
    }
    Ok(hits)
}

fn http_get_text(url: &str) -> Result<String> {
    // Prefer system curl so we stay free of rustls/ring (no C toolchain).
    let out = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--connect-timeout",
            "8",
            "--max-time",
            "20",
            "-A",
            "lpc-llm/0.1 (local knowledge)",
            url,
        ])
        .output()
        .map_err(|e| {
            AppError::msg(format!(
                "search requires `curl` on PATH (failed to spawn: {e})"
            ))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(AppError::msg(format!(
            "search HTTP failed (curl exit {:?}): {}",
            out.status.code(),
            stderr.trim()
        )));
    }
    String::from_utf8(out.stdout).map_err(|e| AppError::msg(format!("search UTF-8: {e}")))
}

/// Minimal URL-encoding for query strings (UTF-8 safe).
fn urlencoding_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                out.push(HEX[(b >> 4) as usize] as char);
                out.push(HEX[(b & 0xf) as usize] as char);
            }
        }
    }
    out
}

#[derive(Debug, Default, Deserialize)]
struct DdgResponse {
    #[serde(default, rename = "Heading")]
    heading: String,
    #[serde(default, rename = "AbstractText")]
    abstract_text: String,
    #[serde(default, rename = "AbstractURL")]
    abstract_url: String,
    #[serde(default, rename = "Answer")]
    answer: String,
    #[serde(default, rename = "RelatedTopics")]
    related_topics: Vec<Value>,
}
