//! Select related nodes: BM25-ish lexical + embedding cosine + graph neighborhood.

use std::collections::{BTreeSet, HashMap};

use crate::project_map::build::{MapMeta, NodeMeta};
use crate::project_map::embed::{cosine, hash_embed, EMBED_DIM};

#[derive(Debug, Clone)]
pub struct QueryOpts {
    pub max_nodes: usize,
    pub graph_hops: usize,
}

impl Default for QueryOpts {
    fn default() -> Self {
        Self {
            max_nodes: 12,
            graph_hops: 1,
        }
    }
}

/// Rank and expand a node id set for the user query.
pub fn select_nodes(meta: &MapMeta, query: &str, opts: &QueryOpts) -> Vec<u32> {
    if meta.nodes.is_empty() || opts.max_nodes == 0 {
        return Vec::new();
    }
    let q_terms = tokenize(query);
    let q_emb = hash_embed(query);

    let mut scored: Vec<(f32, u32)> = meta
        .nodes
        .iter()
        .map(|n| {
            let lex = bm25_like(&q_terms, n);
            let text = format!("{} {} {}", n.name, n.kind, n.file);
            let emb = hash_embed(&text);
            let sim = cosine(&q_emb, &emb);
            let score = lex * 2.0 + sim;
            (score, n.id)
        })
        .filter(|(s, _)| *s > 0.05)
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<u32> = scored
        .into_iter()
        .take(opts.max_nodes.min(8))
        .map(|(_, id)| id)
        .collect();

    if selected.is_empty() {
        // Fallback: first N nodes (stable overview).
        return meta
            .nodes
            .iter()
            .take(opts.max_nodes.min(6))
            .map(|n| n.id)
            .collect();
    }

    expand_graph(meta, &mut selected, opts.graph_hops, opts.max_nodes);
    selected
}

fn expand_graph(meta: &MapMeta, selected: &mut Vec<u32>, hops: usize, max_nodes: usize) {
    let mut set: BTreeSet<u32> = selected.iter().copied().collect();
    let mut frontier = selected.clone();
    for _ in 0..hops {
        let mut next = Vec::new();
        for &id in &frontier {
            for e in &meta.edges {
                if e.from == id && set.insert(e.to) {
                    next.push(e.to);
                }
                if e.to == id && set.insert(e.from) {
                    next.push(e.from);
                }
                if set.len() >= max_nodes {
                    break;
                }
            }
            if set.len() >= max_nodes {
                break;
            }
        }
        frontier = next;
        if frontier.is_empty() || set.len() >= max_nodes {
            break;
        }
    }
    *selected = set.into_iter().take(max_nodes).collect();
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

fn bm25_like(terms: &[String], n: &NodeMeta) -> f32 {
    if terms.is_empty() {
        return 0.0;
    }
    let hay = format!("{} {} {}", n.name.to_ascii_lowercase(), n.file.to_ascii_lowercase(), n.kind);
    let mut score = 0.0f32;
    for t in terms {
        if hay.contains(t) {
            score += 1.0;
            if n.name.eq_ignore_ascii_case(t) {
                score += 2.0;
            }
        }
    }
    score
}

/// Build adjacency for callers (used by synthesize).
#[allow(dead_code)]
pub fn adjacency(meta: &MapMeta) -> HashMap<u32, Vec<u32>> {
    let mut m: HashMap<u32, Vec<u32>> = HashMap::new();
    for e in &meta.edges {
        m.entry(e.from).or_default().push(e.to);
    }
    m
}

#[allow(dead_code)]
pub fn _embed_dim() -> usize {
    EMBED_DIM
}
