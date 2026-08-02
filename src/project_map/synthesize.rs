//! Partial-subgraph summaries for `--project-map` prompt injection.

use crate::error::Result;
use crate::project_map::build::{DecodedNode, MapMeta};
use crate::project_map::fetch::{fetch_nodes, ProjectMapReader};
use crate::project_map::query::{select_nodes, QueryOpts};

#[derive(Debug, Clone)]
pub struct ProjectContextOpts {
    pub max_nodes: usize,
    pub max_chars: usize,
    pub graph_hops: usize,
}

impl Default for ProjectContextOpts {
    fn default() -> Self {
        Self {
            max_nodes: 12,
            max_chars: 2_400,
            graph_hops: 1,
        }
    }
}

/// Select related nodes, DMA-fetch payloads, and synthesize a compact overview.
pub fn synthesize_context(
    reader: &mut ProjectMapReader,
    user_query: &str,
    opts: &ProjectContextOpts,
) -> Result<String> {
    let qopts = QueryOpts {
        max_nodes: opts.max_nodes,
        graph_hops: opts.graph_hops,
    };
    let ids = select_nodes(&reader.meta, user_query, &qopts);
    if ids.is_empty() {
        return Ok(String::new());
    }
    let nodes = match fetch_nodes(reader, &ids) {
        Ok(n) => n,
        Err(e) => {
            // Fallback: meta-only summary without DMA payloads.
            eprintln!("project-map DMA fetch failed ({e}); using meta-only summary");
            return Ok(synthesize_from_meta(&reader.meta, &ids, opts.max_chars));
        }
    };
    Ok(synthesize_from_decoded(
        &reader.meta,
        &ids,
        &nodes,
        opts.max_chars,
    ))
}

fn synthesize_from_decoded(
    meta: &MapMeta,
    ids: &[u32],
    nodes: &[DecodedNode],
    max_chars: usize,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Project map overview ({} nodes, {} edges; showing {}):\n",
        meta.node_count,
        meta.edge_count,
        nodes.len()
    ));

    for n in nodes {
        let line = format!(
            "- {} @ {}:{} — {}\n  deps: {}\n",
            n.name,
            n.file,
            n.line,
            truncate(&n.signature, 120),
            neighbor_names(meta, find_id_by_name(meta, &n.name), 4)
        );
        if out.len() + line.len() > max_chars {
            break;
        }
        out.push_str(&line);
    }

    // Impact range: callers of top-selected nodes.
    if !ids.is_empty() {
        let impact = impact_hints(meta, ids, 6);
        if !impact.is_empty() {
            let block = format!("Impact / callers: {impact}\n");
            if out.len() + block.len() <= max_chars {
                out.push_str(&block);
            }
        }
    }
    out
}

fn synthesize_from_meta(meta: &MapMeta, ids: &[u32], max_chars: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Project map overview (meta-only; {} nodes / {} edges):\n",
        meta.node_count, meta.edge_count
    ));
    for id in ids {
        let Some(n) = meta.nodes.iter().find(|x| x.id == *id) else {
            continue;
        };
        let line = format!(
            "- {} ({}) @ {}:{} — neighbors: {}\n",
            n.name,
            n.kind,
            n.file,
            n.line,
            neighbor_names(meta, Some(*id), 4)
        );
        if out.len() + line.len() > max_chars {
            break;
        }
        out.push_str(&line);
    }
    out
}

fn find_id_by_name(meta: &MapMeta, name: &str) -> Option<u32> {
    meta.nodes.iter().find(|n| n.name == name).map(|n| n.id)
}

fn neighbor_names(meta: &MapMeta, id: Option<u32>, limit: usize) -> String {
    let Some(id) = id else {
        return String::from("(none)");
    };
    let mut names = Vec::new();
    for e in &meta.edges {
        if e.from == id {
            if let Some(n) = meta.nodes.iter().find(|x| x.id == e.to) {
                names.push(n.name.clone());
            }
        }
        if names.len() >= limit {
            break;
        }
    }
    if names.is_empty() {
        "(none)".into()
    } else {
        names.join(", ")
    }
}

fn impact_hints(meta: &MapMeta, ids: &[u32], limit: usize) -> String {
    let mut callers = Vec::new();
    for &id in ids {
        for e in &meta.edges {
            if e.to == id {
                if let Some(n) = meta.nodes.iter().find(|x| x.id == e.from) {
                    let hint = format!("{}→{}", n.name, meta.nodes.iter().find(|x| x.id == id).map(|x| x.name.as_str()).unwrap_or("?"));
                    if !callers.contains(&hint) {
                        callers.push(hint);
                    }
                }
            }
            if callers.len() >= limit {
                break;
            }
        }
        if callers.len() >= limit {
            break;
        }
    }
    callers.join("; ")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect::<String>() + "…"
    }
}
