//! Phase 3: ultra-light router agent (intent → adapter / expert hints).
//!
//! Runs a tiny classifier (default: SmolLM2 360M) under the same `--ram-mib`
//! budget, then **drops** router weights + KV before the main model loads so
//! the two never share RAM concurrently.

use console::style;
use tokenizers::Tokenizer;

use crate::catalog::{self, ModelEntry};
use crate::engine::Engine;
use crate::error::{AppError, Result};
use crate::store::{InstalledAdapter, LocalStore};

/// Default router model (already in the catalog; ChatML).
#[allow(dead_code)]
pub const DEFAULT_ROUTER_MODEL: &str = "smollm2:360m";

/// Structured routing decision produced by the agent.
#[derive(Debug, Clone, Default)]
pub struct RouteDecision {
    /// Adapter name to bind (must exist under `adapters/`), if any.
    pub adapter: Option<String>,
    /// Soft expert-id affinity for MoE prefetch (may be empty).
    pub expert_hints: Vec<usize>,
    /// Free-text domain label from the classifier (debug / logs).
    pub intent: String,
    /// Raw model output (truncated) for debugging.
    #[allow(dead_code)]
    pub raw: String,
}

/// Classify `user_prompt` with the router model, then free its memory.
///
/// Memory exclusivity: the returned [`RouteDecision`] holds no engine state.
/// Callers must ensure the router [`Engine`] is dropped before constructing
/// the main hybrid/eager session.
pub fn classify_intent(
    store: &LocalStore,
    user_prompt: &str,
    available_adapters: &[InstalledAdapter],
    router_name: &str,
) -> Result<RouteDecision> {
    let entry = catalog::find(router_name).ok_or_else(|| {
        AppError::msg(format!(
            "router model `{router_name}` not in catalog — pull it first"
        ))
    })?;
    let installed = store.resolve(&entry)?.ok_or_else(|| {
        AppError::msg(format!(
            "router `{router_name}` is not installed — run `lpc-llm pull {router_name}`"
        ))
    })?;

    eprintln!(
        "{} agent: loading router `{}` (exclusive RAM slice) …",
        style("·").cyan(),
        style(router_name).bold()
    );

    let adapter_names: Vec<&str> = available_adapters.iter().map(|a| a.name.as_str()).collect();
    let classify_prompt = build_classify_prompt(user_prompt, &adapter_names, &entry);

    let compute = crate::device::ComputeContext::from_pref(crate::config::ComputeDevicePref::Cpu)?;
    let mut engine = Engine::load(&installed.model_path, compute)?;
    let tokenizer = Tokenizer::from_file(&installed.tokenizer_path)
        .map_err(|e| AppError::msg(format!("router tokenizer: {e}")))?;

    let mut acc = String::new();
    let raw = engine.generate(&tokenizer, &classify_prompt, 48, 0.1, |piece| {
        acc.push_str(piece);
        Ok(())
    })?;

    // Explicitly drop router weights + any internal KV before returning.
    drop(engine);
    drop(tokenizer);

    eprintln!(
        "{} agent: router released — handing off to main model",
        style("✓").green()
    );

    let decision = parse_route_output(&raw, available_adapters);
    eprintln!(
        "{} agent: intent={} adapter={} experts={:?}",
        style("·").cyan(),
        decision.intent,
        decision.adapter.as_deref().unwrap_or("(none)"),
        decision.expert_hints
    );
    Ok(decision)
}

fn build_classify_prompt(user: &str, adapters: &[&str], entry: &ModelEntry) -> String {
    let adapter_list = if adapters.is_empty() {
        "(none installed)".to_string()
    } else {
        adapters.join(", ")
    };
    let system = format!(
        "You are a routing classifier for a local LLM. \
         Given the user message, reply with EXACTLY one line:\n\
         INTENT=<label> ADAPTER=<name|none> EXPERTS=<comma ids or none>\n\
         Available adapters: {adapter_list}\n\
         Labels: coding, chat, python, rust, math, other.\n\
         Pick ADAPTER only from the available list (or none). \
         EXPERTS are optional MoE expert id hints (0-7) or none."
    );
    // Reuse ChatML / Gemma wrapping via a synthetic single-turn.
    let body = format!("{system}\n\nUser message:\n{user}");
    entry.format_prompt(&body, &[])
}

/// Parse classifier text; falls back to keyword heuristics.
pub fn parse_route_output(raw: &str, adapters: &[InstalledAdapter]) -> RouteDecision {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| l.to_ascii_uppercase().contains("INTENT=") || l.contains("ADAPTER="))
        .unwrap_or(raw.trim());

    let mut intent = String::from("other");
    let mut adapter: Option<String> = None;
    let mut expert_hints = Vec::new();

    for tok in line.split_whitespace() {
        let upper = tok.to_ascii_uppercase();
        if let Some(v) = upper.strip_prefix("INTENT=") {
            intent = v.to_ascii_lowercase();
            // Recover original casing from tok after '='.
            if let Some((_, rest)) = tok.split_once('=') {
                intent = rest.trim().to_ascii_lowercase();
            }
        } else if let Some((_, rest)) = tok.split_once('=') {
            let key = tok[..tok.len() - rest.len() - 1].to_ascii_uppercase();
            if key == "ADAPTER" {
                let name = rest.trim();
                if !name.is_empty() && !name.eq_ignore_ascii_case("none") {
                    if let Some(a) = adapters.iter().find(|a| a.name.eq_ignore_ascii_case(name)) {
                        adapter = Some(a.name.clone());
                    }
                }
            } else if key == "EXPERTS" {
                let name = rest.trim();
                if !name.is_empty() && !name.eq_ignore_ascii_case("none") {
                    for part in name.split(',') {
                        if let Ok(id) = part.trim().parse::<usize>() {
                            expert_hints.push(id);
                        }
                    }
                }
            }
        }
    }

    // Heuristic fallback when the model ignored the schema.
    if adapter.is_none() {
        adapter = heuristic_adapter(raw, adapters).or_else(|| heuristic_adapter(line, adapters));
    }
    if intent == "other" {
        intent = heuristic_intent(raw);
    }
    if expert_hints.is_empty() {
        expert_hints = heuristic_experts(&intent);
    }

    RouteDecision {
        adapter,
        expert_hints,
        intent,
        raw: raw.chars().take(200).collect(),
    }
}

fn heuristic_intent(text: &str) -> String {
    let t = text.to_ascii_lowercase();
    if t.contains("python") || t.contains("pip ") || t.contains("django") {
        "python".into()
    } else if t.contains("rust") || t.contains("cargo") || t.contains("lifetime") {
        "rust".into()
    } else if t.contains("code")
        || t.contains("function")
        || t.contains("compile")
        || t.contains("bug")
    {
        "coding".into()
    } else if t.contains("math") || t.contains("integral") || t.contains("equation") {
        "math".into()
    } else {
        "chat".into()
    }
}

fn heuristic_adapter(text: &str, adapters: &[InstalledAdapter]) -> Option<String> {
    let t = text.to_ascii_lowercase();
    // Prefer name substring match: "python-expert" for python intent, etc.
    let intent = heuristic_intent(&t);
    adapters
        .iter()
        .find(|a| a.name.to_ascii_lowercase().contains(&intent))
        .or_else(|| {
            adapters.iter().find(|a| {
                let n = a.name.to_ascii_lowercase();
                t.split_whitespace().any(|w| w.len() > 3 && n.contains(w))
            })
        })
        .map(|a| a.name.clone())
}

fn heuristic_experts(intent: &str) -> Vec<usize> {
    // Soft affinity only — real Top-K still comes from the gating network.
    match intent {
        "python" | "coding" => vec![0, 1],
        "rust" => vec![2, 3],
        "math" => vec![4, 5],
        _ => Vec::new(),
    }
}

/// Soft MiB estimate for the router while it is loaded (weights ≈ blob size).
pub fn router_ram_hint_mib(router_name: &str) -> usize {
    match router_name {
        "smollm2:360m" => 512,
        _ => 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InstalledAdapter;
    use std::path::PathBuf;

    fn adapters() -> Vec<InstalledAdapter> {
        vec![
            InstalledAdapter {
                name: "python-expert".into(),
                path: PathBuf::from("/tmp/python-expert"),
                base_model: "gemma2:2b".into(),
                recorded_at_unix: 0,
            },
            InstalledAdapter {
                name: "demo-zero".into(),
                path: PathBuf::from("/tmp/demo-zero"),
                base_model: "gemma2:2b".into(),
                recorded_at_unix: 0,
            },
        ]
    }

    #[test]
    fn parse_schema_line() {
        let d = parse_route_output(
            "INTENT=python ADAPTER=python-expert EXPERTS=0,1\nmore junk",
            &adapters(),
        );
        assert_eq!(d.intent, "python");
        assert_eq!(d.adapter.as_deref(), Some("python-expert"));
        assert_eq!(d.expert_hints, vec![0, 1]);
    }

    #[test]
    fn heuristic_picks_python_adapter() {
        let d = parse_route_output("please fix my python django view", &adapters());
        assert_eq!(d.adapter.as_deref(), Some("python-expert"));
        assert_eq!(d.intent, "python");
    }
}
