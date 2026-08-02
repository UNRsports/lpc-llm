//! Built-in model catalog (Ollama-style `family:tag` names).

use serde::{Deserialize, Serialize};

/// How to wrap a user turn for instruct checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptStyle {
    /// ChatML (`<|im_start|>…`) — SmolLM2, Qwen2.5, etc.
    ChatMl,
    /// Gemma instruct turns.
    Gemma,
    /// Pass the user string through unchanged.
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Ollama-style name, e.g. `gemma2:2b`.
    pub name: String,
    pub display: String,
    pub hf_repo: String,
    pub gguf_file: String,
    pub tokenizer_repo: String,
    pub approx_size: String,
    pub min_ram_hint: String,
    pub prompt_style: PromptStyle,
}

impl ModelEntry {
    pub fn format_prompt(&self, user: &str, history: &[(String, String)]) -> String {
        match self.prompt_style {
            PromptStyle::ChatMl => {
                let mut s = String::new();
                for (u, a) in history {
                    s.push_str("<|im_start|>user\n");
                    s.push_str(u);
                    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
                    s.push_str(a);
                    s.push_str("<|im_end|>\n");
                }
                s.push_str("<|im_start|>user\n");
                s.push_str(user);
                s.push_str("<|im_end|>\n<|im_start|>assistant\n");
                s
            }
            PromptStyle::Gemma => {
                let mut s = String::new();
                for (u, a) in history {
                    s.push_str("<start_of_turn>user\n");
                    s.push_str(u);
                    s.push_str("<end_of_turn>\n<start_of_turn>model\n");
                    s.push_str(a);
                    s.push_str("<end_of_turn>\n");
                }
                s.push_str("<start_of_turn>user\n");
                s.push_str(user);
                s.push_str("<end_of_turn>\n<start_of_turn>model\n");
                s
            }
            PromptStyle::Raw => {
                if history.is_empty() {
                    user.to_string()
                } else {
                    let mut s = String::new();
                    for (u, a) in history {
                        s.push_str("User: ");
                        s.push_str(u);
                        s.push_str("\nAssistant: ");
                        s.push_str(a);
                        s.push('\n');
                    }
                    s.push_str("User: ");
                    s.push_str(user);
                    s.push_str("\nAssistant:");
                    s
                }
            }
        }
    }
}

/// Static catalog shipped with the binary.
pub fn catalog() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            name: "smollm2:360m".into(),
            display: "SmolLM2 360M Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/SmolLM2-360M-Instruct-GGUF".into(),
            gguf_file: "SmolLM2-360M-Instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "HuggingFaceTB/SmolLM2-360M-Instruct".into(),
            approx_size: "~260 MB".into(),
            min_ram_hint: "~1 GB".into(),
            prompt_style: PromptStyle::ChatMl,
        },
        ModelEntry {
            name: "gemma2:2b".into(),
            display: "Gemma 2 2B Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/gemma-2-2b-it-GGUF".into(),
            gguf_file: "gemma-2-2b-it-Q4_K_M.gguf".into(),
            // google/gemma-2-2b-it is gated; unsloth mirrors tokenizer.json
            tokenizer_repo: "unsloth/gemma-2-2b-it".into(),
            approx_size: "~1.7 GB".into(),
            min_ram_hint: "~4 GB".into(),
            prompt_style: PromptStyle::Gemma,
        },
        ModelEntry {
            name: "qwen2.5:1.5b".into(),
            display: "Qwen2.5 1.5B Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF".into(),
            gguf_file: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct".into(),
            approx_size: "~1.1 GB".into(),
            min_ram_hint: "~3 GB".into(),
            prompt_style: PromptStyle::ChatMl,
        },
        ModelEntry {
            name: "phi3:mini".into(),
            display: "Phi-3 Mini 4K Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/Phi-3-mini-4k-instruct-GGUF".into(),
            gguf_file: "Phi-3-mini-4k-instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct".into(),
            approx_size: "~2.2 GB".into(),
            min_ram_hint: "~5 GB".into(),
            prompt_style: PromptStyle::ChatMl,
        },
    ]
}

pub fn find(name: &str) -> Option<ModelEntry> {
    catalog().into_iter().find(|e| e.name == name)
}

/// Synthetic catalog entry for locally trained / imported models (not in `catalog()`).
pub fn entry_for_local(name: &str, gguf_file: &str, tokenizer_repo: &str) -> ModelEntry {
    let safe = name.replace([':', '/'], "_");
    ModelEntry {
        name: name.to_string(),
        display: format!("Local / trained model `{name}`"),
        hf_repo: format!("local/{safe}"),
        gguf_file: gguf_file.to_string(),
        tokenizer_repo: tokenizer_repo.to_string(),
        approx_size: "custom".into(),
        min_ram_hint: "varies".into(),
        prompt_style: PromptStyle::Raw,
    }
}
