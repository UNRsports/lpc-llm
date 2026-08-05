//! Built-in model catalog (Ollama-style `family:tag` names).

use serde::{Deserialize, Serialize};

/// How to wrap a user turn for instruct checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptStyle {
    /// ChatML (`<|im_start|>…`) — SmolLM2, Qwen2.5, etc.
    ChatMl,
    /// Gemma 1/2/3 instruct turns (`<start_of_turn>`).
    Gemma,
    /// Gemma 4 IT turns (`<|turn>…`).
    Gemma4,
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
            PromptStyle::Gemma4 => {
                // Official non-thinking format: empty thought channel after model turn
                // so the model does not emit a visible "thought" stub at decode start.
                // See https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
                let mut s = String::from("<bos>");
                for (u, a) in history {
                    s.push_str("<|turn>user\n");
                    s.push_str(u);
                    s.push_str("<turn|>\n<|turn>model\n");
                    s.push_str("<|channel>thought\n<channel|>");
                    s.push_str(strip_gemma4_channels(a));
                    s.push_str("<turn|>\n");
                }
                s.push_str("<|turn>user\n");
                s.push_str(user);
                s.push_str("<turn|>\n<|turn>model\n");
                s.push_str("<|channel>thought\n<channel|>");
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

    /// Continue an unfinished assistant turn (no closing turn marker after `partial`).
    pub fn format_prompt_continue(
        &self,
        user: &str,
        prior_history: &[(String, String)],
        partial: &str,
    ) -> String {
        let mut s = self.format_prompt(user, prior_history);
        // format_prompt already opened the empty thought channel for Gemma4;
        // append only the visible answer fragment.
        s.push_str(strip_gemma4_channels(partial));
        s
    }
}

/// Remove Gemma 4 `<|channel>…<channel|>` wrappers (and a leading bare `thought`) from text.
pub fn strip_gemma4_channels(text: &str) -> &str {
    let mut s = text;
    // Full channel block(s).
    while let Some(start) = s.find("<|channel>") {
        if let Some(rel_end) = s[start..].find("<channel|>") {
            let end = start + rel_end + "<channel|>".len();
            s = s[end..].trim_start();
        } else {
            break;
        }
    }
    // Decoder may surface the channel name alone when special tokens are skipped.
    let trimmed = s.trim_start();
    if let Some(rest) = trimmed.strip_prefix("thought") {
        let rest = rest.trim_start_matches(['\r', '\n', ' ', '\t']);
        return rest;
    }
    s
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
            name: "gemma3:4b".into(),
            display: "Gemma 3 4B Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/google_gemma-3-4b-it-GGUF".into(),
            gguf_file: "google_gemma-3-4b-it-Q4_K_M.gguf".into(),
            tokenizer_repo: "unsloth/gemma-3-4b-it".into(),
            approx_size: "~2.5 GB".into(),
            min_ram_hint: "~6 GB".into(),
            prompt_style: PromptStyle::Gemma,
        },
        ModelEntry {
            name: "gemma3:12b".into(),
            display: "Gemma 3 12B Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/google_gemma-3-12b-it-GGUF".into(),
            gguf_file: "google_gemma-3-12b-it-Q4_K_M.gguf".into(),
            tokenizer_repo: "unsloth/gemma-3-12b-it".into(),
            approx_size: "~7.3 GB".into(),
            min_ram_hint: "~12 GB".into(),
            prompt_style: PromptStyle::Gemma,
        },
        ModelEntry {
            name: "gemma3:27b".into(),
            display: "Gemma 3 27B Instruct (Q4_K_M)".into(),
            hf_repo: "bartowski/google_gemma-3-27b-it-GGUF".into(),
            gguf_file: "google_gemma-3-27b-it-Q4_K_M.gguf".into(),
            tokenizer_repo: "unsloth/gemma-3-27b-it".into(),
            approx_size: "~16.5 GB".into(),
            min_ram_hint: "~20 GB (--hybrid streams layers; raise --ram-mib to pin more)".into(),
            prompt_style: PromptStyle::Gemma,
        },
        ModelEntry {
            name: "gemma4:26b-a4b".into(),
            display: "Gemma 4 26B-A4B Instruct MoE (Q4_K_M)".into(),
            hf_repo: "bartowski/google_gemma-4-26B-A4B-it-GGUF".into(),
            gguf_file: "google_gemma-4-26B-A4B-it-Q4_K_M.gguf".into(),
            // google/gemma-4 is gated; unsloth mirrors tokenizer.json
            tokenizer_repo: "unsloth/gemma-4-26B-A4B-it".into(),
            approx_size: "~17.0 GB disk (Q4_K_M); ~3.8B active / 128 experts Top-8 + shared".into(),
            min_ram_hint: "~16 GB (--hybrid --ram-mib 16384; experts on NVMe, cores+Top-K in RAM)"
                .into(),
            prompt_style: PromptStyle::Gemma4,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_gemma4_channel_and_bare_thought() {
        assert_eq!(
            strip_gemma4_channels("<|channel>thought\n<channel|>Hello"),
            "Hello"
        );
        assert_eq!(strip_gemma4_channels("thought\nHi"), "Hi");
        assert_eq!(strip_gemma4_channels("Hello"), "Hello");
    }

    #[test]
    fn gemma4_prompt_opens_empty_thought_channel() {
        let e = catalog()
            .into_iter()
            .find(|m| m.name == "gemma4:26b-a4b")
            .expect("gemma4 catalog");
        let p = e.format_prompt("ping", &[]);
        assert!(
            p.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "prompt should open empty thought channel, got: {p:?}"
        );
    }
}
