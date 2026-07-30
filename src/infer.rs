//! Interactive chat REPL (eager Candle load or hybrid io_uring prefetch).

use std::io::{self, Write};

use console::style;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokenizers::Tokenizer;

use crate::adapter::AdapterSet;
use crate::catalog::ModelEntry;
use crate::engine::Engine;
use crate::error::{AppError, Result};
use crate::hybrid::{HybridConfig, HybridEngine};
use crate::store::InstalledModel;

enum Backend {
    Eager(Engine),
    Hybrid(HybridEngine),
}

impl Backend {
    fn device_name(&self) -> &str {
        match self {
            Self::Eager(e) => e.device_name(),
            Self::Hybrid(e) => e.device_name(),
        }
    }

    fn architecture(&self) -> &str {
        match self {
            Self::Eager(e) => e.architecture(),
            Self::Hybrid(e) => e.architecture(),
        }
    }

    fn reset_state(&mut self) {
        match self {
            Self::Eager(e) => {
                let _ = e.reset_state();
            }
            Self::Hybrid(e) => e.reset_state(),
        }
    }

    fn first_burst(&self) -> usize {
        match self {
            Self::Eager(_) => usize::MAX,
            Self::Hybrid(e) => e.config().first_burst_tokens,
        }
    }

    fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        match self {
            Self::Eager(e) => e.generate(tokenizer, prompt, max_tokens, temperature, on_token),
            Self::Hybrid(e) => e.generate(tokenizer, prompt, max_tokens, temperature, on_token),
        }
    }

    fn io_hint(&self) -> Option<f64> {
        match self {
            Self::Hybrid(e) => Some(e.io_compute_ratio()),
            Self::Eager(_) => None,
        }
    }
}

pub struct ChatSession {
    backend: Backend,
    tokenizer: Tokenizer,
    entry: ModelEntry,
    history: Vec<(String, String)>,
    max_tokens: usize,
    temperature: f64,
}

impl ChatSession {
    pub fn load_with_config(
        installed: &InstalledModel,
        entry: ModelEntry,
        hybrid: bool,
        cfg: HybridConfig,
        pack_cache: &std::path::Path,
        adapter: Option<AdapterSet>,
    ) -> Result<Self> {
        if adapter.is_some() && !hybrid {
            return Err(AppError::msg(
                "--adapter requires hybrid inference (internal: hybrid flag was false)",
            ));
        }

        let adapter_label = adapter
            .as_ref()
            .map(|a| format!("+adapter:{}", a.name()))
            .unwrap_or_default();

        eprintln!(
            "{} loading {} ({}{}) …",
            style("·").cyan(),
            style(&entry.name).bold(),
            if hybrid {
                "hybrid pack+io_uring"
            } else {
                "eager"
            },
            adapter_label
        );

        let backend = if hybrid {
            Backend::Hybrid(HybridEngine::load_with_config(
                &installed.model_path,
                cfg,
                pack_cache,
                adapter,
            )?)
        } else {
            Backend::Eager(Engine::load(&installed.model_path)?)
        };

        let tokenizer = Tokenizer::from_file(&installed.tokenizer_path)
            .map_err(|e| AppError::msg(format!("tokenizer load: {e}")))?;

        eprintln!(
            "{} ready on {} ({})",
            style("✓").green(),
            backend.device_name(),
            backend.architecture()
        );

        Ok(Self {
            backend,
            tokenizer,
            entry,
            history: Vec::new(),
            // Cap default length so first reply arrives sooner (思考の小分け).
            max_tokens: 96,
            temperature: 0.7,
        })
    }

    pub fn run_repl(&mut self) -> Result<()> {
        println!(
            "{} {} — `/bye` exit, `/clear` reset, `/more` continue last (+96 tokens)",
            style(">>>").cyan().bold(),
            style(&self.entry.display).bold()
        );

        // rustyline edits by Unicode scalar / display width, so Backspace on
        // CJK (e.g. 「化」) removes one character instead of one UTF-8 byte.
        let mut rl = DefaultEditor::new().map_err(|e| {
            AppError::msg(format!("readline init: {e}"))
        })?;
        let mut last_user: Option<String> = None;

        loop {
            let line = match rl.readline(">>> ") {
                Ok(l) => l,
                Err(ReadlineError::Interrupted) => {
                    println!("{}", style("(Ctrl-C — type /bye to exit)").dim());
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!();
                    break;
                }
                Err(e) => return Err(AppError::msg(format!("readline: {e}"))),
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let _ = rl.add_history_entry(line);
            match line {
                "/bye" | "/exit" | "/quit" => break,
                "/clear" => {
                    self.history.clear();
                    last_user = None;
                    self.backend.reset_state();
                    println!("{}", style("(history cleared)").dim());
                    continue;
                }
                "/more" => {
                    if let Some(ref u) = last_user.clone() {
                        // Continue generation with another burst (小分け).
                        let reply = self.generate_burst(u, self.max_tokens)?;
                        if let Some(last) = self.history.last_mut() {
                            last.1.push_str(&reply);
                        }
                    } else {
                        println!("{}", style("(nothing to continue)").dim());
                    }
                    continue;
                }
                _ => {}
            }

            last_user = Some(line.to_string());
            // First burst: shorter cap for faster TTFT feel.
            let burst = self.backend.first_burst().min(self.max_tokens);
            let reply = self.generate_burst(line, burst)?;
            self.history.push((line.to_string(), reply));
            if let Some(r) = self.backend.io_hint() {
                if r > 1.5 {
                    eprintln!(
                        "{}",
                        style(format!(
                            "(hint: I/O wait/compute≈{r:.1} — try --hot-layers N or --ram-mib 6144)"
                        ))
                        .dim()
                    );
                }
            }
        }
        Ok(())
    }

    fn generate_burst(&mut self, user: &str, max_tokens: usize) -> Result<String> {
        let prompt = self.entry.format_prompt(user, &self.history);
        self.backend.reset_state();

        let mut stdout = io::stdout();
        print!("{} ", style("…").green());
        stdout.flush()?;

        let assembled = self.backend.generate(
            &self.tokenizer,
            &prompt,
            max_tokens,
            self.temperature,
            |token| {
                print!("{token}");
                let _ = stdout.flush();
                Ok(())
            },
        )?;

        println!();
        Ok(assembled.trim().to_string())
    }
}
