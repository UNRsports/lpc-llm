//! Interactive chat REPL (eager Candle load or hybrid io_uring prefetch).

use std::io::{self, Write};
use std::path::PathBuf;

use console::style;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokenizers::Tokenizer;

use crate::adapter::AdapterSet;
use crate::agent;
use crate::catalog::ModelEntry;
use crate::commands::run::{resolve_adapter, resolve_adapter_name};
use crate::device::ComputeContext;
use crate::engine::{Engine, GenerateOutcome};
use crate::error::{AppError, Result};
use crate::hybrid::{HybridConfig, HybridEngine};
use crate::knowledge::{
    inject_knowledge, needs_knowledge, spawn_search_job, KnowledgeInjectOpts, KnowledgeStore,
};
use crate::project_map::{synthesize_context, ProjectContextOpts, ProjectMapReader};
use crate::store::{InstalledModel, LocalStore};
use crate::user_adapt::append_turn;

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

    fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        on_token: impl FnMut(&str) -> Result<()>,
    ) -> Result<GenerateOutcome> {
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

    fn set_expert_hints(&mut self, hints: Vec<usize>) {
        if let Self::Hybrid(e) = self {
            e.set_expert_prefetch_hints(hints);
        }
    }
}

/// Optional Phase 7 / 8 session context (knowledge, project-map, logging).
pub struct SessionExtras {
    pub knowledge: Option<KnowledgeStore>,
    pub inject_knowledge: bool,
    pub project_map: Option<ProjectMapReader>,
    pub log_turns: bool,
    pub model_name: String,
}

impl Default for SessionExtras {
    fn default() -> Self {
        Self {
            knowledge: None,
            inject_knowledge: false,
            project_map: None,
            log_turns: true,
            model_name: String::new(),
        }
    }
}

pub struct ChatSession {
    backend: Backend,
    tokenizer: Tokenizer,
    entry: ModelEntry,
    history: Vec<(String, String)>,
    /// Hard cap per generate() call (first reply or `/more`).
    max_tokens: usize,
    temperature: f64,
    extras: SessionExtras,
    knowledge_dir: Option<PathBuf>,
    pending_search: Option<crate::knowledge::SearchJobHandle>,
}

impl ChatSession {
    pub fn load_with_config(
        installed: &InstalledModel,
        entry: ModelEntry,
        hybrid: bool,
        cfg: HybridConfig,
        pack_cache: &std::path::Path,
        adapter: Option<AdapterSet>,
        mut extras: SessionExtras,
        compute: ComputeContext,
        max_tokens: usize,
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
                compute,
            )?)
        } else {
            Backend::Eager(Engine::load(&installed.model_path, compute)?)
        };

        let tokenizer = Tokenizer::from_file(&installed.tokenizer_path)
            .map_err(|e| AppError::msg(format!("tokenizer load: {e}")))?;

        eprintln!(
            "{} ready on {} ({})",
            style("✓").green(),
            backend.device_name(),
            backend.architecture()
        );

        extras.model_name = entry.name.clone();
        let knowledge_dir = extras.knowledge.as_ref().map(|k| k.dir().to_path_buf());

        Ok(Self {
            backend,
            tokenizer,
            entry,
            history: Vec::new(),
            max_tokens: max_tokens.max(1),
            temperature: 0.7,
            extras,
            knowledge_dir,
            pending_search: None,
        })
    }

    /// Phase 3 agent REPL: first user turn → router (exclusive RAM) → drop → main.
    pub fn run_agent_repl(
        store: &LocalStore,
        installed: &InstalledModel,
        entry: ModelEntry,
        cfg: HybridConfig,
        pack_cache: &std::path::Path,
        _force_hybrid: bool,
        explicit_adapter: Option<String>,
        agent_model: String,
        no_user_profile: bool,
        extras: SessionExtras,
        compute: ComputeContext,
        max_tokens: usize,
    ) -> Result<()> {
        println!(
            "{} {} — agent mode (`{}` router, exclusive RAM) — `/bye` exit",
            style(">>>").cyan().bold(),
            style(&entry.display).bold(),
            agent_model
        );

        let mut rl = DefaultEditor::new()
            .map_err(|e| AppError::msg(format!("readline init: {e}")))?;

        // Wait for the first real user turn before loading anything heavy.
        let first_user = loop {
            let line = match rl.readline(">>> ") {
                Ok(l) => l,
                Err(ReadlineError::Interrupted) => {
                    println!("{}", style("(Ctrl-C — type /bye to exit)").dim());
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!();
                    return Ok(());
                }
                Err(e) => return Err(AppError::msg(format!("readline: {e}"))),
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let _ = rl.add_history_entry(line);
            match line {
                "/bye" | "/exit" | "/quit" => return Ok(()),
                "/clear" | "/more" => {
                    println!("{}", style("(load the first message first)").dim());
                    continue;
                }
                _ => break line.to_string(),
            }
        };

        let adapters_for_base: Vec<_> = store
            .list_adapters()?
            .into_iter()
            .filter(|a| a.base_model == entry.name)
            .collect();

        // Router occupies RAM alone; classify_intent drops it before return.
        let decision = agent::classify_intent(store, &first_user, &adapters_for_base, &agent_model)?;

        if explicit_adapter.is_none() {
            if let Some(a) = decision.adapter.as_deref() {
                eprintln!(
                    "{} agent selected adapter `{a}` (intent={})",
                    style("·").cyan(),
                    decision.intent
                );
            }
        }

        let adapter_name = resolve_adapter_name(
            store,
            &entry.name,
            explicit_adapter.as_deref(),
            decision.adapter.as_deref(),
            no_user_profile,
        )?;

        let (adapter_set, cfg) = resolve_adapter(store, &entry.name, adapter_name.as_deref(), cfg)?;
        // Agent always prefers hybrid so LoRA + MoE expert hints apply.
        let use_hybrid = true;

        let mut session = Self::load_with_config(
            installed,
            entry,
            use_hybrid,
            cfg,
            pack_cache,
            adapter_set,
            extras,
            compute,
            max_tokens,
        )?;
        session.backend.set_expert_hints(decision.expert_hints);

        // First turn already collected — generate immediately, then normal REPL.
        let (reply, _truncated) = session.generate_turn(&first_user)?;
        session.maybe_log_turn(&first_user, &reply);
        session.history.push((first_user, reply));
        session.run_repl_continue(rl)?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn set_expert_hints(&mut self, hints: Vec<usize>) {
        self.backend.set_expert_hints(hints);
    }

    pub fn run_repl(&mut self) -> Result<()> {
        println!(
            "{} {} — `/bye` exit, `/clear` reset, `/more` continue last (+{} tokens)",
            style(">>>").cyan().bold(),
            style(&self.entry.display).bold(),
            self.max_tokens
        );

        let rl = DefaultEditor::new().map_err(|e| AppError::msg(format!("readline init: {e}")))?;
        self.run_repl_continue(rl)
    }

    fn run_repl_continue(&mut self, mut rl: DefaultEditor) -> Result<()> {
        let mut last_user: Option<String> = self.history.last().map(|(u, _)| u.clone());

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
                    if last_user.is_some() && self.history.last().is_some() {
                        let (more, truncated) = self.generate_more()?;
                        if more.is_empty() && !truncated {
                            println!("{}", style("(already complete)").dim());
                        }
                    } else {
                        println!("{}", style("(nothing to continue)").dim());
                    }
                    continue;
                }
                _ => {}
            }

            self.maybe_spawn_knowledge_job(line);

            last_user = Some(line.to_string());
            let (reply, _truncated) = self.generate_turn(line)?;
            self.maybe_log_turn(line, &reply);
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

    fn maybe_spawn_knowledge_job(&mut self, user: &str) {
        let Some((gap, query)) = needs_knowledge(user) else {
            return;
        };
        let Some(dir) = self.knowledge_dir.clone() else {
            return;
        };
        // Avoid stacking jobs; join a finished handle so results / errors surface.
        if let Some(h) = self.pending_search.take() {
            let st = h.status();
            if !st.done {
                self.pending_search = Some(h);
                return;
            }
            match h.join() {
                Ok(n) => {
                    if n > 0 {
                        eprintln!(
                            "{} background search `{}` added {} chunk(s)",
                            style("·").cyan(),
                            st.query,
                            n
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "{} background search `{}` failed: {e}",
                        style("·").cyan(),
                        st.query
                    );
                }
            }
        }
        eprintln!(
            "{} knowledge gap ({gap:?}) — background search: {query}",
            style("·").cyan()
        );
        self.pending_search = Some(spawn_search_job(dir, query, vec!["auto".into()]));
    }

    fn maybe_log_turn(&self, user: &str, assistant: &str) {
        if !self.extras.log_turns {
            return;
        }
        if let Ok(store) = LocalStore::open() {
            let _ = append_turn(&store, &self.extras.model_name, user, assistant, None);
        }
    }

    fn enrich_user(&mut self, user: &str) -> Result<String> {
        let mut body = user.to_string();

        if self.extras.inject_knowledge {
            if let Some(ref kstore) = self.extras.knowledge {
                let (enriched, chunks) =
                    inject_knowledge(kstore, &body, &KnowledgeInjectOpts::default())?;
                if !chunks.is_empty() {
                    eprintln!(
                        "{} injected {} knowledge chunk(s)",
                        style("·").cyan(),
                        chunks.len()
                    );
                    body = enriched;
                }
            }
        }

        if let Some(ref mut pm) = self.extras.project_map {
            let ctx = synthesize_context(pm, user, &ProjectContextOpts::default())?;
            if !ctx.is_empty() {
                eprintln!("{} injected project-map overview", style("·").cyan());
                let mut enriched = String::new();
                enriched.push_str("[Project structure]\n");
                enriched.push_str(&ctx);
                enriched.push_str("\n\n[User]\n");
                enriched.push_str(&body);
                body = enriched;
            }
        }

        Ok(body)
    }

    /// Generate a new assistant reply for `user`. Returns (text, truncated_without_eos).
    fn generate_turn(&mut self, user: &str) -> Result<(String, bool)> {
        let enriched = self.enrich_user(user)?;
        let prompt = self.entry.format_prompt(&enriched, &self.history);
        let outcome = self.stream_generate(&prompt, self.max_tokens)?;
        let truncated = !outcome.hit_eos && outcome.tokens_generated >= self.max_tokens;
        if truncated {
            self.print_truncated_hint();
        }
        Ok((outcome.text.trim().to_string(), truncated))
    }

    /// Continue the last assistant turn (`/more`).
    fn generate_more(&mut self) -> Result<(String, bool)> {
        let Some((user, partial)) = self.history.last().cloned() else {
            return Ok((String::new(), false));
        };
        let prior: Vec<(String, String)> = self.history[..self.history.len().saturating_sub(1)]
            .to_vec();
        let enriched = self.enrich_user(&user)?;
        let prompt = self
            .entry
            .format_prompt_continue(&enriched, &prior, &partial);
        let outcome = self.stream_generate(&prompt, self.max_tokens)?;
        let piece = outcome.text;
        let truncated = !outcome.hit_eos && outcome.tokens_generated >= self.max_tokens;
        if let Some(last) = self.history.last_mut() {
            last.1.push_str(&piece);
        }
        if truncated {
            self.print_truncated_hint();
        }
        Ok((piece.trim().to_string(), truncated))
    }

    fn print_truncated_hint(&self) {
        println!(
            "{}",
            style(format!(
                "(truncated — type /more for up to +{} tokens)",
                self.max_tokens
            ))
            .dim()
        );
    }

    fn stream_generate(&mut self, prompt: &str, max_tokens: usize) -> Result<GenerateOutcome> {
        self.backend.reset_state();

        let mut stdout = io::stdout();
        print!("{} ", style("…").green());
        stdout.flush()?;

        let outcome = self.backend.generate(
            &self.tokenizer,
            prompt,
            max_tokens,
            self.temperature,
            |token| {
                print!("{token}");
                let _ = stdout.flush();
                Ok(())
            },
        )?;

        println!();
        Ok(outcome)
    }
}
