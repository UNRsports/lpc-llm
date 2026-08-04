//! lpc-llm — Ollama-like local LLM CUI (pure Rust) + hybrid NVMe prefetch.

mod adapter;
mod agent;
mod catalog;
mod commands;
mod config;
mod device;
mod engine;
mod error;
mod hybrid;
mod i18n;
mod infer;
mod io;
mod job;
mod knowledge;
mod progress;
mod project_map;
mod pull;
mod store;
mod train;
mod user_adapt;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use commands::io_demo::IoArgs;

#[derive(Debug, Parser)]
#[command(
    name = "lpc-llm",
    about = "Local LLM runner (Ollama-style CUI) + hybrid NVMe prefetch I/O",
    long_about = "Pull and run quantized GGUF models via Candle (pure Rust).\n\
                  Subcommands: list / pull / run / rm / show / adapter / train / job / config / …\n\
                  Paths come from config_lpcllm (see `lpc-llm config show`).\n\
                  Defaults: data under ~/.local/share/lpc-llm; private corpora under …/train;\n\
                  binary under ~/.local/bin (user) or /usr/local/bin (system install).\n\
                  Shared install ships the binary only; each user's data stays in their home.\n\
                  `adapter create --from …` resolves files under train_dir when needed.\n\
                  `train scratch|sft|dpo` creates tiny models / preference opts (Phase 5).\n\
                  `job init|run|import|convert` bridges scale-up / RLHF stages (Phase 6).\n\
                  `search` / `knowledge` / `adapter auto-train` — Phase 7 knowledge & user profile.\n\
                  `project-map` / `run --project-map` — Phase 8 NVMe project overview.\n\
                  `setup` — Phase 9 i18n first-run (UI language + compute device)."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// List catalog models and local install status
    List,

    /// Download a model from the catalog (Hugging Face)
    Pull {
        /// Model name, e.g. smollm2:360m or gemma2:2b
        name: String,
    },

    /// Run a model (pulls first if needed); omit name to pick interactively
    Run {
        /// Model name (optional — interactive select if omitted)
        name: Option<String>,
        /// Pull without confirmation when missing
        #[arg(long)]
        pull: bool,
        /// Stream layers via io_uring double-buffer prefetch
        #[arg(long)]
        hybrid: bool,
        /// Force hot (RAM-resident) layer count; default = derive from --ram-mib
        #[arg(long)]
        hot_layers: Option<usize>,
        /// Soft RAM budget for hot layers + 2 prefetch slots (MiB)
        #[arg(long, default_value_t = 4096)]
        ram_mib: usize,
        /// Max new tokens per reply (and per `/more` continuation)
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// Deprecated (ignored). Use `--max-tokens` instead.
        #[arg(long)]
        burst: Option<usize>,
        /// Bind a LoRA / diff adapter by name (forces hybrid)
        #[arg(long)]
        adapter: Option<String>,
        /// Run a lightweight router agent first (SmolLM2) to pick adapter / expert hints.
        /// Router and main model time-share RAM under `--ram-mib` (never co-resident).
        #[arg(long)]
        agent: bool,
        /// Router model for `--agent` (default: smollm2:360m)
        #[arg(long, default_value = "smollm2:360m")]
        agent_model: String,
        /// Do not auto-attach `adapters/user_profile/`
        #[arg(long)]
        no_user_profile: bool,
        /// Inject project-map overview (path or cache hash)
        #[arg(long)]
        project_map: Option<String>,
        /// Inject retrieved chunks from `cache/knowledge/` into prompts
        #[arg(long)]
        knowledge: bool,
        /// Compute backend: auto|cpu|cuda|vulkan (overrides config_lpcllm `[runtime].device`)
        #[arg(long, value_parser = parse_device_pref)]
        device: Option<crate::config::ComputeDevicePref>,
    },

    /// First-run Q&A: UI language + compute device → home config_lpcllm
    Setup,

    /// Manage diff adapters (LoRA)
    Adapter {
        #[command(subcommand)]
        cmd: AdapterCmd,
    },

    /// Web search → persist chunks under `cache/knowledge/`
    Search {
        /// Query string
        query: String,
    },

    /// List / purge local knowledge store
    Knowledge {
        #[command(subcommand)]
        cmd: KnowledgeCmd,
    },

    /// Build / inspect NVMe-resident project structure maps
    #[command(name = "project-map")]
    ProjectMap {
        #[command(subcommand)]
        cmd: ProjectMapCmd,
    },

    /// Phase 5: tiny from-scratch / full SFT / DPO / GGUF export
    Train {
        #[command(subcommand)]
        cmd: TrainCmd,
    },

    /// Phase 6: declarative jobs, import/convert, RLHF stage bridge
    Job {
        #[command(subcommand)]
        cmd: JobCmd,
    },

    /// Show / init path settings (`config_lpcllm`)
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },

    /// Map GGUF layers and benchmark io_uring ping-pong prefetch
    Prefetch {
        name: String,
        #[arg(long)]
        pull: bool,
    },

    /// Remove a model from the local registry (downloaded blobs kept)
    Rm {
        name: String,
    },

    /// Show catalog + local metadata for a model
    Show {
        name: String,
    },

    /// Synthetic NVMe / io_uring double-buffer demo
    Io(IoArgs),
}

#[derive(Debug, Subcommand)]
enum AdapterCmd {
    /// List registered adapters
    List,
    /// Create a LoRA adapter from a local text / JSONL dataset (Phase 4)
    Create {
        /// Training file (cwd path, or bare name under train_dir from config_lpcllm)
        #[arg(long)]
        from: String,
        /// Output adapter name under `adapters/<name>/`
        #[arg(long)]
        out: String,
        /// Catalog base model, e.g. `smollm2:360m`
        #[arg(long)]
        base: String,
        #[arg(long, default_value_t = 8)]
        rank: usize,
        #[arg(long, default_value_t = 16.0)]
        alpha: f64,
        /// AdamW update steps (cycle over tokenized chunks)
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// Max tokens per training chunk
        #[arg(long, default_value_t = 128)]
        max_seq: usize,
        /// Soft RAM budget while loading the hybrid trainer (MiB)
        #[arg(long, default_value_t = 4096)]
        ram_mib: usize,
        /// Train LoRA only on the last N layers (`0` = all layers)
        #[arg(long, default_value_t = 0)]
        last_layers: usize,
        /// Pull the base model without confirmation when missing
        #[arg(long)]
        pull: bool,
    },
    /// Install a zero-filled demo adapter for integration tests
    InstallDemo {
        /// Adapter name (directory under adapters/)
        #[arg(long, default_value = "demo-zero")]
        name: String,
        /// Catalog base model this adapter is shaped for
        #[arg(long, default_value = "gemma2:2b")]
        base: String,
        /// Transformer layer count (gemma2:2b = 26)
        #[arg(long, default_value_t = 26)]
        layers: usize,
        /// Embedding / hidden size (gemma2:2b = 2304)
        #[arg(long, default_value_t = 2304)]
        emb_dim: usize,
        #[arg(long, default_value_t = 8)]
        rank: usize,
    },
    /// Idle-time LoRA update into `adapters/user_profile/` (Phase 7.2)
    AutoTrain {
        /// Catalog base model the profile LoRA is trained against
        #[arg(long)]
        base: String,
        /// Run a single training cycle (default)
        #[arg(long, default_value_t = true)]
        once: bool,
        /// Loop: wait for idle, train, cool-down, repeat
        #[arg(long, default_value_t = false)]
        daemon: bool,
        #[arg(long, default_value_t = 8)]
        min_samples: usize,
        #[arg(long, default_value_t = 4096)]
        ram_mib: usize,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        rank: usize,
        #[arg(long, default_value_t = 8.0)]
        alpha: f64,
        #[arg(long, default_value_t = 128)]
        max_seq: usize,
        #[arg(long, default_value_t = 4)]
        last_layers: usize,
        /// Seconds of system idle required before training (`0` = skip wait)
        #[arg(long, default_value_t = 120)]
        idle_secs: u64,
        #[arg(long, default_value_t = 600)]
        max_train_secs: u64,
        #[arg(long)]
        pull: bool,
    },
}

#[derive(Debug, Subcommand)]
enum KnowledgeCmd {
    /// List stored knowledge chunks
    List,
    /// Delete all knowledge chunks
    Purge,
}

#[derive(Debug, Subcommand)]
enum ProjectMapCmd {
    /// Index a project tree into `cache/projects/<hash>/`
    Build {
        /// Project root directory
        path: String,
    },
    /// Show status for a path or hash
    Status {
        /// Project path or map hash
        path_or_hash: String,
    },
    /// Wipe and rebuild a project map
    Rebuild {
        path: String,
    },
}

#[derive(Debug, Subcommand)]
enum TrainCmd {
    /// Train a tiny Llama-family model from scratch → checkpoint (+ register GGUF)
    Scratch {
        #[arg(long)]
        from: String,
        /// Model name for registration, e.g. `tiny:demo`
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        #[arg(long, default_value_t = 3e-3)]
        lr: f64,
        #[arg(long, default_value_t = 64)]
        max_seq: usize,
        #[arg(long, default_value_t = 1024)]
        ram_mib: usize,
        /// Recompute activations layer-by-layer (lower peak RAM)
        #[arg(long, default_value_t = true)]
        grad_checkpoint: bool,
        #[arg(long, default_value_t = 128)]
        n_embd: usize,
        #[arg(long, default_value_t = 2)]
        n_layers: usize,
        #[arg(long, default_value_t = 4)]
        n_heads: usize,
        #[arg(long, default_value_t = 512)]
        n_ff: usize,
        /// Skip blobs/manifest registration
        #[arg(long)]
        no_register: bool,
    },
    /// Full fine-tune SFT continuing a tiny checkpoint
    Sft {
        /// Checkpoint dir or registered train name under cache/train/
        #[arg(long)]
        ckpt: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 64)]
        max_seq: usize,
        #[arg(long, default_value_t = 1024)]
        ram_mib: usize,
        #[arg(long, default_value_t = true)]
        grad_checkpoint: bool,
        #[arg(long)]
        no_register: bool,
    },
    /// Lightweight preference optimization (DPO)
    Dpo {
        #[arg(long)]
        ckpt: String,
        /// JSONL with prompt/chosen/rejected
        #[arg(long)]
        from: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 5e-4)]
        lr: f64,
        #[arg(long, default_value_t = 64)]
        max_seq: usize,
        #[arg(long, default_value_t = 1024)]
        ram_mib: usize,
        #[arg(long, default_value_t = 0.1)]
        beta: f64,
        #[arg(long, default_value_t = true)]
        grad_checkpoint: bool,
        #[arg(long)]
        no_register: bool,
    },
    /// Export a tiny checkpoint to GGUF (optionally register)
    Export {
        #[arg(long)]
        ckpt: String,
        #[arg(long)]
        out: Option<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        register: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigCmd {
    /// Print resolved data_dir / train_dir / bin_dir
    Show,
    /// Write ~/.config/lpc-llm/config_lpcllm with documented defaults
    Init {
        /// Overwrite an existing user config
        #[arg(long)]
        force: bool,
        /// Interactive i18n Q&A (same as `lpc-llm setup`)
        #[arg(long)]
        interactive: bool,
    },
    /// Print one resolved value (for install scripts): data_dir|train_dir|bin_dir|install.mode
    Get {
        key: String,
    },
    /// Print the default config_lpcllm text to stdout
    Example,
}

#[derive(Debug, Subcommand)]
enum JobCmd {
    /// Write a declarative job JSON template
    Init {
        /// Template: scratch | sft | rlhf | remote
        #[arg(long, default_value = "scratch")]
        template: String,
        #[arg(long, default_value = "job.json")]
        out: String,
    },
    /// Run stages from a job config (local and/or remote.launch)
    Run {
        #[arg(long)]
        config: String,
        /// Skip remote.launch even if present
        #[arg(long)]
        local: bool,
    },
    /// Show job status.json
    Status {
        /// Job name or path to status.json
        name: String,
    },
    /// Import an existing GGUF + tokenizer into blobs/manifest
    Import {
        #[arg(long)]
        gguf: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long)]
        name: String,
    },
    /// Convert checkpoint / external HF tree → GGUF + register
    Convert {
        #[arg(long)]
        from_dir: String,
        #[arg(long)]
        name: String,
        /// builtin (Phase 5 ckpt) or external ($LPC_LLM_CONVERT_CMD)
        #[arg(long, default_value = "builtin")]
        backend: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        None => commands::cmd_menu(),
        Some(Commands::List) => commands::cmd_list(),
        Some(Commands::Pull { name }) => commands::cmd_pull(&name),
        Some(Commands::Run {
            name,
            pull,
            hybrid,
            hot_layers,
            ram_mib,
            max_tokens,
            burst,
            adapter,
            agent,
            agent_model,
            no_user_profile,
            project_map,
            knowledge,
            device,
        }) => {
            if burst.is_some() {
                eprintln!(
                    "warning: --burst is deprecated and ignored; use --max-tokens (default {max_tokens})"
                );
            }
            commands::cmd_run(commands::run::RunOpts {
                name,
                auto_pull: pull,
                hybrid,
                hot_layers,
                ram_mib,
                max_tokens,
                adapter,
                agent,
                agent_model,
                no_user_profile,
                project_map,
                knowledge,
                device,
            })
        }
        Some(Commands::Setup) => commands::cmd_setup(),
        Some(Commands::Search { query }) => commands::cmd_search(&query),
        Some(Commands::Knowledge { cmd }) => match cmd {
            KnowledgeCmd::List => commands::cmd_knowledge_list(),
            KnowledgeCmd::Purge => commands::cmd_knowledge_purge(),
        },
        Some(Commands::ProjectMap { cmd }) => match cmd {
            ProjectMapCmd::Build { path } => commands::cmd_project_map_build(path),
            ProjectMapCmd::Status { path_or_hash } => {
                commands::cmd_project_map_status(path_or_hash)
            }
            ProjectMapCmd::Rebuild { path } => commands::cmd_project_map_rebuild(path),
        },
        Some(Commands::Adapter { cmd }) => match cmd {
            AdapterCmd::List => commands::cmd_adapter_list(),
            AdapterCmd::Create {
                from,
                out,
                base,
                rank,
                alpha,
                steps,
                lr,
                max_seq,
                ram_mib,
                last_layers,
                pull,
            } => commands::cmd_adapter_create(commands::adapter::CreateOpts {
                from,
                out,
                base,
                rank,
                alpha,
                steps,
                lr,
                max_seq,
                ram_mib,
                last_layers,
                pull,
            }),
            AdapterCmd::InstallDemo {
                name,
                base,
                layers,
                emb_dim,
                rank,
            } => commands::cmd_adapter_install_demo(name, base, layers, emb_dim, rank),
            AdapterCmd::AutoTrain {
                base,
                once,
                daemon,
                min_samples,
                ram_mib,
                steps,
                rank,
                alpha,
                max_seq,
                last_layers,
                idle_secs,
                max_train_secs,
                pull,
            } => commands::cmd_adapter_auto_train(crate::user_adapt::AutoTrainOpts {
                base,
                once: once || !daemon,
                daemon,
                min_samples,
                ram_mib,
                steps,
                rank,
                alpha,
                max_seq,
                last_layers,
                idle_secs,
                max_train_secs,
                pull,
            }),
        },
        Some(Commands::Train { cmd }) => match cmd {
            TrainCmd::Scratch {
                from,
                out,
                steps,
                lr,
                max_seq,
                ram_mib,
                grad_checkpoint,
                n_embd,
                n_layers,
                n_heads,
                n_ff,
                no_register,
            } => commands::cmd_train_scratch(commands::train::ScratchOpts {
                from,
                out,
                steps,
                lr,
                max_seq,
                ram_mib,
                grad_checkpoint,
                n_embd,
                n_layers,
                n_heads,
                n_ff,
                no_register,
            }),
            TrainCmd::Sft {
                ckpt,
                from,
                out,
                steps,
                lr,
                max_seq,
                ram_mib,
                grad_checkpoint,
                no_register,
            } => commands::cmd_train_sft(commands::train::SftOpts {
                ckpt,
                from,
                out,
                steps,
                lr,
                max_seq,
                ram_mib,
                grad_checkpoint,
                no_register,
            }),
            TrainCmd::Dpo {
                ckpt,
                from,
                out,
                steps,
                lr,
                max_seq,
                ram_mib,
                beta,
                grad_checkpoint,
                no_register,
            } => commands::cmd_train_dpo(commands::train::DpoOpts {
                ckpt,
                from,
                out,
                steps,
                lr,
                max_seq,
                ram_mib,
                beta,
                grad_checkpoint,
                no_register,
            }),
            TrainCmd::Export {
                ckpt,
                out,
                name,
                register,
            } => commands::cmd_train_export(commands::train::ExportOpts {
                ckpt,
                out_gguf: out,
                name,
                register,
            }),
        },
        Some(Commands::Job { cmd }) => match cmd {
            JobCmd::Init { template, out } => commands::cmd_job_init(template, out),
            JobCmd::Run { config, local } => commands::cmd_job_run(config, local),
            JobCmd::Status { name } => commands::cmd_job_status(name),
            JobCmd::Import {
                gguf,
                tokenizer,
                name,
            } => commands::cmd_job_import(gguf, tokenizer, name),
            JobCmd::Convert {
                from_dir,
                name,
                backend,
            } => commands::cmd_job_convert(from_dir, name, backend),
        },
        Some(Commands::Config { cmd }) => match cmd {
            ConfigCmd::Show => commands::cmd_config_show(),
            ConfigCmd::Init { force, interactive } => {
                commands::cmd_config_init(force, interactive)
            }
            ConfigCmd::Get { key } => commands::cmd_config_get(&key),
            ConfigCmd::Example => commands::cmd_config_example(),
        },
        Some(Commands::Prefetch { name, pull }) => commands::cmd_prefetch(&name, pull),
        Some(Commands::Rm { name }) => commands::cmd_rm(&name),
        Some(Commands::Show { name }) => commands::cmd_show(&name),
        Some(Commands::Io(args)) => commands::io_demo::run(args),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_device_pref(s: &str) -> std::result::Result<crate::config::ComputeDevicePref, String> {
    crate::config::ComputeDevicePref::parse(s)
        .ok_or_else(|| format!("invalid device `{s}` (want auto|cpu|cuda|vulkan)"))
}
