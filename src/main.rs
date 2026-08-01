//! lpc-llm — Ollama-like local LLM CUI (pure Rust) + hybrid NVMe prefetch.

mod adapter;
mod agent;
mod catalog;
mod commands;
mod engine;
mod error;
mod hybrid;
mod infer;
mod io;
mod pull;
mod store;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use commands::io_demo::IoArgs;

#[derive(Debug, Parser)]
#[command(
    name = "lpc-llm",
    about = "Local LLM runner (Ollama-style CUI) + hybrid NVMe prefetch I/O",
    long_about = "Pull and run quantized GGUF models via Candle (pure Rust).\n\
                  Subcommands: list / pull / run / rm / show / adapter / prefetch / io.\n\
                  Model blobs live under ~/.local/share/lpc-llm/blobs (durable);\n\
                  adapters under ~/.local/share/lpc-llm/adapters;\n\
                  engine packs under ~/.local/share/lpc-llm/cache (regenerable).\n\
                  Engine upgrades reuse downloaded weights (e.g. gemma2:2b).\n\
                  `run --hybrid` streams layers via io_uring double buffers.\n\
                  `run --adapter <name>` binds a LoRA side-path (forces hybrid).\n\
                  `run --agent` classifies intent with SmolLM2 then runs the main model.\n\
                  `adapter create --from … --out … --base …` trains a LoRA delta (Phase 4)."
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
        /// Tokens to stream on the first reply burst (TTFT / 思考の小分け)
        #[arg(long, default_value_t = 24)]
        burst: usize,
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
    },

    /// Manage diff adapters (LoRA)
    Adapter {
        #[command(subcommand)]
        cmd: AdapterCmd,
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
        /// Training file (plain text lines, or `.jsonl` with `{"text":"..."}`)
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
            burst,
            adapter,
            agent,
            agent_model,
        }) => commands::cmd_run(
            name,
            pull,
            hybrid,
            hot_layers,
            ram_mib,
            burst,
            adapter,
            agent,
            agent_model,
        ),
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
