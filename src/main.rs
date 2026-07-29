//! lpc-llm — Ollama-like local LLM CUI (pure Rust) + hybrid NVMe prefetch.

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
                  Subcommands: list / pull / run / rm / show / prefetch / io.\n\
                  Model blobs live under ~/.local/share/lpc-llm/blobs (durable);\n\
                  engine packs under ~/.local/share/lpc-llm/cache (regenerable).\n\
                  Engine upgrades reuse downloaded weights (e.g. gemma2:2b).\n\
                  `run --hybrid` streams layers via io_uring double buffers."
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
        }) => commands::cmd_run(name, pull, hybrid, hot_layers, ram_mib, burst),
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
