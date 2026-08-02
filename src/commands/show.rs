use console::style;

use crate::catalog;
use crate::error::{AppError, Result};
use crate::store::LocalStore;

pub fn run(name: &str) -> Result<()> {
    let store = LocalStore::open()?;
    let entry = if let Some(e) = catalog::find(name) {
        e
    } else if let Some(m) = store.resolve_name(name)? {
        catalog::entry_for_local(&m.name, &m.gguf_file, &m.tokenizer_repo)
    } else {
        return Err(AppError::UnknownModel(name.into()));
    };

    println!("{} {}", style("name:").bold(), entry.name);
    println!("{} {}", style("display:").bold(), entry.display);
    println!("{} {}", style("size:").bold(), entry.approx_size);
    println!("{} {}", style("ram hint:").bold(), entry.min_ram_hint);
    println!("{} {}", style("hf repo:").bold(), entry.hf_repo);
    println!("{} {}", style("gguf:").bold(), entry.gguf_file);
    println!("{} {}", style("tokenizer:").bold(), entry.tokenizer_repo);
    println!("{} {:?}", style("prompt:").bold(), entry.prompt_style);

    match store.resolve_name(name)? {
        Some(m) => {
            println!("{} {}", style("status:").bold(), style("local").green());
            println!("{} {}", style("model path:").bold(), m.model_path.display());
            println!(
                "{} {}",
                style("tokenizer path:").bold(),
                m.tokenizer_path.display()
            );
            println!(
                "{} {}",
                style("engine pack cache:").bold(),
                store.pack_cache_dir(&entry.name).display()
            );
            println!("{} {}", style("pulled_at:").bold(), m.pulled_at_unix);
            println!(
                "{}",
                style("note: blobs survive engine upgrades; packs regenerate in cache/").dim()
            );
        }
        None => {
            println!("{} {}", style("status:").bold(), style("not installed").yellow());
            if catalog::find(name).is_some() {
                println!(
                    "{}",
                    style(format!("hint: lpc-llm pull {}", entry.name)).dim()
                );
            }
        }
    }

    Ok(())
}
