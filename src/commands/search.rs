//! CLI: `lpc-llm search` / `lpc-llm knowledge list|purge`.

use console::style;

use crate::error::Result;
use crate::knowledge::{search_query, KnowledgeStore};
use crate::store::LocalStore;

pub fn search(query: &str) -> Result<()> {
    let store = LocalStore::open()?;
    let kstore = KnowledgeStore::open(&store)?;
    eprintln!("{} searching …", style("·").cyan());
    let hits = search_query(query)?;
    if hits.is_empty() {
        println!("{}", style("(no results)").dim());
        return Ok(());
    }
    for (i, h) in hits.iter().enumerate() {
        println!("{}. {}", i + 1, style(&h.title).bold());
        if !h.url.is_empty() {
            println!("   {}", style(&h.url).dim());
        }
        println!("   {}", h.snippet);
        println!();
    }
    let added = kstore.ingest_hits(query, &hits, &["search"])?;
    eprintln!(
        "{} stored {added} new chunk(s) under {}",
        style("✓").green(),
        kstore.dir().display()
    );
    Ok(())
}

pub fn knowledge_list() -> Result<()> {
    let store = LocalStore::open()?;
    let kstore = KnowledgeStore::open(&store)?;
    let chunks = kstore.list()?;
    if chunks.is_empty() {
        println!("{}", style("(no knowledge chunks)").dim());
        return Ok(());
    }
    println!(
        "{:<16} {:<24} {}",
        style("ID").bold(),
        style("QUERY").bold(),
        style("TITLE").bold()
    );
    for c in chunks {
        println!(
            "{:<16} {:<24} {}",
            c.id,
            truncate(&c.query, 24),
            truncate(&c.title, 48)
        );
    }
    Ok(())
}

pub fn knowledge_purge() -> Result<()> {
    let store = LocalStore::open()?;
    let kstore = KnowledgeStore::open(&store)?;
    let n = kstore.purge()?;
    println!("{} purged {n} knowledge chunk(s)", style("✓").green());
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}
