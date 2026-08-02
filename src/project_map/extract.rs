//! Lightweight language front-end: file walk + symbol / call-edge extraction.
//!
//! Pure Rust heuristics (no tree-sitter / C deps) covering Rust, Python, JS/TS,
//! Go, and C-family enough for overview graphs.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{AppError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    CFamily,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub file: PathBuf,
    pub line: usize,
    /// Signature / declaration line (trimmed).
    pub signature: String,
    /// Small body preview for embedding / display.
    pub preview: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Class,
    Trait,
    Type,
    Module,
    Other,
}

#[derive(Debug, Clone)]
pub struct CallEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone)]
pub struct FileExtract {
    pub path: PathBuf,
    pub symbols: Vec<Symbol>,
    pub calls: Vec<CallEdge>,
    pub mtime_unix: u64,
}

const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".cursor",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
];

const MAX_FILE_BYTES: u64 = 512 * 1024;

/// Walk `root` and extract symbols / call edges from supported source files.
pub fn extract_tree(root: &Path) -> Result<Vec<FileExtract>> {
    if !root.is_dir() {
        return Err(AppError::msg(format!(
            "project path is not a directory: {}",
            root.display()
        )));
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<FileExtract>) -> Result<()> {
    let rd = fs::read_dir(dir)?;
    for entry in rd {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') && name != ".env.example" {
            // Still skip hidden dirs; allow nothing special.
        }
        if entry.file_type()?.is_dir() {
            if SKIP_DIRS.iter().any(|s| *s == name) {
                continue;
            }
            walk(root, &path, out)?;
            continue;
        }
        let Some(lang) = detect_lang(&path) else {
            continue;
        };
        let meta = entry.metadata()?;
        if meta.len() == 0 || meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        let (symbols, calls) = extract_file(&rel, lang, &text);
        let mtime_unix = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        out.push(FileExtract {
            path: rel,
            symbols,
            calls,
            mtime_unix,
        });
    }
    Ok(())
}

fn detect_lang(path: &Path) -> Option<Lang> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "rs" => Lang::Rust,
        "py" => Lang::Python,
        "js" | "mjs" | "cjs" => Lang::JavaScript,
        "ts" | "tsx" => Lang::TypeScript,
        "go" => Lang::Go,
        "c" | "h" | "cc" | "cpp" | "hpp" | "cxx" => Lang::CFamily,
        _ => return None,
    })
}

fn extract_file(file: &Path, lang: Lang, text: &str) -> (Vec<Symbol>, Vec<CallEdge>) {
    let mut symbols = Vec::new();
    let mut calls = Vec::new();
    let mut current_fn: Option<String> = None;

    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }

        if let Some(sym) = match lang {
            Lang::Rust => parse_rust_symbol(file, line_no, trimmed),
            Lang::Python => parse_python_symbol(file, line_no, trimmed),
            Lang::JavaScript | Lang::TypeScript => parse_js_symbol(file, line_no, trimmed),
            Lang::Go => parse_go_symbol(file, line_no, trimmed),
            Lang::CFamily => parse_c_symbol(file, line_no, trimmed),
        } {
            if matches!(sym.kind, SymbolKind::Function | SymbolKind::Method) {
                current_fn = Some(sym.name.clone());
            }
            symbols.push(sym);
        }

        if let Some(ref from) = current_fn {
            for to in extract_call_names(trimmed) {
                if to == *from {
                    continue;
                }
                calls.push(CallEdge {
                    from: from.clone(),
                    to,
                });
            }
        }
    }

    // Cap per-file call edges to keep graphs tractable.
    if calls.len() > 256 {
        calls.truncate(256);
    }
    (symbols, calls)
}

fn parse_rust_symbol(file: &Path, line: usize, t: &str) -> Option<Symbol> {
    let t = t.trim_start_matches("pub ").trim_start_matches("async ");
    if let Some(rest) = t.strip_prefix("fn ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("struct ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Struct, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("enum ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Type, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("trait ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Trait, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("type ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Type, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("mod ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Module, file, line, t));
    }
    if t.starts_with("impl ") {
        let name = format!("impl {}", t.trim_start_matches("impl ").chars().take(40).collect::<String>());
        return Some(make_sym(name, SymbolKind::Other, file, line, t));
    }
    None
}

fn parse_python_symbol(file: &Path, line: usize, t: &str) -> Option<Symbol> {
    if let Some(rest) = t.strip_prefix("def ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("async def ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("class ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Class, file, line, t));
    }
    None
}

fn parse_js_symbol(file: &Path, line: usize, t: &str) -> Option<Symbol> {
    if let Some(rest) = t.strip_prefix("function ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("class ") {
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Class, file, line, t));
    }
    if let Some(idx) = t.find(" = function") {
        let left = t[..idx].trim();
        let name = left
            .rsplit([' ', '.'])
            .next()
            .and_then(|s| ident_prefix(s))?;
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    if t.contains("=>") {
        if let Some(idx) = t.find("const ") {
            let rest = t[idx + 6..].trim();
            if let Some(name) = ident_prefix(rest) {
                return Some(make_sym(name, SymbolKind::Function, file, line, t));
            }
        }
    }
    None
}

fn parse_go_symbol(file: &Path, line: usize, t: &str) -> Option<Symbol> {
    if let Some(rest) = t.strip_prefix("func ") {
        // func (r *T) Name( or func Name(
        let rest = rest.trim();
        if rest.starts_with('(') {
            if let Some(end) = rest.find(')') {
                let after = rest[end + 1..].trim();
                let name = ident_prefix(after)?;
                return Some(make_sym(name, SymbolKind::Method, file, line, t));
            }
        }
        let name = ident_prefix(rest)?;
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    if let Some(rest) = t.strip_prefix("type ") {
        let name = ident_prefix(rest)?;
        let kind = if rest.contains("struct") {
            SymbolKind::Struct
        } else if rest.contains("interface") {
            SymbolKind::Trait
        } else {
            SymbolKind::Type
        };
        return Some(make_sym(name, kind, file, line, t));
    }
    None
}

fn parse_c_symbol(file: &Path, line: usize, t: &str) -> Option<Symbol> {
    if t.starts_with("typedef ") || t.starts_with("struct ") || t.starts_with("class ") {
        let kind = if t.contains("class ") {
            SymbolKind::Class
        } else {
            SymbolKind::Struct
        };
        let name = t
            .split_whitespace()
            .nth(1)
            .and_then(|s| ident_prefix(s))?;
        return Some(make_sym(name, kind, file, line, t));
    }
    // Heuristic: return_type name(
    if t.contains('(') && !t.contains('=') && !t.ends_with(';') {
        let before = t.split('(').next()?;
        let name = before.split_whitespace().last().and_then(|s| {
            let s = s.trim_start_matches('*');
            ident_prefix(s)
        })?;
        if matches!(
            name.as_str(),
            "if" | "for" | "while" | "switch" | "return" | "sizeof"
        ) {
            return None;
        }
        return Some(make_sym(name, SymbolKind::Function, file, line, t));
    }
    None
}

fn make_sym(name: String, kind: SymbolKind, file: &Path, line: usize, sig: &str) -> Symbol {
    Symbol {
        name,
        kind,
        file: file.to_path_buf(),
        line,
        signature: sig.chars().take(200).collect(),
        preview: sig.chars().take(120).collect(),
    }
}

fn ident_prefix(s: &str) -> Option<String> {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == 0 {
            if c.is_ascii_alphabetic() || c == '_' {
                out.push(c);
            } else {
                return None;
            }
        } else if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            break;
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn extract_call_names(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_cont(bytes[i]) {
                i += 1;
            }
            let name = &line[start..i];
            // Skip keywords / common noise.
            let skip = matches!(
                name,
                "if" | "for"
                    | "while"
                    | "match"
                    | "return"
                    | "let"
                    | "mut"
                    | "self"
                    | "Self"
                    | "fn"
                    | "def"
                    | "class"
                    | "import"
                    | "from"
                    | "pub"
                    | "async"
                    | "await"
                    | "print"
                    | "println"
                    | "format"
                    | "Ok"
                    | "Err"
                    | "Some"
                    | "None"
            );
            // Look ahead for '('
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if !skip && j < bytes.len() && bytes[j] == b'(' {
                out.push(name.to_string());
            }
        } else {
            i += 1;
        }
        if out.len() >= 8 {
            break;
        }
    }
    out
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
