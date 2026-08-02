//! Large checkpoint → GGUF conversion bridge + GGUF import.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use console::style;

use crate::error::{AppError, Result};
use crate::store::LocalStore;
use crate::train::checkpoint;
use crate::train::{export_and_register, register_gguf_model};

/// Import a ready-made GGUF + tokenizer into the local store.
pub fn import_gguf(
    store: &LocalStore,
    gguf: impl AsRef<Path>,
    tokenizer: impl AsRef<Path>,
    name: &str,
) -> Result<()> {
    register_gguf_model(store, name, gguf, tokenizer)?;
    Ok(())
}

/// Convert training artifacts to GGUF.
///
/// - `builtin`: Phase 5 checkpoint dir → F16 GGUF + register
/// - `external`: run `$LPC_LLM_CONVERT_CMD` (or default hint) for multi-billion HF trees
pub fn convert_checkpoint_to_gguf(
    store: &LocalStore,
    from_dir: impl AsRef<Path>,
    name: &str,
    backend: &str,
) -> Result<PathBuf> {
    let from_dir = from_dir.as_ref();
    match backend {
        "builtin" => {
            // Tiny / Phase 5 checkpoint layout.
            if from_dir.join(checkpoint::CONFIG_FILE).is_file() {
                let installed = export_and_register(store, from_dir, name)?;
                return Ok(installed.model_path);
            }
            // Directory that already contains model.gguf + tokenizer.json
            let gguf = from_dir.join("model.gguf");
            let tok = from_dir.join("tokenizer.json");
            if gguf.is_file() && tok.is_file() {
                let installed = register_gguf_model(store, name, &gguf, &tok)?;
                return Ok(installed.model_path);
            }
            Err(AppError::msg(format!(
                "builtin convert: expected checkpoint ({}) or model.gguf+tokenizer.json under {}",
                checkpoint::CONFIG_FILE,
                from_dir.display()
            )))
        }
        "external" => {
            let cmd = std::env::var("LPC_LLM_CONVERT_CMD").unwrap_or_else(|_| {
                "echo 'Set LPC_LLM_CONVERT_CMD to your HF→GGUF converter \
                 (e.g. python convert_hf_to_gguf.py)' >&2; exit 1"
                    .into()
            });
            let out_dir = store
                .cache_dir()
                .join("convert")
                .join(name.replace([':', '/'], "_"));
            fs::create_dir_all(&out_dir)?;
            eprintln!(
                "{} external convert: backend=LPC_LLM_CONVERT_CMD from={}",
                style("▸").cyan(),
                from_dir.display()
            );
            let status = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .env("LPC_LLM_CONVERT_IN", from_dir.as_os_str())
                .env("LPC_LLM_CONVERT_OUT", out_dir.as_os_str())
                .env("LPC_LLM_CONVERT_NAME", name)
                .status()
                .map_err(|e| AppError::msg(format!("spawn convert cmd: {e}")))?;
            if !status.success() {
                return Err(AppError::msg(format!(
                    "external convert failed (exit {status}); \
                     set LPC_LLM_CONVERT_CMD to write GGUF into $LPC_LLM_CONVERT_OUT"
                )));
            }
            let gguf = find_gguf(&out_dir)?.ok_or_else(|| {
                AppError::msg(format!(
                    "no .gguf produced under {}",
                    out_dir.display()
                ))
            })?;
            let tok = out_dir.join("tokenizer.json");
            if !tok.is_file() {
                // Fall back to tokenizer beside the source tree.
                let alt = from_dir.join("tokenizer.json");
                if alt.is_file() {
                    fs::copy(&alt, &tok)?;
                } else {
                    return Err(AppError::msg(
                        "external convert: tokenizer.json missing in out/from dir",
                    ));
                }
            }
            let installed = register_gguf_model(store, name, &gguf, &tok)?;
            // Optional: trigger hybrid pack on first run (Phase 2 path).
            eprintln!(
                "{} imported converted GGUF as `{}` (hybrid pack builds on first --hybrid run)",
                style("·").cyan(),
                name
            );
            Ok(installed.model_path)
        }
        other => Err(AppError::msg(format!(
            "unknown convert backend `{other}` (use builtin|external)"
        ))),
    }
}

fn find_gguf(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    for ent in fs::read_dir(dir)? {
        let ent = ent?;
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(Some(p));
        }
    }
    Ok(None)
}
