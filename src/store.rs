//! Local data layout — **model module** vs **engine module**.
//!
//! ```text
//! ~/.local/share/lpc-llm/
//!   blobs/          # MODEL MODULE (durable). GGUF + tokenizer by HF repo.
//!                   # Survives engine upgrades; never re-downloaded if present.
//!   cache/          # ENGINE MODULE (regenerable). packs/, derived I/O layout.
//!                   # Safe to delete; rebuilt from blobs on next hybrid run.
//!   manifest.json   # Soft index (rebuildable by scanning blobs + catalog).
//! ```
//!
//! Engine binary upgrades must not require re-pulling Gemma (or any) weights.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::{self, ModelEntry};
use crate::error::{AppError, Result};

/// Soft registry entry. Paths always point into [`LocalStore::blobs_dir`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModel {
    pub name: String,
    /// Absolute path to the GGUF under `blobs/`.
    pub model_path: PathBuf,
    pub tokenizer_repo: String,
    pub tokenizer_path: PathBuf,
    pub hf_repo: String,
    pub gguf_file: String,
    pub pulled_at_unix: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Manifest {
    models: BTreeMap<String, InstalledModel>,
}

pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn open() -> Result<Self> {
        let root = data_dir()?;
        fs::create_dir_all(root.join("blobs"))?;
        fs::create_dir_all(root.join("cache").join("packs"))?;
        // legacy empty dir from earlier layouts
        fs::create_dir_all(root.join("models"))?;
        let store = Self { root };
        // Repair soft index from durable blobs (engine upgrade / wiped manifest).
        store.reconcile()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Durable model assets (GGUF, tokenizer). Do not wipe on engine upgrade.
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    /// Regenerable engine artifacts (layer packs, etc.).
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// Pack cache for one catalog model, versioned by engine so layout can change
    /// without touching the GGUF.
    pub fn pack_cache_dir(&self, model_name: &str) -> PathBuf {
        let safe = model_name.replace([':', '/'], "_");
        self.cache_dir()
            .join("packs")
            .join(safe)
            .join(env!("CARGO_PKG_VERSION"))
    }

    /// Canonical blob path: `blobs/<repo--with--slashes>/<filename>`.
    pub fn blob_path(&self, repo_id: &str, filename: &str) -> PathBuf {
        let safe_repo = repo_id.replace('/', "--");
        self.blobs_dir().join(safe_repo).join(filename)
    }

    fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    fn load(&self) -> Result<Manifest> {
        let path = self.manifest_path();
        if !path.exists() {
            return Ok(Manifest::default());
        }
        let text = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&text)?)
    }

    fn save(&self, manifest: &Manifest) -> Result<()> {
        let path = self.manifest_path();
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(manifest)?)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn list_installed(&self) -> Result<Vec<InstalledModel>> {
        let m = self.load()?;
        Ok(m.models.into_values().collect())
    }

    pub fn get(&self, name: &str) -> Result<Option<InstalledModel>> {
        Ok(self.load()?.models.get(name).cloned())
    }

    pub fn is_installed(&self, name: &str) -> Result<bool> {
        Ok(self.resolve_name(name)?.is_some())
    }

    pub fn record(&self, entry: InstalledModel) -> Result<()> {
        let mut m = self.load()?;
        m.models.insert(entry.name.clone(), entry);
        self.save(&m)
    }

    /// Remove from the soft registry only. **Blobs are never deleted** (model module).
    pub fn remove(&self, name: &str) -> Result<bool> {
        let mut m = self.load()?;
        let removed = m.models.remove(name).is_some();
        if removed {
            self.save(&m)?;
        }
        Ok(removed)
    }

    /// Resolve a catalog entry: valid manifest → discover blobs → `None`.
    /// Never downloads.
    pub fn resolve(&self, entry: &ModelEntry) -> Result<Option<InstalledModel>> {
        if let Some(m) = self.get(&entry.name)? {
            if paths_ok(&m) {
                return Ok(Some(m));
            }
        }
        if let Some(m) = self.discover(entry)? {
            self.record(m.clone())?;
            return Ok(Some(m));
        }
        Ok(None)
    }

    /// Resolve by catalog name (or `None` if unknown / not on disk).
    pub fn resolve_name(&self, name: &str) -> Result<Option<InstalledModel>> {
        match catalog::find(name) {
            Some(entry) => self.resolve(&entry),
            None => {
                // Custom / removed-from-catalog but still in manifest.
                match self.get(name)? {
                    Some(m) if paths_ok(&m) => Ok(Some(m)),
                    _ => Ok(None),
                }
            }
        }
    }

    /// If GGUF + tokenizer blobs already exist for this catalog entry, build an
    /// [`InstalledModel`] without downloading.
    pub fn discover(&self, entry: &ModelEntry) -> Result<Option<InstalledModel>> {
        let model_path = self.blob_path(&entry.hf_repo, &entry.gguf_file);
        let tokenizer_path = self.blob_path(&entry.tokenizer_repo, "tokenizer.json");
        if file_nonempty(&model_path)? && file_nonempty(&tokenizer_path)? {
            return Ok(Some(InstalledModel {
                name: entry.name.clone(),
                model_path,
                tokenizer_repo: entry.tokenizer_repo.clone(),
                tokenizer_path,
                hf_repo: entry.hf_repo.clone(),
                gguf_file: entry.gguf_file.clone(),
                pulled_at_unix: now_unix(),
            }));
        }
        Ok(None)
    }

    /// Rebuild soft manifest entries from durable blobs + current catalog.
    pub fn reconcile(&self) -> Result<usize> {
        let mut added = 0usize;
        let mut m = self.load()?;
        for entry in catalog::catalog() {
            let stale = match m.models.get(&entry.name) {
                Some(inst) => !paths_ok(inst),
                None => true,
            };
            if !stale {
                continue;
            }
            if let Some(found) = self.discover(&entry)? {
                m.models.insert(entry.name.clone(), found);
                added += 1;
            }
        }
        if added > 0 {
            self.save(&m)?;
        }
        Ok(added)
    }
}

fn paths_ok(m: &InstalledModel) -> bool {
    file_nonempty(&m.model_path).unwrap_or(false)
        && file_nonempty(&m.tokenizer_path).unwrap_or(false)
}

fn file_nonempty(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    Ok(path.metadata()?.len() > 0)
}

fn data_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir().ok_or_else(|| {
        AppError::msg("could not resolve XDG data directory (~/.local/share)")
    })?;
    Ok(base.join("lpc-llm"))
}

pub fn now_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
