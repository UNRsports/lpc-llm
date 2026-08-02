//! Local data layout — **model module** vs **engine module** vs **adapters**.
//!
//! Root and `train/` come from [`crate::config::AppConfig`] (`config_lpcllm`).
//! Defaults:
//!
//! ```text
//! ~/.local/share/lpc-llm/          # paths.data_dir (XDG)
//!   blobs/          # MODEL MODULE (durable). GGUF + tokenizer by HF repo.
//!   adapters/       # Diff modules (LoRA). Durable; independent of blobs.
//!   train/          # Private corpora (paths.train_dir; may be relocated)
//!   cache/          # ENGINE MODULE (regenerable). packs/, derived I/O layout.
//!   manifest.json   # Soft index (rebuildable by scanning blobs + catalog).
//! ```
//!
//! Engine binary upgrades must not require re-pulling Gemma (or any) weights.
//! Private data must not live in the git repository.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::{self, ModelEntry};
use crate::config::AppConfig;
use crate::error::Result;

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

/// Soft registry entry for a LoRA / diff adapter under [`LocalStore::adapters_dir`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledAdapter {
    pub name: String,
    /// Absolute path to the adapter directory (`adapter.json` + `weights.bin`).
    pub path: PathBuf,
    pub base_model: String,
    pub recorded_at_unix: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Manifest {
    models: BTreeMap<String, InstalledModel>,
    #[serde(default)]
    adapters: BTreeMap<String, InstalledAdapter>,
}

pub struct LocalStore {
    root: PathBuf,
    train_dir: PathBuf,
}

impl LocalStore {
    pub fn open() -> Result<Self> {
        let cfg = AppConfig::load()?;
        Self::open_with(&cfg)
    }

    pub fn open_with(cfg: &AppConfig) -> Result<Self> {
        let root = cfg.data_dir.clone();
        let train_dir = cfg.train_dir.clone();
        fs::create_dir_all(root.join("blobs"))?;
        fs::create_dir_all(root.join("adapters"))?;
        fs::create_dir_all(root.join("cache").join("packs"))?;
        fs::create_dir_all(root.join("cache").join("knowledge"))?;
        fs::create_dir_all(root.join("cache").join("user_logs"))?;
        fs::create_dir_all(root.join("cache").join("projects"))?;
        fs::create_dir_all(&train_dir)?;
        // legacy empty dir from earlier layouts
        fs::create_dir_all(root.join("models"))?;
        let store = Self { root, train_dir };
        // Repair soft index from durable blobs (engine upgrade / wiped manifest).
        store.reconcile()?;
        store.reconcile_adapters()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Private training corpora (from `config_lpcllm` `paths.train_dir`).
    pub fn train_dir(&self) -> &Path {
        &self.train_dir
    }

    /// Durable model assets (GGUF, tokenizer). Do not wipe on engine upgrade.
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    /// Durable LoRA / diff adapters (`<name>/adapter.json` + `weights.bin`).
    pub fn adapters_dir(&self) -> PathBuf {
        self.root.join("adapters")
    }

    /// Canonical path for one adapter directory.
    pub fn adapter_path(&self, name: &str) -> PathBuf {
        let safe = name.replace([':', '/'], "_");
        self.adapters_dir().join(safe)
    }

    /// Regenerable engine artifacts (layer packs, etc.).
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    /// Web-fetched knowledge chunks (`cache/knowledge/`).
    pub fn knowledge_dir(&self) -> PathBuf {
        self.cache_dir().join("knowledge")
    }

    /// Conversation / edit logs for user-profile auto-train (`cache/user_logs/`).
    pub fn user_logs_dir(&self) -> PathBuf {
        self.cache_dir().join("user_logs")
    }

    /// Project structure graphs (`cache/projects/<hash>/`).
    pub fn projects_dir(&self) -> PathBuf {
        self.cache_dir().join("projects")
    }

    /// One project-map directory under `cache/projects/<hash>/`.
    pub fn project_map_dir(&self, hash: &str) -> PathBuf {
        let safe = hash.replace([':', '/', '\\'], "_");
        self.projects_dir().join(safe)
    }

    /// Canonical user-profile adapter directory (`adapters/user_profile/`).
    pub fn user_profile_adapter_path(&self) -> PathBuf {
        self.adapter_path("user_profile")
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

    pub fn list_adapters(&self) -> Result<Vec<InstalledAdapter>> {
        let m = self.load()?;
        Ok(m.adapters.into_values().collect())
    }

    pub fn get_adapter(&self, name: &str) -> Result<Option<InstalledAdapter>> {
        Ok(self.load()?.adapters.get(name).cloned())
    }

    pub fn record_adapter(&self, entry: InstalledAdapter) -> Result<()> {
        let mut m = self.load()?;
        m.adapters.insert(entry.name.clone(), entry);
        self.save(&m)
    }

    #[allow(dead_code)]
    pub fn remove_adapter(&self, name: &str) -> Result<bool> {
        let mut m = self.load()?;
        let removed = m.adapters.remove(name).is_some();
        if removed {
            self.save(&m)?;
        }
        Ok(removed)
    }

    /// Resolve an adapter by name: valid manifest → discover on disk → `None`.
    pub fn resolve_adapter(&self, name: &str) -> Result<Option<InstalledAdapter>> {
        if let Some(a) = self.get_adapter(name)? {
            if adapter_paths_ok(&a) {
                return Ok(Some(a));
            }
        }
        if let Some(a) = self.discover_adapter(name)? {
            self.record_adapter(a.clone())?;
            return Ok(Some(a));
        }
        Ok(None)
    }

    /// If `adapters/<name>/adapter.json` exists, build an [`InstalledAdapter`].
    pub fn discover_adapter(&self, name: &str) -> Result<Option<InstalledAdapter>> {
        let path = self.adapter_path(name);
        let meta_path = path.join("adapter.json");
        if !file_nonempty(&meta_path)? {
            return Ok(None);
        }
        let text = fs::read_to_string(&meta_path)?;
        let meta: AdapterManifestPeek = serde_json::from_str(&text)?;
        if meta.name != name && meta.name.replace([':', '/'], "_") != name.replace([':', '/'], "_")
        {
            // Allow directory name to be the registry key even if meta.name differs slightly.
        }
        let base_model = meta.base_model;
        Ok(Some(InstalledAdapter {
            name: name.to_string(),
            path,
            base_model,
            recorded_at_unix: now_unix(),
        }))
    }

    /// Scan `adapters/*/adapter.json` and repair the soft index.
    pub fn reconcile_adapters(&self) -> Result<usize> {
        let dir = self.adapters_dir();
        if !dir.exists() {
            return Ok(0);
        }
        let mut added = 0usize;
        let mut m = self.load()?;
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let stale = match m.adapters.get(&name) {
                Some(a) => !adapter_paths_ok(a),
                None => true,
            };
            if !stale {
                continue;
            }
            if let Some(found) = self.discover_adapter(&name)? {
                m.adapters.insert(name, found);
                added += 1;
            }
        }
        if added > 0 {
            self.save(&m)?;
        }
        Ok(added)
    }
}

/// Minimal peek of `adapter.json` for store indexing (full parse lives in `adapter`).
#[derive(Debug, Deserialize)]
struct AdapterManifestPeek {
    name: String,
    base_model: String,
}

fn paths_ok(m: &InstalledModel) -> bool {
    file_nonempty(&m.model_path).unwrap_or(false)
        && file_nonempty(&m.tokenizer_path).unwrap_or(false)
}

fn adapter_paths_ok(a: &InstalledAdapter) -> bool {
    file_nonempty(&a.path.join("adapter.json")).unwrap_or(false)
        && file_nonempty(&a.path.join("weights.bin")).unwrap_or(false)
}

fn file_nonempty(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    Ok(path.metadata()?.len() > 0)
}

pub fn now_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
