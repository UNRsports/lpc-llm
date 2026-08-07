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

/// Result of wiping pack cache or reporting deleted paths.
#[derive(Debug, Default, Clone)]
pub struct WipeReport {
    pub bytes_freed: u64,
    pub paths: Vec<PathBuf>,
}

/// Result of a full model purge (blobs + cache + optional adapters).
#[derive(Debug, Default, Clone)]
pub struct PurgeReport {
    pub name: String,
    pub registry_removed: bool,
    pub bytes_freed: u64,
    pub blob_paths: Vec<PathBuf>,
    pub cache_paths: Vec<PathBuf>,
    pub adapter_paths: Vec<PathBuf>,
    pub adapters_removed: Vec<String>,
    /// Blob paths kept because another model still references them.
    pub skipped_shared: Vec<PathBuf>,
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

    /// All engine pack versions for one catalog model: `cache/packs/<safe_name>/`.
    pub fn pack_cache_model_dir(&self, model_name: &str) -> PathBuf {
        let safe = model_name.replace([':', '/'], "_");
        self.cache_dir().join("packs").join(safe)
    }

    /// Pack cache for one catalog model, versioned by engine so layout can change
    /// without touching the GGUF.
    pub fn pack_cache_dir(&self, model_name: &str) -> PathBuf {
        self.pack_cache_model_dir(model_name)
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
    /// Prefer [`Self::purge_model`] when freeing disk is required.
    pub fn remove(&self, name: &str) -> Result<bool> {
        let mut m = self.load()?;
        let removed = m.models.remove(name).is_some();
        if removed {
            self.save(&m)?;
        }
        Ok(removed)
    }

    /// Wipe regenerable pack cache for one model (`cache/packs/<safe>/`, all engine versions).
    /// Does not touch blobs or the registry.
    pub fn wipe_pack_cache(&self, name: &str) -> Result<WipeReport> {
        let dir = self.pack_cache_model_dir(name);
        let mut report = WipeReport::default();
        if dir.exists() {
            let bytes = dir_size(&dir)?;
            fs::remove_dir_all(&dir)?;
            report.bytes_freed = bytes;
            report.paths.push(dir);
        }
        Ok(report)
    }

    /// Delete durable blobs + pack cache for `name`, then drop the soft registry entry.
    ///
    /// Shared blob / tokenizer paths used by other installed models are skipped.
    /// When `with_adapters` is true, also removes LoRA adapters whose `base_model` matches.
    pub fn purge_model(&self, name: &str, with_adapters: bool) -> Result<PurgeReport> {
        let installed = self.resolve_name(name)?;
        let mut report = PurgeReport {
            name: name.to_string(),
            ..PurgeReport::default()
        };

        let cache = self.wipe_pack_cache(name)?;
        report.bytes_freed += cache.bytes_freed;
        report.cache_paths = cache.paths;

        if let Some(ref m) = installed {
            let model_path = m.model_path.clone();
            let tokenizer_path = m.tokenizer_path.clone();

            if !self.path_used_by_other_models(name, &model_path)? {
                report.bytes_freed += remove_file_and_empty_parents(&model_path, &self.blobs_dir())?;
                report.blob_paths.push(model_path.clone());
                // Incomplete downloads: `foo.gguf` → `foo.part` (see pull.rs).
                let part = model_path.with_extension("part");
                if part.exists() {
                    report.bytes_freed +=
                        remove_file_and_empty_parents(&part, &self.blobs_dir())?;
                    report.blob_paths.push(part);
                }
                if let Some(parent) = model_path.parent() {
                    try_remove_empty_dir_chain(parent, &self.blobs_dir())?;
                }
            } else {
                report.skipped_shared.push(model_path);
            }

            if tokenizer_path != m.model_path {
                if !self.path_used_by_other_models(name, &tokenizer_path)? {
                    report.bytes_freed +=
                        remove_file_and_empty_parents(&tokenizer_path, &self.blobs_dir())?;
                    report.blob_paths.push(tokenizer_path.clone());
                    if let Some(parent) = tokenizer_path.parent() {
                        try_remove_empty_dir_chain(parent, &self.blobs_dir())?;
                    }
                } else {
                    report.skipped_shared.push(tokenizer_path);
                }
            }
        }

        if with_adapters {
            let adapters = self.adapters_for_base(name)?;
            for a in adapters {
                let dir = a.path.clone();
                if dir.exists() {
                    report.bytes_freed += dir_size(&dir)?;
                    fs::remove_dir_all(&dir)?;
                    report.adapter_paths.push(dir);
                }
                let _ = self.remove_adapter(&a.name)?;
                report.adapters_removed.push(a.name);
            }
        }

        report.registry_removed = self.remove(name)?;
        Ok(report)
    }

    /// Soft-registry adapters whose `base_model` equals `base` (exact catalog name).
    pub fn adapters_for_base(&self, base: &str) -> Result<Vec<InstalledAdapter>> {
        let m = self.load()?;
        Ok(m.adapters
            .into_values()
            .filter(|a| a.base_model == base)
            .collect())
    }

    /// True if another installed (or discoverable catalog) model uses `path`
    /// as its GGUF or tokenizer file.
    fn path_used_by_other_models(&self, exclude_name: &str, path: &Path) -> Result<bool> {
        let path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        for m in self.list_installed()? {
            if m.name == exclude_name {
                continue;
            }
            let mp = fs::canonicalize(&m.model_path).unwrap_or_else(|_| m.model_path.clone());
            let tp =
                fs::canonicalize(&m.tokenizer_path).unwrap_or_else(|_| m.tokenizer_path.clone());
            if mp == path || tp == path {
                return Ok(true);
            }
        }
        // Catalog entries not yet in the soft index but with durable blobs on disk.
        for entry in catalog::catalog() {
            if entry.name == exclude_name {
                continue;
            }
            if let Some(found) = self.discover(&entry)? {
                let mp =
                    fs::canonicalize(&found.model_path).unwrap_or_else(|_| found.model_path.clone());
                let tp = fs::canonicalize(&found.tokenizer_path)
                    .unwrap_or_else(|_| found.tokenizer_path.clone());
                if mp == path || tp == path {
                    return Ok(true);
                }
            }
        }
        Ok(false)
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

fn file_size(path: &Path) -> u64 {
    path.metadata().map(|m| m.len()).unwrap_or(0)
}

fn dir_size(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    if path.is_file() {
        return Ok(file_size(path));
    }
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            total = total.saturating_add(dir_size(&p)?);
        } else {
            total = total.saturating_add(file_size(&p));
        }
    }
    Ok(total)
}

/// Remove a file and return its size. Empty-parent cleanup is separate.
fn remove_file_and_empty_parents(path: &Path, stop_at: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let bytes = file_size(path);
    if path.is_dir() {
        let bytes = dir_size(path)?;
        fs::remove_dir_all(path)?;
        return Ok(bytes);
    }
    fs::remove_file(path)?;
    if let Some(parent) = path.parent() {
        try_remove_empty_dir_chain(parent, stop_at)?;
    }
    Ok(bytes)
}

/// Remove empty directories from `start` up to (but not including) `stop_at`.
fn try_remove_empty_dir_chain(start: &Path, stop_at: &Path) -> Result<()> {
    let stop = fs::canonicalize(stop_at).unwrap_or_else(|_| stop_at.to_path_buf());
    let mut cur = start.to_path_buf();
    loop {
        let cur_canon = fs::canonicalize(&cur).unwrap_or_else(|_| cur.clone());
        if cur_canon == stop || !cur_canon.starts_with(&stop) {
            break;
        }
        match fs::remove_dir(&cur) {
            Ok(()) => {
                if let Some(parent) = cur.parent() {
                    cur = parent.to_path_buf();
                } else {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    Ok(())
}

pub fn now_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn format_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let f = n as f64;
    if f >= GIB {
        format!("{:.2} GiB", f / GIB)
    } else if f >= MIB {
        format!("{:.1} MiB", f / MIB)
    } else if f >= KIB {
        format!("{:.0} KiB", f / KIB)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AppConfig, ComputeDevicePref, InstallMode, UiLanguage};

    fn temp_store() -> (PathBuf, LocalStore) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "lpc-llm-purge-test-{}-{}-{}",
            std::process::id(),
            now_unix(),
            seq
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("temp root");
        let cfg = AppConfig {
            data_dir: root.clone(),
            train_dir: root.join("train"),
            bin_dir: root.join("bin"),
            install_mode: InstallMode::User,
            ui_language: UiLanguage::En,
            compute_device: ComputeDevicePref::Auto,
            runtime_device_configured: false,
            loaded_from: Vec::new(),
        };
        let store = LocalStore::open_with(&cfg).expect("open store");
        (root, store)
    }

    fn write_dummy_model(store: &LocalStore, name: &str) -> InstalledModel {
        let entry = catalog::find(name).expect("catalog");
        let model_path = store.blob_path(&entry.hf_repo, &entry.gguf_file);
        let tokenizer_path = store.blob_path(&entry.tokenizer_repo, "tokenizer.json");
        if let Some(p) = model_path.parent() {
            fs::create_dir_all(p).unwrap();
        }
        if let Some(p) = tokenizer_path.parent() {
            fs::create_dir_all(p).unwrap();
        }
        fs::write(&model_path, b"gguf-bytes").unwrap();
        fs::write(&tokenizer_path, b"{}").unwrap();
        let pack = store.pack_cache_dir(name);
        fs::create_dir_all(&pack).unwrap();
        fs::write(pack.join("layers.pack"), b"pack").unwrap();
        let installed = InstalledModel {
            name: name.to_string(),
            model_path,
            tokenizer_repo: entry.tokenizer_repo.clone(),
            tokenizer_path,
            hf_repo: entry.hf_repo.clone(),
            gguf_file: entry.gguf_file.clone(),
            pulled_at_unix: now_unix(),
        };
        store.record(installed.clone()).unwrap();
        installed
    }

    #[test]
    fn soft_remove_keeps_blobs_and_cache() {
        let (root, store) = temp_store();
        let m = write_dummy_model(&store, "smollm2:360m");
        assert!(store.remove("smollm2:360m").unwrap());
        assert!(m.model_path.exists());
        assert!(store.pack_cache_dir("smollm2:360m").join("layers.pack").exists());
        // reconcile will re-register from leftover blobs
        let n = store.reconcile().unwrap();
        assert!(n >= 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn wipe_pack_cache_only() {
        let (root, store) = temp_store();
        let m = write_dummy_model(&store, "smollm2:360m");
        let report = store.wipe_pack_cache("smollm2:360m").unwrap();
        assert!(report.bytes_freed > 0);
        assert!(!store.pack_cache_model_dir("smollm2:360m").exists());
        assert!(m.model_path.exists());
        assert!(store.is_installed("smollm2:360m").unwrap());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn purge_removes_blobs_cache_and_registry() {
        let (root, store) = temp_store();
        let m = write_dummy_model(&store, "smollm2:360m");
        let report = store.purge_model("smollm2:360m", false).unwrap();
        assert!(report.bytes_freed > 0);
        assert!(!m.model_path.exists());
        assert!(!m.tokenizer_path.exists());
        assert!(!store.pack_cache_model_dir("smollm2:360m").exists());
        assert!(!store.is_installed("smollm2:360m").unwrap());
        assert_eq!(store.reconcile().unwrap(), 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn purge_with_adapters_removes_matching_lora() {
        let (root, store) = temp_store();
        let _ = write_dummy_model(&store, "smollm2:360m");
        let adapter_dir = store.adapter_path("demo-lora");
        fs::create_dir_all(&adapter_dir).unwrap();
        fs::write(
            adapter_dir.join("adapter.json"),
            r#"{"name":"demo-lora","base_model":"smollm2:360m","rank":8,"alpha":16.0,"layers":[]}"#,
        )
        .unwrap();
        fs::write(adapter_dir.join("weights.bin"), b"ab").unwrap();
        store
            .record_adapter(InstalledAdapter {
                name: "demo-lora".into(),
                path: adapter_dir.clone(),
                base_model: "smollm2:360m".into(),
                recorded_at_unix: now_unix(),
            })
            .unwrap();

        let report = store.purge_model("smollm2:360m", true).unwrap();
        assert!(report.adapters_removed.iter().any(|n| n == "demo-lora"));
        assert!(!adapter_dir.exists());
        assert!(store.get_adapter("demo-lora").unwrap().is_none());
        let _ = fs::remove_dir_all(root);
    }
}
