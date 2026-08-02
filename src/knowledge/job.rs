//! Background search jobs (non-blocking fetch → parse → persist).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::error::Result;
use crate::knowledge::backend::search_query;
use crate::knowledge::store::KnowledgeStore;

/// Handle to an in-flight or finished background search.
pub struct SearchJobHandle {
    join: Option<JoinHandle<Result<usize>>>,
    status: Arc<Mutex<SearchJobStatus>>,
}

#[derive(Debug, Clone)]
pub struct SearchJobStatus {
    pub query: String,
    pub done: bool,
    pub added: Option<usize>,
    pub error: Option<String>,
}

impl SearchJobHandle {
    pub fn status(&self) -> SearchJobStatus {
        self.status
            .lock()
            .map(|g| g.clone())
            .unwrap_or(SearchJobStatus {
                query: String::new(),
                done: true,
                added: None,
                error: Some("status lock poisoned".into()),
            })
    }

    /// Block until the job finishes; returns chunks added.
    pub fn join(mut self) -> Result<usize> {
        let result = if let Some(h) = self.join.take() {
            match h.join() {
                Ok(r) => r,
                Err(_) => Err(crate::error::AppError::msg("search job thread panicked")),
            }
        } else {
            let st = self.status();
            if let Some(e) = st.error {
                Err(crate::error::AppError::msg(e))
            } else {
                Ok(st.added.unwrap_or(0))
            }
        };
        result
    }
}

impl Drop for SearchJobHandle {
    fn drop(&mut self) {
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

/// Spawn a background thread that searches and writes into `knowledge_dir`.
pub fn spawn_search_job(knowledge_dir: PathBuf, query: String, tags: Vec<String>) -> SearchJobHandle {
    let status = Arc::new(Mutex::new(SearchJobStatus {
        query: query.clone(),
        done: false,
        added: None,
        error: None,
    }));
    let status_thr = Arc::clone(&status);
    let join = thread::spawn(move || {
        let result = (|| {
            let store = KnowledgeStore::open_path(&knowledge_dir)?;
            let hits = search_query(&query)?;
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            store.ingest_hits(&query, &hits, &tag_refs)
        })();
        if let Ok(mut g) = status_thr.lock() {
            g.done = true;
            match &result {
                Ok(n) => g.added = Some(*n),
                Err(e) => g.error = Some(e.to_string()),
            }
        }
        result
    });
    SearchJobHandle {
        join: Some(join),
        status,
    }
}
