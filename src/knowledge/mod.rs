//! Phase 7.1 — Web search, local knowledge store, RAG-style prompt injection.

mod backend;
mod heuristic;
mod inject;
mod job;
mod store;

#[allow(unused_imports)]
pub use backend::{search_query, SearchBackendKind, SearchHit};
#[allow(unused_imports)]
pub use heuristic::{needs_knowledge, KnowledgeGap};
pub use inject::{inject_knowledge, KnowledgeInjectOpts};
pub use job::{spawn_search_job, SearchJobHandle};
#[allow(unused_imports)]
pub use store::{KnowledgeChunk, KnowledgeStore};
