//! Phase 8 — NVMe-resident project structure graph + on-demand symbol fetch.

mod build;
mod embed;
mod extract;
mod fetch;
mod query;
mod synthesize;

#[allow(unused_imports)]
pub use build::{
    build_project_map, load_status, project_hash, rebuild_project_map, resolve_map_dir, MapStatus,
};
#[allow(unused_imports)]
pub use fetch::{fetch_nodes, ProjectMapReader};
#[allow(unused_imports)]
pub use query::{select_nodes, QueryOpts};
pub use synthesize::{synthesize_context, ProjectContextOpts};
