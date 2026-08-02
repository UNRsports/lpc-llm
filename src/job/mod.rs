//! Phase 6: scale-up bridge — declarative jobs, convert/import, RLHF stages.

mod config;
mod convert;
mod rlhf;
mod runner;

#[allow(unused_imports)]
pub use config::{JobConfig, JobStage, RemoteSpec};
pub use convert::{convert_checkpoint_to_gguf, import_gguf};
#[allow(unused_imports)]
pub use rlhf::{default_rlhf_pipeline, RlhfStageKind};
pub use runner::{job_status, run_job, write_template};
