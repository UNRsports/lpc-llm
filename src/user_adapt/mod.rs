//! Phase 7.2 — user habit logging, idle detection, auto LoRA training.

mod auto_train;
mod features;
mod idle;
mod log;

pub use auto_train::{run_auto_train, AutoTrainOpts};
#[allow(unused_imports)]
pub use features::extract_style_features;
#[allow(unused_imports)]
pub use idle::{idle_seconds, wait_until_idle};
#[allow(unused_imports)]
pub use log::{append_turn, build_training_corpus, rotate_logs, UserLogEntry};
