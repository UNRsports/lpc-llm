//! Diff adapters (LoRA) — storage format + side-path runtime binding.
//!
//! Base GGUF / `layers.pack` stay untouched. At inference time each target
//! Linear computes `y = Wq(x) + scale * (x @ Aᵀ) @ Bᵀ`.

mod format;
mod lora;
mod train;

pub use format::{write_demo_adapter, AdapterSet};
pub use lora::{LayerLora, LoraDelta};
pub use train::{load_training_texts, train_adapter, TrainConfig};
