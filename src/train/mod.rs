//! Phase 5: constrained-resource model creation.
//!
//! Tiny Llama-family from-scratch training, GGUF export, local SFT, and DPO.

pub mod checkpoint;
mod data;
mod dpo;
mod gguf_export;
mod memory;
mod scratch;
mod sft;
mod tiny;
mod tokenizer_tiny;

#[allow(unused_imports)]
pub use checkpoint::{load_checkpoint, save_checkpoint, CheckpointMeta};
#[allow(unused_imports)]
pub use data::{load_preference_pairs, load_training_texts, PreferencePair};
pub use dpo::{train_dpo, DpoConfig};
#[allow(unused_imports)]
pub use gguf_export::{
    export_and_register, export_checkpoint_dir, export_gguf, register_gguf_model,
};
pub use scratch::{train_scratch, ScratchConfig};
pub use sft::{train_sft_full, SftConfig};
#[allow(unused_imports)]
pub use tiny::{TinyConfig, TinyModel};
