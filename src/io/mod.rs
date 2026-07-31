//! Low-level I/O building blocks for hybrid LLM inference:
//! locked double buffers + `io_uring` / `O_DIRECT` NVMe prefetch.

pub mod error;
pub mod gguf_map;
pub mod hybrid_pipeline;
pub mod moe;
pub mod nvme;
pub mod pack;
pub mod pipeline;
pub mod prefetch;

#[allow(unused_imports)]
pub use error::{IoError, Result};
pub use gguf_map::GgufLayerMap;
pub use hybrid_pipeline::run_gguf_prefetch_pipeline;
#[allow(unused_imports)]
pub use moe::{ExpertDmaPlan, MoeFamily, MoeInfo, MoeLayout};
pub use nvme::AsyncNvmeReader;
#[allow(unused_imports)]
pub use pack::{ensure_experts_packed, ensure_packed, PackedExperts, PackedWeights};
#[allow(unused_imports)]
pub use pipeline::{run_pipeline, synthesize_layers, LayerExtent, StepStats};
#[allow(unused_imports)]
pub use prefetch::{
    align_up, PrefetchBuffer, PrefetchBufferManager, PrefetchRing, DIRECT_ALIGN,
};

