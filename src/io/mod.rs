//! Low-level I/O building blocks for hybrid LLM inference:
//! locked double buffers + `io_uring` / `O_DIRECT` NVMe prefetch.

pub mod error;
pub mod gguf_map;
pub mod hybrid_pipeline;
pub mod nvme;
pub mod pack;
pub mod pipeline;
pub mod prefetch;

#[allow(unused_imports)]
pub use error::{IoError, Result};
pub use gguf_map::GgufLayerMap;
pub use hybrid_pipeline::run_gguf_prefetch_pipeline;
pub use nvme::AsyncNvmeReader;
pub use pack::ensure_packed;
#[allow(unused_imports)]
pub use pipeline::{run_pipeline, synthesize_layers, LayerExtent, StepStats};
#[allow(unused_imports)]
pub use prefetch::{align_up, PrefetchBuffer, PrefetchBufferManager, DIRECT_ALIGN};

