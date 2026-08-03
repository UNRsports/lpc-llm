//! ash-based Vulkan compute for Q4_K GEMV offload.

mod context;
mod q4k;

pub use context::{probe, VulkanContext};
