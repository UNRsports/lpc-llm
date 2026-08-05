//! ash-based Vulkan compute for Q4_K / Q6_K GEMV offload.

mod context;
mod q4k;
mod q6k;

pub use context::{probe, VulkanContext};
