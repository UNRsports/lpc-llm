//! ash-based Vulkan compute for Q4_K / Q6_K / Q8_0 GEMV offload.

mod context;
mod q4k;
mod q6k;
mod q8_0;

pub use context::{probe, DeviceAct, VulkanContext};
