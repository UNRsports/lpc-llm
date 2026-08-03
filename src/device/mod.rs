//! Phase 9 — compute backend selection and Vulkan QMatMul offload.

mod resolve;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use resolve::{
    detect_cuda, detect_vulkan, resolve_backend, ComputeBackendKind, ResolvedBackend,
};

use std::sync::Arc;

use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Tensor};

use crate::config::ComputeDevicePref;
use crate::error::Result;

/// Shared compute context for inference engines.
#[derive(Clone)]
pub struct ComputeContext {
    pub backend: ResolvedBackend,
    pub candle_device: Device,
    #[cfg(feature = "vulkan")]
    vulkan: Option<Arc<vulkan::VulkanContext>>,
}

impl ComputeContext {
    pub fn from_pref(pref: ComputeDevicePref) -> Result<Self> {
        let kind = ComputeBackendKind::from_pref(pref);
        let backend = resolve_backend(kind)?;
        Self::from_resolved(backend)
    }

    pub fn from_resolved(backend: ResolvedBackend) -> Result<Self> {
        let candle_device = match backend {
            ResolvedBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| {
                        crate::error::AppError::msg(format!("CUDA device: {e}"))
                    })?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("warning: CUDA selected but build lacks `--features cuda`; using CPU");
                    Device::Cpu
                }
            }
            ResolvedBackend::Cpu | ResolvedBackend::Vulkan => Device::Cpu,
        };

        #[cfg(feature = "vulkan")]
        let vulkan = if matches!(backend, ResolvedBackend::Vulkan) {
            match vulkan::VulkanContext::new() {
                Ok(ctx) => {
                    if ctx.gpu_gemv_worthwhile() {
                        eprintln!(
                            "compute: Vulkan Q4_K path ready (VRAM-cached weights; \
                             other dtypes / cold weights → CPU)"
                        );
                    } else {
                        eprintln!(
                            "compute: Vulkan opened but Q4_K uses Candle CPU on this GPU \
                             (faster); attention/norms stay on CPU either way"
                        );
                    }
                    Some(Arc::new(ctx))
                }
                Err(e) => {
                    eprintln!("warning: Vulkan init failed ({e}); QMatMul stays on CPU");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            backend,
            candle_device,
            #[cfg(feature = "vulkan")]
            vulkan,
        })
    }

    pub fn label(&self) -> &'static str {
        match self.backend {
            ResolvedBackend::Cpu => "CPU",
            ResolvedBackend::Cuda => "CUDA",
            ResolvedBackend::Vulkan => "Vulkan",
        }
    }

    pub fn device(&self) -> &Device {
        &self.candle_device
    }

    /// Quantized matmul: Vulkan Q4_K GEMV when available; else Candle CPU.
    pub fn qmatmul(&self, w: &QMatMul, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "vulkan")]
        if let Some(ref vk) = self.vulkan {
            match vk.qmatmul(w, x) {
                Ok(t) => return Ok(t),
                Err(e) => {
                    let msg = e.to_string();
                    if !msg.starts_with("vulkan-skip:") {
                        eprintln!("warning: Vulkan QMatMul fell back to CPU ({e})");
                    }
                }
            }
        }
        Ok(w.forward(x)?)
    }
}
