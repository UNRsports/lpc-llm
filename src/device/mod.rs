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
                            "compute: Vulkan Q4_K path ready (prefill m≥8 fused submit; \
                             decode m=1 → {}; other dtypes / cold weights → CPU)",
                            if ctx.gpu_decode_worthwhile() {
                                "GPU"
                            } else {
                                "Candle CPU (fewer fences)"
                            }
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
        let mut outs = self.qmatmul_multi(&[w], x)?;
        outs.pop()
            .ok_or_else(|| crate::error::AppError::msg("qmatmul returned no output"))
    }

    /// Independent GEMVs against the same activation (fused GPU submit or parallel CPU).
    pub fn qmatmul_multi(&self, ws: &[&QMatMul], x: &Tensor) -> Result<Vec<Tensor>> {
        if ws.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "vulkan")]
        if let Some(ref vk) = self.vulkan {
            if vk.should_try_gpu(x) {
                match vk.qmatmul_multi(ws, x) {
                    Ok(t) => return Ok(t),
                    Err(e) => {
                        let msg = e.to_string();
                        if !msg.starts_with("vulkan-skip:") {
                            eprintln!("warning: Vulkan QMatMul fell back to CPU ({e})");
                        }
                    }
                }
            }
        }
        if ws.len() >= 2 {
            return parallel_qmatmul(ws, x);
        }
        let mut out = Vec::with_capacity(ws.len());
        for w in ws {
            out.push(w.forward(x)?);
        }
        Ok(out)
    }

    pub fn would_use_gpu(&self, x: &Tensor) -> bool {
        #[cfg(feature = "vulkan")]
        if let Some(ref vk) = self.vulkan {
            return vk.should_try_gpu(x);
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = x;
        false
    }

    #[cfg(feature = "vulkan")]
    pub fn vulkan_stats(&self) -> Option<(u64, u64)> {
        self.vulkan.as_ref().map(|vk| vk.stats())
    }

    #[cfg(not(feature = "vulkan"))]
    pub fn vulkan_stats(&self) -> Option<(u64, u64)> {
        let _ = self;
        None
    }

    /// Best-effort: pin a Q4_K weight in VRAM so small-batch GEMV can use the GPU.
    pub fn warm_q4k(&self, w: &QMatMul) {
        #[cfg(feature = "vulkan")]
        if let Some(ref vk) = self.vulkan {
            if let Err(e) = vk.warm_q4k(w) {
                let msg = e.to_string();
                if !msg.starts_with("vulkan-skip:") {
                    eprintln!("warning: Vulkan warm_q4k failed ({e})");
                }
            }
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = w;
    }
}

fn parallel_qmatmul(ws: &[&QMatMul], x: &Tensor) -> Result<Vec<Tensor>> {
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(ws.len());
        for w in ws {
            let w: &QMatMul = *w;
            joins.push(scope.spawn(move || -> Result<Tensor> {
                Ok(w.forward(x)?)
            }));
        }
        let mut out = Vec::with_capacity(joins.len());
        for join in joins {
            let piece = join
                .join()
                .map_err(|_| crate::error::AppError::msg("qmatmul worker panicked"))?;
            out.push(piece?);
        }
        Ok(out)
    })
}
