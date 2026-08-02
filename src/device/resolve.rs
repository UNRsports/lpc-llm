//! Detect and resolve preferred compute backends.

use crate::config::ComputeDevicePref;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendKind {
    Auto,
    Cpu,
    Cuda,
    Vulkan,
}

impl ComputeBackendKind {
    pub fn from_pref(pref: ComputeDevicePref) -> Self {
        match pref {
            ComputeDevicePref::Auto => Self::Auto,
            ComputeDevicePref::Cpu => Self::Cpu,
            ComputeDevicePref::Cuda => Self::Cuda,
            ComputeDevicePref::Vulkan => Self::Vulkan,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBackend {
    Cpu,
    Cuda,
    Vulkan,
}

/// Probe whether a Vulkan physical device with compute queue exists.
pub fn detect_vulkan() -> bool {
    #[cfg(feature = "vulkan")]
    {
        crate::device::vulkan::probe().is_ok()
    }
    #[cfg(not(feature = "vulkan"))]
    {
        false
    }
}

/// True when this binary was built with the `cuda` feature (runtime device may still fail).
pub fn detect_cuda() -> bool {
    cfg!(feature = "cuda")
}

pub fn resolve_backend(kind: ComputeBackendKind) -> Result<ResolvedBackend> {
    match kind {
        ComputeBackendKind::Cpu => Ok(ResolvedBackend::Cpu),
        ComputeBackendKind::Cuda => {
            if detect_cuda() {
                Ok(ResolvedBackend::Cuda)
            } else {
                eprintln!(
                    "warning: CUDA requested but binary built without `--features cuda`; using CPU"
                );
                Ok(ResolvedBackend::Cpu)
            }
        }
        ComputeBackendKind::Vulkan => {
            if detect_vulkan() {
                Ok(ResolvedBackend::Vulkan)
            } else {
                eprintln!("warning: Vulkan requested but no suitable device; using CPU");
                Ok(ResolvedBackend::Cpu)
            }
        }
        ComputeBackendKind::Auto => {
            if detect_vulkan() {
                Ok(ResolvedBackend::Vulkan)
            } else if detect_cuda() {
                Ok(ResolvedBackend::Cuda)
            } else {
                Ok(ResolvedBackend::Cpu)
            }
        }
    }
}
