//! ash-based Vulkan compute: Q4_K / Q6_K dequant+GEMV with VRAM-resident weights.

use std::collections::{HashMap, HashSet, VecDeque};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use ash::{Device, Entry, Instance};
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device as CandleDevice, Tensor};

use super::q4k::{self, pack_constant_q4k, BLOCK_Q4K_SIZE, QK_K};
use super::q6k::BLOCK_Q6K_SIZE;
use crate::error::{AppError, Result};

const SPIRV_Q4K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemv.spv"));
const SPIRV_Q6K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q6k_gemv.spv"));

/// Cap cached weight buffers. Hot (pinned) + recent experts.
const MAX_WEIGHT_CACHE: usize = 4096;

/// Max independent GEMVs recorded into one submit (Top-8 gates, Q/K/V, …).
const FUSED_MAX_OPS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuantKind {
    Q4K,
    Q6K,
}

impl QuantKind {
    fn from_dtype(dt: GgmlDType) -> Option<Self> {
        match dt {
            GgmlDType::Q4K => Some(Self::Q4K),
            GgmlDType::Q6K => Some(Self::Q6K),
            _ => None,
        }
    }

    fn block_size(self) -> usize {
        match self {
            Self::Q4K => BLOCK_Q4K_SIZE,
            Self::Q6K => BLOCK_Q6K_SIZE,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
        }
    }
}

struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

/// One compute pipeline (shared descriptor / pipeline layout with the other kind).
struct QuantPipeline {
    shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

/// Host-visible scratch kept mapped across calls.
struct MappedScratch {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u64,
    ptr: *mut u8,
}

// SAFETY: only touched under `submit_lock`.
unsafe impl Send for MappedScratch {}

struct SubmitResources {
    x: MappedScratch,
    y: MappedScratch,
    u: MappedScratch,
    sets: Vec<vk::DescriptorSet>,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

/// Vulkan compute context (thread-safe via mutex around queue submits).
pub struct VulkanContext {
    _entry: Entry,
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family: u32,
    physical: vk::PhysicalDevice,
    pipeline_q4k: QuantPipeline,
    pipeline_q6k: QuantPipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    submit_lock: Mutex<()>,
    /// Quantized weight blobs keyed by `Arc<QTensor>` pointer.
    weight_cache: Mutex<HashMap<usize, GpuBuffer>>,
    /// LRU order (front = oldest).
    weight_lru: Mutex<VecDeque<usize>>,
    /// Hot-layer keys that LRU must never evict.
    weight_pinned: Mutex<HashSet<usize>>,
    scratch: Mutex<Option<SubmitResources>>,
    /// When false, always fall back to Candle CPU (microbench lost on this GPU).
    gpu_gemv_worthwhile: bool,
    skip_count: AtomicU64,
    submit_count: AtomicU64,
    skip_seen: Mutex<HashSet<String>>,
}

impl VulkanContext {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| AppError::msg(format!("load libvulkan: {e}")))?;
        let app_name = c"lpc-llm";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| AppError::msg(format!("vkCreateInstance: {e}")))?
        };

        let (physical, queue_family) = pick_device(&instance)?;
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        let device_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
        let device = unsafe {
            instance
                .create_device(physical, &device_info, None)
                .map_err(|e| AppError::msg(format!("vkCreateDevice: {e}")))?
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let descriptor_set_layout = create_descriptor_set_layout(&device)?;
        let pipeline_layout = create_pipeline_layout(&device, descriptor_set_layout)?;
        let shader_q4k = create_shader(&device, SPIRV_Q4K)?;
        let shader_q6k = create_shader(&device, SPIRV_Q6K)?;
        let pipeline_q4k = QuantPipeline {
            shader: shader_q4k,
            pipeline: create_compute_pipeline(&device, pipeline_layout, shader_q4k)?,
        };
        let pipeline_q6k = QuantPipeline {
            shader: shader_q6k,
            pipeline: create_compute_pipeline(&device, pipeline_layout, shader_q6k)?,
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (FUSED_MAX_OPS as u32) * 8,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: (FUSED_MAX_OPS as u32) * 2,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets((FUSED_MAX_OPS as u32) + 4)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| AppError::msg(format!("vkCreateDescriptorPool: {e}")))?
        };

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        let command_pool = unsafe {
            device
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| AppError::msg(format!("vkCreateCommandPool: {e}")))?
        };

        let mut ctx = Self {
            _entry: entry,
            instance,
            device,
            queue,
            queue_family,
            physical,
            pipeline_q4k,
            pipeline_q6k,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            command_pool,
            submit_lock: Mutex::new(()),
            weight_cache: Mutex::new(HashMap::new()),
            weight_lru: Mutex::new(VecDeque::new()),
            weight_pinned: Mutex::new(HashSet::new()),
            scratch: Mutex::new(None),
            gpu_gemv_worthwhile: true,
            skip_count: AtomicU64::new(0),
            submit_count: AtomicU64::new(0),
            skip_seen: Mutex::new(HashSet::new()),
        };
        ctx.gpu_gemv_worthwhile = ctx.microbench_gpu_vs_cpu().unwrap_or(false);
        if ctx.gpu_gemv_worthwhile {
            eprintln!(
                "compute: GPU Q4_K/Q6_K for VRAM-warmed weights (hot layers + MoE experts); \
                 expect higher GPU% when experts stay cached"
            );
        }
        Ok(ctx)
    }

    /// True when the naive GPU GEMV path beat Candle-style CPU on a small probe.
    pub fn gpu_gemv_worthwhile(&self) -> bool {
        self.gpu_gemv_worthwhile
    }

    /// Kept for API symmetry with startup banners / future decode policy.
    #[allow(dead_code)]
    pub fn gpu_decode_worthwhile(&self) -> bool {
        self.gpu_gemv_worthwhile
    }

    pub fn stats(&self) -> (u64, u64) {
        (
            self.submit_count.load(Ordering::Relaxed),
            self.skip_count.load(Ordering::Relaxed),
        )
    }

    fn log_skip(&self, reason: &str) {
        self.skip_count.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut seen) = self.skip_seen.lock() {
            if seen.insert(reason.to_string()) {
                eprintln!(
                    "compute: vulkan-skip ({reason}) — further skips of this reason are silent"
                );
            }
        }
    }

    /// Whether this activation batch should attempt the GPU Q4_K path.
    pub fn should_try_gpu(&self, x: &Tensor) -> bool {
        if !self.gpu_gemv_worthwhile {
            return false;
        }
        let _ = x;
        true
    }

    /// True if this Q4_K / Q6_K weight is already resident in the VRAM cache.
    pub fn weight_cached(&self, w: &QMatMul) -> bool {
        let QMatMul::QTensor(qt) = w else {
            return false;
        };
        if QuantKind::from_dtype(qt.dtype()).is_none() {
            return false;
        }
        let key = Arc::as_ptr(qt) as usize;
        self.weight_cache
            .lock()
            .map(|c| c.contains_key(&key))
            .unwrap_or(false)
    }

    /// Upload a Q4_K / Q6_K weight into the VRAM cache without running GEMV.
    /// `pin` marks hot-layer keys that LRU must never evict (experts stay unpinned).
    pub fn warm_quant(&self, w: &QMatMul, pin: bool) -> Result<()> {
        if !self.gpu_gemv_worthwhile {
            return Ok(());
        }
        let QMatMul::QTensor(qt) = w else {
            return Ok(());
        };
        if QuantKind::from_dtype(qt.dtype()).is_none() {
            return Ok(());
        }
        let key = Arc::as_ptr(qt) as usize;
        let w_bytes = qt
            .data()
            .map_err(|e| AppError::msg(format!("QTensor data: {e}")))?;
        self.ensure_weight(key, w_bytes.as_ref())?;
        if pin {
            let mut pinned = self
                .weight_pinned
                .lock()
                .map_err(|_| AppError::msg("weight pin lock poisoned"))?;
            pinned.insert(key);
        }
        Ok(())
    }

    /// Warm + pin (hot attn / shared / router). Prefer `warm_quant` for experts.
    #[allow(dead_code)]
    pub fn warm_q4k(&self, w: &QMatMul) -> Result<()> {
        self.warm_quant(w, true)
    }

    fn microbench_gpu_vs_cpu(&self) -> Result<bool> {
        let n = 256usize;
        let k = 2048usize;
        let w = pack_constant_q4k(n, k, 2, 0.25, 0.0);
        let x = vec![1.0f32; k];
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            let _ = q4k::q4k_gemv_cpu(&w, n, k, &x)?;
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;

        // Warm GPU path once (pipeline / scratch).
        let key = 0xBEEF_u64 as usize;
        let _ = self.q4k_gemm_gpu(key, &w, n as u32, k as u32, 1, &x)?;
        let t1 = std::time::Instant::now();
        for _ in 0..3 {
            let _ = self.q4k_gemm_gpu(key, &w, n as u32, k as u32, 1, &x)?;
        }
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / 3.0;

        let win = gpu_ms < cpu_ms * 0.85;
        eprintln!(
            "compute: Q4_K GEMV probe CPU≈{cpu_ms:.2}ms GPU≈{gpu_ms:.2}ms → {}",
            if win {
                "use GPU when weights are VRAM-cached"
            } else {
                "prefer Candle CPU for Q4_K (GPU path slower on this device)"
            }
        );
        Ok(win)
    }

    /// Multiple independent Q4_K / Q6_K GEMVs against the same activation (one submit).
    pub fn qmatmul_multi(&self, ws: &[&QMatMul], x: &Tensor) -> Result<Vec<Tensor>> {
        if ws.is_empty() {
            return Ok(Vec::new());
        }
        if !self.should_try_gpu(x) {
            self.log_skip("GPU GEMV disabled on this device");
            return Err(AppError::msg(
                "vulkan-skip: GPU GEMV disabled on this device",
            ));
        }

        let x_f32 = x.to_dtype(DType::F32)?;
        let x_dims = x_f32.dims().to_vec();
        if x_dims.is_empty() {
            return Err(AppError::msg("Vulkan quant GEMV: empty input"));
        }
        let last = *x_dims.last().unwrap();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product::<usize>().max(1);
        // Forward path never uploads: only warm_* fills VRAM.

        let mut guards = Vec::with_capacity(ws.len());
        let mut metas: Vec<(usize, u32, u32, QuantKind)> = Vec::with_capacity(ws.len());
        for w in ws {
            let QMatMul::QTensor(qt) = w else {
                self.log_skip("weight already dequantized (Tensor/F16)");
                return Err(AppError::msg(
                    "vulkan-skip: weight already dequantized (Tensor/F16)",
                ));
            };
            let Some(kind) = QuantKind::from_dtype(qt.dtype()) else {
                self.log_skip(&format!("dtype {:?} (Q4_K/Q6_K only)", qt.dtype()));
                return Err(AppError::msg(format!(
                    "vulkan-skip: dtype {:?} (Q4_K/Q6_K only)",
                    qt.dtype()
                )));
            };
            let (n, k) = qt
                .shape()
                .dims2()
                .map_err(|e| AppError::msg(e.to_string()))?;
            if !k.is_multiple_of(QK_K) {
                self.log_skip(&format!("k={k} not multiple of {QK_K}"));
                return Err(AppError::msg(format!(
                    "vulkan-skip: k={k} not multiple of {QK_K}"
                )));
            }
            if last != k {
                return Err(AppError::msg(format!(
                    "Vulkan {} shape mismatch: x last={last} k={k}",
                    kind.name()
                )));
            }
            let key = Arc::as_ptr(qt) as usize;
            let cached = self
                .weight_cache
                .lock()
                .map_err(|_| AppError::msg("weight cache lock poisoned"))?
                .contains_key(&key);
            if !cached {
                self.log_skip("weight not VRAM-cached (CPU; cold)");
                return Err(AppError::msg(
                    "vulkan-skip: weight not VRAM-cached (CPU; cold)",
                ));
            }
            let w_bytes = qt
                .data()
                .map_err(|e| AppError::msg(format!("QTensor data: {e}")))?;
            let expect = n * (k / QK_K) * kind.block_size();
            if w_bytes.as_ref().len() != expect {
                return Err(AppError::msg(format!(
                    "{} byte len {} != expected {expect}",
                    kind.name(),
                    w_bytes.as_ref().len()
                )));
            }
            guards.push(w_bytes);
            metas.push((key, n as u32, k as u32, kind));
        }

        let a_flat = x_f32.reshape((m, last))?.flatten_all()?.to_vec1::<f32>()?;
        let prepared: Vec<(usize, &[u8], u32, u32, QuantKind)> = metas
            .iter()
            .zip(guards.iter())
            .map(|(&(key, n, k, kind), g)| (key, g.as_ref(), n, k, kind))
            .collect();
        let chunks = self.quant_gemm_gpu_multi(&prepared, m as u32, &a_flat)?;
        let mut outs = Vec::with_capacity(chunks.len());
        for (i, c_flat) in chunks.into_iter().enumerate() {
            let n = metas[i].1 as usize;
            let mut out_dims = x_dims.clone();
            *out_dims.last_mut().unwrap() = n;
            outs.push(Tensor::from_vec(c_flat, out_dims, &CandleDevice::Cpu)?);
        }
        Ok(outs)
    }

    fn touch_lru(lru: &mut VecDeque<usize>, key: usize) {
        if let Some(pos) = lru.iter().position(|&k| k == key) {
            lru.remove(pos);
        }
        lru.push_back(key);
    }

    fn ensure_weights_batch(&self, ops: &[(usize, &[u8], u32, u32, QuantKind)]) -> Result<()> {
        let protect: HashSet<usize> = ops.iter().map(|&(k, _, _, _, _)| k).collect();
        for &(key, w_bytes, _, _, _) in ops {
            self.ensure_weight_protected(key, w_bytes, &protect)?;
        }
        Ok(())
    }

    fn ensure_weight(&self, key: usize, bytes: &[u8]) -> Result<()> {
        self.ensure_weight_protected(key, bytes, &HashSet::new())
    }

    fn ensure_weight_protected(
        &self,
        key: usize,
        bytes: &[u8],
        protect: &HashSet<usize>,
    ) -> Result<()> {
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| AppError::msg("weight cache lock poisoned"))?;
            if cache.contains_key(&key) {
                let mut lru = self
                    .weight_lru
                    .lock()
                    .map_err(|_| AppError::msg("weight lru lock poisoned"))?;
                Self::touch_lru(&mut lru, key);
                return Ok(());
            }
            // Evict LRU entries that are not pinned and not in the current fused batch.
            while cache.len() >= MAX_WEIGHT_CACHE {
                let mut lru = self
                    .weight_lru
                    .lock()
                    .map_err(|_| AppError::msg("weight lru lock poisoned"))?;
                let pinned = self
                    .weight_pinned
                    .lock()
                    .map_err(|_| AppError::msg("weight pin lock poisoned"))?;
                let mut victim = None;
                for &cand in lru.iter() {
                    if !protect.contains(&cand) && !pinned.contains(&cand) && cand != key {
                        victim = Some(cand);
                        break;
                    }
                }
                drop(pinned);
                let Some(old_key) = victim else {
                    // Entire cache is protected — skip GPU rather than thrash.
                    return Err(AppError::msg(
                        "vulkan-skip: VRAM weight cache full (protected batch)",
                    ));
                };
                if let Some(pos) = lru.iter().position(|&k| k == old_key) {
                    lru.remove(pos);
                }
                if let Some(old) = cache.remove(&old_key) {
                    drop(lru);
                    self.destroy_buffer(old);
                } else {
                    break;
                }
            }
        }
        let buf = self.upload_bytes(bytes)?;
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| AppError::msg("weight cache lock poisoned"))?;
        let mut lru = self
            .weight_lru
            .lock()
            .map_err(|_| AppError::msg("weight lru lock poisoned"))?;
        if cache.contains_key(&key) {
            Self::touch_lru(&mut lru, key);
            self.destroy_buffer(buf);
            return Ok(());
        }
        cache.insert(key, buf);
        Self::touch_lru(&mut lru, key);
        Ok(())
    }

    fn ensure_scratch(&self) -> Result<()> {
        let mut slot = self
            .scratch
            .lock()
            .map_err(|_| AppError::msg("scratch lock poisoned"))?;
        if slot.is_some() {
            return Ok(());
        }
        let x = self.alloc_mapped(
            64 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let y = self.alloc_mapped(
            64 * 1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let u = self.alloc_mapped(
            (size_of::<[u32; 4]>() * FUSED_MAX_OPS) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        let set_layouts = vec![self.descriptor_set_layout; FUSED_MAX_OPS];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        let sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc)
                .map_err(|e| AppError::msg(format!("allocate_descriptor_sets: {e}")))?
        };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmds = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| AppError::msg(format!("allocate_command_buffers: {e}")))?
        };
        let fence = unsafe {
            self.device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .map_err(|e| AppError::msg(format!("create_fence: {e}")))?
        };
        *slot = Some(SubmitResources {
            x,
            y,
            u,
            sets,
            cmd: cmds[0],
            fence,
        });
        Ok(())
    }

    fn grow_mapped(&self, scratch: &mut MappedScratch, need: u64, usage: vk::BufferUsageFlags) -> Result<()> {
        if scratch.capacity >= need {
            return Ok(());
        }
        let new_cap = need.next_power_of_two().max(4096);
        unsafe {
            self.device.unmap_memory(scratch.memory);
            self.device.destroy_buffer(scratch.buffer, None);
            self.device.free_memory(scratch.memory, None);
        }
        let grown = self.alloc_mapped(new_cap, usage)?;
        *scratch = grown;
        Ok(())
    }

    pub(crate) fn q4k_gemm_gpu(
        &self,
        key: usize,
        w_bytes: &[u8],
        n: u32,
        k: u32,
        m: u32,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let mut outs = self.quant_gemm_gpu_multi(
            &[(key, w_bytes, n, k, QuantKind::Q4K)],
            m,
            x,
        )?;
        outs.pop().ok_or_else(|| AppError::msg("gpu gemm empty"))
    }

    #[allow(dead_code)] // used by gpu_tests
    pub(crate) fn q6k_gemm_gpu(
        &self,
        key: usize,
        w_bytes: &[u8],
        n: u32,
        k: u32,
        m: u32,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let mut outs = self.quant_gemm_gpu_multi(
            &[(key, w_bytes, n, k, QuantKind::Q6K)],
            m,
            x,
        )?;
        outs.pop().ok_or_else(|| AppError::msg("gpu gemm empty"))
    }

    fn quant_gemm_gpu_multi(
        &self,
        ops: &[(usize, &[u8], u32, u32, QuantKind)],
        m: u32,
        x: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        if ops.len() > FUSED_MAX_OPS {
            let mut all = Vec::new();
            for chunk in ops.chunks(FUSED_MAX_OPS) {
                all.extend(self.quant_gemm_gpu_multi(chunk, m, x)?);
            }
            return Ok(all);
        }

        let _guard = self
            .submit_lock
            .lock()
            .map_err(|_| AppError::msg("vulkan submit lock poisoned"))?;

        self.ensure_weights_batch(ops)?;
        self.ensure_scratch()?;

        let w_bufs: Vec<(vk::Buffer, u64)> = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| AppError::msg("weight cache lock poisoned"))?;
            let mut out = Vec::with_capacity(ops.len());
            for &(key, _, _, _, _) in ops {
                let w_buf = cache
                    .get(&key)
                    .ok_or_else(|| {
                        AppError::msg("vulkan-skip: weight cache miss after ensure")
                    })?;
                out.push((w_buf.buffer, w_buf.size));
            }
            out
        };

        let x_bytes = (x.len() * size_of::<f32>()) as u64;
        let y_elems: Vec<usize> = ops
            .iter()
            .map(|&(_, _, n, _, _)| (m as usize) * (n as usize))
            .collect();
        let y_total: usize = y_elems.iter().sum();
        let y_bytes = (y_total * size_of::<f32>()) as u64;
        let u_bytes = (size_of::<[u32; 4]>() * ops.len()) as u64;

        let mut scratch_guard = self
            .scratch
            .lock()
            .map_err(|_| AppError::msg("scratch lock poisoned"))?;
        let scratch = scratch_guard
            .as_mut()
            .ok_or_else(|| AppError::msg("scratch missing after ensure"))?;

        self.grow_mapped(
            &mut scratch.x,
            x_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        self.grow_mapped(
            &mut scratch.y,
            y_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        self.grow_mapped(
            &mut scratch.u,
            u_bytes,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), scratch.x.ptr as *mut f32, x.len());
        }

        let mut y_off_elems = 0usize;
        let mut writes_storage: Vec<(
            [vk::DescriptorBufferInfo; 1],
            [vk::DescriptorBufferInfo; 1],
            [vk::DescriptorBufferInfo; 1],
            [vk::DescriptorBufferInfo; 1],
        )> = Vec::with_capacity(ops.len());
        for (i, &(_, _, n, k, _)) in ops.iter().enumerate() {
            let dims = [n, k, m, 0u32];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dims.as_ptr(),
                    scratch.u.ptr.add(i * size_of::<[u32; 4]>()) as *mut u32,
                    4,
                );
            }
            let y_op_bytes = (y_elems[i] * size_of::<f32>()) as u64;
            writes_storage.push((
                [vk::DescriptorBufferInfo {
                    buffer: scratch.x.buffer,
                    offset: 0,
                    range: x_bytes.max(4),
                }],
                [vk::DescriptorBufferInfo {
                    buffer: w_bufs[i].0,
                    offset: 0,
                    range: w_bufs[i].1,
                }],
                [vk::DescriptorBufferInfo {
                    buffer: scratch.y.buffer,
                    offset: (y_off_elems * size_of::<f32>()) as u64,
                    range: y_op_bytes.max(4),
                }],
                [vk::DescriptorBufferInfo {
                    buffer: scratch.u.buffer,
                    offset: (i * size_of::<[u32; 4]>()) as u64,
                    range: size_of::<[u32; 4]>() as u64,
                }],
            ));
            y_off_elems += y_elems[i];
        }

        let mut writes = Vec::with_capacity(ops.len() * 4);
        for (i, (x_info, w_info, y_info, u_info)) in writes_storage.iter().enumerate() {
            let set = scratch.sets[i];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(x_info),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(w_info),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(y_info),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(u_info),
            );
        }
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        unsafe {
            self.device
                .wait_for_fences(&[scratch.fence], true, 60_000_000_000)
                .map_err(|e| AppError::msg(format!("wait_for_fences(reset): {e}")))?;
            self.device
                .reset_fences(&[scratch.fence])
                .map_err(|e| AppError::msg(format!("reset_fences: {e}")))?;
            self.device
                .reset_command_buffer(scratch.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| AppError::msg(format!("reset_command_buffer: {e}")))?;
        }

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let cmd = scratch.cmd;
        let fence = scratch.fence;
        let sets: Vec<vk::DescriptorSet> = scratch.sets[..ops.len()].to_vec();
        let ns: Vec<u32> = ops.iter().map(|op| op.2).collect();
        let kinds: Vec<QuantKind> = ops.iter().map(|op| op.4).collect();
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| AppError::msg(format!("begin_command_buffer: {e}")))?;
            for i in 0..ops.len() {
                let pipeline = match kinds[i] {
                    QuantKind::Q4K => self.pipeline_q4k.pipeline,
                    QuantKind::Q6K => self.pipeline_q6k.pipeline,
                };
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[sets[i]],
                    &[],
                );
                let groups = (m * ns[i]).div_ceil(64);
                self.device.cmd_dispatch(cmd, groups, 1, 1);
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ,
                    );
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| AppError::msg(format!("end_command_buffer: {e}")))?;
        }

        let cmds = [cmd];
        let submits = [vk::SubmitInfo::default().command_buffers(&cmds)];
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, fence)
                .map_err(|e| AppError::msg(format!("queue_submit: {e}")))?;
            self.device
                .wait_for_fences(&[fence], true, 60_000_000_000)
                .map_err(|e| AppError::msg(format!("wait_for_fences: {e}")))?;
        }
        self.submit_count.fetch_add(1, Ordering::Relaxed);

        let mut outs = Vec::with_capacity(ops.len());
        let mut off = 0usize;
        for &len in &y_elems {
            let mut out = vec![0f32; len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (scratch.y.ptr as *const f32).add(off),
                    out.as_mut_ptr(),
                    len,
                );
            }
            outs.push(out);
            off += len;
        }
        Ok(outs)
    }

    // --- end quant_gemm_gpu_multi ---

    fn alloc_mapped(&self, capacity: u64, usage: vk::BufferUsageFlags) -> Result<MappedScratch> {
        let capacity = capacity.max(4);
        let info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            self.device
                .create_buffer(&info, None)
                .map_err(|e| AppError::msg(format!("create_buffer: {e}")))?
        };
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            &self.instance,
            self.physical,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type);
        let memory = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|e| AppError::msg(format!("allocate_memory: {e}")))?
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| AppError::msg(format!("bind_buffer_memory: {e}")))?;
        }
        let ptr = unsafe {
            self.device
                .map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty())
                .map_err(|e| AppError::msg(format!("map_memory: {e}")))? as *mut u8
        };
        Ok(MappedScratch {
            buffer,
            memory,
            capacity,
            ptr,
        })
    }

    fn upload_bytes(&self, data: &[u8]) -> Result<GpuBuffer> {
        // Q6_K blocks are 210 bytes (not 4-aligned); pad so u32 storage loads are in-bounds.
        let padded = ((data.len() + 3) / 4 * 4).max(4);
        let buf = self.alloc_buffer(
            padded as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .device
                .map_memory(buf.memory, 0, padded as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| AppError::msg(format!("map_memory: {e}")))?;
            std::ptr::write_bytes(ptr as *mut u8, 0, padded);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(buf.memory);
        }
        Ok(buf)
    }

    fn alloc_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<GpuBuffer> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            self.device
                .create_buffer(&info, None)
                .map_err(|e| AppError::msg(format!("create_buffer: {e}")))?
        };
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            &self.instance,
            self.physical,
            reqs.memory_type_bits,
            props,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type);
        let memory = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|e| AppError::msg(format!("allocate_memory: {e}")))?
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| AppError::msg(format!("bind_buffer_memory: {e}")))?;
        }
        Ok(GpuBuffer {
            buffer,
            memory,
            size: size.max(4),
        })
    }

    fn destroy_buffer(&self, buf: GpuBuffer) {
        unsafe {
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }

    fn destroy_mapped(&self, buf: MappedScratch) {
        unsafe {
            self.device.unmap_memory(buf.memory);
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(mut scratch) = self.scratch.lock() {
                if let Some(s) = scratch.take() {
                    self.device.destroy_fence(s.fence, None);
                    // Descriptor set / command buffer freed with pools.
                    self.destroy_mapped(s.x);
                    self.destroy_mapped(s.y);
                    self.destroy_mapped(s.u);
                }
            }
            if let Ok(mut cache) = self.weight_cache.lock() {
                for (_, buf) in cache.drain() {
                    self.device.destroy_buffer(buf.buffer, None);
                    self.device.free_memory(buf.memory, None);
                }
            }
            if let Ok(mut lru) = self.weight_lru.lock() {
                lru.clear();
            }
            if let Ok(mut pinned) = self.weight_pinned.lock() {
                pinned.clear();
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline_q4k.pipeline, None);
            self.device.destroy_pipeline(self.pipeline_q6k.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_shader_module(self.pipeline_q4k.shader, None);
            self.device
                .destroy_shader_module(self.pipeline_q6k.shader, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[allow(dead_code)]
fn batch_rows(x: &Tensor) -> u32 {
    let dims = x.dims();
    if dims.len() <= 1 {
        return 1;
    }
    dims[..dims.len() - 1]
        .iter()
        .product::<usize>()
        .max(1) as u32
}

/// Lightweight probe used by setup / auto detect.
pub fn probe() -> Result<()> {
    let entry =
        unsafe { Entry::load() }.map_err(|e| AppError::msg(format!("load libvulkan: {e}")))?;
    let app_name = c"lpc-llm-probe";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|e| AppError::msg(format!("vkCreateInstance: {e}")))?
    };
    let r = pick_device(&instance);
    unsafe { instance.destroy_instance(None) };
    r.map(|_| ())
}

fn pick_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| AppError::msg(format!("enumerate_physical_devices: {e}")))?
    };
    for physical in devices {
        let _props = unsafe { instance.get_physical_device_properties(physical) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        for (idx, fam) in families.iter().enumerate() {
            if fam.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((physical, idx as u32));
            }
        }
    }
    Err(AppError::msg("no Vulkan compute device found"))
}

fn create_shader(device: &Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    if spirv.len() % 4 != 0 {
        return Err(AppError::msg("SPIR-V length not multiple of 4"));
    }
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    unsafe {
        device
            .create_shader_module(&info, None)
            .map_err(|e| AppError::msg(format!("create_shader_module: {e}")))
    }
}

fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe {
        device
            .create_descriptor_set_layout(&dsl_info, None)
            .map_err(|e| AppError::msg(format!("create_descriptor_set_layout: {e}")))
    }
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let layouts = [descriptor_set_layout];
    let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    unsafe {
        device
            .create_pipeline_layout(&pl_info, None)
            .map_err(|e| AppError::msg(format!("create_pipeline_layout: {e}")))
    }
}

fn create_compute_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    shader: vk::ShaderModule,
) -> Result<vk::Pipeline> {
    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    let create = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create], None)
            .map_err(|e| AppError::msg(format!("create_compute_pipelines: {e:?}")))?
    };
    Ok(pipelines[0])
}

fn find_memory_type(
    instance: &Instance,
    physical: vk::PhysicalDevice,
    type_bits: u32,
    props: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let mem = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..mem.memory_type_count {
        if type_bits & (1 << i) != 0
            && mem.memory_types[i as usize]
                .property_flags
                .contains(props)
        {
            return Ok(i);
        }
    }
    Err(AppError::msg("no suitable Vulkan memory type"))
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use crate::device::vulkan::q4k::{self, pack_constant_q4k};
    use crate::device::vulkan::q6k::{self, pack_constant_q6k};

    #[test]
    fn q4k_gpu_matches_cpu_when_vulkan_available() {
        let Ok(ctx) = VulkanContext::new() else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let n = 4usize;
        let k = 256usize;
        let w = pack_constant_q4k(n, k, 2, 0.25, 0.0);
        let x = vec![1.0f32; k];
        let cpu = q4k::q4k_gemv_cpu(&w, n, k, &x).expect("cpu");
        let gpu = ctx
            .q4k_gemm_gpu(0xDEAD_BEEF, &w, n as u32, k as u32, 1, &x)
            .expect("gpu");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "mismatch at {i}: cpu={a} gpu={b}"
            );
        }
    }

    #[test]
    fn q6k_gpu_matches_cpu_when_vulkan_available() {
        let Ok(ctx) = VulkanContext::new() else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let n = 4usize;
        let k = 256usize;
        let w = pack_constant_q6k(n, k, 40, 0.5, 2);
        let x = vec![1.0f32; k];
        let cpu = q6k::q6k_gemv_cpu(&w, n, k, &x).expect("cpu");
        let gpu = ctx
            .q6k_gemm_gpu(0xC0FFEE, &w, n as u32, k as u32, 1, &x)
            .expect("gpu");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "mismatch at {i}: cpu={a} gpu={b}"
            );
        }
    }
}
