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
use super::q8_0::{self, pack_constant_q8_0, BLOCK_Q8_0_SIZE, QK8_0};
use crate::error::{AppError, Result};

const SPIRV_Q4K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemv.spv"));
const SPIRV_Q6K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q6k_gemv.spv"));
const SPIRV_Q8_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q8_0_gemv.spv"));
const SPIRV_SOFTMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax_f32.spv"));

/// Cap cached weight buffers. Hot (pinned) + recent experts.
const MAX_WEIGHT_CACHE: usize = 4096;

/// Max independent GEMVs recorded into one submit (Top-8 gates+ups=16, Q/K/V, …).
const FUSED_MAX_OPS: usize = 16;

/// Host-side f32 activation prepared for a fused GPU submit.
/// Keeps shape metadata so callers can avoid repeated `Tensor` reshape/`to_vec1`
/// when packing many GEMVs; the GPU path uploads once per fused group and only
/// downloads after the whole submit (DeviceAct = deferred D2H boundary).
#[derive(Clone, Debug)]
pub struct DeviceAct {
    data: Vec<f32>,
    m: u32,
    k: u32,
    /// Logical dims of the original tensor (last dim = k).
    dims: Vec<usize>,
}

impl DeviceAct {
    pub fn from_tensor(x: &Tensor) -> Result<Self> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let dims = x_f32.dims().to_vec();
        if dims.is_empty() {
            return Err(AppError::msg("DeviceAct: empty input"));
        }
        let k = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
        let data = x_f32.reshape((m, k))?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            data,
            m: m as u32,
            k: k as u32,
            dims,
        })
    }

    pub fn m(&self) -> u32 {
        self.m
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(self.data.clone(), self.dims.as_slice(), &CandleDevice::Cpu)
            .map_err(|e| AppError::msg(e.to_string()))
    }

    pub(crate) fn from_flat(
        data: Vec<f32>,
        m: u32,
        n: u32,
        template_dims: &[usize],
    ) -> Result<Self> {
        let mut dims = template_dims.to_vec();
        if dims.is_empty() {
            dims = vec![m as usize, n as usize];
        } else {
            *dims.last_mut().unwrap() = n as usize;
        }
        Ok(Self {
            data,
            m,
            k: n,
            dims,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuantKind {
    Q4K,
    Q6K,
    Q8_0,
}

impl QuantKind {
    fn from_dtype(dt: GgmlDType) -> Option<Self> {
        match dt {
            GgmlDType::Q4K => Some(Self::Q4K),
            GgmlDType::Q6K => Some(Self::Q6K),
            GgmlDType::Q8_0 => Some(Self::Q8_0),
            _ => None,
        }
    }

    fn block_size(self) -> usize {
        match self {
            Self::Q4K => BLOCK_Q4K_SIZE,
            Self::Q6K => BLOCK_Q6K_SIZE,
            Self::Q8_0 => BLOCK_Q8_0_SIZE,
        }
    }

    fn qk(self) -> usize {
        match self {
            Self::Q4K | Self::Q6K => QK_K,
            Self::Q8_0 => QK8_0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
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
    pipeline_q8_0: QuantPipeline,
    pipeline_softmax: QuantPipeline,
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
    /// Dual scratch slots for ping-pong (wait only the slot being reused).
    scratches: Mutex<[Option<SubmitResources>; 2]>,
    scratch_rr: AtomicU64,
    /// When false, always fall back to Candle CPU (microbench lost on this GPU).
    gpu_gemv_worthwhile: bool,
    /// When true, ≥4 fused Q8_0 downs may use GPU (microbench win).
    gpu_q8_0_fused_worthwhile: bool,
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
        let shader_q8_0 = create_shader(&device, SPIRV_Q8_0)?;
        let shader_softmax = create_shader(&device, SPIRV_SOFTMAX)?;
        let pipeline_q4k = QuantPipeline {
            shader: shader_q4k,
            pipeline: create_compute_pipeline(&device, pipeline_layout, shader_q4k)?,
        };
        let pipeline_q6k = QuantPipeline {
            shader: shader_q6k,
            pipeline: create_compute_pipeline(&device, pipeline_layout, shader_q6k)?,
        };
        let pipeline_q8_0 = QuantPipeline {
            shader: shader_q8_0,
            pipeline: create_compute_pipeline(&device, pipeline_layout, shader_q8_0)?,
        };
        let pipeline_softmax = QuantPipeline {
            shader: shader_softmax,
            pipeline: create_compute_pipeline(&device, pipeline_layout, shader_softmax)?,
        };

        // Dual scratch × FUSED_MAX_OPS sets (each set: 3 storage + 1 uniform).
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (FUSED_MAX_OPS as u32) * 2 * 3,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: (FUSED_MAX_OPS as u32) * 2,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets((FUSED_MAX_OPS as u32) * 2)
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
            pipeline_q8_0,
            pipeline_softmax,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            command_pool,
            submit_lock: Mutex::new(()),
            weight_cache: Mutex::new(HashMap::new()),
            weight_lru: Mutex::new(VecDeque::new()),
            weight_pinned: Mutex::new(HashSet::new()),
            scratches: Mutex::new([None, None]),
            scratch_rr: AtomicU64::new(0),
            gpu_gemv_worthwhile: true,
            gpu_q8_0_fused_worthwhile: true,
            skip_count: AtomicU64::new(0),
            submit_count: AtomicU64::new(0),
            skip_seen: Mutex::new(HashSet::new()),
        };
        ctx.gpu_gemv_worthwhile = ctx.microbench_gpu_vs_cpu().unwrap_or(false);
        // Allow Q8_0 fused path during probe; result overwrites the flag.
        ctx.gpu_q8_0_fused_worthwhile = true;
        ctx.gpu_q8_0_fused_worthwhile = ctx.microbench_q8_0_fused().unwrap_or(false);
        if ctx.gpu_gemv_worthwhile {
            eprintln!(
                "compute: GPU Q4_K/Q6_K for VRAM-warmed weights; \
                 Q8_0 fused downs {}",
                if ctx.gpu_q8_0_fused_worthwhile {
                    "enabled (≥4 ops, microbench win)"
                } else {
                    "disabled (CPU faster on this GPU)"
                }
            );
        }
        Ok(ctx)
    }

    /// True when the naive GPU GEMV path beat Candle-style CPU on a small probe.
    pub fn gpu_gemv_worthwhile(&self) -> bool {
        self.gpu_gemv_worthwhile
    }

    /// True when ≥4 fused Q8_0 downs beat parallel CPU on this GPU.
    pub fn gpu_q8_0_fused_worthwhile(&self) -> bool {
        self.gpu_q8_0_fused_worthwhile
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

    /// 8 fused Q8_0 GEMVs (n≈2816, k≈704) vs 8 parallel CPU refs. Win if GPU < CPU * 0.90.
    fn microbench_q8_0_fused(&self) -> Result<bool> {
        let n = 2816usize;
        let k = 704usize;
        let w = pack_constant_q8_0(n, k, 2, 0.25);
        let x = vec![1.0f32; k];
        let rounds = 2usize;

        let t0 = std::time::Instant::now();
        for _ in 0..rounds {
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    let w_ref = &w;
                    let x_ref = &x;
                    scope.spawn(move || {
                        let _ = q8_0::q8_0_gemv_cpu(w_ref, n, k, x_ref);
                    });
                }
            });
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / rounds as f64;

        let key = 0xC8F0_BEEF_u64 as usize;
        let ops: Vec<(usize, &[u8], u32, u32, QuantKind)> = (0..8)
            .map(|i| {
                (
                    key + i,
                    w.as_slice(),
                    n as u32,
                    k as u32,
                    QuantKind::Q8_0,
                )
            })
            .collect();
        let xs: Vec<&[f32]> = (0..8).map(|_| x.as_slice()).collect();

        // Warm GPU path once.
        let _ = self.quant_gemm_gpu_multi_xs(&ops, &xs)?;
        let t1 = std::time::Instant::now();
        for _ in 0..rounds {
            let _ = self.quant_gemm_gpu_multi_xs(&ops, &xs)?;
        }
        let gpu_ms = t1.elapsed().as_secs_f64() * 1000.0 / rounds as f64;

        let win = gpu_ms < cpu_ms * 0.90;
        eprintln!(
            "compute: Q8_0 fused×8 probe CPU≈{cpu_ms:.2}ms GPU≈{gpu_ms:.2}ms → {}",
            if win {
                "use GPU for ≥4 fused Q8_0 downs"
            } else {
                "prefer Candle CPU for Q8_0 downs"
            }
        );
        Ok(win)
    }

    /// Multiple independent Q4_K / Q6_K / Q8_0 GEMVs against the same activation (one submit).
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
        let act = DeviceAct::from_tensor(x)?;
        let acts = self.qmatmul_multi_act(ws, &act)?;
        let mut outs = Vec::with_capacity(acts.len());
        for a in acts {
            outs.push(a.to_tensor()?);
        }
        Ok(outs)
    }

    /// Same as `qmatmul_multi` but keeps results as [`DeviceAct`] (caller downloads once).
    pub fn qmatmul_multi_act(&self, ws: &[&QMatMul], act: &DeviceAct) -> Result<Vec<DeviceAct>> {
        if ws.is_empty() {
            return Ok(Vec::new());
        }
        let mut guards = Vec::with_capacity(ws.len());
        let mut metas: Vec<(usize, u32, u32, QuantKind)> = Vec::with_capacity(ws.len());
        for w in ws {
            let (key, n, k, kind, guard) = self.prepare_one_gemv(w, act.k() as usize)?;
            guards.push(guard);
            metas.push((key, n, k, kind));
        }
        let prepared: Vec<(usize, &[u8], u32, u32, QuantKind)> = metas
            .iter()
            .zip(guards.iter())
            .map(|(&(key, n, k, kind), g)| (key, g.as_ref(), n, k, kind))
            .collect();
        let x_refs: Vec<&[f32]> = (0..prepared.len()).map(|_| act.data()).collect();
        let chunks = self.quant_gemm_gpu_multi_xs(&prepared, &x_refs)?;
        let mut outs = Vec::with_capacity(chunks.len());
        for (i, c_flat) in chunks.into_iter().enumerate() {
            let n = metas[i].1;
            outs.push(DeviceAct::from_flat(c_flat, act.m(), n, act.dims())?);
        }
        Ok(outs)
    }

    /// Fused GEMVs with CPU work overlapped between `queue_submit` and fence wait / D2H.
    #[allow(dead_code)] // Phase 14 MoE / layer overlap
    pub fn qmatmul_multi_overlap<R, F>(
        &self,
        ws: &[&QMatMul],
        x: &Tensor,
        between: F,
    ) -> Result<(Vec<Tensor>, R)>
    where
        F: FnOnce() -> Result<R>,
    {
        if ws.is_empty() {
            let r = between()?;
            return Ok((Vec::new(), r));
        }
        if !self.should_try_gpu(x) {
            self.log_skip("GPU GEMV disabled on this device");
            return Err(AppError::msg(
                "vulkan-skip: GPU GEMV disabled on this device",
            ));
        }
        let act = DeviceAct::from_tensor(x)?;
        let mut guards = Vec::with_capacity(ws.len());
        let mut metas: Vec<(usize, u32, u32, QuantKind)> = Vec::with_capacity(ws.len());
        for w in ws {
            let (key, n, k, kind, guard) = self.prepare_one_gemv(w, act.k() as usize)?;
            guards.push(guard);
            metas.push((key, n, k, kind));
        }
        let prepared: Vec<(usize, &[u8], u32, u32, QuantKind)> = metas
            .iter()
            .zip(guards.iter())
            .map(|(&(key, n, k, kind), g)| (key, g.as_ref(), n, k, kind))
            .collect();
        let x_refs: Vec<&[f32]> = (0..prepared.len()).map(|_| act.data()).collect();
        let (chunks, r) =
            self.quant_gemm_gpu_multi_xs_overlap(&prepared, &x_refs, between)?;
        let mut outs = Vec::with_capacity(chunks.len());
        for (i, c_flat) in chunks.into_iter().enumerate() {
            let n = metas[i].1;
            let da = DeviceAct::from_flat(c_flat, act.m(), n, act.dims())?;
            outs.push(da.to_tensor()?);
        }
        Ok((outs, r))
    }

    /// Independent GEMVs with **per-op activations** (one submit; MoE expert downs).
    pub fn qmatmul_multi_xs(&self, pairs: &[(&QMatMul, &Tensor)]) -> Result<Vec<Tensor>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        if !self.should_try_gpu(pairs[0].1) {
            self.log_skip("GPU GEMV disabled on this device");
            return Err(AppError::msg(
                "vulkan-skip: GPU GEMV disabled on this device",
            ));
        }
        let acts: Vec<DeviceAct> = pairs
            .iter()
            .map(|(_, x)| DeviceAct::from_tensor(x))
            .collect::<Result<Vec<_>>>()?;
        let ws: Vec<&QMatMul> = pairs.iter().map(|(w, _)| *w).collect();
        let out_acts = self.qmatmul_multi_xs_act(&ws, &acts)?;
        let mut outs = Vec::with_capacity(out_acts.len());
        for a in out_acts {
            outs.push(a.to_tensor()?);
        }
        Ok(outs)
    }

    /// Per-op [`DeviceAct`] inputs → fused submit → [`DeviceAct`] outputs (D2H deferred to caller).
    pub fn qmatmul_multi_xs_act(
        &self,
        ws: &[&QMatMul],
        acts: &[DeviceAct],
    ) -> Result<Vec<DeviceAct>> {
        if ws.len() != acts.len() {
            return Err(AppError::msg(format!(
                "qmatmul_multi_xs_act: {} weights vs {} activations",
                ws.len(),
                acts.len()
            )));
        }
        if ws.is_empty() {
            return Ok(Vec::new());
        }
        let mut guards = Vec::with_capacity(ws.len());
        let mut metas: Vec<(usize, u32, u32, QuantKind)> = Vec::with_capacity(ws.len());
        for (i, w) in ws.iter().enumerate() {
            let act = &acts[i];
            let (key, n, k, kind, guard) = self.prepare_one_gemv(w, act.k() as usize)?;
            let expect_x = (act.m() as usize).saturating_mul(k as usize);
            if act.data().len() != expect_x {
                return Err(AppError::msg(format!(
                    "DeviceAct len {} != m*k={expect_x}",
                    act.data().len()
                )));
            }
            guards.push(guard);
            metas.push((key, n, k, kind));
        }
        let prepared: Vec<(usize, &[u8], u32, u32, QuantKind)> = metas
            .iter()
            .zip(guards.iter())
            .map(|(&(key, n, k, kind), g)| (key, g.as_ref(), n, k, kind))
            .collect();
        let x_refs: Vec<&[f32]> = acts.iter().map(|a| a.data()).collect();
        let chunks = self.quant_gemm_gpu_multi_xs(&prepared, &x_refs)?;
        let mut outs = Vec::with_capacity(chunks.len());
        for (i, c_flat) in chunks.into_iter().enumerate() {
            let n = metas[i].1;
            outs.push(DeviceAct::from_flat(
                c_flat,
                acts[i].m(),
                n,
                acts[i].dims(),
            )?);
        }
        Ok(outs)
    }

    /// Validate weight + pull QTensor bytes; weight must already be VRAM-cached.
    fn prepare_one_gemv<'a>(
        &self,
        w: &'a QMatMul,
        x_last: usize,
    ) -> Result<(usize, u32, u32, QuantKind, std::borrow::Cow<'a, [u8]>)> {
        let QMatMul::QTensor(qt) = w else {
            self.log_skip("weight already dequantized (Tensor/F16)");
            return Err(AppError::msg(
                "vulkan-skip: weight already dequantized (Tensor/F16)",
            ));
        };
        let Some(kind) = QuantKind::from_dtype(qt.dtype()) else {
            self.log_skip(&format!("dtype {:?} (unsupported quant)", qt.dtype()));
            return Err(AppError::msg(format!(
                "vulkan-skip: dtype {:?} (unsupported quant)",
                qt.dtype()
            )));
        };
        let (n, k) = qt
            .shape()
            .dims2()
            .map_err(|e| AppError::msg(e.to_string()))?;
        let qk = kind.qk();
        if !k.is_multiple_of(qk) {
            self.log_skip(&format!("k={k} not multiple of {qk}"));
            return Err(AppError::msg(format!(
                "vulkan-skip: k={k} not multiple of {qk}"
            )));
        }
        if x_last != k {
            return Err(AppError::msg(format!(
                "Vulkan {} shape mismatch: x last={x_last} k={k}",
                kind.name()
            )));
        }
        // Tiny output rows (e.g. MoE router n=128): host fence dominates the kernel.
        // Prefer Candle CPU and keep the queue free for large attn / gate GEMVs.
        const MIN_GPU_N: u32 = 256;
        if (n as u32) < MIN_GPU_N {
            self.log_skip(&format!(
                "n={n} < {MIN_GPU_N} (CPU; fence-bound tiny GEMV)"
            ));
            return Err(AppError::msg(format!(
                "vulkan-skip: n={n} < {MIN_GPU_N} (CPU; fence-bound tiny GEMV)"
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
        let expect = n * (k / kind.qk()) * kind.block_size();
        if w_bytes.as_ref().len() != expect {
            return Err(AppError::msg(format!(
                "{} byte len {} != expected {expect}",
                kind.name(),
                w_bytes.as_ref().len()
            )));
        }
        Ok((key, n as u32, k as u32, kind, w_bytes))
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
        let mut slots = self
            .scratches
            .lock()
            .map_err(|_| AppError::msg("scratch lock poisoned"))?;
        for slot_idx in 0..2 {
            if slots[slot_idx].is_some() {
                continue;
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
            slots[slot_idx] = Some(SubmitResources {
                x,
                y,
                u,
                sets,
                cmd: cmds[0],
                fence,
            });
        }
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
        let _ = m;
        let mut outs = self.quant_gemm_gpu_multi_xs(
            &[(key, w_bytes, n, k, QuantKind::Q4K)],
            &[x],
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
        let _ = m;
        let mut outs = self.quant_gemm_gpu_multi_xs(
            &[(key, w_bytes, n, k, QuantKind::Q6K)],
            &[x],
        )?;
        outs.pop().ok_or_else(|| AppError::msg("gpu gemm empty"))
    }

    /// Shared-activation fused GEMVs (all ops read the same `x`).
    #[allow(dead_code)]
    fn quant_gemm_gpu_multi(
        &self,
        ops: &[(usize, &[u8], u32, u32, QuantKind)],
        m: u32,
        x: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        let _ = m;
        let x_refs: Vec<&[f32]> = (0..ops.len()).map(|_| x).collect();
        self.quant_gemm_gpu_multi_xs(ops, &x_refs)
    }

    /// Fused GEMVs with **per-op** activations packed into one submit.
    /// `xs[i].len()` must equal `m_i * k_i` where `m_i = xs[i].len() / k_i`.
    fn quant_gemm_gpu_multi_xs(
        &self,
        ops: &[(usize, &[u8], u32, u32, QuantKind)],
        xs: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>> {
        self.quant_gemm_gpu_multi_xs_overlap(ops, xs, || Ok(()))
            .map(|(outs, _)| outs)
    }

    /// Like [`Self::quant_gemm_gpu_multi_xs`] but runs `between` after submit, before wait+D2H.
    fn quant_gemm_gpu_multi_xs_overlap<R, F>(
        &self,
        ops: &[(usize, &[u8], u32, u32, QuantKind)],
        xs: &[&[f32]],
        between: F,
    ) -> Result<(Vec<Vec<f32>>, R)>
    where
        F: FnOnce() -> Result<R>,
    {
        if ops.is_empty() {
            let r = between()?;
            return Ok((Vec::new(), r));
        }
        if ops.len() != xs.len() {
            return Err(AppError::msg(format!(
                "quant_gemm_gpu_multi_xs: {} ops vs {} xs",
                ops.len(),
                xs.len()
            )));
        }
        if ops.len() > FUSED_MAX_OPS {
            // Chunk without overlap on intermediate chunks; run between on the last chunk only.
            let mut all = Vec::new();
            let n_chunks = ops.chunks(FUSED_MAX_OPS).len();
            let mut between_opt = Some(between);
            let mut last_r: Option<R> = None;
            for (ci, (chunk_ops, chunk_xs)) in ops
                .chunks(FUSED_MAX_OPS)
                .zip(xs.chunks(FUSED_MAX_OPS))
                .enumerate()
            {
                if ci + 1 == n_chunks {
                    let (chunk_outs, r) = self.quant_gemm_gpu_multi_xs_overlap(
                        chunk_ops,
                        chunk_xs,
                        between_opt.take().unwrap(),
                    )?;
                    all.extend(chunk_outs);
                    last_r = Some(r);
                } else {
                    all.extend(self.quant_gemm_gpu_multi_xs(chunk_ops, chunk_xs)?);
                }
            }
            return Ok((all, last_r.unwrap()));
        }

        // Policy: single/tiny Q8_0 stays CPU; ≥4 fused may use GPU when microbench wins.
        let any_q8 = ops.iter().any(|op| op.4 == QuantKind::Q8_0);
        if any_q8 && (ops.len() < 4 || !self.gpu_q8_0_fused_worthwhile) {
            let reason = if ops.len() < 4 {
                format!("Q8_0 fused len={} < 4 (CPU)", ops.len())
            } else {
                "Q8_0 fused microbench lost (CPU)".to_string()
            };
            self.log_skip(&reason);
            return Err(AppError::msg(format!("vulkan-skip: {reason}")));
        }

        let ms: Vec<u32> = ops
            .iter()
            .zip(xs.iter())
            .map(|(&(_, _, _, k, _), x)| {
                let k = k as usize;
                if k == 0 || !x.len().is_multiple_of(k) {
                    0
                } else {
                    (x.len() / k) as u32
                }
            })
            .collect();
        for (i, &m) in ms.iter().enumerate() {
            if m == 0 {
                return Err(AppError::msg(format!(
                    "Vulkan GEMV op {i}: x len {} not multiple of k={}",
                    xs[i].len(),
                    ops[i].3
                )));
            }
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
                let w_buf = cache.get(&key).ok_or_else(|| {
                    AppError::msg("vulkan-skip: weight cache miss after ensure")
                })?;
                out.push((w_buf.buffer, w_buf.size));
            }
            out
        };

        let x_lens: Vec<usize> = xs.iter().map(|x| x.len()).collect();
        let x_total: usize = x_lens.iter().sum();
        let x_bytes = (x_total * size_of::<f32>()) as u64;
        let y_elems: Vec<usize> = ops
            .iter()
            .zip(ms.iter())
            .map(|(&(_, _, n, _, _), &m)| (m as usize) * (n as usize))
            .collect();
        let y_total: usize = y_elems.iter().sum();
        let y_bytes = (y_total * size_of::<f32>()) as u64;
        let u_bytes = (size_of::<[u32; 4]>() * ops.len()) as u64;

        let slot_idx = (self.scratch_rr.fetch_add(1, Ordering::Relaxed) % 2) as usize;
        let mut scratch_guard = self
            .scratches
            .lock()
            .map_err(|_| AppError::msg("scratch lock poisoned"))?;
        let scratch = scratch_guard[slot_idx]
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

        // Pack all activations contiguously (one H2D for the fused group).
        let mut x_off_elems = 0usize;
        let mut x_offsets = Vec::with_capacity(ops.len());
        unsafe {
            for (i, x) in xs.iter().enumerate() {
                x_offsets.push(x_off_elems);
                std::ptr::copy_nonoverlapping(
                    x.as_ptr(),
                    (scratch.x.ptr as *mut f32).add(x_off_elems),
                    x.len(),
                );
                x_off_elems += x_lens[i];
            }
        }

        let mut y_off_elems = 0usize;
        let mut writes_storage: Vec<(
            [vk::DescriptorBufferInfo; 1],
            [vk::DescriptorBufferInfo; 1],
            [vk::DescriptorBufferInfo; 1],
            [vk::DescriptorBufferInfo; 1],
        )> = Vec::with_capacity(ops.len());
        for (i, &(_, _, n, k, _)) in ops.iter().enumerate() {
            let m = ms[i];
            let dims = [n, k, m, 0u32];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dims.as_ptr(),
                    scratch.u.ptr.add(i * size_of::<[u32; 4]>()) as *mut u32,
                    4,
                );
            }
            let x_op_bytes = (x_lens[i] * size_of::<f32>()) as u64;
            let y_op_bytes = (y_elems[i] * size_of::<f32>()) as u64;
            writes_storage.push((
                [vk::DescriptorBufferInfo {
                    buffer: scratch.x.buffer,
                    offset: (x_offsets[i] * size_of::<f32>()) as u64,
                    range: x_op_bytes.max(4),
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

        // Wait ONLY this slot's fence before reuse (other slot may still be in flight).
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
        let last = ops.len().saturating_sub(1);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| AppError::msg(format!("begin_command_buffer: {e}")))?;
            for i in 0..ops.len() {
                let pipeline = match kinds[i] {
                    QuantKind::Q4K => self.pipeline_q4k.pipeline,
                    QuantKind::Q6K => self.pipeline_q6k.pipeline,
                    QuantKind::Q8_0 => self.pipeline_q8_0.pipeline,
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
                let groups = (ms[i] * ns[i]).div_ceil(64);
                self.device.cmd_dispatch(cmd, groups, 1, 1);
                // Between ops: shader→shader only. HOST after the last op (one D2H).
                let (dst_access, dst_stage) = if i == last {
                    (
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ,
                        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                    )
                } else {
                    (
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    )
                };
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(dst_access);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    dst_stage,
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
        }

        // CPU work while GPU runs (submit_lock held; fine for single-threaded decode).
        let r = between()?;

        unsafe {
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
        Ok((outs, r))
    }

    // --- end quant_gemm_gpu_multi_xs ---

    /// Softmax over the last dimension (Candle-compatible f32 max/exp/sum/div).
    #[allow(dead_code)] // Phase 14 Softmax GPU
    pub fn softmax_last_dim_gpu(&self, rows: u32, cols: u32, x: &[f32]) -> Result<Vec<f32>> {
        let expect = (rows as usize).saturating_mul(cols as usize);
        if x.len() != expect {
            return Err(AppError::msg(format!(
                "softmax_last_dim_gpu: x len {} != rows*cols={expect}",
                x.len()
            )));
        }
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }

        let _guard = self
            .submit_lock
            .lock()
            .map_err(|_| AppError::msg("vulkan submit lock poisoned"))?;
        self.ensure_scratch()?;

        let x_bytes = (x.len() * size_of::<f32>()) as u64;
        let y_bytes = x_bytes;
        let u_bytes = size_of::<[u32; 4]>() as u64;

        let slot_idx = (self.scratch_rr.fetch_add(1, Ordering::Relaxed) % 2) as usize;
        let mut scratch_guard = self
            .scratches
            .lock()
            .map_err(|_| AppError::msg("scratch lock poisoned"))?;
        let scratch = scratch_guard[slot_idx]
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
            std::ptr::copy_nonoverlapping(
                x.as_ptr(),
                scratch.x.ptr as *mut f32,
                x.len(),
            );
            let dims = [rows, cols, 0u32, 0u32];
            std::ptr::copy_nonoverlapping(dims.as_ptr(), scratch.u.ptr as *mut u32, 4);
        }

        // Softmax bindings: 0=x, 1=y, 2=unused, 3=dims (same layout types as GEMV).
        let x_info = [vk::DescriptorBufferInfo {
            buffer: scratch.x.buffer,
            offset: 0,
            range: x_bytes.max(4),
        }];
        let y_info = [vk::DescriptorBufferInfo {
            buffer: scratch.y.buffer,
            offset: 0,
            range: y_bytes.max(4),
        }];
        let unused_info = [vk::DescriptorBufferInfo {
            buffer: scratch.y.buffer,
            offset: 0,
            range: 4,
        }];
        let u_info = [vk::DescriptorBufferInfo {
            buffer: scratch.u.buffer,
            offset: 0,
            range: u_bytes,
        }];
        let set = scratch.sets[0];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&x_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&y_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&unused_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&u_info),
        ];
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
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| AppError::msg(format!("begin_command_buffer: {e}")))?;
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_softmax.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_dispatch(cmd, 1, rows, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
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

        let mut out = vec![0f32; x.len()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch.y.ptr as *const f32,
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(out)
    }

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
            if let Ok(mut scratches) = self.scratches.lock() {
                for slot in scratches.iter_mut() {
                    if let Some(s) = slot.take() {
                        self.device.destroy_fence(s.fence, None);
                        // Descriptor set / command buffer freed with pools.
                        self.destroy_mapped(s.x);
                        self.destroy_mapped(s.y);
                        self.destroy_mapped(s.u);
                    }
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
            self.device.destroy_pipeline(self.pipeline_q8_0.pipeline, None);
            self.device
                .destroy_pipeline(self.pipeline_softmax.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_shader_module(self.pipeline_q4k.shader, None);
            self.device
                .destroy_shader_module(self.pipeline_q6k.shader, None);
            self.device
                .destroy_shader_module(self.pipeline_q8_0.shader, None);
            self.device
                .destroy_shader_module(self.pipeline_softmax.shader, None);
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

/// CPU Softmax over the last dimension (Candle-compatible f32).
#[allow(dead_code)]
pub fn softmax_last_dim_f32(rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), rows.saturating_mul(cols));
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let base = r * cols;
        let row = &x[base..base + cols];
        let mut m = f32::NEG_INFINITY;
        for &v in row {
            if v > m {
                m = v;
            }
        }
        let mut sum = 0f32;
        for j in 0..cols {
            let e = (row[j] - m).exp();
            y[base + j] = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for j in 0..cols {
            y[base + j] *= inv;
        }
    }
    y
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

    #[test]
    fn softmax_gpu_matches_cpu_when_vulkan_available() {
        let Ok(ctx) = VulkanContext::new() else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let rows = 3usize;
        let cols = 17usize;
        let mut x = Vec::with_capacity(rows * cols);
        for i in 0..(rows * cols) {
            x.push((i as f32) * 0.1 - 1.5);
        }
        let cpu = softmax_last_dim_f32(rows, cols, &x);
        let gpu = ctx
            .softmax_last_dim_gpu(rows as u32, cols as u32, &x)
            .expect("gpu softmax");
        assert_eq!(cpu.len(), gpu.len());
        for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "softmax mismatch at {i}: cpu={a} gpu={b}"
            );
        }
    }
}
