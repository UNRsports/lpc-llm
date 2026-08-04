//! ash-based Vulkan compute: Q4_K dequant+GEMV with VRAM-resident weights.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use ash::vk;
use ash::{Device, Entry, Instance};
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device as CandleDevice, Tensor};

use super::q4k::{self, pack_constant_q4k, BLOCK_Q4K_SIZE, QK_K};
use crate::error::{AppError, Result};

const SPIRV_Q4K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemv.spv"));

/// Cap cached weight buffers so streamed (rebuilt) tensors cannot grow VRAM forever.
/// MoE keeps many Top-K expert Q4_K mats hot across turns — allow a larger set.
const MAX_WEIGHT_CACHE: usize = 768;

/// Below this batch size, Candle CPU Q4_K is usually faster than H2D+dispatch+D2H
/// unless the weight is already resident in the VRAM cache.
const SMALL_BATCH_CPU: u32 = 8;

struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
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
    set: vk::DescriptorSet,
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
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    shader: vk::ShaderModule,
    submit_lock: Mutex<()>,
    /// Quantized weight blobs keyed by `Arc<QTensor>` pointer.
    weight_cache: Mutex<HashMap<usize, GpuBuffer>>,
    scratch: Mutex<Option<SubmitResources>>,
    /// When false, always fall back to Candle CPU (microbench lost on this GPU).
    gpu_gemv_worthwhile: bool,
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

        let shader = create_shader(&device, SPIRV_Q4K)?;
        let (descriptor_set_layout, pipeline_layout, pipeline) =
            create_pipeline(&device, shader)?;

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 64,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 16,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(16)
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
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            command_pool,
            shader,
            submit_lock: Mutex::new(()),
            weight_cache: Mutex::new(HashMap::new()),
            scratch: Mutex::new(None),
            gpu_gemv_worthwhile: true,
        };
        ctx.gpu_gemv_worthwhile = ctx.microbench_gpu_vs_cpu().unwrap_or(false);
        Ok(ctx)
    }

    /// True when the naive GPU GEMV path beat Candle-style CPU on a small probe.
    pub fn gpu_gemv_worthwhile(&self) -> bool {
        self.gpu_gemv_worthwhile
    }

    /// Upload a Q4_K weight into the VRAM cache without running GEMV.
    /// Enables small-batch decode to hit GPU on the next `qmatmul`.
    pub fn warm_q4k(&self, w: &QMatMul) -> Result<()> {
        if !self.gpu_gemv_worthwhile {
            return Ok(());
        }
        let QMatMul::QTensor(qt) = w else {
            return Ok(());
        };
        if qt.dtype() != GgmlDType::Q4K {
            return Ok(());
        }
        let key = Arc::as_ptr(qt) as usize;
        let w_bytes = qt
            .data()
            .map_err(|e| AppError::msg(format!("QTensor data: {e}")))?;
        self.ensure_weight(key, w_bytes.as_ref())
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

    /// Q4_K path: VRAM-resident quantized weights + GPU dequant+GEMV.
    /// Returns `Err` with message starting `vulkan-skip:` for silent CPU fallback.
    pub fn qmatmul(&self, w: &QMatMul, x: &Tensor) -> Result<Tensor> {
        if !self.gpu_gemv_worthwhile {
            return Err(AppError::msg(
                "vulkan-skip: GPU GEMV slower than CPU on this device",
            ));
        }
        let QMatMul::QTensor(qt) = w else {
            return Err(AppError::msg(
                "vulkan-skip: weight already dequantized (Tensor/F16)",
            ));
        };
        if qt.dtype() != GgmlDType::Q4K {
            return Err(AppError::msg(format!(
                "vulkan-skip: dtype {:?} (Q4_K only)",
                qt.dtype()
            )));
        }
        let (n, k) = qt
            .shape()
            .dims2()
            .map_err(|e| AppError::msg(e.to_string()))?;
        if !k.is_multiple_of(QK_K) {
            return Err(AppError::msg(format!(
                "vulkan-skip: k={k} not multiple of {QK_K}"
            )));
        }
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_dims = x_f32.dims().to_vec();
        if x_dims.is_empty() {
            return Err(AppError::msg("Vulkan Q4_K: empty input"));
        }
        let last = *x_dims.last().unwrap();
        if last != k {
            return Err(AppError::msg(format!(
                "Vulkan Q4_K shape mismatch: x last={last} k={k}"
            )));
        }
        let m: usize = x_dims[..x_dims.len() - 1].iter().product::<usize>().max(1);
        let key = Arc::as_ptr(qt) as usize;

        // Small batches: only use GPU if the weight is already in VRAM (hot layers).
        // Uploading ephemeral streamed tensors every token is far slower than CPU.
        if (m as u32) < SMALL_BATCH_CPU {
            let cached = self
                .weight_cache
                .lock()
                .map_err(|_| AppError::msg("weight cache lock poisoned"))?
                .contains_key(&key);
            if !cached {
                return Err(AppError::msg(
                    "vulkan-skip: small batch / weight not yet in VRAM cache",
                ));
            }
        }

        let a_flat = x_f32
            .reshape((m, k))?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let w_bytes = qt
            .data()
            .map_err(|e| AppError::msg(format!("QTensor data: {e}")))?;
        let expect = n * (k / QK_K) * BLOCK_Q4K_SIZE;
        if w_bytes.len() != expect {
            return Err(AppError::msg(format!(
                "Q4_K byte len {} != expected {expect}",
                w_bytes.len()
            )));
        }

        let c_flat = self.q4k_gemm_gpu(key, &w_bytes, n as u32, k as u32, m as u32, &a_flat)?;
        let mut out_dims = x_dims;
        *out_dims.last_mut().unwrap() = n;
        Ok(Tensor::from_vec(c_flat, out_dims, &CandleDevice::Cpu)?)
    }

    fn ensure_weight(&self, key: usize, bytes: &[u8]) -> Result<()> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| AppError::msg("weight cache lock poisoned"))?;
        if cache.contains_key(&key) {
            return Ok(());
        }
        if cache.len() >= MAX_WEIGHT_CACHE {
            // Evict an arbitrary entry (streamed tensors churn Arc keys).
            if let Some(old_key) = cache.keys().next().copied() {
                if let Some(old) = cache.remove(&old_key) {
                    self.destroy_buffer(old);
                }
            }
        }
        let buf = self.upload_bytes(bytes)?;
        cache.insert(key, buf);
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
            size_of::<[u32; 4]>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        let set_layouts = [self.descriptor_set_layout];
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
            set: sets[0],
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
        let c_len = (m as usize) * (n as usize);
        let _guard = self
            .submit_lock
            .lock()
            .map_err(|_| AppError::msg("vulkan submit lock poisoned"))?;

        self.ensure_weight(key, w_bytes)?;
        self.ensure_scratch()?;

        let (w_buffer, w_size) = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| AppError::msg("weight cache lock poisoned"))?;
            let w_buf = cache
                .get(&key)
                .ok_or_else(|| AppError::msg("weight cache miss after ensure"))?;
            (w_buf.buffer, w_buf.size)
        };

        let x_bytes = (x.len() * size_of::<f32>()) as u64;
        let y_bytes = (c_len * size_of::<f32>()) as u64;
        let dims = [n, k, m, 0u32];

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

        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), scratch.x.ptr as *mut f32, x.len());
            std::ptr::copy_nonoverlapping(
                dims.as_ptr(),
                scratch.u.ptr as *mut u32,
                4,
            );
        }

        let x_info = [vk::DescriptorBufferInfo {
            buffer: scratch.x.buffer,
            offset: 0,
            range: x_bytes.max(4),
        }];
        let w_info = [vk::DescriptorBufferInfo {
            buffer: w_buffer,
            offset: 0,
            range: w_size,
        }];
        let y_info = [vk::DescriptorBufferInfo {
            buffer: scratch.y.buffer,
            offset: 0,
            range: y_bytes.max(4),
        }];
        let u_info = [vk::DescriptorBufferInfo {
            buffer: scratch.u.buffer,
            offset: 0,
            range: size_of::<[u32; 4]>() as u64,
        }];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(scratch.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&x_info),
            vk::WriteDescriptorSet::default()
                .dst_set(scratch.set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&w_info),
            vk::WriteDescriptorSet::default()
                .dst_set(scratch.set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&y_info),
            vk::WriteDescriptorSet::default()
                .dst_set(scratch.set)
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
        let groups = (m * n).div_ceil(64);
        let cmd = scratch.cmd;
        let set = scratch.set;
        let fence = scratch.fence;
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| AppError::msg(format!("begin_command_buffer: {e}")))?;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_dispatch(cmd, groups, 1, 1);
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

        let mut out = vec![0f32; c_len];
        unsafe {
            std::ptr::copy_nonoverlapping(scratch.y.ptr as *const f32, out.as_mut_ptr(), c_len);
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
        let bytes = data.len() as u64;
        let buf = self.alloc_buffer(
            bytes.max(4),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .device
                .map_memory(buf.memory, 0, bytes.max(4), vk::MemoryMapFlags::empty())
                .map_err(|e| AppError::msg(format!("map_memory: {e}")))?;
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
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_shader_module(self.shader, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
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

fn create_pipeline(
    device: &Device,
    shader: vk::ShaderModule,
) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)> {
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
    let descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&dsl_info, None)
            .map_err(|e| AppError::msg(format!("create_descriptor_set_layout: {e}")))?
    };
    let layouts = [descriptor_set_layout];
    let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pl_info, None)
            .map_err(|e| AppError::msg(format!("create_pipeline_layout: {e}")))?
    };
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
    Ok((descriptor_set_layout, pipeline_layout, pipelines[0]))
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
}
