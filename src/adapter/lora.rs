//! Side-path LoRA delta: `Δy = (α/r) · (x @ Aᵀ) @ Bᵀ`.

use candle_core::{DType, Device, Module, Tensor};

use crate::error::{AppError, Result};

/// Target Linear inside a transformer block (GGUF / PEFT-style names).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoraModuleName {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl LoraModuleName {
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AttnQ => "attn_q",
            Self::AttnK => "attn_k",
            Self::AttnV => "attn_v",
            Self::AttnOutput => "attn_output",
            Self::FfnGate => "ffn_gate",
            Self::FfnUp => "ffn_up",
            Self::FfnDown => "ffn_down",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "attn_q" | "q_proj" => Some(Self::AttnQ),
            "attn_k" | "k_proj" => Some(Self::AttnK),
            "attn_v" | "v_proj" => Some(Self::AttnV),
            "attn_output" | "o_proj" => Some(Self::AttnOutput),
            "ffn_gate" | "gate_proj" => Some(Self::FfnGate),
            "ffn_up" | "up_proj" => Some(Self::FfnUp),
            "ffn_down" | "down_proj" => Some(Self::FfnDown),
            _ => None,
        }
    }
}

/// One LoRA pair (A, B) already on device.
///
/// Shapes: `A = [rank, in_features]`, `B = [out_features, rank]`.
#[derive(Debug, Clone)]
pub struct LoraDelta {
    pub a: Tensor, // [rank, in]
    pub b: Tensor, // [out, rank]
    pub scale: f64,
}

impl LoraDelta {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [..., in] → flatten → [N, in] @ Aᵀ [in, rank] @ Bᵀ [rank, out] → reshape
        let a_t = self.a.t()?;
        let b_t = self.b.t()?;
        let dims = x.dims().to_vec();
        let in_f = *dims.last().ok_or_else(|| AppError::msg("LoRA input rank 0"))?;
        let leading: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        let x2 = x.reshape((leading, in_f))?;
        let mid = x2.matmul(&a_t)?;
        let out = mid.matmul(&b_t)?;
        let out_f = self.b.dim(0)?;
        let mut out_shape = dims;
        if let Some(last) = out_shape.last_mut() {
            *last = out_f;
        }
        Ok((out.reshape(out_shape)? * self.scale)?)
    }
}

/// Optional LoRA slots for every Linear in one layer.
#[derive(Debug, Default, Clone)]
pub struct LayerLora {
    pub q: Option<LoraDelta>,
    pub k: Option<LoraDelta>,
    pub v: Option<LoraDelta>,
    pub o: Option<LoraDelta>,
    pub gate: Option<LoraDelta>,
    pub up: Option<LoraDelta>,
    pub down: Option<LoraDelta>,
}

impl LayerLora {
    pub fn set(&mut self, name: LoraModuleName, delta: LoraDelta) {
        match name {
            LoraModuleName::AttnQ => self.q = Some(delta),
            LoraModuleName::AttnK => self.k = Some(delta),
            LoraModuleName::AttnV => self.v = Some(delta),
            LoraModuleName::AttnOutput => self.o = Some(delta),
            LoraModuleName::FfnGate => self.gate = Some(delta),
            LoraModuleName::FfnUp => self.up = Some(delta),
            LoraModuleName::FfnDown => self.down = Some(delta),
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, name: LoraModuleName) -> Option<&LoraDelta> {
        match name {
            LoraModuleName::AttnQ => self.q.as_ref(),
            LoraModuleName::AttnK => self.k.as_ref(),
            LoraModuleName::AttnV => self.v.as_ref(),
            LoraModuleName::AttnOutput => self.o.as_ref(),
            LoraModuleName::FfnGate => self.gate.as_ref(),
            LoraModuleName::FfnUp => self.up.as_ref(),
            LoraModuleName::FfnDown => self.down.as_ref(),
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.q.is_none()
            && self.k.is_none()
            && self.v.is_none()
            && self.o.is_none()
            && self.gate.is_none()
            && self.up.is_none()
            && self.down.is_none()
    }
}

/// Apply `Wq(x)` then optional LoRA delta.
pub fn qmatmul_with_lora(
    w: &candle_core::quantized::QMatMul,
    lora: Option<&LoraDelta>,
    x: &Tensor,
) -> Result<Tensor> {
    let y = w.forward(x)?;
    match lora {
        None => Ok(y),
        Some(d) => Ok((y + d.forward(x)?)?),
    }
}

/// Build a [`LoraDelta`] from raw f16 little-endian bytes (A then referenced by meta).
pub fn delta_from_f16_bytes(
    a_bytes: &[u8],
    a_shape: &[usize],
    b_bytes: &[u8],
    b_shape: &[usize],
    scale: f64,
    device: &Device,
) -> Result<LoraDelta> {
    let a = f16_bytes_to_f32_tensor(a_bytes, a_shape, device)?;
    let b = f16_bytes_to_f32_tensor(b_bytes, b_shape, device)?;
    Ok(LoraDelta { a, b, scale })
}

fn f16_bytes_to_f32_tensor(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let expect = n.saturating_mul(2);
    if bytes.len() < expect {
        return Err(AppError::msg(format!(
            "adapter weights truncated: need {expect} bytes for shape {shape:?}, got {}",
            bytes.len()
        )));
    }
    let mut f32s = Vec::with_capacity(n);
    for i in 0..n {
        let lo = bytes[i * 2] as u16;
        let hi = bytes[i * 2 + 1] as u16;
        let bits = lo | (hi << 8);
        f32s.push(f16_to_f32(bits));
    }
    Ok(Tensor::from_vec(f32s, shape, device)?.to_dtype(DType::F32)?)
}

fn f16_to_f32(bits: u16) -> f32 {
    // IEEE half → f32 (soft float; fine for adapter load).
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mut frac = bits & 0x7fffff;
    if exp == 255 {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    exp -= 127;
    if exp > 15 {
        return sign | 0x7c00; // inf
    }
    if exp < -14 {
        // subnormal / zero
        if exp < -24 {
            return sign;
        }
        frac |= 0x800000;
        let shift = (-14 - exp) as u32 + 13;
        let half = (frac >> shift) as u16;
        return sign | half;
    }
    let half_exp = (exp + 15) as u16;
    let half_frac = (frac >> 13) as u16;
    sign | (half_exp << 10) | half_frac
}
