//! Tiny Llama-family transformer for from-scratch training (Candle).

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear_no_bias, ops, rms_norm, Embedding, Linear, RmsNorm, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};

/// Hyper-parameters for a memory-friendly Llama-style model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_ff: usize,
    pub max_seq: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub tie_embeddings: bool,
}

impl Default for TinyConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            n_embd: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            n_ff: 512,
            max_seq: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            tie_embeddings: true,
        }
    }
}

impl TinyConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_heads
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_embd == 0 || self.n_layers == 0 || self.n_heads == 0 {
            return Err(AppError::msg("n_embd, n_layers, n_heads must be > 0"));
        }
        if !self.n_embd.is_multiple_of(self.n_heads) {
            return Err(AppError::msg("n_embd must be divisible by n_heads"));
        }
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(AppError::msg(
                "n_kv_heads must divide n_heads and be > 0",
            ));
        }
        if self.vocab_size < 4 {
            return Err(AppError::msg("vocab_size must be >= 4"));
        }
        if self.n_ff == 0 {
            return Err(AppError::msg("n_ff must be > 0"));
        }
        Ok(())
    }

    /// Rough parameter count (embeddings + layers + optional untied lm_head).
    pub fn param_count(&self) -> usize {
        let emb = self.vocab_size * self.n_embd;
        let head_dim = self.head_dim();
        let q = self.n_embd * self.n_heads * head_dim;
        let kv = self.n_embd * self.n_kv_heads * head_dim;
        let o = self.n_heads * head_dim * self.n_embd;
        let ffn = self.n_embd * self.n_ff * 2 + self.n_ff * self.n_embd;
        let norms = self.n_embd * (2 * self.n_layers + 1);
        let layer = q + kv * 2 + o + ffn + norms / (self.n_layers + 1).max(1) * 2;
        let layers = (q + kv * 2 + o + ffn + self.n_embd * 2) * self.n_layers;
        let head = if self.tie_embeddings {
            0
        } else {
            self.vocab_size * self.n_embd
        };
        let _ = (layer, norms);
        emb + layers + head + self.n_embd
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gated = ops::silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gated * up)?)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;

        let q = q
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let k = repeat_kv(k, self.n_heads / self.n_kv_heads)?;
        let v = repeat_kv(v, self.n_heads / self.n_kv_heads)?;

        let scale = (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? / scale)?;
        let mask = causal_mask(t, x.device())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, self.n_heads * self.head_dim))?;
        self.o.forward(&y)
    }
}

struct Block {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?, cos, sin)?;
        let x = (x + h)?;
        let h = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
        x + h
    }

    /// Grad-checkpoint style: same math; caller may drop intermediates between layers.
    fn forward_recompute(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        self.forward(x, cos, sin)
    }
}

/// Trainable tiny Llama-family model.
pub struct TinyModel {
    pub cfg: TinyConfig,
    pub varmap: VarMap,
    embed: Embedding,
    blocks: Vec<Block>,
    output_norm: RmsNorm,
    lm_head: Option<Linear>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    device: Device,
}

impl TinyModel {
    pub fn new(cfg: TinyConfig, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Self::from_vb(cfg, varmap, vb, device)
    }

    pub(super) fn from_vb(
        cfg: TinyConfig,
        varmap: VarMap,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let embed = embedding(cfg.vocab_size, cfg.n_embd, vb.pp("token_embd"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let bvb = vb.pp(&format!("blk.{i}"));
            let attn = Attention {
                q: linear_no_bias(cfg.n_embd, cfg.n_heads * head_dim, bvb.pp("attn_q"))?,
                k: linear_no_bias(cfg.n_embd, cfg.n_kv_heads * head_dim, bvb.pp("attn_k"))?,
                v: linear_no_bias(cfg.n_embd, cfg.n_kv_heads * head_dim, bvb.pp("attn_v"))?,
                o: linear_no_bias(cfg.n_heads * head_dim, cfg.n_embd, bvb.pp("attn_output"))?,
                n_heads: cfg.n_heads,
                n_kv_heads: cfg.n_kv_heads,
                head_dim,
            };
            let mlp = Mlp {
                gate: linear_no_bias(cfg.n_embd, cfg.n_ff, bvb.pp("ffn_gate"))?,
                up: linear_no_bias(cfg.n_embd, cfg.n_ff, bvb.pp("ffn_up"))?,
                down: linear_no_bias(cfg.n_ff, cfg.n_embd, bvb.pp("ffn_down"))?,
            };
            blocks.push(Block {
                attn_norm: rms_norm(cfg.n_embd, cfg.rms_norm_eps, bvb.pp("attn_norm"))?,
                attn,
                ffn_norm: rms_norm(cfg.n_embd, cfg.rms_norm_eps, bvb.pp("ffn_norm"))?,
                mlp,
            });
        }
        let output_norm = rms_norm(cfg.n_embd, cfg.rms_norm_eps, vb.pp("output_norm"))?;
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(linear_no_bias(
                cfg.n_embd,
                cfg.vocab_size,
                vb.pp("output"),
            )?)
        };
        let (rope_cos, rope_sin) =
            precompute_rope(cfg.max_seq, head_dim, cfg.rope_theta, device)?;
        Ok(Self {
            cfg,
            varmap,
            embed,
            blocks,
            output_norm,
            lm_head,
            rope_cos,
            rope_sin,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn all_vars(&self) -> Vec<candle_core::Var> {
        self.varmap.all_vars()
    }

    /// Forward tokens `[T]` or `[B,T]` → logits `[B, T, V]`.
    pub fn forward(&self, tokens: &Tensor, grad_checkpoint: bool) -> Result<Tensor> {
        let tokens = if tokens.rank() == 1 {
            tokens.unsqueeze(0)?
        } else {
            tokens.clone()
        };
        let (_b, t) = tokens.dims2()?;
        if t > self.cfg.max_seq {
            return Err(AppError::msg(format!(
                "sequence length {t} exceeds max_seq {}",
                self.cfg.max_seq
            )));
        }
        let cos = self.rope_cos.i(..t)?;
        let sin = self.rope_sin.i(..t)?;
        let mut x = self.embed.forward(&tokens)?;
        for block in &self.blocks {
            x = if grad_checkpoint {
                block.forward_recompute(&x, &cos, &sin)?
            } else {
                block.forward(&x, &cos, &sin)?
            };
        }
        let x = self.output_norm.forward(&x)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&x)?,
            None => {
                // embeddings: [V, C] → weight as [C, V] for x[B,T,C] @ W
                let w = self.embed.embeddings().t()?;
                let (b, t, c) = x.dims3()?;
                let flat = x.reshape((b * t, c))?;
                flat.matmul(&w)?.reshape((b, t, self.cfg.vocab_size))?
            }
        };
        Ok(logits)
    }

    /// Collect named F32 tensors for checkpoint / GGUF export (GGUF naming).
    pub fn named_tensors(&self) -> Result<HashMap<String, Tensor>> {
        let data = self
            .varmap
            .data()
            .lock()
            .map_err(|_| AppError::msg("VarMap lock poisoned"))?;
        let mut out = HashMap::new();
        for (name, var) in data.iter() {
            let gguf_name = varmap_to_gguf_name(name);
            out.insert(gguf_name, var.as_tensor().clone());
        }
        Ok(out)
    }
}

fn varmap_to_gguf_name(name: &str) -> String {
    // candle_nn paths: token_embd.weight, blk.0.attn_q.weight, output_norm.weight, output.weight
    if name.ends_with(".weight") {
        name.to_string()
    } else {
        format!("{name}.weight")
    }
}

fn precompute_rope(
    max_seq: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2 * i) as f32 / head_dim as f32;
        freqs.push(1.0 / theta.powf(exp));
    }
    let freqs = Tensor::new(freqs.as_slice(), device)?.to_dtype(DType::F32)?;
    let t = Tensor::arange(0u32, max_seq as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq, 1))?;
    let freqs = t.broadcast_matmul(&freqs.reshape((1, half))?)?; // [T, half]
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    // Expand to [T, head_dim] with interleaved pairs duplicated for Neox-style rotate.
    let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
    let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
    Ok((cos, sin))
}

fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
    let (_b, _h, _t, d) = x.dims4()?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    // x: [B, H, T, D], cos/sin: [T, D]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1,1,T,D]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let x_embed = (x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?)?;
    Ok(x_embed)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, h, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))
}

fn causal_mask(t: usize, device: &Device) -> CandleResult<Tensor> {
    let mask: Vec<u8> = (0..t)
        .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_vec(mask, (t, t), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> CandleResult<Tensor> {
    let shape = on_false.shape();
    let mask = mask.broadcast_as(shape)?;
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape)?;
    mask.where_cond(&on_true, on_false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let device = Device::Cpu;
        let cfg = TinyConfig {
            vocab_size: 32,
            n_embd: 32,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            n_ff: 64,
            max_seq: 16,
            ..TinyConfig::default()
        };
        let model = TinyModel::new(cfg, &device).unwrap();
        let tokens = Tensor::new(&[1u32, 2, 3, 4], &device).unwrap();
        let logits = model.forward(&tokens, false).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 32]);
    }
}
