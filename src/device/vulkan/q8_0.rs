//! Q8_0 block layout + CPU reference GEMV (matches candle `BlockQ8_0::to_float`).

#![allow(dead_code)]

use crate::error::{AppError, Result};

pub const QK8_0: usize = 32;
pub const BLOCK_Q8_0_SIZE: usize = 34;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    pub d: u16, // f16 bits
    pub qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == BLOCK_Q8_0_SIZE);

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
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

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;
    if exp == 255 {
        return (sign | 0x7c00 | (if frac != 0 { 1 } else { 0 })) as u16;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return (sign | 0x7c00) as u16;
    }
    if e <= 0 {
        if e < -10 {
            return sign as u16;
        }
        let m = (frac | 0x800000) >> (1 - e);
        return (sign | (m >> 13)) as u16;
    }
    (sign | ((e as u32) << 10) | (frac >> 13)) as u16
}

pub fn dequant_block(block: &BlockQ8_0, y: &mut [f32; QK8_0]) {
    let d = f16_bits_to_f32(block.d);
    for j in 0..QK8_0 {
        y[j] = block.qs[j] as f32 * d;
    }
}

fn blocks_from_bytes(bytes: &[u8]) -> Result<&[BlockQ8_0]> {
    if !bytes.len().is_multiple_of(BLOCK_Q8_0_SIZE) {
        return Err(AppError::msg(format!(
            "Q8_0 bytes len {} not multiple of {BLOCK_Q8_0_SIZE}",
            bytes.len()
        )));
    }
    let n = bytes.len() / BLOCK_Q8_0_SIZE;
    let ptr = bytes.as_ptr() as *const BlockQ8_0;
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
}

/// Synthetic Q8_0 with constant dequant value ≈ `q as f32 * d` for tests / microbench.
pub fn pack_constant_q8_0(n: usize, k: usize, q: i8, d_f32: f32) -> Vec<u8> {
    assert!(k.is_multiple_of(QK8_0));
    let blocks_per_row = k / QK8_0;
    let mut out = vec![0u8; n * blocks_per_row * BLOCK_Q8_0_SIZE];
    let d_bits = f32_to_f16_bits(d_f32);
    for blk in 0..(n * blocks_per_row) {
        let base = blk * BLOCK_Q8_0_SIZE;
        out[base] = (d_bits & 0xff) as u8;
        out[base + 1] = (d_bits >> 8) as u8;
        for j in 0..QK8_0 {
            out[base + 2 + j] = q as u8;
        }
    }
    out
}

/// CPU reference: Y[m,n] = X[m,k] @ W[n,k]^T with W as Q8_0.
pub fn q8_0_gemv_cpu(w_bytes: &[u8], n: usize, k: usize, x: &[f32]) -> Result<Vec<f32>> {
    if x.len() != k {
        return Err(AppError::msg(format!(
            "q8_0_gemv_cpu: x len {} != k={k}",
            x.len()
        )));
    }
    if !k.is_multiple_of(QK8_0) {
        return Err(AppError::msg(format!(
            "q8_0_gemv_cpu: k={k} not multiple of {QK8_0}"
        )));
    }
    let blocks = blocks_from_bytes(w_bytes)?;
    let blocks_per_row = k / QK8_0;
    if blocks.len() != n * blocks_per_row {
        return Err(AppError::msg(format!(
            "q8_0_gemv_cpu: blocks {} != n*bpr={}",
            blocks.len(),
            n * blocks_per_row
        )));
    }
    let mut y = vec![0f32; n];
    let mut tmp = [0f32; QK8_0];
    for row in 0..n {
        let mut acc = 0f32;
        for bi in 0..blocks_per_row {
            dequant_block(&blocks[row * blocks_per_row + bi], &mut tmp);
            let x_off = bi * QK8_0;
            for j in 0..QK8_0 {
                acc += tmp[j] * x[x_off + j];
            }
        }
        y[row] = acc;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), BLOCK_Q8_0_SIZE);
    }

    #[test]
    fn q8_0_dequant_constant_matches_formula() {
        let n = 2usize;
        let k = 64usize;
        let q = 3i8;
        let d = 0.5f32;
        let w = pack_constant_q8_0(n, k, q, d);
        let blocks = blocks_from_bytes(&w).unwrap();
        let mut y = [0f32; QK8_0];
        dequant_block(&blocks[0], &mut y);
        let expect = q as f32 * d;
        for (i, v) in y.iter().enumerate() {
            assert!((v - expect).abs() < 1e-4, "i={i} got={v} expect={expect}");
        }
    }

    #[test]
    fn gemv_constant_matches_manual() {
        let n = 2usize;
        let k = 64usize;
        let q = 2i8;
        let d = 0.25f32;
        let w = pack_constant_q8_0(n, k, q, d);
        let x = vec![1.0f32; k];
        let y = q8_0_gemv_cpu(&w, n, k, &x).unwrap();
        // each elem = q*d; row = k * q * d
        let expect = k as f32 * q as f32 * d;
        assert_eq!(y.len(), 2);
        for (i, v) in y.iter().enumerate() {
            assert!((v - expect).abs() < 1e-3, "i={i} got={v} expect={expect}");
        }
    }
}
