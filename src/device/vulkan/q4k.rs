//! Q4_K block layout + CPU reference GEMV (matches candle `BlockQ4K::to_float`).

use crate::error::{AppError, Result};

pub const QK_K: usize = 256;
pub const BLOCK_Q4K_SIZE: usize = 144; // 2+2+12+128

/// Mirror of candle `k_quants::BlockQ4K` (GGML Q4_K).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4K {
    pub d: u16,       // f16 bits
    pub dmin: u16,    // f16 bits
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == BLOCK_Q4K_SIZE);

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

fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one Q4_K block into 256 f32 values (candle-compatible).
pub fn dequant_block(block: &BlockQ4K, y: &mut [f32; QK_K]) {
    let d = f16_bits_to_f32(block.d);
    let min = f16_bits_to_f32(block.dmin);
    let q = &block.qs;
    let mut is = 0usize;
    let mut ys_index = 0usize;
    for j in (0..QK_K).step_by(64) {
        let qchunk = &q[j / 2..j / 2 + 32];
        let (sc, m) = get_scale_min_k4(is, &block.scales);
        let d1 = d * sc as f32;
        let m1 = min * m as f32;
        let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
        let d2 = d * sc as f32;
        let m2 = min * m as f32;
        for &byte in qchunk {
            y[ys_index] = d1 * (byte & 0xF) as f32 - m1;
            ys_index += 1;
        }
        for &byte in qchunk {
            y[ys_index] = d2 * (byte >> 4) as f32 - m2;
            ys_index += 1;
        }
        is += 2;
    }
}

fn blocks_from_bytes(bytes: &[u8]) -> Result<&[BlockQ4K]> {
    if !bytes.len().is_multiple_of(BLOCK_Q4K_SIZE) {
        return Err(AppError::msg(format!(
            "Q4_K bytes len {} not multiple of {BLOCK_Q4K_SIZE}",
            bytes.len()
        )));
    }
    let n = bytes.len() / BLOCK_Q4K_SIZE;
    let ptr = bytes.as_ptr() as *const BlockQ4K;
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
}

/// `Y[m,n] = X[m,k] @ W[n,k]^T` with W stored as Q4_K row-major `(n,k)`.
pub fn q4k_gemm_cpu(w_bytes: &[u8], n: usize, k: usize, x: &[f32], m: usize) -> Result<Vec<f32>> {
    if k == 0 || !k.is_multiple_of(QK_K) {
        return Err(AppError::msg(format!("Q4_K k={k} must be multiple of {QK_K}")));
    }
    let blocks_per_row = k / QK_K;
    let expect = n * blocks_per_row * BLOCK_Q4K_SIZE;
    if w_bytes.len() != expect {
        return Err(AppError::msg(format!(
            "Q4_K size mismatch: got {} want {expect} (n={n} k={k})",
            w_bytes.len()
        )));
    }
    if x.len() != m * k {
        return Err(AppError::msg(format!(
            "Q4_K x len {} want {}",
            x.len(),
            m * k
        )));
    }
    let blocks = blocks_from_bytes(w_bytes)?;
    let mut out = vec![0f32; m * n];
    let mut row_f = [0f32; QK_K];
    for row in 0..n {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        for bi in 0..blocks_per_row {
            dequant_block(&row_blocks[bi], &mut row_f);
            let k0 = bi * QK_K;
            for batch in 0..m {
                let mut acc = 0f32;
                let xrow = &x[batch * k + k0..batch * k + k0 + QK_K];
                for t in 0..QK_K {
                    acc += row_f[t] * xrow[t];
                }
                out[batch * n + row] += acc;
            }
        }
    }
    Ok(out)
}

/// Convenience: single-batch GEMV `y[n] = x[k] @ W[n,k]^T`.
#[allow(dead_code)]
pub fn q4k_gemv_cpu(w_bytes: &[u8], n: usize, k: usize, x: &[f32]) -> Result<Vec<f32>> {
    q4k_gemm_cpu(w_bytes, n, k, x, 1)
}

/// Pack a synthetic Q4_K weight with constant nibble value for tests.
#[cfg(test)]
pub fn pack_constant_q4k(n: usize, k: usize, nibble: u8, d_f32: f32, dmin_f32: f32) -> Vec<u8> {
    assert!(k.is_multiple_of(QK_K));
    let blocks_per_row = k / QK_K;
    let mut out = vec![0u8; n * blocks_per_row * BLOCK_Q4K_SIZE];
    let d_bits = f32_to_f16_bits(d_f32);
    let dmin_bits = f32_to_f16_bits(dmin_f32);
    let nib = nibble & 0xF;
    let qs_byte = nib | (nib << 4);
    // sc=1, m=0 for all 8 groups (candle from_float packing).
    let mut scales = [0u8; 12];
    for j in 0..8 {
        let ls = 1u8;
        let lm = 0u8;
        if j < 4 {
            scales[j] = ls;
            scales[j + 4] = lm;
        } else {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= (ls >> 4) << 6;
            scales[j] |= (lm >> 4) << 6;
        }
    }
    for b in 0..(n * blocks_per_row) {
        let off = b * BLOCK_Q4K_SIZE;
        out[off] = (d_bits & 0xff) as u8;
        out[off + 1] = (d_bits >> 8) as u8;
        out[off + 2] = (dmin_bits & 0xff) as u8;
        out[off + 3] = (dmin_bits >> 8) as u8;
        out[off + 4..off + 16].copy_from_slice(&scales);
        for i in 0..128 {
            out[off + 16 + i] = qs_byte;
        }
    }
    out
}

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn gemv_constant_matches_manual() {
        let n = 2;
        let k = 256;
        let w = pack_constant_q4k(n, k, 3, 0.5, 0.0);
        // sc=1, m=0, d=0.5, nibble=3 → value = 0.5*1*3 - 0 = 1.5
        let x = vec![1.0f32; k];
        let y = q4k_gemv_cpu(&w, n, k, &x).unwrap();
        assert_eq!(y.len(), 2);
        // sum of 256 * 1.5 = 384
        assert!((y[0] - 384.0).abs() < 1e-2, "y0={}", y[0]);
        assert!((y[1] - 384.0).abs() < 1e-2, "y1={}", y[1]);
    }

    #[test]
    fn gemm_batch() {
        let n = 1;
        let k = 256;
        let m = 2;
        let w = pack_constant_q4k(n, k, 1, 1.0, 0.0);
        let mut x = vec![0f32; m * k];
        for i in 0..k {
            x[i] = 1.0;
            x[k + i] = 2.0;
        }
        let y = q4k_gemm_cpu(&w, n, k, &x, m).unwrap();
        // value=1, sum = 256 and 512
        assert!((y[0] - 256.0).abs() < 1e-2);
        assert!((y[1] - 512.0).abs() < 1e-2);
    }
}
