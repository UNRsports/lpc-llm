//! Q6_K block layout + CPU reference GEMV (matches candle `BlockQ6K::to_float`).

#![allow(dead_code)] // CPU reference + pack helpers used by unit / GPU tests.

use crate::error::{AppError, Result};

pub const QK_K: usize = 256;
pub const BLOCK_Q6K_SIZE: usize = 210; // 128 + 64 + 16 + 2

/// Mirror of candle / ggml `block_q6_K`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: u16, // f16 bits
}

const _: () = assert!(std::mem::size_of::<BlockQ6K>() == BLOCK_Q6K_SIZE);

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

/// Dequantize one Q6_K block into 256 f32 values (candle-compatible).
pub fn dequant_block(block: &BlockQ6K, y: &mut [f32; QK_K]) {
    let d = f16_bits_to_f32(block.d);
    let ql = &block.ql;
    let qh = &block.qh;
    let sc = &block.scales;
    for n in (0..QK_K).step_by(128) {
        let idx = n / 128;
        let sc = &sc[8 * idx..];
        let ql = &ql[64 * idx..];
        let qh = &qh[32 * idx..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
            y[n + l] = d * sc[is] as f32 * q1 as f32;
            y[n + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
            y[n + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
            y[n + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
        }
    }
}

fn blocks_from_bytes(bytes: &[u8]) -> Result<&[BlockQ6K]> {
    if !bytes.len().is_multiple_of(BLOCK_Q6K_SIZE) {
        return Err(AppError::msg(format!(
            "Q6_K bytes len {} not multiple of {BLOCK_Q6K_SIZE}",
            bytes.len()
        )));
    }
    let n = bytes.len() / BLOCK_Q6K_SIZE;
    let ptr = bytes.as_ptr() as *const BlockQ6K;
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
}

/// `Y[m,n] = X[m,k] @ W[n,k]^T` with W stored as Q6_K row-major `(n,k)`.
pub fn q6k_gemm_cpu(w_bytes: &[u8], n: usize, k: usize, x: &[f32], m: usize) -> Result<Vec<f32>> {
    if k == 0 || !k.is_multiple_of(QK_K) {
        return Err(AppError::msg(format!("Q6_K k={k} must be multiple of {QK_K}")));
    }
    let blocks_per_row = k / QK_K;
    let expect = n * blocks_per_row * BLOCK_Q6K_SIZE;
    if w_bytes.len() != expect {
        return Err(AppError::msg(format!(
            "Q6_K size mismatch: got {} want {expect} (n={n} k={k})",
            w_bytes.len()
        )));
    }
    if x.len() != m * k {
        return Err(AppError::msg(format!(
            "Q6_K x len {} want {}",
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

pub fn q6k_gemv_cpu(w_bytes: &[u8], n: usize, k: usize, x: &[f32]) -> Result<Vec<f32>> {
    q6k_gemm_cpu(w_bytes, n, k, x, 1)
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    if exp == 0 {
        return sign;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Synthetic Q6_K with constant dequant value ≈ `d * scale * (q-32)` for tests.
pub fn pack_constant_q6k(n: usize, k: usize, q: u8, d_f32: f32, scale: i8) -> Vec<u8> {
    assert!(k.is_multiple_of(QK_K));
    let blocks_per_row = k / QK_K;
    let mut out = vec![0u8; n * blocks_per_row * BLOCK_Q6K_SIZE];
    let d_bits = f32_to_f16_bits(d_f32);
    let q6 = q.min(63);
    // Encode so to_float yields d*scale*(q6-32) for every element.
    // Using the ql/qh packing from from_float inverse for a constant.
    for blk in 0..(n * blocks_per_row) {
        let base = blk * BLOCK_Q6K_SIZE;
        let block = &mut out[base..base + BLOCK_Q6K_SIZE];
        // ql / qh for constant q6 across 256 elems — follow from_float packing.
        let mut l = [0i8; QK_K];
        for e in l.iter_mut() {
            *e = q6 as i8;
        }
        for j in (0..QK_K).step_by(128) {
            let ql_off = if j == 0 { 0 } else { 64 };
            let qh_off = if j == 0 { 128 } else { 160 };
            for l_idx in 0..32 {
                let q1 = (l[j + l_idx] & 0xF) as u8;
                let q2 = (l[j + l_idx + 32] & 0xF) as u8;
                let q3 = (l[j + l_idx + 64] & 0xF) as u8;
                let q4 = (l[j + l_idx + 96] & 0xF) as u8;
                block[ql_off + l_idx] = q1 | (q3 << 4);
                block[ql_off + l_idx + 32] = q2 | (q4 << 4);
                block[qh_off + l_idx] = ((l[j + l_idx] >> 4) as u8)
                    | (((l[j + l_idx + 32] >> 4) as u8) << 2)
                    | (((l[j + l_idx + 64] >> 4) as u8) << 4)
                    | (((l[j + l_idx + 96] >> 4) as u8) << 6);
            }
        }
        for s in 0..16 {
            block[192 + s] = scale as u8;
        }
        block[208] = (d_bits & 0xff) as u8;
        block[209] = (d_bits >> 8) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6k_dequant_constant_matches_formula() {
        let n = 2usize;
        let k = 256usize;
        let q = 40u8; // → (40-32)=8
        let d = 0.5f32;
        let scale = 2i8;
        let w = pack_constant_q6k(n, k, q, d, scale);
        let blocks = blocks_from_bytes(&w).unwrap();
        let mut y = [0f32; QK_K];
        dequant_block(&blocks[0], &mut y);
        let expect = d * scale as f32 * (q as i32 - 32) as f32;
        for (i, v) in y.iter().enumerate() {
            assert!((v - expect).abs() < 1e-4, "i={i} got={v} expect={expect}");
        }
    }
}
