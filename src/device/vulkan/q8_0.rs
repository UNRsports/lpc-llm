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
