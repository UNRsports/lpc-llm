// Y[m, n] = X[m, k] @ W[n, k]^T  with W as GGML Q6_K (210-byte blocks, 256 elems).
// Layout: ql[128] | qh[64] | scales_i8[16] | d_f16
// Bindings: 0=x f32, 1=w u32-packed bytes, 2=y f32, 3=dims {n,k,m,pad}

struct Dims {
    n: u32,
    k: u32,
    m: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const QK_K: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

fn load_u8(byte_off: u32) -> u32 {
    let word = w[byte_off / 4u];
    let shift = (byte_off % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

fn load_u16(byte_off: u32) -> u32 {
    return load_u8(byte_off) | (load_u8(byte_off + 1u) << 8u);
}

fn load_i8(byte_off: u32) -> i32 {
    let u = load_u8(byte_off);
    if ((u & 0x80u) != 0u) {
        return i32(u) - 256;
    }
    return i32(u);
}

fn f16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 0x1fu;
    let frac = h & 0x3ffu;
    var bits: u32;
    if (exp == 0u) {
        if (frac == 0u) {
            bits = sign << 31u;
        } else {
            var e: i32 = 127 - 15 + 1;
            var m: u32 = frac;
            loop {
                if ((m & 0x400u) != 0u) {
                    break;
                }
                m = m << 1u;
                e = e - 1;
            }
            m = m & 0x3ffu;
            bits = (sign << 31u) | (u32(e) << 23u) | (m << 13u);
        }
    } else if (exp == 31u) {
        bits = (sign << 31u) | (0xffu << 23u) | (frac << 13u);
    } else {
        bits = (sign << 31u) | ((exp + 127u - 15u) << 23u) | (frac << 13u);
    }
    return bitcast<f32>(bits);
}

// Candle / llama.cpp dequantize_row_q6k for one 256-elem block, dotted with x.
fn block_dot(block_byte: u32, x_base: u32) -> f32 {
    let d = f16_to_f32(load_u16(block_byte + 208u));
    let ql_base = block_byte;
    let qh_base = block_byte + 128u;
    let sc_base = block_byte + 192u;
    var acc: f32 = 0.0;
    var n: u32 = 0u;
    loop {
        if (n >= QK_K) {
            break;
        }
        let idx = n / 128u;
        let ql = ql_base + 64u * idx;
        let qh = qh_base + 32u * idx;
        let sc = sc_base + 8u * idx;
        var l: u32 = 0u;
        loop {
            if (l >= 32u) {
                break;
            }
            let is_ = l / 16u;
            let q4l = load_u8(ql + l);
            let q4l32 = load_u8(ql + l + 32u);
            let qhb = load_u8(qh + l);
            let q1 = i32((q4l & 0xFu) | ((qhb & 3u) << 4u)) - 32;
            let q2 = i32((q4l32 & 0xFu) | (((qhb >> 2u) & 3u) << 4u)) - 32;
            let q3 = i32((q4l >> 4u) | (((qhb >> 4u) & 3u) << 4u)) - 32;
            let q4 = i32((q4l32 >> 4u) | (((qhb >> 6u) & 3u) << 4u)) - 32;
            let s0 = f32(load_i8(sc + is_));
            let s2 = f32(load_i8(sc + is_ + 2u));
            let s4 = f32(load_i8(sc + is_ + 4u));
            let s6 = f32(load_i8(sc + is_ + 6u));
            let base = x_base + n;
            acc = acc + d * s0 * f32(q1) * x[base + l];
            acc = acc + d * s2 * f32(q2) * x[base + l + 32u];
            acc = acc + d * s4 * f32(q3) * x[base + l + 64u];
            acc = acc + d * s6 * f32(q4) * x[base + l + 96u];
            l = l + 1u;
        }
        n = n + 128u;
    }
    return acc;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.m * dims.n;
    if (idx >= total) {
        return;
    }
    let batch = idx / dims.n;
    let row = idx % dims.n;
    let blocks_per_row = dims.k / QK_K;
    let x_base = batch * dims.k;
    var acc: f32 = 0.0;
    var bi: u32 = 0u;
    loop {
        if (bi >= blocks_per_row) {
            break;
        }
        let block_index = row * blocks_per_row + bi;
        let block_byte = block_index * BLOCK_BYTES;
        acc = acc + block_dot(block_byte, x_base + bi * QK_K);
        bi = bi + 1u;
    }
    y[idx] = acc;
}
