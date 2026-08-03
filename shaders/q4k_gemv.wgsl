// Y[m, n] = X[m, k] @ W[n, k]^T  with W as GGML Q4_K (144-byte blocks, 256 elems).
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
const BLOCK_U32: u32 = 36u; // 144 / 4

fn load_u8(byte_off: u32) -> u32 {
    let word = w[byte_off / 4u];
    let shift = (byte_off % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

fn load_u16(byte_off: u32) -> u32 {
    return load_u8(byte_off) | (load_u8(byte_off + 1u) << 8u);
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

fn get_scale_min_k4(j: u32, scale_base: u32) -> vec2<u32> {
    // scale_base = byte offset of scales[0] within weight buffer
    if (j < 4u) {
        let d = load_u8(scale_base + j) & 63u;
        let m = load_u8(scale_base + j + 4u) & 63u;
        return vec2(d, m);
    }
    let d = (load_u8(scale_base + j + 4u) & 0xFu) | ((load_u8(scale_base + j - 4u) >> 6u) << 4u);
    let m = (load_u8(scale_base + j + 4u) >> 4u) | ((load_u8(scale_base + j) >> 6u) << 4u);
    return vec2(d, m);
}

fn block_dot(block_byte: u32, x_base: u32) -> f32 {
    let d = f16_to_f32(load_u16(block_byte));
    let dmin = f16_to_f32(load_u16(block_byte + 2u));
    let scale_base = block_byte + 4u;
    let qs_base = block_byte + 16u;
    var acc: f32 = 0.0;
    var is_: u32 = 0u;
    var ys_index: u32 = 0u;
    var j: u32 = 0u;
    loop {
        if (j >= QK_K) {
            break;
        }
        let q_off = qs_base + j / 2u;
        let sm1 = get_scale_min_k4(is_, scale_base);
        let d1 = d * f32(sm1.x);
        let m1 = dmin * f32(sm1.y);
        let sm2 = get_scale_min_k4(is_ + 1u, scale_base);
        let d2 = d * f32(sm2.x);
        let m2 = dmin * f32(sm2.y);
        var t: u32 = 0u;
        loop {
            if (t >= 32u) {
                break;
            }
            let byte = load_u8(q_off + t);
            let xv = x[x_base + ys_index];
            acc = acc + (d1 * f32(byte & 0xFu) - m1) * xv;
            ys_index = ys_index + 1u;
            t = t + 1u;
        }
        t = 0u;
        loop {
            if (t >= 32u) {
                break;
            }
            let byte = load_u8(q_off + t);
            let xv = x[x_base + ys_index];
            acc = acc + (d2 * f32(byte >> 4u) - m2) * xv;
            ys_index = ys_index + 1u;
            t = t + 1u;
        }
        is_ = is_ + 2u;
        j = j + 64u;
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
        let block_byte = block_index * 144u;
        acc = acc + block_dot(block_byte, x_base + bi * QK_K);
        bi = bi + 1u;
    }
    y[idx] = acc;
}
