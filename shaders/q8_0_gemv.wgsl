// Y[m, n] = X[m, k] @ W[n, k]^T  with W as GGML Q8_0 (34-byte blocks, 32 elems).
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

const QK8_0: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

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
    if (u >= 128u) {
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

fn block_dot(block_byte: u32, x_base: u32) -> f32 {
    let d = f16_to_f32(load_u16(block_byte));
    var acc: f32 = 0.0;
    var j: u32 = 0u;
    loop {
        if (j >= QK8_0) {
            break;
        }
        let q = load_i8(block_byte + 2u + j);
        acc = acc + d * f32(q) * x[x_base + j];
        j = j + 1u;
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
    let blocks_per_row = dims.k / QK8_0;
    let x_base = batch * dims.k;
    var acc: f32 = 0.0;
    var bi: u32 = 0u;
    loop {
        if (bi >= blocks_per_row) {
            break;
        }
        let block_index = row * blocks_per_row + bi;
        let block_byte = block_index * BLOCK_BYTES;
        acc = acc + block_dot(block_byte, x_base + bi * QK8_0);
        bi = bi + 1u;
    }
    y[idx] = acc;
}
