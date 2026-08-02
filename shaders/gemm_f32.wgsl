// C[m, n] = A[m, k] * B[k, n]  (row-major f32)
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    var acc: f32 = 0.0;
    var t: u32 = 0u;
    loop {
        if (t >= dims.k) {
            break;
        }
        acc = acc + a[row * dims.k + t] * b[t * dims.n + col];
        t = t + 1u;
    }
    c[row * dims.n + col] = acc;
}
