// Softmax over the last dimension (Candle-compatible f32):
//   m = max(x[row, :])
//   e[i] = exp(x[row, i] - m)
//   y[i] = e[i] / sum(e)
// Bindings: 0=x f32, 1=y f32, 2=unused, 3=dims {rows, cols, 0, 0}
// One workgroup per row; workgroup_size 64 reduces then writes.

struct Dims {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> _unused: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let lane = lid.x;
    if (row >= dims.rows) {
        return;
    }
    let base = row * dims.cols;

    // Pass 1: max
    var local_max: f32 = -3.402823e+38;
    var i: u32 = lane;
    loop {
        if (i >= dims.cols) {
            break;
        }
        local_max = max(local_max, x[base + i]);
        i = i + 64u;
    }
    scratch[lane] = local_max;
    workgroupBarrier();
    if (lane < 32u) {
        scratch[lane] = max(scratch[lane], scratch[lane + 32u]);
    }
    workgroupBarrier();
    if (lane < 16u) {
        scratch[lane] = max(scratch[lane], scratch[lane + 16u]);
    }
    workgroupBarrier();
    if (lane < 8u) {
        scratch[lane] = max(scratch[lane], scratch[lane + 8u]);
    }
    workgroupBarrier();
    if (lane < 4u) {
        scratch[lane] = max(scratch[lane], scratch[lane + 4u]);
    }
    workgroupBarrier();
    if (lane < 2u) {
        scratch[lane] = max(scratch[lane], scratch[lane + 2u]);
    }
    workgroupBarrier();
    if (lane == 0u) {
        scratch[0] = max(scratch[0], scratch[1]);
    }
    workgroupBarrier();
    let row_max = scratch[0];

    // Pass 2: sum(exp(x - max))
    var local_sum: f32 = 0.0;
    i = lane;
    loop {
        if (i >= dims.cols) {
            break;
        }
        local_sum = local_sum + exp(x[base + i] - row_max);
        i = i + 64u;
    }
    scratch[lane] = local_sum;
    workgroupBarrier();
    if (lane < 32u) {
        scratch[lane] = scratch[lane] + scratch[lane + 32u];
    }
    workgroupBarrier();
    if (lane < 16u) {
        scratch[lane] = scratch[lane] + scratch[lane + 16u];
    }
    workgroupBarrier();
    if (lane < 8u) {
        scratch[lane] = scratch[lane] + scratch[lane + 8u];
    }
    workgroupBarrier();
    if (lane < 4u) {
        scratch[lane] = scratch[lane] + scratch[lane + 4u];
    }
    workgroupBarrier();
    if (lane < 2u) {
        scratch[lane] = scratch[lane] + scratch[lane + 2u];
    }
    workgroupBarrier();
    if (lane == 0u) {
        scratch[0] = scratch[0] + scratch[1];
    }
    workgroupBarrier();
    let row_sum = scratch[0];
    let inv = select(0.0, 1.0 / row_sum, row_sum > 0.0);

    // Pass 3: write softmax
    i = lane;
    loop {
        if (i >= dims.cols) {
            break;
        }
        y[base + i] = exp(x[base + i] - row_max) * inv;
        i = i + 64u;
    }
}
