//! Compile WGSL gemm shader → SPIR-V for the Vulkan compute path.

fn main() {
    println!("cargo:rerun-if-changed=shaders/gemm_f32.wgsl");
    let wgsl = include_str!("shaders/gemm_f32.wgsl");
    let module = naga::front::wgsl::parse_str(wgsl).expect("parse gemm_f32.wgsl");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validate gemm_f32.wgsl");
    let spv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    )
    .expect("emit SPIR-V");
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("gemm_f32.spv");
    let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&out, bytes).expect("write gemm_f32.spv");
}
