//! Compile WGSL compute shaders → SPIR-V for the Vulkan path.

fn compile_wgsl(name: &str, src: &str, out_name: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
        panic!("parse {name}: {e}");
    });
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate {name}: {e}"));
    let spv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    )
    .unwrap_or_else(|e| panic!("emit SPIR-V {name}: {e}"));
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join(out_name);
    let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&out, bytes).expect("write spv");
}

fn main() {
    println!("cargo:rerun-if-changed=shaders/gemm_f32.wgsl");
    println!("cargo:rerun-if-changed=shaders/q4k_gemv.wgsl");
    println!("cargo:rerun-if-changed=shaders/q6k_gemv.wgsl");
    println!("cargo:rerun-if-changed=shaders/q8_0_gemv.wgsl");
    compile_wgsl(
        "gemm_f32.wgsl",
        include_str!("shaders/gemm_f32.wgsl"),
        "gemm_f32.spv",
    );
    compile_wgsl(
        "q4k_gemv.wgsl",
        include_str!("shaders/q4k_gemv.wgsl"),
        "q4k_gemv.spv",
    );
    compile_wgsl(
        "q6k_gemv.wgsl",
        include_str!("shaders/q6k_gemv.wgsl"),
        "q6k_gemv.spv",
    );
    compile_wgsl(
        "q8_0_gemv.wgsl",
        include_str!("shaders/q8_0_gemv.wgsl"),
        "q8_0_gemv.spv",
    );
}
