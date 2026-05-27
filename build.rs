use std::{fs, path::Path, process::Command};

fn main() {
    // Metal backend on macOS compiles MSL at runtime; no offline shader step needed.
    if cfg!(target_os = "macos") {
        return;
    }

    let shader_src = Path::new("shaders");
    let out_root   = std::env::var("OUT_DIR").unwrap();
    let spv_dir    = format!("{out_root}/shaders");
    fs::create_dir_all(&spv_dir).unwrap();

    let shaders = [
        ("raygen.rgen.glsl",       "raygen.rgen.spv"),
        ("miss.rmiss.glsl",        "miss.rmiss.spv"),
        ("closest_hit.rchit.glsl", "closest_hit.rchit.spv"),
        ("intersection.rint.glsl", "intersection.rint.spv"),
    ];

    let glslc_ok = Command::new("glslc").arg("--version").output().is_ok();
    if !glslc_ok {
        println!(
            "cargo:warning=glslc not found — emitting stub SPIR-V. \
             Install the Vulkan SDK (lunarg.com) for real shaders."
        );
    }

    for (src, spv) in &shaders {
        let src_path = shader_src.join(src);
        let spv_path = format!("{spv_dir}/{spv}");
        println!("cargo:rerun-if-changed={}", src_path.display());

        if glslc_ok {
            let status = Command::new("glslc")
                .args(["--target-env=vulkan1.3", src_path.to_str().unwrap(), "-o", &spv_path])
                .status()
                .expect("glslc failed");
            assert!(status.success(), "Shader compilation failed: {src}");
        } else {
            write_stub_spv(&spv_path);
        }
    }
}

fn write_stub_spv(path: &str) {
    #[rustfmt::skip]
    let header: [u8; 20] = [
        0x03, 0x02, 0x23, 0x07, // magic
        0x00, 0x03, 0x01, 0x00, // version 1.3
        0x00, 0x00, 0x00, 0x00, // generator
        0x01, 0x00, 0x00, 0x00, // id bound
        0x00, 0x00, 0x00, 0x00, // schema
    ];
    fs::write(path, header).unwrap();
}
