//! Metal ray-tracing backend (macOS, Apple Silicon / AMD / Intel with Metal 3+).
//!
//! Architecture:
//!  - One BLAS (primitive acceleration structure) from a triangulated unit cube [0,1]³
//!  - One TLAS (instance acceleration structure) with 4 instances, each translated
//!  - A compute kernel writes ray-traced output to an RGBA8Unorm texture
//!  - A render pass (fullscreen triangle) copies that texture to the BGRA8Unorm drawable

use std::sync::Arc;

use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use core_graphics_types::geometry::CGSize;
use glam::{Mat4, Vec3};
use metal::{
    foreign_types::ForeignType,
    *,
};
use winit::{
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};

use crate::voxel::demo_voxels;
use super::RenderBackend;

// ── MSL shaders ───────────────────────────────────────────────────────────────

const MSL_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct CameraData {
    float4x4 view_inverse;
    float4x4 proj_inverse;
};

constant float3 FACE_NORMALS[6] = {
    float3( 0,  0, -1), float3( 0, 0, 1),
    float3(-1,  0,  0), float3( 1, 0, 0),
    float3( 0, -1,  0), float3( 0, 1, 0),
};

constant float3 PALETTE[4] = {
    float3(0.88, 0.22, 0.20), float3(0.22, 0.78, 0.28),
    float3(0.22, 0.32, 0.90), float3(0.92, 0.80, 0.20),
};

kernel void raytrace_voxels(
    texture2d<float, access::write>  output [[texture(0)]],
    instance_acceleration_structure  tlas   [[buffer(0)]],
    constant CameraData&             camera [[buffer(1)]],
    uint2 tid  [[thread_position_in_grid]],
    uint2 size [[threads_per_grid]]
) {
    if (tid.x >= size.x || tid.y >= size.y) return;

    float2 uv  = (float2(tid) + 0.5) / float2(size);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4 origin    = camera.view_inverse * float4(0, 0, 0, 1);
    float4 target    = camera.proj_inverse * float4(ndc, 1, 1);
    target          /= target.w;
    float4 direction = camera.view_inverse * float4(normalize(target.xyz), 0);

    ray r;
    r.origin       = origin.xyz;
    r.direction    = normalize(direction.xyz);
    r.min_distance = 0.001;
    r.max_distance = 10000.0;

    intersector<instancing, triangle_data> isect;
    isect.force_opacity(forced_opacity::opaque);
    intersection_result<instancing, triangle_data> result = isect.intersect(r, tlas);

    float3 color;
    if (result.type == intersection_type::none) {
        float t = clamp(0.5 * (r.direction.y + 1.0), 0.0, 1.0);
        color = mix(float3(0.95, 0.95, 1.0), float3(0.25, 0.45, 0.85), t);
    } else {
        uint   face  = min(result.primitive_id / 2u, 5u);
        float3 n     = FACE_NORMALS[face];
        if (dot(n, r.direction) > 0.0) n = -n;
        float3 light = normalize(float3(1.5, 3.0, 2.0));
        float  diff  = max(dot(n, light), 0.0);
        float3 base  = PALETTE[result.instance_id & 3u];
        color = base * (0.20 + 0.80 * diff);
    }
    output.write(float4(color, 1.0), tid);
}

// ── Fullscreen blit: copies compute output (RGBA8) → drawable (BGRA8) ─────────

vertex float4 blit_vert(uint vid [[vertex_id]]) {
    const float2 pos[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
    return float4(pos[vid], 0.0, 1.0);
}

fragment float4 blit_frag(
    float4 position [[position]],
    texture2d<float, access::read> src [[texture(0)]]
) {
    return src.read(uint2(position.xy));
}
"#;

// ── Camera UBO ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraData {
    view_inverse: [[f32; 4]; 4],
    proj_inverse: [[f32; 4]; 4],
}

// ── Unit-cube mesh ────────────────────────────────────────────────────────────
// 8 vertices, 12 triangles (2 per face).
// Face ordering matches FACE_NORMALS: –Z, +Z, –X, +X, –Y, +Y.

#[rustfmt::skip]
const VERTICES: &[[f32; 3]] = &[
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    0,1,3,  0,3,2,   // –Z  prims 0,1
    4,6,7,  4,7,5,   // +Z  prims 2,3
    0,4,6,  0,6,2,   // –X  prims 4,5
    1,3,7,  1,7,5,   // +X  prims 6,7
    0,5,4,  0,1,5,   // –Y  prims 8,9
    2,6,7,  2,7,3,   // +Y  prims 10,11
];

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct MetalBackend {
    window:        Arc<Window>,
    device:        Device,
    queue:         CommandQueue,
    layer:         MetalLayer,
    rt_pipeline:   ComputePipelineState,
    blit_pipeline: RenderPipelineState,
    blas:          AccelerationStructure,
    tlas:          AccelerationStructure,
    cam_buf:       Buffer,
    out_tex:       Texture,
    width:         u64,
    height:        u64,
}

impl MetalBackend {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        eprintln!("[metal] device: {}", device.name());
        let queue = device.new_command_queue();

        // ── Attach a CAMetalLayer to the NSView ──────────────────────────────
        let layer = unsafe {
            let RawWindowHandle::AppKit(appkit) = window.window_handle()?.as_raw() else {
                bail!("Expected an AppKit window handle");
            };
            let ns_view = appkit.ns_view.as_ptr() as *mut objc::runtime::Object;

            let layer = MetalLayer::new();
            layer.set_device(&device);
            layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            layer.set_presents_with_transaction(false);
            layer.set_display_sync_enabled(true);

            // wantsLayer must be set before assigning the custom layer
            let () = msg_send![ns_view, setWantsLayer: true];
            let () = msg_send![ns_view, setLayer: layer.as_ptr()];

            layer
        };

        let size = window.inner_size();
        let (width, height) = (size.width as u64, size.height as u64);
        layer.set_drawable_size(CGSize::new(width as f64, height as f64));

        // ── Compile MSL ──────────────────────────────────────────────────────
        let library = device
            .new_library_with_source(MSL_SHADER, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("MSL compile error:\n{e}"))?;

        // Ray-tracing compute pipeline
        let rt_fn = library
            .get_function("raytrace_voxels", None)
            .map_err(|e| anyhow::anyhow!("Missing kernel raytrace_voxels: {e}"))?;
        let rt_pipeline = device
            .new_compute_pipeline_state_with_function(&rt_fn)
            .map_err(|e| anyhow::anyhow!("RT pipeline creation failed: {e}"))?;

        // Fullscreen-blit render pipeline (RGBA8 → BGRA8 drawable)
        let blit_vert = library
            .get_function("blit_vert", None)
            .map_err(|e| anyhow::anyhow!("Missing blit_vert: {e}"))?;
        let blit_frag = library
            .get_function("blit_frag", None)
            .map_err(|e| anyhow::anyhow!("Missing blit_frag: {e}"))?;
        let rp_desc = RenderPipelineDescriptor::new();
        rp_desc.set_vertex_function(Some(&blit_vert));
        rp_desc.set_fragment_function(Some(&blit_frag));
        rp_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let blit_pipeline = device
            .new_render_pipeline_state(&rp_desc)
            .map_err(|e| anyhow::anyhow!("Blit pipeline creation failed: {e}"))?;

        // ── BLAS — triangulated unit cube ────────────────────────────────────
        let vbuf = device.new_buffer_with_data(
            VERTICES.as_ptr() as *const _,
            (VERTICES.len() * std::mem::size_of::<[f32; 3]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ibuf = device.new_buffer_with_data(
            INDICES.as_ptr() as *const _,
            (INDICES.len() * std::mem::size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let tri_geom = AccelerationStructureTriangleGeometryDescriptor::descriptor();
        tri_geom.set_vertex_buffer(Some(&vbuf));
        tri_geom.set_vertex_stride(std::mem::size_of::<[f32; 3]>() as u64);
        tri_geom.set_index_buffer(Some(&ibuf));
        tri_geom.set_index_type(MTLIndexType::UInt16);
        tri_geom.set_triangle_count(INDICES.len() as u64 / 3);
        tri_geom.set_opaque(true);

        let geom_arr = Array::<AccelerationStructureGeometryDescriptor>::from_slice(
            &[&**tri_geom],
        );
        let blas_desc = PrimitiveAccelerationStructureDescriptor::descriptor();
        blas_desc.set_geometry_descriptors(geom_arr);

        let blas_sizes = device.acceleration_structure_sizes_with_descriptor(&blas_desc);
        let blas = device.new_acceleration_structure_with_size(blas_sizes.acceleration_structure_size);
        let blas_scratch = device.new_buffer(
            blas_sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModePrivate,
        );
        build_accel_sync(&queue, &blas, &blas_desc, &blas_scratch);
        eprintln!("[metal] BLAS built ({} triangles)", INDICES.len() / 3);

        // ── TLAS — 4 voxel instances ─────────────────────────────────────────
        let voxels = demo_voxels();
        let instances: Vec<MTLAccelerationStructureInstanceDescriptor> = voxels
            .iter()
            .map(|v| {
                let [tx, ty, tz] = v.aabb_min();
                MTLAccelerationStructureInstanceDescriptor {
                    // Column-major 4×3: 4 columns each 3 rows.
                    transformation_matrix: [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [tx,  ty,  tz ],
                    ],
                    options: MTLAccelerationStructureInstanceOptions::Opaque
                        | MTLAccelerationStructureInstanceOptions::DisableTriangleCulling,
                    mask: 0xFF,
                    intersection_function_table_offset: 0,
                    acceleration_structure_index: 0,
                }
            })
            .collect();

        let inst_buf = device.new_buffer_with_data(
            instances.as_ptr() as *const _,
            (instances.len() * std::mem::size_of::<MTLAccelerationStructureInstanceDescriptor>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let tlas_desc = InstanceAccelerationStructureDescriptor::descriptor();
        tlas_desc.set_instanced_acceleration_structures(
            Array::<AccelerationStructure>::from_slice(&[&*blas]),
        );
        tlas_desc.set_instance_count(instances.len() as u64);
        tlas_desc.set_instance_descriptor_buffer(&inst_buf);

        let tlas_sizes = device.acceleration_structure_sizes_with_descriptor(&tlas_desc);
        let tlas = device.new_acceleration_structure_with_size(tlas_sizes.acceleration_structure_size);
        let tlas_scratch = device.new_buffer(
            tlas_sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModePrivate,
        );
        build_accel_sync(&queue, &tlas, &tlas_desc, &tlas_scratch);
        eprintln!("[metal] TLAS built ({} instances)", instances.len());

        // ── Camera UBO & output texture ──────────────────────────────────────
        let cam = build_camera(width, height);
        let cam_buf = device.new_buffer_with_data(
            &cam as *const CameraData as *const _,
            std::mem::size_of::<CameraData>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_tex = make_output_texture(&device, width, height);

        eprintln!("[metal] backend ready ({}×{})", width, height);
        Ok(Self {
            window, device, queue, layer,
            rt_pipeline, blit_pipeline,
            blas, tlas, cam_buf, out_tex,
            width, height,
        })
    }
}

impl RenderBackend for MetalBackend {
    fn render(&mut self) -> Result<()> {
        let drawable = match self.layer.next_drawable() {
            Some(d) => d,
            None => {
                eprintln!("[metal] next_drawable() returned None");
                return Ok(());
            }
        };

        let cb = self.queue.new_command_buffer();

        // ── Compute: ray-trace → RGBA8Unorm intermediate texture ─────────────
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.rt_pipeline);
            enc.set_texture(0, Some(&self.out_tex));
            enc.set_acceleration_structure(0, Some(&*self.tlas));
            enc.set_buffer(1, Some(&self.cam_buf), 0);
            enc.use_resource(&**self.tlas, MTLResourceUsage::Read);
            enc.use_resource(&**self.blas, MTLResourceUsage::Read);

            let tg = MTLSize { width: 8, height: 8, depth: 1 };
            let groups = MTLSize {
                width:  (self.width  + 7) / 8,
                height: (self.height + 7) / 8,
                depth:  1,
            };
            enc.dispatch_thread_groups(groups, tg);
            enc.end_encoding();
        }

        // ── Render: fullscreen triangle copies intermediate → drawable ─────────
        {
            let rp_desc = RenderPassDescriptor::new();
            let ca = rp_desc.color_attachments().object_at(0).unwrap();
            ca.set_texture(Some(drawable.texture()));
            ca.set_load_action(MTLLoadAction::DontCare);
            ca.set_store_action(MTLStoreAction::Store);

            let enc = cb.new_render_command_encoder(rp_desc);
            enc.set_render_pipeline_state(&self.blit_pipeline);
            enc.set_fragment_texture(0, Some(&self.out_tex));
            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            enc.end_encoding();
        }

        cb.present_drawable(drawable);
        cb.commit();
        Ok(())
    }

    fn handle_resize(&mut self) {
        let size = self.window.inner_size();
        let (w, h) = (size.width as u64, size.height as u64);
        if w == self.width && h == self.height {
            return;
        }
        self.width  = w;
        self.height = h;
        self.layer.set_drawable_size(CGSize::new(w as f64, h as f64));
        self.out_tex = make_output_texture(&self.device, w, h);
        let cam = build_camera(w, h);
        unsafe {
            *(self.cam_buf.contents() as *mut CameraData) = cam;
        }
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_accel_sync(
    queue:   &CommandQueue,
    accel:   &AccelerationStructure,
    desc:    &AccelerationStructureDescriptorRef,
    scratch: &Buffer,
) {
    let cb  = queue.new_command_buffer();
    let enc = cb.new_acceleration_structure_command_encoder();
    enc.build_acceleration_structure(accel, desc, scratch, 0);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn make_output_texture(device: &Device, width: u64, height: u64) -> Texture {
    let desc = TextureDescriptor::new();
    desc.set_texture_type(MTLTextureType::D2);
    desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    desc.set_width(width);
    desc.set_height(height);
    desc.set_usage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
    desc.set_storage_mode(MTLStorageMode::Private);
    device.new_texture(&desc)
}

fn build_camera(width: u64, height: u64) -> CameraData {
    let eye    = Vec3::new(1.5, 4.0, 8.0);
    let center = Vec3::new(1.5, 1.0, 0.0);
    let view   = Mat4::look_at_rh(eye, center, Vec3::Y);
    let aspect = width as f32 / height.max(1) as f32;
    let proj   = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 1000.0);
    CameraData {
        view_inverse: view.inverse().to_cols_array_2d(),
        proj_inverse: proj.inverse().to_cols_array_2d(),
    }
}
