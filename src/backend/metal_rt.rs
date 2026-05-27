//! Metal ray-tracing backend (macOS, Apple Silicon / AMD / Intel with Metal 3+).
//!
//! Rendering architecture:
//!  - One merged-mesh BLAS built from face-culled terrain geometry
//!  - TLAS with a single identity instance wrapping that BLAS
//!  - Per-face data buffer (float4: RGB colour + normal index) indexed by
//!    `primitive_id / 2` in the shader — no per-instance colour needed
//!  - Compute kernel writes to RGBA8Unorm; render pass blits → BGRA8Unorm drawable
//!
//! Camera controls (click to capture cursor):
//!  - Mouse          : look
//!  - W / A / S / D  : fly forward / left / back / right
//!  - Space / E      : fly up       C / Q : fly down
//!  - Shift (held)   : sprint (3× speed)
//!  - Scroll wheel   : adjust base speed
//!  - Escape         : release cursor
//!  - F → P → S      : toggle FPS counter in window title (dev shortcut)

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use core_graphics_types::geometry::CGSize;
use glam::{Mat4, Vec3};
use metal::{foreign_types::ForeignType, *};
use winit::{
    keyboard::KeyCode,
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{CursorGrabMode, Window},
};

use crate::voxel::{build_terrain_mesh, generate_terrain};
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

kernel void raytrace_voxels(
    texture2d<float, access::write>  output [[texture(0)]],
    instance_acceleration_structure  tlas   [[buffer(0)]],
    constant CameraData&             camera [[buffer(1)]],
    device const float4*             faces  [[buffer(2)]],
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
        // Sky gradient
        float t = clamp(0.5 * (r.direction.y + 1.0), 0.0, 1.0);
        color = mix(float3(0.95, 0.93, 0.88), float3(0.30, 0.52, 0.90), t);
    } else {
        // Look up per-face colour and normal from the merged face buffer
        uint   fi         = result.primitive_id / 2u;
        float4 face_entry = faces[fi];
        float3 base       = face_entry.xyz;
        uint   norm_idx   = uint(face_entry.w);

        float3 n = FACE_NORMALS[norm_idx];
        if (dot(n, r.direction) > 0.0) n = -n;

        float3 sun  = normalize(float3(0.8, 1.8, 0.6));
        float  diff = max(dot(n, sun), 0.0);
        // Ambient-occlusion-like: darken bottom faces
        float  ao   = (norm_idx == 4u) ? 0.55 : 1.0;

        color = base * (0.18 + 0.82 * diff) * ao;

        // Simple fog by ray distance
        float fog = 1.0 - exp(-result.distance * 0.006);
        color = mix(color, float3(0.80, 0.82, 0.88), fog * fog);
    }
    output.write(float4(color, 1.0), tid);
}

// Fullscreen blit: RGBA8Unorm intermediate → BGRA8Unorm drawable
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
    face_buf:      Buffer,   // float4 per visible face: [r, g, b, normal_idx]
    out_tex:       Texture,
    width:         u64,
    height:        u64,

    // FPS camera
    cam_pos:    Vec3,
    cam_yaw:    f32,    // radians; 0 = looking toward –Z
    cam_pitch:  f32,    // radians; positive = up

    // Input state
    /// [W, S, A, D, Space/E, C/Q, Shift]
    keys:       [bool; 7],
    mouse_dx:   f32,
    mouse_dy:   f32,
    captured:   bool,
    move_speed: f32,    // base units/s — scroll wheel adjusts this
    last_time:  Instant,

    // FPS counter (toggled by pressing F → P → S)
    show_fps:    bool,
    fps_seq_buf: [Option<KeyCode>; 3],
    frame_count: u32,
    fps_timer:   Instant,
    fps_value:   f32,
}

impl MetalBackend {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        eprintln!("[metal] device: {}", device.name());
        let queue = device.new_command_queue();

        // ── CAMetalLayer ─────────────────────────────────────────────────────
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

        let rt_fn = library.get_function("raytrace_voxels", None)
            .map_err(|e| anyhow::anyhow!("Missing kernel: {e}"))?;
        let rt_pipeline = device.new_compute_pipeline_state_with_function(&rt_fn)
            .map_err(|e| anyhow::anyhow!("RT pipeline: {e}"))?;

        let blit_vert_fn = library.get_function("blit_vert", None)
            .map_err(|e| anyhow::anyhow!("Missing blit_vert: {e}"))?;
        let blit_frag_fn = library.get_function("blit_frag", None)
            .map_err(|e| anyhow::anyhow!("Missing blit_frag: {e}"))?;
        let rp_desc = RenderPipelineDescriptor::new();
        rp_desc.set_vertex_function(Some(&blit_vert_fn));
        rp_desc.set_fragment_function(Some(&blit_frag_fn));
        rp_desc.color_attachments().object_at(0).unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let blit_pipeline = device.new_render_pipeline_state(&rp_desc)
            .map_err(|e| anyhow::anyhow!("Blit pipeline: {e}"))?;

        // ── Terrain generation ────────────────────────────────────────────────
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0xDEAD_BEEF);
        eprintln!("[terrain] seed: {seed:#010x}  (set this in source to reproduce)");
        eprintln!("[terrain] Generating terrain…");

        let voxels = generate_terrain(seed);
        eprintln!("[terrain] {} voxels — building face-culled mesh…", voxels.len());

        let mesh = build_terrain_mesh(&voxels);
        assert!(!mesh.face_data.is_empty(), "terrain mesh has no visible faces");
        eprintln!(
            "[terrain] {} visible faces ({} vertices, {} indices)",
            mesh.face_data.len(),
            mesh.vertices.len(),
            mesh.indices.len(),
        );

        // ── BLAS — merged face-culled terrain mesh ────────────────────────────
        let vbuf = device.new_buffer_with_data(
            mesh.vertices.as_ptr() as *const _,
            (mesh.vertices.len() * std::mem::size_of::<[f32; 3]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ibuf = device.new_buffer_with_data(
            mesh.indices.as_ptr() as *const _,
            (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let tri_geom = AccelerationStructureTriangleGeometryDescriptor::descriptor();
        tri_geom.set_vertex_buffer(Some(&vbuf));
        tri_geom.set_vertex_stride(std::mem::size_of::<[f32; 3]>() as u64);
        tri_geom.set_index_buffer(Some(&ibuf));
        tri_geom.set_index_type(MTLIndexType::UInt32);
        tri_geom.set_triangle_count((mesh.indices.len() / 3) as u64);
        tri_geom.set_opaque(true);

        let geom_arr = Array::<AccelerationStructureGeometryDescriptor>::from_slice(
            &[&**tri_geom],
        );
        let blas_desc = PrimitiveAccelerationStructureDescriptor::descriptor();
        blas_desc.set_geometry_descriptors(geom_arr);

        let blas_sizes = device.acceleration_structure_sizes_with_descriptor(&blas_desc);
        let blas = device.new_acceleration_structure_with_size(
            blas_sizes.acceleration_structure_size,
        );
        let blas_scratch = device.new_buffer(
            blas_sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModePrivate,
        );
        build_accel_sync(&queue, &blas, &blas_desc, &blas_scratch);
        eprintln!("[metal] BLAS built");

        // ── TLAS — single identity instance wrapping the terrain BLAS ─────────
        let identity = MTLAccelerationStructureInstanceDescriptor {
            transformation_matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            options: MTLAccelerationStructureInstanceOptions::Opaque
                | MTLAccelerationStructureInstanceOptions::DisableTriangleCulling,
            mask: 0xFF,
            intersection_function_table_offset: 0,
            acceleration_structure_index: 0,
        };
        let inst_buf = device.new_buffer_with_data(
            &identity as *const MTLAccelerationStructureInstanceDescriptor as *const _,
            std::mem::size_of::<MTLAccelerationStructureInstanceDescriptor>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let tlas_desc = InstanceAccelerationStructureDescriptor::descriptor();
        tlas_desc.set_instanced_acceleration_structures(
            Array::<AccelerationStructure>::from_slice(&[&*blas]),
        );
        tlas_desc.set_instance_count(1);
        tlas_desc.set_instance_descriptor_buffer(&inst_buf);

        let tlas_sizes = device.acceleration_structure_sizes_with_descriptor(&tlas_desc);
        let tlas = device.new_acceleration_structure_with_size(
            tlas_sizes.acceleration_structure_size,
        );
        let tlas_scratch = device.new_buffer(
            tlas_sizes.build_scratch_buffer_size,
            MTLResourceOptions::StorageModePrivate,
        );
        build_accel_sync(&queue, &tlas, &tlas_desc, &tlas_scratch);
        eprintln!("[metal] TLAS built — ready!");

        // ── Per-face data buffer (float4: RGB + normal_idx) ───────────────────
        let face_buf = device.new_buffer_with_data(
            mesh.face_data.as_ptr() as *const _,
            (mesh.face_data.len() * std::mem::size_of::<[f32; 4]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // ── Camera — start outside the world, looking inward ──────────────────
        // Terrain spans x=0..48, z=0..48, heights ~2-18.
        let cam_pos   = Vec3::new(24.0, 14.0, 70.0);
        let cam_yaw   = 0.0_f32;    // looking toward –Z (into the world)
        let cam_pitch = -0.15_f32;  // slight downward tilt

        let cam = camera_data(cam_pos, cam_yaw, cam_pitch, width, height);
        let cam_buf = device.new_buffer_with_data(
            &cam as *const CameraData as *const _,
            std::mem::size_of::<CameraData>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_tex = make_output_texture(&device, width, height);

        eprintln!("[metal] Click window to capture mouse — Escape to release");
        eprintln!("[metal] WASD fly  Space/E for up, Q/C up-down  Shift sprint  Scroll speed");
        eprintln!("[metal] Press F-P-S to toggle FPS counter in title bar");

        Ok(Self {
            window, device, queue, layer,
            rt_pipeline, blit_pipeline,
            blas, tlas,
            cam_buf, face_buf, out_tex,
            width, height,
            cam_pos, cam_yaw, cam_pitch,
            keys: [false; 7],
            mouse_dx: 0.0, mouse_dy: 0.0,
            captured: false,
            move_speed: 8.0,
            last_time: Instant::now(),
            show_fps:    false,
            fps_seq_buf: [None; 3],
            frame_count: 0,
            fps_timer:   Instant::now(),
            fps_value:   0.0,
        })
    }
}

impl RenderBackend for MetalBackend {
    // ── Input ─────────────────────────────────────────────────────────────────

    fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW                            => self.keys[0] = pressed,
            KeyCode::KeyS                            => self.keys[1] = pressed,
            KeyCode::KeyA                            => self.keys[2] = pressed,
            KeyCode::KeyD                            => self.keys[3] = pressed,
            KeyCode::Space | KeyCode::KeyE           => self.keys[4] = pressed,
            KeyCode::KeyC | KeyCode::KeyQ            => self.keys[5] = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys[6] = pressed,
            KeyCode::Escape if pressed => {
                self.captured = false;
                let _ = self.window.set_cursor_grab(CursorGrabMode::None);
                self.window.set_cursor_visible(true);
            }
            _ => {}
        }

        // FPS toggle: detect F → P → S key sequence (any key resets only its slot)
        if pressed {
            self.fps_seq_buf = [self.fps_seq_buf[1], self.fps_seq_buf[2], Some(key)];
            if self.fps_seq_buf == [
                Some(KeyCode::KeyF),
                Some(KeyCode::KeyP),
                Some(KeyCode::KeyS),
            ] {
                self.show_fps = !self.show_fps;
                self.fps_seq_buf = [None; 3];
                if !self.show_fps {
                    self.window.set_title("Voxeller Engine");
                }
            }
        }
    }

    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.captured {
            self.mouse_dx += dx as f32;
            self.mouse_dy += dy as f32;
        }
    }

    fn handle_mouse_button(&mut self, pressed: bool) {
        if pressed && !self.captured {
            self.captured = true;
            let _ = self.window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            self.window.set_cursor_visible(false);
        }
    }

    /// Scroll up → faster, scroll down → slower (multiplicative, clamped).
    fn handle_scroll(&mut self, delta: f32) {
        self.move_speed = (self.move_speed * 1.25_f32.powf(delta)).clamp(1.0, 300.0);
        eprintln!("[metal] speed: {:.1} u/s", self.move_speed);
    }

    // ── Render ────────────────────────────────────────────────────────────────

    fn render(&mut self) -> Result<()> {
        // Delta time
        let now = Instant::now();
        let dt  = now.duration_since(self.last_time).as_secs_f32().min(0.1);
        self.last_time = now;

        // Mouse look
        const SENS: f32 = 0.002;
        self.cam_yaw   += self.mouse_dx * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_dy * SENS)
            .clamp(-std::f32::consts::FRAC_PI_2 * 0.99, std::f32::consts::FRAC_PI_2 * 0.99);
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;

        // World-space forward & right vectors
        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let fwd   = Vec3::new(sy * cp, sp, -cy * cp);
        let right = fwd.cross(Vec3::Y).normalize();

        // Sprint multiplier
        let spd = self.move_speed * dt * if self.keys[6] { 3.0 } else { 1.0 };

        if self.keys[0] { self.cam_pos += fwd     * spd; }  // W
        if self.keys[1] { self.cam_pos -= fwd     * spd; }  // S
        if self.keys[2] { self.cam_pos -= right   * spd; }  // A
        if self.keys[3] { self.cam_pos += right   * spd; }  // D
        if self.keys[4] { self.cam_pos += Vec3::Y * spd; }  // Space / E
        if self.keys[5] { self.cam_pos -= Vec3::Y * spd; }  // C / Q

        // Update camera UBO
        let cam = camera_data(self.cam_pos, self.cam_yaw, self.cam_pitch,
                               self.width, self.height);
        unsafe { *(self.cam_buf.contents() as *mut CameraData) = cam; }

        // Acquire drawable
        let drawable = match self.layer.next_drawable() {
            Some(d) => d,
            None    => { eprintln!("[metal] next_drawable() = None"); return Ok(()); }
        };

        let cb = self.queue.new_command_buffer();

        // Compute: ray-trace → RGBA8Unorm intermediate
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.rt_pipeline);
            enc.set_texture(0, Some(&self.out_tex));
            enc.set_acceleration_structure(0, Some(&*self.tlas));
            enc.set_buffer(1, Some(&self.cam_buf),  0);
            enc.set_buffer(2, Some(&self.face_buf), 0);
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

        // Render: fullscreen triangle → BGRA8Unorm drawable
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

        // FPS counter — update window title once per second when enabled
        self.frame_count += 1;
        let elapsed = self.fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps_value  = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.fps_timer   = Instant::now();

            if self.show_fps {
                self.window.set_title(&format!(
                    "Voxeller Engine  |  {:.0} FPS  |  ({:.1}, {:.1}, {:.1})",
                    self.fps_value,
                    self.cam_pos.x, self.cam_pos.y, self.cam_pos.z,
                ));
            }
        }

        Ok(())
    }

    fn handle_resize(&mut self) {
        let size = self.window.inner_size();
        let (w, h) = (size.width as u64, size.height as u64);
        if w == self.width && h == self.height { return; }
        self.width  = w;
        self.height = h;
        self.layer.set_drawable_size(CGSize::new(w as f64, h as f64));
        self.out_tex = make_output_texture(&self.device, w, h);
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

/// Build CameraData from world-space position and Euler angles.
/// * `yaw`   — around Y; 0 = looking toward –Z
/// * `pitch` — around X; positive = up
fn camera_data(pos: Vec3, yaw: f32, pitch: f32, width: u64, height: u64) -> CameraData {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let fwd    = Vec3::new(sy * cp, sp, -cy * cp);
    let view   = Mat4::look_at_rh(pos, pos + fwd, Vec3::Y);
    let aspect = width as f32 / height.max(1) as f32;
    let proj   = Mat4::perspective_rh(70f32.to_radians(), aspect, 0.05, 2000.0);
    CameraData {
        view_inverse: view.inverse().to_cols_array_2d(),
        proj_inverse: proj.inverse().to_cols_array_2d(),
    }
}
