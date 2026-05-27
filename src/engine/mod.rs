pub(crate) mod accel;
pub(crate) mod memory;
pub(crate) mod pipeline;

use accel::AccelStructures;
use memory::{begin_one_shot, create_image, end_one_shot, upload_buffer};
use pipeline::RtPipeline;

use anyhow::{bail, Result};
use ash::{khr, vk};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use winit::window::Window;

use crate::voxel::demo_voxels;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUBO {
    view_inverse: [[f32; 4]; 4],
    proj_inverse: [[f32; 4]; 4],
}

#[allow(dead_code)]
pub struct Engine {
    window: Arc<Window>,

    // Core Vulkan
    _entry:          ash::Entry,
    instance:        ash::Instance,
    device:          ash::Device,
    phys:            vk::PhysicalDevice,
    graphics_queue:  vk::Queue,
    queue_family:    u32,

    // Extensions
    surface_ext:     khr::surface::Instance,
    swapchain_ext:   khr::swapchain::Device,
    accel_ext:       khr::acceleration_structure::Device,
    rt_ext:          khr::ray_tracing_pipeline::Device,
    #[allow(dead_code)]
    deferred_ext:    khr::deferred_host_operations::Device,

    // Surface + swapchain
    surface:              vk::SurfaceKHR,
    swapchain:            vk::SwapchainKHR,
    swapchain_images:     Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format:     vk::Format,
    swapchain_extent:     vk::Extent2D,

    // Ray-tracing render target (storage image, GENERAL layout)
    storage_image:     vk::Image,
    storage_image_mem: vk::DeviceMemory,
    storage_view:      vk::ImageView,

    // Camera UBO
    camera_buf: vk::Buffer,
    camera_mem: vk::DeviceMemory,

    // Acceleration structures
    accel: AccelStructures,

    // Pipeline + SBT + descriptors
    rt_pipeline: RtPipeline,

    // Commands + sync
    command_pool:    vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_ready:     Vec<vk::Semaphore>,
    render_done:     Vec<vk::Semaphore>,
    in_flight:       Vec<vk::Fence>,
    frame_index:     usize,
}

impl Engine {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        unsafe { Self::init(window) }
    }

    unsafe fn init(window: Arc<Window>) -> Result<Self> {
        // ── Entry & instance ──────────────────────────────────────────────────
        let entry = ash::Entry::linked();

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Voxeller")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let display_handle = window.display_handle().unwrap().as_raw();
        let mut inst_exts =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();
        inst_exts.push(ash::ext::debug_utils::NAME.as_ptr());

        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&inst_exts)
                .enabled_layer_names(&layers),
            None,
        )?;

        // ── Physical device ───────────────────────────────────────────────────
        let required_device_exts: &[*const i8] = &[
            khr::swapchain::NAME.as_ptr(),
            khr::acceleration_structure::NAME.as_ptr(),
            khr::ray_tracing_pipeline::NAME.as_ptr(),
            khr::deferred_host_operations::NAME.as_ptr(),
        ];

        let phys = instance
            .enumerate_physical_devices()?
            .into_iter()
            .find(|&pd| {
                let exts = instance
                    .enumerate_device_extension_properties(pd)
                    .unwrap_or_default();
                required_device_exts.iter().all(|&req| {
                    let req = std::ffi::CStr::from_ptr(req);
                    exts.iter().any(|e| {
                        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == req
                    })
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No GPU with Vulkan raytracing support found.\n\
                     Required extensions: VK_KHR_acceleration_structure, \
                     VK_KHR_ray_tracing_pipeline.\n\
                     (MoltenVK on macOS does NOT support these.)"
                )
            })?;

        // ── Query queue family ────────────────────────────────────────────────
        let queue_family = instance
            .get_physical_device_queue_family_properties(phys)
            .iter()
            .enumerate()
            .find(|(_, p)| {
                p.queue_flags.contains(
                    vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
                )
            })
            .map(|(i, _)| i as u32)
            .ok_or_else(|| anyhow::anyhow!("No suitable queue family"))?;

        // ── Logical device ────────────────────────────────────────────────────
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);

        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
            .buffer_device_address(true);
        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut rt_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
            .ray_tracing_pipeline(true);

        let device = instance.create_device(
            phys,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(required_device_exts)
                .push_next(&mut bda_features)
                .push_next(&mut as_features)
                .push_next(&mut rt_features),
            None,
        )?;

        let graphics_queue = device.get_device_queue(queue_family, 0);

        // ── Extension loaders ─────────────────────────────────────────────────
        let surface_ext   = khr::surface::Instance::new(&entry, &instance);
        let swapchain_ext = khr::swapchain::Device::new(&instance, &device);
        let accel_ext     = khr::acceleration_structure::Device::new(&instance, &device);
        let rt_ext        = khr::ray_tracing_pipeline::Device::new(&instance, &device);
        let deferred_ext  = khr::deferred_host_operations::Device::new(&instance, &device);

        // ── RT pipeline properties (handle sizes for SBT) ─────────────────────
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2   = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
        instance.get_physical_device_properties2(phys, &mut props2);

        // ── Surface ───────────────────────────────────────────────────────────
        let window_handle = window.window_handle().unwrap().as_raw();
        let surface = ash_window::create_surface(
            &entry, &instance, display_handle, window_handle, None,
        )?;

        // ── Swapchain ─────────────────────────────────────────────────────────
        let (swapchain, swapchain_format, swapchain_extent, swapchain_images) =
            create_swapchain(&swapchain_ext, &surface_ext, surface, phys, &window)?;

        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|&img| {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(swapchain_format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
            })
            .collect::<std::result::Result<_, _>>()?;

        // ── Command pool ──────────────────────────────────────────────────────
        let command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;

        // ── Storage image (ray-tracing render target) ─────────────────────────
        let (storage_image, storage_image_mem, storage_view) = create_image(
            &instance, &device, phys,
            swapchain_extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;

        // Transition storage image to GENERAL layout once
        {
            let cb = begin_one_shot(&device, command_pool)?;
            image_barrier(
                &device, cb,
                storage_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );
            end_one_shot(&device, command_pool, graphics_queue, cb)?;
        }

        // ── Camera UBO ────────────────────────────────────────────────────────
        let camera_data = build_camera(swapchain_extent);
        let (camera_buf, camera_mem) = upload_buffer(
            &instance, &device, phys,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            std::slice::from_ref(&camera_data),
        )?;

        // ── Acceleration structures ────────────────────────────────────────────
        let voxels = demo_voxels();
        let accel = AccelStructures::build(
            &instance, &device, phys,
            graphics_queue, command_pool,
            &accel_ext,
            &voxels,
        )?;

        // ── RT pipeline ───────────────────────────────────────────────────────
        let rt_pipeline = RtPipeline::create(
            &instance, &device, phys,
            &rt_ext,
            accel.tlas,
            storage_view,
            camera_buf,
            &rt_props,
        )?;

        // ── Per-frame command buffers ─────────────────────────────────────────
        let command_buffers = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(FRAMES_IN_FLIGHT as u32),
        )?;

        // ── Sync objects ──────────────────────────────────────────────────────
        let sem_info   = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_ready  = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut render_done  = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight    = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            image_ready.push(device.create_semaphore(&sem_info, None)?);
            render_done.push(device.create_semaphore(&sem_info, None)?);
            in_flight.push(device.create_fence(&fence_info, None)?);
        }

        Ok(Self {
            window,
            _entry: entry,
            instance,
            device,
            phys,
            graphics_queue,
            queue_family,
            surface_ext,
            swapchain_ext,
            accel_ext,
            rt_ext,
            deferred_ext,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            storage_image,
            storage_image_mem,
            storage_view,
            camera_buf,
            camera_mem,
            accel,
            rt_pipeline,
            command_pool,
            command_buffers,
            image_ready,
            render_done,
            in_flight,
            frame_index: 0,
        })
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn handle_resize(&mut self) {
        // Recreate swapchain + storage image on next render
        // For brevity this demo just idles until the next frame
        unsafe { self.device.device_wait_idle().ok() };
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe { self.draw_frame() }
    }

    unsafe fn draw_frame(&mut self) -> Result<()> {
        let fi = self.frame_index % FRAMES_IN_FLIGHT;
        let fence   = self.in_flight[fi];
        let img_sem = self.image_ready[fi];
        let rdr_sem = self.render_done[fi];
        let cb      = self.command_buffers[fi];

        self.device.wait_for_fences(&[fence], true, u64::MAX)?;

        let (img_idx, suboptimal) = match self.swapchain_ext.acquire_next_image(
            self.swapchain,
            u64::MAX,
            img_sem,
            vk::Fence::null(),
        ) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()),
            Err(e) => bail!(e),
        };
        if suboptimal {
            return Ok(());
        }

        self.device.reset_fences(&[fence])?;
        self.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;

        // ── Record command buffer ─────────────────────────────────────────────
        self.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // Bind RT pipeline + descriptors
        self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.rt_pipeline.pipeline);
        self.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            self.rt_pipeline.layout,
            0,
            &[self.rt_pipeline.descriptor_set],
            &[],
        );

        // Trace rays → storage image
        self.rt_ext.cmd_trace_rays(
            cb,
            &self.rt_pipeline.raygen_region,
            &self.rt_pipeline.miss_region,
            &self.rt_pipeline.hit_region,
            &self.rt_pipeline.call_region,
            self.swapchain_extent.width,
            self.swapchain_extent.height,
            1,
        );

        let sc_image = self.swapchain_images[img_idx as usize];

        // storage_image: GENERAL → TRANSFER_SRC
        image_barrier(
            &self.device, cb,
            self.storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
        );
        // swapchain image: UNDEFINED → TRANSFER_DST
        image_barrier(
            &self.device, cb,
            sc_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        // Blit (handles format/size differences)
        let region = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                mip_level:        0,
                base_array_layer: 0,
                layer_count:      1,
            },
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.swapchain_extent.width  as i32,
                    y: self.swapchain_extent.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                mip_level:        0,
                base_array_layer: 0,
                layer_count:      1,
            },
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: self.swapchain_extent.width  as i32,
                    y: self.swapchain_extent.height as i32,
                    z: 1,
                },
            ],
        };
        self.device.cmd_blit_image(
            cb,
            self.storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            sc_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
            vk::Filter::NEAREST,
        );

        // storage_image: TRANSFER_SRC → GENERAL (for next frame)
        image_barrier(
            &self.device, cb,
            self.storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        );
        // swapchain image: TRANSFER_DST → PRESENT
        image_barrier(
            &self.device, cb,
            sc_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );

        self.device.end_command_buffer(cb)?;

        // ── Submit ────────────────────────────────────────────────────────────
        let wait_sems   = [img_sem];
        let wait_stages = [vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR];
        let sig_sems    = [rdr_sem];
        let cbs         = [cb];
        self.device.queue_submit(
            self.graphics_queue,
            &[vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cbs)
                .signal_semaphores(&sig_sems)],
            fence,
        )?;

        // ── Present ───────────────────────────────────────────────────────────
        let swapchains  = [self.swapchain];
        let img_indices = [img_idx];
        match self.swapchain_ext.queue_present(
            self.graphics_queue,
            &vk::PresentInfoKHR::default()
                .wait_semaphores(&sig_sems)
                .swapchains(&swapchains)
                .image_indices(&img_indices),
        ) {
            Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
            Err(e) => bail!(e),
        }

        self.frame_index += 1;
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            for &s in &self.image_ready { self.device.destroy_semaphore(s, None); }
            for &s in &self.render_done { self.device.destroy_semaphore(s, None); }
            for &f in &self.in_flight   { self.device.destroy_fence(f, None); }

            self.rt_pipeline.destroy(&self.device);
            self.accel.destroy(&self.device, &self.accel_ext);

            self.device.destroy_buffer(self.camera_buf, None);
            self.device.free_memory(self.camera_mem, None);

            self.device.destroy_image_view(self.storage_view, None);
            self.device.destroy_image(self.storage_image, None);
            self.device.free_memory(self.storage_image_mem, None);

            for &v in &self.swapchain_image_views {
                self.device.destroy_image_view(v, None);
            }
            self.swapchain_ext.destroy_swapchain(self.swapchain, None);
            self.surface_ext.destroy_surface(self.surface, None);

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Insert a pipeline image memory barrier for a layout transition.
#[allow(clippy::too_many_arguments)]
unsafe fn image_barrier(
    device:      &ash::Device,
    cb:          vk::CommandBuffer,
    image:       vk::Image,
    old_layout:  vk::ImageLayout,
    new_layout:  vk::ImageLayout,
    src_access:  vk::AccessFlags,
    dst_access:  vk::AccessFlags,
    src_stage:   vk::PipelineStageFlags,
    dst_stage:   vk::PipelineStageFlags,
) {
    device.cmd_pipeline_barrier(
        cb,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )],
    );
}

fn build_camera(extent: vk::Extent2D) -> CameraUBO {
    let eye    = Vec3::new(1.5, 4.0, 8.0);
    let center = Vec3::new(1.5, 1.0, 0.0);
    let view   = Mat4::look_at_rh(eye, center, Vec3::Y);
    let aspect = extent.width as f32 / extent.height as f32;
    let proj   = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 1000.0);
    CameraUBO {
        view_inverse: view.inverse().to_cols_array_2d(),
        proj_inverse: proj.inverse().to_cols_array_2d(),
    }
}

unsafe fn create_swapchain(
    swapchain_ext: &khr::swapchain::Device,
    surface_ext:   &khr::surface::Instance,
    surface:       vk::SurfaceKHR,
    phys:          vk::PhysicalDevice,
    window:        &Window,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
    let caps     = surface_ext.get_physical_device_surface_capabilities(phys, surface)?;
    let formats  = surface_ext.get_physical_device_surface_formats(phys, surface)?;
    let present_modes = surface_ext.get_physical_device_surface_present_modes(phys, surface)?;

    let format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
        .unwrap_or(vk::SurfaceFormatKHR {
            format:      vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });

    let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    };

    let size = window.inner_size();
    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width:  size.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: size.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };

    let image_count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
        u32::MAX
    } else {
        caps.max_image_count
    });

    let sc = swapchain_ext.create_swapchain(
        &vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true),
        None,
    )?;

    let images = swapchain_ext.get_swapchain_images(sc)?;
    Ok((sc, format.format, extent, images))
}
