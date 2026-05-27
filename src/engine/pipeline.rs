use anyhow::Result;
use ash::{khr, vk};

use super::memory::{align_up, upload_buffer};

macro_rules! spv {
    ($name:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    };
}

pub struct RtPipeline {
    pub pipeline:        vk::Pipeline,
    pub layout:          vk::PipelineLayout,
    pub ds_layout:       vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set:  vk::DescriptorSet,

    pub sbt_buffer: vk::Buffer,
    pub sbt_mem:    vk::DeviceMemory,

    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region:   vk::StridedDeviceAddressRegionKHR,
    pub hit_region:    vk::StridedDeviceAddressRegionKHR,
    pub call_region:   vk::StridedDeviceAddressRegionKHR,
}

impl RtPipeline {
    pub unsafe fn create(
        instance:     &ash::Instance,
        device:       &ash::Device,
        phys:         vk::PhysicalDevice,
        rt_ext:       &khr::ray_tracing_pipeline::Device,
        tlas:         vk::AccelerationStructureKHR,
        storage_view: vk::ImageView,
        camera_buf:   vk::Buffer,
        face_buf:     vk::Buffer,
        rt_props:     &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    ) -> Result<Self> {
        // ── Descriptor set layout ─────────────────────────────────────────────
        let bindings = [
            // 0 – TLAS (raygen)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // 1 – storage image (raygen)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // 2 – camera UBO (raygen)
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // 3 – face data SSBO (closest-hit)
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ];
        let ds_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let ds_layouts = [ds_layout];
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&ds_layouts),
            None,
        )?;

        // ── Shader modules ────────────────────────────────────────────────────
        let raygen_spv = spv!("raygen.rgen.spv");
        let miss_spv   = spv!("miss.rmiss.spv");
        let chit_spv   = spv!("closest_hit.rchit.spv");

        let make_module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            let (prefix, aligned, suffix) = spv.align_to::<u32>();
            assert!(prefix.is_empty() && suffix.is_empty(), "SPIR-V not 4-byte aligned");
            Ok(device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(aligned),
                None,
            )?)
        };

        let raygen_mod = make_module(raygen_spv)?;
        let miss_mod   = make_module(miss_spv)?;
        let chit_mod   = make_module(chit_spv)?;

        let entry = std::ffi::CString::new("main").unwrap();

        let stages = [
            // 0 – raygen
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(raygen_mod)
                .name(&entry),
            // 1 – miss
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(miss_mod)
                .name(&entry),
            // 2 – closest hit
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(chit_mod)
                .name(&entry),
        ];

        // ── Shader groups ─────────────────────────────────────────────────────
        let groups = [
            // 0 – raygen (GENERAL group)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // 1 – miss (GENERAL group)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // 2 – hit (TRIANGLES_HIT_GROUP — no intersection shader needed for triangles)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(layout);

        let pipeline = rt_ext
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
            .map_err(|(_, e)| e)?
            .into_iter()
            .next()
            .unwrap();

        for m in [raygen_mod, miss_mod, chit_mod] {
            device.destroy_shader_module(m, None);
        }

        // ── Shader Binding Table ──────────────────────────────────────────────
        let handle_size  = rt_props.shader_group_handle_size as usize;
        let handle_align = rt_props.shader_group_handle_alignment as usize;
        let base_align   = rt_props.shader_group_base_alignment as usize;
        let entry_stride = align_up(handle_size, handle_align);
        let region_size  = |count: usize| align_up(entry_stride * count, base_align);

        let rg_size   = region_size(1);
        let miss_size = region_size(1);
        let hit_size  = region_size(1);
        let sbt_size  = rg_size + miss_size + hit_size;

        let raw_handles = rt_ext.get_ray_tracing_shader_group_handles(
            pipeline, 0, 3, 3 * handle_size,
        )?;

        let mut sbt_host = vec![0u8; sbt_size];
        sbt_host[..handle_size]
            .copy_from_slice(&raw_handles[..handle_size]);
        sbt_host[rg_size..rg_size + handle_size]
            .copy_from_slice(&raw_handles[handle_size..2 * handle_size]);
        sbt_host[rg_size + miss_size..rg_size + miss_size + handle_size]
            .copy_from_slice(&raw_handles[2 * handle_size..3 * handle_size]);

        let (sbt_buffer, sbt_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &sbt_host,
        )?;

        let sbt_base = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(sbt_buffer),
        );

        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_base,
            stride:         rg_size as u64, // raygen stride == size
            size:           rg_size as u64,
        };
        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_base + rg_size as u64,
            stride:         entry_stride as u64,
            size:           miss_size as u64,
        };
        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_base + rg_size as u64 + miss_size as u64,
            stride:         entry_stride as u64,
            size:           hit_size as u64,
        };
        let call_region = vk::StridedDeviceAddressRegionKHR::default();

        // ── Descriptor pool + set ─────────────────────────────────────────────
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE,              descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER,             descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER,             descriptor_count: 1 },
        ];
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?;
        let descriptor_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&ds_layouts),
        )?[0];

        // Write initial descriptors
        let tlas_arr = [tlas];
        let mut write_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&tlas_arr);
        let img_info = [vk::DescriptorImageInfo {
            image_view:   storage_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler:      vk::Sampler::null(),
        }];
        let cam_info  = [vk::DescriptorBufferInfo { buffer: camera_buf, offset: 0, range: vk::WHOLE_SIZE }];
        let face_info = [vk::DescriptorBufferInfo { buffer: face_buf,   offset: 0, range: vk::WHOLE_SIZE }];

        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .push_next(&mut write_as),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&img_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set).dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&cam_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set).dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&face_info),
            ],
            &[],
        );

        Ok(Self {
            pipeline, layout, ds_layout, descriptor_pool, descriptor_set,
            sbt_buffer, sbt_mem,
            raygen_region, miss_region, hit_region, call_region,
        })
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.ds_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_buffer(self.sbt_buffer, None);
        device.free_memory(self.sbt_mem, None);
    }
}
