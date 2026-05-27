use anyhow::Result;
use ash::{khr, vk};

use super::memory::{align_up, begin_one_shot, create_buffer, end_one_shot, upload_buffer};
use crate::voxel::Voxel;

pub struct AccelStructures {
    pub blas:        vk::AccelerationStructureKHR,
    pub blas_buffer: vk::Buffer,
    pub blas_mem:    vk::DeviceMemory,

    pub tlas:        vk::AccelerationStructureKHR,
    pub tlas_buffer: vk::Buffer,
    pub tlas_mem:    vk::DeviceMemory,
}

impl AccelStructures {
    /// Build a single-AABB BLAS (unit cube [0,1]^3) and a TLAS with one instance per voxel.
    pub unsafe fn build(
        instance:    &ash::Instance,
        device:      &ash::Device,
        phys:        vk::PhysicalDevice,
        queue:       vk::Queue,
        cmd_pool:    vk::CommandPool,
        accel_ext:   &khr::acceleration_structure::Device,
        voxels:      &[Voxel],
    ) -> Result<Self> {
        // ── BLAS ──────────────────────────────────────────────────────────────
        // One AABB geometry: unit cube in object space.
        let aabb_data = vk::AabbPositionsKHR {
            min_x: 0.0, min_y: 0.0, min_z: 0.0,
            max_x: 1.0, max_y: 1.0, max_z: 1.0,
        };
        let (aabb_buf, aabb_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            std::slice::from_ref(&aabb_data),
        )?;

        let aabb_addr = vk::DeviceOrHostAddressConstKHR {
            device_address: device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(aabb_buf),
            ),
        };

        let blas_geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                    .data(aabb_addr)
                    .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as vk::DeviceSize),
            });

        let blas_geometries = [blas_geometry];
        let mut blas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&blas_geometries);

        let primitive_counts = [1u32]; // one AABB
        let mut blas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        accel_ext.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &blas_build_info,
            &primitive_counts,
            &mut blas_sizes,
        );

        let (blas_buffer, blas_mem) = create_buffer(
            instance, device, phys,
            blas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let blas = accel_ext.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(blas_buffer)
                .size(blas_sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )?;

        let (blas_scratch_buf, blas_scratch_mem) = create_buffer(
            instance, device, phys,
            blas_sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let blas_scratch_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(blas_scratch_buf),
        );

        blas_build_info.dst_acceleration_structure = blas;
        blas_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: blas_scratch_addr,
        };

        let blas_ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count:  1,
            primitive_offset: 0,
            first_vertex:     0,
            transform_offset: 0,
        }];

        let cb = begin_one_shot(device, cmd_pool)?;
        accel_ext.cmd_build_acceleration_structures(cb, &[blas_build_info], &[&blas_ranges]);
        // Barrier: AS write -> AS read
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
            &[], &[],
        );
        end_one_shot(device, cmd_pool, queue, cb)?;

        device.destroy_buffer(blas_scratch_buf, None);
        device.free_memory(blas_scratch_mem, None);
        device.destroy_buffer(aabb_buf, None);
        device.free_memory(aabb_mem, None);

        let blas_addr = accel_ext.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(blas),
        );

        // ── TLAS ──────────────────────────────────────────────────────────────
        let instances: Vec<vk::AccelerationStructureInstanceKHR> = voxels
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let [tx, ty, tz] = v.aabb_min();
                vk::AccelerationStructureInstanceKHR {
                    transform: vk::TransformMatrixKHR {
                        // row-major 3×4 matrix stored flat; this is a translation by (tx,ty,tz)
                        matrix: [
                            1.0, 0.0, 0.0, tx,
                            0.0, 1.0, 0.0, ty,
                            0.0, 0.0, 1.0, tz,
                        ],
                    },
                    instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xFF),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: blas_addr,
                    },
                }
            })
            .collect();

        let (inst_buf, inst_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &instances,
        )?;
        let inst_addr = vk::DeviceOrHostAddressConstKHR {
            device_address: device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(inst_buf),
            ),
        };

        let tlas_geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(inst_addr),
            });

        let tlas_geometries = [tlas_geometry];
        let mut tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&tlas_geometries);

        let tlas_prim_counts = [instances.len() as u32];
        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        accel_ext.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &tlas_prim_counts,
            &mut tlas_sizes,
        );

        let (tlas_buffer, tlas_mem) = create_buffer(
            instance, device, phys,
            tlas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let tlas = accel_ext.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(tlas_buffer)
                .size(tlas_sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
            None,
        )?;

        let (tlas_scratch_buf, tlas_scratch_mem) = create_buffer(
            instance, device, phys,
            tlas_sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let tlas_scratch_addr = device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(tlas_scratch_buf),
        );

        tlas_build_info.dst_acceleration_structure = tlas;
        tlas_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: tlas_scratch_addr,
        };

        let tlas_ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count:  instances.len() as u32,
            primitive_offset: 0,
            first_vertex:     0,
            transform_offset: 0,
        }];

        let cb = begin_one_shot(device, cmd_pool)?;
        accel_ext.cmd_build_acceleration_structures(cb, &[tlas_build_info], &[&tlas_ranges]);
        end_one_shot(device, cmd_pool, queue, cb)?;

        device.destroy_buffer(tlas_scratch_buf, None);
        device.free_memory(tlas_scratch_mem, None);
        device.destroy_buffer(inst_buf, None);
        device.free_memory(inst_mem, None);

        Ok(Self { blas, blas_buffer, blas_mem, tlas, tlas_buffer, tlas_mem })
    }

    pub unsafe fn destroy(
        &self,
        device: &ash::Device,
        accel_ext: &khr::acceleration_structure::Device,
    ) {
        accel_ext.destroy_acceleration_structure(self.tlas, None);
        device.destroy_buffer(self.tlas_buffer, None);
        device.free_memory(self.tlas_mem, None);

        accel_ext.destroy_acceleration_structure(self.blas, None);
        device.destroy_buffer(self.blas_buffer, None);
        device.free_memory(self.blas_mem, None);
    }
}

/// Align `value` up to a multiple of `align_to`, where `align_to` is a power of two.
#[allow(dead_code)]
pub(crate) fn aligned(value: usize, align_to: usize) -> usize {
    align_up(value, align_to)
}
