use anyhow::Result;
use ash::{khr, vk};

use super::memory::{align_up, begin_one_shot, create_buffer, end_one_shot, upload_buffer};
use crate::voxel::TerrainMesh;

pub struct AccelStructures {
    pub blas:        vk::AccelerationStructureKHR,
    pub blas_buffer: vk::Buffer,
    pub blas_mem:    vk::DeviceMemory,

    pub tlas:        vk::AccelerationStructureKHR,
    pub tlas_buffer: vk::Buffer,
    pub tlas_mem:    vk::DeviceMemory,

    /// Per-face colour + normal index, read by the closest-hit shader at binding 3.
    /// Layout: faces[i] = [r, g, b, normal_idx as f32], where i = primitive_id / 2.
    pub face_buf: vk::Buffer,
    pub face_mem: vk::DeviceMemory,
}

impl AccelStructures {
    /// Build a single triangle-mesh BLAS from the terrain mesh and a one-instance TLAS.
    pub unsafe fn build(
        instance:  &ash::Instance,
        device:    &ash::Device,
        phys:      vk::PhysicalDevice,
        queue:     vk::Queue,
        cmd_pool:  vk::CommandPool,
        accel_ext: &khr::acceleration_structure::Device,
        mesh:      &TerrainMesh,
    ) -> Result<Self> {
        // ── Face-data SSBO ─────────────────────────────────────────────────────
        // Uploaded once at startup; read-only from the GPU side.
        let (face_buf, face_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &mesh.face_data,
        )?;

        // ── Vertex + index buffers (build inputs, discarded after BLAS build) ──
        let (vtx_buf, vtx_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &mesh.vertices,
        )?;
        let (idx_buf, idx_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            &mesh.indices,
        )?;

        let vtx_addr = vk::DeviceOrHostAddressConstKHR {
            device_address: device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(vtx_buf),
            ),
        };
        let idx_addr = vk::DeviceOrHostAddressConstKHR {
            device_address: device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(idx_buf),
            ),
        };

        let triangle_count = (mesh.indices.len() / 3) as u32;
        let vertex_count   = mesh.vertices.len() as u32;

        // ── BLAS (single triangle geometry) ───────────────────────────────────
        let blas_geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vtx_addr)
                    .vertex_stride(std::mem::size_of::<[f32; 3]>() as vk::DeviceSize)
                    .max_vertex(vertex_count - 1)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(idx_addr),
            });

        let blas_geometries = [blas_geometry];
        let mut blas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&blas_geometries);

        let mut blas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        accel_ext.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &blas_build_info,
            &[triangle_count],
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

        blas_build_info.dst_acceleration_structure = blas;
        blas_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(blas_scratch_buf),
            ),
        };

        let blas_ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count:  triangle_count,
            primitive_offset: 0,
            first_vertex:     0,
            transform_offset: 0,
        }];

        {
            let cb = begin_one_shot(device, cmd_pool)?;
            accel_ext.cmd_build_acceleration_structures(cb, &[blas_build_info], &[&blas_ranges]);
            // Barrier: AS write → AS read (required before TLAS build)
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
        }

        device.destroy_buffer(blas_scratch_buf, None);
        device.free_memory(blas_scratch_mem, None);
        // Vertex + index data baked into the BLAS; buffers no longer needed.
        device.destroy_buffer(vtx_buf, None);
        device.free_memory(vtx_mem, None);
        device.destroy_buffer(idx_buf, None);
        device.free_memory(idx_mem, None);

        let blas_addr = accel_ext.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(blas),
        );

        // ── TLAS (single identity-transform instance referencing the BLAS) ─────
        let tlas_instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                // Row-major 3×4 identity matrix
                matrix: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_addr,
            },
        };

        let (inst_buf, inst_mem) = upload_buffer(
            instance, device, phys,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            std::slice::from_ref(&tlas_instance),
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

        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        accel_ext.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &[1u32],
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

        tlas_build_info.dst_acceleration_structure = tlas;
        tlas_build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(tlas_scratch_buf),
            ),
        };

        let tlas_ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count:  1,
            primitive_offset: 0,
            first_vertex:     0,
            transform_offset: 0,
        }];

        {
            let cb = begin_one_shot(device, cmd_pool)?;
            accel_ext.cmd_build_acceleration_structures(cb, &[tlas_build_info], &[&tlas_ranges]);
            end_one_shot(device, cmd_pool, queue, cb)?;
        }

        device.destroy_buffer(tlas_scratch_buf, None);
        device.free_memory(tlas_scratch_mem, None);
        device.destroy_buffer(inst_buf, None);
        device.free_memory(inst_mem, None);

        Ok(Self {
            blas, blas_buffer, blas_mem,
            tlas, tlas_buffer, tlas_mem,
            face_buf, face_mem,
        })
    }

    pub unsafe fn destroy(
        &self,
        device:    &ash::Device,
        accel_ext: &khr::acceleration_structure::Device,
    ) {
        accel_ext.destroy_acceleration_structure(self.tlas, None);
        device.destroy_buffer(self.tlas_buffer, None);
        device.free_memory(self.tlas_mem, None);

        accel_ext.destroy_acceleration_structure(self.blas, None);
        device.destroy_buffer(self.blas_buffer, None);
        device.free_memory(self.blas_mem, None);

        device.destroy_buffer(self.face_buf, None);
        device.free_memory(self.face_mem, None);
    }
}

#[allow(dead_code)]
pub(crate) fn aligned(value: usize, align_to: usize) -> usize {
    align_up(value, align_to)
}
