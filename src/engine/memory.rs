use anyhow::{bail, Result};
use ash::{vk, Device, Instance};

/// Find a memory type index matching `type_bits` and `required` property flags.
pub unsafe fn find_memory_type(
    instance: &Instance,
    phys: vk::PhysicalDevice,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let props = instance.get_physical_device_memory_properties(phys);
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(required)
        {
            return Ok(i);
        }
    }
    bail!("No suitable memory type found")
}

/// Create a buffer + backing device memory and bind them.
///
/// Pass `SHADER_DEVICE_ADDRESS` in `usage` to enable `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`.
pub unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    phys: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mem_flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buf = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;

    let reqs = device.get_buffer_memory_requirements(buf);
    let type_idx = find_memory_type(instance, phys, reqs.memory_type_bits, mem_flags)?;

    let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default();
    if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
        alloc_flags = alloc_flags.flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    }

    let mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_idx)
            .push_next(&mut alloc_flags),
        None,
    )?;
    device.bind_buffer_memory(buf, mem, 0)?;
    Ok((buf, mem))
}

/// Upload `data` into a freshly allocated HOST_VISIBLE buffer.
pub unsafe fn upload_buffer<T: Copy>(
    instance: &Instance,
    device: &Device,
    phys: vk::PhysicalDevice,
    usage: vk::BufferUsageFlags,
    data: &[T],
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = std::mem::size_of_val(data) as vk::DeviceSize;
    let (buf, mem) = create_buffer(
        instance,
        device,
        phys,
        size,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let ptr = device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())? as *mut T;
    ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
    device.unmap_memory(mem);
    Ok((buf, mem))
}

/// Create a `VK_IMAGE_LAYOUT_UNDEFINED` device-local image + view.
pub unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    phys: vk::PhysicalDevice,
    extent: vk::Extent2D,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        None,
    )?;

    let reqs = device.get_image_memory_requirements(image);
    let type_idx = find_memory_type(
        instance,
        phys,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let mem = device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_idx),
        None,
    )?;
    device.bind_image_memory(image, mem, 0)?;

    let view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            ),
        None,
    )?;
    Ok((image, mem, view))
}

/// Allocate + begin a one-shot command buffer on `pool`.
pub unsafe fn begin_one_shot(device: &Device, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
    let cb = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];
    device.begin_command_buffer(
        cb,
        &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;
    Ok(cb)
}

/// End, submit, and wait for a one-shot command buffer, then free it.
pub unsafe fn end_one_shot(
    device: &Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    cb: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(cb)?;
    let cbs = [cb];
    device.queue_submit(
        queue,
        &[vk::SubmitInfo::default().command_buffers(&cbs)],
        vk::Fence::null(),
    )?;
    device.queue_wait_idle(queue)?;
    device.free_command_buffers(pool, &cbs);
    Ok(())
}

#[inline]
pub fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}
