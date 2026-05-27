pub trait RenderBackend {
    fn render(&mut self) -> anyhow::Result<()>;
    fn handle_resize(&mut self);
    fn request_redraw(&self);
}

#[cfg(target_os = "macos")]
mod metal_rt;
#[cfg(target_os = "macos")]
pub use metal_rt::MetalBackend as Backend;

#[cfg(not(target_os = "macos"))]
mod vulkan;
#[cfg(not(target_os = "macos"))]
pub use vulkan::VulkanBackend as Backend;
