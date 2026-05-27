use winit::keyboard::KeyCode;

pub trait RenderBackend {
    fn render(&mut self) -> anyhow::Result<()>;
    fn handle_resize(&mut self);
    fn request_redraw(&self);

    // Input — backends that don't need these can ignore them.
    fn handle_key(&mut self, _key: KeyCode, _pressed: bool) {}
    fn handle_mouse_motion(&mut self, _dx: f64, _dy: f64) {}
    fn handle_mouse_button(&mut self, _pressed: bool) {}
    /// Scroll-wheel delta (positive = up / zoom-in). Used to change move speed.
    fn handle_scroll(&mut self, _delta: f32) {}
}

#[cfg(target_os = "macos")]
mod metal_rt;
#[cfg(target_os = "macos")]
pub use metal_rt::MetalBackend as Backend;

#[cfg(not(target_os = "macos"))]
mod vulkan;
#[cfg(not(target_os = "macos"))]
pub use vulkan::VulkanBackend as Backend;
