//! Thin adapter that wraps the Vulkan engine as a RenderBackend (non-macOS).
use std::sync::Arc;
use anyhow::Result;
use winit::window::Window;
use super::RenderBackend;

pub struct VulkanBackend(crate::engine::Engine);

impl VulkanBackend {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        Ok(Self(crate::engine::Engine::new(window)?))
    }
}

impl RenderBackend for VulkanBackend {
    fn render(&mut self) -> Result<()>        { self.0.render() }
    fn handle_resize(&mut self)               { self.0.handle_resize() }
    fn request_redraw(&self)                  { self.0.request_redraw() }
}
