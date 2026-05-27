//! Thin adapter that wraps the Vulkan engine as a RenderBackend (non-macOS).
use std::sync::Arc;
use anyhow::Result;
use winit::keyboard::KeyCode;
use winit::window::Window;
use super::RenderBackend;

pub struct VulkanBackend(crate::engine::Engine);

impl VulkanBackend {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        Ok(Self(crate::engine::Engine::new(window)?))
    }
}

impl RenderBackend for VulkanBackend {
    fn render(&mut self) -> Result<()>  { self.0.render() }
    fn handle_resize(&mut self)         { self.0.handle_resize() }
    fn request_redraw(&self)            { self.0.request_redraw() }

    fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        self.0.handle_key(key, pressed);
    }
    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.0.handle_mouse_motion(dx, dy);
    }
    fn handle_mouse_button(&mut self, pressed: bool) {
        self.0.handle_mouse_button(pressed);
    }
    fn handle_scroll(&mut self, delta: f32) {
        self.0.handle_scroll(delta);
    }
}
