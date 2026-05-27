// Bring objc macros (msg_send!, class!, sel!, …) into scope for the Metal backend.
#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

mod backend;
mod voxel;

#[cfg(not(target_os = "macos"))]
mod engine;

use backend::{Backend, RenderBackend};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};

struct App {
    backend: Option<Box<dyn RenderBackend>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Voxeller Engine")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let backend: Box<dyn RenderBackend> = match Backend::new(window) {
            Ok(b) => Box::new(b),
            Err(e) => {
                eprintln!("Backend init failed: {e}");
                event_loop.exit();
                return;
            }
        };
        self.backend = Some(backend);
        // Kick off the render loop — winit on macOS doesn't always auto-fire
        // RedrawRequested after resumed().
        if let Some(b) = &self.backend {
            b.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::RedrawRequested => {
                if let Some(b) = &mut self.backend {
                    if let Err(e) = b.render() {
                        eprintln!("Render error: {e}");
                        event_loop.exit();
                    }
                    b.request_redraw();
                }
            }

            WindowEvent::Resized(_) => {
                if let Some(b) = &mut self.backend {
                    b.handle_resize();
                }
            }

            // Keyboard: forward every physical key press/release to the backend.
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if let Some(b) = &mut self.backend {
                        b.handle_key(key, event.state == ElementState::Pressed);
                    }
                }
            }

            // Left-click: capture / toggle mouse lock.
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                if let Some(b) = &mut self.backend {
                    b.handle_mouse_button(state == ElementState::Pressed);
                }
            }

            // Scroll wheel: adjust fly speed.
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y)  => y,
                    MouseScrollDelta::PixelDelta(pos)  => pos.y as f32 / 40.0,
                };
                if let Some(b) = &mut self.backend {
                    b.handle_scroll(lines);
                }
            }

            _ => {}
        }
    }

    // Device events fire regardless of window focus and work even with a
    // locked cursor, making them the right source for FPS mouse-look deltas.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if let Some(b) = &mut self.backend {
                b.handle_mouse_motion(dx, dy);
            }
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App { backend: None };
    event_loop.run_app(&mut app).expect("Event loop error");
}
