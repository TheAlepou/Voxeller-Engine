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
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
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
        // Request the first frame; winit on macOS doesn't always auto-fire
        // RedrawRequested after resumed(), so we kick the loop manually.
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
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App { backend: None };
    event_loop.run_app(&mut app).expect("Event loop error");
}
