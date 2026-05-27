/// An axis-aligned voxel cube at integer grid coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Voxel {
    pub position: [i32; 3],
}

impl Voxel {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { position: [x, y, z] }
    }

    pub fn aabb_min(&self) -> [f32; 3] {
        self.position.map(|v| v as f32)
    }

    pub fn aabb_max(&self) -> [f32; 3] {
        self.position.map(|v| v as f32 + 1.0)
    }
}

/// The four demo voxels shown on startup.
pub fn demo_voxels() -> Vec<Voxel> {
    vec![
        Voxel::new(0, 0, 0),
        Voxel::new(2, 0, 0),
        Voxel::new(0, 2, 0),
        Voxel::new(2, 2, 0),
    ]
}
