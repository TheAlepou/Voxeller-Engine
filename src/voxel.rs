//! Procedural terrain generation.
//!
//! Produces a 48×48 block world with:
//!  - Multi-octave value noise heightmap with ridge detail
//!  - Three surface layers (grass / mossy-rock, dirt, stone variants)
//!  - Scattered autumn trees with randomised leaf colour
//!
//! Call `generate_terrain(seed)` for a randomised world, then
//! `build_terrain_mesh(&voxels)` to get a face-culled triangle mesh
//! ready for upload to a Metal BLAS.

use std::collections::{HashMap, HashSet};

/// An axis-aligned voxel with a flat-shaded colour.
pub struct Voxel {
    pub position: [i32; 3],
    pub color:    [f32; 3],
}

impl Voxel {
    #[allow(dead_code)]
    pub fn aabb_min(&self) -> [f32; 3] {
        self.position.map(|v| v as f32)
    }
}

/// A face-culled triangle mesh for the entire terrain.
///
/// Each face consists of 2 triangles (4 vertices, 6 indices).
/// `face_data[i]` stores `[r, g, b, normal_idx as f32]` for face `i`,
/// looked up in the ray-tracing shader via `primitive_id / 2`.
pub struct TerrainMesh {
    pub vertices:  Vec<[f32; 3]>,
    pub indices:   Vec<u32>,
    /// Per-face colour + normal index: `[r, g, b, normal_idx_f32]`.
    pub face_data: Vec<[f32; 4]>,
}

// ── Face definitions ──────────────────────────────────────────────────────────
// Each entry: (neighbour_offset, normal_idx, 4 corner positions in local space)
// normal_idx matches FACE_NORMALS in the MSL shader:
//   0 = –Z  1 = +Z  2 = –X  3 = +X  4 = –Y  5 = +Y
#[rustfmt::skip]
const FACE_DEFS: [([i32; 3], u32, [[f32; 3]; 4]); 6] = [
    ([ 0,  0, -1], 0, [[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),  // –Z
    ([ 0,  0,  1], 1, [[1.,0.,1.],[0.,0.,1.],[0.,1.,1.],[1.,1.,1.]]),  // +Z
    ([-1,  0,  0], 2, [[0.,0.,1.],[0.,0.,0.],[0.,1.,0.],[0.,1.,1.]]),  // –X
    ([ 1,  0,  0], 3, [[1.,0.,0.],[1.,0.,1.],[1.,1.,1.],[1.,1.,0.]]),  // +X
    ([ 0, -1,  0], 4, [[0.,0.,0.],[0.,0.,1.],[1.,0.,1.],[1.,0.,0.]]),  // –Y
    ([ 0,  1,  0], 5, [[0.,1.,0.],[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]]),  // +Y
];

// ── Noise helpers ─────────────────────────────────────────────────────────────

/// Deterministic hash of two integers → [0, 1].
fn hash(x: i32, z: i32) -> f32 {
    let h = x.wrapping_mul(1_619).wrapping_add(z.wrapping_mul(31_337));
    let h = h ^ (h >> 13);
    let h = h.wrapping_mul(-1_640_531_527_i32);
    let h = h ^ (h >> 16);
    (h as u32 as f32) / u32::MAX as f32
}

fn smooth(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }

/// Bilinear value noise at (x, z).
fn noise(x: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iz = z.floor() as i32;
    let fx = smooth(x - x.floor());
    let fz = smooth(z - z.floor());
    let ab = hash(ix, iz)     + (hash(ix + 1, iz)     - hash(ix, iz))     * fx;
    let cd = hash(ix, iz + 1) + (hash(ix + 1, iz + 1) - hash(ix, iz + 1)) * fx;
    ab + (cd - ab) * fz
}

/// Fractional Brownian Motion — sums `octaves` noise layers, normalised to [0, 1].
fn fbm(x: f32, z: f32, octaves: u32) -> f32 {
    let (mut v, mut a, mut f, mut m) = (0.0f32, 1.0f32, 1.0f32, 0.0f32);
    for _ in 0..octaves {
        v += noise(x * f, z * f) * a;
        m += a;
        a *= 0.5;
        f *= 2.0;
    }
    v / m
}

fn rgb(r: f32, g: f32, b: f32) -> [f32; 3] { [r, g, b] }

// ── Terrain ───────────────────────────────────────────────────────────────────

/// World footprint side length in voxels.
pub const WORLD: i32 = 48;

/// Generate the full terrain world.
///
/// `seed` shifts the noise domain so every value produces a different landscape.
/// Duplicate positions are deduplicated (trees may overwrite surface cells).
pub fn generate_terrain(seed: u32) -> Vec<Voxel> {
    // Domain offsets derived from the seed — large enough to fully shift the noise
    let sx = (seed & 0xFFFF) as f32;
    let sz = (seed >> 16) as f32;

    let mut grid: HashMap<[i32; 3], [f32; 3]> = HashMap::new();

    for x in 0..WORLD {
        for z in 0..WORLD {
            let (fx, fz) = (x as f32, z as f32);

            // Height field: large hills + fine detail + sharp ridges
            let base   = fbm(fx / 38.0 + sx,                 fz / 38.0 + sz,                 4);
            let detail = fbm(fx / 11.0 + 5.3 + sx,           fz / 11.0 + 5.3 + sz,           3);
            let ridge  = (0.5 - (fbm(fx / 22.0 + 11.0 + sx,  fz / 22.0 + 11.0 + sz,  2) - 0.5).abs()) * 2.0;
            let h = ((base * 10.0 + detail * 4.0 + ridge * 4.5 + 2.5) as i32).clamp(2, 18);

            // Fill the column from y = 0 up to y = h
            for y in 0..=h {
                let gv = noise(fx * 3.1         + sx, fz * 3.1 + sz);
                let dv = noise(fx * 2.3 + 7.0   + sx, fz * 2.3 + 7.0 + sz);
                let sv = noise(fx * 1.4 + 100.0  + sx, fz * 1.4 + y as f32 * 0.7 + sz);

                let col = if y == h {
                    // Top surface — grass at low alt, mossy / bare rock at high alt
                    if h >= 15 {
                        if gv > 0.5 { rgb(0.52, 0.47, 0.38) } else { rgb(0.58, 0.50, 0.40) }
                    } else if h >= 10 {
                        if gv > 0.5 { rgb(0.35, 0.60, 0.28) } else { rgb(0.42, 0.58, 0.33) }
                    } else if gv > 0.5 {
                        rgb(0.30, 0.65, 0.24)
                    } else {
                        rgb(0.38, 0.70, 0.28)
                    }
                } else if y >= h - 2 {
                    // Dirt band
                    if dv > 0.5 { rgb(0.58, 0.42, 0.28) } else { rgb(0.50, 0.36, 0.22) }
                } else {
                    // Stone — orange tones near surface, cool grey-brown deeper
                    if sv > 0.6      { rgb(0.64, 0.50, 0.36) }
                    else if sv > 0.3 { rgb(0.56, 0.43, 0.30) }
                    else             { rgb(0.46, 0.36, 0.24) }
                };
                grid.insert([x, y, z], col);
            }

            // Trees — sparse, mid-altitude only
            let tn = noise(fx * 7.31 + 1.7 + sx, fz * 7.31 + 1.7 + sz);
            if (5..=13).contains(&h) && tn > 0.82 {
                let trunk_h = 2 + (noise(fx * 13.7 + sx, fz * 13.7 + sz) * 2.0) as i32;

                // Trunk
                for ty in 1..=trunk_h {
                    grid.insert([x, h + ty, z], rgb(0.36, 0.25, 0.14));
                }

                // Pick leaf colour: red / orange / yellow / green
                let lv = noise(fx * 5.1 + 2.3 + sx, fz * 5.1 + 2.3 + sz);
                let leaf = if lv < 0.25      { rgb(0.86, 0.22, 0.14) }
                           else if lv < 0.50 { rgb(0.88, 0.54, 0.10) }
                           else if lv < 0.75 { rgb(0.86, 0.73, 0.14) }
                           else               { rgb(0.26, 0.64, 0.20) };

                // Crown: 5×5 at base, 5×5 middle, 3×3 cap
                let top = h + trunk_h;
                for (dy, r) in [(0i32, 2i32), (1, 2), (2, 1)] {
                    for dx in -r..=r {
                        for dz in -r..=r {
                            let (lx, lz, ly) = (x + dx, z + dz, top + dy);
                            if (0..WORLD).contains(&lx) && (0..WORLD).contains(&lz) {
                                grid.insert([lx, ly, lz], leaf);
                            }
                        }
                    }
                }
            }
        }
    }

    grid.into_iter()
        .map(|(position, color)| Voxel { position, color })
        .collect()
}

/// Build a face-culled triangle mesh from the terrain voxel list.
///
/// Only faces whose neighbour cell is air (absent from `voxels`) are emitted,
/// eliminating all invisible interior geometry. Returns a [`TerrainMesh`] whose
/// buffers can be uploaded directly to a Metal BLAS.
pub fn build_terrain_mesh(voxels: &[Voxel]) -> TerrainMesh {
    let occupied: HashSet<[i32; 3]> = voxels.iter().map(|v| v.position).collect();

    let mut vertices:  Vec<[f32; 3]> = Vec::new();
    let mut indices:   Vec<u32>      = Vec::new();
    let mut face_data: Vec<[f32; 4]> = Vec::new();

    for voxel in voxels {
        let [px, py, pz] = voxel.position;
        let base = [px as f32, py as f32, pz as f32];

        for (offset, normal_idx, corners) in &FACE_DEFS {
            let neighbour = [px + offset[0], py + offset[1], pz + offset[2]];
            if occupied.contains(&neighbour) {
                continue; // Hidden by adjacent solid voxel — skip
            }

            // Emit quad as two triangles: [0,1,2] and [0,2,3]
            let v0 = vertices.len() as u32;
            for corner in corners {
                vertices.push([
                    base[0] + corner[0],
                    base[1] + corner[1],
                    base[2] + corner[2],
                ]);
            }
            indices.extend_from_slice(&[v0, v0 + 1, v0 + 2, v0, v0 + 2, v0 + 3]);
            face_data.push([
                voxel.color[0],
                voxel.color[1],
                voxel.color[2],
                *normal_idx as f32,
            ]);
        }
    }

    TerrainMesh { vertices, indices, face_data }
}
