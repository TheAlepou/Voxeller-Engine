#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

// Per-face colour + normal index uploaded by accel.rs.
// faces[i] = vec4(r, g, b, normal_idx)  where i = gl_PrimitiveID / 2
layout(std430, set = 0, binding = 3) readonly buffer FaceData {
    vec4 faces[];
};

// Must match FACE_DEFS normal ordering in voxel.rs:
//   0 = -Z  1 = +Z  2 = -X  3 = +X  4 = -Y  5 = +Y
const vec3 FACE_NORMALS[6] = vec3[6](
    vec3( 0,  0, -1),
    vec3( 0,  0,  1),
    vec3(-1,  0,  0),
    vec3( 1,  0,  0),
    vec3( 0, -1,  0),
    vec3( 0,  1,  0)
);

void main() {
    // Two triangles share a face entry; integer division maps both to the same face.
    uint  fi   = uint(gl_PrimitiveID) / 2u;
    vec4  fe   = faces[fi];
    vec3  base = fe.xyz;
    uint  ni   = uint(fe.w);

    vec3 n = FACE_NORMALS[ni];
    // Ensure the normal faces the incoming ray (back-face flipping).
    if (dot(n, gl_WorldRayDirectionEXT) > 0.0) n = -n;

    const vec3  sun  = normalize(vec3(0.8, 1.8, 0.6));
    const float diff = max(dot(n, sun), 0.0);
    // Subtle AO darkening on the bottom faces of voxels (normal index 4 = -Y).
    const float ao   = (ni == 4u) ? 0.55 : 1.0;
    vec3 color = base * (0.18 + 0.82 * diff) * ao;

    // Exponential fog matching the Metal backend.
    float fog = 1.0 - exp(-gl_HitTEXT * 0.006);
    color = mix(color, vec3(0.80, 0.82, 0.88), fog * fog);

    payload = color;
}
