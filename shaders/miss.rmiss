#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    vec3  dir = normalize(gl_WorldRayDirectionEXT);
    float t   = clamp(0.5 * (dir.y + 1.0), 0.0, 1.0);
    // Horizon matches the fog colour used in closest_hit (0.80, 0.82, 0.88);
    // zenith fades to sky blue, squared for a more natural gradient.
    payload = mix(vec3(0.80, 0.82, 0.88), vec3(0.28, 0.47, 0.82), t * t);
}
