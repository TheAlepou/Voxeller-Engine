#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    // Sky gradient: white horizon, blue zenith
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float t  = 0.5 * (dir.y + 1.0);
    payload  = mix(vec3(0.95, 0.95, 1.0), vec3(0.25, 0.45, 0.85), t);
}
