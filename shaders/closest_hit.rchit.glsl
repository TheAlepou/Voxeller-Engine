#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec3 hitNormal;

void main() {
    // gl_InstanceCustomIndexEXT carries a per-voxel index set on the TLAS instance
    const vec3 palette[4] = vec3[4](
        vec3(0.88, 0.22, 0.20),  // red
        vec3(0.22, 0.78, 0.28),  // green
        vec3(0.22, 0.32, 0.90),  // blue
        vec3(0.92, 0.80, 0.20)   // yellow
    );

    vec3 baseColor = palette[gl_InstanceCustomIndexEXT & 3];
    vec3 normal    = normalize(hitNormal);
    vec3 lightDir  = normalize(vec3(1.5, 3.0, 2.0));

    float ambient  = 0.20;
    float diffuse  = max(dot(normal, lightDir), 0.0);

    payload = baseColor * (ambient + 0.80 * diffuse);
}
