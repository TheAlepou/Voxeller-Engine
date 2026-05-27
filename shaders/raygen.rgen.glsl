#version 460
#extension GL_EXT_ray_tracing : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D outputImage;
layout(binding = 2, set = 0) uniform CameraData {
    mat4 viewInverse;
    mat4 projInverse;
} camera;

layout(location = 0) rayPayloadEXT vec3 payload;

void main() {
    const vec2 pixel = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 uv    = pixel / vec2(gl_LaunchSizeEXT.xy);

    // Flip Y so world +Y is up
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    vec4 origin    = camera.viewInverse * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 target    = camera.projInverse * vec4(ndc, 1.0, 1.0);
    target        /= target.w;
    vec4 direction = camera.viewInverse * vec4(normalize(target.xyz), 0.0);

    payload = vec3(0.0);
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0, 0,      // sbtRecordOffset, sbtRecordStride
        0,         // missIndex
        origin.xyz,
        0.001,
        direction.xyz,
        10000.0,
        0          // payload location
    );

    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(payload, 1.0));
}
