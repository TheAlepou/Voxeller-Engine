#version 460
#extension GL_EXT_ray_tracing : require

// Communicated to the closest-hit shader
hitAttributeEXT vec3 hitNormal;

void main() {
    // Every voxel instance shares a single unit-cube BLAS: [0,1]^3 in object space.
    // The TLAS instance transform positions it in the world.
    const vec3 aabbMin = vec3(0.0);
    const vec3 aabbMax = vec3(1.0);

    vec3 origin = gl_ObjectRayOriginEXT;
    vec3 dir    = gl_ObjectRayDirectionEXT;
    vec3 invDir = 1.0 / dir;

    vec3 t0 = (aabbMin - origin) * invDir;
    vec3 t1 = (aabbMax - origin) * invDir;

    vec3 tMin3 = min(t0, t1);
    vec3 tMax3 = max(t0, t1);

    float tEnter = max(max(tMin3.x, tMin3.y), tMin3.z);
    float tExit  = min(min(tMax3.x, tMax3.y), tMax3.z);

    if (tEnter > tExit || tExit < gl_RayTminEXT || tEnter > gl_RayTmaxEXT)
        return;

    float t = (tEnter >= gl_RayTminEXT) ? tEnter : tExit;
    if (t > gl_RayTmaxEXT) return;

    // Determine which face was hit and set outward normal
    vec3 eps = vec3(1e-4);
    vec3 hit = origin + dir * t;
    if      (abs(hit.x - aabbMin.x) < eps.x) hitNormal = vec3(-1, 0, 0);
    else if (abs(hit.x - aabbMax.x) < eps.x) hitNormal = vec3( 1, 0, 0);
    else if (abs(hit.y - aabbMin.y) < eps.y) hitNormal = vec3(0, -1, 0);
    else if (abs(hit.y - aabbMax.y) < eps.y) hitNormal = vec3(0,  1, 0);
    else if (abs(hit.z - aabbMin.z) < eps.z) hitNormal = vec3(0, 0, -1);
    else                                      hitNormal = vec3(0, 0,  1);

    reportIntersectionEXT(t, 0u);
}
