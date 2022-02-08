#version 450 core

/* ==================== Predefines ==================== */
// Math constants
#define M_EPSILON   0.01
#define M_PI        3.141592653589793116
#define M_2PI       6.283185307179586232
#define M_HALF_PI   1.570796326794896558
#define M_INV_PI    0.318309886183790684
#define M_INV_2PI   0.159154943091895342
#define M_INF       1.0e10

// BSDF types of surface.
#define SurfaceType uint
#define SURFACE_TYPE_DIFFUSE                1
#define SURFACE_TYPE_FRESNEL_REFLECTION     2
#define SURFACE_TYPE_FRESNEL_TRANSMISSION   4
#define SURFACE_TYPE_FRESNEL_DIELECTRIC     6
#define SURFACE_TYPE_MIRROR                 8

// Index of refraction.
#define IOR_VACUUM                          1.0
#define IOR_GLASS                           1.7

/* ==================== Types ==================== */
struct Camera {
    vec3  position;
    vec3  gaze;
    float fov;
    uvec2 resolution;
};

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Material {
    vec3        color;
    float       dummy;
    vec3        emission;
    SurfaceType surfaceType;
};

struct Sphere {
    vec3        position;
    float       radius;
    Material    material;
};

struct IntersectionRecord {
    uint        id;
    vec3        point;
    vec3        normal;
    vec3        surfaceColor;
    SurfaceType surfaceType;
};

struct Complex {
    float r, i;
};

/* ==================== Inputs ==================== */
layout(local_size_x = 8, local_size_y = 6) in;

layout(rgba32f, binding = 0) uniform image2D uImageOut;

layout(std430, binding = 1) readonly buffer SceneObjectBuffer {
    int size;
    Sphere objects[];
} sceneObjects;

layout(std430, binding = 2) readonly buffer SceneLightBuffer {
    int size;
    int lights[];
} sceneLights;

/* ==================== Uniforms ==================== */
uniform Camera uCamera;
uniform int uMaxBounce;
uniform int uRRStartBounce;
uniform int uSpp;

/* ==================== PCG Random ==================== */
// [1] Jarzynski, M., & Olano, M. (2020). Hash Functions for GPU Rendering.
uint _pcgState1D;
uint pcgRandom1D() {
    _pcgState1D = _pcgState1D * 747796405U + 2891336453U;
    uint shifted = ((((_pcgState1D >> 28U)) + 4U) ^ _pcgState1D) * 277803737U;
    return (shifted >> 22U) ^ shifted;
}

uvec2 _pcgState2D;
uvec2 pcgRandom2D() {
    uvec2 v = _pcgState2D * 1664525U + 1013904223U;
    v.x += v.y * 1664525U;
    v.y += v.x * 1664525U;
    v = v ^ (v >> 16U);
    v.x += v.y * 1664525U;
    v.y += v.x * 1664525U;
    v = v ^ (v >> 16U);
    _pcgState2D = v;
    return v;
}

uvec3 _pcgState3D;
uvec3 pcgRandom3D() {
    uvec3 v = _pcgState3D * 1664525U + 1013904223U;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16U;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    _pcgState3D = v;
    return v;
}

float random1D() {
    return float(pcgRandom1D()) / 4294967295.0;
}

vec2 random2D() {
    return vec2(pcgRandom2D()) / 4294967295.0;
}

vec3 random3D() {
    return vec3(pcgRandom3D()) / 4294967295.0;
}

void randomSeed(const in uvec3 seed) {
    _pcgState1D = seed.x;
    _pcgState2D = seed.xy;
    _pcgState3D = seed;
}

/* ==================== Ray Intersection ==================== */
bool intersect(const in Sphere sphere, const in Ray ray, out float t) {
    vec3 op = sphere.position - ray.origin;
    float halfb = dot(ray.direction, op);
    float det = halfb * halfb - dot(op, op) + sphere.radius * sphere.radius;

    if (det < 0.0)
        return false;
    det = sqrt(det);
    float result = halfb - det;
    if (result > M_EPSILON)
        t = result;
    else if ((result = halfb + det) > M_EPSILON)
        t = result;
    else
        return false;    
    return true;
}

bool intersectScene(const in Ray ray, out IntersectionRecord intersection) {
    float t = M_INF;
    int id = -1;
    for (int i = 0; i < sceneObjects.size; ++i) {
        float tmp = 0.0;
        if  (intersect(sceneObjects.objects[i], ray, tmp) && tmp < t) {
            t = tmp;
            id = i;
        }
    }
    if (id == -1)
        return false;

    // Compute the information of intersection.
    vec3 point = ray.origin + t * ray.direction;
    vec3 normal = normalize(point - sceneObjects.objects[id].position);
    vec3 color = sceneObjects.objects[id].material.color;
    SurfaceType type = sceneObjects.objects[id].material.surfaceType;
    intersection = IntersectionRecord(id, point, normal, color, type);
    return true;
}

bool isIntersectedScene(const in Ray ray, const in float maxt) {
    float t;
    for (int i = 0; i < sceneObjects.size; ++i) {
        if (intersect(sceneObjects.objects[i], ray, t) && t < maxt)
            return true;
    }
    return false;
}

/* ==================== Sampling ==================== */
vec2 sampleUniformDisk(const in vec2 u) {
    float r = sqrt(u.x);
    float theta = M_2PI * u.y;
    return vec2(r * cos(theta), r * sin(theta));
}

vec3 sampleUniformSphere(const in vec2 u) {
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = M_2PI * u.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

vec3 sampleUniformHemiSphere(const in vec2 u) {
    float z = u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = M_2PI * u.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

vec3 sampleCosineWeightedHemisphere(const in vec2 u) {
    vec2 v = sampleUniformDisk(u);
    return vec3(v.x, v.y, sqrt(max(0.0, 1.0 - v.x * v.x - v.y * v.y)));
}

/* ==================== Misc ==================== */
bool isBlack(const in vec3 c) {
    return all(equal(c, vec3(0.0)));
}

void swap(inout float a, inout float b) {
    float tmp = a;
    a = b;
    b = tmp;
}

Complex complexDiv(in Complex a, in Complex b) {
    float invDenom = 1.0 / (b.r * b.r + b.i * b.i);
    return Complex(
        (a.r * b.r + a.i * b.i) * invDenom,
        (a.i * b.r - a.r * b.i) * invDenom
    );
}

bool fresnel(in float cosI, in float etaI, in float etaT,
             out float fr, out float ft) {
    float eta = etaI / etaT;
    if (cosI < 0.0) {
        eta = etaT / etaI;
        cosI = abs(cosI);
    }

    float sinI = sqrt(max(0.0, 1.0 - cosI * cosI));
    float sinT = eta * sinI;

    if (sinT >= 1.0) {
        fr = 1.0;
        ft = 0.0;
        return true;
    }
    float cosT = sqrt(1.0 - sinT * sinT);

    float parl = (cosI - eta * cosT) / (cosI + eta * cosT);
    float perp = (eta * cosI - cosT) / (eta * cosI + cosT);
    fr = (parl * parl + perp * perp) * 0.5;
    ft = 1.0 - fr;
    return false;
}

bool schlickFresnel(in float cosI, in float etaI, in float etaT,
                   out float fr, out float ft) {
    float eta = etaI / etaT;
    if (cosI < 0.0) {
        eta = etaT / etaI;
        cosI = abs(cosI);
    }

    float sinI = sqrt(max(0.0, 1.0 - cosI * cosI));
    float sinT = eta * sinI;

    if (sinT >= 1.0) {
        fr = 1.0;
        ft = 0.0;
        return true;
    }
    float r0 = (eta - 1.0) / (eta + 1.0);
    r0 = r0 * r0;
    float t = 1.0 - cosI;
    fr = r0 + (1.0 - r0) * t * t * t * t * t;
    ft = 1.0 - fr;
    return false;
}

/* ==================== Light Sampling ==================== */
vec3 sampleAllLights(in vec3 p, in vec3 n) {
    vec3 c = vec3(0.0);
    for (int i = 0; i < sceneLights.size; ++i) {
        Sphere light = sceneObjects.objects[sceneLights.lights[i]];
        Ray shadowRay;
        shadowRay.origin = p;
        
        vec3 sampledN = sampleUniformSphere(random2D());
        vec3 sampled = sampledN * light.radius + light.position;
        shadowRay.direction = normalize(sampled - p);

        // Avoid self-intersection.
        shadowRay.origin += shadowRay.direction * 0.0001;
        
        vec3 wo = -shadowRay.direction;
        float cosWo = dot(wo, sampledN);
        if (cosWo < 0.0)
            continue;
        float cosWi = dot(shadowRay.direction, faceforward(n, wo, n));
        float len = dot(sampled - p, sampled - p);
        bool occluded = cosWi < 0.0 || isIntersectedScene(shadowRay, sqrt(len) - M_EPSILON);
        if (occluded)
            continue;
        float g = abs(cosWi * cosWo) / len;
        //pdf = 1.0 / (4.0 * M_2PI * light.radius * light.radius);
        c += light.material.emission * 4.0 * M_PI * light.radius * light.radius * g;
    }
    return c;
}

/* ==================== BSDF Sampling ==================== */
// Lambertian diffuse reflection.
vec3 sampleDiffuse(const in vec3 wo, const in vec3 n, const in vec3 c,
                   out vec3 wi, out float pdf) {
    vec3 w = faceforward(n, -wo, n);
    vec3 u = normalize(abs(w.x) > M_EPSILON 
        ? vec3(w.z, 0.0, -w.x) 
        : vec3(0.0, -w.z, w.y));
    vec3 v = cross(w, u);

    vec3 d = sampleCosineWeightedHemisphere(random2D());
    wi = normalize(u * d.x + v * d.y + w * d.z);
    pdf = abs(dot(wi, n)) * M_INV_PI;
    return c * M_INV_PI;
}

// Mirror reflection.
vec3 sampleMirrorReflection(const in vec3 wo, const in vec3 n, const in vec3 c,
                              out vec3 wi, out float pdf) {
    wi = -(reflect(wo, faceforward(n, wo, n)));
    pdf = 1.0;
    return c / abs(dot(wi, n));
}

// Fresnel specular reflection.
vec3 sampleFresnelReflection(const in vec3 wo, const in vec3 n, const in vec3 c,
                             out vec3 wi, out float pdf) {
    const float etaI = IOR_VACUUM;
    const float etaT = IOR_GLASS;

    wi = -(reflect(wo, faceforward(n, wo, n)));
    pdf = 1.0;
    float cosI = dot(wi, n);
    float fr, ft;
    fresnel(cosI, etaI, etaT, fr, ft);

    return c * fr / abs(cosI);
}

// Fresnel specular transmission.
vec3 sampleFresnelTransmission(const in vec3 wo, const in vec3 n, const in vec3 c,
                                out vec3 wi, out float pdf) {
    const float etaI = IOR_VACUUM;
    const float etaT = IOR_GLASS;

    float cosO = dot(wo, n);
    float eta = cosO > 0.0 ? etaI / etaT : etaT / etaI;
    wi = normalize(-refract(wo, faceforward(n, wo, n), eta));
    pdf = 1.0;
    
    float cosI = dot(wi, n);
    float fr, ft;
    schlickFresnel(cosI, etaI, etaT, fr, ft);
    return c * ft * eta * eta / abs(cosI);
}

// Fresnel for dielectric material.
vec3 sampleFresnelDielectric(const in vec3 wo, const in vec3 n, const in vec3 c,
                             out vec3 wi, out float pdf) {
    const float etaI = IOR_VACUUM;
    const float etaT = IOR_GLASS;

    float fr, ft;
    float cosO = dot(wo, n);
    fresnel(cosO, etaI, etaT, fr, ft);

    if (random1D() < fr) {
        wi = -reflect(wo, faceforward(n, wo, n));
        pdf = fr;
        return c * fr / abs(dot(wi, n));
    } else {
        float eta = cosO > 0.0 ? etaI / etaT : etaT / etaI;
        wi = normalize(-refract(wo, faceforward(n, wo, n), eta));
        pdf = ft;
        return c * ft * eta * eta / abs(dot(wi, n));
    }
}

/* ==================== Path Tracing ==================== */
vec3 radiance(in Ray ray) {
    vec3 l = vec3(0.0);
    vec3 throughput = vec3(1.0);
    bool isSpecularBounce = true;

    for (int bounce = 0; bounce < uMaxBounce; ++bounce) {
        // Compute surface intersection.
        IntersectionRecord rec;
        bool intersected = intersectScene(ray, rec);
        if (!intersected)
            return l;

        vec3 wo = -ray.direction;
        vec3 wi, li;
        float pdf;

        // Add emitted light, if intersected with a light.
        vec3 le = sceneObjects.objects[rec.id].material.emission;
        if (!isBlack(le)) {
            if (isSpecularBounce) {
                le *= abs(dot(wo, rec.normal)) * throughput;
                l += le;
            }
            return l;
        }

        // Sample next direction.
        SurfaceType surfaceType = rec.surfaceType;
        if (surfaceType == SURFACE_TYPE_DIFFUSE) {
            li = sampleDiffuse(wo, rec.normal, rec.surfaceColor, wi, pdf);
        } else if (surfaceType == SURFACE_TYPE_FRESNEL_REFLECTION) {
            li = sampleFresnelReflection(wo, rec.normal, rec.surfaceColor, wi, pdf);
        } else if (surfaceType == SURFACE_TYPE_FRESNEL_TRANSMISSION) {
            li = sampleFresnelTransmission(wo, rec.normal, rec.surfaceColor, wi, pdf);
        } else if (surfaceType == SURFACE_TYPE_FRESNEL_DIELECTRIC) {
            li = sampleFresnelDielectric(wo, rec.normal, rec.surfaceColor, wi, pdf);
        } else if (surfaceType == SURFACE_TYPE_MIRROR) {
            li = sampleMirrorReflection(wo, rec.normal, rec.surfaceColor, wi, pdf);
        }

        // For specular bounce, the BSDF function contains a Dirac term.
        isSpecularBounce = surfaceType != SURFACE_TYPE_DIFFUSE;

        // Direct lighting.
        if (surfaceType == SURFACE_TYPE_DIFFUSE) {
            l += sampleAllLights(rec.point, rec.normal) * throughput * rec.surfaceColor;
        }

        if (pdf == 0.0 || isBlack(li))
            return l;
        
        throughput *= li * abs(dot(wi, rec.normal)) / pdf;
        ray = Ray(rec.point, wi);

        // Russian roulette.
        if (uRRStartBounce != -1 && bounce > uRRStartBounce)  {
            float rr = max(0.01, 1.0 - throughput.y);
            if (random1D() < rr)
                break;
            throughput /= 1.0 - rr;
        }
    }
    return l;
}

/* ==================== Camera ==================== */
vec3 _cameraU, _cameraV, _cameraW;
void buildCamera() {
    float w = uCamera.resolution.x;
    float h = uCamera.resolution.y;
    float fov = radians(uCamera.fov);
    _cameraW = normalize(uCamera.gaze - uCamera.position);
    _cameraU = normalize(cross(_cameraW, vec3(0.0, 1.0, 0.0))) * w * fov / h;
    _cameraV = normalize(cross(_cameraU, _cameraW)) * fov;
}

Ray castRay(ivec2 pixelCoords, ivec2 subPixelCoords) {
    vec2 p = vec2(pixelCoords);
    vec2 s = vec2(subPixelCoords) + vec2(0.5);
    vec2 r = random2D() * 2.0;
    vec2 k = sqrt(r);
    k.x = r.x < 1.0 ? k.x - 1.0 : 1.0 - k.x;
    k.y = r.y < 1.0 ? k.y - 1.0 : 1.0 - k.y;
    s = (s + k) * 0.5 + p;
    s = s / vec2(uCamera.resolution) - vec2(0.5);
    vec3 d = _cameraU * s.x + _cameraV * s.y + _cameraW;
    vec3 o = d * 0.1 + uCamera.position;
    d = normalize(d);
    return Ray(o, d);
}

/* ==================== Main ==================== */
void main() {
    // Initialize random generator.
    randomSeed(gl_GlobalInvocationID);

    // Build camera.
    buildCamera();

    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);    
    vec3 color = vec3(0.0);
    
    // 2x2 sub-pixels.
    Ray ray1 = castRay(pixelCoords, ivec2(0, 0));
    Ray ray2 = castRay(pixelCoords, ivec2(0, 1));
    Ray ray3 = castRay(pixelCoords, ivec2(1, 0));
    Ray ray4 = castRay(pixelCoords, ivec2(1, 1));
    for(int s = 0; s < uSpp; s++) {
        vec3 c = radiance(ray1) + radiance(ray2) + radiance(ray3) + radiance(ray4);
        c *= 0.25;
        color += c;
    }
    color /= float(uSpp);

    // Write color to result.
    imageStore(uImageOut, pixelCoords, vec4(color, 1.0));
}