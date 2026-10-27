const T_MIN: f32 = 0.05;
const SUN_T_MAX: f32 = 2000.0;
const NUDGE: f32 = 0.03;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;
const GOLDEN: f32 = 2.39996323;
const SOLID: u32 = 0x80u;
const CANOPY_GROUPS: u32 = 7u;
/// Canopy share a ray sees.
const CANOPY_KEEP: f32 = 0.7;
/// Light share leaves pass.
const LEAF_TRANSMIT: f32 = 0.3;
const CANOPY: u32 = 1u;
const GROUND: u32 = 2u;
/// Ground layers, as ground.rs.
const GROUND_LAYERS: u32 = 8u;
/// Sky fill hue kept.
const SKY_SATURATION: f32 = 0.55;
const SKY_LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

struct Hit { t: f32, normal: vec3<f32>, instance: u32, backface: bool }

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand(seed: u32) -> f32 {
    return f32(pcg(seed)) * (1.0 / 4294967296.0);
}

fn hash_cell(cell: vec3<i32>, level: u32) -> u32 {
    return pcg((u32(cell.x) * 73856093u) ^ (u32(cell.y) * 19349663u) ^ (u32(cell.z) * 83492791u) ^ (level * 2654435761u));
}

fn fib_dir(k: u32) -> vec3<f32> {
    let y = 1.0 - 2.0 * (f32(k) + 0.5) / f32(DIRS);
    let r = sqrt(max(0.0, 1.0 - y * y));
    let phi = f32(k) * GOLDEN;
    return vec3<f32>(r * cos(phi), y, r * sin(phi));
}

fn random_quat(seed: u32) -> vec4<f32> {
    let u = vec3<f32>(rand(seed), rand(seed + 1u), rand(seed + 2u));
    let a = sqrt(1.0 - u.x);
    let b = sqrt(u.x);
    return vec4<f32>(a * sin(TAU * u.y), a * cos(TAU * u.y), b * sin(TAU * u.z), b * cos(TAU * u.z));
}

fn rotate(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

/// Cube maps are left-handed.
fn sky_radiance(dir: vec3<f32>, mip: f32) -> vec3<f32> {
    let l = textureSampleLevel(sky, sky_sampler, vec3<f32>(dir.xy, -dir.z), mip).rgb * params.sky_scale;
    return mix(vec3(dot(l, SKY_LUMA)), l, SKY_SATURATION);
}

/// Solids plus random canopy subset.
fn ray_mask(seed: u32) -> u32 {
    var mask = SOLID;
    for (var g = 0u; g < CANOPY_GROUPS; g++) {
        mask |= select(0u, 1u << g, rand(seed + g) < CANOPY_KEEP);
    }
    return mask;
}

fn trace(origin: vec3<f32>, dir: vec3<f32>, t_max: f32, seed: u32, flags: u32) -> Hit {
    var hit: Hit;
    hit.t = -1.0;
    var rq: ray_query<vertex_return>;
    rayQueryInitialize(&rq, tlas, RayDesc(flags, ray_mask(seed), T_MIN, t_max, origin, dir));
    rayQueryProceed(&rq);
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind != RAY_QUERY_INTERSECTION_TRIANGLE) { return hit; }
    let v = getCommittedHitVertexPositions(&rq);
    let local = cross(v[1] - v[0], v[2] - v[0]);
    if (dot(local, local) < 1e-12) { return hit; }
    let to_world = transpose(mat3x3<f32>(h.world_to_object[0], h.world_to_object[1], h.world_to_object[2]));
    let n = normalize(to_world * local);
    hit.backface = dot(n, dir) > 0.0;
    hit.normal = select(n, -n, hit.backface);
    hit.t = h.t;
    hit.instance = h.instance_custom_data;
    return hit;
}

fn sunlit(p: vec3<f32>, seed: u32) -> bool {
    let shadow = trace(p, params.sun.xyz, SUN_T_MAX, seed + 7u, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return shadow.t < 0.0;
}

/// Sun reflected or transmitted.
fn sun_term(n: vec3<f32>, p: vec3<f32>, canopy: bool, seed: u32) -> vec3<f32> {
    let sun = params.sun.xyz;
    let cos_sun = dot(n, sun);
    let irradiance = params.sun_color * (params.sun.w / PI);
    if (cos_sun > 0.0) {
        return select(vec3<f32>(0.0), irradiance * cos_sun, sunlit(p + n * NUDGE, seed));
    }
    if (!canopy) { return vec3<f32>(0.0); }
    return select(vec3<f32>(0.0), irradiance * (-cos_sun * LEAF_TRANSMIT), sunlit(p + sun * NUDGE, seed));
}

/// Terrain colour from layer weights.
fn ground_albedo(p: vec3<f32>) -> vec3<f32> {
    let side = i32(params.ground.w);
    let texel = clamp(vec2<i32>(floor((p.xz - params.ground.xy) * params.ground.z)), vec2<i32>(0), vec2<i32>(side - 1));
    let packed = textureLoad(ground_weights, texel, 0).xy;
    let a = unpack4x8unorm(packed.x);
    let b = unpack4x8unorm(packed.y);
    let rest = max(1.0 - dot(a, vec4<f32>(1.0)) - dot(b.xyz, vec3<f32>(1.0)), 0.0);
    var weights = array<f32, GROUND_LAYERS>(rest, a.x, a.y, a.z, a.w, b.x, b.y, b.z);
    let mip = f32(textureNumLevels(ground_layers) - 1u);
    let layers = min(GROUND_LAYERS, textureNumLayers(ground_layers));
    var albedo = vec3<f32>(0.0);
    var total = 1e-4;
    for (var i = 0u; i < layers; i++) {
        albedo += textureSampleLevel(ground_layers, sky_sampler, vec2<f32>(0.5), i, mip).rgb * params.tints[i].rgb * weights[i];
        total += weights[i];
    }
    return albedo / total;
}

fn albedo_of(inst: Instance, p: vec3<f32>) -> vec3<f32> {
    if (inst.cover == GROUND) { return ground_albedo(p); }
    return inst.albedo;
}

fn shade(hit: Hit, dir: vec3<f32>, p: vec3<f32>, level: u32, seed: u32) -> vec3<f32> {
    let inst = instances[hit.instance];
    let canopy = inst.cover == CANOPY;
    let n = hit.normal;
    var light = sun_term(n, p, canopy, seed) + bounce_at(p + n * NUDGE, n, level);
    if (canopy) { light += sky_radiance(dir, FAR_SKY_MIP) * LEAF_TRANSMIT; }
    return albedo_of(inst, p) * light;
}

