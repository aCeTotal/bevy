const WORKGROUP: u32 = 64u;
const MAX_GROUPS: u32 = 65535u;
/// Culling planes per view.
const PLANES: u32 = 10u;
const HOLE: u32 = 0xffffffffu;
const INDEX_BITS: u32 = 30u;
const INDEX_MASK: u32 = 0x3fffffffu;
/// Child kinds, as layout.rs.
const CHILD_NODE: u32 = 0u;
const CHILD_CLUSTERS: u32 = 1u;
const CHILD_PART: u32 = 2u;
const TRIANGLE_BITS: u32 = 7u;
const TRIANGLE_MASK: u32 = 0x7fu;
/// Baked limits; garbage never loops.
const MAX_CHILDREN: u32 = 64u;
const CLUSTER_LIMIT: u32 = 128u;
const TWO_SIDED: u32 = 1u;
/// Errors above this are infinite.
const ENDLESS: f32 = 1.0e30;
/// Owner scaled unevenly.
const UNEVEN: u32 = 1u;
const WIND = vec3<f32>(0.8, 0.0, 0.6);
/// Strongest gust over unit reach, in the strongest wind.
const GUST_PEAK: f32 = 1.25;
const STRONGEST: f32 = 3.0;
/// Limb swing per sway metre, rate; phase per metre.
const LIMB = vec2<f32>(0.12, 5.0);
const LIMB_TRAVEL = vec3<f32>(0.55, 0.35, 0.75);
/// Leaf tremble per sway metre, rate; phase per metre.
const FLUTTER = vec2<f32>(0.03, 9.0);
const FLUTTER_TRAVEL = vec3<f32>(1.7, 1.2, 2.3);
/// Leaf tremble fade distance, metres.
const TREMBLE_SEEN = vec2<f32>(20.0, 50.0);
/// Crown rocking across the wind.
const ROCK_ACROSS: f32 = 0.35;
/// Gust front wavelength over TAU and travel, metres and metres per second.
const GUST_FRONT = vec2<f32>(7.0, 6.0);
/// Bend kept in lulls.
const GUST_LULL: f32 = 0.45;
/// Share of the sway reach plants calm over.
const WIND_FADE: f32 = 0.25;
/// Fixed-point steps of packed wind.
const WIND_UNIT: f32 = 4096.0;
/// Golden-ratio hash multiplier.
const SPREAD: u32 = 2654435761u;
/// Spatial hash primes.
const HASH_PRIMES = vec2<u32>(73856093u, 19349663u);
/// Hash bits kept as a fraction.
const HASH_RANGE: f32 = 16777216.0;
const TAU: f32 = 6.2831853;
/// Per-tree sway rates.
const SWAY_RATES = vec2<f32>(1.4, 2.0);

struct Cluster {
    min: vec3<f32>,
    first_vertex: u32,
    max: vec3<f32>,
    first_triangle: u32,
    apex: vec3<f32>,
    counts: u32,
    axis: vec3<f32>,
    cutoff: f32,
    lod: vec4<f32>,
    parent: vec4<f32>,
    errors: vec4<f32>,
}

struct Node {
    min: vec3<f32>,
    first_child: u32,
    max: vec3<f32>,
    child_count: u32,
    lod: vec4<f32>,
    error: vec4<f32>,
}

struct Part {
    rows: array<vec4<f32>, 3>,
    parent: vec4<f32>,
    error: f32,
    root: u32,
    /// Amplitude bits; inverse height and speed, 4096ths.
    wind: vec2<u32>,
}

struct Vertex {
    uv: vec2<f32>,
    normal: u32,
    color: u32,
    tangent: u32,
    /// Packed joints and weight, or zero.
    skin: u32,
}

struct Material {
    base_color: vec4<f32>,
    roughness: f32,
    metallic: f32,
    translucency: f32,
    shading: u32,
    albedo: u32,
    normal: u32,
    orm: u32,
    flags: u32,
}

struct Instance {
    rows: array<vec4<f32>, 3>,
    /// Last frame delta minus identity.
    motion: array<vec4<f32>, 3>,
    root: u32,
    tint: u32,
    /// Far cut error multiplier.
    coarse: f32,
    pad: u32,
    wind: vec4<f32>,
    /// Pose base, bones, frames; no frames is rest.
    pose: vec4<u32>,
    /// Clip start, speed, fps, skin reach.
    play: vec4<f32>,
}

/// One drawn mesh copy.
struct Owner {
    rows: array<vec4<f32>, 3>,
    scale: f32,
    instance: u32,
    flags: u32,
    pad: u32,
    /// Amplitude, inverse height, speed.
    wind: vec4<f32>,
}

struct View {
    clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    previous_clip_from_world: mat4x4<f32>,
    frustum: array<vec4<f32>, PLANES>,
    eye: vec3<f32>,
    lod_scale: f32,
    /// Camera plants sway around, and how far.
    wind_eye: vec3<f32>,
    wind_reach: f32,
    size: vec2<f32>,
    threshold: f32,
    soft_limit: f32,
    time: f32,
    previous_time: f32,
    near: f32,
    instance_extent: u32,
    dynamic_count: u32,
    orthographic: u32,
    occlusion: u32,
    planes: u32,
    /// Debug view, zero shades.
    visualize: u32,
    /// List capacities; buffers grow from GPU peaks.
    static_capacity: u32,
    owners_capacity: u32,
    queue_capacity: u32,
    candidates_capacity: u32,
    /// Each raster path gets half.
    half_visible: u32,
    late_instances_capacity: u32,
    late_nodes_capacity: u32,
    late_clusters_capacity: u32,
    /// Weather sway scale.
    wind_strength: f32,
}

fn place(rows: array<vec4<f32>, 3>, p: vec3<f32>) -> vec3<f32> {
    let h = vec4(p, 1.0);
    return vec3(dot(rows[0], h), dot(rows[1], h), dot(rows[2], h));
}

fn turn(rows: array<vec4<f32>, 3>, d: vec3<f32>) -> vec3<f32> {
    return vec3(dot(rows[0].xyz, d), dot(rows[1].xyz, d), dot(rows[2].xyz, d));
}

fn column(rows: array<vec4<f32>, 3>, i: u32) -> vec3<f32> {
    return vec3(rows[0][i], rows[1][i], rows[2][i]);
}

/// Normal under uneven scale.
fn turn_normal(rows: array<vec4<f32>, 3>, n: vec3<f32>) -> vec3<f32> {
    let c0 = column(rows, 0u);
    let c1 = column(rows, 1u);
    let c2 = column(rows, 2u);
    return normalize(n.x * cross(c1, c2) + n.y * cross(c2, c0) + n.z * cross(c0, c1));
}

fn compose(a: array<vec4<f32>, 3>, b: array<vec4<f32>, 3>) -> array<vec4<f32>, 3> {
    var c: array<vec4<f32>, 3>;
    for (var r = 0u; r < 3u; r++) {
        c[r] = a[r].x * b[0] + a[r].y * b[1] + a[r].z * b[2] + vec4(0.0, 0.0, 0.0, a[r].w);
    }
    return c;
}

fn largest_scale(rows: array<vec4<f32>, 3>) -> f32 {
    return max(length(column(rows, 0u)), max(length(column(rows, 1u)), length(column(rows, 2u))));
}

fn uneven(rows: array<vec4<f32>, 3>) -> bool {
    let s = vec3(length(column(rows, 0u)), length(column(rows, 1u)), length(column(rows, 2u)));
    return max(s.x, max(s.y, s.z)) > min(s.x, min(s.y, s.z)) * 1.001;
}

/// Gusts rolling downwind over the ground, 0 lull to 1 gust.
fn gust_front(xz: vec2<f32>, time: f32) -> f32 {
    let down = dot(xz, WIND.xz) - time * GUST_FRONT.y;
    let side = dot(xz, vec2(-WIND.z, WIND.x));
    let wave = sin(down / GUST_FRONT.x) * 0.6 + sin(down / GUST_FRONT.x * 0.43 + side / GUST_FRONT.x * 0.6 + 1.9) * 0.4;
    return smoothstep(-0.6, 0.9, wave);
}

/// Stable fraction per metre cell.
fn seed_of(p: vec2<f32>) -> f32 {
    let cell = bitcast<vec2<u32>>(vec2<i32>(floor(p))) * HASH_PRIMES;
    return f32(((cell.x ^ cell.y) * SPREAD) >> 8u) / HASH_RANGE;
}

/// Furthest any vertex may sway.
fn sway_margin(wind: vec4<f32>) -> f32 {
    return wind.x * (GUST_PEAK + ROCK_ACROSS + (LIMB.x + FLUTTER.x) * 1.5) * STRONGEST * length(WIND);
}

/// Box growth for sway and limbs.
fn drift(o: Owner, i: Instance) -> f32 {
    return max(sway_margin(o.wind), i.play.w * o.scale);
}

/// Where a copy stands; plants sway about it.
fn root_of(o: Owner) -> vec3<f32> {
    return vec3(o.rows[0].w, o.rows[1].w, o.rows[2].w);
}

/// A part's own wind, else the one it sits in.
fn part_wind(p: Part, inherited: vec4<f32>) -> vec4<f32> {
    let amplitude = bitcast<f32>(p.wind.x);
    if amplitude <= 0.0 {
        return inherited;
    }
    return vec4(amplitude, vec2<f32>(vec2(p.wind.y & 0xffffu, p.wind.y >> 16u)) / WIND_UNIT, 0.0);
}

fn unpack_triangle(t: u32) -> vec3<u32> {
    return vec3(t & 0xffu, (t >> 8u) & 0xffu, (t >> 16u) & 0xffu);
}

fn signed_unit(bits: u32, width: u32) -> f32 {
    let shift = 32u - width;
    let value = f32(bitcast<i32>(bits << shift) >> shift);
    return clamp(value / f32((1u << (width - 1u)) - 1u), -1.0, 1.0);
}

fn unfold(e: vec2<f32>) -> vec3<f32> {
    var n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if n.z < 0.0 {
        n = vec3((1.0 - abs(n.yx)) * sign(n.xy), n.z);
    }
    return normalize(n);
}

fn unpack_direction(p: u32) -> vec3<f32> {
    return unfold(vec2(signed_unit(p & 0xffffu, 16u), signed_unit(p >> 16u, 16u)));
}

fn unpack_tangent(p: u32) -> vec4<f32> {
    let d = unfold(vec2(signed_unit(p & 0xffffu, 16u), signed_unit((p >> 16u) & 0x7fffu, 15u)));
    return vec4(d, select(1.0, -1.0, (p >> 31u) == 1u));
}

fn unpack_color(p: u32) -> vec4<f32> {
    return unpack4x8unorm(p);
}

/// Perspective barycentrics and screen derivatives.
struct Barycentrics {
    weights: vec3<f32>,
    ddx: vec3<f32>,
    ddy: vec3<f32>,
}

fn barycentrics(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, ndc: vec2<f32>, size: vec2<f32>) -> Barycentrics {
    let inv_w = 1.0 / vec3(p0.w, p1.w, p2.w);
    let n0 = p0.xy * inv_w.x;
    let n1 = p1.xy * inv_w.y;
    let n2 = p2.xy * inv_w.z;
    let a = n2 - n1;
    let b = n0 - n1;
    let inv_det = 1.0 / (a.x * b.y - a.y * b.x);
    var ddx = vec3(n1.y - n2.y, n2.y - n0.y, n0.y - n1.y) * inv_det * inv_w;
    var ddy = vec3(n2.x - n1.x, n0.x - n2.x, n1.x - n0.x) * inv_det * inv_w;
    let delta = ndc - n0;
    let inv_interp = inv_w.x + delta.x * (ddx.x + ddx.y + ddx.z) + delta.y * (ddy.x + ddy.y + ddy.z);
    let interp = 1.0 / inv_interp;
    let weights = interp * (vec3(inv_w.x, 0.0, 0.0) + delta.x * ddx + delta.y * ddy);
    ddx *= 2.0 / size.x;
    ddy *= -2.0 / size.y;
    let dx = interp * (1.0 / (1.0 + (ddx.x + ddx.y + ddx.z) * interp));
    let dy = interp * (1.0 / (1.0 + (ddy.x + ddy.y + ddy.z) * interp));
    let step_x = dx / interp * (weights + ddx * interp) - weights;
    let step_y = dy / interp * (weights + ddy * interp) - weights;
    return Barycentrics(weights, step_x, step_y);
}

fn pixel_ndc(pixel: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let uv = pixel / size;
    return vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn depth_of(v: u64) -> f32 {
    return bitcast<f32>(u32(v >> 32u));
}

/// Visibility id of scene depth.
const BACKGROUND: u32 = 0xffffffffu;

fn id_of(v: u64) -> u32 {
    return u32(v);
}

