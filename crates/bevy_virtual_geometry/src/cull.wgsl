const CLUSTER_ARGS: u32 = 0u;
const SOFT_ARGS: u32 = 3u;
const HARD_ARGS: u32 = 6u;
const SEED_ARGS: u32 = 9u;
/// Traversal levels; last holds late.
const MAX_LEVELS: u32 = 32u;
const LATE_SLOT: u32 = MAX_LEVELS - 1u;
/// Cull reasons, as counter slots.
const OFF_SCREEN: u32 = 0u;
const HIDDEN: u32 = 1u;
const BACK_FACING: u32 = 2u;
/// Exact foliage within this, metres.
const FINE_REACH: f32 = 10.0;

struct Counters {
    owners: atomic<u32>,
    candidates: atomic<u32>,
    soft: atomic<u32>,
    hard: atomic<u32>,
    overflow: atomic<u32>,
    triangles: atomic<u32>,
    late_instances: atomic<u32>,
    late_nodes: atomic<u32>,
    late_clusters: atomic<u32>,
    soft_base: u32,
    hard_base: u32,
    soft_done: u32,
    hard_done: u32,
    late_count: u32,
    /// Culled objects: off-screen, hidden.
    objects_culled: array<atomic<u32>, 2>,
    nodes_culled: array<atomic<u32>, 2>,
    /// Off-screen, hidden, back-facing.
    clusters_culled: array<atomic<u32>, 3>,
    /// Queue length per level.
    levels: array<atomic<u32>, MAX_LEVELS>,
}

/// Peak slots: owners, queue level, candidates, raster path, late instances, nodes, clusters.
const PEAKS: u32 = 7u;

/// Hierarchy level and pass.
struct Step {
    level: u32,
    late: u32,
}

@group(1) @binding(0) var<storage, read_write> args: array<u32>;
@group(1) @binding(1) var<storage, read_write> steps_even: array<u32>;
@group(1) @binding(2) var<storage, read_write> steps_odd: array<u32>;
/// Next level's indirect arguments.
@group(1) @binding(3) var<storage, read_write> steps_next: array<atomic<u32>>;
/// Largest demand per list over this frame's views.
@group(1) @binding(4) var<storage, read_write> peaks: array<atomic<u32>, PEAKS>;
@group(2) @binding(3) var<storage, read> clusters: array<Cluster>;
@group(2) @binding(4) var<storage, read> nodes: array<Node>;
@group(2) @binding(5) var<storage, read> children: array<vec2<u32>>;
@group(2) @binding(6) var<storage, read> parts: array<Part>;
@group(2) @binding(7) var<storage, read> instances: array<Instance>;
@group(3) @binding(0) var<uniform> view: View;
@group(3) @binding(1) var<storage, read_write> owners: array<Owner>;
@group(3) @binding(2) var<storage, read_write> queue: array<vec2<u32>>;
@group(3) @binding(3) var<storage, read_write> candidates: array<vec2<u32>>;
@group(3) @binding(4) var<storage, read_write> visible: array<vec2<u32>>;
@group(3) @binding(5) var<storage, read_write> counters: Counters;
@group(3) @binding(6) var hzb: texture_2d<f32>;
@group(3) @binding(7) var<storage, read_write> late_instances: array<u32>;
@group(3) @binding(8) var<storage, read_write> late_nodes: array<vec2<u32>>;
@group(3) @binding(9) var<storage, read_write> late_clusters: array<vec2<u32>>;
var<immediate> step: Step;

fn thread(wid: vec3<u32>, groups: vec3<u32>, lid: u32) -> u32 {
    return (wid.y * groups.x + wid.x) * WORKGROUP + lid;
}

fn dispatch(at: u32, count: u32, size: u32) {
    let groups = (count + size - 1u) / size;
    args[at] = min(groups, MAX_GROUPS);
    args[at + 1u] = (groups + MAX_GROUPS - 1u) / MAX_GROUPS;
    args[at + 2u] = 1u;
}

fn overflow() {
    atomicAdd(&counters.overflow, 1u);
}

/// Index fits capped list.
fn claim(at: u32, cap: u32) -> bool {
    if at >= cap {
        overflow();
        return false;
    }
    return true;
}

/// Takes `count` level slots.
fn reserve_nodes(level: u32, count: u32) -> u32 {
    if count == 0u {
        return 0u;
    }
    let at = atomicAdd(&counters.levels[level], count);
    atomicMax(&steps_next[level * 3u], (min(at + count, view.queue_capacity) + WORKGROUP - 1u) / WORKGROUP);
    return at;
}

fn store_node(level: u32, at: u32, item: vec2<u32>) {
    if claim(at, view.queue_capacity) {
        queue[(level & 1u) * view.queue_capacity + at] = item;
    }
}

fn push_node(level: u32, item: vec2<u32>) {
    store_node(level, reserve_nodes(level, 1u), item);
}

fn new_owner(o: Owner) -> u32 {
    let at = atomicAdd(&counters.owners, 1u);
    if !claim(at, view.owners_capacity) {
        return HOLE;
    }
    owners[at] = o;
    return at;
}

/// Box behind a culling plane.
fn outside(rows: array<vec4<f32>, 3>, lo: vec3<f32>, hi: vec3<f32>, margin: f32) -> bool {
    let center = place(rows, (lo + hi) * 0.5);
    let half = (hi - lo) * 0.5;
    let extent = vec3(dot(abs(rows[0].xyz), half), dot(abs(rows[1].xyz), half), dot(abs(rows[2].xyz), half)) + margin;
    for (var i = 0u; i < view.planes; i++) {
        let p = view.frustum[i];
        if dot(p.xyz, center) + p.w + dot(abs(p.xyz), extent) < 0.0 {
            return true;
        }
    }
    return false;
}

/// Clip position in pyramid space.
fn pyramid_clip(world: vec3<f32>, motion: array<vec4<f32>, 3>) -> vec4<f32> {
    if step.late == 0u {
        return view.previous_clip_from_world * vec4(world + place(motion, world), 1.0);
    }
    return view.clip_from_world * vec4(world, 1.0);
}

/// Local box corners.
struct Bounds {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

/// Box screen rectangle, nearest depth.
struct Footprint {
    lo: vec2<f32>,
    hi: vec2<f32>,
    nearest: f32,
    valid: bool,
}

fn footprint(rows: array<vec4<f32>, 3>, b: Bounds, motion: array<vec4<f32>, 3>) -> Footprint {
    var f = Footprint(vec2(1e30), vec2(-1e30), 0.0, true);
    for (var k = 0u; k < 8u; k++) {
        let corner = select(b.lo, b.hi, vec3((k & 1u) != 0u, (k & 2u) != 0u, (k & 4u) != 0u));
        let clip = pyramid_clip(place(rows, corner), motion);
        if clip.w <= view.near {
            f.valid = false;
            return f;
        }
        let ndc = clip.xyz / clip.w;
        let pixel = (ndc.xy * vec2(0.5, -0.5) + 0.5) * view.size;
        f.lo = min(f.lo, pixel);
        f.hi = max(f.hi, pixel);
        f.nearest = max(f.nearest, ndc.z);
    }
    return f;
}

/// Box behind all pyramid depth.
fn hidden(o: Owner, lo: vec3<f32>, hi: vec3<f32>, margin: f32) -> bool {
    if view.occlusion == 0u {
        return false;
    }
    let pad = vec3(margin / max(o.scale, 1e-6));
    let f = footprint(o.rows, Bounds(lo - pad, hi + pad), instances[o.instance].motion);
    if !f.valid {
        return false;
    }
    let lo_px = clamp(f.lo, vec2(0.0), view.size - 1.0) * 0.5;
    let hi_px = clamp(f.hi, vec2(0.0), view.size - 1.0) * 0.5;
    let levels = textureNumLevels(hzb);
    let level = min(u32(max(ceil(log2(max(max(hi_px.x - lo_px.x, hi_px.y - lo_px.y), 1.0))), 0.0)), levels - 1u);
    let size = vec2<f32>(textureDimensions(hzb, level));
    let scale = exp2(f32(level));
    let a = vec2<u32>(min(lo_px / scale, size - 1.0));
    let b = vec2<u32>(min(hi_px / scale, size - 1.0));
    let farthest = min(min(textureLoad(hzb, a, level).x, textureLoad(hzb, vec2(b.x, a.y), level).x),
        min(textureLoad(hzb, vec2(a.x, b.y), level).x, textureLoad(hzb, b, level).x));
    return f.nearest < farthest;
}

fn projected(sphere: vec4<f32>, error: f32, cap: f32) -> f32 {
    return projected_at(length(sphere.xyz - view.eye) - sphere.w, error, cap);
}

/// Error pixels; distant foliage coarsens.
fn projected_at(distance: f32, error: f32, cap: f32) -> f32 {
    if view.orthographic != 0u {
        return error * view.lod_scale / cap;
    }
    let near = max(distance, view.near);
    let ramp = near / FINE_REACH;
    let coarse = clamp(ramp * ramp, 1.0, cap);
    return error * view.lod_scale / (near * coarse);
}


fn owner_projected(o: Owner, sphere: vec4<f32>, error: f32) -> f32 {
    return projected(vec4(place(o.rows, sphere.xyz), sphere.w * o.scale), error * o.scale, instances[o.instance].coarse);
}

fn margin_of(o: Owner) -> f32 {
    return drift(o, instances[o.instance]);
}
