/// Counter slots raster reads.
const SOFT_COUNT: u32 = 2u;
const HARD_COUNT: u32 = 3u;
const SOFT_BASE: u32 = 9u;
const HARD_BASE: u32 = 10u;

@group(2) @binding(0) var<storage, read> positions: array<f32>;
@group(2) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(2) @binding(2) var<storage, read> triangles: array<u32>;
@group(2) @binding(3) var<storage, read> clusters: array<Cluster>;
@group(2) @binding(7) var<storage, read> instances: array<Instance>;
@group(2) @binding(8) var<storage, read> materials: array<Material>;
@group(3) @binding(0) var<uniform> view: View;
@group(3) @binding(1) var<storage, read> owners: array<Owner>;
@group(3) @binding(4) var<storage, read> visible: array<vec2<u32>>;
@group(3) @binding(5) var<storage, read> counts: array<u32>;
@group(3) @binding(7) var<storage, read> visibility_read: array<u64>;

fn visible_at(pixel: vec2<u32>) -> u64 {
    return visibility_read[pixel.y * u32(view.size.x) + pixel.x];
}

fn position_of(index: u32) -> vec3<f32> {
    return vec3(positions[index * 3u], positions[index * 3u + 1u], positions[index * 3u + 2u]);
}

/// World position of cluster vertex.
fn world_of(o: Owner, index: u32, time: f32) -> vec3<f32> {
    return world_in(o, instances[o.instance], index, time);
}

/// Same, instance already loaded.
fn world_in(o: Owner, instance: Instance, index: u32, time: f32) -> vec3<f32> {
    let world = place(o.rows, local_at(index, instance, time));
    return world + sway(world, root_of(o), o.wind, time);
}

fn two_sided(c: Cluster) -> bool {
    return (materials[c.counts >> 16u].flags & TWO_SIDED) != 0u;
}

/// One screen pixel's surface.
struct Surface {
    owner: Owner,
    cluster: Cluster,
    corners: vec3<u32>,
    weights: Barycentrics,
    world: array<vec3<f32>, 3>,
}

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2(f32(i >> 1u), f32(i & 1u)) * 2.0;
    return vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
}

fn surface_at(id: u32, pixel: vec2<f32>) -> Surface {
    let item = visible[id >> TRIANGLE_BITS];
    let o = owners[item.x];
    let c = clusters[item.y];
    let corners = unpack_triangle(triangles[c.first_triangle + (id & TRIANGLE_MASK)]) + c.first_vertex;
    let instance = instances[o.instance];
    var world: array<vec3<f32>, 3>;
    var clip: array<vec4<f32>, 3>;
    for (var k = 0u; k < 3u; k++) {
        world[k] = world_in(o, instance, corners[k], view.time);
        clip[k] = view.clip_from_world * vec4(world[k], 1.0);
    }
    let weights = barycentrics(clip[0], clip[1], clip[2], pixel_ndc(pixel, view.size), view.size);
    return Surface(o, c, corners, weights, world);
}

fn world_position(s: Surface) -> vec3<f32> {
    let w = s.weights.weights;
    return s.world[0] * w.x + s.world[1] * w.y + s.world[2] * w.z;
}

/// Corner normal, posed when animated.
fn corner_normal(index: u32, i: Instance) -> vec3<f32> {
    return turned_at(index, i, unpack_direction(vertices[index].normal), view.time);
}

fn surface_normal(s: Surface) -> vec3<f32> {
    let w = s.weights.weights;
    let i = instances[s.owner.instance];
    let local = corner_normal(s.corners.x, i) * w.x + corner_normal(s.corners.y, i) * w.y + corner_normal(s.corners.z, i) * w.z;
    return turn_normal(s.owner.rows, local);
}

/// Gusty bend growing with height, calm far from the camera.
fn sway(world: vec3<f32>, root: vec3<f32>, wind: vec4<f32>, time: f32) -> vec3<f32> {
    let calm = saturate((view.wind_reach - distance(root, view.wind_eye)) / (view.wind_reach * WIND_FADE));
    let reach = max(world.y - root.y, 0.0) * wind.y;
    let t = time * wind.z;
    // Own phase per tree.
    let seed = seed_of(root.xz);
    let phase = seed * TAU + t * mix(SWAY_RATES.x, SWAY_RATES.y, seed);
    let gust = sin(phase) * 0.6 + sin(phase * 2.31 + 1.7) * 0.25 + sin(t * 0.37 + seed * TAU) * 0.4;
    // Limbs swing together; leaves tremble only up close.
    let limb = dot(world, LIMB_TRAVEL) + t * LIMB.y * (0.8 + 0.4 * seed);
    let swing = vec3(sin(limb), 0.3 * sin(limb * 1.3 + 0.7), cos(limb * 0.9 + 1.1)) * LIMB.x * reach;
    let near = 1.0 - smoothstep(TREMBLE_SEEN.x, TREMBLE_SEEN.y, distance(world, view.wind_eye));
    let leaf = dot(world, FLUTTER_TRAVEL) + time * FLUTTER.y * (0.8 + 0.4 * seed);
    let shake = vec3(sin(leaf), 0.5 * sin(leaf * 1.37 + 2.1), cos(leaf * 0.83)) * FLUTTER.x * near;
    let rock = vec3(-WIND.z, 0.0, WIND.x) * sin(phase * 0.73 + 2.9) * ROCK_ACROSS * reach * reach;
    let front = gust_front(root.xz, time);
    let bend = WIND * gust * reach * reach * (GUST_LULL + (1.0 - GUST_LULL) * front);
    return (bend + rock + (swing + shake) * (0.4 + 0.6 * front)) * wind.x * calm * view.wind_strength;
}
