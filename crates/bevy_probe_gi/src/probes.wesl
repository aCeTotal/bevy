enable wgpu_ray_query;
enable wgpu_ray_query_vertex_return;

const LEVELS: u32 = 4u;
const CELLS: vec3<u32> = vec3<u32>(32u, 12u, 32u);
const PER_LEVEL: u32 = 12288u;
/// Fine levels; coarse ones alternate.
const FINE_LEVELS: u32 = 2u;
const RAYS: u32 = 8u;
/// Fixed directions per probe.
const DIRS: u32 = 256u;
/// Updates per ray sphere.
const PHASES: u32 = DIRS / RAYS;
/// Updates a coarser prior counts.
const PRIOR: u32 = PHASES / 2u;
/// History kept per finished sphere.
const KEEP: f32 = 0.5;
const INSIDE_LIMIT: f32 = 0.25;
const T_MAX: f32 = 1200.0;
const FAR_SKY_MIP: f32 = 4.0;
/// Radiance above this is glitch.
const RADIANCE_CAP: f32 = 1e6;
/// Calm sweeps before a probe rests.
const SETTLED: f32 = 4.0;
/// A resting probe traces one frame in this many.
const RESTING_RATE: u32 = 16u;
/// Relative change a calm sweep stays under.
const CALM: f32 = 0.02;
const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

var<private> FACES: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
);

struct Window { origin: vec3<f32>, spacing: f32 }
/// Ground corner, texel scale, side.
struct Params {
    windows: array<Window, 4>,
    sun: vec4<f32>,
    sun_color: vec3<f32>,
    sky_scale: f32,
    ground: vec4<f32>,
    tints: array<vec4<f32>, GROUND_LAYERS>,
    frame: u32,
    /// Bumped when sun or sky change.
    epoch: u32,
    /// Levels whose window moved since last traced.
    moved: u32,
}
struct Instance { albedo: vec3<f32>, cover: u32 }
/// Faces: light, sky seen.
struct Probe {
    cell: vec4<i32>,
    faces: array<vec4<f32>, 6>,
    partial: array<vec4<f32>, 6>,
    /// Coverage, inside sum, updates.
    tally: vec4<f32>,
}
/// Trilinear faces and found weight.
struct Sampled { faces: array<vec4<f32>, 6>, weight: f32 }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var tlas: acceleration_structure<vertex_return>;
@group(0) @binding(3) var<storage, read_write> probes: array<Probe>;
@group(0) @binding(4) var out0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var out1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var out2: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var out3: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var sky: texture_cube<f32>;
@group(0) @binding(9) var sky_sampler: sampler;
@group(0) @binding(10) var ground_weights: texture_2d<u32>;
@group(0) @binding(11) var ground_layers: texture_2d_array<f32>;

fn base_of(level: u32) -> vec3<i32> {
    let w = params.windows[level];
    return vec3<i32>(round(w.origin / w.spacing));
}

fn slot_of(level: u32, cell: vec3<i32>) -> u32 {
    let n = vec3<i32>(CELLS);
    let c = vec3<u32>(((cell % n) + n) % n);
    return level * PER_LEVEL + c.x + c.y * CELLS.x + c.z * CELLS.x * CELLS.y;
}

fn inside_window(level: u32, cell: vec3<i32>) -> bool {
    let local = cell - base_of(level);
    return all(local >= vec3<i32>(0)) && all(local < vec3<i32>(CELLS));
}

/// Ambient cube along normal.
fn cube(faces: array<vec4<f32>, 6>, n: vec3<f32>) -> vec3<f32> {
    let nn = n * n;
    let x = select(faces[1].xyz, faces[0].xyz, n.x >= 0.0);
    let y = select(faces[3].xyz, faces[2].xyz, n.y >= 0.0);
    let z = select(faces[5].xyz, faces[4].xyz, n.z >= 0.0);
    return x * nn.x + y * nn.y + z * nn.z;
}

/// Stored bounce at hit.
fn bounce_at(p: vec3<f32>, n: vec3<f32>, from_level: u32) -> vec3<f32> {
    for (var level = from_level; level < LEVELS; level++) {
        let cell = vec3<i32>(floor(p / params.windows[level].spacing));
        if (!inside_window(level, cell)) { continue; }
        let rec = probes[slot_of(level, cell)];
        if (any(rec.cell.xyz != cell)) { continue; }
        return cube(rec.faces, n);
    }
    return sky_radiance(n, FAR_SKY_MIP);
}

fn sample_level(level: u32, p: vec3<f32>) -> Sampled {
    let c = p / params.windows[level].spacing - 0.5;
    let c0 = vec3<i32>(floor(c));
    let f = fract(c);
    var s: Sampled;
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<i32>(vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u));
        let cell = c0 + corner;
        if (!inside_window(level, cell)) { continue; }
        let rec = probes[slot_of(level, cell)];
        if (any(rec.cell.xyz != cell)) { continue; }
        let wf = select(1.0 - f, f, corner == vec3<i32>(1));
        let wk = wf.x * wf.y * wf.z;
        for (var i = 0u; i < 6u; i++) { s.faces[i] += rec.faces[i] * wk; }
        s.weight += wk;
    }
    for (var i = 0u; i < 6u; i++) { s.faces[i] /= max(s.weight, 1e-6); }
    return s;
}

/// Finer level, else probe above.
fn output_faces(level: u32, cell: vec3<i32>, pos: vec3<f32>, rec: Probe) -> array<vec4<f32>, 6> {
    for (var finer = 0u; finer < level; finer++) {
        let s = sample_level(finer, pos);
        if (s.weight > 0.5) { return s.faces; }
    }
    if (rec.tally.x > INSIDE_LIMIT) {
        let up = cell + vec3<i32>(0, 1, 0);
        let above = probes[slot_of(level, up)];
        if (all(above.cell.xyz == up)) { return above.faces; }
    }
    return rec.faces;
}

/// Fresh probe from coarser level.
fn seeded(level: u32, cell: vec3<i32>, pos: vec3<f32>) -> Probe {
    var rec: Probe;
    rec.cell = vec4<i32>(cell, 0);
    if (level + 1u >= LEVELS) { return rec; }
    let s = sample_level(level + 1u, pos);
    if (s.weight < 0.5) { return rec; }
    for (var f = 0u; f < 6u; f++) { rec.partial[f] = s.faces[f] * f32(RAYS * PRIOR / 4u); }
    rec.tally.z = f32(PRIOR);
    return rec;
}

/// Whether a sweep barely moved the light.
fn calm(before: array<vec4<f32>, 6>, after: array<vec4<f32>, 6>) -> bool {
    var change = 0.0;
    var brightest = 1e-4;
    for (var f = 0u; f < 6u; f++) {
        change = max(change, abs(dot(after[f].rgb - before[f].rgb, LUMA)));
        brightest = max(brightest, dot(before[f].rgb, LUMA));
    }
    return change < CALM * brightest;
}

/// Folds sweep sums into faces.
fn fold(rec: ptr<function, Probe>, keep: f32) {
    let rays = f32(RAYS) * (*rec).tally.z;
    for (var f = 0u; f < 6u; f++) {
        (*rec).faces[f] = mix((*rec).partial[f] * (4.0 / rays), (*rec).faces[f], keep);
    }
    (*rec).tally.x = mix((*rec).tally.y / rays, (*rec).tally.x, keep);
}

fn store(level: u32, local: vec3<u32>, faces: array<vec4<f32>, 6>) {
    for (var f = 0u; f < 6u; f++) {
        let positive = (f % 2u) == 0u;
        let at = vec3<i32>(vec3<u32>(local.x, local.y + select(CELLS.y, 0u, positive), local.z + (f / 2u) * CELLS.z));
        switch level {
            case 0u: { textureStore(out0, at, faces[f]); }
            case 1u: { textureStore(out1, at, faces[f]); }
            case 2u: { textureStore(out2, at, faces[f]); }
            default: { textureStore(out3, at, faces[f]); }
        }
    }
}

@compute @workgroup_size(64)
fn trace_probes(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x;
    if (g >= (FINE_LEVELS + 1u) * PER_LEVEL) { return; }
    var level = g / PER_LEVEL;
    if (level >= FINE_LEVELS) { level = FINE_LEVELS + (params.frame & 1u); }
    let i = g % PER_LEVEL;
    let local = vec3<u32>(i % CELLS.x, (i / CELLS.x) % CELLS.y, i / (CELLS.x * CELLS.y));
    let cell = base_of(level) + vec3<i32>(local);
    let slot = slot_of(level, cell);
    let pos = (vec3<f32>(cell) + 0.5) * params.windows[level].spacing;
    var rec = probes[slot];
    if (any(rec.cell.xyz != cell)) {
        rec = seeded(level, cell, pos);
    }
    let probe = hash_cell(cell, level);
    // Sun or sky changed since the last sweep: wake.
    if (rec.cell.w != 0 && u32(rec.cell.w) != params.epoch + 1u) {
        rec.tally.w = 0.0;
    }
    // Resting probes trace rarely; re-store when moved.
    if (rec.tally.w >= SETTLED && (params.frame + probe) % RESTING_RATE != 0u) {
        if ((params.moved >> level) & 1u) != 0u {
            store(level, local, output_faces(level, cell, pos, rec));
        }
        return;
    }
    // Coarse levels tick half rate.
    let tick = select(params.frame, params.frame >> 1u, level >= FINE_LEVELS);
    let q = random_quat(probe);
    var sum: array<vec4<f32>, 6>;
    var inside = 0.0;
    for (var r = 0u; r < RAYS; r++) {
        // Fixed rays: no flicker.
        let k = r * PHASES + tick % PHASES;
        let dir = rotate(fib_dir(k), q);
        let seed = pcg(probe + k);
        let hit = trace(pos, dir, T_MAX, seed, RAY_FLAG_NONE);
        var l = vec4<f32>(sky_radiance(dir, 0.0), 1.0);
        if (hit.t >= 0.0) {
            l = vec4<f32>(shade(hit, dir, pos + dir * hit.t, level, seed), 0.0);
            inside += f32(hit.backface && instances[hit.instance].cover != CANOPY);
        }
        l = clamp(select(l, vec4<f32>(0.0), l != l), vec4<f32>(0.0), vec4<f32>(RADIANCE_CAP));
        for (var f = 0u; f < 6u; f++) { sum[f] += l * max(0.0, dot(dir, FACES[f])); }
    }
    for (var f = 0u; f < 6u; f++) { rec.partial[f] += sum[f]; }
    rec.tally.y += inside;
    rec.tally.z += 1.0;
    let first = rec.cell.w == 0;
    let done = rec.tally.z >= f32(PHASES);
    if (first || done) {
        let before = rec.faces;
        fold(&rec, select(0.0, KEEP, done && !first));
        if (done && !first) {
            rec.tally.w = select(0.0, rec.tally.w + 1.0, calm(before, rec.faces));
        }
    }
    if (done) {
        rec.cell.w = i32(params.epoch + 1u);
        rec.partial = array<vec4<f32>, 6>();
        rec.tally = vec4<f32>(rec.tally.x, 0.0, 0.0, rec.tally.w);
    }
    probes[slot] = rec;
    store(level, local, output_faces(level, cell, pos, rec));
}
