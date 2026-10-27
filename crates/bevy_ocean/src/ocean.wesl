import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    mesh_functions,
    mesh_view_bindings::{view, globals, lights},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::main_pass_post_lighting_processing,
    pbr_lighting::directional_light_radiance,
    view_transformations::position_world_to_clip,
};
import bevy_render::maths::PI;
import package::light::{Surface, main_light, reflectance, water_light};
import package::optics::{IOR, Medium, Below, below, scene_light, mirror, kept, backscatter};

const CASCADES: u32 = 3u;
/// Cascade patch edges, grid side.
const SIZES = array<f32, 3>(512.0, 96.0, 17.0);
const SIDE: f32 = 256.0;
/// Hulls kept out, as surface.rs.
const MAX_HULLS: u32 = 16u;
/// Patch edge where waves fade.
const EDGE_START: f32 = 0.7;
/// Detail fade footprints, in texels.
const DETAIL_FADE = vec2<f32>(1.0, 4.0);
/// Glassiest water, before unresolved waves.
const ROUGH_FLOOR: f32 = 0.06;
const FOAM_ROUGH: f32 = 0.6;
/// Waves shrink, breaking, under this depth.
const SHOAL_DEPTH: f32 = 2.0;
/// Shore swell: wavenumber, height, frequency.
const SWELL = vec3<f32>(0.45, 0.4, 1.3);
/// Swell rises under this depth.
const SWELL_DEPTH: f32 = 6.0;
/// Swell crest bend and break-up scales.
const SWELL_BEND: f32 = 0.004;
const SWELL_BREAKS: f32 = 0.02;
/// Foam hugs the crest.
const CREST_SHARPNESS: f32 = 6.0;
const TAU: f32 = 6.2831853;
/// Foam scale and far mean.
const FOAM_GRAIN: f32 = 1.6;
const FOAM_MEAN: f32 = 0.45;
/// Beach slope floor for swell spacing.
const MIN_SLOPE: f32 = 0.02;
/// Seabed slope under which no swell runs.
const FLAT_BED: f32 = 0.004;
/// Waves break under this depth.
const BREAK_DEPTH: f32 = 1.6;
/// Swash film depth on the beach.
const SURF_DEPTH: f32 = 0.5;
const FOAM = vec3<f32>(0.82, 0.85, 0.84);
/// Clear Atlantic extinction and deep blue albedo.
const WATER = Medium(vec3<f32>(0.45, 0.15, 0.075), vec3<f32>(0.005, 0.02, 0.055));
/// Beach film foam share.
const WASH: f32 = 0.45;
/// Cascades whose crests break.
const CAP_CASCADES: u32 = 2u;
/// Surface squeeze whitening crests, and gain.
const CAP_FOLD: f32 = 0.72;
const CAP_GAIN: f32 = 5.0;
/// Light through backlit crests.
const CREST_GLOW = vec3<f32>(0.02, 0.09, 0.07);
/// Deep ground when unmapped.
const ABYSS: f32 = -1000.0;
/// Fading seabed chart edge share.
const CHART_EDGE: f32 = 0.2;
/// Sea share showing the sea.
const SEA_SHOWN = vec2<f32>(0.3, 0.6);
/// Caustics fade out by this depth.
const CAUSTIC_DEPTH: f32 = 8.0;
/// Focus range and sunlit share.
const CAUSTIC_RANGE = vec2<f32>(0.3, 3.0);
const CAUSTIC_SHARE: f32 = 0.7;

/// Hull frame and planform, as surface.rs.
struct Hull {
    rows: array<vec4<f32>, 3>,
    outline: vec4<f32>,
}

struct Ocean {
    /// Vertex share per cascade; hull count.
    reach: vec4<f32>,
    variance: vec4<f32>,
    /// Seabed corner, inverse extent, mapped.
    seabed: vec4<f32>,
    /// Calm level.
    level: vec4<f32>,
    hulls: array<Hull, MAX_HULLS>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<storage, read> ocean: Ocean;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var shapes: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var sea_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var slopes: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var seabed: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var seabed_sampler: sampler;

/// Swell height, slope and crest.
struct Surf {
    height: f32,
    slope: vec2<f32>,
    crest: f32,
}

/// Integer hash in 0..1.
fn hash(i: vec2<i32>) -> f32 {
    let s = ((bitcast<u32>(i.x) * 1597334677u) ^ (bitcast<u32>(i.y) * 3812015801u)) * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f32(((w >> 22u) ^ w) >> 8u) / 16777216.0;
}

/// Smooth value noise in 0..1.
fn noise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

/// Bottom drag and swell bearing.
struct Shore {
    depth: f32,
    /// Share of the seabed charted here.
    known: f32,
    /// Waves surviving the bottom.
    calm: f32,
    /// Uphill on the seabed.
    up: vec2<f32>,
    slope: f32,
    /// Sea share, low inland.
    sea: f32,
}

fn shore_at(p: vec2<f32>) -> Shore {
    let u = (p - ocean.seabed.xy) * ocean.seabed.z;
    let edges = smoothstep(vec2(0.0), vec2(CHART_EDGE), u) * smoothstep(vec2(1.0), vec2(1.0 - CHART_EDGE), u);
    let known = edges.x * edges.y * ocean.seabed.w;
    let chart = textureSampleLevel(seabed, seabed_sampler, u, 0.0);
    let ground = select(vec3(ABYSS, 0.0, 0.0), chart.xyz, ocean.seabed.w > 0.0);
    let depth = ocean.level.x - ground.x;
    let slope = length(ground.yz);
    let calm = mix(1.0, smoothstep(0.0, SHOAL_DEPTH, depth), known);
    let up = select(vec2(0.0), ground.yz / slope, slope > 0.0);
    return Shore(depth, known, calm, up, slope, mix(1.0, chart.w, known));
}

/// Swell rolling up the beach.
fn swell(p: vec2<f32>, shore: Shore, time: f32) -> Surf {
    let rise = smoothstep(SWELL_DEPTH, SWELL_DEPTH * 0.3, shore.depth) * smoothstep(0.0, 0.4, shore.depth);
    let along = shore.depth / max(shore.slope, MIN_SLOPE);
    let phase = SWELL.x * along + SWELL.z * time + noise(p * SWELL_BEND) * TAU;
    let breaks = noise(p * SWELL_BREAKS);
    // Flats have no beach to break on.
    let beach = smoothstep(FLAT_BED, MIN_SLOPE, shore.slope);
    let amplitude = SWELL.y * rise * breaks * shore.known * beach;
    let c = cos(phase);
    let s = sin(phase);
    let breaking = smoothstep(BREAK_DEPTH, BREAK_DEPTH * 0.4, shore.depth);
    let crest = pow(max(s, 0.0), CREST_SHARPNESS) * smoothstep(0.3, 0.8, breaks) * breaking * shore.known * beach;
    return Surf(amplitude * s, -shore.up * (SWELL.x * amplitude * c), rise * crest);
}

/// Cascade grid coordinate.
fn grid(rest: vec2<f32>, cascade: u32) -> vec2<f32> {
    var sizes = SIZES;
    return rest / sizes[cascade] + 0.5 / SIDE;
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    let index = vertex_in.instance_index;
    let v = decompress_vertex(vertex_in, index);
    let model = mesh_functions::get_world_from_local(index);
    let rest = mesh_functions::mesh_position_local_to_world(model, vec4(v.position, 1.0));
    let local = abs(v.position.xz);
    let fade = 1.0 - smoothstep(EDGE_START, 1.0, max(local.x, local.y));
    let shore = shore_at(rest.xz);
    var moved = vec3(0.0);
    for (var c = 0u; c < CASCADES; c++) {
        moved += textureSampleLevel(shapes, sea_sampler, grid(rest.xz, c), c, 0.0).xyz * ocean.reach[c];
    }
    let surf = swell(rest.xz, shore, globals.time);
    var out: VertexOutput;
    out.world_position = rest + vec4((moved * shore.calm + vec3(0.0, surf.height, 0.0)) * fade, 0.0);
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = vec3(0.0, 1.0, 0.0);
    out.uv = rest.xz;
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = index;
    return out;
}

/// Inside a hull's planform.
fn sheltered(p: vec3<f32>) -> bool {
    for (var i = 0u; i < u32(ocean.reach.w); i++) {
        let h = ocean.hulls[i];
        let q = vec4(p, 1.0);
        let along = dot(h.rows[0], q);
        let across = abs(dot(h.rows[2], q));
        let o = h.outline;
        let width = select(mix(o.z, o.w, -along / o.y), o.z * (1.0 - along / o.x), along >= 0.0);
        if along < o.x && along > -o.y && across < width {
            return true;
        }
    }
    return false;
}

/// Bubbly foam pattern, evened out afar.
fn froth(p: vec2<f32>, footprint: f32, time: f32) -> f32 {
    let drift = vec2(time * 0.07, time * 0.05);
    let near = smoothstep(0.35, 0.7, noise(p + drift) * 0.5 + noise(p * 2.3 - drift) * 0.3 + noise(p * 5.1) * 0.2);
    return mix(near, FOAM_MEAN, smoothstep(0.3, 1.5, footprint * FOAM_GRAIN));
}

/// Resolved slopes, stretch, whitecaps, unresolved variance.
struct Detail {
    slope: vec2<f32>,
    stretch: vec2<f32>,
    /// Stretch of the breaking cascades.
    squeeze: vec2<f32>,
    caps: f32,
    lost: f32,
}

/// Resolved cascade share.
fn resolved(cascade: u32, footprint: f32) -> f32 {
    var sizes = SIZES;
    let texel = sizes[cascade] / SIDE;
    return 1.0 - smoothstep(DETAIL_FADE.x * texel, DETAIL_FADE.y * texel, footprint);
}

fn detail(rest: vec2<f32>, footprint: f32) -> Detail {
    var d = Detail(vec2(0.0), vec2(0.0), vec2(0.0), 0.0, 0.0);
    for (var c = 0u; c < CASCADES; c++) {
        let keep = resolved(c, footprint);
        let uv = grid(rest, c);
        let s = textureSample(slopes, sea_sampler, uv, c);
        d.slope += s.xy * keep;
        d.stretch += s.zw * keep;
        let breaks = f32(c < CAP_CASCADES) * keep;
        d.squeeze += s.zw * breaks;
        d.caps = max(d.caps, textureSample(shapes, sea_sampler, uv, c).w * breaks);
        d.lost += ocean.variance[c] * (1.0 - keep);
    }
    return d;
}

/// Wave-focused sunlight on ground.
fn caustics(seen: Below, footprint: f32) -> f32 {
    let L = lights.directional_lights[main_light()].direction_to_light;
    let strength = smoothstep(CAUSTIC_DEPTH, 0.0, seen.depth) * saturate(L.y * 4.0) * CAUSTIC_SHARE;
    if strength <= 0.0 {
        return 1.0;
    }
    let down = refract(-L, vec3(0.0, 1.0, 0.0), 1.0 / IOR);
    let rest = seen.ground.xz + down.xz * (seen.depth / down.y);
    var curvature = 0.0;
    for (var c = 1u; c < CASCADES; c++) {
        let uv = grid(rest, c);
        let du = vec2(1.0 / SIDE, 0.0);
        let dx = textureSampleLevel(slopes, sea_sampler, uv + du, c, 0.0).x - textureSampleLevel(slopes, sea_sampler, uv - du, c, 0.0).x;
        let dz = textureSampleLevel(slopes, sea_sampler, uv + du.yx, c, 0.0).y - textureSampleLevel(slopes, sea_sampler, uv - du.yx, c, 0.0).y;
        var sizes = SIZES;
        curvature += (dx + dz) * SIDE / (2.0 * sizes[c]) * resolved(c, footprint);
    }
    let focus = 1.0 + seen.depth * (1.0 - 1.0 / IOR) * curvature;
    let gain = clamp(1.0 / max(abs(focus), 1e-3), CAUSTIC_RANGE.x, CAUSTIC_RANGE.y);
    return mix(1.0, gain, strength);
}

/// Sun through thin backlit crests.
fn crest_glow(pos: vec3<f32>, n: vec3<f32>, V: vec3<f32>, foam: f32) -> vec3<f32> {
    let main = main_light();
    let sun = lights.directional_lights[main].direction_to_light;
    let behind = pow(saturate(dot(sun, -V)), 4.0) * pow(0.5 - 0.5 * dot(sun, n), 3.0);
    let lift = max(pos.y - ocean.level.x, 0.0);
    return CREST_GLOW * directional_light_radiance(main, pos) * (behind * lift * (1.0 - foam) / PI);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let pos = in.world_position.xyz;
    let rest = in.uv;
    let footprint = length(fwidth(rest));
    let shore = shore_at(rest);
    let d = detail(rest, footprint);
    let surf = swell(rest, shore, globals.time);
    let slope = d.slope * shore.calm + surf.slope;
    let stretch = max(vec2(1.0) + d.stretch * shore.calm, vec2(0.2));
    let n = normalize(vec3(-slope.x / stretch.x, 1.0, -slope.y / stretch.y));
    let V = normalize(view.world_position - pos);
    let seen = below(in.position.xy, pos, n, V, ocean.level.x);
    let wash = (1.0 - smoothstep(0.0, SURF_DEPTH, pos.y - seen.ground.y)) * WASH;
    let squeezed = (1.0 + d.squeeze.x * shore.calm) * (1.0 + d.squeeze.y * shore.calm);
    let caps = max(d.caps * shore.calm, saturate((CAP_FOLD - squeezed) * CAP_GAIN));
    let foam = saturate(max(max(caps, surf.crest), wash) * froth(rest * FOAM_GRAIN, footprint, globals.time));
    let lost = d.lost * shore.calm * shore.calm;
    let roughness = mix(sqrt(sqrt(ROUGH_FLOOR * ROUGH_FLOOR + lost)), FOAM_ROUGH, foam);
    let mirrored = mirror(pos, n, V, roughness);

    let albedo = mix(backscatter(WATER, seen), FOAM, foam);
    let lit = water_light(in.position.xy, pos, Surface(n, V, albedo, roughness, mirrored.a));
    let toward = reflectance(roughness, max(dot(n, V), 1e-4));
    let through = seen.light * kept(WATER, seen) * caustics(seen, footprint) * (1.0 - foam);
    let glow = crest_glow(pos, n, V, foam) * view.exposure;

    var out: FragmentOutput;
    out.color = vec4(lit + toward * mirrored.rgb * mirrored.a + (1.0 - toward) * through + glow, 1.0);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    let shown = smoothstep(SEA_SHOWN.x, SEA_SHOWN.y, shore.sea) * f32(!sheltered(pos));
    out.color = vec4(mix(scene_light(in.position.xy), out.color.rgb, shown), 1.0);
    return out;
}
