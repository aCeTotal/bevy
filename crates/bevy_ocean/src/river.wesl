import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    mesh_functions,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::main_pass_post_lighting_processing,
};
import package::light::{Surface, reflectance, water_light};
import package::optics::{Medium, below, mirror, kept, backscatter, pixel, scene_light, scene_point, on_screen};
import package::flow::{caustics, drifting, flow_over, ripple, slicks, streak_lines};

/// Seconds a fine and a broad flow phase last.
const PERIOD: f32 = 1.6;
const LONG_PERIOD: f32 = 12.0;
/// Ripple wavelength: per width, bounds; broad swell likewise.
const RIPPLE = vec3<f32>(0.03, 0.3, 0.9);
const BROAD = vec3<f32>(0.15, 1.2, 5.0);
/// Ripple steepness terms; broad swell base and per m/s.
const STEEP = vec3<f32>(0.1, 0.5, 0.5);
const STEEP_BROAD = vec2<f32>(0.03, 0.1);
/// Footprint per wavelength fading ripples out.
const RIPPLE_FADE = vec2<f32>(0.15, 0.6);
/// Ripples kept on glassy slicks.
const SLICK_CALM: f32 = 0.2;
/// Flow streaking: stretch, speed band.
const STREAK_STRETCH: f32 = 4.0;
/// Standing chop: steepness, wavelength, Froude.
const STANDING = vec4<f32>(0.6, 2.0, 0.2, 0.6);
const STREAK_SPEED = vec2<f32>(0.1, 1.0);
/// Current lost at the banks.
const BANK_DRAG: f32 = 0.6;
/// Channel depth terms, as atlas::water.
const DEPTH_GAIN: f32 = 0.4;
const DEPTH_POWER: f32 = 0.55;
const SHALLOWING = vec2<f32>(0.001, 0.006);
const MEAN_DEPTH: f32 = 0.6;
/// Depth speed-up bounds.
const SQUEEZE = vec2<f32>(0.6, 1.4);
/// Froude band breaking water white.
const WHITE = vec2<f32>(0.6, 1.4);
/// Breaking depth over stones, metres.
const SHOAL: f32 = 0.1;
/// Wake length, probes, fan spread.
const WAKE = vec3<f32>(5.0, 8.0, 0.3);
/// Ripples raised in wakes.
const WAKE_RIPPLE: f32 = 0.8;
/// Caustic gain and fade depth.
const CAUSTIC_LIGHT = vec2<f32>(6.0, 1.2);
/// Parting probe and push, metres.
const PART = vec2<f32>(1.0, 0.6);
/// Surface fall band of cascades.
const CASCADE = vec2<f32>(0.08, 0.25);
/// Current band that lets foam form.
const CHURN = vec2<f32>(0.3, 1.0);
/// Foam albedo.
const FOAM = vec3<f32>(0.74, 0.76, 0.75);
/// Foam patch, bubble sizes, edge, opacity.
const FOAM_CELLS = vec2<f32>(1.2, 0.22);
const FOAM_EDGE: f32 = 0.6;
const FOAM_OPACITY: f32 = 0.85;
/// Most foam cover.
const FOAM_COVER: f32 = 0.9;
/// Scum lines: albedo, opacity, share left in calm open water.
const SCUM = vec3<f32>(0.5, 0.5, 0.46);
const SCUM_OPACITY: f32 = 0.25;
const SCUM_CALM: f32 = 0.3;
/// Silty lowland water and clear peaty upland water.
const TURBID = Medium(vec3<f32>(1.1, 1.5, 2.4), vec3<f32>(0.05, 0.045, 0.03));
const CLEAR = Medium(vec3<f32>(0.45, 0.8, 1.7), vec3<f32>(0.035, 0.028, 0.016));
/// Falls clearing rivers, and lake clarity.
const CLEARING = vec2<f32>(0.002, 0.01);
const LAKE_CLARITY: f32 = 0.6;
/// Bank and tree heights.
const BANK_LIP: f32 = 0.5;
const TREES: f32 = 14.0;
/// Banks and leaves seen in reflection.
const SHORE_ALBEDO = vec3<f32>(0.05, 0.07, 0.035);
/// Far depth pull per metre.
const PULL: f32 = 0.004;
const PULL_FROM: f32 = 250.0;
const UP = vec3<f32>(0.0, 1.0, 0.0);
const GRAVITY: f32 = 9.81;
/// Terrain level ranges and morph start, as cdlod.rs.
const RANGE0: f32 = 192.0;
const TOP: f32 = 11.0;
const MORPH_START: f32 = 0.7;
/// Narrowest carved width per sample, as world banks.
const NARROWEST: f32 = 0.5;
/// Farthest surface drawn, as pieces.rs.
const FARTHEST: f32 = 40000.0;
/// Depth of unseen beds.
const UNSEEN: f32 = 1e3;
/// Calm bank share of half-width.
const EDDIES = vec2<f32>(0.55, 0.85);
/// Widths per ground spacing resolving a bed.
const RESOLVED = vec2<f32>(2.0, 6.0);

/// Current axes and stretch.
struct Frame {
    along: vec2<f32>,
    across: vec2<f32>,
    stretch: f32,
}

fn frame_of(current: vec2<f32>) -> Frame {
    let speed = length(current);
    let along = select(vec2(1.0, 0.0), current / max(speed, 1e-4), speed > 1e-3);
    return Frame(along, vec2(-along.y, along.x), mix(1.0, STREAK_STRETCH, smoothstep(STREAK_SPEED.x, STREAK_SPEED.y, speed)));
}

/// Where a point sits on its river: metres downstream and left, and those axes in the world.
struct Channel {
    at: vec2<f32>,
    downstream: vec2<f32>,
    left: vec2<f32>,
}

/// River space from metres downstream and left, or plain ground axes on a lake.
fn channel_of(pos: vec3<f32>, frame: Frame, place: vec2<f32>) -> Channel {
    if place.x < 0.0 {
        return Channel(pos.xz, vec2(1.0, 0.0), vec2(0.0, 1.0));
    }
    return Channel(place, frame.along, frame.across);
}

/// Detail kept at this footprint per wavelength.
fn seen_fade(ratio: f32) -> f32 {
    return 1.0 - smoothstep(RIPPLE_FADE.x, RIPPLE_FADE.y, ratio);
}

/// Depth under a point, deep if unseen.
fn depth_at(q: vec3<f32>) -> f32 {
    let clip = position_world_to_clip(q);
    if clip.w <= 0.0 {
        return UNSEEN;
    }
    let frag = pixel(clip.xy / clip.w);
    let ground = scene_point(frag);
    let blocked = distance(view.world_position, ground) < distance(view.world_position, q);
    return select(UNSEEN, q.y - ground.y, on_screen(frag) > 0.0 && !blocked);
}

/// Stone near the surface.
fn shallow(q: vec3<f32>) -> f32 {
    return 1.0 - smoothstep(0.0, SHOAL, depth_at(q));
}

/// Water parted by stones.
struct Parting {
    wake: f32,
    aside: f32,
}

fn parting(pos: vec3<f32>, frame: Frame, depth_speed: vec2<f32>) -> Parting {
    let speed = depth_speed.y;
    let along = vec3(frame.along.x, 0.0, frame.along.y);
    let across = vec3(frame.across.x, 0.0, frame.across.y);
    // Fanned wakes, longer when fast.
    let fan = fract(sin(dot(pos.xz, vec2(12.9898, 78.233))) * 43758.547) * 2.0 - 1.0;
    let length = WAKE.x * (0.3 + speed);
    var wake = 1.0 - smoothstep(0.0, SHOAL, depth_speed.x);
    for (var k = 0; k < i32(WAKE.y); k++) {
        let t = (f32(k) + 0.5) / WAKE.y;
        let behind = pos - along * length * t + across * length * t * WAKE.z * fan;
        wake = max(wake, shallow(behind) * (1.0 - t));
    }
    let aside = shallow(pos - across * PART.x) - shallow(pos + across * PART.x);
    return Parting(wake, aside * PART.y);
}

/// Reflected sky hidden by banks.
fn hidden(R: vec3<f32>, frame: Frame, shore: vec4<f32>) -> f32 {
    let across = shore.xyz;
    let half = shore.w;
    let run = length(R.xz);
    let toward = dot(R.xz, frame.across) / max(run, 1e-4);
    let left = toward >= 0.0;
    let gap = half * select(1.0 + across.x, 1.0 - across.x, left) / max(abs(toward), 0.05);
    let height = BANK_LIP + TREES * select(across.z, across.y, left);
    return smoothstep(-0.1, 0.1, height / max(gap, 0.5) - R.y / max(run, 1e-4));
}

/// Farthest a channel this wide stays carved and drawn, as pieces.rs.
fn carved(width: f32) -> f32 {
    return min(RANGE0 * exp2(min(floor(log2(max(width / NARROWEST, 1.0))), TOP)), FARTHEST);
}

/// Far ribbons pulled forward.
@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    let index = vertex_in.instance_index;
    let v = decompress_vertex(vertex_in, index);
    let world = mesh_functions::mesh_position_local_to_world(mesh_functions::get_world_from_local(index), vec4(v.position, 1.0));
    let eye = view.world_position - world.xyz;
    let range = length(eye);
    var out: VertexOutput;
    out.world_position = world;
    out.position = position_world_to_clip(world.xyz + eye / max(range, 1e-3) * PULL * max(range - PULL_FROM, 0.0));
    out.world_normal = UP;
    @if(VERTEX_UVS_A)
    out.uv = v.uv;
    @if(VERTEX_UVS_B)
    out.uv_b = v.uv_b;
    @if(VERTEX_TANGENTS)
    out.world_tangent = v.tangent;
    @if(VERTEX_COLORS)
    out.color = v.color;
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = index;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let pos = in.world_position.xyz;
    let width = in.uv_b.x;
    let fall = in.uv_b.y;
    let across = in.color.xyz;
    let frame = frame_of(in.uv);
    let V = normalize(view.world_position - pos);
    let range = distance(pos, view.world_position);
    let footprint = length(fwidth(pos.xz));
    // Lakes carry no course.
    var downstream = -1.0;
    @if(VERTEX_TANGENTS) {
        downstream = in.world_tangent.w;
    }
    let river = f32(downstream >= 0.0);
    let channel = channel_of(pos, frame, vec2(downstream, across.x * width * 0.5));

    // Depth through the calm surface.
    let flat_seen = below(in.position.xy, pos, UP, V, pos.y);
    let mean = DEPTH_GAIN * pow(max(width, 1.0), DEPTH_POWER) / (1.0 + max(fall - SHALLOWING.x, 0.0) / SHALLOWING.y) * MEAN_DEPTH;
    // Coarse ground hides the bed.
    let resolved = smoothstep(RESOLVED.x, RESOLVED.y, width * RANGE0 / max(range, RANGE0));
    let depth = mix(mean, flat_seen.depth, resolved);
    let squeeze = clamp(sqrt(mean / max(depth, 0.05)), SQUEEZE.x, SQUEEZE.y);
    let speed = length(in.uv) * (1.0 - BANK_DRAG * across.x * across.x) * squeeze;
    let froude = speed / sqrt(GRAVITY * max(depth, 0.03));
    let rough = max(smoothstep(WHITE.x, WHITE.y, froude), smoothstep(CASCADE.x, CASCADE.y, fall));
    // Only where the bed shows.
    var parted = Parting(0.0, 0.0);
    if resolved > 0.0 {
        parted = parting(pos, frame, vec2(depth, speed));
    }
    let wake = parted.wake * resolved;
    let churned = max(rough, wake * smoothstep(CHURN.x, CHURN.y, speed)) * (1.0 - smoothstep(EDDIES.x, EDDIES.y, abs(across.x)));

    // Ripples ride the current.
    let fine = flow_over(PERIOD);
    let slow = flow_over(LONG_PERIOD);
    let q = channel.at + vec2(0.0, parted.aside * resolved);
    let glassy = slicks(q, vec2(speed, width), slow) * river * (1.0 - wake);
    let ruffled = 1.0 - glassy * (1.0 - SLICK_CALM);
    let wave = clamp(width * RIPPLE.x, RIPPLE.y, RIPPLE.z);
    let broad = clamp(width * BROAD.x, BROAD.y, BROAD.z);
    let steep = (STEEP.x + STEEP.y * speed + STEEP.z * churned + WAKE_RIPPLE * wake) * seen_fade(footprint / wave);
    let steep_broad = (STEEP_BROAD.x + STEEP_BROAD.y * speed) * seen_fade(footprint / broad);
    // Standing waves on fast shallows.
    let chop = STANDING.x * smoothstep(STANDING.z, STANDING.w, froude) * seen_fade(footprint / wave);
    let standing = ripple(q / (wave * STANDING.y) + 7.7, frame.stretch) * chop;
    let g = drifting(q / wave, speed / wave, frame.stretch, fine) * steep + drifting(q / broad + 3.1, speed / broad, frame.stretch, slow) * steep_broad + standing;
    let slope = (channel.downstream * g.x + channel.left * g.y) * ruffled;
    let n = normalize(vec3(-slope.x, 1.0, -slope.y));
    let seen = below(in.position.xy, pos, n, V, pos.y);

    // Patchy drifting foam on rough water.
    let patches = drifting(q / FOAM_CELLS.x, speed / FOAM_CELLS.x, 1.0, fine);
    let bubbles = drifting(q / FOAM_CELLS.y, speed / FOAM_CELLS.y, 1.0, fine);
    let breakup = saturate(0.5 + (patches.x + patches.y) * 1.5);
    let cover = churned * FOAM_COVER;
    let froth = saturate(0.6 + (bubbles.x - bubbles.y) * 2.0);
    let foam = smoothstep(1.0 - cover, 1.0 - cover + FOAM_EDGE, breakup) * saturate(cover * 1.6) * froth * FOAM_OPACITY;
    // Scum lines gather on calm eddy lines.
    let eddy = smoothstep(EDDIES.x - 0.25, EDDIES.x, abs(across.x)) * (1.0 - smoothstep(EDDIES.y, 1.0, abs(across.x)));
    let gather = (SCUM_CALM + (1.0 - SCUM_CALM) * eddy) * (1.0 - smoothstep(CHURN.x, CHURN.y, speed));
    let scum = streak_lines(q, vec2(speed, footprint), slow) * gather * SCUM_OPACITY * river * (1.0 - foam);
    let covered = foam + scum;
    let roughness = mix(0.04, 0.6, covered);

    let R = reflect(-V, n);
    let mirrored = mirror(pos, n, V, roughness);
    let shore = hidden(R, frame, vec4(across, width * 0.5)) * (1.0 - mirrored.a);
    let clarity = max(smoothstep(CLEARING.x, CLEARING.y, fall), in.color.w * LAKE_CLARITY);
    let water_body = Medium(mix(TURBID.extinction, CLEAR.extinction, clarity), mix(TURBID.deep, CLEAR.deep, clarity));
    let surface = backscatter(water_body, seen) * (1.0 - covered) + SCUM * scum + FOAM * foam;
    let lit = water_light(in.position.xy, pos, Surface(n, V, surface, roughness, mirrored.a + shore));
    let banks = water_light(in.position.xy, pos, Surface(UP, UP, SHORE_ALBEDO, 1.0, 1.0));
    let toward = reflectance(roughness, max(dot(n, V), 1e-4));
    // Sun focused on the bed.
    let focus = caustics(q, vec2(speed, footprint), fine) * CAUSTIC_LIGHT.x * exp(-depth / CAUSTIC_LIGHT.y) * river;
    let through = seen.light * (1.0 + focus) * kept(water_body, seen) * (1.0 - covered);

    let water = lit + toward * (mirrored.rgb * mirrored.a + banks * shore) * (1.0 - covered) + (1.0 - toward) * through;
    let shown = 1.0 - smoothstep(carved(width) * MORPH_START, carved(width), range);

    var out: FragmentOutput;
    out.color = vec4(mix(scene_light(in.position.xy), water, shown), 1.0);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
