import bevy_pbr::render::{
    mesh_view_bindings::{view, lights, view_transmission_texture, view_transmission_sampler},
    view_transformations::{depth_ndc_to_view_z, ndc_to_uv, uv_to_ndc, position_ndc_to_world, position_world_to_clip, position_world_to_ndc},
};
import bevy_pbr::prepass::utils::prepass_depth;
import package::light::main_light;

/// Water's refractive index.
const IOR: f32 = 1.333;
/// Longest refracted bend, metres.
const BEND_REACH: f32 = 6.0;
/// Farthest depth sample, kept finite.
const FAR_DEPTH: f32 = 1e-7;
/// Mirror march steps and refinement halvings.
const MIRROR_STEPS: u32 = 20u;
const MIRROR_REFINE: u32 = 4u;
/// Mirror reach; steps grow cubically.
const MIRROR_REACH: f32 = 1000.0;
/// Hit depth tolerance per step length.
const MIRROR_THICKNESS: f32 = 2.0;
/// Roughness where mirroring fades out.
const MIRROR_ROUGH = vec2<f32>(0.6, 0.85);
/// Screen border share where mirroring fades.
const MIRROR_EDGE: f32 = 0.04;

/// Water extinction and deep albedo.
struct Medium {
    extinction: vec3<f32>,
    deep: vec3<f32>,
}

/// What lies under the surface.
struct Below {
    /// Exposed scene radiance, before the water.
    light: vec3<f32>,
    /// Water crossed by the refracted ray.
    path: f32,
    /// Ground depth under the calm level.
    depth: f32,
    ground: vec3<f32>,
}

/// Rendered-region share at pixel.
fn region_uv(frag: vec2<f32>) -> vec2<f32> {
    return (frag - view.main_pass_viewport.xy) / view.main_pass_viewport.zw;
}

/// Rendered pixel of NDC point.
fn pixel(ndc: vec2<f32>) -> vec2<f32> {
    return ndc_to_uv(ndc) * view.main_pass_viewport.zw + view.main_pass_viewport.xy;
}

/// Opaque scene point at a pixel.
fn scene_point(frag: vec2<f32>) -> vec3<f32> {
    let depth = max(prepass_depth(vec4(frag, 0.0, 0.0), 0u), FAR_DEPTH);
    return position_ndc_to_world(vec3(uv_to_ndc(region_uv(frag)), depth));
}

/// Exposed opaque radiance at a pixel.
fn scene_light(frag: vec2<f32>) -> vec3<f32> {
    let uv = frag / vec2<f32>(textureDimensions(view_transmission_texture));
    return textureSampleLevel(view_transmission_texture, view_transmission_sampler, uv, 0.0).rgb;
}

/// Refracted scene under the surface.
fn below(frag: vec2<f32>, pos: vec3<f32>, n: vec3<f32>, V: vec3<f32>, level: f32) -> Below {
    let straight = scene_point(frag);
    let bent = refract(-V, n, 1.0 / IOR);
    let guess = pos + bent * min(distance(pos, straight), BEND_REACH);
    var at = pixel(position_world_to_ndc(guess).xy);
    var ground = scene_point(at);
    // Off-screen or blocked rays look straight.
    let uv = region_uv(at);
    let off = any(uv < vec2(0.0)) || any(uv > vec2(1.0));
    if off || distance(view.world_position, ground) < distance(view.world_position, pos) {
        at = frag;
        ground = straight;
    }
    return Below(scene_light(at), distance(pos, ground), max(level - ground.y, 0.0), ground);
}

/// Refracted sun cosine underwater.
fn sun_under(L: vec3<f32>) -> f32 {
    return sqrt(1.0 - (1.0 - L.y * L.y) / (IOR * IOR));
}

/// Light surviving sky to eye.
fn kept(m: Medium, b: Below) -> vec3<f32> {
    let down = b.depth / sun_under(lights.directional_lights[main_light()].direction_to_light);
    return exp(-m.extinction * (b.path + down));
}

/// Back-scattered body albedo.
fn backscatter(m: Medium, b: Below) -> vec3<f32> {
    return m.deep * (1.0 - exp(-m.extinction * b.path));
}

/// Screen border fade.
fn on_screen(frag: vec2<f32>) -> f32 {
    let uv = region_uv(frag);
    let edge = min(uv, 1.0 - uv);
    return smoothstep(0.0, MIRROR_EDGE, min(edge.x, edge.y));
}

/// Ray depth past scene, pixel.
fn past_scene(p: vec3<f32>) -> vec3<f32> {
    let clip = position_world_to_clip(p);
    if clip.w <= 0.0 {
        return vec3(-1.0);
    }
    let frag = pixel(clip.xy / clip.w);
    let scene = -depth_ndc_to_view_z(prepass_depth(vec4(frag, 0.0, 0.0), 0u));
    return vec3(frag, clip.w - scene);
}

/// Mirrored radiance and confidence.
fn mirror(pos: vec3<f32>, n: vec3<f32>, V: vec3<f32>, roughness: f32) -> vec4<f32> {
    let fade = 1.0 - smoothstep(MIRROR_ROUGH.x, MIRROR_ROUGH.y, roughness);
    let R = reflect(-V, n);
    if fade <= 0.0 || R.y <= 0.0 {
        return vec4(0.0);
    }
    var near = 0.0;
    for (var i = 1u; i <= MIRROR_STEPS; i++) {
        let s = (f32(i) - 0.5) / f32(MIRROR_STEPS);
        let far = MIRROR_REACH * s * s * s;
        let probe = past_scene(pos + R * far);
        if on_screen(probe.xy) <= 0.0 {
            break;
        }
        if probe.z > 0.0 && probe.z < (far - near) * MIRROR_THICKNESS {
            return refine(pos, R, vec2(near, far), fade);
        }
        near = far;
    }
    return vec4(0.0);
}

/// Refines hit, reads scene.
fn refine(pos: vec3<f32>, R: vec3<f32>, span: vec2<f32>, fade: f32) -> vec4<f32> {
    var s = span;
    var hit = past_scene(pos + R * s.y);
    for (var i = 0u; i < MIRROR_REFINE; i++) {
        let mid = 0.5 * (s.x + s.y);
        let probe = past_scene(pos + R * mid);
        let past = probe.z > 0.0;
        s = select(vec2(mid, s.y), vec2(s.x, mid), past);
        hit = select(hit, probe, past);
    }
    return vec4(scene_light(hit.xy), fade * on_screen(hit.xy));
}
