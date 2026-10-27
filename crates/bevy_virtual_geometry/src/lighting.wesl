import bevy_pbr::render::mesh_view_bindings::{view as camera, lights};
import bevy_pbr::render::{
    clustered_forward::{unpack_clusterable_object_index_ranges, view_fragment_cluster_index},
    pbr_functions::{calculate_diffuse_color, calculate_F0_dielectric, calculate_tbn_mikktspace, calculate_view, main_pass_post_lighting_processing},
    pbr_lighting::{directional_light, directional_light_radiance, F_AB, LAYER_BASE, LightingInput, perceptualRoughnessToRoughness},
    pbr_types::{pbr_input_new, STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE, STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT},
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    shadows::fetch_directional_shadow,
};
import bevy_render::maths::PI;
@if(IRRADIANCE_VOLUME)
import bevy_pbr::light_probe::irradiance_volume::{ambient_cube_toward, irradiance_volume_cube};
@if(ENVIRONMENT_MAP)
import bevy_pbr::light_probe::environment_map::environment_map_light;
import bevy_pbr::light_probe::environment_map::radiance_sample_direction;
@if(SCREEN_SPACE_AMBIENT_OCCLUSION)
import bevy_pbr::render::mesh_view_bindings::screen_space_ambient_occlusion_texture;
@if(SCREEN_SPACE_AMBIENT_OCCLUSION)
import bevy_pbr::ssao::utils::ssao_multibounce;

/// Grazing floor, as Bevy.
const MIN_N_DOT_V: f32 = 0.0001;
/// Sky light kept at a plant's root and crown.
const CROWN_SHADE = vec2<f32>(0.35, 1.0);
/// Rise, in crown heights, of that shading.
const CROWN_RISE = vec2<f32>(0.15, 1.3);
/// Leaf brightness and warmth spread between plants.
const LEAF_SPREAD = vec2<f32>(0.22, 0.14);
/// Leaf turning: tilt, rate; leaves per metre.
const LEAF_TURN = vec2<f32>(0.3, 3.0);
const LEAF_CELLS: f32 = 4.0;

struct Shaded {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

/// Where a pixel gathers light.
struct Spot {
    world: vec3<f32>,
    /// Geometric normal, for shadow bias.
    normal: vec3<f32>,
    pixel: vec2<f32>,
}

/// Diffuse, specular occlusion with SSAO.
fn occlusion(look: Look, pixel: vec2<f32>, n_dot_v: f32) -> vec4<f32> {
    var o = vec4(vec3(look.diffuse_occlusion), look.specular_occlusion);
    @if(SCREEN_SPACE_AMBIENT_OCCLUSION) {
        let ssao = textureLoad(screen_space_ambient_occlusion_texture, vec2<i32>(pixel), 0i).r;
        let roughness = perceptualRoughnessToRoughness(look.roughness);
        let specular = saturate(pow(n_dot_v + ssao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ssao);
        o = vec4(min(o.xyz, ssao_multibounce(ssao, look.base_color.rgb)), min(o.w, specular));
    }
    return o;
}

/// Directional light `i` reaching the spot.
fn shadowed(i: u32, at: Spot, view_z: f32) -> f32 {
    if (lights.directional_lights[i].flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) == 0u {
        return 1.0;
    }
    return fetch_directional_shadow(i, vec4(at.world, 1.0), at.normal, view_z, at.pixel);
}

/// Sky light left under a plant's own crown.
fn crown_light(o: Owner, world: vec3<f32>) -> f32 {
    if o.wind.x <= 0.0 {
        return 1.0;
    }
    let rise = (world.y - root_of(o).y) * o.wind.y;
    return mix(CROWN_SHADE.x, CROWN_SHADE.y, smoothstep(CROWN_RISE.x, CROWN_RISE.y, rise));
}

/// Own leaf tint of a plant.
fn leaf_tint(o: Owner) -> vec3<f32> {
    let seed = seed_of(root_of(o).xz);
    let warm = fract(seed * 7.13) - 0.5;
    return (1.0 + (seed - 0.5) * LEAF_SPREAD.x) * vec3(1.0 + warm * LEAF_SPREAD.y, 1.0, 1.0 - warm * LEAF_SPREAD.y);
}

/// Leaves turning in the wind, catching light in passing waves.
fn turned_leaves(n: vec3<f32>, world: vec3<f32>, o: Owner) -> vec3<f32> {
    if o.wind.x <= 0.0 {
        return n;
    }
    // Each leaf keeps its own beat.
    let leaf = seed_of(vec2(world.x + world.y * 0.7, world.z - world.y * 0.4) * LEAF_CELLS);
    let beat = view.time * LEAF_TURN.y * (0.7 + 0.6 * leaf) * sqrt(view.wind_strength) + leaf * TAU;
    let turn = vec3(sin(beat), 0.4 * sin(beat * 1.7 + 1.3), cos(beat * 0.77 + 0.6)) * LEAF_TURN.x;
    // Leaves smaller than a pixel only shimmer.
    let near = 1.0 - smoothstep(TREMBLE_SEEN.x, TREMBLE_SEEN.y, distance(world, view.wind_eye));
    return normalize(n + turn * (0.35 + 0.65 * gust_front(world.xz, view.time)) * min(view.wind_strength, 2.0) * near);
}

/// Surface in Bevy BRDF terms.
fn brdf_input(look: Look, world: vec3<f32>, toward: vec3<f32>) -> LightingInput {
    let n_dot_v = max(dot(look.normal, toward), MIN_N_DOT_V);
    var input: LightingInput;
    input.layers[LAYER_BASE].NdotV = n_dot_v;
    input.layers[LAYER_BASE].N = look.normal;
    input.layers[LAYER_BASE].R = reflect(-toward, look.normal);
    input.layers[LAYER_BASE].perceptual_roughness = look.roughness;
    input.layers[LAYER_BASE].roughness = perceptualRoughnessToRoughness(look.roughness);
    input.P = world;
    input.V = toward;
    input.diffuse_color = calculate_diffuse_color(look.base_color.rgb, look.metallic, 0.0, look.transmission);
    input.metallic = look.metallic;
    input.F0_dielectric = calculate_F0_dielectric(vec3(look.reflectance));
    input.F0_metallic = look.base_color.rgb;
    input.F_ab = F_AB(look.roughness, n_dot_v);
    return input;
}

/// Light `i` through leaves, Lambert lobe.
fn backlight(i: u32, look: Look, world: vec3<f32>) -> vec3<f32> {
    let through = saturate(-dot(look.normal, lights.directional_lights[i].direction_to_light)) / PI;
    return look.base_color.rgb * (1.0 - look.metallic) * look.transmission * through * directional_light_radiance(i, world);
}

/// Sky specular, else surroundings.
@if(ENVIRONMENT_MAP)
fn reflection(input: ptr<function, LightingInput>, sky: vec3<f32>, seen: vec4<f32>) -> vec3<f32> {
    let f0 = mix((*input).F0_dielectric, (*input).F0_metallic, (*input).metallic);
    return mix((f0 * (*input).F_ab.x + (*input).F_ab.y) * seen.rgb, sky, seen.a);
}

/// Sun, sky and bounce light.
fn lit(look: Look, at: Spot) -> vec3<f32> {
    var input = brdf_input(look, at.world, calculate_view(vec4(at.world, 1.0), false));
    let occluded = occlusion(look, at.pixel, input.layers[LAYER_BASE].NdotV);
    let depth_row = vec4(camera.view_from_world[0].z, camera.view_from_world[1].z, camera.view_from_world[2].z, camera.view_from_world[3].z);
    let view_z = dot(depth_row, vec4(at.world, 1.0));
    var light = vec3(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        light += (directional_light(i, &input, true) + backlight(i, look, at.world)) * shadowed(i, at, view_z);
    }
    var ranges = unpack_clusterable_object_index_ranges(view_fragment_cluster_index(at.pixel, view_z, false));
    var bounced = false;
    // Surroundings light, sky seen.
    var seen = vec4(0.0, 0.0, 0.0, 1.0);
    @if(IRRADIANCE_VOLUME) {
        let around = irradiance_volume_cube(at.world, look.normal, &ranges);
        let through = look.base_color.rgb * look.transmission;
        light += (ambient_cube_toward(around, look.normal).rgb * input.diffuse_color + ambient_cube_toward(around, -look.normal).rgb * through) * occluded.xyz;
        let base = input.layers[LAYER_BASE];
        seen = ambient_cube_toward(around, radiance_sample_direction(base.N, base.R, base.roughness));
        bounced = true;
    }
    @if(ENVIRONMENT_MAP) {
        let sky = environment_map_light(&input, &ranges, bounced);
        light += sky.diffuse * occluded.xyz + reflection(&input, sky.specular, seen) * occluded.w;
    }
    return camera.exposure * light;
}

@fragment
fn shade(@builtin(position) p: vec4<f32>) -> Shaded {
    let v = visible_at(vec2<u32>(p.xy));
    if id_of(v) == BACKGROUND {
        discard;
    }
    let s = surface_at(id_of(v), p.xy);
    let m = materials[s.cluster.counts >> 16u];
    let world = world_position(s);
    var normal = surface_normal(s);
    if two_sided(s.cluster) && dot(normal, view.eye - world) < 0.0 {
        normal = -normal;
    }
    if view.visualize != 0u {
        return Shaded(depth_of(v), vec4(visualized(id_of(v), s, normal), 1.0));
    }
    var look = look_of(m, s, normal);
    let crown = crown_light(s.owner, world);
    look.diffuse_occlusion *= crown;
    look.specular_occlusion *= crown;
    if m.shading == FOLIAGE {
        look.base_color = vec4(look.base_color.rgb * leaf_tint(s.owner), look.base_color.a);
        look.normal = turned_leaves(look.normal, world, s.owner);
    }
    let color = lit(look, Spot(world, normal, p.xy));
    // Fog reads only placement.
    var pbr = pbr_input_new();
    pbr.material.flags = STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT | STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE;
    pbr.frag_coord = vec4(p.xy, depth_of(v), 1.0);
    pbr.world_position = vec4(world, 1.0);
    return Shaded(depth_of(v), main_pass_post_lighting_processing(pbr, vec4(color, 1.0)));
}
