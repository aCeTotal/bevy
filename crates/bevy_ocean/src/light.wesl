import bevy_pbr::render::{
    mesh_view_bindings::{view, lights},
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    pbr_lighting::{F_AB, EnvBRDFApprox, LightingInput, LAYER_BASE, directional_light, perceptualRoughnessToRoughness},
    clustered_forward::{view_fragment_cluster_index, unpack_clusterable_object_index_ranges},
    shadows::fetch_directional_shadow,
    view_transformations::position_world_to_view,
};
@if(ENVIRONMENT_MAP) import bevy_pbr::light_probe::environment_map::environment_map_light;

/// Reflectance giving water's 2% Fresnel.
const REFLECTANCE: f32 = 0.35;

/// Water surface lighting inputs.
struct Surface {
    N: vec3<f32>,
    V: vec3<f32>,
    /// Body and foam albedo.
    albedo: vec3<f32>,
    roughness: f32,
    /// Sky reflection the mirror replaces.
    mirrored: f32,
}

fn casts_shadows(light: u32) -> bool {
    return (lights.directional_lights[light].flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u;
}

/// The shadowing light, else the first.
fn main_light() -> u32 {
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        if casts_shadows(i) {
            return i;
        }
    }
    return 0u;
}

/// Roughness-averaged reflectance toward eye.
fn reflectance(roughness: f32, NdotV: f32) -> vec3<f32> {
    return EnvBRDFApprox(vec3(0.16 * REFLECTANCE * REFLECTANCE), F_AB(roughness, NdotV));
}

/// Exposed sun, moon and sky light.
fn water_light(frag: vec2<f32>, pos: vec3<f32>, s: Surface) -> vec3<f32> {
    var input: LightingInput;
    let NdotV = max(dot(s.N, s.V), 1e-4);
    input.layers[LAYER_BASE].NdotV = NdotV;
    input.layers[LAYER_BASE].N = s.N;
    input.layers[LAYER_BASE].R = reflect(-s.V, s.N);
    input.layers[LAYER_BASE].perceptual_roughness = s.roughness;
    input.layers[LAYER_BASE].roughness = perceptualRoughnessToRoughness(s.roughness);
    input.P = pos;
    input.V = s.V;
    input.diffuse_color = s.albedo;
    input.F0_dielectric = vec3(0.16 * REFLECTANCE * REFLECTANCE);
    input.F_ab = F_AB(s.roughness, NdotV);
    let view_z = position_world_to_view(pos).z;
    var light = vec3(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        var shadow = 1.0;
        if casts_shadows(i) {
            shadow = fetch_directional_shadow(i, vec4(pos, 1.0), s.N, view_z, frag);
        }
        light += directional_light(i, &input, true) * shadow;
    }
    @if(ENVIRONMENT_MAP) {
        let orthographic = view.clip_from_view[3].w == 1.0;
        var ranges = unpack_clusterable_object_index_ranges(view_fragment_cluster_index(frag, view_z, orthographic));
        let sky = environment_map_light(&input, &ranges, false);
        light += sky.diffuse + sky.specular * (1.0 - s.mirrored);
    }
    return light * view.exposure;
}
