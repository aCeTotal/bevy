import package::light_probe::{LightProbeQueryResult, light_probe_iterator_new, light_probe_iterator_next};
import package::render::mesh_view_bindings::{
    irradiance_volumes,
    irradiance_volume,
    irradiance_volume_sampler,
};
import package::render::clustered_forward::ClusterableObjectIndexRanges;

@if(LIGHTMAP) import package::render::mesh_view_types::LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE;

/// Lookup lift off surfaces, in cells.
const LIFT: f32 = 0.5;
/// No light, open sky.
const OPEN = vec4<f32>(0.0, 0.0, 0.0, 1.0);

/// Irradiance per face; sky visibility in alpha.
///
/// See:
/// https://advances.realtimerendering.com/s2006/Mitchell-ShadingInValvesSourceEngine.pdf
/// Slide 28, "Ambient Cube Basis"
struct AmbientCube {
    faces: array<vec4<f32>, 6>,
}

/// Finest volume holding a point, and the next coarser.
struct Nested {
    fine: LightProbeQueryResult,
    coarse: LightProbeQueryResult,
}

/// World-to-volume scale; larger is finer.
fn fineness(volume: LightProbeQueryResult) -> f32 {
    return length(volume.light_from_world[0].xyz);
}

/// Nested volumes: the finest wins, fading into the next coarser.
fn nested_volumes(world_position: vec3<f32>, ranges: ptr<function, ClusterableObjectIndexRanges>) -> Nested {
    var nested: Nested;
    nested.fine.texture_index = -1;
    nested.coarse.texture_index = -1;
    var iterator = light_probe_iterator_new(world_position, true, ranges);
    while (true) {
        let volume = light_probe_iterator_next(&iterator);
        if (volume.texture_index < 0) {
            break;
        }
        @if(LIGHTMAP)
        if ((volume.flags & LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE) == 0u) {
            continue;
        }
        if (nested.fine.texture_index < 0 || fineness(volume) > fineness(nested.fine)) {
            nested.coarse = nested.fine;
            nested.fine = volume;
            continue;
        }
        if (nested.coarse.texture_index < 0 || fineness(volume) > fineness(nested.coarse)) {
            nested.coarse = volume;
        }
    }
    return nested;
}

fn atlas_size(index: i32) -> vec3<f32> {
    @if(MULTIPLE_LIGHT_PROBES_IN_ARRAY)
    return vec3<f32>(textureDimensions(irradiance_volumes[index]));
    @else
    return vec3<f32>(textureDimensions(irradiance_volume));
}

fn atlas_sample(index: i32, uvw: vec3<f32>) -> vec4<f32> {
    @if(MULTIPLE_LIGHT_PROBES_IN_ARRAY)
    return textureSampleLevel(irradiance_volumes[index], irradiance_volume_sampler, uvw, 0.0);
    @else
    return textureSampleLevel(irradiance_volume, irradiance_volume_sampler, uvw, 0.0);
}

/// Atlas point, lifted along `N` and clamped against bleed.
fn atlas_uvw(volume: LightProbeQueryResult, world_position: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    let atlas = atlas_size(volume.texture_index);
    let cells = atlas / vec3(1.0, 2.0, 3.0);
    let local = (volume.light_from_world * vec4(world_position, 1.0)).xyz;
    let lift = normalize((volume.light_from_world * vec4(N, 0.0)).xyz * cells) * LIFT;
    return clamp((local + 0.5) * cells + lift, vec3(0.5), cells - vec3(0.5)) / atlas;
}

/// Face `face` of one volume; the bottom half of each slice holds negative faces.
fn volume_face(volume: LightProbeQueryResult, uvw: vec3<f32>, face: u32) -> vec4<f32> {
    let offset = vec3(0.0, f32(face % 2u) * 0.5, f32(face / 2u) / 3.0);
    return atlas_sample(volume.texture_index, uvw + offset) * vec4(vec3(volume.intensity), 1.0);
}

/// One volume's light toward `dir`.
fn volume_toward(volume: LightProbeQueryResult, uvw: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let negative = vec3<u32>(dir < vec3(0.0));
    let dd = dir * dir;
    return volume_face(volume, uvw, negative.x) * dd.x
        + volume_face(volume, uvw, 2u + negative.y) * dd.y
        + volume_face(volume, uvw, 4u + negative.z) * dd.z;
}

/// Light and sky visibility toward `dir`, read at `world_position` lifted along `N`.
fn irradiance_volume_toward(
    world_position: vec3<f32>,
    N: vec3<f32>,
    dir: vec3<f32>,
    ranges: ptr<function, ClusterableObjectIndexRanges>,
) -> vec4<f32> {
    let nested = nested_volumes(world_position, ranges);
    if (nested.fine.texture_index < 0) {
        return OPEN;
    }
    let fine = volume_toward(nested.fine, atlas_uvw(nested.fine, world_position, N), dir);
    if (nested.fine.weight >= 1.0 || nested.coarse.texture_index < 0) {
        return fine;
    }
    let coarse = volume_toward(nested.coarse, atlas_uvw(nested.coarse, world_position, N), dir);
    return mix(coarse, fine, nested.fine.weight);
}

/// Every face, for reading several directions.
fn irradiance_volume_cube(
    world_position: vec3<f32>,
    N: vec3<f32>,
    ranges: ptr<function, ClusterableObjectIndexRanges>,
) -> AmbientCube {
    var cube = AmbientCube(array(OPEN, OPEN, OPEN, OPEN, OPEN, OPEN));
    let nested = nested_volumes(world_position, ranges);
    if (nested.fine.texture_index < 0) {
        return cube;
    }
    let fine = atlas_uvw(nested.fine, world_position, N);
    for (var face = 0u; face < 6u; face++) {
        cube.faces[face] = volume_face(nested.fine, fine, face);
    }
    if (nested.fine.weight >= 1.0 || nested.coarse.texture_index < 0) {
        return cube;
    }
    let coarse = atlas_uvw(nested.coarse, world_position, N);
    for (var face = 0u; face < 6u; face++) {
        cube.faces[face] = mix(volume_face(nested.coarse, coarse, face), cube.faces[face], nested.fine.weight);
    }
    return cube;
}

/// Cube read toward `dir`.
fn ambient_cube_toward(cube: AmbientCube, dir: vec3<f32>) -> vec4<f32> {
    let dd = dir * dir;
    let x = select(cube.faces[0], cube.faces[1], dir.x < 0.0);
    let y = select(cube.faces[2], cube.faces[3], dir.y < 0.0);
    let z = select(cube.faces[4], cube.faces[5], dir.z < 0.0);
    return x * dd.x + y * dd.y + z * dd.z;
}

/// Diffuse light along `N`.
/// Irradiance along `N`, with coverage.
fn irradiance_volume_light(
    world_position: vec3<f32>,
    N: vec3<f32>,
    ranges: ptr<function, ClusterableObjectIndexRanges>,
) -> vec4<f32> {
    let nested = nested_volumes(world_position, ranges);
    if (nested.fine.texture_index < 0) {
        return vec4<f32>(0.0);
    }
    let fine = volume_toward(nested.fine, atlas_uvw(nested.fine, world_position, N), N).rgb;
    if (nested.fine.weight >= 1.0) {
        return vec4<f32>(fine, 1.0);
    }
    if (nested.coarse.texture_index < 0) {
        return vec4<f32>(fine, nested.fine.weight);
    }
    let coarse = volume_toward(nested.coarse, atlas_uvw(nested.coarse, world_position, N), N).rgb;
    return vec4<f32>(mix(coarse, fine, nested.fine.weight), 1.0);
}
