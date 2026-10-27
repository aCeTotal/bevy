const NO_TEXTURE: u32 = 0xffffffffu;
const FOLIAGE: u32 = 1u;
const LAYERED: u32 = 2u;
const TEXTURES: u32 = 512u;
/// Bevy default.
const REFLECTANCE: f32 = 0.5;
/// Needles scatter; little sheen.
const LEAF_REFLECTANCE: f32 = 0.15;
const LEAF_ROUGHNESS: f32 = 0.85;

@group(2) @binding(9) var textures: binding_array<texture_2d<f32>, TEXTURES>;
@group(2) @binding(10) var texture_sampler: sampler;
@group(2) @binding(11) var layer_albedo: texture_2d_array<f32>;
@group(2) @binding(12) var layer_surface: texture_2d_array<f32>;
/// Repeats per metre, metalness, reflectance per layer.
@group(2) @binding(14) var<storage, read> layers: array<vec4<f32>>;

/// Attribute and screen slopes.
struct Uv {
    at: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
}

/// Material inputs of one surface.
struct Look {
    base_color: vec4<f32>,
    roughness: f32,
    metallic: f32,
    diffuse_occlusion: f32,
    specular_occlusion: f32,
    transmission: f32,
    reflectance: f32,
    normal: vec3<f32>,
}

fn sample(index: u32, uv: Uv, fallback: vec4<f32>) -> vec4<f32> {
    if index == NO_TEXTURE {
        return fallback;
    }
    return textureSampleGrad(textures[index], texture_sampler, uv.at, uv.dx, uv.dy);
}

/// Mip bias as gradient scale.
fn sharpening() -> f32 {
    return exp2(camera.mip_bias);
}

fn uv_of(s: Surface) -> Uv {
    let a = vertices[s.corners.x].uv;
    let b = vertices[s.corners.y].uv;
    let c = vertices[s.corners.z].uv;
    let w = s.weights;
    let sharp = sharpening();
    let at = a * w.weights.x + b * w.weights.y + c * w.weights.z;
    return Uv(at, (a * w.ddx.x + b * w.ddx.y + c * w.ddx.z) * sharp, (a * w.ddy.x + b * w.ddy.y + c * w.ddy.z) * sharp);
}

fn color_of(s: Surface) -> vec4<f32> {
    let w = s.weights.weights;
    return unpack_color(vertices[s.corners.x].color) * w.x + unpack_color(vertices[s.corners.y].color) * w.y
        + unpack_color(vertices[s.corners.z].color) * w.z;
}

/// Corner tangent, posed when animated.
fn corner_tangent(index: u32, i: Instance) -> vec4<f32> {
    let t = unpack_tangent(vertices[index].tangent);
    return vec4(turned_at(index, i, t.xyz, view.time), t.w);
}

fn tangent_of(s: Surface) -> vec4<f32> {
    let w = s.weights.weights;
    let i = instances[s.owner.instance];
    let t = corner_tangent(s.corners.x, i) * w.x + corner_tangent(s.corners.y, i) * w.y + corner_tangent(s.corners.z, i) * w.z;
    return vec4(normalize(turn(s.owner.rows, t.xyz)), sign(t.w));
}

/// Tangent-space normal map.
fn mapped(m: Material, s: Surface, uv: Uv, normal: vec3<f32>) -> vec3<f32> {
    if m.normal == NO_TEXTURE {
        return normal;
    }
    let tbn = calculate_tbn_mikktspace(normal, tangent_of(s));
    let n = sample(m.normal, uv, vec4(0.5, 0.5, 1.0, 1.0)).xyz * 2.0 - 1.0;
    return normalize(tbn * n);
}

/// Metallic-roughness textures.
fn standard(m: Material, s: Surface, normal: vec3<f32>) -> Look {
    let uv = uv_of(s);
    let orm = sample(m.orm, uv, vec4(1.0));
    let base = m.base_color * color_of(s) * sample(m.albedo, uv, vec4(1.0));
    let leaf = m.shading == FOLIAGE;
    let transmission = select(0.0, m.translucency, leaf);
    let roughness = select(m.roughness * orm.g, max(m.roughness * orm.g, LEAF_ROUGHNESS), leaf);
    let reflectance = select(REFLECTANCE, LEAF_REFLECTANCE, leaf);
    return Look(base, roughness, m.metallic * orm.b, orm.r, orm.r, transmission, reflectance, mapped(m, s, uv, normal));
}

/// Mesh point and screen slopes.
fn local_of(s: Surface) -> array<vec3<f32>, 3> {
    let a = position_of(s.corners.x);
    let b = position_of(s.corners.y);
    let c = position_of(s.corners.z);
    let w = s.weights;
    return array(a * w.weights.x + b * w.weights.y + c * w.weights.z, a * w.ddx.x + b * w.ddx.y + c * w.ddx.z, a * w.ddy.x + b * w.ddy.y + c * w.ddy.z);
}

/// Normal map bent onto face.
fn bend(n: vec3<f32>, t: vec3<f32>, surface: vec4<f32>) -> vec3<f32> {
    let xy = surface.rg * 2.0 - 1.0;
    let z = sqrt(max(1.0 - dot(xy, xy), 0.0));
    return normalize(t * xy.x + cross(n, t) * xy.y + n * z);
}

/// Per-face texture-array layers.
fn layered(s: Surface, normal: vec3<f32>) -> Look {
    let corner = vertices[s.corners.x];
    let layer = min(u32(corner.uv.x + 0.5), arrayLength(&layers) - 1u);
    let constants = layers[layer];
    let tangent = unpack_tangent(corner.tangent).xyz;
    let frame = mat2x3(tangent, cross(tangent, unpack_direction(corner.normal)));
    let scale = constants.x;
    let p = local_of(s);
    let uv = (p[0] * frame) * scale;
    let dx = (p[1] * frame) * scale * sharpening();
    let dy = (p[2] * frame) * scale * sharpening();
    let albedo = textureSampleGrad(layer_albedo, texture_sampler, uv, layer, dx, dy);
    let surface = textureSampleGrad(layer_surface, texture_sampler, uv, layer, dx, dy);
    let painted = unpack_color(corner.color);
    let tint = select(painted.rgb, unpack_color(instances[s.owner.instance].tint).rgb, painted.a == 0.0);
    let t = normalize(turn(s.owner.rows, tangent));
    let across = normalize(t - normal * dot(normal, t));
    let roughness = max(surface.b, 1.0 - max(tint.r, max(tint.g, tint.b)));
    return Look(vec4(albedo.rgb * tint, 1.0), roughness, constants.y, surface.a, surface.a, 0.0, constants.z, bend(normal, across, surface));
}

fn look_of(m: Material, s: Surface, normal: vec3<f32>) -> Look {
    if m.shading == LAYERED {
        return layered(s, normal);
    }
    return standard(m, s, normal);
}
