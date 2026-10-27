/// Debug views, as visualize.rs.
const SHOW_TRIANGLES: u32 = 2u;
const SHOW_INSTANCES: u32 = 3u;
const SHOW_RASTER: u32 = 4u;
const SHOW_DETAIL: u32 = 5u;
const SOFTWARE = vec3<f32>(0.2, 0.8, 0.3);
const HARDWARE = vec3<f32>(0.95, 0.5, 0.1);
const FINE = vec3<f32>(0.15, 0.35, 1.0);
const COARSE = vec3<f32>(1.0, 0.25, 0.1);
/// Blue-to-red error span, metres.
const DETAIL_ERRORS = vec2<f32>(0.001, 4.0);
const DEBUG_LIGHT = vec3<f32>(0.36, 0.8, 0.48);

/// Saturated pseudo-random colour.
fn hue(seed: u32) -> vec3<f32> {
    let h = (seed ^ (seed >> 15u)) * SPREAD;
    let angle = f32(h >> 8u) / HASH_RANGE;
    let rgb = saturate(abs(fract(vec3(angle) + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return rgb * (0.55 + 0.45 * f32(h & 255u) / 255.0);
}

/// Debug colour, lit for shape.
fn visualized(id: u32, s: Surface, normal: vec3<f32>) -> vec3<f32> {
    let slot = id >> TRIANGLE_BITS;
    let cluster = visible[slot].y;
    let error = log2(max(s.cluster.errors.x * s.owner.scale, DETAIL_ERRORS.x) / DETAIL_ERRORS.x);
    var color = hue((cluster * SPREAD) ^ s.owner.instance);
    switch view.visualize {
        case SHOW_TRIANGLES: {
            color = hue((id * SPREAD) ^ cluster);
        }
        case SHOW_INSTANCES: {
            color = hue(s.owner.instance);
        }
        case SHOW_RASTER: {
            color = select(SOFTWARE, HARDWARE, slot >= view.half_visible);
        }
        case SHOW_DETAIL: {
            color = mix(FINE, COARSE, saturate(error / log2(DETAIL_ERRORS.y / DETAIL_ERRORS.x)));
        }
        default: {}
    }
    return color * (0.35 + 0.65 * max(dot(normal, DEBUG_LIGHT), 0.0));
}
