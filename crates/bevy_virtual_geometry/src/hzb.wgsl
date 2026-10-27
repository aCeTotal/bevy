/// Farthest-depth occlusion pyramid.
@group(0) @binding(0) var<storage, read> visibility: array<u64>;
@group(0) @binding(1) var finer: texture_2d<f32>;
@group(0) @binding(2) var coarser: texture_storage_2d<r32float, write>;

/// Source is the visibility buffer.
override FIRST: bool;
/// Texels the source holds.
var<immediate> source: vec2<u32>;

fn depth_at(p: vec2<u32>) -> f32 {
    let q = min(p, source - 1u);
    if FIRST {
        return depth_of(visibility[q.y * source.x + q.x]);
    }
    return textureLoad(finer, q, 0).x;
}

@compute @workgroup_size(8, 8)
fn hzb(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.xy * 2u;
    if any(p >= source) {
        return;
    }
    let far = min(min(depth_at(p), depth_at(p + vec2(1u, 0u))), min(depth_at(p + vec2(0u, 1u)), depth_at(p + 1u)));
    textureStore(coarser, id.xy, vec4(far));
}
