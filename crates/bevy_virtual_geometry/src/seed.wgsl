/// Starts visibility from the scene's depth.
@group(0) @binding(0) var<storage, read_write> blank: array<u64>;
@group(0) @binding(1) var scene: texture_depth_2d;
/// Pixels the visibility holds.
var<immediate> size: vec2<u32>;

@compute @workgroup_size(8, 8)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= size) {
        return;
    }
    let depth = textureLoad(scene, id.xy, 0);
    blank[id.y * size.x + id.x] = (u64(bitcast<u32>(depth)) << 32u) | u64(BACKGROUND);
}
