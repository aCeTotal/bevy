import bevy_render::view::View;
import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_pbr::render::view_transformations::uv_to_ndc;

struct PreviousViewUniforms {
    view_from_world: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> previous_view: PreviousViewUniforms;
@group(0) @binding(2) var final_depth: texture_depth_2d;
@group(0) @binding(3) var opaque_depth: texture_depth_2d;

/// Camera motion of water drawn over the opaque scene.
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(final_depth, pixel, 0);
    if depth <= textureLoad(opaque_depth, pixel, 0) {
        discard;
    }
    let world = view.world_from_clip * vec4(uv_to_ndc(in.uv), depth, 1.0);
    let now = view.unjittered_clip_from_world * world;
    let before = previous_view.unjittered_clip_from_world * world;
    return vec4((now.xy / now.w - before.xy / before.w) * vec2(0.5, -0.5), 0.0, 1.0);
}
