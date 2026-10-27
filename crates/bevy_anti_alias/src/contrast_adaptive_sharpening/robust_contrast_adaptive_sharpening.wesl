import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_core_pipeline::tonemapping::rcas::{Cross, rcas};

struct CASUniforms {
    sharpness: f32,
};

@group(0) @binding(0) var screenTexture: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> uniforms: CASUniforms;

/// The CAS node runs after tonemapping, so the input will be in the range of 0 to 1.
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Algorithm uses minimal 3x3 pixel neighborhood.
    //    b
    //  d e f
    //    h
    let b = textureSample(screenTexture, samp, in.uv, vec2<i32>(0, -1)).rgb;
    let d = textureSample(screenTexture, samp, in.uv, vec2<i32>(-1, 0)).rgb;
    // We need the alpha value of the pixel we're working on for the output
    let e = textureSample(screenTexture, samp, in.uv).rgba;
    let f = textureSample(screenTexture, samp, in.uv, vec2<i32>(1, 0)).rgb;
    let h = textureSample(screenTexture, samp, in.uv, vec2<i32>(0, 1)).rgb;
    return vec4<f32>(rcas(Cross(b, d, e.rgb, f, h), uniforms.sharpness), e.w);
}
