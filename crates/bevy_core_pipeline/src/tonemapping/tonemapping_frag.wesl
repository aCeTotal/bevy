import bevy_render::{
    view::View,
    maths::powsafe,
};
import package::{
    fullscreen_vertex_shader::FullscreenVertexOutput,
    tonemapping::{tone_mapping, screen_space_dither},
};
@if(SHARPEN)
import package::tonemapping::rcas::{Cross, rcas};
@if(COMPOSITING_SPACE_SRGB)
import bevy_render::color_operations::{srgb_to_linear, linear_to_srgb};
@if(COMPOSITING_SPACE_OKLAB)
import bevy_render::color_operations::{oklab_to_linear_rgb, linear_rgb_to_oklab};

@group(0) @binding(0) var<uniform> view: View;

@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;
@group(0) @binding(3) var dt_lut_texture: texture_3d<f32>;
@group(0) @binding(4) var dt_lut_sampler: sampler;

/// Tonemapped colour of one pixel.
fn displayed(pixel: vec2<i32>) -> vec4<f32> {
    let edge = vec2<i32>(textureDimensions(hdr_texture)) - 1;
    let hdr_color = textureLoad(hdr_texture, clamp(pixel, vec2(0), edge), 0);

    var scene_color = hdr_color;
@if(COMPOSITING_SPACE_SRGB)
    scene_color = vec4(srgb_to_linear(scene_color.rgb), scene_color.a);
@if(COMPOSITING_SPACE_OKLAB)
    scene_color = vec4(oklab_to_linear_rgb(scene_color.rgb), scene_color.a);

    var output_rgb = tone_mapping(scene_color, view.color_grading).rgb;

@if(DEBAND_DITHER) {
    output_rgb = powsafe(output_rgb.rgb, 1.0 / 2.2);
    output_rgb = output_rgb + screen_space_dither(vec2<f32>(pixel) + 0.5);
    // This conversion back to linear space is required because our output texture format is
    // SRGB; the GPU will assume our output is linear and will apply an SRGB conversion.
    output_rgb = powsafe(output_rgb.rgb, 2.2);
}

@if(COMPOSITING_SPACE_SRGB)
    output_rgb = linear_to_srgb(output_rgb);
@if(COMPOSITING_SPACE_OKLAB)
    output_rgb = linear_rgb_to_oklab(output_rgb);

    return vec4<f32>(output_rgb, hdr_color.a);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let center = displayed(pixel);
@if(SHARPEN) {
    let cross = Cross(
        displayed(pixel + vec2(0, -1)).rgb,
        displayed(pixel + vec2(-1, 0)).rgb,
        center.rgb,
        displayed(pixel + vec2(1, 0)).rgb,
        displayed(pixel + vec2(0, 1)).rgb,
    );
    return vec4(rcas(cross, f32(constants::SHARPNESS_MILLI) / 1000.0), center.a);
}
    return center;
}
