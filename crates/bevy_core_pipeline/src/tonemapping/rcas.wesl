// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// This is set at the limit of providing unnatural results for sharpening.
const FSR_RCAS_LIMIT = 0.1875;
// -4.0 instead of -1.0 to avoid issues with MSAA.
const peakC = vec2<f32>(10.0, -40.0);

/// A pixel and its four edge neighbours.
struct Cross {
    b: vec3<f32>,
    d: vec3<f32>,
    e: vec3<f32>,
    f: vec3<f32>,
    h: vec3<f32>,
}

/// Robust Contrast Adaptive Sharpening (RCAS)
/// Based on the following implementation:
/// https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/ea97a113b0f9cadf519fbcff315cc539915a3acd/src/ffx-fsr2-api/shaders/ffx_fsr1.h#L672
/// RCAS uses a 5 tap filter in a cross pattern (same as CAS),
///    W                b
///  W 1 W  for taps  d e f
///    W                h
/// Where 'W' is the negative lobe weight.
///  output = (W*(b+d+f+h)+e)/(4*W+1)
/// RCAS solves for 'W' by seeing where the signal might clip out of the {0 to 1} input range,
///  0 == (W*(b+d+f+h)+e)/(4*W+1) -> W = -e/(b+d+f+h)
///  1 == (W*(b+d+f+h)+e)/(4*W+1) -> W = (1-e)/(b+d+f+h-4)
/// Then chooses the 'W' which results in no clipping, limits 'W', and multiplies by the 'sharp' amount.
/// RCAS uses 4x the maximum and 4x the minimum (depending on equation) in place of the individual taps,
/// as well as switching from 'e' to either the minimum or maximum (depending on side), to help in energy conservation.
/// With `RCAS_DENOISE`, a normalized highpass of luma reduces sharpening on grain.
/// Input must be in the range of 0 to 1, so after tonemapping.
fn rcas(c: Cross, sharpness: f32) -> vec3<f32> {
    // Min and max of ring.
    let mn4 = min(min(c.b, c.d), min(c.f, c.h));
    let mx4 = max(max(c.b, c.d), max(c.f, c.h));
    // Limiters
    // 4.0 to avoid issues with MSAA.
    let hitMin = mn4 / (4.0 * mx4);
    let hitMax = (peakC.x - mx4) / (peakC.y + 4.0 * mn4);
    let lobeRGB = max(-hitMin, hitMax);
    var lobe = max(-FSR_RCAS_LIMIT, min(0.0, max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)))) * sharpness;
@if(RCAS_DENOISE) {
    // Luma times 2.
    let bL = c.b.b * 0.5 + (c.b.r * 0.5 + c.b.g);
    let dL = c.d.b * 0.5 + (c.d.r * 0.5 + c.d.g);
    let eL = c.e.b * 0.5 + (c.e.r * 0.5 + c.e.g);
    let fL = c.f.b * 0.5 + (c.f.r * 0.5 + c.f.g);
    let hL = c.h.b * 0.5 + (c.h.r * 0.5 + c.h.g);
    // Noise detection.
    var noise = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    noise = saturate(abs(noise) / (max(max(bL, dL), max(fL, hL)) - min(min(bL, dL), min(fL, hL))));
    noise = 1.0 - 0.5 * noise;
    // Apply noise removal.
    lobe *= noise;
}
    return (lobe * c.b + lobe * c.d + lobe * c.f + lobe * c.h + c.e) / (4.0 * lobe + 1.0);
}
