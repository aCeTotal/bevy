/// Grid points per side and cascades, as waves.rs.
const SIDE: u32 = 256u;
const BITS: u32 = 8u;
const CASCADES: u32 = 3u;
const POINTS: u32 = SIDE * SIDE;
/// Crest sharpening, 0 to 1.
const CHOP: f32 = 0.9;
/// Foam-starting squeeze and gain.
const FOAM_FOLD: f32 = 0.78;
const FOAM_GAIN: f32 = 4.0;
const TAU: f32 = 6.2831853;

struct Clock {
    /// Seconds into the loop.
    time: f32,
    /// Foam kept since the last step.
    fade: f32,
}

@group(0) @binding(0) var<uniform> clock: Clock;
/// Per slot: start amplitude and its mirror's conjugate.
@group(0) @binding(1) var<storage, read> starts: array<vec4<f32>>;
/// Per slot: wavevector, inverse length, frequency.
@group(0) @binding(2) var<storage, read> waves: array<vec4<f32>>;
/// Four packed complex field pairs per cascade.
@group(0) @binding(3) var<storage, read_write> fields: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> foam: array<f32>;
@group(0) @binding(5) var shape: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(6) var slope: texture_storage_2d_array<rgba16float, write>;

fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// Times the imaginary unit.
fn quarter(a: vec2<f32>) -> vec2<f32> {
    return vec2(-a.y, a.x);
}

fn turn(angle: f32) -> vec2<f32> {
    return vec2(cos(angle), sin(angle));
}

fn field(cascade: u32, pair: u32, slot: u32) -> u32 {
    return (cascade * 4u + pair) * POINTS + slot;
}

/// Spectrum at the clock's time, four packed pairs.
@compute @workgroup_size(64)
fn spectra(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= CASCADES * POINTS {
        return;
    }
    let cascade = index / POINTS;
    let slot = index % POINTS;
    let start = starts[index];
    let wave = waves[index];
    let phase = turn(wave.w * clock.time);
    let h = mul(start.xy, phase) + mul(start.zw, vec2(phase.x, -phase.y));
    let u = wave.xy * wave.z;
    let ih = quarter(h);
    fields[field(cascade, 0u, slot)] = h * (1.0 - CHOP * u.x);
    fields[field(cascade, 1u, slot)] = ih * (CHOP * u.y) - h * wave.x;
    fields[field(cascade, 2u, slot)] = ih * (wave.y - CHOP * wave.x * u.x);
    fields[field(cascade, 3u, slot)] = h * (-CHOP * wave.y * u.y) - ih * (CHOP * wave.y * u.x);
}

var<workgroup> line: array<vec2<f32>, SIDE>;

/// One line of `SIDE` points, in place, positive exponent.
fn transform(base: u32, step: u32, thread: u32) {
    for (var i = thread; i < SIDE; i += SIDE / 2u) {
        line[reverseBits(i) >> (32u - BITS)] = fields[base + i * step];
    }
    workgroupBarrier();
    for (var half = 1u; half < SIDE; half <<= 1u) {
        let k = thread % half;
        let a = (thread / half) * 2u * half + k;
        let t = mul(line[a + half], turn(TAU * f32(k) / f32(2u * half)));
        let low = line[a];
        workgroupBarrier();
        line[a] = low + t;
        line[a + half] = low - t;
        workgroupBarrier();
    }
    for (var i = thread; i < SIDE; i += SIDE / 2u) {
        fields[base + i * step] = line[i];
    }
}

/// Lines of one pair: x picks the row, y the pair, z the cascade.
@compute @workgroup_size(128)
fn rows(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) thread: u32) {
    transform(field(group.z, group.y, group.x * SIDE), 1u, thread);
}

@compute @workgroup_size(128)
fn columns(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) thread: u32) {
    transform(field(group.z, group.y, group.x), SIDE, thread);
}

/// Unpacks fields into the textures the sea reads, folding foam in.
@compute @workgroup_size(64)
fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= CASCADES * POINTS {
        return;
    }
    let cascade = index / POINTS;
    let slot = index % POINTS;
    let a = fields[field(cascade, 0u, slot)];
    let b = fields[field(cascade, 1u, slot)];
    let c = fields[field(cascade, 2u, slot)];
    let d = fields[field(cascade, 3u, slot)];
    let fold = (1.0 + c.y) * (1.0 + d.x) - d.y * d.y;
    let fresh = clamp((FOAM_FOLD - fold) * FOAM_GAIN, 0.0, 1.0);
    let kept = max(foam[index] * clock.fade, fresh);
    foam[index] = kept;
    let at = vec2<i32>(i32(slot % SIDE), i32(slot / SIDE));
    textureStore(shape, at, cascade, vec4(a.y, a.x, b.x, kept));
    textureStore(slope, at, cascade, vec4(b.y, c.x, c.y, d.x));
}
