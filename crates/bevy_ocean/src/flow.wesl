import bevy_pbr::render::mesh_view_bindings::globals;

/// Ripple octaves.
const OCTAVES: u32 = 4u;
/// Scum line length and spacing, metres; breaks and line sharpness.
const STREAK = vec2<f32>(14.0, 1.8);
const STREAK_BREAKS: f32 = 9.0;
const STREAK_SHARP: f32 = 0.08;
/// Glassy patch length and breadth per width, least metres; glassy band.
const SLICK = vec3<f32>(0.9, 0.3, 6.0);
const SLICK_BAND = vec2<f32>(0.45, 0.7);
/// Caustic cells, metres, finer cells and sharpness.
const CAUSTIC = vec3<f32>(0.9, 0.55, 6.0);

/// Integer hash in 0..1.
fn hash(i: vec2<i32>) -> f32 {
    let s = ((bitcast<u32>(i.x) * 1597334677u) ^ (bitcast<u32>(i.y) * 3812015801u)) * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f32(((w >> 22u) ^ w) >> 8u) / 16777216.0;
}

/// Value noise with its gradient.
fn noise(p: vec2<f32>) -> vec3<f32> {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = hash(i);
    let b = hash(i + vec2(1, 0));
    let c = hash(i + vec2(0, 1));
    let d = hash(i + vec2(1, 1));
    let value = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    let grad = du * vec2(b - a + (a - b - c + d) * u.y, c - a + (a - b - c + d) * u.x);
    return vec3(value, grad);
}

/// Ripple slope, downstream along x, stretched by `stretch`.
fn ripple(q: vec2<f32>, stretch: f32) -> vec2<f32> {
    let p = vec2(q.x / stretch, q.y);
    var g = vec2(0.0);
    var scale = 1.0;
    for (var i = 0u; i < OCTAVES; i++) {
        let turn = f32(i) * 0.9;
        let r = mat2x2(cos(turn), sin(turn), -sin(turn), cos(turn));
        g += transpose(r) * noise(r * p * scale + f32(i) * 7.3).yz;
        scale *= 2.03;
    }
    g /= f32(OCTAVES);
    return vec2(g.x / stretch, g.y);
}

/// Two crossfaded advection phases.
struct Flow {
    phases: vec2<f32>,
    blend: f32,
}

/// Phases restarting every `period` seconds.
fn flow_over(period: f32) -> Flow {
    let cycle = globals.time / period;
    let phases = fract(vec2(cycle, cycle + 0.5));
    return Flow(phases * period, abs(phases.x - 0.5) * 2.0);
}

/// Ripple slope drifting `drift` metres a second downstream.
fn drifting(q: vec2<f32>, drift: f32, stretch: f32, f: Flow) -> vec2<f32> {
    let a = ripple(q - vec2(drift * f.phases.x, 0.0), stretch);
    let b = ripple(q - vec2(drift * f.phases.y, 0.0) + 0.37, stretch);
    return mix(a, b, f.blend);
}

/// Drifting noise value in 0..1.
fn drifting_value(q: vec2<f32>, drift: f32, f: Flow) -> f32 {
    let a = noise(q - vec2(drift * f.phases.x, 0.0)).x;
    let b = noise(q - vec2(drift * f.phases.y, 0.0) + 5.3).x;
    return mix(a, b, f.blend);
}

/// Glassy share of drifting slicks; `channel` holds drift and width.
fn slicks(q: vec2<f32>, channel: vec2<f32>, f: Flow) -> f32 {
    let size = max(vec2(SLICK.x, SLICK.y) * channel.y, vec2(SLICK.z, SLICK.z * SLICK.y / SLICK.x));
    let n = drifting_value(q / size, channel.x / size.x, f);
    return smoothstep(SLICK_BAND.x, SLICK_BAND.y, n);
}

/// Thin scum lines along the current, evened out where pixels outgrow them; `seen` holds drift and footprint.
fn streak_lines(q: vec2<f32>, seen: vec2<f32>, f: Flow) -> f32 {
    let n = drifting_value(q / STREAK, seen.x / STREAK.x, f);
    let line = 1.0 - abs(n * 2.0 - 1.0);
    let edge = max(STREAK_SHARP, seen.y / STREAK.y);
    let breaks = smoothstep(0.3, 0.6, drifting_value(q / STREAK_BREAKS + 11.0, seen.x / STREAK_BREAKS, f));
    return smoothstep(1.0 - edge, 1.0, line) * breaks * STREAK_SHARP / edge;
}

/// Drifting caustic net on the bed.
fn caustics(q: vec2<f32>, seen: vec2<f32>, f: Flow) -> f32 {
    let a = 1.0 - abs(drifting_value(q / CAUSTIC.x, seen.x / CAUSTIC.x, f) * 2.0 - 1.0);
    let b = 1.0 - abs(drifting_value(q / CAUSTIC.y + 4.7, seen.x / CAUSTIC.y, f) * 2.0 - 1.0);
    let net = pow(a * b, CAUSTIC.z);
    return net * (1.0 - smoothstep(CAUSTIC.x * 0.05, CAUSTIC.x * 0.2, seen.y));
}
