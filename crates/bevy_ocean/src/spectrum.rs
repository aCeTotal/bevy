use bevy_math::{ops, Vec2};
use core::f32::consts::{PI, TAU};

/// Grid points per side.
pub(crate) const SIDE: usize = 256;
pub(crate) const GRAVITY: f32 = 9.81;
/// Wind speed and fetch.
const WIND_SPEED: f32 = 8.5;
const FETCH: f32 = 100_000.0;
/// Wind bearing, radians.
pub(crate) const WIND_BEARING: f32 = 0.6;
/// JONSWAP peak sharpening.
const GAMMA: f32 = 3.3;
const SIGMA: (f32, f32) = (0.07, 0.09);
/// Sea repeats after this, seconds.
pub(crate) const LOOP: f32 = 256.0;
const SEED: u32 = 0x5EA5_EED5;
/// Swell height, period, bearing, spread.
const SWELL: (f32, f32, f32, f32) = (1.0, 11.0, 0.35, 24.0);
/// Swell band width per peak frequency.
const SWELL_BAND: f32 = 0.07;

/// Wavelengths one cascade holds.
#[derive(Clone, Copy)]
pub(crate) struct Band {
    /// Patch edge, metres.
    pub size: f32,
    pub longest: f32,
    pub shortest: f32,
}

/// Peak frequency and energy scale.
struct Sea {
    peak: f32,
    alpha: f32,
}

fn fetch_limited() -> Sea {
    let alpha = 0.076 * ops::powf(WIND_SPEED * WIND_SPEED / (FETCH * GRAVITY), 0.22);
    let peak = 22.0 * ops::cbrt(GRAVITY * GRAVITY / (WIND_SPEED * FETCH));
    Sea { peak, alpha }
}

/// JONSWAP energy per angular frequency.
fn jonswap(sea: &Sea, w: f32) -> f32 {
    let sigma = if w <= sea.peak { SIGMA.0 } else { SIGMA.1 };
    let (off, w2, ratio) = (w - sea.peak, w * w, sea.peak / w);
    let peak = ops::exp(-off * off / (2.0 * sigma * sigma * sea.peak * sea.peak));
    let ratio2 = ratio * ratio;
    sea.alpha * GRAVITY * GRAVITY / (w2 * w2 * w)
        * ops::exp(-1.25 * ratio2 * ratio2)
        * ops::powf(GAMMA, peak)
}

/// Donelan-Banner spreading over heading.
fn spreading(sea: &Sea, w: f32, heading: f32) -> f32 {
    let r = w / sea.peak;
    let beta = match r {
        r if r < 0.95 => 2.61 * ops::powf(r, 1.3),
        r if r < 1.6 => 2.28 * ops::powf(r, -1.3),
        r => ops::powf(10.0, -0.4 + 0.8393 * ops::exp(-0.567 * ops::ln(r * r))),
    };
    let cosh = ops::cosh(beta * heading);
    beta / (2.0 * ops::tanh(beta * PI)) / (cosh * cosh)
}

/// Wrapped angle of a wavevector off a bearing.
fn off(k: Vec2, bearing: f32) -> f32 {
    let a = ops::atan2(k.y, k.x) - bearing;
    (a + PI).rem_euclid(TAU) - PI
}

/// Swell energy per frequency, heading.
fn swell(w: f32, heading: f32) -> f32 {
    let (height, period, _, power) = SWELL;
    let peak = TAU / period;
    let width = SWELL_BAND * peak;
    let band = ops::exp(-0.5 * ((w - peak) / width) * ((w - peak) / width)) / (width * TAU.sqrt());
    // Near unit-area cos^2s spread.
    let spread = ops::powf(ops::cos(heading * 0.5), 2.0 * power) / (2.0 * (PI / power).sqrt());
    height * height / 16.0 * band * spread
}

/// Deep-water frequency, looping.
pub(crate) fn frequency(k: f32) -> f32 {
    let step = TAU / LOOP;
    ((GRAVITY * k).sqrt() / step).floor() * step
}

/// Wavevector of a grid slot.
pub(crate) fn wavevector(band: &Band, slot: usize) -> Vec2 {
    let signed = |i: usize| {
        if i < SIDE / 2 {
            i as f32
        } else {
            i as f32 - SIDE as f32
        }
    };
    Vec2::new(signed(slot % SIDE), signed(slot / SIDE)) * (TAU / band.size)
}

/// Height spectrum variance of a slot.
fn energy(sea: &Sea, band: &Band, k: Vec2) -> f32 {
    let len = k.length();
    let held = len > 0.0 && (TAU / band.longest..TAU / band.shortest).contains(&len);
    if !held {
        return 0.0;
    }
    let w = (GRAVITY * len).sqrt();
    let dk = TAU / band.size;
    let sea_state = jonswap(sea, w) * spreading(sea, w, off(k, WIND_BEARING)) + swell(w, off(k, SWELL.2));
    sea_state * GRAVITY / (2.0 * w) / len * dk * dk
}

fn hash(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

/// Two standard normals, Box-Muller, as a complex number.
fn gaussian(seed: u32) -> Vec2 {
    let u = |s: u32| (hash(s) as f32 + 1.0) / (u32::MAX as f32 + 2.0);
    let (a, b) = (u(seed), u(seed ^ 0x9E37_79B9));
    Vec2::from_angle(TAU * b) * (-2.0 * ops::ln(a)).sqrt()
}

/// Starting amplitudes of a cascade.
pub(crate) fn amplitudes(band: &Band, cascade: u32) -> Vec<Vec2> {
    let sea = fetch_limited();
    let salt = hash(SEED ^ cascade);
    (0..SIDE * SIDE)
        .map(|slot| {
            gaussian(hash(salt ^ slot as u32))
                * (energy(&sea, band, wavevector(band, slot)) / 4.0).sqrt()
        })
        .collect()
}

/// Mean squared slope a cascade adds.
pub(crate) fn slope_variance(band: &Band) -> f32 {
    let sea = fetch_limited();
    (0..SIDE * SIDE)
        .map(|slot| wavevector(band, slot))
        .map(|k| k.length_squared() * energy(&sea, band, k))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHOLE: Band = Band {
        size: 400.0,
        longest: 400.0,
        shortest: 6.3,
    };

    #[test]
    fn significant_height_fits_a_breeze_on_swell() {
        let sea = fetch_limited();
        let variance: f32 = (0..SIDE * SIDE)
            .map(|slot| energy(&sea, &WHOLE, wavevector(&WHOLE, slot)))
            .sum();
        let height = 4.0 * variance.sqrt();
        assert!(
            (1.4..2.4).contains(&height),
            "significant height {height} m"
        );
    }

    #[test]
    fn spreading_integrates_to_one() {
        let sea = fetch_limited();
        let steps = 4096;
        let total: f32 = (0..steps)
            .map(|i| {
                spreading(
                    &sea,
                    sea.peak * 1.3,
                    -PI + TAU * (i as f32 + 0.5) / steps as f32,
                )
            })
            .sum::<f32>()
            * TAU
            / steps as f32;
        assert!((total - 1.0).abs() < 0.01, "spread {total}");
    }
}
