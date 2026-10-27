use crate::smoothstep;
use crate::spectrum::{Band, SIDE};
use bevy_ecs::resource::Resource;
use bevy_math::{Vec2, Vec3, Vec3Swizzles, Vec4};
use half::f16;

/// Cascades, longest waves first.
pub(crate) const CASCADES: usize = 3;
pub(crate) const BANDS: [Band; CASCADES] = [
    Band {
        size: 512.0,
        longest: 512.0,
        shortest: 24.0,
    },
    Band {
        size: 96.0,
        longest: 24.0,
        shortest: 4.5,
    },
    Band {
        size: 17.0,
        longest: 4.5,
        shortest: 0.15,
    },
];
/// Patch share where waves fade.
const EDGE_START: f32 = 0.7;
/// Walks back to the water's source.
const INVERT_STEPS: usize = 4;
/// Bytes of one half-float rgba texel.
const TEXEL_BYTES: usize = 8;

/// Drawn patch the surface rides.
#[derive(Clone, Copy, Default)]
pub(crate) struct Patch {
    pub center: Vec2,
    pub half: f32,
    /// Vertex share of each cascade.
    pub reach: [f32; CASCADES],
    /// Waves animate.
    pub coastal: bool,
}

/// The sea state floating things ride.
#[derive(Resource, Default)]
pub struct Waves {
    /// Displacement and foam per cascade, read back as half floats.
    shapes: Vec<u8>,
    pub(crate) patch: Patch,
}

impl Waves {
    /// Takes the latest read-back fields.
    pub(crate) fn read(&mut self, shapes: &[u8]) {
        self.shapes.clear();
        self.shapes.extend_from_slice(shapes);
    }

    fn texel(&self, cascade: usize, x: i32, y: i32) -> Vec4 {
        let (x, y) = (
            x.rem_euclid(SIDE as i32) as usize,
            y.rem_euclid(SIDE as i32) as usize,
        );
        let at = ((cascade * SIDE + y) * SIDE + x) * TEXEL_BYTES;
        let half = |i: usize| {
            f16::from_le_bytes([self.shapes[at + 2 * i], self.shapes[at + 2 * i + 1]]).to_f32()
        };
        Vec4::new(half(0), half(1), half(2), half(3))
    }

    /// Bilinear, wrapping, in grid units.
    fn sample(&self, cascade: usize, at: Vec2) -> Vec4 {
        let base = at.floor();
        let f = at - base;
        let (x, y) = (base.x as i32, base.y as i32);
        let texel = |dx: i32, dy: i32| self.texel(cascade, x + dx, y + dy);
        texel(0, 0)
            .lerp(texel(1, 0), f.x)
            .lerp(texel(0, 1).lerp(texel(1, 1), f.x), f.y)
    }

    /// Displacement of water resting at `rest`.
    fn offset(&self, rest: Vec2, calm: f32) -> Vec3 {
        if self.shapes.len() < CASCADES * SIDE * SIDE * TEXEL_BYTES {
            return Vec3::ZERO;
        }
        let local = (rest - self.patch.center).abs() / self.patch.half;
        let fade = 1.0 - smoothstep(EDGE_START, 1.0, local.max_element());
        let sum: Vec3 = BANDS
            .iter()
            .zip(self.patch.reach)
            .enumerate()
            .map(|(c, (band, w))| self.sample(c, rest / band.size * SIDE as f32).truncate() * w)
            .sum();
        sum * calm * fade
    }

    /// Water displacement over `p`, calmed.
    pub fn surface(&self, p: Vec2, calm: f32) -> Vec3 {
        let mut rest = p;
        for _ in 0..INVERT_STEPS {
            rest = p - self.offset(rest, calm).xz();
        }
        self.offset(rest, calm)
    }

    /// Wave height over `p`.
    pub fn height(&self, p: Vec2, calm: f32) -> f32 {
        self.surface(p, calm).y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use core::f32::consts::TAU;

    /// A smooth swell in every cascade.
    fn sea() -> Waves {
        let patch = Patch {
            center: Vec2::ZERO,
            half: 1.0e4,
            reach: [1.0; CASCADES],
            coastal: true,
        };
        let mut shapes = Vec::new();
        for i in 0..CASCADES * SIDE * SIDE {
            let (x, y) = ((i % SIDE) as f32, ((i / SIDE) % SIDE) as f32);
            let phase = TAU * (x + 2.0 * y) / SIDE as f32;
            let texel = [
                0.4 * ops::cos(phase),
                0.5 * ops::sin(phase),
                0.2 * ops::cos(phase),
                0.0,
            ];
            shapes.extend(
                texel
                    .into_iter()
                    .flat_map(|v| f16::from_f32(v).to_le_bytes()),
            );
        }
        Waves { shapes, patch }
    }

    #[test]
    fn height_finds_the_water_above() {
        let sea = sea();
        let p = Vec2::new(31.0, -12.0);
        let h = sea.height(p, 1.0);
        let mut rest = p;
        for _ in 0..32 {
            rest = p - sea.offset(rest, 1.0).xz();
        }
        let exact = sea.offset(rest, 1.0);
        assert!(
            (rest + exact.xz() - p).length() < 1e-3 && (h - exact.y).abs() < 0.02,
            "height {h}, exact {}",
            exact.y
        );
    }
}
