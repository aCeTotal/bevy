use crate::waves::Waves;
use crate::{smoothstep, Ocean, OceanSeabed};
use bevy_asset::Assets;
use bevy_ecs::component::Component;
use bevy_ecs::system::{Query, Res};
use bevy_image::Image;
use bevy_math::{ops, Quat, Vec2, Vec3, Vec3Swizzles, Vec4};
use bevy_time::Time;
use bevy_transform::components::Transform;
use core::f32::consts::TAU;

/// Depth-limited waves, as ocean.wesl.
const SHOAL_DEPTH: f32 = 2.0;
/// Longest integration step, seconds.
const STEP: f32 = 1.0 / 120.0;
/// Natural periods, seconds: heave, pitch, roll.
const PERIODS: Vec3 = Vec3::new(1.4, 1.8, 2.6);
/// Damping ratios; roll rings longest.
const DAMPING: Vec3 = Vec3::new(0.55, 0.45, 0.12);
/// Steepest pitch and roll, radians.
const TILT_LIMIT: f32 = 0.4;
/// Waterplane samples along and across.
const SAMPLES: (usize, usize) = (5, 3);
/// Mooring natural period and damping.
const MOORING: (f32, f32) = (14.0, 0.6);
/// Orbital drag rate per second.
const DRAG: f32 = 1.2;

/// A hull moored on the sea.
#[derive(Component, Clone, Copy, Debug)]
#[require(Transform, Buoyancy)]
pub struct Floating {
    /// Mooring point, x and z.
    pub at: Vec2,
    /// Bow bearing, radians.
    pub heading: f32,
    /// Waterplane half length and beam.
    pub hull: Vec2,
    /// Hull depth under calm water.
    pub draft: f32,
    /// Bow, stern and beam extents.
    pub planform: Vec4,
}

/// How a floating hull moves now.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct Buoyancy {
    /// Heave, pitch and roll.
    pose: Vec3,
    rate: Vec3,
    /// Last water plane.
    water: Option<Vec3>,
    /// Surge and sway off the mooring.
    drift: Vec2,
    drift_rate: Vec2,
    /// Last water displacement.
    flow: Option<Vec2>,
}

impl Buoyancy {
    /// Settles toward the water plane.
    fn settle(&mut self, water: Vec3, flow: Vec2, dt: f32) {
        let water_rate = self.water.map_or(Vec3::ZERO, |w| (water - w) / dt);
        let current = self.flow.map_or(Vec2::ZERO, |w| (flow - w) / dt);
        (self.water, self.flow) = (Some(water), Some(flow));
        let natural = Vec3::splat(TAU) / PERIODS;
        let mooring = TAU / MOORING.0;
        let steps = (dt / STEP).ceil().max(1.0);
        let step = dt / steps;
        for _ in 0..steps as usize {
            self.rate += (natural * natural * (water - self.pose)
                - 2.0 * DAMPING * natural * (self.rate - water_rate))
                * step;
            self.pose += self.rate * step;
            let spring =
                mooring * mooring * self.drift + 2.0 * MOORING.1 * mooring * self.drift_rate;
            self.drift_rate += (DRAG * (current - self.drift_rate) - spring) * step;
            self.drift += self.drift_rate * step;
        }
    }
}

/// Fitted water plane under hull.
fn water_plane(waves: &Waves, center: Vec2, f: &Floating, calm: f32) -> Vec3 {
    let fore = Vec2::from_angle(f.heading);
    let side = fore.perp();
    let spread = |i: usize, n: usize| i as f32 / (n - 1) as f32 * 2.0 - 1.0;
    let (mut sum, mut along, mut across, mut x2, mut z2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, j) in (0..SAMPLES.0).flat_map(|i| (0..SAMPLES.1).map(move |j| (i, j))) {
        let (x, z) = (
            spread(i, SAMPLES.0) * f.hull.x,
            spread(j, SAMPLES.1) * f.hull.y,
        );
        let h = waves.height(center + fore * x + side * z, calm);
        (sum, along, across, x2, z2) = (
            sum + h,
            along + h * x,
            across + h * z,
            x2 + x * x,
            z2 + z * z,
        );
    }
    let count = (SAMPLES.0 * SAMPLES.1) as f32;
    Vec3::new(sum / count, ops::atan(along / x2), -ops::atan(across / z2))
}

/// Hulls ride waves on moorings.
pub(crate) fn ride_waves(
    time: Res<Time>,
    (waves, ocean, seabed, images): (Res<Waves>, Res<Ocean>, Res<OceanSeabed>, Res<Assets<Image>>),
    mut hulls: Query<(&Floating, &mut Buoyancy, &mut Transform)>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    for (f, mut motion, mut transform) in &mut hulls {
        let center = f.at + motion.drift;
        let depth = seabed
            .height(&images, center)
            .map_or(f32::INFINITY, |ground| ocean.level - ground);
        let calm = smoothstep(0.0, SHOAL_DEPTH, depth);
        motion.settle(
            water_plane(&waves, center, f, calm),
            waves.surface(center, calm).xz(),
            dt,
        );
        let (pitch, roll) = (
            motion.pose.y.clamp(-TILT_LIMIT, TILT_LIMIT),
            motion.pose.z.clamp(-TILT_LIMIT, TILT_LIMIT),
        );
        let turn = Quat::from_rotation_y(-f.heading)
            * Quat::from_rotation_z(pitch)
            * Quat::from_rotation_x(roll);
        let flotation = Vec3::new(center.x, ocean.level + motion.pose.x, center.y);
        *transform =
            Transform::from_translation(flotation - turn * Vec3::Y * f.draft).with_rotation(turn);
    }
}
