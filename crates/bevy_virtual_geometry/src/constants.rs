use crate::clock::Clock;
use crate::debug::VirtualGeometryDebug;
use crate::instances::Instances;
use crate::readout::LodBias;
use crate::records::GpuView;
use crate::view::ViewGeometry;
use crate::wind::{WindReach, WindStrength};
use bevy_camera::{Camera3d, MainPassResolutionOverride};
use bevy_core_pipeline::prepass::PreviousViewData;
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{Has, With};
use bevy_ecs::system::{Query, Res};
use bevy_light::CascadeShadowConfig;
use bevy_math::{Mat4, UVec2, Vec3, Vec4, Vec4Swizzles};
use bevy_pbr::{ExtractedDirectionalLight, KeptRestore, LightEntity, StillCasters};
use bevy_render::camera::TemporalJitter;
use bevy_render::renderer::RenderQueue;
use bevy_render::view::ExtractedView;

/// Largest visible cut error, pixels.
const THRESHOLD: f32 = 1.0;
/// Shadow texels hide more error.
const SHADOW_THRESHOLD: f32 = 4.0;
/// Hardware raster above this, pixels.
const SOFT_LIMIT: f32 = 128.0;
/// Shadow filter reach, metres.
const RECEIVER_MARGIN: f32 = 4.0;

fn normalized(p: Vec4) -> Vec4 {
    p / p.xyz().length()
}

/// Left, right, bottom, top planes.
fn sides(m: Mat4) -> [Vec4; 4] {
    let (x, y, w) = (m.row(0), m.row(1), m.row(3));
    [w + x, w - x, w + y, w - y].map(normalized)
}

fn near(m: Mat4) -> Vec4 {
    normalized(m.row(3) - m.row(2))
}

/// Appends culling planes.
fn cull_by(c: &mut GpuView, planes: impl IntoIterator<Item = Vec4>) {
    for p in planes {
        c.frustum[c.planes as usize] = p.to_array();
        c.planes += 1;
    }
}

/// Where shadows must land.
struct Receiver {
    sides: [Vec4; 4],
    eye: Vec3,
    forward: Vec3,
}

impl Receiver {
    fn new(view: &ExtractedView) -> Self {
        let view_from_world = view.world_from_view.affine().inverse();
        let sides = sides(view.clip_from_view * Mat4::from(view_from_world));
        Self {
            sides,
            eye: view.world_from_view.translation(),
            forward: *view.world_from_view.forward(),
        }
    }

    /// Planes cast shadows can reach.
    fn planes(&self, (near, far): (f32, f32), light: Vec3) -> impl Iterator<Item = Vec4> + '_ {
        let ahead = self.forward.dot(self.eye);
        let slice = [
            self.forward.extend(-ahead - near),
            (-self.forward).extend(ahead + far),
        ];
        self.sides
            .into_iter()
            .chain(slice)
            .filter(move |p| p.xyz().dot(light) <= 0.0)
            .map(|p| p + Vec4::W * RECEIVER_MARGIN)
    }
}

/// Camera depths a cascade shades.
fn slice(config: &CascadeShadowConfig, cascade: usize) -> (f32, f32) {
    let near = cascade.checked_sub(1).map_or(config.minimum_distance, |i| {
        config.bounds[i] * (1.0 - config.overlap_proportion)
    });
    (near, config.bounds[cascade])
}

/// Culling and raster constants.
fn constants(
    view: &ExtractedView,
    size: UVec2,
    (jitter, previous): (Option<&TemporalJitter>, Option<&PreviousViewData>),
) -> GpuView {
    let parallel = view.clip_from_view.w_axis.w == 1.0;
    // Bevy's arithmetic, so motion cancels exactly.
    let view_from_world = view.world_from_view.to_matrix().inverse();
    let unjittered = view.clip_from_view * view_from_world;
    let mut clip_from_view = view.clip_from_view;
    if let Some(jitter) = jitter {
        jitter.jitter_projection(&mut clip_from_view, size.as_vec2());
    }
    let clip_from_world = clip_from_view * view_from_world;
    let mut c = GpuView {
        clip_from_world: clip_from_world.to_cols_array_2d(),
        world_from_clip: clip_from_world.inverse().to_cols_array_2d(),
        unjittered_clip_from_world: unjittered.to_cols_array_2d(),
        previous_clip_from_world: previous
            .map_or(unjittered, |p| p.unjittered_clip_from_world)
            .to_cols_array_2d(),
        eye: view.world_from_view.translation().to_array(),
        lod_scale: size.y as f32 * 0.5 * view.clip_from_view.y_axis.y,
        size: size.as_vec2().to_array(),
        soft_limit: SOFT_LIMIT,
        near: view.clip_from_view.w_axis.z,
        orthographic: u32::from(parallel),
        ..bytemuck::Zeroable::zeroed()
    };
    cull_by(
        &mut c,
        sides(clip_from_world)
            .into_iter()
            .chain((!parallel).then(|| near(clip_from_world))),
    );
    c
}

/// Per-view inputs to the constants.
type Views<'w, 's> = Query<
    'w,
    's,
    (
        &'static ViewGeometry,
        &'static ExtractedView,
        Option<&'static TemporalJitter>,
        Option<&'static PreviousViewData>,
        Option<&'static MainPassResolutionOverride>,
        (
            Option<&'static LightEntity>,
            Has<StillCasters>,
            Has<KeptRestore>,
        ),
    ),
>;

type Eyes<'w, 's> = Query<'w, 's, (Entity, &'static ExtractedView), With<Camera3d>>;

type Suns<'w, 's> = Query<'w, 's, &'static ExtractedDirectionalLight>;

/// Instances, clock, bias, view.
type Frame<'w> = (
    Res<'w, Instances>,
    Res<'w, Clock>,
    Res<'w, LodBias>,
    Res<'w, VirtualGeometryDebug>,
);

pub(crate) fn prepare_constants(
    (views, eyes, suns): (Views, Eyes, Suns),
    (instances, clock, bias, visualize): Frame,
    (queue, wind, strength): (Res<RenderQueue>, Res<WindReach>, Res<WindStrength>),
) {
    let Some((camera_entity, eye)) = eyes.iter().next() else {
        return;
    };
    let receiver = Receiver::new(eye);
    for (geometry, view, jitter, previous, scaled, (light, still, movers)) in &views {
        let size = scaled.map_or(view.viewport.zw(), |s| s.0);
        let mut c = constants(view, size, (jitter, previous));
        let camera = light.is_none();
        if let Some(&LightEntity::Directional { light_entity, cascade_index }) = light
            && let Ok(sun) = suns.get(light_entity)
            // Kept cascades cover their whole box, for later views.
            && !sun.cascades.get(&camera_entity).and_then(|c| c.get(cascade_index)).is_some_and(|c| c.cached)
        {
            cull_by(
                &mut c,
                receiver.planes(
                    slice(&sun.cascade_shadow_config, cascade_index),
                    *view.world_from_view.forward(),
                ),
            );
        }
        c.occlusion = u32::from(camera && c.orthographic == 0);
        c.threshold = if camera { THRESHOLD } else { SHADOW_THRESHOLD } * bias.0;
        // Shadows rasterize in hardware only.
        c.soft_limit = if camera { SOFT_LIMIT } else { 0.0 };
        c.visualize = if camera { *visualize as u32 } else { 0 };
        c.wind_eye = receiver.eye.to_array();
        c.wind_reach = wind.0.max(f32::EPSILON);
        c.wind_strength = strength.0;
        c.time = clock.now;
        c.previous_time = clock.before;
        c.instance_extent = if movers { 0 } else { instances.extent };
        c.static_capacity = instances.static_capacity;
        c.capacity = geometry.work.capacity.0;
        c.dynamic_count = if still { 0 } else { instances.dynamic_count };
        queue.write_buffer(&geometry.uniform, 0, bytemuck::bytes_of(&c));
    }
}
