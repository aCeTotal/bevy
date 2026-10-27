use super::{
    resources::{AtmosphereLutPipelines, AtmosphereTextures},
    ExtractedAtmosphere, GpuAtmosphereSettings,
};
use crate::{
    light_probe::refilter::RewrittenEnvironmentMaps, ExtractedDirectionalLight, GpuScatteringMedium,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::{ops, Vec3};
use bevy_platform::hash::FixedHasher;
use bevy_render::{
    render_asset::RenderAssets, render_resource::PipelineCache, view::ExtractedView,
};
use core::hash::{BuildHasher, Hash, Hasher};

use super::environment::{AtmosphereEnvironmentMap, AtmosphereProbePipeline};

/// Eye height share that refreshes the sky.
const ALTITUDE_STEP: f32 = 0.02;
/// Height where steps stop shrinking, metres.
const ALTITUDE_FLOOR: f32 = 10.0;
/// Eye travel that refreshes the sky, radians.
const HORIZON_STEP: f32 = 3.0e-4;

/// What a view's atmosphere recomputes this frame.
#[derive(Component, Default)]
pub struct AtmosphereRefresh {
    /// Transmittance and multiscattering.
    pub planet: bool,
    /// Sky view and environment: eye place, lights.
    pub sky: bool,
    /// Aerial view: the whole view.
    pub view: bool,
    /// Planet, sky and view keys computed.
    computed: Option<(u64, u64, u64)>,
}

fn hash_floats(state: &mut impl Hasher, values: &[f32]) {
    values.iter().for_each(|v| v.to_bits().hash(state));
}

/// Inputs of the transmittance and multiscattering LUTs.
fn planet_key(
    (atmosphere, settings): (&ExtractedAtmosphere, &GpuAtmosphereSettings),
    (medium, textures): (&GpuScatteringMedium, &AtmosphereTextures),
) -> u64 {
    let mut state = FixedHasher.build_hasher();
    hash_floats(
        &mut state,
        &[atmosphere.inner_radius, atmosphere.outer_radius],
    );
    hash_floats(&mut state, &atmosphere.ground_albedo.to_array());
    medium.density_lut_view.id().hash(&mut state);
    medium.scattering_lut_view.id().hash(&mut state);
    textures.transmittance_lut.texture.id().hash(&mut state);
    textures.multiscattering_lut.texture.id().hash(&mut state);
    (
        settings.transmittance_lut_size,
        settings.multiscattering_lut_size,
    )
        .hash(&mut state);
    let samples = (
        settings.transmittance_lut_samples,
        settings.multiscattering_lut_dirs,
    );
    (samples, settings.multiscattering_lut_samples).hash(&mut state);
    state.finish()
}

/// Eye height and place over the planet, in steps.
fn eye_cell(view: &ExtractedView, atmosphere: &ExtractedAtmosphere) -> [i32; 4] {
    let eye = atmosphere
        .world_to_atmosphere
        .transform_point3(view.world_from_view.translation());
    let height = (eye.length() - atmosphere.inner_radius).max(0.0) + ALTITUDE_FLOOR;
    let up = (eye.normalize_or(Vec3::Y) / HORIZON_STEP).round();
    let level = ops::floor(ops::ln(height) / ops::ln_1p(ALTITUDE_STEP));
    [level as i32, up.x as i32, up.y as i32, up.z as i32]
}

/// Inputs of the sky view and environment.
fn sky_key(
    (planet, settings, view): (u64, &GpuAtmosphereSettings, &ExtractedView),
    (atmosphere, textures): (&ExtractedAtmosphere, &AtmosphereTextures),
    lights: &Query<&ExtractedDirectionalLight>,
) -> u64 {
    let mut state = FixedHasher.build_hasher();
    planet.hash(&mut state);
    eye_cell(view, atmosphere).hash(&mut state);
    hash_floats(&mut state, &atmosphere.world_to_atmosphere.to_cols_array());
    textures.sky_view_lut.texture.id().hash(&mut state);
    (settings.sky_view_lut_size, settings.sky_view_lut_samples).hash(&mut state);
    for light in lights {
        hash_floats(&mut state, &light.transform.back().to_array());
        hash_floats(
            &mut state,
            &[
                light.color.red,
                light.color.green,
                light.color.blue,
                light.color.alpha,
            ],
        );
        hash_floats(
            &mut state,
            &[
                light.illuminance,
                light.sun_disk_angular_size,
                light.sun_disk_intensity,
            ],
        );
    }
    state.finish()
}

/// Inputs of the aerial view beyond the sky.
fn view_key(
    (sky, settings, view): (u64, &GpuAtmosphereSettings, &ExtractedView),
    textures: &AtmosphereTextures,
) -> u64 {
    let mut state = FixedHasher.build_hasher();
    sky.hash(&mut state);
    hash_floats(
        &mut state,
        &view.world_from_view.to_matrix().to_cols_array(),
    );
    hash_floats(&mut state, &view.clip_from_view.to_cols_array());
    textures.aerial_view_lut.texture.id().hash(&mut state);
    let marching = (
        settings.aerial_view_lut_size,
        settings.aerial_view_lut_samples,
        settings.sky_max_samples,
        settings.rendering_method,
    );
    marching.hash(&mut state);
    hash_floats(&mut state, &[settings.aerial_view_lut_max_distance]);
    state.finish()
}

type Views<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        (
            &'static ExtractedAtmosphere,
            &'static GpuAtmosphereSettings,
            &'static ExtractedView,
        ),
        &'static AtmosphereTextures,
        Option<&'static mut AtmosphereRefresh>,
        Option<&'static AtmosphereEnvironmentMap>,
    ),
>;

/// Pipelines every LUT pass needs.
fn compiled(
    cache: &PipelineCache,
    luts: &AtmosphereLutPipelines,
    probe: &AtmosphereProbePipeline,
) -> bool {
    [
        luts.transmittance_lut,
        luts.multiscattering_lut,
        luts.sky_view_lut,
        luts.aerial_view_lut,
        probe.environment,
    ]
    .into_iter()
    .all(|id| cache.get_compute_pipeline(id).is_some())
}

/// Marks what changed since computed.
pub(super) fn refresh_atmosphere(
    mut commands: Commands,
    mut views: Views,
    lights: Query<&ExtractedDirectionalLight>,
    (media, cache): (Res<RenderAssets<GpuScatteringMedium>>, Res<PipelineCache>),
    (luts, probe): (Res<AtmosphereLutPipelines>, Res<AtmosphereProbePipeline>),
    mut rewritten: ResMut<RewrittenEnvironmentMaps>,
) {
    let ready = compiled(&cache, &luts, &probe);
    for (entity, (atmosphere, settings, view), textures, refresh, environment) in &mut views {
        let Some(medium) = media.get(atmosphere.medium) else {
            continue;
        };
        let planet = planet_key((atmosphere, settings), (medium, textures));
        let sky = sky_key((planet, settings, view), (atmosphere, textures), &lights);
        let keys = (planet, sky, view_key((sky, settings, view), textures));
        let Some(mut refresh) = refresh else {
            commands.entity(entity).insert(AtmosphereRefresh {
                planet: true,
                sky: true,
                view: true,
                computed: None,
            });
            continue;
        };
        refresh.planet = refresh.computed.is_none_or(|c| c.0 != keys.0);
        refresh.sky = refresh.computed.is_none_or(|c| c.1 != keys.1);
        refresh.view = refresh.computed != Some(keys);
        if !ready {
            continue;
        }
        refresh.computed = Some(keys);
        if let Some(environment) = environment.filter(|_| refresh.sky) {
            rewritten.mark(environment.environment_map.id());
        }
    }
}
