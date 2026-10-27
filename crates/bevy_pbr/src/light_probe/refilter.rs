use super::generate::{GeneratorPipelines, RenderEnvironmentMap};
use bevy_asset::AssetId;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_platform::collections::HashSet;
use bevy_render::render_resource::{PipelineCache, TextureId};

/// Environment map sources rewritten on the GPU.
///
/// Generated environment maps refilter only when their source texture changes
/// or a GPU writer marks it here; marks made in a frame count from the next,
/// once the writes have run.
#[derive(Resource, Default)]
pub struct RewrittenEnvironmentMaps {
    pending: HashSet<AssetId<Image>>,
    ready: HashSet<AssetId<Image>>,
}

impl RewrittenEnvironmentMaps {
    /// Marks `source` as rewritten this frame.
    pub fn mark(&mut self, source: AssetId<Image>) {
        self.pending.insert(source);
    }
}

/// Whether an environment map filters this frame.
#[derive(Component, Default)]
pub struct EnvironmentMapFilter {
    /// Source texture last filtered.
    filtered: Option<TextureId>,
    /// Filters this frame.
    pub due: bool,
}

/// Pipelines a filter pass needs.
fn compiled(cache: &PipelineCache, p: &GeneratorPipelines) -> bool {
    [
        p.downsample_first,
        p.downsample_second,
        p.copy,
        p.radiance,
        p.irradiance,
    ]
    .into_iter()
    .all(|id| cache.get_compute_pipeline(id).is_some())
}

/// Filters only changed or rewritten sources.
pub fn schedule_filtering(
    mut commands: Commands,
    mut maps: Query<(
        Entity,
        &RenderEnvironmentMap,
        Option<&mut EnvironmentMapFilter>,
    )>,
    rewritten: Res<RewrittenEnvironmentMaps>,
    (cache, pipelines): (Res<PipelineCache>, Option<Res<GeneratorPipelines>>),
) {
    let ready = pipelines.is_some_and(|p| compiled(&cache, &p));
    for (entity, map, filter) in &mut maps {
        let Some(mut filter) = filter else {
            commands.entity(entity).insert(EnvironmentMapFilter {
                filtered: None,
                due: true,
            });
            continue;
        };
        let texture = map.environment_map.texture.id();
        filter.due = filter.filtered != Some(texture) || rewritten.ready.contains(&map.source);
        if ready && filter.due {
            filter.filtered = Some(texture);
        }
    }
}

/// This frame's marks become readable next frame.
pub fn rotate_rewritten(mut rewritten: ResMut<RewrittenEnvironmentMaps>) {
    let rewritten = &mut *rewritten;
    core::mem::swap(&mut rewritten.ready, &mut rewritten.pending);
    rewritten.pending.clear();
}
