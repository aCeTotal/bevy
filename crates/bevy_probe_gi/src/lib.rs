#![cfg_attr(docsrs, feature(doc_cfg))]

//! Diffuse bounce light from ray-traced irradiance probe clipmaps.

mod clipmap;
mod ground;
mod occluder;
mod scene;
mod tracer;

pub use clipmap::ProbeGi;
pub use ground::{ProbeGround, GROUND_LAYERS};
pub use occluder::{Cover, ProbeOccluder};

use bevy_app::{App, Plugin, Startup, Update};
use bevy_asset::Assets;
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_math::Vec3;
use bevy_render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy_render::mesh::allocator::MeshAllocatorSettings;
use bevy_render::render_resource::BufferUsages;
use bevy_render::renderer::{RenderGraph, RenderGraphSystems};
use bevy_render::settings::WgpuFeatures;
use bevy_render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};
use bevy_shader::Shader;
use bevy_utils::default;

/// Share of the shadowing light that reaches the ground, for probe rays.
#[derive(Resource, Clone, Copy, Debug, ExtractResource)]
#[extract_app(RenderApp)]
pub struct ProbeSunlight(pub Vec3);

/// Ordering point of the probe tracing pass in [`Core3d`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProbeGiTrace;

/// Traces probes around [`ProbeGi`] cameras against [`ProbeOccluder`] meshes.
pub struct ProbeGiPlugin;

impl ProbeGiPlugin {
    /// Device features tracing needs.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::EXPERIMENTAL_RAY_QUERY
            | WgpuFeatures::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    }
}

impl Plugin for ProbeGiPlugin {
    fn build(&self, app: &mut App) {
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(tracer::shader());
        app.add_plugins((
            ExtractResourcePlugin::<ProbeGround>::default(),
            ExtractResourcePlugin::<ProbeSunlight>::default(),
        ))
        .add_systems(Startup, clipmap::spawn_volumes)
        .add_systems(Update, clipmap::follow_focus);
        let slabs = MeshAllocatorSettings {
            extra_buffer_usages: BufferUsages::BLAS_INPUT,
            ..default()
        };
        app.sub_app_mut(RenderApp)
            .insert_resource(slabs)
            .insert_resource(tracer::ProbeShader(shader))
            .init_resource::<scene::Scene>()
            .init_resource::<clipmap::Clipmap>()
            .init_resource::<tracer::Light>()
            .add_systems(RenderStartup, tracer::setup)
            .add_systems(
                ExtractSchedule,
                (
                    scene::extract_occluders,
                    clipmap::extract_clipmap,
                    tracer::extract_light,
                ),
            )
            .add_systems(
                Render,
                (
                    scene::build.in_set(RenderSystems::PrepareResources),
                    tracer::prepare.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(RenderGraph, scene::record.in_set(RenderGraphSystems::Begin))
            .add_systems(
                Core3d,
                tracer::trace
                    .in_set(ProbeGiTrace)
                    .before(Core3dSystems::Prepass),
            );
    }
}
