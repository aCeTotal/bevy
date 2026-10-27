#![cfg_attr(docsrs, feature(doc_cfg))]

//! Clustered LOD geometry: GPU-culled, occlusion-tested and rasterized
//! in software or mesh shaders, then shaded from a visibility buffer.

extern crate alloc;

mod arrays;
mod assembly;
mod asset;
mod capacity;
mod clock;
mod constants;
mod debug;
mod draw;
mod instances;
mod layers;
mod library;
mod pipelines;
mod pyramid;
mod readback;
mod readout;
/// GPU layout of clustered LOD hierarchies.
pub mod records;
mod shade;
mod shadow;
mod slab;
mod view;
mod wind;

pub use assembly::{Assembled, Assemblies, Refs};
pub use asset::{Clip, MeshRecords, Shading, VirtualGeometry, VirtualMaterial};
pub use debug::VirtualGeometryDebug;
pub use instances::{affine_rows, Play, Previous, VirtualGeometry3d};
pub use layers::VirtualLayers;
pub use readout::{Drawn, VirtualGeometryCounts};
pub use wind::{Wind, WindReach, WindStrength};

use bevy_app::{App, First, Plugin};
use bevy_asset::{AssetApp, Assets};
use bevy_core_pipeline::core_3d::main_opaque_pass_3d;
use bevy_core_pipeline::prepass::node::late_prepass;
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_pbr::{per_view_shadow_pass, restore_kept_shadows, StillShadowSystems, EARLY_SHADOW_PASS};
use bevy_render::extract_resource::ExtractResourcePlugin;
use bevy_render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};
use bevy_shader::Shader;

/// Ordering points of the virtual geometry passes in [`Core3d`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VirtualGeometrySystems {
    /// Sun shadow raster.
    Shadows,
    /// Culling, raster and resolve into the prepass.
    Raster,
    /// Visibility buffer shading.
    Shade,
}

/// Draws [`VirtualGeometry3d`] entities and [`Assemblies`].
pub struct VirtualGeometryPlugin;

impl Plugin for VirtualGeometryPlugin {
    fn build(&self, app: &mut App) {
        let (assemblies, feed) = assembly::channel();
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(shade::shader());
        let counts = VirtualGeometryCounts::default();
        app.init_asset::<VirtualGeometry>()
            .insert_resource(assemblies)
            .insert_resource(counts.clone())
            .init_resource::<VirtualGeometryDebug>()
            .add_plugins((
                ExtractResourcePlugin::<VirtualLayers>::default(),
                ExtractResourcePlugin::<VirtualGeometryDebug>::default(),
                ExtractResourcePlugin::<WindReach>::default(),
                ExtractResourcePlugin::<WindStrength>::default(),
            ))
            .init_resource::<WindReach>()
            .init_resource::<WindStrength>()
            .add_systems(First, instances::remember);
        app.sub_app_mut(RenderApp)
            .insert_resource(feed)
            .insert_resource(counts)
            .insert_resource(shade::ShaderHandle(shader))
            .init_resource::<shade::ShadeLibrary>()
            .init_resource::<library::Library>()
            .init_resource::<library::Arrivals>()
            .init_resource::<assembly::Placed>()
            .init_resource::<clock::Clock>()
            .init_resource::<view::ShadowWork>()
            .init_resource::<readout::LodBias>()
            .init_resource::<capacity::Capacities>()
            .add_systems(
                RenderStartup,
                (
                    pipelines::setup_pipelines,
                    instances::setup_instances,
                    shade::setup_shading,
                    readout::setup_readout,
                ),
            )
            .add_systems(
                ExtractSchedule,
                (
                    library::extract_arrivals,
                    instances::extract_virtual,
                    clock::extract_clock,
                ),
            )
            .add_systems(
                Render,
                (
                    (
                        library::prepare_library,
                        assembly::prepare_assemblies,
                        library::upload_library,
                    )
                        .chain()
                        .in_set(RenderSystems::PrepareAssets),
                    (
                        instances::prepare_dynamic,
                        view::prepare_views,
                        layers::prepare_layers,
                    )
                        .in_set(RenderSystems::PrepareResources),
                    constants::prepare_constants
                        .after(instances::prepare_dynamic)
                        .after(view::prepare_views)
                        .in_set(RenderSystems::PrepareResources),
                    shade::queue_shading.in_set(RenderSystems::Queue),
                    readout::read_readout.in_set(RenderSystems::Prepare),
                    readout::map_readout.in_set(RenderSystems::Cleanup),
                    (view::prepare_groups, shade::prepare_shade_library)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                (
                    shadow::draw_still_shadows
                        .in_set(VirtualGeometrySystems::Shadows)
                        .in_set(StillShadowSystems),
                    shadow::draw_shadows
                        .in_set(VirtualGeometrySystems::Shadows)
                        .after(restore_kept_shadows)
                        .before(per_view_shadow_pass::<EARLY_SHADOW_PASS>),
                    draw::draw_geometry
                        .in_set(VirtualGeometrySystems::Raster)
                        .after(late_prepass)
                        .in_set(Core3dSystems::Prepass),
                    shade::shade_geometry
                        .in_set(VirtualGeometrySystems::Shade)
                        .before(main_opaque_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                ),
            );
    }
}
