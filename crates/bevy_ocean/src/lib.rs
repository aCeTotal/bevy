#![cfg_attr(docsrs, feature(doc_cfg))]

//! FFT ocean waves, refracting and mirroring water, and buoyancy.
//!
//! Cameras seeing water need `DepthPrepass` and `ScreenSpaceTransmission`.

mod buoyancy;
mod fft;
mod material;
mod motion;
mod river;
mod seabed;
mod spectrum;
mod surface;
mod waves;

pub use buoyancy::{Buoyancy, Floating};
pub use material::{OceanMaterial, OceanSurface};
pub use river::{RiverMaterial, RiverSurface};
pub use seabed::{OceanSeabed, SeabedChart, SEABED_SIDE};
pub use waves::Waves;

use bevy_app::{App, Plugin, PostUpdate, Startup};
use bevy_asset::embedded_asset;
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_math::Vec2;
use bevy_pbr::main_transmissive_pass_3d;
use bevy_pbr::MaterialPlugin;
use bevy_render::extract_resource::ExtractResourcePlugin;
use bevy_render::renderer::{RenderGraph, RenderGraphSystems};
use bevy_render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy_shader::load_shader_library;
use bevy_transform::TransformSystems;
use fft::WaveStep;
use material::OceanFrame;

/// Sea placement; insert before startup.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Ocean {
    /// Calm sea level.
    pub level: f32,
    /// Patch centre, x and z.
    pub focus: Vec2,
    /// Patch cell size, metres.
    pub spacing: f32,
    /// Waves move and show.
    pub waves: bool,
}

impl Ocean {
    /// Half-width of the wave patch drawn.
    pub fn reach(&self) -> f32 {
        surface::CELLS as f32 * 0.5 * self.spacing
    }
}

/// Per-frame ocean work.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OceanSystems;

/// Draws the sea, floats hulls.
pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "optics.wesl");
        load_shader_library!(app, "light.wesl");
        load_shader_library!(app, "flow.wesl");
        embedded_asset!(app, "ocean.wesl");
        embedded_asset!(app, "river.wesl");
        embedded_asset!(app, "motion.wesl");
        embedded_asset!(app, "fft.wesl");
        app.add_plugins((
            MaterialPlugin::<OceanMaterial>::default(),
            MaterialPlugin::<RiverMaterial>::default(),
            ExtractResourcePlugin::<OceanFrame>::default(),
            ExtractResourcePlugin::<WaveStep>::default(),
        ))
        .init_resource::<Waves>()
        .init_resource::<OceanSeabed>()
        .add_systems(Startup, surface::setup_ocean)
        .add_systems(
            PostUpdate,
            (
                surface::follow_focus,
                surface::stir_sea,
                surface::follow_hulls,
                buoyancy::ride_waves,
                surface::chart_sea,
            )
                .chain()
                .in_set(OceanSystems)
                .before(TransformSystems::Propagate),
        );
        app.sub_app_mut(RenderApp)
            .add_systems(RenderStartup, (motion::setup_motion, fft::setup_fft))
            .add_systems(
                Render,
                (
                    material::upload_frame.in_set(RenderSystems::PrepareResources),
                    fft::prepare_fft.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(RenderGraph, fft::run_fft.in_set(RenderGraphSystems::Begin))
            .add_systems(
                Core3d,
                motion::water_motion
                    .after(main_transmissive_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
