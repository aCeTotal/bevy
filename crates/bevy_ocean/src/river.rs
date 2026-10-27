use bevy_asset::Asset;
use bevy_pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy_reflect::TypePath;
use bevy_render::render_resource::AsBindGroup;
use bevy_shader::ShaderRef;

/// River water: UV0 current, UV1 width and fall.
pub type RiverMaterial = ExtendedMaterial<StandardMaterial, RiverSurface>;

const SHADER: &str = "embedded://bevy_ocean/river.wesl";

/// Flow-mapped ripples and rapids.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct RiverSurface {}

impl MaterialExtension for RiverSurface {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn reads_view_transmission_texture() -> bool {
        true
    }
}
