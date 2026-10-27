use bevy_asset::Handle;
use bevy_ecs::change_detection::DetectChanges;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res};
use bevy_image::Image;
use bevy_render::extract_resource::ExtractResource;
use bevy_render::render_resource::{Buffer, BufferInitDescriptor, BufferUsages};
use bevy_render::renderer::RenderDevice;
use bevy_render::RenderApp;

/// Texture-array layers that layered materials pick per face.
///
/// A face's first corner names its layer in `uv.x`; corners with zero
/// alpha take the instance tint.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct VirtualLayers {
    /// Albedo array.
    pub albedo: Handle<Image>,
    /// Normal xy, roughness, occlusion array.
    pub surface: Handle<Image>,
    /// Repeats per metre and metalness, per layer.
    pub layers: Vec<[f32; 2]>,
    /// Fresnel reflectance of every layer.
    pub reflectance: f32,
}

/// Per-layer constants on the GPU.
#[derive(Resource)]
pub(crate) struct LayerTable {
    pub buffer: Buffer,
    /// Bumped on rewrite.
    pub generation: u32,
}

/// Rewrites the table when layers change.
pub(crate) fn prepare_layers(
    mut commands: Commands,
    layers: Option<Res<VirtualLayers>>,
    table: Option<Res<LayerTable>>,
    device: Res<RenderDevice>,
) {
    let fresh = layers.as_ref().is_some_and(DetectChanges::is_changed);
    if table.is_some() && !fresh {
        return;
    }
    let reflectance = layers.as_ref().map_or(0.5, |l| l.reflectance);
    let mut rows: Vec<[f32; 4]> = layers
        .iter()
        .flat_map(|l| &l.layers)
        .map(|&[scale, metallic]| [scale, metallic, reflectance, 0.0])
        .collect();
    if rows.is_empty() {
        rows.push([1.0, 0.0, reflectance, 0.0]);
    }
    let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("virtual_geometry_layers"),
        contents: bytemuck::cast_slice(&rows),
        usage: BufferUsages::STORAGE,
    });
    let generation = table.map_or(0, |t| t.generation + 1);
    commands.insert_resource(LayerTable { buffer, generation });
}
