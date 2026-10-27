use crate::pipelines::PYRAMID;
use crate::view::entries;
use bevy_math::UVec2;
use bevy_render::render_resource::*;
use bevy_render::renderer::RenderDevice;
use bevy_utils::default;

/// Farthest-depth mip chain for occlusion.
pub(crate) struct Pyramid {
    pub whole: TextureView,
    /// Each level with its source extent.
    levels: Vec<(TextureView, UVec2)>,
}

impl Pyramid {
    /// Power-of-two chain; levels fill rounded-up halves.
    pub fn new(device: &RenderDevice, size: UVec2) -> Self {
        let first = (size + 1) / 2;
        let extent = UVec2::new(first.x.next_power_of_two(), first.y.next_power_of_two());
        let count = extent.max_element().trailing_zeros() + 1;
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("virtual_geometry_pyramid"),
            size: Extent3d {
                width: extent.x,
                height: extent.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: PYRAMID,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let level = |k| {
            texture.create_view(&TextureViewDescriptor {
                base_mip_level: k,
                mip_level_count: Some(1),
                ..default()
            })
        };
        let sources = core::iter::successors(Some(size), |s| Some((*s + 1) / 2));
        Self {
            whole: texture.create_view(&TextureViewDescriptor::default()),
            levels: (0..count)
                .zip(sources)
                .map(|(k, source)| (level(k), source))
                .collect(),
        }
    }

    /// Each level reads the one above.
    pub fn groups(
        &self,
        device: &RenderDevice,
        (layout, visibility, unused): (&BindGroupLayout, &Buffer, &TextureView),
    ) -> Vec<(BindGroup, UVec2)> {
        let finer = core::iter::once(unused).chain(self.levels.iter().map(|(view, _)| view));
        let level = |(finer, (coarser, source)): (&TextureView, &(TextureView, UVec2))| {
            let pairs = [
                (0, visibility.as_entire_binding()),
                (1, BindingResource::TextureView(finer)),
                (2, BindingResource::TextureView(coarser)),
            ];
            (
                device.create_bind_group("virtual_geometry_pyramid", layout, &entries(&pairs)),
                *source,
            )
        };
        finer.zip(&self.levels).map(level).collect()
    }
}
