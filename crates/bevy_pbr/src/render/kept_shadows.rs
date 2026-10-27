use bevy_ecs::{component::Component, entity::Entity, schedule::SystemSet, system::Query};
use bevy_light::cascade::{Cascade, Rebuild, REBUILD_TILES};
use bevy_math::{Mat4, UVec4, Vec2, Vec3};
use bevy_render::{
    render_resource::{
        Extent3d, Origin3d, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
        TextureViewDescriptor,
    },
    renderer::{RenderContext, RenderDevice},
    texture::CachedTexture,
};

/// Directional shadow maps kept across frames.
#[derive(Default)]
pub struct KeptDirectionalShadows {
    texture: Option<CachedTexture>,
    /// Shadowed lights' cascade counts.
    layout: Vec<(Entity, usize)>,
}

impl KeptDirectionalShadows {
    /// The array; true when all layers redraw.
    pub fn texture(
        &mut self,
        device: &RenderDevice,
        descriptor: &TextureDescriptor,
        layout: Vec<(Entity, usize)>,
    ) -> (CachedTexture, bool) {
        let kept = self
            .texture
            .as_ref()
            .filter(|t| t.texture.size() == descriptor.size)
            .cloned();
        let redraw = kept.is_none() || self.layout != layout;
        let texture = kept.unwrap_or_else(|| {
            let texture = device.create_texture(descriptor);
            CachedTexture {
                default_view: texture.create_view(&TextureViewDescriptor::default()),
                texture,
            }
        });
        self.texture = Some(texture.clone());
        self.layout = layout;
        (texture, redraw)
    }
}

/// Shadow view drawing only still casters.
#[derive(Component)]
pub struct StillCasters;

/// Kept layer refilled from its still layer.
#[derive(Component)]
pub struct KeptRestore {
    pub texture: Texture,
    /// Still and drawn layers.
    pub layers: (u32, u32),
}

/// Still casters draw before kept layers restore.
#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub struct StillShadowSystems;

fn layer(texture: &Texture, z: u32) -> TexelCopyTextureInfo<'_> {
    TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: Origin3d { x: 0, y: 0, z },
        aspect: TextureAspect::All,
    }
}

/// Copies still casters under the moving ones.
pub fn restore_kept_shadows(restores: Query<&KeptRestore>, mut ctx: RenderContext) {
    for restore in &restores {
        let size = Extent3d {
            depth_or_array_layers: 1,
            ..restore.texture.size()
        };
        ctx.command_encoder().copy_texture_to_texture(
            layer(&restore.texture, restore.layers.0),
            layer(&restore.texture, restore.layers.1),
            size,
        );
    }
}

/// Still layer's box and tile this frame, if drawn.
pub(crate) fn still_draw(cascade: &Cascade, redraw: bool) -> Option<(Rebuild, Option<u32>)> {
    if let Some(r) = &cascade.rebuild {
        return Some((r.clone(), (!redraw).then_some(r.tile)));
    }
    let current = Rebuild {
        world_from_cascade: cascade.world_from_cascade,
        clip_from_cascade: cascade.clip_from_cascade,
        clip_from_world: cascade.clip_from_world,
        tile: 0,
    };
    (cascade.age == 0 || redraw).then_some((current, None))
}

/// Clip zoom and viewport drawing one tile.
pub(crate) fn tile_projection(tile: Option<u32>, size: u32) -> (Mat4, UVec4) {
    let Some(tile) = tile else {
        return (Mat4::IDENTITY, UVec4::new(0, 0, size, size));
    };
    let (x, y) = (tile % REBUILD_TILES, tile / REBUILD_TILES);
    let tiles = REBUILD_TILES as f32;
    // Clip y grows up, tile rows down.
    let center = Vec2::new(2.0 * x as f32 + 1.0 - tiles, tiles - 2.0 * y as f32 - 1.0) / tiles;
    let zoom = Mat4::from_translation((-tiles * center).extend(0.0))
        * Mat4::from_scale(Vec3::new(tiles, tiles, 1.0));
    let side = size / REBUILD_TILES;
    (zoom, UVec4::new(x * side, y * side, side, side))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec4Swizzles;

    /// Framebuffer pixel of clip point `p`.
    fn pixel((zoom, viewport): (Mat4, UVec4), p: Vec2) -> Vec2 {
        let clip = (zoom * p.extend(0.0).extend(1.0)).xy();
        let (origin, size) = (viewport.xy().as_vec2(), viewport.zw().as_vec2());
        origin + Vec2::new(clip.x + 1.0, 1.0 - clip.y) * 0.5 * size
    }

    #[test]
    fn tiles_land_where_the_whole_box_does() {
        let size = 2048;
        let side = (size / REBUILD_TILES) as f32;
        for p in [Vec2::new(-0.7, 0.6), Vec2::new(0.3, -0.2), Vec2::new(0.9, 0.95)] {
            let whole = pixel(tile_projection(None, size), p);
            let cell = (whole / side).floor();
            let tile = cell.y as u32 * REBUILD_TILES + cell.x as u32;
            let tiled = pixel(tile_projection(Some(tile), size), p);
            assert!((tiled - whole).length() < 1e-3, "{tiled} vs {whole}");
        }
    }
}
