use bevy_asset::Handle;
use bevy_ecs::resource::Resource;
use bevy_image::Image;
use bevy_math::{Vec2, Vec3};
use bevy_render::extract_resource::ExtractResource;
use bevy_render::RenderApp;

/// Most ground layers a splat map blends.
pub const GROUND_LAYERS: usize = 8;

/// Splat-mapped terrain colour where rays hit [`Cover::Ground`](crate::Cover::Ground).
///
/// Each texel packs weights of layers 1 to 7 as unorm8 in two words; layer 0
/// takes what they leave. A layer's colour is its texture's coarsest mip times its tint.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct ProbeGround {
    /// `Rg32Uint` packed weights.
    pub weights: Handle<Image>,
    /// Layer albedo array.
    pub layers: Handle<Image>,
    /// Colour calibration per layer.
    pub tints: [Vec3; GROUND_LAYERS],
    /// World corner of texel zero, x and z.
    pub origin: Vec2,
    /// Metres per texel.
    pub texel: f32,
}
