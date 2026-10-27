use bevy_ecs::resource::Resource;
use bevy_render::extract_resource::ExtractResource;
use bevy_render::RenderApp;

/// Bend in wind.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Wind {
    /// Sway at full height, metres.
    pub amplitude: f32,
    /// Height where bend is full.
    pub height: f32,
    /// Gust rate.
    pub speed: f32,
}

/// Distance from the camera within which plants sway; beyond it they stand
/// still, so far shadows stay valid between redraws.
#[derive(Resource, Clone, Copy, Debug, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WindReach(pub f32);

impl Default for WindReach {
    fn default() -> Self {
        WindReach(200.0)
    }
}

/// Sway scale of the weather, one in a normal breeze.
#[derive(Resource, Clone, Copy, Debug, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WindStrength(pub f32);

impl Default for WindStrength {
    fn default() -> Self {
        WindStrength(1.0)
    }
}
