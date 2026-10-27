use bevy_ecs::resource::Resource;
use bevy_render::extract_resource::ExtractResource;
use bevy_render::RenderApp;

/// Debug view replacing shading.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, Debug, ExtractResource)]
#[extract_app(RenderApp)]
pub enum VirtualGeometryDebug {
    /// Lit as usual.
    #[default]
    Off,
    /// Colour per cluster.
    Clusters,
    /// Colour per triangle.
    Triangles,
    /// Colour per instance.
    Instances,
    /// Software or hardware rasterized.
    Raster,
    /// Simplification depth per cluster.
    Detail,
}

/// Cycle order, back to shading.
const CYCLE: [VirtualGeometryDebug; 6] = [
    VirtualGeometryDebug::Off,
    VirtualGeometryDebug::Clusters,
    VirtualGeometryDebug::Triangles,
    VirtualGeometryDebug::Instances,
    VirtualGeometryDebug::Raster,
    VirtualGeometryDebug::Detail,
];

impl VirtualGeometryDebug {
    /// The view after this one.
    pub fn next(self) -> VirtualGeometryDebug {
        let at = CYCLE.iter().position(|&v| v == self).unwrap_or(0);
        CYCLE[(at + 1) % CYCLE.len()]
    }

    /// What the colours mean.
    pub fn label(self) -> Option<&'static str> {
        match self {
            VirtualGeometryDebug::Off => None,
            VirtualGeometryDebug::Clusters => Some("clusters"),
            VirtualGeometryDebug::Triangles => Some("triangles"),
            VirtualGeometryDebug::Instances => Some("instances"),
            VirtualGeometryDebug::Raster => Some("raster: software green, hardware orange"),
            VirtualGeometryDebug::Detail => Some("detail: fine blue, coarse red"),
        }
    }
}
