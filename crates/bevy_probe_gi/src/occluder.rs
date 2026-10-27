use bevy_ecs::component::Component;
use bevy_math::Vec3;

/// How probe rays treat a surface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cover {
    /// Opaque.
    Solid,
    /// Leaf cards some rays pass.
    Canopy,
    /// Terrain, coloured by [`ProbeGround`](crate::ProbeGround).
    Ground,
}

/// A mesh probe rays hit.
#[derive(Component, Clone, Copy, Debug)]
pub struct ProbeOccluder {
    /// Diffuse albedo, linear.
    pub albedo: Vec3,
    /// How rays treat it.
    pub cover: Cover,
}
