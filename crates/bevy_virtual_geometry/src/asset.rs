use crate::records::{GpuCluster, GpuNode, GpuPart, GpuVertex};
use alloc::sync::Arc;
use bevy_asset::{Asset, Handle};
use bevy_image::Image;
use bevy_reflect::TypePath;

/// How a material shades.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Shading {
    /// Metallic-roughness surface.
    #[default]
    Standard,
    /// Thin leaves passing light.
    Foliage,
    /// Per-face layers of [`VirtualLayers`](crate::VirtualLayers).
    Layered,
}

/// Surface of clusters.
#[derive(Clone, Debug)]
pub struct VirtualMaterial {
    /// Linear colour factor.
    pub base_color: [f32; 4],
    /// Perceptual roughness factor.
    pub roughness: f32,
    /// Metallic factor.
    pub metallic: f32,
    /// Light share passing through.
    pub translucency: f32,
    /// Shading model.
    pub shading: Shading,
    /// Lit from both sides.
    pub two_sided: bool,
    /// Base colour texture.
    pub albedo: Option<Handle<Image>>,
    /// Tangent-space normal texture.
    pub normal: Option<Handle<Image>>,
    /// Occlusion, roughness, metallic texture.
    pub orm: Option<Handle<Image>>,
}

/// One mesh in GPU layout, indices local.
#[derive(Clone, Default)]
pub struct MeshRecords {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Attributes in lockstep with positions.
    pub vertices: Vec<GpuVertex>,
    /// Packed cluster-local triangles.
    pub triangles: Vec<u32>,
    /// Leaf ranges sorted finest first.
    pub clusters: Vec<GpuCluster>,
    /// Hierarchy, root first.
    pub nodes: Vec<GpuNode>,
    /// Node children, packed.
    pub children: Vec<[u32; 2]>,
    /// Placed meshes.
    pub parts: Vec<GpuPart>,
}

/// Skinning matrices of an animation.
#[derive(Clone, Default)]
pub struct Clip {
    /// Rows per bone per frame.
    pub poses: Vec<[[f32; 4]; 3]>,
    /// Bones per frame.
    pub bones: u32,
    /// Frames in the loop.
    pub frames: u32,
    /// Frames per second.
    pub fps: f32,
    /// Farthest a vertex strays, mesh units.
    pub reach: f32,
}

/// Clustered LOD hierarchies with their surfaces.
#[derive(Asset, TypePath, Clone)]
pub struct VirtualGeometry {
    /// Parts place earlier meshes.
    pub meshes: Arc<[MeshRecords]>,
    /// Mesh drawn per variant.
    pub roots: Vec<u32>,
    /// Materials clusters index.
    pub materials: Vec<VirtualMaterial>,
    /// Animations skinned vertices play.
    pub clips: Arc<[Clip]>,
}
