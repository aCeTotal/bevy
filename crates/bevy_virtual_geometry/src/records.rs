use bytemuck::{Pod, Zeroable};

/// Child is a node.
pub const CHILD_NODE: u32 = 0;
/// Child is a cluster range.
pub const CHILD_CLUSTERS: u32 = 1;
/// Child is a placed part.
pub const CHILD_PART: u32 = 2;
/// Child index bits.
pub const CHILD_INDEX_BITS: u32 = 30;
/// Texture index meaning none.
pub const NO_TEXTURE: u32 = u32::MAX;
/// Material flag for two-sided surfaces.
pub const TWO_SIDED: u32 = 1;
/// Most triangles in one cluster.
pub const CLUSTER_TRIANGLES: usize = 128;
/// Most vertices in one cluster.
pub const CLUSTER_VERTICES: usize = 128;
/// Traversable levels, as cull.wgsl.
pub(crate) const MAX_LEVELS: u32 = 32;
/// Fixed counters, as cull.wgsl.
pub(crate) const FIXED_COUNTERS: usize = 21;
/// Poses binding, past shade textures.
pub(crate) const POSES_BINDING: u32 = 13;
/// Bytes of one indirect dispatch.
pub(crate) const SLOT_BYTES: u64 = 12;
/// Culling planes, as common.wgsl.
pub(crate) const PLANES: usize = 10;

/// One cluster; indices local to its mesh.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCluster {
    /// Box corner.
    pub min: [f32; 3],
    /// First vertex.
    pub first_vertex: u32,
    /// Box corner.
    pub max: [f32; 3],
    /// First packed triangle.
    pub first_triangle: u32,
    /// Normal cone apex.
    pub apex: [f32; 3],
    /// Vertices, triangles, material.
    pub counts: u32,
    /// Normal cone axis.
    pub axis: [f32; 3],
    /// Normal cone cutoff.
    pub cutoff: f32,
    /// Own error sphere.
    pub lod: [f32; 4],
    /// Parent error sphere.
    pub parent: [f32; 4],
    /// Own and parent error.
    pub errors: [f32; 4],
}

/// One hierarchy node; indices local to its mesh.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuNode {
    /// Box corner.
    pub min: [f32; 3],
    /// First child entry.
    pub first_child: u32,
    /// Box corner.
    pub max: [f32; 3],
    /// Child entries.
    pub child_count: u32,
    /// Error sphere.
    pub lod: [f32; 4],
    /// Largest parent error below.
    pub error: [f32; 4],
}

/// A placed copy of another mesh.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPart {
    /// Rows of the placement.
    pub rows: [[f32; 4]; 3],
    /// Proxy error sphere.
    pub parent: [f32; 4],
    /// Proxy error.
    pub error: f32,
    /// Placed mesh, local link index.
    pub root: u32,
    /// Own wind, as [`pack_wind`]; zero keeps the surrounding wind.
    pub wind: [u32; 2],
}

/// Packed vertex attributes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Octahedral normal.
    pub normal: u32,
    /// RGBA8 colour.
    pub color: u32,
    /// Octahedral tangent, sign on top.
    pub tangent: u32,
    /// Packed joints and weight, or zero.
    pub skin: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuMaterial {
    pub base_color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub translucency: f32,
    pub shading: u32,
    pub albedo: u32,
    pub normal: u32,
    pub orm: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuInstance {
    /// Rows of world from local.
    pub rows: [[f32; 4]; 3],
    /// Last frame delta minus identity.
    pub motion: [[f32; 4]; 3],
    /// Root node, or hole.
    pub root: u32,
    pub tint: u32,
    /// Far cut error multiplier.
    pub coarse: f32,
    pub pad: u32,
    /// Amplitude, inverse height, speed.
    pub wind: [f32; 4],
    /// Pose base, bones, frames; no frames is rest.
    pub pose: [u32; 4],
    /// Clip start, speed, fps, skin reach.
    pub play: [f32; 4],
}

/// Per-view geometry pass constants.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuView {
    pub clip_from_world: [[f32; 4]; 4],
    pub world_from_clip: [[f32; 4]; 4],
    pub unjittered_clip_from_world: [[f32; 4]; 4],
    pub previous_clip_from_world: [[f32; 4]; 4],
    pub frustum: [[f32; 4]; PLANES],
    pub eye: [f32; 3],
    /// Error to pixels, unit distance.
    pub lod_scale: f32,
    /// Camera plants sway around.
    pub wind_eye: [f32; 3],
    pub wind_reach: f32,
    pub size: [f32; 2],
    /// Allowed cut error, pixels.
    pub threshold: f32,
    /// Hardware raster above this, pixels.
    pub soft_limit: f32,
    pub time: f32,
    pub previous_time: f32,
    pub near: f32,
    pub instance_extent: u32,
    pub dynamic_count: u32,
    /// One for parallel projections.
    pub orthographic: u32,
    /// One where occlusion culls.
    pub occlusion: u32,
    /// Frustum planes in use.
    pub planes: u32,
    /// Debug view, zero shades.
    pub visualize: u32,
    pub static_capacity: u32,
    /// Owners, queue level, candidates, raster path, late instances, nodes, clusters.
    pub capacity: [u32; 7],
    /// Weather sway scale.
    pub wind_strength: f32,
    /// Uniform size, as WGSL rounds.
    pub pad: [u32; 2],
}

/// Unit vector, octahedron folded.
fn octahedral(v: [f32; 3]) -> (f32, f32) {
    let sum = v[0].abs() + v[1].abs() + v[2].abs();
    let (x, y) = (v[0] / sum, v[1] / sum);
    if v[2] >= 0.0 {
        return (x, y);
    }
    ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum())
}

/// Signed value in `bits` bits.
fn snorm(c: f32, bits: u32) -> u32 {
    let scale = ((1 << (bits - 1)) - 1) as f32;
    let mask = (1u32 << bits) - 1;
    ((c.clamp(-1.0, 1.0) * scale).round() as i32 as u32) & mask
}

/// Octahedral direction in two snorm16.
pub fn pack_direction(v: [f32; 3]) -> u32 {
    let (x, y) = octahedral(v);
    snorm(x, 16) | (snorm(y, 16) << 16)
}

/// Snorm direction, sign on top.
pub fn pack_tangent(t: [f32; 4]) -> u32 {
    let (x, y) = octahedral([t[0], t[1], t[2]]);
    let sign = if t[3] < 0.0 { 1 << 31 } else { 0 };
    snorm(x, 16) | (snorm(y, 15) << 16) | sign
}

/// Three cluster-local corners.
pub fn pack_triangle(t: [u8; 3]) -> u32 {
    t[0] as u32 | ((t[1] as u32) << 8) | ((t[2] as u32) << 16)
}

/// RGBA8 colour.
pub fn pack_color(c: [u8; 4]) -> u32 {
    u32::from_le_bytes(c)
}

/// Vertices, triangles and material of a cluster.
pub fn pack_counts(vertices: u32, triangles: u32, material: u32) -> u32 {
    vertices | (triangles << 8) | (material << 16)
}

/// Child entry of a node.
pub fn pack_child(kind: u32, index: u32, count: u32) -> [u32; 2] {
    [(kind << CHILD_INDEX_BITS) | index, count]
}

/// Fixed-point steps of packed wind, as common.wgsl.
const WIND_UNIT: f32 = 4096.0;

/// Sway amplitude in metres, height of full bend, gust speed.
pub fn pack_wind(amplitude: f32, height: f32, speed: f32) -> [u32; 2] {
    let unit = |v: f32| (v * WIND_UNIT).round().clamp(0.0, f32::from(u16::MAX)) as u32;
    [
        amplitude.to_bits(),
        unit(1.0 / height.max(f32::EPSILON)) | (unit(speed) << 16),
    ]
}

/// Row-major 3x4, vec4 padded.
pub fn rows(m: [f32; 12]) -> [[f32; 4]; 3] {
    [
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
    ]
}
