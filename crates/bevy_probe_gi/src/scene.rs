use crate::clipmap::Focus;
use crate::occluder::{Cover, ProbeOccluder};
use bevy_asset::AssetId;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Query, Res, ResMut};
use bevy_math::{Affine3A, Vec2, Vec3};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_platform::collections::HashMap;
use bevy_render::mesh::allocator::{MeshAllocator, MeshBufferSlice};
use bevy_render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy_render::render_asset::RenderAssets;
use bevy_render::render_resource::*;
use bevy_render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;

/// Occluder range, metres.
const REACH: f32 = 2_200.0;
/// Triangles of new BLAS per frame.
const BUILD_BUDGET: u32 = 400_000;
const MIN_CAPACITY: u32 = 4_096;
/// Mask bit of solid geometry.
const SOLID: u8 = 0x80;
/// Canopy instances spread over these bits.
const CANOPY_GROUPS: u8 = 7;

/// One placed occluder.
#[derive(Clone, PartialEq)]
struct Placed {
    mesh: AssetId<Mesh>,
    transform: Affine3A,
    albedo: Vec3,
    cover: Cover,
}

/// Per-instance shading data.
#[derive(ShaderType, Clone, Copy)]
pub(crate) struct Record {
    albedo: Vec3,
    cover: u32,
}

/// BLAS source geometry.
struct Source<'a> {
    size: BlasTriangleGeometrySizeDescriptor,
    stride: u64,
    vertices: MeshBufferSlice<'a>,
    indices: MeshBufferSlice<'a>,
}

#[derive(Resource, Default)]
pub(crate) struct Scene {
    placed: Vec<Placed>,
    /// Occluders the TLAS holds.
    built: Vec<Placed>,
    blas: HashMap<AssetId<Mesh>, Blas>,
    tlas: Option<Tlas>,
    capacity: u32,
    filled: u32,
    pub records: StorageBuffer<Vec<Record>>,
    /// New BLAS to build with the TLAS this frame.
    pending: Option<Vec<AssetId<Mesh>>>,
}

impl Scene {
    pub fn tlas(&self) -> Option<&Tlas> {
        self.tlas.as_ref()
    }
}

pub(crate) fn extract_occluders(
    mut scene: ResMut<Scene>,
    focus: Focus,
    occluders: Extract<Query<(&GlobalTransform, &Mesh3d, &ProbeOccluder)>>,
) {
    let Some(gi) = focus.iter().next() else {
        return;
    };
    let focus = Vec2::new(gi.focus.x, gi.focus.z);
    scene.placed.clear();
    for (t, mesh, o) in occluders.iter() {
        let p = t.translation();
        if Vec2::new(p.x, p.z).distance_squared(focus) > REACH * REACH {
            continue;
        }
        scene.placed.push(Placed {
            mesh: mesh.id(),
            transform: t.affine(),
            albedo: o.albedo,
            cover: o.cover,
        });
    }
}

/// Canopies spread over group bits.
fn mask(cover: Cover, slot: usize) -> u8 {
    if cover == Cover::Canopy {
        1 << (slot % usize::from(CANOPY_GROUPS))
    } else {
        SOLID
    }
}

fn rows(m: &Affine3A) -> [f32; 12] {
    let (c, t) = (m.matrix3, m.translation);
    [
        c.x_axis.x, c.y_axis.x, c.z_axis.x, t.x, c.x_axis.y, c.y_axis.y, c.z_axis.y, t.y,
        c.x_axis.z, c.y_axis.z, c.z_axis.z, t.z,
    ]
}

fn source<'a>(
    mesh: &RenderMesh,
    allocator: &'a MeshAllocator,
    id: AssetId<Mesh>,
) -> Option<Source<'a>> {
    let RenderMeshBufferInfo::Indexed {
        count,
        index_format,
    } = mesh.buffer_info
    else {
        return None;
    };
    let vertices = allocator.mesh_vertex_slice(&id)?;
    let indices = allocator.mesh_index_slice(&id)?;
    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: vertices.range.len() as u32,
        index_format: Some(index_format),
        index_count: Some(count),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    Some(Source {
        size,
        stride: mesh.layout.0.layout().array_stride,
        vertices,
        indices,
    })
}

fn flags() -> AccelerationStructureFlags {
    AccelerationStructureFlags::PREFER_FAST_TRACE
        | AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN
}

fn new_blas(device: &RenderDevice, size: &BlasTriangleGeometrySizeDescriptor) -> Blas {
    let descriptor = CreateBlasDescriptor {
        label: Some("probe_gi_blas"),
        flags: flags(),
        update_mode: AccelerationStructureUpdateMode::Build,
    };
    device.wgpu_device().create_blas(
        &descriptor,
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    )
}

fn new_tlas(device: &RenderDevice, capacity: u32) -> Tlas {
    let descriptor = CreateTlasDescriptor {
        label: Some("probe_gi_tlas"),
        max_instances: capacity,
        flags: flags(),
        update_mode: AccelerationStructureUpdateMode::Build,
    };
    device.wgpu_device().create_tlas(&descriptor)
}

fn geometry<'a>(s: &'a Source<'a>) -> BlasTriangleGeometry<'a> {
    BlasTriangleGeometry {
        size: &s.size,
        vertex_buffer: s.vertices.buffer,
        first_vertex: s.vertices.range.start,
        vertex_stride: s.stride,
        index_buffer: Some(s.indices.buffer),
        first_index: Some(s.indices.range.start),
        transform_buffer: None,
        transform_buffer_offset: None,
    }
}

/// Allocates BLAS for meshes within budget.
fn queue_blas(
    scene: &mut Scene,
    meshes: &RenderAssets<RenderMesh>,
    allocator: &MeshAllocator,
    device: &RenderDevice,
) -> Vec<AssetId<Mesh>> {
    let mut fresh = Vec::new();
    let mut budget = BUILD_BUDGET;
    for p in &scene.placed {
        if scene.blas.contains_key(&p.mesh) {
            continue;
        }
        let Some(src) = meshes
            .get(p.mesh)
            .and_then(|m| source(m, allocator, p.mesh))
        else {
            continue;
        };
        let triangles = src.size.index_count.unwrap_or(0) / 3;
        if triangles > budget && budget < BUILD_BUDGET {
            break;
        }
        budget = budget.saturating_sub(triangles);
        scene.blas.insert(p.mesh, new_blas(device, &src.size));
        fresh.push(p.mesh);
    }
    fresh
}

/// Fills the TLAS and instance records.
fn place(scene: &mut Scene, device: &RenderDevice) {
    let needed = scene.placed.len() as u32;
    if scene.tlas.is_none() || needed > scene.capacity {
        scene.capacity = needed.max(MIN_CAPACITY).next_power_of_two();
        scene.tlas = Some(new_tlas(device, scene.capacity));
        scene.filled = 0;
    }
    let Some(tlas) = scene.tlas.as_mut() else {
        return;
    };
    let records = scene.records.get_mut();
    records.clear();
    for p in &scene.placed {
        let Some(blas) = scene.blas.get(&p.mesh) else {
            continue;
        };
        let slot = records.len();
        tlas[slot] = Some(TlasInstance::new(
            blas,
            rows(&p.transform),
            slot as u32,
            mask(p.cover, slot),
        ));
        records.push(Record {
            albedo: p.albedo,
            cover: p.cover as u32,
        });
    }
    for slot in records.len()..scene.filled as usize {
        tlas[slot] = None;
    }
    scene.filled = records.len() as u32;
}

pub(crate) fn build(
    mut scene: ResMut<Scene>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let scene = &mut *scene;
    scene.blas.retain(|id, _| meshes.get(*id).is_some());
    let fresh = queue_blas(scene, &meshes, &allocator, &device);
    if fresh.is_empty() && scene.placed == scene.built {
        scene.pending = None;
        return;
    }
    scene.built.clone_from(&scene.placed);
    place(scene, &device);
    scene.records.write_buffer(&device, &queue);
    scene.pending = Some(fresh);
}

/// Builds new BLAS and the TLAS ahead of tracing.
pub(crate) fn record(
    scene: Res<Scene>,
    (meshes, allocator): (Res<RenderAssets<RenderMesh>>, Res<MeshAllocator>),
    mut ctx: RenderContext,
) {
    let Some(fresh) = &scene.pending else {
        return;
    };
    let sources: Vec<(&Blas, Source)> = fresh
        .iter()
        .filter_map(|id| {
            Some((
                scene.blas.get(id)?,
                source(meshes.get(*id)?, &allocator, *id)?,
            ))
        })
        .collect();
    let entries: Vec<BlasBuildEntry> = sources
        .iter()
        .map(|(blas, s)| BlasBuildEntry {
            blas,
            geometry: BlasGeometries::TriangleGeometries(vec![geometry(s)]),
        })
        .collect();
    ctx.command_encoder()
        .build_acceleration_structures(&entries, scene.tlas.iter());
}
