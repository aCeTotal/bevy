use crate::capacity::{Capacities, Capacity, ITEM_BYTES, PEAKS};
use crate::instances::Instances;
use crate::library::Library;
use crate::pipelines::{Pipelines, RASTER_DEPTH};
use crate::pyramid::Pyramid;
use crate::records::{GpuView, FIXED_COUNTERS, MAX_LEVELS, POSES_BINDING, SLOT_BYTES};
use alloc::sync::Arc;
use bevy_camera::{Camera3d, MainPassResolutionOverride};
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::query::With;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Query, Res, ResMut};
use bevy_math::UVec2;
use bevy_pbr::ShadowView;
use bevy_render::camera::ExtractedCamera;
use bevy_render::render_resource::*;
use bevy_render::renderer::RenderDevice;

/// Peak words read back.
pub(crate) const PEAK_BYTES: u64 = PEAKS as u64 * 4;
/// Fixed counters, then one per level.
const COUNTER_BYTES: u64 = (FIXED_COUNTERS as u64 + MAX_LEVELS as u64) * 4;
const ARG_BYTES: u64 = 4 * SLOT_BYTES;
/// Depth and id per pixel.
const PIXEL_BYTES: u64 = 8;

pub(crate) struct Groups {
    pub library: BindGroup,
    pub cull: BindGroup,
    /// Main and per-level arguments, for prepares.
    pub args: BindGroup,
    /// Next-level arguments, by level parity.
    pub toward: [BindGroup; 2],
    pub raster: BindGroup,
    /// Camera views only.
    pub screen: Option<ScreenGroups>,
}

/// Bindings of a camera's targets.
pub(crate) struct ScreenGroups {
    pub target: BindGroup,
    pub resolve: BindGroup,
    /// Per level: bindings and source extent.
    pub pyramid: Vec<(BindGroup, UVec2)>,
}

/// Buffers views take turns on.
pub(crate) struct Work {
    owners: Buffer,
    queue: Buffer,
    candidates: Buffer,
    visible: Buffer,
    pub counters: Buffer,
    pub args: Buffer,
    /// Per-level arguments, even and odd.
    pub steps: [Buffer; 2],
    late_instances: Buffer,
    late_nodes: Buffer,
    late_clusters: Buffer,
    /// Largest list demand this frame.
    pub peaks: Buffer,
    pub capacity: Capacity,
    /// Camera views only.
    pub screen: Option<Screen>,
}

/// A camera's raster targets.
pub(crate) struct Screen {
    pub size: UVec2,
    /// Nearest depth and id per pixel.
    pub visibility: Buffer,
    pub raster_depth: TextureView,
    pyramid: Pyramid,
}

/// One view's constants and bindings.
#[derive(Component)]
pub(crate) struct ViewGeometry {
    pub work: Arc<Work>,
    pub uniform: Buffer,
    pub groups: Option<Groups>,
    /// Library and instance buffers bound.
    generation: (u32, u32),
}

/// Work shared by every shadow view.
#[derive(Resource, Default)]
pub(crate) struct ShadowWork(Option<Arc<Work>>);

impl ShadowWork {
    pub fn work(&self) -> Option<&Arc<Work>> {
        self.0.as_ref()
    }
}

fn buffer(device: &RenderDevice, label: &str, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn texture(
    device: &RenderDevice,
    size: UVec2,
    (format, usage): (TextureFormat, TextureUsages),
) -> TextureView {
    let size = Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: 1,
    };
    let descriptor = TextureDescriptor {
        label: Some("virtual_geometry_target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    };
    device
        .create_texture(&descriptor)
        .create_view(&TextureViewDescriptor::default())
}

fn screen(device: &RenderDevice, size: UVec2) -> Screen {
    Screen {
        size,
        visibility: buffer(
            device,
            "virtual_geometry_visibility",
            u64::from(size.x) * u64::from(size.y) * PIXEL_BYTES,
            BufferUsages::STORAGE,
        ),
        raster_depth: texture(
            device,
            size,
            (RASTER_DEPTH, TextureUsages::RENDER_ATTACHMENT),
        ),
        pyramid: Pyramid::new(device, size),
    }
}

fn work(device: &RenderDevice, screen: Option<Screen>, capacity: Capacity) -> Arc<Work> {
    let storage = BufferUsages::STORAGE;
    let bytes = |list: usize| u64::from(capacity.0[list]) * ITEM_BYTES[list];
    // Shadows never defer work.
    let late = |list: usize| {
        if screen.is_some() {
            bytes(list)
        } else {
            ITEM_BYTES[list]
        }
    };
    Arc::new(Work {
        owners: buffer(device, "virtual_geometry_owners", bytes(0), storage),
        queue: buffer(device, "virtual_geometry_queue", bytes(1), storage),
        candidates: buffer(device, "virtual_geometry_candidates", bytes(2), storage),
        visible: buffer(device, "virtual_geometry_visible", bytes(3), storage),
        counters: buffer(
            device,
            "virtual_geometry_counters",
            COUNTER_BYTES,
            storage | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        ),
        args: buffer(
            device,
            "virtual_geometry_args",
            ARG_BYTES,
            storage | BufferUsages::INDIRECT,
        ),
        steps: core::array::from_fn(|_| steps(device)),
        late_instances: buffer(device, "virtual_geometry_late_instances", late(4), storage),
        late_nodes: buffer(device, "virtual_geometry_late_nodes", late(5), storage),
        late_clusters: buffer(device, "virtual_geometry_late_clusters", late(6), storage),
        peaks: buffer(
            device,
            "virtual_geometry_peaks",
            PEAK_BYTES,
            storage | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        ),
        capacity,
        screen,
    })
}

/// Per-level arguments, initially empty.
fn steps(device: &RenderDevice) -> Buffer {
    let slots = [[0u32, 1, 1]; MAX_LEVELS as usize];
    let usage = BufferUsages::STORAGE | BufferUsages::INDIRECT;
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("virtual_geometry_steps"),
        contents: bytemuck::cast_slice(&slots),
        usage,
    })
}

fn view_geometry(device: &RenderDevice, work: Arc<Work>) -> ViewGeometry {
    let uniform = buffer(
        device,
        "virtual_geometry_view",
        size_of::<GpuView>() as u64,
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );
    ViewGeometry {
        work,
        uniform,
        groups: None,
        generation: (0, 0),
    }
}

type Cameras<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static ExtractedCamera,
        Option<&'static MainPassResolutionOverride>,
        Option<&'static ViewGeometry>,
    ),
    With<Camera3d>,
>;
type Shadows<'w, 's> = Query<'w, 's, (Entity, Option<&'static ViewGeometry>), With<ShadowView>>;

pub(crate) fn prepare_views(
    mut commands: Commands,
    (cameras, shadows): (Cameras, Shadows),
    (mut shared, capacities): (ResMut<ShadowWork>, Res<Capacities>),
    device: Res<RenderDevice>,
) {
    for (entity, camera, scaled, existing) in &cameras {
        let Some(size) = scaled.map(|s| s.0).or(camera.physical_viewport_size) else {
            continue;
        };
        let current = existing
            .map(|v| &v.work)
            .filter(|w| w.capacity == capacities.camera);
        if current
            .and_then(|w| w.screen.as_ref())
            .is_some_and(|s| s.size == size)
        {
            continue;
        }
        let w = work(&device, Some(screen(&device, size)), capacities.camera);
        commands.entity(entity).insert(view_geometry(&device, w));
    }
    if shadows.is_empty() {
        return;
    }
    let stale = shared
        .0
        .as_ref()
        .is_none_or(|w| w.capacity != capacities.shadow);
    if stale {
        shared.0 = Some(work(&device, None, capacities.shadow));
    }
    let Some(w) = shared.0.as_ref() else { return };
    for (entity, existing) in &shadows {
        if stale || existing.is_none() {
            commands
                .entity(entity)
                .insert(view_geometry(&device, w.clone()));
        }
    }
}

impl ViewGeometry {
    /// What resolve and shading read.
    pub fn read_entries(&self) -> Option<[(u32, BindingResource<'_>); 5]> {
        let s = self.work.screen.as_ref()?;
        Some([
            (0, self.uniform.as_entire_binding()),
            (1, self.work.owners.as_entire_binding()),
            (4, self.work.visible.as_entire_binding()),
            (5, self.work.counters.as_entire_binding()),
            (7, s.visibility.as_entire_binding()),
        ])
    }
}

pub(crate) fn entries<'a>(pairs: &[(u32, BindingResource<'a>)]) -> Vec<BindGroupEntry<'a>> {
    pairs
        .iter()
        .map(|(binding, resource)| BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect()
}

fn screen_groups(
    device: &RenderDevice,
    v: &ViewGeometry,
    (s, pipelines): (&Screen, &Pipelines),
) -> Option<ScreenGroups> {
    let canvas = [(0, s.visibility.as_entire_binding())];
    Some(ScreenGroups {
        target: device.create_bind_group(
            "virtual_geometry_target",
            &pipelines.target,
            &entries(&canvas),
        ),
        resolve: device.create_bind_group(
            "virtual_geometry_resolve",
            &pipelines.resolve_view,
            &entries(&v.read_entries()?),
        ),
        pyramid: s.pyramid.groups(
            device,
            (&pipelines.pyramid, &s.visibility, &pipelines.unused),
        ),
    })
}

fn groups(
    device: &RenderDevice,
    v: &ViewGeometry,
    (pipelines, library, instances): (&Pipelines, &Library, &Instances),
) -> Option<Groups> {
    let b = library.buffers.as_ref()?;
    let w = &v.work;
    let lib = [
        (0, b.positions.as_entire_binding()),
        (1, b.vertices.as_entire_binding()),
        (2, b.triangles.as_entire_binding()),
        (3, b.clusters.as_entire_binding()),
        (4, b.nodes.as_entire_binding()),
        (5, b.children.as_entire_binding()),
        (6, b.parts.as_entire_binding()),
        (7, instances.buffer.as_entire_binding()),
        (8, b.materials.as_entire_binding()),
        (POSES_BINDING, b.poses.as_entire_binding()),
    ];
    let read = [
        (0, v.uniform.as_entire_binding()),
        (1, w.owners.as_entire_binding()),
        (4, w.visible.as_entire_binding()),
        (5, w.counters.as_entire_binding()),
    ];
    let occluders = w
        .screen
        .as_ref()
        .map_or(&pipelines.unused, |s| &s.pyramid.whole);
    let cull = [
        (0, v.uniform.as_entire_binding()),
        (1, w.owners.as_entire_binding()),
        (2, w.queue.as_entire_binding()),
        (3, w.candidates.as_entire_binding()),
        (4, w.visible.as_entire_binding()),
        (5, w.counters.as_entire_binding()),
        (6, BindingResource::TextureView(occluders)),
        (7, w.late_instances.as_entire_binding()),
        (8, w.late_nodes.as_entire_binding()),
        (9, w.late_clusters.as_entire_binding()),
    ];
    Some(Groups {
        library: device.create_bind_group(
            "virtual_geometry_library",
            &pipelines.library,
            &entries(&lib),
        ),
        cull: device.create_bind_group(
            "virtual_geometry_cull",
            &pipelines.cull_view,
            &entries(&cull),
        ),
        args: device.create_bind_group(
            "virtual_geometry_args",
            &pipelines.cull_args,
            &entries(&[
                (0, w.args.as_entire_binding()),
                (1, w.steps[0].as_entire_binding()),
                (2, w.steps[1].as_entire_binding()),
                (4, w.peaks.as_entire_binding()),
            ]),
        ),
        toward: w.steps.each_ref().map(|s| {
            device.create_bind_group(
                "virtual_geometry_toward",
                &pipelines.toward,
                &entries(&[(3, s.as_entire_binding())]),
            )
        }),
        raster: device.create_bind_group(
            "virtual_geometry_raster",
            &pipelines.raster_view,
            &entries(&read),
        ),
        screen: w
            .screen
            .as_ref()
            .and_then(|s| screen_groups(device, v, (s, pipelines))),
    })
}

pub(crate) fn prepare_groups(
    mut views: Query<&mut ViewGeometry>,
    pipelines: Res<Pipelines>,
    library: Res<Library>,
    instances: Res<Instances>,
    device: Res<RenderDevice>,
) {
    for mut v in &mut views {
        let generation = (library.generation, instances.generation);
        if v.groups.is_some() && v.generation == generation {
            continue;
        }
        v.groups = groups(&device, &v, (&pipelines, &library, &instances));
        v.generation = generation;
    }
}
