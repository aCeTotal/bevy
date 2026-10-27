use crate::arrays::ClipRef;
use crate::asset::VirtualGeometry;
use crate::library::Library;
use crate::records::GpuInstance;
use crate::wind::Wind;
use bevy_asset::Handle;
use bevy_camera::visibility::{InheritedVisibility, Visibility};
use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Query, Res, ResMut};
use bevy_math::{Affine3A, DMat3};
use bevy_platform::collections::HashMap;
use bevy_render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_render::Extract;
use bevy_transform::components::{GlobalTransform, Transform};

/// Copies the buffer starts with, still and moving.
const FIRST_CAPACITY: (u32, u32) = (1 << 12, 1 << 12);
/// Root marking an empty slot.
const HOLE: u32 = u32::MAX;
/// Cut error left as built.
const EXACT: f32 = 1.0;

/// Entity drawn from virtual geometry.
#[derive(Component, Clone)]
#[require(Transform, Visibility, Previous)]
pub struct VirtualGeometry3d {
    /// Hierarchies to draw.
    pub geometry: Handle<VirtualGeometry>,
    /// Which root.
    pub variant: u32,
    /// Bend in wind.
    pub wind: Wind,
    /// Colour of tinted layers, RGBA8.
    pub tint: u32,
}

/// Clip a virtual entity plays.
#[derive(Component, Clone, Copy, Debug)]
pub struct Play {
    /// Clip index within its geometry.
    pub clip: u32,
    /// Shader clock when the clip began.
    pub start: f32,
    /// Playback rate.
    pub speed: f32,
}

#[derive(Resource)]
pub(crate) struct Instances {
    /// Static slot per key.
    slots: HashMap<u64, u32>,
    free: Vec<u32>,
    /// Static slots ever used.
    pub extent: u32,
    dynamic: Vec<GpuInstance>,
    /// Still slots, then moving ones.
    pub buffer: Buffer,
    pub static_capacity: u32,
    dynamic_capacity: u32,
    pub dynamic_count: u32,
    /// Bumped when the buffer is replaced.
    pub generation: u32,
}

fn instance_buffer(device: &RenderDevice, (still, moving): (u32, u32)) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("virtual_geometry_instances"),
        size: u64::from(still + moving) * size_of::<GpuInstance>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub(crate) fn setup_instances(mut commands: Commands, device: Res<RenderDevice>) {
    commands.insert_resource(Instances {
        slots: HashMap::default(),
        free: Vec::new(),
        extent: 0,
        dynamic: Vec::new(),
        buffer: instance_buffer(&device, FIRST_CAPACITY),
        static_capacity: FIRST_CAPACITY.0,
        dynamic_capacity: FIRST_CAPACITY.1,
        dynamic_count: 0,
        generation: 0,
    });
}

/// Row-major 3x4 of an affine.
pub fn affine_rows(m: Affine3A) -> [[f32; 4]; 3] {
    let (c, t) = (m.matrix3, m.translation);
    [
        [c.x_axis.x, c.y_axis.x, c.z_axis.x, t.x],
        [c.x_axis.y, c.y_axis.y, c.z_axis.y, t.y],
        [c.x_axis.z, c.y_axis.z, c.z_axis.z, t.z],
    ]
}

fn wind(w: Wind) -> [f32; 4] {
    [w.amplitude, 1.0 / w.height.max(f32::EPSILON), w.speed, 0.0]
}

/// How a copy draws.
struct Look {
    wind: Wind,
    tint: u32,
    coarse: f32,
    /// Clip and its playback, if animated.
    clip: Option<(ClipRef, Play)>,
}

/// Clip words; zero when still.
fn animation(clip: Option<(ClipRef, Play)>) -> ([u32; 4], [f32; 4]) {
    clip.map_or(([0; 4], [0.0; 4]), |(c, p)| {
        (
            [c.base, c.bones, c.frames, 0],
            [p.start, p.speed, c.fps, c.reach],
        )
    })
}

/// Last frame delta minus identity.
fn motion(now: Affine3A, before: Affine3A) -> [[f32; 4]; 3] {
    if now == before {
        return [[0.0; 4]; 3];
    }
    let delta = before.as_daffine3() * now.as_daffine3().inverse();
    affine_rows(Affine3A::from_mat3_translation(
        (delta.matrix3 - DMat3::IDENTITY).as_mat3(),
        delta.translation.as_vec3(),
    ))
}

fn instance(root: u32, now: Affine3A, before: Affine3A, look: Look) -> GpuInstance {
    let (pose, play) = animation(look.clip);
    GpuInstance {
        pose,
        play,
        rows: affine_rows(now),
        motion: motion(now, before),
        root,
        tint: look.tint,
        coarse: look.coarse,
        pad: 0,
        wind: wind(look.wind),
    }
}

fn hole() -> GpuInstance {
    GpuInstance {
        root: HOLE,
        ..bytemuck::Zeroable::zeroed()
    }
}

impl Instances {
    fn write(&self, at: u32, item: &GpuInstance, queue: &RenderQueue) {
        queue.write_buffer(
            &self.buffer,
            u64::from(at) * size_of::<GpuInstance>() as u64,
            bytemuck::bytes_of(item),
        );
    }

    /// Room for `still` and `moving` copies, still ones kept.
    fn grow(
        &mut self,
        (still, moving): (u32, u32),
        (device, queue): (&RenderDevice, &RenderQueue),
    ) {
        let buffer = instance_buffer(device, (still, moving));
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("virtual_geometry_instances"),
        });
        let kept = u64::from(self.static_capacity) * size_of::<GpuInstance>() as u64;
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, kept);
        queue.submit([encoder.finish()]);
        (self.buffer, self.static_capacity, self.dynamic_capacity) = (buffer, still, moving);
        self.generation += 1;
    }

    /// Draws a still root under `key`.
    pub fn set(
        &mut self,
        key: u64,
        (root, transform, coarse): (u32, Affine3A, f32),
        gpu: (&RenderDevice, &RenderQueue),
    ) {
        if self.free.is_empty() && self.extent == self.static_capacity {
            self.grow((self.static_capacity * 2, self.dynamic_capacity), gpu);
        }
        let queue = gpu.1;
        let slot = self.free.pop().unwrap_or(self.extent);
        self.extent = self.extent.max(slot + 1);
        let look = Look {
            wind: Wind::default(),
            tint: 0,
            coarse,
            clip: None,
        };
        self.write(slot, &instance(root, transform, transform, look), queue);
        self.slots.insert(key, slot);
    }

    pub fn remove(&mut self, key: u64, queue: &RenderQueue) {
        let Some(slot) = self.slots.remove(&key) else {
            return;
        };
        self.write(slot, &hole(), queue);
        self.free.push(slot);
    }
}

/// World transform drawn last frame.
#[derive(Component, Clone, Copy, Default)]
pub struct Previous(Option<Affine3A>);

/// Keeps last transform for motion.
pub(crate) fn remember(mut drawn: Query<(&GlobalTransform, &mut Previous)>) {
    drawn
        .par_iter_mut()
        .for_each(|(now, mut before)| before.0 = Some(now.affine()));
}

/// Virtual entity with its motion.
type Drawn = (
    &'static VirtualGeometry3d,
    &'static GlobalTransform,
    &'static Previous,
    &'static InheritedVisibility,
    Option<&'static Play>,
);

pub(crate) fn extract_virtual(
    mut instances: ResMut<Instances>,
    library: Res<Library>,
    query: Extract<Query<Drawn>>,
) {
    instances.dynamic.clear();
    for (v, transform, previous, visible, play) in query.iter() {
        let loaded = library.loaded.get(&v.geometry.id());
        let root = loaded.and_then(|l| l.roots.get(v.variant as usize));
        let (Some(&root), true) = (root, visible.get()) else {
            continue;
        };
        let now = transform.affine();
        let clip = play.and_then(|p| Some((*loaded?.clips.get(p.clip as usize)?, *p)));
        let look = Look {
            wind: v.wind,
            tint: v.tint,
            coarse: EXACT,
            clip,
        };
        instances
            .dynamic
            .push(instance(root, now, previous.0.unwrap_or(now), look));
    }
}

pub(crate) fn prepare_dynamic(
    mut instances: ResMut<Instances>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
) {
    let count = instances.dynamic.len() as u32;
    if count > instances.dynamic_capacity {
        let still = instances.static_capacity;
        instances.grow((still, count.next_power_of_two()), (&device, &queue));
    }
    instances.dynamic_count = count;
    if count == 0 {
        return;
    }
    let offset = u64::from(instances.static_capacity) * size_of::<GpuInstance>() as u64;
    queue.write_buffer(
        &instances.buffer,
        offset,
        bytemuck::cast_slice(&instances.dynamic),
    );
}
