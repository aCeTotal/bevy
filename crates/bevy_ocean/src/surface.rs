use crate::buoyancy::Floating;
use crate::fft::WaveStep;
use crate::material::{Hull, OceanFrame, OceanMaterial, OceanSurface, MAX_HULLS};
use crate::spectrum::SIDE;
use crate::waves::{Waves, BANDS, CASCADES};
use crate::{smoothstep, Ocean, OceanSeabed};
use bevy_asset::{Assets, RenderAssetUsages};
use bevy_camera::visibility::Visibility;
use bevy_ecs::change_detection::DetectChangesMut;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::observer::On;
use bevy_ecs::query::{Has, With};
use bevy_ecs::system::{Commands, Local, Query, Res, ResMut};
use bevy_image::Image;
use bevy_light::NotShadowCaster;
use bevy_math::{Vec2, Vec3, Vec3Swizzles, Vec4};
use bevy_mesh::{Indices, Mesh, Mesh3d, PrimitiveTopology};
use bevy_pbr::{ExtendedMaterial, MeshMaterial3d, StandardMaterial};
use bevy_render::gpu_readback::{Readback, ReadbackComplete};
use bevy_render::renderer::RenderDevice;
use bevy_time::Time;
use bevy_transform::components::Transform;

/// Cells per patch side.
pub(crate) const CELLS: u32 = 320;
/// Flat ring reach, in patch half-widths.
const RING_REACH: f32 = 750.0;
/// Texel spans vertices follow, cells.
const FOLLOW: (f32, f32) = (1.0, 2.0);
/// Cascades vertices never follow.
const VERTEX_CASCADES: usize = 2;

/// Sea surface riding the focus.
#[derive(Component)]
pub(crate) struct OceanMesh;

/// Displaced part, while waves move.
#[derive(Component)]
pub(crate) struct Swell;

/// Reads the sea back while hulls ride it.
#[derive(Component)]
pub(crate) struct WaveReader;

fn mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    let count = positions.len();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; count])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; count])
    .with_inserted_indices(Indices::U32(indices))
}

/// Unit grid over -1..1.
fn patch() -> Mesh {
    let side = CELLS + 1;
    let at = |i: u32| (i as f32 / CELLS as f32) * 2.0 - 1.0;
    let positions = (0..side * side)
        .map(|i| [at(i % side), 0.0, at(i / side)])
        .collect();
    let quad = |c: u32| {
        let i = c / CELLS * side + c % CELLS;
        [i, i + side, i + 1, i + 1, i + side, i + side + 1]
    };
    mesh(positions, (0..CELLS * CELLS).flat_map(quad).collect())
}

/// Flat skirt sharing the patch edge.
fn ring() -> Mesh {
    let at = |i: u32| (i as f32 / CELLS as f32) * 2.0 - 1.0;
    // Edge points, and flipped winding.
    let sides: [(fn(f32, f32) -> [f32; 3], bool); 4] = [
        (|t, r| [t * r, 0.0, -r], true),
        (|t, r| [r, 0.0, t * r], true),
        (|t, r| [t * r, 0.0, r], false),
        (|t, r| [-r, 0.0, t * r], false),
    ];
    let edge = |side: fn(f32, f32) -> [f32; 3]| {
        (0..=CELLS).flat_map(move |i| [side(at(i), 1.0), side(at(i), RING_REACH)])
    };
    let positions: Vec<[f32; 3]> = sides.iter().flat_map(|&(s, _)| edge(s)).collect();
    let quad = |s: u32, i: u32| {
        let k = 2 * (s * (CELLS + 1) + i);
        if sides[s as usize].1 {
            [k, k + 2, k + 1, k + 2, k + 3, k + 1]
        } else {
            [k, k + 1, k + 2, k + 2, k + 1, k + 3]
        }
    };
    let indices = (0..4)
        .flat_map(|s| (0..CELLS).flat_map(move |i| quad(s, i)))
        .collect();
    mesh(positions, indices)
}

/// Asset stores the sea fills.
type Stores<'w> = (
    ResMut<'w, Assets<Mesh>>,
    ResMut<'w, Assets<OceanMaterial>>,
    ResMut<'w, Assets<Image>>,
);

pub(crate) fn setup_ocean(
    mut commands: Commands,
    (ocean, seabed, device): (Res<Ocean>, Res<OceanSeabed>, Res<RenderDevice>),
    (mut meshes, mut materials, mut images): Stores,
) {
    let (surface, frame) = OceanSurface::new((&mut images, &device), &ocean, &seabed);
    commands.insert_resource(WaveStep::new(surface.shape.clone(), surface.slope.clone()));
    commands.insert_resource(frame);
    let look = materials.add(ExtendedMaterial {
        base: StandardMaterial::default(),
        extension: surface,
    });
    let placed = Transform::from_xyz(0.0, ocean.level, 0.0);
    commands.spawn((
        OceanMesh,
        Swell,
        Mesh3d(meshes.add(patch())),
        MeshMaterial3d(look.clone()),
        placed,
        NotShadowCaster,
    ));
    commands.spawn((
        OceanMesh,
        Mesh3d(meshes.add(ring())),
        MeshMaterial3d(look),
        placed,
        NotShadowCaster,
    ));
}

/// Vertex share per cascade.
fn reach(spacing: f32) -> [f32; CASCADES] {
    core::array::from_fn(|c| {
        let texel = BANDS[c].size / SIDE as f32;
        let follows = 1.0 - smoothstep(texel * FOLLOW.0, texel * FOLLOW.1, spacing);
        if c < VERTEX_CASCADES {
            follows
        } else {
            0.0
        }
    })
}

/// Snaps the surface to the focus.
pub(crate) fn follow_focus(
    ocean: Res<Ocean>,
    mut waves: ResMut<Waves>,
    mut meshes: Query<(&mut Transform, &mut Visibility, Has<Swell>), With<OceanMesh>>,
) {
    let spacing = ocean.spacing;
    let snap = |v: f32| (v / spacing).round() * spacing;
    let half = ocean.reach();
    let center = Vec2::new(snap(ocean.focus.x), snap(ocean.focus.y));
    waves.patch.center = center;
    waves.patch.half = half;
    waves.patch.reach = if ocean.waves {
        reach(spacing)
    } else {
        [0.0; CASCADES]
    };
    waves.patch.coastal = ocean.waves;
    for (mut t, mut vis, swell) in &mut meshes {
        *t = Transform::from_xyz(center.x, ocean.level, center.y)
            .with_scale(Vec3::new(half, 1.0, half));
        vis.set_if_neq(if swell && !ocean.waves {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        });
    }
}

/// Runs the GPU sea where waves show or carry hulls.
pub(crate) fn stir_sea(
    time: Res<Time>,
    (waves, hulls): (Res<Waves>, Query<(), With<Floating>>),
    mut step: ResMut<WaveStep>,
) {
    step.tick(&time, waves.patch.coastal || !hulls.is_empty());
}

/// Reads the sea back to the CPU only while hulls float on it.
pub(crate) fn follow_hulls(
    mut commands: Commands,
    (hulls, readers): (Query<(), With<Floating>>, Query<Entity, With<WaveReader>>),
    step: Res<WaveStep>,
) {
    match (hulls.is_empty(), readers.single()) {
        (false, Err(_)) => {
            commands
                .spawn((WaveReader, Readback::texture(step.shape.clone())))
                .observe(|done: On<ReadbackComplete>, mut waves: ResMut<Waves>| {
                    waves.read(&done.data)
                });
        }
        (true, Ok(reader)) => commands.entity(reader).despawn(),
        _ => {}
    }
}

/// Charts reach, seabed and hulls.
pub(crate) fn chart_sea(
    (waves, ocean, seabed): (Res<Waves>, Res<Ocean>, Res<OceanSeabed>),
    hulls: Query<(&Transform, &Floating)>,
    mut frame: ResMut<OceanFrame>,
    mut nearest: Local<Vec<(f32, Hull)>>,
) {
    nearest.clear();
    nearest.extend(hulls.iter().map(|(t, f)| {
        (
            t.translation.xz().distance_squared(ocean.focus),
            Hull::of(t, f),
        )
    }));
    nearest.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    nearest.truncate(MAX_HULLS);
    let state = &mut frame.state;
    state.reach = Vec4::new(
        waves.patch.reach[0],
        waves.patch.reach[1],
        waves.patch.reach[2],
        nearest.len() as f32,
    );
    state.seabed = seabed.frame();
    state.level.x = ocean.level;
    state.hulls = core::array::from_fn(|i| nearest.get(i).map_or_else(Hull::default, |n| n.1));
}
