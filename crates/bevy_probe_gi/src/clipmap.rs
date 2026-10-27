use bevy_asset::AssetId;
use bevy_asset::{Assets, RenderAssetUsages};
use bevy_camera::Camera3d;
use bevy_ecs::component::Component;
use bevy_ecs::query::With;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Query, ResMut};
use bevy_image::Image;
use bevy_light::{IrradianceVolume, LightProbe};
use bevy_math::{UVec3, Vec3};
use bevy_render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy_render::Extract;
use bevy_transform::components::Transform;
use bevy_utils::default;

/// Clipmap levels, finest first.
pub(crate) const LEVELS: usize = 4;
/// Probes per axis per level.
pub(crate) const CELLS: UVec3 = UVec3::new(32, 12, 32);
/// Metres between probes, per level.
pub(crate) const SPACING: [f32; LEVELS] = [2.0, 8.0, 32.0, 128.0];
/// Share of probes below the focus.
const BELOW: f32 = 0.3;
/// Edge blend width, unit cube.
const FALLOFF: f32 = 0.15;

/// Lights a camera's surroundings with ray-traced probes centred on `focus`.
#[derive(Component, Clone, Copy, Default, Debug)]
#[require(Camera3d)]
pub struct ProbeGi {
    /// Point the probe clipmap follows, usually what the camera looks at.
    pub focus: Vec3,
}

/// Clipmap level of a volume.
#[derive(Component)]
pub(crate) struct Level(usize);

/// Window origins and images per level.
#[derive(Resource, Default, Clone)]
pub(crate) struct Clipmap {
    pub origins: [Vec3; LEVELS],
    pub images: [AssetId<Image>; LEVELS],
}

fn voxels() -> Image {
    let size = Extent3d {
        width: CELLS.x,
        height: CELLS.y * 2,
        depth_or_array_layers: CELLS.z * 3,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D3,
        &[0; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST;
    image
}

pub(crate) fn spawn_volumes(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    for (level, spacing) in SPACING.iter().enumerate() {
        let voxels = images.add(voxels());
        let probe = LightProbe {
            falloff: Vec3::splat(FALLOFF),
        };
        let volume = IrradianceVolume {
            voxels,
            intensity: 1.0,
            ..default()
        };
        commands.spawn((
            Level(level),
            probe,
            volume,
            Transform::from_scale(CELLS.as_vec3() * *spacing),
        ));
    }
}

/// Snaps each window to the focus.
pub(crate) fn follow_focus(gi: Query<&ProbeGi>, mut volumes: Query<(&Level, &mut Transform)>) {
    let Some(gi) = gi.iter().next() else { return };
    for (level, mut t) in &mut volumes {
        let spacing = SPACING[level.0];
        let extent = CELLS.as_vec3() * spacing;
        let corner = gi.focus - extent * Vec3::new(0.5, BELOW, 0.5);
        let placed = (corner / spacing).floor() * spacing + extent * 0.5;
        if t.translation != placed {
            t.translation = placed;
        }
    }
}

pub(crate) fn extract_clipmap(
    mut clipmap: ResMut<Clipmap>,
    volumes: Extract<Query<(&Level, &Transform, &IrradianceVolume)>>,
) {
    for (level, t, volume) in volumes.iter() {
        clipmap.origins[level.0] = t.translation - CELLS.as_vec3() * SPACING[level.0] * 0.5;
        clipmap.images[level.0] = volume.voxels.id();
    }
}

/// The focus, for occluder reach.
pub(crate) type Focus<'w, 's> =
    Extract<'w, 's, Query<'static, 'static, &'static ProbeGi, With<Camera3d>>>;
