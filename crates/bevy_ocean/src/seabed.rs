use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_ecs::resource::Resource;
use bevy_ecs::world::{FromWorld, World};
use bevy_image::{Image, ImageSampler, ImageSamplerDescriptor};
use bevy_math::{UVec2, Vec2, Vec4};
use bevy_render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use half::f16;

/// Texels per seabed chart side.
pub const SEABED_SIDE: u32 = 512;
const TEXEL_BYTES: usize = 8;

/// Slope smoothing reach, texels.
const SLOPE_BLUR: usize = 16;

/// One box pass along an axis.
fn box_pass(values: &mut [f32], side: usize, (along, across): (usize, usize)) {
    let mut line = vec![0.0f32; side];
    let reach = SLOPE_BLUR as isize;
    let width = (2 * SLOPE_BLUR + 1) as f32;
    for l in 0..side {
        line.iter_mut()
            .enumerate()
            .for_each(|(k, x)| *x = values[l * across + k * along]);
        let at = |k: isize| line[k.clamp(0, side as isize - 1) as usize];
        let mut sum: f32 = (-reach..=reach).map(at).sum();
        for k in 0..side as isize {
            values[l * across + k as usize * along] = sum / width;
            sum += at(k + reach + 1) - at(k - reach);
        }
    }
}

/// Heights smoothed for swell bearings.
fn calmed(heights: &[f32], side: usize) -> Vec<f32> {
    let mut values = heights.to_vec();
    for _ in 0..2 {
        box_pass(&mut values, side, (1, side));
        box_pass(&mut values, side, (side, 1));
    }
    values
}

/// Seabed height and slope chart.
#[derive(Resource, Clone)]
pub struct OceanSeabed {
    pub(crate) heights: Handle<Image>,
    /// Texel zero corner; `f32::MAX` uncharted.
    pub(crate) origin: Vec2,
    /// Metres the chart spans.
    pub(crate) extent: f32,
}

/// Seabed chart encoded off-thread.
pub struct SeabedChart {
    texels: Vec<u8>,
    origin: Vec2,
    extent: f32,
}

impl SeabedChart {
    /// Charts square `heights` and sea `shares` spanning `extent`.
    pub fn new((heights, shares): (&[f32], &[f32]), origin: Vec2, extent: f32) -> SeabedChart {
        let side = SEABED_SIDE as usize;
        assert_eq!(
            heights.len(),
            side * side,
            "seabed chart must be SEABED_SIDE squared"
        );
        let calm = calmed(heights, side);
        let at = |x: usize, y: usize| calm[y.min(side - 1) * side + x.min(side - 1)];
        let span = 2.0 * extent / SEABED_SIDE as f32;
        let texels = (0..side * side)
            .flat_map(|i| {
                let (x, y) = (i % side, i / side);
                let dx = (at(x + 1, y) - at(x.saturating_sub(1), y)) / span;
                let dz = (at(x, y + 1) - at(x, y.saturating_sub(1))) / span;
                [heights[i], dx, dz, shares[i]].map(|v| f16::from_f32(v).to_bits())
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        SeabedChart {
            texels,
            origin,
            extent,
        }
    }
}

impl FromWorld for OceanSeabed {
    fn from_world(world: &mut World) -> OceanSeabed {
        let size = Extent3d {
            width: SEABED_SIDE,
            height: SEABED_SIDE,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0; TEXEL_BYTES],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::default(),
        );
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
        let heights = world.resource_mut::<Assets<Image>>().add(image);
        OceanSeabed {
            heights,
            origin: Vec2::splat(f32::MAX),
            extent: 1.0,
        }
    }
}

impl OceanSeabed {
    /// Follows render space when its origin moves.
    pub fn shift(&mut self, by: Vec2) {
        if self.origin.x < f32::MAX {
            self.origin -= by;
        }
    }

    /// Swaps in a new chart.
    pub fn install(&mut self, images: &mut Assets<Image>, chart: SeabedChart) {
        let Some(mut image) = images.get_mut(&self.heights) else {
            return;
        };
        image.data = Some(chart.texels);
        (self.origin, self.extent) = (chart.origin, chart.extent);
    }

    /// Corner, inverse extent, charted flag.
    pub(crate) fn frame(&self) -> Vec4 {
        let charted = self.origin.x < f32::MAX;
        self.origin
            .extend(1.0 / self.extent)
            .extend(if charted { 1.0 } else { 0.0 })
    }

    /// Charted ground height under `p`.
    pub fn height(&self, images: &Assets<Image>, p: Vec2) -> Option<f32> {
        let image = images.get(&self.heights)?;
        let data = image.data.as_deref()?;
        let side = UVec2::splat(SEABED_SIDE).as_vec2();
        let t = (p - self.origin) / self.extent * side - 0.5;
        if t.cmplt(Vec2::ZERO).any() || t.cmpge(side - 1.0).any() {
            return None;
        }
        let (base, f) = (t.floor(), t.fract());
        let at = |dx: f32, dy: f32| {
            let i = ((base.y + dy) * side.x + base.x + dx) as usize * TEXEL_BYTES;
            f16::from_le_bytes([data[i], data[i + 1]]).to_f32()
        };
        let bottom = at(0.0, 0.0) + (at(1.0, 0.0) - at(0.0, 0.0)) * f.x;
        let top = at(0.0, 1.0) + (at(1.0, 1.0) - at(0.0, 1.0)) * f.x;
        Some(bottom + (top - bottom) * f.y)
    }
}
