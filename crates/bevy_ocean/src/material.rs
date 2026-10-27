use crate::buoyancy::Floating;
use crate::spectrum;
use crate::spectrum::SIDE;
use crate::waves::{BANDS, CASCADES};
use crate::{Ocean, OceanSeabed};
use bevy_asset::{Asset, Assets, Handle, RenderAssetUsages};
use bevy_ecs::change_detection::DetectChanges;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Local, Res};
use bevy_image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy_math::Vec4;
use bevy_pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy_reflect::TypePath;
use bevy_render::extract_resource::ExtractResource;
use bevy_render::render_resource::encase::StorageBuffer;
use bevy_render::render_resource::{
    AsBindGroup, Buffer, BufferDescriptor, BufferUsages, Extent3d, ShaderType, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_render::RenderApp;
use bevy_shader::ShaderRef;
use bevy_transform::components::Transform;
use bevy_utils::default;

/// Water seen through, lit and reflected.
pub type OceanMaterial = ExtendedMaterial<StandardMaterial, OceanSurface>;

const SHADER: &str = "embedded://bevy_ocean/ocean.wesl";
/// Hulls the sea keeps out.
pub(crate) const MAX_HULLS: usize = 16;
const TEXEL_BYTES: usize = 8;

/// Hull frame and its planform.
#[derive(ShaderType, Clone, Copy, Default, Debug)]
pub struct Hull {
    /// World to hull, three rows.
    rows: [Vec4; 3],
    /// Bow, stern and beam extents.
    outline: Vec4,
}

impl Hull {
    pub(crate) fn of(transform: &Transform, floating: &Floating) -> Hull {
        let into = transform.to_matrix().inverse();
        Hull {
            rows: [into.row(0), into.row(1), into.row(2)],
            outline: floating.planform,
        }
    }
}

/// Per-frame sea inputs.
#[derive(ShaderType, Clone, Debug)]
pub struct OceanState {
    /// Vertex share per cascade; hull count.
    pub(crate) reach: Vec4,
    /// Slope variance per cascade.
    variance: Vec4,
    /// Seabed corner, inverse extent, mapped.
    pub(crate) seabed: Vec4,
    /// Calm level.
    pub(crate) level: Vec4,
    pub(crate) hulls: [Hull; MAX_HULLS],
}

/// Wave fields and seabed inputs.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct OceanSurface {
    /// Per-frame inputs, uploaded separately.
    #[storage(100, read_only, buffer)]
    state: Buffer,
    /// Displacement and foam per cascade.
    #[texture(101, dimension = "2d_array")]
    #[sampler(102)]
    pub(crate) shape: Handle<Image>,
    /// Slopes and stretch per cascade.
    #[texture(103, dimension = "2d_array")]
    pub(crate) slope: Handle<Image>,
    /// Seabed height and slope.
    #[texture(104)]
    #[sampler(105)]
    seabed: Handle<Image>,
}

impl MaterialExtension for OceanSurface {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    fn reads_view_transmission_texture() -> bool {
        true
    }
}

/// Half-float cascade field array.
fn field(images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: SIDE as u32,
        height: SIDE as u32,
        depth_or_array_layers: CASCADES as u32,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0; TEXEL_BYTES],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..ImageSamplerDescriptor::linear()
    });
    images.add(image)
}

/// Per-frame sea inputs and their GPU copy.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct OceanFrame {
    pub(crate) state: OceanState,
    buffer: Buffer,
}

impl OceanSurface {
    pub(crate) fn new(
        (images, device): (&mut Assets<Image>, &RenderDevice),
        ocean: &Ocean,
        seabed: &OceanSeabed,
    ) -> (OceanSurface, OceanFrame) {
        let variance = BANDS.map(|b| spectrum::slope_variance(&b));
        let state = OceanState {
            reach: Vec4::ZERO,
            variance: Vec4::new(variance[0], variance[1], variance[2], 0.0),
            seabed: seabed.frame(),
            level: Vec4::new(ocean.level, 0.0, 0.0, 0.0),
            hulls: [Hull::default(); MAX_HULLS],
        };
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ocean_state"),
            size: OceanState::min_size().get(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface = OceanSurface {
            state: buffer.clone(),
            shape: field(images),
            slope: field(images),
            seabed: seabed.heights.clone(),
        };
        (surface, OceanFrame { state, buffer })
    }
}

/// Uploads changed sea inputs.
pub(crate) fn upload_frame(
    frame: Res<OceanFrame>,
    queue: Res<RenderQueue>,
    mut bytes: Local<Vec<u8>>,
) {
    if !frame.is_changed() {
        return;
    }
    let mut staged = StorageBuffer::new(core::mem::take(&mut *bytes));
    staged.write(&frame.state).unwrap();
    queue.write_buffer(&frame.buffer, 0, staged.as_ref());
    *bytes = staged.into_inner();
}
