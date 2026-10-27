use crate::spectrum::{self, LOOP, SIDE};
use crate::waves::{BANDS, CASCADES};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::change_detection::DetectChanges;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res, ResMut};
use bevy_image::Image;
use bevy_math::{ops, Vec2, Vec4};
use bevy_render::diagnostic::RecordDiagnostics;
use bevy_render::extract_resource::ExtractResource;
use bevy_render::render_asset::RenderAssets;
use bevy_render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_2d_array, uniform_buffer,
};
use bevy_render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
    BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
    StorageTextureAccess, TextureFormat, UniformBuffer,
};
use bevy_render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy_render::texture::GpuImage;
use bevy_render::RenderApp;
use bevy_time::Time;
use bevy_utils::default;

const POINTS: usize = SIDE * SIDE;
/// Field pairs per cascade, as fft.wesl.
const PAIRS: usize = 4;
/// Seconds foam takes to fade.
const FOAM_LIFE: f32 = 3.0;
const WORKGROUP: u32 = 64;

/// Sea time for one step.
#[derive(ShaderType, Clone, Copy, Default, PartialEq)]
struct Clock {
    /// Seconds into the loop.
    time: f32,
    /// Foam kept since the last step.
    fade: f32,
}

/// What the GPU steps this frame, and where it writes.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub(crate) struct WaveStep {
    clock: Clock,
    active: bool,
    /// Displacement and foam per cascade.
    pub(crate) shape: Handle<Image>,
    /// Slopes and stretch per cascade.
    slope: Handle<Image>,
}

impl WaveStep {
    pub(crate) fn new(shape: Handle<Image>, slope: Handle<Image>) -> WaveStep {
        WaveStep {
            clock: Clock::default(),
            active: false,
            shape,
            slope,
        }
    }

    /// Advances the clock while waves are needed.
    pub(crate) fn tick(&mut self, time: &Time, active: bool) {
        self.active = active;
        if active {
            self.clock = Clock {
                time: (time.elapsed_secs_f64() % f64::from(LOOP)) as f32,
                fade: ops::exp(-time.delta_secs() / FOAM_LIFE),
            };
        }
    }
}

/// Spectrum, fields and pipelines of the GPU sea.
#[derive(Resource)]
pub(crate) struct WaveFft {
    layout: BindGroupLayoutDescriptor,
    /// Spectra, rows, columns, unpack.
    stages: [CachedComputePipelineId; 4],
    clock: UniformBuffer<Clock>,
    buffers: [Buffer; 4],
    bind_group: Option<BindGroup>,
}

/// Start amplitudes with their mirrors' conjugates, and wave tables, per slot.
fn tables() -> (Vec<Vec4>, Vec<Vec4>) {
    let mirror = |slot: usize| (SIDE - slot % SIDE) % SIDE + (SIDE - slot / SIDE) % SIDE * SIDE;
    let mut starts = Vec::with_capacity(CASCADES * POINTS);
    let mut waves = Vec::with_capacity(CASCADES * POINTS);
    for (band, cascade) in BANDS.iter().zip(0..) {
        let h0 = spectrum::amplitudes(band, cascade);
        starts.extend((0..POINTS).map(|s| h0[s].extend(h0[mirror(s)].x).extend(-h0[mirror(s)].y)));
        waves.extend((0..POINTS).map(|slot| {
            let k = spectrum::wavevector(band, slot);
            let len = k.length();
            let inverse = if len > 0.0 { 1.0 / len } else { 0.0 };
            k.extend(inverse).extend(spectrum::frequency(len))
        }));
    }
    (starts, waves)
}

fn storage(device: &RenderDevice, label: &'static str, data: &[Vec4]) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::STORAGE,
    })
}

fn scratch(device: &RenderDevice, label: &'static str, size: usize) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

pub(crate) fn setup_fft(
    mut commands: Commands,
    (device, cache, assets): (Res<RenderDevice>, Res<PipelineCache>, Res<AssetServer>),
) {
    let layout = BindGroupLayoutDescriptor::new(
        "ocean_fft",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<Clock>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
            ),
        ),
    );
    let shader = load_embedded_asset!(assets.as_ref(), "fft.wesl");
    let stages = ["spectra", "rows", "columns", "unpack"].map(|entry| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("ocean_fft_{entry}").into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    });
    let (starts, waves) = tables();
    let buffers = [
        storage(&device, "ocean_starts", &starts),
        storage(&device, "ocean_waves", &waves),
        scratch(
            &device,
            "ocean_fields",
            CASCADES * PAIRS * POINTS * size_of::<Vec2>(),
        ),
        scratch(&device, "ocean_foam", CASCADES * POINTS * size_of::<f32>()),
    ];
    commands.insert_resource(WaveFft {
        layout,
        stages,
        clock: UniformBuffer::default(),
        buffers,
        bind_group: None,
    });
}

/// Uploads the clock; binds the fields once their images exist.
pub(crate) fn prepare_fft(
    step: Option<Res<WaveStep>>,
    mut fft: ResMut<WaveFft>,
    (images, cache): (Res<RenderAssets<GpuImage>>, Res<PipelineCache>),
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
) {
    let Some(step) = step else { return };
    let fft = &mut *fft;
    if step.is_changed() {
        fft.clock.set(step.clock);
        fft.clock.write_buffer(&device, &queue);
    }
    if fft.bind_group.is_some() {
        return;
    }
    let (Some(shape), Some(slope), Some(clock)) = (
        images.get(&step.shape),
        images.get(&step.slope),
        fft.clock.binding(),
    ) else {
        return;
    };
    let [starts, waves, fields, foam] = &fft.buffers;
    fft.bind_group = Some(device.create_bind_group(
        "ocean_fft",
        &cache.get_bind_group_layout(&fft.layout),
        &BindGroupEntries::sequential((
            clock,
            starts.as_entire_binding(),
            waves.as_entire_binding(),
            fields.as_entire_binding(),
            foam.as_entire_binding(),
            &shape.texture_view,
            &slope.texture_view,
        )),
    ));
}

/// Steps the sea on the GPU.
pub(crate) fn run_fft(
    step: Option<Res<WaveStep>>,
    (fft, cache): (Res<WaveFft>, Res<PipelineCache>),
    mut ctx: RenderContext,
) {
    if !step.is_some_and(|s| s.active) {
        return;
    }
    let Some(bind_group) = &fft.bind_group else {
        return;
    };
    let [Some(spectra), Some(rows), Some(columns), Some(unpack)] =
        fft.stages.map(|id| cache.get_compute_pipeline(id))
    else {
        return;
    };
    let points = (CASCADES * POINTS) as u32 / WORKGROUP;
    let lines = (SIDE as u32, PAIRS as u32, CASCADES as u32);
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("ocean_fft"),
            timestamp_writes: None,
        });
    let span = diagnostics.pass_span(&mut pass, "ocean_fft");
    pass.set_bind_group(0, bind_group, &[]);
    for (pipeline, groups) in [
        (spectra, (points, 1, 1)),
        (rows, lines),
        (columns, lines),
        (unpack, (points, 1, 1)),
    ] {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
    span.end(&mut pass);
}
