use crate::clipmap::{Clipmap, ProbeGi, CELLS, LEVELS, SPACING};
use crate::ground::{ProbeGround, GROUND_LAYERS};
use crate::scene::{Record, Scene};
use crate::ProbeSunlight;
use bevy_asset::{AssetId, Handle};
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Query, Res, ResMut, SystemParam};
use bevy_image::Image;
use bevy_light::{DirectionalLight, EnvironmentMapLight};
use bevy_math::{Vec3, Vec4};
use bevy_render::diagnostic::RecordDiagnostics;
use bevy_render::render_asset::RenderAssets;
use bevy_render::render_resource::binding_types::{
    acceleration_structure_vertex_return, sampler, storage_buffer_read_only, storage_buffer_sized,
    texture_2d, texture_2d_array, texture_cube, texture_storage_3d, uniform_buffer,
};
use bevy_render::render_resource::*;
use bevy_render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy_render::texture::GpuImage;
use bevy_render::Extract;
use bevy_shader::Shader;
use bevy_transform::components::GlobalTransform;
use bevy_utils::default;

const WORKGROUP: u32 = 64;
/// Per-frame levels, as probes.wesl.
const FINE_LEVELS: u32 = 2;
/// Fine levels plus one coarse.
const TRACED: u32 = CELLS.x * CELLS.y * CELLS.z * (FINE_LEVELS + 1);
/// Records of every level.
const PROBES: u32 = CELLS.x * CELLS.y * CELLS.z * LEVELS as u32;
/// Bytes per probe record.
const RECORD: u64 = 224;
const KERNEL: &str = include_str!("probes.wesl");
const TRACING: &str = include_str!("rays.wesl");

/// Probe kernel plus ray helpers.
pub(crate) fn shader() -> Shader {
    Shader::from_wesl([KERNEL, TRACING].concat(), "bevy_probe_gi/probes.wesl")
}

/// Probe shader from the main world.
#[derive(Resource)]
pub(crate) struct ProbeShader(pub Handle<Shader>);

#[derive(ShaderType, Clone, Copy, Default)]
struct Window {
    origin: Vec3,
    spacing: f32,
}

#[derive(ShaderType, Clone, Default)]
struct Params {
    windows: [Window; LEVELS],
    /// Toward sun; w is lux.
    sun: Vec4,
    sun_color: Vec3,
    sky_scale: f32,
    /// Ground corner, texel scale, side.
    ground: Vec4,
    /// Colour calibration per ground layer.
    tints: [Vec4; GROUND_LAYERS],
    frame: u32,
    /// Bumped when sun or sky change; settled probes wake.
    epoch: u32,
    /// Levels whose window moved since last traced.
    moved: u32,
}

/// Sun and sky, extracted.
#[derive(Resource, Default, Clone, PartialEq)]
pub(crate) struct Light {
    sun: Vec4,
    color: Vec3,
    /// None until the sky is lit.
    sky: Option<AssetId<Image>>,
    sky_scale: f32,
}

/// Device-side handles the tracer needs.
#[derive(SystemParam)]
pub(crate) struct Gpu<'w> {
    device: Res<'w, RenderDevice>,
    queue: Res<'w, RenderQueue>,
    cache: Res<'w, PipelineCache>,
    images: Res<'w, RenderAssets<GpuImage>>,
}

#[derive(Resource)]
pub(crate) struct Tracer {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    params: UniformBuffer<Params>,
    records: Buffer,
    bind_group: Option<BindGroup>,
    frame: u32,
    epoch: u32,
    lit: Light,
    /// Window origins as last traced.
    traced: [Vec3; LEVELS],
}

pub(crate) fn setup(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    shader: Res<ProbeShader>,
) {
    let voxels = || texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly);
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer::<Params>(false),
            storage_buffer_read_only::<Vec<Record>>(false),
            acceleration_structure_vertex_return(),
            storage_buffer_sized(false, None),
            voxels(),
            voxels(),
            voxels(),
            voxels(),
            texture_cube(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Uint),
            texture_2d_array(TextureSampleType::Float { filterable: true }),
        ),
    );
    let layout = BindGroupLayoutDescriptor::new("probe_gi", &entries);
    let descriptor = ComputePipelineDescriptor {
        label: Some("probe_gi".into()),
        layout: vec![layout.clone()],
        shader: shader.0.clone(),
        entry_point: Some("trace_probes".into()),
        ..default()
    };
    let pipeline = cache.queue_compute_pipeline(descriptor);
    let records = device.create_buffer(&BufferDescriptor {
        label: Some("probe_gi_records"),
        size: RECORD * u64::from(PROBES),
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    commands.insert_resource(Tracer {
        layout,
        pipeline,
        params: UniformBuffer::default(),
        records,
        bind_group: None,
        frame: 0,
        epoch: 0,
        lit: Light::default(),
        traced: [Vec3::NAN; LEVELS],
    });
}

/// The shadowing light and the sky the probe camera sees.
type Lights<'w, 's> = (
    Extract<'w, 's, Query<'static, 'static, (&'static DirectionalLight, &'static GlobalTransform)>>,
    Extract<
        'w,
        's,
        Query<'static, 'static, &'static EnvironmentMapLight, bevy_ecs::query::With<ProbeGi>>,
    >,
    Extract<'w, 's, Option<Res<'static, ProbeSunlight>>>,
);

pub(crate) fn extract_light(mut light: ResMut<Light>, (suns, sky, sunlight): Lights) {
    if let Some((d, t)) = suns.iter().find(|(d, _)| d.shadow_maps_enabled) {
        light.sun = t.back().extend(d.illuminance);
        let linear = d.color.to_linear();
        light.color = Vec3::new(linear.red, linear.green, linear.blue)
            * sunlight.as_ref().map_or(Vec3::ONE, |s| s.0);
    }
    if let Ok(env) = sky.single() {
        light.sky = Some(env.specular_map.id());
        light.sky_scale = env.intensity;
    }
}

/// Scene, windows, light, ground.
type Inputs<'w> = (
    Res<'w, Scene>,
    Res<'w, Clipmap>,
    Res<'w, Light>,
    Option<Res<'w, ProbeGround>>,
);

fn ground_params(ground: &ProbeGround) -> (Vec4, [Vec4; GROUND_LAYERS]) {
    let side = ground.texel.recip();
    let corner = ground.origin.extend(side).extend(0.0);
    (corner, ground.tints.map(|t| t.extend(0.0)))
}

pub(crate) fn prepare(
    mut tracer: ResMut<Tracer>,
    (scene, clipmap, light, ground): Inputs,
    gpu: Gpu,
) {
    tracer.bind_group = None;
    let (Some(tlas), Some(sky), Some(ground)) = (
        scene.tlas(),
        light.sky.and_then(|id| gpu.images.get(id)),
        ground,
    ) else {
        return;
    };
    let (Some(weights), Some(albedo)) = (
        gpu.images.get(ground.weights.id()),
        gpu.images.get(ground.layers.id()),
    ) else {
        return;
    };
    let views: Vec<&TextureView> = clipmap
        .images
        .iter()
        .filter_map(|id| gpu.images.get(*id))
        .map(|i| &i.texture_view)
        .collect();
    if views.len() != LEVELS {
        return;
    }
    let tracer = &mut *tracer;
    if tracer.lit != *light {
        tracer.lit = light.clone();
        tracer.epoch = tracer.epoch.wrapping_add(1);
    }
    tracer.frame = tracer.frame.wrapping_add(1);
    let coarse = FINE_LEVELS as usize + (tracer.frame & 1) as usize;
    let mut moved = 0;
    for level in (0..FINE_LEVELS as usize).chain([coarse]) {
        if tracer.traced[level] != clipmap.origins[level] {
            moved |= 1 << level;
            tracer.traced[level] = clipmap.origins[level];
        }
    }
    let windows = core::array::from_fn(|i| Window {
        origin: clipmap.origins[i],
        spacing: SPACING[i],
    });
    let (mut corner, tints) = ground_params(&ground);
    corner.w = weights.texture_descriptor.size.width as f32;
    let params = Params {
        windows,
        sun: light.sun,
        sun_color: light.color,
        sky_scale: light.sky_scale,
        ground: corner,
        tints,
        frame: tracer.frame,
        epoch: tracer.epoch,
        moved,
    };
    tracer.params.set(params);
    tracer.params.write_buffer(&gpu.device, &gpu.queue);
    let (Some(params), Some(records)) = (tracer.params.binding(), scene.records.binding()) else {
        return;
    };
    let entries = BindGroupEntries::sequential((
        params,
        records,
        BindingResource::AccelerationStructure(tlas),
        tracer.records.as_entire_binding(),
        views[0],
        views[1],
        views[2],
        views[3],
        &sky.texture_view,
        &sky.sampler,
        &weights.texture_view,
        &albedo.texture_view,
    ));
    tracer.bind_group = Some(gpu.device.create_bind_group(
        "probe_gi",
        &gpu.cache.get_bind_group_layout(&tracer.layout),
        &entries,
    ));
}

/// Traces this frame's share of probes.
pub(crate) fn trace(tracer: Res<Tracer>, cache: Res<PipelineCache>, mut ctx: RenderContext) {
    let (Some(bind_group), Some(pipeline)) = (
        tracer.bind_group.as_ref(),
        cache.get_compute_pipeline(tracer.pipeline),
    ) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "probe_gi");
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("probe_gi"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(TRACED / WORKGROUP, 1, 1);
    drop(pass);
    span.end(ctx.command_encoder());
}
