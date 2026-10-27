use crate::records::POSES_BINDING;
use bevy_core_pipeline::prepass::{MOTION_VECTOR_PREPASS_FORMAT, NORMAL_PREPASS_FORMAT};
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res};
use bevy_pbr::CORE_3D_SHADOW_MAP_FORMAT;
use bevy_render::render_resource::*;
use bevy_render::renderer::RenderDevice;
use bevy_utils::default;

const COMMON: &str = include_str!("common.wgsl");
const READ: &str = include_str!("read.wgsl");
const SKIN: &str = include_str!("skin.wgsl");
const CULL: &str = include_str!("cull.wgsl");
const HIERARCHY: &str = include_str!("hierarchy.wgsl");
const CLUSTERS: &str = include_str!("clusters.wgsl");
const SOFT: &str = include_str!("soft.wgsl");
const HARD: &str = include_str!("hard.wgsl");
const HZB: &str = include_str!("hzb.wgsl");
const SEED: &str = include_str!("seed.wgsl");
const RESOLVE: &str = include_str!("resolve.wgsl");
/// Hardware raster depth format.
pub(crate) const RASTER_DEPTH: TextureFormat = TextureFormat::Depth32Float;
const VIEW_DEPTH: TextureFormat = TextureFormat::Depth32Float;
/// Traversal step immediate bytes.
const STEP_BYTES: u32 = 8;
/// Pyramid source extent bytes.
const SOURCE_BYTES: u32 = 8;
/// Format of the occlusion pyramid.
pub(crate) const PYRAMID: TextureFormat = TextureFormat::R32Float;
const ALL: ShaderStages = ShaderStages::COMPUTE
    .union(ShaderStages::MESH)
    .union(ShaderStages::FRAGMENT);

fn buffer(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    let ty = BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    BindGroupLayoutEntry {
        binding,
        visibility: ALL,
        ty,
        count: None,
    }
}

fn reads(binding: u32) -> BindGroupLayoutEntry {
    buffer(binding, BufferBindingType::Storage { read_only: true })
}

fn writes(binding: u32) -> BindGroupLayoutEntry {
    buffer(binding, BufferBindingType::Storage { read_only: false })
}

fn uniform(binding: u32) -> BindGroupLayoutEntry {
    buffer(binding, BufferBindingType::Uniform)
}

fn pyramid_read(binding: u32) -> BindGroupLayoutEntry {
    let ty = BindingType::Texture {
        sample_type: TextureSampleType::Float { filterable: false },
        view_dimension: TextureViewDimension::D2,
        multisampled: false,
    };
    BindGroupLayoutEntry {
        binding,
        visibility: ALL,
        ty,
        count: None,
    }
}

fn pyramid_write(binding: u32) -> BindGroupLayoutEntry {
    let ty = BindingType::StorageTexture {
        access: StorageTextureAccess::WriteOnly,
        format: PYRAMID,
        view_dimension: TextureViewDimension::D2,
    };
    BindGroupLayoutEntry {
        binding,
        visibility: ALL,
        ty,
        count: None,
    }
}

/// Geometry pipelines and layouts.
#[derive(Resource)]
pub(crate) struct Pipelines {
    pub library: BindGroupLayout,
    pub cull_view: BindGroupLayout,
    pub cull_args: BindGroupLayout,
    /// Next-level arguments a cull writes.
    pub toward: BindGroupLayout,
    pub raster_view: BindGroupLayout,
    pub resolve_view: BindGroupLayout,
    pub target: BindGroupLayout,
    pub blank: BindGroupLayout,
    pub pyramid: BindGroupLayout,
    /// Bound where no pyramid exists.
    pub unused: TextureView,
    pub seed: ComputePipeline,
    pub pyramid_first: ComputePipeline,
    pub pyramid_down: ComputePipeline,
    pub prepare_late: ComputePipeline,
    pub seed_late: ComputePipeline,
    pub cull_instances: ComputePipeline,
    pub cull_nodes: ComputePipeline,
    pub prepare_clusters: ComputePipeline,
    pub cull_clusters: ComputePipeline,
    pub prepare_raster: ComputePipeline,
    pub soft: ComputePipeline,
    pub hard: RenderPipeline,
    /// Depth-only raster into sun shadows.
    pub cast: RenderPipeline,
    pub resolve: RenderPipeline,
}

struct Modules {
    seed: ShaderModule,
    cull: ShaderModule,
    pyramid: ShaderModule,
    soft: ShaderModule,
    hard: ShaderModule,
    resolve: ShaderModule,
}

fn module(device: &RenderDevice, label: &str, parts: &[&str]) -> ShaderModule {
    let source = ShaderSource::Wgsl(parts.concat().into());
    device
        .wgpu_device()
        .create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source,
        })
}

fn compute(
    device: &RenderDevice,
    (layout, module): (&PipelineLayout, &ShaderModule),
    entry: &str,
) -> ComputePipeline {
    specialized(device, (layout, module), (entry, &[]))
}

fn specialized(
    device: &RenderDevice,
    (layout, module): (&PipelineLayout, &ShaderModule),
    (entry, constants): (&str, &[(&str, f64)]),
) -> ComputePipeline {
    device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions {
            constants,
            ..default()
        },
        cache: None,
    })
}

fn layout(
    device: &RenderDevice,
    groups: &[Option<&BindGroupLayout>],
    immediate_size: u32,
) -> PipelineLayout {
    let bind_group_layouts: Vec<_> = groups.iter().map(|g| g.map(|l| &**l)).collect();
    device
        .wgpu_device()
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &bind_group_layouts,
            immediate_size,
        })
}

fn depth(format: TextureFormat) -> DepthStencilState {
    DepthStencilState {
        format,
        depth_write_enabled: Some(true),
        depth_compare: Some(CompareFunction::GreaterEqual),
        stencil: default(),
        bias: default(),
    }
}

/// Mesh-shader raster, hardware list.
fn mesh(
    device: &RenderDevice,
    (layout, module): (&PipelineLayout, &ShaderModule),
    (primitive, format, fragment): (PrimitiveState, TextureFormat, Option<RawFragmentState>),
) -> RenderPipeline {
    let raw = device
        .wgpu_device()
        .create_mesh_pipeline(&RawMeshPipelineDescriptor {
            label: Some("virtual_geometry_hard"),
            layout: Some(layout),
            task: None,
            mesh: RawMeshState {
                module,
                entry_point: Some("raster_hard"),
                compilation_options: default(),
            },
            primitive,
            depth_stencil: Some(depth(format)),
            multisample: default(),
            fragment,
            multiview: None,
            cache: None,
        });
    RenderPipeline::from(raw)
}

fn resolve(
    device: &RenderDevice,
    layout: &PipelineLayout,
    module: &ShaderModule,
) -> RenderPipeline {
    let target = |format| {
        Some(ColorTargetState {
            format,
            blend: None,
            write_mask: ColorWrites::ALL,
        })
    };
    device.create_render_pipeline(&RawRenderPipelineDescriptor {
        label: Some("virtual_geometry_resolve"),
        layout: Some(layout),
        vertex: RawVertexState {
            module,
            entry_point: Some("fullscreen"),
            compilation_options: default(),
            buffers: &[],
        },
        primitive: default(),
        depth_stencil: Some(depth(VIEW_DEPTH)),
        multisample: default(),
        fragment: Some(RawFragmentState {
            module,
            entry_point: Some("resolve"),
            compilation_options: default(),
            targets: &[
                target(NORMAL_PREPASS_FORMAT),
                target(MOTION_VECTOR_PREPASS_FORMAT),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// One-texel pyramid stand-in.
fn unused(device: &RenderDevice) -> TextureView {
    let size = Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let usage = TextureUsages::TEXTURE_BINDING;
    let descriptor = TextureDescriptor {
        label: Some("virtual_geometry_unused"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: PYRAMID,
        usage,
        view_formats: &[],
    };
    device
        .create_texture(&descriptor)
        .create_view(&TextureViewDescriptor::default())
}

pub(crate) fn setup_pipelines(mut commands: Commands, device: Res<RenderDevice>) {
    let library = device.create_bind_group_layout(
        "virtual_geometry_library",
        &(0..9).chain([POSES_BINDING]).map(reads).collect::<Vec<_>>(),
    );
    let cull_entries = [
        uniform(0),
        writes(1),
        writes(2),
        writes(3),
        writes(4),
        writes(5),
        pyramid_read(6),
        writes(7),
        writes(8),
        writes(9),
    ];
    let cull_view = device.create_bind_group_layout("virtual_geometry_cull_view", &cull_entries);
    let pyramid = device.create_bind_group_layout(
        "virtual_geometry_pyramid",
        &[reads(0), pyramid_read(1), pyramid_write(2)],
    );
    let cull_args = device.create_bind_group_layout(
        "virtual_geometry_cull_args",
        &[writes(0), writes(1), writes(2), writes(4)],
    );
    let toward = device.create_bind_group_layout("virtual_geometry_toward", &[writes(3)]);
    let read_view = [uniform(0), reads(1), reads(4), reads(5)];
    let raster_view = device.create_bind_group_layout("virtual_geometry_raster_view", &read_view);
    let resolve_entries = [&read_view[..], &[reads(7)]].concat();
    let resolve_view =
        device.create_bind_group_layout("virtual_geometry_resolve_view", &resolve_entries);
    let target = device.create_bind_group_layout("virtual_geometry_target", &[writes(0)]);
    let scene = BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let blank = device.create_bind_group_layout("virtual_geometry_blank", &[writes(0), scene]);
    let modules = Modules {
        seed: module(&device, "virtual_geometry_seed", &[COMMON, SEED]),
        cull: module(
            &device,
            "virtual_geometry_cull",
            &[CULL, HIERARCHY, CLUSTERS, COMMON],
        ),
        pyramid: module(&device, "virtual_geometry_pyramid", &[COMMON, HZB]),
        soft: module(
            &device,
            "virtual_geometry_soft",
            &[SOFT, COMMON, READ, SKIN],
        ),
        hard: module(
            &device,
            "virtual_geometry_hard",
            &[HARD, COMMON, READ, SKIN],
        ),
        resolve: module(
            &device,
            "virtual_geometry_resolve",
            &[RESOLVE, COMMON, READ, SKIN],
        ),
    };
    let work = layout(
        &device,
        &[None, Some(&toward), Some(&library), Some(&cull_view)],
        STEP_BYTES,
    );
    let plain = layout(
        &device,
        &[None, None, Some(&library), Some(&cull_view)],
        STEP_BYTES,
    );
    let prepare = layout(
        &device,
        &[None, Some(&cull_args), Some(&library), Some(&cull_view)],
        STEP_BYTES,
    );
    let raster = layout(
        &device,
        &[Some(&target), None, Some(&library), Some(&raster_view)],
        0,
    );
    let casting = layout(
        &device,
        &[None, None, Some(&library), Some(&raster_view)],
        0,
    );
    let cleared = layout(&device, &[Some(&blank)], SOURCE_BYTES);
    let visible = RawFragmentState {
        module: &modules.hard,
        entry_point: Some("raster_hard_fragment"),
        compilation_options: default(),
        targets: &[],
    };
    let built = layout(&device, &[Some(&pyramid)], SOURCE_BYTES);
    let resolved = layout(
        &device,
        &[None, None, Some(&library), Some(&resolve_view)],
        0,
    );
    let (w, p, plain) = (
        (&work, &modules.cull),
        (&prepare, &modules.cull),
        (&plain, &modules.cull),
    );
    commands.insert_resource(Pipelines {
        seed: compute(&device, (&cleared, &modules.seed), "seed"),
        pyramid_first: specialized(
            &device,
            (&built, &modules.pyramid),
            ("hzb", &[("FIRST", 1.0)]),
        ),
        pyramid_down: specialized(
            &device,
            (&built, &modules.pyramid),
            ("hzb", &[("FIRST", 0.0)]),
        ),
        prepare_late: compute(&device, p, "prepare_late"),
        seed_late: compute(&device, plain, "seed_late"),
        cull_instances: compute(&device, w, "cull_instances"),
        cull_nodes: compute(&device, w, "cull_nodes"),
        prepare_clusters: compute(&device, p, "prepare_clusters"),
        cull_clusters: compute(&device, plain, "cull_clusters"),
        prepare_raster: compute(&device, p, "prepare_raster"),
        soft: compute(&device, (&raster, &modules.soft), "raster_soft"),
        hard: mesh(
            &device,
            (&raster, &modules.hard),
            (default(), RASTER_DEPTH, Some(visible)),
        ),
        cast: mesh(
            &device,
            (&casting, &modules.hard),
            (
                PrimitiveState {
                    unclipped_depth: true,
                    ..default()
                },
                CORE_3D_SHADOW_MAP_FORMAT,
                None,
            ),
        ),
        resolve: resolve(&device, &resolved, &modules.resolve),
        library,
        cull_view,
        cull_args,
        toward,
        raster_view,
        resolve_view,
        unused: unused(&device),
        target,
        blank,
        pyramid,
    });
}
