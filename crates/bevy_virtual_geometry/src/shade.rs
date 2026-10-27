use crate::instances::Instances;
use crate::layers::{LayerTable, VirtualLayers};
use crate::library::Library;
use crate::records::POSES_BINDING;
use crate::view::{entries, ViewGeometry};
use bevy_asset::Handle;
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Component;
use bevy_ecs::query::With;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Local, Query, Res, ResMut};
use bevy_mesh::{
    Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayout, MeshVertexBufferLayoutRef,
    MeshVertexBufferLayouts, VertexBufferLayout,
};
use bevy_pbr::{MeshPipeline, MeshPipelineKey, MeshViewBindGroup, ViewKeyCache};
use bevy_platform::collections::HashMap;
use bevy_render::diagnostic::RecordDiagnostics;
use bevy_render::render_asset::RenderAssets;
use bevy_render::render_resource::*;
use bevy_render::renderer::{RenderContext, RenderDevice, ViewQuery};
use bevy_render::texture::{FallbackImage, GpuImage};
use bevy_render::view::{ExtractedView, ViewDepthStencilTexture, ViewTarget};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_utils::default;

/// Texture slots in the shading pass.
const TEXTURES: usize = 512;
const FULLSCREEN: u32 = 3;
const ANISOTROPY: u16 = 16;
/// Layer constants binding.
const LAYERS_BINDING: u32 = 14;
const LIGHTING: &str = include_str!("lighting.wesl");
const LOOK: &str = include_str!("look.wesl");
const VISUALIZE: &str = include_str!("visualize.wgsl");
const COMMON: &str = include_str!("common.wgsl");
const READ: &str = include_str!("read.wgsl");
const SKIN: &str = include_str!("skin.wgsl");

/// Lighting plus shared shader code.
pub(crate) fn shader() -> Shader {
    Shader::from_wesl(
        [LIGHTING, LOOK, VISUALIZE, COMMON, READ, SKIN].concat(),
        "bevy_virtual_geometry/lighting.wesl",
    )
}

#[derive(Resource)]
pub(crate) struct Shading {
    shader: Handle<Shader>,
    library: BindGroupLayoutDescriptor,
    view: BindGroupLayoutDescriptor,
    sampler: Sampler,
}

fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(
    binding: u32,
    view_dimension: TextureViewDimension,
    count: Option<u32>,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension,
            multisampled: false,
        },
        count: count.and_then(core::num::NonZeroU32::new),
    }
}

fn library_entries() -> Vec<BindGroupLayoutEntry> {
    let read = BufferBindingType::Storage { read_only: true };
    let mut library: Vec<_> = (0..9)
        .chain([POSES_BINDING, LAYERS_BINDING])
        .map(|b| buffer_entry(b, read))
        .collect();
    library.push(texture_entry(
        9,
        TextureViewDimension::D2,
        Some(TEXTURES as u32),
    ));
    library.push(BindGroupLayoutEntry {
        binding: 10,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    });
    library.extend([11, 12].map(|b| texture_entry(b, TextureViewDimension::D2Array, None)));
    library
}

pub(crate) fn setup_shading(
    mut commands: Commands,
    device: Res<RenderDevice>,
    shader: Res<ShaderHandle>,
) {
    let read = BufferBindingType::Storage { read_only: true };
    let view = [
        buffer_entry(0, BufferBindingType::Uniform),
        buffer_entry(1, read),
        buffer_entry(4, read),
        buffer_entry(5, read),
        buffer_entry(7, read),
    ];
    let sampler = device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        anisotropy_clamp: ANISOTROPY,
        ..default()
    });
    commands.insert_resource(Shading {
        shader: shader.0.clone(),
        library: BindGroupLayoutDescriptor::new(
            "virtual_geometry_shade_library",
            &library_entries(),
        ),
        view: BindGroupLayoutDescriptor::new("virtual_geometry_shade_view", &view),
        sampler,
    });
}

/// Shader handle from the main world.
#[derive(Resource)]
pub(crate) struct ShaderHandle(pub Handle<Shader>);

/// Shading pipeline of one view.
#[derive(Component)]
pub(crate) struct ShadePipeline(CachedRenderPipelineId);

/// Library bindings, rebuilt on change.
#[derive(Resource, Default)]
pub(crate) struct ShadeLibrary {
    group: Option<BindGroup>,
    /// Library, instance, layer table and ready textures it was built with.
    built: (u32, u32, u32, usize),
}

/// Positions, normals and uvs: what the mesh pipeline expects.
fn stand_in_layout(layouts: &mut MeshVertexBufferLayouts) -> MeshVertexBufferLayoutRef {
    let attribute = |format, offset, shader_location| VertexAttribute {
        format,
        offset,
        shader_location,
    };
    let attributes = vec![
        attribute(VertexFormat::Float32x3, 0, 0),
        attribute(VertexFormat::Float32x3, 12, 1),
        attribute(VertexFormat::Float32x2, 24, 2),
    ];
    let ids = vec![
        Mesh::ATTRIBUTE_POSITION.id,
        Mesh::ATTRIBUTE_NORMAL.id,
        Mesh::ATTRIBUTE_UV_0.id,
    ];
    layouts.insert(MeshVertexBufferLayout::new(
        ids,
        VertexBufferLayout {
            array_stride: 32,
            step_mode: VertexStepMode::Vertex,
            attributes,
        },
        MeshAttributeCompressionFlags::empty(),
    ))
}

fn descriptor(
    shading: &Shading,
    mesh: &MeshPipeline,
    key: MeshPipelineKey,
    layout: &MeshVertexBufferLayoutRef,
) -> Option<RenderPipelineDescriptor> {
    let base = mesh.specialize(key, layout).ok()?;
    let fragment = base.fragment?;
    let mut shader_defs: Vec<ShaderDefVal> = fragment.shader_defs;
    shader_defs.push("STANDARD_MATERIAL_DIFFUSE_TRANSMISSION".into());
    let view = mesh.get_view_layout(key.into());
    let depth = DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: Some(false),
        depth_compare: Some(CompareFunction::Equal),
        stencil: default(),
        bias: default(),
    };
    Some(RenderPipelineDescriptor {
        label: Some("virtual_geometry_shade".into()),
        layout: vec![
            view.main_layout,
            view.binding_array_layout,
            shading.library.clone(),
            shading.view.clone(),
        ],
        vertex: VertexState {
            shader: shading.shader.clone(),
            shader_defs: shader_defs.clone(),
            entry_point: Some("fullscreen".into()),
            buffers: vec![],
            constants: vec![],
        },
        depth_stencil: Some(depth),
        fragment: Some(FragmentState {
            shader: shading.shader.clone(),
            shader_defs,
            entry_point: Some("shade".into()),
            targets: fragment.targets,
            constants: vec![],
        }),
        ..default()
    })
}

pub(crate) fn queue_shading(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedView), With<ViewGeometry>>,
    keys: Res<ViewKeyCache>,
    (shading, mesh, cache): (Res<Shading>, Res<MeshPipeline>, Res<PipelineCache>),
    mut layouts: ResMut<MeshVertexBufferLayouts>,
    mut known: Local<HashMap<MeshPipelineKey, CachedRenderPipelineId>>,
) {
    let layout = stand_in_layout(&mut layouts);
    for (entity, view) in &views {
        let Some(&key) = keys.get(&view.retained_view_entity) else {
            continue;
        };
        let id = match known.get(&key) {
            Some(id) => *id,
            None => {
                let Some(d) = descriptor(&shading, &mesh, key, &layout) else {
                    continue;
                };
                let id = cache.queue_render_pipeline(d);
                known.insert(key, id);
                id
            }
        };
        commands.entity(entity).insert(ShadePipeline(id));
    }
}

type Sources<'w> = (
    Res<'w, Library>,
    Res<'w, Instances>,
    Res<'w, RenderAssets<GpuImage>>,
    Res<'w, FallbackImage>,
);

pub(crate) fn prepare_shade_library(
    mut shade: ResMut<ShadeLibrary>,
    (shading, layers, table): (Res<Shading>, Option<Res<VirtualLayers>>, Res<LayerTable>),
    (library, instances, images, fallback): Sources,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    let Some(b) = library.buffers.as_ref() else {
        return;
    };
    let arrays: Vec<&GpuImage> = layers
        .iter()
        .flat_map(|l| [images.get(l.albedo.id()), images.get(l.surface.id())])
        .flatten()
        .collect();
    let ready = library
        .textures
        .iter()
        .filter(|id| images.get(**id).is_some())
        .count()
        + arrays.len();
    let built = (
        library.generation,
        instances.generation,
        table.generation,
        ready,
    );
    if shade.group.is_some() && shade.built == built {
        return;
    }
    let [albedo, surface] = match arrays[..] {
        [albedo, surface] => [albedo, surface],
        _ => [&fallback.d2_array; 2],
    };
    let textures = library
        .textures
        .iter()
        .map(|id| images.get(*id).unwrap_or(&fallback.d2));
    let views: Vec<&WgpuTextureView> = textures
        .chain(core::iter::repeat(&fallback.d2))
        .take(TEXTURES)
        .map(|i| &*i.texture_view)
        .collect();
    let pairs = [
        (0, b.positions.as_entire_binding()),
        (1, b.vertices.as_entire_binding()),
        (2, b.triangles.as_entire_binding()),
        (3, b.clusters.as_entire_binding()),
        (4, b.nodes.as_entire_binding()),
        (5, b.children.as_entire_binding()),
        (6, b.parts.as_entire_binding()),
        (7, instances.buffer.as_entire_binding()),
        (8, b.materials.as_entire_binding()),
        (9, BindingResource::TextureViewArray(&views)),
        (10, BindingResource::Sampler(&shading.sampler)),
        (11, BindingResource::TextureView(&albedo.texture_view)),
        (12, BindingResource::TextureView(&surface.texture_view)),
        (POSES_BINDING, b.poses.as_entire_binding()),
        (LAYERS_BINDING, table.buffer.as_entire_binding()),
    ];
    let layout = cache.get_bind_group_layout(&shading.library);
    shade.group =
        Some(device.create_bind_group("virtual_geometry_shade_library", &layout, &entries(&pairs)));
    shade.built = built;
}

type Inputs = (
    &'static ViewTarget,
    &'static ViewDepthStencilTexture,
    &'static MeshViewBindGroup,
    &'static ShadePipeline,
    &'static ViewGeometry,
);

/// Lights virtual geometry before opaque.
pub(crate) fn shade_geometry(
    view: ViewQuery<Inputs>,
    (shade, shading): (Res<ShadeLibrary>, Res<Shading>),
    (cache, device): (Res<PipelineCache>, Res<RenderDevice>),
    mut ctx: RenderContext,
) {
    let (target, depth, mesh_view, pipeline, geometry) = view.into_inner();
    let (Some(library), Some(pipeline), Some(read), Some(screen)) = (
        shade.group.as_ref(),
        cache.get_render_pipeline(pipeline.0),
        geometry.read_entries(),
        geometry.work.screen.as_ref(),
    ) else {
        return;
    };
    let view_group = device.create_bind_group(
        "virtual_geometry_shade_view",
        &cache.get_bind_group_layout(&shading.view),
        &entries(&read),
    );
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("virtual_geometry_shade"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let span = diagnostics.pass_span(&mut pass, "virtual_geometry_shade");
    pass.set_viewport(
        0.0,
        0.0,
        screen.size.x as f32,
        screen.size.y as f32,
        0.0,
        1.0,
    );
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &mesh_view.main, &mesh_view.main_offsets);
    pass.set_bind_group(1, &mesh_view.binding_array, &[]);
    pass.set_bind_group(2, library, &[]);
    pass.set_bind_group(3, &view_group, &[]);
    pass.draw(0..FULLSCREEN, 0..1);
    span.end(&mut pass);
}
