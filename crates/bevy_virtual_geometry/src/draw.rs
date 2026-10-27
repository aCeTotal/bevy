use crate::instances::Instances;
use crate::library::Library;
use crate::pipelines::Pipelines;
use crate::readout::Readout;
use crate::records::{MAX_LEVELS, SLOT_BYTES};
use crate::view::{entries, Groups, Screen, ScreenGroups, ViewGeometry, Work};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::system::{Res, ResMut};
use bevy_render::diagnostic::RecordDiagnostics;
use bevy_render::render_resource::*;
use bevy_render::renderer::{RenderContext, ViewQuery};
use bevy_render::view::ViewDepthStencilTexture;

const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65535;
/// Side of seed and pyramid tiles.
const TILE: u32 = 8;
/// Byte offsets of indirect arguments.
const CLUSTER_ARGS: u64 = 0;
const SOFT_ARGS: u64 = SLOT_BYTES;
pub(crate) const HARD_ARGS: u64 = 2 * SLOT_BYTES;
const SEED_ARGS: u64 = 3 * SLOT_BYTES;
/// Late-instance slot, odd steps.
const LATE_ARGS: u64 = (MAX_LEVELS as u64 - 1) * SLOT_BYTES;
/// Deepest level sparing late slot.
const DEEPEST: u32 = MAX_LEVELS - 2;
const FULLSCREEN: u32 = 3;
/// Depth kept for late pass.
const EARLY_DEPTH: Operations<f32> = Operations {
    load: LoadOp::Clear(0.0),
    store: StoreOp::Store,
};
const LATE_DEPTH: Operations<f32> = Operations {
    load: LoadOp::Load,
    store: StoreOp::Discard,
};

/// Culling pass, as the shader's step.late.
#[derive(Clone, Copy)]
enum Phase {
    /// Tests against last frame's pyramid.
    Early,
    /// Retests what the early pass deferred.
    Late,
}

/// A camera view's resources.
struct Camera<'a> {
    work: &'a Work,
    groups: &'a Groups,
    screen: &'a Screen,
    bound: &'a ScreenGroups,
    /// Scene depth into visibility.
    seed: &'a BindGroup,
}

fn bind(pass: &mut ComputePass, pipeline: &ComputePipeline, (level, phase): (u32, Phase)) {
    pass.set_pipeline(pipeline);
    pass.set_immediates(0, bytemuck::bytes_of(&[level, phase as u32]));
}

/// Levels write next level's arguments.
fn traverse(
    pass: &mut ComputePass,
    (p, g, w): (&Pipelines, &Groups, &Work),
    (steps, phase): (u32, Phase),
) {
    for level in 0..steps.min(DEEPEST) {
        let parity = level as usize & 1;
        bind(pass, &p.cull_nodes, (level, phase));
        pass.set_bind_group(1, Some(&*g.toward[1 - parity]), &[]);
        pass.dispatch_workgroups_indirect(&w.steps[parity], u64::from(level) * SLOT_BYTES);
    }
    pass.set_bind_group(1, Some(&*g.args), &[]);
    bind(pass, &p.prepare_clusters, (0, phase));
    pass.dispatch_workgroups(1, 1, 1);
    bind(pass, &p.cull_clusters, (0, phase));
    pass.dispatch_workgroups_indirect(&w.args, CLUSTER_ARGS);
    bind(pass, &p.prepare_raster, (0, phase));
    pass.dispatch_workgroups(1, 1, 1);
}

fn begin_cull<'a>(encoder: &'a mut CommandEncoder, g: &Groups) -> ComputePass<'a> {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("virtual_geometry_cull"),
        timestamp_writes: None,
    });
    pass.set_bind_group(2, Some(&*g.library), &[]);
    pass.set_bind_group(3, Some(&*g.cull), &[]);
    pass
}

/// Culls against last frame's pyramid.
pub(crate) fn cull_early(
    encoder: &mut CommandEncoder,
    (p, g, w): (&Pipelines, &Groups, &Work),
    (total, steps): (u32, u32),
) {
    encoder.clear_buffer(&w.counters, 0, None);
    let mut pass = begin_cull(encoder, g);
    bind(&mut pass, &p.cull_instances, (0, Phase::Early));
    pass.set_bind_group(1, Some(&*g.toward[0]), &[]);
    let groups = total.div_ceil(WORKGROUP);
    pass.dispatch_workgroups(groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS), 1);
    traverse(&mut pass, (p, g, w), (steps, Phase::Early));
}

/// Retests deferred work, current pyramid.
fn cull_late(encoder: &mut CommandEncoder, (p, g, w): (&Pipelines, &Groups, &Work), steps: u32) {
    let mut pass = begin_cull(encoder, g);
    pass.set_bind_group(1, Some(&*g.args), &[]);
    bind(&mut pass, &p.prepare_late, (0, Phase::Late));
    pass.dispatch_workgroups(1, 1, 1);
    bind(&mut pass, &p.seed_late, (0, Phase::Late));
    pass.dispatch_workgroups_indirect(&w.args, SEED_ARGS);
    bind(&mut pass, &p.cull_instances, (0, Phase::Late));
    pass.set_bind_group(1, Some(&*g.toward[0]), &[]);
    pass.dispatch_workgroups_indirect(&w.steps[1], LATE_ARGS);
    traverse(&mut pass, (p, g, w), (steps, Phase::Late));
}

/// Starts visibility from scene depth.
fn seed(encoder: &mut CommandEncoder, p: &Pipelines, c: &Camera) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("virtual_geometry_seed"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&p.seed);
    pass.set_bind_group(0, Some(&**c.seed), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&c.screen.size.to_array()));
    pass.dispatch_workgroups(
        c.screen.size.x.div_ceil(TILE),
        c.screen.size.y.div_ceil(TILE),
        1,
    );
}

fn soft(encoder: &mut CommandEncoder, p: &Pipelines, c: &Camera) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("virtual_geometry_soft"),
        timestamp_writes: None,
    });
    pass.set_bind_group(0, Some(&*c.bound.target), &[]);
    pass.set_bind_group(2, Some(&*c.groups.library), &[]);
    pass.set_bind_group(3, Some(&*c.groups.raster), &[]);
    pass.set_pipeline(&p.soft);
    pass.dispatch_workgroups_indirect(&c.work.args, SOFT_ARGS);
}

fn hard(encoder: &mut CommandEncoder, p: &Pipelines, (c, depth_ops): (&Camera, Operations<f32>)) {
    let depth = RenderPassDepthStencilAttachment {
        view: &c.screen.raster_depth,
        depth_ops: Some(depth_ops),
        stencil_ops: None,
    };
    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("virtual_geometry_hard"),
        color_attachments: &[],
        depth_stencil_attachment: Some(depth),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&p.hard);
    pass.set_bind_group(0, Some(&*c.bound.target), &[]);
    pass.set_bind_group(2, Some(&*c.groups.library), &[]);
    pass.set_bind_group(3, Some(&*c.groups.raster), &[]);
    pass.draw_mesh_tasks_indirect(&c.work.args, HARD_ARGS);
}

/// Farthest depths of the visibility buffer.
fn build_pyramid(encoder: &mut CommandEncoder, p: &Pipelines, bound: &ScreenGroups) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("virtual_geometry_pyramid"),
        timestamp_writes: None,
    });
    let pipelines = std::iter::once(&p.pyramid_first).chain(std::iter::repeat(&p.pyramid_down));
    for (pipeline, (group, source)) in pipelines.zip(&bound.pyramid) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&**group), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&source.to_array()));
        let texels = (*source + 1) / 2;
        pass.dispatch_workgroups(texels.x.div_ceil(TILE), texels.y.div_ceil(TILE), 1);
    }
}

/// Two-pass occlusion culling and raster.
fn rasterize(encoder: &mut CommandEncoder, p: &Pipelines, c: &Camera, load: (u32, u32)) {
    seed(encoder, p, c);
    cull_early(encoder, (p, c.groups, c.work), load);
    soft(encoder, p, c);
    hard(encoder, p, (c, EARLY_DEPTH));
    // Late pass and next frame.
    build_pyramid(encoder, p, c.bound);
    cull_late(encoder, (p, c.groups, c.work), load.1);
    soft(encoder, p, c);
    hard(encoder, p, (c, LATE_DEPTH));
}

type Targets = (
    &'static ViewGeometry,
    &'static ViewDepthStencilTexture,
    &'static ViewPrepassTextures,
);

/// Rasterizes virtual geometry over the prepass.
pub(crate) fn draw_geometry(
    view: ViewQuery<Targets>,
    (pipelines, library, instances): (Res<Pipelines>, Res<Library>, Res<Instances>),
    mut readout: ResMut<Readout>,
    mut ctx: RenderContext,
) {
    let (v, depth, prepass) = view.into_inner();
    let total = instances.extent + instances.dynamic_count;
    let (Some(groups), Some(screen), Some(normal), Some(motion), Some(scene)) = (
        v.groups.as_ref(),
        v.work.screen.as_ref(),
        prepass.normal.as_ref(),
        prepass.motion_vectors.as_ref(),
        prepass.depth.as_ref(),
    ) else {
        return;
    };
    let Some(bound) = groups.screen.as_ref() else {
        return;
    };
    if total == 0 {
        return;
    }
    let pairs = [
        (0, screen.visibility.as_entire_binding()),
        (1, BindingResource::TextureView(&scene.texture.default_view)),
    ];
    let seed = ctx.render_device().create_bind_group(
        "virtual_geometry_seed",
        &pipelines.blank,
        &entries(&pairs),
    );
    let camera = Camera {
        work: &v.work,
        groups,
        screen,
        bound,
        seed: &seed,
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "virtual_geometry_raster");
    ctx.command_encoder().clear_buffer(&v.work.peaks, 0, None);
    rasterize(
        ctx.command_encoder(),
        &pipelines,
        &camera,
        (total, library.depth),
    );
    readout.capture_camera(ctx.command_encoder(), (&v.work.counters, &v.work.peaks));
    span.end(ctx.command_encoder());
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("virtual_geometry_resolve"),
        color_attachments: &[Some(normal.get_attachment()), Some(motion.get_attachment())],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let span = diagnostics.pass_span(&mut pass, "virtual_geometry_resolve");
    pass.set_viewport(
        0.0,
        0.0,
        screen.size.x as f32,
        screen.size.y as f32,
        0.0,
        1.0,
    );
    pass.set_render_pipeline(&pipelines.resolve);
    pass.set_bind_group(2, &groups.library, &[]);
    pass.set_bind_group(3, &bound.resolve, &[]);
    pass.draw(0..FULLSCREEN, 0..1);
    span.end(&mut pass);
    drop(pass);
    // SSAO and upscaling sample this copy.
    ctx.command_encoder().copy_texture_to_texture(
        depth.texture().as_image_copy(),
        scene.texture.texture.as_image_copy(),
        prepass.size,
    );
}
