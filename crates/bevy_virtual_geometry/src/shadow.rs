use crate::draw::{cull_early, HARD_ARGS};
use crate::instances::Instances;
use crate::library::Library;
use crate::pipelines::Pipelines;
use crate::readout::Readout;
use crate::view::{ShadowWork, ViewGeometry};
use bevy_ecs::query::{Has, With, Without};
use bevy_ecs::system::{Query, Res, ResMut};
use bevy_pbr::{KeptRestore, LightEntity, ShadowView, StillCasters};
use bevy_render::diagnostic::RecordDiagnostics;
use bevy_render::render_resource::{CommandEncoder, RenderPassDescriptor, StoreOp};
use bevy_render::renderer::RenderContext;
use bevy_render::view::ExtractedView;

/// One view's culling and depth raster.
fn cast(
    encoder: &mut CommandEncoder,
    (pipelines, steps): (&Pipelines, u32),
    (v, shadow, view): (&ViewGeometry, &ShadowView, &ExtractedView),
    total: u32,
) {
    let Some(g) = v.groups.as_ref().filter(|_| total > 0) else {
        return;
    };
    let depth = shadow.depth_attachment.get_attachment(StoreOp::Store);
    cull_early(encoder, (pipelines, g, &v.work), (total, steps));
    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("virtual_geometry_cast"),
        color_attachments: &[],
        depth_stencil_attachment: Some(depth),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let [x, y, w, h] = view.viewport.as_vec4().to_array();
    pass.set_viewport(x, y, w, h, 0.0, 1.0);
    pass.set_pipeline(&pipelines.cast);
    pass.set_bind_group(2, Some(&*g.library), &[]);
    pass.set_bind_group(3, Some(&*g.raster), &[]);
    pass.draw_mesh_tasks_indirect(&v.work.args, HARD_ARGS);
}

/// Still copies into kept sun shadows.
pub(crate) fn draw_still_shadows(
    views: Query<(&ViewGeometry, &ShadowView, &ExtractedView), With<StillCasters>>,
    (pipelines, library): (Res<Pipelines>, Res<Library>),
    (instances, shared): (Res<Instances>, Res<ShadowWork>),
    mut ctx: RenderContext,
) {
    let Some(work) = shared.work() else {
        return;
    };
    ctx.command_encoder().clear_buffer(&work.peaks, 0, None);
    if views.is_empty() {
        return;
    }
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "virtual_geometry_still_shadows");
    for view in &views {
        cast(
            ctx.command_encoder(),
            (&pipelines, library.depth),
            view,
            instances.extent,
        );
    }
    span.end(ctx.command_encoder());
}

/// Sun shadow depth raster.
pub(crate) fn draw_shadows(
    views: Query<
        (
            &ViewGeometry,
            &ShadowView,
            &ExtractedView,
            &LightEntity,
            Has<KeptRestore>,
        ),
        Without<StillCasters>,
    >,
    (pipelines, library): (Res<Pipelines>, Res<Library>),
    (instances, shared, mut readout): (Res<Instances>, Res<ShadowWork>, ResMut<Readout>),
    mut ctx: RenderContext,
) {
    let Some(work) = shared.work() else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "virtual_geometry_shadows");
    for (v, shadow, view, light, movers) in &views {
        if !matches!(light, LightEntity::Directional { .. }) {
            continue;
        }
        let still = if movers { 0 } else { instances.extent };
        let total = still + instances.dynamic_count;
        cast(
            ctx.command_encoder(),
            (&pipelines, library.depth),
            (v, shadow, view),
            total,
        );
    }
    readout.capture_shadow(ctx.command_encoder(), &work.peaks);
    span.end(ctx.command_encoder());
}
