use bevy_asset::{load_embedded_asset, AssetServer};
use bevy_camera::{MainPassResolutionOverride, Viewport};
use bevy_core_pipeline::prepass::{
    MotionVectorPrepass, PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms,
    ViewPrepassTextures, MOTION_VECTOR_PREPASS_FORMAT,
};
use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::query::With;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res};
use bevy_pbr::Transmissive3d;
use bevy_render::camera::ExtractedCamera;
use bevy_render::render_phase::ViewSortedRenderPhases;
use bevy_render::render_resource::binding_types::{texture_depth_2d, uniform_buffer};
use bevy_render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, FragmentState, PipelineCache, RenderPassDescriptor,
    RenderPipelineDescriptor, ShaderStages,
};
use bevy_render::renderer::{RenderContext, ViewQuery};
use bevy_render::view::{
    ExtractedView, ViewDepthStencilTexture, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy_utils::default;

/// Pipeline writing water motion vectors.
#[derive(Resource)]
pub(crate) struct WaterMotion {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedRenderPipelineId,
}

pub(crate) fn setup_motion(
    mut commands: Commands,
    (cache, fullscreen, assets): (Res<PipelineCache>, Res<FullscreenShader>, Res<AssetServer>),
) {
    let layout = BindGroupLayoutDescriptor::new(
        "water_motion",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<PreviousViewData>(true),
                texture_depth_2d(),
                texture_depth_2d(),
            ),
        ),
    );
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("water_motion".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: load_embedded_asset!(assets.as_ref(), "motion.wesl"),
            targets: vec![Some(ColorTargetState {
                format: MOTION_VECTOR_PREPASS_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(WaterMotion { layout, pipeline });
}

/// What the motion pass reads of a view.
type MotionView = (
    &'static ExtractedView,
    &'static ExtractedCamera,
    &'static ViewDepthStencilTexture,
    &'static ViewPrepassTextures,
    (
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
    ),
    Option<&'static MainPassResolutionOverride>,
);

/// Gives water the camera motion of its own surface, not of the ground beneath.
pub(crate) fn water_motion(
    view: ViewQuery<MotionView, With<MotionVectorPrepass>>,
    (motion, cache): (Res<WaterMotion>, Res<PipelineCache>),
    (uniforms, previous): (Res<ViewUniforms>, Res<PreviousViewUniforms>),
    phases: Res<ViewSortedRenderPhases<Transmissive3d>>,
    mut ctx: RenderContext,
) {
    let (extracted, camera, depth, prepass, (offset, previous_offset), resolution) =
        view.into_inner();
    let water = phases.get(&extracted.retained_view_entity);
    if water.is_none_or(|phase| phase.items.is_empty()) {
        return;
    }
    let (Some(pipeline), Some(target), Some(final_depth), Some(opaque_depth)) = (
        cache.get_render_pipeline(motion.pipeline),
        prepass.motion_vectors.as_ref(),
        depth.attachment.depth_stencil_views().depth_only_view(),
        prepass.depth_only_view(),
    ) else {
        return;
    };
    let (Some(view_binding), Some(previous_binding)) =
        (uniforms.uniforms.binding(), previous.uniforms.binding())
    else {
        return;
    };
    let bind_group = ctx.render_device().create_bind_group(
        "water_motion",
        &cache.get_bind_group_layout(&motion.layout),
        &BindGroupEntries::sequential((view_binding, previous_binding, final_depth, opaque_depth)),
    );
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("water_motion"),
        color_attachments: &[Some(target.get_attachment())],
        ..default()
    });
    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution)
    {
        pass.set_camera_viewport(&viewport);
    }
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[offset.offset, previous_offset.offset]);
    pass.draw(0..3, 0..1);
}
