use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Res, ResMut};
use bevy_render::Extract;
use bevy_time::Time;

/// Shader clock, now and last frame.
#[derive(Resource, Default)]
pub(crate) struct Clock {
    pub now: f32,
    pub before: f32,
}

pub(crate) fn extract_clock(mut clock: ResMut<Clock>, time: Extract<Res<Time>>) {
    clock.before = clock.now;
    clock.now = time.elapsed_secs_wrapped();
}
