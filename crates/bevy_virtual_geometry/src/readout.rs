use crate::capacity::{Capacities, PEAKS};
use crate::readback::Readback;
use crate::records::FIXED_COUNTERS;
use crate::view::PEAK_BYTES;
use alloc::sync::Arc;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res, ResMut};
use bevy_render::render_resource::{Buffer, CommandEncoder, PollType};
use bevy_render::renderer::RenderDevice;
use std::sync::Mutex;

const COUNTER_BYTES: u64 = (FIXED_COUNTERS * 4) as u64;
/// Threshold growth per saturated readback.
const BIAS_GROWTH: f32 = 1.5;
/// Threshold relaxation per calm readback.
const BIAS_DECAY: f32 = 0.97;
const MAX_BIAS: f32 = 16.0;

/// Main view's last GPU counts.
#[derive(Clone, Copy, Default, Debug)]
pub struct Drawn {
    /// Mesh copies traversed.
    pub copies: u32,
    /// Clusters rasterized.
    pub clusters: u32,
    /// Triangles rasterized.
    pub triangles: u32,
    /// Lists that ran out of room.
    pub overflow: u32,
    /// Off-screen and hidden objects.
    pub objects_culled: [u32; 2],
    /// Off-screen and hidden nodes.
    pub nodes_culled: [u32; 2],
    /// Off-screen, hidden, back-facing.
    pub clusters_culled: [u32; 3],
}

/// Latest counts for the main world.
#[derive(Resource, Clone, Default)]
pub struct VirtualGeometryCounts(pub Arc<Mutex<Drawn>>);

/// Counters and list peaks on their way back.
#[derive(Resource)]
pub(crate) struct Readout {
    camera: Readback,
    shadow: Readback,
}

pub(crate) fn setup_readout(mut commands: Commands, device: Res<RenderDevice>) {
    commands.insert_resource(Readout {
        camera: Readback::new(
            &device,
            "virtual_geometry_readout",
            COUNTER_BYTES + PEAK_BYTES,
        ),
        shadow: Readback::new(&device, "virtual_geometry_shadow_peaks", PEAK_BYTES),
    });
}

impl Readout {
    /// Copies the camera's counters and peaks.
    pub fn capture_camera(
        &mut self,
        encoder: &mut CommandEncoder,
        (counters, peaks): (&Buffer, &Buffer),
    ) {
        self.camera
            .capture(encoder, &[(counters, COUNTER_BYTES), (peaks, PEAK_BYTES)]);
    }

    /// Copies the shadow cascades' peaks.
    pub fn capture_shadow(&mut self, encoder: &mut CommandEncoder, peaks: &Buffer) {
        self.shadow.capture(encoder, &[(peaks, PEAK_BYTES)]);
    }
}

/// Maps this frame's copies.
pub(crate) fn map_readout(readout: Res<Readout>) {
    readout.camera.map();
    readout.shadow.map();
}

/// Cut threshold; grows only where lists cannot.
#[derive(Resource)]
pub(crate) struct LodBias(pub f32);

impl Default for LodBias {
    fn default() -> Self {
        LodBias(1.0)
    }
}

fn drawn(v: &[u32]) -> Drawn {
    Drawn {
        copies: v[0],
        clusters: v[2] + v[3],
        triangles: v[5],
        overflow: v[4],
        objects_culled: [v[14], v[15]],
        nodes_culled: [v[16], v[17]],
        clusters_culled: [v[18], v[19], v[20]],
    }
}

fn peaks(v: &[u32]) -> [u32; PEAKS] {
    core::array::from_fn(|k| v[k])
}

/// Publishes counts, grows lists to demand, frees buffers.
pub(crate) fn read_readout(
    readout: Res<Readout>,
    (counts, mut capacities, mut bias): (
        Res<VirtualGeometryCounts>,
        ResMut<Capacities>,
        ResMut<LodBias>,
    ),
    device: Res<RenderDevice>,
) {
    device.poll(PollType::Poll).ok();
    let limit = device.limits().max_storage_buffer_binding_size;
    let capacities = &mut *capacities;
    let mut fits = None;
    readout.camera.drain(|bytes| {
        let words: &[u32] = bytemuck::cast_slice(bytes);
        let (counters, demand) = words.split_at(FIXED_COUNTERS);
        if let Ok(mut shared) = counts.0.lock() {
            *shared = drawn(counters);
        }
        fits = Some(fits.unwrap_or(true) & capacities.camera.fit(peaks(demand), limit));
    });
    readout.shadow.drain(|bytes| {
        fits = Some(
            fits.unwrap_or(true)
                & capacities
                    .shadow
                    .fit(peaks(bytemuck::cast_slice(bytes)), limit),
        );
    });
    if let Some(fits) = fits {
        capacities.saturated = !fits;
    }
    bias.0 = if capacities.saturated {
        (bias.0 * BIAS_GROWTH).min(MAX_BIAS)
    } else {
        (bias.0 * BIAS_DECAY).max(1.0)
    };
}
