use alloc::sync::Arc;
use bevy_render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, MapMode,
};
use bevy_render::renderer::RenderDevice;
use core::sync::atomic::{AtomicU8, Ordering};

/// Copies in flight.
const RING: usize = 3;
const FREE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;

struct Slot {
    buffer: Buffer,
    state: Arc<AtomicU8>,
}

/// GPU bytes read frames later.
pub(crate) struct Readback {
    slots: Vec<Slot>,
    next: usize,
}

impl Readback {
    pub fn new(device: &RenderDevice, label: &'static str, bytes: u64) -> Readback {
        let usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        let slot = |_| Slot {
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            }),
            state: Arc::new(AtomicU8::new(FREE)),
        };
        Readback {
            slots: (0..RING).map(slot).collect(),
            next: 0,
        }
    }

    /// Copies `sources` back to back into a free slot.
    pub fn capture(&mut self, encoder: &mut CommandEncoder, sources: &[(&Buffer, u64)]) {
        let slot = &self.slots[self.next];
        if slot.state.load(Ordering::Acquire) != FREE {
            return;
        }
        let mut at = 0;
        for &(source, bytes) in sources {
            encoder.copy_buffer_to_buffer(source, 0, &slot.buffer, at, bytes);
            at += bytes;
        }
        slot.state.store(COPIED, Ordering::Release);
        self.next = (self.next + 1) % RING;
    }

    /// Maps this frame's copies.
    pub fn map(&self) {
        for slot in self
            .slots
            .iter()
            .filter(|s| s.state.load(Ordering::Acquire) == COPIED)
        {
            slot.state.store(MAPPING, Ordering::Release);
            let state = slot.state.clone();
            slot.buffer
                .slice(..)
                .map_async(MapMode::Read, move |result| {
                    state.store(
                        if result.is_ok() { MAPPED } else { FREE },
                        Ordering::Release,
                    );
                });
        }
    }

    /// Visits mapped copies, then frees them.
    pub fn drain(&self, mut visit: impl FnMut(&[u8])) {
        for slot in self
            .slots
            .iter()
            .filter(|s| s.state.load(Ordering::Acquire) == MAPPED)
        {
            if let Ok(bytes) = slot.buffer.slice(..).get_mapped_range() {
                visit(&bytes);
            }
            slot.buffer.unmap();
            slot.state.store(FREE, Ordering::Release);
        }
    }
}
