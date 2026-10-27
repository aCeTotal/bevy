use bevy_render::render_resource::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use core::ops::Range;

/// Capacity multiple when a buffer grows.
const GROWTH: u64 = 2;

/// Any slab, for uploading together.
pub(crate) trait Upload {
    fn grow(&mut self, device: &RenderDevice, encoder: &mut CommandEncoder) -> bool;
    fn flush(&mut self, queue: &RenderQueue);
    fn bytes(&self) -> u64;
}

/// GPU array with reusable holes; writes stage until flushed.
pub(crate) struct Slab<T> {
    len: u32,
    free: Vec<Range<u32>>,
    staged: Vec<(u32, Vec<T>)>,
    buffer: Option<Buffer>,
    label: &'static str,
    item: PhantomData<T>,
}

impl<T: Pod + Zeroable> Slab<T> {
    pub fn new(label: &'static str) -> Self {
        Self {
            len: 0,
            free: Vec::new(),
            staged: Vec::new(),
            buffer: None,
            label,
            item: PhantomData,
        }
    }

    /// Room for `count`, first fit.
    pub fn reserve(&mut self, count: u32) -> Range<u32> {
        let Some(at) = self.free.iter().position(|r| r.len() >= count as usize) else {
            let start = self.len;
            self.len += count;
            return start..self.len;
        };
        let start = self.free[at].start;
        self.free[at].start += count;
        if self.free[at].is_empty() {
            self.free.remove(at);
        }
        start..start + count
    }

    /// Stores `items` from `at` onwards.
    pub fn write(&mut self, at: u32, items: impl IntoIterator<Item = T>) {
        let items: Vec<T> = items.into_iter().collect();
        if !items.is_empty() {
            self.staged.push((at, items));
        }
    }

    pub fn release(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }
        let at = self.free.partition_point(|r| r.start < range.start);
        self.free.insert(at, range);
        self.free.dedup_by(|next, prev| {
            let touching = prev.end == next.start;
            if touching {
                prev.end = next.end;
            }
            touching
        });
    }

    pub fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().expect("synced before binding")
    }
}

impl<T: Pod + Zeroable> Upload for Slab<T> {
    /// Room for every item, old contents copied; true if replaced.
    fn grow(&mut self, device: &RenderDevice, encoder: &mut CommandEncoder) -> bool {
        let needed = (u64::from(self.len.max(1))) * size_of::<T>() as u64;
        let old = self.buffer.take_if(|b| b.size() < needed);
        if self.buffer.is_some() {
            return false;
        }
        let size = old
            .as_ref()
            .map_or(needed, |b| (b.size() * GROWTH).max(needed));
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(self.label),
            size,
            usage,
            mapped_at_creation: false,
        });
        if let Some(old) = &old {
            encoder.copy_buffer_to_buffer(old, 0, &buffer, 0, old.size());
        }
        self.buffer = Some(buffer);
        true
    }

    /// Uploads staged writes, after growth landed.
    fn flush(&mut self, queue: &RenderQueue) {
        let Some(buffer) = self.buffer.as_ref() else {
            return;
        };
        for (at, items) in self.staged.drain(..) {
            queue.write_buffer(
                buffer,
                u64::from(at) * size_of::<T>() as u64,
                bytemuck::cast_slice(&items),
            );
        }
    }

    fn bytes(&self) -> u64 {
        self.buffer.as_ref().map_or(0, |b| b.size())
    }
}
