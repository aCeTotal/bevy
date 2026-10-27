use bevy_ecs::resource::Resource;

/// Bytes per item of each list, in peak order.
pub(crate) const ITEM_BYTES: [u64; PEAKS] = [80, 16, 8, 16, 4, 8, 8];
/// Lists the GPU reports demand for.
pub(crate) const PEAKS: usize = 7;
/// Demand above this share of capacity grows the list.
const HEADROOM: u64 = 4;

/// Item capacities of one work set's lists, in peak order: owners, queue level,
/// candidates, raster path, late instances, late nodes, late clusters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Capacity(pub [u32; PEAKS]);

impl Default for Capacity {
    fn default() -> Self {
        Capacity([1 << 18, 1 << 20, 1 << 21, 1 << 19, 36_864, 1 << 18, 1 << 19])
    }
}

impl Capacity {
    /// Room for `demand` with headroom; false where a list would pass `limit` bytes.
    pub fn fit(&mut self, demand: [u32; PEAKS], limit: u64) -> bool {
        let mut fits = true;
        for ((cap, &need), bytes) in self.0.iter_mut().zip(&demand).zip(ITEM_BYTES) {
            if u64::from(need) * HEADROOM <= u64::from(*cap) * (HEADROOM - 1) {
                continue;
            }
            let wanted = (u64::from(need) * 2).next_power_of_two();
            let most = limit / bytes;
            fits &= wanted <= most;
            *cap = wanted.min(most) as u32;
        }
        fits
    }
}

/// Capacities the camera and the shared shadow work need.
#[derive(Resource, Default)]
pub(crate) struct Capacities {
    pub camera: Capacity,
    pub shadow: Capacity,
    /// Some list hit the device limit.
    pub saturated: bool,
}
