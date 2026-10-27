use crate::arrays::{Links, Spans};
use crate::asset::{MeshRecords, VirtualGeometry};
use crate::instances::Instances;
use crate::library::Library;
use alloc::sync::Arc;
use bevy_asset::AssetId;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Res, ResMut};
use bevy_math::Affine3A;
use bevy_platform::collections::HashMap;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use crossbeam_channel::{Receiver, Sender};

/// Loaded geometry an assembly links to.
#[derive(Default)]
pub struct Refs {
    /// Geometry and variant per local part mesh.
    pub parts: Vec<(AssetId<VirtualGeometry>, u32)>,
    /// Geometry and material per local material.
    pub materials: Vec<(AssetId<VirtualGeometry>, u32)>,
}

/// Mesh placing loaded geometry, drawn in place.
pub struct Assembled {
    /// The mesh; part roots and materials index `refs`.
    pub mesh: MeshRecords,
    /// World from mesh.
    pub transform: Affine3A,
    /// Far cut error multiplier.
    pub coarse: f32,
    /// What the mesh links to.
    pub refs: Arc<Refs>,
}

type Change = (u64, Option<Assembled>);

/// Keyed still meshes to draw, sent to the render world.
#[derive(Resource, Clone)]
pub struct Assemblies(Sender<Change>);

impl Assemblies {
    /// Draws `assembled` under `key`, replacing what was there.
    pub fn set(&self, key: u64, assembled: Assembled) {
        self.0.send((key, Some(assembled))).ok();
    }

    /// Stops drawing `key`.
    pub fn remove(&self, key: u64) {
        self.0.send((key, None)).ok();
    }
}

/// Changes for the render world.
#[derive(Resource)]
pub(crate) struct Feed(Receiver<Change>);

pub(crate) fn channel() -> (Assemblies, Feed) {
    let (tx, rx) = crossbeam_channel::unbounded();
    (Assemblies(tx), Feed(rx))
}

/// Stored assemblies, and unplaceable ones.
#[derive(Resource, Default)]
pub(crate) struct Placed {
    spans: HashMap<u64, Spans>,
    waiting: Vec<(u64, Assembled)>,
}

/// Global materials, roots and depths.
type Tables = (Vec<u32>, Vec<u32>, Vec<u32>);

fn tables(refs: &Refs, library: &Library) -> Option<Tables> {
    let material =
        |&(id, m): &(AssetId<VirtualGeometry>, u32)| Some(library.loaded.get(&id)?.material + m);
    let root = |&(id, v): &(AssetId<VirtualGeometry>, u32)| {
        library.loaded.get(&id)?.roots.get(v as usize).copied()
    };
    let depth = |&(id, v): &(AssetId<VirtualGeometry>, u32)| {
        library.loaded.get(&id)?.depths.get(v as usize).copied()
    };
    let materials = refs.materials.iter().map(material).collect::<Option<_>>()?;
    let roots = refs.parts.iter().map(root).collect::<Option<_>>()?;
    let depths = refs.parts.iter().map(depth).collect::<Option<_>>()?;
    Some((materials, roots, depths))
}

/// Swaps changed assemblies in.
pub(crate) fn prepare_assemblies(
    feed: Res<Feed>,
    mut placed: ResMut<Placed>,
    (mut library, mut instances): (ResMut<Library>, ResMut<Instances>),
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
) {
    let placed = &mut *placed;
    for (key, change) in feed.0.try_iter() {
        placed.waiting.retain(|(k, _)| *k != key);
        if let Some(spans) = placed.spans.remove(&key) {
            library.release(spans);
        }
        instances.remove(key, &queue);
        placed.waiting.extend(change.map(|a| (key, a)));
    }
    for (key, a) in core::mem::take(&mut placed.waiting) {
        let Some((materials, roots, depths)) = tables(&a.refs, &library) else {
            placed.waiting.push((key, a));
            continue;
        };
        let spans = library.place(
            &a.mesh,
            &Links {
                materials: &materials,
                roots: &roots,
                depths: &depths,
            },
        );
        instances.set(
            key,
            (spans.root(), a.transform, a.coarse),
            (&device, &queue),
        );
        placed.spans.insert(key, spans);
    }
}
