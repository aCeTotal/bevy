use crate::arrays::{Appended, Arrays, Links, Spans};
use crate::asset::{MeshRecords, VirtualGeometry};
use crate::records::NO_TEXTURE;
use bevy_asset::{AssetEvent, AssetId, Assets, Handle};
use bevy_ecs::message::MessageReader;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Res, ResMut};
use bevy_image::Image;
use bevy_platform::collections::HashMap;
use bevy_render::render_resource::Buffer;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_render::Extract;

/// GPU buffers of loaded hierarchies.
pub(crate) struct Buffers {
    pub positions: Buffer,
    pub vertices: Buffer,
    pub triangles: Buffer,
    pub clusters: Buffer,
    pub nodes: Buffer,
    pub children: Buffer,
    pub parts: Buffer,
    pub materials: Buffer,
    pub poses: Buffer,
}

#[derive(Resource, Default)]
pub(crate) struct Library {
    arrays: Arrays,
    /// Global indices per geometry.
    pub loaded: HashMap<AssetId<VirtualGeometry>, Appended>,
    pub textures: Vec<AssetId<Image>>,
    /// Table slot per texture.
    slots: HashMap<AssetId<Image>, u32>,
    pub buffers: Option<Buffers>,
    /// Node queue passes a frame needs.
    pub depth: u32,
    /// Bumped when buffers are replaced.
    pub generation: u32,
}

/// Geometries finished loading this frame.
#[derive(Resource, Default)]
pub(crate) struct Arrivals(Vec<(AssetId<VirtualGeometry>, VirtualGeometry)>);

pub(crate) fn extract_arrivals(
    mut arrivals: ResMut<Arrivals>,
    mut events: Extract<MessageReader<AssetEvent<VirtualGeometry>>>,
    assets: Extract<Res<Assets<VirtualGeometry>>>,
) {
    for event in events.read() {
        let AssetEvent::Added { id } = event else {
            continue;
        };
        let Some(g) = assets.get(*id) else { continue };
        arrivals.0.push((*id, g.clone()));
    }
}

fn buffers(a: &Arrays) -> Buffers {
    Buffers {
        positions: a.positions.buffer().clone(),
        vertices: a.vertices.buffer().clone(),
        triangles: a.triangles.buffer().clone(),
        clusters: a.clusters.buffer().clone(),
        nodes: a.nodes.buffer().clone(),
        children: a.children.buffer().clone(),
        parts: a.parts.buffer().clone(),
        materials: a.materials.buffer().clone(),
        poses: a.poses.buffer().clone(),
    }
}

impl Library {
    fn texture(&mut self, image: Option<&Handle<Image>>) -> u32 {
        let Some(id) = image.map(Handle::id) else {
            return NO_TEXTURE;
        };
        *self.slots.entry(id).or_insert_with(|| {
            self.textures.push(id);
            self.textures.len() as u32 - 1
        })
    }

    fn add(&mut self, id: AssetId<VirtualGeometry>, g: &VirtualGeometry) {
        let slots: Vec<[u32; 3]> = g
            .materials
            .iter()
            .map(|m| {
                [m.albedo.as_ref(), m.normal.as_ref(), m.orm.as_ref()].map(|t| self.texture(t))
            })
            .collect();
        let appended = self
            .arrays
            .append((&g.meshes, &g.roots), (&g.materials, &slots), &g.clips);
        self.loaded.insert(id, appended);
    }

    /// Stores a mesh placing loaded roots.
    pub fn place(&mut self, mesh: &MeshRecords, links: &Links) -> Spans {
        self.arrays.place(mesh, links).0
    }

    pub fn release(&mut self, spans: Spans) {
        self.arrays.release(spans);
    }
}

/// Appends arrivals.
pub(crate) fn prepare_library(mut library: ResMut<Library>, mut arrivals: ResMut<Arrivals>) {
    for (id, g) in core::mem::take(&mut arrivals.0) {
        library.add(id, &g);
    }
}

/// Uploads changes, rebinding on growth.
pub(crate) fn upload_library(
    mut library: ResMut<Library>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let library = &mut *library;
    library.depth = library.arrays.depth;
    if !library.arrays.sync(&device, &queue) {
        return;
    }
    library.buffers = Some(buffers(&library.arrays));
    library.generation += 1;
}
