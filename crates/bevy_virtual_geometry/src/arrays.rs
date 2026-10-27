use crate::asset::{Clip, MeshRecords, Shading, VirtualMaterial};
use crate::records::*;
use crate::slab::{Slab, Upload};
use bevy_render::render_resource::CommandEncoderDescriptor;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use core::ops::Range;
use tracing::info;

/// Material index bits in cluster counts.
const MATERIAL_SHIFT: u32 = 16;
const LOW_MASK: u32 = (1 << MATERIAL_SHIFT) - 1;
const INDEX_MASK: u32 = (1 << CHILD_INDEX_BITS) - 1;

/// Loaded hierarchies, GPU layout.
pub(crate) struct Arrays {
    pub positions: Slab<[f32; 3]>,
    /// Lockstep with positions.
    pub vertices: Slab<GpuVertex>,
    pub triangles: Slab<u32>,
    pub clusters: Slab<GpuCluster>,
    pub nodes: Slab<GpuNode>,
    pub children: Slab<[u32; 2]>,
    pub parts: Slab<GpuPart>,
    pub materials: Slab<GpuMaterial>,
    /// Skinning matrices of every clip.
    pub poses: Slab<[[f32; 4]; 3]>,
    /// Most node levels any root descends.
    pub depth: u32,
}

impl Default for Arrays {
    fn default() -> Self {
        Self {
            positions: Slab::new("virtual_geometry_positions"),
            vertices: Slab::new("virtual_geometry_vertices"),
            triangles: Slab::new("virtual_geometry_triangles"),
            clusters: Slab::new("virtual_geometry_clusters"),
            nodes: Slab::new("virtual_geometry_nodes"),
            children: Slab::new("virtual_geometry_children"),
            parts: Slab::new("virtual_geometry_parts"),
            materials: Slab::new("virtual_geometry_materials"),
            poses: Slab::new("virtual_geometry_poses"),
            depth: 0,
        }
    }
}

/// Where a clip sits in the poses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClipRef {
    pub base: u32,
    pub bones: u32,
    pub frames: u32,
    pub fps: f32,
    /// Farthest a vertex strays, mesh units.
    pub reach: f32,
}

/// Global indices of one geometry.
pub(crate) struct Appended {
    pub roots: Vec<u32>,
    /// Node levels below each root.
    pub depths: Vec<u32>,
    /// First global material.
    pub material: u32,
    pub clips: Vec<ClipRef>,
}

/// Global indices a mesh refers to.
pub(crate) struct Links<'a> {
    /// Per local material.
    pub materials: &'a [u32],
    /// Root node per local part mesh.
    pub roots: &'a [u32],
    /// Node levels per local part mesh.
    pub depths: &'a [u32],
}

/// Ranges one placed mesh holds.
pub(crate) struct Spans {
    vertices: Range<u32>,
    triangles: Range<u32>,
    clusters: Range<u32>,
    nodes: Range<u32>,
    children: Range<u32>,
    parts: Range<u32>,
}

impl Spans {
    pub fn root(&self) -> u32 {
        self.nodes.start
    }
}

fn shading(s: Shading) -> u32 {
    match s {
        Shading::Standard => 0,
        Shading::Foliage => 1,
        Shading::Layered => 2,
    }
}

fn material(m: &VirtualMaterial, textures: [u32; 3]) -> GpuMaterial {
    GpuMaterial {
        base_color: m.base_color,
        roughness: m.roughness,
        metallic: m.metallic,
        translucency: m.translucency,
        shading: shading(m.shading),
        albedo: textures[0],
        normal: textures[1],
        orm: textures[2],
        flags: if m.two_sided { TWO_SIDED } else { 0 },
    }
}

fn rebased_cluster(c: &GpuCluster, s: &Spans, materials: &[u32]) -> GpuCluster {
    let material = materials[(c.counts >> MATERIAL_SHIFT) as usize];
    GpuCluster {
        first_vertex: s.vertices.start + c.first_vertex,
        first_triangle: s.triangles.start + c.first_triangle,
        counts: (c.counts & LOW_MASK) | (material << MATERIAL_SHIFT),
        ..*c
    }
}

fn rebased_child(c: [u32; 2], s: &Spans) -> [u32; 2] {
    let (kind, index) = (c[0] >> CHILD_INDEX_BITS, c[0] & INDEX_MASK);
    let base = match kind {
        CHILD_NODE => s.nodes.start,
        CHILD_CLUSTERS => s.clusters.start,
        _ => s.parts.start,
    };
    pack_child(kind, base + index, c[1])
}

/// Node levels below `node`, parts included.
fn levels(m: &MeshRecords, node: usize, depths: &[u32]) -> u32 {
    let n = &m.nodes[node];
    let children = &m.children[n.first_child as usize..(n.first_child + n.child_count) as usize];
    let below = |c: &[u32; 2]| match c[0] >> CHILD_INDEX_BITS {
        CHILD_NODE => levels(m, (c[0] & INDEX_MASK) as usize, depths),
        CHILD_PART => depths[m.parts[(c[0] & INDEX_MASK) as usize].root as usize],
        _ => 0,
    };
    1 + children.iter().map(below).max().unwrap_or(0)
}

impl Arrays {
    /// Appends a geometry; returns its indices.
    pub fn append(
        &mut self,
        (meshes, roots): (&[MeshRecords], &[u32]),
        (materials, textures): (&[VirtualMaterial], &[[u32; 3]]),
        clips: &[Clip],
    ) -> Appended {
        let range = self.materials.reserve(materials.len() as u32);
        self.materials.write(
            range.start,
            materials.iter().zip(textures).map(|(m, t)| material(m, *t)),
        );
        let globals: Vec<u32> = range.clone().collect();
        let mut placed = Vec::with_capacity(meshes.len());
        let mut depths = Vec::with_capacity(meshes.len());
        for mesh in meshes {
            let (spans, depth) = self.place(
                mesh,
                &Links {
                    materials: &globals,
                    roots: &placed,
                    depths: &depths,
                },
            );
            placed.push(spans.root());
            depths.push(depth);
        }
        let pick = |all: &[u32]| roots.iter().map(|&r| all[r as usize]).collect();
        let clips = clips.iter().map(|c| self.clip(c)).collect();
        Appended {
            roots: pick(&placed),
            depths: pick(&depths),
            material: range.start,
            clips,
        }
    }

    fn clip(&mut self, c: &Clip) -> ClipRef {
        let range = self.poses.reserve(c.poses.len() as u32);
        self.poses.write(range.start, c.poses.iter().copied());
        ClipRef {
            base: range.start,
            bones: c.bones,
            frames: c.frames,
            fps: c.fps,
            reach: c.reach,
        }
    }

    /// Stores a mesh; returns its spans and depth.
    pub fn place(&mut self, m: &MeshRecords, links: &Links) -> (Spans, u32) {
        let vertices = self.positions.reserve(m.positions.len() as u32);
        self.vertices.reserve(m.positions.len() as u32);
        let s = Spans {
            vertices,
            triangles: self.triangles.reserve(m.triangles.len() as u32),
            clusters: self.clusters.reserve(m.clusters.len() as u32),
            nodes: self.nodes.reserve(m.nodes.len() as u32),
            children: self.children.reserve(m.children.len() as u32),
            parts: self.parts.reserve(m.parts.len() as u32),
        };
        self.positions
            .write(s.vertices.start, m.positions.iter().copied());
        self.vertices
            .write(s.vertices.start, m.vertices.iter().copied());
        self.triangles
            .write(s.triangles.start, m.triangles.iter().copied());
        self.clusters.write(
            s.clusters.start,
            m.clusters
                .iter()
                .map(|c| rebased_cluster(c, &s, links.materials)),
        );
        self.parts.write(
            s.parts.start,
            m.parts.iter().map(|p| GpuPart {
                root: links.roots[p.root as usize],
                ..*p
            }),
        );
        self.children.write(
            s.children.start,
            m.children.iter().map(|&c| rebased_child(c, &s)),
        );
        self.nodes.write(
            s.nodes.start,
            m.nodes.iter().map(|n| GpuNode {
                first_child: s.children.start + n.first_child,
                ..*n
            }),
        );
        let depth = if m.nodes.is_empty() {
            0
        } else {
            levels(m, 0, links.depths)
        };
        self.depth = self.depth.max(depth);
        (s, depth)
    }

    pub fn release(&mut self, s: Spans) {
        self.positions.release(s.vertices.clone());
        self.vertices.release(s.vertices);
        self.triangles.release(s.triangles);
        self.clusters.release(s.clusters);
        self.nodes.release(s.nodes);
        self.children.release(s.children);
        self.parts.release(s.parts);
    }

    fn slabs(&mut self) -> [&mut dyn Upload; 9] {
        [
            &mut self.positions,
            &mut self.vertices,
            &mut self.triangles,
            &mut self.clusters,
            &mut self.nodes,
            &mut self.children,
            &mut self.parts,
            &mut self.materials,
            &mut self.poses,
        ]
    }

    /// Uploads changes; true if buffers changed.
    pub fn sync(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("virtual_geometry_growth"),
        });
        let grown = self
            .slabs()
            .into_iter()
            .fold(false, |any, s| s.grow(device, &mut encoder) | any);
        if grown {
            queue.submit([encoder.finish()]);
            info!(
                "virtual geometry buffers grew to {} MiB",
                self.slabs().iter().map(|s| s.bytes()).sum::<u64>() >> 20
            );
        }
        self.slabs().into_iter().for_each(|s| s.flush(queue));
        grown
    }
}
