enable wgpu_mesh_shader;

@group(0) @binding(0) var<storage, read_write> visibility: array<atomic<u64>>;

struct Corner {
    @builtin(position) position: vec4<f32>,
}

struct Face {
    @builtin(triangle_indices) indices: vec3<u32>,
    @builtin(cull_primitive) cull: bool,
    @per_primitive @location(0) @interpolate(flat) id: u32,
}

struct Mesh {
    @builtin(vertices) vertices: array<Corner, 128>,
    @builtin(primitives) primitives: array<Face, 128>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh: Mesh;
var<workgroup> clip: array<vec4<f32>, 128>;

/// Offscreen, backfacing or between centres.
fn culled(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, both: bool) -> bool {
    if a.w <= 0.0 || b.w <= 0.0 || c.w <= 0.0 {
        return false;
    }
    let na = a.xy / a.w;
    let nb = b.xy / b.w;
    let nc = c.xy / c.w;
    let area = (nb.x - na.x) * (nc.y - na.y) - (nb.y - na.y) * (nc.x - na.x);
    if area == 0.0 || (area < 0.0 && !both) {
        return true;
    }
    let lo = min(na, min(nb, nc));
    let hi = max(na, max(nb, nc));
    if any(hi < vec2(-1.0)) || any(lo > vec2(1.0)) {
        return true;
    }
    let pixel_lo = round((lo * vec2(0.5, -0.5) + 0.5) * view.size);
    let pixel_hi = round((hi * vec2(0.5, -0.5) + 0.5) * view.size);
    return pixel_lo.x == pixel_hi.x || pixel_lo.y == pixel_hi.y;
}

@mesh(mesh)
@workgroup_size(128)
fn raster_hard(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let index = counts[HARD_BASE] + wid.y * groups.x + wid.x;
    let live = index < min(counts[HARD_COUNT], view.half_visible);
    let item = visible[view.half_visible + select(0u, index, live)];
    let o = owners[item.x];
    let c = clusters[item.y];
    let vertex_count = select(0u, min(c.counts & 0xffu, CLUSTER_LIMIT), live);
    let triangle_count = select(0u, min((c.counts >> 8u) & 0xffu, CLUSTER_LIMIT), live);
    if lid == 0u {
        mesh.vertex_count = vertex_count;
        mesh.primitive_count = triangle_count;
    }
    if lid < vertex_count {
        let position = view.clip_from_world * vec4(world_of(o, c.first_vertex + lid, view.time), 1.0);
        mesh.vertices[lid].position = position;
        clip[lid] = position;
    }
    workgroupBarrier();
    if lid >= triangle_count {
        return;
    }
    // Indices past the vertex count are hardware undefined behaviour.
    let raw = unpack_triangle(triangles[c.first_triangle + lid]);
    let t = min(raw, vec3(max(vertex_count, 1u) - 1u));
    mesh.primitives[lid].indices = t;
    mesh.primitives[lid].cull = any(raw != t) || culled(clip[t.x], clip[t.y], clip[t.z], two_sided(c));
    mesh.primitives[lid].id = ((view.half_visible + index) << TRIANGLE_BITS) | lid;
}

@fragment
@early_depth_test(force)
fn raster_hard_fragment(@builtin(position) p: vec4<f32>, @per_primitive @location(0) @interpolate(flat) id: u32) {
    atomicMax(&visibility[u32(p.y) * u32(view.size.x) + u32(p.x)], (u64(bitcast<u32>(p.z)) << 32u) | u64(id));
}
