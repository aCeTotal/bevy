@group(0) @binding(0) var<storage, read_write> visibility: array<atomic<u64>>;

var<workgroup> screen: array<vec3<f32>, 128>;

/// Screen corners of one triangle.
struct Corners {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
}

fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

/// Columns a row may cover.
fn span(row: vec3<f32>, along: vec3<f32>, width: f32) -> vec2<f32> {
    let cross = -row / along;
    let starts = select(vec3(0.0), floor(cross), along > vec3(0.0));
    let ends = select(vec3(width), ceil(cross), along < vec3(0.0));
    return vec2(max(max(starts.x, starts.y), starts.z), min(min(ends.x, ends.y), ends.z));
}

fn write(pixel: vec2<u32>, depth: f32, id: u32) {
    atomicMax(&visibility[pixel.y * u32(view.size.x) + pixel.x], (u64(bitcast<u32>(depth)) << 32u) | u64(id));
}

/// Nearest-wins atomic coverage.
fn rasterize(t: Corners, id: u32, both: bool) {
    let area = edge(t.a.xy, t.b.xy, t.c.xy);
    if area == 0.0 || (area > 0.0 && !both) {
        return;
    }
    let lo = max(floor(min(t.a.xy, min(t.b.xy, t.c.xy)) - 0.5), vec2(0.0));
    let hi = min(ceil(max(t.a.xy, max(t.b.xy, t.c.xy)) - 0.5), view.size - 1.0);
    // Bounded work; broken corners never loop.
    if !all(hi - lo < vec2(2.0 * view.soft_limit)) || any(lo != lo) || any(hi != hi) {
        return;
    }
    let flip = sign(area);
    // Edges step linearly per pixel.
    let along = vec3(t.b.y - t.c.y, t.c.y - t.a.y, t.a.y - t.b.y) * flip;
    let down = vec3(t.c.x - t.b.x, t.a.x - t.c.x, t.b.x - t.a.x) * flip;
    let first = lo + 0.5;
    var row = vec3(edge(t.b.xy, t.c.xy, first), edge(t.c.xy, t.a.xy, first), edge(t.a.xy, t.b.xy, first)) * flip;
    let depth = vec3(t.a.z, t.b.z, t.c.z) / abs(area);
    for (var y = lo.y; y <= hi.y; y += 1.0) {
        let columns = span(row, along, hi.x - lo.x);
        var w = row + along * columns.x;
        for (var x = lo.x + columns.x; x <= lo.x + columns.y; x += 1.0) {
            if all(w >= vec3(0.0)) {
                write(vec2<u32>(vec2(x, y)), saturate(dot(w, depth)), id);
            }
            w += along;
        }
        row += down;
    }
}

/// Threads per cluster, slots each.
const LANES: u32 = 64u;
const SLOTS: u32 = 2u;

/// Cluster vertex in pixels.
fn screen_of(o: Owner, instance: Instance, index: u32) -> vec3<f32> {
    let clip = view.clip_from_world * vec4(world_in(o, instance, index, view.time), 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3((ndc.x * 0.5 + 0.5) * view.size.x, (0.5 - ndc.y * 0.5) * view.size.y, ndc.z);
}

@compute @workgroup_size(64)
fn raster_soft(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let index = counts[SOFT_BASE] + wid.y * groups.x + wid.x;
    if index >= min(counts[SOFT_COUNT], view.half_visible) {
        return;
    }
    let item = visible[index];
    let o = owners[item.x];
    let c = clusters[item.y];
    let instance = instances[o.instance];
    let vertex_count = min(c.counts & 0xffu, CLUSTER_LIMIT);
    let triangle_count = min((c.counts >> 8u) & 0xffu, CLUSTER_LIMIT);
    // Index loads overlap the transforms.
    let last = c.first_triangle + triangle_count - 1u;
    var packed: array<u32, SLOTS>;
    for (var k = 0u; k < SLOTS; k++) {
        packed[k] = triangles[min(c.first_triangle + lid + k * LANES, last)];
    }
    let both = two_sided(c);
    for (var v = lid; v < vertex_count; v += LANES) {
        screen[v] = screen_of(o, instance, c.first_vertex + v);
    }
    workgroupBarrier();
    for (var k = 0u; k < SLOTS; k++) {
        let slot = lid + k * LANES;
        if slot < triangle_count {
            let t = unpack_triangle(packed[k]);
            rasterize(Corners(screen[t.x], screen[t.y], screen[t.z]), (index << TRIANGLE_BITS) | slot, both);
        }
    }
}
