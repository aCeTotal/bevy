@compute @workgroup_size(1)
fn prepare_clusters() {
    dispatch(CLUSTER_ARGS, min(atomicLoad(&counters.candidates), view.candidates_capacity), WORKGROUP);
}

/// Parent too coarse, self fine.
fn in_cut(o: Owner, c: Cluster) -> bool {
    if owner_projected(o, c.lod, c.errors.x) > view.threshold {
        return false;
    }
    return c.errors.y >= ENDLESS || owner_projected(o, c.parent, c.errors.y) > view.threshold;
}

/// Cluster facing wholly away.
fn backfacing(o: Owner, c: Cluster, margin: f32) -> bool {
    if c.cutoff >= 1.0 || margin > 0.0 || (o.flags & UNEVEN) != 0u || view.orthographic != 0u {
        return false;
    }
    let apex = place(o.rows, c.apex);
    let axis = normalize(turn(o.rows, c.axis));
    return dot(normalize(apex - view.eye), axis) >= c.cutoff;
}

fn append_visible(item: vec2<u32>, hard: bool) {
    var at: u32;
    if hard {
        at = atomicAdd(&counters.hard, 1u);
    } else {
        at = atomicAdd(&counters.soft, 1u);
    }
    if claim(at, view.half_visible) {
        visible[select(0u, view.half_visible, hard) + at] = item;
    }
}

fn defer_cluster(item: vec2<u32>) {
    if step.late != 0u {
        atomicAdd(&counters.clusters_culled[HIDDEN], 1u);
        return;
    }
    let at = atomicAdd(&counters.late_clusters, 1u);
    if claim(at, view.late_clusters_capacity) {
        late_clusters[at] = item;
    }
}

@compute @workgroup_size(64)
fn cull_clusters(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = thread(wid, groups, lid);
    if i >= min(atomicLoad(&counters.candidates), view.candidates_capacity) {
        return;
    }
    let item = candidates[i];
    let o = owners[item.x];
    let c = clusters[item.y];
    let margin = margin_of(o);
    if !in_cut(o, c) {
        return;
    }
    if outside(o.rows, c.min, c.max, margin) {
        atomicAdd(&counters.clusters_culled[OFF_SCREEN], 1u);
        return;
    }
    if backfacing(o, c, margin) {
        atomicAdd(&counters.clusters_culled[BACK_FACING], 1u);
        return;
    }
    if hidden(o, c.min, c.max, margin) {
        defer_cluster(item);
        return;
    }
    let center = place(o.rows, (c.min + c.max) * 0.5);
    let radius = length(c.max - c.min) * 0.5 * o.scale + margin;
    let distance = length(center - view.eye);
    let parallel = view.orthographic != 0u;
    let crosses_near = !parallel && distance - radius <= view.near;
    let pixels = 2.0 * radius * view.lod_scale / select(max(distance, view.near), 1.0, parallel);
    append_visible(item, crosses_near || pixels > view.soft_limit);
    atomicAdd(&counters.triangles, (c.counts >> 8u) & 0xffu);
}

@compute @workgroup_size(1)
fn prepare_raster() {
    let soft = min(atomicLoad(&counters.soft), view.half_visible);
    let hard = min(atomicLoad(&counters.hard), view.half_visible);
    counters.soft_base = counters.soft_done;
    counters.hard_base = counters.hard_done;
    dispatch(SOFT_ARGS, soft - counters.soft_base, 1u);
    dispatch(HARD_ARGS, hard - counters.hard_base, 1u);
    counters.soft_done = soft;
    counters.hard_done = hard;
    var deepest = 0u;
    // Clean level arguments for next.
    for (var level = 0u; level < MAX_LEVELS; level++) {
        steps_even[level * 3u] = 0u;
        steps_odd[level * 3u] = 0u;
        deepest = max(deepest, atomicLoad(&counters.levels[level]));
    }
    let demand = array<u32, PEAKS>(
        atomicLoad(&counters.owners),
        deepest,
        atomicLoad(&counters.candidates),
        max(atomicLoad(&counters.soft), atomicLoad(&counters.hard)),
        atomicLoad(&counters.late_instances),
        atomicLoad(&counters.late_nodes),
        atomicLoad(&counters.late_clusters),
    );
    for (var k = 0u; k < PEAKS; k++) {
        atomicMax(&peaks[k], demand[k]);
    }
}
