/// Late pass retests; misses hide.
fn defer_instance(slot: u32) {
    if step.late != 0u {
        atomicAdd(&counters.objects_culled[HIDDEN], 1u);
        return;
    }
    let at = atomicAdd(&counters.late_instances, 1u);
    if claim(at, view.late_instances_capacity) {
        late_instances[at] = slot;
    }
}

/// Owner for an admitted instance.
fn admit(slot: u32) {
    let instance = instances[slot];
    if instance.root == HOLE {
        return;
    }
    let root = nodes[instance.root];
    let flags = select(0u, UNEVEN, uneven(instance.rows));
    let o = Owner(instance.rows, largest_scale(instance.rows), slot, flags, 0u, instance.wind);
    let margin = drift(o, instance);
    if outside(instance.rows, root.min, root.max, margin) {
        atomicAdd(&counters.objects_culled[OFF_SCREEN], 1u);
        return;
    }
    if hidden(o, root.min, root.max, margin) {
        defer_instance(slot);
        return;
    }
    let index = new_owner(o);
    if index != HOLE {
        push_node(0u, vec2(index, instance.root));
    }
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = thread(wid, groups, lid);
    if step.late == 1u {
        if i < min(counters.late_count, view.late_instances_capacity) {
            admit(late_instances[i]);
        }
        return;
    }
    if i >= view.instance_extent + view.dynamic_count {
        return;
    }
    admit(select(view.static_capacity + i - view.instance_extent, i, i < view.instance_extent));
}

/// Enters part unless proxy suffices.
fn enter_part(o: Owner, index: u32, next: u32) {
    let part = parts[index];
    let center = place(o.rows, part.parent.xyz);
    let radius = part.parent.w * o.scale;
    let error = part.error * o.scale;
    if projected(vec4(center, radius), error, instances[o.instance].coarse) <= view.threshold {
        return;
    }
    let rows = compose(o.rows, part.rows);
    let flags = o.flags | select(0u, UNEVEN, uneven(part.rows));
    let child = new_owner(Owner(rows, o.scale * largest_scale(part.rows), o.instance, flags, 0u, part_wind(part, o.wind)));
    if child != HOLE {
        push_node(next, vec2(child, part.root));
    }
}

/// Slots children need, ranges kept.
struct Demand {
    nodes: u32,
    clusters: u32,
    /// Bit per child: range drawn.
    keep: vec2<u32>,
}

/// Range may hold cut clusters.
fn wanted(o: Owner, first: u32) -> bool {
    let c = clusters[first];
    if c.errors.y < ENDLESS && owner_projected(o, c.parent, c.errors.y) <= view.threshold {
        return false;
    }
    let farthest = length(place(o.rows, c.parent.xyz) - view.eye) + c.parent.w * o.scale;
    return projected_at(farthest, c.errors.x * o.scale, instances[o.instance].coarse) <= view.threshold;
}

fn demand(o: Owner, node: Node) -> Demand {
    var d = Demand(0u, 0u, vec2(0u));
    for (var k = 0u; k < min(node.child_count, MAX_CHILDREN); k++) {
        let child = children[node.first_child + k];
        let kind = child.x >> INDEX_BITS;
        d.nodes += u32(kind == CHILD_NODE);
        if kind == CHILD_CLUSTERS && wanted(o, child.x & INDEX_MASK) {
            d.clusters += child.y;
            d.keep[k / 32u] |= 1u << (k % 32u);
        }
    }
    return d;
}

fn store_clusters(first: vec2<u32>, count: u32, at: u32) {
    for (var k = 0u; k < count; k++) {
        if claim(at + k, view.candidates_capacity) {
            candidates[at + k] = first + vec2(0u, k);
        }
    }
}

/// Pushes children in bulk.
fn visit_children(owner: u32, o: Owner, node: Node, next: u32) {
    let need = demand(o, node);
    var node_at = reserve_nodes(next, need.nodes);
    var cluster_at = 0u;
    if need.clusters > 0u {
        cluster_at = atomicAdd(&counters.candidates, need.clusters);
    }
    for (var k = 0u; k < min(node.child_count, MAX_CHILDREN); k++) {
        let child = children[node.first_child + k];
        let kind = child.x >> INDEX_BITS;
        let index = child.x & INDEX_MASK;
        if kind == CHILD_NODE {
            store_node(next, node_at, vec2(owner, index));
            node_at += 1u;
            continue;
        }
        if kind == CHILD_PART {
            enter_part(o, index, next);
            continue;
        }
        if (need.keep[k / 32u] & (1u << (k % 32u))) == 0u {
            continue;
        }
        store_clusters(vec2(owner, index), child.y, cluster_at);
        cluster_at += child.y;
    }
}

/// Defers hidden node to late.
fn defer_node(item: vec2<u32>) {
    if step.late != 0u {
        atomicAdd(&counters.nodes_culled[HIDDEN], 1u);
        return;
    }
    let at = atomicAdd(&counters.late_nodes, 1u);
    if claim(at, view.late_nodes_capacity) {
        late_nodes[at] = item;
    }
}

@compute @workgroup_size(64)
fn cull_nodes(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = thread(wid, groups, lid);
    if i >= min(atomicLoad(&counters.levels[step.level]), view.queue_capacity) {
        return;
    }
    let item = queue[(step.level & 1u) * view.queue_capacity + i];
    let o = owners[item.x];
    let node = nodes[item.y];
    let margin = margin_of(o);
    if outside(o.rows, node.min, node.max, margin) {
        atomicAdd(&counters.nodes_culled[OFF_SCREEN], 1u);
        return;
    }
    if node.error.x < ENDLESS && owner_projected(o, node.lod, node.error.x) <= view.threshold {
        return;
    }
    if hidden(o, node.min, node.max, margin) {
        defer_node(item);
        return;
    }
    visit_children(item.x, o, node, step.level + 1u);
}

/// Seeds the late pass.
@compute @workgroup_size(1)
fn prepare_late() {
    let nodes_left = min(atomicLoad(&counters.late_nodes), view.late_nodes_capacity);
    let clusters_left = min(atomicLoad(&counters.late_clusters), view.late_clusters_capacity);
    counters.late_count = min(atomicLoad(&counters.late_instances), view.late_instances_capacity);
    for (var level = 1u; level < MAX_LEVELS; level++) {
        atomicStore(&counters.levels[level], 0u);
    }
    atomicStore(&counters.levels[0], nodes_left);
    atomicStore(&counters.candidates, clusters_left);
    steps_even[0] = (nodes_left + WORKGROUP - 1u) / WORKGROUP;
    steps_odd[LATE_SLOT * 3u] = (counters.late_count + WORKGROUP - 1u) / WORKGROUP;
    dispatch(SEED_ARGS, max(nodes_left, clusters_left), WORKGROUP);
}

@compute @workgroup_size(64)
fn seed_late(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = thread(wid, groups, lid);
    if i < min(atomicLoad(&counters.late_nodes), view.late_nodes_capacity) {
        queue[i] = late_nodes[i];
    }
    if i < min(atomicLoad(&counters.late_clusters), view.late_clusters_capacity) {
        candidates[i] = late_clusters[i];
    }
}
