struct Resolved {
    @builtin(frag_depth) depth: f32,
    @location(0) normal: vec4<f32>,
    @location(1) motion: vec2<f32>,
}

/// Corner travel since last frame.
fn travel(o: Owner, index: u32) -> vec3<f32> {
    let instance = instances[o.instance];
    let base = place(o.rows, local_at(index, instance, view.time));
    let earlier = place(o.rows, local_at(index, instance, view.previous_time));
    let root = root_of(o);
    let shift = place(instance.motion, earlier);
    let gone = sway(earlier + shift, root + place(instance.motion, root), o.wind, view.previous_time);
    return base - earlier + sway(base, root, o.wind, view.time) - gone - shift;
}

/// Precise even far from origin.
fn motion_of(s: Surface) -> vec2<f32> {
    let w = s.weights.weights;
    let now = world_position(s);
    var moved = vec3(0.0);
    for (var k = 0u; k < 3u; k++) {
        moved += travel(s.owner, s.corners[k]) * w[k];
    }
    let current = view.unjittered_clip_from_world * vec4(now, 1.0);
    let previous = view.previous_clip_from_world * vec4(now, 1.0) - view.previous_clip_from_world * vec4(moved, 0.0);
    return (current.xy / current.w - previous.xy / previous.w) * vec2(0.5, -0.5);
}

@fragment
fn resolve(@builtin(position) p: vec4<f32>) -> Resolved {
    let v = visible_at(vec2<u32>(p.xy));
    if id_of(v) == BACKGROUND {
        discard;
    }
    let s = surface_at(id_of(v), p.xy);
    var normal = surface_normal(s);
    if two_sided(s.cluster) && dot(normal, view.eye - s.world[0]) < 0.0 {
        normal = -normal;
    }
    return Resolved(depth_of(v), vec4(normal * 0.5 + 0.5, 1.0), motion_of(s));
}

