@group(2) @binding(13) var<storage, read> poses: array<Pose>;

/// Skin word: flag, joints, first weight.
const SKINNED: u32 = 0x80000000u;
const JOINT_MASK: u32 = 0xffu;
const SECOND_SHIFT: u32 = 8u;
const WEIGHT_SHIFT: u32 = 16u;
const WEIGHT_ONE: f32 = 255.0;

/// Skinning matrix rows.
struct Pose {
    rows: array<vec4<f32>, 3>,
}

/// Clip frames around now and their blend.
struct Frames {
    a: u32,
    b: u32,
    t: f32,
}

fn frames_at(i: Instance, time: f32) -> Frames {
    let count = f32(i.pose.z);
    let f = (time - i.play.x) * i.play.y * i.play.z;
    let wrapped = f - floor(f / count) * count;
    let a = min(u32(wrapped), i.pose.z - 1u);
    return Frames(a, (a + 1u) % i.pose.z, fract(wrapped));
}

/// Joint matrix between two frames.
fn joint(i: Instance, f: Frames, j: u32) -> array<vec4<f32>, 3> {
    let a = poses[i.pose.x + f.a * i.pose.y + j].rows;
    let b = poses[i.pose.x + f.b * i.pose.y + j].rows;
    return array(mix(a[0], b[0], f.t), mix(a[1], b[1], f.t), mix(a[2], b[2], f.t));
}

/// Skin word; zero when not playing.
fn skin_of(index: u32, i: Instance) -> u32 {
    if i.pose.z == 0u {
        return 0u;
    }
    return vertices[index].skin;
}

/// Mesh point, posed when animated.
fn local_at(index: u32, i: Instance, time: f32) -> vec3<f32> {
    let p = position_of(index);
    let skin = skin_of(index, i);
    if (skin & SKINNED) == 0u {
        return p;
    }
    let f = frames_at(i, time);
    let w = f32((skin >> WEIGHT_SHIFT) & JOINT_MASK) / WEIGHT_ONE;
    let first = place(joint(i, f, skin & JOINT_MASK), p);
    return mix(place(joint(i, f, (skin >> SECOND_SHIFT) & JOINT_MASK), p), first, w);
}

/// Mesh direction turned by the main joint.
fn turned_at(index: u32, i: Instance, d: vec3<f32>, time: f32) -> vec3<f32> {
    let skin = skin_of(index, i);
    if (skin & SKINNED) == 0u {
        return d;
    }
    return turn(joint(i, frames_at(i, time), skin & JOINT_MASK), d);
}
