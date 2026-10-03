"""Fiber spaghetti generator: a seeded, deterministic pile of patch cords spilling out of a pulled-out tray (storyboard S6).

Usage from Blender Python (also embedded as a text datablock in fiber_spaghetti_generator.blend):

    import fiber_spaghetti as FS
    ob = FS.build_pile(name="pile", count=9072, seed=44, length_range=(1.0, 3.0), spill_radius=1.2, slack=1.0,
                       palette=FS.DEFAULT_PALETTE, parent=None, collection=coll)

All lengths in metres. The result is ONE mesh object holding the strand polylines (edges only) plus a Geometry Nodes modifier
(NG_fiber_spaghetti: Mesh to Curve -> Curve to Mesh with a round profile, noise squirm). Cost: about count * 23 * 2 * profile_resolution
triangles at evaluation time (9,072 strands at profile resolution 4 is about 1.7 M triangles evaluated, 0 stored).
"""
import math

import bmesh
import bpy
import numpy as np

DEFAULT_PALETTE = [(1.0, 0.82, 0.05), (1.0, 0.82, 0.05), (1.0, 0.82, 0.05), (1.0, 0.82, 0.05), (1.0, 0.82, 0.05), (0.0, 0.62, 0.62), (0.0, 0.62, 0.62), (0.95, 0.35, 0.03), (0.05, 0.15, 0.8),
                   (0.1, 0.7, 0.2), (0.85, 0.1, 0.1), (0.9, 0.9, 0.88), (0.5, 0.1, 0.7)]   # yellow-heavy: SMF patch cords, some OM3/OM4 aqua, colour-coded runs


def generate_paths(count=9072, seed=44, length_range=(1.0, 3.0), spill_radius=1.2, slack=1.0, n_pts=32, tray_halfwidth=0.22,
                   tray_depth=0.6, tray_z=0.35, fiber_radius=0.001):
    """Return (paths (count, n_pts, 3), u (count, n_pts), colour_index (count,)).

    Frame: floor at z = 0, tray lip at y = 0 (tray body at y > 0, spill toward -Y), strands start inside the tray, cross the lip,
    fall to the floor around (0, -spill_radius/2) and wander with `slack` (0 = straight, 2 = very loopy).
    """
    rng = np.random.default_rng(seed)
    K = 14
    paths = np.zeros((count, K, 3))
    # start in the tray, lip crossing
    paths[:, 0, 0] = rng.uniform(-tray_halfwidth, tray_halfwidth, count)
    paths[:, 0, 1] = rng.uniform(0.15, tray_depth, count)
    paths[:, 0, 2] = tray_z + rng.uniform(0.0, 0.05, count)
    lip_x = paths[:, 0, 0] * rng.uniform(0.8, 1.0, count)
    paths[:, 1] = np.stack([lip_x, np.zeros(count) - 0.01, tray_z + 0.03 + rng.uniform(0, 0.03, count)], axis=1)
    # landing point on the floor
    cx, cy = 0.0, -spill_radius * 0.55
    ang = rng.uniform(0, 2 * np.pi, count)
    rad = spill_radius * 0.5 * np.sqrt(rng.uniform(0.02, 1.0, count))
    lx = cx + rad * np.cos(ang) * 1.0
    ly = np.minimum(cy + rad * np.sin(ang) * 0.9, -0.05)
    cur = np.stack([lx, ly], axis=1)
    L_tot = rng.uniform(length_range[0], length_range[1], count)
    fan = np.hypot(lx - lip_x, ly) if False else 0.0
    L_floor = np.maximum(L_tot - 0.6, 0.4) * (0.8 + 0.5 * slack)
    step = L_floor / (K - 3)
    th = rng.uniform(0, 2 * np.pi, count)
    heading_sigma = 0.3 + 0.8 * slack
    pts2 = [cur.copy()]
    for k in range(K - 3):
        th = th + rng.normal(0, heading_sigma, count)
        nxt = pts2[-1] + np.stack([np.cos(th), np.sin(th)], axis=1) * step[:, None]
        # keep inside the spill radius: pull toward the centre when outside
        d = nxt - np.array([cx, cy])
        r = np.linalg.norm(d, axis=1)
        out = r > spill_radius
        nxt[out] = np.array([cx, cy]) + d[out] / r[out, None] * spill_radius * 0.98
        th = np.where(out, np.arctan2(-d[:, 1], -d[:, 0]) + rng.normal(0, 0.6, count), th)
        pts2.append(nxt)
    pts2 = np.array(pts2)   # (K-2, count, 2)
    for k in range(K - 2):
        xy = pts2[k]
        r = np.linalg.norm(xy - np.array([cx, cy]), axis=1)
        h = fiber_radius + (0.14 * (1 - np.clip(r / spill_radius, 0, 1)) ** 1.4) * rng.uniform(0.15, 1.0, count)
        paths[:, 2 + k] = np.stack([xy[:, 0], xy[:, 1], h], axis=1)
    # intermediate drop point between the lip and the first floor point (gravity arc)
    mid = (paths[:, 1] + paths[:, 2]) / 2
    mid[:, 2] = np.maximum(paths[:, 2, 2], tray_z * 0.45 + rng.uniform(-0.05, 0.05, count))
    # resample: Catmull-Rom through the K waypoints, then equal-count resample
    out = np.zeros((count, n_pts, 3))
    per = 5
    t = np.linspace(0, 1, per, endpoint=False)[None, :, None]
    segs = []
    P = np.concatenate([paths[:, :2], mid[:, None], paths[:, 2:]], axis=1)   # (count, K+1, 3)
    Q = np.concatenate([2 * P[:, :1] - P[:, 1:2], P, 2 * P[:, -1:] - P[:, -2:-1]], axis=1)
    for i in range(1, Q.shape[1] - 2):
        p0, p1, p2, p3 = Q[:, i - 1][:, None], Q[:, i][:, None], Q[:, i + 1][:, None], Q[:, i + 2][:, None]
        segs.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    full = np.concatenate(segs + [P[:, -1:][:, :1]], axis=1)                 # (count, M, 3)
    M_ = full.shape[1]
    s_ = np.linspace(0, M_ - 1, n_pts)
    i0 = np.floor(s_).astype(int)
    i1 = np.minimum(i0 + 1, M_ - 1)
    f = (s_ - i0)[None, :, None]
    out = full[:, i0] * (1 - f) + full[:, i1] * f
    out[:, :, 2] = np.maximum(out[:, :, 2], fiber_radius)
    u = np.broadcast_to(np.linspace(0, 1, n_pts)[None, :], (count, n_pts))
    col = rng.integers(0, 1 << 30, count)
    return out, u, col


def _material(name="MAT_interconnect_fiber_pile", glow=0.25):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "fiber_color"
    at.attribute_type = "GEOMETRY"
    nt.links.new(at.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(at.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = glow
    b.inputs["Roughness"].default_value = 0.4
    return m


def make_node_group(name="NG_fiber_spaghetti"):
    ng = bpy.data.node_groups.get(name)
    if ng:
        return ng
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    it = ng.interface
    it.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    it.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    s = it.new_socket("Fiber Radius", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 0.001; s.min_value = 1e-5
    s = it.new_socket("Profile Resolution", in_out="INPUT", socket_type="NodeSocketInt"); s.default_value = 4; s.min_value = 3; s.max_value = 16
    s = it.new_socket("Squirm Amplitude", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 0.0; s.min_value = 0.0
    s = it.new_socket("Squirm Speed", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 1.0
    s = it.new_socket("Squirm Noise Scale", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 3.0
    it.new_socket("Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    N = ng.nodes
    L = ng.links
    gi = N.new("NodeGroupInput")
    go = N.new("NodeGroupOutput")
    pos = N.new("GeometryNodeInputPosition")
    nattr = N.new("GeometryNodeInputNamedAttribute")
    nattr.data_type = "FLOAT"
    nattr.inputs["Name"].default_value = "squirm_w"
    tm = N.new("GeometryNodeInputSceneTime")
    vscale = N.new("ShaderNodeVectorMath"); vscale.operation = "SCALE"
    wmul = N.new("ShaderNodeMath"); wmul.operation = "MULTIPLY"
    noise = N.new("ShaderNodeTexNoise"); noise.noise_dimensions = "4D"
    noise.inputs["Detail"].default_value = 1.0
    sub = N.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.5)
    a2 = N.new("ShaderNodeMath"); a2.operation = "MULTIPLY"      # amp * weight
    a3 = N.new("ShaderNodeMath"); a3.operation = "MULTIPLY"; a3.inputs[1].default_value = 2.0
    vs2 = N.new("ShaderNodeVectorMath"); vs2.operation = "SCALE"
    setp = N.new("GeometryNodeSetPosition")
    m2c = N.new("GeometryNodeMeshToCurve")
    scr = N.new("GeometryNodeSetCurveRadius")
    circ = N.new("GeometryNodeCurvePrimitiveCircle")
    circ.inputs["Radius"].default_value = 1.0
    c2m = N.new("GeometryNodeCurveToMesh")
    smooth = N.new("GeometryNodeSetShadeSmooth")
    setm = N.new("GeometryNodeSetMaterial")
    L.new(pos.outputs[0], vscale.inputs[0])
    L.new(gi.outputs["Squirm Noise Scale"], vscale.inputs[3])
    L.new(vscale.outputs[0], noise.inputs["Vector"])
    L.new(tm.outputs["Seconds"], wmul.inputs[0])
    L.new(gi.outputs["Squirm Speed"], wmul.inputs[1])
    L.new(wmul.outputs[0], noise.inputs["W"])
    L.new(noise.outputs["Color"], sub.inputs[0])
    L.new(gi.outputs["Squirm Amplitude"], a2.inputs[0])
    L.new(nattr.outputs["Attribute"], a2.inputs[1])
    L.new(a2.outputs[0], a3.inputs[0])
    L.new(sub.outputs[0], vs2.inputs[0])
    L.new(a3.outputs[0], vs2.inputs[3])
    L.new(gi.outputs["Geometry"], setp.inputs["Geometry"])
    L.new(vs2.outputs[0], setp.inputs["Offset"])
    L.new(setp.outputs["Geometry"], m2c.inputs["Mesh"])
    L.new(m2c.outputs["Curve"], scr.inputs["Curve"])
    L.new(gi.outputs["Fiber Radius"], scr.inputs["Radius"])
    L.new(gi.outputs["Profile Resolution"], circ.inputs["Resolution"])
    L.new(scr.outputs["Curve"], c2m.inputs["Curve"])
    L.new(circ.outputs["Curve"], c2m.inputs["Profile Curve"])
    L.new(c2m.outputs["Mesh"], smooth.inputs["Geometry"])
    L.new(smooth.outputs["Geometry"], setm.inputs["Geometry"])
    L.new(gi.outputs["Material"], setm.inputs["Material"])
    L.new(setm.outputs["Geometry"], go.inputs["Geometry"])
    return ng


def _identifier(ng, name):
    for it in ng.interface.items_tree:
        if it.item_type == "SOCKET" and it.name == name and it.in_out == "INPUT":
            return it.identifier
    raise KeyError(name)


def build_pile(name="fiber_pile", count=9072, seed=44, length_range=(1.0, 3.0), spill_radius=1.2, slack=1.0, palette=DEFAULT_PALETTE,
               fiber_radius=0.001, profile_res=4, n_pts=32, parent=None, collection=None, squirm_amp=0.0, squirm_speed=1.0):
    paths, u, col = generate_paths(count, seed, length_range, spill_radius, slack, n_pts=n_pts, fiber_radius=fiber_radius)
    count_, n, _ = paths.shape
    me = bpy.data.meshes.new(name)
    me.vertices.add(count_ * n)
    me.vertices.foreach_set("co", paths.astype(np.float32).ravel())
    ne = count_ * (n - 1)
    me.edges.add(ne)
    idx = np.arange(count_ * n).reshape(count_, n)
    ev = np.stack([idx[:, :-1], idx[:, 1:]], axis=-1).reshape(-1, 2)
    me.edges.foreach_set("vertices", ev.astype(np.int32).ravel())
    pal = np.array(palette, dtype=np.float32)
    ci = col % len(pal)
    colors = np.concatenate([pal[ci], np.ones((count_, 1), np.float32)], axis=1)
    colors = np.repeat(colors, n, axis=0)
    a = me.attributes.new("fiber_color", "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", colors.ravel())
    w = np.clip(u, 0, 1) ** 1.5
    a2 = me.attributes.new("squirm_w", "FLOAT", "POINT")
    a2.data.foreach_set("value", w.astype(np.float32).ravel())
    me.update()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if parent is not None:
        ob.parent = parent
    ng = make_node_group()
    md = ob.modifiers.new("fiber_gn", "NODES")
    md.node_group = ng
    md[_identifier(ng, "Fiber Radius")] = float(fiber_radius)
    md[_identifier(ng, "Profile Resolution")] = int(profile_res)
    md[_identifier(ng, "Squirm Amplitude")] = float(squirm_amp)
    md[_identifier(ng, "Squirm Speed")] = float(squirm_speed)
    md[_identifier(ng, "Material")] = _material()
    return ob


def evaluated_tris(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    m = ev.to_mesh()
    n = sum(len(p.vertices) - 2 for p in m.polygons)
    ev.to_mesh_clear()
    return n
