"""wires group: colored wire bundle (tray header -> external board), the external tan board with two trimmers and
the strip under it, the white cable (4-pin housing, two single wires back into the tray, one flat striped pair
with a free end), the colored yoke-lead loops inside the tray, and the cables over the main board (HV cable,
grey cable, red leads). All dimensions/paths from config/model/wires.toml (mm, world frame).
Helper scripts (uv, not Blender): fit_wire.py, ray_depth.py (paths from traces.json)."""

import math

import bmesh
import bpy
import lib
from mathutils import Matrix, Vector


def _mats(P):
    out = {}
    spec = P.get("insulation", {})
    for k, v in P["colors"].items():
        m = lib.mat_pbr(f"wires_{k}", tuple(v[:3]), roughness=v[3])
        if k in spec.get("names", []):
            m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = spec["specular_ior_level"]
        out[k] = m
    return out


def _stripe_mat(name, base, stripe, width_frac, period_mm, duty, length_mm, spec):
    """Insulation with a dashed stripe along the wire. Needs the curve's UV-as-generated coordinates:
    UV.x runs 0..1 along each spline, UV.y 0..1 around the circumference (use_uv_as_generated)."""
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Specular IOR Level"].default_value = spec
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    along = nt.nodes.new("ShaderNodeMath")
    along.operation = "MULTIPLY"
    along.inputs[1].default_value = length_mm / period_mm
    nt.links.new(sep.outputs["X"], along.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath")
    fr.operation = "FRACT"
    nt.links.new(along.outputs[0], fr.inputs[0])
    dash = nt.nodes.new("ShaderNodeMath")
    dash.operation = "LESS_THAN"
    dash.inputs[1].default_value = duty
    nt.links.new(fr.outputs[0], dash.inputs[0])
    band = nt.nodes.new("ShaderNodeMath")
    band.operation = "LESS_THAN"
    band.inputs[1].default_value = width_frac
    nt.links.new(sep.outputs["Y"], band.inputs[0])
    both = nt.nodes.new("ShaderNodeMath")
    both.operation = "MULTIPLY"
    nt.links.new(dash.outputs[0], both.inputs[0])
    nt.links.new(band.outputs[0], both.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*base, 1.0)
    mix.inputs["B"].default_value = (*stripe, 1.0)
    nt.links.new(both.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    m.diffuse_color = (*base, 1.0)
    return m


def _path_len(pts):
    return sum((Vector(pts[i + 1]) - Vector(pts[i])).length for i in range(len(pts) - 1))


def _tube_multi(name, splines, r_mm, coll, mats, mat_idx):
    """One curve object with several splines (e.g. a flat two-conductor ribbon), per-spline material index."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = 8
    cu.bevel_depth = r_mm * lib.MM
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    if hasattr(cu, "use_uv_as_generated"):
        cu.use_uv_as_generated = True
    for pts, mi in zip(splines, mat_idx):
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = Vector(lib.mm(*p))
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        sp.material_index = mi
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    for m in mats:
        ob.data.materials.append(m)
    return ob


def _offset_pair(pts, half_sep):
    """Two parallel copies of a path, offset +-half_sep mm sideways (horizontal normal to the tangent)."""
    P = [Vector(p) for p in pts]
    a, b = [], []
    for i in range(len(P)):
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        s = t.cross(Vector((0, 0, 1)))
        s = s.normalized() if s.length > 1e-6 else Vector((1, 0, 0))
        a.append(tuple(P[i] + s * half_sep))
        b.append(tuple(P[i] - s * half_sep))
    return a, b


def _board(name, corners, thick, coll, mat):
    """Plate from 4 top-surface corners (mm), extruded thick mm along -normal."""
    c = [Vector(lib.mm(*p)) for p in corners]
    n = (c[1] - c[0]).cross(c[3] - c[0]).normalized()
    if n.z < 0:
        n = -n
    bm = bmesh.new()
    top = [bm.verts.new(p) for p in c]
    bot = [bm.verts.new(p - n * thick * lib.MM) for p in c]
    bm.faces.new(top)
    bm.faces.new(bot[::-1])
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new([top[i], bot[i], bot[j], top[j]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat), n


def _cyl_on(name, c_mm, n, d, h, coll, mat):
    """Cylinder standing on a point of a plane with normal n (base at c_mm)."""
    ob = lib.cylinder(name, d / 2, h, (0, 0, h / 2), coll, mat=mat, verts=24)
    R = Vector((0, 0, 1)).rotation_difference(n).to_matrix().to_4x4()
    ob.data.transform(Matrix.Translation(Vector(lib.mm(*c_mm))) @ R)
    return ob


def pot_frame(c_mm, n, h, e1):
    """Top-center (mm) and in-plane axes (e_u along the board's long edge, e_v = n x e_u) of a pot top."""
    eu = (e1 - n * e1.dot(n)).normalized()
    ev = n.cross(eu)
    top = Vector(c_mm) + n * h
    return top, eu, ev


def _disc_label(name, center_mm, eu, ev, n, r_mm, coll, mat, offset_mm=0.05, seg=32):
    """Textured disc (UV = planar map of the square TL..BL = c -/+ eu r +/- ev r, image top along +ev)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    c = Vector(center_mm) + n * offset_mm
    vs = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        p = c + (eu * math.cos(a) + ev * math.sin(a)) * r_mm
        vs.append((bm.verts.new(p * lib.MM), (0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))))
    f = bm.faces.new([v for v, _ in vs])
    for loop, (_, t) in zip(f.loops, vs):
        loop[uv].uv = t
    return lib._obj_from_bm(name, bm, coll, mat)


def _tex_mat(name, tex, fallback):
    p = lib.ROOT / tex
    return lib.mat_image(name, tex, roughness=0.6) if p.exists() else fallback


def _housing(P, coll, mat, dark):
    k = P["ext_conn4"]
    sx = k["size"][0]
    ob = lib.box("wires.ext_conn4", k["size"], (0, 0, 0), coll, mat, bevel_mm=0.3)
    sl = k["slot"]  # slots on the +x face (mating side), one per pin along z
    for i, dz in enumerate(k["pin_dz"]):
        cut = lib.box(f"_slot{i}", (sl[2] * 2, sl[0], sl[1]), (sx / 2, 0, dz), coll)
        lib.boolean(ob, cut)
    lib.place(ob, (k["cx"], k["cy"], k["cz"]), (0, 0, k["yaw_deg"]))
    # dark pin contacts inside the slots: one object (4 small boxes)
    bm = bmesh.new()
    for dz in k["pin_dz"]:
        g = bmesh.ops.create_cube(bm, size=1.0)["verts"]
        bmesh.ops.scale(bm, vec=Vector(lib.mm(0.3, 0.8, 0.8)), verts=g)
        bmesh.ops.translate(bm, vec=Vector(lib.mm(sx / 2 - sl[2] + 0.2, 0, dz)), verts=g)
    pins = lib._obj_from_bm("wires.ext_conn4_pins", bm, coll, dark)
    lib.place(pins, (k["cx"], k["cy"], k["cz"]), (0, 0, k["yaw_deg"]))
    return ob


def build(P: dict, coll) -> None:
    M = _mats(P)
    eb = P["ext_board"]
    _, n = _board("wires.ext_board", eb["corners"], eb["thick"], coll, M["board"])
    C = [Vector(p) for p in eb["corners"]]
    lib.label_quad("wires.label_ext_board", eb["corners"], coll,
                   _tex_mat("wires_tex_ext_board", eb.get("texture", ""), M["board"]), offset_mm=0.05)
    e1 = (C[1] - C[0]).normalized()
    for p in P.get("ext_pot", []):
        _cyl_on(f"wires.{p['name']}", p["c"], n, p["d"], p["h"], coll, M["pot"])
        top, eu, ev = pot_frame(p["c"], n, p["h"], e1)
        if p.get("texture") and (lib.ROOT / p["texture"]).exists():
            _disc_label(f"wires.label_{p['name']}", top, eu, ev, n, p["d"] / 2, coll,
                        lib.mat_image(f"wires_tex_{p['name']}", p["texture"]))
    s = P.get("ext_strip")
    if s:
        ob = lib.box("wires.ext_strip", s["size"], (0, 0, 0), coll, M["board"])
        lib.place(ob, s["c"], (0, s.get("tilt_deg", 0.0), s.get("yaw_deg", 0.0)))

    b = P["bundle"]
    for col, w in b["wires"].items():
        lib.tube_path(f"wires.bundle_{col}", _wire_pts(b, w), b["r"], coll, M[col])

    _housing(P, coll, M["conn"], M.get("grey", M["conn"]))

    spec = P.get("insulation", {}).get("specular_ior_level", 0.3)
    st = P.get("stripes", {})
    white = tuple(P["colors"]["white"][:3])
    pink = tuple(st.get("color", [0.6, 0.12, 0.12]))
    for name, w in P.get("wire", {}).items():
        if name == "white_pair":
            a, bb = _offset_pair(w["pts"], w["sep"] / 2)
            L = _path_len(w["pts"])
            ma = _stripe_mat("wires_stripe_dash", white, pink, st["width_frac"], st["dash_period"], st["dash_duty"],
                             L, spec)
            mb = _stripe_mat("wires_stripe_x", white, pink, st["width_frac"], st["x_period"], st["x_duty"], L, spec)
            _tube_multi("wires.white_pair", [a, bb], w["r"], coll, [ma, mb], [0, 1])
        elif w.get("stripe"):
            L = _path_len(w["pts"])
            m = _stripe_mat(f"wires_stripe_{name}", white, pink, st["width_frac"], st["dash_period"],
                            st["dash_duty"], L, spec)
            _tube_multi(f"wires.{name}", [w["pts"]], w["r"], coll, [m], [0])
        else:
            lib.tube_path(f"wires.{name}", w["pts"], w["r"], coll, M[w.get("color", "white")])

    t = P.get("loop_tie")
    if t:
        ob = lib.box("wires.loop_tie", t["size"], (0, 0, 0), coll, M["cream"], bevel_mm=0.4)
        lib.place(ob, t["c"], (0, 0, t["yaw_deg"]))


def _wire_pts(b, w):
    if "pts" in w:
        return w["pts"]
    mid = [[a + d for a, d in zip(p, w["offset"])] for p in b["path"]]
    return [w["start"], *mid, w["end"]]
