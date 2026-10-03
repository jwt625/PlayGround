"""Eggs and plate family: egg, cracked egg, fried egg (shape keys + vertex groups + UVs for the egg-fry shader), plate, spatula,
frying pan, plate with eggs, pan with egg. Clay style.

Run: Blender -b --python scripts/assets/props/build_eggs_plate.py -- assets/components/props
Each item is its own ASSET_<id> collection with ROOT_<id>; roots are laid out along +X in the file (reset root location on append).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import props_lib as L  # noqa: E402
from props_lib import C, mm  # noqa: E402

OUT, PREV = L.outdirs()
PI = math.pi
EGG_A, EGG_B = 28.5, 22.0  # half length, half width (mm): 57 x 44 mm (task brief)
EGG_W = 3.5  # asymmetry parameter (Hugelschaeger-type egg curve), design choice
R_MAX = 50.0  # fully spread white radius (mm), estimate: fried large egg about 100 mm across
NS = 96
NR = 36
NYS = 48
NYR = 10
RY = 15.0  # yolk radius (mm), estimate


def egg_r(h):
    """Egg outer radius at axial position h (mm, -a..a); blunt end at -a, pointed end at +a."""
    a, b, w = EGG_A, EGG_B, EGG_W
    v = (a * a - h * h) / (a * a + 2 * w * h + w * w)
    return b * math.sqrt(max(v, 0.0))


def smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def mats():
    return dict(
        shell=L.clay("MAT_props_egg_shell", (0.93, 0.89, 0.78), bump=0.25, scale=400, ring=60),
        white=L.clay("MAT_props_egg_white", (0.96, 0.96, 0.94), bump=0.15, scale=300, rough=0.45, ring=40),
        yolk=L.clay("MAT_props_egg_yolk", (1.0, 0.58, 0.04), bump=0.1, scale=300, rough=0.35, ring=40),
        plate=L.clay("MAT_props_plate_clay", (0.88, 0.9, 0.95), bump=0.3, scale=120, ring=25),
        steel=C.principled("MAT_props_spatula_steel", (0.7, 0.7, 0.72), metallic=1.0, rough=0.3),
        handle=L.clay("MAT_props_spatula_handle", (0.45, 0.22, 0.1), bump=0.3, scale=150, ring=30),
        pan=L.clay("MAT_props_pan_clay", (0.2, 0.2, 0.22), bump=0.25, scale=150, ring=30),
        pan_handle=L.clay("MAT_props_pan_handle", (0.08, 0.08, 0.09), bump=0.25, scale=150, ring=30),
    )


# ------------------------------------------------------------------ egg
def egg_profile(n=40, h0=-EGG_A, h1=EGG_A):
    hs = [h0 + (h1 - h0) * (0.5 - 0.5 * math.cos(PI * i / n)) for i in range(n + 1)]
    return [(egg_r(h), h) for h in hs]


def egg_obj(name, mat):
    prof = [(mm(r), mm(h)) for r, h in egg_profile(44)]
    xf = Matrix.Translation((0, 0, mm(EGG_B))) @ Matrix.Rotation(PI / 2, 4, "Y")
    return L.lathe(name, prof, 64, xf=xf, mat=mat)


def shell_half(name, mat, which, jag_seed):
    """Half egg shell (thickness 0.9 mm) with a jagged break edge; which = 'blunt' or 'point'. Opening up (+Z), origin at bottom."""
    rnd = random.Random(jag_seed)
    ph = [rnd.uniform(0, 2 * PI) for _ in range(3)]
    n = 72
    nr = 24
    cut0 = 4.0 if which == "point" else -4.0

    def cut(theta):
        return cut0 + 3.2 * math.sin(7 * theta + ph[0]) * 1.0 + 2.0 * math.sin(13 * theta + ph[1]) + 1.2 * math.sin(23 * theta + ph[2])
    t = 0.9
    bm = bmesh.new()
    outer, inner = [], []
    for j in range(nr + 1):
        f = j / nr
        ro, ri, zo, zi = [], [], [], []
        for i in range(n):
            th = 2 * PI * i / n
            if which == "point":
                h_end, h_cut = EGG_A, cut(th)
                h = h_end + (h_cut - h_end) * f
                h = min(h, EGG_A - 1e-6)
                # axial coordinate -> local z (tip at bottom): z = EGG_A - h
                z = EGG_A - h
            else:
                h_end, h_cut = -EGG_A, cut(th)
                h = h_end + (h_cut - h_end) * f
                h = max(h, -EGG_A + 1e-6)
                z = h + EGG_A
            r = egg_r(h)
            ring_o = Vector((r * math.cos(th), r * math.sin(th), z))
            # inner surface: offset inward along radius and up (approximate normal offset)
            rin = max(r - t, 0.0)
            zin = z + (t if j == 0 else 0.0) * (1 - f)
            ring_i = Vector((rin * math.cos(th), rin * math.sin(th), zin))
            ro.append(ring_o)
            ri.append(ring_i)
        if j == 0:
            outer.append(bm.verts.new(Vector((0, 0, ro[0].z)) * 0.001))
            inner.append(bm.verts.new(Vector((0, 0, ri[0].z)) * 0.001))
        else:
            outer.append([bm.verts.new(p * 0.001) for p in ro])
            inner.append([bm.verts.new(p * 0.001) for p in ri])
    for j in range(nr):
        a, b = outer[j], outer[j + 1]
        c, d = inner[j], inner[j + 1]
        for i in range(n):
            k = (i + 1) % n
            if j == 0:
                bm.faces.new((a, b[i], b[k]))
                bm.faces.new((c, d[k], d[i]))
            else:
                bm.faces.new((a[i], b[i], b[k], a[k]))
                bm.faces.new((c[k], d[k], d[i], c[i]))
    for i in range(n):  # rim
        k = (i + 1) % n
        bm.faces.new((outer[-1][i], outer[-1][k], inner[-1][k], inner[-1][i]))
    o = L.obj_from_bm(name, bm, mat, True, True, None)
    # shift so that the lowest point sits at z = 0
    zmin = min(v.co.z for v in o.data.vertices)
    o.data.transform(Matrix.Translation((0, 0, -zmin)))
    return o


# ------------------------------------------------------------------ fried egg
def outline_R(theta, rs):
    return rs * (1 + 0.07 * math.sin(2 * theta + 1.3) + 0.05 * math.sin(3 * theta + 0.4) + 0.03 * math.sin(5 * theta + 2.1))


BUBBLES = None


def bubbles():
    global BUBBLES
    if BUBBLES is None:
        rnd = random.Random(7)
        BUBBLES = []
        while len(BUBBLES) < 7:
            r = rnd.uniform(22, 44)
            th = rnd.uniform(0, 2 * PI)
            sig = rnd.uniform(2.2, 3.6)
            h = rnd.uniform(1.8, 3.0)
            c = (r * math.cos(th), r * math.sin(th))
            if all(math.hypot(c[0] - b[0][0], c[1] - b[0][1]) > 11 for b in BUBBLES):
                BUBBLES.append((c, sig, h))
    return BUBBLES


def white_z(rho, th, spread, crisp, bub):
    tc = 15.0 + (5.5 - 15.0) * spread
    te = 5.0 + (0.9 - 5.0) * spread
    z = te + (tc - te) * (1 - rho * rho) ** 1.5
    z += crisp * (1.0 * smooth(0.86, 1.0, rho) + 0.6 * smooth(0.9, 1.0, rho) * math.sin(17 * th + 0.8))
    if bub:
        x, y = rho * R_MAX * math.cos(th), rho * R_MAX * math.sin(th)
        for (cx, cy), sig, h in bubbles():
            z += bub * h * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig))
    return z


def egg_state(spread, crisp, dome, bub):
    """Vertex coordinates (mm, list of Vector) of the fried egg mesh for a state. Order: white top (centre, rings), white bottom ring + centre, yolk (apex, rings)."""
    pts = []
    rs = 24.0 + (R_MAX - 24.0) * spread
    pts.append(Vector((0, 0, white_z(0, 0, spread, crisp, bub))))
    for j in range(1, NR + 1):
        rho = j / NR
        for i in range(NS):
            th = 2 * PI * i / NS
            R = outline_R(th, rs)
            R += crisp * (1.2 * math.sin(11 * th) * smooth(0.85, 1.0, rho) - 1.0 * smooth(0.9, 1.0, rho))
            pts.append(Vector((rho * R * math.cos(th), rho * R * math.sin(th), white_z(rho, th, spread, crisp, bub))))
    for i in range(NS):
        th = 2 * PI * i / NS
        R = outline_R(th, rs) + crisp * (1.2 * math.sin(11 * th) - 1.0)
        pts.append(Vector((R * math.cos(th), R * math.sin(th), 0.0)))
    pts.append(Vector((0, 0, 0)))
    # yolk
    ry = RY * (0.92 + 0.08 * spread)
    rho_y = ry / rs
    zbase = white_z(min(rho_y, 1.0), 0.0, spread, 0.0, 0.0) - 0.6
    H = 3.0 + 6.5 * dome
    pts.append(Vector((0, 0, zbase + H)))
    for j in range(1, NYR + 1):
        f = j / NYR
        r = ry * f
        z = zbase + H * (1 - f ** 2.3) ** 0.8
        for i in range(NYS):
            th = 2 * PI * i / NYS
            pts.append(Vector((r * math.cos(th), r * math.sin(th), z)))
    return pts


def fried_egg_obj(name, M):
    bm = bmesh.new()
    vs = []
    vs.append(bm.verts.new((0, 0, 0)))
    for j in range(1, NR + 1):
        for i in range(NS):
            vs.append(bm.verts.new((j, i, 0)))
    bot0 = len(vs)
    for i in range(NS):
        vs.append(bm.verts.new((0, 0, 0)))
    botc = len(vs)
    vs.append(bm.verts.new((0, 0, 0)))
    y0 = len(vs)
    vs.append(bm.verts.new((0, 0, 0)))
    for j in range(1, NYR + 1):
        for i in range(NYS):
            vs.append(bm.verts.new((0, 0, 0)))
    faces_white, faces_yolk = [], []

    def W(j, i):
        return vs[0] if j == 0 else vs[1 + (j - 1) * NS + i % NS]
    for j in range(NR):
        for i in range(NS):
            if j == 0:
                f = bm.faces.new((W(0, 0), W(1, i), W(1, i + 1)))
            else:
                f = bm.faces.new((W(j, i), W(j + 1, i), W(j + 1, i + 1), W(j, i + 1)))
            f.material_index = 0
    for i in range(NS):  # side wall
        k = (i + 1) % NS
        f = bm.faces.new((W(NR, i), vs[bot0 + i], vs[bot0 + k], W(NR, k)))
        f.material_index = 0
        f = bm.faces.new((vs[botc], vs[bot0 + k], vs[bot0 + i]))
        f.material_index = 0

    def Y(j, i):
        return vs[y0] if j == 0 else vs[y0 + 1 + (j - 1) * NYS + i % NYS]
    for j in range(NYR):
        for i in range(NYS):
            if j == 0:
                f = bm.faces.new((Y(0, 0), Y(1, i), Y(1, i + 1)))
            else:
                f = bm.faces.new((Y(j, i), Y(j + 1, i), Y(j + 1, i + 1), Y(j, i + 1)))
            f.material_index = 1
    bm.verts.ensure_lookup_table()
    # UVs: UV_topdown (planar, unit square, texture domain = spread-1 disc) and UV_radial (u = angle, v = radius)
    uv1 = bm.loops.layers.uv.new("UV_topdown")
    uv2 = bm.loops.layers.uv.new("UV_radial")
    vidx = {v: k for k, v in enumerate(vs)}

    def param(k):
        if k == 0:
            return 0.0, 0.0, "w"
        if k < bot0:
            j = (k - 1) // NS + 1
            i = (k - 1) % NS
            return j / NR, 2 * PI * i / NS, "w"
        if k < botc:
            return 1.0, 2 * PI * (k - bot0) / NS, "w"
        if k == botc:
            return 0.0, 0.0, "w"
        if k == y0:
            return 0.0, 0.0, "y"
        j = (k - y0 - 1) // NYS + 1
        i = (k - y0 - 1) % NYS
        return j / NYR, 2 * PI * i / NYS, "y"
    for f in bm.faces:
        for l in f.loops:
            rho, th, kind = param(vidx[l.vert])
            rr = rho * (RY / R_MAX if kind == "y" else 1.0)
            l[uv1].uv = (0.5 + 0.5 * rr * math.cos(th), 0.5 + 0.5 * rr * math.sin(th))
            l[uv2].uv = ((th / (2 * PI)) % 1.0, rr)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    # fix seam u in UV_radial for the last column
    for f in bm.faces:
        us = [l[uv2].uv[0] for l in f.loops]
        if max(us) - min(us) > 0.5:
            for l in f.loops:
                if l[uv2].uv[0] < 0.25:
                    l[uv2].uv[0] += 1.0
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(M["white"])
    me.materials.append(M["yolk"])
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    # shape keys
    basis_pts = egg_state(0, 0, 0, 0)
    n = len(basis_pts)
    o.shape_key_add(name="Basis", from_mix=False)
    for kb_name, st in (("spread", (1, 0, 0, 0)), ("edge_crisp", (0, 1, 0, 0)), ("yolk_dome", (0, 0, 1, 0)), ("bubbles", (0, 0, 0, 1))):
        pts = egg_state(*st)
        kb = o.shape_key_add(name=kb_name, from_mix=False)
        kb.slider_min, kb.slider_max = 0.0, 1.0
    # assign: Basis; each key stores basis + (state - basis) so that unit values add to the full state
    for kb in o.data.shape_keys.key_blocks:
        if kb.name == "Basis":
            kb.data.foreach_set("co", [c for p in basis_pts for c in (mm(p.x), mm(p.y), mm(p.z))])
    for kb_name, st in (("spread", (1, 0, 0, 0)), ("edge_crisp", (0, 1, 0, 0)), ("yolk_dome", (0, 0, 1, 0)), ("bubbles", (0, 0, 0, 1))):
        pts = egg_state(*st)
        kb = o.data.shape_keys.key_blocks[kb_name]
        kb.data.foreach_set("co", [c for p in pts for c in (mm(p.x), mm(p.y), mm(p.z))])
    o.data.vertices.foreach_set("co", [c for p in basis_pts for c in (mm(p.x), mm(p.y), mm(p.z))])
    o.data.update()
    # vertex groups
    vg = {nme: o.vertex_groups.new(name=nme) for nme in ("white", "yolk", "edge_dist", "radial", "rim", "bubble_zone", "yolk_edge")}
    fin = egg_state(1, 0, 1, 1)
    for k in range(n):
        rho, th, kind = param(k)
        if kind == "w":
            vg["white"].add([k], 1.0, "REPLACE")
            vg["radial"].add([k], rho, "REPLACE")
            vg["edge_dist"].add([k], smooth(0.55, 1.0, rho), "REPLACE")
            if rho > 0.93:
                vg["rim"].add([k], 1.0, "REPLACE")
            x, y = fin[k].x, fin[k].y
            bz = 0.0
            for (cx, cy), sig, h in bubbles():
                bz = max(bz, math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (1.6 * sig) ** 2)))
            if bz > 0.02:
                vg["bubble_zone"].add([k], bz, "REPLACE")
        else:
            vg["yolk"].add([k], 1.0, "REPLACE")
            vg["radial"].add([k], rho, "REPLACE")
            if rho > 0.999:
                vg["yolk_edge"].add([k], 1.0, "REPLACE")
    return o


# ------------------------------------------------------------------ plate, spatula, pan
def plate_obj(name, mat):
    prof = [(0, 2.0), (78, 2.0), (80, 0.0), (90, 0.0), (92, 2.0), (120, 10.0), (133, 22.0), (135, 25.0), (133, 26.5), (130, 25.5), (100, 14.0), (0, 13.0)]
    return L.lathe(name, [(mm(r), mm(h)) for r, h in prof], 96, closed_profile=True, mat=mat, sharp_deg=None)


def spatula(coll, root, M, pre):
    parts = []
    # blade lies flat; origin at the handle centre; blade extends along -Y
    z0 = 0.0
    bar = L.rbox(pre + "blade_bar", (mm(92), mm(26), mm(1.4)), (0, -mm(211), mm(z0)), mm(0.5), 1, M["steel"])
    parts.append(bar)
    for k in range(5):
        x = (k - 2) * 19.5
        parts.append(L.rbox(pre + "blade_tine_%d" % k, (mm(12), mm(80), mm(1.4)), (mm(x), -mm(264), mm(z0)), mm(0.5), 1, M["steel"]))
    # neck: bent flat bar rising from the blade to the handle
    a = Vector((0, -mm(310 - 26 - 80 + 4), mm(z0)))
    ptsyz = [(-(310 - 106 - 80 + 4) + 0, z0)]
    neck_pts = [Vector((0, -mm(y), mm(z))) for y, z in ((210, 0), (198, 0.4), (175, 6), (150, 22), (125, 36), (100, 38), (70, 38))]
    neck_pts = L.catmull(neck_pts, 6)
    bm = bmesh.new()
    L.tube_path_bm(bm, neck_pts, [(mm(a), mm(b)) for a, b in L.superellipse2d(22, 2.6, 10, 3.0)], binormal=Vector((1, 0, 0)), cap=True)
    parts.append(L.obj_from_bm(pre + "neck", bm, M["steel"], True, True, 50))
    prof = [(0, -62), (9.5, -62), (11.5, -45), (11.5, 40), (10, 56), (7, 62), (0, 62)]
    h = L.lathe(pre + "handle", [(mm(r), mm(z)) for r, z in prof], 32, xf=Matrix.Translation((0, 0, mm(38))) @ Matrix.Rotation(PI / 2, 4, "X"), mat=M["handle"])
    # handle runs along y: lathe Z -> after rotate +90 about X: Z -> -Y
    parts.append(h)
    rv = C.cylinder(pre + "rivet", mm(2.5), mm(26), loc=(0, -mm(95), mm(38)), axis="Z", verts=12, mat=M["steel"])
    parts.append(rv)
    for o in parts:
        C.add(o, coll, root)
    L.hook_at("grip", coll, root, (0, 0, mm(38)), size=0.03)
    L.hook_at("blade_tip", coll, root, (0, -mm(304), 0), size=0.02)


def pan_obj(coll, root, M, pre):
    prof = [(0, 0.0), (92, 0.0), (100, 3.0), (120, 40.0), (123, 44.0), (121, 45.0), (118, 43.0), (97, 7.0), (92, 3.5), (0, 3.5)]
    p = L.lathe(pre + "body", [(mm(r), mm(h)) for r, h in prof], 96, closed_profile=True, mat=M["pan"], sharp_deg=None)
    C.add(p, coll, root)
    # handle along -Y from the rim
    pts = [Vector((0, -mm(y), mm(z))) for y, z in ((118, 40), (150, 43), (230, 48), (310, 50))]
    pts = L.catmull(pts, 6)
    bm = bmesh.new()
    L.tube_path_bm(bm, pts, [(mm(a), mm(b)) for a, b in L.superellipse2d(24, 12, 12, 2.8)], binormal=Vector((1, 0, 0)), cap=True,
                   scale_fn=lambda t: 1.0 - 0.25 * t)
    hnd = L.obj_from_bm(pre + "handle", bm, M["pan_handle"], True, True, 50)
    C.add(hnd, coll, root)
    for k, y in enumerate((140, 165)):
        rv = C.cylinder(pre + "rivet_%d" % k, mm(4), mm(3), loc=(0, -mm(y), mm(51 if k else 45)), axis="Z", verts=12, mat=M["steel"] if k >= 0 else None)
        C.add(rv, coll, root)
    L.hook_at("grip", coll, root, (0, -mm(270), mm(49)), size=0.03)
    L.hook_at("egg_center", coll, root, (0, 0, mm(3.5)), size=0.02)


def new_item(aid, accuracy="B"):
    coll, root = C.new_asset(aid, accuracy=accuracy)
    return coll, root


def main():
    C.reset()
    M = mats()
    items = []
    xcur = [0.0]

    def place(root, width_m):
        root.location = (xcur[0], 0, 0)
        xcur[0] += width_m + 0.15
    # ---- egg (lying on its side along X, resting on z = 0)
    coll, root = new_item("egg")
    o = egg_obj("egg_shell", M["shell"])
    C.add(o, coll, root)
    L.hook_at("crack_point", coll, root, (0, 0, mm(2 * EGG_B)), size=0.01)
    place(root, 0.06)
    items.append(dict(asset_id="egg", coll=coll, root=root, meta=dict(accuracy="B", origin="bottom centre of the footprint; egg lies along X (blunt end -X, pointed end +X)",
                                                                  hooks={"HOOK_crack_point": "top of the shell where it cracks"})))
    # ---- cracked egg
    coll, root = new_item("egg_cracked")
    ha = shell_half("egg_cracked_half_blunt", M["shell"], "blunt", 3)
    hb = shell_half("egg_cracked_half_point", M["shell"], "point", 5)
    ha.data.transform(Matrix.Translation((-mm(30), 0, 0)) @ Matrix.Rotation(0.12, 4, "Y"))
    hb.data.transform(Matrix.Translation((mm(34), mm(5), 0)) @ Matrix.Rotation(-0.35, 4, "Y"))
    for h in (ha, hb):
        zmin = min(v.co.z for v in h.data.vertices)
        h.data.transform(Matrix.Translation((0, 0, -zmin)))
    C.add(ha, coll, root)
    C.add(hb, coll, root)
    yk = C.sphere("egg_cracked_yolk", mm(14), loc=(-mm(30), 0, mm(16)), scale=(1, 1, 0.82), mat=M["yolk"], seg=32, rings=16)
    C.add(yk, coll, root)
    place(root, 0.1)
    items.append(dict(asset_id="egg_cracked", coll=coll, root=root, meta=dict(accuracy="B", origin="bottom centre between the halves; halves open upward, jagged break edge, thickness 0.9 mm (estimate)", hooks={})))
    # ---- fried egg
    coll, root = new_item("fried_egg")
    fe = fried_egg_obj("fried_egg", M)
    C.add(fe, coll, root)
    L.add_prop(root, "p_spread", 1.0, 0.0, 1.0, "white spread 0 (compact, thick) to 1 (fully spread, 100 mm); drives shape key 'spread'")
    L.add_prop(root, "p_edge_crisp", 1.0, 0.0, 1.0, "edge crisping 0..1 (raised, frilled, shrunken rim); drives 'edge_crisp'")
    L.add_prop(root, "p_yolk_dome", 1.0, 0.0, 1.0, "yolk dome 0 (flat 3 mm) to 1 (domed 9.5 mm); drives 'yolk_dome'")
    L.add_prop(root, "p_bubbles", 1.0, 0.0, 1.0, "bubble blisters 0..1; drives 'bubbles'")
    key = fe.data.shape_keys
    for kb_name, prop in (("spread", "p_spread"), ("edge_crisp", "p_edge_crisp"), ("yolk_dome", "p_yolk_dome"), ("bubbles", "p_bubbles")):
        L.driver(key, 'key_blocks["%s"].value' % kb_name, None, root, [prop], "v0")
        key.key_blocks[kb_name].value = 1.0
    L.hook_at("yolk_top", coll, root, (0, 0, mm(15)), size=0.01)
    L.hook_at("pan_contact", coll, root, (0, 0, 0), size=0.01)
    place(root, 0.11)
    items.append(dict(asset_id="fried_egg", coll=coll, root=root, meta=dict(accuracy="B", origin="bottom centre (z = 0 underside), egg lies on the XY plane; diameter about 100 mm at p_spread=1 (estimate)",
                                                                       hooks={"HOOK_yolk_top": "approximate yolk apex at full state", "HOOK_pan_contact": "underside centre"})))
    fried = fe
    # ---- plate
    coll, root = new_item("plate")
    C.add(plate_obj("plate_body", M["plate"]), coll, root)
    L.hook_at("food_center", coll, root, (0, 0, mm(13)), size=0.02)
    L.hook_at("grip_edge", coll, root, (0, -mm(135), mm(20)), size=0.02)
    place(root, 0.27)
    items.append(dict(asset_id="plate", coll=coll, root=root, meta=dict(accuracy="B", origin="bottom centre", hooks={"HOOK_food_center": "centre of the well (z = 13 mm)", "HOOK_grip_edge": "front rim for the carrying hand"})))
    # ---- spatula
    coll, root = new_item("spatula")
    spatula(coll, root, M, "spatula_")
    place(root, 0.1)
    items.append(dict(asset_id="spatula", coll=coll, root=root, meta=dict(accuracy="C", origin="handle centre (mounting origin), blade extends along -Y, lying flat; length about 304 mm", hooks={"HOOK_grip": "handle centre", "HOOK_blade_tip": "front edge of the blade"})))
    # ---- frying pan
    coll, root = new_item("frying_pan")
    pan_obj(coll, root, M, "frying_pan_")
    place(root, 0.25)
    items.append(dict(asset_id="frying_pan", coll=coll, root=root, meta=dict(accuracy="C", origin="bottom centre of the pan, handle along -Y", hooks={"HOOK_grip": "handle", "HOOK_egg_center": "pan floor centre"})))
    # ---- plate with eggs (three fried eggs on the plate, own mesh copies without drivers)
    coll, root = new_item("plate_with_eggs")
    C.add(plate_obj("plate_with_eggs_plate", M["plate"]), coll, root)
    for k, (x, y, rz) in enumerate(((-45, 20, 0.4), (40, 30, 2.0), (0, -50, 4.1))):
        e = bpy.data.objects.new("plate_with_eggs_egg_%d" % (k + 1), fried.data.copy())
        e.data.shape_keys.animation_data_clear()
        for kb in e.data.shape_keys.key_blocks:
            if kb.name != "Basis":
                kb.value = 1.0 if kb.name != "spread" else 0.55
        bpy.context.scene.collection.objects.link(e)
        e.location = (mm(x), mm(y), mm(13))
        e.scale = (0.8, 0.8, 0.8)
        e.rotation_euler = (0, 0, rz)
        C.add(e, coll, root)
    L.hook_at("grip_edge", coll, root, (0, -mm(135), mm(20)), size=0.02)
    place(root, 0.27)
    items.append(dict(asset_id="plate_with_eggs", coll=coll, root=root, meta=dict(accuracy="B", origin="bottom centre of the plate; three fried eggs (spread 0.55, scaled 0.8) as independent mesh copies (no drivers)", hooks={"HOOK_grip_edge": "front rim"})))
    # ---- pan with egg
    coll, root = new_item("pan_with_egg")
    pan_obj(coll, root, M, "pan_with_egg_")
    e = bpy.data.objects.new("pan_with_egg_egg", fried.data.copy())
    e.data.shape_keys.animation_data_clear()
    for kb in e.data.shape_keys.key_blocks:
        if kb.name != "Basis":
            kb.value = 0.85
    bpy.context.scene.collection.objects.link(e)
    e.location = (0, 0, mm(3.5))
    C.add(e, coll, root)
    place(root, 0.25)
    items.append(dict(asset_id="pan_with_egg", coll=coll, root=root, meta=dict(accuracy="C", origin="bottom centre of the pan", hooks={"HOOK_grip": "handle"})))

    meta = dict(
        category="props", asset_family="eggs_plate", date="2026-10-02", accuracy="B (egg and plate dimensions from the brief; fried egg profile is a design)",
        description="Clay eggs and kitchenware: egg, cracked egg halves with yolk, fried egg (shape keys, vertex groups, two UV maps for the external egg-fry shader), plate, spatula, frying pan, plate with 3 eggs, pan with egg. Each is an ASSET_ collection with its own root; roots are spread along +X in the file.",
        sources=[{"what": "egg 57 x 44 mm and plate 270 mm diameter", "url": "task brief (DevLog props assignment); not independently verified", "accessed": "2026-10-02"}],
        dimensions_mm=[{"item": "egg length x width", "value": "57 x 44", "provenance": "task brief", "level": "B"},
                       {"item": "egg shell thickness (cracked halves)", "value": 0.9, "provenance": "estimate +-0.2", "level": "C"},
                       {"item": "fried egg white diameter at p_spread=1", "value": "about 100 (outline 24..50 mm radius range)", "provenance": "estimate +-15", "level": "C"},
                       {"item": "yolk radius", "value": "15 (at spread 0, 15 at spread 1)", "provenance": "estimate +-3", "level": "C"},
                       {"item": "plate diameter", "value": 270, "provenance": "task brief", "level": "B"},
                       {"item": "pan top diameter / depth", "value": "246 / 45", "provenance": "estimate", "level": "C"},
                       {"item": "spatula length", "value": 304, "provenance": "estimate", "level": "C"}],
        egg_fry_shader_interface={
            "object": "fried_egg (one mesh, two material slots: 0 = white incl. side wall, 1 = yolk; replace with the egg-fry shader)",
            "shape_keys": {"Basis": "freshly dropped: compact lump, radius about 24 mm, centre 15 mm thick, flat yolk 3 mm", "spread": "0..1 white spreads to 100 mm, thins to 0.9 mm at the edge",
                           "edge_crisp": "0..1 rim raised about 1.6 mm, frilled (17 and 11 lobe sinusoids), shrunk about 1 mm", "yolk_dome": "0..1 yolk height 3 mm to 9.5 mm", "bubbles": "0..1 seven blisters 1.8-3.0 mm high, sigma 2.2-3.6 mm"},
            "root_custom_properties": "p_spread, p_edge_crisp, p_yolk_dome, p_bubbles (0..1, drive the keys; all 1.0 in the file = fully cooked look)",
            "vertex_groups": {"white": "1 on white verts", "yolk": "1 on yolk verts", "radial": "normalised radius 0..1 (white: ring index / 36; yolk: 0..1 over the dome)", "edge_dist": "0 inside rho 0.55, smoothstep to 1 at the rim: drive edge browning",
                              "rim": "1 on the outer rings (rho > 0.93) and side wall", "bubble_zone": "Gaussian mask around the 7 bubble centres (1.6 sigma) at full state", "yolk_edge": "1 on the yolk base ring"},
            "uv_maps": {"UV_topdown": "planar top-down, unit square, white disc rho*0.5 around (0.5, 0.5) independent of the state; yolk maps to the central 0.3 radius", "UV_radial": "u = angle / 2pi (0..1, seam handled), v = radius 0..1 for the white; yolk v = 0..0.3"},
            "mesh": "white: centre + 36 rings x 96 segments + side wall + underside fan; yolk: apex + 10 rings x 48 segments",
        },
        hooks="see items",
        custom_properties={"fried_egg.p_spread": "0..1", "fried_egg.p_edge_crisp": "0..1", "fried_egg.p_yolk_dome": "0..1", "fried_egg.p_bubbles": "0..1"},
        materials=[m.name for m in M.values()],
        simplifications=["fried egg preview material is plain clay (white and yolk); animated look comes from the external shader", "yolk is an open dome sunk 0.6 mm into the white (no inner yolk membrane)", "spatula tines are separate thin boxes, neck is a swept flat bar",
                         "pan is a lathe with a swept handle, no interior coating"],
        intended_usage="S4: one fried egg per optical engine (scaled by the assembler, egg-fry shader plugged in), Gary carries plate_with_eggs, eggs fly and land on the Manager; spatula/pan optional gags.",
    )
    blend = os.path.join(OUT, "eggs_plate.blend")
    L.write_family(blend, items, meta, PREV, previews=False)
    L.reopen(blend)
    pn = []
    for aid, close in (("egg", []), ("egg_cracked", []), ("fried_egg", []), ("plate", []), ("spatula", []), ("frying_pan", []), ("plate_with_eggs", []), ("pan_with_egg", [])):
        coll = bpy.data.collections["ASSET_" + aid]
        views = L.std_views()
        if aid == "fried_egg":
            views = [{"name": "front", "dir": "front", "fit": 1.0}, {"name": "three_quarter", "dir": "three_quarter", "fit": 1.0}, {"name": "top", "dir": "top", "fit": 1.0}]
        pn += L.render_views([coll], PREV, "eggs_plate_" + aid, views)
    # fried egg key states
    root = bpy.data.objects["ROOT_fried_egg"]
    coll = bpy.data.collections["ASSET_fried_egg"]
    for nm, vals in (("state_raw", (0, 0, 0, 0)), ("state_half", (0.5, 0.3, 0.5, 0.0)), ("state_cooked_close", (1, 1, 1, 1))):
        for p, v in zip(("p_spread", "p_edge_crisp", "p_yolk_dome", "p_bubbles"), vals):
            L.set_prop(root, p, float(v))
        views = [{"name": nm, "dir": (0.0, -0.7, 0.8), "fit": 1.0}] if nm != "state_cooked_close" else [{"name": nm, "target": Vector((mm(30), -mm(10), mm(3))), "cam_dir": Vector((0.3, -0.7, 0.5)), "dist": 0.11}]
        pn += L.render_views([coll], PREV, "eggs_plate_fried_egg", views)
    L.contact_sheet(pn, os.path.join(L.SCRATCH, "sheet_eggs.png"), 6, (300, 225))


main()
