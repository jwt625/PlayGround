"""OUNOLIGHT HATCHERY v3: assembly line with real T / H parts (workstream M, DevLog-019). Blender 4.2, headless, metres, Z up.

Line along local +X (camera side -Y, back wall +Y). Five stations, every station acting in parallel over one sequence
(DUR frames from F0) so the scene can cut to any station at any time:
  S1 DIE ATTACH        x -2.2  gantry bond head + collet places the PIC, then a DFB, on the TX carrier
  S2 ACTIVE ALIGN      x -1.1  two 6-axis stages with tweezers place the micro isolator and Si lens 2 in front of the lit
                               DFB (alignment dither, power readout rising), then UV light guides flash (UV cure)
  S3 WIRE BOND         x  0.0  ball-wedge bond head (horn, capillary, clamp, EFO): HS GSSG bonds appear one by one
  S4 DISPENSE + PRESS  x  1.15 needle dispenses TIM dots on the DSP; press lowers the top shell + wings onto the bird
  S5 TEST / EYES ON    x  1.95 eye-check cameras, eye-diagram monitor, eyes light up, PASS, the bird hops off
Parts: TX subassembly objects appended from assets/components/optics_tx/optics_tx.blend, the bird from
assets/hero/pigeon_osfp_dr8.blend, both scaled x2.6 (hall scale = S_BIRD). Tool ends (collet, tweezers, capillary, horn,
needle) are modeled true size inside the x2.6 work frame; machine bodies are human scale (children with scale 1/2.6).

Usage: scripts/bslot.sh -b --factory-startup --python scripts/assembly/build_assembly.py -- [out.blend]
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
for p in (SCRIPTS, os.path.join(SCRIPTS, "env")):
    if p not in sys.path:
        sys.path.insert(0, p)
import env_lib as E  # noqa: E402
import vfx  # noqa: E402
from env_lib import MB  # noqa: E402

PI = math.pi
S = 2.6                                   # hall scale (S_BIRD)
F0, DUR = 1, 288                          # v4.2: 288 build frames = 184 film frames (6.133 s) at S's speed 1.5652 (2x v4.1)
DFB_ON = 40                               # DFB off during pick and place; on when tweezer B leaves Si lens 2 (rise starts t39)
TX_BLEND = os.path.join(ROOT, "assets", "components", "optics_tx", "optics_tx.blend")
HERO_BLEND = os.path.join(ROOT, "assets", "hero", "pigeon_osfp_dr8.blend")
OUT_DIR = os.path.join(ROOT, "assets", "components", "assembly")
EYE_TEX = os.path.join(OUT_DIR, "textures", "eye_pam4.png")
EYE_ATLAS = os.path.join(OUT_DIR, "textures", "eye_pam4_atlas.png")     # v4: 24 varying PAM4 frames (make_eye_atlas.py)
XS = {"S1": -2.2, "S2": -1.1, "S3": 0.0, "S4": 1.15, "S5": 1.95}
CHUCK_Z = 0.98                            # top of the work chuck (micro stations)
WORK_Y = -0.05
BENCH_Z = 0.80                            # bird bench top
OY, OZ = 4.7e-3, 3.07e-3                  # sub (T frame) -> work frame offsets: carrier centred, carrier bottom on the chuck
EVENTS = {}                               # name -> frame offset from F0 (for SFX / camera)
LOG = []


def f(t):
    return F0 + t


def s2w(p):
    """T subassembly frame (m, +X = component normal) -> work frame W (m true size, +Z up)."""
    return Vector((-p[2], p[1] + OY, p[0] + OZ))


def w2s(v):
    return Vector((v[2] - OZ, v[1] - OY, -v[0]))


def key(o, path, frame, value, interp="BEZIER"):
    setattr(o, path, value)
    o.keyframe_insert(path, frame=frame)
    if o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            if fc.data_path == path:
                for kp in fc.keyframe_points:
                    if abs(kp.co.x - frame) < 0.01:
                        kp.interpolation = interp


def key_val(sock, frame, value, interp="LINEAR"):
    sock.default_value = value
    sock.keyframe_insert("default_value", frame=frame)
    nt = sock.id_data
    if nt.animation_data and nt.animation_data.action:
        for fc in nt.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                if abs(kp.co.x - frame) < 0.01:
                    kp.interpolation = interp


def emis_mat(name, color, strength, dark=False):
    """Emissive material; dark=True gives a near-black base so the object reads OFF when the keyed emission is 0."""
    m = E.principled(name, (0.015, 0.015, 0.017) if dark else color, 0.0, 0.4, color, strength)
    return m


def bsdf(m):
    return next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def child_inv(parent, name, c):
    """Empty under `parent` with scale 1/S: children modeled in real metres inside a x2.6 work frame."""
    e = E.empty(name, c, (0, 0, 0), (0, 0, 0), 0.05, parent=parent)
    e.scale = (1 / S, 1 / S, 1 / S)
    return e


# ==================================================================== parts
def append_tx():
    """Append every object of T's blend; return name -> object (originals, used as templates)."""
    with bpy.data.libraries.load(TX_BLEND, link=False) as (src, dst):
        dst.objects = list(src.objects)
    out = {}
    tmp = bpy.data.collections.new("ASM_tmp_tx")
    bpy.context.scene.collection.children.link(tmp)
    for o in dst.objects:
        if o is None:
            continue
        tmp.objects.link(o)                 # in the scene so matrix_world is evaluated (orphans keep a stale one)
        out[o.name] = o
    bpy.context.view_layer.update()
    return out


def tx_copy(tpl, names, parent, c, prefix):
    """Linked-data copies of template objects under `parent` (the sub frame); keeps the template's local transform."""
    objs = {}
    for n in names:
        t = tpl[n]
        o = t.copy()
        o.name = prefix + n
        c.objects.link(o)
        o.parent = parent
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = t.matrix_world.copy()          # template parent (OPTICS_TX_ROOT) is identity at the origin
        o.hide_render = t.hide_render
        objs[n] = o
    return objs


def group(name, objs, parent, c, center_sub):
    """Empty at center (sub frame) holding objs, so a part moves as one."""
    g = E.empty(name, c, tuple(center_sub), (0, 0, 0), 0.0005, parent=parent)
    for o in objs:
        mw = o.matrix_basis.copy()
        o.parent = g
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = Matrix.Translation(-Vector(center_sub)) @ mw
    return g


def bbox_sub(o):
    vs = [o.matrix_basis @ Vector(v) for v in o.bound_box]
    lo = Vector([min(v[i] for v in vs) for i in range(3)])
    hi = Vector([max(v[i] for v in vs) for i in range(3)])
    return lo, hi


def center_of(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = -lo
    for o in objs:
        a, b = bbox_sub(o)
        lo = Vector([min(lo[i], a[i]) for i in range(3)])
        hi = Vector([max(hi[i], b[i]) for i in range(3)])
    return (lo + hi) / 2, lo, hi


def split_loose(o, c):
    """Split a mesh object into one object per connected component (same parent / transform). Returns the parts."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.verts.ensure_lookup_table()
    seen, comps = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            x = stack.pop()
            comp.append(x.index)
            for e in x.link_edges:
                y = e.other_vert(x)
                if y.index not in seen:
                    seen.add(y.index)
                    stack.append(y)
        comps.append(comp)
    parts = []
    src = o.data
    for k, comp in enumerate(comps):
        sel = set(comp)
        b2 = bm.copy()
        b2.verts.ensure_lookup_table()
        bmesh.ops.delete(b2, geom=[v for v in b2.verts if v.index not in sel], context="VERTS")
        me = bpy.data.meshes.new("%s_%02d" % (o.name, k))
        b2.to_mesh(me)
        b2.free()
        for m in src.materials:
            me.materials.append(m)
        p = bpy.data.objects.new(me.name, me)
        c.objects.link(p)
        p.parent = o.parent
        p.matrix_basis = o.matrix_basis.copy()
        parts.append(p)
    bm.free()
    bpy.data.objects.remove(o, do_unlink=True)
    return parts


def set_origin(o, p_local):
    """Move the object origin to p_local (object space) without moving the geometry."""
    o.data.transform(Matrix.Translation(-Vector(p_local)))
    o.matrix_basis = o.matrix_basis @ Matrix.Translation(Vector(p_local))


def place_in_W(tpl_obj, center_sub, target_w):
    """Matrix (W frame) that puts a T template object's part centre at target_w, laid flat like on the carrier."""
    return (Matrix.Translation(Vector(target_w) - s2w(center_sub)) @ Matrix.Translation(Vector((0, OY, OZ))) @
            Matrix.Rotation(-PI / 2, 4, "Y") @ tpl_obj.matrix_world)


def spare(tpl, n, W, c, name, xy, z_surface):
    o = tpl[n].copy()
    o.name = name
    c.objects.link(o)
    o.parent = W
    o.matrix_parent_inverse = Matrix.Identity(4)
    cc_, lo, hi = center_of([tpl[n]])
    o.matrix_basis = place_in_W(tpl[n], cc_, (xy[0], xy[1], z_surface + (cc_.x - lo.x)))
    return o


def append_hero():
    """Append the hero's collections (objects keep their collection membership: needed for the lid split)."""
    with bpy.data.libraries.load(HERO_BLEND, link=False) as (src, dst):
        dst.collections = list(src.collections)
    cols = [cc for cc in dst.collections if cc is not None]
    objs = []
    for cc in cols:
        for o in cc.objects:
            if o not in objs:
                objs.append(o)
    return objs, cols


def hero_copy(objs, root_parent, c, prefix, lid=None, lid_colls=()):
    """Copy the hero (linked data) under a new root empty; objects of lid_colls go under `lid`."""
    mp = {}
    for o in objs:
        n = o.copy()
        n.name = prefix + o.name
        c.objects.link(n)
        mp[o] = n
    for o, n in mp.items():
        if o.parent is not None and o.parent in mp:
            n.parent = mp[o.parent]
            n.matrix_parent_inverse = o.matrix_parent_inverse.copy()
    for o, n in mp.items():
        if o.name == "HERO_ROOT":
            n.parent = root_parent
            n.matrix_parent_inverse = Matrix.Identity(4)
            n.matrix_basis = Matrix.Identity(4)
    if lid is not None:
        for o, n in mp.items():
            cn = o.users_collection[0].name if o.users_collection else ""
            if cn in lid_colls and (o.parent is None or o.parent.name == "HERO_ROOT"):
                mb = n.matrix_basis.copy()      # relative to HERO_ROOT (identity under the root) -> same under lid (at origin)
                n.parent = lid
                n.matrix_parent_inverse = o.matrix_parent_inverse.copy()
                n.matrix_basis = mb
    return {o.name: n for o, n in mp.items()}


def own_material(o, suffix):
    """Single-user copy of the object's materials (so one bird can glow while another stays dark)."""
    for i, m in enumerate(o.data.materials):
        if m is None:
            continue
        o.data = o.data.copy() if o.data.users > 1 else o.data
        nm = m.copy()
        nm.name = m.name + suffix
        o.data.materials[i] = nm
    return [m for m in o.data.materials if m]


# ==================================================================== set
def build_set(root, cset, cl):
    mb = MB()
    mb.box((0.0, 0.2, -0.05), (7.6, 4.4, 0.1), m=0)                       # floor
    mb.box((0.0, 1.25, 1.6), (7.6, 0.12, 3.2), m=1)                       # back wall
    for x in [-3.6 + 0.9 * i for i in range(9)]:
        mb.box((x, 1.18, 1.6), (0.04, 0.03, 3.2), m=2)
    mb.box((0.0, 1.17, 0.12), (7.6, 0.04, 0.24), m=2)                     # skirting
    mb.box((0.0, 0.2, 3.25), (7.6, 4.4, 0.1), m=2)                        # ceiling
    for x in (-2.6, -0.9, 0.8, 2.5):
        mb.box((x, 0.0, 3.19), (1.4, 0.18, 0.02), m=3)                    # light troffers
    for y in (-0.62, -0.74):
        mb.box((0.0, y, 0.002), (7.2, 0.05, 0.004), m=4)                  # walkway lines
    E.obj("asm_room", mb, [vfx.wet_ground_material("ASM_floor", (0.05, 0.055, 0.06), wetness=0.15, coverage=0.0, rough_dry=0.4,
                                                    rough_wet=0.25, grain=0.05),
                           "cladding_light", "metal_dark", "lamp_lens_cool", "paint_yellow"], cset, parent=root)
    import street_kit as SK
    sg = SK.neon_sign(cset, (0.0, 1.18, 2.55), 0.0, "OUNOLIGHT HATCHERY", "neon_mag", 0.30, name="asm_sign",
                      light=((1.0, 0.1, 0.7), 20.0, cl))
    sg["root"].parent = root
    for o in cl.objects:
        if o.name.startswith("asm_sign") and o.parent is None:
            o.parent = root
    for x in (-2.6, -0.9, 0.8, 2.5):
        E.light("asm_ceiling_%+.1f" % x, "AREA", (x, 0.0, 3.1), 140, (0.85, 0.9, 1.0), cl, size=1.4, rot=(0, 0, 0))
    E.light("asm_front_fill", "AREA", (0.0, -3.2, 2.2), 90, (0.7, 0.75, 0.9), cl, size=6, target=(0, 0.3, 1.0))
    lab_m = E.principled("ASM_label", (0.9, 0.95, 1.0), 0, 0.4, (0.85, 0.95, 1.0), 1.5)
    labels = {"S1": "DIE ATTACH", "S2": "ACTIVE ALIGN + UV CURE", "S3": "WIRE BOND", "S4": "DISPENSE + LID PRESS",
              "S5": "TEST / EYES ON"}
    for k, (st, lab) in enumerate(labels.items()):
        x = XS[st]
        pb = MB()
        pb.box((x, 1.17, 1.95), (0.86 if st in ("S1", "S2", "S3") else 0.7, 0.02, 0.16), m=0)
        E.obj("asm_plate_%s" % st, pb, ["sign_back"], cset, parent=root)
        E.text("asm_lbl_%s" % st, "%s  %s" % (st, lab), 0.05 if len(lab) < 14 else 0.04, cset, (x, 1.158, 1.95), (PI / 2, 0, 0),
               mat=lab_m, fnt="Arial Bold", parent=root)


def machine_base(name, x, c, root, color_badge="neon_cyan", badge=""):
    """Cabinet 0.8 x 0.7 x 0.85 + granite slab to z 0.95 + chuck to CHUCK_Z; badge text on the front."""
    mb = MB()
    mb.box((x, 0.05, 0.425), (0.8, 0.7, 0.85), m=0)
    mb.box((x, 0.05, 0.9), (0.8, 0.7, 0.1), m=1)
    mb.box((x, -0.302, 0.45), (0.7, 0.004, 0.6), m=2)                     # front panel inset
    for s in (-1, 1):
        mb.box((x + s * 0.38, 0.05, 0.02), (0.06, 0.66, 0.04), m=3)       # levelling rails
    mb.box((x, WORK_Y, 0.965), (0.24, 0.17, 0.03), m=3)                   # work stage / chuck plate
    mb.box((x, -0.304, 0.78), (0.7, 0.004, 0.012), m=5)                   # status LED strip
    o = E.obj(name + "_base", mb, ["ac_white", E.principled("ASM_granite", (0.018, 0.019, 0.022), 0.0, 0.55),
                                   "metal_paint", "metal_dark", "galv", "led_cyan"], c, parent=root)
    if badge:
        E.text(name + "_badge", badge, 0.045, c, (x - 0.25, -0.306, 0.68), (PI / 2, 0, 0), mat=color_badge, fnt="Arial Bold",
               parent=root)
    return o


def frame_posts(name, x, c, root, h=1.75):
    """Aluminium extrusion frame (open front) around a micro station + ring light over the chuck."""
    mb = MB()
    for sx in (-0.38, 0.38):
        for y in (-0.28, 0.38):
            if y < 0:
                continue
            mb.box((x + sx, y, 0.95 + h / 2 - 0.475), (0.04, 0.04, h - 0.95), m=0)
    mb.box((x, 0.38, h - 0.02), (0.8, 0.04, 0.04), m=0)
    for sx in (-0.38, 0.38):
        mb.box((x + sx, 0.05, h - 0.02), (0.04, 0.7, 0.04), m=0)
    mb.box((x, -0.28, h - 0.02), (0.8, 0.04, 0.04), m=0)
    mb.box((x, 0.05, h + 0.005), (0.8, 0.7, 0.01), m=1)                    # top cover
    return E.obj(name + "_frame", mb, ["galv", "metal_dark"], c, parent=root)


def microscope(name, x, c, root, cl, z0=1.22, ring=True):
    mb = MB()
    mb.cyl((x, WORK_Y, z0 + 0.17), 0.022, 0.34, m=0)
    mb.cyl((x, WORK_Y, z0), 0.03, 0.03, m=1)
    mb.box((x, WORK_Y + 0.1, z0 + 0.25), (0.04, 0.2, 0.05), m=0)
    mb.box((x, WORK_Y + 0.2, z0 + 0.25), (0.05, 0.04, 0.6), m=1)
    E.obj(name + "_scope", mb, ["camera_white", "metal_dark"], c, parent=root)
    if ring:
        rb = MB()
        rb.ring((x, WORK_Y, z0 - 0.02), 0.05, 0.035, 0.008, m=0, seg=32)
        E.obj(name + "_ringlight", rb, ["lamp_lens_cool"], c, parent=root, shadow=False)
        E.light(name + "_ring_lamp", "AREA", (x, WORK_Y, z0 - 0.03), 5.0, (0.9, 0.95, 1.0), cl, size=0.08, rot=(0, 0, 0))


def monitor(name, loc, c, root, yaw=0.0, w=0.32, h=0.2):
    mb = MB()
    with mb.at(loc, rz=yaw):
        mb.box((0, 0.012, 0), (w + 0.02, 0.02, h + 0.02), m=0)
        mb.box((0, 0.06, -h / 2 - 0.1), (0.03, 0.03, 0.2), m=0)
    E.obj(name + "_monitor", mb, ["plastic_dark"], c, parent=root)
    sb = MB()
    sb.quad((-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2), m=0)
    return sb


# ==================================================================== stations
def tx_names_all(tpl):
    return [n for n in tpl if tpl[n].type == "MESH" and not n.startswith("tx_beam_")]


SUB_BASE = ["tx_carrier", "tx_carrier_traces", "tx_carrier_pads", "tx_carrier_vias", "tx_laser_bench_aln", "tx_laser_bench_traces"]
PIC = ["tx_pic_siph_die", "tx_pic_bond_pads", "tx_pic_die_attach"]


def ld(n, k):
    return [x % k for x in n]


DFB = ["tx_dfb_submount_aln_%s", "tx_dfb_submount_metal_%s", "tx_dfb_laser_die_%s", "tx_dfb_facet_glow_%s"]
LENS1, LENS2 = "tx_si_lens1_collimator_%s", "tx_si_lens2_focuser_%s"
ISO = ["tx_isolator_polarizers_%s", "tx_isolator_garnet_%s", "tx_isolator_halfwave_plate_%s"]
EPOXY = "tx_laser_optics_epoxy_%s"
LBONDS = "tx_laser_wirebonds_dc_%s"


def work_frame(name, x, c, root):
    W = E.empty(name + "_W", c, (x, WORK_Y, CHUCK_Z), (0, 0, 0), 0.02, parent=root)
    W.scale = (S, S, S)
    P = E.empty(name + "_P", c, (0, OY, OZ), (0, -PI / 2, 0), 0.01, parent=W)
    return W, P


def tool_empty(name, W, c, loc_w):
    t = E.empty(name, c, tuple(loc_w), (0, 0, 0), 0.002, parent=W)
    return t


def key_path(o, pts, interp="BEZIER"):
    """pts = [(t, Vector)] in o's parent space."""
    for t, v in pts:
        key(o, "location", f(t), Vector(v), interp)


def follow(part_g, tool, tip_off_w, pts_w, P):
    """Key a part group (sub frame, under P) to ride under a tool tip: part centre = tool point + tip_off_w (W)."""
    for t, v in pts_w:
        key(part_g, "location", f(t), w2s(Vector(v) + Vector(tip_off_w)))


def station_S1(tpl, c, root, cl):
    x = XS["S1"]
    machine_base("S1", x, c, root, badge="DB-26  DIE BONDER")
    frame_posts("S1", x, c, root)
    microscope("S1", x + 0.16, c, root, cl, z0=1.30, ring=True)
    W, P = work_frame("S1", x, c, root)
    base = tx_copy(tpl, SUB_BASE + ld(DFB, "LD2"), P, c, "S1_")
    pic = tx_copy(tpl, PIC, P, c, "S1_")
    dfb1 = tx_copy(tpl, ld(DFB, "LD1"), P, c, "S1_")
    for n in ("tx_dfb_facet_glow_LD1", "tx_dfb_facet_glow_LD2"):
        for d in (base, dfb1):
            if n in d:
                d[n].hide_render = True
    cp, lo, hi = center_of(pic.values())
    gp = group("S1_pic_grp", list(pic.values()), P, c, cp)
    cd, lod, hid = center_of(dfb1.values())
    gd = group("S1_dfb_grp", list(dfb1.values()), P, c, cd)
    dfb_parts_tpl = [tpl[n] for n in ld(DFB, "LD1")[:3]]
    pic_top_w = s2w(Vector((hi.x, cp.y, cp.z)))          # collet tip touches the PIC top centre
    dfb_top_w = s2w(Vector((hid.x, cd.y, cd.z)))
    # trays (waffle packs) left/right of the chuck, work frame; spare PICs / DFB carriers in pockets
    tray_pic = Vector((0.026, 0.0, 0.0012))
    tray_dfb = Vector((-0.024, -0.004, 0.0012))
    tb = MB()
    for ctr, n in ((tray_pic, 2), (tray_dfb, 3)):
        tb.box((ctr.x, ctr.y, 0.0006), (0.016, 0.016, 0.0012), m=0)
        for i in range(n):
            for j in range(n):
                p = 0.016 / n
                tb.box((ctr.x - 0.008 + p * (i + 0.5), ctr.y - 0.008 + p * (j + 0.5), 0.00121), (p * 0.8, p * 0.8, 0.0001), m=1)
    E.obj("S1_trays", tb, [E.principled("ASM_waffle", (0.01, 0.01, 0.012), 0.0, 0.5), "metal_dark"], c, parent=W)
    for i in range(2):                     # spare PICs (shared mesh) in the PIC tray
        for j in range(2):
            if (i, j) != (0, 0):
                spare(tpl, "tx_pic_siph_die", W, c, "S1_spare_pic_%d%d" % (i, j),
                      (tray_pic.x - 0.004 + 0.008 * i, tray_pic.y - 0.004 + 0.008 * j), 0.0012)
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                for n in ld(DFB, "LD1")[:3]:
                    o = spare(tpl, n, W, c, "S1_spare_%s_%d%d" % (n, i, j), (0, 0), 0.0)
                    cc_, _, _ = center_of(dfb_parts_tpl)
                    lo_ = min(bbox_sub(t)[0].x for t in dfb_parts_tpl)
                    o.matrix_basis = place_in_W(tpl[n], cc_, (tray_dfb.x - 0.0053 + 0.0053 * i, tray_dfb.y - 0.0053 + 0.0053 * j,
                                                              0.0012 + (cc_.x - lo_)))
    # bond head (tool frame W, true size) -> collet; body child in real metres
    tool = tool_empty("S1_collet_tip", W, c, pic_top_w)
    cb = MB()
    cb.cyl((0, 0, 0.0015), 0.0006, 0.003, m=0, r2=0.0012, seg=20)        # collet tip (vacuum), true size
    cb.cyl((0, 0, 0.007), 0.0016, 0.008, m=1, seg=20)                     # tool shank
    E.obj("S1_collet", cb, ["rubber", "chrome"], c, parent=tool)
    body = child_inv(tool, "S1_head_body", c)
    hb = MB()
    z1 = 0.011 * S
    hb.box((0, 0, z1 + 0.09), (0.05, 0.05, 0.18), m=0)                    # Z slide / bond head
    hb.box((0, 0.13, z1 + 0.16), (0.04, 0.26, 0.04), m=0)                 # arm to the gantry carriage
    hb.box((0, 0.27, z1 + 0.16), (0.1, 0.03, 0.22), m=1)                  # carriage plate (hides the beam joint)
    hb.cyl((0.045, 0, z1 + 0.06), 0.012, 0.12, m=2)                       # bond-head camera
    hb.cyl((0.045, 0, z1 - 0.002), 0.008, 0.006, m=3)
    E.obj("S1_head", hb, ["ac_white", "metal_paint", "camera_white", "lens_black"], c, parent=body)
    gb = MB()                              # gantry: posts + beam (fixed)
    yb = WORK_Y + 0.27 + 0.03
    zb = CHUCK_Z + z1 + 0.16
    for s in (-1, 1):
        gb.box((x + s * 0.34, yb + 0.02, (0.95 + zb + 0.05) / 2), (0.05, 0.05, zb + 0.05 - 0.95), m=0)
    gb.box((x, yb + 0.02, zb), (0.74, 0.06, 0.08), m=0)
    gb.box((x, yb - 0.012, zb + 0.03), (0.7, 0.006, 0.012), m=1)
    E.obj("S1_gantry", gb, ["metal_paint", "galv"], c, parent=root)
    # timeline: PIC under the collet at the tray -> over the carrier -> place (t28) -> release -> DFB tray -> pick -> place LD1
    up = Vector((0, 0, 0.006))
    A = tray_pic + Vector((0, 0, pic_top_w.z - 0.0))
    tdz = 0.0012 + (hid.x - lod.x)                       # collet tip height when the DFB sits in the tray
    tray_dfb = Vector((tray_dfb.x, tray_dfb.y, 0.0))
    pts = [(0, A + up), (16, pic_top_w + up), (24, pic_top_w + Vector((0, 0, 0.0006))), (28, pic_top_w), (31, pic_top_w),
           (38, pic_top_w + up), (54, tray_dfb + Vector((0, 0, tdz)) + up), (60, tray_dfb + Vector((0, 0, tdz))),
           (63, tray_dfb + Vector((0, 0, tdz))), (69, tray_dfb + Vector((0, 0, tdz)) + up),
           (82, dfb_top_w + up), (88, dfb_top_w), (92, dfb_top_w), (100, dfb_top_w + up)]
    key_path(tool, pts)
    off_p = s2w(cp) - pic_top_w
    follow(gp, tool, off_p, [p for p in pts if p[0] <= 28], P)
    key(gp, "location", f(31), cp)
    off_d = s2w(cd) - dfb_top_w
    tdfb = tray_dfb + Vector((0, 0, tdz))
    key(gd, "location", f(0), w2s(tdfb + off_d))
    key(gd, "location", f(60), w2s(tdfb + off_d))
    follow(gd, tool, off_d, [p for p in pts if 63 <= p[0] <= 88], P)
    key(gd, "location", f(92), cd)
    EVENTS.update({"S1_pic_place": 28, "S1_dfb_pick": 60, "S1_dfb_place": 88})
    E.hook("S1_work", c, tuple(Vector((x, WORK_Y, CHUCK_Z))), 0.0, "station", parent=root)
    return W


def tweezers(name, tool, c, appr, grip_w):
    """Tweezer tips at the tool origin (true size, W frame). appr = unit xy vector from the part toward the arm; prongs
    spread across it. Arm + hexapod head in real metres (child with scale 1/S). Returns (prongs, head_world_offset)."""
    a = Vector((appr[0], appr[1], 0.0)).normalized()
    sp = Vector((-a.y, a.x, 0.0))
    up = Vector((0, 0, 1))
    tb = MB()
    for s in (-1, 1):
        g = sp * (s * (grip_w / 2 + 0.00012))
        p1 = a * 0.004 + up * 0.009 + sp * (s * (grip_w / 2 + 0.0006))
        p2 = a * 0.009 + up * 0.014 + sp * (s * 0.0009)
        tb.seg(g, p1, 0.0005, 0.0003, m=0, up=tuple(a))
        tb.seg(p1, p2, 0.0007, 0.0004, m=0, up=tuple(a))
    pr = E.obj(name + "_prongs", tb, [E.principled("ASM_prong_steel", (0.7, 0.72, 0.75), 0.3, 0.25)], c, parent=tool)
    bb = MB()
    q0 = a * 0.009 + up * 0.014
    q1 = a * 0.04 + up * 0.05
    bb.seg(q0, q1, 0.0012, 0.0007, m=0, up=tuple(a))
    E.obj(name + "_body", bb, [E.principled("ASM_brushed_steel", (0.5, 0.52, 0.55), 0.3, 0.3)], c, parent=tool)
    body = child_inv(tool, name + "_arm", c)
    ab = MB()
    p0 = q1 * S
    p1 = a * 0.27 + up * 0.17
    ab.rod(p0, p1, 0.003, m=0, seg=12, r2=0.008)
    ab.box((p1.x, p1.y, p1.z), (0.04, 0.04, 0.04), m=1)
    ab.cyl((p1.x + a.x * 0.03, p1.y + a.y * 0.03, p1.z - 0.025), 0.06, 0.025, m=1, seg=6)   # hexapod moving platform
    E.obj(name + "_armmesh", ab, ["ac_white", "metal_paint"], c, parent=body)
    return pr, a * 0.30


def hexapod_base(name, center_xy, c, root, r=0.09):
    mb = MB()
    bx, by = center_xy
    zt = CHUCK_Z + 0.17 - 0.04
    mb.cyl((bx, by, 0.97), r, 0.04, m=0, seg=6)
    for k in range(6):
        a0 = k * PI / 3 + (0.3 if k % 2 else -0.3)
        a1 = (k // 2) * 2 * PI / 3 + PI / 3 * (1 if k % 2 else 0) + 0.4
        p0 = Vector((bx + 0.8 * r * math.cos(a0), by + 0.8 * r * math.sin(a0), 0.99))
        p1 = Vector((bx + 0.05 * math.cos(a1), by + 0.05 * math.sin(a1), zt))
        mb.rod(p0, p1, 0.006, m=1, seg=8)
        mb.rod(p0, p0 + (p1 - p0) * 0.45, 0.01, m=2, seg=8)
    E.obj(name + "_hexapod", mb, ["metal_paint", "chrome", "plastic_dark"], c, parent=root)


def station_S2(tpl, c, root, cl):
    x = XS["S2"]
    machine_base("S2", x, c, root, badge="AA-6  ACTIVE ALIGNER")
    frame_posts("S2", x, c, root)
    microscope("S2", x, c, root, cl, z0=1.26)
    W, P = work_frame("S2", x, c, root)
    names = SUB_BASE + PIC + ld(DFB, "LD1") + ld(DFB, "LD2") + [LBONDS % "LD1", LBONDS % "LD2", EPOXY % "LD1", EPOXY % "LD2",
                                                               LENS1 % "LD1", LENS1 % "LD2", LENS2 % "LD2"] + ld(ISO, "LD2")
    base = tx_copy(tpl, names, P, c, "S2_")
    iso = tx_copy(tpl, ld(ISO, "LD1"), P, c, "S2_")
    l2 = tx_copy(tpl, [LENS2 % "LD1"], P, c, "S2_")
    beam = tx_copy(tpl, ["tx_beam_cores_laser"], P, c, "S2_")["tx_beam_cores_laser"]
    beam.hide_render = False
    ci, loi, hii = center_of(iso.values())
    gi = group("S2_iso_grp", list(iso.values()), P, c, ci)
    cl2, lol, hil = center_of(l2.values())
    gl = group("S2_lens2_grp", list(l2.values()), P, c, cl2)
    iso_w, l2_w = s2w(ci), s2w(cl2)
    # tweezers: A (isolator) from +X, B (lens 2) from -X; tool origin = part centre (tips grip the part sides)
    tA = tool_empty("S2_tweezA_tip", W, c, iso_w)
    tB = tool_empty("S2_tweezB_tip", W, c, l2_w)
    apA, apB = (0.0, 1.0), (-0.8, -0.6)       # A from the back (DFB side), B from the front-left (PIC side)
    _, offA = tweezers("S2_tweezA", tA, c, apA, (hii.z - loi.z))     # grip across W x (= sub z extent)
    _, offB = tweezers("S2_tweezB", tB, c, apB, (hil.z - lol.z))
    for nm, tw, off in (("S2_A", iso_w, offA), ("S2_B", l2_w, offB)):
        cw = Vector((x, WORK_Y, CHUCK_Z)) + S * tw + off
        hexapod_base(nm, (cw.x, cw.y), c, root)
    hi_ = Vector((0.0, -0.0008, 0.007))
    near = Vector((0, 0, 0.0005))
    dz = lambda a, b=0.0, cc=0.0: Vector((a, b, cc))
    hi_ = Vector((0.0, 0.0012, 0.007))
    ptsA = [(0, iso_w + hi_), (13, iso_w + near), (18, iso_w), (20, iso_w + dz(0.00005, 0, 0)), (22, iso_w + dz(-0.00004, 0.00003)),
            (24, iso_w + dz(0.00001, -0.00001)), (26, iso_w), (28, iso_w), (38, iso_w + dz(0.0, 0.0012, 0.0065)),
            (52, iso_w + dz(0.0, 0.0012, 0.0065))]
    ptsB = [(0, l2_w + dz(-0.0016, -0.0012, 0.008)), (14, l2_w + dz(-0.0016, -0.0012, 0.008)), (27, l2_w + near), (31, l2_w),
            (33, l2_w + dz(-0.00004, 0.00003)), (35, l2_w + dz(0.00003, -0.00002)), (36, l2_w), (39, l2_w),
            (49, l2_w + dz(-0.0016, -0.0012, 0.0065)), (60, l2_w + dz(-0.0016, -0.0012, 0.0065))]
    key_path(tA, ptsA)
    key_path(tB, ptsB)
    follow(gi, tA, Vector(), [p for p in ptsA if p[0] <= 26], P)
    key(gi, "location", f(28), ci)
    follow(gl, tB, Vector(), [p for p in ptsB if p[0] <= 36], P)
    key(gl, "location", f(39), cl2)
    # prong release: open the tips a little (prongs scale in Y about the tip) at release
    for t_, frames in ((tA, (26, 28)), (tB, (36, 39))):
        pr = next(o for o in t_.children if o.name.endswith("_prongs"))
        key(pr, "scale", f(frames[0]), Vector((1, 1, 1)))
        key(pr, "scale", f(frames[1]), Vector((1.35, 1.35, 1)))
    # UV cure: two light-guide tips aimed at the parts + violet light, epoxy glows; t29-36
    uv_m = emis_mat("ASM_uv_tip", (0.45, 0.2, 1.0), 0.0, dark=True)
    ub = MB()
    tips = []
    aim = Vector((x, WORK_Y, CHUCK_Z)) + (S * iso_w)
    for s in (-1, 1):
        tip = aim + Vector((s * 0.03, 0.04, 0.05))
        d = (aim - tip).normalized()
        base_p = Vector((x + s * 0.3, WORK_Y + 0.3, 1.5))
        mid = Vector((x + s * 0.1, WORK_Y + 0.15, 1.2))
        ub.tube([base_p, mid, tip - d * 0.06, tip], 0.006, seg=10, m=0)
        tips.append((tip, d))
    E.obj("S2_uv_guides", ub, ["cable"], c, parent=root)
    tb = MB()
    for tip, d in tips:
        tb.rod(tip - d * 0.03, tip, 0.008, m=0, seg=16)
        tb.rod(tip, tip + d * 0.002, 0.005, m=1, seg=16)
    E.obj("S2_uv_tips", tb, ["chrome", uv_m], c, parent=root)
    uvl = E.light("S2_uv_flash", "POINT", tuple(Vector((x, WORK_Y - 0.02, CHUCK_Z + 0.03)) + S * iso_w), 0.0, (0.5, 0.25, 1.0), cl,
                  radius=0.01, shadow=False)
    es = bsdf(uv_m).inputs["Emission Strength"]
    for t_, v in ((0, 0.0), (55, 0.0), (56, 40.0), (63, 40.0), (65, 0.0)):
        key_val(es, f(t_), v)
        uvl.data.energy = {0.0: 0.0, 40.0: 1.2}[v]
        uvl.data.keyframe_insert("energy", frame=f(t_))
    ep = base[EPOXY % "LD1"]
    epm = own_material(ep, "_S2glow")
    b = bsdf(epm[0])
    b.inputs["Emission Color"].default_value = (0.55, 0.3, 1.0, 1.0)
    for t_, v in ((0, 0.0), (55, 0.0), (57, 6.0), (63, 6.0), (66, 0.0)):
        key_val(b.inputs["Emission Strength"], f(t_), v)
    EVENTS.update({"S2_iso_place": 18, "S2_lens2_place": 31, "S2_tool_leaves_lens": 39, "S2_dfb_on": DFB_ON, "S2_power_peak": DFB_ON + 14,
                   "S2_uv_on": 56, "S2_uv_off": 64})
    # v4: DFB LD1 facet glow off until DFB_ON (own material copy); LD2 stays dark in this station
    base["tx_dfb_facet_glow_LD2"].hide_render = True
    for m in own_material(base["tx_dfb_facet_glow_LD1"], "_S2"):
        for nd in m.node_tree.nodes:
            if nd.type == "BSDF_PRINCIPLED":
                nd.inputs["Base Color"].default_value = (0.03, 0.02, 0.02, 1.0)
            inp = nd.inputs.get("Emission Strength") or (nd.inputs.get("Strength") if nd.type == "EMISSION" else None)
            if inp is not None:
                v0 = max(inp.default_value, 1.0)
                for t_, v in ((0, 0.0), (DFB_ON - 1, 0.0), (DFB_ON, v0 * 1.5), (DFB_ON + 3, v0)):
                    key_val(inp, f(t_), v, "CONSTANT" if t_ == DFB_ON - 1 else "LINEAR")
    # laser on: facet glow on; beam faint until aligned, brighter after lens 2
    bm_ = own_material(beam, "_S2")
    for m in bm_:
        for nd in m.node_tree.nodes:
            inp = nd.inputs.get("Emission Strength") or (nd.inputs.get("Strength") if nd.type == "EMISSION" else None)
            if inp is not None:
                v0 = inp.default_value
                for t_, v in ((0, 0.0), (DFB_ON - 1, 0.0), (DFB_ON, 1.0)):
                    key_val(inp, f(t_), v0 * v, "CONSTANT" if t_ == DFB_ON - 1 else "LINEAR")
    # visible laser core LD1 (viz only, 1311 nm is invisible): DFB facet -> lens 1 -> isolator -> lens 2 -> PIC edge SSC
    fg = tpl["tx_dfb_facet_glow_LD1"]
    fgc, _, _ = center_of([fg])
    b0 = s2w(fgc)
    b1 = s2w(Vector((fgc.x, -2.25e-3, fgc.z)))
    rb = MB()
    rb.rod(b0, b1, 0.000035, m=0, seg=8)
    rb.rod(b0, b1, 0.00009, m=1, seg=8)
    bcore = emis_mat("ASM_beam_core", (1.0, 0.25, 0.15), 0.0, dark=True)
    bhalo = emis_mat("ASM_beam_halo", (1.0, 0.06, 0.03), 0.0, dark=True)
    if hasattr(bhalo, "surface_render_method"):
        bhalo.surface_render_method = "BLENDED"
    bsdf(bhalo).inputs["Alpha"].default_value = 0.35
    E.obj("S2_beam_viz_LD1", rb, [bcore, bhalo], c, parent=W, shadow=False)
    for m_, vals in ((bcore, (0.0, 0.0, 20.0, 14.0)), (bhalo, (0.0, 0.0, 6.0, 4.0))):
        for t_, v in zip((0, DFB_ON - 1, DFB_ON, DFB_ON + 3), vals):
            key_val(bsdf(m_).inputs["Emission Strength"], f(t_), v, "CONSTANT" if t_ == DFB_ON - 1 else "LINEAR")
    # monitor: COUPLED POWER bar rising (alignment feedback)
    sb = monitor("S2", (x - 0.28, 0.3, 1.42), c, root)
    scr_m = E.principled("ASM_screen_dark", (0.004, 0.006, 0.01), 0.0, 0.3, (0.01, 0.02, 0.03), 1.0)
    scr = E.obj("S2_screen", sb, [scr_m], c, parent=root, loc=(x - 0.28, 0.3 - 0.003, 1.42), shadow=False)
    bar = MB()
    bar.quad((0, 0, -0.02), (0.26, 0, -0.02), (0.26, 0, 0.02), (0, 0, 0.02), m=0)
    bo = E.obj("S2_power_bar", bar, [emis_mat("ASM_bar_green", (0.2, 1.0, 0.35), 4.0)], c, parent=root,
               loc=(x - 0.28 - 0.13, 0.3 - 0.005, 1.40), shadow=False)
    for t_, v in ((0, 0.0), (DFB_ON - 1, 0.0), (DFB_ON + 3, 0.5), (DFB_ON + 7, 0.8), (DFB_ON + 11, 0.95), (DFB_ON + 14, 1.0)):
        key(bo, "scale", f(t_), Vector((max(v, 0.001), 1, 1)), "CONSTANT" if t_ == DFB_ON - 1 else "BEZIER")
    lab_m = emis_mat("ASM_screen_text", (0.8, 0.95, 1.0), 2.0)
    E.text("S2_screen_lbl", "COUPLED POWER", 0.022, c, (x - 0.28, 0.3 - 0.005, 1.47), (PI / 2, 0, 0), mat=lab_m, fnt="Arial Bold",
           parent=root)
    # gel-pak with spare isolators + lenses (shared meshes)
    gb_ = MB()
    gp0 = Vector((0.0, -0.017, 0.0))
    gb_.box((gp0.x, gp0.y, 0.0006), (0.024, 0.012, 0.0012), m=0)
    E.obj("S2_gelpak", gb_, [E.principled("ASM_gelpak", (0.03, 0.035, 0.04), 0.0, 0.2)], c, parent=W)
    iso_tpl = [tpl[n % "LD1"] for n in ISO]
    ci_t, lo_t, _ = center_of(iso_tpl)
    for i in range(5):
        for n in ISO:
            o = spare(tpl, n % "LD1", W, c, "S2_spare_%s_%d" % (n % "LD1", i), (0, 0), 0.0)
            o.matrix_basis = place_in_W(tpl[n % "LD1"], ci_t, (gp0.x - 0.008 + 0.004 * i, gp0.y - 0.003, 0.0012 + ci_t.x - lo_t.x))
        spare(tpl, LENS2 % "LD1", W, c, "S2_spare_lens2_%d" % i, (gp0.x - 0.008 + 0.004 * i, gp0.y + 0.003), 0.0012)
    E.light("S2_key", "AREA", tuple(aim + Vector((0.22, -0.12, 0.2))), 1.5, (1.0, 0.95, 0.9), cl, size=0.12, target=tuple(aim))
    E.hook("S2_work", c, tuple(Vector((x, WORK_Y, CHUCK_Z)) + S * iso_w), 0.0, "station", parent=root)
    return W, S * iso_w, S * l2_w


def station_S3(tpl, c, root, cl):
    x = XS["S3"]
    machine_base("S3", x, c, root, badge="WB-26  BALL BONDER")
    frame_posts("S3", x, c, root)
    microscope("S3", x - 0.12, c, root, cl, z0=1.30)
    W, P = work_frame("S3", x, c, root)
    names = [n for n in tx_names_all(tpl) if n.startswith(("tx_carrier", "tx_laser", "tx_pic", "tx_dfb", "tx_si_lens", "tx_isolator"))]
    names += ["tx_wirebonds_dc_ls"]
    base = tx_copy(tpl, names, P, c, "S3_")
    for n in ("tx_dfb_facet_glow_LD1", "tx_dfb_facet_glow_LD2"):
        base[n].hide_render = True
    hs = tx_copy(tpl, ["tx_wirebonds_hs_diff_gssg"], P, c, "S3_")["tx_wirebonds_hs_diff_gssg"]
    balls = tx_copy(tpl, ["tx_wirebond_balls"], P, c, "S3_")["tx_wirebond_balls"]
    hs.data = hs.data.copy()
    balls.data = balls.data.copy()
    wires = split_loose(hs, c)
    bl = split_loose(balls, c)
    # endpoints in sub frame: PIC pad end = max Y (first bond, ball), carrier end = min Y (stitch)
    info = []
    for w in wires:
        vs = [w.matrix_basis @ v.co for v in w.data.vertices]
        ymax = max(v.y for v in vs)
        ymin = min(v.y for v in vs)
        e1 = [v for v in vs if v.y > ymax - 0.02e-3]
        e2 = [v for v in vs if v.y < ymin + 0.02e-3]
        p1 = sum(e1, Vector()) / len(e1)
        p2 = sum(e2, Vector()) / len(e2)
        top = max(v.x for v in vs)
        info.append((p1.z, w, p1, p2, top))
    info.sort(key=lambda r: r[0])
    LOG.append("S3 HS wires %d, balls %d" % (len(wires), len(bl)))
    # balls near a wire end ride with that wire
    bcen = []
    for b in bl:
        vs = [b.matrix_basis @ v.co for v in b.data.vertices]
        bcen.append((b, sum(vs, Vector()) / len(vs)))
    K0, PERIOD, T1 = 0, 5, 2                 # v4.2: every HS wire bonded in sequence (same 5 frames per bond)
    capw = []
    for k, (_, w, p1, p2, top) in enumerate(info):
        set_origin(w, w.matrix_basis.inverted() @ p1)
        ts = T1 + PERIOD * (k - K0)
        mine = [b for b, bc in bcen if (bc - p1).length < 0.06e-3 or (bc - p2).length < 0.06e-3]
        if k < K0:
            continue
        if ts > 200:
            ts = 200
        key(w, "scale", f(0), Vector((0, 0, 0)), "CONSTANT")
        key(w, "scale", f(ts + 1), Vector((0.05, 0.05, 0.05)), "LINEAR")
        key(w, "scale", f(ts + 3), Vector((1, 1, 1)), "LINEAR")
        for b in mine:
            bc = [bc_ for bb, bc_ in bcen if bb is b][0]
            set_origin(b, b.matrix_basis.inverted() @ bc)
            t_on = ts + 1 if (bc - p1).length < (bc - p2).length else ts + 3
            key(b, "scale", f(0), Vector((0, 0, 0)), "CONSTANT")
            key(b, "scale", f(t_on), Vector((1, 1, 1)), "CONSTANT")
        if ts < DUR + 10:
            capw.append((ts, s2w(p1), s2w(p2), s2w(Vector((top, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)))))
    LOG.append("S3 bonds animated %d (from wire %d, %d frames each)" % (len(capw), K0, PERIOD))
    # capillary path (tool frame W): above pad1 -> touch -> loop top -> touch pad2 (stitch) -> rise
    h = Vector((0, 0, 0.00025))
    tool = tool_empty("S3_capillary_tip", W, c, capw[0][1] + h)
    pts = []
    for ts, a, b, top in capw:
        pts += [(ts, a + h), (ts + 1, a), (ts + 2, top + Vector((0, 0, 0.00012))), (ts + 3, b), (ts + 4, b + h)]
    pts = [p for p in pts if p[0] >= 0]
    key_path(tool, pts, "LINEAR")
    # capillary (true size), wire clamp, EFO, horn + head (real metres)
    cb = MB()
    cb.cyl((0, 0, 0.0015), 0.000033, 0.003, m=0, r2=0.000794, seg=20)    # bottleneck cone, 66 um tip (US 7,407,080 class)
    cb.cyl((0, 0, 0.0071), 0.000794, 0.0082, m=0, seg=20)                 # 1/16 in shank, 11.1 mm total (D)
    cb.rod((0, 0, 0.0), (0, 0, 0.026), 0.0000127, m=1, seg=6)             # 25 um Au wire through the tool
    cb.box((0.0009, 0, 0.0145), (0.0008, 0.002, 0.003), m=2)              # wire clamp jaws
    cb.box((-0.0009, 0, 0.0145), (0.0008, 0.002, 0.003), m=2)
    E.obj("S3_capillary", cb, [E.principled("ASM_ceramic", (0.75, 0.72, 0.68), 0.0, 0.35), 
                               E.principled("ASM_gold_wire", (1.0, 0.72, 0.3), 1.0, 0.25), "chrome"], c, parent=tool)
    spark_m = emis_mat("ASM_efo_spark", (0.6, 0.75, 1.0), 0.0, dark=True)
    sp = MB()
    sp.ellipsoid((0, 0, -0.00012), (0.0001, 0.0001, 0.0001), seg=10, rings=6, m=0)
    E.obj("S3_efo_spark", sp, [spark_m], c, parent=tool, shadow=False)
    for ts, a, b, top in capw:
        for dt, v in ((4, 0.0), (4.5, 120.0), (5, 0.0)):
            key_val(bsdf(spark_m).inputs["Emission Strength"], f(ts) + dt, v)
    body = child_inv(tool, "S3_head_body", c)
    hb = MB()
    zc = 0.0115 * S
    hb.rod((0, 0, zc), (0, 0.19, zc), 0.004, m=0, seg=16, r2=0.011)        # ultrasonic horn (transducer) toward the head
    hb.cyl((0, 0.2, zc), 0.016, 0.03, axis="y", m=1, seg=20)
    hb.box((0, 0.3, zc + 0.04), (0.12, 0.17, 0.14), m=2)                  # bond head body
    hb.cyl((0, 0.31, zc + 0.15), 0.035, 0.03, axis="x", m=3, seg=24)      # wire spool
    hb.rod((0.0, 0.31, zc + 0.15), (0, 0.0, zc + 0.08), 0.0006, m=3, seg=4)
    hb.rod((0.0, 0.0, zc + 0.08), (0, 0.0, 0.026 * S), 0.0005, m=3, seg=4)
    hb.rod((0.03, 0.2, zc - 0.01), (0.004, 0.006, 0.0007 * S), 0.0012, m=4, seg=8)   # EFO wand
    E.obj("S3_head", hb, ["chrome", "metal_paint", "ac_white", E.principled("ASM_spool_gold", (1.0, 0.7, 0.28), 1.0, 0.3),
                          "metal_dark"], c, parent=body)
    # heated workholder ring + status
    wb = MB()
    wb.box((x, WORK_Y, 0.958), (0.16, 0.12, 0.016), m=0)
    wb.box((x, WORK_Y - 0.061, 0.958), (0.12, 0.002, 0.004), m=1)
    E.obj("S3_workholder", wb, ["metal_dark", E.principled("ASM_heater", (1.0, 0.3, 0.05), 0.0, 0.4, (1.0, 0.3, 0.05), 3.0)], c,
          parent=root)
    E.text("S3_heater_lbl", "HEATER ON", 0.02, c, (x + 0.2, -0.306, 0.84), (PI / 2, 0, 0), mat="neon_orange", fnt="Arial Bold",
           parent=root)
    mid = capw[min(6, len(capw) - 1)]
    EVENTS.update({"S3_bond_first": T1, "S3_bond_period": PERIOD, "S3_bonds_in_window": len([1 for p in capw if p[0] < DUR])})
    for k, cw in enumerate(capw):                 # first-bond touchdown of every wire (SFX)
        EVENTS["S3_bond_%02d" % (k + 1)] = cw[0] + 1
    aim3 = Vector((x, WORK_Y, CHUCK_Z)) + S * mid[1]
    E.light("S3_key", "AREA", tuple(aim3 + Vector((0.2, -0.2, 0.18))), 4.0, (1.0, 0.95, 0.9), cl, size=0.12, target=tuple(aim3))
    E.hook("S3_work", c, tuple(Vector((x, WORK_Y, CHUCK_Z)) + S * mid[1]), 0.0, "station", parent=root)
    return W, [(ts, S * a, S * b) for ts, a, b, _ in capw]


def bird_stage(hero_objs, c, root, name, loc, yaw, lid_colls=()):
    br = E.empty(name + "_root", c, loc, (0, 0, yaw), 0.05, parent=root)
    br.scale = (S, S, S)
    lid = None
    if lid_colls:
        lid = E.empty(name + "_lid", c, (0, 0, 0), (0, 0, 0), 0.02, parent=br)
    mp = hero_copy(hero_objs, br, c, name + "_", lid=lid, lid_colls=lid_colls)
    for n, o in mp.items():
        if n.startswith("leg_tx_band_text_OUTNOLIGHT"):
            o.hide_render = True             # H's band text still says OUTNOLIGHT (rename is H's); illegible at these shots anyway
    return br, lid, mp


def bench(c, root):
    mb = MB()
    x0, x1 = 0.72, 2.40
    mb.box(((x0 + x1) / 2, 0.05, BENCH_Z / 2), (x1 - x0, 0.62, BENCH_Z), m=0)
    mb.box(((x0 + x1) / 2, 0.05, BENCH_Z - 0.019), (x1 - x0 + 0.02, 0.64, 0.04), m=1)   # top plate 1 mm proud (no z-fight)
    mb.box(((x0 + x1) / 2, -0.2725, BENCH_Z - 0.06), (x1 - x0, 0.004, 0.012), m=2)
    xs = XS["S4"]
    mb.box((xs, WORK_Y, BENCH_Z + 0.009), (0.34, 0.28, 0.016), m=3)        # pallet (S4)
    mb.box((xs, WORK_Y - 0.136, BENCH_Z + 0.022), (0.33, 0.018, 0.018), m=4)   # yellow edge strip, offset from the pallet faces
    mb.box((XS["S5"], WORK_Y + 0.01, BENCH_Z + 0.009), (0.52, 0.44, 0.016), m=3)   # S5 fixture base plate
    mb.box((2.62, 0.05, 0.25), (0.36, 0.5, 0.5), m=1)                      # DISPATCH step
    E.obj("asm_bench", mb, ["ac_white", "metal_dark", "led_cyan", "metal_dark", "paint_yellow"], c, parent=root)
    E.text("asm_dispatch_lbl", "DISPATCH", 0.05, c, (2.62, -0.202, 0.36), (PI / 2, 0, 0), mat="neon_green", fnt="Arial Bold",
           parent=root)


def station_S4(hero_objs, c, root, cl):
    x = XS["S4"]
    yaw = 0.0
    br, lid, mp = bird_stage(hero_objs, c, root, "S4_bird", (x, WORK_Y, BENCH_Z + 0.017), yaw,
                             lid_colls=("shell_top", "fins"))
    for n, o in mp.items():                 # S4 bird eyes dark
        if n.startswith(("eye_tx_iris_glow_ring", "eye_rx_iris_glow_ring")):
            for m in own_material(o, "_S4"):
                b = bsdf(m)
                b.inputs["Emission Strength"].default_value = 0.0
    LIFT = 0.085                            # true metres in the bird frame (x2.6 = 0.22 m)
    for t_, z in ((0, LIFT), (154, LIFT), (164, 0.0)):
        key(lid, "location", f(t_), Vector((0, 0, z)))
    # press: columns + crosshead + cylinder (fixed); rod + platen + cups move with the lid (in hall metres)
    zt = BENCH_Z + 0.017 + 0.0938 * S                                     # shell top (head) when closed
    pb = MB()
    for s in (-1, 1):
        pb.box((x + s * 0.24, 0.2, (BENCH_Z + 1.62) / 2), (0.05, 0.05, 1.62 - BENCH_Z), m=0)
    pb.box((x, 0.2, 1.62), (0.56, 0.08, 0.08), m=0)
    pb.box((x, 0.06, 1.62), (0.12, 0.22, 0.06), m=0)
    pb.cyl((x, -0.03, 1.5), 0.045, 0.22, m=1, seg=24)
    E.obj("S4_press_frame", pb, ["metal_paint", "chrome"], c, parent=root)
    plat = E.empty("S4_platen", c, (0, 0, 0), (0, 0, 0), 0.02, parent=br)    # v4: platen rises again after the close
    plat.scale = (1 / S, 1 / S, 1 / S)
    for t_, z in ((0, LIFT), (154, LIFT), (164, 0.0), (168, 0.0), (178, LIFT)):
        key(plat, "location", f(t_), Vector((0, 0, z)))
    qb = MB()
    zl = 0.0938 * S                                                       # local (hall metres) above the bird root
    qb.box((0, -0.02, zl + 0.05), (0.3, 0.22, 0.02), m=0)
    qb.rod((0, 0.02 - 0.05, zl + 0.06), (0, -0.03 + 0.0 + 0.0, 1.5 - 0.11 - (BENCH_Z + 0.017)), 0.012, m=1, seg=12)
    for cx, cy, cz in ((0.0, -0.114, zl), (0.0, 0.0, 0.0669 * S), (0.0, 0.06, 0.06 * S)):
        qb.cyl((cx, cy, (cz + zl + 0.04) / 2), 0.006, zl + 0.04 - cz, m=1, seg=10)
        qb.cyl((cx, cy, cz + 0.003), 0.014, 0.006, m=2, seg=16, r2=0.008)
    E.obj("S4_platen_mesh", qb, ["ac_white", "chrome", "rubber"], c, parent=plat)
    EVENTS.update({"S4_press_down": 154, "S4_lid_closed": 164, "S4_platen_up": 168, "S4_platen_home": 178})
    # dispenser: syringe on a small arm; needle tip traces TIM dots on the DSP thermal pad (t12-40)
    pad_top = 0.09006
    pts_b = [(-0.003, -0.046), (0.003, -0.046), (0.003, -0.0425), (-0.003, -0.0425), (-0.003, -0.039), (0.003, -0.039),
             (0.003, -0.0405), (0.0, -0.0425)]
    cu = bpy.data.curves.new("S4_tim_bead", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts_b) - 1)
    for i, (px, py) in enumerate(pts_b):
        sp.points[i].co = (px, py, pad_top + 0.0003, 1.0)
    cu.bevel_depth = 0.0004
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    bead = bpy.data.objects.new("S4_tim_bead", cu)
    c.objects.link(bead)
    bead.parent = br
    cu.materials.append(E.principled("ASM_tim_grey", (0.55, 0.56, 0.6), 0.2, 0.35))
    cu.bevel_factor_end = 0.0
    cu.keyframe_insert("bevel_factor_end", frame=f(124))
    cu.bevel_factor_end = 1.0
    cu.keyframe_insert("bevel_factor_end", frame=f(150))
    for fc in cu.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    # needle tip follows the bead (bird frame, true metres) via an empty under the bird root
    nt = E.empty("S4_needle_tip", c, (0, -0.046, pad_top + 0.02), (0, 0, 0), 0.002, parent=br)
    seglen = [0.0]
    for i in range(1, len(pts_b)):
        seglen.append(seglen[-1] + math.dist(pts_b[i], pts_b[i - 1]))
    key(nt, "location", f(0), Vector((pts_b[0][0] + 0.004, pts_b[0][1], pad_top + 0.03)))
    key(nt, "location", f(116), Vector((pts_b[0][0] + 0.004, pts_b[0][1], pad_top + 0.03)))
    key(nt, "location", f(122), Vector((pts_b[0][0], pts_b[0][1], pad_top + 0.0012)))
    for i, (px, py) in enumerate(pts_b):
        key(nt, "location", f(124 + 26 * seglen[i] / seglen[-1]), Vector((px, py, pad_top + 0.0012)), "LINEAR")
    key(nt, "location", f(153), Vector((pts_b[-1][0] + 0.004, pts_b[-1][1], pad_top + 0.035)))
    key(nt, "location", f(160), Vector((-0.045, -0.04, 0.12)))
    nb = MB()
    nb.cyl((0, 0, 0.004), 0.00025, 0.008, m=0, seg=10)                    # needle (true size)
    nb.cyl((0, 0, 0.0095), 0.0012, 0.003, m=1, seg=12, r2=0.0005)
    E.obj("S4_needle", nb, ["chrome", E.principled("ASM_hub_blue", (0.05, 0.2, 0.8), 0.0, 0.4)], c, parent=nt)
    nbody = E.empty("S4_syringe_body", c, (0, 0, 0), (0, 0, 0), 0.02, parent=nt)
    nbody.scale = (1 / S, 1 / S, 1 / S)
    sb = MB()
    sb.cyl((0, 0, 0.011 * S + 0.05), 0.012, 0.1, m=0, seg=16)
    sb.box((0.0, 0.0, 0.011 * S + 0.12), (0.03, 0.03, 0.04), m=1)
    sb.seg((0, 0.0, 0.011 * S + 0.12), (-0.34, 0.12, 0.011 * S + 0.12), 0.025, 0.025, m=1)   # arm toward -X (clear of the S5 screens)
    sb.box((-0.34, 0.12, 0.011 * S + 0.12), (0.07, 0.07, 0.09), m=1)
    E.obj("S4_syringe", sb, [E.principled("ASM_syringe", (0.8, 0.82, 0.85), 0.0, 0.1), "ac_white"], c, parent=nbody)
    EVENTS.update({"S4_dispense_start": 124, "S4_dispense_end": 150})
    E.hook("S4_work", c, (x, WORK_Y, BENCH_Z + 0.15), 0.0, "station", parent=root)
    return br


def eye_atlas_material(name, atlas, phase_off, n=24, cols=6):
    """Screen material reading one cell of the varying PAM4 eye atlas; cell = floor(phase + offset) mod n.
    Inputs keyed by the builder: value nodes 'phase' (advances 1 cell per build frame) and 'eye_on' (0 dark .. 1 lit)."""
    rows = math.ceil(n / cols)
    m, bld, out_ = vfx.nodes.new_material(name)
    nt = m.node_tree
    N = nt.nodes
    L = nt.links.new
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L(tc.outputs["UV"], sep.inputs[0])
    ph = N.new("ShaderNodeValue")
    ph.name = ph.label = "phase"

    def mth(op, a, b_):
        nd = N.new("ShaderNodeMath")
        nd.operation = op
        for i, v in enumerate((a, b_)):
            if isinstance(v, (int, float)):
                nd.inputs[i].default_value = v
            else:
                L(v, nd.inputs[i])
        return nd.outputs[0]
    idx = mth("FLOOR", mth("MODULO", mth("ADD", ph.outputs[0], phase_off + 1000.0 * n), float(n)), 0.0)
    col = mth("MODULO", idx, float(cols))
    row = mth("FLOOR", mth("DIVIDE", idx, float(cols)), 0.0)
    u = mth("DIVIDE", mth("ADD", sep.outputs[0], col), float(cols))
    v = mth("DIVIDE", mth("ADD", sep.outputs[1], mth("SUBTRACT", float(rows - 1), row)), float(rows))
    cmb = N.new("ShaderNodeCombineXYZ")
    L(u, cmb.inputs[0])
    L(v, cmb.inputs[1])
    tex = N.new("ShaderNodeTexImage")
    tex.image = atlas
    tex.interpolation = "Linear"
    L(cmb.outputs[0], tex.inputs[0])
    on = N.new("ShaderNodeValue")
    on.name = on.label = "eye_on"
    emn = N.new("ShaderNodeEmission")
    L(tex.outputs["Color"], emn.inputs["Color"])
    L(mth("MULTIPLY", on.outputs[0], 2.5), emn.inputs["Strength"])
    L(emn.outputs[0], out_.inputs["Surface"])
    return m, ph.outputs[0], on.outputs[0]


def quad_uv(o):
    me = o.data
    uvl = me.uv_layers[0] if me.uv_layers else me.uv_layers.new()
    for poly in me.polygons:
        for li, (u, v) in zip(poly.loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
            uvl.data[li].uv = (u, v)


def loose_centers(me):
    """Centres (mesh space) of the connected components of a mesh (pads)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    seen, out = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            x_ = stack.pop()
            comp.append(x_.co.copy())
            for e in x_.link_edges:
                y_ = e.other_vert(x_)
                if y_.index not in seen:
                    seen.add(y_.index)
                    stack.append(y_)
        out.append(sum(comp, Vector()) / len(comp))
    bm.free()
    return out


def loose_parts(me):
    """(centre, size) per connected component (mesh space)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    seen, out = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            x_ = stack.pop()
            comp.append(x_.co.copy())
            for e in x_.link_edges:
                y_ = e.other_vert(x_)
                if y_.index not in seen:
                    seen.add(y_.index)
                    stack.append(y_)
        lo = Vector([min(c_[i] for c_ in comp) for i in range(3)])
        hi = Vector([max(c_[i] for c_ in comp) for i in range(3)])
        out.append(((lo + hi) / 2, hi - lo))
    bm.free()
    return out


def screen_material(name, base_img, reveal_img=None, axis="u"):
    """Instrument screen: base texture (+ optional RGBA layer revealed along u (left->right) or v (top->bottom)).
    Value nodes keyed by the builder: 'screen_on' (0.03 dark .. 1 lit), 'reveal' (0..1)."""
    m, bld, out_ = vfx.nodes.new_material(name)
    nt = m.node_tree
    N, L = nt.nodes, nt.links.new
    tc = N.new("ShaderNodeTexCoord")
    tb = N.new("ShaderNodeTexImage")
    tb.image = base_img
    L(tc.outputs["UV"], tb.inputs[0])
    col = tb.outputs["Color"]
    on = N.new("ShaderNodeValue")
    on.name = on.label = "screen_on"
    rv = N.new("ShaderNodeValue")
    rv.name = rv.label = "reveal"
    if reveal_img is not None:
        tr = N.new("ShaderNodeTexImage")
        tr.image = reveal_img
        L(tc.outputs["UV"], tr.inputs[0])
        sep = N.new("ShaderNodeSeparateXYZ")
        L(tc.outputs["UV"], sep.inputs[0])
        coord = sep.outputs[0]
        if axis == "v":
            inv = N.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            L(sep.outputs[1], inv.inputs[1])
            coord = inv.outputs[0]
        lt = N.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        L(coord, lt.inputs[0])
        L(rv.outputs[0], lt.inputs[1])
        a = N.new("ShaderNodeMath")
        a.operation = "MULTIPLY"
        L(lt.outputs[0], a.inputs[0])
        L(tr.outputs["Alpha"], a.inputs[1])
        mx = N.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        L(a.outputs[0], mx.inputs["Factor"])
        L(col, mx.inputs[6])
        L(tr.outputs["Color"], mx.inputs[7])
        col = mx.outputs[2]
    emn = N.new("ShaderNodeEmission")
    L(col, emn.inputs["Color"])
    mul = N.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    L(on.outputs[0], mul.inputs[0])
    mul.inputs[1].default_value = 1.4
    L(mul.outputs[0], emn.inputs["Strength"])
    L(emn.outputs[0], out_.inputs["Surface"])
    return m, on.outputs[0], rv.outputs[0]


def key_sock(sock, keys):
    """keys = [(t, value, interp)] on a node socket (CONSTANT holds where asked)."""
    for t_, v, it in keys:
        key_val(sock, f(t_), v, it)


def test_head(name, c, root, label, side_sign):
    """Free-space optical test head (adapted from motorized optical probe heads; no fiber on the birds): housing with a
    2x4 lens-array window matching the bird's 2x4 lenslet eye, white ring LED, carriage block on top, pigtail to the cable
    chain. Local +X points at the eye; origin = window centre. side_sign: which local Y side faces the camera (+1 / -1)."""
    mb = MB()
    mb.box((-0.035, 0, 0), (0.07, 0.05, 0.042), m=0)
    mb.box((-0.002, 0, 0), (0.004, 0.026, 0.022), m=1)
    for i in range(4):
        for j in range(2):
            mb.ellipsoid((0.0005, -0.003 + 0.002 * i, -0.001 + 0.002 * j), (0.0006, 0.0008, 0.0008), seg=10, rings=6, m=2)
    mb.ring((0.0, 0, 0), 0.016, 0.0135, 0.002, axis="x", seg=24, m=3)
    mb.box((-0.035, 0, 0.03), (0.05, 0.03, 0.018), m=4)                   # carriage under the rail
    mb.tube([(-0.07, 0, 0.0), (-0.09, 0, 0.012), (-0.1, 0, 0.04)], 0.004, seg=8, m=5)
    led = emis_mat(name + "_led", (0.85, 0.9, 1.0), 0.3)
    o = E.obj(name, mb, ["ac_white", "metal_dark", "lens_black", led, "metal_paint", "cable"], c, parent=root)
    rz = 0.0 if side_sign < 0 else PI
    E.text(name + "_lbl", label, 0.0075, c, (-0.035, side_sign * 0.0256, 0.004), (PI / 2, 0, rz), mat="plastic_dark",
           fnt="Arial Bold", parent=o)
    return o, led


def instrument(name, ctr, size, c, root, label, sub, ports=0, port_kind="coax", screen=True, rows=1, knobs=True):
    """Generic bench instrument (parody label, no trade dress): grey box, dark bezel, small panel screen, knobs, ports."""
    x, y, z = ctr
    w, d, h = size
    mb = MB()
    mb.box((x, y + d / 2, z + h / 2), (w, d, h), m=0)
    mb.box((x, y - 0.002, z + h / 2), (w - 0.01, 0.004, h - 0.01), m=1)
    if screen:
        mb.box((x - w * 0.18, y - 0.005, z + h * 0.58), (w * 0.5, 0.002, h * 0.55), m=2)
    for k in range(3 if knobs else 0):
        mb.cyl((x + w * 0.2 + k * 0.03, y - 0.008, z + h * 0.7), 0.009, 0.012, axis="y", m=3, seg=14)
    per = max(1, ports // rows)
    port_pos = []
    for k in range(ports):
        r_, i_ = divmod(k, per)
        px = x - w / 2 + 0.03 + i_ * (w - 0.06) / max(1, per - 1)
        pz = z + 0.03 + r_ * 0.024
        if port_kind == "coax":
            mb.cyl((px, y - 0.008, pz), 0.0045 if rows == 1 else 0.004, 0.012, axis="y", m=4, seg=10)
        else:
            mb.box((px, y - 0.006, pz), (0.012, 0.008, 0.012), m=5)
        port_pos.append(Vector((px, y - 0.008, pz)))
    E.obj(name, mb, ["ac_white", "plastic_dark", E.principled("ASM_panel_screen", (0.02, 0.03, 0.035), 0.0, 0.2,
                                                                (0.05, 0.07, 0.08), 0.6),
                     "metal_dark", "galv", "lens_black"], c, parent=root)
    E.text(name + "_lbl", label, 0.017, c, (x - w / 2 + 0.012, y - 0.0045, z + h - 0.02), (PI / 2, 0, 0),
           mat=E.principled("ASM_inst_label", (0.82, 0.84, 0.86), 0.0, 0.5),
           fnt="Arial Bold", align="LEFT", parent=root)
    E.text(name + "_sub", sub, 0.011, c, (x - w / 2 + 0.012, y - 0.0045, z + h - 0.038), (PI / 2, 0, 0),
           mat=E.principled("ASM_inst_sub", (0.5, 0.52, 0.55), 0.0, 0.5),
           fnt="Arial Bold", align="LEFT", parent=root)
    return port_pos


def station_S5(hero_objs, c, root, cl):
    """Final test (v4.1): module compliance board (MCB) socket under the feet -> coax launches -> BERT (electrical side);
    TX eye -> free-space reference-receiver head -> sampling scope (optical PAM4 eye, TDECQ / OMA / ER);
    RX eye <- free-space reference-transmitter head through a VOA (sensitivity sweep, BER at the host electrical output);
    full link loopback panel (per-lane pre-FEC BER, live eye) -> PASS. Refs: research/assembly_refs.md section 6."""
    x = XS["S5"]
    yaw = math.radians(30.0)
    R = Matrix.Rotation(yaw, 3, "Z")
    # v4.2 link-test steps (build frames): seat -> power on -> CMIS -> heads dock -> TX lanes 1-8 -> RX sweep -> full link -> PASS
    T_SEAT0, T_SEAT1, T_SEATED = 168, 178, 180
    T_PWR, T_EYES = 182, 184
    T_CMIS_ID, T_CMIS_READY, T_CMIS_DDM = 190, 194, 198
    T_IN0, T_IN1 = 198, 206
    T_ON, LANE_DT = 210, 5                                   # TX eye: lane k shown from T_ON + LANE_DT * (k - 1)
    T_RX0, T_RX1 = 250, 262
    T_LINK = 260
    T_PASS, T_OFF, T_OUT0, T_OUT1 = 270, 275, 275, 281
    H1A, H1B, H2A, H2B = 280, 287, 290, 299
    MCB_T = 0.0016                                          # board thickness (true m, x2.6 in the hall)
    PIN = 0.001                                             # pogo pin height under the pads (true m)
    board_top_w = BENCH_Z + 0.017 + MCB_T * S
    seat = Vector((x, WORK_Y, board_top_w + (PIN - 0.0003) * S))   # bird root: foot pads (z 0.3 mm) rest on the pin tips
    SF = E.empty("S5_socket_frame", c, tuple(seat), (0, 0, yaw), 0.02, parent=root)
    SF.scale = (S, S, S)

    def w_of(p):                                            # socket-frame (true m) -> asm_root local (hall m)
        return seat + R @ (Vector(p) * S)
    # ---- bird: waits on the bench, hops into the socket, seats
    br, _, mp = bird_stage(hero_objs, c, root, "S5_bird", tuple(seat), yaw)
    p_wait = Vector((x - 0.33, WORK_Y - 0.03, BENCH_Z + 0.001))
    key(br, "location", f(0), p_wait)
    key(br, "location", f(T_SEAT0), p_wait)
    n = 6
    for i in range(n + 1):
        u = i / n
        pos = p_wait.lerp(seat + Vector((0, 0, 0.003)), u) + Vector((0, 0, 4 * 0.09 * u * (1 - u)))
        key(br, "location", f(T_SEAT0 + (T_SEAT1 - T_SEAT0) * u), pos, "LINEAR")
    key(br, "location", f(T_SEATED), seat)
    key(br, "location", f(H1A), seat)
    key(br, "scale", f(T_SEAT0 - 2), Vector((S, S, S)))
    key(br, "scale", f(T_SEAT0), Vector((S * 1.03, S * 1.03, S * 0.9)))
    key(br, "scale", f(T_SEAT0 + 2), Vector((S * 0.97, S * 0.97, S * 1.06)))
    key(br, "scale", f(T_SEAT1), Vector((S * 1.02, S * 1.02, S * 0.95)))
    key(br, "scale", f(T_SEATED), Vector((S, S, S)))
    # ---- eyes light once the module is powered in the socket
    eye_ms = []
    for nme, o in mp.items():
        if nme.startswith(("eye_tx_iris_glow_ring", "eye_rx_iris_glow_ring")):
            eye_ms += own_material(o, "_S5")
    for m in eye_ms:
        b = bsdf(m)
        b.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1.0)
        for t_, v in ((0, 0.0), (T_EYES, 0.0), (T_EYES + 2, 8.0), (T_EYES + 4, 18.0), (T_EYES + 7, 12.0)):
            key_val(b.inputs["Emission Strength"], f(t_), v)
    disc = {}
    for side_, col in ((-1, (1.0, 0.08, 0.05)), (1, (0.1, 0.85, 1.0))):
        m_ = emis_mat("ASM_eye_disc_%s" % ("tx" if side_ < 0 else "rx"), col, 0.0, dark=True)
        db = MB()
        db.cyl((side_ * 0.01135, -0.049, 0.0815), 0.0024, 0.0003, axis="x", m=0, seg=24)
        E.obj("S5_eye_disc_%s" % ("tx" if side_ < 0 else "rx"), db, [m_], c, parent=br, shadow=False)
        for t_, v in ((0, 0.0), (T_EYES, 0.0), (T_EYES + 2, 10.0), (T_EYES + 4, 25.0), (T_EYES + 7, 16.0)):
            key_val(bsdf(m_).inputs["Emission Strength"], f(t_), v)
    # ---- MCB (v4.1 board upgrade): socket + pogo pins under all 60 contacts; HS: 16 differential pairs (8 TX + 8 RX) routed
    # as matched pairs to 32 vertical-launch coax connectors (flange + screws, hex body, threaded interface) in staggered
    # rows on both side edges; LS: USB-to-I2C bridge (U1) + USB-C, I2C header, DIP switch / jumpers for LPWn/PRSn and
    # INT/RSTn, status LEDs, test points, EEPROM, oscillator; power: banana jacks -> fuse / TVS -> current sense + monitor
    # -> DC-DC + inductor + capacitor bank -> wide 3.3 V pour to the VCC pads. Refs: research/assembly_refs.md section 7.
    parts_raw = {}
    for nme, o in mp.items():
        if nme in ("foot_tx_edge_pads_x26", "foot_rx_edge_pads_x26", "pcb_belly_lowspeed_vcc_pads_x8"):
            M_ = o.matrix_parent_inverse @ o.matrix_basis
            parts_raw[nme] = [(M_ @ cc, sz) for cc, sz in loose_parts(o.data)]
    pads = {k: [p_ for p_, _ in v] for k, v in parts_raw.items()}
    LOG.append("S5 contacts: " + ", ".join("%s %d" % (k, len(v)) for k, v in pads.items()))
    bx0, bx1, by0, by1 = -0.068, 0.068, -0.075, 0.05
    zb = -PIN                                                # board top in the socket frame
    Z = lambda dz: zb + dz
    hb = MB()      # board, landing pads, socket nest, mounting hardware
    pin = MB()     # pogo pins
    tr = MB()      # gold traces / pours
    cp = MB()      # components
    hb.box(((bx0 + bx1) / 2, (by0 + by1) / 2, zb - MCB_T / 2), (bx1 - bx0, by1 - by0, MCB_T), m=0)
    feet = pads.get("foot_tx_edge_pads_x26", []) + pads.get("foot_rx_edge_pads_x26", [])
    for pc in feet:
        hb.box((pc.x, pc.y, Z(0.00002)), (0.0006, 0.0011, 0.00004), m=1)
        pin.cyl((pc.x, pc.y, zb + PIN * 0.45), 0.00012, PIN * 0.9, m=0, seg=8)
        pin.cyl((pc.x, pc.y, zb + PIN * 0.95), 0.00007, PIN * 0.1, m=1, seg=6)
    for sx in (-1, 1):
        fx = [pc.x for pc in feet if pc.x * sx > 0]
        fy = [pc.y for pc in feet if pc.x * sx > 0]
        if not fx:
            continue
        lo_x, hi_x, lo_y, hi_y = min(fx) - 0.003, max(fx) + 0.003, min(fy) - 0.004, max(fy) + 0.003
        for (a, b_) in (((lo_x, lo_y), (hi_x, lo_y)), ((lo_x, hi_y), (hi_x, hi_y)), ((lo_x, lo_y), (lo_x, hi_y)),
                        ((hi_x, lo_y), (hi_x, hi_y))):
            hb.seg((a[0], a[1], Z(0.0006)), (b_[0], b_[1], Z(0.0006)), 0.0012, 0.0012, m=2)
        hb.cyl((lo_x, lo_y, Z(0.0015)), 0.0007, 0.003, m=3, seg=10)
        hb.cyl((hi_x, hi_y, Z(0.0015)), 0.0007, 0.003, m=3, seg=10)
    for mx_, my_ in ((bx0 + 0.004, by0 + 0.004), (bx1 - 0.004, by0 + 0.004), (bx0 + 0.004, by1 - 0.004), (bx1 - 0.004, by1 - 0.004)):
        hb.ring((mx_, my_, Z(0.00005)), 0.0022, 0.0013, 0.0001, seg=20, m=3)           # mounting hole + screw
        hb.cyl((mx_, my_, Z(0.0004)), 0.0016, 0.0008, m=3, seg=6)
    # belly pedestal: identify VCC (merged, widest) vs low-speed pads
    bel = parts_raw.get("pcb_belly_lowspeed_vcc_pads_x8", [])
    vcc_pads, ls_pads = [], []
    if bel:
        wmax = max(sz.x for _, sz in bel)
        for p_, sz in bel:
            (vcc_pads if sz.x > 0.75 * wmax and len(vcc_pads) < 2 else ls_pads).append(p_)
        bxs = [p_.x for p_, _ in bel]
        bys = [p_.y for p_, _ in bel]
        zt = min(p_.z for p_, _ in bel)
        hb.box(((min(bxs) + max(bxs)) / 2, (min(bys) + max(bys)) / 2, (zb + zt - PIN) / 2),
               (max(bxs) - min(bxs) + 0.003, max(bys) - min(bys) + 0.003, zt - PIN - zb), m=2)
        for p_, _ in bel:
            pin.cyl((p_.x, p_.y, zt - PIN * 0.5), 0.00012, PIN, m=0, seg=8)
    LOG.append("S5 belly pads: VCC %d, low-speed %d" % (len(vcc_pads), len(ls_pads)))
    tz = Z(0.00003)

    def trace(pts, w=0.00028, m=0):
        for a, b_ in zip(pts[:-1], pts[1:]):
            tr.seg((a[0], a[1], tz), (b_[0], b_[1], tz), w, 0.00004, m=m)
    # high-speed: per side, signal pads (G S S G pattern) -> 8 matched pairs -> staggered vertical launches
    launches = []
    for sx, key_ in ((-1, "foot_tx_edge_pads_x26"), (1, "foot_rx_edge_pads_x26")):
        pl = sorted(pads.get(key_, []), key=lambda q: q.y)
        sig = [pc for i, pc in enumerate(pl) if i % 3 != 0][:16]
        for pc in [pc for i, pc in enumerate(pl) if i % 3 == 0]:
            hb.cyl((pc.x + sx * 0.0012, pc.y, Z(0.00004)), 0.00025, 0.00008, m=1, seg=8)   # GND via
        for k in range(8):
            if 2 * k + 1 >= len(sig):
                break
            yk = -0.040 + k * 0.0068
            for j, pc in enumerate((sig[2 * k], sig[2 * k + 1])):
                dy = (j - 0.5) * 0.0005
                cx = sx * (0.050 if j == 0 else 0.059)
                cy = yk + (-0.0017 if j == 0 else 0.0017)
                trace([(pc.x, pc.y), (sx * 0.018, pc.y + dy * 0), (sx * 0.024, yk + dy), (sx * 0.044, yk + dy),
                       (cx - sx * 0.003, cy), (cx, cy)])
                launches.append((sx, k, j, Vector((cx, cy, zb))))
    vb = MB()      # vertical launch connectors (1.85 mm style, compression mount)
    for sx, k, j, lp in launches:
        x_, y_ = lp.x, lp.y
        vb.box((x_, y_, Z(0.0003)), (0.0056, 0.0030, 0.0006), m=0)                    # flange
        for s2 in (-1, 1):
            vb.cyl((x_ + s2 * 0.0021, y_, Z(0.00075)), 0.00055, 0.0003, m=1, seg=6)  # screw heads
        vb.cyl((x_, y_, Z(0.0017)), 0.0019, 0.0022, m=0, seg=6)                       # hex body
        vb.cyl((x_, y_, Z(0.0043)), 0.0012, 0.003, m=0, seg=14)                       # threaded interface
        for tz_ in (0.0034, 0.0040, 0.0046, 0.0052):
            vb.cyl((x_, y_, Z(tz_)), 0.00128, 0.00015, m=0, seg=14)
        vb.cyl((x_, y_, Z(0.0068)), 0.0016, 0.002, m=1, seg=6)                        # cable nut
    # low-speed block (front-right corner): U1 USB-I2C bridge, J1 USB-C, U2 EEPROM, Y1 oscillator, J2 I2C header,
    # SW1 DIP, JP1/JP2, D1-D3 LEDs, TP1-TP4
    def chip(cx_, cy_, w, l, h, m=0, leads=0, lead_axis="x"):
        cp.box((cx_, cy_, Z(h / 2)), (w, l, h), m=m)
        for i in range(leads):
            u = (i + 0.5) / leads - 0.5
            for s2 in (-1, 1):
                if lead_axis == "x":
                    cp.box((cx_ + s2 * (w / 2 + 0.0003), cy_ + u * l * 0.9, Z(0.00012)), (0.0007, 0.0003, 0.00024), m=1)
                else:
                    cp.box((cx_ + u * w * 0.9, cy_ + s2 * (l / 2 + 0.0003), Z(0.00012)), (0.0003, 0.0007, 0.00024), m=1)
    chip(0.032, -0.060, 0.005, 0.005, 0.0009, leads=6)                               # U1 (QFN-like)
    cp.box((0.032, -0.0722, Z(0.0016)), (0.009, 0.0075, 0.0032), m=1)                 # J1 USB-C shell
    cp.box((0.032, -0.0759, Z(0.0016)), (0.0066, 0.0002, 0.0012), m=4)
    chip(0.0445, -0.060, 0.0039, 0.0049, 0.0015, leads=4)                             # U2 EEPROM SOIC-8
    cp.box((0.022, -0.060, Z(0.0006)), (0.0032, 0.0025, 0.0012), m=1)                 # Y1 oscillator can
    cp.box((0.056, -0.058, Z(0.0012)), (0.0052, 0.0078, 0.0025), m=4)                 # J2 I2C header 2x3
    for i in range(2):
        for jj in range(3):
            cp.box((0.0547 + i * 0.00254, -0.0605 + jj * 0.00254, Z(0.0045)), (0.0006, 0.0006, 0.004), m=2)
    cp.box((0.044, -0.0685, Z(0.0012)), (0.0100, 0.0060, 0.0025), m=3)                # SW1 DIP (4)
    for i in range(4):
        cp.box((0.0405 + i * 0.00235, -0.0685 + (0.001 if i % 2 else -0.001), Z(0.0028)), (0.0011, 0.0016, 0.0008), m=5)
    for i in range(2):                                                                 # JP1 / JP2
        cp.box((0.056 + i * 0.004, -0.0685, Z(0.0012)), (0.0026, 0.0052, 0.0025), m=4)
        cp.box((0.056 + i * 0.004, -0.0675, Z(0.0032)), (0.0024, 0.0026, 0.0018), m=4)
    led_m = emis_mat("ASM_board_led", (0.25, 1.0, 0.4), 0.0, dark=True)
    for i in range(3):                                                                 # D1 PRSn, D2 INT, D3 PG
        cp.box((0.019 + i * 0.0028, -0.0690, Z(0.0004)), (0.0016, 0.0008, 0.0008), m=9)
    for i in range(4):                                                                 # TP1-TP4
        cp.cyl((0.064, -0.054 + i * 0.0035, Z(0.0018)), 0.0003, 0.0036, m=2, seg=6)
        cp.ring((0.064, -0.054 + i * 0.0035, Z(0.0038)), 0.0009, 0.0005, 0.0004, seg=12, m=2)
    for k, p_ in enumerate(sorted(ls_pads, key=lambda q: q.x)):                       # SCL, SDA, LPWn/PRSn, INT/RSTn -> U1
        trace([(p_.x, p_.y), (p_.x, -0.017 - 0.0006 * k), (0.016 + 0.0006 * k, -0.044), (0.029 + 0.0008 * k, -0.0565)])
    trace([(0.032, -0.0625), (0.032, -0.068)], 0.0004)                                # U1 <-> USB
    trace([(0.0345, -0.060), (0.0425, -0.060)], 0.0003)
    trace([(0.0295, -0.060), (0.0236, -0.060)], 0.0003)
    trace([(0.0345, -0.0585), (0.0535, -0.0585)], 0.0003)
    trace([(0.034, -0.062), (0.040, -0.0665)], 0.0003)
    # power block (front-left corner): J3 red / J4 black banana jacks, F1 fuse, D4 TVS, R1 sense + U4 monitor, U3 DC-DC,
    # L1 inductor, C1/C2 bulk, MLCC bank, wide pour to the VCC pads
    jacks = []
    for i, col_m in enumerate((6, 4)):
        jx, jy = -0.060, -0.067 + i * 0.010
        cp.cyl((jx, jy, Z(0.0006)), 0.0029, 0.0012, m=1, seg=6)
        cp.cyl((jx, jy, Z(0.0037)), 0.0023, 0.005, m=col_m, seg=18)
        cp.cyl((jx, jy, Z(0.0063)), 0.0010, 0.0003, m=7, seg=12)
        jacks.append(Vector((jx, jy, Z(0.0064))))
    cp.box((-0.051, -0.069, Z(0.0004)), (0.0032, 0.0016, 0.0008), m=5)                 # F1
    cp.box((-0.051, -0.0615, Z(0.0011)), (0.0043, 0.0026, 0.0022), m=0)                # D4 TVS
    cp.box((-0.043, -0.069, Z(0.0003)), (0.0063, 0.0032, 0.0006), m=4)                 # R1 2512 sense
    for s2 in (-1, 1):
        cp.box((-0.043 + s2 * 0.0029, -0.069, Z(0.00035)), (0.0006, 0.0032, 0.0007), m=1)
    chip(-0.043, -0.061, 0.003, 0.003, 0.001, leads=3)                                 # U4 current monitor
    chip(-0.0335, -0.0645, 0.004, 0.004, 0.0009, leads=4)                              # U3 DC-DC
    cp.box((-0.0255, -0.0645, Z(0.0015)), (0.005, 0.005, 0.003), m=8)                  # L1 shielded inductor
    for cxx in (-0.034, -0.027):                                                        # C1 / C2 bulk caps
        cp.cyl((cxx, -0.0548, Z(0.0028)), 0.0025, 0.0055, m=7, seg=18)
        cp.cyl((cxx, -0.0548, Z(0.00565)), 0.0025, 0.0001, m=4, seg=18)
    for i in range(6):                                                                  # MLCC bank 0805
        cp.box((-0.0395 + i * 0.0028, -0.0715, Z(0.0005)), (0.0012, 0.002, 0.001), m=10)
        for s2 in (-1, 1):
            cp.box((-0.0395 + i * 0.0028, -0.0715 + s2 * 0.0009, Z(0.0005)), (0.00125, 0.0003, 0.00105), m=1)
    trace([(jacks[0].x, jacks[0].y), (-0.051, -0.069), (-0.043, -0.069), (-0.0335, -0.067)], 0.0016, m=1)
    trace([(jacks[1].x, jacks[1].y), (-0.051, -0.0615)], 0.0016, m=1)
    for k, p_ in enumerate(vcc_pads):                                                   # 3.3 V pour to the VCC pads
        trace([(-0.0255, -0.0645), (-0.020, -0.052), (p_.x - 0.004, -0.030 + 0.002 * k), (p_.x, -0.016), (p_.x, p_.y)], 0.0022, m=1)
    pcb = E.principled("ASM_mcb_pcb", (0.04, 0.09, 0.07), 0.0, 0.45)
    E.obj("S5_mcb_board", hb, [pcb, "chrome", E.principled("ASM_socket_body", (0.03, 0.03, 0.035), 0.0, 0.5), "galv"], c, parent=SF)
    E.obj("S5_mcb_pogo", pin, ["galv", E.principled("ASM_pin_gold", (1.0, 0.75, 0.35), 1.0, 0.25)], c, parent=SF)
    E.obj("S5_mcb_traces", tr, [E.principled("ASM_trace_gold", (0.9, 0.62, 0.28), 1.0, 0.35),
                                E.principled("ASM_pour_copper", (0.75, 0.45, 0.25), 1.0, 0.4)], c, parent=SF, shadow=False)
    E.obj("S5_mcb_launches", vb, [E.principled("ASM_launch_steel", (0.78, 0.76, 0.7), 1.0, 0.3), "galv"], c, parent=SF)
    E.obj("S5_mcb_parts", cp, [E.principled("ASM_ic_black", (0.02, 0.02, 0.022), 0.0, 0.45), "galv",
                               E.principled("ASM_pin_gold", (1.0, 0.75, 0.35), 1.0, 0.25),
                               E.principled("ASM_dip_red", (0.55, 0.05, 0.04), 0.0, 0.4),
                               E.principled("ASM_plastic_blk", (0.015, 0.015, 0.017), 0.0, 0.5),
                               E.principled("ASM_plastic_wht", (0.85, 0.85, 0.82), 0.0, 0.5),
                               E.principled("ASM_banana_red", (0.6, 0.03, 0.03), 0.0, 0.35),
                               E.principled("ASM_alu_can", (0.7, 0.72, 0.75), 0.8, 0.35),
                               E.principled("ASM_inductor", (0.08, 0.08, 0.085), 0.3, 0.5), led_m,
                               E.principled("ASM_mlcc_tan", (0.55, 0.45, 0.3), 0.0, 0.5)], c, parent=SF)
    key_sock(bsdf(led_m).inputs["Emission Strength"], [(0, 0.0, "LINEAR"), (T_PWR - 1, 0.0, "CONSTANT"), (T_PWR, 6.0, "LINEAR")])
    silk = E.principled("ASM_silk", (0.85, 0.87, 0.85), 0.0, 0.6)
    for txt, (lx, ly), sz in (("MCB  800G  MODULE COMPLIANCE BOARD", (0.0, by0 + 0.0035), 0.0028),
                              ("U1 USB-I2C", (0.032, -0.0538), 0.0013), ("U2", (0.0445, -0.0565), 0.0013),
                              ("Y1", (0.022, -0.0575), 0.0013), ("J2 I2C", (0.056, -0.0532), 0.0013),
                              ("SW1 LPWn/PRSn INT/RSTn", (0.046, -0.0725), 0.0011), ("D1 D2 D3", (0.0218, -0.0712), 0.0011),
                              ("TP1-4", (0.0625, -0.0425), 0.0011), ("J3 +3V3", (-0.060, -0.0715), 0.0012),
                              ("J4 GND", (-0.060, -0.0518), 0.0012), ("U3 DC-DC", (-0.0335, -0.0598), 0.0011),
                              ("L1", (-0.0255, -0.0610), 0.0011), ("R1 SENSE  U4 I-MON", (-0.045, -0.0735), 0.0011),
                              ("TX LANES 1-8", (-0.0545, 0.0135), 0.0014), ("RX LANES 1-8", (0.0545, 0.0135), 0.0014)):
        E.text("S5_mcb_silk_%s" % txt.split()[0], txt, sz, c, (lx, ly, Z(0.00006)), (0, 0, 0), mat=silk, fnt="Arial Bold", parent=SF)
    # ---- instruments (parody names) on the back of the bench; bench supply + CMIS host on top
    yf = 0.20
    instrument("S5_dca", (x - 0.36, yf, BENCH_Z + 0.001), (0.36, 0.16, 0.19), c, root, "KEYSIGHTLESS",
               "SAMPLING SCOPE  OPTICAL IN", ports=2, port_kind="optical")
    instrument("S5_voa", (x - 0.36, yf, BENCH_Z + 0.191), (0.36, 0.16, 0.06), c, root, "VIAVOID", "VOA  +  OPTICAL SWITCH",
               ports=4, port_kind="optical", screen=False)
    bert_ports = instrument("S5_bert", (x + 0.36, yf, BENCH_Z + 0.001), (0.36, 0.16, 0.19), c, root, "ANRITSOON",
                            "BERT  PPG / ED  32 x COAX", ports=32, port_kind="coax", rows=2)
    def vis(o_, t_a, t_b=None):                              # object exists only in [t_a, t_b) (scale keys, CONSTANT)
        sc = o_.scale.copy()
        key(o_, "scale", f(0), Vector((0, 0, 0)) if t_a > 0 else sc, "CONSTANT")
        key(o_, "scale", f(max(t_a, 0)), sc, "CONSTANT")
        if t_b is not None:
            key(o_, "scale", f(t_b), Vector((0, 0, 0)), "CONSTANT")
    psu_c = Vector((x + 0.25, yf, BENCH_Z + 0.191))
    instrument("S5_psu", tuple(psu_c), (0.14, 0.13, 0.1), c, root, "RIGOLD", "DC SUPPLY  CH1", ports=0, screen=False, knobs=False)
    ro_m = emis_mat("ASM_psu_readout", (0.9, 0.92, 0.95), 2.0)
    E.text("S5_psu_volt", "3.30 V", 0.016, c, (psu_c.x - 0.03, yf - 0.0046, psu_c.z + 0.05), (PI / 2, 0, 0), mat=ro_m,
           fnt="Andale Mono", parent=root)
    cur_steps = (("0.00 A", 0, T_PWR + 1), ("0.9 A", T_PWR + 1, T_PWR + 3), ("2.1 A", T_PWR + 3, T_PWR + 5),
                 ("3.2 A", T_PWR + 5, T_PWR + 7), ("3.8 A", T_PWR + 7, None))
    for k, (txt, ta, tb) in enumerate(cur_steps):            # supply current rises at power-on (3.8 A x 3.3 V = 12.5 W < 16 W max)
        o_ = E.text("S5_psu_amp_%d" % k, txt, 0.016, c, (psu_c.x - 0.03, yf - 0.0046, psu_c.z + 0.027), (PI / 2, 0, 0), mat=ro_m,
                    fnt="Andale Mono", parent=root)
        vis(o_, ta, tb)
    pj = MB()
    psu_jacks = []
    for i, col_m in enumerate((0, 1)):
        jp = Vector((psu_c.x + 0.035 + i * 0.022, yf - 0.006, psu_c.z + 0.016))
        pj.cyl(tuple(jp), 0.007, 0.012, axis="y", m=col_m, seg=16)
        psu_jacks.append(jp - Vector((0, 0.006, 0)))
    E.obj("S5_psu_jacks", pj, [E.principled("ASM_banana_red", (0.6, 0.03, 0.03), 0.0, 0.35),
                               E.principled("ASM_plastic_blk", (0.015, 0.015, 0.017), 0.0, 0.5)], c, parent=root)
    host_c = Vector((x + 0.445, yf, BENCH_Z + 0.191))
    instrument("S5_cmis_host", tuple(host_c), (0.19, 0.13, 0.15), c, root, "CMIS HOST", "USB  I2C (TWI)", ports=0, screen=False, knobs=False)
    hq = MB()
    hq.quad((-0.085, 0, -0.05), (0.085, 0, -0.05), (0.085, 0, 0.05), (-0.085, 0, 0.05), m=0)
    E.obj("S5_cmis_screen", hq, [E.principled("ASM_cmis_bg", (0.01, 0.012, 0.014), 0.0, 0.3, (0.012, 0.014, 0.016), 1.0)], c,
          parent=root, loc=(host_c.x, yf - 0.0046, host_c.z + 0.058), shadow=False)
    cm = emis_mat("ASM_cmis_text", (0.85, 0.88, 0.9), 2.2)
    cm_ok = emis_mat("ASM_cmis_ok", (0.35, 0.85, 0.5), 2.5)
    # module ID + CMIS state + DDM (values inside the DR8 datasheet ranges: case 0-70 C, VCC 3.135-3.465 V, TX bias DDM 0-100 mA,
    # average launch power -2.9..4.0 dBm per lane; CMIS ModuleReady state per OIF-CMIS 5.4 6.3.2.5.6)
    lines = (("CMIS  TWI A0h  LINK UP", T_PWR + 2, cm), ("ID    OSFP 800G DR8", T_CMIS_ID, cm), ("VEND  OUNOLIGHT", T_CMIS_ID, cm),
             ("STATE ModuleReady", T_CMIS_READY, cm_ok), ("TEMP  42.5 C", T_CMIS_DDM, cm), ("VCC   3.29 V", T_CMIS_DDM, cm),
             ("BIAS  58 mA  (L1)", T_CMIS_DDM + 1, cm), ("TXPWR 1.2 dBm (L1)", T_CMIS_DDM + 1, cm))
    for k, (txt, ta, mat_) in enumerate(lines):
        o_ = E.text("S5_cmis_line_%d" % k, txt, 0.0088, c, (host_c.x - 0.079, yf - 0.0050, host_c.z + 0.098 - k * 0.0118),
                    (PI / 2, 0, 0), mat=mat_, fnt="Andale Mono", align="LEFT", parent=root)
        vis(o_, ta)
    # ---- cables: 32 coax launches -> BERT; banana leads -> supply; USB -> CMIS host
    cab = MB()
    order = sorted(launches, key=lambda r: (r[0], r[1], r[2]))
    for idx, (sx, k, j, lp) in enumerate(order):
        top = w_of(Vector((lp.x, lp.y, Z(0.0078))))
        p_end = bert_ports[idx] - Vector((0, 0.006, 0))
        rise = top + Vector((0, 0, 0.03 + 0.004 * k))
        if sx < 0:
            ctrl = [top, rise, Vector((x - 0.16, 0.08 + 0.003 * idx, BENCH_Z + 0.09)), Vector((x + 0.05, 0.13, BENCH_Z + 0.07)),
                    Vector((p_end.x, p_end.y - 0.05, p_end.z + 0.02)), p_end]
        else:
            ctrl = [top, rise, Vector((p_end.x - 0.02, p_end.y - 0.07, p_end.z + 0.04)), p_end]
        cab.tube(ctrl, 0.0022, seg=6, m=0)
    E.obj("S5_coax_cables", cab, [E.principled("ASM_coax_blue", (0.08, 0.1, 0.14), 0.3, 0.45)], c, parent=root)
    lead = MB()
    for i, jk in enumerate(jacks):
        a = w_of(jk)
        b_ = psu_jacks[i]
        lead.tube([a, a + Vector((0, 0, 0.012)), a + Vector((-0.03, 0.14, 0.02)), Vector((x - 0.12, 0.17 + 0.01 * i, BENCH_Z + 0.06)),
                   Vector((x + 0.1, 0.185, BENCH_Z + 0.17)), b_ - Vector((0, 0.04, 0)), b_], 0.0028, seg=8, m=i)
    usb_a = w_of(Vector((0.032, -0.0765, Z(0.0016))))
    usb_dir = (R @ Vector((0, -1, 0))).normalized()
    hp_ = host_c + Vector((0.095, 0.075, 0.03))               # USB into the host's right side (clear of its screen)
    lead.tube([usb_a, usb_a + usb_dir * 0.03, usb_a + usb_dir * 0.05 + Vector((0, 0, 0.03)), Vector((hp_.x + 0.06, hp_.y - 0.1, BENCH_Z + 0.05)),
               Vector((hp_.x + 0.05, hp_.y, hp_.z)), hp_ + Vector((0.006, 0, 0))], 0.0025, seg=8, m=2)
    E.obj("S5_power_usb_leads", lead, [E.principled("ASM_lead_red", (0.6, 0.04, 0.03), 0.0, 0.45),
                                       E.principled("ASM_lead_blk", (0.02, 0.02, 0.022), 0.0, 0.45),
                                       E.principled("ASM_usb_grey", (0.35, 0.36, 0.38), 0.0, 0.5)], c, parent=root)
    # ---- free-space test heads on linear stages (towers on the bench)
    beams = []
    for side, label in ((-1, "REF RX  >  DCA"), (1, "REF TX + VOA")):
        eye = Vector((side * 0.0113, -0.049, 0.0815))
        eye_w = seat + R @ (eye * S)
        out = (R @ Vector((side, 0.0, 0.0))).normalized()
        to_eye = -out
        ly = Vector((0, 0, 1)).cross(to_eye)               # head local +Y in world
        side_sign = 1 if ly.dot(Vector((0, -1, 0))) > 0 else -1
        nm = "S5_test_head_%s" % ("tx" if side < 0 else "rx")
        h, led = test_head(nm, c, root, label, side_sign)
        h.rotation_euler = to_eye.to_track_quat("X", "Z").to_euler()
        dock = eye_w + out * 0.022
        away = eye_w + out * 0.14
        for t_, p_ in ((0, away), (T_IN0, away), (T_IN1, dock), (T_OUT0, dock), (T_OUT1, away)):
            key(h, "location", f(t_), p_)
        key_sock(bsdf(led).inputs["Emission Strength"], [(0, 0.3, "LINEAR"), (T_ON - 1, 0.3, "CONSTANT"), (T_ON, 6.0, "LINEAR"),
                                                        (T_OFF, 6.0, "CONSTANT"), (T_OFF + 1, 0.3, "LINEAR")])
        # stage: rail along `out` above the head path + column down to the bench
        st = MB()
        r0 = dock + out * 0.0 + Vector((0, 0, 0.044))
        r1 = eye_w + out * 0.24 + Vector((0, 0, 0.044))
        st.seg(r0 - out * 0.01, r1, 0.03, 0.012, m=0)
        st.box((r1.x, r1.y, (BENCH_Z + r1.z) / 2), (0.04, 0.04, r1.z - BENCH_Z), m=1)
        st.box((r1.x, r1.y, BENCH_Z + 0.006), (0.09, 0.09, 0.012), m=1)
        E.obj(nm + "_stage", st, ["galv", "metal_paint"], c, parent=root)
        bm = emis_mat("ASM_test_beam_%s" % ("tx" if side < 0 else "rx"), (1.0, 0.1, 0.06) if side < 0 else (0.1, 0.85, 1.0), 0.0,
                      dark=True)
        rb = MB()
        rb.rod(eye_w + out * 0.001, dock - out * 0.001, 0.0012, m=0, seg=10)
        E.obj(nm + "_beam_viz", rb, [bm], c, parent=root, shadow=False)
        key_sock(bsdf(bm).inputs["Emission Strength"], [(0, 0.0, "LINEAR"), (T_ON - 1, 0.0, "CONSTANT"), (T_ON, 12.0, "LINEAR"),
                                                       (T_OFF, 12.0, "CONSTANT"), (T_OFF + 1, 0.0, "LINEAR")])
    # ---- three screens on a bar above the instruments: TX (DCA) | FULL LINK | RX (BERT)
    tex = os.path.join(OUT_DIR, "textures")
    img = lambda n_: bpy.data.images.load(os.path.join(tex, n_))
    atlas = bpy.data.images.load(EYE_ATLAS)
    SW, SH, SZ, SY, DX = 0.36, 0.24, 1.33, 0.33, 0.37
    XC = x + 0.05                                            # screen group centre (clear of the S4 press column)
    bar = MB()
    for px in (XC - 1.5 * DX, XC + 1.5 * DX):
        bar.box((px, SY + 0.03, (BENCH_Z + SZ + SH / 2) / 2), (0.03, 0.03, SZ + SH / 2 - BENCH_Z), m=0)
    bar.box((XC, SY + 0.03, SZ + SH / 2 + 0.03), (3 * DX + 0.06, 0.03, 0.03), m=0)
    for k in range(3):
        bar.box((XC + (k - 1) * DX, SY + 0.012, SZ), (SW + 0.02, 0.02, SH + 0.02), m=1)
    E.obj("S5_screen_bar", bar, ["metal_paint", "plastic_dark"], c, parent=root)
    screens = (("tx", img("scr_tx.png"), None, "u"), ("link", img("scr_link_base.png"), img("scr_link_rows.png"), "v"),
               ("rx", img("scr_rx_base.png"), img("scr_rx_curve.png"), "u"))
    socks = {}
    for k, (nm, bi, ri, ax) in enumerate(screens):
        m, on, rv = screen_material("ASM_screen_" + nm, bi, ri, ax)
        q = MB()
        q.quad((-SW / 2, 0, -SH / 2), (SW / 2, 0, -SH / 2), (SW / 2, 0, SH / 2), (-SW / 2, 0, SH / 2), m=0)
        so = E.obj("S5_screen_" + nm, q, [m], c, parent=root, loc=(XC + (k - 1) * DX, SY, SZ), shadow=False)
        quad_uv(so)
        socks[nm] = (on, rv)
    t_on = {"tx": T_ON, "link": T_LINK, "rx": T_RX0}
    for nm, (on, rv) in socks.items():
        key_sock(on, [(0, 0.03, "LINEAR"), (t_on[nm] - 1, 0.03, "CONSTANT"), (t_on[nm], 1.0, "LINEAR"), (T_OFF, 1.0, "CONSTANT"),
                      (T_OFF + 1, 0.03, "LINEAR")])
    # RX: VOA sweep draws the waterfall left -> right; FULL LINK: lanes fill top -> bottom (8 rows, 1.5 frames each)
    key_sock(socks["rx"][1], [(0, 0.0, "LINEAR"), (T_RX0, 0.08, "LINEAR"), (T_RX1, 1.0, "LINEAR")])
    v0, dv = 120 / 600, 44 / 600                            # row layout of make_test_screens.py (ROW_Y0, ROW_DY over H)
    key_sock(socks["link"][1], [(0, 0.0, "LINEAR"), (T_LINK, v0 - 0.002, "LINEAR"), (T_LINK + 10, v0 + 8 * dv, "LINEAR")])
    # live eyes: TX optical eye (DCA) and host RX eye on the FULL LINK panel (atlas, varying)
    for nm, (cx, cz, ew, eh, off) in (("tx", (XC - DX, SZ + 0.0104, 0.33, 0.15, 0.0)),       # boxes of make_test_screens.py
                                      ("link", (XC + 0.09, SZ + 0.0244, 0.15, 0.12, 11.0))):
        m, ph, on = eye_atlas_material("ASM_eye_screen_" + nm, atlas, off)
        q = MB()
        q.quad((-ew / 2, 0, -eh / 2), (ew / 2, 0, -eh / 2), (ew / 2, 0, eh / 2), (-ew / 2, 0, eh / 2), m=0)
        so = E.obj("S5_eye_" + nm, q, [m], c, parent=root, loc=(cx, SY - 0.0015, cz), shadow=False)
        quad_uv(so)
        ph.default_value = 0.0
        ph.keyframe_insert("default_value", frame=f(0))
        ph.default_value = float(DUR + 20)
        ph.keyframe_insert("default_value", frame=f(DUR + 20))
        for fc in m.node_tree.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        key_sock(on, [(0, 0.0, "LINEAR"), (t_on[nm] - 1, 0.0, "CONSTANT"), (t_on[nm], 1.0, "LINEAR"), (T_OFF, 1.0, "CONSTANT"),
                      (T_OFF + 1, 0.0, "LINEAR")])
    # TX screen: lane label + per-lane readout step 1..8 (values inside the DR8 datasheet limits: TDECQ <= 3.4 dB,
    # OMA outer -0.8..4.2 dBm, ER >= 3.5 dB), PASS while the TX test runs
    LANES = ((1.6, 1.9, 4.6), (1.5, 2.1, 4.4), (1.7, 1.8, 4.7), (1.4, 2.2, 4.5), (1.6, 2.0, 4.6), (1.8, 1.7, 4.3), (1.5, 2.0, 4.8),
             (1.6, 1.9, 4.5))
    sl = XC - DX - SW / 2
    lane_m = emis_mat("ASM_tx_lane_text", (0.85, 0.88, 0.9), 2.2)
    for k, (td, oma, er) in enumerate(LANES):
        ta = T_ON + LANE_DT * k
        tb = ta + LANE_DT if k < 7 else T_OFF + 1
        o1 = E.text("S5_tx_lane_lbl_%d" % (k + 1), "PAM4  LANE %d" % (k + 1), 0.0115, c, (sl + 0.016, SY - 0.004, SZ + 0.078),
                    (PI / 2, 0, 0), mat=lane_m, fnt="Arial Bold", align="LEFT", parent=root)
        o2 = E.text("S5_tx_lane_val_%d" % (k + 1), "TDECQ %.1f dB   OMA %.1f dBm   ER %.1f dB" % (td, oma, er), 0.0112, c,
                    (sl + 0.012, SY - 0.004, SZ - 0.088), (PI / 2, 0, 0), mat=lane_m, fnt="Arial Bold", align="LEFT", parent=root)
        vis(o1, ta, tb)
        vis(o2, ta, tb)
    txp_m = emis_mat("ASM_tx_pass", (0.35, 0.85, 0.5), 3.0)
    o3 = E.text("S5_tx_pass", "PASS", 0.0135, c, (sl + 0.325, SY - 0.004, SZ - 0.088), (PI / 2, 0, 0), mat=txp_m, fnt="Arial Bold",
                parent=root)
    vis(o3, T_ON + 2, T_OFF + 1)
    # RX and FULL LINK verdicts + big PASS (text objects, keyed)
    pass_m = emis_mat("ASM_pass_green", (0.35, 0.85, 0.5), 0.0, dark=True)
    po = E.text("S5_pass", "PASS", 0.05, c, (XC + 0.088, SY - 0.004, SZ - 0.075), (PI / 2, 0, 0), mat=pass_m, fnt="Arial Bold",
                parent=root)
    rxv = emis_mat("ASM_rx_verdict", (0.35, 0.85, 0.5), 0.0, dark=True)
    ro = E.text("S5_rx_verdict", "SENSITIVITY  PASS", 0.0145, c, (XC + DX - 0.04, SY - 0.004, SZ - 0.05), (PI / 2, 0, 0),
                mat=rxv, fnt="Arial Bold", parent=root)
    for o_, t_a in ((po, T_PASS), (ro, T_RX1 + 1)):          # text exists only while lit (no grey ghost when off)
        sc = o_.scale.copy()
        key(o_, "scale", f(0), Vector((0, 0, 0)), "CONSTANT")
        key(o_, "scale", f(t_a), sc, "CONSTANT")
        key(o_, "scale", f(T_OFF + 1), Vector((0, 0, 0)), "CONSTANT")
    key_sock(bsdf(pass_m).inputs["Emission Strength"], [(0, 0.0, "LINEAR"), (T_PASS - 1, 0.0, "CONSTANT"), (T_PASS, 5.0, "LINEAR"),
                                                       (T_OFF, 5.0, "CONSTANT"), (T_OFF + 1, 0.0, "LINEAR")])
    key_sock(bsdf(rxv).inputs["Emission Strength"], [(0, 0.0, "LINEAR"), (T_RX1, 0.0, "CONSTANT"), (T_RX1 + 1, 3.0, "LINEAR"),
                                                    (T_OFF, 3.0, "CONSTANT"), (T_OFF + 1, 0.0, "LINEAR")])
    # ---- hop off after the heads retract: to the bench end, then onto DISPATCH
    p1 = Vector((x + 0.36, WORK_Y - 0.08, BENCH_Z + 0.001))
    p2 = Vector((2.62, WORK_Y - 0.03, 0.5))
    key(br, "scale", f(H1A - 2), Vector((S, S, S)))
    key(br, "scale", f(H1A), Vector((S * 1.03, S * 1.03, S * 0.9)))
    key(br, "scale", f(H1A + 2), Vector((S * 0.97, S * 0.97, S * 1.06)))
    key(br, "scale", f(H1B - 1), Vector((S, S, S)))
    key(br, "scale", f(H1B), Vector((S * 1.03, S * 1.03, S * 0.92)))
    key(br, "scale", f(H2A), Vector((S, S, S)))
    key(br, "rotation_euler", f(H1A - 2), Vector((0, 0, yaw)))
    key(br, "rotation_euler", f(H1A + 2), Vector((0, 0, math.radians(75))))
    key(br, "rotation_euler", f(H2A), Vector((0, 0, math.radians(90))))
    for (a, b_, t0_, t1_, hgt) in ((seat, p1, H1A, H1B, 0.12), (p1, p2, H2A, H2B, 0.1)):
        for i in range(n + 1):
            u = i / n
            pos = a.lerp(b_, u) + Vector((0, 0, 4 * hgt * u * (1 - u)))
            key(br, "location", f(t0_ + (t1_ - t0_) * u), pos, "LINEAR")
    EVENTS.update({"S5_seat_hop": T_SEAT0, "S5_feet_seated": T_SEATED, "S5_power_on": T_PWR, "S5_eyes_on": T_EYES,
                   "S5_eyes_full": T_EYES + 4, "S5_cmis_id": T_CMIS_ID, "S5_cmis_ready": T_CMIS_READY, "S5_cmis_ddm": T_CMIS_DDM,
                   "S5_heads_in": T_IN0, "S5_heads_docked": T_IN1, "S5_eye_diagram_on": T_ON, "S5_tx_lanes_done": T_ON + 8 * LANE_DT,
                   "S5_rx_sweep_on": T_RX0, "S5_rx_sweep_done": T_RX1, "S5_full_link_on": T_LINK, "S5_full_link_lanes_done": T_LINK + 10,
                   "S5_pass": T_PASS, "S5_eye_diagram_off": T_OFF, "S5_heads_out": T_OUT0, "S5_heads_away": T_OUT1,
                   "S5_hop1": H1A, "S5_land1": H1B, "S5_hop2": H2A, "S5_land2": H2B})
    for k in range(8):
        EVENTS["S5_tx_lane_%d" % (k + 1)] = T_ON + LANE_DT * k
    for k, t_ in enumerate((T_PWR + 1, T_PWR + 3, T_PWR + 5, T_PWR + 7)):
        EVENTS["S5_psu_current_step_%d" % (k + 1)] = t_
    E.hook("S5_work", c, tuple(seat + Vector((0, 0, 0.12))), 0.0, "station", parent=root)
    E.hook("S5_socket", c, tuple(seat), yaw, "station", parent=root)
    E.hook("S5_screens", c, (XC, SY, SZ), 0.0, "station", parent=root)
    return br


# ==================================================================== camera suggestion
def camera_plan(s2_iso_l, s3_caps):
    """Suggested shots (local to asm_root; frame offsets from the sequence start). Shot A: S2 lens + isolator + UV (macro);
    shot B: S3 wire bonding (macro, follows the bonding row); shot C: S4 press closing -> truck to S5 eyes on + hop."""
    wA = Vector((XS["S2"], WORK_Y, CHUCK_Z)) + s2_iso_l
    shots = []
    # v4.2 (2x): A 0-71, B 72-121, C 122-167, D 168-287 (build frames; film frames = build / 1.5652 at S's speed)
    shots.append({"name": "A_align_macro", "cut_in": True, "keys": [
        (0, tuple(wA + Vector((0.17, -0.05, 0.08))), tuple(wA + Vector((0, -0.0006, 0.0008))), 560.0),
        (40, tuple(wA + Vector((0.16, -0.03, 0.072))), tuple(wA + Vector((0, -0.0010, 0.0007))), 600.0),
        (71, tuple(wA + Vector((0.155, -0.02, 0.068))), tuple(wA + Vector((0, -0.0014, 0.0006))), 600.0)]})
    inwin = [p for p in s3_caps if 72 <= p[0] <= 121]
    a0 = inwin[0][1] if inwin else s3_caps[0][1]
    a1 = inwin[-1][1] if inwin else s3_caps[-1][1]
    w0 = Vector((XS["S3"], WORK_Y, CHUCK_Z)) + a0
    w1 = Vector((XS["S3"], WORK_Y, CHUCK_Z)) + a1
    off = Vector((0.03, -0.13, 0.085))
    shots.append({"name": "B_bond_macro", "cut_in": True, "keys": [
        (72, tuple(w0 + off), tuple(w0 + Vector((0, -0.0015, 0.0004))), 330.0),
        (121, tuple(w1 + off), tuple(w1 + Vector((0, -0.0015, 0.0004))), 330.0)]})
    b4 = Vector((XS["S4"], WORK_Y, BENCH_Z + 0.2))
    shots.append({"name": "C_press_to_test", "cut_in": True, "keys": [          # dispense close, then the press closes the lid
        (122, tuple(b4 + Vector((0.12, -0.45, 0.33))), tuple(b4 + Vector((0.0, -0.07, 0.06))), 60.0),
        (148, tuple(b4 + Vector((0.10, -0.42, 0.31))), tuple(b4 + Vector((0.0, -0.07, 0.06))), 60.0),
        (156, tuple(b4 + Vector((0.15, -1.25, 0.30))), tuple(b4 + Vector((0.03, 0, 0.02))), 55.0),
        (167, tuple(b4 + Vector((0.22, -1.2, 0.27))), tuple(b4 + Vector((0.08, 0, -0.01))), 55.0)]})
    x5 = XS["S5"]
    seat = Vector((x5, WORK_Y, BENCH_Z + 0.017 + 0.0016 * S + 0.0007 * S))
    t1 = seat + Vector((-0.04, 0, 0.035))
    t2 = seat + Vector((-0.01, 0, 0.04))
    ins = Vector((x5 + 0.345, 0.20, BENCH_Z + 0.255))          # supply + CMIS host on the BERT
    txs = Vector((x5 + 0.05 - 0.37, 0.33, 1.33))              # TX screen centre
    shots.append({"name": "D_test_stand", "cut_in": True, "keys": [
        (168, tuple(t1 + Vector((0.05, -0.40, 0.13))), tuple(t1), 40.0),                       # hop into the MCB socket, seat
        (184, tuple(t2 + Vector((0.04, -0.36, 0.12))), tuple(t2), 40.0),                       # power on: board LEDs, eyes
        (189, tuple(ins + Vector((0.12, -0.44, 0.06))), tuple(ins), 34.0),                     # supply current + CMIS / DDM
        (202, tuple(ins + Vector((0.11, -0.41, 0.055))), tuple(ins), 34.0),
        (208, (x5 + 0.03, -1.15, 1.3), (x5 + 0.03, 0.1, 1.02), 32.0),                          # heads docked
        (213, tuple(txs + Vector((0.0, -0.58, -0.06))), tuple(txs + Vector((0, 0, -0.01))), 40.0),   # TX lanes 1-8
        (248, tuple(txs + Vector((0.02, -0.55, -0.06))), tuple(txs + Vector((0, 0, -0.01))), 40.0),
        (254, (x5 + 0.05, -0.75, 1.42), (x5 + 0.05, 0.25, 1.12), 28.0),                        # RX sweep, full link, PASS
        (273, (x5 + 0.05, -0.71, 1.41), (x5 + 0.05, 0.25, 1.12), 28.0),
        (279, tuple(seat + Vector((0.12, -1.1, 0.3))), tuple(seat + Vector((0.08, 0, 0.1))), 42.0),   # retract, hop off
        (287, tuple(seat + Vector((0.22, -1.15, 0.27))), tuple(seat + Vector((0.25, 0, 0.07))), 42.0)]})
    shots.append({"name": "overview", "cut_in": True, "optional": True, "keys": [
        (0, (0.9, -3.6, 1.75), (-0.1, 0.2, 1.0), 26.0)]})
    return shots


# ==================================================================== main
def build(out_blend):
    scn = E.reset()
    E.HOOKS.clear()
    scn.frame_start, scn.frame_end = F0, F0 + DUR - 1
    scn.render.fps = 30
    top = E.coll("ENV_assembly")
    cset = E.coll("ASM_set", top)
    cst = E.coll("ASM_stations", top)
    cl = E.coll("ASM_lights", top)
    hc = E.coll("ASM_hooks", top)
    root = E.empty("assembly_root", top, (0, 0, 0), (0, 0, 0), 0.5)
    build_set(root, cset, cl)
    tpl = append_tx()
    LOG.append("TX template objects %d" % len(tpl))
    station_S1(tpl, cst, root, cl)
    W2, iso_l, l2_l = station_S2(tpl, cst, root, cl)
    W3, caps = station_S3(tpl, cst, root, cl)
    for o in list(tpl.values()):            # drop the templates (copies share their mesh data)
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(bpy.data.collections["ASM_tmp_tx"])
    bench(cst, root)
    hero, hero_cols = append_hero()
    keep = [o for o in hero if not o.name.startswith(("tx_beam", "rx_viz_beam")) and not (o.type == "MESH" and o.hide_render)]
    LOG.append("hero objects %d, kept %d" % (len(hero), len(keep)))
    station_S4(keep, cst, root, cl)
    station_S5(keep, cst, root, cl)
    for o in hero:
        bpy.data.objects.remove(o, do_unlink=True)
    for cc in hero_cols:
        bpy.data.collections.remove(cc)
    # hooks into the hooks collection
    for o in list(cst.objects):
        if o.name.startswith("HOOK_"):
            cst.objects.unlink(o)
            hc.objects.link(o)
    shots = camera_plan(iso_l, caps)
    bpy.ops.file.pack_all()
    bpy.context.view_layer.update()
    tris = E.count_tris(top.all_objects)
    E.save_blend(out_blend)
    side = {
        "asset": "assembly_hatchery", "date": "2026-10-10", "workstream": "M (DevLog-019)",
        "blend": E.rel(out_blend), "root": "assembly_root", "collection": "ENV_assembly",
        "frame_start": F0, "duration_frames": DUR, "fps": 30, "scale": S,
        "stations_x": XS, "chuck_z": CHUCK_Z, "bench_z": BENCH_Z, "events": EVENTS, "camera_plan": shots,
        "hooks": E.hooks_json() if hasattr(E, "hooks_json") else {}, "tris": tris, "log": LOG,
        "notes": ["TX parts appended from assets/components/optics_tx/optics_tx.blend; bird from assets/hero/pigeon_osfp_dr8.blend",
                  "leg band text OUTNOLIGHT hidden on both birds (H to rename)",
                  "beam on S2 is viz only (1311 nm invisible): the scene carries the beam note"],
    }
    E.write_json(os.path.splitext(out_blend)[0] + ".json", side)
    print("ASM built", E.rel(out_blend), "objects", len(top.all_objects), "tris", tris)
    for l in LOG:
        print("ASM", l)
    print("ASM events", json.dumps(EVENTS))


if __name__ == "__main__":
    a = E.argv_after_dashes()
    build(a[0] if a else os.path.join(OUT_DIR, "assembly_hatchery.blend"))
