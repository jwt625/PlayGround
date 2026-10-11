"""Simple form-factor models for the tree intro (bpy). Real size in metres (1 mm = 0.001), no logos.

Convention: length along +X (front / optical port at +X), width along Y, height along Z, origin at the centre of
the main body. Each builder returns one joined mesh object with material slots: 0 body (neutral satin metal),
1 dark (latch / plastic / cable / port windows).

Dimensions (W x H x L, mm) and their sources; APPROX = no primary dimension in the project research, shape and
size approximate (the card shows an "APPROX." tag):

| Key | W x H x L | Source | Grade |
|---|---|---|---|
| osfp | 22.58 x 13.00 x 100.40 | OSFP MSA Rev 5.2 Fig 3-2 (research/hero_800g_osfp_dr8.md); fins drawn open-top | A |
| qsfp28 / qsfp_plus / compatible | 18.35 x 8.5 x 72.4 (nose 19 x 13.5 x 20 max) | QSFP112 MSA Rev 2.1.2 Fig 12 + sec 6.2 (Type 1 = QSFP28) | A |
| sfp | 13.7 x 8.95 x 56.5 | Phoenix Contact SFP module datasheet (vendor) | B |
| cfp | 82 x 13.6 x 144.8 | Cisco 100GBASE CFP data sheet (c78-633027); height 13.6 also CFP MSA Rev 1.4 history | B |
| xenpak | 36 x 18 x 121 | Cisco XENPAK-10GB-ZR product listing (icecat) | C |
| gbic | 30 x 12 x 65 | product listings disagree (ZyXEL 65x30x10, D-Link ~57x30x12) | APPROX |
| qsfp_dd / zr400 | 18.35 x 8.5 x 89 | width/height = QSFP cage; length not in project research | APPROX |
| cfp2_dco | 41.5 x 12.4 x 107.5 | not verified (CFP2 MSA hardware spec not fetched) | APPROX |
| xpo | 45 x 16 x 120 + cold plate | "about two OSFP footprints" [2nd]; no primary dimensions | APPROX |
| one_by_nine | 25 x 10 x 39 + 9 pins | no spec found | APPROX |
| aec | QSFP-DD-like plug + cable | stylized | APPROX |
| cpo | 20 x 20 engine on substrate + fiber ribbon | stylized | APPROX |
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

MM = 0.001

DIMS = {  # key: (W, H, L) mm, grade
    "osfp": ((22.58, 13.0, 100.4), "A"),
    "qsfp28": ((18.35, 8.5, 72.4), "A"),
    "qsfp_plus": ((18.35, 8.5, 72.4), "A"),
    "compatible": ((18.35, 8.5, 72.4), "A"),
    "sfp": ((13.7, 8.95, 56.5), "B"),
    "cfp": ((82.0, 13.6, 144.8), "B"),
    "xenpak": ((36.0, 18.0, 121.0), "C"),
    "gbic": ((30.0, 12.0, 65.0), "APPROX"),
    "qsfp_dd": ((18.35, 8.5, 89.0), "APPROX"),
    "zr400": ((18.35, 8.5, 89.0), "APPROX"),
    "cfp2_dco": ((41.5, 12.4, 107.5), "APPROX"),
    "xpo": ((45.0, 16.0, 120.0), "APPROX"),
    "one_by_nine": ((25.0, 10.0, 39.0), "APPROX"),
    "aec": ((18.35, 8.5, 70.0), "APPROX"),
    "cpo": ((20.0, 3.0, 20.0), "APPROX"),
}


def approx(key: str) -> bool:
    return DIMS[key][1] == "APPROX"


# ------------------------------------------------------------------ materials
def materials():
    mats = {}
    if "INTRO_ff_body" not in bpy.data.materials:
        m = bpy.data.materials.new("INTRO_ff_body")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.44, 0.46, 0.49, 1)
        b.inputs["Metallic"].default_value = 0.3
        b.inputs["Roughness"].default_value = 0.6
        m.diffuse_color = (0.44, 0.46, 0.49, 1)
    if "INTRO_ff_dark" not in bpy.data.materials:
        m = bpy.data.materials.new("INTRO_ff_dark")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.035, 0.037, 0.04, 1)
        b.inputs["Roughness"].default_value = 0.55
        m.diffuse_color = (0.035, 0.037, 0.04, 1)
    mats["body"] = bpy.data.materials["INTRO_ff_body"]
    mats["dark"] = bpy.data.materials["INTRO_ff_dark"]
    return mats


# ------------------------------------------------------------------ mesh builder
class MB:
    """Accumulates boxes / cylinders (mm, centre-based) into one bmesh with per-face material index."""

    def __init__(self):
        self.bm = bmesh.new()

    def box(self, c, s, mat=0, bevel=0.0):
        r = bmesh.ops.create_cube(self.bm, size=1.0)
        vs = r["verts"]
        bmesh.ops.scale(self.bm, vec=Vector(s) * MM, verts=vs)
        bmesh.ops.translate(self.bm, vec=Vector(c) * MM, verts=vs)
        for f in {f for v in vs for f in v.link_faces}:
            f.material_index = mat
        return vs

    def cyl(self, c, r, depth, axis="X", mat=0, seg=16):
        res = bmesh.ops.create_cone(self.bm, cap_ends=True, segments=seg, radius1=r * MM, radius2=r * MM,
                                    depth=depth * MM)
        vs = res["verts"]
        rot = {"X": Matrix.Rotation(math.pi / 2, 3, "Y"), "Y": Matrix.Rotation(math.pi / 2, 3, "X"),
               "Z": Matrix.Identity(3)}[axis]
        bmesh.ops.rotate(self.bm, cent=(0, 0, 0), matrix=rot, verts=vs)
        bmesh.ops.translate(self.bm, vec=Vector(c) * MM, verts=vs)
        for f in {f for v in vs for f in v.link_faces}:
            f.material_index = mat
        return vs

    def obj(self, name, coll, bevel_mm=0.35):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        mats = materials()
        me.materials.append(mats["body"])
        me.materials.append(mats["dark"])
        for p in me.polygons:
            p.use_smooth = False
        o = bpy.data.objects.new(name, me)
        coll.objects.link(o)
        if bevel_mm > 0:
            bv = o.modifiers.new("bevel", "BEVEL")
            bv.width = bevel_mm * MM
            bv.segments = 2
            bv.limit_method = "ANGLE"
        return o


def _fins(mb, x0, x1, w, z0, n, fin_h, t=0.8):
    """n fins along X (running lengthwise) on top of a body of width w, base at z0."""
    for i in range(n):
        y = -w / 2 + (i + 0.5) * w / n
        mb.box(((x0 + x1) / 2, y, z0 + fin_h / 2), (x1 - x0, t, fin_h))


def _lc_duplex(mb, x, z, w_gap=6.25):
    for y in (-w_gap / 2, w_gap / 2):
        mb.box((x, y, z), (0.8, 4.6, 4.6), mat=1)


def _pull_tab(mb, x_front, w, z, length=30.0):
    mb.box((x_front + length / 2, 0, z), (length, w * 0.55, 1.2), mat=1)
    mb.box((x_front + length - 2.5, 0, z), (5.0, w * 0.75, 2.4), mat=1)


# ------------------------------------------------------------------ builders (body centred at the origin)
def b_qsfp(mb, key):
    w, h, L = DIMS[key][0]
    body = L - 20.0
    mb.box((-L / 2 + body / 2, 0, 0), (body, w, h))
    mb.box((L / 2 - 10.0, 0, (13.5 - h) / 2 * 0.6), (20.0, 19.0, 13.5 * 0.8))   # nose outside the cage
    mb.box((-L / 2 - 1.0, 0, -h / 2 + 0.4), (2.5, w * 0.8, 0.8), mat=1)          # paddle-card edge
    _lc_duplex(mb, L / 2 + 0.1, 0.6)
    _pull_tab(mb, L / 2, w, -1.2)


def b_qsfp_dd(mb, key):
    w, h, L = DIMS[key][0]
    body = L - 20.0
    mb.box((-L / 2 + body / 2, 0, 0), (body, w, h))
    mb.box((L / 2 - 10.0, 0, 1.2), (20.0, 19.0, 11.0))
    mb.box((-L / 2 - 1.0, 0, -h / 2 + 0.4), (2.5, w * 0.8, 0.8), mat=1)
    mb.box((-L / 2 - 1.0, 0, -h / 2 + 2.4), (2.5, w * 0.8, 0.8), mat=1)          # second contact row (DD)
    if key == "zr400":
        _fins(mb, L / 2 - 19, L / 2 - 1, 18.0, 6.7, 6, 3.5)                       # nose heatsink (stylized)
    _lc_duplex(mb, L / 2 + 0.1, 0.8)
    _pull_tab(mb, L / 2, w, -1.4)


def b_osfp(mb, key="osfp"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, -h / 4), (L, w, h / 2))
    _fins(mb, -L / 2 + 1, L / 2 - 1, w - 1.0, 0.0, 9, h / 2 - 0.3, t=1.0)         # finned heatsink (top shown open)
    mb.box((0, -w / 2 + 0.4, h / 4), (L, 0.8, h / 2))
    mb.box((0, w / 2 - 0.4, h / 4), (L, 0.8, h / 2))
    mb.box((-L / 2 + 6, 0, h / 2 - 0.4), (12, w, 0.8))                              # rear top bridge (fins visible)
    mb.box((L / 2 + 0.1, 0, 0), (0.8, 12.0, 3.6), mat=1)                           # MPO receptacle window
    mb.box((-L / 2 - 1.0, 0, -h / 2 + 0.6), (2.5, w * 0.8, 0.8), mat=1)
    _pull_tab(mb, L / 2, w, -h / 2 + 1.0, 35.0)


def b_sfp(mb, key="sfp"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, 0), (L, w, h))
    _lc_duplex(mb, L / 2 + 0.1, 0.4, 6.25 * 0.7)
    # bail latch: U loop in front
    for y in (-w / 2 + 0.6, w / 2 - 0.6):
        mb.box((L / 2 + 4.0, y, -h / 2 + 1.5), (8.0, 1.0, 1.0), mat=1)
    mb.box((L / 2 + 8.0, 0, -h / 2 + 1.5), (1.0, w, 1.0), mat=1)


def b_gbic(mb, key="gbic"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, 0), (L, w, h))
    mb.box((L / 2 - 6, 0, 0), (12, w + 2.0, h + 1.0), mat=1)                       # plastic front collar
    for y in (-6.5, 6.5):
        mb.box((L / 2 + 0.1, y, 0), (0.8, 9.0, 7.5), mat=1)                         # SC duplex windows
    for y in (-w / 2 - 1.2, w / 2 + 1.2):
        mb.box((L / 2 - 16, y, 0), (14, 1.0, 4.0), mat=1)                           # side latch tabs


def b_xenpak(mb, key="xenpak"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, -h / 4), (L, w, h / 2))
    _fins(mb, -L / 2 + 25, L / 2 - 2, w - 2, 0.0, 8, h / 2, t=1.4)                 # finned rear section
    mb.box((L / 2 - 1.5, 0, 0), (3.0, w + 2, h + 2))                                # front bezel
    for y in (-7.0, 7.0):
        mb.box((L / 2 + 0.2, y, 0), (0.8, 9.0, 7.5), mat=1)                          # SC duplex
    for y in (-w / 2 + 4, w / 2 - 4):
        mb.cyl((L / 2 + 1.5, y, -h / 2 + 3.5), 1.6, 3.0, "X", mat=1, seg=12)        # bezel screws


def b_cfp(mb, key="cfp"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, 0), (L, w, h))
    mb.box((L / 2 - 3, 0, 0), (6, w + 2, h + 1))                                    # front flange
    mb.box((L / 2 + 0.2, 0, 0), (0.8, w - 10, h - 4), mat=1)                         # front rail
    for y in (-8.0, 8.0):
        mb.box((L / 2 + 0.5, y, 0), (0.8, 12.0, 8.0))                                # LC duplex windows (light)
        mb.box((L / 2 + 0.7, y, 0), (0.8, 10.0, 6.5), mat=1)
    for y in (-w / 2 + 4.5, w / 2 - 4.5):
        mb.cyl((L / 2 + 3.0, y, 0), 3.8, 6.0, "X", mat=1, seg=20)                   # thumbscrews


def b_cfp2(mb, key="cfp2_dco"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, -h / 4), (L, w, h / 2))
    _fins(mb, -L / 2 + 2, L / 2 - 12, w - 1, 0.0, 10, h / 2, t=1.2)
    mb.box((L / 2 - 6, 0, h / 4), (12, w, h / 2))
    for y in (-5.5, 5.5):
        mb.box((L / 2 + 0.1, y, 0), (0.8, 4.6, 4.6), mat=1)
    _pull_tab(mb, L / 2, w, -h / 2 + 1, 25.0)


def b_xpo(mb, key="xpo"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, 0), (L, w, h))
    mb.box((-4, 0, h / 2 + 2.5), (L - 18, w - 2, 5.0))                              # integrated liquid cold plate
    for y in (-w / 4, w / 4):
        mb.cyl((-L / 2 - 4, y, h / 2 + 2.5), 3.0, 14.0, "X", mat=1, seg=16)         # coolant fittings (in/out)
        mb.cyl((-L / 2 - 12, y, h / 2 + 2.5), 4.2, 6.0, "X", mat=0, seg=16)
    mb.box((L / 2 + 0.1, 0, -1.0), (0.8, w - 8, 4.0), mat=1)                         # front ports
    _pull_tab(mb, L / 2, w, -h / 2 + 1, 25.0)


def b_one_by_nine(mb, key="one_by_nine"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, 0), (L, w, h))
    for y in (-6.0, 6.0):
        mb.box((L / 2 + 3.5, y, 0), (7.0, 9.5, 9.0))                                 # SC receptacle snouts
        mb.box((L / 2 + 7.1, y, 0), (0.6, 7.0, 6.5), mat=1)
    for i in range(9):                                                                # 9 through-hole pins, one row
        mb.box((-L / 2 + 4 + i * 2.54, 0, -h / 2 - 2.0), (0.5, 0.5, 4.0), mat=1)
    for x in (-L / 2 + 6, L / 2 - 6):
        mb.box((x, 0, -h / 2 - 1.5), (1.2, w - 4, 3.0))                              # mounting posts (stylized)


def b_aec(mb, key="aec"):
    w, h, L = DIMS[key][0]
    mb.box((0, 0, 0), (L, w, h))
    mb.box((-L / 2 - 1.0, 0, -h / 2 + 0.4), (2.5, w * 0.8, 0.8), mat=1)
    mb.box((L / 2 + 6, 0, 0), (12, w * 0.8, h * 1.1), mat=1)                         # cable boot
    n = 10
    for i in range(n):                                                                # cable stub curving down
        a = i / (n - 1) * math.pi / 3
        r = 45.0
        mb.cyl((L / 2 + 12 + r * math.sin(a), 0, -r * (1 - math.cos(a))), 4.0, 9.0, "X", mat=1, seg=14)


def b_cpo(mb, key="cpo"):
    mb.box((0, 0, -1.0), (34, 34, 1.4))                                               # package substrate
    mb.box((0, 0, 0.4), (20, 20, 1.4), mat=1)                                         # switch die / lid (dark)
    for k, (x, y) in enumerate(((0, 13.5), (0, -13.5), (13.5, 0), (-13.5, 0))):     # four optical engines
        sx, sy = (8, 4) if y else (4, 8)
        mb.box((x, y, 0.2), (sx, sy, 1.2))
    mb.box((24, 0, 0.0), (14, 6, 0.6), mat=1)                                         # fiber ribbon stub


BUILDERS = {
    "one_by_nine": b_one_by_nine, "gbic": b_gbic, "xenpak": b_xenpak, "cfp": b_cfp, "cfp2_dco": b_cfp2,
    "xpo": b_xpo, "sfp": b_sfp, "qsfp_plus": b_qsfp, "qsfp28": b_qsfp, "compatible": b_qsfp,
    "qsfp_dd": b_qsfp_dd, "zr400": b_qsfp_dd, "osfp": b_osfp, "aec": b_aec, "cpo": b_cpo,
}


def build(key: str, coll, name: str | None = None):
    mb = MB()
    BUILDERS[key](mb, key)
    o = mb.obj(name or f"FF_{key}", coll, bevel_mm=0.25 if key in ("sfp", "cpo") else 0.4)
    o["ff_key"] = key
    o["dims_mm_WxHxL"] = list(DIMS[key][0])
    o["dims_grade"] = DIMS[key][1]
    return o
