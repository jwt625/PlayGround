"""Thin-film lithium niobate (x-cut LNOI on Si) Mach-Zehnder modulator die, 6 x 6 x 0.5 mm.

Die frame (before placement): x = east (optical axis, input facet at west), y = north, z = up.
Crystal: x-cut, so the crystal z axis is in-plane and must be perpendicular to the waveguides
(die +y). The GSG traveling-wave electrode puts its field across the gaps along die y, i.e. along
crystal z (r33). The gold "+Z" arrow on the die points to die +y.

Layout (mm), after Wang et al. Opt. Express 26, 1547 (2018) and Xu et al. Optica 9, 61 (2022):
input edge coupler -> 1x2 MMI -> two arms that fan out around the west pad set (arm offset 0.40 mm)
-> S-bends that take each arm under its own ground conductor into the electrode gaps -> two
straight arms (48 um apart) in the two gaps of a straight traveling-wave GSG electrode
(x in [-1.6, 1.6]; ground / signal / ground = 300 / 40 / 300 um, 8 um gaps) -> the mirror image
-> 2x1 MMI -> output edge coupler. Each arm crosses only under its adjacent ground (cladding
crossing, not drawn), never under the signal line. At both ends the electrode tapers in line
(gaps 8 -> 40 um) to rectangular GSG pads at 150 um pitch: RF input at the west end, termination
at the east end; probes land along the optical axis direction, above the fibers.
Square 6 mm die, so it fits its fixture pocket in all four rotations.
"""
import math

import bmesh
import bpy
from mathutils import Vector

import common as C

DIE = 6.0
T = 0.5
YC = -0.4
XE, XT, XP = 1.60, 1.85, 1.95           # end of straight section, end of taper, end of pads (|x|)
WS, G, WG = 0.040, 0.008, 0.300         # straight section: signal, gap, ground
PS, PG = 0.080, 0.040                   # pads: signal pad width, pad gap (ground contacts at +-150 um)
H_AU = 0.0015                            # electrode thickness drawn (1.5 um)
PADS = [(-(XT + XP) / 2, YC), ((XT + XP) / 2, YC)]   # pad-set centres (signal pad)
ARM_OUT = 0.40                           # arm offset where the arms go around the pads


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def electrode_polys(n=200):
    """Three conductors (ground, signal, ground) of the straight CPW with in-line tapers to pads."""
    xs = [-XP + 2 * XP * i / n for i in range(n + 1)]

    def u(x):  # 0 in the straight section, 1 on the pads
        return _smooth((abs(x) - XE) / (XT - XE))

    hs = [WS / 2 + (PS / 2 - WS / 2) * u(x) for x in xs]
    gi = [WS / 2 + G + ((PS / 2 + PG) - (WS / 2 + G)) * u(x) for x in xs]
    go = WS / 2 + G + WG
    sig = [(x, YC - h) for x, h in zip(xs, hs)] + [(x, YC + h) for x, h in reversed(list(zip(xs, hs)))]
    gn = [(x, YC + g) for x, g in zip(xs, gi)] + [(XP, YC + go), (-XP, YC + go)]
    gs = [(-XP, YC - go), (XP, YC - go)] + [(x, YC - g) for x, g in reversed(list(zip(xs, gi)))]
    return [sig, gn, gs]


def _rect(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def fiducial_polys():
    polys = []
    # L-shaped corner mark, north-west corner
    cx, cy, L, w = -2.72, 2.72, 0.42, 0.05
    polys.append([(cx, cy), (cx + L, cy), (cx + L, cy - w), (cx + w, cy - w), (cx + w, cy - L), (cx, cy - L)])
    # alignment crosses at the other three corners
    for (x, y) in [(2.72, -2.72), (-2.72, -2.72), (2.72, 2.20)]:
        polys.append(_rect(x, y, 0.20, 0.03))
        polys.append(_rect(x, y, 0.03, 0.20))
    # +Z arrow (points to die +y), north of the modulator
    ax, ay0, ay1, sw, hw, hl = 0.0, 0.55, 1.75, 0.06, 0.22, 0.30
    polys.append([(ax - sw / 2, ay0), (ax + sw / 2, ay0), (ax + sw / 2, ay1 - hl), (ax + hw / 2, ay1 - hl),
                  (ax, ay1), (ax - hw / 2, ay1 - hl), (ax - sw / 2, ay1 - hl)])
    return polys


def _sbend(x0, x1, y0, y1, w, n=40):
    top, bot = [], []
    for i in range(n + 1):
        u = i / n
        x = x0 + (x1 - x0) * u
        y = y0 + (y1 - y0) * (u - math.sin(2 * math.pi * u) / (2 * math.pi))
        top.append((x, y + w / 2))
        bot.append((x, y - w / 2))
    if x1 < x0:
        top, bot = bot, top
    return bot + list(reversed(top))


def waveguide_polys(w=0.002):
    """Edge couplers, MMIs, arms around the pad sets, S-bends under the grounds, arms in the gaps."""
    off = WS / 2 + G / 2
    mmi_l, mmi_w = 0.060, 0.008
    m_off = mmi_w / 4
    polys = []
    for sx in (-1, 1):
        xm = sx * 2.66
        polys.append(_rect(sx * (DIE / 2 + abs(xm) + mmi_l / 2) / 2, YC, DIE / 2 - abs(xm) - mmi_l / 2, w))
        polys.append(_rect(xm, YC, mmi_l, mmi_w))
        xa = sx * (2.66 - mmi_l / 2)
        for sy in (1, -1):
            polys.append(_sbend(xa, sx * 2.10, YC + sy * m_off, YC + sy * ARM_OUT, w))
            polys.append(_rect(sx * 1.825, YC + sy * ARM_OUT, 0.55, w))
            polys.append(_sbend(sx * 1.55, sx * 1.20, YC + sy * ARM_OUT, YC + sy * off, w))
    for sy in (1, -1):
        polys.append(_rect(0, YC + sy * off, 2.4, w))
    polys.append(_rect(0, -2.45, DIE, w))  # reference straight waveguide
    return polys


def materials():
    m = {}
    m["au"] = C.principled("au_die", color=(1.0, 0.70, 0.27), metallic=1.0, rough=0.08, rough_var=0.02, rough_scale=3.0)
    # LNOI surfaces as colored mirrors: normal-incidence reflectance of the stack computed with a
    # transfer-matrix model (LN 300 nm slab or 600 nm rib on 4.7 um SiO2 on Si, CIE 1931, D65),
    # see color_stack.py. Linear sRGB.
    # Trench-defined rib process: the unetched 600 nm film covers the field, each waveguide is a
    # 1.5 um rib flanked by two 3 um trenches etched down to a 300 nm slab, so a waveguide reads
    # as a ~7.5 um line of slab colour on film colour (true dimensions, no exaggeration).
    m["top"] = C.principled("lnoi_film600", color=(0.374, 0.32, 0.369), metallic=1.0, rough=0.04,
                            rough_var=0.012, rough_scale=1.5)
    m["trench"] = C.principled("lnoi_slab300", color=(0.26, 0.409, 0.323), metallic=1.0, rough=0.4)  # ion-milled, rougher
    m["rib"] = C.principled("lnoi_rib600", color=(0.374, 0.32, 0.369), metallic=1.0, rough=0.035)
    m["facet"] = C.principled("si_facet", color=(0.03, 0.03, 0.035), rough=0.06, ior=3.9)
    m["diced"] = C.principled("si_diced", color=(0.05, 0.05, 0.055), rough=0.35, ior=3.9,
                              bump=0.4, bump_scale=60.0, bump_dist=0.0005)
    return m


def build(name="die", mats=None, rot_z_deg=0.0, loc=(0, 0, 0), label="MZM-07  X-CUT LN"):
    mats = mats or materials()
    root = C.empty(name, loc=loc, rot=(0, 0, math.radians(rot_z_deg)))
    # body with per-face materials
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(DIE, DIE, T), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0, -T / 2), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=[e for e in bm.edges], offset=0.006, segments=2, affect="EDGES")
    for f in bm.faces:
        nrm = f.normal
        if nrm.z > 0.9:
            f.material_index = 0
        elif abs(nrm.x) > 0.9:
            f.material_index = 1   # polished facets (east / west)
        else:
            f.material_index = 2
    me = bpy.data.meshes.new(name + "_body")
    bm.to_mesh(me)
    bm.free()
    body = C.link(bpy.data.objects.new(name + "_body", me))
    for k in ("top", "facet", "diced"):
        me.materials.append(mats[k])
    parts = [body]
    parts.append(C.extrude_polys(name + "_rf", electrode_polys(), 0.0, H_AU, mats["au"]))
    parts.append(C.extrude_polys(name + "_fid", fiducial_polys(), 0.0, H_AU, mats["au"]))
    parts.append(C.extrude_polys(name + "_trench", waveguide_polys(w=0.0075), 0.0, 0.0002, mats["trench"]))
    parts.append(C.extrude_polys(name + "_wg", waveguide_polys(w=0.0015), 0.0, 0.0004, mats["rib"]))
    txt = [
        C.text_mesh(name + "_t1", "+Z", 0.32, (0.36, 1.35, 0), h=H_AU, mat=mats["au"], align="LEFT"),
        C.text_mesh(name + "_t2", label, 0.20, (0.0, -0.05 + 2.25 - 0.0, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t3", "IN", 0.13, (-2.78, YC - 0.22, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t4", "OUT", 0.13, (2.74, YC - 0.22, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t5", "RF IN", 0.12, (-1.9, YC + ARM_OUT + 0.17, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t6", "TERM", 0.12, (1.9, YC + ARM_OUT + 0.17, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t7", "REF", 0.11, (0.0, -2.30, 0), h=H_AU, mat=mats["au"]),
    ]
    parts += txt
    C.parent(parts, root)
    return root
