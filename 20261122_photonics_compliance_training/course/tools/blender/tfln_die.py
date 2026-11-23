"""Thin-film lithium niobate (x-cut LNOI on Si) Mach-Zehnder modulator die, 6 x 6 x 0.5 mm.

Die frame (before placement): x = east (optical axis, input facet at west), y = north, z = up.
Crystal: x-cut, so the crystal z axis is in-plane and must be perpendicular to the waveguides
(die +y). The GSG traveling-wave electrode puts its field across the gaps along die y, i.e. along
crystal z (r33). The gold "+Z" arrow on the die points to die +y.

Layout (mm), after Wang et al. Opt. Express 26, 1547 (2018) and Xu et al. Optica 9, 61 (2022):
input edge coupler (west facet, y = -0.4) -> 1x2 MMI at x = -2.66 -> two arms (S-bends out to +-0.40 mm)
-> S-bends (x = -1.55..-1.20) into the two gaps of the active GSG section (x in [-1.40, 1.40]; ground /
signal / ground = 300 / 40 / 300 um, 8 um gaps, arms 48 um apart) -> the mirror image -> 2x1 MMI at
x = +2.66 -> output edge coupler (east facet). Outside the active section the CPW tapers (|x| =
1.40..1.58) to the feed geometry, signal 80 um and gaps 40 um, grounds unchanged. At the input (west)
end the feed bends 90 degrees north (0.45 mm centreline radius, vertical run at x = -2.05) to the RF
input pad set; at the output (east) end it bends south (x = +2.05) to the termination pad set. Each
line ends 0.95 mm inside the edge (|y| = 2.05) and its last 150 um are the GSG pads (contacts 90 um
from the line end at 150 um pitch). So the input probe lands from the north and the termination
(port 2) probe from the south, clear of the lensed fibers on the west and east facets. At each bend
one arm passes under the CPW (oxide-clad crossing, not drawn): the north arm at the input bend, the
south arm at the output bend; with off-axis pads one crossing per end is unavoidable for a push-pull
GSG electrode. A straight reference waveguide runs at y = -2.45.
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
XE, RB = 1.60, 0.45                     # end of the straight section (|x|), CPW bend radius (centreline)
XV = XE + RB                            # x of the vertical runs to the pads (|x|)
Y_PAD_END = 2.05                        # |y| of the outer end of each pad set (0.95 mm from the die edge)
XA = 1.40                               # |x| where the active (narrow-gap) section ends; taper to XE
L_PAD, D_TIP = 0.150, 0.090             # pad-set length at the line end, contact distance from the line end
WS, G, WG = 0.040, 0.008, 0.300         # active section: signal, gap, ground (arms in the 8 um gaps)
PS, PG = 0.080, 0.040                   # feed, bends and pads: signal 80 um, gaps 40 um (contacts at +-150 um)
PITCH = 0.150                           # GSG pitch (ground contact centre to signal centre)
H_AU = 0.0015                            # electrode thickness drawn (1.5 um)
# pad-set centres (signal pad) and the direction (unit, die frame) from the pads toward the probe body
PADS = [(-XV, Y_PAD_END - D_TIP), (XV, -(Y_PAD_END - D_TIP))]
PAD_DIRS = [(0.0, 1.0), (0.0, -1.0)]
ARM_OUT = 0.40                           # arm offset outside the electrode


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def cpw_path(step=0.004):
    """Centreline of the CPW from the north (input) pad end to the south (termination) pad end:
    list of (x, y, tx, ty, s) with unit tangent and arc length."""
    pts = []

    def add_line(p0, p1):
        n = max(2, int(math.dist(p0, p1) / step))
        t = ((p1[0] - p0[0]) / math.dist(p0, p1), (p1[1] - p0[1]) / math.dist(p0, p1))
        for i in range(n):
            u = i / n
            pts.append((p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u, t[0], t[1]))

    def add_arc(c, a0, a1):
        n = max(8, int(abs(a1 - a0) * RB / step))
        sg = 1.0 if a1 > a0 else -1.0
        for i in range(n):
            a = a0 + (a1 - a0) * i / n
            pts.append((c[0] + RB * math.cos(a), c[1] + RB * math.sin(a), -sg * math.sin(a), sg * math.cos(a)))

    add_line((-XV, Y_PAD_END), (-XV, YC + RB))
    add_arc((-XE, YC + RB), math.pi, 1.5 * math.pi)
    add_line((-XE, YC), (XE, YC))
    add_arc((XE, YC - RB), 0.5 * math.pi, 0.0)
    add_line((XV, YC - RB), (XV, -Y_PAD_END))
    pts.append((XV, -Y_PAD_END, 0.0, -1.0))
    out, s = [], 0.0
    for i, p in enumerate(pts):
        if i:
            s += math.dist(p[:2], pts[i - 1][:2])
        out.append((p[0], p[1], p[2], p[3], s))
    return out


def electrode_polys():
    """Three conductors (ground, signal, ground) along the bent CPW: narrow gaps (8 um) in the active
    straight section, tapering at |x| = 1.40..1.58 mm to the 80 / 40 um feed geometry that runs through the
    bends to the line ends, where it forms the GSG pad sets (150 um pitch). Outer ground edges fixed."""
    path = cpw_path()

    def u(x):  # 0 in the active section, 1 in feed, bends and pads
        return _smooth((abs(x) - XA) / (XE - 0.02 - XA))

    def side(f):
        return [(x - ty * f(x), y + tx * f(x)) for x, y, tx, ty, s in path]

    def hs(x):
        return WS / 2 + (PS / 2 - WS / 2) * u(x)

    def gi(x):
        return WS / 2 + G + ((PS / 2 + PG) - (WS / 2 + G)) * u(x)

    def go(x):
        return WS / 2 + G + WG

    sig = side(hs) + list(reversed(side(lambda x: -hs(x))))
    gl = side(go) + list(reversed(side(gi)))
    gr = side(lambda x: -gi(x)) + list(reversed(side(lambda x: -go(x))))
    return [sig, gl, gr]


def skate_polys(length=0.025, width=0.012):
    """Faint probe skate marks on the six pads (50 um overtravel, 2:1 skate), ending at each contact."""
    polys = []
    for (px, py), (dx, dy) in zip(PADS, PAD_DIRS):
        nx, ny = -dy, dx
        for k in (-1, 0, 1):
            cx, cy = px + nx * k * PITCH, py + ny * k * PITCH
            # the contact slid forward (away from the probe body): mark lies on the probe side of the contact
            x0, y0, x1, y1 = cx, cy, cx + dx * length, cy + dy * length
            polys.append([(x0 - nx * width / 2, y0 - ny * width / 2), (x1 - nx * width / 2, y1 - ny * width / 2),
                          (x1 + nx * width / 2, y1 + ny * width / 2), (x0 + nx * width / 2, y0 + ny * width / 2)])
    return polys


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
    """Edge couplers, MMIs, arms out at +-ARM_OUT, S-bends into the gaps (crossing under the grounds), arms in
    the gaps. Near each CPW bend one arm passes under the bent CPW (oxide-clad crossing)."""
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
    m["au_scrub"] = C.principled("au_scrub", color=(0.85, 0.62, 0.30), metallic=1.0, rough=0.45)
    # LNOI surfaces as colored mirrors: normal-incidence reflectance of the stack computed with a
    # transfer-matrix model (LN 300 nm slab or 600 nm rib on 4.7 um SiO2 on Si, CIE 1931, D65),
    # see color_stack.py. Linear sRGB.
    # Trench-defined rib process: the unetched 600 nm film covers the field, each waveguide is a
    # 1.5 um rib flanked by two 3 um trenches etched down to a 300 nm slab, so a waveguide reads
    # as a ~7.5 um line of slab colour on film colour (true dimensions, no exaggeration).
    m["top"] = C.principled("lnoi_film600", color=(0.374, 0.32, 0.369), metallic=1.0, rough=0.04,
                            rough_var=0.012, rough_scale=1.5)
    m["trench"] = C.principled("lnoi_slab300", color=(0.26, 0.409, 0.323), metallic=1.0, rough=0.7)  # ion-milled
    m["rib"] = C.principled("lnoi_rib600", color=(0.374, 0.32, 0.369), metallic=1.0, rough=0.035)
    m["facet"] = C.principled("si_facet", color=(0.03, 0.03, 0.035), rough=0.06, ior=3.9)
    m["diced"] = C.principled("si_diced", color=(0.05, 0.05, 0.055), rough=0.35, ior=3.9,
                              bump=0.4, bump_scale=60.0, bump_dist=0.0005)
    return m


def build(name="die", mats=None, rot_z_deg=0.0, loc=(0, 0, 0), label="MZM-07  X-CUT LN", skate=False):
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
    if skate:  # probed die: faint scrub marks where the contacts slid on the gold
        parts.append(C.extrude_polys(name + "_skate", skate_polys(), H_AU, 0.00005, mats["au_scrub"]))
    parts.append(C.extrude_polys(name + "_trench", waveguide_polys(w=0.0075), 0.0, 0.0002, mats["trench"]))
    parts.append(C.extrude_polys(name + "_wg", waveguide_polys(w=0.0015), 0.0, 0.0004, mats["rib"]))
    txt = [
        C.text_mesh(name + "_t1", "+Z", 0.32, (0.36, 1.35, 0), h=H_AU, mat=mats["au"], align="LEFT"),
        C.text_mesh(name + "_t2", label, 0.20, (0.0, -0.05 + 2.25 - 0.0, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t3", "IN", 0.13, (-2.78, YC - 0.22, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t4", "OUT", 0.13, (2.74, YC - 0.22, 0), h=H_AU, mat=mats["au"]),
        C.text_mesh(name + "_t5", "RF IN", 0.12, (PADS[0][0] + 0.48, PADS[0][1], 0), h=H_AU, mat=mats["au"],
                    align="LEFT"),
        C.text_mesh(name + "_t6", "TERM", 0.12, (PADS[1][0] - 0.48, PADS[1][1], 0), h=H_AU, mat=mats["au"],
                    align="RIGHT"),
        C.text_mesh(name + "_t7", "REF", 0.11, (0.0, -2.30, 0), h=H_AU, mat=mats["au"]),
    ]
    parts += txt
    C.parent(parts, root)
    return root
