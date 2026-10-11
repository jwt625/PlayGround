"""SUPERSEDED reference (2026-10-09): module-envelope layout v1 that produced outputs/hero/pigeon_osfp_dr8_exploded_module_layout.png.
The current hero is built by build_hero_pigeon.py. Kept only to reproduce that reference render; run it with an
output folder other than assets/hero (e.g. -- outputs/hero/work/module_layout_v1) so the pigeon asset is not overwritten.

Build the hero asset: 800G OSFP DR8 modeled as a cyberpunk surveillance pigeon (Blender 4.2, headless).

Run from the project folder (always through the Blender slot wrapper):
  scripts/bslot.sh -b --factory-startup --python scripts/assets/build_hero_pigeon.py -- assets/hero

Writes assets/hero/pigeon_osfp_dr8.blend, assets/hero/pigeon_osfp_dr8.glb, assets/hero/pigeon_osfp_dr8_parts.json.

Coordinates (meters in Blender; all numbers below are millimetres, converted by MM):
  origin HERO_ROOT = module bottom centre on the forward-stop plane B (OSFP MSA Rev 5.22 Fig 3-8/3-10),
  -Y = front face (head), +Y = card edge, +Z = up. A viewer facing the front face looks along +Y, so the
  viewer's left = -X = TX = the pigeon's RIGHT eye/foot (MSA Fig 14-46 MPO-16 lane map, Fig 15-1/3-21 pads).
  research/hero_component_dimensions.md uses x from the front face and +y = TX: Y = x - 89.79, X = -y.
Every dimension carries a grade in DevLog-003 (A spec, B datasheet/paper, C photo-measured, D estimate).
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

MM = 0.001
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["assets/hero"]
OUT_DIR = ARGS[0]
ASSET = "pigeon_osfp_dr8"

# ------------------------------------------------------------------ envelope (MSA Rev 5.22)
W = 22.58            # A Fig 3-2
HW = W / 2
H = 13.00            # A Fig 3-2
YF = -89.79          # A derived: 100.40 (Fig 3-2) - 10.61 (Fig 3-8) ahead of B
YREAR = 10.61        # A Fig 3-8 rear of top lip behind B
YTAPER = -0.89       # A derived: rear taper starts 11.50 (Fig 3-8) ahead of the rear lip
YNECK = -70.69       # A derived: 69.8 min in-cage length (Fig 3-2) ahead of the taper start = cage interface line
YHSK1 = -2.20        # A Fig 3-15 heat sink ends 2.20 min before B
ZHS0, ZHS1 = 9.20, 9.50      # A Fig 3-16 ex.1: heat sink floor and base plate
ZCOV0 = 12.70        # A Fig 3-16 ex.1: vents 3.20 high, top skin 0.30
NVENT, VENT_W, FIN_T, SIDE_RAIL = 10, 1.80, 0.30, 1.00   # A Fig 3-16 ex.1 (ref dims)
WALL = 1.00          # A Fig 3-16 example side wall
FLOOR = 1.00         # D (wall thickness of the example applied to the floor)
ZSPLIT = 6.10        # D shell split height (aligned with the latch pocket top 6.10 max, Fig 3-26)
ZNOSE = 9.20         # D nose roof height (heat-sink vent inlets stay open above it)
ZPCB0, ZPCB1 = 3.40, 4.40    # A Fig 3-8 (3.90 centre, 1.00 thick), Fig 3-16 (3.40/4.40)
Y_TIP = 9.80         # A Fig 3-8 PCB tip behind B
Y_E = Y_TIP - 1.40   # A Fig 3-21 signal pad leading edge (datum E side)
LATCH_Y = -44.59     # A Fig 3-26
LATCH_Z = (2.95, 6.10)       # A Fig 3-26
LATCH_DEPTH = 0.65   # A Fig 3-26
GUARD_Y = 9.62       # A Fig 3-8 bottom guard end
TAPER_DROP = 2.00    # A Fig 3-8

# ------------------------------------------------------------------ head / stylized parts (D unless noted)
EYE_DX = 3.80        # D eye centre offset (TX at -X); MPO-16 fiber halves would be at +/-1.0 (0.25 pitch)
EYE_Z = 6.00         # D
SOCKET_R = 2.40      # D
IRIS_RIN = 1.85      # D
LENS_PITCH = 0.75    # D 2x4 lenslet pitch
LENS_R = 0.33        # D
FACE_T = 1.50        # D face plate thickness
EDGE_PITCH_TX = 0.25  # B FAU / edge-coupler pitch 250 um (research claim 3)
EDGE_PITCH_RX = 0.50  # B octal TIA 500 um pitch (Semtech GN1825, research claim 4)
TAB_TIP_Y = -139.1   # A (ref) MSA App. B Fig B-1 pull tab tip ahead of B
TAB_W = 10.0         # D leash strap width

# ------------------------------------------------------------------ internals (updated from research/hero_component_dimensions.md)
#   research layout (x from the front face): PCB front ~30, TX PIC 43-51, RX blocks 42-51 at y +/-6.8, DSP 52-66,
#   power/MCU 66-89 (grades C/D, +/-5 mm). Model Y = x - 89.79.
def RY(x):
    return x - 89.79


INT = dict(
    pcb_front_y=RY(30.0),                                       # C/D research layout step 3
    bridge=dict(y0=YF + FACE_T, y1=RY(30.0) - 0.2, z1=4.40),   # D metal bridge (photo: fibres run over it)
    relay=dict(w=2.4, h=0.5, z_tx=5.00, z_rx=5.45),            # D stylized glass relay ("optic nerve")
    tx_pic=dict(cx=0.0, y0=RY(40.0), lx=7.0, ly=7.5, t=0.75),  # B Credo Carmel8 7 x 7.5 (8 MZM + 2 laser SiPh DR8 PIC);
    #                                                             x position C/D (photo PIC area x 43-51, shifted 3 mm forward
    #                                                             so the back-edge lasers clear the DSP); t D (0.15-0.78)
    lane_pitch=0.78,                                            # D 8 MZMs across the 7.0 mm PIC
    mzm_len=2.2,                                                # B (low) TW-MZM arm 2-2.5 mm
    dfb=dict(lx=0.30, ly=1.0, t=0.10),                          # B/D die: 1 mm cavity x 0.3 x 0.1 (Casela / MACOM classes)
    dfb_sub=dict(lx=1.0, ly=1.5, t=0.25),                       # D AlN chip-on-submount
    dfb_spacer=dict(lx=1.4, ly=1.7),                            # D raises the laser to the PIC waveguide height
    rx_cx=6.8,                                                  # C photo RX block lateral position (y +/-6.8)
    rx_y0=RY(41.0),                                             # C/D RX block front (photo x 42-51, shifted with the PIC)
    rx_pd=dict(lx=2.0, ly=0.25, t=0.15, pitch=0.25),            # B/D 1x8 PIN array (Lasermate 1x4 at 250 um, doubled)
    rx_pd_sub=dict(lx=2.6, ly=0.8, t=0.25),                     # D ceramic PD submount
    rx_prism_len=2.9,                                           # C/D lens/prism block (photo ~5 L, shortened), 2.4 W
    rx_tia=dict(lx=4.2, ly=1.2, t=0.20, pitch=0.50),            # A pitch (GN1825 500 um) / D size
    dsp=dict(cx=0.0, y0=RY(52.0), lx=13.0, ly=14.0, sub_t=0.6, die=(9.0, 9.5, 0.75)),  # B 13 x 14 (Credo Dove 800); H D
    thermal_pad_t=1.0,                                          # D 0.3-1.0
)


# ================================================================== mesh builder (mm)
class MB:
    def __init__(self):
        self.v, self.f = [], []

    def add(self, verts, faces):
        o = len(self.v)
        self.v += [tuple(p) for p in verts]
        self.f += [[i + o for i in fc] for fc in faces]
        return self

    def box(self, x0, x1, y0, y1, z0, z1):
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        z0, z1 = min(z0, z1), max(z0, z1)
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return self.add(v, f)

    def boxc(self, cx, cy, cz, sx, sy, sz):
        return self.box(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2)

    def prism(self, prof, a0, a1, plane):
        """Extrude polygon prof [(u, w)] between a0..a1. plane: 'yz' (along X), 'xy' (along Z), 'xz' (along Y)."""
        def P(u, w, a):
            return {"yz": (a, u, w), "xy": (u, w, a), "xz": (u, a, w)}[plane]
        n = len(prof)
        v = [P(u, w, a0) for u, w in prof] + [P(u, w, a1) for u, w in prof]
        f = [list(range(n))[::-1], list(range(n, 2 * n))]
        for i in range(n):
            j = (i + 1) % n
            f.append([i, j, n + j, n + i])
        return self.add(v, f)

    def cyl(self, c, r, h, axis="z", segs=24, r2=None):
        r2 = r if r2 is None else r2
        v = []
        for k, (rr, s) in enumerate(((r, -0.5), (r2, 0.5))):
            for i in range(segs):
                a = 2 * math.pi * i / segs
                p, q = rr * math.cos(a), rr * math.sin(a)
                if axis == "z":
                    v.append((c[0] + p, c[1] + q, c[2] + s * h))
                elif axis == "y":
                    v.append((c[0] + p, c[1] + s * h, c[2] + q))
                else:
                    v.append((c[0] + s * h, c[1] + p, c[2] + q))
        f = [list(range(segs))[::-1], list(range(segs, 2 * segs))]
        for i in range(segs):
            j = (i + 1) % segs
            f.append([i, j, segs + j, segs + i])
        return self.add(v, f)

    def ring(self, c, r_in, r_out, h, axis="y", segs=32):
        v = []
        for s in (-0.5, 0.5):
            for rr in (r_out, r_in):
                for i in range(segs):
                    a = 2 * math.pi * i / segs
                    p, q = rr * math.cos(a), rr * math.sin(a)
                    v.append((c[0] + p, c[1] + s * h, c[2] + q) if axis == "y" else (c[0] + p, c[1] + q, c[2] + s * h))
        n = segs
        f = []
        for i in range(n):
            j = (i + 1) % n
            f.append([0 * n + i, 0 * n + j, 2 * n + j, 2 * n + i])      # outer side
            f.append([1 * n + j, 1 * n + i, 3 * n + i, 3 * n + j])      # inner side
            f.append([0 * n + j, 0 * n + i, 1 * n + i, 1 * n + j])      # front annulus
            f.append([2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i])      # back annulus
        return self.add(v, f)

    def ellipsoid(self, c, rx, ry, rz, segs=16, rings=10, hemi=None):
        """hemi: None full, '-y' keeps the half toward -Y (dome facing front), '+z' keeps the top half."""
        v, f = [], []
        lat0, lat1 = -math.pi / 2, math.pi / 2
        if hemi in ("-y", "+z"):
            lat0 = 0.0
        rows = []
        for r in range(rings + 1):
            lat = lat0 + (lat1 - lat0) * r / rings
            row = []
            for i in range(segs):
                lon = 2 * math.pi * i / segs
                a, b, cc = math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)
                if hemi == "-y":
                    p = (c[0] + rx * a, c[1] - ry * cc, c[2] + rz * b)
                else:
                    p = (c[0] + rx * a, c[1] + ry * b, c[2] + rz * cc)
                row.append(len(v))
                v.append(p)
            rows.append(row)
        for r in range(rings):
            for i in range(segs):
                j = (i + 1) % segs
                f.append([rows[r][i], rows[r][j], rows[r + 1][j], rows[r + 1][i]])
        if lat0 == 0.0:
            f.append(rows[0][::-1])
        return self.add(v, f)

    def tube(self, pts, r, sides=6):
        pts = [Vector(p) for p in pts]
        v, f = [], []
        n = len(pts)
        prev_n = None
        for k in range(n):
            t = (pts[min(k + 1, n - 1)] - pts[max(k - 1, 0)]).normalized()
            if prev_n is None:
                up = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
                nn = t.cross(up).normalized()
            else:
                nn = (prev_n - t * prev_n.dot(t)).normalized()
            prev_n = nn
            bb = t.cross(nn).normalized()
            for i in range(sides):
                a = 2 * math.pi * i / sides
                v.append(tuple(pts[k] + (nn * math.cos(a) + bb * math.sin(a)) * r))
        for k in range(n - 1):
            for i in range(sides):
                j = (i + 1) % sides
                f.append([k * sides + i, k * sides + j, (k + 1) * sides + j, (k + 1) * sides + i])
        f.append(list(range(sides))[::-1])
        f.append([(n - 1) * sides + i for i in range(sides)])
        return self.add(v, f)

    def loft(self, sections, npts=16, power=2.0):
        """sections: [(x, y, z, half_w_x, half_h_z)] along the path; superellipse cross-sections in XZ."""
        v, f = [], []
        for (x, y, z, a, b) in sections:
            for i in range(npts):
                t = 2 * math.pi * i / npts
                ct, st = math.cos(t), math.sin(t)
                px = a * math.copysign(abs(ct) ** (2 / power), ct)
                pz = b * math.copysign(abs(st) ** (2 / power), st)
                v.append((x + px, y, z + pz))
        ns = len(sections)
        for k in range(ns - 1):
            for i in range(npts):
                j = (i + 1) % npts
                f.append([k * npts + i, k * npts + j, (k + 1) * npts + j, (k + 1) * npts + i])
        f.append(list(range(npts))[::-1])
        f.append([(ns - 1) * npts + i for i in range(npts)])
        return self.add(v, f)

    def hull8(self, front, back):
        """Hexahedron from 4 front + 4 back corner points (each ordered around the face)."""
        v = list(front) + list(back)
        f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        return self.add(v, f)


# ================================================================== scene, materials, objects
bpy.ops.wm.read_factory_settings(use_empty=True)
SCN = bpy.context.scene
TOP = bpy.data.collections.new("HERO_pigeon_osfp_dr8")
SCN.collection.children.link(TOP)
COLL = {}
for cname in ("shell_top", "shell_bottom", "fins", "head_eyes", "beak", "leash", "pcb", "dsp",
              "tx_optics", "rx_optics", "electronics", "feet"):
    c = bpy.data.collections.new(cname)
    TOP.children.link(c)
    COLL[cname] = c

ROOT = bpy.data.objects.new("HERO_ROOT", None)
ROOT.empty_display_type = "PLAIN_AXES"
ROOT.empty_display_size = 0.02
TOP.objects.link(ROOT)
ROOT["asset_id"] = ASSET
ROOT["units"] = "meters; origin = module bottom centre on forward-stop plane B; -Y front (head); -X = TX = pigeon's right"

MATS = {}


def mat(name, color, metal=0.0, rough=0.5, emit=0.0, ecol=None, alpha=1.0, coat=0.0):
    if name in MATS:
        return MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.08
    if emit:
        b.inputs["Emission Color"].default_value = (*(ecol or color), 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.surface_render_method = "BLENDED"
        m.use_transparency_overlap = True
    m.diffuse_color = (*color, alpha)
    MATS[name] = m
    return m


M = dict(
    shell_top=mat("gunmetal_top", (0.055, 0.058, 0.064), 1.0, 0.34, coat=0.4),
    shell_bot=mat("gunmetal_bottom", (0.075, 0.077, 0.082), 1.0, 0.42, coat=0.2),
    fins=mat("fin_metal", (0.13, 0.135, 0.145), 1.0, 0.26),
    feather_tip=mat("feather_tip_chrome", (0.30, 0.31, 0.34), 1.0, 0.16, coat=0.6),
    emi=mat("emi_spring_steel", (0.30, 0.31, 0.33), 1.0, 0.3),
    face=mat("face_black_chrome", (0.05, 0.05, 0.056), 1.0, 0.28, coat=0.7),
    iris=mat("iris_black", (0.006, 0.006, 0.008), 0.6, 0.3),
    bezel=mat("bezel_chrome", (0.45, 0.46, 0.5), 1.0, 0.12),
    glass=mat("glass_optic", (0.80, 0.90, 0.95), 0.0, 0.04, alpha=0.30),
    eye_tx=mat("eye_tx_red", (1.0, 0.03, 0.02), 0.0, 0.3, emit=4.0),
    eye_rx=mat("eye_rx_cyan", (0.02, 0.75, 1.0), 0.0, 0.3, emit=4.0),
    glow_tx=mat("retina_tx_red", (0.6, 0.02, 0.02), 0.0, 0.5, emit=3.0),
    glow_rx=mat("retina_rx_cyan", (0.02, 0.45, 0.6), 0.0, 0.5, emit=3.0),
    ray_tx=mat("ray_tx_red", (1.0, 0.1, 0.06), 0.0, 0.5, emit=6.0),
    ray_rx=mat("ray_rx_cyan", (0.1, 0.9, 1.0), 0.0, 0.5, emit=6.0),
    beak=mat("beak_dark_chrome", (0.16, 0.16, 0.18), 1.0, 0.15, coat=0.8),
    cere=mat("cere_ceramic", (0.86, 0.85, 0.80), 0.0, 0.45),
    leash=mat("pull_tab_yellow_107U", (0.98, 0.83, 0.12), 0.0, 0.45),
    pcb=mat("solder_mask_black", (0.008, 0.011, 0.012), 0.0, 0.38, coat=0.12),
    pcb_core=mat("fr4_core", (0.35, 0.33, 0.18), 0.0, 0.6),
    gold=mat("gold_enig", (1.0, 0.72, 0.30), 1.0, 0.18),
    copper=mat("copper", (0.95, 0.55, 0.35), 1.0, 0.3),
    silicon=mat("silicon_die", (0.13, 0.15, 0.22), 0.4, 0.18),
    si_wg=mat("si_waveguide", (0.55, 0.6, 0.75), 0.3, 0.2, emit=0.6, ecol=(1.0, 0.15, 0.1)),
    inp=mat("inp_die", (0.06, 0.055, 0.065), 0.4, 0.22),
    ge_pd=mat("ge_monitor_pd", (0.25, 0.22, 0.28), 0.5, 0.25),
    ceramic=mat("ceramic_white", (0.88, 0.87, 0.84), 0.0, 0.45),
    mold=mat("mold_black", (0.025, 0.025, 0.03), 0.0, 0.55),
    dsp_lid=mat("dsp_lid_nickel", (0.55, 0.55, 0.57), 1.0, 0.3),
    tpad=mat("thermal_pad_gray", (0.55, 0.57, 0.60), 0.0, 0.9),
    mlcc=mat("mlcc_tan", (0.55, 0.45, 0.30), 0.0, 0.5),
    tin=mat("tin_terminal", (0.75, 0.75, 0.75), 1.0, 0.3),
    inductor=mat("inductor_gray", (0.32, 0.33, 0.35), 0.2, 0.7),
    xtal=mat("crystal_can", (0.7, 0.7, 0.72), 1.0, 0.25),
    claw=mat("claw_black_chrome", (0.04, 0.04, 0.045), 1.0, 0.12, coat=0.8),
    toe_glow_tx=mat("toe_tx_red", (0.9, 0.05, 0.03), 0.0, 0.4, emit=2.5),
    toe_glow_rx=mat("toe_rx_cyan", (0.03, 0.75, 0.9), 0.0, 0.4, emit=2.5),
    pd_active=mat("pd_active_dark", (0.02, 0.05, 0.12), 0.2, 0.15),
)

PARTS = []   # (name, collection, explode, tris)


def obj_from_mb(name, mb, material, cname, explode=(0, 0, 0), origin=None, smooth=False, bevel=0.0,
                parent=None, label=None):
    if not mb.v:
        raise ValueError("empty mesh " + name)
    xs, ys, zs = zip(*mb.v)
    if origin is None:
        origin = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    verts = [((x - origin[0]) * MM, (y - origin[1]) * MM, (z - origin[2]) * MM) for x, y, z in mb.v]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], mb.f)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.validate()
    ob = bpy.data.objects.new(name, me)
    COLL[cname].objects.link(ob)
    par = parent or ROOT
    ob.parent = par
    pw = Vector(par.location) / MM if par is not ROOT else Vector((0, 0, 0))   # parents are direct children of ROOT
    ob.location = ((origin[0] - pw[0]) * MM, (origin[1] - pw[1]) * MM, (origin[2] - pw[2]) * MM)
    me.materials.append(material)
    for p in me.polygons:
        p.use_smooth = smooth
    if bevel:
        md = ob.modifiers.new("bevel", "BEVEL")
        md.width = bevel * MM
        md.segments = 2
        md.limit_method = "ANGLE"
        md.angle_limit = math.radians(40)
        md.harden_normals = False
    ob["explode"] = [e * MM for e in explode]
    ob["group"] = cname
    if label:
        ob["label"] = label
    PARTS.append(ob)
    return ob


def hook(name, loc, parent=None, cname=None, doc=""):
    h = bpy.data.objects.new(name, None)
    h.empty_display_type = "ARROWS"
    h.empty_display_size = 0.004
    (COLL[cname] if cname else TOP).objects.link(h)
    par = parent or ROOT
    h.parent = par
    pw = Vector(par.location) / MM if par is not ROOT else Vector((0, 0, 0))   # parents are direct children of ROOT
    h.location = ((loc[0] - pw[0]) * MM, (loc[1] - pw[1]) * MM, (loc[2] - pw[2]) * MM)
    h["doc"] = doc
    bpy.context.view_layer.update()
    return h


# explode offsets per group (mm); tuned for the exploded render
EX = dict(
    shell_top=(0, 12, 36), fins=(0, 16, 52), shell_bottom=(0, 6, -26), head_eyes=(0, -24, 4), beak=(0, -36, -1),
    leash=(0, -14, -24), pcb=(0, 0, 0), dsp=(0, 0, 14), tpad=(0, 0, 23), tx_optics=(0, 0, 10), dfb=(0, 3, 20),
    rx_optics=(3, 0, 10), electronics=(0, 0, 7), feet=(0, 9, 0), ruff=(0, -4, -26), relay=(0, -10, 10),
)

# ================================================================== SHELL BOTTOM
mb = MB()
mb.box(-HW, HW, YF, 0, 0, FLOOR)                                           # floor (body + nose)
for s in (-1, 1):
    xo, xi = s * HW, s * (HW - WALL)
    # side wall split around the latch pocket (pocket: 2.0 long ahead of the latch plane, z 2.95-6.10)
    mb.box(xo, xi, YF, LATCH_Y - 2.0, FLOOR, ZSPLIT)
    mb.box(xo, xi, LATCH_Y, 0, FLOOR, ZSPLIT)
    mb.box(xo, xi, LATCH_Y - 2.0, LATCH_Y, FLOOR, LATCH_Z[0])
    mb.box(xi + s * 0.0, xo - s * LATCH_DEPTH, LATCH_Y - 2.0, LATCH_Y, LATCH_Z[0], ZSPLIT)   # pocket back wall
    # forward-stop shoulder (vertical side wall faces at B, Fig 3-10) and rear guard side rails
    mb.box(xo, s * (HW - 1.6), 0, GUARD_Y, 0, 3.0)
mb.box(-HW + 1.6, HW - 1.6, 0.0, GUARD_Y - 0.4, 0, 1.0)                    # bottom guard plate 1.00 max (Fig 3-8)
shell_bottom = obj_from_mb("shell_bottom", mb, M["shell_bot"], "shell_bottom", EX["shell_bottom"], bevel=0.15,
                           label="shell (2-piece)")

# ================================================================== SHELL TOP (closed-top heat sink skin + walls + rear tongue)
mb = MB()
for s in (-1, 1):
    xo, xi = s * HW, s * (HW - WALL)
    mb.box(xo, xi, YNECK, YTAPER, ZSPLIT, ZHS0)                               # body side walls (upper half)
    mb.box(xo, xi, YF + FACE_T, YNECK, ZSPLIT, ZNOSE)                          # nose side walls (upper half)
    mb.box(xo, s * (HW - SIDE_RAIL), YNECK, YHSK1, ZHS1, ZCOV0)               # heat sink side rails (Fig 3-16)
mb.box(-HW, HW, YNECK, YHSK1, ZHS0, ZHS1)                                   # heat sink base plate
mb.box(-HW + WALL, HW - WALL, YF + FACE_T, YNECK, ZNOSE - 1.0, ZNOSE)        # nose roof
mb.box(-HW, HW, YNECK, YHSK1 + 0.0, ZCOV0, H)                               # closed-top skin 0.30 (Fig 3-16 ex.1)
# rear: solid trailing block from the heat sink end to the taper, then the tapered tongue over the card edge
mb.box(-HW, HW, YHSK1, YTAPER, ZHS0, H)
mb.prism([(YTAPER, 8.20), (YREAR, 8.20), (YREAR, H - TAPER_DROP), (YTAPER, H)], -HW, HW, "yz")   # Fig 3-8 tongue
for s in (-1, 1):
    mb.box(s * HW, s * (HW - WALL), YTAPER, 0.0, ZSPLIT, ZHS0)
shell_top = obj_from_mb("shell_top", mb, M["shell_top"], "shell_top", EX["shell_top"], bevel=0.15,
                        label="shell (2-piece)")

# rear vent slots on the tongue (Fig 3-2/3-13: 4 rounded slots), shown as dark inlays on the taper surface
mb = MB()
for i, xc in enumerate((-6.9, -2.3, 2.3, 6.9)):
    # inlay follows the taper plane z(y) = H - TAPER_DROP * (y - YTAPER) / (YREAR - YTAPER)
    y0, y1 = 2.0, 8.6
    z = lambda y: H - TAPER_DROP * (y - YTAPER) / (YREAR - YTAPER)
    mb.prism([(y0, z(y0) - 0.05), (y1, z(y1) - 0.05), (y1, z(y1) + 0.02), (y0, z(y0) + 0.02)], xc - 1.5, xc + 1.5, "yz")
vents = obj_from_mb("shell_top_rear_vents", mb, M["iris"], "shell_top", EX["shell_top"])

# ================================================================== FINS (vent ribs = feathers), feather-tip skin scallops
mb = MB()
x0 = -HW + SIDE_RAIL
fin_xs = []
for k in range(NVENT - 1):
    xa = x0 + VENT_W * (k + 1) + FIN_T * k
    fin_xs.append(xa + FIN_T / 2)
    # rib from the neck to the heat-sink end; front end raked like a quill, rear end tapered like a feather tip
    prof = [(YNECK + 1.2, ZHS1), (YHSK1, ZHS1), (YHSK1 + 1.6, ZHS1 + 1.2), (YHSK1 + 1.6, ZCOV0 - 0.4),
            (YHSK1, ZCOV0), (YNECK, ZCOV0), (YNECK, ZCOV0 - 0.6)]
    mb.prism(prof, xa, xa + FIN_T, "yz")
fins = obj_from_mb("fins_heatsink_ribs", mb, M["fins"], "fins", EX["fins"], label="heat-sink fins = feathers")

# feather tips: pointed chrome tips at the rear end of each vent channel, inside the envelope (below the skin, behind YHSK1)
mb = MB()
for k in range(NVENT):
    xa = x0 + (VENT_W + FIN_T) * k
    xb = xa + VENT_W
    xc = (xa + xb) / 2
    # flat pointed plate sitting on the trailing block top, tip pointing +Y
    poly = [(xa + 0.05, YHSK1 - 3.0), (xb - 0.05, YHSK1 - 3.0), (xb - 0.05, YHSK1 - 0.6), (xc, YHSK1 + 1.3), (xa + 0.05, YHSK1 - 0.6)]
    mb.prism(poly, H - 0.06, H + 0.02, "xy")
    # feather vane barbs: 3 short chevrons per channel on the skin (inlaid 0.06 mm)
    for j in range(3):
        yb = YHSK1 - 8.0 - j * 7.0
        mb.prism([(xa + 0.15, yb - 0.5), (xc, yb + 0.6), (xb - 0.15, yb - 0.5), (xb - 0.15, yb - 0.2), (xc, yb + 0.9),
                  (xa + 0.15, yb - 0.2)], H - 0.04, H + 0.015, "xy")
feather_tips = obj_from_mb("fins_feather_tips", mb, M["feather_tip"], "fins", EX["fins"])

# ================================================================== NECK RUFF (EMI spring fingers at the cage interface line)
mb = MB()
pitch = 1.45      # D (half the open-top fin pitch 2.90, Fig 3-20)
L_f = 4.8         # D finger length
y0r = YNECK
def finger(p0, outward, along, width):
    # bowed spring finger: 5 points, rising 0.55 mm outward at mid-length
    pts = []
    for i in range(5):
        t = i / 4
        bow = 0.55 * math.sin(math.pi * t) + 0.12
        pts.append(p0 + along * (L_f * t) + outward * bow)
    for i in range(4):
        a, b = pts[i], pts[i + 1]
        side = outward.cross(along).normalized() * (width / 2)
        th = outward * 0.08
        mb.hull8([a - side, a + side, a + side + th, a - side + th], [b - side, b + side, b + side + th, b - side + th])
along = Vector((0, 1, 0))
nx = int((W - 1.0) / pitch)
for i in range(nx):
    x = -HW + 0.5 + pitch * (i + 0.5)
    finger(Vector((x, y0r, H)), Vector((0, 0, 1)), along, 1.0)
    finger(Vector((x, y0r, 0.0)), Vector((0, 0, -1)), along, 1.0)
nz = int((H - 1.0) / pitch)
for i in range(nz):
    z = 0.5 + pitch * (i + 0.5)
    finger(Vector((HW, y0r, z)), Vector((1, 0, 0)), along, 1.0)
    finger(Vector((-HW, y0r, z)), Vector((-1, 0, 0)), along, 1.0)
mb.box(-HW - 0.12, HW + 0.12, y0r - 0.6, y0r, -0.12, H + 0.12)   # collar band
# collar band is a full box: hollow it by drawing only its 4 walls instead
mb.v, mb.f = mb.v[:-8], mb.f[:-6]
mb.box(-HW - 0.12, HW + 0.12, y0r - 0.6, y0r, H, H + 0.12)
mb.box(-HW - 0.12, HW + 0.12, y0r - 0.6, y0r, -0.12, 0.0)
mb.box(-HW - 0.12, -HW, y0r - 0.6, y0r, 0, H)
mb.box(HW, HW + 0.12, y0r - 0.6, y0r, 0, H)
ruff = obj_from_mb("neck_ruff_emi_fingers", mb, M["emi"], "shell_bottom", EX["ruff"], label="EMI fingers = neck ruff")

# ================================================================== HEAD: face plate with eye sockets, brow / skull fairing
eye_c = {"tx": (-EYE_DX, EYE_Z), "rx": (EYE_DX, EYE_Z)}
mb = MB()
mb.box(-HW, HW, YF, YF + FACE_T, 0.0, ZNOSE)
face = obj_from_mb("head_face_plate", mb, M["face"], "head_eyes", EX["head_eyes"], bevel=0.12, label="head (front face)")
# cut eye sockets with booleans
for key, (ex, ez) in eye_c.items():
    cmb = MB().cyl((ex, YF + FACE_T / 2, ez), SOCKET_R, FACE_T + 2.0, axis="y", segs=48)
    cut = obj_from_mb("_cut_" + key, cmb, M["iris"], "head_eyes")
    md = face.modifiers.new("socket_" + key, "BOOLEAN")
    md.object = cut
    md.operation = "DIFFERENCE"
    md.solver = "EXACT"
bpy.context.view_layer.objects.active = face
for md in list(face.modifiers):
    if md.type == "BOOLEAN":
        with bpy.context.temp_override(object=face, active_object=face):
            bpy.ops.object.modifier_apply(modifier=md.name)
for key in eye_c:
    o = bpy.data.objects["_cut_" + key]
    PARTS.remove(o)
    bpy.data.objects.remove(o, do_unlink=True)

# brow: angry V-shaped visor across the top of the face (does not reach the sockets)
brow_bottom = [(-HW, 8.95), (-EYE_DX - 1.0, 8.75), (-EYE_DX + 0.6, 8.55), (-1.3, 7.75), (0.0, 7.35),
               (1.3, 7.75), (EYE_DX - 0.6, 8.55), (EYE_DX + 1.0, 8.75), (HW, 8.95)]
brow_top = [(HW, 10.1), (5.0, 10.9), (0.0, 10.3), (-5.0, 10.9), (-HW, 10.1)]
mb = MB()
mb.prism(brow_bottom + brow_top, YF - 0.0, YF + 3.2, "xz")
# skull plates: swept fairing over the nose roof, fading out before the vent inlets at the neck
mb.prism([(YF + 3.2, ZNOSE), (YNECK - 3.0, ZNOSE), (YNECK - 3.0, ZNOSE + 0.25), (YF + 9.0, ZNOSE + 1.35), (YF + 3.2, 10.6)],
         -HW + 0.4, HW - 0.4, "yz")
# central crest ridge (creepy skull seam)
mb.prism([(YF + 0.5, 10.25), (YF + 12.0, ZNOSE + 1.1), (YNECK - 3.0, ZNOSE + 0.25), (YNECK - 3.0, ZNOSE + 0.45),
          (YF + 12.0, ZNOSE + 1.55), (YF + 1.2, 10.75)], -0.35, 0.35, "yz")
brow = obj_from_mb("head_brow_skull_fairing", mb, M["shell_top"], "head_eyes", EX["head_eyes"], bevel=0.1,
                   label="brow / skull fairing")

# ================================================================== EYES (2x4 lenslet arrays, iris rings, prism rearrangers)
HOOKS = {}
eye_objs = {}
EYE_LENS = {}
for key, (ex, ez) in eye_c.items():
    lens_mat = M["eye_tx"] if key == "tx" else M["eye_rx"]
    glow_mat = M["glow_tx"] if key == "tx" else M["glow_rx"]
    ray_mat = M["ray_tx"] if key == "tx" else M["ray_rx"]
    tag = "TX (pigeon's right eye)" if key == "tx" else "RX (pigeon's left eye)"
    mb = MB().ring((ex, YF + 0.75 + 0.15, ez), IRIS_RIN, SOCKET_R, FACE_T - 0.3, axis="y", segs=48)
    obj_from_mb("eye_%s_iris_ring" % key, mb, M["iris"], "head_eyes", EX["head_eyes"])
    mb = MB().ring((ex, YF - 0.05, ez), SOCKET_R, SOCKET_R + 0.35, 0.4, axis="y", segs=48)
    obj_from_mb("eye_%s_bezel" % key, mb, M["bezel"], "head_eyes", EX["head_eyes"])
    mb = MB().ring((ex, YF + 0.32, ez), IRIS_RIN - 0.16, IRIS_RIN + 0.02, 0.06, axis="y", segs=48)
    obj_from_mb("eye_%s_iris_glow_ring" % key, mb, glow_mat, "head_eyes", EX["head_eyes"])
    # lenslet substrate + 8 lens domes (2 rows x 4 columns)
    ys0 = YF + 0.55
    mb = MB().box(ex - 1.6, ex + 1.6, ys0, ys0 + 0.4, ez - 0.85, ez + 0.85)
    obj_from_mb("eye_%s_lenslet_substrate" % key, mb, M["glass"], "head_eyes", EX["head_eyes"])
    mb = MB()
    lens_pts = []
    for r in range(2):
        for c in range(4):
            lx = ex + (c - 1.5) * LENS_PITCH
            lz = ez + (r - 0.5) * LENS_PITCH
            lens_pts.append((lx, lz))
            mb.ellipsoid((lx, ys0, lz), LENS_R, 0.17, LENS_R, segs=16, rings=5, hemi="-y")
    lens = obj_from_mb("eye_%s_lenslet_array_2x4" % key, mb, lens_mat, "head_eyes", EX["head_eyes"], smooth=True,
                       label="2x4 lenslet array " + key.upper())
    eye_objs[key] = lens
    EYE_LENS[key] = lens_pts
    HOOKS["eye_" + key] = hook("HOOK_eye_" + key, (ex, YF - 0.2, ez), lens, "head_eyes",
                               doc="%s; local -Y = beam direction out of the eye (8 collimated lanes)" % tag)

# ================================================================== BEAK (stylized latch): upper + lower mandible, cere, hinge
HINGE = (0.0, YF - 0.2, 2.90)
hinge = hook("HOOK_beak_hinge", HINGE, None, "beak",
             doc="rotate about local X to open the lower mandible (positive X rotation opens downward-forward); upper mandible fixed")
hinge.empty_display_type = "SINGLE_ARROW"
hinge["explode"] = [e * MM for e in EX["beak"]]
secs = []
n = 9
for i in range(n):
    t = i / (n - 1)
    y = YF + 0.6 - 6.2 * t
    z = 3.85 + 0.55 * math.sin(math.pi * t * 0.7) - 1.55 * t ** 2.4
    a = 1.45 * (1 - t) ** 0.8 + 0.07
    b = 0.95 * (1 - t) ** 0.7 + 0.06
    secs.append((0.0, y, z, a, b))
mb = MB().loft(secs, npts=18, power=2.4)
upper = obj_from_mb("beak_upper_mandible", mb, M["beak"], "beak", EX["beak"], smooth=True, label="beak = latch")
secs = []
for i in range(n):
    t = i / (n - 1)
    y = YF + 0.2 - 5.3 * t
    z = 2.30 - 0.15 * t - 0.12 * math.sin(math.pi * t)
    a = 1.30 * (1 - t) ** 0.8 + 0.06
    b = 0.62 * (1 - t) ** 0.7 + 0.05
    secs.append((0.0, y, z, a, b))
mb = MB().loft(secs, npts=18, power=2.4)
lower = obj_from_mb("beak_lower_mandible", mb, M["beak"], "beak", EX["beak"], smooth=True, parent=hinge,
                    origin=HINGE)
mb = MB().ellipsoid((0.0, YF - 0.15, 4.95), 0.85, 0.45, 0.38, segs=16, rings=8)
cere = obj_from_mb("beak_cere", mb, M["cere"], "beak", EX["beak"], smooth=True)

# ================================================================== LEASH (pull tab, MSA Table 3-3 yellow for 1310 nm <= 500 m)
mb = MB()
zt0, zt1 = -0.75, -0.35
# strap under the nose (front region may extend 1.6 below the bottom, Fig 3-2 note 1), then out to the reference tip (App. B Fig B-1)
mb.box(-TAB_W / 2, TAB_W / 2, YNECK - 0.5, YF - 1.0, zt0, zt1)      # under the nose only (outside the cage)
pts = []
segs = 10
for i in range(segs + 1):
    t = i / segs
    y = YF - 1.0 + (TAB_TIP_Y + 8.0 - (YF - 1.0)) * t
    z = (zt0 + zt1) / 2 - 1.0 * math.sin(math.pi * t) * 0.6 - 0.2 * t
    pts.append((y, z))
for i in range(segs):
    (ya, za), (yb, zb) = pts[i], pts[i + 1]
    mb.prism([(ya, za - 0.2), (yb, zb - 0.2), (yb, zb + 0.2), (ya, za + 0.2)], -TAB_W / 2, TAB_W / 2, "yz")
leash = obj_from_mb("leash_pull_tab_strap", mb, M["leash"], "leash", EX["leash"], label="pull tab = leash")
# end loop (handle)
yl, zl = TAB_TIP_Y + 8.0, pts[-1][1]
mb = MB()
outer = 6.5
for i in range(24):
    a0, a1 = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24
    def P(a, r):
        return (r * math.cos(a) * 1.0, yl - 4.0 + r * math.sin(a) * 0.9 - 0.0)
    for (r_in, r_out) in ((4.2, outer),):
        p0, p1, p2, p3 = P(a0, r_in), P(a1, r_in), P(a1, r_out), P(a0, r_out)
        mb.prism([p0, p1, p2, p3], zl - 0.2, zl + 0.2, "xy")
loop = obj_from_mb("leash_end_loop", mb, M["leash"], "leash", EX["leash"])
HOOKS["leash_end"] = hook("HOOK_leash_end", (0.0, yl - 4.0 - outer * 0.9, zl), loop, "leash",
                          doc="end of the pull-tab loop; the handler's grip point")

# ================================================================== PCB (paddle card) + 60 pads (MSA Fig 3-21/3-22, Fig 15-1)
yp0 = INT["pcb_front_y"]
half = [(9.745, yp0), (9.745, 1.9), (8.125, 2.6), (8.125, 4.0), (9.385, 4.6), (9.385, Y_TIP - 0.25), (9.135, Y_TIP)]
outline = half + [(-x, y) for x, y in reversed(half)]
mb = MB().prism(outline, ZPCB0, ZPCB1, "xy")
pcb = obj_from_mb("pcb_paddle_card", mb, M["pcb"], "pcb", EX["pcb"], label="paddle-card PCB")


def pin_type(n):
    gnd = {1, 4, 7, 10, 13, 18, 21, 24, 27, 30, 31, 34, 37, 40, 43, 48, 51, 54, 57, 60}
    if n in gnd:
        return "GND"
    if n in (15, 16, 45, 46):
        return "VCC"
    if n in (14, 17, 44, 47):
        return "LS"
    if 2 <= n <= 12 or 49 <= n <= 59:
        return "TX"
    return "RX"


def pin_x(n):
    return -8.85 + (60 - n) * 0.60 if n >= 31 else -8.55 + (n - 1) * 0.60


PAD_REAR = Y_E - 1.50
PAD_FRONT = {"GND": Y_TIP - 0.55, "VCC": Y_TIP - 0.85, "LS": Y_E, "TX": Y_E, "RX": Y_E}
mb_top, mb_bot = MB(), MB()
pad_t = 0.035   # D (visual thickness; real plating is ~4.6 um, sec 3.6)
for n in range(1, 61):
    tp = pin_type(n)
    if n in (16, 46):
        continue                      # power pads are merged pairs (Fig 3-22/3-23)
    if n in (15, 45):
        xc = (pin_x(n) + pin_x(n + 1)) / 2
        w = 0.98
    else:
        xc, w = pin_x(n), 0.38
    target = mb_top if n >= 31 else mb_bot
    z0, z1 = (ZPCB1, ZPCB1 + pad_t) if n >= 31 else (ZPCB0 - pad_t, ZPCB0)
    target.box(xc - w / 2, xc + w / 2, PAD_REAR, PAD_FRONT[tp], z0, z1)
obj_from_mb("pcb_edge_pads_top_31_60", mb_top, M["gold"], "pcb", EX["pcb"], label="60-pad edge (feet)")
obj_from_mb("pcb_edge_pads_bottom_1_30", mb_bot, M["gold"], "pcb", EX["pcb"])
HOOKS["i2c_pads"] = hook("HOOK_i2c_pads", (0.0, Y_E - 0.6, (ZPCB0 + ZPCB1) / 2), pcb, "pcb",
                         doc="middle of the edge: SCL(14) SDA(47) INT/RSTn(44) LPWn/PRSn(17) + VCC 15/16/45/46")

# ================================================================== FEET: toe fairings + 8 toes per foot (one per lane)
lane_pairs = {"tx": {"top": [(59, 58), (56, 55), (53, 52), (50, 49)], "bot": [(2, 3), (5, 6), (8, 9), (11, 12)]},
              "rx": {"top": [(41, 42), (38, 39), (35, 36), (32, 33)], "bot": [(19, 20), (22, 23), (25, 26), (28, 29)]}}
FAIR_Y0, FAIR_Y1 = 2.75, 4.95    # D: ahead of the connector mating zone (3.40 min from E, Fig 3-24)
TOE_TIP_Y = PAD_REAR - 0.12
for key in ("tx", "rx"):
    glow = M["toe_glow_tx"] if key == "tx" else M["toe_glow_rx"]
    xs_all = [pin_x(p) for side in lane_pairs[key].values() for pr in side for p in pr]
    xa, xb = min(xs_all) - 0.25, max(xs_all) + 0.25
    xa, xb = max(xa, -8.0), min(xb, 8.0)
    mbf, mbt, mbg = MB(), MB(), MB()
    for side, zs in (("top", 1), ("bot", -1)):
        zb = ZPCB1 if zs > 0 else ZPCB0
        # fairing: thin plate with a scalloped rear edge (one scallop per toe)
        mbf.prism([(xa, FAIR_Y0), (xb, FAIR_Y0), (xb, FAIR_Y1 - 0.4), (xa, FAIR_Y1 - 0.4)], zb, zb + zs * 0.30, "xy")
        for (pa, pb) in lane_pairs[key][side]:
            xc = (pin_x(pa) + pin_x(pb)) / 2
            secs = []
            for i in range(7):
                t = i / 6
                y = FAIR_Y1 - 0.9 + (TOE_TIP_Y - (FAIR_Y1 - 0.9)) * t
                arch = 0.30 * math.sin(math.pi * min(1.0, t * 1.15))
                z = zb + zs * (0.18 + arch - 0.14 * t)
                secs.append((xc, y, z, 0.26 * (1 - t) + 0.035, 0.15 * (1 - t) + 0.03))
            mbt.loft(secs, npts=10, power=2.0)
            # glowing knuckle band on each toe (lane indicator)
            mbg.box(xc - 0.16, xc + 0.16, FAIR_Y0 + 0.35, FAIR_Y0 + 0.75, zb + zs * 0.30, zb + zs * 0.36)   # lane light on the fairing
    obj_from_mb("feet_%s_toe_fairing" % key, mbf, M["shell_top"], "feet", EX["feet"], bevel=0.05)
    toes = obj_from_mb("feet_%s_toes_x8" % key, mbt, M["claw"], "feet", EX["feet"], smooth=True,
                       label=("TX foot (8 toes)" if key == "tx" else "RX foot (8 toes)"))
    obj_from_mb("feet_%s_toe_lane_lights" % key, mbg, glow, "feet", EX["feet"], smooth=True)
    xs_pad = [pin_x(p) for side in lane_pairs[key].values() for pr in side for p in pr]
    HOOKS["foot_" + key] = hook("HOOK_foot_" + key, (sum(xs_pad) / len(xs_pad), Y_E, (ZPCB0 + ZPCB1) / 2), toes, "feet",
                                doc="%s pad group centre (8 lanes: 4 pairs top, 4 pairs bottom)" % key.upper())

# ================================================================== INTERNALS (research/hero_component_dimensions.md layout)
import random

ZT = ZPCB1
OCC = []   # occupied PCB-top rectangles (x0, x1, y0, y1) for the passive scatter


def occ(x0, x1, y0, y1, pad=0.25):
    OCC.append((min(x0, x1) - pad, max(x0, x1) + pad, min(y0, y1) - pad, max(y0, y1) + pad))


def catmull(pts, per=6):
    P = [Vector(p) for p in pts]
    P = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(per):
            t = k / per
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return [tuple(v) for v in out]


def bond(mb, p0, p1, h=0.25, r=0.0125, n=9):
    """Gold ball-wedge wire bond as a parabolic arc (25 um wire, D)."""
    pts = []
    ztop = max(p0[2], p1[2]) + h
    for i in range(n):
        t = i / (n - 1)
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        zl = p0[2] + (p1[2] - p0[2]) * t
        z = zl + (ztop - zl) * 4 * t * (1 - t) * (1.0 if t < 0.5 else 1.0)
        pts.append((x, y, z))
    mb.tube(pts, r, sides=4)


# ---- metal optical bridge in the nose (the real module routes fibres over it; here it carries the glass relays)
b = INT["bridge"]
mb = MB().box(-HW + WALL, HW - WALL, b["y0"], b["y1"], b["z1"] - 0.8, b["z1"])
for s in (-1, 1):
    mb.box(s * (HW - WALL), s * (HW - WALL - 1.2), b["y0"], b["y1"], FLOOR, b["z1"] - 0.8)
obj_from_mb("shell_bottom_optical_bridge", mb, M["shell_bot"], "shell_bottom", EX["shell_bottom"], bevel=0.1)

# ---- TX PIC (SiPh): die, splitter trees, MZM arms, electrodes, monitor PDs, fan-in, edge couplers
p = INT["tx_pic"]
pf, pb = p["y0"], p["y0"] + p["ly"]
px0, px1 = p["cx"] - p["lx"] / 2, p["cx"] + p["lx"] / 2
zt = ZT + p["t"]
occ(px0, px1, pf, pb)
mb = MB().box(px0, px1, pf, pb, ZT, zt)
pic = obj_from_mb("tx_pic_siph_die", mb, M["silicon"], "tx_optics", EX["tx_optics"], label="TX SiPh PIC")
LP = INT["lane_pitch"]
lanes = [p["cx"] + (i - 3.5) * LP for i in range(8)]
gcs = [sum(lanes[0:4]) / 4, sum(lanes[4:8]) / 4]                  # laser / splitter-tree centres
edge_x = [p["cx"] + (i - 3.5) * EDGE_PITCH_TX for i in range(8)]
zw = zt + 0.012
wg = MB()
rw = 0.016
y_s1, y_s2, y_mmi = pb - 0.3, pb - 0.9, pb - 1.5
y_a0, y_a1 = pb - 1.7, pb - 1.7 - INT["mzm_len"]
y_comb, y_fan0, y_fan1 = y_a1 - 0.2, y_a1 - 0.5, pf + 0.35
for g, gc in enumerate(gcs):
    wg.tube([(gc, pb, zw), (gc, y_s1, zw)], rw, 4)                    # laser input
    for s1 in (-1, 1):
        x1 = gc + s1 * LP
        wg.tube(catmull([(gc, y_s1, zw), (gc + s1 * LP * 0.5, (y_s1 + y_s2) / 2, zw), (x1, y_s2, zw)], 4), rw, 4)
        for s2 in (-1, 1):
            x2 = x1 + s2 * LP / 2
            wg.tube(catmull([(x1, y_s2, zw), (x1 + s2 * LP * 0.25, (y_s2 + y_mmi) / 2, zw), (x2, y_mmi, zw)], 4), rw, 4)
mzm_el = MB()
mpd = MB()
for i, lx in enumerate(lanes):
    wg.tube([(lx, y_mmi, zw), (lx, y_a0 + 0.1, zw)], rw, 4)
    for s in (-1, 1):                                                 # two MZM arms
        wg.tube([(lx, y_a0 + 0.1, zw), (lx + s * 0.11, y_a0, zw), (lx + s * 0.11, y_a1, zw), (lx, y_comb, zw)], rw, 4)
    wg.tube(catmull([(lx, y_comb, zw), (lx, y_fan0, zw), ((lx + edge_x[i]) / 2, (y_fan0 + y_fan1) / 2, zw),
                     (edge_x[i], y_fan1, zw), (edge_x[i], pf + 0.02, zw)], 5), rw, 4)
    # edge-coupler taper at the front facet
    wg.tube([(edge_x[i], pf + 0.35, zw), (edge_x[i], pf + 0.01, zw)], 0.03, 4)
    # G-S-G travelling-wave electrodes (gold) over the arms
    for dx, w in ((-0.28, 0.18), (0.0, 0.12), (0.28, 0.18)):
        mzm_el.box(lx + dx - w / 2, lx + dx + w / 2, y_a1, y_a0, zt, zt + 0.01)
    # RF feed from the electrode back end to the back-edge bond pad
    mzm_el.box(lx - 0.03, lx + 0.03, y_a0, pb - 0.2, zt, zt + 0.008)
    mzm_el.box(lx - 0.08, lx + 0.08, pb - 0.24, pb - 0.08, zt, zt + 0.012)
    # monitor PD tap (Ge) after the combiner
    wg.tube([(lx, y_comb - 0.05, zw), (lx + 0.22, y_comb - 0.2, zw)], rw * 0.8, 4)
    mpd.box(lx + 0.17, lx + 0.33, y_comb - 0.32, y_comb - 0.16, zt, zt + 0.02)
    mzm_el.box(lx + 0.33, lx + 0.39, y_comb - 0.28, y_comb - 0.2, zt, zt + 0.008)
obj_from_mb("tx_pic_waveguides_splitters", wg, M["si_wg"], "tx_optics", EX["tx_optics"], label="splitter waveguides")
obj_from_mb("tx_mzm_x8_electrodes", mzm_el, M["gold"], "tx_optics", EX["tx_optics"], label="MZM x8")
obj_from_mb("tx_monitor_pds_x8", mpd, M["ge_pd"], "tx_optics", EX["tx_optics"])

# ---- 2 CW DFB lasers butt-coupled at the PIC back edge (spacer + AlN submount + InP die), facet glow, bonds
d, ds, dsp_ = INT["dfb"], INT["dfb_sub"], INT["dfb_spacer"]
z_sp1 = zt - d["t"] - ds["t"]
mb_sp, mb_sub, mb_die, mb_glow, mb_lb = MB(), MB(), MB(), MB(), MB()
for gc in gcs:
    mb_sp.box(gc - dsp_["lx"] / 2, gc + dsp_["lx"] / 2, pb + 0.02, pb + 0.02 + dsp_["ly"], ZT, z_sp1)
    mb_sub.box(gc - ds["lx"] / 2, gc + ds["lx"] / 2, pb + 0.02, pb + 0.02 + ds["ly"], z_sp1, z_sp1 + ds["t"])
    mb_die.box(gc - d["lx"] / 2, gc + d["lx"] / 2, pb + 0.005, pb + 0.005 + d["ly"], z_sp1 + ds["t"], zt)
    mb_glow.ellipsoid((gc, pb + 0.01, zt - 0.02), 0.07, 0.05, 0.035, segs=10, rings=6)
    mb_sub.box(gc + 0.22, gc + 0.45, pb + 0.5, pb + 1.3, z_sp1 + ds["t"], z_sp1 + ds["t"] + 0.01)   # submount pad (gold on AlN)
    bond(mb_lb, (gc, pb + 0.6, zt), (gc + 0.33, pb + 0.9, z_sp1 + ds["t"] + 0.01), h=0.18)
    bond(mb_lb, (gc + 0.33, pb + 1.1, z_sp1 + ds["t"] + 0.01), (gc + 0.9, pb + 2.1, ZT + 0.02), h=0.3)
    occ(gc - dsp_["lx"] / 2, gc + dsp_["lx"] / 2, pb, pb + dsp_["ly"])
obj_from_mb("tx_dfb_spacers", mb_sp, M["tin"], "tx_optics", EX["dfb"])
obj_from_mb("tx_dfb_submounts_aln", mb_sub, M["ceramic"], "tx_optics", EX["dfb"])
obj_from_mb("tx_dfb_laser_dies_x2", mb_die, M["inp"], "tx_optics", EX["dfb"], label="CW DFB x2")
obj_from_mb("tx_dfb_facet_glow", mb_glow, M["eye_tx"], "tx_optics", EX["dfb"], smooth=True)
obj_from_mb("tx_dfb_wirebonds", mb_lb, M["gold"], "tx_optics", EX["dfb"])

# ---- TX RF wire bonds: PIC back-edge pads -> PCB pads behind the lasers (DSP drives the MZMs directly)
mb_b, mb_pp = MB(), MB()
y_pcbpad = pb + dsp_["ly"] + 0.45
for lx in lanes:
    bond(mb_b, (lx, pb - 0.16, zt + 0.012), (lx * 1.08, y_pcbpad, ZT + 0.03), h=0.55)
    mb_pp.box(lx * 1.08 - 0.12, lx * 1.08 + 0.12, y_pcbpad - 0.15, y_pcbpad + 0.25, ZT, ZT + 0.025)
for s in (-1, 1):                                                     # bias / ground bonds on the PIC sides
    for k in range(5):
        yk = pf + 1.2 + k * 1.1
        mzm_el2 = (s * (p["lx"] / 2 - 0.12), yk, zt)
        bond(mb_b, mzm_el2, (s * (p["lx"] / 2 + 0.7), yk, ZT + 0.03), h=0.3)
        mb_pp.box(s * (p["lx"] / 2 + 0.7) - 0.12, s * (p["lx"] / 2 + 0.7) + 0.12, yk - 0.12, yk + 0.12, ZT, ZT + 0.025)
        mb_pp.box(s * (p["lx"] / 2 - 0.12) - 0.07, s * (p["lx"] / 2 - 0.12) + 0.07, yk - 0.07, yk + 0.07, zt, zt + 0.01)
occ(-p["lx"] / 2 - 1.0, p["lx"] / 2 + 1.0, pf, y_pcbpad + 0.3)
obj_from_mb("tx_pic_wirebonds", mb_b, M["gold"], "tx_optics", EX["tx_optics"])
obj_from_mb("pcb_tx_bond_pads", mb_pp, M["gold"], "pcb", EX["pcb"])

# ---- RX: 45-degree turning prism, 1x8 PIN PD array on submount, octal TIA, fan-out bonds
rx = INT["rx_cx"]
ry0 = INT["rx_y0"]
pdd, pds, tia = INT["rx_pd"], INT["rx_pd_sub"], INT["rx_tia"]
y_pd = ry0 + INT["rx_prism_len"] - 0.45                              # PD array centre (under the 45 deg face)
z_pd_top = ZT + pds["t"] + pdd["t"]
z_rx = INT["relay"]["z_rx"]
mb = MB().box(rx - pds["lx"] / 2, rx + pds["lx"] / 2, y_pd - pds["ly"] / 2, y_pd + pds["ly"] / 2, ZT, ZT + pds["t"])
obj_from_mb("rx_pd_submount", mb, M["ceramic"], "rx_optics", EX["rx_optics"])
mb = MB().box(rx - pdd["lx"] / 2, rx + pdd["lx"] / 2, y_pd - pdd["ly"] / 2, y_pd + pdd["ly"] / 2, ZT + pds["t"], z_pd_top)
obj_from_mb("rx_pin_pd_array_1x8", mb, M["inp"], "rx_optics", EX["rx_optics"], label="PIN PD array (8 ch)")
pd_x = [rx + (i - 3.5) * pdd["pitch"] for i in range(8)]
mb_act, mb_rpad = MB(), MB()
for x in pd_x:
    mb_act.cyl((x, y_pd - 0.03, z_pd_top + 0.004), 0.045, 0.008, axis="z", segs=12)
    mb_rpad.box(x - 0.04, x + 0.04, y_pd + 0.05, y_pd + 0.11, z_pd_top, z_pd_top + 0.008)
obj_from_mb("rx_pd_active_areas", mb_act, M["pd_active"], "rx_optics", EX["rx_optics"])
y_tia0 = y_pd + pds["ly"] / 2 + 0.35
y_tia1 = y_tia0 + tia["ly"]
mb = MB().box(rx - tia["lx"] / 2, rx + tia["lx"] / 2, y_tia0, y_tia1, ZT, ZT + tia["t"])
obj_from_mb("rx_octal_tia", mb, M["silicon"], "rx_optics", EX["rx_optics"], label="octal TIA")
tia_in = [rx + (i - 3.5) * tia["pitch"] for i in range(8)]
mb_rb = MB()
for i in range(8):
    mb_rpad.box(tia_in[i] - 0.06, tia_in[i] + 0.06, y_tia0 + 0.06, y_tia0 + 0.18, ZT + tia["t"], ZT + tia["t"] + 0.008)
    bond(mb_rb, (pd_x[i], y_pd + 0.08, z_pd_top + 0.008), (tia_in[i], y_tia0 + 0.12, ZT + tia["t"] + 0.008), h=0.2)
    # TIA outputs to the PCB (towards the DSP)
    mb_rpad.box(tia_in[i] - 0.06, tia_in[i] + 0.06, y_tia1 - 0.18, y_tia1 - 0.06, ZT + tia["t"], ZT + tia["t"] + 0.008)
    mb_rpad.box(tia_in[i] - 0.11, tia_in[i] + 0.11, y_tia1 + 0.45, y_tia1 + 0.75, ZT, ZT + 0.025)
    bond(mb_rb, (tia_in[i], y_tia1 - 0.12, ZT + tia["t"] + 0.008), (tia_in[i], y_tia1 + 0.6, ZT + 0.025), h=0.2)
obj_from_mb("rx_bond_pads", mb_rpad, M["gold"], "rx_optics", EX["rx_optics"])
obj_from_mb("rx_wirebonds", mb_rb, M["gold"], "rx_optics", EX["rx_optics"])
# turning prism: slanted face through (y_pd, z_rx) at 45 deg, bottom clears the PD top
yb_front = ry0
prof = [(yb_front, z_pd_top + 0.05), (y_pd - (z_rx - z_pd_top - 0.05), z_pd_top + 0.05), (y_pd + 0.45, z_rx + 0.45),
        (yb_front, z_rx + 0.45)]
mb = MB().prism(prof, rx - 1.2, rx + 1.2, "yz")
obj_from_mb("rx_turning_prism", mb, M["glass"], "rx_optics", EX["rx_optics"])
mb = MB()
for s in (-1, 1):
    mb.box(rx + s * 1.2, rx + s * 1.45, yb_front + 0.1, y_pd - 0.3, ZT, z_rx + 0.3)   # prism side supports
obj_from_mb("rx_prism_supports", mb, M["ceramic"], "rx_optics", EX["rx_optics"])
occ(rx - 2.2, rx + 2.2, ry0, y_tia1 + 0.8)

# ---- glass relays ("optic nerves") from the eye prisms back to the chips, with 8 lane rays each
RL = INT["relay"]
y_prism_back = YF + 3.8
paths = {
    "tx": [(-EYE_DX, y_prism_back, RL["z_tx"]), (-EYE_DX, -78.0, RL["z_tx"]), (-2.4, -66.0, RL["z_tx"]),
           (-0.4, -56.0, RL["z_tx"]), (0.0, pf - 2.5, RL["z_tx"]), (0.0, pf, RL["z_tx"])],
    "rx": [(EYE_DX, y_prism_back, RL["z_rx"]), (EYE_DX, -78.0, RL["z_rx"]), (5.4, -66.0, RL["z_rx"]),
           (6.6, -56.0, RL["z_rx"]), (rx, yb_front - 2.5, RL["z_rx"]), (rx, yb_front, RL["z_rx"])],
}
for key, path in paths.items():
    cname = "tx_optics" if key == "tx" else "rx_optics"
    pts = catmull(path, 6)
    for (x, y, z) in pts:
        if y > INT["pcb_front_y"] - 1.0:
            occ(x - RL["w"] / 2, x + RL["w"] / 2, y - 1.0, y + 1.0)
    secs = [(x, y, z, RL["w"] / 2, RL["h"] / 2) for (x, y, z) in pts]
    mb = MB().loft(secs, npts=16, power=8.0)
    obj_from_mb("%s_glass_relay_optic_nerve" % key, mb, M["glass"], cname, EX["relay"],
                label="glass relay (optic nerve)")
    # supports on the bridge
    mbs = MB()
    for yy in (-80.0, -70.0, -62.0):
        # nearest path point
        q = min(pts, key=lambda P: abs(P[1] - yy))
        mbs.box(q[0] - 1.0, q[0] + 1.0, yy - 0.5, yy + 0.5, INT["bridge"]["z1"], q[2] - RL["h"] / 2)
    obj_from_mb("%s_relay_supports" % key, mbs, M["ceramic"], cname, EX["shell_bottom"])
    # eye prism (rearranger): 2x4 lens grid (front) <- 1x8 row (back)
    ex, ez = eye_c[key]
    zr = path[0][2]
    yfp = YF + 0.95
    front = [(ex - 1.6, yfp, ez - 0.85), (ex + 1.6, yfp, ez - 0.85), (ex + 1.6, yfp, ez + 0.85), (ex - 1.6, yfp, ez + 0.85)]
    back = [(ex - 1.2, y_prism_back, zr - 0.25), (ex + 1.2, y_prism_back, zr - 0.25), (ex + 1.2, y_prism_back, zr + 0.25),
            (ex - 1.2, y_prism_back, zr + 0.25)]
    mb = MB().hull8(front, back)
    obj_from_mb("eye_%s_prism_rearranger_1x8_to_2x4" % key, mb, M["glass"], "head_eyes", EX["head_eyes"],
                label="prism 1x8 -> 2x4")
    # rays: lane i, row pitch 0.25 at the chip/relay; lens index: lanes 1-4 top row, 5-8 bottom row
    ray_mat = M["ray_tx"] if key == "tx" else M["ray_rx"]
    mbr = MB()
    for i in range(8):
        u = (i - 3.5) * 0.25
        li = 4 + i if i < 4 else i - 4
        lx, lz = EYE_LENS[key][li]
        y_mid = (yfp + y_prism_back) / 2
        prism_part = [(lx, YF + 0.45, lz), (lx, yfp, lz), (lx, y_mid, lz), (path[0][0] + u, y_mid + 0.6, zr),
                      (path[0][0] + u, y_prism_back, zr)]
        rel = [(x + u, y, z) for (x, y, z) in pts]
        tail = []
        if key == "tx":
            tail = [(edge_x[i], pf + 0.05, zw)]
        else:
            tail = [(pd_x[i], y_pd, z_rx), (pd_x[i], y_pd, z_pd_top + 0.01)]
        mbr.tube(prism_part + rel[1:] + tail, 0.03, sides=5)
    obj_from_mb("%s_beam_rays_x8" % key, mbr, ray_mat, cname, EX["relay"])

# ---- DSP (brain): substrate, die, thermal pad; thermal boss on the lid underside
dd = INT["dsp"]
dy0, dy1 = dd["y0"], dd["y0"] + dd["ly"]
dcy = (dy0 + dy1) / 2
mb = MB().box(-dd["lx"] / 2, dd["lx"] / 2, dy0, dy1, ZT, ZT + dd["sub_t"])
obj_from_mb("dsp_package_substrate", mb, mat("dsp_substrate", (0.07, 0.09, 0.06), 0.0, 0.45), "dsp", EX["dsp"])
dx_, dyy, dz_ = dd["die"]
z_die1 = ZT + dd["sub_t"] + dz_
mb = MB().box(-dx_ / 2, dx_ / 2, dcy - dyy / 2, dcy + dyy / 2, ZT + dd["sub_t"], z_die1)
obj_from_mb("dsp_die_8x100g_pam4", mb, M["silicon"], "dsp", EX["dsp"], label="DSP (brain)")
mb = MB()
for i in range(16):                                                     # die-side capacitors on the substrate
    a = i / 16
    xx = -dd["lx"] / 2 + 0.6 + (dd["lx"] - 1.2) * a
    mb.box(xx - 0.25, xx + 0.25, dy0 + 0.35, dy0 + 0.6, ZT + dd["sub_t"], ZT + dd["sub_t"] + 0.25)
    mb.box(xx - 0.25, xx + 0.25, dy1 - 0.6, dy1 - 0.35, ZT + dd["sub_t"], ZT + dd["sub_t"] + 0.25)
obj_from_mb("dsp_substrate_caps", mb, M["mlcc"], "dsp", EX["dsp"])
tp_t = INT["thermal_pad_t"]
mb = MB().box(-dx_ / 2 - 0.8, dx_ / 2 + 0.8, dcy - dyy / 2 - 0.8, dcy + dyy / 2 + 0.8, z_die1, z_die1 + tp_t)
obj_from_mb("dsp_thermal_pad", mb, M["tpad"], "dsp", EX["tpad"], smooth=False, bevel=0.3, label="thermal pad")
mb = MB().box(-dx_ / 2 - 0.6, dx_ / 2 + 0.6, dcy - dyy / 2 - 0.6, dcy + dyy / 2 + 0.6, z_die1 + tp_t, ZHS0 + 0.01)
obj_from_mb("shell_top_thermal_boss", mb, M["shell_top"], "shell_top", EX["shell_top"], bevel=0.15)
occ(-dd["lx"] / 2, dd["lx"] / 2, dy0, dy1)
# RF traces DSP -> TX pads and RX TIA (copper under mask, slightly raised)
mb = MB()
for lx in lanes:
    mb.box(lx * 1.08 - 0.05, lx * 1.08 + 0.05, y_pcbpad + 0.25, dy0, ZT, ZT + 0.012)
for x in tia_in:
    mb.box(x - 0.05, x + 0.05, y_tia1 + 0.75, dy0 + 0.5, ZT, ZT + 0.012)
obj_from_mb("pcb_rf_traces", mb, mat("trace_under_mask", (0.05, 0.07, 0.06), 0.6, 0.3), "pcb", EX["pcb"])

# ---- electronics (power, management): positions in the research power zone x 66-89 (Y -23.8 .. -0.8)
def ic(name, cx, cy, lx, ly, h, material, label=None, ex_key="electronics"):
    occ(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2)
    m_ = MB().box(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, ZT, ZT + h)
    return obj_from_mb(name, m_, material, "electronics", EX[ex_key], bevel=0.04, label=label)


ic("elec_crystal_2016", 8.2, dcy + 2.5, 1.6, 2.0, 0.6, M["xtal"], "crystal")
ic("elec_mcu_ufqfpn48_7x7", 0.0, -9.5, 7.0, 7.0, 0.55, M["mold"], "MCU")
ic("elec_eeprom_dfn_2x3", -6.4, -9.5, 2.0, 3.0, 0.55, M["mold"], "EEPROM")
ic("elec_pmic_qfn_4x4", 1.6, -18.5, 4.0, 4.0, 0.8, M["mold"], "PMIC")
ic("elec_buck_mp28164_qfn_2x3", -3.0, -18.5, 2.0, 3.0, 0.8, M["mold"], "buck converter")
ic("elec_load_switch_dfn_2x3", 0.0, -2.6, 3.0, 2.0, 0.55, M["mold"], "load switch")
mb = MB()
for cx in (-6.4, 6.4):
    mb.cyl((cx, -18.5, ZT + 1.0), 2.0, 2.0, axis="z", segs=28)
    occ(cx - 2.0, cx + 2.0, -20.5, -16.5)
obj_from_mb("elec_power_inductors_x2", mb, M["inductor"], "electronics", EX["electronics"], smooth=True, label="inductors")
# IC pins / exposed pads (tin) for the QFN/DFN parts: thin strips along edges
mb = MB()
for (cx, cy, lx, ly) in ((0.0, -9.5, 7.0, 7.0), (1.6, -18.5, 4.0, 4.0), (-3.0, -18.5, 2.0, 3.0), (-6.4, -9.5, 2.0, 3.0),
                         (0.0, -2.6, 3.0, 2.0)):
    n_side = max(2, int(lx / 0.5))
    for k in range(n_side):
        xx = cx - lx / 2 + (k + 0.5) * lx / n_side
        mb.box(xx - 0.12, xx + 0.12, cy - ly / 2 - 0.15, cy - ly / 2 + 0.05, ZT, ZT + 0.06)
        mb.box(xx - 0.12, xx + 0.12, cy + ly / 2 - 0.05, cy + ly / 2 + 0.15, ZT, ZT + 0.06)
obj_from_mb("elec_ic_terminals", mb, M["tin"], "electronics", EX["electronics"])

# passives: seeded scatter (positions D; sizes A: EIA 0402 1.0 x 0.5 x 0.5, 0201 0.6 x 0.3 x 0.3)
rng = random.Random(1311)
mb_body, mb_term, mb_res = MB(), MB(), MB()
placed = 0
def free(x0, x1, y0, y1):
    for (a0, a1, b0, b1) in OCC:
        if not (x1 < a0 or x0 > a1 or y1 < b0 or y0 > b1):
            return False
    return True
specs = [(1.0, 0.5, 0.5, 60, "c"), (0.6, 0.3, 0.3, 50, "c"), (0.6, 0.3, 0.25, 24, "r")]
for (L_, W_, H_, cnt, kind) in specs:
    k = tries = 0
    while k < cnt and tries < cnt * 200:
        tries += 1
        cx = rng.uniform(-9.2, 9.2)
        cy = rng.uniform(INT["pcb_front_y"] + 0.8, -0.8)
        rot = rng.random() < 0.5
        lx, ly = (L_, W_) if rot else (W_, L_)
        if not free(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2):
            continue
        occ(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, pad=0.12)
        k += 1
        if kind == "r":
            mb_res.box(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, ZT, ZT + H_)
            continue
        if rot:
            mb_body.box(cx - lx / 2 + 0.15 * L_, cx + lx / 2 - 0.15 * L_, cy - ly / 2, cy + ly / 2, ZT, ZT + H_)
            for s in (-1, 1):
                mb_term.box(cx + s * lx / 2 - (0.15 * L_ if s > 0 else 0), cx + s * lx / 2 + (0.15 * L_ if s < 0 else 0),
                            cy - ly / 2, cy + ly / 2, ZT, ZT + H_)
        else:
            mb_body.box(cx - lx / 2, cx + lx / 2, cy - ly / 2 + 0.15 * L_, cy + ly / 2 - 0.15 * L_, ZT, ZT + H_)
            for s in (-1, 1):
                mb_term.box(cx - lx / 2, cx + lx / 2, cy + s * ly / 2 - (0.15 * L_ if s > 0 else 0),
                            cy + s * ly / 2 + (0.15 * L_ if s < 0 else 0), ZT, ZT + H_)
# bottom-side MLCCs (0402)
for k in range(22):
    cx = rng.uniform(-8.8, 8.8)
    cy = rng.uniform(-50.0, -3.0)
    mb_body.box(cx - 0.25, cx + 0.25, cy - 0.35, cy + 0.35, ZPCB0 - 0.5, ZPCB0)
    for s in (-1, 1):
        mb_term.box(cx - 0.25, cx + 0.25, cy + s * 0.35 - (0.15 if s > 0 else 0), cy + s * 0.35 + (0.15 if s < 0 else 0),
                    ZPCB0 - 0.5, ZPCB0)
obj_from_mb("elec_mlcc_bodies", mb_body, M["mlcc"], "electronics", EX["electronics"], label="MLCCs")
obj_from_mb("elec_mlcc_terminals", mb_term, M["tin"], "electronics", EX["electronics"])
obj_from_mb("elec_resistors_0201", mb_res, M["mold"], "electronics", EX["electronics"])

# ================================================================== finish: smoothing, export
bpy.context.view_layer.update()
# smooth curved surfaces, keep hard edges sharp (sets sharp edges + smooth faces, no modifier: glTF-clean)
meshes = [o for o in PARTS if o.type == "MESH"]
with bpy.context.temp_override(selected_editable_objects=meshes, selected_objects=meshes, active_object=meshes[0],
                               object=meshes[0]):
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(38), keep_sharp_edges=True)
tris = 0
deps = bpy.context.evaluated_depsgraph_get()
rows = []
for ob in PARTS:
    ev = ob.evaluated_get(deps)
    me = ev.to_mesh()
    t = sum(len(p.vertices) - 2 for p in me.polygons)
    ev.to_mesh_clear()
    tris += t
    rows.append(dict(name=ob.name, group=ob["group"], tris=t, explode_m=list(ob["explode"]), label=ob.get("label", "")))
os.makedirs(OUT_DIR, exist_ok=True)
blend = os.path.abspath(os.path.join(OUT_DIR, ASSET + ".blend"))
glb = os.path.abspath(os.path.join(OUT_DIR, ASSET + ".glb"))
bpy.context.preferences.filepaths.save_version = 0   # no .blend1 backups
bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=False, export_apply=True, export_extras=True,
                          export_yup=True, export_materials="EXPORT", export_cameras=False, export_lights=False)
meta = dict(asset=ASSET, triangles=tris, objects=len(PARTS),
            hooks=sorted(o.name for o in bpy.data.objects if o.name.startswith("HOOK_")),
            collections=list(COLL.keys()), parts=rows)
with open(os.path.join(OUT_DIR, ASSET + "_parts.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
print("HERO_BUILD_DONE tris=%d objects=%d blend=%s glb=%s" % (tris, len(PARTS), blend, glb))
