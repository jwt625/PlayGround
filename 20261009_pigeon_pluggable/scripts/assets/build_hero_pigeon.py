"""Build the hero asset: 800G OSFP DR8 as a cyberpunk surveillance rock pigeon (Blender 4.2, headless).

Run from the project folder (always through the Blender slot wrapper):
  scripts/bslot.sh -b --factory-startup --python scripts/assets/build_hero_pigeon.py -- assets/hero

Writes assets/hero/pigeon_osfp_dr8.blend, assets/hero/pigeon_osfp_dr8.glb, assets/hero/pigeon_osfp_dr8_parts.json.

Two stages (2026-10-09 correction, DevLog-003):
  1. Component stage: every electronic / optical part is built at true mm size in the module frame used by the
     module-layout v1 (origin on forward stop B, -Y front, +Z up). These are the models of
     outputs/hero/pigeon_osfp_dr8_exploded_module_layout.png, unchanged.
  2. Pigeon stage: rigid placements of those parts into a rock-pigeon anatomy (head = DSP brain + side eyes,
     chest = power organs on a sternum PCB, wings = heatsink fins, legs/feet = edge-connector lanes), plus the
     reshaped 2-piece skin and routed flex wiring. Pigeon frame: origin on the ground between the feet,
     -Y = beak, +Z up, -X = pigeon's RIGHT = TX (red eye, right foot); +X = LEFT = RX (cyan).
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
TAB_W = 6.0          # D leash strap width (pigeon scale)

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
              "tx_optics", "rx_optics", "electronics", "feet", "wiring"):
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
    fins=mat("fin_metal", (0.34, 0.35, 0.38), 1.0, 0.30),
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
    beak=mat("beak_dark_chrome", (0.28, 0.28, 0.31), 1.0, 0.28, coat=0.8),
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
    skin_top=mat("skin_gunmetal_back", (0.10, 0.105, 0.115), 1.0, 0.36, coat=0.35),
    skin_bot=mat("skin_dark_polymer_belly", (0.035, 0.037, 0.042), 0.0, 0.55, coat=0.2),
    skin_neck=mat("skin_neck_iridescent", (0.05, 0.16, 0.12), 0.9, 0.25, coat=0.6),
    seam=mat("panel_seam_black", (0.008, 0.008, 0.01), 0.3, 0.5),
    polymer=mat("polymer_dark", (0.03, 0.03, 0.035), 0.0, 0.6),
    band=mat("leg_band_orange", (0.95, 0.35, 0.03), 0.0, 0.45),
    ink_white=mat("ink_white", (0.92, 0.92, 0.9), 0.0, 0.5),
    flex=mat("polyimide_flex_amber", (0.62, 0.30, 0.04), 0.0, 0.32, coat=0.5),
    covert=mat("covert_brushed_gunmetal", (0.22, 0.23, 0.26), 0.85, 0.36),
    stiffener=mat("fr4_stiffener_dark", (0.06, 0.08, 0.06), 0.0, 0.5),
    connector=mat("connector_lcp_black", (0.02, 0.02, 0.025), 0.0, 0.45),
    skin_inner=mat("bead_blasted_alu", (0.42, 0.43, 0.45), 0.9, 0.62),
    coverlay=mat("coverlay_amber", (0.72, 0.42, 0.08), 0.0, 0.25, alpha=0.55),
)
_b = M["skin_neck"].node_tree.nodes["Principled BSDF"]
if "Thin Film Thickness" in _b.inputs:                 # iridescent neck (Blender 4.2 thin-film interference)
    _b.inputs["Thin Film Thickness"].default_value = 520.0
    _b.inputs["Thin Film IOR"].default_value = 1.6

# ---- film v2 hero materials (DevLog-010 section 9): hero signature + keyable channels for the scene. Names are an
# interface for the film scene (documented in DevLog-003 section 16); change them only together with the scene.
HM = dict(
    # (1) signature: glowing amber EEPROM leg band; iridescent neck with a Fresnel rim accent (crowd proxies have neither)
    band=mat("hero_leg_band_glow_amber", (1.0, 0.36, 0.02), 0.0, 0.4, emit=1.8, ecol=(1.0, 0.36, 0.02)),
    neck=mat("hero_neck_rim_accent", (0.05, 0.16, 0.12), 0.9, 0.25, coat=0.6),
    # (2) keyable emission (key "Emission Strength"; colours preset): right = TX = red, left = RX = cyan
    pads_r=mat("hero_foot_pads_R_tx", (1.0, 0.72, 0.30), 1.0, 0.18, emit=0.0001, ecol=(1.0, 0.05, 0.03)),
    pads_l=mat("hero_foot_pads_L_rx", (1.0, 0.72, 0.30), 1.0, 0.18, emit=0.0001, ecol=(0.03, 0.75, 1.0)),
    rim_r=mat("hero_eye_rim_R_tx", (0.6, 0.02, 0.02), 0.0, 0.5, emit=3.0),
    rim_l=mat("hero_eye_rim_L_rx", (0.02, 0.45, 0.6), 0.0, 0.5, emit=3.0),
    # (5) wing heatsink panels: same look as fin_metal / covert, own materials so the scene can key "Alpha" (dithered)
    wing_fins=mat("hero_wing_fins_fadeable", (0.34, 0.35, 0.38), 1.0, 0.30),
    wing_covert=mat("hero_wing_covert_fadeable", (0.22, 0.23, 0.26), 0.85, 0.36),
    # (4) pull tab: woven nylon webbing (yellow 107U-like), sheen + weave bump instead of a flat plastic slab
    tab=mat("hero_pull_tab_webbing", (0.80, 0.50, 0.025), 0.0, 0.78),
)
for _m in (HM["pads_r"], HM["pads_l"]):
    _m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.0
for _m in (HM["wing_fins"], HM["wing_covert"]):
    _m.surface_render_method = "DITHERED"             # alpha < 1 renders as a stochastic fade in EEVEE Next
_b = HM["neck"].node_tree.nodes["Principled BSDF"]
if "Thin Film Thickness" in _b.inputs:
    _b.inputs["Thin Film Thickness"].default_value = 520.0
    _b.inputs["Thin Film IOR"].default_value = 1.6
_nt = HM["neck"].node_tree
_lw = _nt.nodes.new("ShaderNodeLayerWeight")
_lw.inputs["Blend"].default_value = 0.25
_mm = _nt.nodes.new("ShaderNodeMath")
_mm.operation = "MULTIPLY"
_mm.name = _mm.label = "rim_strength"
_mm.inputs[1].default_value = 0.8                    # key this input to brighten / dim the rim accent
_nt.links.new(_lw.outputs["Fresnel"], _mm.inputs[0])
_nt.links.new(_mm.outputs["Value"], _b.inputs["Emission Strength"])
_b.inputs["Emission Color"].default_value = (0.25, 0.95, 0.75, 1)
_b = HM["tab"].node_tree.nodes["Principled BSDF"]
_b.inputs["Sheen Weight"].default_value = 0.3
_b.inputs["Sheen Tint"].default_value = (1.0, 0.95, 0.7, 1)
_nt = HM["tab"].node_tree
_tc = _nt.nodes.new("ShaderNodeTexCoord")
_wv = _nt.nodes.new("ShaderNodeTexWave")
_wv.wave_type = "BANDS"
_wv.inputs["Scale"].default_value = 2500.0          # object coords in metres: ~0.4 mm weave period
_wv.inputs["Distortion"].default_value = 1.5
_bp = _nt.nodes.new("ShaderNodeBump")
_bp.inputs["Strength"].default_value = 0.25
_bp.inputs["Distance"].default_value = 0.00005
_nt.links.new(_tc.outputs["Object"], _wv.inputs["Vector"])
_nt.links.new(_wv.outputs["Fac"], _bp.inputs["Height"])
_nt.links.new(_bp.outputs["Normal"], _b.inputs["Normal"])

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


# ---- decals (v2 polish): textured planes for silkscreen / via fields / laser markings. The build writes the decal
# layout to assets/hero/textures/decal_layout.json; scripts/assets/make_hero_textures.py draws the PNGs from it;
# the next build maps them (planes are skipped while a texture is missing).
TEXDIR = os.path.abspath(os.path.join(OUT_DIR, "textures"))
DECAL_LAYOUT = {}


def make_decal(name, origin, u_axis, v_axis, size, layout, cname, explode=(0, 0, 0), lift=0.0):
    """Plane spanning origin + u*size[0] + v*size[1] (mm, world or module frame); normal = u x v (the readable side)."""
    DECAL_LAYOUT[name] = dict(size_mm=list(size), **layout)
    path = os.path.join(TEXDIR, name + ".png")
    if not os.path.exists(path):
        return None
    o, u, v = Vector(origin), Vector(u_axis).normalized(), Vector(v_axis).normalized()
    n = u.cross(v).normalized()
    o = o + n * lift
    corners = [o, o + u * size[0], o + u * size[0] + v * size[1], o + v * size[1]]
    c = sum(corners, Vector()) / 4
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple((p - c) * MM) for p in corners], [], [[0, 1, 2, 3]])
    uvl = me.uv_layers.new(name="UVMap")
    for li, uvv in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        uvl.data[li].uv = uvv
    m = bpy.data.materials.new("decal_" + name)
    m.use_nodes = True
    nt_ = m.node_tree
    b = nt_.nodes["Principled BSDF"]
    tex = nt_.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Linear"
    nt_.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    nt_.links.new(tex.outputs["Alpha"], b.inputs["Alpha"])
    b.inputs["Roughness"].default_value = 0.45
    try:
        m.blend_method = "CLIP"
        m.alpha_threshold = 0.5
    except Exception:
        pass
    m.surface_render_method = "DITHERED"
    me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    COLL[cname].objects.link(ob)
    ob.parent = ROOT
    ob.location = tuple(c * MM)
    ob["explode"] = [e * MM for e in explode]
    ob["group"] = cname
    PARTS.append(ob)
    return ob


# explode offsets per group (mm); tuned for the exploded render
EX = dict(
    shell_top=(0, 12, 36), fins=(0, 16, 52), shell_bottom=(0, 6, -26), head_eyes=(0, -24, 4), beak=(0, -36, -1),
    leash=(0, -14, -24), pcb=(0, 0, 0), dsp=(0, 0, 14), tpad=(0, 0, 23), tx_optics=(0, 0, 10), dfb=(0, 3, 20),
    rx_optics=(3, 0, 10), electronics=(0, 0, 7), feet=(0, 9, 0), ruff=(0, -4, -26), relay=(0, -10, 10),
)

# ================================================================== EYES (2x4 lenslet arrays, iris rings, prism rearrangers)
HOOKS = {}
eye_objs = {}
EYE_LENS = {}
eye_c = {"tx": (-EYE_DX, EYE_Z), "rx": (EYE_DX, EYE_Z)}    # module-frame eye centres (component stage)
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
# v2 polish (photos 15.1): nickel lid with chamfered corners over the die (Cisco 400G style), laser-marked, then a
# torn grey thermal-putty blob (Innolight 400G/800G style). Lid 0.2 mm above the die; putty 0.8 mm so the stack top
# stays at die top + 1.0 (the copper spreader and strap are unchanged).
LID_W, LID_L, LID_C = 11.6, 12.2, 1.0
z_lid0, z_lid1 = ZT + dd["sub_t"], z_die1 + 0.2
lid_poly = [(-LID_W / 2 + LID_C, dcy - LID_L / 2), (LID_W / 2 - LID_C, dcy - LID_L / 2), (LID_W / 2, dcy - LID_L / 2 + LID_C),
            (LID_W / 2, dcy + LID_L / 2 - LID_C), (LID_W / 2 - LID_C, dcy + LID_L / 2), (-LID_W / 2 + LID_C, dcy + LID_L / 2),
            (-LID_W / 2, dcy + LID_L / 2 - LID_C), (-LID_W / 2, dcy - LID_L / 2 + LID_C)]
mb = MB().prism(lid_poly, z_lid0, z_lid1, "xy")
obj_from_mb("dsp_lid_nickel", mb, M["dsp_lid"], "dsp", EX["dsp"], bevel=0.08, label="DSP lid")
DSP_LID_DECAL = make_decal("dsp_lid_marking", (-LID_W / 2 + 0.6, dcy - LID_L / 2 + 0.6, z_lid1), (1, 0, 0), (0, 1, 0),
                           (LID_W - 1.2, LID_L - 1.2), dict(kind="marking", bg=None, ink=[30, 30, 32],
                           lines=["NARROWCOM", "NC-0800D  PAM4", "2541  TWN", "KD9X7 01"], pin1=True), "dsp", EX["dsp"],
                           lift=0.004)
# film v2 (3): switchable parody brand on the DSP lid. Two materials with the same node setup, different stamp textures:
# hero_dsp_mark_NARROWCOM (assigned) and hero_dsp_mark_MERVALL (fake user, swap ob.data.materials[0] in the scene).
DECAL_LAYOUT["dsp_lid_marking_mervall"] = dict(size_mm=[LID_W - 1.2, LID_L - 1.2], kind="marking", bg=None, ink=[30, 30, 32],
                                               lines=["MERVALL", "MV-0800D  PAM4", "2541  TWN", "KD9X7 01"], pin1=True)
DSP_MARK_MATS = {}
if DSP_LID_DECAL is not None:
    _m0 = DSP_LID_DECAL.data.materials[0]
    _m0.name = "hero_dsp_mark_NARROWCOM"
    DSP_MARK_MATS["NARROWCOM"] = _m0
    _pm = os.path.join(TEXDIR, "dsp_lid_marking_mervall.png")
    if os.path.exists(_pm):
        _m1 = _m0.copy()
        _m1.name = "hero_dsp_mark_MERVALL"
        for _n in _m1.node_tree.nodes:
            if _n.type == "TEX_IMAGE":
                _n.image = bpy.data.images.load(_pm)
        _m1.use_fake_user = True
        DSP_MARK_MATS["MERVALL"] = _m1
    print("DSP_MARKS", sorted(m_.name for m_ in DSP_MARK_MATS.values()))
import random as _rnd
_r = _rnd.Random(87)
mbp = MB()
mbp.ellipsoid((0.25, dcy + 0.3, z_lid1), 4.9, 5.4, tp_t - 0.2, segs=28, rings=8, hemi="+z")
mbp.ellipsoid((-3.6, dcy - 3.9, z_lid1), 1.7, 1.3, 0.35, segs=14, rings=5, hemi="+z")          # torn smear at the edge
mbp.ellipsoid((3.9, dcy + 4.4, z_lid1), 1.2, 1.6, 0.30, segs=14, rings=5, hemi="+z")
vv = []
for (x, y, z) in mbp.v:
    k = (z - z_lid1) / max(tp_t - 0.2, 1e-6)
    j = 0.10 + 0.25 * k
    vv.append((x + _r.uniform(-j, j), y + _r.uniform(-j, j), max(z_lid1, z + _r.uniform(-0.06, 0.06) * k)))
mbp.v = vv
obj_from_mb("dsp_thermal_pad", mbp, M["tpad"], "dsp", EX["tpad"], smooth=True, label="thermal putty")

# ---- electronics (power, management): same part models as the module layout v1, one object per part
ICS = {}


def ic(name, cx, cy, lx, ly, h, material, label=None):
    m_ = MB().box(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, ZT, ZT + h)
    body = obj_from_mb(name, m_, material, "electronics", (0, 0, 0), bevel=0.04, label=label)
    mt = MB()
    n_side = max(2, int(lx / 0.5))
    for k in range(n_side):
        xx = cx - lx / 2 + (k + 0.5) * lx / n_side
        mt.box(xx - 0.12, xx + 0.12, cy - ly / 2 - 0.15, cy - ly / 2 + 0.05, ZT, ZT + 0.06)
        mt.box(xx - 0.12, xx + 0.12, cy + ly / 2 - 0.05, cy + ly / 2 + 0.15, ZT, ZT + 0.06)
    term = obj_from_mb(name + "_terminals", mt, M["tin"], "electronics", (0, 0, 0))
    parts = [body, term]
    dec = make_decal("mark_" + name, (cx - lx / 2 + 0.12, cy - ly / 2 + 0.12, ZT + h), (1, 0, 0), (0, 1, 0),
                     (lx - 0.24, ly - 0.24), dict(kind="marking", bg=None, ink=[150, 150, 150], lines=MARKINGS.get(name, [])),
                     "electronics", lift=0.004)
    if dec is not None:
        parts.append(dec)
    ICS[name] = ((cx, cy, ZT), parts)
    return body


# laser-marked top text (generic / parody only; no real vendor marks)
MARKINGS = {
    "elec_crystal_2016": ["156.25", "2532"],
    "elec_mcu_ufqfpn48_7x7": ["MCU-L4", "QFN48", "2538 C"],
    "elec_eeprom_dfn_2x3": ["E2K", "2541"],
    "elec_pmic_qfn_4x4": ["PM3V3", "2537A"],
    "elec_buck_mp28164_qfn_2x3": ["BK28", "2536"],
    "elec_load_switch_dfn_2x3": ["LS10", "2539"],
}


ic("elec_crystal_2016", 0.0, 0.0, 1.6, 2.0, 0.6, M["xtal"], "crystal")
ic("elec_mcu_ufqfpn48_7x7", 0.0, 0.0, 7.0, 7.0, 0.55, M["mold"], "MCU")
ic("elec_eeprom_dfn_2x3", 0.0, 0.0, 2.0, 3.0, 0.55, M["mold"], "EEPROM")
ic("elec_pmic_qfn_4x4", 0.0, 0.0, 4.0, 4.0, 0.8, M["mold"], "PMIC")
ic("elec_buck_mp28164_qfn_2x3", 0.0, 0.0, 2.0, 3.0, 0.8, M["mold"], "buck converter")
ic("elec_load_switch_dfn_2x3", 0.0, 0.0, 3.0, 2.0, 0.55, M["mold"], "load switch")
for nm in ("elec_power_inductor_L1", "elec_power_inductor_L2"):
    mb = MB().cyl((0.0, 0.0, ZT + 1.0), 2.0, 2.0, axis="z", segs=28)
    ICS[nm] = ((0.0, 0.0, ZT), [obj_from_mb(nm, mb, M["inductor"], "electronics", (0, 0, 0), smooth=True,
                                            label="power inductor")])

# ---- carriers for the side-mounted optics (module coords; they move rigidly with the TX / RX groups)
mb = MB().box(-4.6, 4.6, pf - 0.3, y_pcbpad + 0.45, ZPCB0, ZPCB1)
tx_carrier = obj_from_mb("pcb_tx_pic_carrier", mb, M["pcb"], "pcb", (0, 0, 0), bevel=0.05, label="TX carrier")
mb = MB().box(rx - 2.6, rx + 2.6, ry0 - 0.3, y_tia1 + 1.0, ZPCB0, ZPCB1)
rx_carrier = obj_from_mb("pcb_rx_carrier", mb, M["pcb"], "pcb", (0, 0, 0), bevel=0.05, label="RX carrier")
# carrier silkscreen / via decals (module frame, move with the TX / RX groups)
_txL = (y_pcbpad + 0.45) - (pf - 0.3)
TXC_DECAL = make_decal("silk_tx_carrier", (-4.6, pf - 0.3, ZPCB1), (1, 0, 0), (0, 1, 0), (9.2, _txL), dict(
    kind="pcb", outline=[[0, 0], [9.2, 0], [9.2, _txL], [0, _txL]],
    vias=[[0.15, 0.3, 0.75, _txL - 0.3], [8.45, 0.3, 9.05, _txL - 0.3]], fiducials=[[0.45, 0.45], [8.75, _txL - 0.45]],
    texts=[["TXC-8", 0.45, _txL - 1.2, 0.35, 90]]), "pcb", lift=0.004)
_rxL = (y_tia1 + 1.0) - (ry0 - 0.3)
RXC_DECAL = make_decal("silk_rx_carrier", (rx - 2.6, ry0 - 0.3, ZPCB1), (1, 0, 0), (0, 1, 0), (5.2, _rxL), dict(
    kind="pcb", outline=[[0, 0], [5.2, 0], [5.2, _rxL], [0, _rxL]],
    vias=[[0.1, 0.2, 0.5, _rxL - 0.2], [4.7, 0.2, 5.1, _rxL - 0.2]], fiducials=[[0.3, 0.3], [4.9, _rxL - 0.3]], texts=[]),
    "pcb", lift=0.004)
# TIA flip-chip bumps (visible at the die edge) and DSP BGA balls (module coords, move with their groups)
mb = MB()
for i in range(16):
    x = rx - tia["lx"] / 2 + 0.15 + (tia["lx"] - 0.3) * i / 15
    for yy in (y_tia0 + 0.12, y_tia1 - 0.12):
        mb.ellipsoid((x, yy, ZT + 0.01), 0.045, 0.045, 0.03, segs=8, rings=4)
tia_bumps = obj_from_mb("rx_tia_flipchip_bumps", mb, M["tin"], "rx_optics", (0, 0, 0))
BGA_GAP = 0.25
mb = MB()
pitch_b = 0.8
nx_b, ny_b = int(dd["lx"] / pitch_b), int(dd["ly"] / pitch_b)
for i in range(nx_b):
    for j in range(ny_b):
        if 2 < i < nx_b - 3 and 2 < j < ny_b - 3:
            continue                                                   # perimeter rows only (interior hidden)
        x = -dd["lx"] / 2 + pitch_b * (i + 0.5) + (dd["lx"] - nx_b * pitch_b) / 2
        y = dy0 + pitch_b * (j + 0.5) + (dd["ly"] - ny_b * pitch_b) / 2
        mb.ellipsoid((x, y, ZT - BGA_GAP / 2), 0.2, 0.2, BGA_GAP / 2 + 0.02, segs=8, rings=4)
dsp_bga = obj_from_mb("dsp_bga_balls", mb, M["tin"], "dsp", (0, 0, 0))

# ================================================================== PIGEON ASSEMBLY (2026-10-09 correction)
# All components above keep their true mm size; only rigid placements change. Pigeon frame: origin on the ground
# between the feet, -Y = beak, +Z up, -X = pigeon's right = TX. Proportions from the Commons side photo
# (references/figures/commons_feral_pigeon_columba_livia_side.jpg), scaled to 124 mm beak-to-tail (grade C/D).
from mathutils import Matrix

bpy.context.view_layer.update()
R_ID = Matrix.Identity(3)
R_TX = Matrix(((0, 0, -1), (0, 1, 0), (1, 0, 0)))     # module +Z -> -X (outward right), +X -> +Z, Y kept
R_RX = Matrix(((0, 0, 1), (0, 1, 0), (-1, 0, 0)))     # module +Z -> +X (outward left), +X -> -Z, Y kept
R_FLIP = Matrix(((-1, 0, 0), (0, 1, 0), (0, 0, -1)))  # component on a board underside


def T_of(R, P0, Q):
    return Matrix.Translation(Vector(Q) * MM) @ R.to_4x4() @ Matrix.Translation(-Vector(P0) * MM)


def place(objs, R, P0, Q):
    T = T_of(R, P0, Q)
    for o in objs:
        o.matrix_world = T @ o.matrix_world
    bpy.context.view_layer.update()
    return T


def tp(T, p):
    """Transform a module-frame point (mm) to the pigeon frame (mm)."""
    v = T @ (Vector(p) * MM)
    return tuple(c / MM for c in v)


def objs(*names):
    return [bpy.data.objects[n] for n in names]


def setex(obs, ex):
    for o in obs:
        o["explode"] = [e * MM for e in ex]


# ---- key anatomy points (mm)
HC = Vector((0.0, -44.0, 81.5))        # head centre (photo: head centre ~ (Y -45, Z 86) at this scale)
HA = (12.0, 14.0, 12.0)                # head semi-axes (sized to enclose DSP + side optics; photo head r ~ 10.7)
# eyes: v2 (2026-10-10) moved from (HC.y-3.5, HC.z-1.0) to (HC.y-5.0, HC.z) so the delivered T / X subassemblies fit inside
# the inner skin (scripts/assets/fit_eye_subassy.py grid search with the 2026-10-10 02:21 TX (carrier top at sub X -1.75):
# worst ellipsoid level TX 0.974 / RX 0.710 with 0.25 mm clearance; yaw 0 = optical axis along +-X; aperture recessed 1.0 mm
# inside the skin mid-surface in an eye socket; the eye-cere ring covers the RX carrier front edge)
E_Y, E_Z = HC.y - 5.0, HC.z
EYE_OFF = -1.0
X_S = HA[0] * math.sqrt(1 - ((E_Y - HC.y) / HA[1]) ** 2 - ((E_Z - HC.z) / HA[2]) ** 2)
XW_TX, XR_RX = -7.45, 8.0              # TX PIC waveguide plane / RX turning-prism plane
Y_PIC_F = E_Y + 1.8                    # TX PIC front edge
Y_RX_F = E_Y + 2.0                     # RX turning prism front face
ZB0 = HC.z + 4.9                       # head board bottom (1.0 thick), 1.3 mm above the side carriers

# ---- head: TX optics group on the right side of the head, PIC face outward, lasers deeper (behind the PIC)
tx_names = ["tx_pic_siph_die", "tx_pic_waveguides_splitters", "tx_mzm_x8_electrodes", "tx_monitor_pds_x8",
            "tx_dfb_spacers", "tx_dfb_submounts_aln", "tx_dfb_laser_dies_x2", "tx_dfb_facet_glow", "tx_dfb_wirebonds",
            "tx_pic_wirebonds", "pcb_tx_bond_pads", "pcb_tx_pic_carrier"] + (["silk_tx_carrier"] if TXC_DECAL else [])
T_TX = place(objs(*tx_names), R_TX, (p["cx"], pf, zt), (XW_TX, Y_PIC_F, E_Z))
rx_names = ["rx_pd_submount", "rx_pin_pd_array_1x8", "rx_pd_active_areas", "rx_octal_tia", "rx_bond_pads", "rx_wirebonds",
            "rx_turning_prism", "rx_prism_supports", "pcb_rx_carrier", "rx_tia_flipchip_bumps"] + (["silk_rx_carrier"] if RXC_DECAL else [])
T_RX = place(objs(*rx_names), R_RX, (rx, ry0, z_rx), (XR_RX, Y_RX_F, E_Z))
# eyes (lenslet arrays face sideways like a pigeon's lateral eyes)
for key, s, ang in (("tx", -1, -90.0), ("rx", 1, 90.0)):
    names = ["eye_%s_iris_ring" % key, "eye_%s_bezel" % key, "eye_%s_iris_glow_ring" % key,
             "eye_%s_lenslet_substrate" % key, "eye_%s_lenslet_array_2x4" % key]
    T_E = place(objs(*names), Matrix.Rotation(math.radians(ang), 3, "Z"), (s * EYE_DX, YF, EYE_Z), (s * (X_S + 1.0), E_Y, E_Z))
    EYE_LENS[key] = [tp(T_E, (lx, YF + 0.55, lz)) for (lx, lz) in EYE_LENS[key]]
# head board (rigid PCB under the brain), DSP on top, MCU (hypothalamus) underneath
mb = MB().prism([(-5.8, HC.y - 8.7), (5.8, HC.y - 8.7), (6.8, HC.y - 7.7), (6.8, HC.y + 7.0), (5.0, HC.y + 8.0), (5.0, HC.y + 10.2),
                  (-5.0, HC.y + 10.2), (-5.0, HC.y + 8.0), (-6.8, HC.y + 7.0), (-6.8, HC.y - 7.7)], ZB0, ZB0 + 1.0, "xy")
head_board = obj_from_mb("pcb_head_board", mb, M["pcb"], "pcb", (0, 0, 0), bevel=0.08, label="head board")
dsp_names = ["dsp_package_substrate", "dsp_die_8x100g_pam4", "dsp_substrate_caps", "dsp_thermal_pad", "dsp_bga_balls",
             "dsp_lid_nickel"] + (["dsp_lid_marking"] if DSP_LID_DECAL else [])
place(objs(*dsp_names), R_ID, (0.0, dcy, ZT), (0.0, HC.y + 0.5, ZB0 + 1.0 + BGA_GAP))
MCU_C = (-1.3, HC.y + 1.5)             # v2: shifted to clear the RX flex connector J2 and the TX DC connector J6
place(ICS["elec_mcu_ufqfpn48_7x7"][1], R_FLIP, ICS["elec_mcu_ufqfpn48_7x7"][0], (MCU_C[0], MCU_C[1], ZB0))
# audit fix 2: DSP reference crystal on the head-board underside beside the MCU, short REFCLK route up to the DSP BGA
place(ICS["elec_crystal_2016"][1], R_FLIP, ICS["elec_crystal_2016"][0], (4.9, HC.y + 0.5, ZB0))
setex(ICS["elec_crystal_2016"][1], (0, -6, 8))
mb = MB().box(3.2, 4.15, HC.y + 0.43, HC.y + 0.57, ZB0 - 0.02, ZB0)
mb.cyl((3.2, HC.y + 0.5, ZB0 - 0.015), 0.12, 0.03, axis="z", segs=10)
obj_from_mb("pcb_head_refclk_trace_via", mb, M["gold"], "pcb", (0, -6, 8), label="REFCLK")
Z_PAD_TOP = ZB0 + 1.0 + BGA_GAP + dd["sub_t"] + dd["die"][2] + INT["thermal_pad_t"]

# ---- new eye prisms (1x8 chip row <-> 2x4 lenslet grid, 45 deg fold) and lane rays
x_lens_back = X_S + 1.0 - 0.95
for key in ("tx", "rx"):
    s = -1 if key == "tx" else 1
    e = E_Y
    if key == "tx":
        poly = [(-x_lens_back, e - 1.6), (-x_lens_back, e + 1.6), (-7.95, e + 1.6), (-7.95, Y_PIC_F), (-7.0, Y_PIC_F),
                (-7.0, e + 0.4), (-9.0, e - 1.6)]
    else:
        poly = [(x_lens_back, e - 1.6), (x_lens_back, e + 1.6), (XR_RX + 1.2, e + 1.6), (XR_RX + 1.2, Y_RX_F),
                (XR_RX - 1.0, Y_RX_F), (XR_RX - 1.0, e + 0.4), (XR_RX + 0.8, e - 1.6)]   # clears the RX carrier
    mb = MB().prism(poly, E_Z - 1.1, E_Z + 1.1, "xy")
    obj_from_mb("eye_%s_prism_rearranger_1x8_to_2x4" % key, mb, M["glass"], "head_eyes", (0, 0, 0),
                label="prism 1x8 -> 2x4")
    mbr = MB()
    for i in range(8):
        li = 4 + i if i < 4 else i - 4
        lx3, ly3, lz3 = EYE_LENS[key][li]
        x_out = s * (X_S + 1.0 + 0.6)
        if key == "tx":
            p0 = tp(T_TX, (edge_x[i], pf + 0.05, zw))
            pts = [p0, (p0[0], e + 0.9, p0[2]), (p0[0] - 0.8, e - 0.1, (p0[2] + lz3) / 2), (-x_lens_back, ly3, lz3),
                   (x_out, ly3, lz3)]
        else:
            q0 = tp(T_RX, (pd_x[i], ry0, z_rx))
            q1 = tp(T_RX, (pd_x[i], y_pd, z_rx))
            q2 = tp(T_RX, (pd_x[i], y_pd, z_pd_top + 0.01))
            pts = [(x_out, ly3, lz3), (x_lens_back, ly3, lz3), (q0[0] + 0.8, e - 0.1, (q0[2] + lz3) / 2),
                   (q0[0], e + 0.9, q0[2]), q0, q1, q2]
        mbr.tube(pts, 0.03, sides=5)
    obj_from_mb("%s_beam_rays_x8" % key, mbr, M["ray_tx"] if key == "tx" else M["ray_rx"],
                "tx_optics" if key == "tx" else "rx_optics", (0, 0, 0))

# ---- T / X subassembly integration (DevLog-005 section 3). Interface frame: origin at the centre of the eye's 2x4
# lenslet outer surface, +X out of the eye, +Z up. When assets/components/optics_tx|rx/<name>.blend exists, its
# objects replace the hero's own eye + optics group for that side; hooks are renamed HOOK_<side>_<hook>.
COMP_DIR = os.path.abspath(os.path.join(OUT_DIR, "..", "components"))
SUBASSY = {}
SUB_K = 0.7                     # fraction of the subassembly's own explode vectors used in the hero exploded view
SUB_GROUP_EX = {"tx": (-12.0, -6.0, 4.0), "rx": (12.0, -6.0, 4.0)}   # whole optics group (mm, pigeon frame)


def eye_frame(side_key):
    """Aperture point (mm) and rotation sub -> pigeon: sub +X = out of the eye (horizontal), +Z up."""
    s_ = -1 if side_key == "tx" else 1
    A_ = Vector((s_ * (X_S + EYE_OFF), E_Y, E_Z))
    R_ = Matrix.Rotation(math.pi, 3, "Z") if side_key == "tx" else Matrix.Identity(3)   # T: sub +Y -> pigeon -Y
    return A_, R_


def sub_pt(side_key, p_sub):
    A_, R_ = eye_frame(side_key)
    return tuple(A_ + R_ @ Vector(p_sub))


def import_subassembly(side_key):
    folder = os.path.join(COMP_DIR, "optics_" + side_key)
    path = os.path.join(folder, "optics_%s.blend" % side_key)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    new = [o for o in dst.objects if o is not None and o.type not in ("CAMERA", "LIGHT")]
    for im in bpy.data.images:                       # make library-relative texture paths absolute, then pack
        if im.filepath.startswith("//") and im.packed_file is None and im.library is None:
            cand = os.path.join(folder, im.filepath[2:])
            if os.path.isfile(cand):
                im.filepath = cand
                im.reload()
                im.pack()
    A_, R_ = eye_frame(side_key)
    T_ = Matrix.Translation(A_ * MM) @ R_.to_4x4()
    cname = "tx_optics" if side_key == "tx" else "rx_optics"
    G_ = Vector(SUB_GROUP_EX[side_key]) * MM
    hooks_, meshes_ = {}, []
    for o in new:
        COLL[cname].objects.link(o)
    for o in new:
        if o.parent is None or o.parent not in new:
            mw = o.matrix_world.copy()
            o.parent = ROOT
            o.matrix_world = T_ @ mw
            o.name = "%s_optics_root" % side_key
            continue
        if o.name.startswith("HOOK_"):
            base = o.name.split(".")[0][5:]
            o.name = "HOOK_%s_%s" % (side_key, base)
            hooks_[base] = o
            for k_ in ("explode", "explode_gltf"):
                if k_ in o:
                    del o[k_]
            continue
        if not o.name.startswith(side_key + "_"):
            o.name = "%s_%s" % (side_key, o.name)
        if o.type != "MESH":
            continue
        ex_sub = Vector(o.get("explode", (0.0, 0.0, 0.0)))
        if "explode_gltf" in o:
            del o["explode_gltf"]                    # sub-frame copy; the hero explode below replaces it
        viz = ("beam" in o.name) or o.name.startswith(side_key + "_viz_")
        o["explode"] = list(G_ + (Vector((0, 0, 0)) if viz else R_ @ ex_sub * SUB_K))
        o["group"] = cname
        o["subassy"] = side_key
        if viz:
            o["viz_only"] = 1                         # 1311 nm beams are invisible: hidden in hero renders
            o.hide_render = True
        PARTS.append(o)
        meshes_.append(o)
    bpy.context.view_layer.update()
    # flatten: every object becomes a direct child of ROOT (world transform kept), so the renderers' explode offsets
    # (added to .location in parent space) act in the pigeon frame, as for all hero parts
    for o in new:
        if o.parent is not ROOT:
            mw = o.matrix_world.copy()
            o.parent = ROOT
            o.matrix_world = mw
    bpy.context.view_layer.update()
    print("SUBASSY %s: %d objects (%d meshes) from %s, hooks %s" % (side_key, len(new), len(meshes_),
                                                                     os.path.relpath(path, COMP_DIR), sorted(hooks_)))
    return dict(objects=new, meshes=meshes_, hooks=hooks_, path=path)


def remove_objects(names):
    for nm in names:
        o = bpy.data.objects.get(nm)
        if o is None:
            continue
        for ch in list(o.children):
            ch.parent = ROOT
        if o in PARTS:
            PARTS.remove(o)
        bpy.data.objects.remove(o, do_unlink=True)


def ring_on_axis(mb_, side_key, r_in, r_out, x0, x1, segs=64):
    """Annulus around the eye axis between sub-frame X x0..x1 (mm), appended to mb_ in pigeon coordinates."""
    m_ = MB().ring((0.0, (x0 + x1) / 2, 0.0), r_in, r_out, x1 - x0, axis="y", segs=segs)
    off = len(mb_.v)
    for (p_, y_, q_) in m_.v:                          # local (p, y, q) -> sub (y, p, q)
        mb_.v.append(sub_pt(side_key, (y_, p_, q_)))
    mb_.f.extend([[i + off for i in f_] for f_ in m_.f])
    return mb_


EYE_RING = dict(glow=(2.45, 2.62), iris=(2.45, 3.25), cere=(3.25, 4.40))   # D (stylized pigeon eye + eye cere)
for side_key in ("tx", "rx"):
    # remove the hero's own v1 eye + optics group first (several names are reused by the subassemblies)
    own = [o.name for o in PARTS if o.name.startswith("eye_%s_" % side_key)] + ["%s_beam_rays_x8" % side_key]
    own += tx_names if side_key == "tx" else rx_names
    keep_hook = bpy.data.objects.get("HOOK_eye_" + side_key)
    keep_hook.parent = ROOT
    remove_objects([n for n in own if not n.startswith("HOOK_")])
    DECAL_LAYOUT.pop("silk_%s_carrier" % side_key, None)
    sa = import_subassembly(side_key)
    SUBASSY[side_key] = sa
    A_k, R_k = eye_frame(side_key)                 # position = HOOK_<side>_eye_aperture; local +X = out of the eye
    keep_hook.matrix_world = Matrix.Translation(A_k * MM) @ R_k.to_4x4()
    _d = (keep_hook.matrix_world.translation - sa["hooks"]["eye_aperture"].matrix_world.translation).length / MM
    print("EYE_HOOK %s aperture match %.4f mm" % (side_key, _d))
    keep_hook["doc"] = ("%s eye aperture (= HOOK_%s_eye_aperture): local +X = beam direction out of the eye"
                        % ("TX (pigeon's right eye)" if side_key == "tx" else "RX (pigeon's left eye)", side_key))
    # pigeon eye dressing around the delivered lenslet plate: glowing inner rim, orange iris ring, pale eye cere
    s_ = -1 if side_key == "tx" else 1
    ex_ring = (s_ * 18.0, -4.0, -9.0)        # below the optics group so it does not hide it in the exploded view
    obj_from_mb("eye_%s_iris_ring" % side_key, ring_on_axis(MB(), side_key, *EYE_RING["iris"], -0.9, 1.10), M["iris"],
                "head_eyes", ex_ring)
    obj_from_mb("eye_%s_iris_glow_ring" % side_key, ring_on_axis(MB(), side_key, *EYE_RING["glow"], 1.10, 1.16),
                HM["rim_r"] if side_key == "tx" else HM["rim_l"], "head_eyes", ex_ring)
    obj_from_mb("eye_%s_bezel" % side_key, ring_on_axis(MB(), side_key, *EYE_RING["cere"], -1.8, 1.0), M["bezel"],
                "head_eyes", ex_ring, label="eye cere (covers the carrier front edge)")

# ---- beak (stylized latch) at the front of the head, tipped slightly down
beak_objs = [hinge, upper, cere]
place(beak_objs, Matrix.Rotation(math.radians(12), 3, "X") @ (Matrix.Identity(3) * 1.4), (0.0, YF, 3.4),
      (0.0, HC.y - 13.2, HC.z - 2.6))   # beak scaled 1.4x to pigeon proportion (stylized part, D)


# ================================================================== SKIN (2-piece shell reshaped into the pigeon)
def catmull_n(rows, per=6):
    """Catmull-Rom over tuples of any length."""
    P = [list(r) for r in rows]
    P = [[2 * a - b for a, b in zip(P[0], P[1])]] + P + [[2 * a - b for a, b in zip(P[-1], P[-2])]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(per):
            t = k / per
            out.append([0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t ** 3)
                        for a, b, c, d in zip(p0, p1, p2, p3)])
    out.append(P[-2])
    return out


def path_frames(rows, per=6):
    """rows: (y, z, a, b) control sections of a path in the YZ plane -> list of (centre, Xaxis, Naxis, a, b)."""
    S = catmull_n(rows, per)
    fr = []
    for k, (y, z, a, b) in enumerate(S):
        k0, k1 = max(k - 1, 0), min(k + 1, len(S) - 1)
        T = Vector((0, S[k1][0] - S[k0][0], S[k1][1] - S[k0][1])).normalized()
        X = Vector((1, 0, 0))
        N = X.cross(T).normalized()
        fr.append((Vector((0, y, z)), X, N, max(a, 0.05), max(b, 0.05)))
    return fr


def sect_pt(f, t, pw=2.3):
    c, X, N, a, b = f
    ct, st = math.cos(t), math.sin(t)
    return c + X * (a * math.copysign(abs(ct) ** (2 / pw), ct)) + N * (b * math.copysign(abs(st) ** (2 / pw), st))


def half_surface(frames, half, n=28, pw=2.3):
    mb_ = MB()
    t0 = 0.0 if half == "top" else math.pi
    rows = []
    for f in frames:
        rows.append([tuple(sect_pt(f, t0 + math.pi * i / n, pw)) for i in range(n + 1)])
    v, fcs = [], []
    for r in rows:
        v += r
    w = n + 1
    for k in range(len(rows) - 1):
        for i in range(n):
            fcs.append([k * w + i, k * w + i + 1, (k + 1) * w + i + 1, (k + 1) * w + i])
    mb_.add(v, fcs)
    return mb_


def solidify(ob, t=0.6):
    md = ob.modifiers.new("skin_thickness", "SOLIDIFY")
    md.thickness = t * MM
    md.offset = 0.0
    md.use_rim = True
    md.use_even_offset = True


BODY_ROWS = [(-41.6, 45.0, 1.5, 2.0), (-40.0, 45.0, 6.0, 7.0), (-37.0, 45.0, 13.0, 15.0), (-31.0, 43.5, 18.0, 21.5),
             (-22.0, 42.0, 21.0, 25.0), (-12.0, 39.0, 21.0, 25.0), (-2.0, 36.0, 19.5, 23.0), (8.0, 33.0, 16.5, 19.0),
             (18.0, 30.0, 12.5, 14.0), (28.0, 27.0, 8.5, 9.0), (36.0, 25.0, 5.0, 5.0), (40.0, 24.0, 2.5, 2.5),
             (41.5, 24.0, 0.6, 0.6)]
NECK_ROWS = [(-42.5, 76.5, 9.5, 9.0), (-38.5, 70.0, 11.8, 11.0), (-36.0, 62.0, 14.8, 13.5), (-33.0, 54.0, 17.0, 15.5),
             (-30.0, 47.0, 18.0, 16.0)]
BODY_FR = path_frames(BODY_ROWS, 5)
NECK_FR = path_frames(NECK_ROWS, 5)


def body_x_surface(y, z):
    """Half-width of the body skin at (y, z) (0 outside); used to wrap wings and route ribbons."""
    best = 0.0
    for (c, X, N, a, b) in BODY_FR:
        if abs(c.y - y) < 2.6:
            dz = (z - c.z) / b
            if abs(dz) < 1:
                best = max(best, a * (1 - abs(dz) ** 2.3) ** (1 / 2.3))
    return best


def head_half(half, nu=36, nv=14):
    mb_ = MB()
    v, fcs = [], []
    for j in range(nv + 1):
        lat = (math.pi / 2) * j / nv * (1 if half == "top" else -1)
        for i in range(nu):
            lon = 2 * math.pi * i / nu
            v.append((HC.x + HA[0] * math.cos(lat) * math.cos(lon), HC.y + HA[1] * math.cos(lat) * math.sin(lon),
                      HC.z + HA[2] * math.sin(lat)))
    for j in range(nv):
        for i in range(nu):
            i2 = (i + 1) % nu
            fcs.append([j * nu + i, j * nu + i2, (j + 1) * nu + i2, (j + 1) * nu + i])
    mb_.add(v, fcs)
    return mb_


SKIN = {}
for half, cname, mat_k in (("top", "shell_top", "skin_top"), ("bottom", "shell_bottom", "skin_bot")):
    o = obj_from_mb("skin_%s_body" % ("back" if half == "top" else "belly"), half_surface(BODY_FR, half), M[mat_k], cname,
                    (0, 0, 0), smooth=True, label="shell %s half" % ("top/back" if half == "top" else "belly"))
    solidify(o)
    o2 = obj_from_mb("skin_%s_neck" % ("back" if half == "top" else "front"), half_surface(NECK_FR, half, pw=2.0),
                     HM["neck"], cname, (0, 0, 0), smooth=True, label="iridescent neck")
    solidify(o2)
    o3 = obj_from_mb("skin_head_%s" % ("cranium" if half == "top" else "jaw"), head_half(half), M[mat_k], cname, (0, 0, 0),
                     smooth=True)
    solidify(o3)
    SKIN[half] = [o, o2, o3]

# eye openings (v2 integration): cut the head skin around each eye axis (radius = eye-cere outer - 0.35) and bake the
# solidify + boolean into the mesh so the GLB carries no cutter objects
_cutters = []
for side_key in ("tx", "rx"):
    m_ = MB().cyl((0.0, 1.5, 0.0), EYE_RING["cere"][1] - 0.35, 9.0, axis="y", segs=64)   # skin overlaps the cere rim
    mc_ = MB()
    mc_.add([sub_pt(side_key, (y_, p_, q_)) for (p_, y_, q_) in m_.v], m_.f)
    _cutters.append(obj_from_mb("cutter_eye_%s" % side_key, mc_, M["bezel"], "head_eyes"))
for half in ("top", "bottom"):
    o3 = SKIN[half][2]
    for c_ in _cutters:
        md = o3.modifiers.new("eye_cut_" + c_.name[-2:], "BOOLEAN")
        md.operation = "DIFFERENCE"
        md.solver = "EXACT"
        md.object = c_
    bpy.context.view_layer.update()
    dg_ = bpy.context.evaluated_depsgraph_get()
    me_new = bpy.data.meshes.new_from_object(o3.evaluated_get(dg_))
    me_old = o3.data
    o3.modifiers.clear()
    o3.data = me_new
    bpy.data.meshes.remove(me_old)
    print("EYE_CUT %s faces=%d" % (o3.name, len(me_new.polygons)))
remove_objects([c_.name for c_ in _cutters])

# seams (panel lines); hex socket-head screws (Cisco 400G photo) with internal screw bosses (v2 polish)
mb_seam = MB()
SCR = {"top": (MB(), MB(), MB()), "bottom": (MB(), MB(), MB())}     # heads, hex recesses, inner bosses


def cyl_axis(mb_, c, n, r, h, segs=16, poly=None):
    n = Vector(n).normalized()
    e1 = n.cross(Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))).normalized()
    e2 = n.cross(e1)
    k = poly or segs
    v_, f_ = [], []
    for s_ in (0.0, 1.0):
        for i in range(k):
            ang = 2 * math.pi * i / k
            v_.append(tuple(Vector(c) + n * (h * s_) + (e1 * math.cos(ang) + e2 * math.sin(ang)) * r))
    f_.append(list(range(k))[::-1])
    f_.append([k + i for i in range(k)])
    for i in range(k):
        j = (i + 1) % k
        f_.append([i, j, k + j, k + i])
    mb_.add(v_, f_)


def screw(p, n, half):
    hd, rc, bs = SCR[half]
    p, n = Vector(p), Vector(n).normalized()
    cyl_axis(hd, p, n, 0.62, 0.30, segs=18)                       # head, sits on the skin
    cyl_axis(rc, p + n * 0.22, n, 0.30, 0.10, poly=6)             # hex socket (dark), slightly proud of the head top
    cyl_axis(bs, p - n * 0.3, -n, 0.95, 2.0, segs=16)             # boss behind the skin (visible through the x-ray skin)


for yq in (-26.0, -6.0, 16.0):
    f = min(BODY_FR, key=lambda F: abs(F[0].y - yq))
    f2 = (f[0], f[1], f[2], f[3] + 0.32, f[4] + 0.32)
    for half_t0 in (0.0, math.pi):
        pts = [tuple(sect_pt(f2, half_t0 + math.pi * (0.04 + 0.92 * i / 24))) for i in range(25)]
        mb_seam.tube(pts, 0.14, sides=5)
    for tq in (0.35, 1.2, 1.94, 2.79, 3.5, 4.2, 5.2, 5.9):
        c_, X_, N_, a_, b_ = f
        nrm = X_ * (math.cos(tq) / a_) + N_ * (math.sin(tq) / b_)
        screw(sect_pt((c_, X_, N_, a_ + 0.3, b_ + 0.3), tq), nrm, "top" if tq < math.pi else "bottom")
for half_t0 in (0.0, math.pi):                                           # longitudinal split seam along the flank
    pts = [tuple(sect_pt((F[0], F[1], F[2], F[3] + 0.3, F[4] + 0.3), half_t0)) for F in BODY_FR[2:-3]]
    mb_seam.tube(pts, 0.16, sides=5)
lat_s = math.radians(38)
pts = [(HC.x + (HA[0] + 0.3) * math.cos(lat_s) * math.cos(a), HC.y + (HA[1] + 0.3) * math.cos(lat_s) * math.sin(a),
        HC.z + (HA[2] + 0.3) * math.sin(lat_s)) for a in [2 * math.pi * i / 40 for i in range(41)]]
mb_seam.tube(pts, 0.13, sides=5)
for a in (0.4, 2.0, 3.6, 5.0):
    q_ = Vector((HC.x + (HA[0] + 0.3) * math.cos(lat_s) * math.cos(a), HC.y + (HA[1] + 0.3) * math.cos(lat_s) * math.sin(a),
                 HC.z + (HA[2] + 0.3) * math.sin(lat_s)))
    nrm = Vector(((q_.x - HC.x) / HA[0] ** 2, (q_.y - HC.y) / HA[1] ** 2, (q_.z - HC.z) / HA[2] ** 2))
    screw(q_, nrm, "top")
obj_from_mb("skin_panel_seams", mb_seam, M["seam"], "shell_top", (0, 0, 0), smooth=True)
for half, cn in (("top", "shell_top"), ("bottom", "shell_bottom")):
    hd, rc, bs = SCR[half]
    obj_from_mb("skin_screws" if half == "top" else "skin_screws_belly", hd, M["bezel"], cn, (0, 0, 0))
    obj_from_mb("skin_screw_sockets_" + half, rc, M["seam"], cn, (0, 0, 0))
    obj_from_mb("skin_screw_bosses_" + half, bs, M["skin_inner"], cn, (0, 0, 0))

# ================================================================== WINGS (finned heatsink split in two) + TAIL FAN
W_S = Vector((0, -27.0, 58.0))          # shoulder (side view)
W_T = Vector((0, 50.0, 21.0))           # folded wing tip (photo ~ (Y 44-50, Z 18-21))
U = (W_T - W_S).normalized()
V = Vector((0, -U.z, U.y))               # in-plane "up" (perpendicular to U in YZ)
if V.z < 0:
    V = -V
WING_L = (W_T - W_S).length
FIN_T = 0.30                             # A ref (Fig 3-16 fin thickness), reused for the feather fins


def wing_pt(u, v):
    return W_S + U * u + V * v


def blade(mb_, side, root, ang_deg, length, width, layer, nl=12, nw=3, tip_round=0.15):
    """Feather fin: tapered blade in the wing plane, wrapped onto the body (X pushed out to clear the skin)."""
    d = U * math.cos(math.radians(ang_deg)) + V * math.sin(math.radians(ang_deg))
    q = U * -math.sin(math.radians(ang_deg)) + V * math.cos(math.radians(ang_deg))
    clr = 0.6 + 0.38 * layer
    grid_top, grid_bot = [], []
    for i in range(nl + 1):
        s_ = i / nl
        half = width / 2 * (1 - s_ ** 3 * 0.9) * (1 - max(0, s_ - 0.85) / 0.15 * (1 - tip_round))
        rt, rb = [], []
        for j in range(nw + 1):
            t_ = -1 + 2 * j / nw
            P = root + d * (length * s_) + q * (half * t_)
            xb = body_x_surface(P.y, P.z)
            x_in = xb + clr + 0.35 * (1 - t_ * t_) * 0          # flat blade
            rb.append((side * x_in, P.y, P.z))
            rt.append((side * (x_in + FIN_T), P.y, P.z))
        grid_top.append(rt)
        grid_bot.append(rb)
    v, fcs = [], []
    w = nw + 1
    for g in (grid_bot, grid_top):
        for r in g:
            v += r
    off = len(grid_bot) * w
    for i in range(nl):
        for j in range(nw):
            a, b_, c_, d_ = i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j
            fcs.append([a, d_, c_, b_])
            fcs.append([off + a, off + b_, off + c_, off + d_])
    for i in range(nl):                                                   # side walls
        for j in (0, nw):
            a, b_ = i * w + j, (i + 1) * w + j
            fcs.append([a, b_, off + b_, off + a])
    for j in range(nw):
        for i in (0, nl):
            a, b_ = i * w + j, i * w + j + 1
            fcs.append([a, b_, off + b_, off + a])
    mb_.add(v, fcs)


WING_OBJS = {}
for side, key in ((-1, "right"), (1, "left")):
    mb_f, mb_c = MB(), MB()
    # primaries (longest, reach the wing tip), then secondaries/tertials, layered outward like stacked fins
    # (u_root, v_root, angle deg, length, width) in the wing plane; primaries end at the folded wing tip W_T
    prim = [(33.0, 0.0, -0.5, 52.0, 7.5), (31.0, -1.8, -1.5, 52.0, 7.5), (29.0, -3.6, -2.5, 51.0, 7.5),
            (27.0, -5.4, -3.5, 49.0, 7.0)]
    sec = [(20.0, -7.0, -6.0, 36.0, 8.0), (15.0, -8.5, -8.5, 35.0, 8.0), (10.0, -10.0, -11.0, 33.0, 7.5),
           (5.0, -11.0, -13.5, 30.0, 7.0)]
    k = 0
    for (ur, vr, ang, ln, wd) in prim + sec:
        blade(mb_f, side, wing_pt(ur, vr), ang, ln, wd, layer=k)
        k += 1
    # coverts / heat-spreader plate (outermost layer): the thermal contact of the wing
    blade(mb_c, side, wing_pt(-1.0, -5.5), -8.0, 34.0, 14.0, layer=k + 0.3, nl=10, nw=5, tip_round=0.7)
    WING_OBJS[key] = [obj_from_mb("wing_%s_flight_feather_fins" % key, mb_f, HM["wing_fins"], "fins", (side * 30, 6, 18),
                                  label="wing = heatsink fins"),
                      obj_from_mb("wing_%s_covert_heat_spreader" % key, mb_c, HM["wing_covert"], "fins", (side * 30, 6, 18))]

TA0 = Vector((0, 37.0, 22.5))            # tail base
TA1 = Vector((0, 66.0, 6.5))             # tail tip (photo ~ (Y 64, Z 6))
TD = (TA1 - TA0).normalized()
mb_t = MB()
for k, ang in enumerate((-15, -11, -7, -3.5, 0, 3.5, 7, 11, 15)):
    a_ = math.radians(ang)
    d = TD * math.cos(a_) + Vector((1, 0, 0)) * math.sin(a_)
    d = (d + Vector((0, 0, -1)) * 0.22 * abs(k - 4) / 4).normalized()   # outer feathers droop: folded V tail
    q = Vector((1, 0, 0)) * math.cos(a_) - TD * math.sin(a_)
    up = d.cross(q).normalized()
    if up.z < 0:
        up = -up
    lift = (4 - abs(k - 4)) * 1.2                                       # centre feathers on top (folded fan)
    root = TA0 + up * lift
    ln, wd = 31.0 - abs(ang) * 0.12, 7.0
    sec_ = []
    for i in range(9):
        s_ = i / 8
        half = wd / 2 * (0.55 + 0.45 * s_) * (1 - max(0, s_ - 0.88) / 0.12 * 0.6)
        P = root + d * (ln * s_)
        sec_.append([P - q * half, P + q * half, P + q * half + up * FIN_T, P - q * half + up * FIN_T])
    for i in range(8):
        mb_t.hull8(sec_[i], sec_[i + 1])
obj_from_mb("tail_fan_fins", mb_t, M["fins"], "fins", (0, 12, 4), label="tail fan")

# ================================================================== LEGS + FEET = the OSFP 60-contact card edge (MSA Rev 5.22 Fig 15-1, 3-21/22)
# Pin names and classes verbatim from MSA Fig 15-1 (top side 31-60, bottom side 1-30).
PIN_NAME = {
    60: "GND", 59: "TX1p", 58: "TX1n", 57: "GND", 56: "TX3p", 55: "TX3n", 54: "GND", 53: "TX5p", 52: "TX5n", 51: "GND",
    50: "TX7p", 49: "TX7n", 48: "GND", 47: "SDA", 46: "VCC", 45: "VCC", 44: "INT/RSTn", 43: "GND", 42: "RX8n", 41: "RX8p",
    40: "GND", 39: "RX6n", 38: "RX6p", 37: "GND", 36: "RX4n", 35: "RX4p", 34: "GND", 33: "RX2n", 32: "RX2p", 31: "GND",
    1: "GND", 2: "TX2p", 3: "TX2n", 4: "GND", 5: "TX4p", 6: "TX4n", 7: "GND", 8: "TX6p", 9: "TX6n", 10: "GND",
    11: "TX8p", 12: "TX8n", 13: "GND", 14: "SCL", 15: "VCC", 16: "VCC", 17: "LPWn/PRSn", 18: "GND", 19: "RX7n", 20: "RX7p",
    21: "GND", 22: "RX5n", 23: "RX5p", 24: "GND", 25: "RX3n", 26: "RX3p", 27: "GND", 28: "RX1n", 29: "RX1p", 30: "GND",
}


def pin_class(n):
    nm = PIN_NAME[n]
    if nm == "GND":
        return "GND"
    if nm == "VCC":
        return "VCC"
    if nm.startswith("TX"):
        return "TX"
    if nm.startswith("RX"):
        return "RX"
    return "LS"


PAD_LEN = {"GND": 2.35, "VCC": 2.05, "TX": 1.50, "RX": 1.50, "LS": 1.50}   # A: Fig 3-21 (0.55 / 0.85 / 1.40 from tip to a common 2.90 end)
LEN_CLASS = {"GND": "long (mates 1st)", "VCC": "medium (mates 2nd)", "TX": "short (mates last)", "RX": "short (mates last)",
             "LS": "short (mates last)"}
PAD_W, PAD_PITCH, VCC_W = 0.38, 0.60, 0.98                                  # A Fig 3-21 / 3-22
TOE_PINS = {   # base -> tip along the toe; every toe ends in a GND pad that continues into the talon
    "tx": {"outer": [60, 59, 58, 57, 56, 55, 54], "middle": [53, 52, 51, 50, 49, 48],
           "inner": [8, 9, 10, 11, 12, 13], "hallux": [1, 2, 3, 4, 5, 6, 7]},
    "rx": {"outer": [31, 32, 33, 34, 35, 36, 37], "middle": [38, 39, 40, 41, 42, 43],
           "inner": [18, 19, 20, 21, 22, 23, 24], "hallux": [25, 26, 27, 28, 29, 30]},
}
BELLY_ROWS = {"top": [47, 46, 45, 44], "bottom": [14, 15, 16, 17]}
_all = [n for f in TOE_PINS.values() for t in f.values() for n in t] + [n for r in BELLY_ROWS.values() for n in r]
assert sorted(_all) == list(range(1, 61)), "every OSFP contact exactly once"
_cnt = {c: sum(1 for n in range(1, 61) if pin_class(n) == c) for c in ("TX", "RX", "LS", "VCC", "GND")}
assert len(_all) == 60 and _cnt == {"TX": 16, "RX": 16, "LS": 4, "VCC": 4, "GND": 20}, _cnt
for key, cls in (("tx", "TX"), ("rx", "RX")):
    assert sum(1 for t in TOE_PINS[key].values() for n in t if pin_class(n) == cls) == 16
    assert all(pin_class(t[-1]) == "GND" for t in TOE_PINS[key].values())
print("PIN_ASSERT_OK total=60 counts=%s" % _cnt)
PINMAP = []

FOOT = {-1: Vector((-7.4, -12.5, 1.65)), 1: Vector((7.4, -12.5, 1.65))}   # foot centre / ankle ball
HIP = {-1: Vector((-6.6, -6.0, 15.5)), 1: Vector((6.6, -6.0, 15.5))}
TOES = [("hallux", 172.0, 8.5), ("inner", -22.0, 12.0), ("middle", -4.0, 16.0), ("outer", 26.0, 12.5)]   # C photo
Z_SIG, Z_GND = 0.30, 0.30        # pad bottoms coplanar (real card pads are coplanar; mating order = pad length)
A0, B0 = 1.3, 0.85               # toe half-width / half-height (toe width 2.6 holds the 2.35 GND pad)
Z_BASE = 1.55
Z_DIST = 0.335 + B0              # toe centre height over the flat pad zone (underside at 0.335)
FOOT_INFO = {}


def toe_zc(dist_from_base, ln, n_pads):
    flat_start = ln - (n_pads * PAD_PITCH + 0.6)
    return Z_BASE - (Z_BASE - Z_DIST) * min(1.0, max(0.0, dist_from_base) / max(flat_start, 0.5))


for side in (-1, 1):
    key = "tx" if side < 0 else "rx"
    F, Hp = FOOT[side], HIP[side]
    mb_bone, mb_toe, mb_claw, mb_gnd, mb_sig, mb_joint = MB(), MB(), MB(), MB(), MB(), MB()
    mb_bone.tube([tuple(Hp), tuple(Hp.lerp(F, 0.5) + Vector((0, -0.4, 0))), tuple(F + Vector((0, 0, 0.6)))], 1.05, sides=12)
    mb_joint.ellipsoid(tuple(F + Vector((0, 0, 0.35))), 1.45, 1.45, 1.2, segs=16, rings=8)
    mb_joint.ellipsoid(tuple(Hp.lerp(F, 0.5)), 1.35, 1.35, 1.0, segs=14, rings=6)
    mb_joint.box(F.x - 1.6, F.x + 1.6, F.y - 1.1, F.y + 1.4, 0.40, 0.75)          # metatarsal pad (no contact)
    toe_info = {}
    for (tn, ang, ln) in TOES:
        a_ = math.radians(ang) * side                                   # inner toes point toward the midline
        d = Vector((math.sin(a_), -math.cos(a_), 0.0))
        if tn == "hallux":
            d = Vector((-math.sin(math.radians(8)) * side, 1.0, 0.0)).normalized()
        q = Vector((d.y, -d.x, 0.0))
        if q.x < 0:
            q = -q
        pins = TOE_PINS[key][tn]
        npd = len(pins)
        rings = []
        for i in range(11):
            s_ = ln * i / 10
            P = F + d * s_
            zc = toe_zc(s_, ln, npd)
            rings.append([tuple(Vector((P.x, P.y, zc)) + q * (A0 * math.cos(2 * math.pi * j / 16)) +
                                Vector((0, 0, B0 * math.sin(2 * math.pi * j / 16)))) for j in range(16)])
        v, fcs = [], []
        for r in rings:
            v += r
        for i in range(len(rings) - 1):
            for j in range(16):
                j2 = (j + 1) % 16
                fcs.append([i * 16 + j, i * 16 + j2, (i + 1) * 16 + j2, (i + 1) * 16 + j])
        fcs.append(list(range(16))[::-1])
        fcs.append([(len(rings) - 1) * 16 + j for j in range(16)])
        mb_toe.add(v, fcs)
        for s_k in (0.3 * (ln - npd * PAD_PITCH),):                     # knuckle ring ahead of the pad zone
            P = F + d * s_k
            zc = toe_zc(s_k, ln, npd)
            ring_v = []
            for dy_ in (0.0, 0.35):
                for j in range(16):
                    ca, sa = math.cos(2 * math.pi * j / 16), math.sin(2 * math.pi * j / 16)
                    ring_v.append(tuple(P + d * dy_ + q * ((A0 + 0.12) * ca) + Vector((0, 0, zc + (B0 + 0.12) * sa))))
            mb_joint.add(ring_v, [[j, (j + 1) % 16, 16 + (j + 1) % 16, 16 + j] for j in range(16)])
        # talon: curved cone from the toe tip down to the ground; its gold tip is part of the distal GND pad
        tipP = F + d * ln
        cl = []
        for i in range(7):
            s_ = i / 6
            P = tipP + d * (1.9 * s_)
            z = Z_DIST + 0.25 * math.sin(math.pi * s_ * 0.6) - (Z_DIST + 0.25 * math.sin(math.pi * 0.6) - 0.06) * s_ ** 1.6
            cl.append((P.x, P.y, z, 0.55 * (1 - s_) + 0.06, 0.5 * (1 - s_) + 0.06))
        mbc = MB().loft(cl[:5], npts=10)
        mb_claw.add(mbc.v, mbc.f)
        mbg = MB().loft(cl[4:], npts=10)
        mb_gnd.add(mbg.v, mbg.f)
        mb_gnd.ellipsoid((cl[-1][0], cl[-1][1], 0.09), 0.16, 0.16, 0.09, segs=10, rings=5)
        # pads: a row ALONG the toe at the MSA pitch; each pad runs ACROSS the toe with its MSA length class
        pad_c = []
        for j, n in enumerate(pins):
            cls = pin_class(n)
            s_from_tip = 0.3 + PAD_W / 2 + (npd - 1 - j) * PAD_PITCH
            C = tipP - d * s_from_tip
            L = PAD_LEN[cls]
            zb = Z_GND if cls == "GND" else Z_SIG
            hw_, hl_ = d * (PAD_W / 2), q * (L / 2)
            c4 = [C - hw_ - hl_, C + hw_ - hl_, C + hw_ + hl_, C - hw_ + hl_]
            mb_sig.hull8([tuple(Vector((p_.x, p_.y, zb))) for p_ in c4], [tuple(Vector((p_.x, p_.y, 0.335))) for p_ in c4])
            pad_c.append((C.x, C.y, zb))
            PINMAP.append(dict(pin=n, name=PIN_NAME[n], cls=cls, length_class=LEN_CLASS[cls], length_mm=L,
                               width_mm=PAD_W, side="top (31-60)" if n >= 31 else "bottom (1-30)",
                               location="%s foot (%s), %s toe, pad %d of %d from base%s" % (
                                   "right" if key == "tx" else "left", key.upper(), tn, j + 1, npd,
                                   " = talon tip GND" if j == npd - 1 else ""),
                               centre_mm=[round(C.x, 3), round(C.y, 3), round(zb, 3)],
                               label_dir=[round(q.x * (1 if j % 2 == 0 else -1), 3), round(q.y * (1 if j % 2 == 0 else -1), 3)]))
        toe_info[tn] = (d, q, ln, pins, pad_c)
    lab = "TX foot (right): 26 contacts" if key == "tx" else "RX foot (left): 26 contacts"
    exf = (0, 0, 0)                     # feet stay on the ground so the leg flexes stay attached in the exploded view
    FOOT_INFO[key] = toe_info
    obj_from_mb("leg_%s_tarsus_bone" % key, mb_bone, M["shell_top"], "feet", exf, smooth=True)
    obj_from_mb("leg_%s_joints_and_pads" % key, mb_joint, M["claw"], "feet", exf, smooth=True)
    toes = obj_from_mb("foot_%s_toes_x4" % key, mb_toe, M["shell_top"], "feet", exf, smooth=True, label=lab)
    obj_from_mb("foot_%s_claws" % key, mb_claw, M["claw"], "feet", exf, smooth=True)
    _pm = HM["pads_r"] if key == "tx" else HM["pads_l"]
    obj_from_mb("foot_%s_talon_tip_gnd" % key, mb_gnd, _pm, "feet", exf, smooth=True, label="talon tips = GND")
    obj_from_mb("foot_%s_edge_pads_x26" % key, mb_sig, _pm, "feet", exf, label="toe pads (G-S-S-G)")
    HOOKS["foot_" + key] = hook("HOOK_foot_" + key, tuple(F), toes, "feet",
                                doc="%s foot centre; 26 OSFP contacts on 4 toes (see pinmap json)" % key.upper())

# leg band on the right (TX) leg = EEPROM (orange band, OUNOLIGHT)
Fr, Hr = FOOT[-1], HIP[-1]
BC = Hr.lerp(Fr, 0.50)                  # film v2: between the leg clips (0.30 / 0.68) so the band is not hidden
ax_ = (Hr - Fr).normalized()
e1 = Vector((1, 0, 0)) - ax_ * ax_.x
e1.normalize()
e2 = ax_.cross(e1).normalized()
mb = MB()
nseg = 32
v, fcs = [], []
for hz in (-1.6, 1.6):
    for rr in (2.95, 2.45):               # film v2: band rings the tarsus AND the leg flex (flex reaches ~2.3 mm off-axis)
        for j in range(nseg):
            a_ = 2 * math.pi * j / nseg
            v.append(tuple(BC + ax_ * hz + (e1 * math.cos(a_) + e2 * math.sin(a_)) * rr))
n_ = nseg
for j in range(n_):
    j2 = (j + 1) % n_
    fcs += [[j, j2, 2 * n_ + j2, 2 * n_ + j], [n_ + j2, n_ + j, 3 * n_ + j, 3 * n_ + j2],
            [j2, j, n_ + j, n_ + j2], [2 * n_ + j, 2 * n_ + j2, 3 * n_ + j2, 3 * n_ + j]]
mb.add(v, fcs)
band = obj_from_mb("leg_tx_band_eeprom_carrier", mb, HM["band"], "feet", (-6, 0, 0), smooth=True, label="leg band = EEPROM")
eeprom_parts = ICS["elec_eeprom_dfn_2x3"][1]
place(eeprom_parts, R_TX, ICS["elec_eeprom_dfn_2x3"][0], tuple(BC + e1 * -2.95))
setex(eeprom_parts, (-6, 0, 0))
# band text: Blender text -> mesh, wrapped around the band (front face, toward -Y)
cu = bpy.data.curves.new("band_text", "FONT")
cu.body = "OUNOLIGHT"
cu.size = 0.62 * MM
cu.extrude = 0.03 * MM
cu.align_x = "CENTER"
cu.align_y = "CENTER"
tob = bpy.data.objects.new("_tmp_text", cu)
SCN.collection.objects.link(tob)
deps_ = bpy.context.evaluated_depsgraph_get()
tme = bpy.data.meshes.new_from_object(tob.evaluated_get(deps_))
bpy.data.objects.remove(tob, do_unlink=True)
tv = [Vector(vv.co) / MM for vv in tme.vertices]
mbtx = MB()
R_band = 2.97
wrap = []
for vv in tv:
    ang = -math.pi / 2 + vv.x / R_band          # centred on the -Y (front) side of the band
    r_ = R_band + vv.z
    wrap.append(tuple(BC + ax_ * vv.y + (e1 * math.cos(ang) + e2 * math.sin(ang)) * r_))
mbtx.add(wrap, [list(p_.vertices) for p_ in tme.polygons])
bpy.data.meshes.remove(tme)
obj_from_mb("leg_tx_band_text_OUNOLIGHT", mbtx, M["ink_white"], "feet", (-6, 0, 0))

# belly cluster: the 4 low-speed + 4 VCC contacts (MSA middle group) on a small plate between the legs.
# True lateral positions from Fig 3-21 (top pins 44-47, bottom pins 14-17); VCC pairs 45/46 and 15/16 are one
# merged 0.98 mm pad each (Fig 3-22/3-23). Rows: top-side numbering in front, bottom-side numbering behind.
yb_ = -9.0
zb_ = min(sect_pt(F_, 1.5 * math.pi).z for F_ in BODY_FR if abs(F_[0].y - yb_) < 2.6) + 0.2
mb = MB().box(-2.2, 2.2, yb_ - 2.6, yb_ + 2.6, zb_ - 0.45, zb_ + 0.3)
obj_from_mb("pcb_belly_contact_plate", mb, M["polymer"], "pcb", (0, 0, -6))


def card_x(n):
    return -8.85 + (60 - n) * 0.60 if n >= 31 else -8.55 + (n - 1) * 0.60    # A Fig 3-21 (pin 60 at 8.85 TX side)


mb = MB()
z_pb = zb_ - 0.45
BELLY_C = {}
for row, sgn in (("top", -1), ("bottom", 1)):
    y_common = yb_ + sgn * 0.25                       # common inner end of the row (like the 2.90 mm line on the card)
    done = set()
    for n in BELLY_ROWS[row]:
        cls = pin_class(n)
        L = PAD_LEN[cls]
        if n in (45, 46, 15, 16):
            pair = (45, 46) if n in (45, 46) else (15, 16)
            if pair in done:
                continue
                continue
            done.add(pair)
            xc = (card_x(pair[0]) + card_x(pair[1])) / 2
            w = VCC_W
        else:
            xc, w = card_x(n), PAD_W
        y0, y1 = sorted((y_common, y_common + sgn * L))
        zb2 = z_pb - 0.035                                # coplanar pads; mating order is set by pad length
        mb.box(xc - w / 2, xc + w / 2, y0, y1, zb2, z_pb + 0.01)
        for nn in ((45, 46) if n in (45, 46) else (15, 16) if n in (15, 16) else (n,)):
            BELLY_C[nn] = (xc, (y0 + y1) / 2, zb2)
for row in ("top", "bottom"):
    for n in BELLY_ROWS[row]:
        cls = pin_class(n)
        C = BELLY_C[n]
        PINMAP.append(dict(pin=n, name=PIN_NAME[n], cls=cls, length_class=LEN_CLASS[cls], length_mm=PAD_LEN[cls],
                           width_mm=VCC_W if cls == "VCC" else PAD_W, side="top (31-60)" if n >= 31 else "bottom (1-30)",
                           location="belly cluster, %s-side row%s" % (row, " (merged VCC pad 45/46)" if n in (45, 46) else
                                                                     (" (merged VCC pad 15/16)" if n in (15, 16) else "")),
                           centre_mm=[round(C[0], 3), round(C[1], 3), round(C[2], 3)],
                           label_dir=[0.0, -1.0 if row == "top" else 1.0]))
lsp = obj_from_mb("pcb_belly_lowspeed_vcc_pads_x8", mb, M["gold"], "pcb", (0, 0, -6), label="belly: 4 low-speed + 4 VCC")
HOOKS["i2c_pads"] = hook("HOOK_i2c_pads", (0.0, yb_, z_pb), lsp, "pcb",
                         doc="belly cluster: SCL 14, SDA 47, LPWn/PRSn 17, INT/RSTn 44, VCC 15/16/45/46")
assert len(PINMAP) == 60 and len({r["pin"] for r in PINMAP}) == 60
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, ASSET + "_pinmap.json"), "w") as fh:
    json.dump(dict(source="OSFP MSA Rev 5.22 Fig 15-1 (names), Fig 3-21/3-22 (pad width 0.38, pitch 0.60, VCC 0.98, "
                          "lengths GND 2.35 / VCC 2.05 / signal 1.50 mm)",
                   frame="pigeon frame, mm; origin on the ground between the feet, -Y beak, +Z up, -X pigeon's right",
                   counts=_cnt, pins=sorted(PINMAP, key=lambda r: r["pin"])), fh, indent=1)

# ================================================================== CHEST: sternum PCB (keel) with the power organs
KEEL = [(-35.0, 40.0), (-31.0, 50.0), (-21.0, 53.0), (-9.0, 47.0), (-5.0, 31.0), (-9.0, 22.0), (-21.0, 19.5), (-32.0, 27.0)]
mb = MB().prism([(y, z) for (y, z) in KEEL], -0.5, 0.5, "yz")
obj_from_mb("pcb_sternum_board", mb, M["pcb"], "pcb", (0, 0, 0), bevel=0.08, label="sternum PCB")
CHEST = {  # name: (face sign, Y, Z)  face -1 = right side (-X), +1 = left side (+X)
    "elec_power_inductor_L1": (-1, -19.0, 38.0), "elec_power_inductor_L2": (1, -19.0, 38.0),
    "elec_load_switch_dfn_2x3": (-1, -12.0, 26.5),
    "elec_pmic_qfn_4x4": (-1, -27.5, 43.5), "elec_buck_mp28164_qfn_2x3": (-1, -12.0, 38.0),   # camera-side face
}
for nm, (fs, yy, zz) in CHEST.items():
    P0, parts = ICS[nm]
    place(parts, R_TX if fs < 0 else R_RX, P0, (fs * 0.5, yy, zz))
    setex(parts, (-10, -6, 0) if fs < 0 else (12, -2, -26))   # +X-face parts drop below the keel so they stay visible


def in_poly(y, z, poly):
    c = False
    for i in range(len(poly)):
        (y1, z1), (y2, z2) = poly[i], poly[(i + 1) % len(poly)]
        if (z1 > z) != (z2 > z) and y < (y2 - y1) * (z - z1) / (z2 - z1) + y1:
            c = not c
    return c


rng = random.Random(1311)
occ_k = {-1: [], 1: []}
for nm, (fs, yy, zz) in CHEST.items():
    occ_k[fs].append((yy - 2.6, yy + 2.6, zz - 2.6, zz + 2.6))
mb_body, mb_term = MB(), MB()
for (L_, W_, H_, cnt) in ((1.0, 0.5, 0.5, 60), (0.6, 0.3, 0.3, 56)):   # v2: denser (photos: dense 0201/0402 clusters)
    k = tries = 0
    while k < cnt and tries < 4000:
        tries += 1
        fs = rng.choice((-1, 1))
        yy, zz = rng.uniform(-34.0, -6.0), rng.uniform(20.5, 52.0)
        rot = rng.random() < 0.5
        ly, lz = (L_, W_) if rot else (W_, L_)
        if not all(in_poly(yy + a, zz + b, KEEL) for a in (-ly, ly) for b in (-lz, lz)):
            continue
        if any(not (yy + ly < a0 or yy - ly > a1 or zz + lz < b0 or zz - lz > b1) for (a0, a1, b0, b1) in occ_k[fs]):
            continue
        occ_k[fs].append((yy - ly / 2 - 0.15, yy + ly / 2 + 0.15, zz - lz / 2 - 0.15, zz + lz / 2 + 0.15))
        k += 1
        x0, x1 = (fs * 0.5, fs * (0.5 + H_))
        t_ = 0.15 * L_
        if rot:
            mb_body.box(x0, x1, yy - ly / 2 + t_, yy + ly / 2 - t_, zz - lz / 2, zz + lz / 2)
            mb_term.box(x0, x1, yy - ly / 2, yy - ly / 2 + t_, zz - lz / 2, zz + lz / 2)
            mb_term.box(x0, x1, yy + ly / 2 - t_, yy + ly / 2, zz - lz / 2, zz + lz / 2)
        else:
            mb_body.box(x0, x1, yy - ly / 2, yy + ly / 2, zz - lz / 2 + t_, zz + lz / 2 - t_)
            mb_term.box(x0, x1, yy - ly / 2, yy + ly / 2, zz - lz / 2, zz - lz / 2 + t_)
            mb_term.box(x0, x1, yy - ly / 2, yy + ly / 2, zz + lz / 2 - t_, zz + lz / 2)
obj_from_mb("elec_mlcc_bodies", mb_body, M["mlcc"], "electronics", (0, 0, 0), label="MLCCs")
obj_from_mb("elec_mlcc_terminals", mb_term, M["tin"], "electronics", (0, 0, 0))

# ================================================================== WIRING: flex circuits (constant width, routed curves, terminations, clips)
# Rules applied: constant-width polyimide (0.10 mm) with copper traces; bends only about the ribbon width axis
# (direction changes in the ribbon plane are cut-outline curves); no twist; stacked layers 0.45 mm apart in the neck;
# every flex ends on a stiffener + bond area (carriers), a flex-to-board connector, or directly on the contact pads.
FLEX_T = 0.10
TR_PITCH = 0.11            # D: 50 um lines / 60 um spaces class flex
WIRES = []


def ribbon(name, ctrl, wdirs, width, n_tr=0, cname="wiring", film="flex", tr_mat="copper", thick=FLEX_T, per=10,
           label=None, tr_w=None, lanes=None, tr_end=None, explode=(0, 0, 0)):
    pts = catmull(ctrl, per)
    m, n = len(pts), len(ctrl)
    P = [Vector(q) for q in pts]
    fr = []
    for k in range(m):
        f_ = k / (m - 1) * (n - 1)
        i = min(int(f_), n - 2)
        w = Vector(wdirs[i]).lerp(Vector(wdirs[i + 1]), f_ - i)
        T = (P[min(k + 1, m - 1)] - P[max(k - 1, 0)]).normalized()
        w = (w - T * w.dot(T)).normalized()
        fr.append((P[k], T, w, T.cross(w).normalized()))
    if lanes is None:
        lanes = [-width / 2 + width / (n_tr + 1) * (j + 1) for j in range(n_tr)]
    tws = tr_w or [0.06] * n_tr
    kend = [m - 1] * n_tr
    vias = []
    if tr_end:
        for j, tgt in enumerate(tr_end):
            if tgt is None:
                continue
            kb = min(range(m), key=lambda k: ((fr[k][0] + fr[k][2] * lanes[j]).x - tgt[0]) ** 2 +
                     ((fr[k][0] + fr[k][2] * lanes[j]).y - tgt[1]) ** 2)
            kend[j] = kb
            Pk, Tk, wk, Nk = fr[kb]
            vias.append(tuple(Pk + wk * lanes[j] + Nk * (thick / 2 + 0.01)))
    mf, mt_ = MB(), MB()
    for k in range(m - 1):
        sec = []
        for (Pk, Tk, wk, Nk) in (fr[k], fr[k + 1]):
            hw = width / 2
            sec.append([tuple(Pk - wk * hw - Nk * thick / 2), tuple(Pk + wk * hw - Nk * thick / 2),
                        tuple(Pk + wk * hw + Nk * thick / 2), tuple(Pk - wk * hw + Nk * thick / 2)])
        mf.hull8(sec[0], sec[1])
        for j in range(n_tr):
            if k >= kend[j]:
                continue
            s2 = []
            for (Pk, Tk, wk, Nk) in (fr[k], fr[k + 1]):
                c = Pk + wk * lanes[j] + Nk * (thick / 2)
                t2 = tws[j] / 2
                s2.append([tuple(c - wk * t2), tuple(c + wk * t2), tuple(c + wk * t2 + Nk * 0.018), tuple(c - wk * t2 + Nk * 0.018)])
            mt_.hull8(s2[0], s2[1])
    if film == "flex" and m > 6:
        # coverlay (v2 polish): translucent amber polyimide over the traces, ends left bare as bond / contact fingers
        mc = MB()
        for k in range(2, m - 3):
            sec = []
            for (Pk, Tk, wk, Nk) in (fr[k], fr[k + 1]):
                hw = width / 2 - 0.02
                z0_, z1_ = thick / 2 + 0.022, thick / 2 + 0.045
                sec.append([tuple(Pk - wk * hw + Nk * z0_), tuple(Pk + wk * hw + Nk * z0_), tuple(Pk + wk * hw + Nk * z1_),
                            tuple(Pk - wk * hw + Nk * z1_)])
            mc.hull8(sec[0], sec[1])
        WIRES.append(obj_from_mb(name + "_coverlay", mc, M["coverlay"], cname, explode))
    o1 = obj_from_mb(name, mf, M[film], cname, explode, label=label)
    WIRES.append(o1)
    if n_tr:
        WIRES.append(obj_from_mb(name + "_traces", mt_, M[tr_mat], cname, explode))
    if vias:
        mv = MB()
        for v_ in vias:
            mv.cyl(v_, 0.045, 0.03, axis="z", segs=10)
        WIRES.append(obj_from_mb(name + "_vias", mv, M["gold"], cname, explode))
    return fr


def block(name, x0, x1, y0, y1, z0, z1, material="connector", cname="wiring", label=None):
    o = obj_from_mb(name, MB().box(x0, x1, y0, y1, z0, z1), M[material], cname, (0, 0, 0), bevel=0.05, label=label)
    if material == "connector":
        # contact rows (v2 polish): gold pins along both long sides at 0.3 mm pitch, near the bottom edge
        mp = MB()
        lx_, ly_ = x1 - x0, y1 - y0
        npin = int(max(lx_, ly_) / 0.3) - 1
        for k in range(npin):
            t_ = (k + 1) / (npin + 1)
            if lx_ >= ly_:
                xc_ = x0 + lx_ * t_
                for yy_ in (y0 - 0.04, y1):
                    mp.box(xc_ - 0.06, xc_ + 0.06, yy_, yy_ + 0.04, z0, z0 + min(0.35, (z1 - z0) * 0.4))
            else:
                yc_ = y0 + ly_ * t_
                for xx_ in (x0 - 0.04, x1):
                    mp.box(xx_, xx_ + 0.04, yc_ - 0.06, yc_ + 0.06, z0, z0 + min(0.35, (z1 - z0) * 0.4))
        if mp.v:
            WIRES.append(obj_from_mb(name + "_pins", mp, M["gold"], cname, (0, 0, 0)))
    WIRES.append(o)
    return o


def clip(name, c, T, W, N, hw, hn, t=0.28, L=0.8):
    mb_ = MB()
    c, T, W, N = Vector(c), Vector(T).normalized(), Vector(W).normalized(), Vector(N).normalized()
    for (a, b, wa, wb) in ((W * -hw, W * hw, N * hn, N * (hn + t)), (W * -hw, W * hw, N * -(hn + t), N * -hn),
                           (W * -(hw + t), W * -hw, N * -(hn + t), N * (hn + t)), (W * hw, W * (hw + t), N * -(hn + t), N * (hn + t))):
        face0 = [c - T * L / 2 + a + wa, c - T * L / 2 + b + wa, c - T * L / 2 + b + wb, c - T * L / 2 + a + wb]
        face1 = [p_ + T * L for p_ in face0]
        mb_.hull8([tuple(p_) for p_ in face0], [tuple(p_) for p_ in face1])
    WIRES.append(obj_from_mb(name, mb_, M["connector"], "wiring", (0, 0, 0), label=None))


XA, YA, ZA = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def gsg(nsig):
    """Lane classes for nsig signals with grounds between and outside (G S G S ... G)."""
    out = []
    for i in range(nsig):
        out += ["G", "S"]
    return out + ["G"]


# ---- head: line-side flexes, v2 (2026-10-10) reconnected to the T / X subassembly hooks (DevLog-005 section 3).
# Pad fields come from the delivered parts: TX = assets/components/optics_tx/optics_tx_parts.json (rf_pads: 25 pads,
# GSSG x8 at 0.27 pitch on the carrier back edge; dc_nets + laser_nets: 28 + 4 pads on the carrier -Z edge);
# RX = rx_flex_pads measured from optics_rx.glb (8 x GSSG: G at lane +-0.19, S at +-0.07, lane pitch 0.50; DC/ctrl pads
# at +-2.30 / +-2.62 / +-2.94). Each flex lies on its pad field (bonded), bends only about its width axis and ends in
# a flex-to-board connector on the head-board underside: J1 (TX RF, vertical riser), J6 (TX DC), J2 (RX).
TXJ = json.load(open(os.path.join(COMP_DIR, "optics_tx", "optics_tx_parts.json")))
FLEX_LIFT = 0.02 + FLEX_T / 2 + 0.005                     # pad thickness + half film + glue line
# TX RF: pads on the carrier top face (sub X carrier_top), field centre HOOK_tx_flex_rf, exit sub -Y = pigeon +Y
A_tx, _ = eye_frame("tx")
TXH = {h_["name"]: h_["location_mm"] for h_ in TXJ["hooks"]}
x_tx_f = sub_pt("tx", (TXJ["stack_x_mm"]["carrier_top"] + FLEX_LIFT, 0, 0))[0]   # flex on the carrier top-face pads
y_rf_edge = sub_pt("tx", TXH["HOOK_flex_rf"])[1]
TXRF_Z = [p_["z_mm"] for p_ in TXJ["rf_pads"]]
TXRF_W = (max(TXRF_Z) - min(TXRF_Z)) + 0.27 + 0.4
tx_rf_lanes = [z_ for z_ in TXRF_Z]
tx_rf_tw = [0.12 if p_["kind"] == "G" else 0.09 for p_ in TXJ["rf_pads"]]
J1 = (-6.75, -6.05, y_rf_edge - 0.20, y_rf_edge + 0.65)   # x0, x1, y0, y1
y_j1 = (J1[2] + J1[3]) / 2
ribbon("flex_tx_rf_dsp_to_pic", [(x_tx_f, y_rf_edge - 0.6, E_Z), (x_tx_f, y_rf_edge - 0.1, E_Z),
                                 (x_tx_f + 0.08, y_rf_edge + 0.17, E_Z), (x_tx_f + 0.36, y_j1, E_Z),
                                 (J1[0] - 0.55, y_j1, E_Z), (J1[0], y_j1, E_Z)],
       [ZA] * 6, TXRF_W, len(tx_rf_lanes), lanes=tx_rf_lanes, tr_w=tx_rf_tw,
       label="TX RF flex: 8 GSSG lanes (25 traces) DSP -> PIC carrier")
block("conn_head_tx_rf", J1[0], J1[1], J1[2], J1[3], E_Z - TXRF_W / 2 - 0.25, ZB0, label="flex-to-board connectors")
# TX DC: pads on the carrier outer face along its -Z edge, exit sub -Z (down), wrap under the carrier, rise to J6
TXDC_SUBY = [n_["field_y_mm"] for n_ in TXJ["laser_nets"]] + [n_["field_y_mm"] for n_ in TXJ["dc_nets"]]
TXDC_Y = [sub_pt("tx", (0, y_, 0))[1] for y_ in TXDC_SUBY]
y_dc_c = sub_pt("tx", TXH["HOOK_flex_dc"])[1]
TXDC_W = (max(TXDC_Y) - min(TXDC_Y)) + 0.6
tx_dc_lanes = [y_ - y_dc_c for y_ in TXDC_Y]
tx_dc_tw = [0.09] * len(TXJ["laser_nets"]) + [0.06] * len(TXJ["dc_nets"])
z_dc_edge = sub_pt("tx", TXH["HOOK_flex_dc"])[2]
J6 = (-6.6, -5.6, y_dc_c - TXDC_W / 2 - 0.25, y_dc_c + TXDC_W / 2 + 0.25)
x_dc_up = (J6[0] + J6[1]) / 2
ribbon("flex_tx_dc_pic_laser", [(x_tx_f, y_dc_c, z_dc_edge + 0.6), (x_tx_f, y_dc_c, z_dc_edge + 0.05),
                                (x_tx_f + 0.08, y_dc_c, z_dc_edge - 0.23), (x_tx_f + 0.36, y_dc_c, z_dc_edge - 0.32),
                                (x_dc_up - 0.8, y_dc_c, z_dc_edge - 0.32), (x_dc_up - 0.25, y_dc_c, z_dc_edge),
                                (x_dc_up, y_dc_c, z_dc_edge + 0.7), (x_dc_up, y_dc_c, ZB0 - 4.0), (x_dc_up, y_dc_c, ZB0 - 1.25)],
       [YA] * 9, TXDC_W, len(tx_dc_lanes), lanes=tx_dc_lanes, tr_w=tx_dc_tw,
       label="TX DC flex: 28 PIC (heaters, MPDs) + 4 laser (LD1/LD2 +/-)")
block("conn_head_tx_dc", J6[0], J6[1], J6[2], J6[3], ZB0 - 1.25, ZB0)
# RX: pads on the carrier top (sub Z -1.45) at sub X -6.85..-6.53, exit sub -X (inward), bend up to J2
def _rx_pad_rows():
    """Pad centres / widths along the carrier Y axis, measured from the imported rx_flex_pads mesh (X's file)."""
    ob = bpy.data.objects["rx_flex_pads"]
    A_, R_ = eye_frame("rx")
    me = ob.data
    vw = [R_.transposed() @ (ob.matrix_world @ v.co / MM - A_) for v in me.vertices]
    iv = sorted((min(vw[i].y for i in p_.vertices), max(vw[i].y for i in p_.vertices)) for p_ in me.polygons)
    rows = []
    for lo, hi in iv:
        if rows and lo <= rows[-1][1] + 1e-4:
            rows[-1][1] = max(rows[-1][1], hi)
        else:
            rows.append([lo, hi])
    return [((lo + hi) / 2, hi - lo) for lo, hi in rows]


RX_PADS = _rx_pad_rows()
RX_DC = [(c_, w_) for c_, w_ in RX_PADS if w_ > 0.15]            # 6 DC / control pads (0.20 wide)
RX_RF = [(c_, w_) for c_, w_ in RX_PADS if w_ <= 0.15]           # 8 x G-S-S-G: G 0.09 wide, S 0.055 wide (pairs 0.10 c-c)
assert len(RX_RF) == 32 and len(RX_DC) == 6, (len(RX_RF), len(RX_DC))
rx_lanes = [c_ for c_, _ in RX_RF] + [c_ for c_, _ in RX_DC]
rx_tw = [min(0.9 * w_, 0.12) for _, w_ in RX_RF] + [0.12] * len(RX_DC)
print("RX_PADS rf=%d dc=%d pair_cc=%.3f" % (len(RX_RF), len(RX_DC), RX_RF[2][0] - RX_RF[1][0]))
RX_F_W = 6.3
p_rx0 = Vector(sub_pt("rx", (-6.45, 0.0, -1.45 + 0.007 + FLEX_T / 2 + 0.005)))
x_rx_edge = sub_pt("rx", (-6.95, 0, 0))[0]
J2 = (x_rx_edge - 0.96, x_rx_edge + 0.04, E_Y - RX_F_W / 2 - 0.2, E_Y + RX_F_W / 2 + 0.2)
x_rx_up = (J2[0] + J2[1]) / 2
ribbon("flex_rx_tia_to_dsp", [tuple(p_rx0), (x_rx_edge + 0.04, E_Y, p_rx0.z), (x_rx_up + 0.22, E_Y, p_rx0.z + 0.09),
                              (x_rx_up + 0.05, E_Y, p_rx0.z + 0.39), (x_rx_up, E_Y, p_rx0.z + 1.1), (x_rx_up, E_Y, ZB0 - 1.25)],
       [YA] * 6, RX_F_W, len(rx_lanes), lanes=rx_lanes, tr_w=rx_tw,
       label="RX flex: 8 differential GSSG TIA outputs (32) + 6 DC/control (GND, VPD_BIAS, VTIA; RSTn, SDA, SCL)")
block("conn_head_rx", J2[0], J2[1], J2[2], J2[3], ZB0 - 1.25, ZB0)
print("LINE_FLEX tx_rf=%d traces w=%.2f  tx_dc=%d traces w=%.2f  rx=%d traces w=%.2f" % (
    len(tx_rf_lanes), TXRF_W, len(tx_dc_lanes), TXDC_W, len(rx_lanes), RX_F_W))

# ---- host-side lanes: head board -> neck (stacked) -> chest -> leg -> ankle -> 4 toe branches -> the 26 foot pads
GROUP_ORDER = {-1: ["hallux", "outer", "middle", "inner"], 1: ["inner", "middle", "outer", "hallux"]}   # -X .. +X
S1, S2, S3 = Vector((0, -37.0, 72.0)), Vector((0, -33.5, 63.5)), Vector((0, -30.5, 57.0))
Y_CONN = {"tx": HC.y + 5.6, "pwr": HC.y + 6.8, "rx": HC.y + 8.0}
STACK = {"tx": -0.45, "pwr": 0.0, "rx": 0.45}
for key in ("tx", "rx"):
    side = -1 if key == "tx" else 1
    Hp, F = HIP[side], FOOT[side]
    groups = GROUP_ORDER[side]
    gw = {g: TR_PITCH * len(TOE_PINS[key][g]) + 0.12 for g in groups}
    width = sum(gw.values()) + 0.1 * 3
    lanes_all, gcent, cur = [], {}, -width / 2
    for g in groups:
        npd = len(TOE_PINS[key][g])
        gcent[g] = cur + gw[g] / 2
        for j in range(npd):
            lanes_all.append(gcent[g] - gw[g] / 2 + 0.06 + TR_PITCH * (j + 0.5))
        cur += gw[g] + 0.1
    o_ = STACK[key]
    D_ = F - Hp
    nf = Vector((0, -0.905, 0.425)) * 1.55
    end = Vector((F.x, F.y - 1.75, F.z + 2.35))
    ctrl = [(0.0, Y_CONN[key], ZB0 - 1.25), (0.0, Y_CONN[key], ZB0 - 2.8), tuple(S1 + Vector((0, o_, 0))),
            tuple(S2 + Vector((0, o_, 0))), tuple(S3 + Vector((0, o_, 0))), (side * 2.6, -28.0 + o_, 51.0),
            (side * 5.0, -24.5, 44.0), (side * 5.0, -17.0, 31.0), (side * 5.8, -9.0, 21.0), (Hp.x + side * 0.1, -8.0, 15.5),
            tuple(Hp + D_ * 0.3 + nf), tuple(Hp + D_ * 0.7 + nf), tuple(end)]
    fr = ribbon("flex_%s_host_lanes_neck_leg" % key, ctrl, [XA] * len(ctrl), width, len(lanes_all), lanes=lanes_all,
                label=("neck/leg flex: 8 TX pairs + 10 GND" if key == "tx" else "neck/leg flex: 8 RX pairs + 10 GND"))
    block("conn_head_%s_host" % key, -width / 2 - 0.3, width / 2 + 0.3, Y_CONN[key] - 0.45, Y_CONN[key] + 0.45, ZB0 - 1.25, ZB0)
    # clips on the leg (flex + tarsus)
    for tq in (0.3, 0.68):
        cb = Hp + D_ * tq
        cr = cb + nf
        c = (cb + cr) / 2
        Nn = (cr - cb).normalized()
        clip("clip_leg_%s_%d" % (key, int(tq * 100)), c, D_, Vector(XA), Nn, width / 2 + 0.15, (cr - cb).length / 2 + 1.05)
    # toe branches split from the leg flex at the ankle front; traces end in vias on the pads below
    Pend, Tend, wend, Nend = fr[-1]
    for g in groups:
        d, q, ln, pins, pad_c = FOOT_INFO[key][g]
        npd = len(pins)
        st = Pend + wend * gcent[g]
        top = lambda s_: toe_zc(s_, ln, npd) + B0 + FLEX_T / 2 + 0.02
        if g == "hallux":
            c_ = [tuple(st), tuple(st + Vector((side * 0.9, -0.2, -0.7))), (F.x + side * 2.7, F.y - 0.3, 1.95),
                  (F.x + side * 2.4, F.y + 1.6, 2.2)]
        else:
            c_ = [tuple(st), tuple(st + Tend * 0.6)]
        for s_ in (1.6, 0.5 * ln, ln - 0.25):
            Pq = F + d * s_
            c_.append((Pq.x, Pq.y, top(s_)))
        bw = TR_PITCH * npd + 0.12
        bl = [-bw / 2 + 0.06 + TR_PITCH * (j + 0.5) for j in range(npd)]
        ribbon("flex_%s_toe_%s" % (key, g), c_, [XA] * len(c_), bw, npd, lanes=bl, tr_end=[(pc[0], pc[1]) for pc in pad_c],
               per=8)

# ---- power + I2C: head board -> neck (middle layer) -> sternum top-edge connector
# Audit fix 3: rail widths sized for current (D, rough IPC-2152 rule ~2.5 A per mm of 35 um outer copper at 10 C rise):
# DSP core rail (~10 A after the buck) 4.0 mm, 3V3 (~3 A) 1.2 mm, two GND returns 1.8 mm each, I2C/control 0.06 mm.
PWR_TW = [1.2, 4.0, 1.8, 1.8, 0.06, 0.06, 0.06, 0.06]
PWR_LANES, _c = [], -5.1
for _w in PWR_TW:
    PWR_LANES.append(_c + _w / 2)
    _c += _w + 0.25
PWR_W = _c - 0.25 + 5.1 + 0.4
PWR_LANES = [l_ - (PWR_W - 0.4) / 2 + 5.1 - 0.0 for l_ in PWR_LANES]
ribbon("flex_power_i2c_neck", [(0.0, Y_CONN["pwr"], ZB0 - 1.25), (0.0, Y_CONN["pwr"], ZB0 - 2.8), tuple(S1), tuple(S2),
                               tuple(S3), (0.0, -26.5, 55.6), (0.0, -23.25, 53.7)],
       [XA] * 7, PWR_W, 8, label="power + I2C flex (3V3, DSP core, 2x GND, SCL, SDA, LPWn, INT)",
       lanes=PWR_LANES, tr_w=PWR_TW)
block("conn_head_pwr", -PWR_W / 2 - 0.3, PWR_W / 2 + 0.3, Y_CONN["pwr"] - 0.45, Y_CONN["pwr"] + 0.45, ZB0 - 1.25, ZB0)
block("conn_sternum_top_pwr", -PWR_W / 2 - 0.3, PWR_W / 2 + 0.3, -24.5, -22.0, 51.6, 53.6)
for nm_, Sx in (("clip_neck_upper", S1), ("clip_neck_lower", S2)):
    clip(nm_, Sx, Vector((0, 0.35, -0.94)), Vector(XA), Vector((0, 0.94, 0.35)), PWR_W / 2 + 0.2, 0.45 + 0.05 + 0.12)
# ---- belly contact plate -> sternum bottom-edge connector (SCL, SDA, LPWn/PRSn, INT/RSTn, VCC 45/46, VCC 15/16)
# audit fix 3: VCC traces 1.2 mm each (module ~16 W / 3.3 V ~ 5 A over two traces)
ribbon("flex_belly_lowspeed_vcc", [(0.0, yb_, zb_ + 0.35), (0.0, yb_ - 0.2, zb_ + 2.0), (0.0, -10.4, 17.5), (0.0, -11.0, 20.6)],
       [XA] * 4, 3.6, 6, label="belly flex: 4 low-speed + 2 VCC (1.2 mm)", tr_w=[0.06, 0.06, 0.06, 0.06, 1.2, 1.2],
       lanes=[-1.55, -1.35, -1.15, -0.95, -0.05, 1.25])
block("conn_sternum_bottom_belly", -2.5, 2.5, -12.2, -9.8, 20.6, 22.4)
block("flex_belly_stiffener", -1.9, 1.9, yb_ - 1.0, yb_ + 1.0, zb_ + 0.3, zb_ + 0.5, "stiffener")
# audit fix 3: GND return taps from the leg flexes (10 GND traces each) to the sternum bottom connector
for side in (-1, 1):
    ribbon("flex_%s_gnd_tap_to_sternum" % ("tx" if side < 0 else "rx"),
           [(side * 5.2, -10.6, 21.5), (side * 4.0, -11.0, 21.6), (side * 2.6, -11.0, 21.5)], [YA] * 3, 1.6, 2,
           tr_w=[0.6, 0.6], label="GND return taps")
# sternum GND ring (stitched ground plane edge) on both faces
cy_k = sum(p_[0] for p_ in KEEL) / len(KEEL)
cz_k = sum(p_[1] for p_ in KEEL) / len(KEEL)
mb = MB()
for fs in (-1, 1):
    for i in range(len(KEEL)):
        (y1, z1), (y2, z2) = KEEL[i], KEEL[(i + 1) % len(KEEL)]
        def ins(y, z, d=0.9):
            v_ = Vector((cy_k - y, cz_k - z))
            v_.normalize()
            return (y + v_.x * d, z + v_.y * d)
        quad = [(y1, z1), (y2, z2), ins(y2, z2), ins(y1, z1)]
        x0_, x1_ = sorted((fs * 0.5, fs * 0.535))
        mb.prism(quad, x0_, x1_, "yz")
obj_from_mb("pcb_sternum_gnd_ring", mb, M["copper"], "pcb", (0, 0, 0), label="sternum GND")
# audit fix 4: EEPROM leg band on the MCU I2C bus (SCL, SDA, VCC, GND) up the outside of the right tarsus
ee_p = BC + e1 * -3.65                   # film v2: band outer radius 2.95 + EEPROM body
ribbon("flex_eeprom_band_i2c", [tuple(ee_p + ax_ * 0.2), tuple(Hr.lerp(Fr, 0.35) + e1 * -2.6), tuple(Hr + e1 * -1.6 + ax_ * -0.4),
                                (-4.2, -10.0, 19.8), (-2.4, -11.0, 20.9)],
       [tuple(e2), tuple(e2), tuple(e2), YA, YA], 0.9, 4, tr_w=[0.06, 0.06, 0.12, 0.12], label="EEPROM I2C flex")

# ---- board decals (v2 polish): head board top / bottom and both sternum faces
dy0_p, dy1_p = HC.y + 0.5 - dd["ly"] / 2, HC.y + 0.5 + dd["ly"] / 2
def _hb(x, y, flip=False):
    return [round((6.8 - x) if flip else (x + 6.8), 3), round(y - (HC.y - 8.0), 3)]


HB_POLY = [(-5.8, HC.y - 8.7), (5.8, HC.y - 8.7), (6.8, HC.y - 7.7), (6.8, HC.y + 7.0), (5.0, HC.y + 8.0), (5.0, HC.y + 10.2),
           (-5.0, HC.y + 10.2), (-5.0, HC.y + 8.0), (-6.8, HC.y + 7.0), (-6.8, HC.y - 7.7)]   # v2: front extended 0.7 for J2


def _rect(x0, x1, y0, y1, flip=False):
    a, b = _hb(x0, y0, flip), _hb(x1, y1, flip)
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]


hb_top = make_decal("silk_head_top", (-6.8, HC.y - 8.0, ZB0 + 1.0), (1, 0, 0), (0, 1, 0), (13.6, 18.2), dict(
    kind="pcb", outline=[_hb(x, y) for x, y in HB_POLY],
    rects=[[*_rect(-6.55, 6.55, dy0_p - 0.1, dy1_p + 0.1), "U1"]],
    vias=[[0.3, 0.15, 13.3, 1.2], [2.0, 16.0, 11.6, 16.6]], fiducials=[[2.4, 17.4], [11.2, 17.4], [0.7, 0.7]],
    tps=[[12.6, 0.7, ""]], texts=[["OUNOLIGHT", 6.8, 17.55, 0.75, 0], ["94V-0  2541  HB-01B", 6.8, 15.85, 0.42, 0]]),
    "pcb", (0, -6, 12), lift=0.004)
rf_tx_c = ((J1[0] + J1[1]) / 2, (J1[2] + J1[3]) / 2)
rf_rx_c = ((J2[0] + J2[1]) / 2, (J2[2] + J2[3]) / 2)
bga_c = (0.0, HC.y + 0.5)
enig = []
for (cx_, cy_), n_ in ((rf_tx_c, 8), (rf_rx_c, 8)):
    d_ = Vector((bga_c[0] - cx_, bga_c[1] - cy_))
    L_ = d_.length
    u_ = d_.normalized()
    w_ = Vector((-u_.y, u_.x))
    for k in range(n_):                               # 8 differential RF pairs (one ENIG line drawn per pair)
        off = (k - (n_ - 1) / 2) * 0.16
        p0_ = Vector((cx_, cy_)) + w_ * off
        p1_ = Vector((cx_, cy_)) + u_ * (L_ * 0.45) + w_ * off * 1.6
        p2_ = Vector(bga_c) - u_ * 3.6 + w_ * off * 2.2
        enig.append([_hb(q_.x, q_.y, True) for q_ in (p0_, p1_, p2_)])
hb_bot = make_decal("silk_head_bottom", (6.8, HC.y - 8.0, ZB0), (-1, 0, 0), (0, 1, 0), (13.6, 18.2), dict(
    kind="pcb", outline=[_hb(x, y, True) for x, y in HB_POLY],
    rects=[[*_rect(MCU_C[0] - 3.65, MCU_C[0] + 3.65, MCU_C[1] - 3.65, MCU_C[1] + 3.65, True), "U2"], [*_rect(4.0, 5.8, HC.y - 0.6, HC.y + 1.6, True), "Y1"],
           [*_rect(-2.42, 2.42, Y_CONN["tx"] - 0.5, Y_CONN["tx"] + 0.5, True), "J3"],
           [*_rect(-PWR_W / 2 - 0.35, PWR_W / 2 + 0.35, Y_CONN["pwr"] - 0.5, Y_CONN["pwr"] + 0.5, True), "J4"],
           [*_rect(-2.42, 2.42, Y_CONN["rx"] - 0.5, Y_CONN["rx"] + 0.5, True), "J5"],
           [*_rect(J1[0] - 0.05, J1[1] + 0.05, J1[2] - 0.05, J1[3] + 0.05, True), "J1"],
           [*_rect(J2[0] - 0.05, J2[1] + 0.05, J2[2] - 0.05, J2[3] + 0.05, True), "J2"],
           [*_rect(J6[0] - 0.05, J6[1] + 0.05, J6[2] - 0.05, J6[3] + 0.05, True), "J6"]],
    enig=enig, vias=[[0.3, 0.2, 13.3, 1.4], [0.3, 1.6, 2.6, 7.0], [11.0, 1.6, 13.3, 7.0]],
    fiducials=[[0.7, 0.7], [12.9, 0.7], [7.0, 17.6]], tps=[[2.0, 3.0, "TP5"], [11.6, 3.0, "TP6"]],
    texts=[["OUNOLIGHT HB-01 REV B", 6.8, 0.9, 0.5, 0]]), "pcb", (0, -6, 12), lift=0.004)


def _st(y, z, face):
    return [round((-4.0 - y) if face < 0 else (y + 36.0), 3), round(z - 19.0, 3)]


ST_PARTS = {-1: [("elec_power_inductor_L1", "L1", "circle"), ("elec_pmic_qfn_4x4", "U3", (4.0, 4.0)),
                 ("elec_buck_mp28164_qfn_2x3", "U4", (2.0, 3.0)), ("elec_load_switch_dfn_2x3", "Q1", (3.0, 2.0))],
            1: [("elec_power_inductor_L2", "L2", "circle")]}
ST_DECALS = []
for face in (-1, 1):
    rects, circles = [], []
    for nm, ref, fp in ST_PARTS[face]:
        fs_, yy_, zz_ = CHEST[nm]
        if fp == "circle":
            circles.append([*_st(yy_, zz_, face), 2.2, ref])
        else:
            zw, yw = fp                       # R_TX / R_RX map module X -> pigeon Z, module Y -> pigeon Y
            a, b = _st(yy_ - yw / 2 - 0.25, zz_ - zw / 2 - 0.25, face), _st(yy_ + yw / 2 + 0.25, zz_ + zw / 2 + 0.25, face)
            rects.append([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]), ref])
    for (y0_, y1_, z0_, z1_, ref) in ((-24.6, -21.9, 51.5, 53.7, "J6"), (-12.3, -9.7, 20.5, 22.5, "J7")):
        a, b = _st(y0_, z0_, face), _st(y1_, z1_, face)
        rects.append([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]), ref])
    lay = dict(kind="pcb", outline=[_st(y, z, face) for y, z in KEEL], rects=rects, circles=circles, ring_vias=0.45,
               fiducials=[_st(-31.5, 32.0, face), _st(-10.5, 44.5, face), _st(-22.0, 50.8, face)],
               tps=[[*_st(-8.2, 43.0, face), "3V3"], [*_st(-8.2, 39.5, face), "VCORE"], [*_st(-8.2, 36.0, face), "GND"]] if face < 0 else [],
               texts=[["OUNOLIGHT ST-02 REV A", _st(-21.5, 23.6, face)[0], 4.6, 0.6, 0], ["94V-0  2541", _st(-21.5, 22.0, face)[0], 3.0, 0.45, 0]])
    if face < 0:
        d_ = make_decal("silk_sternum_right", (-0.5, -4.0, 19.0), (0, -1, 0), (0, 0, 1), (32.0, 35.0), lay, "pcb", lift=0.004)
    else:
        d_ = make_decal("silk_sternum_left", (0.5, -36.0, 19.0), (0, 1, 0), (0, 0, 1), (32.0, 35.0), lay, "pcb", lift=0.004)
    if d_ is not None:
        ST_DECALS.append(d_)
if hb_top is not None:
    setex([hb_top], (0, -6, 12))
# backside 0201 decoupling and small passives on the head-board underside (v2 polish)
_occ_hb = [(MCU_C[0] - 3.75, MCU_C[0] + 3.75, MCU_C[1] - 3.75, MCU_C[1] + 3.75), (3.95, 5.85, HC.y - 0.65, HC.y + 1.65), (3.1, 4.2, HC.y + 0.35, HC.y + 0.65),
           (-2.45, 2.45, Y_CONN["tx"] - 0.55, Y_CONN["tx"] + 0.55), (-PWR_W / 2 - 0.4, PWR_W / 2 + 0.4, Y_CONN["pwr"] - 0.55, Y_CONN["pwr"] + 0.55),
           (-2.45, 2.45, Y_CONN["rx"] - 0.55, Y_CONN["rx"] + 0.55), (J1[0] - 0.1, J1[1] + 0.1, J1[2] - 0.1, J1[3] + 0.1), (J2[0] - 0.1, J2[1] + 0.1, J2[2] - 0.1, J2[3] + 0.1),
           (J6[0] - 0.1, J6[1] + 0.1, J6[2] - 0.1, J6[3] + 0.1)]
_rh = random.Random(442)
mbc_b, mbc_t = MB(), MB()
_k = 0
_tries = 0
while _k < 70 and _tries < 6000:
    _tries += 1
    x_, y_ = _rh.uniform(-6.5, 6.5), _rh.uniform(HC.y - 7.7, HC.y + 9.9)
    rot_ = _rh.random() < 0.5
    lx_, ly_ = (0.6, 0.3) if rot_ else (0.3, 0.6)
    pts4 = [(x_ - lx_ / 2, y_ - ly_ / 2), (x_ + lx_ / 2, y_ + ly_ / 2), (x_ - lx_ / 2, y_ + ly_ / 2), (x_ + lx_ / 2, y_ - ly_ / 2)]
    if not all(in_poly(px, py, HB_POLY) for px, py in pts4):
        continue
    if any(not (x_ + lx_ / 2 < a0 or x_ - lx_ / 2 > a1 or y_ + ly_ / 2 < b0 or y_ - ly_ / 2 > b1) for (a0, a1, b0, b1) in _occ_hb):
        continue
    _occ_hb.append((x_ - lx_ / 2 - 0.12, x_ + lx_ / 2 + 0.12, y_ - ly_ / 2 - 0.12, y_ + ly_ / 2 + 0.12))
    _k += 1
    t_ = 0.09
    if rot_:
        mbc_b.box(x_ - lx_ / 2 + t_, x_ + lx_ / 2 - t_, y_ - ly_ / 2, y_ + ly_ / 2, ZB0 - 0.3, ZB0)
        mbc_t.box(x_ - lx_ / 2, x_ - lx_ / 2 + t_, y_ - ly_ / 2, y_ + ly_ / 2, ZB0 - 0.3, ZB0)
        mbc_t.box(x_ + lx_ / 2 - t_, x_ + lx_ / 2, y_ - ly_ / 2, y_ + ly_ / 2, ZB0 - 0.3, ZB0)
    else:
        mbc_b.box(x_ - lx_ / 2, x_ + lx_ / 2, y_ - ly_ / 2 + t_, y_ + ly_ / 2 - t_, ZB0 - 0.3, ZB0)
        mbc_t.box(x_ - lx_ / 2, x_ + lx_ / 2, y_ - ly_ / 2, y_ - ly_ / 2 + t_, ZB0 - 0.3, ZB0)
        mbc_t.box(x_ - lx_ / 2, x_ + lx_ / 2, y_ + ly_ / 2 - t_, y_ + ly_ / 2, ZB0 - 0.3, ZB0)
obj_from_mb("pcb_head_backside_0201_bodies", mbc_b, M["mlcc"], "electronics", (0, -6, 12))
obj_from_mb("pcb_head_backside_0201_terminals", mbc_t, M["tin"], "electronics", (0, -6, 12))

# ---- thermal path: DSP thermal pad -> copper spreader -> strap over the back of the head, down the neck -> spine -> wings
ribbon("thermal_strap_dsp_to_spine", [(0.0, HC.y + 0.5, Z_PAD_TOP + 0.25), (0.0, HC.y + 7.0, Z_PAD_TOP + 0.6),
                                       (0.0, HC.y + 11.4, HC.z + 6.5), (0.0, HC.y + 12.5, HC.z + 1.0), (0.0, -33.0, 74.0),
                                       (0.0, -27.5, 66.5), (0.0, -16.0, 64.0)], [XA] * 7, 6.0, 0, cname="fins",
       film="copper", thick=0.5, label="thermal strap (DSP -> wings)")
mb = MB().box(-5.0, 5.0, HC.y + 0.5 - 5.0, HC.y + 0.5 + 5.0, Z_PAD_TOP, Z_PAD_TOP + 0.5)
obj_from_mb("thermal_spreader_on_dsp", mb, M["copper"], "fins", (0, 0, 18))
for side in (-1, 1):
    S3w = wing_pt(8.0, -4.0)
    xw = side * (body_x_surface(S3w.y, S3w.z) + 0.6 + 0.38 * 8.3)
    ribbon("thermal_strap_spine_to_wing_%s" % ("right" if side < 0 else "left"),
           [(0.0, -16.0, 64.0), (side * 6.0, -17.0, 64.5), (side * 12.0, (S3w.y - 17.0) / 2, (S3w.z + 64.0) / 2 + 2.0),
            (xw, S3w.y, S3w.z)], [(0, 1, 0)] * 4, 5.0, 0, cname="fins", film="copper", thick=0.5)

# ================================================================== LEASH (pull tab) from under the tail
# Owner decision 2026-10-10: the pull tab comes out of the tail (stylized; on a real OSFP it is at the front/optical end).
# Strap 52.6 mm long from under the tail base; loop end ~20 mm past the tail tip (MSA App. B length not applied).
mb = MB()
ctrl = [(0.0, 34.0, 18.5), (0.0, 46.0, 12.5), (0.0, 62.0, 6.5), (0.0, 78.0, 3.2)]
pts = catmull(ctrl, 8)
# film v2: webbing cross-section cupped 0.6 mm, a mid-length twist up to 25 deg (zero at both ends so the root and the
# loop stay aligned) and a slight lengthwise wave (1.0 mm), all D (styling)
NW_, TT_ = 9, 0.4
top_, bot_ = [], []
for i, q in enumerate(pts):
    t_ = i / (len(pts) - 1)
    Pq = Vector(q)
    Tq = (Vector(pts[min(i + 1, len(pts) - 1)]) - Vector(pts[max(i - 1, 0)])).normalized()
    Nq = Vector((0.0, -Tq.z, Tq.y))
    if Nq.z < 0:
        Nq = -Nq
    Wq = Vector((1.0, 0.0, 0.0))
    tw = math.radians(25.0) * math.sin(math.pi * t_)
    Wr = Wq * math.cos(tw) + Nq * math.sin(tw)
    Nr = Nq * math.cos(tw) - Wq * math.sin(tw)
    Pq = Pq + Nq * (1.0 * math.sin(2 * math.pi * 1.5 * t_) * math.sin(math.pi * t_))
    for j in range(NW_):
        u_ = -1 + 2 * j / (NW_ - 1)
        c_ = Pq + Wr * (u_ * TAB_W / 2) + Nr * (0.6 * u_ * u_)
        top_.append(tuple(c_ + Nr * TT_ / 2))
        bot_.append(tuple(c_ - Nr * TT_ / 2))
nL = len(pts)
off_ = nL * NW_
fc_ = []
for i in range(nL - 1):
    for j in range(NW_ - 1):
        a0, a1, b0, b1 = i * NW_ + j, i * NW_ + j + 1, (i + 1) * NW_ + j, (i + 1) * NW_ + j + 1
        fc_.append([a0, a1, b1, b0])
        fc_.append([off_ + a0, off_ + b0, off_ + b1, off_ + a1])
    for j in (0, NW_ - 1):
        a0, b0 = i * NW_ + j, (i + 1) * NW_ + j
        fc_.append([a0, b0, off_ + b0, off_ + a0])
for i in (0, nL - 1):
    for j in range(NW_ - 1):
        a0, a1 = i * NW_ + j, i * NW_ + j + 1
        fc_.append([a0, off_ + a0, off_ + a1, a1])
mb.add(top_ + bot_, fc_)
leash = obj_from_mb("leash_pull_tab_strap", mb, HM["tab"], "leash", (0, 10, -4), smooth=True, label="leash (pull tab)")
yl, zl = pts[-1][1], pts[-1][2]
mb = MB()
for i in range(24):
    a0, a1 = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24
    P_ = lambda a, r: (r * math.cos(a), yl + 4.0 + r * math.sin(a))
    mb.prism([P_(a0, 3.0), P_(a1, 3.0), P_(a1, 4.6), P_(a0, 4.6)], zl - 0.2, zl + 0.2, "xy")
loop = obj_from_mb("leash_end_loop", mb, HM["tab"], "leash", (0, 10, -4))
HOOKS["leash_end"] = hook("HOOK_leash_end", (0.0, yl + 4.0 + 4.6, zl), loop, "leash", doc="end of the pull-tab loop behind the tail")

# ================================================================== explode offsets (pigeon frame, mm)
setex(SKIN["top"], (0, 0, 42))
setex(SKIN["bottom"], (24, 4, -12))
setex(objs("skin_panel_seams", "skin_screws", "skin_screw_sockets_top", "skin_screw_bosses_top"), (0, 0, 42))
setex(objs("skin_screws_belly", "skin_screw_sockets_bottom", "skin_screw_bosses_bottom"), (24, 4, -12))
# T / X subassemblies: explode set at import (group offset SUB_GROUP_EX + 0.7 x their own explode vectors, rotated into
# the pigeon frame); eye rings at import (s * 18, -4, -9)
setex(objs(*dsp_names), (0, -6, 20))
setex(objs("dsp_thermal_pad"), (0, -6, 26))
setex(objs("thermal_spreader_on_dsp"), (0, -6, 31))
setex(ICS["elec_mcu_ufqfpn48_7x7"][1], (0, -6, 8))
setex(objs("pcb_head_board"), (0, -6, 12))
setex(beak_objs + [lower], (0, -18, -2))
hinge["explode"] = [0, -18 * MM, -2 * MM]
setex(objs("pcb_sternum_board"), (0, 0, 0))
# @@PIGEON_STAGE@@
# ================================================================== finish: smoothing, export
bpy.context.view_layer.update()
# smooth curved surfaces, keep hard edges sharp (sets sharp edges + smooth faces, no modifier: glTF-clean)
meshes = [o for o in PARTS if o.type == "MESH" and o.get("subassy") is None]
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
os.makedirs(TEXDIR, exist_ok=True)
with open(os.path.join(TEXDIR, "decal_layout.json"), "w") as fh:
    json.dump(DECAL_LAYOUT, fh, indent=1)
print("DECALS layout=%d mapped=%d" % (len(DECAL_LAYOUT), sum(1 for o in PARTS if o.name in DECAL_LAYOUT)))
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
