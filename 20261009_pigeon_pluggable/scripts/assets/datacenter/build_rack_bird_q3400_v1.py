"""FROZEN v1 (2026-10-10), superseded by build_rack_bird_q3400.py v2 (tiered loft). rack_bird_q3400_v1: bird switch rack (stylized Quantum-X800 Q3400-RA look) with pigeonhole cells for the 2.6x cyber-pigeons.

800 W x 1068 D x 2236 H mm network cabinet (depth/height matched to the GB300 rack). Top: true-scale 1U patch panel, 1U
management switch, 1U horizontal manager. Below: three bird-scale Q3400-style chassis (one block row = 2 x 3 portrait
cells each, 18 cells), true-scale vertical finger cable managers on both sides, rear PDUs and power cords. Rack top: landing
pad with charging coil and a charging/perch rail. Sources: research/racks_refs.md sections 3-5. Scale: DevLog-007 section 3.
Frame: origin floor centre of the footprint, FRONT = +X, width along Y (+Y = right seen from the front), Z up.
Hooks: HOOK_cell_<unit>_<row>_<col> (unit 1..3 top->bottom, row 0 upper / 1 lower, col 0..2 left->right seen from the
front); origin = bird feet origin on the cell floor, 60 mm behind the faceplate; +X = out of the cell. HOOK_landing_pad,
HOOK_perch_<i>. Custom props on cell hooks: fill_index (0 = first filled, top-left), unit, row, col, cell_label.
Run: scripts/bslot.sh -b --python scripts/assets/datacenter/build_rack_bird_q3400.py -- assets/components/datacenter
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import rk_lib as K  # noqa: E402
import rk_q3400 as Q  # noqa: E402
from rk_lib import MB, Part  # noqa: E402

OUT, KV = K.parse_args()
AID = "rack_bird_q3400_v1"   # frozen v1 (2026-10-10); renamed so it never overwrites v2
PREV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(OUT)), "..", "..", "outputs", "assets", "datacenter"))
m = lambda x: x * 0.001

W, D, H = 800.0, 1068.0, 2236.0
U1, UH = 82.0, 44.45
TOP = H - 20.0                       # top frame underside
FACE = 505.0                         # chassis faceplate front plane (x)
FEET_IN = 60.0                       # bird feet origin behind the faceplate (DevLog-004: birds stand 6 cm inside)
S = Q.BIRD
CH_TOP = TOP - 3 * UH                # top of chassis 1 (below 3U of true-scale units)
CH_Z = [CH_TOP - (u + 1) * S["H"] for u in range(3)]   # chassis bottoms, unit 1 (top) .. 3
MGR_Y0, MGR_Y1 = 354.0, 392.0        # finger manager band |y| (Eaton SRCABLEVRTL class, 2 in wide)
S_BIRD = 2.6

K.reset()
L = K.palette()
L["pad_glow"] = K.M("pad_glow_cyan", (0.1, 0.9, 1.0), 0.0, 0.3, (0.1, 0.9, 1.0), 12.0)
L["pad_amber"] = K.M("pad_glow_amber", (1.0, 0.55, 0.05), 0.0, 0.3, (1.0, 0.55, 0.05), 6.0)
L["deck"] = K.M("deck_antislip", (0.035, 0.036, 0.04), 0.2, 0.8)
coll, root = K.new_asset(AID, accuracy="C")
cc = {k: K.sub(coll, k) for k in ("FRAME", "SIDE_PANELS", "CHASSIS", "TOP_UNITS", "CABLE_MGMT", "CABLES", "REAR", "LANDING_PAD")}


def fb(mb, x0, x1, y0, y1, z0, z1, mi=0):
    mb.box2((m(x0), m(y0), m(z0)), (m(x1), m(y1), m(z1)), mi)


# ------------------------------------------------------------------ cabinet frame
fr = MB()
hw, hd = W / 2, D / 2
fb(fr, -hd, hd, -hw, hw, 0, U1 - 4)                                       # plinth
fb(fr, -hd, hd, -hw, hw, TOP, H)                                          # top frame
for sy in (-1, 1):
    fb(fr, hd - 30, hd, sy * (hw - 6), sy * hw, U1, TOP)                  # thin front stiles
    fb(fr, -hd, -hd + 44, sy * (hw - 44), sy * hw, U1, TOP)               # rear posts
    fb(fr, -hd, hd, sy * (hw - 40), sy * hw, U1, U1 + 30)                 # bottom side rails
    fb(fr, -hd, hd, sy * (hw - 40), sy * hw, TOP - 30, TOP)               # top side rails
    fb(fr, FACE - 860, FACE - 830, sy * (hw - 52), sy * (hw - 44), U1, TOP)   # chassis rear support rails
    fb(fr, FACE - 30, FACE - 4, sy * 350, sy * (MGR_Y0 - 1), U1, TOP)     # chassis front mounting rails
rf = MB()
fb(rf, -hd + 40, hd - 40, -hw + 40, hw - 40, H - 1, H + 0.5)              # perforated roof skin
sp = MB()
for sy in (-1, 1):
    fb(sp, -hd + 4, hd - 4, sy * (hw - 3), sy * hw, U1, TOP)
K.instantiate([Part("frame", fr, [L["frame"]], bevel=0.0012), Part("roof", rf, [L["perf_black"]], uv_tile=0.012)],
              cc["FRAME"], root, AID + "_frame")
K.instantiate([Part("side_panels", sp, [L["panel"]])], cc["SIDE_PANELS"], root, AID + "_side")

# ------------------------------------------------------------------ bird chassis x3
cells_out = []
fill = 0
for u in range(3):
    name = "%s_chassis_u%d" % (AID, u + 1)
    e = K.empty(name, cc["CHASSIS"], root, (m(FACE), 0, m(CH_Z[u])))
    parts, lay, labels = Q.chassis(S, L, port_offset=u * 3)
    K.instantiate(parts, cc["CHASSIS"], e, name, key="bird_chassis")
    bpy.context.view_layer.update()
    Q.label_texts(S, L, labels, cc["CHASSIS"], e, name, "NVYDIA\nQX-3400B")   # parody badge (owner rule)
    for c in sorted(lay["cells"], key=lambda c: (c["row"], c["col"])):
        yc = (c["y0"] + c["y1"]) / 2
        zf = CH_Z[u] + c["z0"]
        label = 2 * (u * 3 + c["col"]) + 1 + c["row"]
        h = K.hook("cell_%d_%d_%d" % (u + 1, c["row"], c["col"]), cc["CHASSIS"], root, (m(FACE - FEET_IN), m(yc), m(zf)),
                   fill_index=fill, unit=u + 1, row=c["row"], col=c["col"], cell_label=label,
                   cell_opening_m=[round(m(c["y1"] - c["y0"]), 4), round(m(c["z1"] - c["z0"]), 4), round(m(S["cell_depth"]), 4)])
        cells_out.append(dict(hook=h.name, fill_index=fill, unit=u + 1, row=c["row"], col=c["col"], cell_label=label,
                              feet_origin_m=[round(m(FACE - FEET_IN), 4), round(m(yc), 4), round(m(zf), 4)],
                              opening_center_m=[round(m(FACE), 4), round(m(yc), 4), round(m(zf + (c["z1"] - c["z0"]) / 2), 4)]))
        fill += 1

# ------------------------------------------------------------------ true-scale top units (19 in) + bottom brush panel
BW, EW = 440.0, 482.6


def unit_patch_panel():
    mb = MB()   # 0 black, 1 adapter blue, 2 port hole, 3 label
    fb(mb, -2, 0, -BW / 2, BW / 2, 0.5, UH - 0.5, 0)
    for s in (-1, 1):
        fb(mb, -2.5, 0.5, *sorted((s * BW / 2, s * EW / 2)), 1, UH - 1, 0)
    for g in range(4):
        for k in range(6):
            y = -180 + g * 96 + k * 13.6
            fb(mb, -6, 1.5, y - 6.0, y + 6.0, 16, 28, 1)
            fb(mb, -5, 1.7, y - 4.6, y - 0.6, 18.5, 25.5, 2)
            fb(mb, -5, 1.7, y + 0.6, y + 4.6, 18.5, 25.5, 2)
        fb(mb, -0.5, 0.6, -186 + g * 96, -186 + g * 96 + 80, 6, 11, 3)
    return [Part("front", mb, [L["black"], L["qd_blue"], L["port_hole"], L["label"]])]


def unit_mgmt_switch():
    mb = MB()   # 0 black, 1 rj45, 2 hole, 3 led, 4 nickel
    fb(mb, -2, 0, -BW / 2, BW / 2, 0.5, UH - 0.5, 0)
    for s in (-1, 1):
        fb(mb, -2.5, 0.5, *sorted((s * BW / 2, s * EW / 2)), 1, UH - 1, 0)
    for g in range(3):
        for c in range(8):
            for r in range(2):
                y = -196 + g * 104 + c * 12.4
                z = 12 + r * 17
                fb(mb, -3, 0.8, y - 5.6, y + 5.6, z - 7, z + 7, 1)
                fb(mb, -2, 1.0, y - 4.6, y + 4.6, z - 5.5, z + 4.0, 2)
    for k in range(4):
        y = 128 + k * 21
        fb(mb, -6, 1.5, y - 9.75, y + 9.75, 15.75, 26.25, 4)
        fb(mb, -5, 1.7, y - 8.5, y + 8.5, 16.75, 25.25, 2)
    for k in range(4):
        fb(mb, -0.5, 1.0, 210, 213, 10 + k * 7, 12.5 + k * 7, 3)
    return [Part("front", mb, [L["black"], L["rj45"], L["port_hole"], L["led_g"], L["nickel"]])]


def unit_brush():
    mb = MB()   # 0 black, 1 brush grey, 2 steel
    fb(mb, -2, 0, -BW / 2, BW / 2, 0.5, UH - 0.5, 0)
    for s in (-1, 1):
        fb(mb, -2.5, 0.5, *sorted((s * BW / 2, s * EW / 2)), 1, UH - 1, 0)
    fb(mb, -1, 0.8, -200, 200, 14, 30, 1)
    return [Part("front", mb, [L["black"], L["grey"], L["steel"]])]


def filler(z0, z1):
    mb = MB()
    for s in (-1, 1):
        fb(mb, FACE - 2, FACE, *sorted((s * EW / 2, s * (MGR_Y0 - 1))), z0, z1, 0)
    return mb


top_units = [("patch_panel", unit_patch_panel(), TOP - UH), ("mgmt_switch", unit_mgmt_switch(), TOP - 2 * UH),
             ("horizontal_manager", unit_brush(), TOP - 3 * UH), ("bottom_brush", unit_brush(), CH_Z[2] - UH)]
for nm, parts, z in top_units:
    e = K.empty("%s_%s" % (AID, nm), cc["TOP_UNITS"], root, (m(FACE), 0, m(z)))
    K.instantiate(parts, cc["TOP_UNITS"], e, "%s_%s" % (AID, nm), key=nm if nm != "bottom_brush" else "horizontal_manager")
    body = MB()
    fb(body, -420, -2, -BW / 2, BW / 2, 1, UH - 1, 0)
    K.instantiate([Part("body", body, [L["dark_steel"]])], cc["TOP_UNITS"], e, "%s_%s_body" % (AID, nm), key="unit_body")
fl = filler(TOP - 3 * UH, TOP)
fl.extend(filler(CH_Z[2] - UH, CH_Z[2]))
fl2 = MB()
fb(fl2, FACE - 2, FACE, -MGR_Y0 + 1, MGR_Y0 - 1, U1, CH_Z[2] - UH, 0)
fl.extend(fl2)
K.instantiate([Part("fillers", fl, [L["perf_black"]], uv_tile=m(9))], cc["TOP_UNITS"], root, AID + "_fillers")

# ------------------------------------------------------------------ vertical finger cable managers (both sides)
mg = MB()   # 0 black plastic, 1 dark steel spine
for sy in (-1, 1):
    y0, y1 = sorted((sy * MGR_Y0, sy * MGR_Y1))
    fb(mg, 420, 432, y0, y1, U1 + 10, TOP - 10, 1)                         # spine
    n = int((TOP - U1 - 40) / UH)
    for k in range(n + 1):
        z = U1 + 20 + k * UH
        fb(mg, 432, 516, y0 + 2, y1 - 2, z - 4.5, z + 4.5, 0)               # finger
        fb(mg, 512, 524, y0, y1, z - 7, z + 7, 0)                           # finger tip
K.instantiate([Part("finger_managers", mg, [L["black"], L["dark_steel"]], bevel=0.0015)], cc["CABLE_MGMT"], root,
              AID + "_cable_mgr")

# ------------------------------------------------------------------ cables: mgmt RJ-45 of each chassis -> right manager -> mgmt switch
lay0 = Q.layout(S)
yrc = (lay0["groups_y"][-1][1] + S["W"] / 2) / 2
uz_rel = S["H"] * 0.46
mg_y = (MGR_Y0 + MGR_Y1) / 2
z_sw = TOP - 2 * UH + 21
for u in range(3):
    for k, dz in enumerate((7.5, -7.5)):
        z = CH_Z[u] + uz_rel + dz
        xo = 470 + 9 * k + 6 * u
        pts = [(m(FACE + 3), m(yrc + 9), m(z)), (m(FACE + 22), m(yrc + 12), m(z - 6)), (m(xo + 25), m(mg_y), m(z + 30)),
               (m(xo), m(mg_y), m(z + 160)), (m(xo), m(mg_y), m(z_sw - 120)), (m(FACE + 26), m(300 - 12 * (2 * u + k)), m(z_sw)),
               (m(FACE + 3), m(116 + 12.4 * (2 * u + k)), m(z_sw - 8.5))]
        K.cable("%s_cable_mgmt_u%d_%d" % (AID, u + 1, k), pts, m(3.0), L["cable_blue"], cc["CABLES"], root)
# yellow simplex patch cords from the patch panel into the left manager (decorative)
for k in range(5):
    y = -180 + k * 13.6
    z = TOP - UH + 22
    xo = 470 + 8 * k
    pts = [(m(FACE + 4), m(y), m(z)), (m(FACE + 28), m(y - 20), m(z - 10)), (m(xo + 20), m(-mg_y), m(z - 60)),
           (m(xo), m(-mg_y), m(z - 300 - 80 * k)), (m(xo), m(-mg_y), m(U1 + 60))]
    K.cable("%s_fiber_patch_%d" % (AID, k), pts, m(1.5), L["cable_yellow"], cc["CABLES"], root)

# ------------------------------------------------------------------ rear: PDUs and power cords
rr = MB()   # 0 black, 1 outlet black-grey, 2 led
for sy in (-1, 1):
    y0, y1 = sorted((sy * 330, sy * 380))
    fb(rr, -hd + 40, -hd + 90, y0, y1, 200, 2000, 0)
    for k in range(30):
        z = 260 + k * 56
        fb(rr, -hd + 37, -hd + 40, y0 + 8, y1 - 8, z, z + 30, 1)
    fb(rr, -hd + 37, -hd + 40, y0 + 18, y1 - 18, 1940, 1960, 2)
K.instantiate([Part("pdus", rr, [L["black"], L["grey"], L["led_b"]])], cc["REAR"], root, AID + "_rear")
psu_w = S["W"] * 0.12
ph = (S["H"] - 8) / S["psu_rows"]
xr = FACE - S["D"]
for u in range(3):
    for sy in (-1, 1):
        for i in range(S["psu_rows"]):
            yi = sy * (S["W"] / 2 - 2 - psu_w / 2) - sy * psu_w * 0.12
            z = CH_Z[u] + 4 + i * ph + ph / 2
            zt = 260 + int((z - 200) / 56) * 56 + 15
            pts = [(m(xr - 6), m(yi), m(z)), (m(xr - 60), m(yi), m(z - 20)), (m(-hd + 120), m(sy * 345), m(zt - 40)),
                   (m(-hd + 42), m(sy * 355), m(zt))]
            K.cable("%s_power_cord_u%d_%s%d" % (AID, u + 1, "R" if sy > 0 else "L", i), pts, m(4.5), L["cable_black"], cc["REAR"],
                    root)

# ------------------------------------------------------------------ landing pad + charging/perch rail (rack top)
PAD_X0, PAD_X1 = -80.0, 520.0
PAD_C = ((PAD_X0 + PAD_X1) / 2, 0.0)
ZD = H + 20.0
dk = MB()
fb(dk, PAD_X0, PAD_X1, -380, 380, H, ZD, 0)
for xx in (PAD_X0 + 40, PAD_X1 - 40):                                    # deck feet rails
    for sy in (-1, 1):
        fb(dk, xx - 20, xx + 20, sy * 360 - 20, sy * 360 + 20, H - 1, H + 1, 1)
glow = MB()
cx, cy = PAD_C
ring = [(m(cx + 200 * math.cos(a)), m(cy + 200 * math.sin(a)), m(ZD + 2)) for a in [2 * math.pi * i / 48 for i in range(48)]]
glow.tube(ring, m(5), 8, 0, closed=True)
for sx, sy in ((1, 0), (0, 1)):
    for sgn in (-1, 1):
        a0, a1 = 225, 290
        glow.box2((m(cx + sgn * sx * a0 - (4 if sy else 0)), m(cy + sgn * sy * a0 - (4 if sx else 0)), m(ZD)),
                  (m(cx + sgn * sx * a1 + (4 if sy else 0)), m(cy + sgn * sy * a1 + (4 if sx else 0)), m(ZD + 2.5)), 1)
for sy in (-1, 1):                                                       # edge LED strips
    glow.box2((m(PAD_X0 + 10), m(sy * 372 - 4), m(ZD)), (m(PAD_X1 - 10), m(sy * 372 + 4), m(ZD + 2)), 0)
for xx in (PAD_X0 + 6, PAD_X1 - 6):
    glow.box2((m(xx - 4), m(-360), m(ZD)), (m(xx + 4), m(360), m(ZD + 2)), 0)
for k in range(2):                                                       # approach chevrons pointing +X (aisle)
    x0 = PAD_X1 - 120 + k * 45
    for sy in (-1, 1):
        a = (m(x0), m(sy * 70), m(ZD + 0.5))
        b = (m(x0 + 40), m(0), m(ZD + 0.5))
        glow.tube([a, b], m(5), 6, 1)
coil = MB()
for r in (40, 58, 76, 94):
    coil.tube([(m(cx + r * math.cos(a)), m(cy + r * math.sin(a)), m(ZD + 1.5)) for a in [2 * math.pi * i / 36 for i in range(36)]],
              m(3), 6, 0, closed=True)
coil.cyl((m(cx), m(cy), m(ZD + 1)), m(28), m(2.5), "z", 24, 1)
K.instantiate([Part("deck", dk, [L["deck"], L["dark_steel"]], bevel=0.002)], cc["LANDING_PAD"], root, AID + "_landing")
K.instantiate([Part("glow", glow, [L["pad_glow"], L["pad_amber"]])], cc["LANDING_PAD"], root, AID + "_pad")
K.instantiate([Part("charge_coil", coil, [L["copper"], L["gold"]], smooth=True)], cc["LANDING_PAD"], root, AID + "_pad")
# charging / perch rail along the front edge: 3 lanes (front brass = GND for the claws, gold = power, gold = signal)
RX, RZ = 492.0, H + 95.0
rl = MB()
fb(rl, RX - 18, RX + 18, -372, 372, RZ - 22, RZ, 0)
for sy in (-1, 1):
    fb(rl, RX - 10, RX + 10, sy * 210 - 10, sy * 210 + 10, ZD, RZ - 22, 0)      # posts between perch spots
lanes = MB()
for k, (x0, x1) in enumerate(((RX + 6, RX + 16), (RX - 5, RX + 5), (RX - 16, RX - 6))):
    fb(lanes, x0, x1, -366, 366, RZ, RZ + 0.8, 0 if k == 0 else 1)
perch_y = [-315.0, -105.0, 105.0, 315.0]                 # 0.21 m spacing: 0.13 m wide birds stay 0.08 m apart
for y in perch_y:
    fb(lanes, RX + 18, RX + 19.2, y - 6, y + 6, RZ - 14, RZ - 8, 2)
K.instantiate([Part("perch_rail", rl, [L["dark_steel"]], bevel=0.0015),
               Part("perch_lanes", lanes, [L["brass"], L["gold"], L["pad_glow"]])], cc["LANDING_PAD"], root, AID + "_rail")
K.hook("landing_pad", cc["LANDING_PAD"], root, (m(cx), m(cy), m(ZD)), note="deck top centre; +X faces the aisle")
for i, y in enumerate(perch_y):
    K.hook("perch_%d" % i, cc["LANDING_PAD"], root, (m(RX), m(y), m(RZ + 0.8)), note="rail top; feet origin; +X faces the aisle")
K.hook("rack_front_center", cc["FRAME"], root, (m(hd), 0, m(1100)))
K.hook("rack_top_center", cc["FRAME"], root, (0, 0, m(H)))
bpy.context.view_layer.update()

# ------------------------------------------------------------------ save, export, meta
os.makedirs(OUT, exist_ok=True)
bb = K.bbox(coll)
meta = dict(
    asset_id=AID, date="2026-10-10", units="m",
    frame="origin floor centre of footprint; FRONT = +X; width along Y (+Y = right seen from the front); Z up. glTF: Y-up",
    size_m=[round(bb[1][i] - bb[0][i], 4) for i in range(3)], bbox_m=bb, triangles=K.count_tris(coll),
    cabinet_mm=dict(W=W, D=D, H=H), chassis=dict(count=3, size_mm=[S["W"], S["H"], S["D"]], z_bottom_mm=[round(z, 2) for z in CH_Z],
                                                faceplate_x_mm=FACE),
    cells=dict(count=len(cells_out), per_chassis="2 rows x 3 cols", opening_mm=[S["cell_w"], S["cell_h"], S["cell_depth"]],
               fill_order="fill_index 0..17: unit 1 (top) row 0 col 0..2, row 1, unit 2 ..., unit 3 row 1 col 2 last (top to bottom, left to right seen from the front)",
               feet_origin="cell floor, 60 mm behind the faceplate, centred in width; bird +x (beak) = hook +X (out of the cell)",
               list=cells_out),
    bird_envelope_m=dict(scale=S_BIRD, beak_to_tail=0.32, height=0.245, width=0.13, beak_ahead_of_feet=0.16, tail_behind_feet=0.17,
                         source="crude film bird primitive (scripts/build_crude_film.py, DevLog-004) x 2.6"),
    landing_pad=dict(hook="HOOK_landing_pad", deck_m=[round(m(PAD_X1 - PAD_X0), 3), 0.76], deck_top_z_m=m(ZD),
                     glow_objects=[AID + "_pad_glow (material MAT_dc_pad_glow_cyan / MAT_dc_pad_glow_amber: animate emission for charging)"],
                     perch_hooks=["HOOK_perch_%d" % i for i in range(len(perch_y))],
                     rail_lanes="front brass = GND (claws), middle gold = power (palm pads), rear gold = signal (toe pads)"),
    sources="research/racks_refs.md sections 3-5",
    accuracy=dict(cabinet="C (800 mm network cabinet; depth/height matched to the GB300 rack)", chassis_front_grammar="B (Q3400-RA figure)",
                  chassis_scale="stylized (DevLog-007 section 3)", top_units="C (true-scale 1U generic)", managers="B (2 in finger manager class)"),
    collections=dict(FRAME="cabinet", SIDE_PANELS="side skins", CHASSIS="3 bird chassis + cell hooks", TOP_UNITS="patch panel, mgmt switch, brush panels, fillers",
                     CABLE_MGMT="vertical finger managers", CABLES="mgmt Cat6 + fibre patch cords", REAR="PDUs + power cords",
                     LANDING_PAD="deck, glow ring, coil, perch rail + hooks"),
    simplifications=["no logos (parody badge NVYDIA QX-3400B)", "cells are portrait (stylized); real OSFP cages are landscape", "chassis internals not modelled",
                     "cage lip latch (beak bite bar) not modelled on crowd cells"],
    hooks=K.hooks_of(coll),
)
K.save_blend(os.path.join(OUT, AID + ".blend"))
K.export_glb(coll, os.path.join(OUT, AID + ".glb"))
K.write_json(os.path.join(OUT, AID + ".json"), meta)
print("ASSET", AID, "tris", meta["triangles"], "size_m", meta["size_m"], "cells", len(cells_out))


# ------------------------------------------------------------------ previews (bird envelope proxies, preview only, not saved)
if int(KV.get("previews", 1)):
    os.makedirs(PREV, exist_ok=True)
    bp = MB()
    for c in cells_out:
        K.bird_proxy(bp, c["feet_origin_m"])
    K.bird_proxy(bp, (m(PAD_C[0]), m(PAD_C[1]), m(ZD)))
    for y in (perch_y[0], perch_y[3]):
        K.bird_proxy(bp, (m(RX), m(y), m(RZ + 0.8)))
    bm = K.bird_mats()
    pc = bpy.data.collections.new("PREVIEW_birds")
    bpy.context.scene.collection.children.link(pc)
    K.obj_from_mb("PREVIEW_bird_proxies", bp, bm, pc, smooth=True)
    K.preview_setup(res=(900, 1125), samples=16, world=0.25, exposure=-0.3)
    K.floor(20)
    K.light("key", "AREA", (3.4, -1.8, 3.0), 220, 2.0, (0, 0, 1.1))
    K.light("fill", "AREA", (2.8, 2.4, 1.4), 90, 2.0, (0, 0, 1.0))
    K.light("rear", "AREA", (-3.2, 1.6, 2.8), 220, 2.0, (0, 0, 1.1))
    K.light("rear_fill", "AREA", (-2.8, -2.2, 1.4), 70, 2.0, (0, 0, 1.0))
    V = lambda n: os.path.join(PREV, "%s_%s.png" % (AID, n))
    K.render_view(V("front"), (5.0, 0, 1.22), (0, 0, 1.22), ortho=2.7)
    K.render_view(V("three_quarter"), (2.9, -2.0, 1.8), (0, 0, 1.15), lens=32)
    K.render_view(V("rear"), (-5.0, 0, 1.22), (0, 0, 1.22), ortho=2.7)
    K.render_view(V("three_quarter_rear"), (-2.8, 2.0, 1.7), (0, 0, 1.1), lens=32)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 1200, 800
    K.render_view(V("cells_closeup"), (1.35, 0.35, 1.55), (0.45, 0, 1.45), lens=35)
    K.render_view(V("landing_pad"), (1.5, -0.9, 3.0), (0.2, 0, 2.3), lens=35)
    pc.hide_render = True
    K.render_view(V("cells_empty_closeup"), (1.35, 0.35, 1.55), (0.45, 0, 1.45), lens=35)
