"""datahall_environment: raised floor (600 mm tiles, perforated cold-aisle tiles, cut tiles over under-floor trays), under-floor and overhead
cable trays, ceiling grid with LED strips, cold-aisle containment (roof + sliding end doors), two rows of rack stubs, signage, floor markings.

Overrides (key=value after the output dir): L=20 (aisle length, m), racks=32 (per row, even), ceil=3.6 (ceiling height, m)
Run: Blender -b --python scripts/assets/datacenter/build_datahall_environment.py -- assets/components/datacenter
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_rack as R
import dc_parts as P
import common as C
import dc_common as D
from dc_common import MB
import bpy

OUT, KV = D.parse_args()
AID = "datahall_environment"
LEN = float(KV.get("L", 20.0))
NR = int(KV.get("racks", 2 * int(LEN / 1.2)))
NR += NR % 2
CEIL = float(KV.get("ceil", 3.6))
C.reset()
D.reset_mats()
L = D.lib()
coll, root = C.new_asset(AID, accuracy="B")
root["p_aisle_length"] = LEN
root["p_racks_per_row"] = NR
root["p_ceiling_height"] = CEIL
root["p_door_a"] = 0.0
root["p_door_b"] = 0.0
cs = {k: C.sub_collection(coll, k) for k in ("FLOOR", "UNDERFLOOR", "RACKS", "OVERHEAD", "CEILING", "CONTAINMENT", "SIGNAGE", "WALLS")}
TILE = 0.6
XH = math.ceil((LEN / 2 + 2.4) / 0.6) * 0.6   # hall half length (x), multiple of the tile pitch
YH = 4.2                    # hall half width (y)
NX, NY = int(round(2 * XH / TILE)), int(round(2 * YH / TILE))
tile_m = D.M("floor_tile", (0.62, 0.63, 0.62), 0.1, 0.55)
tile_edge = D.M("floor_tile_edge", (0.05, 0.05, 0.055), 0.5, 0.5)
conc = D.M("concrete", (0.32, 0.32, 0.31), 0.0, 0.9)
wall_m = D.M("wall_paint", (0.78, 0.8, 0.8), 0.0, 0.85)
ceil_m = D.M("ceiling_tile", (0.9, 0.9, 0.88), 0.0, 0.9)
tbar = D.M("tbar_white", (0.85, 0.85, 0.85), 0.5, 0.5)
led = D.M("led_strip", (1, 1, 1), 0.0, 0.3, (1, 0.97, 0.9), 12.0)
poly = D.M("polycarbonate_roof", (0.7, 0.85, 0.95), 0.0, 0.1, alpha=0.22)
yellow = D.M("paint_yellow", (0.95, 0.75, 0.05), 0.0, 0.5)
fiber_y = D.M("fiber_trough_yellow", (0.95, 0.75, 0.05), 0.0, 0.5)
sign_bg = D.M("sign_face", (0.02, 0.05, 0.12), 0.0, 0.5)
sign_tx = D.M("sign_text_emit", (1, 1, 1), 0.0, 0.5, (1, 1, 1), 3.0)
exit_tx = D.M("sign_exit_emit", (0.1, 1, 0.3), 0.0, 0.5, (0.1, 1, 0.3), 4.0)
perf = D.perforated_material("perforated_tile", (0.2, 0.21, 0.22), 0.8, 0.45, pitch=0.012, radius=0.0034, scale_axes=(1, 1, 0))
cable_blue = D.M("cable_blue", (0.05, 0.2, 0.8), 0.0, 0.4)

# ---------------- floor: tiles, perforated tiles, cut tiles, pedestals, slab
cut = set()
for ix in (-2.7, 3.9, -7.5):
    for iy in (-2.1, 2.1):
        cut.add((round(ix / 0.3), round(iy / 0.3)))      # tile centre indices in 0.3 m units
nt = MB()
pt = MB()
ped = MB()
for i in range(NX):
    for j in range(NY):
        cx = -XH + (i + 0.5) * TILE
        cy = -YH + (j + 0.5) * TILE
        key = (round(cx / 0.3), round(cy / 0.3))
        for sx in (-1, 1):
            for sy in (-1, 1):
                ped.cyl((cx + sx * TILE / 2, cy + sy * TILE / 2, -0.26), 0.025, 0.45, "z", 8, 0) if (sx > 0 and sy > 0) else None
        if key in cut:
            continue
        if abs(cy) < 0.6 and abs(cx) < LEN / 2 + 0.3:
            pt.box((cx, cy, -0.0175), (TILE - 0.006, TILE - 0.006, 0.035), 0)
        else:
            nt.box((cx, cy, -0.0175), (TILE - 0.006, TILE - 0.006, 0.035), 0)
ped.box((0, 0, -0.485), (0.001, 0.001, 0.001), 0)
for sx in range(NX + 1):
    pass
P.instantiate([P.Part("tiles", nt, [tile_m], bevel=(0.002, 1)), P.Part("tiles_perforated", pt, [perf]),
               P.Part("pedestals", ped, [L["steel"]])], cs["FLOOR"], root, AID + "_floor")
sl = MB()
sl.box((0, 0, -0.51), (2 * XH, 2 * YH, 0.02), 0)
P.instantiate([P.Part("subfloor_slab", sl, [conc])], cs["FLOOR"], root, AID + "_floor")
# under-floor stringers (grid at tile edges) so the void does not look empty through cut tiles
st = MB()
for i in range(NX + 1):
    x = -XH + i * TILE
    st.box((x, 0, -0.05), (0.02, 2 * YH, 0.016), 0)
for j in range(NY + 1):
    y = -YH + j * TILE
    st.box((0, y, -0.05), (2 * XH, 0.02, 0.016), 0)
P.instantiate([P.Part("stringers", st, [L["steel"]])], cs["FLOOR"], root, AID + "_floor")

# ---------------- under-floor cable baskets with cables (along x) at y = -2.1 and +2.1
ub = MB()
cb = MB()
cbl = MB()
for yy in (-2.1, 2.1):
    for dy in (-0.15, 0.15):
        ub.box((0, yy + dy, -0.2), (2 * XH - 0.2, 0.012, 0.1), 0)
    for k in range(int((2 * XH - 0.4) / 0.1)):
        ub.box((-XH + 0.2 + k * 0.1, yy, -0.25), (0.006, 0.3, 0.006), 0)
    for q, (r_, mi) in enumerate(((0.03, 0), (0.025, 1), (0.02, 0), (0.03, 1))):
        cb.cyl((0, yy - 0.1 + q * 0.065, -0.215 + (0.01 if q % 2 else 0)), r_, 2 * XH - 0.5, "x", 12, mi)
P.instantiate([P.Part("baskets", ub, [L["steel"]])], cs["UNDERFLOOR"], root, AID + "_underfloor")
P.instantiate([P.Part("cables", cb, [L["hose"], D.M("fiber_trough_yellow", (0.95, 0.75, 0.05), 0, 0.5)], smooth=True)], cs["UNDERFLOOR"], root, AID + "_underfloor")

# ---------------- racks: two rows facing the cold aisle
RW, RD = R.RACK_W * 0.001, R.RACK_D * 0.001
rowy = 0.6 + RD / 2
for side, sy in (("a", -1), ("b", 1)):
    for i in range(NR):
        x = (i - NR / 2 + 0.5) * RW
        R.rack_stub(cs["RACKS"], root, "%s_rack_%s%02d" % (AID, side, i), L, loc=(x, sy * rowy, 0),
                    rot=(0, 0, math.pi if sy < 0 else 0), cache_key="stub")
XR = NR * RW / 2

# ---------------- cold-aisle containment: roof panels + frame, sliding end doors
rz = R.RACK_H * 0.001 + 0.004
rf = MB()
pm = MB()
for i in range(NR):
    x = (i - NR / 2 + 0.5) * RW
    pm.box((x, 0, rz), (RW - 0.01, 1.4, 0.006), 0)
for i in range(NR + 1):
    x = (i - NR / 2) * RW
    rf.box((x, 0, rz), (0.03, 1.4, 0.03), 0)
for sy in (-1, 1):
    rf.box((0, sy * 0.7, rz), (2 * XR, 0.03, 0.03), 0)
P.instantiate([P.Part("roof_frame", rf, [L["alu"]], bevel=(0.001, 1)), P.Part("roof_panels", pm, [poly])],
              cs["CONTAINMENT"], root, AID + "_containment")
DH = 2.15
for tag, sx in (("a", -1), ("b", 1)):
    xe = sx * (XR + 0.05)
    fr = MB()
    for yy in (-0.62, 0.62):
        fr.box((xe, yy, DH / 2), (0.06, 0.06, DH), 0)
    fr.box((xe, 0, DH), (0.06, 1.3, 0.06), 0)
    fr.box((xe, 0, rz - 0.04 + 0.0), (0.06, 1.3, 0.05), 0) if False else None
    fr.box((xe, -0.675, (DH + R.RACK_H * 0.001) / 2), (0.06, 0.06, R.RACK_H * 0.001 - DH + 0.0), 0)
    fr.box((xe, 0.675, (DH + R.RACK_H * 0.001) / 2), (0.06, 0.06, R.RACK_H * 0.001 - DH + 0.0), 0)
    fp = MB()
    fp.box((xe, 0, (DH + R.RACK_H * 0.001) / 2 + 0.03), (0.01, 1.3, R.RACK_H * 0.001 - DH - 0.06), 0)
    P.instantiate([P.Part("end_frame", fr, [L["alu"]], bevel=(0.001, 1)), P.Part("end_transom_glass", fp, [poly])],
                  cs["CONTAINMENT"], root, "%s_door_%s" % (AID, tag))
    for lf, ys in (("l", -1), ("r", 1)):
        d = MB()
        d.box((0, 0, DH / 2 - 0.03), (0.04, 0.6, DH - 0.06), 0)
        gl = MB()
        gl.box((0, 0, DH / 2 - 0.03), (0.012, 0.52, DH - 0.18), 0)
        hd = MB()
        hd.box((sx * -0.04, -ys * 0.24, 1.05), (0.03, 0.025, 0.4), 0)
        o = P.instantiate([P.Part("frame", d, [L["alu"]], bevel=(0.001, 1)), P.Part("glass", gl, [poly]),
                           P.Part("handle", hd, [L["black"]])], cs["CONTAINMENT"], root, "%s_door_%s_%s" % (AID, tag, lf))
        # parent the 3 objects to a leaf empty so one driver moves them
        leaf = bpy.data.objects.new("%s_door_%s_%s_leaf" % (AID, tag, lf), None)
        cs["CONTAINMENT"].objects.link(leaf)
        leaf.parent = root
        leaf.location = (xe, ys * 0.3, 0)
        for oo in o.values():
            oo.parent = leaf
            oo.location = (0, 0, 0)
        D.drive(leaf, "location", 1, root, "p_door_" + tag, "%.2f * (0.3 + p * 0.55)" % ys)

# ---------------- ceiling grid + LED strips + hall walls
zc = CEIL
cg = MB()
ct = MB()
ct.box((0, 0, zc + 0.02), (2 * XH, 2 * YH, 0.02), 0)
for i in range(NX + 1):
    cg.box((-XH + i * TILE, 0, zc), (0.025, 2 * YH, 0.03), 0)
for j in range(NY + 1):
    cg.box((0, -YH + j * TILE, zc), (2 * XH, 0.025, 0.03), 0)
ls = MB()
for yy in (-2.27, 0.0, 2.27):
    k = 0
    xs = -LEN / 2 + 0.6
    while xs < LEN / 2:
        ls.box((xs, yy, zc - 0.03), (1.2, 0.07, 0.03), 0)
        xs += 2.4
P.instantiate([P.Part("ceiling_plane", ct, [ceil_m]), P.Part("tbar_grid", cg, [tbar])], cs["CEILING"], root, AID + "_ceiling")
P.instantiate([P.Part("led_strips", ls, [led])], cs["CEILING"], root, AID + "_ceiling")
wl = MB()
wl.box((-XH, 0, zc / 2 - 0.25), (0.1, 2 * YH, zc + 0.5), 0)
wl.box((XH, 0, zc / 2 - 0.25), (0.1, 2 * YH, zc + 0.5), 0)
wl.box((0, -YH, zc / 2 - 0.25), (2 * XH, 0.1, zc + 0.5), 0)
wl.box((0, YH, zc / 2 - 0.25), (2 * XH, 0.1, zc + 0.5), 0)
P.instantiate([P.Part("walls", wl, [wall_m])], cs["WALLS"], root, AID + "_hall")

# ---------------- overhead ladders (along x above each row, cross ladders), rods, cable bundles
oz = 2.62
ld = MB()
rd = MB()
for yy in (-rowy, rowy):
    for dy in (-0.15, 0.15):
        ld.box((0, yy + dy, oz), (2 * XR, 0.04, 0.06), 0)
    x = -XR + 0.15
    while x < XR:
        ld.box((x, yy, oz), (0.025, 0.3, 0.025), 0)
        x += 0.3
    x = -XR + 0.9
    while x < XR:
        for dy in (-0.15, 0.15):
            rd.cyl((x, yy + dy, (oz + zc) / 2), 0.004, zc - oz, "z", 6, 0)
        x += 3.0
xc = -XR + 3.0
while xc < XR - 2.0:
    for dx in (-0.15, 0.15):
        ld.box((xc + dx, 0, oz), (0.04, 2 * (rowy + 0.15) + 0.0, 0.06), 0)
    y = -rowy
    while y < rowy:
        ld.box((xc, y, oz), (0.3, 0.025, 0.025), 0)
        y += 0.3
    xc += 6.0
P.instantiate([P.Part("ladders", ld, [L["steel"]], bevel=(0.0005, 1)), P.Part("rods", rd, [L["steel"]])], cs["OVERHEAD"], root, AID + "_overhead")
ob = MB()
for yy in (-rowy, rowy):
    ob.cyl((0, yy - 0.05, oz + 0.05), 0.03, 2 * XR - 0.3, "x", 12, 0)
    ob.cyl((0, yy + 0.05, oz + 0.04), 0.022, 2 * XR - 0.3, "x", 12, 1)
P.instantiate([P.Part("overhead_cables", ob, [L["hose"], fiber_y], smooth=True)], cs["OVERHEAD"], root, AID + "_overhead")

# ---------------- signage and floor markings
def sign(name, body, loc, rot, w=1.2, h=0.3, mat=sign_tx):
    b = MB()
    b.box((0, 0, 0), (w, 0.02, h), 0)
    o = b.obj("%s_sign_%s" % (AID, name), [sign_bg])
    D.put(o, cs["SIGNAGE"], root, loc, rot)
    t = D.text("%s_signtext_%s" % (AID, name), body, h * 0.55, (0, -0.0115, 0), (math.pi / 2, 0, 0), mat, cs["SIGNAGE"], o, extrude=0.001)
    return o

sign("cold_a", "COLD AISLE A1", (-XR - 0.12, 0, 2.38), (0, 0, -math.pi / 2), 1.2, 0.3)
sign("cold_b", "COLD AISLE A1", (XR + 0.12, 0, 2.38), (0, 0, math.pi / 2), 1.2, 0.3)
sign("hot_l", "HOT AISLE", (XR + 0.4, -2.27, 2.5), (0, 0, math.pi / 2), 1.0, 0.25)
sign("hot_r", "HOT AISLE", (XR + 0.4, 2.27, 2.5), (0, 0, math.pi / 2), 1.0, 0.25)
sign("exit", "EXIT", (XH - 0.06, 3.3, 2.3), (0, 0, math.pi / 2), 0.6, 0.2, exit_tx)
fm = MB()
hz = MB()
for sy in (-1, 1):
    fm.box((0, sy * 2.55, 0.002), (2 * XR, 0.05, 0.004), 0)                  # safety line, hot aisle outer edge
    for sxx in (-1, 1):
        for k in range(8):
            hz.box((sxx * (XR + 0.35) + 0, sy * 0.3 - sy * 0.0 + (k - 3.5) * 0.0, 0.002), (0.0, 0.0, 0.0), 0)
for sxx in (-1, 1):
    for k in range(6):
        hz.box((sxx * (XR + 0.3), -0.55 + k * 0.22, 0.002), (0.3, 0.11, 0.004), 0)     # threshold hazard stripes (yellow)
P.instantiate([P.Part("safety_lines", fm, [yellow]), P.Part("threshold_stripes", hz, [yellow])], cs["SIGNAGE"], root, AID + "_markings")

# ---------------- hooks
C.hook("aisle_center", cs["SIGNAGE"], root, (0, 0, 0))
C.hook("eye_height_start_cold", cs["SIGNAGE"], root, (-XR + 0.8, 0, 1.65), (math.pi / 2, 0, -math.pi / 2 * 0 + math.pi / 2 * 0))
C.hook("eye_height_start_hot", cs["SIGNAGE"], root, (-XR + 0.8, -2.27, 1.65))
C.hook("door_a_center", cs["CONTAINMENT"], root, (-XR - 0.05, 0, 1.0))
C.hook("door_b_center", cs["CONTAINMENT"], root, (XR + 0.05, 0, 1.0))
C.hook("cut_tile_first", cs["UNDERFLOOR"], root, (-2.7, -2.1, 0))
C.hook("rack_row_a_start", cs["RACKS"], root, (-XR + RW / 2, -rowy, 0))
C.hook("rack_row_b_start", cs["RACKS"], root, (-XR + RW / 2, rowy, 0))
bpy.context.view_layer.update()
blend = os.path.join(OUT, AID + ".blend")
meta = dict(
    sources=[
        dict(what="Raised-floor tile 600 x 600 mm, 35 mm panel, pedestal void (common data-hall practice); ANSI/TIA-942 hot/cold aisle", url="https://en.wikipedia.org/wiki/Raised_floor", accessed="2026-10-02"),
        dict(what="Rack envelope 600 x 1068 x 2236 mm", url="https://www.supermicro.com/en/products/system/gpu/48u/srs-gb200-nvl72", accessed="2026-10-02"),
        dict(what="Hot/cold aisle containment layout: cold aisle ~1.2 m (2 tiles), hot aisles ~1.2 m", url="https://en.wikipedia.org/wiki/Hot_aisle/cold_aisle_containment", accessed="2026-10-02"),
    ],
    dimensions=[D.dim("floor tile", "600 x 600 (panel 594 x 594 x 35)", "mm", "raised-floor standard", "A"),
                D.dim("under-floor void", "~485 mm to slab", "mm", "estimate (typical 300-900)", "C"),
                D.dim("cold aisle width", 1200, "mm", "2 tiles", "B"), D.dim("hot aisle width", "~1200", "mm", "estimate", "C"),
                D.dim("aisle length", LEN, "m", "param L (default 20); racks per row %d x 600 mm = %.1f m" % (NR, NR * 0.6), "-"),
                D.dim("ceiling height", CEIL, "m", "param (typical 3-4 m)", "C"),
                D.dim("overhead ladder", "300 mm wide, 2.62 m AFF", "mm", "estimate", "C"),
                D.dim("containment door", "1.2 m wide x 2.15 m, two sliding leaves", "m", "estimate", "C")],
    simplifications=["racks are stubs (closed carcass + perforated door); one pair of rows only", "walls are plain boxes; no CRAH units, PDUs or busway",
                     "LED strips are emissive only (EEVEE does not light from them: add area lights)", "perforation is a shader (dithered alpha), not geometry"],
    hooks_doc={"HOOK_aisle_center": "origin marker", "HOOK_eye_height_start_cold / hot": "camera starts at eye height 1.65 m near the aisle start",
               "HOOK_door_a_center / b_center": "containment end doors", "HOOK_cut_tile_first": "first cut floor tile over an under-floor tray",
               "HOOK_rack_row_a_start / b_start": "first rack of each row"},
    custom_properties_doc={"p_door_a / p_door_b": "end-door opening 0..1 (drivers move the two sliding leaves)",
                           "p_aisle_length / p_racks_per_row / p_ceiling_height": "build-time parameters (rebuild with L=, racks=, ceil=)"},
    origin="aisle centre on the floor (z=0 = tile tops); aisle along x, cold aisle centred on y=0", intended_usage="S1 hook / wide establishing; eye-height walk",
)
D.use_visible_bbox()
C.finish(AID, blend, coll, meta)
D.extend_meta(blend, dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll), custom_properties=D.custom_props(root)))
pv = os.path.join(OUT, "previews")
lights = [((x, y, zc - 0.3), 90, 1.5) for x in (-8, -3, 2, 7) for y in (-2.27, 0.0, 2.27)]
for nm, loc, tgt, lens in (
        ("eye_cold_aisle", (-XR + 0.7, 0, 1.65), (XR, 0, 1.4), 28),
        ("eye_hot_aisle", (-XR + 0.7, -2.27, 1.65), (XR, -2.27, 1.4), 28),
        ("three_quarter", (-XH * 0.75, -YH * 0.8, 6.0), (2, 0, 1.0), 26),
        ("top", (0, -0.1, 24), (0, 0, 0), 35),
        ("front", (0, -9, 1.7), (0, 0, 1.4), 28),
        ("cut_tile_closeup", (-2.0, -3.4, 1.0), (-2.7, -2.1, -0.15), 35)):
    if nm in ('top', 'three_quarter'):
        cs['CEILING'].hide_render = True
    else:
        cs['CEILING'].hide_render = False
    D.render_views(coll, pv, AID, [dict(name=nm, loc=loc, target=tgt, lens=lens)], floor=False, lights=lights, sun=0.0, world=0.15,
                   clip=(0.05, 200))
