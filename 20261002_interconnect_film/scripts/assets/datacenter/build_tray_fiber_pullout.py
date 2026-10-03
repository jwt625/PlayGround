"""tray_fiber_pullout: S6 hardware. Rack stub with a 2U pull-out fiber management drawer: slide rails with stops, spools, routing guides,
splice cassettes, LC/MPO patch panel at the front, folding cable management arm. HOOK_slide (0 closed .. 1 open) driven by root p_slide.

Run: Blender -b --python scripts/assets/datacenter/build_tray_fiber_pullout.py -- assets/components/datacenter
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_parts as P
import common as C
import dc_common as D
from dc_common import MB
import bpy

OUT, KV = D.parse_args()
AID = "tray_fiber_pullout"
C.reset()
D.reset_mats()
L = D.lib()
m = lambda v: v * 0.001
coll, root = C.new_asset(AID, accuracy="C")
T = 0.55   # slide travel (m); estimate for a 600 mm class rail
root["p_slide"] = 0.0
cs = {k: C.sub_collection(coll, k) for k in ("RACK_STUB", "DRAWER_moving", "SLIDES", "CMA", "VARIANT_lid", "FIBERS")}
aqua = D.M("fiber_aqua", (0.0, 0.7, 0.75), 0.0, 0.35)
yellow = D.M("fiber_yellow", (0.95, 0.75, 0.05), 0.0, 0.4)
orange = D.M("fiber_orange", (0.95, 0.4, 0.05), 0.0, 0.4)
lc_blue = D.M("lc_blue", (0.05, 0.25, 0.85), 0.0, 0.35)
apc_green = D.M("apc_green", (0.1, 0.65, 0.15), 0.0, 0.35)
clear = D.M("clear_plastic", (0.8, 0.9, 0.95), 0.0, 0.1, alpha=0.18)

# ------------------------------------------------ rack stub (19 in, 4 posts, 1000 mm deep, 1500 mm high)
RW, RD, RH = 600.0, 1000.0, 1500.0
fr = MB()
for sx in (-1, 1):
    for sy in (-1, 1):
        fr.boxmm((sx * (RW / 2 - 18), sy * (RD / 2 - 18), RH / 2 + 40), (36, 36, RH - 80), 0)
for z in (60, RH - 20):
    fr.boxmm((0, -RD / 2 + 18, z), (RW, 36, 40), 0)
    fr.boxmm((0, RD / 2 - 18, z), (RW, 36, 40), 0)
    for sx in (-1, 1):
        fr.boxmm((sx * (RW / 2 - 18), 0, z), (36, RD - 72, 40), 0)
fr.boxmm((0, 0, 35), (RW - 8, RD - 8, 10), 1)          # base plate
for sx in (-1, 1):
    fr.boxmm((sx * (RW / 2 - 1), 0, RH / 2 + 40), (2, RD - 40, RH - 90), 2)    # side panels
fr.boxmm((0, RD / 2 - 1, RH / 2 + 40), (RW - 40, 2, RH - 90), 2)              # back panel
parts = [P.Part("frame", fr, [L["frame"], L["dark_steel"], L["panel"]], bevel=(0.0012, 1))]
cs_ = MB()
for sx in (-1, 1):
    for sy in (-1, 1):
        cs_.cyl((sx * 0.25, sy * 0.4, 0.035), 0.035, 0.026, "x", 16, 0)
parts.append(P.Part("casters", cs_, [L["hose"]]))
rm = D.perforated_material("ru_rail_holes", (0.55, 0.56, 0.58), 0.9, 0.4, pitch=0.01588, radius=0.0035, scale_axes=(0, 0, 1))
for sx in (-1, 1):
    r = MB()
    r.boxmm((sx * 245.0, -RD / 2 + 70, RH / 2 + 40), (30, 2.5, RH - 120))
    parts.append(P.Part("ru_rail_%s" % ("L" if sx < 0 else "R"), r, [rm]))
P.instantiate(parts, cs["RACK_STUB"], root, AID + "_stub")
# neighbouring 1U blanks and a 1U LC patch panel (static) above and below the drawer
zU0 = 700.0                    # bottom of the drawer (mm); 2U = 88.9 mm
front_y = -RD / 2 + 70
bl = MB()
bl.boxmm((0, front_y - 1.5, zU0 - 22.2), (482.6, 3, 42.5), 0)
bl.boxmm((0, front_y - 1.5, zU0 - 66.7), (482.6, 3, 42.5), 0)
bl.boxmm((0, front_y - 1.5, zU0 + 88.9 + 22.2), (482.6, 3, 42.5), 0)
P.instantiate([P.Part("blanks", bl, [L["panel"]], bevel=(0.0006, 1))], cs["RACK_STUB"], root, AID + "_stub")

# ------------------------------------------------ drawer (moving) : parent HOOK_slide, local y=0 at the closed front plane
hs = C.hook("slide", cs["DRAWER_moving"], root, (0, m(front_y), m(zU0)))
D.drive(hs, "location", 1, root, "p_slide", "%.5f - p * %.4f" % (m(front_y), T))
H = 85.0
DW, DD = 440.0, 800.0
dr = MB()
dr.boxmm((0, -1.5, H / 2), (482.6, 3.0, H), 0)                       # front panel, 2U
parts = [P.Part("front_panel", dr, [L["tray_front"]], bevel=(mm := 0.0006, 1))]
pan = MB()
pan.boxmm((0, DD / 2, 3), (DW, DD, 1.5), 0)
for sx in (-1, 1):
    pan.boxmm((sx * (DW / 2 - 0.75), DD / 2, 18), (1.5, DD, 36), 0)
pan.boxmm((0, DD - 0.75, 18), (DW, 1.5, 36), 0)
parts.append(P.Part("tray_base", pan, [L["steel"]], bevel=(0.0004, 1)))
# front handles and the slide latch pull
hd = MB()
for sx in (-1, 1):
    hd.tube([(sx * 0.236, -0.0035, 0.025), (sx * 0.236, -0.026, 0.028), (sx * 0.236, -0.026, 0.060), (sx * 0.236, -0.0035, 0.063)], 0.0028, 8, 0)
parts.append(P.Part("handles", hd, [L["alu"]], smooth=True))
# patch panel: 2 rows x 10 LC duplex (blue), 3 MPO adapters (green APC), labels
lc = MB()   # housing blue, opening black
for row in range(2):
    for i in range(10):
        x = -215 + i * 22.5
        z = 24 + row * 36
        lc.boxmm((x, -4, z), (21, 7, 13), 0)
        lc.boxmm((x - 4.8, -7.6, z), (5.5, 0.6, 4.8), 1)
        lc.boxmm((x + 4.8, -7.6, z), (5.5, 0.6, 4.8), 1)
mp = MB()
for i in range(3):
    x = 40 + i * 52
    for row in range(2):
        z = 24 + row * 36
        mp.boxmm((x, -4, z), (36, 7, 14), 0)
        mp.boxmm((x, -7.6, z), (12.5, 0.6, 3.0), 1)
parts.append(P.Part("lc_adapters", lc, [lc_blue, L["black"]], bevel=(0.0003, 1)))
parts.append(P.Part("mpo_adapters", mp, [apc_green, L["black"]], bevel=(0.0003, 1)))
lb = MB()
lb.boxmm((-110, -3.2, 80.5), (215, 0.4, 5), 0)
lb.boxmm((130, -3.2, 80.5), (150, 0.4, 5), 0)
parts.append(P.Part("labels", lb, [L["label"]]))
# spools, routing guides, splice cassettes
sp = MB()   # spool core+flanges mat0
for (x, y) in ((-100, 250), (80, 330)):
    sp.cyl((m(x), m(y), m(10)), 0.040, 0.016, "z", 32, 0)         # lower flange r=40 (mm: 40)
    sp.cyl((m(x), m(y), m(18)), 0.028, 0.020, "z", 32, 0)         # core
    sp.cyl((m(x), m(y), m(27)), 0.040, 0.004, "z", 32, 0)         # upper flange
parts.append(P.Part("spools", sp, [L["grey"]], smooth=True))
gd = MB()   # bend-radius guides: pegs with flanges, along the rear and front edges
for x in range(-200, 201, 50):
    for y in (150, 460):
        gd.cyl((m(x), m(y), m(12)), 0.006, 0.020, "z", 12, 0)
        gd.cyl((m(x), m(y), m(22.5)), 0.0095, 0.002, "z", 12, 0)
for sx in (-1, 1):
    for t in range(9):
        a = math.pi * t / 8
        gd.cyl((m(sx * 170 + sx * 30 * math.sin(a)), m(480 - 30 * (1 - math.cos(a)) * 0 + 0), m(12)), 0.004, 0.02, "z", 8, 0)
parts.append(P.Part("routing_guides", gd, [L["black"]], smooth=True))
sc = MB()
for i in range(3):
    sc.boxmm((140, 60 + i * 0, 8 + i * 0), (0, 0, 0), 0) if False else None
for i in range(4):
    sc.boxmm((-170 + i * 36, 120, 11), (30, 90, 12), 0)
    for q in range(6):
        sc.boxmm((-170 + i * 36, 100 + q * 7, 17.5), (24, 0.8, 1), 1)
parts.append(P.Part("splice_cassettes", sc, [L["grey"], L["label"]], bevel=(0.0005, 1)))
objs = P.instantiate(parts, cs["DRAWER_moving"], hs, AID + "_drawer")
# fibers: patch-panel backs to spools (curves) + two spool coils + bundle exiting rear
fc = [(yellow, 0.0011), (aqua, 0.0011), (orange, 0.0011)]
for i in range(9):
    mat, r = fc[i % 3]
    x0 = -200 + i * 22
    pts = [(x0, 12, 26 + (i % 2) * 36), (x0, 60, 14), (x0 + (i - 4) * 4, 150, 14), (-100 + 40 * math.cos(i), 250 + 40 * math.sin(i), 14 + i % 3)]
    D.curve("%s_fiber_%d" % (AID, i), [(m(a), m(b), m(c)) for a, b, c in pts], r, mat, cs["FIBERS"], hs)
for k, (x, y) in enumerate(((-100, 250), (80, 330))):
    for turn in range(6):
        rr = 0.031 + 0.0014 * (turn % 2)
        pts = [(m(x) + rr * math.cos(a), m(y) + rr * math.sin(a), m(18) + 0.0026 * (turn - 2.5)) for a in [i * math.pi / 6 for i in range(12)]]
        D.curve("%s_coil_%d_%d" % (AID, k, turn), pts, 0.0011, [yellow, aqua][k], cs["FIBERS"], hs, cyclic=True)
for i in range(6):
    pts = [(m(80), m(330 + 31), m(18)), (m(100 + 8 * i), m(420), m(14 + i)), (m(60 + 20 * i), m(500), m(14)), (m(60 + 20 * i), m(DD + 40), m(14 + 2 * i))]
    D.curve("%s_bundle_%d" % (AID, i), pts, 0.0011, [yellow, aqua, orange][i % 3], cs["FIBERS"], hs)
lid = MB()
lid.boxmm((0, DD / 2, 37), (DW, DD, 1.2), 0)
lo = P.instantiate([P.Part("lid", lid, [clear])], cs["VARIANT_lid"], hs, AID + "_drawer")
D.set_collection_visible(cs["VARIANT_lid"], False)

# ------------------------------------------------ slides: outer fixed (rack), middle driven T/2, inner on drawer; stops
for sx, side in ((-1, "L"), (1, "R")):
    x = sx * (DW / 2 + 4)
    ou = MB()
    ou.boxmm((x + sx * 6, front_y + 350, zU0 + 42), (4, 700, 30), 0)
    for yy in (front_y + 4, front_y + 696):
        ou.boxmm((x + sx * 3, yy, zU0 + 42), (10, 8, 34), 1)       # end stops (rack side)
    P.instantiate([P.Part("outer_" + side, ou, [L["dark_steel"], L["brass"]], bevel=(0.0004, 1))], cs["SLIDES"], root, AID + "_slide")
    md = MB()
    md.boxmm((x, front_y + 300, zU0 + 42), (3.5, 560, 26), 0)
    mo = P.instantiate([P.Part("middle_" + side, md, [L["alu"]], bevel=(0.0004, 1))], cs["SLIDES"], root, AID + "_slide")["middle_" + side]
    mo.location = (0, 0, 0)
    mo.location.y = 0
    D.drive(mo, "location", 1, root, "p_slide", "-p * %.4f" % (T / 2))
    inn = MB()
    inn.boxmm((sx * (DW / 2 + 1.8), 330, 42), (3.0, 620, 22), 0)
    inn.boxmm((sx * (DW / 2 + 1.8), 620, 42), (4, 10, 28), 1)       # inner stop block
    P.instantiate([P.Part("inner_" + side, inn, [L["alu"], L["brass"]], bevel=(0.0004, 1))], cs["SLIDES"], hs, AID + "_slide")

# ------------------------------------------------ cable management arm (two links + hub), driven by p_slide
L_ = 0.45
P0 = (0.25, m(front_y) + 0.88)        # fixed pivot (rack rear right)
zc_ = m(zU0 + 44)
P1x, P1y0 = -0.15, m(front_y) + 0.77           # drawer pivot at closed (relative frame: drawer origin y = front_y)
# solver empty with chained custom properties (short driver expressions)
sv = bpy.data.objects.new(AID + "_cma_solver", None)
cs["CMA"].objects.link(sv)
sv.parent = root
for k_ in ("dist", "hh", "jx", "jy"):
    sv[k_] = 0.0
dxc = P1x - P0[0]
dy_ = "(%.5f - p * %.4f)" % (P1y0 - P0[1], T)
PV = {"p": (root, '["p_slide"]')}
D.drive_multi(sv, '["dist"]', None, PV, "sqrt(%.6f + %s*%s)" % (dxc * dxc, dy_, dy_))
SV = lambda n: (sv, '["%s"]' % n)
D.drive_multi(sv, '["hh"]', None, {"d": SV("dist")}, "sqrt(max(%.6f - d*d/4, 0.0001))" % (L_ * L_))
D.drive_multi(sv, '["jx"]', None, dict(PV, d=SV("dist"), h=SV("hh")), "%.5f + %.5f/2 + (%s/d)*h" % (P0[0], dxc, dy_))
D.drive_multi(sv, '["jy"]', None, dict(PV, d=SV("dist"), h=SV("hh")), "%.5f + %s/2 - (%.5f/d)*h" % (P0[1], dy_, dxc))
cma = MB()
cma.boxmm((L_ / 2 * 1000, 0, 0), (L_ * 1000, 26, 36), 0)
cma.boxmm((L_ / 2 * 1000, 0, 19), (L_ * 1000 - 8, 20, 2), 1)
a = P.instantiate([P.Part("link_A", cma, [L["dark_steel"], L["grey"]], bevel=(0.0006, 1))], cs["CMA"], root, AID + "_cma")["link_A"]
a.location = (P0[0], P0[1], zc_)
D.drive_multi(a, "rotation_euler", 2, {"x": SV("jx"), "y": SV("jy")}, "atan2(y - %.5f, x - %.5f)" % (P0[1], P0[0]))
hub = MB()
hub.cyl((0, 0, 0), 0.020, 0.044, "z", 20, 0)
hubo = P.instantiate([P.Part("hub", hub, [L["nickel"]], smooth=True)], cs["CMA"], root, AID + "_cma")["hub"]
for idx, nm in ((0, "jx"), (1, "jy")):
    D.drive_multi(hubo, "location", idx, {"v": SV(nm)}, "v")
hubo.location.z = zc_
b = P.instantiate([P.Part("link_B", cma, [L["dark_steel"], L["grey"]], bevel=(0.0006, 1))], cs["CMA"], root, AID + "_cma")["link_B"]
for idx, nm in ((0, "jx"), (1, "jy")):
    D.drive_multi(b, "location", idx, {"v": SV(nm)}, "v")
b.location.z = zc_
D.drive_multi(b, "rotation_euler", 2, dict(PV, x=SV("jx"), y=SV("jy")),
              "atan2(%.5f - p * %.4f + %.5f - y, %.5f - x)" % (P1y0 - P0[1] + P0[1], T, 0.0, P1x))
# drawer-side and rack-side brackets
br = MB()
br.boxmm((0, 0, 0), (36, 30, 40), 0)
P.instantiate([P.Part("bracket_rack", br, [L["dark_steel"]])], cs["CMA"], root, AID + "_cma")["bracket_rack"].location = (P0[0], P0[1], zc_)
bo = P.instantiate([P.Part("bracket_drawer", br, [L["dark_steel"]])], cs["CMA"], hs, AID + "_cma")["bracket_drawer"]
bo.location = (P1x, 0.77, 0.044)
C.hook("fiber_exit", cs["CMA"], hs, (P1x, 0.77, 0.044))
C.hook("fiber_rack_anchor", cs["CMA"], root, (P0[0], P0[1], zc_))
C.hook("patch_panel_center", cs["DRAWER_moving"], hs, (0, -0.01, 0.043))
C.hook("slide_open_front", cs["DRAWER_moving"], root, (0, m(front_y) - T, m(zU0) + 0.044))
bpy.context.view_layer.update()

blend = os.path.join(OUT, AID + ".blend")
meta = dict(
    sources=[
        dict(what="EIA-310: 19 in panel 482.6 mm, 1U = 44.45 mm (2U = 88.9 mm), mounting hole pitch 15.875 mm (5/8 in)", url="https://en.wikipedia.org/wiki/19-inch_rack", accessed="2026-10-02"),
        dict(what="LC duplex adapter (~21 x 13 mm outline per duplex) and MPO adapter outlines: estimates from public connector datasheet knowledge; not re-read", url="n/a", accessed="2026-10-02"),
        dict(what="Sliding fiber enclosure with cable management arm, spools and splice cassettes: generic product class (public vendor catalogues), geometry estimated", url="n/a", accessed="2026-10-02"),
    ],
    dimensions=[
        D.dim("drawer front panel", "482.6 x 85 x 3", "mm", "EIA-310 2U", "A"),
        D.dim("drawer body", "440 x 800 x 36", "mm", "estimate (range 420-450 x 600-850)", "C"),
        D.dim("slide travel", T * 1000, "mm", "estimate for a 600 mm class rail", "C"),
        D.dim("LC duplex adapters", "20 (2 rows x 10), 21 x 13 mm each", "mm", "estimate", "C"),
        D.dim("MPO adapters", "6 (2 rows x 3), 36 x 14 mm each", "mm", "estimate", "C"),
        D.dim("rack stub", "600 x 1000 x 1500", "mm", "generic 19 in rack; estimate", "C"),
        D.dim("CMA link length", 360, "mm", "estimate; 2 links + hub, geometry solved analytically in drivers", "C"),
    ],
    simplifications=["fibers are curves: panel-back runs, two spool coils, a bundle ending at the rear of the drawer (assembler's fiber spaghetti attaches at HOOK_fiber_exit)",
                     "CMA has no fiber bundle inside (links only)", "no logos"],
    hooks_doc={"HOOK_slide": "drawer root: location.y driven by p_slide (0 closed, 1 open, travel 0.55 m); animate p_slide on the root",
               "HOOK_fiber_exit": "fiber bundle exit at the drawer rear (moves with the drawer)", "HOOK_fiber_rack_anchor": "CMA rack-side pivot (fixed)",
               "HOOK_patch_panel_center": "front panel centre", "HOOK_slide_open_front": "fully open front plane marker"},
    custom_properties_doc={"p_slide": "0..1 drawer travel; drives drawer, middle slide members and the CMA links via drivers (simple expressions, no Python needed)"},
    variants={"lid": "unhide collection VARIANT_lid (transparent clear cover)"},
    origin="bottom centre of the rack stub footprint, front at -y", intended_usage="S6 tray pull-out (animate p_slide); fiber spill FX attaches at HOOK_fiber_exit",
)
D.use_visible_bbox()
C.finish(AID, blend, coll, meta)
D.extend_meta(blend, dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll), custom_properties=D.custom_props(root)))
pv = os.path.join(OUT, "previews")
lights = [((1.5, -2.5, 2.5), 80, 2.0)]
res = []
for pval, nm in ((0.0, "closed"), (1.0, "open")):
    root["p_slide"] = pval
    bpy.context.view_layer.update()
    D.render_views(coll, pv, AID, [dict(name=nm + "_three_quarter", loc=(1.9, -2.1, 1.7), target=(0, -0.25, 0.8), lens=35),
                                   dict(name=nm + "_top", loc=(0, -0.3, 3.3), target=(0, -0.2, 0.7), lens=35)],
                   floor=True, floor_size=8, lights=lights)
root["p_slide"] = 1.0
bpy.context.view_layer.update()
D.render_views(coll, pv, AID, [dict(name="front", loc=(0, -2.4, 1.0), target=(0, -0.2, 0.8), lens=40),
                               dict(name="panel_closeup", loc=(0.15, -1.3, 0.95), target=(0.0, -0.78, 0.76), lens=40)],
               floor=True, floor_size=8, lights=lights)
