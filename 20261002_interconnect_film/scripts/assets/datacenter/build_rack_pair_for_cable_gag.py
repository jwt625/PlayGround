"""rack_pair_for_cable_gag: two racks side by side with parametric centre gap (p_gap, 1..5 m), overhead cable ladder, 14 port hooks per side.

Run: Blender -b --python scripts/assets/datacenter/build_rack_pair_for_cable_gag.py -- assets/components/datacenter [gap=2.0]
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
AID = "rack_pair_for_cable_gag"
GAP = float(KV.get("gap", 2.0))
C.reset()
D.reset_mats()
L = D.lib()
m = lambda v: v * 0.001
coll, root = C.new_asset(AID, accuracy="C")
root["p_gap"] = GAP
cs = {k: C.sub_collection(coll, k) for k in ("RACK_A", "RACK_B", "LADDER", "PORTS", "HOOKS")}
RH = R.RACK_H
# racks (stub: carcass + perforated front door), driven positions
ra = R.rack_stub(cs["RACK_A"], root, AID + "_rack_a", L, loc=(-GAP / 2, 0, 0), cache_key="stub")
rb = R.rack_stub(cs["RACK_B"], root, AID + "_rack_b", L, loc=(GAP / 2, 0, 0), cache_key="stub")
D.drive(ra, "location", 0, root, "p_gap", "-p / 2")
D.drive(rb, "location", 0, root, "p_gap", "p / 2")
# side port plates (14 cable glands: 2 rows x 7) on the facing sides
ports = []
for side, rack, sx in (("A", ra, 1), ("B", rb, -1)):
    pl = MB()
    gl = MB()
    x = sx * (R.RACK_W / 2 + 1.5)
    pl.boxmm((x, 0, 1525), (3, 760, 460), 0)
    names = []
    for row in range(2):
        for i in range(7):
            y = -360 + i * 120
            z = 1350 + row * 350
            gl.cyl((m(x + sx * 12), m(y), m(z)), 0.030, 0.024, "x", 24, 0)
            gl.cyl((m(x + sx * 25), m(y), m(z)), 0.019, 0.002, "x", 20, 1)
    P.instantiate([P.Part("port_plate", pl, [L["dark_steel"]], bevel=(0.0008, 1)),
                   P.Part("port_glands", gl, [L["black"], L["hose"]], smooth=True)], cs["PORTS"], rack, "%s_%s" % (AID, side.lower()))
    for row in range(2):
        for i in range(7):
            n = row * 7 + i
            h = C.hook("port_%s_%02d" % (side, n), cs["HOOKS"], rack, (m(x + sx * 27), m(-360 + i * 120), m(1350 + row * 350)),
                       (0, 0, math.pi / 2 * sx))
            h.empty_display_size = 0.04
    C.hook("port_" + side, cs["HOOKS"], rack, (m(x + sx * 27), 0, m(1525)), (0, 0, math.pi / 2 * sx))
# overhead ladder across the gap (rails scale with p_gap, rungs by array count)
zl = RH * 0.001 + 0.045
rl = MB()
for yy in (-0.15, 0.15):
    rl.box((0, yy, 0), (1.0, 0.04, 0.06), 0)
rails = rl.obj(AID + "_ladder_rails", [L["steel"]], bevel=(0.0008, 1))
D.put(rails, cs["LADDER"], root, loc=(0, 0, zl))
D.drive(rails, "scale", 0, root, "p_gap", "p")
rg = MB()
rg.box((0, 0, 0), (0.025, 0.30, 0.025), 0)
rung = rg.obj(AID + "_ladder_rungs", [L["steel"]], bevel=(0.0005, 1))
D.put(rung, cs["LADDER"], root, loc=(-GAP / 2 + 0.15, 0, zl))
am = rung.modifiers.new("Array", "ARRAY")
am.use_relative_offset = False
am.use_constant_offset = True
am.constant_offset_displace = (0.3, 0, 0)
am.count = max(2, int(GAP / 0.3))
D.drive(rung, "location", 0, root, "p_gap", "-p / 2 + 0.15")
D.drive_multi(rung, 'modifiers["Array"].count', None, {"p": (root, '["p_gap"]')}, "floor(p / 0.3)")
# ladder supports on each rack top
sp = MB()
for sx in (-1, 1):
    for yy in (-0.15, 0.15):
        sp.box((0, yy, RH * 0.001 + 0.0225), (0.08, 0.05, 0.045), 0)
sa = sp.obj(AID + "_ladder_support_a", [L["dark_steel"]], bevel=(0.0006, 1))
D.put(sa, cs["LADDER"], ra, loc=(R.RACK_W / 2 * 0.001 - 0.04, 0, 0))
sb = P.instantiate([P.Part("x", sp, [L["dark_steel"]], bevel=(0.0006, 1))], cs["LADDER"], rb, AID + "_ladder_support_b", key="sup")["x"]
sb.location = (-R.RACK_W / 2 * 0.001 + 0.04, 0, 0)
# bundle mount: saddle clamp plates at the ladder centre and near each rack (P-clamp style)
cl = MB()
cl.box((0, 0, 0.012), (0.06, 0.12, 0.008), 0)
cl.cyl((0, 0, 0.03), 0.032, 0.03, "y", 20, 1) if False else None
mt = cl.obj(AID + "_bundle_saddle", [L["alu"]], bevel=(0.0004, 1))
D.put(mt, cs["LADDER"], root, loc=(0, 0, zl + 0.03))
C.hook("bundle_mount_ladder", cs["HOOKS"], root, (0, 0, zl + 0.05))
C.hook("rack_a_top", cs["HOOKS"], ra, (0, 0, RH * 0.001))
C.hook("rack_b_top", cs["HOOKS"], rb, (0, 0, RH * 0.001))
C.hook("pair_center_floor", cs["HOOKS"], root, (0, 0, 0))
bpy.context.view_layer.update()
blend = os.path.join(OUT, AID + ".blend")
meta = dict(
    sources=[dict(what="Rack envelope 600 x 1068 x 2236 mm (NVL72 / MGX ORV3 listing)", url="https://www.supermicro.com/en/products/system/gpu/48u/srs-gb200-nvl72", accessed="2026-10-02"),
             dict(what="14-cable bundle, ports 2 rows x 7 at z 1.35 / 1.70 m, y -0.36..0.36 m (storyboard crude s1)", url="scripts/build_crude_film.py s1", accessed="2026-10-02")],
    dimensions=[D.dim("rack", "600 x 1068 x 2236", "mm", "NVL72 / MGX ORV3 external", "B"),
                D.dim("rack centre gap", "p_gap = 1..5 m", "m", "param (storyboard 1, 2, 5 m)", "-"),
                D.dim("port plates", "760 x 460 mm, 14 glands o60 mm, spacing 120 / 350 mm", "mm", "estimate", "C"),
                D.dim("ladder", "300 mm wide, rungs every 300 mm, atop rack tops", "mm", "estimate; typical ladder tray", "C")],
    simplifications=["racks are closed carcasses with a perforated front door (no interior)", "no cables (bundle supplied by interconnect agent)"],
    hooks_doc={"HOOK_port_A / HOOK_port_B": "centre of each 14-port plate (facing sides, +z up, rotated to face the other rack)",
               "HOOK_port_A_00..13 / HOOK_port_B_00..13": "each gland (row-major: row 0 z=1.35 m, row 1 z=1.70 m; y -0.36..0.36)",
               "HOOK_bundle_mount_ladder": "saddle on the overhead ladder centre", "HOOK_rack_a_top / rack_b_top": "rack tops"},
    custom_properties_doc={"p_gap": "centre-to-centre gap in metres (1..5); drives rack A/B x, ladder rail scale and rung count (drivers)"},
    origin="centre between racks, floor z=0, fronts at -y", intended_usage="S1 cable-length gag; animate p_gap (or rack B position) for the stretch",
)
D.use_visible_bbox()
C.finish(AID, blend, coll, meta)
D.extend_meta(blend, dict(material_slots=D.mat_slots(), hooks=D.hook_list(coll)[:6] + ["... %d hooks total" % len(D.hook_list(coll))], custom_properties=D.custom_props(root)))
pv = os.path.join(OUT, "previews")
lights = [((0, -4, 4), 120, 3.0)]
for g, nm in ((2.0, "gap2"), (1.0, "gap1"), (5.0, "gap5")):
    root["p_gap"] = g
    bpy.context.view_layer.update()
    D.render_views(coll, pv, AID, [dict(name=nm + "_three_quarter", loc=(g * 0.9 + 3.0, -g * 0.9 - 3.5, 2.6), target=(0, 0, 1.2), lens=35)],
                   floor=True, floor_size=30, lights=lights)
root["p_gap"] = 2.0
bpy.context.view_layer.update()
D.render_views(coll, pv, AID, [dict(name="front", loc=(0, -8, 1.4), target=(0, 0, 1.2), lens=40),
                               dict(name="top", loc=(0, -0.1, 9), target=(0, 0, 0), lens=35),
                               dict(name="ports_closeup", loc=(0, -1.8, 1.7), target=(0.0, 0, 1.55), lens=45)],
               floor=True, floor_size=30, lights=lights)
