"""Rack-style sampling oscilloscope / digital communications analyzer (generic, no brand marks), inspired by the public
dimensions of the Keysight 86100D DCA-X mainframe class: wide, deep, plug-in modules with optical (FC/PC) and electrical inputs.
Run: Blender -b --python build_sampling_scope_dca.py -- <out_dir>
Origin: bottom centre of the footprint (underside of feet at z = 0). -Y = front.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import lo_common as L
from lo_common import C, M, S

OUT = C.argv_after_dashes()[0]
AID = "sampling_scope_dca"
L.begin(AID, "B")
root = S.root

W, D, FEET = 426.0, 530.0, 13.0
Z0, Z1 = FEET, 234.0
HB = Z1 - Z0
YF = -D / 2
YP = YF - 3.0
SCR_W, SCR_H = 216.0, 135.0
SCR_CX, SCR_CZ = -78.0, 157.0
root["p_screen_aspect"] = 1.6
root["p_rack_ears"] = 0

m_shell = M("case_light")
m_panel = M("panel_light")
m_dark = M("panel_dark")
m_bezel = M("abs_textured")
m_glass = M("glass_dark")
m_screen = L.screen_material("MAT_lab_office_screen")
m_ink = M("ink_white")
m_ink_dark = C.principled("MAT_lab_office_ink_dark", (0.05, 0.05, 0.06), rough=0.6)
S.slots.add("MAT_lab_office_ink_dark")

# ---------------------------------------------------------------- shell + panel
L.box("shell", (W, D, HB), (0, 0, (Z0 + Z1) / 2), m_shell, r=4.0, seg=2)
L.box("front_panel", (W - 8, 3.0, HB - 8), (0, YF - 1.5, (Z0 + Z1) / 2), m_panel, r=2.5)
L.box("front_trim_top", (W - 8, 3.5, 6.0), (0, YF - 1.7, Z1 - 7.0), M("case_grey"), r=1.0)
L.box("front_trim_bottom", (W - 8, 3.5, 6.0), (0, YF - 1.7, Z0 + 7.0), M("case_grey"), r=1.0)
# top and side vents (single merged meshes)
L.multi_box("top_vents", [((100.0, 3.2, 0.6), (-80.0 + 0, -100.0 + i * 12.0, Z1 + 0.1)) for i in range(18)], M("abs_black"), r=0.2)
L.multi_box("side_vents_l", [((0.6, 120.0, 3.2), (-W / 2 - 0.1, -40.0, 60.0 + i * 13.0)) for i in range(10)], M("abs_black"), r=0.2)
L.multi_box("side_vents_r", [((0.6, 120.0, 3.2), (W / 2 + 0.1, -40.0, 60.0 + i * 13.0)) for i in range(10)], M("abs_black"), r=0.2)

# ---------------------------------------------------------------- display
bz = 9.0
ww, wh = SCR_W + 8, SCR_H + 8
for nm, size, loc in (("bezel_top", (ww + 2 * bz, 3.0, bz), (SCR_CX, 0, SCR_CZ + wh / 2 + bz / 2)),
                      ("bezel_bottom", (ww + 2 * bz, 3.0, bz), (SCR_CX, 0, SCR_CZ - wh / 2 - bz / 2)),
                      ("bezel_left", (bz, 3.0, wh), (SCR_CX - ww / 2 - bz / 2, 0, SCR_CZ)),
                      ("bezel_right", (bz, 3.0, wh), (SCR_CX + ww / 2 + bz / 2, 0, SCR_CZ))):
    L.box(nm, size, (loc[0], YP - 1.5, loc[2]), m_bezel, r=1.0)
L.box("glass_backing", (ww, 1.0, wh), (SCR_CX, YP - 0.5, SCR_CZ), m_glass)
scr = L.uvquad("screen", SCR_W, SCR_H, (SCR_CX, YP - 1.6, SCR_CZ), m_screen, facing="-y")
scr.name = AID + "_screen"
scr["is_image_screen"] = True
scr.visible_shadow = False
# softkeys under the display
sk = L.box("softkey_1", (26.0, 3.0, 8.0), (SCR_CX - 5 * 20.0 / 1.0 * 0.0 - 105.0, YP - 1.5, SCR_CZ - wh / 2 - bz - 9.0), M("key_grey"), r=1.0)
for i in range(1, 8):
    L.dup(sk, "softkey_%d" % (i + 1), (SCR_CX - 105.0 + i * 30.0, YP - 1.5, SCR_CZ - wh / 2 - bz - 9.0))

# ---------------------------------------------------------------- control area (below display): keypad grid, knob, arrows
KZ = 62.0
keys = []
for r_ in range(3):
    for c_ in range(8):
        keys.append(((16.0, 2.6, 9.0), (-196.0 + c_ * 20.0 + 8.0, YP - 1.3, KZ + 14.0 - r_ * 15.0)))
L.multi_box("keypad_keys", keys, M("key_grey"), r=0.8)
# colour-coded function keys
L.box("key_run", (22.0, 3.0, 9.0), (-196.0 + 8.0, YP - 1.5, KZ - 24.0), M("key_green"), r=1.0)
L.box("key_stop", (22.0, 3.0, 9.0), (-196.0 + 8.0 + 26.0, YP - 1.5, KZ - 24.0), M("key_red"), r=1.0)
L.box("key_autoscale", (22.0, 3.0, 9.0), (-196.0 + 8.0 + 52.0, YP - 1.5, KZ - 24.0), M("key_yellow"), r=1.0)
kn = L.knob("main_knob", (-5.0, YP, KZ - 2.0), 14.0, 15.0, M("knob_black"), seg=40)
L.knob("small_knob", (-40.0, YP, KZ - 2.0), 8.0, 10.0, M("knob_black"), seg=32)
# power + USB + ESD jack bottom-left
L.cyl("power_button", 6.5, 3.5, (-188.0, YP - 1.5, 30.0), M("key_grey"), axis="-y", seg=32, bev=0.5)
L.lathe("power_led_ring", [(7.5, 0), (8.6, 0), (8.6, 0.8), (7.5, 0.8)], (-188.0, YP - 1.0, 30.0), M("led_green"), axis="-y", seg=40)
u1 = L.box("usb_a_1", (14.0, 4.0, 6.5), (-160.0, YP - 0.5, 30.0), M("abs_black"), r=0.5)
L.dup(u1, "usb_a_2", (-138.0, YP - 0.5, 30.0))
L.cyl("esd_ground_jack", 4.0, 5.0, (-112.0, YP - 2.5, 30.0), M("chrome"), axis="-y", seg=16, bev=0.4)
L.cyl("esd_ground_jack_hole", 1.8, 5.2, (-112.0, YP - 5.0, 30.0), M("abs_black"), axis="-y", seg=12)

# ---------------------------------------------------------------- plug-in modules (right half): A optical+electrical, B dual electrical
MX = 125.0
for nm, zc, h in (("module_a", 178.0, 94.0), ("module_b", 72.0, 94.0)):
    L.box(nm + "_plate", (170.0, 3.0, h), (MX, YP - 1.5, zc), m_dark, r=2.0)
    L.box(nm + "_handle", (6.0, 5.0, h - 30.0), (MX + 77.0, YP - 5.5, zc), M("alu_dark"), r=1.5)
    for sz in (-1, 1):
        L.cyl(nm + "_screw_%s" % ("t" if sz > 0 else "b"), 3.0, 2.0, (MX - 77.0, YP - 3.8, zc + sz * (h / 2 - 7.0)), M("steel"), axis="-y", seg=12, bev=0.3)

# FC/PC bulkhead on module A (square flange 19 mm, barrel with ceramic ferrule); dust cap hanging beside
fc_x, fc_z = MX - 40.0, 178.0
flange = L.box("optical_in_flange", (22.0, 2.6, 22.0), (fc_x, YP - 4.3, fc_z), M("chrome"), r=1.2)
bar = L.lathe("optical_in_barrel", [(2.8, 0), (5.2, 0), (5.2, 14.0), (4.8, 15.0), (3.2, 15.0), (2.8, 13.5)], (fc_x, YP - 5.6, fc_z), M("chrome"), axis="-y", seg=32)
L.cyl("optical_in_sleeve", 2.6, 12.0, (0, -8.0, 0), M("ptfe"), axis="-y", seg=24, parent=bar)
L.cyl("optical_in_ferrule", 1.25, 14.0, (0, -8.5, 0), M("ceramic_white"), axis="-y", seg=16, parent=bar)
for sx in (-1, 1):
    for sz in (-1, 1):
        L.cyl("optical_in_screw", 1.2, 1.0, (fc_x + sx * 8.3, YP - 5.9, fc_z + sz * 8.3), M("steel"), axis="-y", seg=10)
# dust cap on a tether
cap = L.lathe("optical_dust_cap", [(0, 0), (6.4, 0), (6.4, 14.0), (5.4, 14.0), (5.4, 15.0), (0, 15.0)], (fc_x + 28.0, YP - 6.0, fc_z - 14.0), M("abs_black"), axis="-y", seg=24)
L.tube("optical_dust_cap_tether", L.catmull([(fc_x + 9, YP - 6.0, fc_z), (fc_x + 20, YP - 9.0, fc_z - 8), (fc_x + 27, YP - 12, fc_z - 14)], 5), 0.8, (0, 0, 0), M("abs_black"), seg=6)
# electrical inputs (2.92 mm class: 8 mm hex nut with 3 mm bore), two on module A, two on module B
def eport(name, x, z, nut=8.0):
    b = L.lathe(name, [(1.8, 0), (nut / 2, 0), (nut / 2, 10.0), (nut / 2 - 0.6, 11.0), (3.4, 11.0), (3.4, 9.0)], (x, YP - 4.0, z), M("chrome"), axis="-y", seg=6 if False else 24)
    L.cyl(name + "_die", 1.7, 7.0, (0, -5.0, 0), M("ptfe"), axis="-y", seg=16, parent=b)
    L.cyl(name + "_pin", 0.45, 8.0, (0, -6.5, 0), M("gold"), axis="-y", seg=8, parent=b)
    return b
e_a1 = eport("elec_in_1", MX + 20.0, 196.0)
e_a2 = eport("elec_in_2", MX + 20.0, 158.0)
e_b1 = eport("elec_in_3", MX - 20.0, 90.0)
e_b2 = eport("elec_in_4", MX + 20.0, 90.0)
eclk = eport("clock_in", MX + 55.0, 90.0, nut=7.0)
eport("trigger_in", MX - 55.0, 54.0, nut=7.0)
eport("trigger_out", MX + 20.0, 54.0, nut=7.0)
# status LEDs
for i in range(4):
    L.cyl("module_led_%d" % (i + 1), 1.3, 0.8, (MX + 30.0 + i * 6.0, YP - 3.4, 211.0), M(["led_green", "led_green", "led_amber", "led_red"][i]), axis="-y", seg=10)
# printed labels
tx = [("OPTICAL IN", 3.0, (fc_x, YP - 3.2, fc_z + 16.0)), ("ELEC IN 1", 2.6, (MX + 20, YP - 3.2, 205.0)), ("ELEC IN 2", 2.6, (MX + 20, YP - 3.2, 167.0)),
      ("ELEC IN 3", 2.6, (MX - 20, YP - 3.2, 99.0)), ("ELEC IN 4", 2.6, (MX + 20, YP - 3.2, 99.0)), ("CLOCK", 2.6, (MX + 55, YP - 3.2, 99.0)),
      ("TRIG IN", 2.6, (MX - 55, YP - 3.2, 63.0)), ("TRIG OUT", 2.6, (MX + 20, YP - 3.2, 63.0)), ("ESD", 2.4, (-112, YP - 3.2, 22.0)),
      ("POWER", 2.4, (-188, YP - 3.2, 19.0)), ("USB", 2.4, (-149, YP - 3.2, 22.0)), ("MODULE A", 3.0, (MX, YP - 3.2, 222.0)), ("MODULE B", 3.0, (MX, YP - 3.2, 116.0))]
L.labels("panel_labels", [(t, s, (p[0], p[1], p[2])) for (t, s, p) in tx], m_ink)
L.labels("panel_labels_dark", [("DIGITAL COMMUNICATIONS ANALYZER", 3.0, (-78.0, YP - 3.2, Z1 - 12.0))], m_ink_dark)

# ---------------------------------------------------------------- feet, handles, rack ears (variant)
for sx in (-1, 1):
    for sy in (-1, 1):
        L.box("foot_%s%s" % ("l" if sx < 0 else "r", "f" if sy < 0 else "b"), (50.0, 50.0, FEET), (sx * 160.0, sy * 200.0, FEET / 2), M("rubber"), r=3.0)
L.push_group("variant_rack_ears", (0, 0, 0), sub=True)
S.coll.hide_render = True
for sx in (-1, 1):
    L.box("rack_ear_%s" % ("l" if sx < 0 else "r"), (28.0, 3.0, HB), (sx * (W / 2 + 14.0 - 4.0 + 4.0), YF - 1.5 + 0.0, (Z0 + Z1) / 2), M("steel"), r=1.0)
    for sz in (-1, 1):
        L.cyl("rack_ear_hole", 3.2, 3.2, (sx * (W / 2 + 14.0 - 4.0 + 4.0), YF - 2.2, (Z0 + Z1) / 2 + sz * 80.0), M("abs_black"), axis="-y", seg=16)
L.pop_group()

# ---------------------------------------------------------------- rear panel (+Y)
YR = D / 2
L.box("rear_plate", (W - 12, 3.0, HB - 12), (0, YR + 1.5, (Z0 + Z1) / 2), m_dark, r=2.0)
FX, FZ = -110.0, 123.0
L.cyl("fan_recess", 60.0, 2.0, (FX, YR + 3.0, FZ), M("abs_black"), axis="y", seg=48)
for k, rr in enumerate((14.0, 28.0, 42.0, 57.0)):
    L.tube("fan_ring_%d" % k, L.circle_pts(rr, 64, (FX, YR + 4.6, FZ), plane="xz"), 1.0, (0, 0, 0), M("alu_dark"), seg=6, closed=True, cap=False)
for k in range(8):
    a = math.pi * k / 4
    L.box("fan_spoke_%d" % k, (57.0, 1.6, 1.6), (FX + 28.5 * math.cos(a), YR + 4.6, FZ + 28.5 * math.sin(a)), M("alu_dark"), rot=(0, -math.degrees(a), 0))
L.cyl("fan_hub", 9.0, 3.0, (FX, YR + 3.5, FZ), M("abs_black"), axis="y", seg=24)
L.box("iec_inlet", (30.0, 6.0, 22.0), (150.0, YR + 5.0, 60.0), M("abs_black"), r=1.0)
L.box("iec_inlet_recess", (24.0, 3.0, 16.0), (150.0, YR + 7.2, 60.0), M("abs_dark"), r=0.5)
L.box("rear_power_switch", (20.0, 3.0, 12.0), (150.0, YR + 4.5, 85.0), M("abs_black"), r=1.0)
io = [((16.0, 3.0, 12.0), (60.0 + i * 24.0, YR + 4.5, 160.0)) for i in range(5)]
L.multi_box("rear_io_ports", io, M("chrome"), r=0.5)
L.multi_box("rear_io_cavities", [((12.0, 2.0, 7.0), (60.0 + i * 24.0, YR + 5.6, 160.0)) for i in range(5)], M("abs_black"))
L.multi_box("rear_vent_slots", [((130.0, 1.0, 3.0), (110.0, YR + 3.4, 195.0 + (i * 6.0) - 0.0)) for i in range(5)] , M("abs_black"), r=0.3)
L.box("rating_label", (70.0, 0.5, 26.0), (140.0, YR + 3.2, 112.0), M("paper"), r=0.3, seg=1)

# ---------------------------------------------------------------- hooks
L.hook("screen_center", (SCR_CX, YP - 1.8, SCR_CZ), note="screen centre; screen faces -Y")
L.hook("optical_in", (fc_x, YP - 21.0, fc_z), note="FC/PC mouth (fiber attach point), axis -Y")
L.hook("electrical_in_1", (MX + 20.0, YP - 15.0, 196.0), note="electrical input 1 mouth")
L.hook("electrical_in_2", (MX + 20.0, YP - 15.0, 158.0), note="electrical input 2 mouth")
L.hook("dust_cap", (fc_x + 28.0, YP - 6.0, fc_z - 14.0), note="dust cap (removable, swing to the side)")

# ---------------------------------------------------------------- dimensions
L.dim("height without feet (body)", HB, "mm", "Keysight 86100D datasheet: 221 mm without front connectors and rear feet (5990-5824, via search snippet 2026-10-02)", "B")
L.dim("height with feet", Z1, "mm", "Keysight 86100D datasheet: 234 mm with front connectors and rear feet", "B")
L.dim("width", W, "mm", "Keysight 86100D datasheet: 426 mm", "B")
L.dim("depth (body)", D, "mm", "Keysight 86100D datasheet: 530 mm without front connectors and rear feet (601 mm with them)", "B")
L.dim("rack ear width (variant)", 28.0, "mm", "EIA-310 19 in rack: 482.6 mm overall (426 + 2 x 28); ear shape generic", "B")
L.dim("screen active area", "216 x 135", "mm", "estimate (about 10 in, 16:10) to map the 384x240 eye textures without stretch", "C")
L.dim("FC/PC adapter flange", "22 x 22", "mm", "estimate; ferrule OD 2.5 mm is standard (SC/FC ceramic ferrule)", "B")
L.dim("electrical connector (2.92 mm class)", "8 mm hex nut", "mm", "estimate", "C")
L.dim("module plate size", "170 x 94", "mm", "estimate", "C")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="mainframe dimensions", url="https://www.keysight.com/us/en/assets/7018-02546/data-sheets-archived/5990-5824.pdf (86100D datasheet)", access_date="2026-10-02"),
             dict(what="front-panel arrangement (display left, plug-in module bays right, optical and electrical inputs): generic from the DCA class, not a copy of any model", url="n/a", access_date="2026-10-02")],
    simplifications=["no brand marks or model names; the only front text is generic words", "module internals not modelled", "rack ears in hidden VARIANT_rack_ears (hide_render on)",
                     "keypad is a merged key grid (one mesh) with no individual key legends"],
    custom_properties={"p_screen_aspect": "1.6", "p_rack_ears": "informational: unhide collection ITEM_variant_rack_ears to show the ears"},
    screens=[dict(object=AID + "_screen", material="MAT_lab_office_screen", image_node="SCREEN_IMAGE", toggle_node="SCREEN_FAC (0 idle, 1 image)", aspect="16:10",
                  note="same convention as bench_oscilloscope")],
    usage="S1/S2 wall-mounted sampling scope (generic) showing the eye; use HOOK_screen_center for pushes, HOOK_optical_in for the fiber."))

L.plug_image(m_screen, L.tex_paths("eye_s2", 100)[99])
views = [("front", dict(loc=(0, -1.2, 0.14), target=(0, 0, 0.12), lens=50)),
         ("three_quarter", dict(loc=(0.9, -1.1, 0.55), target=(0, 0, 0.1), lens=45)),
         ("top", dict(loc=(0, -0.05, 1.5), target=(0, 0, 0), lens=50)),
         ("back", dict(loc=(-0.9, 1.5, 0.6), target=(0, 0.1, 0.12), lens=45)),
         ("closeup_inputs", dict(loc=(0.22, -0.9, 0.22), target=(0.13, -0.27, 0.13), lens=50))]
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=True, sun=1.4, world=0.5)
