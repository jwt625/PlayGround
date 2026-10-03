"""Bench clutter set: probes with clips, DMM, DMM leads, bench PSU, function generator, soldering station, microscope, laptop,
ESD parts bins, cable spool, coffee mug, notebook. Each item is its own empty + sub-collection (ITEM_<name>) with the origin at the
bottom centre of the item (rest on z = 0), -Y = front. No brands; generic labels only.
Run: Blender -b --python build_lab_clutter.py -- <out_dir>
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import lo_common as L
from lo_common import C, M, S

OUT = C.argv_after_dashes()[0]
AID = "lab_clutter"
L.begin(AID, "B")
root = S.root
PI = math.pi
screens = []


def lcd(name, w, h, loc, color, tint=None, facing="-y", rot=None):
    """Small LCD: backing glass + screen plane with an own empty-image screen material."""
    mat = L.screen_material("MAT_lab_office_screen_%s" % name, idle=color)
    p = L.uvquad(name + "_screen", w, h, loc, mat, facing=facing, rot=rot)
    p.visible_shadow = False
    p["is_image_screen"] = True
    screens.append(p.name)
    return p


# ================================================================= 1. probes with cables and clips
L.push_group("probes_clips", (-700, -300, 0))
cab = L.catmull([(0, 0, 3.0), (60, -40, 2.2), (130, 20, 2.2), (220, 60, 2.2), (300, -10, 2.2), (330, -90, 2.2)], 8)
L.tube("probe_cable", cab, 2.0, (0, 0, 0), M("abs_black"), seg=8)
pb = L.lathe("probe_body", [(0, 0), (4.8, 0), (5.4, 6), (5.4, 70), (4.2, 90), (3.2, 104), (0, 104)], (330, -90, 6.0), M("abs_dark"), axis="-y", seg=24)
L.cyl("probe_grip", 5.6, 40.0, (0, -30, 0), M("key_blue"), axis="y", seg=24, parent=pb)
L.lathe("probe_tip", [(0, 0), (2.2, 0), (1.6, 6), (0.7, 10), (0.4, 16), (0, 18)], (0, -104, 0), M("steel"), axis="-y", seg=12, parent=pb)
L.hook("probe_tip", (0, -122, 0), parent=pb, note="probe needle tip")
L.box("probe_comp_box", (24, 52, 16), (160, 40, 8.5), M("abs_dark"), r=3)
L.lathe("probe_bnc_plug", [(0, 0), (6.5, 0), (6.5, 18), (0, 18)], (0, 14, 6.0), M("chrome"), axis="y", seg=20)
# IC hook clip (micro-grabber) and alligator clip on short leads
for k, (cx, cy, nm) in enumerate(((230, -80, "grabber"), (280, -140, "alligator"))):
    cv = L.catmull([(cx - 80, cy + 60, 2.0), (cx - 40, cy + 20, 2.0), (cx, cy, 3.0)], 6)
    L.tube("clip_lead_%s" % nm, cv, 1.1, (0, 0, 0), M("key_red" if k == 0 else "abs_black"), seg=6)
    if nm == "grabber":
        L.box("grabber_body", (9, 36, 9), (cx, cy - 12, 5.5), M("key_red"), r=1.5)
        L.cyl("grabber_hook", 1.6, 10, (cx, cy - 33, 6), M("steel"), axis="y", seg=8)
    else:
        L.box("alligator_jaw_top", (10, 40, 3), (cx, cy - 18, 12), M("steel"), r=1.0, rot=(-8, 0, 0))
        L.box("alligator_jaw_bottom", (10, 40, 3), (cx, cy - 18, 5.5), M("steel"), r=1.0, rot=(8, 0, 0))
        L.box("alligator_boot", (12, 24, 12), (cx, cy + 6, 6.5), M("abs_black"), r=2.5)
L.pop_group()

# ================================================================= 2. DMM (handheld, standing on its rear bail)
L.push_group("dmm", (-380, 200, 0))
yel = C.principled("MAT_lab_office_holster_yellow", (0.85, 0.65, 0.05), rough=0.7)
S.slots.add("MAT_lab_office_holster_yellow")
L.box("holster", (96, 50, 196), (0, 3, 98), yel, r=12, seg=3)
L.box("body", (84, 40, 184), (0, -3, 98), M("abs_dark"), r=8, seg=2)
L.box("face_plate", (78, 1.5, 178), (0, -23.5, 98), M("panel_dark"), r=2)
L.box("lcd_window", (68, 1.5, 40), (0, -24.4, 160), M("glass_dark"), r=1)
lcd("dmm", 60, 32, (0, -25.5, 160), (0.35, 0.42, 0.3))
d = L.lathe("dial", [(0, 0), (27, 0), (27, 8), (23, 12), (0, 12)], (0, -24.5, 108), M("knob_black"), axis="-y", seg=40)
L.box("dial_pointer", (3, 0.5, 18), (0, -12.2, 12), M("ink_white"), parent=d)
for i in range(5):
    L.box("fn_button_%d" % (i + 1), (11, 3, 7), (-30 + i * 15, -25.5, 139), M("key_blue" if i == 0 else "key_grey"), r=1.2)
for i, (jm) in enumerate(("abs_black", "abs_black", "key_red", "key_red")):
    L.lathe("jack_%d" % (i + 1), [(2.2, 0), (6.5, 0), (6.5, 5), (4.8, 6), (2.2, 6)], (-31.5 + i * 21, -23.5, 40), M(jm), axis="-y", seg=20)
L.box("rear_bail", (60, 4, 70), (0, 30, 40), M("abs_dark"), r=1, rot=(-18, 0, 0))
L.labels("dmm_labels", [("V", 3.0, (-30, -24.3, 56)), ("COM", 3.0, (-10, -24.3, 56)), ("A", 3.0, (10, -24.3, 56)), ("MA", 3.0, (31, -24.3, 56))], M("ink_white"))
L.hook("dmm_jack_com", (-10.5, -34.0, 40), note="COM jack mouth (lead attach)")
L.pop_group()

# ================================================================= 3. DMM leads
L.push_group("multimeter_leads", (-100, -300, 0))
for k, (col, sgn) in enumerate((("key_red", 1), ("abs_black", -1))):
    pts = [(sgn * 10, 0, 6), (sgn * 30, -50, 3), (sgn * 90, -90, 2), (sgn * 160, -40, 2), (sgn * 230, 40, 2), (sgn * 300, 30, 2), (sgn * 340, -30, 2)]
    L.tube("lead_%s" % ("red" if sgn > 0 else "black"), L.catmull(pts, 8), 1.8, (0, 0, 0), M(col), seg=8)
    L.lathe("banana_plug_%s" % ("red" if sgn > 0 else "black"), [(0, 0), (6, 0), (6, 24), (0, 24)], (sgn * 10, 12, 8.5), M(col), axis="y", seg=16)
    L.cyl("banana_pin_%s" % ("red" if sgn > 0 else "black"), 2.0, 14, (sgn * 10, 28, 8.5), M("chrome"), axis="y", seg=10)
    pb_ = L.lathe("test_probe_%s" % ("red" if sgn > 0 else "black"), [(0, 0), (5, 0), (7, 10), (7, 60), (3, 100), (0, 100)], (sgn * 340, -30, 8.0), M(col), axis="-y", seg=16)
    L.cyl("test_tip_%s" % ("red" if sgn > 0 else "black"), 0.9, 28, (0, -112, 0), M("steel"), axis="-y", seg=8, parent=pb_)
L.pop_group()

# ================================================================= 4. bench PSU (2 channel, 230 x 130 x 350)
L.push_group("bench_psu", (-900, 200, 0))
PW, PH, PD = 230.0, 130.0, 350.0
L.box("case", (PW, PD, PH), (0, 0, 8 + PH / 2), M("case_grey"), r=3)
L.box("front_panel", (PW - 4, 4, PH - 4), (0, -PD / 2 - 2, 8 + PH / 2), M("panel_dark"), r=2)
for sx in (-1, 1):
    L.box("foot_%s" % ("l" if sx < 0 else "r"), (40, 60, 8), (sx * 90, -100, 4), M("rubber"), r=2)
    L.box("foot_b_%s" % ("l" if sx < 0 else "r"), (40, 60, 8), (sx * 90, 100, 4), M("rubber"), r=2)
YP_ = -PD / 2 - 4
L.box("display_window", (100, 2, 52), (-40, YP_ - 1, 8 + 100), M("glass_dark"), r=1)
lcd("psu", 92, 46, (-40, YP_ - 2.2, 8 + 100), (0.0, 0.05, 0.1))
for i, x in enumerate((20, 55, 90)):
    L.knob("knob_%d" % (i + 1), (x, YP_, 8 + 100), 10, 14, M("knob_black"), seg=28)
for r_ in range(2):
    for c_ in range(4):
        L.box("btn_%d%d" % (r_, c_), (16, 3, 9), (-72 + c_ * 22, YP_ - 1.5, 8 + 60 - r_ * 14), M("key_grey"), r=1.2)
for i, (nm, mt) in enumerate((("pos", "key_red"), ("neg", "abs_black"), ("gnd", "key_green"))):
    L.lathe("terminal_%s" % nm, [(0, 0), (8, 0), (8, 5), (6.5, 14), (0, 14)], (-60 + i * 28, YP_, 8 + 28), M(mt), axis="-y", seg=20)
    L.cyl("terminal_%s_hole" % nm, 2.0, 1.5, (-60 + i * 28, YP_ - 14.2, 8 + 28), M("abs_black"), axis="y", seg=10)
L.cyl("power_switch", 7, 4, (90, YP_ - 2, 8 + 25), M("key_red"), axis="-y", seg=24, bev=0.6)
L.labels("psu_labels", [("CH1", 3.5, (-70, YP_ - 0.2, 8 + 80)), ("CH2", 3.5, (-30, YP_ - 0.2, 8 + 80)), ("V", 3.0, (20, YP_ - 0.2, 8 + 84)),
                        ("A", 3.0, (55, YP_ - 0.2, 8 + 84)), ("OUT", 3.0, (90, YP_ - 0.2, 8 + 84))], M("ink_white"))
L.cyl("fan_grille", 40, 2, (0, PD / 2 + 1, 8 + 65), M("abs_black"), axis="y", seg=32)
for k in range(3):
    L.tube("fan_ring_%d" % k, L.circle_pts(12 + k * 13, 40, (0, PD / 2 + 3, 8 + 65), plane="xz"), 0.8, (0, 0, 0), M("alu_dark"), seg=6, closed=True, cap=False)
L.box("iec_inlet", (30, 6, 22), (-80, PD / 2 + 2, 8 + 45), M("abs_black"), r=1)
L.hook("psu_terminal_pos", (-60, YP_ - 16, 8 + 28), note="positive output terminal mouth")
L.pop_group()

# ================================================================= 5. function generator (230 x 105 x 290)
L.push_group("function_generator", (-600, 200, 0))
FW, FH, FD = 230.0, 105.0, 290.0
L.box("case", (FW, FD, FH), (0, 0, 8 + FH / 2), M("case_grey"), r=3)
L.box("front_panel", (FW - 4, 4, FH - 4), (0, -FD / 2 - 2, 8 + FH / 2), M("panel_dark"), r=2)
YPf = -FD / 2 - 4
for sx in (-1, 1):
    L.box("foot_%s" % ("l" if sx < 0 else "r"), (40, 60, 8), (sx * 90, -80, 4), M("rubber"), r=2)
L.box("display_window", (100, 2, 56), (-50, YPf - 1, 8 + 68), M("glass_dark"), r=1)
lcd("function_generator", 92, 50, (-50, YPf - 2.2, 8 + 68), (0.0, 0.07, 0.1))
L.knob("main_knob", (70, YPf, 8 + 62), 17, 16, M("knob_black"), seg=36)
keys = [((14, 3, 8), (-95 + c_ * 18, YPf - 1.5, 8 + 28 - r_ * 12)) for r_ in range(2) for c_ in range(5)]
L.multi_box("keypad", keys, M("key_grey"), r=1.0)
for i in range(2):
    L.bnc_jack("bnc_out_%d" % (i + 1), (30 + i * 36, YPf, 8 + 22))
L.cyl("power_switch", 6, 4, (95, YPf - 2, 8 + 18), M("key_green"), axis="-y", seg=20, bev=0.5)
L.labels("fg_labels", [("OUTPUT", 3.0, (48, YPf - 0.2, 8 + 36)), ("WAVE", 3.0, (-86, YPf - 0.2, 8 + 40))], M("ink_white"))
L.hook("fg_output_bnc", (30, YPf - 14, 8 + 22), note="output 1 BNC mouth")
L.pop_group()

# ================================================================= 6. soldering station
L.push_group("soldering_station", (-100, 200, 0))
L.box("base", (110, 130, 105), (0, 0, 52.5), M("abs_dark"), r=8, seg=3)
L.box("base_panel", (96, 3, 70), (0, -66.5, 60), M("panel_dark"), r=2)
lcd("soldering_station", 40, 22, (0, -68.2, 82), (0.08, 0.02, 0.0))
L.knob("temp_knob", (0, -68, 52), 14, 14, M("knob_black"), seg=32)
L.cyl("power_switch", 5, 3, (-38, -68, 30), M("key_red"), axis="-y", seg=16)
L.cyl("iron_socket", 8, 8, (38, -68, 32), M("chrome"), axis="-y", seg=16)
# iron holder: tray with sponge and spring coil
L.box("holder_base", (90, 150, 22), (140, -20, 11), M("alu_dark"), r=4)
L.box("sponge", (60, 40, 14), (140, -70, 28), M("key_yellow"), r=3)
coil = [(140 + 14 * math.cos(a), 20 + 14 * math.sin(a) * 0.0, 22 + a * 8.0) for a in [i * 0.5 for i in range(0, 25)]]
L.tube("holder_post", [(120, 55, 22), (120, 55, 100)], 4.0, (0, 0, 0), M("steel"), seg=8)
L.tube("holder_post_2", [(160, 55, 22), (160, 55, 100)], 4.0, (0, 0, 0), M("steel"), seg=8)
L.tube("holder_cradle", [(120, 55, 100), (120, -10, 100), (160, -10, 100), (160, 55, 100)], 3.0, (0, 0, 0), M("steel"), seg=8)
L.tube("holder_spring", L.catmull([(140 + 12 * math.cos(a * 0.9), -10 + a * 3.0, 100 + 12 * math.sin(a * 0.9)) for a in range(0, 24)], 2), 1.2, (0, 0, 0), M("steel"), seg=6)
iron = L.lathe("iron_handle", [(0, 0), (8, 0), (9.5, 6), (9.5, 90), (7.5, 130), (6, 150), (0, 150)], (140, 40, 104), M("abs_black"), axis="-y", seg=24)
L.cyl("iron_grip", 10, 60, (0, -60, 0), M("key_blue"), axis="y", seg=24, parent=iron)
L.lathe("iron_tube", [(5, 150), (4, 165), (3.2, 200), (0, 200)], (0, 0, 0), M("steel"), axis="-y", seg=16, parent=iron)
L.lathe("iron_tip", [(2.2, 200), (1.2, 225), (0.4, 240), (0, 242)], (0, 0, 0), M("copper"), axis="-y", seg=12, parent=iron)
L.hook("iron_tip", (0, -242, 0), parent=iron, note="soldering tip (smoke / glow origin)")
cable = L.catmull([(140, 40 + 10, 104), (140, 110, 60), (100, 110, 3), (60, 80, 3), (40, 40, 5), (38, -62, 32)], 8)
L.tube("iron_cable", cable, 3.0, (0, 0, 0), M("abs_black"), seg=8)
L.pop_group()

# ================================================================= 7. microscope (boom stand inspection scope)
L.push_group("microscope", (250, 200, 0))
L.box("base_plate", (240, 300, 28), (0, 0, 14), M("case_grey"), r=6, seg=3)
L.cyl("stage_ring", 70, 6, (0, -30, 31), M("alu_dark"), seg=40, bev=1.0)
pcb = L.box("pcb_sample", (110, 80, 1.6), (0, -30, 35), M("pcb_green"), r=0.4)
L.multi_box("pcb_parts", [((12, 12, 2.5), (-30 + 22 * i, -30 + (-1) ** i * 12, 37.5)) for i in range(5)], M("abs_black"), r=0.5)
L.cyl("post", 20, 420, (0, 105, 28 + 210), M("alu"), seg=32, bev=1.5)
L.cyl("post_collar", 28, 30, (0, 105, 28 + 15), M("abs_dark"), seg=32, bev=2.0)
L.box("boom_clamp", (50, 50, 56), (0, 105, 340), M("abs_dark"), r=5)
L.box("boom_arm", (36, 190, 34), (0, 40, 340), M("abs_dark"), r=7, seg=3)
L.cyl("focus_knob_l", 22, 14, (-34, 105, 340), M("knob_black"), axis="x", seg=28, bev=1.0)
L.cyl("focus_knob_r", 22, 14, (34, 105, 340), M("knob_black"), axis="x", seg=28, bev=1.0)
hd = L.box("head_body", (90, 80, 70), (0, -30, 340), M("abs_dark"), r=8, seg=3)
L.cyl("objective_turret", 30, 22, (0, -30, 289), M("alu_dark"), seg=28, bev=1.5)
L.cyl("objective", 15, 56, (0, -30, 250), M("alu"), seg=24, bev=1.0)
L.cyl("objective_lens", 9, 2, (0, -30, 221.5), M("glass_dark"), seg=20)
L.tube("ring_light", L.circle_pts(28, 40, (0, -30, 236), plane="xy"), 4.0, (0, 0, 0), M("light_panel"), seg=8, closed=True, cap=False)
for sx in (-1, 1):
    L.cyl("eyepiece_tube_%s" % ("l" if sx < 0 else "r"), 12, 70, (sx * 30, -10, 400), M("abs_dark"), seg=20, rot=(-35, 0, 0), bev=1.0)
    L.cyl("eyepiece_%s" % ("l" if sx < 0 else "r"), 15, 30, (sx * 30, 13, 437), M("knob_black"), seg=20, rot=(-35, 0, 0), bev=2.0)
    L.cyl("eyecup_%s" % ("l" if sx < 0 else "r"), 17, 12, (sx * 30, 22, 448), M("rubber"), seg=20, rot=(-35, 0, 0), r2=14)
L.hook("microscope_eyepiece", (0, 20, 450), note="between the eyepieces (camera / glint)")
L.hook("microscope_stage_center", (0, -30, 36), note="sample on the stage")
L.pop_group()

# ================================================================= 8. laptop (open 105 deg, 14 in class: 320 x 225 x 16 base)
L.push_group("laptop", (700, 200, 0))
LW, LD, LH = 320.0, 225.0, 16.0
L.box("base", (LW, LD, LH), (0, 0, LH / 2 + 2), M("alu_dark"), r=3, seg=2)
for sx in (-1, 1):
    for sy in (-1, 1):
        L.cyl("foot_%d%d" % (sx, sy), 6, 2, (sx * 130, sy * 85, 1), M("rubber"), seg=12)
L.multi_box("keys", [((17.0, 17.0, 2.5), (-132 + c * 19.0, -LD / 2 + 128.0 - r_ * 19.0 + 0.0, LH + 3.2)) for r_ in range(5) for c in range(14)], M("abs_black"), r=1.0)
L.box("trackpad", (105, 70, 1.0), (0, -LD / 2 + 45.0, LH + 2.6), M("glass_dark"), r=2)
L.box("deck_cutout", (275, 100, 0.5), (0, 20, LH + 2.3), M("abs_black"))
HY, HZ = LD / 2, LH + 2
al = math.radians(105)
u = (0, -math.cos(al), math.sin(al))
n_ = (0, -math.sin(al), -math.cos(al))
n_ = (0, -math.sin(al), -math.cos(al))
lc = (0, HY + u[1] * LD / 2, HZ + u[2] * LD / 2)
L.box("lid", (LW, LD, 7.0), (lc[0] + n_[0] * -3.5, lc[1] + n_[1] * -3.5, lc[2] + n_[2] * -3.5), M("alu_dark"), r=3, rot=(-105, 0, 0))
L.box("lid_bezel", (LW - 8, LD - 8, 1.0), (lc[0] + n_[0] * 0.5, lc[1] + n_[1] * 0.5, lc[2] + n_[2] * 0.5), M("abs_black"), r=1, rot=(-105, 0, 0))
lcd("laptop", LW - 20.0, LD - 30.0, (lc[0] + n_[0] * 1.3, lc[1] + n_[1] * 1.3 - 0.0, lc[2] + n_[2] * 1.3), (0.02, 0.025, 0.03), facing="z", rot=(75, 0, 0))
L.cyl("hinge_barrel", 5, LW - 40, (0, HY, HZ), M("alu_dark"), axis="x", seg=16)
L.hook("laptop_screen_center", (lc[0] + n_[0] * 1.5, lc[1] + n_[1] * 1.5, lc[2] + n_[2] * 1.5), note="laptop screen centre")
L.pop_group()

# ================================================================= 9. ESD parts bins (4 open-front storage bins on a rail)
L.push_group("parts_bins", (300, -300, 0))
L.box("bin_rail", (460, 8, 20), (0, 90, 140), M("alu_dark"), r=1.5)
bin_cols = [("plastic_blue",), ("abs_black",), ("plastic_blue",), ("abs_black",)]
for i, (cm,) in enumerate(bin_cols):
    cx = -165 + i * 110
    mt = M(cm)
    # tray: bottom, back, two sides, low front lip, label holder
    L.box("bin_%d_bottom" % (i + 1), (100, 150, 3), (cx, 15, 62), mt, r=0.8)
    L.box("bin_%d_back" % (i + 1), (100, 3, 75), (cx, 88, 62 + 36), mt, r=0.8)
    for sx in (-1, 1):
        L.box("bin_%d_side_%s" % (i + 1, "l" if sx < 0 else "r"), (3, 150, 55), (cx + sx * 48.5, 15, 62 + 26), mt, r=0.8)
    L.box("bin_%d_front" % (i + 1), (100, 3, 35), (cx, -58, 62 + 17), mt, r=0.8)
    L.box("bin_%d_label" % (i + 1), (60, 1, 14), (cx, -59.8, 62 + 20), M("paper"), r=0.2, seg=1)
    # parts inside: SMD reels? chips, resistors
    if i % 2 == 0:
        L.multi_box("bin_%d_chips" % (i + 1), [((8 + (k % 3) * 2, 8, 2.5), (cx - 30 + (k % 5) * 14, -20 + (k // 5) * 18 - 10, 66.5)) for k in range(15)], M("abs_black"), r=0.5)
        L.multi_box("bin_%d_pins" % (i + 1), [((10 + (k % 3) * 2, 1.0, 1.0), (cx - 30 + (k % 5) * 14, -20 + (k // 5) * 18 - 10 + 4.3, 66.5)) for k in range(15)], M("steel"))
    else:
        for k in range(12):
            a = k * 2.4
            L.cyl("bin_%d_res_%d" % (i + 1, k), 1.8, 14, (cx - 25 + (k % 4) * 17, -10 + (k // 4) * 25, 66 + (k % 2)), M("plastic_cream"), axis="x", seg=8, rot=(0, 0, a * 25))
for sx in (-1, 1):
    L.box("rail_leg_%s" % ("l" if sx < 0 else "r"), (10, 60, 300), (sx * 240, 60, 150), M("alu_dark"), r=1.5)
L.pop_group()

# ================================================================= 10. cable spool
L.push_group("cable_spool", (650, -300, 0))
L.cyl("flange_a", 90, 3, (0, 0, 45 + 33), M("plastic_blue"), axis="y", seg=48, bev=0.6)
L.cyl("flange_b", 90, 3, (0, 70, 45 + 33), M("plastic_blue"), axis="y", seg=48, bev=0.6)
L.cyl("core", 32, 70, (0, 35, 45 + 33), M("plastic_blue"), axis="y", seg=32)
L.cyl("wound_cable", 74, 64, (0, 35, 45 + 33), M("key_red"), axis="y", seg=48)
L.lathe("wound_cable_ridges", [(74.2, -30), (74.2, 30)], (0, 35, 78), M("abs_black"), axis="y", seg=96)
L.tube("spool_loose_end", L.catmull([(0, 0, 111), (30, -40, 100), (90, -90, 30), (150, -70, 3), (200, -20, 3)], 8), 2.2, (0, 0, 0), M("key_red"), seg=8)
L.cyl("spool_stand_foot", 6, 6, (0, 35, 3), M("abs_black"), seg=12)
L.pop_group()

# ================================================================= 11. mug, 12. notebook
L.push_group("coffee_mug", (850, -300, 0))
prof = [(0, 0), (38, 0), (41, 3), (41, 92), (39.5, 95), (37.5, 92), (37.5, 6), (0, 6)]
L.lathe("mug_body", [(0, 0), (37, 0), (41, 4), (41, 92), (40.2, 95), (38.2, 95), (37.5, 92), (37.5, 8), (0, 8)], (0, 0, 0), M("ceramic_white"), seg=48)
L.cyl("coffee", 37.4, 2, (0, 0, 78), M("coffee"), seg=48)
hp = [(40.5 + 28 * math.sin(a), 0, 50 + 38 * math.cos(a)) for a in [(-2.3 + 4.6 * i / 14.0) for i in range(15)]]
hp = [(40.0 + 30.0 * math.sin(t), 0.0, 52.0 + 36.0 * math.cos(t)) for t in [(-2.2 + 4.4 * i / 16.0) for i in range(17)]]
L.tube("mug_handle", hp, 4.5, (0, 0, 0), M("ceramic_white"), seg=10)
L.hook("mug_rim", (0, 0, 95), note="mug rim centre (steam origin)")
L.pop_group()

L.push_group("notebook", (1000, -300, 0))
L.box("pages", (143, 203, 11), (0, 0, 7.5), M("paper"), r=0.6)
L.box("cover_bottom", (148, 210, 1.6), (0, 0, 0.8), M("cover_blue"), r=0.8)
L.box("cover_top", (148, 210, 1.6), (0, 0, 14.2), M("cover_blue"), r=0.8)
L.box("spine", (4, 210, 14.5), (-74, 0, 7.25), M("cover_blue"), r=1.5)
L.box("elastic_band", (3, 212, 0.8), (50, 0, 15.2), M("abs_black"))
L.box("bookmark_ribbon", (3, 25, 0.4), (-30, -115, 8), M("key_red"))
pen = L.cyl("pen", 5.0, 145, (0, 0, 20), M("plastic_blue"), axis="y", seg=16, rot=(0, 0, 12))
L.box("pen_clip", (1.5, 40, 2.5), (1.2, 22, 24.2), M("chrome"), r=0.4, rot=(0, 0, 12))
L.pop_group()

# ================================================================= dimensions
L.dim("handheld DMM", "96 x 50 x 196 with holster", "mm", "estimate; typical handheld DMM about 90 x 45 x 190 mm", "C")
L.dim("bench PSU", "230 x 350 x 138", "mm", "estimate for a 2-channel bench supply", "C")
L.dim("function generator", "230 x 290 x 113", "mm", "estimate", "C")
L.dim("soldering station base", "110 x 130 x 105", "mm", "estimate", "C")
L.dim("microscope", "240 x 300 x 450 incl. eyepieces", "mm", "estimate for a boom-stand inspection microscope", "C")
L.dim("laptop", "320 x 225 x 16 (14 in class), lid at 105 deg", "mm", "estimate; screen active 300 x 195", "C")
L.dim("ESD bins", "100 x 150 x 75 each", "mm", "estimate for small open-front bins", "C")
L.dim("cable spool", "180 dia x 70", "mm", "estimate", "C")
L.dim("mug", "82 dia x 95", "mm", "estimate; typical 11 oz mug", "B")
L.dim("notebook", "148 x 210 x 15 (A5)", "mm", "ISO A5 paper size", "A")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="all clutter sizes are generic estimates informed by common catalogue sizes", url="n/a", access_date="2026-10-02")],
    simplifications=["no brand marks; labels are generic words", "knobs/buttons not articulated", "bin contents are merged meshes", "laptop keys are one merged mesh"],
    custom_properties={},
    items=["probes_clips", "dmm", "multimeter_leads", "bench_psu", "function_generator", "soldering_station", "microscope", "laptop", "parts_bins", "cable_spool", "coffee_mug", "notebook"],
    screens=[dict(object=n, material="MAT_lab_office_screen_*", image_node="SCREEN_IMAGE", toggle_node="SCREEN_FAC", note="optional; idle dark by default") for n in screens],
    usage="Each item is an empty lab_clutter_<item> in its own collection ITEM_<item>; the layout in the file is a display row only: move each item empty to the bench (z = 0 at the item's feet)."))

L.plug_image(bpy.data.materials["MAT_lab_office_screen_laptop"], L.tex_paths("graph_s2", 100)[99])
L.plug_image(bpy.data.materials["MAT_lab_office_screen_psu"], L.tex_paths("spec_s5", 40)[39])
views = [("front", dict(loc=(100, -2.6, 0.9), target=(0.1, 0, 0.1), lens=35)),
         ("three_quarter", dict(loc=(1.8, -1.7, 1.1), target=(0.1, 0.0, 0.1), lens=35)),
         ("top", dict(loc=(0.1, -0.05, 3.3), target=(0.1, -0.05, 0), lens=35)),
         ("closeup_instruments", dict(loc=(-0.5, -0.9, 0.5), target=(-0.5, 0.2, 0.1), lens=40)),
         ("closeup_microscope_laptop", dict(loc=(0.55, -0.6, 0.55), target=(0.5, 0.2, 0.15), lens=35))]
views[0][1]["loc"] = (0.1, -2.6, 0.9)
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=True)
