"""Generic 4-channel bench oscilloscope (12.1 in class screen), no brand marks. Inspired by public photos/datasheets of
Keysight InfiniiVision 4000 X and Tektronix 4 Series MSO class instruments.
Run: Blender -b --python build_bench_oscilloscope.py -- <out_dir>
Origin: bottom centre of the instrument footprint (underside of the feet on z = 0). -Y = front (screen side).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import lo_common as L
from lo_common import C, M, S

OUT = C.argv_after_dashes()[0]
AID = "bench_oscilloscope"
L.begin(AID, "B")
root = S.root

# ---------------------------------------------------------------- envelope (mm)
W, D, FEET = 400.0, 150.0, 12.0
Z0, Z1 = FEET, 232.0           # body bottom/top
HB = Z1 - Z0
YF = -D / 2                     # front face of the shell (y)
YP = YF - 3.0                   # front of the panel plate (3 mm proud)
SCR_W, SCR_H = 256.0, 160.0     # 12.1 in diagonal at 16:10 (computed)
SCR_CX, SCR_CZ = -62.0, 134.0
root["p_screen_aspect"] = 1.6
root["p_screen_emission_strength"] = 1.0
root["p_channels"] = 4

# ---------------------------------------------------------------- materials
m_shell = M("case_grey")
m_panel = M("panel_dark")
m_bezel = M("abs_textured")
m_glass = M("glass_dark")
m_screen = L.screen_material("MAT_lab_office_screen")
m_ink = M("ink_white")
m_knob = M("knob_black")
m_alu = M("alu")

# ---------------------------------------------------------------- shell and panel
L.box("shell", (W, D, HB), (0, 0, (Z0 + Z1) / 2), m_shell, r=7.0, seg=3)
L.box("front_panel", (W - 12, 3.0, HB - 12), (0, YF - 1.5 + 0.0, (Z0 + Z1) / 2), m_panel, r=3.0, seg=2)
# side grips / rear shroud
L.box("rear_shroud", (W - 20, 6, HB - 20), (0, D / 2 + 3, (Z0 + Z1) / 2), M("abs_dark"), r=3.0)
# top vent slots
for k in range(14):
    L.box("vent_slot_%02d" % k, (4.0, 70.0, 0.6), (-95 + k * 14.5, 30.0, Z1 + 0.1), M("abs_black"), r=0.2, seg=1)

# ---------------------------------------------------------------- screen: bezel, glass, screen plane
bz = 8.0
win_w, win_h = SCR_W + 8, SCR_H + 8
for nm, size, loc in (("bezel_top", (win_w + 2 * bz, 2.5, bz), (SCR_CX, 0, SCR_CZ + win_h / 2 + bz / 2)),
                      ("bezel_bottom", (win_w + 2 * bz, 2.5, bz), (SCR_CX, 0, SCR_CZ - win_h / 2 - bz / 2)),
                      ("bezel_left", (bz, 2.5, win_h), (SCR_CX - win_w / 2 - bz / 2, 0, SCR_CZ)),
                      ("bezel_right", (bz, 2.5, win_h), (SCR_CX + win_w / 2 + bz / 2, 0, SCR_CZ))):
    L.box(nm, size, (loc[0], YP - 1.25, loc[2]), m_bezel, r=0.8, seg=2)
L.box("glass_backing", (win_w, 1.0, win_h), (SCR_CX, YP - 0.5, SCR_CZ), m_glass)
scr = L.uvquad("screen", SCR_W, SCR_H, (SCR_CX, YP - 1.4, SCR_CZ), m_screen, facing="-y")
scr.name = AID + "_screen"
scr["is_image_screen"] = True
scr.visible_shadow = False
# softkey row
sk_names = []
btn = L.box("softkey_tmpl", (30.0, 3.0, 9.0), (SCR_CX, YP - 1.5, 40.0), M("key_grey"), r=1.0, seg=2)
btn.name = AID + "_softkey_1"
sk_names.append(btn)
for i in range(1, 6):
    sk_names.append(L.dup(btn, "softkey_%d" % (i + 1), (SCR_CX + (i - 2.5) * 42.0 + (SCR_CX - SCR_CX), YP - 1.5, 40.0)))
btn.location.x = (SCR_CX + (0 - 2.5) * 42.0) * 0.001

# ---------------------------------------------------------------- right control column
CXR = 136.0  # column centre x
ZR = {}

# group separators (thin printed lines)
for z in (205.0, 140.0, 85.0):
    L.box("sep_%d" % int(z), (106.0, 0.3, 0.5), (CXR, YP - 0.15, z), m_ink, r=0.1, seg=1)

# run controls
run_btn = L.box("run_button", (22.0, 3.0, 9.0), (100.0, YP - 1.5, 214.0), M("key_green"), r=1.0, seg=2)
runs = [run_btn,
        L.box("single_button", (22.0, 3.0, 9.0), (126.0, YP - 1.5, 214.0), M("key_yellow"), r=1.0, seg=2),
        L.box("autoscale_button", (22.0, 3.0, 9.0), (152.0, YP - 1.5, 214.0), M("key_grey"), r=1.0, seg=2),
        L.box("default_button", (22.0, 3.0, 9.0), (178.0, YP - 1.5, 214.0), M("key_grey"), r=1.0, seg=2)]

# horizontal group: scale knob, position knob, two buttons
h_scale = L.knob("horizontal_scale_knob", (162.0, YP, 172.0), 13.0, 14.0, m_knob, seg=40)
h_pos = L.knob("horizontal_position_knob", (112.0, YP, 172.0), 8.0, 11.0, m_knob, seg=32)
L.box("zoom_button", (22.0, 3.0, 8.0), (112.0, YP - 1.5, 150.0), M("key_grey"), r=1.0)
L.box("search_button", (22.0, 3.0, 8.0), (140.0, YP - 1.5, 150.0), M("key_grey"), r=1.0)
# trigger group: level knob and three buttons
t_level = L.knob("trigger_level_knob", (162.0, YP, 118.0), 11.0, 13.0, m_knob, seg=36)
for j, z in enumerate((128.0, 112.0, 96.0)):
    L.box("trigger_button_%d" % (j + 1), (26.0, 3.0, 8.0), (112.0, YP - 1.5, z), M("key_grey"), r=1.0)
L.box("force_button", (20.0, 3.0, 8.0), (162.0, YP - 1.5, 94.0), M("key_red"), r=1.0)

# vertical group: 4 channels, pitch 24 mm (estimate)
BNC_Z = 22.0
chx = [108.0, 132.0, 156.0, 180.0]
ch_mats = ["ch1", "ch2", "ch3", "ch4"]
v_scale, v_pos, ch_btn, bncs = [], [], [], []
for i, x in enumerate(chx):
    cm = M(ch_mats[i])
    ch_btn.append(L.box("channel_button_%d" % (i + 1), (13.0, 3.0, 8.0), (x, YP - 1.5, 77.0), cm, r=1.0))
    v_scale.append(L.knob("vertical_scale_knob_%d" % (i + 1), (x, YP, 60.0), 9.0, 11.0, m_knob, cap_mat=cm, seg=32))
    v_pos.append(L.knob("vertical_position_knob_%d" % (i + 1), (x, YP, 41.0), 6.0, 8.0, m_knob, cap_mat=cm, seg=28))
    bncs.append(L.bnc_jack("bnc_ch%d" % (i + 1), (x, YP, BNC_Z)))
    L.box("bnc_color_ring_%d" % (i + 1), (17.0, 0.4, 17.0), (x, YP - 0.2, BNC_Z), cm, r=2.0, seg=2)

# lower-left strip: USB-A x2, probe comp terminals, power button
usb_tmpl = L.box("usb_a_1", (14.0, 4.0, 6.5), (-176.0, YP - 0.5, 24.0), M("abs_black"), r=0.5)
L.box("usb_a_1_tongue", (11.0, 3.0, 1.6), (-176.0, YP - 1.2, 24.0), M("gold"), r=0.2, seg=1)
usb2 = L.dup(usb_tmpl, "usb_a_2", (-152.0, YP - 0.5, 24.0))
L.box("usb_a_2_tongue", (11.0, 3.0, 1.6), (-152.0, YP - 1.2, 24.0), M("gold"), r=0.2, seg=1)
# probe compensation terminal and ground
L.cyl("probe_comp_ring", 5.0, 3.0, (-70.0, YP - 1.2, 24.0), M("chrome"), axis="-y", seg=24, bev=0.4)
L.cyl("probe_comp_pin", 1.2, 5.0, (-70.0, YP - 3.0, 24.0), M("gold"), axis="-y", seg=12)
L.cyl("probe_ground_ring", 5.0, 3.0, (-48.0, YP - 1.2, 24.0), M("chrome"), axis="-y", seg=24, bev=0.4)
L.cyl("probe_ground_post", 1.6, 6.0, (-48.0, YP - 3.0, 24.0), M("chrome"), axis="-y", seg=12)
# power button + LED ring
pwr = L.cyl("power_button", 5.5, 3.5, (-112.0, YP - 1.4, 24.0), M("key_grey"), axis="-y", seg=32, bev=0.5)
L.lathe("power_led_ring", [(6.5, 0), (7.6, 0), (7.6, 0.8), (6.5, 0.8)], (-112.0, YP - 0.9, 24.0), M("led_green"), axis="-y", seg=40)
# small status LEDs near the window
for i in range(3):
    L.cyl("status_led_%d" % (i + 1), 1.2, 0.8, (-186.0 + i * 5.0, YP - 0.4, 226.0), M(["led_green", "led_amber", "led_red"][i]), axis="-y", seg=10)

# ---------------------------------------------------------------- printed labels (flat text meshes)
tx = []
tx += [("HORIZONTAL", 3.2, (CXR, 0, 201.0)), ("TRIGGER", 3.2, (CXR, 0, 136.0)), ("VERTICAL", 3.2, (CXR, 0, 82.0))]
for i, x in enumerate(chx):
    tx.append((str(i + 1), 3.2, (x, 0, 69.0)))
tx += [("RUN/STOP", 2.4, (100.0, 0, 207.5)), ("SINGLE", 2.4, (126.0, 0, 207.5)), ("AUTO", 2.4, (152.0, 0, 207.5)),
       ("DEFAULT", 2.4, (178.0, 0, 207.5)), ("SCALE", 2.4, (162.0, 0, 154.0)), ("POS", 2.4, (112.0, 0, 184.0)),
       ("LEVEL", 2.4, (162.0, 0, 102.0)), ("FORCE", 2.4, (162.0, 0, 88.5)), ("ZOOM", 2.4, (112.0, 0, 144.5)),
       ("SEARCH", 2.4, (140.0, 0, 144.5)), ("PROBE COMP", 2.4, (-59.0, 0, 15.0)), ("USB", 2.4, (-164.0, 0, 15.0)),
       ("POWER", 2.4, (-112.0, 0, 14.0))]
tx = [(t, s, (p[0], YP - 0.15, p[2])) for (t, s, p) in tx]
L.labels("panel_labels", tx, m_ink)
# BNC input spec label
L.labels("bnc_labels", [("1 MOHM 14 PF 300 V", 2.0, (CXR - 10, YP - 0.15, 9.0))], M("ink_grey"))

# ---------------------------------------------------------------- handle (carry handle, folded back over the top) and hubs
hub_x = W / 2 + 1.0
hub_z = 150.0
hubs = []
for sgn in (-1, 1):
    hubs.append(L.cyl("handle_hub_%s" % ("l" if sgn < 0 else "r"), 15.0, 9.0, (sgn * (hub_x + 4.0), 0.0, hub_z), M("abs_dark"), axis="x", seg=36, bev=1.0))
arm = [(0, 0, hub_z), (0, 28, hub_z + 38), (0, 55, hub_z + 78), (0, 72, hub_z + 98), (0, 80, hub_z + 106)]
hpts = L.catmull([(0, y, z) for (_, y, z) in arm], 5)
path_l = [(-(hub_x + 8.0), y, z) for (_, y, z) in hpts]
path_r = [((hub_x + 8.0), y, z) for (_, y, z) in hpts]
# arch across the top: from left arm end to right arm end
top_pts = L.catmull([path_l[-1], (-(hub_x - 20), 84, hub_z + 106 + 1), (0, 86, hub_z + 107), ((hub_x - 20), 84, hub_z + 107), path_r[-1]], 6)
full = path_l + top_pts[1:] + path_r[::-1][1:]
handle = L.tube("handle", full, 6.5, (0, 0, 0), M("abs_dark"), seg=12)

# ---------------------------------------------------------------- feet and tilt legs
for sx in (-1, 1):
    for sy in (-1, 1):
        L.box("foot_%s%s" % ("l" if sx < 0 else "r", "f" if sy < 0 else "b"), (44.0, 38.0, FEET), (sx * 150.0, sy * 52.0, FEET / 2), M("rubber"), r=3.0, seg=2)
for sx in (-1, 1):
    L.box("tilt_leg_%s" % ("l" if sx < 0 else "r"), (36.0, 22.0, 5.0), (sx * 90.0, -48.0, Z0 + 1.8), M("abs_dark"), r=1.2)
    L.cyl("tilt_hinge_%s" % ("l" if sx < 0 else "r"), 3.0, 36.0, (sx * 90.0, -34.0, Z0 + 2.5), M("alu_dark"), axis="x", seg=12)

# ---------------------------------------------------------------- rear panel (+Y)
YR = D / 2 + 6.0
# fan grille: dark recess, concentric wire rings and spokes
FANX, FANZ, FR_ = -85.0, 125.0, 48.0
L.cyl("fan_recess", FR_ + 3.0, 2.0, (FANX, YR - 0.8, FANZ), M("abs_black"), axis="y", seg=48)
for k, rr in enumerate((12.0, 24.0, 36.0, 47.0)):
    L.tube("fan_ring_%d" % k, L.circle_pts(rr, 56, (FANX, YR + 1.2, FANZ), plane="xz"), 0.9, (0, 0, 0), M("alu_dark"), seg=6, closed=True, cap=False)
for k in range(8):
    a = math.pi * k / 4
    L.box("fan_spoke_%d" % k, (47.0, 1.4, 1.4), (FANX + 23.5 * math.cos(a), YR + 1.2, FANZ + 23.5 * math.sin(a)), M("alu_dark"), rot=(0, -math.degrees(a), 0))
L.cyl("fan_hub", 8.0, 3.0, (FANX, YR + 0.5, FANZ), M("abs_black"), axis="y", seg=24)
# IEC C14 inlet with fuse drawer
L.box("iec_inlet", (28.0, 6.0, 20.0), (130.0, YR - 1.0, 70.0), M("abs_black"), r=1.0)
L.box("iec_inlet_recess", (22.0, 3.0, 14.0), (130.0, YR + 1.2, 70.0), M("abs_dark"), r=0.5)
for sx, sz, dz in ((-6, 3, 0), (6, 3, 0), (0, -3.5, 0)):
    L.box("iec_pin", (2.2, 3.0, 4.0), (130.0 + sx, YR + 1.8, 70.0 + sz), M("chrome"), r=0.2, seg=1)
L.box("fuse_drawer", (28.0, 3.0, 8.0), (130.0, YR + 0.2, 52.0), M("abs_dark"), r=0.6)
# I/O panel: LAN, USB-B, video out, trigger out, Kensington slot, ground lug
io = L.box("io_plate", (120.0, 2.0, 40.0), (20.0, YR + 0.2, 70.0), M("panel_dark"), r=1.0)
L.box("lan_port", (16.0, 3.0, 14.0), (-20.0, YR + 1.2, 72.0), M("chrome"), r=0.5)
L.box("lan_port_cavity", (12.0, 2.0, 9.0), (-20.0, YR + 2.2, 71.0), M("abs_black"))
L.box("usb_b_port", (12.0, 3.0, 11.0), (4.0, YR + 1.2, 72.0), M("chrome"), r=0.5)
L.box("usb_b_cavity", (8.0, 2.0, 7.0), (4.0, YR + 2.2, 72.0), M("abs_black"))
L.box("video_port", (22.0, 3.0, 9.0), (34.0, YR + 1.2, 72.0), M("chrome"), r=0.5)
L.box("video_port_cavity", (18.0, 2.0, 5.0), (34.0, YR + 2.2, 72.0), M("abs_black"))
L.bnc_jack("bnc_trig_out", (62.0, YR + 0.2, 72.0))
bpy.data.objects[AID + "_bnc_trig_out"].rotation_euler = (0, 0, 0)
L.box("kensington_slot", (7.0, 3.0, 3.0), (-60.0, YR + 1.2, 85.0), M("abs_black"), r=0.5)
L.cyl("ground_lug", 4.0, 5.0, (-170.0, YR + 2.0, 40.0), M("chrome"), axis="y", seg=16, bev=0.4)
L.labels("rear_labels", [("LAN", 2.4, (-20, YR + 1.4, 82.0)), ("USB", 2.4, (4, YR + 1.4, 82.0)), ("VIDEO", 2.4, (34, YR + 1.4, 82.0)),
                         ("TRIG OUT", 2.4, (62, YR + 1.4, 60.0))], M("ink_grey"), facing="-y")
rl = bpy.data.objects[AID + "_rear_labels"]
# regulatory label rectangle
L.box("rating_label", (60.0, 0.5, 22.0), (125.0, YR + 0.2, 105.0), M("paper"), r=0.3, seg=1)

# ---------------------------------------------------------------- passive probe on CH1 (lies on the bench plane, tip at the end)
L.push_group("probe_ch1", (0, 0, 0))
PX0 = chx[0]
yb = YP - 13.6
plug = L.cyl("probe_bnc_plug", 6.5, 18.0, (PX0, yb - 9.0, BNC_Z), M("chrome"), axis="y", seg=24, bev=0.6)
L.cyl("probe_bnc_collar", 8.0, 10.0, (PX0, yb - 22.0, BNC_Z), M("abs_black"), axis="y", seg=24, bev=0.8)
cab = L.catmull([(PX0, yb - 24, BNC_Z), (PX0, yb - 60, BNC_Z - 8), (PX0 + 4, yb - 110, 5.0), (PX0 - 20, yb - 190, 2.6),
                 (PX0 - 90, yb - 260, 2.6), (PX0 - 150, yb - 330, 2.6), (PX0 - 160, yb - 380, 3.0)], 6)
L.tube("probe_cable", cab, 2.0, (0, 0, 0), M("abs_black"), seg=8)
cx0, cy0 = PX0 - 160, yb - 380
L.box("probe_comp_box", (24.0, 52.0, 16.0), (cx0, cy0 - 20.0, 8.5), M("abs_dark"), r=3.0)
L.box("probe_comp_box_dial", (6.0, 12.0, 2.0), (cx0, cy0 - 20.0, 17.4), M("alu"), r=0.6)
# probe body and tip
body = L.lathe("probe_body", [(0, 0), (4.8, 0), (5.4, 6), (5.4, 70), (4.2, 90), (3.2, 104), (0, 104)], (cx0 - 3.0, cy0 - 70.0, 6.0), M("abs_dark"), axis="-y", seg=24)
L.cyl("probe_body_grip", 5.6, 40.0, (0, -30.0, 0), M("key_blue"), axis="y", seg=24, parent=body)
L.lathe("probe_tip", [(0, 0), (2.2, 0), (1.6, 6), (0.7, 10), (0.4, 16), (0, 18)], (0, -104.0, 0), M("steel"), axis="-y", seg=12, parent=body)
tip = L.hook("probe_tip", (0, -122.0, 0), parent=body, note="probe needle tip (probe lies on the bench plane z=0 of this asset)")
# ground lead
gl = L.catmull([(cx0 - 3.0 + 4, cy0 - 78.0, 6.5), (cx0 + 12, cy0 - 90, 4), (cx0 + 22, cy0 - 120, 2.0), (cx0 + 20, cy0 - 150, 2.0)], 6)
L.tube("probe_ground_lead", gl, 0.9, (0, 0, 0), M("abs_black"), seg=6)
L.box("probe_ground_clip", (8.0, 22.0, 6.0), (cx0 + 20, cy0 - 160.0, 4.0), M("steel"), r=1.0)

L.pop_group()

# ---------------------------------------------------------------- hooks
L.hook("screen_center", (SCR_CX, YP - 1.6, SCR_CZ), rot_deg=(90, 0, 0), note="screen centre; arrows: local -Z of the empty... use as camera aim point; screen faces -Y")
L.hook("ch1_bnc", (chx[0], YP - 13.6, BNC_Z), note="CH1 BNC mouth (probe attach)")
L.hook("trigger_level_knob", (162.0, YP - 14.0, 118.0), note="trigger level knob front")
L.hook("power_button", (-112.0, YP - 3.0, 24.0), note="power button")

# ---------------------------------------------------------------- dimensions
L.dim("overall width (shell)", W, "mm", "estimate; envelope class from Tektronix 4 Series MSO spec 405 mm (https://www.tek.com/en/manual/oscilloscope/4-series-mso-specifications-and-performance-verification-4-series-mso)", "B")
L.dim("overall depth (shell, excl. knobs and feet)", D, "mm", "estimate; Tektronix 4 Series MSO lists 155 mm depth from feet to knobs (same URL)", "B")
L.dim("body height (shell)", HB, "mm", "estimate for a 12.1 in class bench scope", "C")
L.dim("feet height", FEET, "mm", "estimate", "C")
L.dim("screen diagonal", 12.1, "in", "task class (Keysight InfiniiVision 4000 X-Series: 12.1 in capacitive touch screen, https://www.keysight.com/zz/en/products/oscilloscopes/infiniivision-2-4-channel-digital-oscilloscopes/infiniivision-4000-x-series-oscilloscopes.html)", "A")
L.dim("active screen area", "256 x 160", "mm", "computed from 12.1 in diagonal at 16:10 (matches the 384x240 eye textures without stretch); Tektronix 13.3 in 16:9 display area is 289 x 165 mm for comparison", "A")
L.dim("channel BNC pitch", 24.0, "mm", "estimate (not found in a primary source)", "C")
L.dim("BNC barrel OD", 9.6, "mm", "BNC connector standard practice (MIL-STD-348 class dimensions), not re-verified", "B")
L.dim("knob diameters", "26 (horizontal), 22 (trigger), 18 (vertical scale), 12 (position)", "mm", "estimate", "C")
L.dim("handle", "tube 13 mm, arch folded back over the top", "mm", "estimate", "C")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="screen class, envelope size comparison", url="https://www.tek.com/en/manual/oscilloscope/4-series-mso-specifications-and-performance-verification-4-series-mso", access_date="2026-10-02"),
             dict(what="12.1 in screen class, front-panel group layout (vertical, horizontal, trigger, run control)", url="https://www.keysight.com/zz/en/products/oscilloscopes/infiniivision-2-4-channel-digital-oscilloscopes/infiniivision-4000-x-series-oscilloscopes.html", access_date="2026-10-02"),
             dict(what="physical envelope numbers other than the above are generic estimates (accuracy C)", url="n/a", access_date="2026-10-02")],
    simplifications=["no brand marks or real model names; labels are generic words", "knobs have no knurling; pointer marks are small boxes",
                     "rear I/O panel is schematic; no internal parts", "passive probe is a simplified lathe-built probe on the bench plane"],
    custom_properties={"p_screen_aspect": "1.6 (16:10)", "p_screen_emission_strength": "informational; set the SCREEN_EMISSION node Strength in the material",
                       "p_channels": "4"},
    screens=[dict(object=AID + "_screen", material="MAT_lab_office_screen", image_node="SCREEN_IMAGE", toggle_node="SCREEN_FAC (0 idle dark, set 1.0 after plugging)",
                  aspect="16:10 (384x240 eye sequence maps 1:1)", note="assembler: img.source='SEQUENCE'; image_user.frame_duration=N; frame_offset to start; emission strength node SCREEN_EMISSION")],
    usage="S1 and S2 bench scope showing the eye (eye_s1 / eye_s2 sequences); HOOK_screen_center for camera pushes; HOOK_probe_tip for Pulse/probe gags."))

# ---------------------------------------------------------------- previews with the eye sequence plugged (not saved)
L.plug_image(m_screen, L.tex_paths("eye_s1", 150)[149], as_sequence=False)
views = [("front", dict(loc=(0, -1.35, 0.14), target=(0, 0, 0.12), lens=50)),
         ("three_quarter", dict(loc=(0.75, -0.95, 0.5), target=(0.0, -0.1, 0.1), lens=40)),
         ("top", dict(loc=(0, -0.12, 1.0), target=(0, -0.12, 0), lens=50)),
         ("back", dict(loc=(-0.6, 0.9, 0.35), target=(0, 0.05, 0.12), lens=40)),
         ("closeup_panel", dict(loc=(0.14, -0.45, 0.13), target=(0.14, -0.08, 0.1), lens=50)),
         ("closeup_screen", dict(loc=(-0.06, -0.55, 0.135), target=(-0.06, -0.08, 0.134), lens=50))]
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=True)
