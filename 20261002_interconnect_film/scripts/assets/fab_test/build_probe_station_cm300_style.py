"""probe_station_cm300_style: CM300xi-ULN style 300 mm probe station (generic labels, no vendor marks).

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/fab_test/build_probe_station_cm300_style.py -- assets/components/fab_test
All coordinates are millimetres in asset space (origin: floor centre of the cabinet footprint, -Y front, Z up).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft  # noqa: E402
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
AID = "probe_station_cm300_style"
A = ft.Asset(AID, accuracy="B")
MM = ft.MM


def n(part):
    return AID + "_" + part


# ------------------------------------------------------------------ key coordinates (mm)
CY = 40.0           # y of the probe centre / deck opening centre
ZD = 1060.0         # deck top (FormFactor drawing: 1060 mm floor to deck)
ZC = 1017.0         # chuck top at probing height (platen to chuck top 43 mm, public datasheet)
WAFER_T = 0.775
W2 = 520.0          # half width of cabinet (1040 mm per drawing)
D2 = 582.5          # half depth (1165 mm per drawing)
FRONT = -D2

# ------------------------------------------------------------------ sources / dimension table
A.src("FormFactor CM300xi-ULN Facility Planning Guide PN 188-101-6 (station drawings p7-8, dimensions p4-5)",
      "https://psirep.com/system/files/188-101-6-cm300xi-uln-fpg.pdf_-_188-101-cm300xi-uln-fpg.pdf",
      "cabinet width/depth, deck height 1060, top 1478/1547, light tower 2146, weight, clearances")
A.src("FormFactor CM300xi-ULN product page / datasheet (chuck 300 mm, chuck-to-platen 43 mm, XY travel 301 x 501 mm)",
      "https://www.formfactor.com/product/probe-systems/300-mm-systems/cm300xi-uln/",
      "stage travel, chuck size, platen height")
A.src("Wentao reference photo of the CM300xi ULN front (drawer open)", "references/formfactor-CM300.png",
      "layout of fascia, drawer, deck, positioners, microscope column; proportions via 300 mm wafer scale in the photo")
A.dim("cabinet width", 1040, "mm", "FPG p7 front view (feet span; 1154 overall at base, 1600 with side shelf)", "A")
A.dim("cabinet depth", 1165, "mm", "FPG p8 side view (1165 body, 1279 / 1535 with accessories)", "A")
A.dim("deck height above floor", 1060, "mm", "FPG p7 front view", "A")
A.dim("top of microscope arch above floor", 1478, "mm", "FPG p7 front view", "A")
A.dim("highest point without light tower", 1547, "mm", "FPG p7 front view (monitor top, used here)", "A")
A.dim("overall height with light tower", 2146, "mm", "FPG p8 side view", "A")
A.dim("XY stage travel", "301 x 501", "mm", "FormFactor CM300xi-ULN datasheet", "A")
A.dim("chuck diameter (top plate)", 320, "mm", "300 mm wafer chuck; exact OD not published, wafer 300 mm + rim", "C")
A.dim("chuck top to platen", 43, "mm", "FormFactor datasheet 'chuck to platen height'", "A")
A.dim("platen ring OD / aperture", "524 / 330", "mm", "estimated from photo (ring about 480 mm wide vs 300 mm wafer)", "C")
A.dim("drawer opening", "470 x 245", "mm", "estimated from photo scale (300 mm wafer = 206 px)", "C")
A.dim("positioner body", "80 x 64 x 48", "mm", "estimated from photo (about 85 mm wide vs wafer 300 mm)", "C")
A.dim("fascia posts diameter", 96, "mm", "estimated from photo", "C")
A.dim("monitor", "500 x 300", "mm", "estimated (about 24 inch)", "C")

# ------------------------------------------------------------------ materials
m_black = A.mat("black_paint")
m_plastic = A.mat("black_plastic")
m_anod = A.mat("black_anodized")
m_deck = A.mat("deck_silver")
m_silver = A.mat("silver")
m_steel = A.mat("steel")
m_dark = A.mat("steel_dark")
m_chrome = A.mat("chrome")
m_rubber = A.mat("rubber")
m_glass = A.mat("glass")
m_red = A.mat("red")
m_yel = A.mat("yellow")
m_white = A.mat("label_white")
m_tung = A.mat("tungsten")
m_gold = A.mat("gold")
m_led_g = A.mat("green_led")
m_led_r = A.mat("red_led")
m_wp = A.mat("white_plastic")

R = A.root
# custom props (documented in metadata)
A.prop("p_stage_x", 0.0, "Wafer stage X offset under the fixed probe head (mm, +X to viewer's right)", -150.5, 150.5, "mm")
A.prop("p_stage_y", 0.0, "Wafer stage Y offset (mm, +Y away from viewer / toward the back)", -250.5, 250.5, "mm")
A.prop("p_stage_theta", 0.0, "Wafer stage theta rotation about Z (degrees)", -180.0, 180.0, "deg")
A.prop("p_drawer", 0.0,
       "0 = door closed, chuck under the probe head; 0..0.1 door slides down; 0.1..1 chuck/tray slides out the front "
       "(500 mm) and lowers 60 mm; 1 = fully open load position", 0.0, 1.0, "")

# ------------------------------------------------------------------ cabinet shell
for dx in (-460, 460):
    for dy in (-520, 520):
        A.cyl(n("foot"), 30, 60, (dx, dy, 0), "black_anodized", anchor="b", seg=24)
A.box(n("cabinet_floor"), (1040, 1165, 20), (0, 0, 60), "black_plastic", anchor="b")
for sx in (-1, 1):
    A.box(n("cabinet_side_panel"), (24, 1068, 975), (sx * (W2 - 12), 48, 80), "black_paint", anchor="b", bev=3)
A.box(n("cabinet_back_panel"), (1040, 24, 975), (0, D2 - 12, 80), "black_paint", anchor="b", bev=3)
A.box(n("cabinet_inner_deck_support"), (990, 1100, 12), (0, 0, 1023), "black_plastic", anchor="b")
# side panel vent slots (instanced)
slot = A.box(n("vent_slot_proto"), (3, 150, 14), (W2 + 1, 150, 360), "steel_dark")
for sx in (-1, 1):
    for k in range(10):
        A.dup(slot, n("vent_slot"), (sx * (W2 + 1), -80 + k * 36 + 150, 360 + 120), parent=R)
A.parts[n("vent_slot_proto")].hide_render = True
A.parts[n("vent_slot_proto")].hide_viewport = True

# front panel with drawer opening
panel = A.box(n("front_panel"), (848, 24, 955), (0, FRONT + 12, 80), "black_paint", anchor="b", bev=3)
cutter = A.box("cutter", (470, 60, 245), (0, FRONT + 12, 790), "steel", anchor="b")
A.cut(panel, [cutter])
# vent slats on the lower front (instanced)
vslat = A.box(n("vent_slat_proto"), (22, 3, 140), (0, FRONT - 1.5, 160), "steel_dark", anchor="b", bev=1)
for k in range(15):
    A.dup(vslat, n("vent_slat"), (-350 + k * 50, FRONT - 1.5, 160 + 70))
A.parts[n("vent_slat_proto")].hide_render = True
A.parts[n("vent_slat_proto")].hide_viewport = True
# fascia posts (cylinders)
for sx in (-1, 1):
    A.cyl(n("post"), 48, 1000, (sx * (W2 - 48), FRONT + 48, 62), "black_paint", anchor="b", seg=48, bev=4)
    A.cyl(n("post_cap"), 49, 6, (sx * (W2 - 48), FRONT + 48, 1062), "black_anodized", anchor="b", seg=48)
# fascia panel inserts and controls
for sx in (-1, 1):
    A.box(n("fascia_insert"), (170, 5, 225), (sx * 330, FRONT - 2.5, 810), "black_plastic", anchor="b", bev=10, bev_seg=3)
    A.cyl(n("knob_base"), 30, 6, (sx * 330, FRONT - 8, 925), "black_anodized", axis="y", seg=40)
    A.cyl(n("knob"), 24, 22, (sx * 330, FRONT - 20, 925), "silver", axis="y", seg=48, bev=3)
    A.cyl(n("knob_cap"), 15, 3, (sx * 330, FRONT - 32.5, 925), "steel_dark", axis="y", seg=32)
A.cyl(n("small_button"), 8, 10, (-330, FRONT - 9, 845), "silver", axis="y", seg=24, bev=1.5)
# thumb lever slot (left)
A.box(n("lever_slot"), (22, 4, 120), (-397, FRONT - 6, 905), "steel_dark", anchor="c", bev=6, bev_seg=3)
A.box(n("lever"), (8, 14, 18), (-397, FRONT - 13, 940), "rubber", anchor="c", bev=2)
# emergency stop (right)
A.cyl(n("estop_ring"), 34, 6, (355, FRONT - 7, 838), "yellow", axis="y", seg=40)
A.cyl(n("estop_housing"), 28, 10, (355, FRONT - 15, 838), "black_anodized", axis="y", seg=40, bev=2)
A.cyl(n("estop_button"), 22, 18, (355, FRONT - 28, 838), "red", axis="y", seg=40, bev=6, bev_seg=3)
# generic labels (not vendor marks)
A.text(n("label_left"), "WAFER PROBE", 11, (-330, FRONT - 6, 1000), "label_white", rot=(math.pi / 2, 0, 0), extrude=0.3)
A.text(n("label_right_1"), "CM300-STYLE", 13, (330, FRONT - 6, 1000), "label_white", rot=(math.pi / 2, 0, 0), extrude=0.3)
A.text(n("label_right_2"), "ULN", 8, (330, FRONT - 6, 983), "label_white", rot=(math.pi / 2, 0, 0), extrude=0.3)

# drawer opening trim (silver frame) and guide rails
tx, tz = 250, 245
for z in (790 - 4, 1035 + 0):
    A.box(n("opening_trim_h"), (500, 8, 8), (0, FRONT - 3, z), "silver", bev=1.5)
for sx in (-1, 1):
    A.box(n("opening_trim_v"), (8, 8, 255), (sx * 246, FRONT - 3, 790 - 4 + 127), "silver", bev=1.5)
    A.box(n("drawer_guide_rail"), (10, 520, 14), (sx * 232, FRONT + 150, 800), "steel", bev=1.5)

# drawer door (slides down into the hollow lower body)
HD = A.hook("drawer_door", (0, FRONT + 38, 790 + 122.5), R)
A.box(n("drawer_door"), (470, 12, 245), (0, FRONT + 38, 790), "black_paint", anchor="b", parent=HD, bev=2)
A.box(n("drawer_door_handle"), (110, 8, 10), (0, FRONT + 30, 805), "silver", parent=HD, bev=2)
A.drive(HD, "location", 2, "%.6f-0.245*min(1,p_drawer/0.1)" % ((790 + 122.5) * MM), ["p_drawer"])

# ------------------------------------------------------------------ deck
deck = A.box(n("deck"), (1040, 1140, 25), (0, 0, 1035), "deck_silver", anchor="b", bev=2)
hole = ft.C.cylinder("cutter", 262 * MM, 80 * MM, loc=(0, CY * MM, 1048 * MM), verts=160)
ft.C.add(hole, A.coll, R)
A.cut(deck, [hole])
A.box(n("deck_front_lip"), (884, 40, 32), (0, FRONT - 28, 1028), "deck_silver", anchor="b", bev=9, bev_seg=4)
# platen ring and probe card ring (centre of the deck)
A.cyl(n("platen_ring"), 262, 30, (0, CY, 1034), "deck_silver", anchor="b", r_in=165, bev=3, seg=96)
A.cyl(n("platen_rim"), 270, 6, (0, CY, 1057), "black_anodized", anchor="b", r_in=255, seg=96)
A.cyl(n("probe_ring"), 188, 14, (0, CY, 1064), "steel", anchor="b", r_in=166, bev=2, seg=96)
# deck screws (instanced)
scr = A.cyl(n("screw_proto"), 4.5, 2.5, (0, 0, 1061), "steel_dark", anchor="b", seg=12)
scr.hide_render = scr.hide_viewport = True
for k in range(24):
    a = 2 * math.pi * k / 24
    A.dup(scr, n("deck_screw"), (275 * math.cos(a), CY + 275 * math.sin(a), 1061 + 1.25), parent=R)
# slanted rails on the deck (left and right)
for sx, rot in ((-1, math.radians(75)), (1, math.radians(-75))):
    A.box(n("deck_rail"), (22, 300, 4), (sx * 395, CY - 40, 1060), "silver", anchor="b", bev=1, rot_z=0.0)
    for k in range(5):
        A.box(n("deck_rail_slot"), (6, 28, 1.5), (sx * 395, CY - 150 + k * 55, 1064), "steel_dark", anchor="b")

# ------------------------------------------------------------------ stage chain (drawer -> x -> y -> theta)
H_DR = A.hook("drawer", (0, CY, ZC), R, size=0.08)
H_X = A.hook("stage_x", (0, CY, ZC), H_DR, size=0.06)
H_Y = A.hook("stage_y", (0, CY, ZC), H_X, size=0.05)
H_T = A.hook("stage_theta", (0, CY, ZC), H_Y, size=0.04)
H_W = A.hook("wafer_slot", (0, CY, ZC), H_T, size=0.1)
A.drive(H_DR, "location", 1, "%.6f-0.50*max(0,(p_drawer-0.1)/0.9)" % (CY * MM), ["p_drawer"])
A.drive(H_DR, "location", 2, "%.6f-0.06*max(0,(p_drawer-0.1)/0.9)" % (ZC * MM), ["p_drawer"])
A.drive(H_X, "location", 0, "p_stage_x*0.001", ["p_stage_x"])
A.drive(H_Y, "location", 1, "p_stage_y*0.001", ["p_stage_y"])
A.drive(H_T, "rotation_euler", 2, "p_stage_theta*0.01745329", ["p_stage_theta"])

# drawer tray (moves with the drawer only)
TY = CY - 20
A.box(n("drawer_tray"), (440, 360, 14), (0, TY, 860), "silver", anchor="b", parent=H_DR, bev=2)
A.box(n("drawer_front_bar"), (440, 26, 46), (0, TY - 193, 850), "white_plastic", anchor="b", parent=H_DR, bev=4)
A.box(n("drawer_front_slot"), (130, 4, 12), (0, TY - 207, 872), "black_plastic", parent=H_DR, bev=3)
for sx in (-1, 1):
    A.cyl(n("drawer_latch"), 14, 12, (sx * 190, TY - 206, 872), "black_anodized", axis="y", parent=H_DR, seg=32, bev=2)
    A.box(n("drawer_clamp_block"), (66, 90, 60), (sx * 176, TY - 120, 874), "silver", anchor="b", parent=H_DR, bev=3)
    A.box(n("drawer_clamp_pad"), (50, 40, 14), (sx * 176, TY - 140, 934), "black_anodized", anchor="b", parent=H_DR, bev=2)
    A.box(n("drawer_warning_label"), (22, 1, 22), (sx * 176, TY - 165.5, 900), "yellow", parent=H_DR)

# stage carriages (x then y then theta), brushed silver
A.box(n("stage_x_carriage"), (340, 440, 22), (0, CY, 890), "steel", anchor="b", parent=H_X, bev=2)
A.box(n("stage_y_carriage"), (330, 330, 22), (0, CY, 912), "steel", anchor="b", parent=H_Y, bev=2)
A.cyl(n("stage_theta_ring"), 142, 22, (0, CY, 934), "silver", anchor="b", parent=H_T, seg=96, bev=1.5)
A.cyl(n("chuck_skirt"), 158, 40, (0, CY, 956), "silver", anchor="b", parent=H_T, seg=96, bev=2)
# chuck top plate with vacuum grooves (lathe)
# grooves from outside to inside
pts = []
for rg in (150, 120, 90, 60, 30):
    pts += [(rg + 1.5, 22), (rg + 1.0, 20.6), (rg - 1.0, 20.6), (rg - 1.5, 22)]
pts += [(0, 22)]
prof = [(0, 0), (160, 0), (160, 21), (158, 22)] + pts
A.lathe(n("chuck_top"), prof, (0, CY, 995), "silver", parent=H_T, seg=120, sharp_deg=40)
# chuck wafer-lift pins and centre dot (decor)
A.cyl(n("chuck_ring_band"), 160.5, 4, (0, CY, 1008), "black_anodized", parent=H_T, seg=96, r_in=159)

# ------------------------------------------------------------------ gantry arch and microscope column
APTH = [(-400, 1062), (-400, 1290)]
for k in range(1, 11):
    a = math.radians(180 - 90 * k / 10)
    APTH.append((-250 + 150 * math.cos(a), 1290 + 150 * math.sin(a)))
for k in range(0, 11):
    a = math.radians(90 - 90 * k / 10)
    APTH.append((250 + 150 * math.cos(a), 1290 + 150 * math.sin(a)))
APTH.append((400, 1062))
A.sweep_rect(n("gantry_arch"), APTH, 70, 120, CY, "black_paint")
for sx in (-1, 1):
    A.box(n("gantry_foot"), (110, 150, 12), (sx * 400, CY, 1060), "black_anodized", anchor="b", bev=2)
A.box(n("scope_slide_rail"), (300, 40, 14), (0, CY, 1395), "silver", anchor="b", bev=1.5)
A.box(n("scope_carriage"), (280, 140, 120), (0, CY, 1275), "black_anodized", anchor="b", bev=6)
A.box(n("scope_column"), (100, 100, 175), (0, CY, 1100), "black_paint", anchor="b", bev=8)
A.box(n("scope_label_plate"), (60, 2, 40), (0, CY - 51, 1225), "label_white")
A.text(n("scope_label_text"), "VUE-STYLE", 9, (0, CY - 52.5, 1225), "black_plastic", rot=(math.pi / 2, 0, 0), extrude=0.2)
A.box(n("scope_bracket"), (90, 12, 90), (0, CY - 55, 1130), "black_anodized", anchor="b", bev=3)
A.cyl(n("scope_flange"), 58, 10, (0, CY, 1090), "silver", anchor="b", seg=64, bev=1.5)
A.cyl(n("scope_barrel"), 38, 28, (0, CY, 1062), "steel_dark", anchor="b", seg=64, bev=1.5)
A.cyl(n("scope_barrel_ring"), 42, 6, (0, CY, 1062), "chrome", anchor="b", seg=64, r_in=34, bev=1)
A.cyl(n("scope_lens"), 30, 1.5, (0, CY, 1060.5), "glass", anchor="b", seg=48)
# ring light (emissive) and light object
m_ring = A.custom_mat("ring_light", (1, 1, 1), emit=(1.0, 0.98, 0.9), emit_strength=6.0)
A.cyl(n("scope_ring_light"), 36, 4, (0, CY, 1057), "steel", anchor="b", seg=64, r_in=31, bev=0.5).data.materials.clear()
A.parts[n("scope_ring_light")].data.materials.append(m_ring)
# microscope cable (to the arch)
A.tube_path(n("scope_cable"), [(60, CY + 20, 1250), (110, CY + 30, 1340), (170, CY + 40, 1480), (250, CY + 50, 1500),
                               (330, CY + 55, 1440), (385, CY + 55, 1380)], 5, "rubber", seg=12)
A.cyl(n("scope_cable_gland"), 8, 22, (62, CY + 20, 1255), "silver", anchor="c", seg=16)
# light object
L = bpy.data.lights.new(n("light"), "AREA")
L.energy = 4.0
L.shape = "DISK"
L.size = 0.12
lo = bpy.data.objects.new(n("light"), L)
A.cur.objects.link(lo)
lo.parent = R
lo.location = (0, CY * MM, 1054 * MM)
lo.rotation_euler = (math.pi, 0, 0)

# ------------------------------------------------------------------ four positioners
pos_data = []
tips = []
for i, (phi_d, tipoff) in enumerate(((155, (-0.45, 0.45)), (205, (-0.45, -0.45)), (25, (0.45, 0.45)), (-25, (0.45, -0.45)))):
    phi = math.radians(phi_d)
    u = Vector((math.cos(phi), math.sin(phi), 0))
    t = Vector((-math.sin(phi), math.cos(phi), 0))

    def P(a, b, z, u=u, t=t):
        a = a - 60 if a >= 130 else a
        return (u.x * a + t.x * b, CY + u.y * a + t.y * b, z + 4)

    tag = "pos%d" % (i + 1)
    A.box(n(tag + "_base"), (130, 110, 8), P(280, 0, 1060), "silver", anchor="b", bev=1.5, rot_z=phi)
    A.box(n(tag + "_body"), (80, 64, 48), P(285, 0, 1068), "black_anodized", anchor="b", bev=4, rot_z=phi)
    A.box(n(tag + "_front_mount"), (22, 50, 38), P(238, 0, 1072), "silver", anchor="b", bev=2, rot_z=phi)
    # knobs
    A.tube_path(n(tag + "_knob_side"), [P(280, 32, 1098), P(280, 52, 1098)], 14, "silver", seg=24)
    A.tube_path(n(tag + "_knob_side_cap"), [P(280, 52, 1098), P(280, 56, 1098)], 8, "steel_dark", seg=16)
    A.tube_path(n(tag + "_knob_end"), [P(325, 0, 1092), P(325 + 24, 0, 1092)], 12, "silver", seg=24)
    A.tube_path(n(tag + "_knob_top"), [P(272, 0, 1116), P(272, 0, 1132)], 11, "silver", seg=24)
    # arm to the probe tip
    A.tube_path(n(tag + "_arm"), [P(236, 0, 1092), P(130, 0, 1066 + 8), P(20, 0, 1040)], 3.5, "steel", seg=12)
    A.box(n(tag + "_holder"), (16, 8, 8), P(14, 0, 1032), "silver", anchor="b", bev=1, rot_z=phi)
    tip = (tipoff[0], CY + tipoff[1], ZC + WAFER_T)
    tips.append(tip)
    A.tube_path(n(tag + "_needle"), [P(14, 0, 1034), (tip[0], tip[1], 1026),
                                     (tip[0], tip[1], tip[2] + 1.0), tip], [0.7, 0.45, 0.3, 0.05], "tungsten", seg=8)
    # cables with strain relief
    A.tube_path(n(tag + "_cable"), [P(330, 0, 1100), P(360, 0, 1140), P(378, 14, 1195), P(356, 45, 1220), P(300, 72, 1190),
                                   P(262, 88, 1120), P(252, 90, 1072)], 3.0, "rubber", seg=10)
    A.tube_path(n(tag + "_cable_relief"), [P(325 + 24, 0, 1092), P(325 + 44, 0, 1098)], 5.5, "silver", seg=14)
    A.tube_path(n(tag + "_cable_conn"), [P(252, 90, 1072), P(252, 90, 1064)], 7.5, "silver", seg=14)

# probe centre hook (fixed): top of wafer at the deck-opening centre
H_PC = A.hook("probe_center", (0, CY, ZC + WAFER_T), R, size=0.06)

# ------------------------------------------------------------------ monitor, arm, keyboard shelf, light tower
MX, MY, MZ = 300, 430, 1385
A.tube_path(n("monitor_arm"), [(390, 470, 1062), (390, 470, 1250), (330, 455, 1330), (300, MY + 20, MZ)], 16, "black_anodized", seg=24)
A.box(n("monitor_body"), (500, 30, 300), (MX, MY, MZ), "black_plastic", bev=6, bev_seg=3)
A.box(n("monitor_bezel_back"), (210, 14, 160), (MX, MY + 20, MZ), "black_anodized", bev=4)
# screen plane with UV (faces -Y)
scr_mat = bpy.data.materials.new("MAT_fab_test_screen")
scr_mat.use_nodes = True
nt = scr_mat.node_tree
for nd in list(nt.nodes):
    nt.nodes.remove(nd)
out = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Strength"].default_value = 1.0
mix = nt.nodes.new("ShaderNodeMix")
mix.data_type = "RGBA"
mix.label = "MIX_use_image"
fac = nt.nodes.new("ShaderNodeValue")
fac.label = "MIX_use_image (0 = default UI, 1 = image)"
fac.name = "MIX_use_image"
fac.outputs[0].default_value = 0.0
img = nt.nodes.new("ShaderNodeTexImage")
img.name = "IMG_screen"
img.label = "IMG_screen (assign the image sequence here)"
brick = nt.nodes.new("ShaderNodeTexBrick")
brick.inputs["Color1"].default_value = (0.02, 0.05, 0.12, 1)
brick.inputs["Color2"].default_value = (0.03, 0.08, 0.20, 1)
brick.inputs["Mortar"].default_value = (0.2, 0.4, 0.6, 1)
brick.inputs["Scale"].default_value = 6.0
brick.inputs["Mortar Size"].default_value = 0.03
tc = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(tc.outputs["UV"], brick.inputs["Vector"])
nt.links.new(tc.outputs["UV"], img.inputs["Vector"])
nt.links.new(brick.outputs["Color"], mix.inputs[6])
nt.links.new(img.outputs["Color"], mix.inputs[7])
nt.links.new(fac.outputs[0], mix.inputs[0])
nt.links.new(mix.outputs[2], em.inputs["Color"])
nt.links.new(em.outputs[0], out.inputs["Surface"])
sw, sh = 468, 270
sy_ = MY - 15.5
scr_o = A.obj_from_mesh(n("screen"), [((MX + sw / 2) * MM, sy_ * MM, (MZ - sh / 2) * MM), ((MX - sw / 2) * MM, sy_ * MM, (MZ - sh / 2) * MM),
                                      ((MX - sw / 2) * MM, sy_ * MM, (MZ + sh / 2) * MM), ((MX + sw / 2) * MM, sy_ * MM, (MZ + sh / 2) * MM)],
                        [(0, 1, 2, 3)], scr_mat, smooth=False)
uv = scr_o.data.uv_layers.new(name="UVMap")
for li, uvc in zip(scr_o.data.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
    uv.data[li].uv = uvc
A.mats["MAT_fab_test_screen"] = scr_mat
# keyboard shelf
A.box(n("shelf"), (330, 240, 10), (700, -260, 1075), "silver", anchor="b", bev=2)
A.box(n("shelf_arm"), (60, 180, 14), (560, -260, 1062), "black_anodized", anchor="b", bev=2)
A.box(n("keyboard"), (290, 110, 14), (700, -280, 1085), "black_plastic", anchor="b", bev=3)
key = A.box(n("key_proto"), (17, 15, 5), (700, -280, 1099), "steel_dark", anchor="b", bev=1, bev_seg=1)
key.hide_render = key.hide_viewport = True
for r in range(4):
    for c in range(14):
        A.dup(key, n("key"), (700 - 130 + c * 19.6, -280 - 36 + r * 21, 1099), parent=R)
A.cyl(n("mouse_pad"), 40, 2, (700, -190, 1085), "rubber", anchor="b", seg=24)
# light tower
A.cyl(n("tower_pole"), 8, 1050, (-470, 460, 1096), "steel", anchor="b", seg=16)
A.cyl(n("tower_base"), 22, 14, (-470, 460, 1060), "black_anodized", anchor="b", seg=24)
for k, mk in enumerate(("red_led", "yellow", "green_led")):
    A.cyl(n("tower_light_%d" % k), 24, 34, (-470, 460, 2025 + k * 36), mk, anchor="b", seg=32, bev=1.5)
A.cyl(n("tower_cap"), 22, 8, (-470, 460, 2133), "black_anodized", anchor="b", seg=24)

# ------------------------------------------------------------------ hooks documented
A.preview_fit = 0.62
A.preview_target_off = (0.15, 0, 0.0)
meta = {
    "description": "CM300xi-ULN style 300 mm semi-automated probe station at real size. Generic labels only (no vendor marks).",
    "scene_usage": "S5 wafer-level test: wafers fly into the open drawer (p_drawer = 1), the drawer closes (p_drawer 1 -> 0), "
                   "the stage steps wafer dies under the fixed probe point via p_stage_x / p_stage_y, then reopens.",
    "origin": "centre of cabinet footprint on the floor (z = 0); front is -Y",
    "mounting": "Parent a wafer_300mm root (origin bottom centre) to HOOK_wafer_slot with identity transform.",
    "hook_notes": {
        "HOOK_drawer": "slides along -Y by 500 mm and drops 60 mm as p_drawer goes 0.1..1 (driver on its location)",
        "HOOK_stage_x": "child of HOOK_drawer; x location = p_stage_x mm (driver)",
        "HOOK_stage_y": "child of HOOK_stage_x; y location = p_stage_y mm (driver)",
        "HOOK_stage_theta": "child of HOOK_stage_y; z rotation = p_stage_theta deg (driver)",
        "HOOK_wafer_slot": "child of HOOK_stage_theta; top of the chuck, wafer bottom sits here (wafer origin is bottom centre)",
        "HOOK_probe_center": "fixed (child of root): top surface of a 775 um wafer on the chuck at the probe point; the four needle tips land within about 1 mm",
        "HOOK_drawer_door": "front door; z location driven by p_drawer (opens over p_drawer 0..0.1)",
    },
    "driver_note": "Hooks have drivers on p_* of the root empty. Animate the p_* properties (preferred); to keyframe a hook "
                   "directly, remove or mute the driver on that channel first.",
    "simplifications": [
        "Interior of the cabinet is empty beyond the stage chain (hollow shell, dark).",
        "Positioner internals, probe-card holder and platen details simplified; needle tips are 4 converging tungsten needles within about 1 mm.",
        "Only the semi-automated station without the MHU loader is modelled; no equipment rack, thermal unit or cables to instruments.",
        "Monitor, keyboard and light tower placed per the datasheet envelope; arm geometry is generic.",
    ],
    "intended_screen": "probe_station_cm300_style_screen (MAT_fab_test_screen: IMG_screen node left empty; set MIX_use_image to 1 after plugging an image)",
}
A.finish(OUT, meta, preview=False)


def setup_open():
    """Preview-only state: drawer open with a SiPh wafer on the chuck (wafer appended from its own blend)."""
    if getattr(setup_open, "done", False):
        return
    setup_open.done = True
    wb = os.path.join(OUT, "wafer_300mm_siph.blend")
    if os.path.exists(wb):
        with bpy.data.libraries.load(wb, link=False) as (src, dst):
            dst.collections = ["ASSET_wafer_300mm_siph"]
        wc = dst.collections[0]
        A.coll.children.link(wc)
        for o in wc.objects:
            if o.name == "ROOT_wafer_300mm_siph":
                o.parent = H_W
                o.location = (0, 0, 0)
    R["p_drawer"] = 1.0


def setup_closed_stage():
    R["p_drawer"] = 0.0
    R["p_stage_x"] = 12.0
    R["p_stage_y"] = -9.0


A.preview_fit = 0.9
A.preview_target_off = (0.0, 0.0, -0.35)
ft.render_previews(A, os.path.join(OUT, "previews"), ("front", "three_quarter", "top"), [
    ("photo_view", (0, -2500, 1800), (0, -200, 1010), 70, setup_open),
    ("positioners", (0, -700, 1350), (0, CY, 1060), 50, setup_closed_stage),
    ("needles", (0, CY - 130, 1085), (0, CY, 1030), 85, None, 1e-3),
])
print("PROBE_STATION_BUILT")
