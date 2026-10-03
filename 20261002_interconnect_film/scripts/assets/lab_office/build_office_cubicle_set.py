"""Office cubicle set: two cubicles (fabric partitions, L-desks, monitors, rigged chairs, pedestals, filing cabinets) plus a common
area (water cooler, plant, printer on a cabinet), wall clock, exit sign and a back wall. Optional extras: phone, mouse, keyboard, bins.
Run: Blender -b --python build_office_cubicle_set.py -- <out_dir>
Origin: floor level, centre of the shared middle partition (x = 0, y = 0). -Y = open front of the cubicles, +Y = back wall.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import lo_common as L
import lo_furniture as F
from lo_common import C, M, S

OUT = C.argv_after_dashes()[0]
AID = "office_cubicle_set"
L.begin(AID, "B")
root = S.root
S._stack = getattr(S, "_stack", [])
PI = math.pi

CW, CD, PH = 1830.0, 1830.0, 1500.0   # 6 ft x 6 ft cubicle, 1.5 m panels (estimate / common office module)
BACK_Y = CD / 2 + 0.0


def group(name, loc, rot=(0, 0, 0), hide=False):
    return L.push_group(name, loc, rot)


# ---------------------------------------------------------------- floor and back wall
L.push_group("floor_and_wall", (0, 0, 0))
L.box("floor_carpet_tile", (7800.0, 4600.0, 12.0), (900.0, 300.0, -6.0), M("carpet"), r=1.0)
L.box("back_wall", (7800.0, 100.0, 2800.0), (900.0, BACK_Y + 650.0 + 50.0, 1400.0), M("wall_paint"))
L.box("skirting", (7800.0, 14.0, 90.0), (900.0, BACK_Y + 650.0 - 7.0, 45.0), M("case_light"), r=1.5)
L.pop_group()

# ---------------------------------------------------------------- partitions
L.push_group("partitions", (0, 0, 0))
fab = M("cubicle_fabric")
rail = M("alu_dark")


def panel(name, length, loc, along="x"):
    """Fabric partition panel: alu frame, tackable fabric inside, top cap; along = x or y."""
    sx, sy = (length, 62.0) if along == "x" else (62.0, length)
    L.box(name + "_fabric", (sx - 4.0 if along == "x" else 50.0, 50.0 if along == "x" else sy - 4.0, PH - 90.0), (loc[0], loc[1], 50.0 + (PH - 90.0) / 2), fab, r=3.0)
    L.box(name + "_top_cap", (sx, sy, 36.0), (loc[0], loc[1], PH - 18.0), rail, r=4.0)
    L.box(name + "_base", (sx, sy, 50.0), (loc[0], loc[1], 25.0), rail, r=3.0)


for k, x in enumerate((-CW, 0.0, CW)):
    panel("panel_side_%d" % (k + 1), CD, (x, 0.0), along="y")
    L.box("post_front_%d" % (k + 1), (70.0, 70.0, PH), (x, -CD / 2 + 35.0, PH / 2), rail, r=5.0)
    L.box("post_back_%d" % (k + 1), (70.0, 70.0, PH), (x, CD / 2 - 35.0, PH / 2), rail, r=5.0)
for k, x in enumerate((-CW / 2, CW / 2)):
    panel("panel_back_%d" % (k + 1), CW - 70.0, (x, CD / 2 - 31.0), along="x")
# name plate + pinboard items on a back panel
L.box("name_plate_1", (200.0, 6.0, 60.0), (-CW / 2, CD / 2 - 62.0, 1300.0), M("paper"), r=0.5)
L.box("pinned_note_1", (100.0, 3.0, 100.0), (-CW / 2 - 300.0, CD / 2 - 58.0, 1100.0), M("key_yellow"), r=0.4)
L.box("pinned_note_2", (110.0, 3.0, 80.0), (CW / 2 + 250.0, CD / 2 - 58.0, 1150.0), M("plastic_white"), r=0.4)
L.pop_group()


# ---------------------------------------------------------------- one workstation
def workstation(idx, cx):
    L.push_group("workstation_%d" % idx, (cx, 0, 0))
    DZ = 730.0
    top = M("wood_light")
    # L-desk: main run along the back panel, return along the left panel
    L.box("desk_main", (CW - 140.0, 750.0, 30.0), (0, CD / 2 - 70.0 - 375.0, DZ - 15.0), top, r=3.0)
    L.box("desk_return", (700.0, 1100.0, 30.0), (-CW / 2 + 70.0 + 350.0 + (CW - 140.0) / 2 - (CW - 140.0) / 2, CD / 2 - 70.0 - 750.0 - 550.0 + 100.0, DZ - 15.0), top, r=3.0)
    for sx in (-1, 1):
        L.box("desk_panel_leg_%s" % ("l" if sx < 0 else "r"), (25.0, 700.0, DZ - 30.0), (sx * (CW / 2 - 100.0), CD / 2 - 70.0 - 375.0, (DZ - 30.0) / 2), top, r=2.0)
    L.box("desk_modesty_panel", (CW - 300.0, 20.0, 380.0), (0, CD / 2 - 90.0, DZ - 30.0 - 190.0), top, r=2.0)
    # pedestal (3 drawers) under the main desk, right
    px = CW / 2 - 100.0 - 220.0
    L.box("pedestal", (400.0, 500.0, 600.0), (px, CD / 2 - 70.0 - 300.0, 300.0), M("frame_grey"), r=2.0)
    for i in range(3):
        L.box("pedestal_drawer_%d" % (i + 1), (388.0, 16.0, 175.0), (px, CD / 2 - 70.0 - 560.0, 100.0 + i * 190.0), M("frame_grey"), r=2.0)
        L.box("pedestal_handle_%d" % (i + 1), (140.0, 10.0, 10.0), (px, CD / 2 - 70.0 - 575.0, 140.0 + i * 190.0), M("alu"), r=3.0)
    # monitor 24 in on a stand
    mx, my = 120.0, CD / 2 - 70.0 - 300.0
    L.cyl("monitor_stand_base", 100.0, 8.0, (mx, my, DZ + 4.0), M("abs_black"), seg=32, bev=1.5)
    L.box("monitor_stand_neck", (50.0, 20.0, 260.0), (mx, my + 20.0, DZ + 130.0), M("abs_black"), r=4.0)
    L.box("monitor_body", (545.0, 18.0, 322.0), (mx, my - 10.0, DZ + 300.0), M("abs_black"), r=4.0)
    L.box("monitor_glass", (531.0, 1.0, 299.0), (mx, my - 19.5, DZ + 300.0), M("glass_dark"))
    # keyboard, mouse, phone, mug
    kx, ky = mx - 20.0, my - 250.0
    L.box("keyboard_base", (440.0, 140.0, 18.0), (kx, ky, DZ + 9.0), M("case_light"), r=4.0)
    L.multi_box("keyboard_keys", [((26.0, 26.0, 4.0), (kx - 195.0 + c * 28.0, ky - 45.0 + r * 28.0, DZ + 18.0 + 2.0)) for r in range(4) for c in range(14)], M("abs_dark"), r=1.0)
    L.box("mouse", (62.0, 108.0, 32.0), (kx + 300.0, ky - 10.0, DZ + 16.0), M("case_light"), r=14.0, seg=3)
    L.box("phone_base", (200.0, 220.0, 55.0), (-CW / 2 + 70.0 + 120.0 + 450.0, my - 20.0, DZ + 27.5), M("abs_dark"), r=8.0, seg=3, rot=(0, 0, 0))
    L.box("phone_handset", (60.0, 200.0, 40.0), (-CW / 2 + 70.0 + 120.0 + 450.0, my - 20.0, DZ + 70.0), M("abs_black"), r=14.0, seg=3)
    # paper tray + notebook
    L.box("paper_tray", (260.0, 330.0, 60.0), (-CW / 2 + 70.0 + 200.0, my + 20.0, DZ + 30.0), M("alu_dark"), r=3.0)
    L.box("paper_stack", (230.0, 300.0, 40.0), (-CW / 2 + 70.0 + 200.0, my + 20.0, DZ + 20.0 + 10.0), M("paper"), r=1.0)
    # chair at the desk (faces +Y to the desk), tucked in
    F.make_chair("chair", (0.0, 0.0 - 100.0 + 0.0, 0.0), 180.0 + (12.0 if idx == 1 else -15.0), fabric="fabric_green" if idx == 2 else "fabric_blue", swivel_deg=0.0)
    L.hook("desk_surface_center", (0, my - 150.0, DZ), note="centre of the desk work area")
    L.hook("monitor_screen_center", (mx, my - 20.0, DZ + 300.0), note="monitor glass centre")
    L.pop_group()


workstation(1, -CW / 2)
workstation(2, CW / 2)

# ---------------------------------------------------------------- filing cabinets (lateral, 4 drawer) beside the cubicles
for k, (x, y) in enumerate(((-CW - 600.0, CD / 2 - 300.0), (CW + 600.0, CD / 2 - 300.0))):
    L.push_group("filing_cabinet_%d" % (k + 1), (x, y, 0))
    L.box("body", (900.0, 450.0, 1300.0), (0, 0, 650.0), M("case_grey"), r=3.0)
    for i in range(4):
        zc = 160.0 + i * 310.0
        L.box("drawer_%d_front" % (i + 1), (880.0, 16.0, 290.0), (0, -233.0, zc), M("case_grey"), r=2.0)
        L.box("drawer_%d_handle" % (i + 1), (200.0, 12.0, 14.0), (0, -248.0, zc + 80.0), M("alu"), r=4.0)
        L.box("drawer_%d_label" % (i + 1), (110.0, 1.0, 40.0), (0, -241.5, zc + 20.0), M("paper"), r=0.3, seg=1)
    L.box("lock_bar", (14.0, 10.0, 1200.0), (410.0, -240.0, 650.0), M("alu_dark"), r=3.0)
    L.pop_group()

# ---------------------------------------------------------------- water cooler
L.push_group("water_cooler", (3200.0, 150.0, 0))
L.box("body", (320.0, 340.0, 950.0), (0, 0, 475.0), M("plastic_white"), r=30.0, seg=3)
L.box("drip_tray", (230.0, 110.0, 12.0), (0, -170.0, 360.0), M("alu_dark"), r=3.0)
L.box("tap_panel", (230.0, 20.0, 180.0), (0, -171.0, 600.0), M("abs_dark"), r=8.0)
L.cyl("tap_blue", 18.0, 40.0, (-50.0, -195.0, 570.0), M("key_blue"), axis="-y", seg=16, bev=2.0)
L.cyl("tap_red", 18.0, 40.0, (50.0, -195.0, 570.0), M("key_red"), axis="-y", seg=16, bev=2.0)
L.box("cup_dispenser", (110.0, 90.0, 220.0), (140.0, -190.0, 700.0), M("plastic_white"), r=6.0)
L.lathe("bottle", [(0, 0), (130.0, 0), (136.0, 10), (138.0, 300), (125.0, 360), (60.0, 400), (48.0, 420), (48.0, 460), (0, 460)], (0, 0, 950.0), M("water_blue"), seg=40)
L.cyl("bottle_cap", 50.0, 26.0, (0, 0, 1420.0 - 0.0), M("plastic_blue"), seg=24, bev=2.0)
L.hook("water_cooler_tap", (-50.0, -215.0, 570.0), note="blue tap spout")
L.pop_group()

# ---------------------------------------------------------------- plant (potted, generic leafy plant)
L.push_group("plant", (2750.0, -450.0, 0))
L.lathe("pot", [(0, 0), (140.0, 0), (180.0, 330.0), (190.0, 350.0), (172.0, 350.0), (172.0, 330.0), (0, 320.0)], (0, 0, 0), M("terracotta"), seg=32)
L.cyl("soil", 170.0, 6.0, (0, 0, 322.0), M("soil"), seg=32)
L.tube("trunk", [(0, 0, 320.0), (10, 5, 600.0), (-10, 0, 900.0)], 18.0, (0, 0, 0), M("wood_light"), seg=10, r_end=8.0)
leaf = C.sphere(AID + "_leaf_template", 1.0, (0, 0, 0.002), scale=(60.0 * 0.001, 24.0 * 0.001, 6.0 * 0.001), seg=12, rings=6, mat=M("plant_green"))
C.add(leaf, S.coll, S.root)
import random
rnd = random.Random(7)
for i in range(60):
    a = rnd.uniform(0, 2 * PI)
    r_ = rnd.uniform(40.0, 330.0)
    z = rnd.uniform(520.0, 1250.0)
    spread = 120.0 + (z - 520.0) * 0.18
    x, y = math.cos(a) * min(r_, spread), math.sin(a) * min(r_, spread)
    o = bpy.data.objects.new(AID + "_leaf_%02d" % i, leaf.data)
    o.location = (x * 0.001, y * 0.001, z * 0.001)
    o.rotation_euler = (rnd.uniform(-0.6, 0.6), rnd.uniform(-0.5, 0.5), a + rnd.uniform(-0.4, 0.4))
    o.scale = (rnd.uniform(1.2, 2.2), rnd.uniform(1.2, 2.0), 1.0)
    S.coll.objects.link(o)
    o.parent = S.root
L.pop_group()

# ---------------------------------------------------------------- printer on a cabinet
L.push_group("printer", (3950.0, 250.0, 0))
L.box("stand", (700.0, 520.0, 700.0), (0, 0, 350.0), M("frame_grey"), r=3.0)
L.box("stand_door_l", (340.0, 14.0, 640.0), (-175.0, -265.0, 350.0), M("case_grey"), r=2.0)
L.box("stand_door_r", (340.0, 14.0, 640.0), (175.0, -265.0, 350.0), M("case_grey"), r=2.0)
L.box("body", (500.0, 440.0, 380.0), (0, 0, 700.0 + 190.0), M("case_light"), r=12.0, seg=3)
L.box("lower_body_band", (504.0, 444.0, 120.0), (0, 0, 700.0 + 60.0), M("abs_dark"), r=8.0)
L.box("paper_tray_front", (440.0, 20.0, 70.0), (0, -225.0, 700.0 + 60.0), M("case_grey"), r=3.0)
L.box("output_tray", (440.0, 240.0, 8.0), (0, -160.0, 700.0 + 300.0), M("case_grey"), r=2.0, rot=(8, 0, 0))
L.box("control_panel", (230.0, 90.0, 12.0), (80.0, -190.0, 700.0 + 342.0), M("abs_black"), r=4.0, rot=(-22, 0, 0))
L.box("scanner_lid", (500.0, 440.0, 30.0), (0, 0, 700.0 + 395.0), M("abs_dark"), r=6.0)
for i in range(6):
    L.box("panel_button_%d" % (i + 1), (22.0, 16.0, 6.0), (10.0 + i * 28.0, -188.0, 700.0 + 352.0), M("key_grey"), r=1.5, rot=(-22, 0, 0))
L.hook("printer_output", (0, -200.0, 700.0 + 304.0), note="paper output tray")
L.pop_group()

# ---------------------------------------------------------------- wall clock and exit sign
L.push_group("wall_clock", (900.0, BACK_Y + 650.0 - 5.0, 2250.0))
L.lathe("rim", [(160.0, 0), (166.0, 8), (166.0, 34), (160.0, 34), (150.0, 30), (150.0, 0)], (0, 0, 0), M("abs_black"), axis="-y", seg=64)
L.cyl("face", 150.0, 3.0, (0, -6.0, 0), M("plastic_white"), axis="y", seg=64)
L.multi_box("hour_marks", [((6.0 if i % 3 else 10.0, 2.0, 22.0 if i % 3 else 30.0), (0, 0, 0)) for i in range(0)], M("abs_black"))
for i in range(12):
    a = math.radians(i * 30.0)
    L.box("mark_%02d" % i, (8.0 if i % 3 else 12.0, 2.0, 24.0 if i % 3 else 34.0), (125.0 * math.sin(a), -9.0, 125.0 * math.cos(a)), M("abs_black"), r=0.5, rot=(0, -math.degrees(a), 0))
hh = L.box("hand_hour", (9.0, 3.0, 70.0), (math.sin(math.radians(300.0)) * 35.0, -12.0, math.cos(math.radians(300.0)) * 35.0), M("abs_black"), r=1.0, rot=(0, -300.0, 0))
mh = L.box("hand_minute", (6.0, 3.0, 105.0), (math.sin(math.radians(60.0)) * 52.0, -15.0, math.cos(math.radians(60.0)) * 52.0), M("abs_black"), r=1.0, rot=(0, -60.0, 0))
L.box("hand_second", (2.5, 2.0, 120.0), (0, -17.0, 0.0), M("key_red"), r=0.4, rot=(0, -180.0, 0))
L.cyl("hand_pivot", 7.0, 5.0, (0, -17.0, 0), M("key_red"), axis="y", seg=16)
L.hook("clock_center", (0, -20.0, 0), note="clock face centre (hands are static meshes at 10:10; replace/animate as needed)")
L.pop_group()

L.push_group("exit_sign", (3950.0, BACK_Y + 650.0 - 5.0, 2450.0))
L.box("housing", (320.0, 50.0, 140.0), (0, -25.0, 0), M("plastic_white"), r=6.0, seg=2)
L.box("legend_panel", (290.0, 2.0, 110.0), (0, -50.5, 0), M("exit_green"), r=2.0)
L.labels("legend", [("EXIT", 70.0, (0, -52.0, 0))], M("ink_white"))
L.pop_group()

# ---------------------------------------------------------------- extras: waste bins and a coat stand? (small)
for k, x in enumerate((-CW / 2 + 300.0, CW / 2 + 300.0)):
    L.push_group("waste_bin_%d" % (k + 1), (x, 250.0, 0))
    L.lathe("bin", [(0, 0), (120.0, 0), (140.0, 300.0), (136.0, 300.0), (116.0, 8.0), (0, 8.0)], (0, 0, 0), M("abs_black"), seg=32)
    L.pop_group()

# ---------------------------------------------------------------- dimensions
L.dim("cubicle module", "1830 x 1830 (6 ft x 6 ft)", "mm", "common office cubicle module (estimate; not a primary-source dimension)", "C")
L.dim("panel height", PH, "mm", "estimate; low/medium cubicle panels 1.2-1.65 m", "C")
L.dim("desk height", 730.0, "mm", "standard 29 in working height (about 720-740 mm)", "B")
L.dim("desk main run", "1690 x 750", "mm", "estimate", "C")
L.dim("filing cabinet", "900 x 450 x 1300", "mm", "estimate for a lateral 4-drawer cabinet", "C")
L.dim("water cooler incl. bottle", "320 x 340 x 1470", "mm", "estimate; 5 gal bottle about 280 mm dia x 500 mm", "C")
L.dim("printer", "500 x 440 x 400 on a 700 mm cabinet", "mm", "estimate", "C")
L.dim("wall clock", "332 dia", "mm", "estimate", "C")
L.dim("monitor", "24 in class, 531 x 299 active", "mm", "computed from 24 in 16:9", "A")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="generic office dimensions (cubicle 6 ft module, desk height 29 in)", url="https://kavela.furniture/en/office-furniture-dimensions-guide/", access_date="2026-10-02")],
    simplifications=["partitions have no cable raceway detail", "plant leaves are linked-duplicate ellipsoids", "clock hands are static at 10:10", "monitors are dark panels (no image slot)"],
    custom_properties={}, rigs=[dict(name="chair rigs", note="workstation_N/chair: armature with bones base, swivel (pose Y rotation), back_tilt (pose X rotation)")],
    usage="Optional office background (cubicles) for scenes that need a workplace; the exit sign and clock are on the back wall."))

hide_none = []
views = [("front", dict(loc=(0.9, -4.3, 1.5), target=(0.9, 0.5, 1.0), lens=24)),
         ("three_quarter", dict(loc=(4.6, -3.6, 2.6), target=(0.9, 0.3, 0.8), lens=26)),
         ("top", dict(loc=(0.9, 0.0, 8.0), target=(0.9, 0.0, 0), lens=24)),
         ("closeup_workstation", dict(loc=(-0.6, -1.3, 1.5), target=(-0.5, 0.8, 0.9), lens=30)),
         ("closeup_common", dict(loc=(3.0, -2.4, 1.5), target=(3.4, 0.3, 0.9), lens=28))]
for (x, y) in [(-1.0, -0.6), (3.0, -0.8), (0.9, 1.6)]:
    ld = bpy.data.lights.new("PREVIEW_room_light", "AREA")
    ld.energy = 900.0
    ld.size = 1.5
    o = bpy.data.objects.new("PREVIEW_room_light", ld)
    o.location = (x, y, 2.6)
    bpy.context.scene.collection.objects.link(o)
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=False, lights="none", world=0.9)
