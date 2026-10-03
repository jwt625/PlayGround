"""Conference room: table (storyboard 5.6 x 2.2 m and realistic 2.4 x 1.2 m variants), rigged office chairs, wall TV on a mount, door,
glass partition, carpet, ceiling with light panels, window blinds, wall whiteboard. Seats: 4 on the far side (+Y), 1 at the head (-X end).
Run: Blender -b --python build_conference_room.py -- <out_dir>
Origin: centre of the floor (z = 0 is the carpet surface). -Y = camera side (front wall is a hidden variant), +Y = window wall.
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
AID = "conference_room"
L.begin(AID, "B")
root = S.root
S._stack = getattr(S, "_stack", [])

RX, RY, RZ = 8400.0, 6000.0, 2800.0     # interior size
WT = 150.0                              # wall thickness
TABLE_TOP_Z = 760.0
root["p_table_variant"] = "storyboard"
root["p_room_size_mm"] = (RX, RY, RZ)


def enter_collection(name, hide=False):
    c = bpy.data.collections.new(name)
    S.coll.children.link(c)
    S._stack.append((S.coll, S.root))
    S.coll = c
    if hide:
        c.hide_render = True   # (hide_viewport left on so matrices evaluate; unhide_render to use)
    return c


def leave():
    S.coll, S.root = S._stack.pop()


wall_m = M("wall_paint")
acc_m = M("wall_accent")

# ---------------------------------------------------------------- floor and skirting
enter_collection("ITEM_floor")
L.box("carpet", (RX, RY, 14.0), (0, 0, -7.0), M("carpet"), r=1.0)
for nm, size, loc in (("skirting_back", (RX, 12.0, 90.0), (0, RY / 2 - 6.0, 45.0)), ("skirting_left", (12.0, RY, 90.0), (-RX / 2 + 6.0, 0, 45.0)),
                      ("skirting_right", (12.0, RY, 90.0), (RX / 2 - 6.0, 0, 45.0))):
    L.box(nm, size, loc, M("case_light"), r=1.5)
leave()

# ---------------------------------------------------------------- walls
enter_collection("ITEM_walls")
# back wall (+Y) with three window openings: sill 900, head 2300, 1800 wide, pillars between
WIN_W, SILL, HEAD = 1800.0, 900.0, 2300.0
pil = (RX - 3 * WIN_W) / 4.0
yb = RY / 2 + WT / 2
L.box("wall_back_sill", (RX + 2 * WT, WT, SILL), (0, yb, SILL / 2), wall_m)
L.box("wall_back_head", (RX + 2 * WT, WT, RZ - HEAD), (0, yb, HEAD + (RZ - HEAD) / 2), wall_m)
wx = []
x = -RX / 2
for k in range(4):
    L.box("wall_back_pillar_%d" % k, (pil + (WT if k in (0, 3) else 0.0), WT, HEAD - SILL), (x + pil / 2 + (-WT / 2 if k == 0 else (WT / 2 if k == 3 else 0.0)), yb, (SILL + HEAD) / 2), wall_m)
    x += pil
    if k < 3:
        wx.append(x + WIN_W / 2)
        x += WIN_W
# right wall (+X): plain with accent
xr = RX / 2 + WT / 2
L.box("wall_right", (WT, RY + 2 * WT, RZ), (xr, 0, RZ / 2), wall_m)
L.box("wall_right_accent", (8.0, 3000.0, RZ - 200.0), (RX / 2 - 4.0, 0, RZ / 2), acc_m)
# left wall (-X): door opening at y=-1500 (900 wide), glass partition y in [-600, 1800]
xl = -RX / 2 - WT / 2
DOOR_Y, DOOR_W, DOOR_H = -1900.0, 900.0, 2100.0
GL_Y0, GL_Y1 = -900.0, 1500.0
ya = -RY / 2 - WT
L.box("wall_left_front", (WT, (DOOR_Y - DOOR_W / 2) - ya, RZ), (xl, (ya + DOOR_Y - DOOR_W / 2) / 2, RZ / 2), wall_m)
L.box("wall_left_door_head", (WT, DOOR_W, RZ - DOOR_H), (xl, DOOR_Y, DOOR_H + (RZ - DOOR_H) / 2), wall_m)
L.box("wall_left_between", (WT, (GL_Y0 - 50.0) - (DOOR_Y + DOOR_W / 2), RZ), (xl, ((GL_Y0 - 50.0) + (DOOR_Y + DOOR_W / 2)) / 2, RZ / 2), wall_m)
L.box("wall_left_back", (WT, (RY / 2 + WT) - (GL_Y1 + 50.0), RZ), (xl, ((RY / 2 + WT) + (GL_Y1 + 50.0)) / 2, RZ / 2), wall_m)
L.box("wall_left_glass_sill", (WT, GL_Y1 - GL_Y0, 100.0), (xl, (GL_Y0 + GL_Y1) / 2, 50.0), wall_m)
L.box("wall_left_glass_head", (WT, GL_Y1 - GL_Y0, RZ - 2400.0), (xl, (GL_Y0 + GL_Y1) / 2, 2400.0 + (RZ - 2400.0) / 2), wall_m)
# front wall (-Y): hidden variant so the camera can sit inside
enter_collection("VARIANT_front_wall", hide=True)
L.box("wall_front", (RX + 2 * WT, WT, RZ), (0, -RY / 2 - WT / 2, RZ / 2), wall_m)
leave()
leave()

# ---------------------------------------------------------------- windows: frames, glass, blinds
enter_collection("ITEM_windows")
for k, cx in enumerate(wx):
    wz0, wz1 = SILL, HEAD
    L.box("window_frame_top_%d" % k, (WIN_W, 80.0, 50.0), (cx, RY / 2 - 20.0, wz1 - 25.0), M("alu_dark"), r=3.0)
    L.box("window_frame_bottom_%d" % k, (WIN_W, 90.0, 50.0), (cx, RY / 2 - 20.0, wz0 + 25.0), M("alu_dark"), r=3.0)
    for sx in (-1, 1):
        L.box("window_frame_side_%d%s" % (k, "l" if sx < 0 else "r"), (50.0, 80.0, wz1 - wz0), (cx + sx * (WIN_W / 2 - 25.0), RY / 2 - 20.0, (wz0 + wz1) / 2), M("alu_dark"), r=3.0)
    L.box("window_mullion_%d" % k, (40.0, 70.0, wz1 - wz0 - 100.0), (cx, RY / 2 - 20.0, (wz0 + wz1) / 2), M("alu_dark"), r=3.0)
    L.box("window_sill_%d" % k, (WIN_W + 100.0, 180.0, 24.0), (cx, RY / 2 - 70.0, wz0 + 12.0), M("case_light"), r=3.0)
    L.box("window_glass_%d" % k, (WIN_W - 100.0, 6.0, wz1 - wz0 - 100.0), (cx, RY / 2 + 10.0, (wz0 + wz1) / 2), M("glass_clear"))
    # venetian blind: headrail, slats (one merged mesh), bottom rail, cord
    top = wz1 - 60.0
    drop = 1000.0 if k != 1 else 1300.0
    L.box("blind_headrail_%d" % k, (WIN_W - 100.0, 50.0, 36.0), (cx, RY / 2 - 85.0, top + 18.0), M("alu"), r=3.0)
    n = int(drop // 28.0)
    L.multi_box("blind_slats_%d" % k, [((WIN_W - 130.0, 22.0, 2.2), (cx, RY / 2 - 85.0, top - i * 28.0)) for i in range(n)], M("case_light"), origin=(cx, RY / 2 - 85.0, top))
    L.box("blind_bottom_rail_%d" % k, (WIN_W - 130.0, 30.0, 10.0), (cx, RY / 2 - 85.0, top - n * 28.0 - 4.0), M("alu"), r=2.0)
    for sx in (-1, 1):
        L.box("blind_ladder_%d%s" % (k, "l" if sx < 0 else "r"), (3.0, 3.0, n * 28.0), (cx + sx * (WIN_W / 2 - 200.0), RY / 2 - 70.0, top - n * 14.0), M("ptfe"))
    L.cyl("blind_wand_%d" % k, 6.0, 900.0, (cx + WIN_W / 2 - 80.0, RY / 2 - 95.0, top - 450.0), M("plastic_white"), seg=10)
leave()

# ---------------------------------------------------------------- left wall: door (origin at the hinge), frame, glass partition
enter_collection("ITEM_door_partition")
hx, hy = -RX / 2, DOOR_Y + DOOR_W / 2   # hinge at the +Y side of the opening, opens into the room (+X)
L.box("door_frame_top", (WT, DOOR_W + 80.0, 50.0), (-RX / 2 - WT / 2 + 15.0, DOOR_Y, DOOR_H + 25.0), M("wood_light"), r=2.0)
for sy in (-1, 1):
    L.box("door_frame_side_%s" % ("a" if sy < 0 else "b"), (WT - 20.0, 40.0, DOOR_H), (-RX / 2 - WT / 2 + 15.0, DOOR_Y + sy * (DOOR_W / 2 + 20.0), DOOR_H / 2), M("wood_light"), r=2.0)
g, door = L.push_group("door", (hx - 40.0 + 0.0, hy, 0.0), sub=False)
L.box("leaf", (40.0, DOOR_W - 6.0, DOOR_H - 20.0), (20.0, -(DOOR_W - 6.0) / 2, (DOOR_H - 20.0) / 2 + 10.0), M("wood_light"), r=2.0)
L.box("vision_panel", (42.0, 220.0, 900.0), (20.0, -(DOOR_W - 6.0) / 2, 1450.0), M("glass_tint"))
L.cyl("lever_rose", 28.0, 8.0, (50.0, -(DOOR_W - 60.0), 1000.0), M("alu"), axis="x", seg=24, bev=1.0)
L.box("lever", (10.0, 130.0, 20.0), (62.0, -(DOOR_W - 60.0) + 45.0 - 65.0 + 65.0 - 60.0, 1000.0), M("alu"), r=4.0)
for z in (250.0, 1050.0, 1850.0):
    L.cyl("hinge", 10.0, 120.0, (20.0, -5.0, z), M("alu_dark"), seg=12)
L.hook("door_handle", (62.0, -(DOOR_W - 120.0), 1000.0), parent=door, note="lever handle")
L.pop_group()
# glass partition (full height 2400, aluminium frame, mid rail)
gy = (GL_Y0 + GL_Y1) / 2
gl = GL_Y1 - GL_Y0
L.box("partition_glass", (12.0, gl - 100.0, 2300.0), (-RX / 2 + 20.0, gy, 100.0 + 1150.0), M("glass_clear"))
for k in range(5):
    yy = GL_Y0 + 25.0 + k * (gl - 50.0) / 4.0
    L.box("partition_mullion_%d" % k, (50.0, 50.0, 2400.0), (-RX / 2 + 20.0, yy, 1200.0), M("alu_dark"), r=3.0)
L.box("partition_rail_mid", (50.0, gl, 50.0), (-RX / 2 + 20.0, gy, 1100.0), M("alu_dark"), r=3.0)
L.box("partition_privacy_band", (14.0, gl - 100.0, 300.0), (-RX / 2 + 20.0, gy, 1500.0), C.principled("MAT_lab_office_glass_frosted", (0.9, 0.93, 0.95), rough=0.6, alpha=0.7), r=0.0)
S.slots.add("MAT_lab_office_glass_frosted")
leave()

# ---------------------------------------------------------------- ceiling: slab, grid, light panels
enter_collection("ITEM_ceiling")
L.box("ceiling_slab", (RX + 2 * WT, RY + 2 * WT, 40.0), (0, 0, RZ + 20.0), M("ceiling_tile"))
grid = []
for i in range(int(RX // 600.0) + 1):
    grid.append(((20.0, RY, 6.0), (-RX / 2 + i * 600.0, 0, RZ - 3.0)))
for j in range(int(RY // 600.0) + 1):
    grid.append(((RX, 20.0, 6.0), (0, -RY / 2 + j * 600.0, RZ - 3.0)))
L.multi_box("ceiling_grid", grid, M("case_light"))
LP = []
lpos = [(-2400.0, -1200.0), (0.0, -1200.0), (2400.0, -1200.0), (-2400.0, 1200.0), (0.0, 1200.0), (2400.0, 1200.0)]
for k, (lx_, ly_) in enumerate(lpos):
    L.box("light_panel_%d" % (k + 1), (590.0, 590.0, 10.0), (lx_ + 10.0 - 10.0, ly_, RZ - 6.0), M("light_panel"), r=1.0)
    L.hook("light_%d" % (k + 1), (lx_, ly_, RZ - 20.0), rot_deg=(0, 0, 0), note="ceiling light panel centre (add an area light facing -Z, about 600 x 600 mm)")
# HVAC diffuser and smoke detector
L.box("hvac_diffuser", (590.0, 590.0, 14.0), (-1200.0, 0.0, RZ - 8.0), M("case_light"), r=2.0)
L.multi_box("hvac_diffuser_slots", [((560.0, 6.0, 2.0), (-1200.0, -230.0 + i * 46.0, RZ - 17.0)) for i in range(11)], M("abs_black"), origin=(-1200.0, 0.0, RZ - 8.0))
L.cyl("smoke_detector", 55.0, 28.0, (1200.0, 0.0, RZ - 14.0), M("plastic_white"), seg=24, bev=3.0)
leave()

# ---------------------------------------------------------------- right wall: TV on a mount, whiteboard on the back wall
enter_collection("ITEM_wall_tv")
TVW, TVH = 1660.0, 934.0   # 75 in 16:9 (computed)
tvx = RX / 2
tvz = 1500.0
L.box("tv_wall_plate", (10.0, 420.0, 340.0), (tvx - 5.0, 0, tvz), M("abs_black"), r=2.0)
L.box("tv_mount_arm", (60.0, 100.0, 40.0), (tvx - 40.0, 0, tvz), M("abs_black"), r=3.0)
L.box("tv_body", (46.0, TVW, TVH), (tvx - 85.0, 0, tvz), M("abs_black"), r=4.0)
L.box("tv_glass_border", (2.0, TVW - 20.0, TVH - 20.0), (tvx - 108.5, 0, tvz), M("glass_dark"))
tvs = L.uvquad("tv_screen", TVW - 24.0, TVH - 24.0, (tvx - 110.0, 0, tvz), L.screen_material("MAT_lab_office_screen"), facing="-x")
tvs.name = AID + "_tv_screen"
tvs.visible_shadow = False
tvs["is_image_screen"] = True
L.hook("tv_screen_center", (tvx - 110.5, 0, tvz), rot_deg=(0, 0, 0), note="TV glass centre; screen faces -X; material MAT_lab_office_screen (empty SCREEN_IMAGE, SCREEN_FAC=0)")
# camera/soundbar below the TV
L.box("video_bar", (60.0, 800.0, 70.0), (tvx - 60.0, 0, tvz - TVH / 2 - 60.0), M("abs_dark"), r=6.0)
L.cyl("video_bar_lens", 14.0, 4.0, (tvx - 91.0, 0, tvz - TVH / 2 - 60.0), M("glass_dark"), axis="x", seg=16)
leave()

enter_collection("ITEM_wall_whiteboard")
wbx, wbz = 1900.0, 1550.0
WBW, WBH = 1800.0, 1200.0
L.box("wb_frame", (WBW, 20.0, WBH), (wbx, RY / 2 - 10.0, wbz), M("alu"), r=2.0)
L.box("wb_surface", (WBW - 40.0, 2.0, WBH - 40.0), (wbx, RY / 2 - 21.0, wbz), M("laminate_white"))
L.box("wb_tray", (900.0, 60.0, 12.0), (wbx, RY / 2 - 40.0, wbz - WBH / 2 - 4.0), M("alu"), r=2.0)
L.hook("wall_whiteboard_center", (wbx, RY / 2 - 24.0, wbz), rot_deg=(90, 0, 0), note="small wall whiteboard (use whiteboard_big for the film whiteboard)")
leave()

# ---------------------------------------------------------------- table variants
def sofa(): pass


def storyboard_table():
    enter_collection("VARIANT_table_storyboard")
    TW, TD, TT = 5600.0, 2200.0, 50.0
    L.box("table_top", (TW, TD, TT), (0, 0, TABLE_TOP_Z - TT / 2), M("wood_table"), r=5.0, seg=3)
    L.box("table_edge_inlay", (TW - 100.0, 30.0, 2.0), (0, -TD / 2 + 60.0, TABLE_TOP_Z + 0.5), M("alu_dark"))
    for sx in (-1, 1):
        L.box("table_leg_panel_%s" % ("l" if sx < 0 else "r"), (70.0, 1500.0, TABLE_TOP_Z - TT - 60.0), (sx * 2050.0, 0, 60.0 + (TABLE_TOP_Z - TT - 60.0) / 2), M("wood_table"), r=4.0)
        L.box("table_foot_%s" % ("l" if sx < 0 else "r"), (200.0, 1700.0, 60.0), (sx * 2050.0, 0, 30.0), M("alu_dark"), r=6.0)
    L.box("table_modesty_rail", (4100.0, 18.0, 300.0), (0, 600.0, TABLE_TOP_Z - TT - 160.0), M("wood_table"), r=2.0)
    # cable cubbies (flush pop-up boxes) and a conference phone
    for k, x in enumerate((-1600.0, 0.0, 1600.0)):
        L.box("cubby_%d" % (k + 1), (180.0, 130.0, 8.0), (x, 0.0, TABLE_TOP_Z + 4.0), M("alu_dark"), r=3.0)
        L.box("cubby_lid_%d" % (k + 1), (150.0, 100.0, 2.0), (x, 0.0, TABLE_TOP_Z + 9.0), M("abs_black"), r=1.0)
    ph = L.lathe("conference_phone", [(0, 0), (120.0, 0), (125.0, 12.0), (95.0, 22.0), (0, 24.0)], (700.0, 0.0, TABLE_TOP_Z), M("abs_black"), seg=32)
    for a in (90, 210, 330):
        L.box("conference_phone_arm", (170.0, 40.0, 20.0), (700.0 + 90.0 * math.cos(math.radians(a)), 90.0 * math.sin(math.radians(a)), TABLE_TOP_Z + 8.0), M("abs_black"), r=6.0, rot=(0, 0, a))
    L.hook("table_center", (0, 0, TABLE_TOP_Z), note="centre of the table top")
    L.hook("table_far_edge_center", (0, TD / 2, TABLE_TOP_Z), note="far (vendor) edge centre")
    L.hook("table_near_edge_center", (0, -TD / 2, TABLE_TOP_Z), note="near (camera side) edge centre")
    L.hook("table_head_edge_center", (-TW / 2, 0, TABLE_TOP_Z), note="head (-X) end centre")
    # chairs: far side 4, head 1, near side 4 (separate sub-collection)
    far_x = (-1950.0, -650.0, 650.0, 1950.0)
    ch = []
    for k, x in enumerate(far_x):
        e = F.make_chair("chair_far_%d" % (k + 1), (x, 1560.0, 0.0), 0.0, fabric="fabric_blue", swivel_deg=[8, -6, 10, -12][k])
        L.hook("seat_far_%d" % (k + 1), (x, 1560.0, 480.0), note="vendor seat %d (far side, facing -Y); chair root sits at floor level under it" % (k + 1))
    e = F.make_chair("chair_head", (-3300.0, 0.0, 0.0), 90.0, fabric="fabric_blue", swivel_deg=0.0)
    L.hook("seat_head", (-3300.0, 0.0, 480.0), note="head-of-table seat (-X end, facing +X)")
    enter_collection("ITEM_chairs_near")
    for k, x in enumerate((-1950.0, -650.0, 650.0, 1950.0)):
        F.make_chair("chair_near_%d" % (k + 1), (x, -1560.0, 0.0), 180.0, fabric="fabric_blue", swivel_deg=[-5, 9, -8, 6][k])
    leave()
    leave()


def realistic_table():
    enter_collection("VARIANT_table_realistic", hide=True)
    TW, TD, TT = 2400.0, 1200.0, 30.0
    L.box("table_top_small", (TW, TD, TT), (0, 0, 740.0 - TT / 2), M("wood_table"), r=4.0, seg=3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            L.box("table_small_leg_%s%s" % ("l" if sx < 0 else "r", "f" if sy < 0 else "b"), (60.0, 60.0, 740.0 - TT), (sx * (TW / 2 - 120.0), sy * (TD / 2 - 120.0), (740.0 - TT) / 2), M("alu_dark"), r=4.0)
    for sy in (-1, 1):
        L.box("table_small_apron_%s" % ("f" if sy < 0 else "b"), (TW - 300.0, 20.0, 90.0), (0, sy * (TD / 2 - 120.0), 740.0 - TT - 45.0), M("wood_table"), r=2.0)
    for k, (x, y, yaw) in enumerate([(-600.0, 960.0, 0.0), (600.0, 960.0, 0.0), (-600.0, -960.0, 180.0), (600.0, -960.0, 180.0), (-1700.0, 0.0, 90.0), (1700.0, 0.0, -90.0)]):
        F.make_chair("chair_real_%d" % (k + 1), (x, y, 0.0), yaw, fabric="fabric_grey")
    L.hook("table_small_center", (0, 0, 740.0), note="realistic variant table centre; 6 chairs around it (unhide VARIANT_table_realistic, hide VARIANT_table_storyboard, p_table_variant)")
    leave()


storyboard_table()
realistic_table()

# ---------------------------------------------------------------- hooks (room level)
L.hook("camera_front", (0, -2800.0, 1500.0), rot_deg=(90, 0, 0), note="suggested wide camera spot inside the room, looking +Y; unhide the front wall variant only when the camera is outside")
L.hook("room_center_floor", (0, 0, 0), note="floor centre / origin")
L.hook("door_hinge", (hx - 40.0, hy, 0.0), note="door hinge axis (door empty rotates about Z)")
L.hook("window_center", (wx[1], RY / 2, 1600.0), note="middle window")

# ---------------------------------------------------------------- dimensions
L.dim("room interior", "%.0f x %.0f x %.0f" % (RX, RY, RZ), "mm", "design choice around the 5.6 x 2.2 m storyboard table (clearances 1.4 m at the TV end)", "C")
L.dim("storyboard table top", "5600 x 2200 x 50, top at 760", "mm", "storyboard (DevLog-001 S3); height from standard 29-30 in (737-762 mm) conference tables (search 2026-10-02)", "B")
L.dim("realistic table", "2400 x 1200 x 30, top at 740", "mm", "standard 8 ft x 4 ft 8-person conference table (Fargo Woodworks / Arcgrove guides, 2026-10-02)", "B")
L.dim("chair seat height", "470 (+ about 35 seat thickness top at 500)", "mm", "standard 16-20 in office chair seat heights (search 2026-10-02)", "B")
L.dim("chair base radius", 300, "mm", "estimate; five-star base about 660 mm diameter is common", "C")
L.dim("door", "900 x 2100 x 40", "mm", "common interior door size (estimate)", "B")
L.dim("windows", "3 x 1800 x 1400 (sill 900)", "mm", "estimate", "C")
L.dim("TV", "75 in 16:9 = 1660 x 934", "mm", "computed from diagonal", "A")
L.dim("ceiling", 2800.0, "mm", "estimate; common office ceiling 2.7-3.0 m; 600 x 600 tile grid is a common metric standard", "B")
L.dim("wall whiteboard", "1800 x 1200", "mm", "common size", "B")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="conference table sizes, table and chair heights", url="https://fargowoodworks.com/the-complete-seating-capacity-guide-for-conference-tables/ ; https://www.arcgrove.com/blogs/posts/how-to-choose-the-right-size-conference-table", access_date="2026-10-02"),
             dict(what="storyboard table size 5.6 x 2.2 m", url="DevLog/DevLog-001-story-scenes-assets-proposal.md and scripts/build_crude_film.py (s3)", access_date="2026-10-02")],
    simplifications=["TV and wall whiteboard are generic; walls are boxes around the openings", "blind slats are static", "front wall is a hidden variant (VARIANT_front_wall, hide_render on)",
                     "no Blender lights stored: use HOOK_light_N positions", "chairs: mesh parts bone-parented to a 3-bone armature, no cloth/foam deformation"],
    custom_properties={"p_table_variant": "storyboard (default). Switch by toggling hide_render/hide_viewport of VARIANT_table_storyboard and VARIANT_table_realistic",
                       "p_room_size_mm": "8400 x 6000 x 2800 interior"},
    rigs=[dict(name="<chair>_rig armatures", bones=["base", "swivel (pose rotation_euler Y)", "back_tilt (pose rotation_euler X)"],
               note="every chair has its own armature under its chair_* empty; rotate the chair_* empty for yaw (heading)")],
    screens=[dict(object=AID + "_tv_screen", material="MAT_lab_office_screen", image_node="SCREEN_IMAGE", toggle_node="SCREEN_FAC", aspect="16:9", note="optional TV content, same recipe as the scopes")],
    usage="S3 conference table with four vendors on the far side (HOOK_seat_far_1..4), the head seat HOOK_seat_head, wall TV, door for the entrance."))

# ---------------------------------------------------------------- previews: lights, hide ceiling slab for overview shots
def room_lights():
    lights = []
    for (x, y) in [(-2.4, 0.0), (2.4, 0.0), (0.0, 1.2), (0.0, -1.2)]:
        ld = bpy.data.lights.new("PREVIEW_room_light", "AREA")
        ld.energy = 900.0
        ld.size = 1.5
        o = bpy.data.objects.new("PREVIEW_room_light", ld)
        o.location = (x, y, 2.6)
        bpy.context.scene.collection.objects.link(o)
    return lights


room_lights()
hide_top = [o.name for o in S.top_coll.all_objects if o.name.startswith(AID + "_ceiling") or o.name.startswith(AID + "_light_panel") or o.name.startswith(AID + "_hvac") or o.name.startswith(AID + "_smoke")]
views = [("front", dict(loc=(0, -2.9, 1.55), target=(0, 1.0, 1.0), lens=22, hide=hide_top)),
         ("three_quarter", dict(loc=(-3.9, -2.8, 2.3), target=(0.5, 0.6, 0.9), lens=24, hide=hide_top)),
         ("top", dict(loc=(0, -0.2, 9.0), target=(0, 0, 0), lens=35, hide=hide_top)),
         ("closeup_table_far", dict(loc=(0.3, -0.9, 1.25), target=(0.3, 1.4, 0.7), lens=30, hide=hide_top)),
         ("closeup_tv_door", dict(loc=(-1.0, -2.8, 1.6), target=(3.5, 0.0, 1.3), lens=22, hide=hide_top)),
         ("closeup_door_partition", dict(loc=(1.2, 1.0, 1.5), target=(-4.2, -0.4, 1.2), lens=24, hide=hide_top)),
         ("ceiling_windows", dict(loc=(0, -2.9, 1.7), target=(0, 3.0, 1.9), lens=18))]
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=False, lights="none", world=0.9)
