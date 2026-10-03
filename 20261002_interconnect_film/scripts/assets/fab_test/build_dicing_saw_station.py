"""dicing_saw_station: generic fully-automatic 300 mm blade dicing saw: enclosure, X chuck table with theta, spindle + flange + blade, water nozzles,
alignment camera, load port, control screen. Origin: floor centre; front -Y.
Hooks: HOOK_blade_spin (p_spin deg), HOOK_cut_line, HOOK_table / HOOK_wafer_slot, HOOK_spray_l/r.
Run: Blender -b --python scripts/assets/fab_test/build_dicing_saw_station.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("dicing_saw_station", accuracy="B")
MM = ft.MM
A.src("DISCO DFD6361 fully automatic dicing saw for 300 mm: 1200 x 1550 x 1800 mm, about 2050 kg, X cutting range 310 mm, spindle 6,000-60,000 rpm, 2 inch blade Z stroke 14.7 mm",
      "https://www.disco.co.jp/eg/products/dicer/dfd6361.html (summarised via web search 2026-10-02)", "enclosure size, travel, spindle speed (layout itself generic)")
A.src("Task brief: blade thickness 20-50 um, blade OD about 55 mm (2 inch hubless / hub blade ~55.56 mm)", "assets brief", "blade geometry")
A.dim("machine W x D x H", "1200 x 1550 x 1800", "mm", "DISCO DFD6361 public listing", "A")
A.dim("blade OD / thickness", "55.6 / 0.03", "mm", "2 inch blade; 30 um within 20-50 um brief", "B")
A.dim("flange OD", 56, "mm", "2 inch flange, estimate", "C")
A.dim("chuck table diameter", 330, "mm", "300 mm wafer + frame clips, estimate", "C")
A.dim("X cutting range", 310, "mm", "DFD6361 listing", "A")
A.dim("spindle speed", "6,000 - 60,000", "rpm", "DFD6361 listing (animation does not need exact)", "A")
A.prop("p_spin", 0.0, "blade rotation angle in degrees (keyframe 0 -> N*360); 60,000 rpm = 360000 deg/s", None, None, "deg")
A.prop("p_table_x", 0.0, "chuck table X position (mm), cutting direction", -160.0, 160.0, "mm")
A.prop("p_table_theta", 0.0, "chuck table theta (deg)", -95.0, 95.0, "deg")
A.prop("p_index_y", 0.0, "spindle index (Y) offset (mm)", -150.0, 150.0, "mm")
A.prop("p_spray", 1.0, "water spray on (>= 0.5) / off (< 0.5): driver on hide_render of the water jets", 0.0, 1.0)
paint = A.custom_mat("saw_paint", (0.30, 0.34, 0.38), metallic=0.1, rough=0.5, coat=0.2)
water = A.custom_mat("saw_water", (0.7, 0.85, 1.0), rough=0.05, alpha=0.35)
gm = A.custom_mat("saw_glass", (0.5, 0.65, 0.75), rough=0.05, alpha=0.1)
ZD = 960.0
CYc = 60.0
ZW = 1060.0 + 0.08 + 0.775
# enclosure
for sx in (-1, 1):
    for sy in (-1, 1):
        A.cyl("dicing_saw_station_foot", 40, 60, (sx * 540, sy * 700, 0), "black_anodized", anchor="b", seg=24)
A.box("dicing_saw_station_cabinet", (1200, 1550, ZD - 60), (0, 0, 60), paint, anchor="b", bev=4)
for k in range(3):
    A.box("dicing_saw_station_cabinet_door", (360, 4, 760), (-400 + k * 400, -777, 110), "grey_plastic", anchor="b", bev=2)
A.box("dicing_saw_station_deck", (1180, 1500, 20), (0, 0, ZD), "steel", anchor="b", bev=2)
for (x, y) in ((-590, -740), (590, -740), (-590, 740), (590, 740)):
    A.box("dicing_saw_station_hood_post", (40, 40, 800), (x, y, ZD + 20), "silver", anchor="b", bev=3)
A.box("dicing_saw_station_hood_roof", (1200, 1520, 20), (0, 0, ZD + 820), paint, anchor="b", bev=3)
A.box("dicing_saw_station_hood_back", (1180, 20, 800), (0, 750, ZD + 20), paint, anchor="b")
for sx in (-1, 1):
    A.box("dicing_saw_station_hood_side", (20, 1480, 800), (sx * 590, 0, ZD + 20), paint, anchor="b")
vg = A.variant("hood_glass")
A.cur = vg
A.box("dicing_saw_station_glass_front", (1160, 4, 780), (0, -740, ZD + 30), gm, anchor="b")
A.cur = A.coll
# table chain
HT = A.hook("table", (0, CYc, 1040), A.root, size=0.1)
A.drive(HT, "location", 0, "p_table_x*0.001", ["p_table_x"])
HTH = A.hook("table_theta", (0, CYc, 1040), HT, size=0.06)
A.drive(HTH, "rotation_euler", 2, "p_table_theta*0.01745329", ["p_table_theta"])
A.box("dicing_saw_station_x_rail", (700, 30, 20), (0, CYc - 160, ZD + 20), "steel", anchor="b", bev=1)
A.box("dicing_saw_station_x_rail", (700, 30, 20), (0, CYc + 160, ZD + 20), "steel", anchor="b", bev=1)
A.box("dicing_saw_station_table_carriage", (480, 460, 40), (0, CYc, ZD + 40), "steel_dark", anchor="b", bev=2, parent=HT)
A.cyl("dicing_saw_station_theta_ring", 175, 20, (0, CYc, 1000), "silver", anchor="b", seg=64, bev=1, parent=HTH)
A.cyl("dicing_saw_station_chuck", 165, 40, (0, CYc, 1020), "ceramic", anchor="b", seg=96, bev=0.8, parent=HTH)
A.cyl("dicing_saw_station_frame_plate", 205, 4, (0, CYc, 1056), "steel", anchor="b", seg=96, r_in=168, parent=HTH)
for k in range(4):
    a = math.pi / 2 * k + math.pi / 4
    A.box("dicing_saw_station_frame_clamp", (30, 14, 18), (212 * math.cos(a), CYc + 212 * math.sin(a), 1056), "black_anodized", anchor="b", bev=1.5, rot_z=a, parent=HTH)
A.hook("wafer_slot", (0, CYc, 1060), HTH, size=0.1)    # tape-frame origin (tape bottom) sits here
# spindle assembly (index Y)
HI = A.hook("index_y", (0, 0, 0), A.root, size=0.05)
A.drive(HI, "location", 1, "p_index_y*0.001", ["p_index_y"])
ZB = ZW + 27.8 - 0.3
YB = CYc
A.box("dicing_saw_station_z_column", (160, 140, 520), (0, 560, ZD + 20), "black_anodized", anchor="b", bev=4, parent=HI)
A.box("dicing_saw_station_spindle_mount", (140, 120, 160), (0, 480, ZB - 80), "steel_dark", anchor="b", bev=4, parent=HI)
A.cyl("dicing_saw_station_spindle_housing", 36, 260, (0, YB + 260, ZB), "silver", axis="y", parent=HI, seg=48, bev=1)
A.cyl("dicing_saw_station_spindle_nose", 28, 50, (0, YB + 110, ZB), "steel_dark", axis="y", parent=HI, seg=48)
HS = A.hook("blade_spin", (0, YB + 70, ZB), HI, size=0.08)
A.drive(HS, "rotation_euler", 1, "p_spin*0.01745329", ["p_spin"])
A.cyl("dicing_saw_station_flange_back", 23, 6, (0, YB + 44, ZB), "chrome", axis="y", parent=HS, seg=48)
A.cyl("dicing_saw_station_blade", 27.8, 0.03, (0, YB, ZB), "silver", axis="y", parent=HS, seg=160, r_in=19.5)
A.cyl("dicing_saw_station_flange_front", 23, 6, (0, YB - 4, ZB), "chrome", axis="y", parent=HS, seg=48)
A.cyl("dicing_saw_station_blade_nut", 8, 6, (0, YB - 9, ZB), "steel_dark", axis="y", parent=HS, seg=6)
for k in range(6):
    a = math.pi / 3 * k
    A.cyl("dicing_saw_station_flange_bolt", 1.6, 2, (16 * math.cos(a), YB - 7.5, ZB + 16 * math.sin(a)), "black_anodized", axis="y", parent=HS, seg=10)
A.box("dicing_saw_station_flange_mark", (3, 0.5, 8), (0, YB - 7.6, ZB + 20), "red", parent=HS)
# blade guard + water nozzles
A.box("dicing_saw_station_blade_guard", (80, 60, 40), (0, YB + 10, ZB + 44), "black_anodized", anchor="b", bev=3, parent=HI)
for sg, nm in ((-1, "l"), (1, "r")):
    tipx = sg * 0.0
    y = YB + (-14 if sg < 0 else 14)
    A.tube_path("dicing_saw_station_nozzle_" + nm, [(sg * 0 + 40, y, ZB + 55), (36, y, ZB + 20), (8, y, ZW + 6)], 1.6, "steel", seg=10, parent=HI)
    jet = A.tube_path("dicing_saw_station_spray_" + nm, [(8, y, ZW + 6), (3, y, ZW + 0.5)], [0.3, 3.0], water, seg=14, parent=HI, caps=False)
    fcv = jet.driver_add("hide_render"); fcv.driver.type = "SCRIPTED"; fcv.driver.expression = "p_spray < 0.5"
    vv = fcv.driver.variables.new(); vv.name = "p_spray"; vv.type = "SINGLE_PROP"; vv.targets[0].id = A.root; vv.targets[0].data_path = '["p_spray"]'
    A.hook("spray_" + nm, (8, y, ZW + 6), HI, size=0.02)
A.hook("cut_line", (0, YB, ZW), HI, size=0.04)   # X axis = cutting direction
# alignment microscope on a bridge
A.box("dicing_saw_station_bridge", (30, 40, 300), (-260, 400, ZD + 20), "black_anodized", anchor="b", bev=2)
A.box("dicing_saw_station_scope_arm", (30, 360, 30), (-260, 220, ZD + 290), "black_anodized", anchor="b", bev=2)
A.cyl("dicing_saw_station_scope_body", 32, 140, (-260, 40, ZD + 150), "black_anodized", anchor="b", seg=40, bev=1)
A.cyl("dicing_saw_station_scope_lens", 24, 14, (-260, 40, ZD + 140), "glass", anchor="b", seg=32)
# load port (left front): FOUP seat
A.box("dicing_saw_station_loadport_base", (520, 420, 40), (-690, -560, ZD - 100), "black_anodized", anchor="b", bev=3)
A.box("dicing_saw_station_loadport_plate", (440, 360, 10), (-690, -560, ZD - 60), "steel", anchor="b", bev=1.5)
for (x, y) in ((0, 100), (-80, -90), (80, -90)):
    A.cyl("dicing_saw_station_loadport_pin", 6, 10, (-690 + x * 1.0, -560 + y * 1.0, ZD - 50), "steel_dark", anchor="b", seg=16)
A.hook("load_port", (-690, -560, ZD - 50), A.root, size=0.08)
# control screen on arm and light tower
A.box("dicing_saw_station_screen_arm", (50, 50, 300), (560, -760, ZD - 300 + 160), "black_anodized", anchor="b", bev=3)
A.box("dicing_saw_station_monitor", (420, 30, 270), (560, -780, ZD + 80), "black_plastic", anchor="b", bev=4)
A.screen_plane("dicing_saw_station_screen", (560, ZD + 215), (390, 235), -796.5)
A.hook("screen", (560, -796.5, ZD + 215), A.root, size=0.03)
A.cyl("dicing_saw_station_tower_pole", 9, 300, (-560, -720, ZD + 840), "steel", anchor="b", seg=16)
for k, mk in enumerate(("red_led", "yellow", "green_led")):
    A.cyl("dicing_saw_station_tower_light", 26, 34, (-560, -720, ZD + 1140 + k * 36), mk, anchor="b", seg=32, bev=1.5)
A.preview_fit = 1.05
A.preview_hide = ["VARIANT_hood_glass"]
meta = {"description": "Generic 300 mm dicing saw (envelope per DISCO DFD6361): chuck table, spindle with 2 inch blade, water jets, microscope, load port.",
        "origin": "floor centre; front -Y; the saw cuts along X (table moves in X)",
        "scene_usage": "S5 'DICING + TAPE': mount wafer_tape_frame on HOOK_wafer_slot (origin bottom of tape), animate p_spin and p_table_x, p_index_y per street.",
        "variants": {"VARIANT_hood_glass": "front glass panel (alpha 0.1); hide for open shots (previews are rendered with it hidden)"},
        "simplifications": ["Single spindle (real DFD6361 has two facing spindles)", "Blade has no hub/nickel detail; the rim glow is not modelled", "Water jets are static cones toggled by p_spray, no particles", "Interior mechanics reduced to blocks"]}
A.finish(OUT, meta, views=("front", "three_quarter", "top"),
         closeups=[("blade_close", (60, -80, ZW + 55), (0, CYc, ZW + 20), 70, None, 1e-3)])
