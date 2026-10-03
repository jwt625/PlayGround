"""die_bonder_station: generic flip-chip / thermo-compression bonder: granite base, X rails + Y bridge gantry, bond head with heated
tool and pick-up collet, heated substrate chuck, die tray, vision cameras, enclosure frame, monitor, light tower.
Origin: floor centre; front -Y; head rest position above the substrate chuck (x = +150).
Run: Blender -b --python scripts/assets/fab_test/build_die_bonder_station.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("die_bonder_station", accuracy="C")
MM = ft.MM
A.src("Wentao's flip-chip bonder photo (bond head over a wafer on a black chuck)", "/Users/wentaojiang/Documents/GitHub/jwt625.github.io/assets/images/2025/20251219_CPO/flip-chip-bonder-wafer.webp", "head/collet/chuck layout")
A.src("ASMPT FIREBIRD TCB / NANO Lite, Finetech FINEPLACER femto pro public pages: placement accuracy 1-2 um, bonding above 350 C", "https://www.asmpt.com/en/innovation/thermo-compression-bonding/", "capability context (no dimensions)")
A.dim("machine footprint (this model)", "1200 x 1000 (enclosure 1240 x 1060)", "mm", "generic estimate; public TCB brochures do not list footprints in the sources used", "C")
A.dim("overall height (enclosure top, light tower)", "1750 / 2100", "mm", "estimate", "C")
A.dim("granite top height", 940, "mm", "estimate", "C")
A.dim("heated chuck top", 1000, "mm", "estimate", "C")
A.dim("head travel", "X -480..0, Y +-300, Z 0..45 (relative to rest)", "mm", "estimate", "C")
A.dim("collet tip", "6 x 7 mm pad on a 4 mm taper", "mm", "sized for a 5 x 6 mm EIC", "C")
CX = 150.0   # chuck / rest x
ZCH = 1000.0
grey = A.custom_mat("machine_paint", (0.26, 0.30, 0.34), metallic=0.1, rough=0.5, coat=0.2)
granite = A.custom_mat("granite", (0.025, 0.027, 0.03), metallic=0.0, rough=0.25)
heat = A.custom_mat("heater_glow", (0.05, 0.03, 0.02), emit=(1.0, 0.3, 0.05), emit_strength=0.0)
b = heat.node_tree.nodes["Principled BSDF"]
fc = b.inputs["Emission Strength"].driver_add("default_value")
fc.driver.type = "SCRIPTED"; fc.driver.expression = "p_heat*2.0"
v = fc.driver.variables.new(); v.name = "p_heat"; v.type = "SINGLE_PROP"; v.targets[0].id = A.root; v.targets[0].data_path = '["p_heat"]'
A.prop("p_head_x", 0.0, "head X offset from the rest position over the chuck (mm, negative toward the die tray, -480 = pick position)", -520.0, 120.0, "mm")
A.prop("p_head_y", 0.0, "head Y offset (mm)", -300.0, 300.0, "mm")
A.prop("p_head_z", 0.0, "head lowering (mm), 0 = up, 45 = bond contact", 0.0, 60.0, "mm")
A.prop("p_heat", 0.0, "heater glow of chuck and bond tool (0..1)", 0.0, 1.0)
# base
for sx in (-1, 1):
    for sy in (-1, 1):
        A.cyl("die_bonder_station_foot", 35, 60, (sx * 540, sy * 440, 0), "black_anodized", anchor="b", seg=24)
A.box("die_bonder_station_cabinet", (1200, 1000, 760), (0, 0, 60), grey, anchor="b", bev=4)
for k in range(3):
    A.box("die_bonder_station_cabinet_door", (360, 4, 640), (-400 + k * 400, -502, 110), "grey_plastic", anchor="b", bev=2)
    A.box("die_bonder_station_cabinet_handle", (10, 8, 120), (-400 + k * 400 + 150, -508, 380), "silver", anchor="b", bev=2)
A.box("die_bonder_station_granite", (1100, 900, 120), (0, 0, 820), granite, anchor="b", bev=2)
# substrate stage, heated chuck, substrate with PIC
A.box("die_bonder_station_stage", (300, 300, 40), (CX, 0, 940), "steel", anchor="b", bev=2)
A.cyl("die_bonder_station_chuck_body", 90, 20, (CX, 0, 980), "black_anodized", anchor="b", seg=64, bev=1)
A.cyl("die_bonder_station_chuck_heater", 85, 3, (CX, 0, 997), heat, anchor="b", seg=64)
for k in range(4):
    a = math.pi / 2 * k + math.pi / 4
    A.cyl("die_bonder_station_chuck_vacuum_port", 4, 3, (CX + 70 * math.cos(a), 70 * math.sin(a), 1000), "steel_dark", anchor="b", seg=12)
A.box("die_bonder_station_substrate", (60, 60, 1.4), (CX, 0, 1000), "pcb_green", anchor="b", bev=0.2)
A.box("die_bonder_station_pic", (9, 7, 0.775), (CX, 0, 1001.4), "silicon", anchor="b")
A.box("die_bonder_station_pic_film", (9, 7, 0.02), (CX, 0, 1002.175), "gold", anchor="b")
# die tray (waffle) with EICs
A.box("die_bonder_station_tray", (170, 170, 12), (CX - 480, -30, 940), "steel", anchor="b", bev=2)
d = A.box("die_bonder_station_eic_proto", (5, 6, 0.3), (0, 0, 952), "silicon", anchor="b")
d.hide_render = d.hide_viewport = True
for i in range(6):
    for j in range(6):
        A.dup(d, "die_bonder_station_tray_eic", (CX - 480 - 62 + i * 25, -30 - 62 + j * 25, 952 + 0.15))
# upward-looking alignment camera
A.cyl("die_bonder_station_up_camera", 28, 90, (CX - 220, -30, 940), "black_anodized", anchor="b", seg=32, bev=1.5)
A.cyl("die_bonder_station_up_lens", 22, 10, (CX - 220, -30, 1030), "glass", anchor="b", seg=32)
# rails (X) and bridge
for sy in (-1, 1):
    A.box("die_bonder_station_x_rail", (1000, 40, 30), (-30, sy * 430, 940), "steel", anchor="b", bev=1.5)
    A.box("die_bonder_station_x_rail_cover", (1000, 60, 16), (-30, sy * 430, 970), "grey_plastic", anchor="b")
BR = A.frame("bridge_frame", (0, 0, 0), A.root)
A.drive(BR, "location", 0, "p_head_x*0.001", ["p_head_x"])
for sy in (-1, 1):
    A.box("die_bonder_station_bridge_foot", (160, 90, 270), (CX - 90, sy * 430, 940), "black_anodized", anchor="b", bev=3, parent=BR)
A.box("die_bonder_station_bridge_beam", (110, 940, 120), (CX - 90, 0, 1210), "steel", anchor="b", bev=4, parent=BR)
A.box("die_bonder_station_bridge_rail", (14, 900, 10), (CX - 90 + 62, 0, 1250), "silver", anchor="b", parent=BR)
# head xy
HXY = A.hook("head_xy", (CX, 0, 1270), A.root, size=0.1)
A.drive(HXY, "location", 0, "%.6f+p_head_x*0.001" % (CX * MM), ["p_head_x"])
A.drive(HXY, "location", 1, "p_head_y*0.001", ["p_head_y"])
A.box("die_bonder_station_head_carriage", (190, 170, 150), (CX - 20, 0, 1220), "black_anodized", anchor="b", bev=4, parent=HXY)
A.cyl("die_bonder_station_top_camera", 26, 110, (CX + 60, 40, 1250), "black_anodized", anchor="b", parent=HXY, seg=32, bev=1)
A.cyl("die_bonder_station_top_camera_lens", 20, 14, (CX + 60, 40, 1236), "glass", anchor="b", parent=HXY, seg=32)
HZ = A.hook("head_z", (CX, 0, 1270), HXY, size=0.08)
A.drive(HZ, "location", 2, "-p_head_z*0.001", ["p_head_z"])   # relative to HXY (rest: 0)
A.box("die_bonder_station_z_housing", (90, 90, 260), (CX + 15, 0, 1130), "steel", anchor="b", bev=4, parent=HZ)
A.box("die_bonder_station_z_motor", (60, 60, 90), (CX + 15, 0, 1390), "black_anodized", anchor="b", bev=3, parent=HZ)
A.box("die_bonder_station_tool_cooling_plate", (70, 70, 14), (CX, 0, 1116), "silver", anchor="b", bev=1, parent=HZ)
A.box("die_bonder_station_tool_heater", (60, 60, 36), (CX, 0, 1080), heat, anchor="b", bev=1, parent=HZ)
for sx in (-1, 1):
    A.cyl("die_bonder_station_tool_cable_gland", 8, 20, (CX + sx * 35, 0, 1100), "silver", axis="x", parent=HZ, seg=16)
A.cyl("die_bonder_station_collet", 12, 36, (CX, 0, 1044), "ceramic", anchor="b", parent=HZ, seg=32, r_top=24)
A.box("die_bonder_station_collet_pad", (6, 7, 0.5), (CX, 0, 1043.5), "steel", anchor="b", parent=HZ)
A.cyl("die_bonder_station_vacuum_tube", 3, 60, (CX + 28, 0, 1120), "rubber", anchor="b", parent=HZ, seg=12)
tip = A.hook("collet_tip", (CX, 0, 1043.5), HZ, size=0.03)
A.hook("bond_point", (CX, 0, 1002.2), A.root, size=0.03)
A.hook("die_pick_point", (CX - 480, -30, 952.3), A.root, size=0.03)
A.hook("substrate_top", (CX, 0, 1001.4), A.root, size=0.03)
# enclosure frame, top and glass
post_pts = [(-610, -520), (610, -520), (-610, 520), (610, 520)]
for (x, y) in post_pts:
    A.box("die_bonder_station_frame_post", (40, 40, 800), (x, y, 940), "silver", anchor="b", bev=3)
for (x0, y0, x1, y1) in ((-610, -520, 610, -520), (-610, 520, 610, 520)):
    A.box("die_bonder_station_frame_beam_x", (1220, 40, 40), (0, y0, 1740), "silver", anchor="b", bev=3)
for sx in (-1, 1):
    A.box("die_bonder_station_frame_beam_y", (40, 1040, 40), (sx * 610, 0, 1740), "silver", anchor="b", bev=3)
A.box("die_bonder_station_roof", (1240, 1060, 20), (0, 0, 1780), grey, anchor="b", bev=3)
A.box("die_bonder_station_ioniser", (900, 40, 30), (0, -480, 1700), "white_plastic", anchor="b", bev=3)
vg = A.variant("enclosure_glass")
A.cur = vg
gm = A.custom_mat("enclosure_glass", (0.5, 0.65, 0.75), rough=0.05, alpha=0.1)
A.box("die_bonder_station_glass_front", (1180, 4, 780), (0, -520, 960), gm, anchor="b")
A.box("die_bonder_station_glass_back", (1180, 4, 780), (0, 520, 960), gm, anchor="b")
for sx in (-1, 1):
    A.box("die_bonder_station_glass_side", (4, 1000, 780), (sx * 610, 0, 960), gm, anchor="b")
A.cur = A.coll
# monitor, light tower
A.box("die_bonder_station_monitor_arm", (40, 40, 400), (740, -480, 940), "black_anodized", anchor="b", bev=3)
A.box("die_bonder_station_monitor", (420, 24, 260), (740, -500, 1340), "black_plastic", anchor="b", bev=4)
A.box("die_bonder_station_monitor_screen", (390, 2, 230), (740, -513, 1355), A.custom_mat("bonder_screen", (0.02, 0.06, 0.1), emit=(0.1, 0.4, 0.7), emit_strength=0.6), anchor="b")
A.cyl("die_bonder_station_tower_pole", 8, 500, (-690, 520, 1780), "steel", anchor="b", seg=16)
for k, mk in enumerate(("red_led", "yellow", "green_led")):
    A.cyl("die_bonder_station_tower_light", 24, 34, (-690, 520, 2280 + k * 36), mk, anchor="b", seg=32, bev=1.5)
A.preview_fit = 1.0
meta = {"description": "Generic flip-chip / thermo-compression bonder (gantry, heated tool, collet, heated chuck, die tray, vision).",
        "origin": "floor centre; head rest over the substrate chuck at x = +150 mm",
        "scene_usage": "S5 EIC/PIC BONDING station: p_head_x -480 to pick an EIC at the tray, then p_head_x 0 over the substrate, p_head_z 0 -> 45 to bond, p_heat to glow.",
        "variants": {"VARIANT_enclosure_glass": "transparent glass panels (alpha 0.1); hide for unobstructed shots"},
        "simplifications": ["Generic geometry from the flip-chip bonder photo, no vendor machine copied", "Granite base, gantry, enclosure are plain boxes", "Interior cabling omitted"]}
A.finish(OUT, meta, views=("front", "three_quarter", "top"),
         closeups=[("head_close", (CX - 400, -520, 1200), (CX, 0, 1040), 60)])
