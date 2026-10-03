"""fau_attach_station: fibre-array-unit (FAU) active-alignment and attach bench: granite top, PIC chuck, XYZ + tilt stack carrying a
fibre holder with FAU and ribbon, UV/epoxy dispenser, UV lamp, microscope, power meter with a screen.
Origin: floor centre; front -Y; the PIC sits at x = 0, FAU approaches from -X.
Run: Blender -b --python scripts/assets/fab_test/build_fau_attach_station.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("fau_attach_station", accuracy="C")
MM = ft.MM
A.src("Wentao's Broadcom CPO assembly flow slide (OFC 2026): FAU attach is a step in the OE assembly", "/Users/wentaojiang/Documents/GitHub/jwt625.github.io/assets/images/2026/OFC2026/IMG_4084.JPG", "context")
A.src("Standard fibre array pitch 250 um, 125 um cladding (SMF-28 / G.652 class); generic active alignment bench concept (hexapod or XYZ + tilt, power meter feedback, UV epoxy)", "n/a", "fibre pitch and radius")
A.dim("bench footprint", "900 x 700", "mm", "estimate", "C")
A.dim("overall height (microscope top)", 1350, "mm", "estimate", "C")
A.dim("PIC / chuck height", "900 / ~872 top", "mm", "estimate", "C")
A.dim("FAU glass block", "8.0 x 4.0 x 1.5", "mm", "typical 8-channel V-groove FAU, estimate", "C")
A.dim("fibre pitch / cladding radius", "0.25 / 0.0625 (true), drawn 0.125", "mm", "standard; radius doubled for visibility", "B")
A.dim("PIC", "9 x 7 x 0.775", "mm", "project PIC", "B")
A.prop("p_fau_x", 0.0, "FAU stage X offset (mm), + toward the PIC (+X)", -30.0, 30.0, "mm")
A.prop("p_fau_y", 0.0, "FAU stage Y offset (mm)", -20.0, 20.0, "mm")
A.prop("p_fau_z", 0.0, "FAU stage Z offset (mm)", -20.0, 20.0, "mm")
A.prop("p_fau_tilt", 0.0, "FAU pitch tilt about Y (deg)", -10.0, 10.0, "deg")
A.prop("p_dispense", 0.0, "dispenser lowering (mm), 0 = up, 40 = at the joint", 0.0, 40.0, "mm")
A.prop("p_uv", 0.0, "UV lamp glow (0..1)", 0.0, 1.0)
paint = A.custom_mat("bench_paint", (0.24, 0.28, 0.32), metallic=0.1, rough=0.5, coat=0.2)
granite = A.custom_mat("granite", (0.025, 0.027, 0.03), metallic=0.0, rough=0.25)
uvm = A.custom_mat("uv_glow", (0.1, 0.05, 0.3), emit=(0.4, 0.1, 1.0), emit_strength=0.0)
b = uvm.node_tree.nodes["Principled BSDF"]
fc = b.inputs["Emission Strength"].driver_add("default_value")
fc.driver.type = "SCRIPTED"; fc.driver.expression = "p_uv*6.0"
v = fc.driver.variables.new(); v.name = "p_uv"; v.type = "SINGLE_PROP"; v.targets[0].id = A.root; v.targets[0].data_path = '["p_uv"]'
ZT = 800.0     # granite top
ZP = 872.0     # PIC top surface
# base
for sx in (-1, 1):
    for sy in (-1, 1):
        A.cyl("fau_attach_station_foot", 30, 30, (sx * 400, sy * 300, 0), "black_anodized", anchor="b", seg=24)
A.box("fau_attach_station_cabinet", (900, 700, 720), (0, 0, 30), paint, anchor="b", bev=4)
for k in range(2):
    A.box("fau_attach_station_cabinet_door", (400, 4, 620), (-220 + k * 440, -352, 80), "grey_plastic", anchor="b", bev=2)
A.box("fau_attach_station_granite", (860, 660, 50), (0, 0, 750), granite, anchor="b", bev=2)
# PIC stage
A.box("fau_attach_station_pic_stage_base", (220, 220, 40), (0, 0, ZT), "blue_anodized", anchor="b", bev=2)
A.box("fau_attach_station_pic_stage_top", (160, 160, 20), (0, 0, ZT + 40), "steel", anchor="b", bev=2)
A.cyl("fau_attach_station_chuck", 35, 8, (0, 0, ZT + 60), "black_anodized", anchor="b", seg=48, bev=0.5)
A.box("fau_attach_station_carrier", (40, 40, 3), (0, 0, ZT + 68), "steel", anchor="b", bev=0.3)
A.box("fau_attach_station_pic", (9, 7, 0.775), (0, 0, ZP - 0.795), "silicon", anchor="b")
A.box("fau_attach_station_pic_film", (9, 7, 0.02), (0, 0, ZP - 0.02), "pcb_blue", anchor="b")
for k in range(8):
    A.box("fau_attach_station_pic_grating_port", (0.3, 0.12, 0.01), (-4.3, (k - 3.5) * 0.25, ZP), "gold", anchor="b")
# fixed XYZ tower to the left (decor: Y,Z stages) and moving carriage
A.box("fau_attach_station_xyz_base", (200, 160, 40), (-300, 0, ZT), "blue_anodized", anchor="b", bev=2)
A.box("fau_attach_station_xyz_z_column", (90, 110, 120), (-340, 0, ZT + 40), "blue_anodized", anchor="b", bev=2)
HX = A.hook("fau_xyz", (-300, 0, ZP), A.root, size=0.06)
A.drive(HX, "location", 0, "%.6f+p_fau_x*0.001" % (-300 * MM), ["p_fau_x"])
A.drive(HX, "location", 1, "p_fau_y*0.001", ["p_fau_y"])
A.drive(HX, "location", 2, "%.6f+p_fau_z*0.001" % (ZP * MM), ["p_fau_z"])
A.box("fau_attach_station_xyz_carriage", (110, 110, 70), (-300, 0, ZP - 120), "blue_anodized", anchor="b", bev=2, parent=HX)
A.box("fau_attach_station_xyz_arm", (160, 40, 40), (-250, 0, ZP - 60), "steel", anchor="b", bev=2, parent=HX)
HT = A.hook("fau_tilt", (-190, 0, ZP), HX, size=0.04)
A.drive(HT, "rotation_euler", 1, "p_fau_tilt*0.01745329", ["p_fau_tilt"])
A.box("fau_attach_station_tilt_block", (50, 40, 36), (-190, 0, ZP - 40), "steel_dark", anchor="b", bev=2, parent=HT)
A.box("fau_attach_station_fibre_holder", (36, 30, 12), (-150, 0, ZP - 14), "silver", anchor="b", bev=1, parent=HT)
A.box("fau_attach_station_fibre_holder_clamp", (30, 24, 6), (-150, 0, ZP - 2), "steel", anchor="b", bev=0.8, parent=HT)
A.box("fau_attach_station_fau_glass", (4.0, 8.0, 1.5), (-6.0, 0, ZP - 1.0), "glass_tint", anchor="b", parent=HT)
A.box("fau_attach_station_fau_lid", (6.0, 8.0, 1.0), (-8.0, 0, ZP + 0.5), "glass_tint", anchor="b", parent=HT)
for k in range(8):
    y = (k - 3.5) * 0.25
    A.tube_path("fau_attach_station_fibre", [(-4.0, y, ZP - 0.4), (-12.0, y, ZP - 0.4), (-60, y, ZP - 0.3), (-140, y, ZP + 1.0)], 0.125, "fiber_glass", seg=8, parent=HT)
A.tube_path("fau_attach_station_ribbon", [(-148, 0, ZP + 1.0), (-190, 0, ZP + 6), (-260, 0, ZP - 40), (-300, -60, ZP - 160), (-330, -200, ZP - 70),
                                          (-100, -280, ZP - 60), (320, -300, ZP - 20)], 2.0, "fiber_yellow", seg=10)
A.hook("fau_attach", (-4.5, 0, ZP - 0.4), HT, size=0.01)
A.hook("pic_edge_target", (-4.5, 0, ZP - 0.4), A.root, size=0.01)
# dispenser (right/back), moves down with p_dispense
A.box("fau_attach_station_disp_column", (50, 50, 360), (60, 260, ZT), "steel", anchor="b", bev=2)
HD = A.hook("dispenser", (-2, 40, ZP + 40), A.root, size=0.04)
A.drive(HD, "location", 2, "%.6f-p_dispense*0.001" % ((ZP + 40) * MM), ["p_dispense"])
A.box("fau_attach_station_disp_slide", (40, 90, 40), (40, 150, ZP + 40), "black_anodized", anchor="b", bev=2, parent=HD)
A.cyl("fau_attach_station_syringe", 8, 70, (-2, 40, ZP + 40), "fiber_yellow", anchor="b", parent=HD, seg=24, bev=0.6)
A.cyl("fau_attach_station_syringe_piston", 5, 40, (-2, 40, ZP + 110), "silver", anchor="b", parent=HD, seg=16)
A.cyl("fau_attach_station_needle_hub", 3, 8, (-2, 40, ZP + 32), "silver", anchor="b", parent=HD, seg=12)
A.tube_path("fau_attach_station_needle", [(-2, 40, ZP + 33), (-2, 40, ZP + 6), (-5, 5, ZP - 0.5)], 0.3, "steel", seg=8, parent=HD)
A.hook("epoxy_tip", (-5, 5, ZP - 0.5), HD, size=0.01)
# UV lamp gooseneck
A.cyl("fau_attach_station_uv_base", 40, 25, (200, 200, ZT), "black_anodized", anchor="b", seg=32, bev=1)
A.tube_path("fau_attach_station_uv_guide", [(200, 200, ZT + 25), (200, 200, ZT + 150), (140, 150, ZT + 200), (30, 40, ZP + 40), (-3, 8, ZP + 8)], 4.0, "black_plastic", seg=12)
A.cyl("fau_attach_station_uv_tip", 3.5, 3, (-3, 8, ZP + 6), uvm, anchor="b", seg=16)
A.hook("uv_tip", (-3, 8, ZP + 6), A.root, size=0.01)
# microscope: column behind, objective above the joint; side camera
A.cyl("fau_attach_station_scope_post", 28, 520, (-60, 290, ZT), "steel", anchor="b", seg=32, bev=1)
A.box("fau_attach_station_scope_arm", (40, 300, 40), (-60, 140, ZT + 480), "black_anodized", anchor="b", bev=3)
A.box("fau_attach_station_scope_body", (90, 90, 150), (-60, 0, ZT + 330), "black_anodized", anchor="b", bev=4)
A.cyl("fau_attach_station_scope_objective", 26, 80, (-60, 0, ZT + 250), "steel_dark", anchor="b", seg=40, bev=1)
A.cyl("fau_attach_station_scope_lens", 18, 2, (-60, 0, ZT + 249), "glass", anchor="b", seg=32)
A.cyl("fau_attach_station_side_camera", 22, 120, (-60, -300, ZP - 20), "black_anodized", axis="y", seg=32, bev=1)
# power meter instrument and screen
A.box("fau_attach_station_meter", (260, 220, 120), (340, -250, ZT), "black_paint", anchor="b", bev=4)
for k in range(3):
    A.cyl("fau_attach_station_meter_knob", 14, 14, (260 + k * 40, -362, ZT + 30), "silver", axis="y", seg=24, bev=1)
A.box("fau_attach_station_meter_ports", (50, 6, 30), (430, -362, ZT + 30), "steel_dark")
smat = bpy.data.materials.new("MAT_fab_test_screen")
smat.use_nodes = True
nt = smat.node_tree
for n_ in list(nt.nodes):
    nt.nodes.remove(n_)
o = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
fv = nt.nodes.new("ShaderNodeValue"); fv.name = "MIX_use_image"; fv.label = "MIX_use_image (0 default, 1 image)"; fv.outputs[0].default_value = 0.0
im = nt.nodes.new("ShaderNodeTexImage"); im.name = "IMG_screen"; im.label = "IMG_screen (assign image sequence)"
tc = nt.nodes.new("ShaderNodeTexCoord"); br = nt.nodes.new("ShaderNodeTexGradient")
rp = nt.nodes.new("ShaderNodeValToRGB"); rp.color_ramp.elements[0].color = (0.0, 0.05, 0.02, 1); rp.color_ramp.elements[1].color = (0.1, 0.9, 0.3, 1)
nt.links.new(tc.outputs["UV"], im.inputs["Vector"]); nt.links.new(tc.outputs["UV"], br.inputs["Vector"])
nt.links.new(br.outputs["Fac"], rp.inputs["Fac"])
nt.links.new(rp.outputs["Color"], mx.inputs[6]); nt.links.new(im.outputs["Color"], mx.inputs[7]); nt.links.new(fv.outputs[0], mx.inputs[0])
nt.links.new(mx.outputs[2], em.inputs["Color"]); nt.links.new(em.outputs[0], o.inputs["Surface"])
A.mats["MAT_fab_test_screen"] = smat
sy_ = -250 - 110 - 2.0
scr = A.obj_from_mesh("fau_attach_station_screen", [((340 + 100) * MM, sy_ * MM, (ZT + 50) * MM), ((340 - 100) * MM, sy_ * MM, (ZT + 50) * MM),
                      ((340 - 100) * MM, sy_ * MM, (ZT + 105) * MM), ((340 + 100) * MM, sy_ * MM, (ZT + 105) * MM)], [(0, 1, 2, 3)], smat, smooth=False)
uv = scr.data.uv_layers.new(name="UVMap")
for li, c in zip(scr.data.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
    uv.data[li].uv = c
A.hook("meter_screen", (340, -362, ZT + 78), A.root, size=0.03)
# monitor
A.box("fau_attach_station_monitor_arm", (30, 30, 300), (-400, 260, ZT), "black_anodized", anchor="b", bev=2)
A.box("fau_attach_station_monitor", (360, 24, 220), (-400, 240, ZT + 280), "black_plastic", anchor="b", bev=4)
A.box("fau_attach_station_monitor_screen", (330, 2, 190), (-400, 227, ZT + 295), A.custom_mat("fau_monitor", (0.02, 0.05, 0.08), emit=(0.1, 0.35, 0.5), emit_strength=0.5), anchor="b")
A.preview_fit = 1.0
meta = {"description": "FAU active-alignment and UV-attach bench (generic): PIC chuck, XYZ+tilt FAU positioner, dispenser, UV lamp, microscope, power meter.",
        "origin": "floor centre; PIC at x = 0; FAU approaches from -X",
        "scene_usage": "S5 'FAU ATTACH' station: animate p_fau_x toward the PIC, p_dispense down and up, p_uv on.",
        "simplifications": ["Fibre ribbon is rigid and moves with the stage only for the first segment; the far end does not flex", "Stage internals are blocks", "Generic geometry; no vendor hexapod copied"],
        "screen": "fau_attach_station_screen with MAT_fab_test_screen: IMG_screen left empty (set MIX_use_image to 1 after plugging the sequence)"}
A.finish(OUT, meta, views=("front", "three_quarter", "top"),
         closeups=[("joint_close", (-30, -35, ZP + 12), (-5, 0, ZP), 60)])
