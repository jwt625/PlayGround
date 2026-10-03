"""eic_pic_bonding_close: close-up of an EIC held by a bond collet above a PIC on a substrate (true size, mm-scale).
Origin: substrate underside centre (z = 0). Hooks: HOOK_eic (moves with p_gap), HOOK_collet_tip, HOOK_bond_point.
Run: Blender -b --python scripts/assets/fab_test/build_eic_pic_bonding_close.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("eic_pic_bonding_close", accuracy="C")
MM = ft.MM
A.src("Project PIC die about 7 x 9 mm (storyboard); EIC 5 x 6 mm is an estimate", "DevLog/DevLog-001-story-scenes-assets-proposal.md", "die sizes")
A.src("Wentao's flip-chip bonder photo (tool, collet, chuck)", "/Users/wentaojiang/Documents/GitHub/jwt625.github.io/assets/images/2025/20251219_CPO/flip-chip-bonder-wafer.webp", "tool shape")
A.dim("substrate", "60 x 60 x 1.6", "mm", "estimate", "C")
A.dim("PIC", "9 x 7 x 0.775 (x along 9)", "mm", "project PIC ~7 x 9 mm, 775 um silicon", "B")
A.dim("EIC", "6 x 5 x 0.3", "mm", "estimate", "C")
A.dim("microbump pitch (scaled up)", 0.2, "mm", "real 25-55 um pitch; exaggerated so bumps are visible, 24 x 29 array", "C")
A.dim("collet", "tip 6.4 x 5.4 mm, taper to 4 mm radius body", "mm", "estimate", "C")
A.prop("p_gap", 3.0, "EIC / collet height above its bonded position (mm), 0 = bonded", 0.0, 10.0, "mm")
A.prop("p_heat", 0.0, "bond tool heater glow (0..1)", 0.0, 1.0)
heat = A.custom_mat("heater_glow", (0.05, 0.03, 0.02), emit=(1.0, 0.3, 0.05), emit_strength=0.0)
b = heat.node_tree.nodes["Principled BSDF"]
fc = b.inputs["Emission Strength"].driver_add("default_value")
fc.driver.type = "SCRIPTED"; fc.driver.expression = "p_heat*2.0"
v = fc.driver.variables.new(); v.name = "p_heat"; v.type = "SINGLE_PROP"; v.targets[0].id = A.root; v.targets[0].data_path = '["p_heat"]'
A.box("eic_pic_bonding_close_substrate", (60, 60, 1.6), (0, 0, 0), "pcb_green", anchor="b", bev=0.3)
for k in range(40):  # substrate pads ring
    a = 2 * math.pi * k / 40
    A.box("eic_pic_bonding_close_substrate_pad", (1.2, 0.6, 0.05), (14 * math.cos(a) * 1.2, 11 * math.sin(a) * 1.2, 1.6), "gold", anchor="b", rot_z=a)
A.box("eic_pic_bonding_close_pic", (9, 7, 0.775), (0, 0, 1.6), "silicon", anchor="b")
A.box("eic_pic_bonding_close_pic_film", (9, 7, 0.02), (0, 0, 2.375), "pcb_blue", anchor="b")
pad = A.box("eic_pic_bonding_close_pic_pad", (0.3, 0.3, 0.02), (-3.9, 0, 2.395), "gold", anchor="b")
for k in range(1, 12):
    A.dup(pad, "eic_pic_bonding_close_pic_pad", (-3.9 + 0.0, -3.0 + k * 0.5, 2.395 + 0.01))
# PIC bump pads under the EIC footprint (copper) on the film
cu = A.cyl("eic_pic_bonding_close_ubump_proto", 0.05, 0.05, (0, 0, 2.395), "copper", anchor="b", seg=10)
cu.hide_render = cu.hide_viewport = True
HE = A.hook("eic", (0, 0, 2.395), A.root, size=0.004)
A.drive(HE, "location", 2, "%.7f+p_gap*0.001" % (2.395 * MM), ["p_gap"])
A.box("eic_pic_bonding_close_eic", (6, 5, 0.3), (0, 0, 2.395 + 0.05), "silicon", anchor="b", parent=HE)
A.box("eic_pic_bonding_close_eic_film", (6, 5, 0.02), (0, 0, 2.395 + 0.35), "black_plastic", anchor="b", parent=HE)
ub = A.cyl("eic_pic_bonding_close_ubump", 0.045, 0.05, (0, 0, 2.395), "copper", anchor="b", seg=8, parent=HE)
nx, ny = 24, 22
first = True
for i in range(nx):
    for j in range(ny):
        x, y = (i - (nx - 1) / 2) * 0.2, (j - (ny - 1) / 2) * 0.2
        if first:
            A.parts["eic_pic_bonding_close_ubump"].location = (x * MM, y * MM, (2.395 + 0.025) * MM - HE.location.z)
            first = False
        else:
            A.dup(ub, "eic_pic_bonding_close_ubump", (x, y, 2.395 + 0.025), parent=HE)
# collet and bond tool above the EIC (moves with the EIC)
A.box("eic_pic_bonding_close_collet_pad", (6.4, 5.4, 0.6), (0, 0, 2.395 + 0.37), "steel", anchor="b", parent=HE)
A.cyl("eic_pic_bonding_close_collet", 3.6, 7, (0, 0, 2.395 + 0.97), "ceramic", anchor="b", parent=HE, seg=32, r_top=7)
A.cyl("eic_pic_bonding_close_tool_heater", 9, 6, (0, 0, 2.395 + 7.97), heat, anchor="b", parent=HE, seg=32, bev=0.4)
A.cyl("eic_pic_bonding_close_tool_body", 11, 10, (0, 0, 2.395 + 13.97), "black_anodized", anchor="b", parent=HE, seg=32, bev=0.6)
A.tube_path("eic_pic_bonding_close_vacuum_tube", [(0, 0, 2.395 + 24), (10, 0, 2.395 + 30), (25, 0, 2.395 + 30)], 1.5, "rubber", seg=10, parent=HE)
A.hook("collet_tip", (0, 0, 2.395 + 0.37), HE, size=0.003)
A.hook("bond_point", (0, 0, 2.395), A.root, size=0.003)
A.preview_fit = 1.6
A.preview_shadows = True
meta = {"description": "EIC above a PIC on a substrate, held by a bond collet and heated tool; gap driven by p_gap.",
        "origin": "centre of the substrate underside (z = 0)",
        "scene_usage": "S5 'EIC / PIC BONDING' close-up; animate p_gap 3 -> 0 then p_heat up.",
        "simplifications": ["Microbump pitch exaggerated to 0.2 mm", "EIC bump array sits on the EIC underside; the PIC has only edge pads", "Tool body is generic"]}
A.finish(OUT, meta, views=("three_quarter", "top"),
         closeups=[("bumps_close", (4.0, -14.0, 3.4), (0, 0, 2.9), 85)])
