"""reflow_oven_line: 10-zone convection reflow oven (8 heating + 2 cooling), conveyor, entry/exit tunnels, exhausts, N2 ports,
with a substrate board on the conveyor. Flow direction +X, front -Y, origin floor centre.
Run: Blender -b --python scripts/assets/fab_test/build_reflow_oven_line.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, line_lib as LL  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("reflow_oven_line", accuracy="B")
MM = ft.MM
A.src("Heller 1936 MK5 reflow oven listing: 590 x 152 x 160 cm, 10 heating zones (top and bottom) + 3 cooling zones, max PCB width 560 mm, N2 version", "https://smtnet.com/company/index.cfm?fuseaction=view_company&company_id=59090&component=catalog&catalog_id=271835", "overall envelope, zone concept")
A.src("Wentao's CPO assembly flow slide (reflow on substrate as first step of the reverse trip)", "/Users/wentaojiang/Documents/GitHub/jwt625.github.io/assets/images/2026/OFC2026/IMG_4084.JPG", "context")
A.dim("oven length (this model)", 4500, "mm", "task brief 3-5 m; Heller 1936 MK5 is 5.9 m", "B")
A.dim("oven width x height", "1400 x 1550", "mm", "scaled from Heller 1936 MK5 (1520 x 1600)", "B")
A.dim("zones", "8 heating + 2 cooling = 10 (zone length 400 mm)", "-", "task brief 8-12 zones", "B")
A.dim("conveyor belt-top height", 950, "mm", "SMEMA-style board conveyor height, typical", "B")
A.dim("conveyor width (rail to rail)", 600, "mm", "Heller max PCB 560 mm", "B")
A.dim("board", "300 x 220 x 1.6 + package", "mm", "generic substrate board", "C")
NZ, ZL = 10, 400.0
XL = 2250.0     # half length of the oven body
ZB = 950.0      # belt top
Wd = 1400.0
paint = A.mat("grey_plastic")
body_mat = A.custom_mat("oven_paint", (0.16, 0.19, 0.23), metallic=0.1, rough=0.5, coat=0.2)
lid_mat = A.custom_mat("oven_lid", (0.30, 0.33, 0.37), metallic=0.1, rough=0.45, coat=0.2)
A.prop("p_glow", 1.0, "glow multiplier of the 8 heating-zone slits (0..1)", 0.0, 1.0)
A.prop("p_glow_cool", 0.3, "glow multiplier of the 2 cooling-zone slits (0..1)", 0.0, 1.0)
A.prop("p_board_x", 0.0, "position of the board carriage along the line in mm from x = -2500 (0 .. 5000)", 0.0, 5000.0, "mm")

# lower body with through tunnel (cut)
body = A.box("reflow_oven_line_body", (2 * XL, Wd, 970), (0, 0, 100), body_mat, anchor="b", bev=3)
tun = ft.C.cube("cutter", (2 * XL * MM + 0.2, 640 * MM, 80 * MM), loc=(0, 0, (ZB + 10) * MM))
ft.C.add(tun, A.coll, A.root)
A.cut(body, [tun])
for sx in (-1, 1):
    for sy in (-1, 1):
        A.cyl("reflow_oven_line_foot", 40, 100, (sx * (XL - 120), sy * (Wd / 2 - 100), 0), "black_anodized", anchor="b", seg=24)
# entry/exit tunnel hoods
for sg, nm in ((-1, "entry"), (1, "exit")):
    hood = A.box("reflow_oven_line_%s_hood" % nm, (260, 780, 160), (sg * (XL + 130 - 20), 0, ZB - 45), lid_mat, anchor="b", bev=3)
    cutter = ft.C.cube("cutter", (300 * MM, 640 * MM, 80 * MM), loc=(sg * (XL + 130 - 20) * MM, 0, (ZB + 10) * MM))
    ft.C.add(cutter, A.coll, A.root)
    A.cut(hood, [cutter])
    A.box("reflow_oven_line_%s_curtain" % nm, (4, 650, 14), (sg * (XL + 262 - 20), 0, ZB + 60), "rubber", anchor="b")
# zone lids with glow slits
glow_names = []
for i in range(NZ):
    x = -XL + ZL * (i + 0.5)
    heat = i < 8
    A.box("reflow_oven_line_zone_%02d_lid" % (i + 1), (ZL - 4, Wd, 480), (x, 0, 1070), lid_mat, anchor="b", bev=3)
    mname = "oven_glow_%02d" % (i + 1)
    col = (1.0, 0.32, 0.05) if heat else (0.2, 0.6, 1.0)
    m = A.custom_mat(mname, (0.1, 0.1, 0.1), emit=col, emit_strength=1.5 if heat else 0.6)
    slit = A.box("reflow_oven_line_zone_%02d_glow" % (i + 1), (ZL - 60, 4, 26), (x, -Wd / 2 - 1, 1190), m, anchor="c", bev=0.5)
    # driver: emission strength follows p_glow / p_glow_cool
    b = m.node_tree.nodes["Principled BSDF"]
    fc = b.inputs["Emission Strength"].driver_add("default_value")
    d = fc.driver
    d.type = "SCRIPTED"
    pn = "p_glow" if heat else "p_glow_cool"
    d.expression = "%s*%s" % (pn, "1.5" if heat else "0.6")
    v = d.variables.new(); v.name = pn; v.type = "SINGLE_PROP"; v.targets[0].id = A.root; v.targets[0].data_path = '["%s"]' % pn
    glow_names.append("MAT_fab_test_" + mname)
    A.box("reflow_oven_line_zone_%02d_handle" % (i + 1), (140, 18, 14), (x, -Wd / 2 - 20, 1420), "silver", bev=3)
    A.cyl("reflow_oven_line_zone_%02d_fan" % (i + 1), 95 if heat else 140, 90, (x, 0, 1550), "steel_dark", anchor="b", seg=40, bev=2)
    A.cyl("reflow_oven_line_zone_%02d_fan_cap" % (i + 1), 60 if heat else 100, 20, (x, 0, 1640), "black_anodized", anchor="b", seg=32, bev=2)
# exhaust stacks (entry and exit zones) with elbows
for xs in (-XL + 130, XL - 130):
    A.cyl("reflow_oven_line_exhaust_stack", 70, 420, (xs, Wd / 2 - 250, 1550), "silver", anchor="b", seg=32, bev=2)
    A.cyl("reflow_oven_line_exhaust_collar", 85, 30, (xs, Wd / 2 - 250, 1560), "steel", anchor="b", seg=32)
    A.tube_path("reflow_oven_line_exhaust_duct", [(xs, Wd / 2 - 250, 1970), (xs, Wd / 2 - 250, 2050), (xs + 30, Wd / 2 - 100, 2120), (xs + 200, Wd / 2 + 150, 2120)], 70, "silver", seg=28)
# N2 ports and control cabinet at the front near the entry
for k in range(2):
    A.cyl("reflow_oven_line_n2_port", 18, 90, (-XL + 80 + k * 90, -Wd / 2 - 45, 600), "gold", axis="y", seg=20)
    A.cyl("reflow_oven_line_n2_valve", 30, 20, (-XL + 80 + k * 90, -Wd / 2 - 100, 600), "red", axis="y", seg=24, bev=2)
ctrl = A.box("reflow_oven_line_control_box", (420, 260, 420), (-XL + 260, -Wd / 2 - 140, 100), "black_paint", anchor="b", bev=4)
A.box("reflow_oven_line_control_screen_frame", (300, 14, 200), (-XL + 260, -Wd / 2 - 275, 400), "black_plastic", bev=3)
A.box("reflow_oven_line_status_lamp_g", (40, 40, 40), (-XL + 120, -Wd / 2 - 140, 520), "green_led", anchor="b", bev=3)
# conveyors (through the oven + extensions) and board
HB = A.hook("board", (-2500, 0, ZB), A.root, size=0.1)
A.drive(HB, "location", 0, "-2.5+p_board_x*0.001", ["p_board_x"])
LL.belt_conveyor(A, "reflow_oven_line_conveyor_in", -3000, -XL - 20, 0, ZB)
LL.belt_conveyor(A, "reflow_oven_line_conveyor_in_through", -XL - 20, XL + 20, 0, ZB, legs=False)
LL.belt_conveyor(A, "reflow_oven_line_conveyor_out", XL + 20, 3000, 0, ZB)
LL.make_board(A, "reflow_oven_line_board", -2500, 0, ZB, parent=HB)
A.hook("conveyor_start", (-3000, 0, ZB), A.root)
A.hook("conveyor_end", (3000, 0, ZB), A.root)
for i in range(NZ):
    A.hook("zone_%02d_center" % (i + 1), (-XL + ZL * (i + 0.5), 0, ZB + 40), A.root, size=0.04)
A.preview_fit = 1.2
meta = {"description": "10-zone convection reflow oven (8 heat + 2 cool), entry/exit hoods, conveyor with a substrate board.",
        "origin": "centre of the floor footprint of the oven body; +X is the conveyor flow direction",
        "emissive_materials": glow_names + ["driven by p_glow (heating, 8 W/m2-style strength 8) and p_glow_cool via drivers on Emission Strength; edit the material directly to override"],
        "board_motion": "HOOK_board moves along +X with p_board_x (mm); it is not occluded: the body hides it inside the tunnel by geometry",
        "scene_usage": "S5 REFLOW ON SUBSTRATE station: board enters from the left, glow slits pulse, board exits right.",
        "simplifications": ["Interior of the tunnel is empty (no nozzles or plenum)", "Control cabinet, screen frame and N2 ports are generic", "Zone lids are single boxes", "Real ovens are longer (Heller 1936 MK5: 5.9 m)"]}
A.finish(OUT, meta, views=("front", "three_quarter", "top"),
         closeups=[("entry_close", (-3300, -1100, 1500), (-2200, 0, 1000), 45), ("glow_close", (-600, -2300, 1500), (-200, -700, 1190), 40)])
