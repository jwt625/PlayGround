"""conveyor_modules: three short line modules in a row (straight 800 mm, stopper/buffer with sensor tower, transfer lift) plus a
board on the first module. Flow +X, front -Y, origin = centre of the row footprint on the floor.
Run: Blender -b --python scripts/assets/fab_test/build_conveyor_modules.py -- assets/components/fab_test
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, line_lib as LL  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("conveyor_modules", accuracy="C")
A.src("SMEMA-style SMT board conveyors (generic): belt-top height 950 mm, 600 mm rail spacing for boards up to 560 mm (Heller max PCB width)", "https://smtnet.com/company/index.cfm?fuseaction=view_company&company_id=59090&component=catalog&catalog_id=271835", "width")
A.dim("module length", 800, "mm", "generic modular conveyor, estimate", "C")
A.dim("belt-top height / rail spacing", "950 / 600", "mm", "SMEMA-style typical; Heller max board 560 mm", "B")
ZB = 950.0
A.prop("p_board_x", 0.0, "board position along the row (mm from the row start x = -1200)", 0.0, 2400.0, "mm")
A.prop("p_stopper", 0.0, "stopper pin raised (0..1) on module 2", 0.0, 1.0)
LL.belt_conveyor(A, "conveyor_modules_m1", -1200, -400, 0, ZB)
LL.belt_conveyor(A, "conveyor_modules_m2", -400, 400, 0, ZB)
LL.belt_conveyor(A, "conveyor_modules_m3", 400, 1200, 0, ZB)
# module 2: stopper and sensor tower
HS = A.hook("stopper", (250, 0, ZB), A.root, size=0.04)
A.box("conveyor_modules_stopper_pin", (12, 60, 40), (250, 0, ZB), "red", anchor="b", bev=1.5, parent=HS)
A.drive(HS, "location", 2, "%.6f+0.025*p_stopper-0.025" % (ZB * ft.MM), ["p_stopper"])
A.box("conveyor_modules_sensor_post", (30, 30, 180), (150, -330, ZB), "black_anodized", anchor="b", bev=2)
A.box("conveyor_modules_sensor_head", (60, 40, 40), (150, -330, ZB + 180), "blue_anodized", anchor="b", bev=3)
A.cyl("conveyor_modules_sensor_lamp", 10, 20, (150, -330, ZB + 220), "green_led", anchor="b", seg=20)
# module 3: transfer lift cylinder
A.box("conveyor_modules_lift_block", (300, 360, 70), (800, 0, ZB - 160), "black_anodized", anchor="b", bev=3)
for sx in (-1, 1):
    A.cyl("conveyor_modules_lift_rod", 12, 80, (800 + sx * 100, 0, ZB - 90), "chrome", anchor="b", seg=20)
A.box("conveyor_modules_control_box", (240, 160, 300), (-800, -370, 500), "black_paint", anchor="b", bev=3)
HB = A.hook("board", (-1200, 0, ZB), A.root, size=0.1)
A.drive(HB, "location", 0, "-1.2+p_board_x*0.001", ["p_board_x"])
LL.make_board(A, "conveyor_modules_board", -1000, 0, ZB, parent=HB)
A.hook("row_start", (-1200, 0, ZB), A.root)
A.hook("row_end", (1200, 0, ZB), A.root)
A.preview_fit = 1.0
meta = {"description": "Three 800 mm board-conveyor modules (straight, stopper/sensor, transfer lift) with a board; origin floor centre; +X flow.",
        "scene_usage": "Between line stations in S5 reverse trip, board carried along the line.",
        "simplifications": ["Rollers and drive internals not modelled", "Generic dimensions"]}
A.finish(OUT, meta, views=("front", "three_quarter", "top"))
