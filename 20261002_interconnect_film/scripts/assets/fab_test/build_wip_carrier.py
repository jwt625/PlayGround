"""wip_carrier: work-in-progress carrier (tray with 2 x 3 substrate pockets) with six substrates, and a stack option via p_stack.
Origin: bottom centre; footprint 520 x 380 mm.
Run: Blender -b --python scripts/assets/fab_test/build_wip_carrier.py -- assets/components/fab_test
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, line_lib as LL  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("wip_carrier", accuracy="C")
A.src("Generic SMT/OSAT process carrier (anodized aluminium tray with pockets); dimensions are estimates", "n/a", "concept")
A.dim("tray L x W x H", "520 x 380 x 12", "mm", "estimate", "C")
A.dim("pocket", "160 x 150, 2 x 3", "mm", "estimate", "C")
tray = A.box("wip_carrier_tray", (520, 380, 12), (0, 0, 0), "blue_anodized", anchor="b", bev=3)
for ix in range(3):
    for iy in range(2):
        x, y = (ix - 1) * 165, (iy - 0.5) * 170
        A.box("wip_carrier_pocket_frame", (172, 158, 3), (x, y, 12), "black_anodized", anchor="b", bev=0.8)
        LL.make_board(A, "wip_carrier_sub_%d%d" % (ix, iy), x, y, 14.5, size=(150, 140), pkg=True)
for sx in (-1, 1):
    A.box("wip_carrier_handle", (30, 120, 14), (sx * 275, 0, 0), "steel", anchor="b", bev=3)
A.hook("carrier_top", (0, 0, 14.5), A.root, size=0.05)
for ix in range(3):
    for iy in range(2):
        A.hook("pocket_%d%d" % (ix, iy), ((ix - 1) * 165, (iy - 0.5) * 170, 14.5), A.root, size=0.015)
A.preview_fit = 0.8
meta = {"description": "WIP carrier tray with six substrate boards.", "origin": "bottom centre of the tray",
        "scene_usage": "rides on the conveyor modules between stations", "simplifications": ["Generic geometry"]}
A.finish(OUT, meta, views=("three_quarter", "top"))
