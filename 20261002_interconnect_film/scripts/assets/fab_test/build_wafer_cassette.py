"""wafer_cassette: open 25-slot 300 mm wafer cassette (H-bar style, generic) with 25 wafer objects.
Origin: bottom centre of footprint; open side (loading side) is -Y.
Run: Blender -b --python scripts/assets/fab_test/build_wafer_cassette.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, wafer_lib as W  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("wafer_cassette", accuracy="C")
A.src("Supsemi 12 inch wafer cassette listing and general SEMI cassette practice (pitch 10 mm, 25 slots); dimensions are generic estimates",
      "https://www.supsemi.com/en/product/Semiconductor-Lab-Wafer-Frame-Box-Cassette-6-8-12-inches.html", "concept only")
A.dim("overall W x D x H", "335 x 330 x 300", "mm", "estimate (wafer 300 mm + side walls, 25 slots at 10 mm)", "C")
A.dim("slot pitch / count", "10 / 25", "mm / -", "common 300 mm pitch (FOUP spec ePAK)", "B")
W_, D_, H_ = 335.0, 330.0, 300.0
white = "white_plastic"
A.box("wafer_cassette_base", (W_, D_, 6), (0, 0, 0), "grey_plastic", anchor="b", bev=1)
A.box("wafer_cassette_top_bar", (W_, 20, 8), (0, D_ / 2 - 10, H_ - 8), "grey_plastic", anchor="b", bev=1)
for sx in (-1, 1):
    A.box("wafer_cassette_side_wall", (6, D_, H_ - 6), (sx * (W_ / 2 - 3), 0, 6), white, anchor="b", bev=1)
    A.box("wafer_cassette_rear_post", (14, 14, H_ - 6), (sx * (W_ / 2 - 12), D_ / 2 - 7, 6), "grey_plastic", anchor="b", bev=1)
A.box("wafer_cassette_rear_bar", (W_, 8, 40), (0, D_ / 2 - 4, 6), "grey_plastic", anchor="b", bev=1)
A.box("wafer_cassette_handle", (120, 12, 12), (0, D_ / 2 + 12, H_ - 26), "orange", anchor="b", bev=3)
rib = A.box("wafer_cassette_rib_proto", (12, 280, 2.5), (0, 0, 0), white, bev=0.3)
rib.hide_render = rib.hide_viewport = True
z_first = 20.0
for sx in (-1, 1):
    for k in range(25):
        A.dup(rib, "wafer_cassette_rib", (sx * (W_ / 2 - 6 - 6), -5, z_first + k * 10 - 1.25))
mat = A.mat("silicon")
w1 = W.disc(A, "wafer_cassette_wafer_01", mat, z0=z_first, N=240, center=(0, 0))
for k in range(1, 25):
    A.dup(w1, "wafer_cassette_wafer_%02d" % (k + 1), (0, 0, z_first + k * 10 + W.T_WAFER / 2))
for k in range(25):
    A.hook("wafer_slot_%02d" % (k + 1), (0, 0, z_first + k * 10), A.root, size=0.02)
A.preview_fit = 0.9
meta = {"description": "Open 300 mm wafer cassette (generic) with 25 wafers.", "origin": "bottom centre; wafers load/unload toward -Y",
        "simplifications": ["Generic geometry; real cassette dimensions per SEMI vary by vendor", "Wafers are reduced 240-segment discs, blank"]}
A.finish(OUT, meta, views=("front", "three_quarter", "top"))
