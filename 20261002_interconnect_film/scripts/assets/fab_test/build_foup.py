"""foup: 300 mm FOUP (SEMI E47.1 style) with 25 wafer slots at 10 mm pitch, door (p_door_open) and 25 wafer objects.
Origin: bottom centre of the footprint (z = 0 = underside); front (door) is -Y.
Run: Blender -b --python scripts/assets/fab_test/build_foup.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, wafer_lib as W  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("foup", accuracy="B")
MM = ft.MM
A.src("ePAK eFOUP300 product page: 420 x 342 x 338 mm with handle and flange; 388 x 342 x 332 mm without; 25 wafers; 10 mm pitch (+-0.5); first wafer 44 mm from datum; SEMI E47.1 / E57 kinematic coupling",
      "https://www.epak.com/products/efoup-wafer-handling/", "overall dimensions, slot pitch, first slot height")
A.src("SEMI E47.1 mechanical specification for FOUPs (summary via store listing)", "https://store-us.semi.org/products/e04701-semi-e47-1-mechanical-specification-for-foups-used-to-transport-and-store-300-mm-wafers", "standard reference")
A.dim("body W x D x H (no handles)", "388 x 342 x 332", "mm", "ePAK eFOUP300", "B")
A.dim("with handles + flange W x H", "420 x 338", "mm", "ePAK eFOUP300 (H listed 338 mm)", "B")
A.dim("slot pitch / count", "10 / 25", "mm / -", "ePAK; SEMI E47.1", "A")
A.dim("first wafer height above datum", 44, "mm", "ePAK", "B")
A.dim("wall thickness, door opening, door size", "5 / 346 x 296 / 340 x 290", "mm", "estimates", "C")
W_, D_, H_ = 388.0, 342.0, 332.0
t = 5.0
shell = "foup_poly_clear"
A.prop("p_door_open", 0.0, "0 = door closed, 1 = door slid out 250 mm toward -Y and 30 mm down (open)", 0.0, 1.0)
y0 = -D_ / 2
A.box("foup_bottom", (W_, D_, t), (0, 0, 0), shell, anchor="b", bev=1)
A.box("foup_top", (W_, D_, t), (0, 0, H_ - t), shell, anchor="b", bev=1)
A.box("foup_back", (W_, t, H_), (0, D_ / 2 - t / 2, 0), shell, anchor="b", bev=1)
for sx in (-1, 1):
    A.box("foup_side", (t, D_, H_), (sx * (W_ / 2 - t / 2), 0, 0), shell, anchor="b", bev=1)
# front frame around the opening (opening 346 x 296, centred at z = 166)
fz = 166.0
for (sx, sz, w, h) in ((0, fz + 148 + 10, W_, 20), (0, fz - 148 - 10, W_, 20), (-1, fz, 21, 296), (1, fz, 21, 296)):
    X = sx * (173 + 10.5) if sx else 0
    A.box("foup_front_frame", (w, 16, h), (X, y0 + 8, sz), "black_plastic", bev=1.5)
# slot ribs (instanced) on both inner walls: 25 per side
rib = A.box("foup_slot_rib_proto", (8, 300, 3), (0, 0, 0), "white_plastic", bev=0.3)
rib.hide_render = rib.hide_viewport = True
wafers = []
z_first = 44.0
for sx in (-1, 1):
    for k in range(25):
        A.dup(rib, "foup_slot_rib", (sx * (W_ / 2 - t - 4), 12, z_first + k * 10 - 1.5))
# door
HD = A.hook("door", (0, y0 - 5, fz), A.root, size=0.08)
A.box("foup_door", (340, 12, 290), (0, y0 - 5, fz), "black_plastic", parent=HD, bev=2)
for sx in (-1, 1):
    A.cyl("foup_door_keyhole", 11, 2, (sx * 110, y0 - 12, fz), "steel_dark", axis="y", parent=HD, seg=24)
    A.box("foup_door_keyslot", (3, 1, 14), (sx * 110, y0 - 13.2, fz), "black_paint", parent=HD)
A.drive(HD, "location", 1, "%.6f-0.25*p_door_open" % ((y0 - 5) * MM), ["p_door_open"])
A.drive(HD, "location", 2, "%.6f-0.03*p_door_open" % (fz * MM), ["p_door_open"])
# robotic flange (top) and side handles
A.box("foup_top_flange_plate", (130, 100, 6), (0, 0, H_), "orange", anchor="b", bev=1)
for sx in (-1, 1):
    A.box("foup_top_flange_rail", (14, 100, 5), (sx * 34, 0, H_ + 6), "orange", anchor="b", bev=1)
    A.box("foup_handle_bar", (16, 110, 34), (sx * (W_ / 2 + 8), 6, 150), "orange", anchor="b", bev=3)
# kinematic coupling grooves (3 pads on the underside)
for (x, y) in ((0, 100), (-80, -90), (80, -90)):
    A.box("foup_kinematic_pad", (30, 40, 4), (x, y, -4), "steel_dark", anchor="b", bev=1)
# wafers: first real, others linked duplicates
mat = A.mat("silicon")
w1 = W.disc(A, "foup_wafer_01", mat, z0=z_first, N=240, center=(0, 8))
wafers.append(w1)
for k in range(1, 25):
    wafers.append(A.dup(w1, "foup_wafer_%02d" % (k + 1), (0, 8, z_first + k * 10 + W.T_WAFER / 2)))
for k in range(25):
    A.hook("wafer_slot_%02d" % (k + 1), (0, 8, z_first + k * 10), A.root, size=0.02)
A.preview_fit = 0.9
A.preview_shadows = True
meta = {"description": "300 mm FOUP, translucent shell, 25 slots at 10 mm pitch, front door driven by p_door_open, 25 individually movable wafer objects (reduced 240-segment discs).",
        "origin": "bottom centre of footprint, z = 0 at the underside datum; door at -Y",
        "usage": "wafers can be lifted out individually (foup_wafer_NN, origin at bbox centre; HOOK_wafer_slot_NN = slot bottom). Wafer discs are blank (no die map); swap for wafer_300mm_siph if needed.",
        "simplifications": ["Shell is plain boxes (no moulded features, latches, RFID, purge ports)", "Front frame and door are generic", "Slot comb ribs are simplified"]}
A.finish(OUT, meta, views=("front", "three_quarter", "top"),
         closeups=[("slots_close", (-200, -260, 200), (-120, -60, 120), 60)])
