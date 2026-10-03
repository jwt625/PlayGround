"""wafer_300mm: blank 300 mm mirror-silicon wafer (SEMI M1 style): notch at 6 o'clock, bevelled edge, laser mark.
Run: Blender -b --python scripts/assets/fab_test/build_wafer_300mm.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, wafer_lib as W  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("wafer_300mm", accuracy="A")
A.src("SEMI M1 polished single-crystal silicon wafers (public summaries: 300 mm diameter, 775 um thickness, notch 1.0 mm deep, 90 deg)",
      "https://en.wikipedia.org/wiki/Wafer_(electronics)", "diameter 300 mm, thickness 775 um, notch geometry")
A.dim("diameter", 300, "mm", "SEMI M1 / standard 300 mm wafer", "A")
A.dim("thickness", 0.775, "mm", "SEMI M1 nominal 775 um", "A")
A.dim("notch depth / included angle", "1.0 / 90", "mm / deg", "SEMI M1 notch (V notch at 6 o'clock = -Y)", "A")
A.dim("edge chamfer", "0.30 x 0.25 + round", "mm", "typical bevel, estimated", "C")
A.dim("laser mark", "NC-SIPH-001, 2 mm characters", "text", "generic placeholder text", "C")
mat = A.mat("silicon")
disc = W.disc(A, "wafer_300mm_disc", mat)
A.text("wafer_300mm_lasermark", "NC-BLANK-001", 2.0, (0, -139, W.T_WAFER + 0.01), "steel_dark")
A.hook("wafer_center_top", (0, 0, W.T_WAFER), A.root, size=0.05)
A.hook("wafer_notch", (0, -W.R_WAFER, W.T_WAFER / 2), A.root, size=0.02)
A.hook("wafer_bottom_center", (0, 0, 0), A.root, size=0.03)
A.preview_fit = 1.15
A.preview_shadows = False
meta = {"description": "Blank 300 mm silicon wafer, mirror finish, notch at -Y (front). Origin: bottom centre (sits on z = 0).",
        "scene_usage": "Flies in/out of the probe station drawer, sits on the chuck at HOOK_wafer_slot (identity parent transform); also used standalone in the FOUP/cassette.",
        "origin": "bottom centre of the wafer (z = 0 is the back surface)",
        "simplifications": ["Edge profile is a simple chamfer/round (no SEMI T-type edge exactness)", "Mirror look is a metallic principled shader; no thin-film iridescence"]}
A.finish(OUT, meta, views=("three_quarter", "top"),
         closeups=[("notch_edge", (0, -190, 28), (0, -149, 0.4), 90), ("edge_close", (95, -150, 8), (70, -132, 0.4), 80)])
