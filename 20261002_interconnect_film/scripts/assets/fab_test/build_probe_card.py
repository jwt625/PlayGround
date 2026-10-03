"""probe_card: vertical (MEMS-style) 300 mm probe card, operating pose (needles down).
Origin: centre of the needle-tip plane (z = 0 at the tips); card extends upward. Mount: tips touch the wafer top at HOOK_probe_center.
Variants: VARIANT_full_needles (4 x 4 sites x 24 needles = 384) and VARIANT_light (1 site, 24 needles).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("probe_card", accuracy="C")
A.src("Vertical/MEMS probe card construction (PCB + stiffener + space transformer + guide plates + needles), generic from public vendor descriptions",
      "https://www.formfactor.com/ , https://www.technoprobe.com/ (concept only, no dimensions copied)", "construction concept")
A.src("Die pad layout of wafer_300mm_siph (12 pads per row, two rows, pitch 0.5455 mm; die 7 x 9 mm; site pitch 7.1 x 9.1 mm)",
      "scripts/assets/fab_test/wafer_lib.py", "needle positions match the film's own die pads")
A.dim("PCB diameter / thickness", "410 / 6", "mm", "estimate: 300 mm probe cards use roughly 400-450 mm PCBs", "C")
A.dim("stiffener OD / ID / height", "330 / 250 / 12", "mm", "estimate", "C")
A.dim("PCB window", "40 x 50", "mm", "estimate for the 4 x 4 site head", "C")
A.dim("space transformer", "52 x 62 x 3", "mm", "estimate", "C")
A.dim("needle length / radius", "6.5 / 0.04", "mm", "typical vertical MEMS probe, estimate", "C")
A.dim("needle pitch within a site", 0.5455, "mm", "matches wafer_300mm_siph die pads (not a real probe-card limit; real fine pitch down to ~0.05 mm)", "B")
A.dim("sites", "4 x 4 at 7.1 x 9.1 mm", "mm", "wafer_300mm_siph die pitch", "B")

ZW = 9.5   # PCB bottom
pcb = A.cyl("probe_card_pcb", 205, 6, (0, 0, ZW), "pcb_green", anchor="b", seg=128, bev=0.6)
win = ft.C.cube("cutter", (40 * ft.MM, 50 * ft.MM, 20 * ft.MM), loc=(0, 0, (ZW + 3) * ft.MM))
ft.C.add(win, A.coll, A.root)
A.cut(pcb, [win])
A.cyl("probe_card_pcb_edge_gold", 205.2, 1.0, (0, 0, ZW + 2.5), "gold", anchor="b", seg=128)
A.cyl("probe_card_stiffener", 165, 12, (0, 0, ZW + 6), "steel", anchor="b", seg=128, r_in=125, bev=0.8)
A.cyl("probe_card_stiffener_inner_lip", 126, 4, (0, 0, ZW + 6), "silver", anchor="b", seg=96, r_in=118)
A.box("probe_card_space_transformer", (52, 62, 3), (0, 0, ZW - 3), "ceramic", anchor="b", bev=0.3)
A.box("probe_card_head_frame", (60, 70, 2), (0, 0, ZW - 5), "steel_dark", anchor="b", bev=0.3)
for sx in (-1, 1):
    for sy in (-1, 1):
        A.cyl("probe_card_head_screw", 1.6, 2.5, (sx * 27, sy * 32, ZW - 5), "steel", anchor="b", seg=12)
A.box("probe_card_guide_upper", (32, 40, 0.4), (0, 0, 5.0), "glass_tint", anchor="b")
A.box("probe_card_guide_lower", (32, 40, 0.4), (0, 0, 1.0), "glass_tint", anchor="b")
for (bx, by, bsx, bsy) in ((0, 20.5, 34, 1.0), (16.5, 0, 1.0, 42), (-16.5, 0, 1.0, 42)):
    A.box("probe_card_guide_spacer_wall", (bsx, bsy, 3.4), (bx, by, 1.4), "steel_dark", anchor="b")
# peripheral contact pads (instanced)
pad = A.box("probe_card_ring_pad_proto", (5, 2.4, 0.12), (190, 0, ZW + 6), "gold", anchor="b")
pad.hide_render = pad.hide_viewport = True
for k in range(120):
    a = 2 * math.pi * k / 120
    A.dup(pad, "probe_card_ring_pad", (178 * math.cos(a), 178 * math.sin(a), ZW + 6), rot_z=a)
# mounting screws on the stiffener
scr = A.cyl("probe_card_screw_proto", 3.0, 3.0, (0, 0, ZW + 18), "steel_dark", anchor="b", seg=16, bev=0.4)
scr.hide_render = scr.hide_viewport = True
for k in range(12):
    a = 2 * math.pi * (k + 0.5) / 12
    A.dup(scr, "probe_card_stiffener_screw", (145 * math.cos(a), 145 * math.sin(a), ZW + 18))
# coax connectors on the PCB rim
for k in range(6):
    a = math.radians(30 + 60 * k)
    x, y = 185 * math.cos(a), 185 * math.sin(a)
    A.cyl("probe_card_coax_body", 4.5, 10, (x, y, ZW + 6), "silver", anchor="b", seg=24, bev=0.4)
    A.cyl("probe_card_coax_dielectric", 2.2, 1.5, (x, y, ZW + 16), "white_plastic", anchor="b", seg=16)
    A.cyl("probe_card_coax_pin", 0.6, 3, (x, y, ZW + 17.5), "gold", anchor="b", seg=8)
# tester-side connector blocks (4 on the rim)
for k in range(4):
    a = math.radians(90 * k + 0)
    A.box("probe_card_edge_connector", (18, 40, 8), (193 * math.cos(a), 193 * math.sin(a), ZW + 6), "black_plastic", anchor="b", bev=0.8, rot_z=a)
# silk label
A.text("probe_card_silk_text", "VPC-300  4x4x24N", 7, (0, -105, ZW + 6.05), "label_white", rot=(0, 0, 0), extrude=0.05)
A.text("probe_card_silk_text2", "GENERIC TEST CARD", 4, (0, -115, ZW + 6.05), "label_white", rot=(0, 0, 0), extrude=0.05)
# needles: 4 x 4 sites x 2 rows x 12 pads
PX = 7.0 - 1.0
pad_x = [-PX / 2 + k * PX / 11 for k in range(12)]
pad_y = [4.5 - 0.35, -4.5 + 0.35]
def needle_path(x, y):
    return [(x, y, 6.5), (x, y, 4.2), (x + 0.15, y, 2.8), (x, y, 0.6), (x, y, 0.0)]


# simple explicit placement: build one needle at origin, then linked duplicates offset by (x, y)
def place_needles(coll, sites, tag):
    base = A.tube_path("probe_card_needle_%s" % tag, needle_path(0.0, 0.0), [0.04, 0.04, 0.04, 0.03, 0.01], "tungsten", seg=6, coll=coll)
    # base mesh is recentred; use its mesh with explicit offsets (needle spans z 0..6.5, centre z 3.25, x offset 0.0375)
    cz = base.location.z
    cx = base.location.x
    n = 0
    for (sx, sy) in sites:
        for ry in pad_y:
            for px in pad_x:
                x, y = sx + px, sy + ry
                if n == 0:
                    base.location = (x * ft.MM + 0.0, y * ft.MM, cz)
                    base.location.x = x * ft.MM + cx
                else:
                    o = bpy.data.objects.new("probe_card_needle_%s" % tag, base.data)
                    coll.objects.link(o)
                    o.parent = A.root
                    o.location = (x * ft.MM + cx, y * ft.MM, cz)
                n += 1
    return n


vfull = A.variant("full_needles")
vlight = A.variant("light")
sites_full = [((i - 1.5) * 7.1, (j - 1.5) * 9.1) for j in range(4) for i in range(4)]
n_full = place_needles(vfull, sites_full, "full")
n_light = place_needles(vlight, [(0.0, 0.0)], "light")
A.hook("probe_center", (0, 0, 0), A.root, size=0.03)
A.hook("site_origin_lower_left", (sites_full[0][0], sites_full[0][1], 0), A.root, size=0.01)
A.hook("pcb_top", (0, 0, ZW + 6), A.root, size=0.03)
A.prop("p_overdrive_um", 0.0, "documentation only: overdrive 0..100 um; assembler moves the card or the stage by this amount", 0.0, 100.0, "um")
A.preview_fit = 1.0
A.preview_target_off = (0, 0, 0.012)
A.preview_shadows = False
meta = {"description": "Vertical probe card (400+ mm PCB, stiffener ring, central window, space transformer, guide plates, tungsten-style needles).",
        "origin": "centre of the needle-tip plane (z = 0 at the tips, card extends +Z). Mount: put HOOK_probe_center on the probe station's HOOK_probe_center, tips touching the wafer top (overdrive by moving z)",
        "scene_usage": "Optional mounting above the wafer in S5 close-ups of the probe tips; the CM300-style station itself uses four positioner needles.",
        "variants": {"VARIANT_full_needles": "%d needles, 4 x 4 sites x 24, matching wafer_300mm_siph die pads" % n_full,
                     "VARIANT_light": "%d needles, single site" % n_light, "note": "hide the unused variant collection (the card body is shared); no cantilever variant built"},
        "simplifications": ["No traces or SMD parts on the PCB", "Needles are straight-ish tubes with a small bend, not real MEMS geometry", "Pogo/interposer between PCB and space transformer omitted"]}
A.finish(OUT, meta, views=("three_quarter", "top"),
         closeups=[("needles_close", (6, -26, 2.6), (0, -2, 2.4), 70), ("window_close", (0, -140, 120), (0, 0, 8), 60)])
