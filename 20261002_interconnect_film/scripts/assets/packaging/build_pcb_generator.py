"""Build pcb_generator demo: ASSET_pcb_generator holds (1) a 120 x 80 mm demo board and (2) the S2 compute-tray board (440 x 350 mm) with a backplane-connector row.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_pcb_generator.py -- assets/components/packaging
The reusable function is scripts/assets/packaging/pcb_gen.py (class PCB). Sub-collections VARIANT_demo_board (visible) and VARIANT_tray_board (hidden).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import pk  # noqa: E402
import pcb_gen  # noqa: E402
import parts  # noqa: E402
from pk import MB, MM, PI  # noqa: E402

C = pk.C
OUT = C.argv_after_dashes()[0]
AID = "pcb_generator"
C.reset()
pk.reset_mats()
coll, root = C.new_asset(AID, accuracy="B")
cd = C.sub_collection(coll, "VARIANT_demo_board")
ct = C.sub_collection(coll, "VARIANT_tray_board")
ct.hide_render = ct.hide_viewport = True
dims = pk.Dims()

# ------------------------------------------------------------------ demo board
b = pcb_gen.PCB(cd, root, "pcb_generator_demo", 120.0, 80.0, 1.6, "mask_green")
for (x, y) in ((-54, -34), (54, -34), (-54, 34), (54, 34)):
    b.hole(x, y, 3.2)
b.fiducials()
bga_c = (-22.0, 0.0)
b.pads_grid(bga_c[0], bga_c[1], 14, 14, 1.0, 0.5, skip=lambda i, j, x, y: abs(i - 6.5) < 2.6 and abs(j - 6.5) < 2.6)
b.outline(bga_c[0] - 8.2, bga_c[1] - 8.2, bga_c[0] + 8.2, bga_c[1] + 8.2)
# connector pad row on the right edge and 12 differential pairs routed with 45-degree bends
for k in range(24):
    b.pad_rect(49.0, -17.25 + k * 1.5, 2.0, 0.7)
for k in range(12):
    p0 = (bga_c[0] + 6.6, -5.5 + k * 1.0 - 0.0)
    p1 = (47.6, -17.25 + k * 3.0 + 0.75)
    path = pcb_gen.route45(p0, p1, first="x")
    b.diff_pair(path, 0.1, 0.13)
b.vias([(-22 + (i - 5.5) * 2.0, 11.5) for i in range(12)] + [(-22 + (i - 5.5) * 2.0, -11.5) for i in range(12)])
b.text("U1", -22.0, 10.0, 2.0)
b.text("J1", 42.0, 20.0, 2.0)
b.text("PCB GENERATOR DEMO  120 X 80", 0.0, -37.0, 1.6)
objs_d = b.finish()
counts_d = dict(b.counts)

# ------------------------------------------------------------------ S2 compute-tray board
TW, TD = 440.0, 350.0
t = pcb_gen.PCB(ct, root, "pcb_generator_tray", TW, TD, 2.4, "mask_green")
for x in (-205, -100, 0, 100, 205):
    for y in (-160, 160):
        t.hole(x, y, 3.2)
t.fiducials(6.0)
t.hole(-205, 0, 3.2)
t.hole(205, 0, 3.2)
SC = TW / 1.9 / 1.0
conn_x = [(k - 1.5) * 0.5 * SC for k in range(4)]
conn_y = 0.7 * SC / 1.0 - 0.1 * SC + 0.0        # rear row (crude y 0.7 relative to tray centre 0.1)
conn_y = (0.7 - 0.1) * SC
rows_y = [(0.42 - 0.1) * SC, (0.15 - 0.1) * SC, (-0.12 - 0.1) * SC]
hooks_conn, hooks_rt = [], []
for k, cx in enumerate(conn_x):
    # backplane connector pad field: 2 columns x 16 rect pads (1.2 x 0.6) at 2.0 mm pitch plus a shield-row
    for i in range(16):
        for col in (-1, 1):
            t.pad_rect(cx + (i - 7.5) * 4.0, conn_y + col * 1.5, 1.4, 0.7)
    t.outline(cx - 36.0, conn_y - 8.0, cx + 36.0, conn_y + 8.0, 0.2)
    t.text("J%d" % (k + 1), cx, conn_y - 11.0, 4.0)
    # retimer BGA footprints (17 x 17 at 0.8 mm, pad 0.4) in three rows, first row connected to the connector by 16 diff pairs
    for r, ry in enumerate(rows_y):
        t.pads_grid(cx, ry, 17, 17, 0.8, 0.4, skip=lambda i, j, x, y: (abs(i - 8) < 2.5 and abs(j - 8) < 2.5))
        t.outline(cx - 8.5, ry - 8.5, cx + 8.5, ry + 8.5, 0.2)
        t.text("U%d" % (k * 3 + r + 1), cx - 10.5, ry + 9.5, 2.2)
        hooks_rt.append((r, k, cx, ry))
    for i in range(16):
        xs = cx + (i - 7.5) * 4.0
        xt = cx + (i - 7.5) * 0.8
        path = pcb_gen.route45((xs, conn_y - 2.5), (xt, rows_y[0] + 8.2), first="y")
        t.diff_pair(path, 0.1, 0.13)
    hooks_conn.append((k, cx))
t.text("COMPUTE TRAY BOARD 440 X 350 MM", 0.0, -TD / 2 + 10.0, 5.0)
objs_t = t.finish()
counts_t = dict(t.counts)
for (k, cx) in hooks_conn:
    C.hook("conn_%d" % k, ct, root, loc=(cx * MM, conn_y * MM, 2.4 * MM))
for (r, k, cx, ry) in hooks_rt:
    C.hook("retimer_r%d_c%d" % (r, k), ct, root, loc=(cx * MM, ry * MM, 2.4 * MM))
C.hook("board_center_top", coll, root, loc=(0, 0, 1.6 * MM))

dims.add("signal trace width", 0.1, "mm", "4 mil, typical high-speed outer-layer microstrip trace", "B")
dims.add("differential pair gap", 0.13, "mm", "5 mil edge-coupled pair (85-100 ohm class)", "B")
dims.add("via drill / pad", "0.20 / 0.45", "mm", "typical HDI-class via", "B")
dims.add("copper thickness", 35, "um", "1 oz outer layers", "B")
dims.add("solder mask over copper", 20, "um", "typical LPI mask thickness over traces", "B")
dims.add("demo board", "120 x 80 x 1.6", "mm", "illustrative", "C")
dims.add("tray board", "440 x 350 x 2.4", "mm", "plausible compute-tray board: fits a 19 in rack (440 mm usable width); depth scaled from the crude S2 tray ratio (1.9 : 1.5 approx 1.27); not a specific product", "C")
dims.add("backplane connector pitch on tray", round(0.5 * SC, 1), "mm", "crude v0.2 layout 0.5 / 1.9 of the tray width", "C")
dims.add("retimer footprint", "17 x 17 pads, 0.8 mm pitch, 15 mm body outline 17 mm", "mm", "matches narrowcom_retimer_chip", "B")
dims.add("demo board features", counts_d, "", "traces/diff pairs/vias/pads/holes", "A")
dims.add("tray board features", counts_t, "", "traces/diff pairs/vias/pads/holes", "A")

meta = {
    "title": "PCB generator demo boards", "accuracy_level": "B (real trace widths) / C (board sizes)",
    "origin": "board bottom center at z = 0; top surface at z = thickness (1.6 mm demo, 2.4 mm tray); both variants overlap at the origin (toggle collections)",
    "api": "scripts/assets/packaging/pcb_gen.py class PCB: hole, fiducials, trace, diff_pair, vias, pads_grid, pad_rect, text, outline, finish; route45() for 45-degree routes",
    "variants": {"VARIANT_demo_board": "visible by default", "VARIANT_tray_board": "S2 tray board with 4 backplane-connector pad rows and 12 retimer BGA footprints (hooks HOOK_conn_k, HOOK_retimer_r<row>_c<k>); hidden by default"},
    "sources": [{"what": "typical high-speed PCB design values (4 mil trace, 5 mil gap, 1 oz Cu, 0.2/0.45 via)", "note": "generic industry practice, no single source"}],
    "instancing": "vias and BGA pads are Geometry Nodes instances; traces are mitered ribbons",
    "simplifications": ["outer layer only", "no inner layers or via barrels", "masked traces are drawn as a raised mask bulge in a lighter green"],
    "scene_usage": "S2 tray board under the connector row; S4/S3 board regions; any board needing real trace geometry",
}
blend = os.path.join(OUT, AID + ".blend")
C.save(blend)
pv = os.path.join(OUT, "previews")
pngs = []
pngs += pk.render_views(cd, pv, AID, [dict(name="three_quarter", loc=(70, -110, 90), tgt=(0, 0, 1), lens=50, floor=True),
                                      dict(name="top", loc=(0, -0.1, 175), tgt=(0, 0, 1.6), lens=50),
                                      dict(name="front", loc=(0, -190, 30), tgt=(0, 0, 2), lens=50, floor=True),
                                      dict(name="closeup_bga_traces", loc=(-4, -16, 16), tgt=(-6, 0, 1.7), lens=70, floor=True),
                                      dict(name="closeup_hole_pads", loc=(40, -36, 8), tgt=(49, -30, 1.7), lens=70, floor=True)])
cd.hide_render = cd.hide_viewport = True
ct.hide_render = ct.hide_viewport = False
bpy.context.view_layer.update()
pngs += pk.render_views(ct, pv, AID, [dict(name="tray_top", loc=(0, -1, 520), tgt=(0, 0, 2.4), lens=50),
                                      dict(name="tray_three_quarter", loc=(300, -520, 420), tgt=(0, 60, 2), lens=50, floor=True),
                                      dict(name="tray_closeup_connector_row", loc=(-120, 40, 150), tgt=(-115, 120, 2.4), lens=60, floor=True),
                                      dict(name="tray_closeup_retimer_pads", loc=(-100, 40, 60), tgt=(-115, 95, 2.4), lens=60, floor=True)])
ct.hide_render = ct.hide_viewport = True
cd.hide_render = cd.hide_viewport = False
C.save(blend)
bb, tris = pk.eval_stats(cd)
hooks, mnames, props = pk.collect_auto_meta(coll, root)
meta.update({"asset_id": AID, "bbox_mm": bb, "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)], "triangles_including_instances_demo_board": tris,
             "dimension_table": dims.rows, "hooks": hooks, "material_slots": mnames, "custom_properties": props,
             "previews": [os.path.relpath(p, OUT) for p in pngs]})
pk.write_json(blend, meta)
print("DONE", AID, tris, counts_d, counts_t)
