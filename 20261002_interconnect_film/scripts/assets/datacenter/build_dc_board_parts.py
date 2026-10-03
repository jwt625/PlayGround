"""dc_board_parts: family of 14 dummy board components (VRM, chokes, caps, heat sink, SOCAMM, DIMM, NIC/DPU card, cold plate,
QD pair, fan, PCIe slot, BMC card, coin cell, E1.S bank) for the S2 rack-interior shots. New 2026-10-02.

Run: Blender -b --python scripts/assets/datacenter/build_dc_board_parts.py -- assets/components/datacenter
One ASSET_<part> collection each (real size, mm), root empty at the footprint bottom centre. Sources: references/datacenter/SOURCES_s02_dummy_parts.json
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_board_parts_lib as B
import common as C
import dc_common as D
import bpy
from mathutils import Vector

OUT, KV = D.parse_args()
AID = "dc_board_parts"
C.reset()
D.reset_mats()
M = B.mats()
roots = {}
colls = {}
for name, fn in B.PARTS.items():
    mb, ms = fn(M)
    coll, root = C.new_asset(name, accuracy=B.PART_META[name][0])
    o = mb.obj("%s_body" % name, ms, bevel=(0.0003, 1))
    C.add(o, coll, root)
    C.hook("airflow" if name == "fan_module" else "origin_top_center", coll, root, (0, 0, 0))
    roots[name], colls[name] = root, coll
bpy.context.view_layer.update()

# preview layout: grid of the parts (positions only for previews; moved back before saving)
meta_parts = {}
for name in B.PARTS:
    bb = C.bbox_mm(colls[name])
    meta_parts[name] = dict(size_mm=[round(bb[1][k] - bb[0][k], 1) for k in range(3)], accuracy=B.PART_META[name][0],
                            provenance=B.PART_META[name][1], triangles=C.count_tris(colls[name]))
blend = os.path.join(OUT, AID + ".blend")
meta = dict(asset_id=AID, family=list(B.PARTS), parts=meta_parts, units="1 Blender unit = 1 m; real-size features (apply the rack stack detail scale 4 at assembly)",
            origin="each part: bottom centre of the footprint on the board top surface (z = 0), +y toward the backplane, -y front",
            sources=json.load(open(os.path.join(os.path.dirname(OUT), "..", "..", "references", "datacenter", "SOURCES_s02_dummy_parts.json")))["sources"]
            if os.path.exists(os.path.join(OUT, "..", "..", "references", "datacenter", "SOURCES_s02_dummy_parts.json")) else "references/datacenter/SOURCES_s02_dummy_parts.json",
            hooks={"HOOK_origin_top_center": "each part, at the origin", "HOOK_airflow": "fan_module origin; airflow along +/-y"},
            material_slots=D.mat_slots(), simplifications=["no text, logos or part markings", "blades, pins, contacts are decorative"],
            intended_usage="S2 rack interior board dressing at detail scale 4 (see tray_dummy_layouts)")
C.write_meta(os.path.splitext(blend)[0] + ".json", meta)
C.save(blend)
# previews: grid
xs = 0.0
order = list(B.PARTS)
for i, name in enumerate(order):
    r, c = divmod(i, 5)
    roots[name].location = (c * 0.16 - 0.32, -r * 0.16, 0)
tmp = bpy.data.collections.new("PREV_ALL")
bpy.context.scene.collection.children.link(tmp)
for name in order:
    tmp.children.link(colls[name])
bpy.context.view_layer.update()
D.render_views(tmp, os.path.join(OUT, "previews"), AID, [
    dict(name="grid_three_quarter", loc=(0.1, -0.75, 0.55), target=(0.0, -0.16, 0.0), lens=40),
    dict(name="grid_top", loc=(0.0, -0.17, 0.9), target=(0.0, -0.17, 0.0), lens=40)],
    floor=True, floor_color=(0.02, 0.1, 0.05), sun=1.6, world=0.6, lights=[((0.2, -0.5, 0.6), 80, 0.6)], clip=(0.01, 20))
print("DONE", AID)
