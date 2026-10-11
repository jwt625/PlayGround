"""Re-import the hero GLB into an empty scene and report counts, extent and extras (verification).

Run: scripts/bslot.sh -b --factory-startup --python scripts/assets/check_hero_glb.py -- assets/hero/pigeon_osfp_dr8.glb
"""
import sys

import bpy
from mathutils import Vector

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
objs = list(bpy.data.objects)
meshes = [o for o in objs if o.type == "MESH"]
deps = bpy.context.evaluated_depsgraph_get()
tris = 0
for o in meshes:
    me = o.evaluated_get(deps).to_mesh()
    tris += sum(len(p.vertices) - 2 for p in me.polygons)
    o.evaluated_get(deps).to_mesh_clear()


def bbox(names):
    pts = []
    for o in meshes:
        if o.name in names:
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


lo, hi = bbox({o.name for o in meshes})
print("GLB_CHECK objects=%d meshes=%d materials=%d tris=%d" % (len(objs), len(meshes), len(bpy.data.materials), tris))
print("GLB_CHECK whole-asset extent (m): x %.5f  y %.5f  z %.5f" % tuple(hi - lo))
print("GLB_CHECK min (m):", tuple(round(v, 5) for v in lo), "max:", tuple(round(v, 5) for v in hi))
missing = [o.name for o in meshes if "explode" not in o]
print("GLB_CHECK meshes without explode extra:", missing)
hooks = sorted(o.name for o in objs if o.name.startswith("HOOK_"))
print("GLB_CHECK hooks:", hooks)
for nm in ("tx_eye_plate_2x4", "rx_lenslet_array_2x4", "foot_tx_toes_x4", "foot_rx_toes_x4"):
    l2, h2 = bbox({nm})
    print("GLB_CHECK %s centre x = %.4f m" % (nm, (l2.x + h2.x) / 2))
