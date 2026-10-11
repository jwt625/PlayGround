"""Import each datacenter GLB into an empty scene and report object count, HOOK nodes, bounds (Blender Z-up after import).
Run: scripts/bslot.sh -b --python scripts/assets/datacenter/check_rack_glbs.py -- assets/components/datacenter
"""
import os
import sys

import bpy
from mathutils import Vector

d = sys.argv[sys.argv.index("--") + 1]
for f in sorted(os.listdir(d)):
    if not f.endswith(".glb"):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.join(d, f))
    objs = list(bpy.context.scene.objects)
    hooks = [o for o in objs if o.name.startswith("HOOK_")]
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    lo = [round(min(p[i] for p in pts), 3) for i in range(3)]
    hi = [round(max(p[i] for p in pts), 3) for i in range(3)]
    cell = next((o for o in hooks if o.name.startswith("HOOK_cell_1_0_0")), None)
    extra = ""
    if cell:
        fx = (cell.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
        extra = " HOOK_cell_1_0_0 at %s facing %s fill_index=%s" % ([round(v, 3) for v in cell.matrix_world.translation],
                                                                   [round(v, 3) for v in fx], cell.get("fill_index"))
    print("GLBCHECK %s objects=%d meshes=%d hooks=%d bounds=%s..%s%s" % (f, len(objs), sum(o.type == "MESH" for o in objs),
                                                                           len(hooks), lo, hi, extra))
