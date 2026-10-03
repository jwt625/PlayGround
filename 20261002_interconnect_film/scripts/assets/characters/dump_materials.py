"""Blender -b <asset>.blend --python dump_materials.py -- <out.json>: material slots per object, evaluated triangle count."""
import json
import sys
import bpy
out = sys.argv[sys.argv.index("--") + 1]
dg = bpy.context.evaluated_depsgraph_get()
d = {"materials": {}, "evaluated_triangles": 0, "objects": []}
for o in bpy.data.objects:
    if o.type == "MESH":
        d["materials"][o.name] = [m.name if m else None for m in o.data.materials]
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        d["evaluated_triangles"] += sum(len(p.vertices) - 2 for p in me.polygons)
        e.to_mesh_clear()
        d["objects"].append(o.name)
d["actions"] = sorted(a.name for a in bpy.data.actions)
d["blend_size_mb"] = None
json.dump(d, open(out, "w"), indent=1)
