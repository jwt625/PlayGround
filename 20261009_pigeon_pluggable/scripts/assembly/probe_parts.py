"""Read-only probe: list objects (name, type, collection, world bbox mm, tris) of a .blend. Usage:
scripts/bslot.sh -b <file.blend> --python scripts/assembly/probe_parts.py"""
import bpy
from mathutils import Vector
dg = bpy.context.evaluated_depsgraph_get()
for c in bpy.data.collections:
    print("COLL", c.name, len(c.objects), [ch.name for ch in c.children])
for o in sorted(bpy.data.objects, key=lambda o: o.name):
    if o.type == "MESH":
        bb = [o.matrix_world @ Vector(v) for v in o.bound_box]
        lo = [min(p[i] for p in bb) * 1000 for i in range(3)]
        hi = [max(p[i] for p in bb) * 1000 for i in range(3)]
        tris = sum(len(p.vertices) - 2 for p in o.data.polygons)
        print("OBJ %-44s %-6s %-18s lo %s hi %s tris %d mats %s par %s" % (o.name, o.type, o.users_collection[0].name if o.users_collection else "-",
              tuple(round(v, 2) for v in lo), tuple(round(v, 2) for v in hi), tris, [m.name for m in o.data.materials if m][:3], o.parent.name if o.parent else "-"))
    else:
        print("OBJ %-44s %-6s %-18s loc %s par %s" % (o.name, o.type, o.users_collection[0].name if o.users_collection else "-", tuple(round(v * 1000, 2) for v in o.matrix_world.translation), o.parent.name if o.parent else "-"))
