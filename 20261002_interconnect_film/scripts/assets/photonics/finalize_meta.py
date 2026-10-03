"""Post-process: open every photonics .blend, add per-variant bbox sizes (mm), evaluated triangles (incl. GN instances), hook list,
material slot names and a missing-file check to the asset JSON. Run: Blender -b --python finalize_meta.py -- <out_dir>"""
import json
import os
import sys

import bpy
from mathutils import Vector

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
ASSETS = ["microring_modulator_cell", "microring_array_closeup", "pic_die", "eic_die", "pic_eic_stack", "oe_module_cpo",
          "oe_module_npo", "fau_v_groove", "els_laser_source", "grating_coupler_closeup"]


def bbox(coll):
    pts = []
    for o in coll.all_objects:
        if o.type in {"MESH", "CURVE", "FONT"} and not (o.name.endswith("_proto") or "_proto" in o.name):
            for v in o.bound_box:
                pts.append(o.matrix_world @ Vector(v))
    if not pts:
        return None
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    return [round((hi[k] - lo[k]) * 1000, 4) for k in range(3)], [round(lo[k] * 1000, 4) for k in range(3)]


for a in ASSETS:
    bp = os.path.join(out_dir, a + ".blend")
    bpy.ops.wm.open_mainfile(filepath=bp)
    bpy.context.view_layer.update()
    top = [c for c in bpy.data.collections if c.name == "ASSET_" + a][0]
    dg = bpy.context.evaluated_depsgraph_get()
    tris = 0
    for o in top.all_objects:
        if o.type == "MESH":
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            tris += sum(len(p.vertices) - 2 for p in me.polygons)
            ev.to_mesh_clear()
    for inst in dg.object_instances:
        if inst.is_instance and inst.object.type == "MESH" and inst.parent and inst.parent.name in top.all_objects:
            tris += sum(len(p.vertices) - 2 for p in inst.object.data.polygons)
    variants = {}
    for c in top.children:
        bb = bbox(c)
        variants[c.name] = dict(size_mm=bb[0] if bb else None, min_corner_mm=bb[1] if bb else None)
    jp = os.path.join(out_dir, a + ".json")
    j = json.load(open(jp))
    j["variants_size_mm"] = variants
    j["triangles_evaluated_incl_instances"] = tris
    j["hooks"] = sorted(o.name for o in top.all_objects if o.name.startswith("HOOK_"))
    j["material_slots"] = sorted(m.name for m in bpy.data.materials if m.name.startswith("MAT_photonics_") and m.users > 0)
    j["custom_properties_on_root"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in bpy.data.objects["ROOT_" + a].items()}
    j["missing_external_files"] = [i.filepath for i in bpy.data.images if i.source == "FILE"] + [l.filepath for l in bpy.data.libraries]
    j["size_mm_note"] = "size_mm / bbox_mm cover the union of all variants laid out in the file; use variants_size_mm per variant"
    j["scene_scale_note"] = j.get("scene_scale_note", "1 Blender unit = 1 m; scaled variants carry their scale in geometry (or a scaled parent empty for fau_v_groove/pic_eic_stack insets)")
    j["devlog"] = "DevLog/v1/DevLog-003-photonics.md"
    json.dump(j, open(jp, "w"), indent=2, sort_keys=True)
    print(a, "tris", tris, "hooks", len(j["hooks"]), "mats", len(j["material_slots"]), "missing", j["missing_external_files"])
