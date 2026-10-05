"""Export the model as glTF binary (.glb, textures embedded) for web/desktop 3D viewers.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/blender/export_glb.py -- <out_dir> [--with-mat]
Writes <out_dir>/crt_model.glb (object only) and, with --with-mat, <out_dir>/crt_model_with_mat.glb.
Units: meters, glTF Y-up (the exporter converts from Blender's Z-up). Procedural shader nodes that glTF cannot
represent (e.g. noise-based copper) export as their base color; image textures export as is.
"""
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
import build_all  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
out = Path(args[0]).resolve()
out.mkdir(parents=True, exist_ok=True)
print("BUILD_RESULT", build_all.build_all())
for o in bpy.data.objects:
    if o.type == "CURVE":  # wires: convert to meshes so every viewer shows them
        bpy.context.view_layer.objects.active = o
for o in [o for o in bpy.data.objects if o.type == "CURVE"]:
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.convert(target="MESH")


def export(path, include_mat):
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name.startswith("_eval"):
            continue
        if o.name.startswith("_env") and not include_mat:
            continue
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_image_format="AUTO")
    print("EXPORTED", path, path.stat().st_size)


export(out / "crt_model.glb", False)
if "--with-mat" in args:
    export(out / "crt_model_with_mat.glb", True)
