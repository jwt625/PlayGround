"""Export the model as glTF binary (.glb, textures embedded) for web/desktop 3D viewers.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/blender/export_glb.py -- <out_dir> [--with-mat]
       [--tray-frame]
Writes <out_dir>/<model_name>.glb (object only) and, with --with-mat, <out_dir>/<model_name>_with_env.glb
(model_name from config/scene.yaml). By default the scene is first rotated from the tray frame into the
gravity ("scene") frame of config/frames.json, so viewers show it standing as photographed; --tray-frame keeps
the tray frame (tray level, open top up). Textures are embedded as JPEG (quality 90) unless --png. Units: meters, glTF Y-up (the exporter converts from Blender's Z-up). Procedural shader nodes that glTF cannot
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
import json  # noqa: E402

from mathutils import Matrix  # noqa: E402

NAME = next((ln.split(":", 1)[1].split("#")[0].strip() for ln in (ROOT / "config" / "scene.yaml").read_text().splitlines()
             if ln.startswith("model_name:")), "model")
if "--tray-frame" not in args:
    Rs = json.loads((ROOT / "config" / "frames.json").read_text())["frames"]["scene"]["R"]
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = Rs[j][i]  # world -> scene: R^T (columns of R are the scene axes in world)
    for o in bpy.data.objects:
        if o.parent is None and not o.name.startswith("_eval"):
            o.matrix_world = M @ o.matrix_world
for o in bpy.data.objects:
    if o.type == "CURVE":  # wires: convert to meshes so every viewer shows them
        bpy.context.view_layer.objects.active = o
for o in [o for o in bpy.data.objects if o.type == "CURVE"]:
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.convert(target="MESH")


IMG_FMT = "AUTO" if "--png" in args else "JPEG"  # photo textures as JPEG q90 (v1: 23 MB with PNG)


def export(path, include_mat):
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        if o.type != "MESH" or o.name.startswith("_eval"):
            continue
        if o.name.startswith("_env") and not include_mat:
            continue
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_image_format=IMG_FMT, export_jpeg_quality=90)
    print("EXPORTED", path, path.stat().st_size)


export(out / f"{NAME}.glb", False)
if "--with-mat" in args:
    export(out / f"{NAME}_with_env.glb", True)
