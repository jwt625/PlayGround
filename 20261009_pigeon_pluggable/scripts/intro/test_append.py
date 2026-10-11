"""Smoke test / usage example of the intro append API in a fresh scene (remote location, own camera, frame offset).

Run: scripts/bslot.sh -b --factory-startup --python scripts/intro/test_append.py -- <out_dir>
Renders frames 300 (start) and 435 (end) of a scene where the intro starts at frame 300, placed at (0, -800, 0).
"""

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_intro as BI

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
root = BI.append_intro(location=(0.0, -800.0, 0.0))
BI.offset_animation(root, 300)
cam = bpy.data.objects.new("FILM_cam", bpy.data.cameras.new("FILM_cam"))
scn.collection.objects.link(cam)
scn.camera = cam
BI.key_camera(cam, root, f_start=300)
scn.render.engine = "BLENDER_EEVEE_NEXT"
scn.eevee.taa_render_samples = 4
scn.render.resolution_x, scn.render.resolution_y = 540, 674
scn.view_settings.view_transform = "Standard"
for f in (300, 300 + BI.F_MOVE_END):
    scn.frame_set(f)
    print("cam", f, tuple(round(v, 3) for v in cam.matrix_world.translation))
    scn.render.filepath = str(out / f"append_test_{f}.png")
    bpy.ops.render.render(write_still=True)
print("objects in INTRO_tree:", len(bpy.data.collections[BI.COLL].all_objects))
