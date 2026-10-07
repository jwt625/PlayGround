"""Build the full model and save it as a .blend (relative texture paths).
Usage: scripts/bslot.sh -b --factory-startup --python scripts/blender/save_model.py -- <out.blend>"""
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
import build_all  # noqa: E402

out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
print("BUILD_RESULT", build_all.build_all())
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"
out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out))
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_mainfile()
print("SAVED", out)
