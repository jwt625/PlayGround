"""Render selected frames of the open .blend to PNGs.
Usage: blender -b <file.blend> --python scripts/render_frames.py -- <out_dir> <frame> [<frame> ...]
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out, frames = argv[0], [int(f) for a in argv[1:] for f in a.split()]
os.makedirs(out, exist_ok=True)
scn = bpy.context.scene
for f in frames:
    scn.frame_set(f)
    scn.render.filepath = os.path.join(out, "f%04d.png" % f)
    bpy.ops.render.render(write_still=True)
