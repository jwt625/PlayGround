"""Render one scene blend: applies the bloom/vignette/grain compositor group, then renders PNG frames.

Usage: blender -b scenes/v1/sNN_*.blend --python scripts/film_v1/render_scene.py -- <out_dir> [res_pct] [comp_preset] [frame ...]
With frame numbers it renders just those frames (stills); otherwise the whole 1..frame_end animation.
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, "scripts", "assets", "materials_vfx"))
import render_presets as RP  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
pct = int(args[1]) if len(args) > 1 else 100
comp = args[2] if len(args) > 2 else "cartoon"
frames = [int(a) for a in args[3:]]
scn = bpy.context.scene
with bpy.data.libraries.load(os.path.join(PROJ, "assets", "components", "materials_vfx", "lighting_and_world.blend"), link=False) as (src, dst):
    dst.node_groups = ["NG_comp_post"]
if comp != "off":
    RP.setup_comp(scn, comp)
preset = os.environ.get("RENDER_PRESET")
mb, shutter = scn.render.use_motion_blur, scn.render.motion_blur_shutter  # keep the scene's own motion blur through the preset
if preset:  # draft | standard | hero (see render_presets.py); draft = 8 samples, 50 percent resolution
    RP.apply_render_preset(scn, preset)
    if os.environ.get("RENDER_PCT_OVERRIDE") != "0":
        scn.render.resolution_percentage = pct
    scn.render.use_motion_blur, scn.render.motion_blur_shutter = mb, shutter
else:
    scn.render.resolution_percentage = pct
scn.render.image_settings.file_format = "PNG"
os.makedirs(out, exist_ok=True)
scn.render.filepath = os.path.join(out, "f_")
if frames:
    for f in frames:
        scn.frame_set(f)
        scn.render.filepath = os.path.join(out, "f_%04d" % f)
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
print("RENDERED", out, scn.frame_end)
