"""Render the model from new viewpoints (an orbit around the case plus a few close-ups), EEVEE.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/blender/render_orbit.py -- --out <dir> [--res 960x720]
Cameras look at the case center; the orbit uses elevations 20, 45 and 75 deg and 6 azimuths. Lighting: the same
calibrated rig as render_views (uniform white world).
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "blender"))
import build_all  # noqa: E402
import render_views as rv  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--res", default="960x720")
ap.add_argument("--samples", type=int, default=32)
a = ap.parse_args(argv)
out = Path(a.out)
if not out.is_absolute():
    out = ROOT / out
out.mkdir(parents=True, exist_ok=True)
build_all.build_all()
scene = bpy.context.scene
W, H = (int(x) for x in a.res.split("x"))
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.eevee.taa_render_samples = a.samples
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
rv.setup_world(scene)
cd = bpy.data.cameras.new("_orbit_cam")
cam = bpy.data.objects.new("_orbit_cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
cd.lens = 50
cd.clip_start = 0.005
target = Vector((0.0, 0.0, 0.02))
shots = []
for el in (20, 45, 75):
    for az in range(0, 360, 60):
        shots.append((f"orbit_el{el:02d}_az{az:03d}", el, az, 0.42))
shots += [("close_yoke", 50, 300, 0.16), ("close_flyback", 60, 250, 0.16), ("close_screen", 55, 160, 0.2)]
targets = {"close_yoke": Vector((0.035, 0.0, 0.03)), "close_flyback": Vector((0.07, -0.035, 0.02)),
           "close_screen": Vector((-0.05, 0.0, 0.02))}
for name, el, az, dist in shots:
    t = targets.get(name, target)
    e, z = math.radians(el), math.radians(az)
    cam.location = t + dist * Vector((math.cos(e) * math.cos(z), math.cos(e) * math.sin(z), math.sin(e)))
    cam.rotation_euler = (t - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out / f"{name}.png")
    bpy.ops.render.render(write_still=True)
print("ORBIT_DONE", len(shots))
