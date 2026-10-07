"""Render the model from new viewpoints (an orbit around the tray plus a few close-ups), EEVEE.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/blender/render_orbit.py -- --out <dir> [--res 960x720]
Cameras look at the tray center; the orbit is about gravity-up (scene frame) at elevations 15, 40 and 70 deg and
6 azimuths, 1.6 m away; close-ups of the package, front bay, rear bay and front panel. The environment's
backdrop and ground are hidden (table and plinth stay). Lighting: the same
calibrated rig as render_views (uniform white world).
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

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
for o in bpy.data.objects:  # orbits circle the tray: the backdrop and ground would block or fill most shots
    if o.name in ("_env.backdrop", "_env.ground"):
        o.hide_render = True
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
# Orbit about gravity-up (scene frame of config/frames.json) around the tray center; close-ups of the package,
# the front and rear bays and the front panel. Positions in the tray (world) frame, meters.
import json  # noqa: E402

fr = json.loads((ROOT / "config" / "frames.json").read_text())["frames"]
Rs = fr["scene"]["R"]
ex, ey, up = (Vector((Rs[0][k], Rs[1][k], Rs[2][k])) for k in range(3))  # scene axes in world
pkg = Vector(fr["package"]["t_mm"]) * 1e-3
pkg_n = Vector((fr["package"]["R"][0][2], fr["package"]["R"][1][2], fr["package"]["R"][2][2]))
target = Vector((0.0, 0.39, -0.035))
shots = []
for el in (15, 40, 70):
    for az in range(0, 360, 60):
        shots.append((f"orbit_el{el:02d}_az{az:03d}", el, az, 1.6, target))
close = {"close_package": (pkg, 0.32), "close_front_bay": (Vector((0.0, 0.17, -0.03)), 0.55),
         "close_rear_bay": (Vector((0.0, 0.69, -0.03)), 0.5), "close_front_panel": (Vector((0.0, 0.0, -0.04)), 0.6)}
tray_z = Vector((0.0, 0.0, 1.0))


def look_at(cam_obj, eye, t, upv):
    f = (t - eye).normalized()
    r = f.cross(upv)
    if r.length < 1e-6:
        r = f.cross(ey)
    r.normalize()
    u = r.cross(f)
    M = Matrix(((r.x, u.x, -f.x, eye.x), (r.y, u.y, -f.y, eye.y), (r.z, u.z, -f.z, eye.z), (0, 0, 0, 1)))
    cam_obj.matrix_world = M


for name, el, az, dist, t in shots:
    e, z = math.radians(el), math.radians(az)
    d = math.cos(e) * math.cos(z) * ex + math.cos(e) * math.sin(z) * ey + math.sin(e) * up
    look_at(cam, t + dist * d, t, up)
    scene.render.filepath = str(out / f"{name}.png")
    bpy.ops.render.render(write_still=True)
for name, (t, dist) in close.items():
    d = (pkg_n if name == "close_package" else (tray_z * 0.8 - ey * 0.3 if name == "close_front_panel" else tray_z))
    d = (d.normalized() + 0.25 * up).normalized()
    look_at(cam, t + dist * d, t, up)
    scene.render.filepath = str(out / f"{name}.png")
    bpy.ops.render.render(write_still=True)
shots += list(close)
print("ORBIT_DONE", len(shots))
