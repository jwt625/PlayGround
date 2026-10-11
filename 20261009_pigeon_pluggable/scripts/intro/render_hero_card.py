"""Render the hero card image (cyber pigeon) from the hero GLB: studio 3/4 view, 800x600, neutral grey backdrop.

Run: scripts/bslot.sh -b --factory-startup --python scripts/intro/render_hero_card.py
Output: assets/components/intro/photos/cyber_pigeon.png
"""

import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
GLB = ROOT / "assets" / "hero" / "pigeon_osfp_dr8.glb"
OUT = ROOT / "assets" / "components" / "intro" / "photos" / "cyber_pigeon.png"

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=str(GLB))
objs = [o for o in scn.objects if o.type == "MESH"]
hidden = []
for o in objs:
    n = o.name.lower()
    if any(k in n for k in ("leash", "explode_ghost")):
        o.hide_render = True
        hidden.append(o.name)
print("hidden:", hidden)
pts = [o.matrix_world @ Vector(c) for o in objs if not o.hide_render for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr, size = (lo + hi) / 2, hi - lo
print("bbox lo", tuple(lo), "hi", tuple(hi))
L = max(size)

# backdrop: curved-ish floor + wall, mid grey
bpy.ops.mesh.primitive_plane_add(size=L * 30, location=(ctr.x, ctr.y, lo.z))
floor = bpy.context.object
bpy.ops.mesh.primitive_plane_add(size=L * 30, location=(ctr.x, ctr.y + L * 3.0, ctr.z), rotation=(math.pi / 2, 0, 0))
wall = bpy.context.object
m = bpy.data.materials.new("backdrop")
m.use_nodes = True
b = m.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.34, 0.35, 0.37, 1)
b.inputs["Roughness"].default_value = 0.8
for o in (floor, wall):
    o.data.materials.append(m)

# camera: 3/4 front view (beak along -Y in the GLB), slightly above
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = 70
cam = bpy.data.objects.new("cam", cam_d)
scn.collection.objects.link(cam)
direction = Vector((-0.75, -1.0, 0.32)).normalized()
cam.location = ctr + direction * L * 2.5
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
scn.camera = cam


def area(name, loc, energy, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size, ld.color = energy, size, color
    lo_ = bpy.data.objects.new(name, ld)
    scn.collection.objects.link(lo_)
    lo_.location = loc
    lo_.rotation_euler = (ctr - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


k = L * L  # energy scales with distance^2
area("key", ctr + Vector((-1.6, -1.4, 1.6)) * L, 140 * k, L * 1.5)
area("fill", ctr + Vector((1.8, -1.2, 0.6)) * L, 45 * k, L * 2.0, (0.9, 0.95, 1.0))
area("rim", ctr + Vector((0.6, 1.8, 1.4)) * L, 160 * k, L * 1.2)

w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.055, 1)
scn.world = w
scn.render.engine = "BLENDER_EEVEE_NEXT"
scn.eevee.taa_render_samples = 32
scn.render.resolution_x, scn.render.resolution_y = 800, 600
scn.view_settings.view_transform = "Standard"
scn.render.image_settings.file_format = "PNG"
scn.render.filepath = str(OUT)
OUT.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.render.render(write_still=True)
print("wrote", OUT)
