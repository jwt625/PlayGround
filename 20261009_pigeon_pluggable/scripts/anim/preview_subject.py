"""Preview stills of the Subject (hoodie variant, cup, actions). Usage:
scripts/bslot.sh -b --factory-startup --python scripts/anim/preview_subject.py -- <out_dir> [frames...]
"""
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys as KY  # noqa: E402
import subject as SJ  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else "outputs/assets/characters/preview"
FRAMES = [int(x) for x in argv[1:]] or [1, 40, 95, 128, 175, 205]
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 16
sc.render.resolution_x = sc.render.resolution_y = 540
sc.render.fps = 30
sc.frame_start, sc.frame_end = 1, 220
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.055, 0.07, 1)
sc.world = w
me = bpy.data.meshes.new("ground")
me.from_pydata([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)], [], [(0, 1, 2, 3)])
g = bpy.data.objects.new("ground", me)
sc.collection.objects.link(g)
gm = bpy.data.materials.new("g")
gm.diffuse_color = (0.2, 0.2, 0.2, 1)
me.materials.append(gm)
for nm, loc, e, rot in (("key", (-3, -4, 5), 900, (0.9, 0, -0.6)), ("fill", (4, -3, 2.5), 300, (1.2, 0, 0.9)), ("rim", (0, 4, 3), 500, (-1.1, 0, 0))):
    ld = bpy.data.lights.new(nm, "AREA")
    ld.energy = e
    ld.size = 3
    lo = bpy.data.objects.new(nm, ld)
    lo.location = loc
    lo.rotation_euler = rot
    sc.collection.objects.link(lo)

s = SJ.append_subject()
SJ.place(s, 1, (0, 2.0, 0), 0.0)
SJ.idle(s, 1, 400)
fe = SJ.walk_path(s, 1, [(0, 2.0), (0.2, 0.6), (0.0, -0.4)])
print("WALK end", fe)
fs = SJ.sip_coffee(s, fe + 10)
print("SIP end", fs)
SJ.look_at(s, fs + 4, fs + 40, (1.5, -1.0, 1.0))
SJ.glance_over_shoulder(s, fs + 44, "R")
KY.flush()
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = 40
cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam


def aim(o, pos, tgt):
    o.location = pos
    d = Vector(tgt) - Vector(pos)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


views = {"front": ((1.2, -4.2, 1.4), (0, 0, 1.0)), "back": ((-1.6, 3.8, 1.5), (0, 0.3, 1.0)), "side": ((4.0, -0.5, 1.4), (0, 0.3, 1.0))}
for f in FRAMES:
    sc.frame_set(f)
    for vn, (p, t) in views.items():
        loc = s.root.matrix_world.translation
        aim(cam, Vector(p) + Vector((loc.x, loc.y, 0)), Vector(t) + Vector((loc.x, loc.y, 0)))
        sc.render.filepath = os.path.join(OUT, "subj_%s_f%03d.png" % (vn, f))
        bpy.ops.render.render(write_still=True)
print("PREVIEW done", OUT, s.root.matrix_world.translation, s.arm.matrix_local)
