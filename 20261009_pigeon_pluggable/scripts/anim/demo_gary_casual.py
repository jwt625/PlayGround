"""v3 demo: Gary goes about his day (walk while sipping, tap a rack, check watch, read phone while humming, stretch and
yawn) while street birds planted on a post top, a railing and a wire watch him. Prints perch-contact and face-readability
checks.

scripts/bslot.sh -b --factory-startup --python scripts/anim/demo_gary_casual.py -- stills 40,110,150,200,255 | anim
"""
import math
import os
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import behaviors as BH  # noqa: E402
import keys as KY  # noqa: E402
import pigeon_rig as PR  # noqa: E402
import subject as SJ  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT_DIR, "outputs", "assets", "characters")
RACK = os.path.join(ROOT_DIR, "assets", "components", "datacenter", "rack_bird_q3400.blend")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = argv[0] if argv else "stills"
STILLS = [int(x) for x in argv[1].split(",")] if len(argv) > 1 else [40, 110, 150, 200, 255]
D2R = math.pi / 180
F1 = 300
T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 16
sc.render.resolution_x, sc.render.resolution_y = 540, 674
sc.render.fps = 30
sc.frame_start, sc.frame_end = 1, F1
sc.view_settings.view_transform = "AgX"
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.03, 0.035, 0.05, 1)
sc.world = w


def mat(name, color, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def prim(kind, name, loc, size, m, rot=(0, 0, 0)):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    if kind == "box":
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    else:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=size[0], radius2=size[0], depth=size[1])
    bm.to_mesh(me)
    bm.free()
    me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    sc.collection.objects.link(o)
    o.location, o.rotation_euler = loc, rot
    return o


prim("box", "ground", (0, 0, -0.05), (20, 20, 0.1), mat("ground", (0.06, 0.06, 0.065)))
prim("box", "wall", (0, 4.2, 2.0), (10, 0.2, 4.0), mat("wall", (0.12, 0.08, 0.06), 0.9))
M_METAL = mat("metal", (0.35, 0.36, 0.38), 0.35, 0.8)
post = prim("cyl", "post", (-0.9, 2.6, 0.6), (0.06, 1.2), M_METAL)
RAIL = (Vector((-1.6, 3.0, 1.0)), Vector((0.9, 3.0, 1.0)), 0.02)
rail = prim("cyl", "rail", (RAIL[0] + RAIL[1]) / 2, (RAIL[2], (RAIL[1] - RAIL[0]).length), M_METAL, rot=(0, math.pi / 2, 0))
for x in (-1.6, 0.9):
    prim("cyl", "rail_post", (x, 3.0, 0.5), (0.02, 1.0), M_METAL)
WIRE = (Vector((-2.0, 3.5, 1.75)), Vector((1.6, 3.5, 1.75)), 0.004)
prim("cyl", "wire", (WIRE[0] + WIRE[1]) / 2, (WIRE[2], (WIRE[1] - WIRE[0]).length), mat("wire", (0.05, 0.05, 0.05)), rot=(0, math.pi / 2, 0))
with bpy.data.libraries.load(RACK, link=False) as (src, dst):
    dst.collections = [c for c in src.collections if c.startswith("ASSET_")][:1]
rack = bpy.data.objects.new("rack", None)
rack.instance_type = "COLLECTION"
rack.instance_collection = dst.collections[0]
rack.location = (1.0, 1.62, 0.0)
rack.rotation_euler = (0, 0, -math.pi / 2)
sc.collection.objects.link(rack)
for nm, loc, e, rot, col in (("key", (-1.5, -2.0, 3.5), 900, (0.8, 0, -0.6), (1.0, 0.9, 0.8)), ("fill", (2.5, -2.5, 2.0), 300, (1.1, 0, 0.8), (0.7, 0.8, 1.0)),
                             ("rim", (0.0, 3.5, 3.0), 400, (-0.9, 0, 0), (0.8, 0.9, 1.0))):
    ld = bpy.data.lights.new(nm, "AREA")
    ld.energy, ld.size, ld.color = e, 2.5, col
    lo = bpy.data.objects.new(nm, ld)
    lo.location, lo.rotation_euler = loc, rot
    sc.collection.objects.link(lo)
cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
cd.clip_start = 0.02
cd.lens = 32
tgt = bpy.data.objects.new("tgt", None)
sc.collection.objects.link(tgt)
con = cam.constraints.new("TRACK_TO")
con.target = tgt
con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
for f, p, t in ((1, (0.0, -2.3, 1.5), (-0.9, 1.3, 1.05)), (95, (0.9, -1.9, 1.5), (0.35, 1.0, 1.1)), (F1, (0.8, -1.7, 1.5), (0.4, 1.0, 1.15))):
    cam.location, tgt.location = p, t
    cam.keyframe_insert("location", frame=f)
    tgt.keyframe_insert("location", frame=f)

# ---------------------------------------------------------------- Gary's day
s = SJ.append_subject()
SJ.place(s, 1, (-2.0, 1.5, 0), SJ.yaw_face((-2.0, 1.5), (0.35, 0.9)))
SJ.idle(s, 1, F1 + 60)
f = SJ.walk_sipping(s, 1, [(-2.0, 1.5), (-0.8, 1.1), (0.35, 0.9)], sips=1, end_yaw=45 * D2R)   # ends turned toward the lens
log("walk_sipping stops", f)
f = SJ.tap_rack(s, f + 14, (0.62, 1.10, 1.30), side="L", taps=2)
log("tap_rack ends", f)
f = SJ.check_watch(s, f + 2)
log("check_watch ends", f)
f_ph = f + 2
f = SJ.glance_phone(s, f_ph, dur=56)
SJ.idle_hum(s, f_ph, f)
log("glance_phone + hum ends", f)
f = SJ.stretch_yawn(s, f + 2)
log("stretch_yawn ends", f)
KY.flush()

# ---------------------------------------------------------------- street birds on perches
lib = PR.proxy_library(log=log)
birds = {}
b1 = PR.make_proxy_rig("PB_post", lib, (-0.9, 2.6, 1.3), -100 * D2R)
PR.rest_pose(b1, 1)
PR.place(b1, 1, (-0.9, 2.6, 1.3), -100 * D2R)
r1 = PR.plant_on_surface(b1, 1, [post])
b2 = PR.make_proxy_rig("PB_rail", lib, (-0.3, 3.0, 1.1), -80 * D2R)
PR.rest_pose(b2, 1)
PR.place(b2, 1, (-0.3, 3.0, 1.1), -80 * D2R)
r2 = PR.plant_on_wire(b2, 1, RAIL[0], RAIL[1], radius=RAIL[2])
b3 = PR.make_proxy_rig("PB_wire", lib, (0.4, 3.5, 1.8), -90 * D2R)
PR.rest_pose(b3, 1)
PR.place(b3, 1, (0.4, 3.5, 1.8), -90 * D2R)
r3 = PR.plant_on_wire(b3, 1, WIRE[0], WIRE[1], radius=WIRE[2])
b4 = PR.make_proxy_rig("PB_wire2", lib, (-1.0, 3.5, 1.8), -90 * D2R)
PR.rest_pose(b4, 1)
PR.place(b4, 1, (-1.0, 3.5, 1.8), -90 * D2R)
r4 = PR.plant_on_wire(b4, 1, WIRE[0], WIRE[1], radius=WIRE[2])
birds = {"PB_post": b1, "PB_rail": b2, "PB_wire": b3, "PB_wire2": b4}
KY.flush()
specs = [BH.spec_from_rig(k, r, kind="street", group="street", can_turn=False, yaw_lim=110 * D2R, roll_lim=50 * D2R) for k, r in birds.items()]
track = SJ.head_track(s, 1, F1, step=6)
sch = BH.schedule(specs, 1, F1, seed=3, subject=track)
BH.apply(sch, birds)
for r in birds.values():
    PR.breathe(r, 1, F1, seed=len(r.name))

# ---------------------------------------------------------------- checks
sc.frame_set(1)
bpy.context.view_layer.update()


def sole_world(r):
    M = r.root.matrix_world
    return [M @ q for q in PR.sole_points(r)]


def seg_dist(p, a, b):
    d = b - a
    u = max(0.0, min(1.0, (p - a).dot(d) / d.length_squared))
    return (p - (a + d * u)).length


post_top = post.location.z + 0.6
g1 = min(p.z for p in sole_world(b1)) - post_top
g2 = min(seg_dist(p, RAIL[0], RAIL[1]) for p in sole_world(b2)) - RAIL[2]
g3 = min(seg_dist(p, WIRE[0], WIRE[1]) for p in sole_world(b3)) - WIRE[2]
g4 = min(seg_dist(p, WIRE[0], WIRE[1]) for p in sole_world(b4)) - WIRE[2]
log("PERCH contact (lowest sole point to surface, mm; 0 = touching): post %.2f, rail %.2f, wire %.2f, wire2 %.2f"
    % (g1 * 1e3, g2 * 1e3, g3 * 1e3, g4 * 1e3), "plant results", r1["dz"], r2["yaw"], r3["yaw"])
bad = SJ.unreadable_frames(s, 1, F1, cam, max_deg=100, step=5)
log("Gary face turned > 100 deg from the lens at frames:", bad)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "demo_gary_casual.blend"), compress=True)
if MODE == "stills":
    for fr in STILLS:
        sc.frame_set(fr)
        sc.render.filepath = os.path.join(OUT, "work_v3", "casual_f%03d.png" % fr)
        bpy.ops.render.render(write_still=True)
elif MODE == "anim":
    sc.render.filepath = os.path.join(OUT, "work_v3", "frames", "f_")
    bpy.ops.render.render(animation=True)
log("DONE")
