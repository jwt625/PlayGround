"""v2 demo: the Subject takes the hero out of its real rack cell (R's bird rack, HOOK_bird_1_0_1) onto his palm, raises it,
the bird flops on its back and the hand tilts its belly/feet toward the camera, back upright, back into the cell, then a
glance into the lens. Mode 's6' renders stills of the S6 helpers (walk to mark, slow look-around, startle-freeze).

scripts/bslot.sh -b --factory-startup --python scripts/anim/demo_hold_bird.py -- anim | stills 30,60,95,120,170 | s6
"""
import math
import os
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import keys as KY  # noqa: E402
import pigeon_rig as PR  # noqa: E402
import subject as SJ  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT_DIR, "outputs", "assets", "characters")
RACK = os.path.join(ROOT_DIR, "assets", "components", "datacenter", "rack_bird_q3400.blend")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = argv[0] if argv else "stills"
STILLS = [int(x) for x in argv[1].split(",")] if len(argv) > 1 else [30, 60, 95, 120, 170]
D2R = math.pi / 180
T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 16
sc.render.resolution_x = sc.render.resolution_y = 540
sc.render.fps = 30
sc.view_settings.view_transform = "AgX"
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.025, 0.04, 1)
sc.world = w
gm = bpy.data.materials.new("ground")
gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.05, 0.05, 0.055, 1)
me = bpy.data.meshes.new("ground")
me.from_pydata([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)], [], [(0, 1, 2, 3)])
me.materials.append(gm)
sc.collection.objects.link(bpy.data.objects.new("ground", me))
for nm, loc, e, rot, col in (("key", (2.2, 1.6, 3.2), 700, (0.75, 0, 2.2), (1.0, 0.9, 0.8)), ("fill", (2.0, -2.0, 2.0), 250, (1.1, 0, 0.8), (0.7, 0.8, 1.0)),
                             ("top", (0.9, 0.0, 3.5), 300, (0, 0, 0), (1, 1, 1))):
    ld = bpy.data.lights.new(nm, "AREA")
    ld.energy, ld.size, ld.color = e, 2.0, col
    lo = bpy.data.objects.new(nm, ld)
    lo.location, lo.rotation_euler = loc, rot
    sc.collection.objects.link(lo)
cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
cd.clip_start = 0.01


def aim(p, t, lens):
    cam.location = p
    cam.rotation_euler = (Vector(t) - Vector(p)).to_track_quat("-Z", "Y").to_euler()
    cd.lens = lens


if MODE == "s6":
    sc.frame_start, sc.frame_end = 1, 200
    aim((1.2, -4.6, 1.55), (0.0, 0.0, 1.2), 35)
    s = SJ.append_subject()
    SJ.place(s, 1, (-2.2, 1.0, 0), SJ.yaw_face((-2.2, 1.0), (0.0, 0.0)))
    SJ.idle(s, 1, 400)
    fe = SJ.walk_to_mark(s, 1, (0.0, 0.0), face=tuple(cam.location))
    log("walk_to_mark settled", fe)
    f2 = SJ.look_around_slow(s, fe + 2, fe + 72, tuple(cam.location), span_deg=40)
    f3 = SJ.startle_freeze(s, f2 + 4, 200, look=cam)
    KY.flush()
    bad = SJ.unreadable_frames(s, 1, 200, cam, max_deg=100, step=5)
    log("frames with the face turned > 100 deg from the lens:", bad)
    log("face angle at", [(f, round(SJ.face_angle(s, f, cam.location))) for f in (fe, fe + 20, fe + 40, fe + 60, f2 + 30, 195)])
    for f in (fe - 10, fe + 20, fe + 45, f2 + 12, 195):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(OUT, "work_v2", "s6_f%03d.png" % f)
        bpy.ops.render.render(write_still=True)
    log("DONE")
    raise SystemExit

# ---------------------------------------------------------------- rack, hero, subject
with bpy.data.libraries.load(RACK, link=False) as (src, dst):
    dst.collections = [c for c in src.collections if c.startswith("ASSET_")][:1]
sc.collection.children.link(dst.collections[0])
bpy.context.view_layer.update()
hook = next(o for o in dst.collections[0].all_objects if o.name.startswith("HOOK_bird_1_0_1"))
cell_pos = tuple(hook.matrix_world.translation)
xa = hook.matrix_world.col[0].xyz
cell_yaw = math.atan2(xa.y, xa.x)
log("cell", [round(v, 3) for v in cell_pos], "yaw", round(cell_yaw / D2R, 1))
hero = PR.build_hero_rig("HERO", loc=cell_pos, yaw=cell_yaw, log=log)
PR.rest_pose(hero, 1)
PR.place(hero, 1, cell_pos, cell_yaw)
F1 = 190
sc.frame_start, sc.frame_end = 1, F1
s = SJ.append_subject()
sp = (cell_pos[0] + 0.45, cell_pos[1] - 0.30, 0.0)
SJ.place(s, 1, sp, SJ.yaw_face(sp, cell_pos))
SJ.idle(s, 1, 400)
SJ.cup_show(s, 1, False)
aim((1.0, 1.45, 1.6), (0.8, -0.08, 1.38), 30)
present = (Vector(cam.location) - Vector((sp[0], sp[1], 1.3))).normalized()
res = SJ.hold_bird(s, 8, 165, cell_pos, cell_yaw, rig=hero, present_dir=tuple(present), end="cell")
log("hold_bird frames", res["frames"])
PR.head_look_at(hero, res["frames"]["raised"], Vector((sp[0], sp[1], 1.5)), dur=3)
PR.head_snap(hero, res["frames"]["flip0"] - 2, 0.0, 0.0, 0.0)
PR.blinks(hero, 1, F1, seed=4)
fg = SJ.glance_at_camera(s, 168, cam, hold=14)
KY.flush()
log("glance ends", fg)
# feet-on-palm check: carrier origin vs palm surface while perched / on back
hookR = next(o for o in s.objs if o.name.startswith("HOOK_hand_R"))
pf = SJ.palm_frame(s)
chk = []
for f in (res["frames"]["on"] + 4, res["frames"]["raised"], res["frames"]["present"] + 5):
    sc.frame_set(f)
    P = hookR.matrix_world @ pf
    chk.append((f, round((res["carrier"].matrix_world.translation - P).length, 4), round((hero.root.matrix_world.translation - P).length, 4)))
log("carrier/hero-root distance to palm centre (perched 0; on back = %.3f)" % (0.066 * 2.6), chk)
# clearance: sampled hero shell vertices vs Gary's evaluated outfit/body meshes (excluding the right hand/forearm region)
from mathutils.kdtree import KDTree  # noqa: E402
shell = [o for o in hero.glb if o.type == "MESH" and o.name.startswith(("skin_", "wing_", "tail_fan", "beak_", "foot_", "leg_"))]
gary = [o for o in s.objs if o.type == "MESH" and not o.hide_render and o.name.split("_")[-1] in
        ("hardhat", "bib", "overalls", "belt", "pouches", "head", "body", "shirt", "straps", "pocket", "pen", "buckle", "bib_buckles")]
worst = []
for f in range(res["frames"]["hop0"], res["frames"]["lowered"], 3):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    hand = (hookR.matrix_world @ pf)
    kd_pts = []
    for o in gary:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        mw = ev.matrix_world
        kd_pts += [(mw @ v.co, o.name) for v in m.vertices]
        ev.to_mesh_clear()
    kd_pts = [p for p in kd_pts if (p[0] - hand).length > 0.16]   # the palm/fingers/forearm hold the bird by design
    kd = KDTree(len(kd_pts))
    for i, p in enumerate(kd_pts):
        kd.insert(p[0], i)
    kd.balance()
    dmin = (9.0, "", "")
    for o in shell:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        mw = ev.matrix_world
        for v in list(m.vertices)[::7]:
            _, i, d = kd.find(mw @ v.co)
            if d < dmin[0]:
                dmin = (d, o.name, kd_pts[i][1])
        ev.to_mesh_clear()
    worst.append((f, round(dmin[0], 3), dmin[1], dmin[2]))
log("bird-to-Gary clearance (m, excl. 0.16 m around the palm):", [w[:2] for w in worst])
log("closest parts", [w for w in worst if w[1] < 0.03])
log("min clearance", min(worst, key=lambda x: x[1]))
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "demo_hold_bird.blend"), compress=True)
if MODE == "stills":
    for f in STILLS:
        sc.frame_set(f)
        sc.render.filepath = os.path.join(OUT, "work_v2", "hold_f%03d.png" % f)
        bpy.ops.render.render(write_still=True)
elif MODE == "anim":
    sc.render.filepath = os.path.join(OUT, "work_v2", "frames", "f_")
    bpy.ops.render.render(animation=True)
log("DONE")
