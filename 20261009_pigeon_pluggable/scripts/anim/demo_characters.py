"""Workstream C demo: street birds re-pairing (link events), a proxy bird take-off/flight/landing, the hero rig (look-at,
TX disable), a small head-only rack wall, and the Subject walking in, stopping, looking, sipping coffee.

Usage (one Blender process):
  scripts/bslot.sh -b --factory-startup --python scripts/anim/demo_characters.py -- stills 20,60,100,130,165
  scripts/bslot.sh -b --factory-startup --python scripts/anim/demo_characters.py -- anim
Outputs: outputs/assets/characters/demo_characters.blend, demo_still_fNNN.png, demo_frames/ (anim; encode then delete),
demo_link_events.json.
"""
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import behaviors as BH  # noqa: E402
import keys as KY  # noqa: E402
import pigeon_rig as PR  # noqa: E402
import subject as SJ  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT_DIR, "outputs", "assets", "characters")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = argv[0] if argv else "stills"
STILLS = [int(x) for x in argv[1].split(",")] if len(argv) > 1 else [20, 60, 100, 130, 165]
F0, F1 = 1, 180
D2R = math.pi / 180
SEED = 11
T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 16
sc.render.resolution_x = sc.render.resolution_y = 540
sc.render.fps = 30
sc.frame_start, sc.frame_end = F0, F1
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.012, 0.016, 0.03, 1)
sc.world = w


def mat(name, color, rough=0.7, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


def box(name, c, size, m):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(m)
    o = bpy.data.objects.new(name, me)
    sc.collection.objects.link(o)
    o.location = c
    return o


M_GROUND = mat("ground", (0.035, 0.035, 0.04), 0.55)
M_WALL = mat("wall", (0.09, 0.05, 0.04), 0.9)
M_RAIL = mat("rail", (0.25, 0.26, 0.28), 0.35, 0.8)
M_CRATE = mat("crate", (0.16, 0.10, 0.05), 0.8)
M_SHELF = mat("shelf", (0.10, 0.11, 0.12), 0.6, 0.5)
M_BEAM = mat("beam", (1.0, 0.08, 0.05), 0.5, 0.0, 8.0)
box("ground", (0, 0, -0.05), (30, 30, 0.1), M_GROUND)
box("wall", (0, 3.4, 2.0), (14, 0.2, 4.0), M_WALL)
box("rail_top", (0, 0, 0.985), (3.4, 0.05, 0.03), M_RAIL)
for x in (-1.6, -0.55, 0.55, 1.6):
    box("rail_post", (x, 0, 0.49), (0.04, 0.04, 0.98), M_RAIL)
box("crate", (-0.95, -0.85, 0.25), (0.5, 0.5, 0.5), M_CRATE)
SHELF_Z = (1.5, 1.95, 2.4)
for z in SHELF_Z:
    box("shelf", (-0.2, 3.05, z - 0.015), (2.6, 0.5, 0.03), M_SHELF)
for nm, loc, e, rot, sz, col in (("lamp", (0.6, -0.6, 4.2), 900, (0.15, 0.1, 0), 2.0, (1.0, 0.82, 0.6)),
                                 ("fill", (2.5, -4.5, 2.0), 250, (1.25, 0, 0.5), 3.0, (0.6, 0.75, 1.0)),
                                 ("rim", (-1.0, 2.6, 3.2), 350, (-0.9, 0, 0.3), 2.0, (0.7, 0.85, 1.0))):
    ld = bpy.data.lights.new(nm, "AREA")
    ld.energy, ld.size, ld.color = e, sz, col
    lo = bpy.data.objects.new(nm, ld)
    lo.location, lo.rotation_euler = loc, rot
    sc.collection.objects.link(lo)

# ------------------------------------------------------------------ subject
s = SJ.append_subject()
SJ.place(s, F0, (-2.0, 1.35, 0), SJ.yaw_face((-2.0, 1.35), (0.3, 1.2)))
SJ.idle(s, F0, F1 + 60)
f_stop = SJ.walk_path(s, F0, [(-2.0, 1.35), (-0.8, 1.31), (0.3, 1.2)])
log("subject walk ends", f_stop)

# ------------------------------------------------------------------ birds
lib = PR.proxy_library(log=log)
log("proxy parts", sorted(lib))
RAIL_Z = 1.0
street = {}
specs = []
rail = [("st0", -1.15, -95), ("st1", -0.5, -70), ("st2", 0.2, -110), ("fly", 0.9, -80)]
for bid, x, yd in rail:
    r = PR.make_proxy_rig(bid, lib, (x, 0.0, RAIL_Z), yd * D2R)
    PR.rest_pose(r, F0)
    PR.place(r, F0, (x, 0.0, RAIL_Z), yd * D2R)
    PR.breathe(r, F0, F1, seed=sum(map(ord, bid)))
    street[bid] = r
    specs.append(BH.BirdSpec(bid, (x, 0.0, RAIL_Z), yd * D2R, kind="street", yaw_lim=110 * D2R, roll_lim=50 * D2R, group="street",
                             blocked=[(14, F1)] if bid == "fly" else [(146, F1)] if bid == "st0" else [], can_turn=False if bid == "st0" else None))
hero = PR.build_hero_rig("HERO", loc=(-0.95, -0.85, 0.5), yaw=-40 * D2R, log=log)
PR.rest_pose(hero, F0)
PR.place(hero, F0, (-0.95, -0.85, 0.5), -40 * D2R)
PR.breathe(hero, F0, F1, seed=3)
street["HERO"] = hero
specs.append(BH.BirdSpec("HERO", (-0.95, -0.85, 0.5), -40 * D2R, kind="street", yaw_lim=110 * D2R, roll_lim=50 * D2R,
                         group="street", blocked=[(146, F1)], can_turn=False))
rack = {}
for row, z in enumerate(SHELF_Z):
    for col in range(4):
        bid = "rk%d%d" % (row, col)
        x = -1.2 + col * 0.65 + (0.2 if row % 2 else 0.0)
        r = PR.make_proxy_rig(bid, lib, (x, 2.95, z), -90 * D2R, mode="head")
        PR.rest_pose(r, F0)
        rack[bid] = r
        specs.append(BH.BirdSpec(bid, (x, 2.95, z), -90 * D2R, kind="rack", group="rack"))
log("rigs: street %d, rack %d" % (len(street), len(rack)))

# flyer: take-off from the railing, loop out over the street, land on the pavement near the Subject's stop point
FLY = street["fly"]
fl = PR.flight(FLY, 20, [(0.9, 0.0, RAIL_Z), (1.4, -0.6, 1.7), (2.0, -0.3, 2.0), (2.0, 0.6, 1.3), (1.3, 0.75, 0.0)],
               speed=2.4, end_yaw=160 * D2R, seed=SEED)
log("flight", fl, "leg_L rot y at 33/60/90:", [round(KY.value(FLY.leg["L"], "rotation_euler", 1, f), 2) for f in (33, 60, 90)])
# st0 sidesteps along the railing away from st1 late (2 steps); blocked in the scheduler from 146 (positions assumed static)
PR.perch_shuffle(street["st0"], 150, steps=2, direction=-1)

# subject actions after the stop: look at the landed bird, sip coffee
f_turn = SJ.turn(s, f_stop + 1, "R", 90)        # face the street (camera side)
log("subject turn ends", f_turn)
SJ.look_at(s, f_turn + 1, f_turn + 18, (1.0, 0.95, 0.15), ramp=5, track=False)
f_sip = SJ.sip_coffee(s, f_turn + 14, hold=16)
log("subject sip ends", f_sip)
KY.flush()

# ------------------------------------------------------------------ behaviours (subject head track sampled from the rig)
track = SJ.head_track(s, F0, F1, step=3)
sched = BH.schedule(specs, F0, F1, seed=SEED, subject=track, force_observe=[(f_stop + 4, f_stop + 30)],
                    forced_links=[(3, "st1", "st0", 26), (36, "st2", "st1", 24), (70, "st2", "st0", 22)],
                    params=BH.Params(observe_every=(90, 160), p_link=0.6, link_hold=(20, 36)), log=log)
log("schedule", sched.stats())
BH.apply(sched, dict(street, **rack), blinks=True, log=log)
# the flyer before take-off: one look toward the street, then airborne; after landing: head tracks the Subject
PR.head_snap(FLY, 6, -40 * D2R, 0.2, 0.0)
f_w = PR.walk(FLY, fl["f_end"] + 4, [(1.3, 0.75, 0.0), (1.15, 0.85, 0.0), (0.95, 0.95, 0.0)])
log("flyer walk ends", f_w)
PR.head_track(FLY, f_w + 2, F1, lambda f: track[min(len(track) - 1, max(0, (f - F0) // 3))][1], seed=5)
# hero: TX eye turns to the camera, TX disable blink
cam_end = Vector((1.8, -3.75, 1.6))
fe, k, y, ro, ex = PR.head_look_at(hero, 148, cam_end, eye="tx", dur=3)
PR.tx_disable(hero, 156, 170)
PR.head_snap(hero, 172, y * 0.4, 0.0, 0.0)
KY.flush()
sched.events_json(os.path.join(OUT, "demo_link_events.json"))

# link beams (demo VFX only): thin emissive bars TX eye -> RX eye during each link event, CONSTANT visibility keys
n_beam = 0
for e in sched.links:
    a, b = Vector(e.tx_eye), Vector(e.rx_eye)
    d = b - a
    import bmesh
    me = bpy.data.meshes.new("beam")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.004, radius2=0.004, depth=d.length)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M_BEAM)
    o = bpy.data.objects.new("beam_%03d" % n_beam, me)
    sc.collection.objects.link(o)
    o.matrix_world = Matrix.Translation((a + b) / 2) @ d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    for f, v in ((F0, 0.0), (e.f0, 1.0), (e.f1, 0.0)):
        o.scale = (v, v, v)
        o.keyframe_insert("scale", frame=f)
    for fc in o.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    n_beam += 1
log("beams", n_beam)

# ------------------------------------------------------------------ camera + beam note
cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
tgt = bpy.data.objects.new("cam_tgt", None)
sc.collection.objects.link(tgt)
con = cam.constraints.new("TRACK_TO")
con.target = tgt
con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
def flyer_pos(f):
    return Vector(KY.values(FLY.root, "location", f))


cam_keys = [(F0, Vector((0.9, -2.35, 1.35)), Vector((0.1, 0.0, 1.08)), 32), (22, Vector((1.0, -2.4, 1.38)), Vector((0.5, 0.0, 1.1)), 32)]
for f in range(32, 103, 6):            # follow the flyer: constant offset (bird stays mid-frame), low on the landing
    fp = flyer_pos(min(f, 100))
    off = Vector((0.35, -2.2, 0.0))
    cp = fp + off
    cp.z = max(0.55, fp.z - 0.05)
    cam_keys.append((f, cp, fp + Vector((0.0, 0.0, 0.08)), 35))
cam_keys += [(122, Vector((1.9, -3.9, 1.6)), Vector((0.2, 0.6, 0.95)), 34), (F1, Vector((1.7, -3.6, 1.6)), Vector((0.2, 0.6, 1.05)), 36)]
for f, p, t, lens in cam_keys:
    cam.location = p
    cam.keyframe_insert("location", frame=f)
    tgt.location = t
    tgt.keyframe_insert("location", frame=f)
    cd.lens = lens
    cd.keyframe_insert("lens", frame=f)
cd.clip_start = 0.01
note = bpy.data.curves.new("note", "FONT")
note.body = "* beams invisible (1311 nm); rendered for viewing"
note.size = 0.00095
no = bpy.data.objects.new("beam_note", note)
sc.collection.objects.link(no)
no.parent = cam
for f, p, t, lens in cam_keys:      # keep the note at the bottom-left corner as the lens changes (half width 0.025*36/lens)
    hw = 0.03 * 18.0 / lens
    no.location = (-0.94 * hw, -0.94 * hw, -0.03)
    no.keyframe_insert("location", frame=f)
no.data.materials.append(mat("note", (0.9, 0.9, 0.9), 0.5, 0.0, 1.0))
vis = [0.0] * (F1 + 2)
for e in sched.links:
    for f in range(int(e.f0), int(e.f1)):
        if F0 <= f <= F1:
            vis[f] = 1.0
prev = None
for f in range(F0, F1 + 1):
    if vis[f] != prev:
        no.scale = (vis[f],) * 3
        no.keyframe_insert("scale", frame=f)
        prev = vis[f]
for fc in no.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "CONSTANT"

os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "demo_characters.blend"), compress=True)
log("saved blend")

# ------------------------------------------------------------------ render
if MODE == "flystills":
    cam.constraints.clear()
    for o in (cam, tgt):
        o.animation_data_clear()
    cd.animation_data_clear()
    cd.lens = 50
    for f in STILLS:
        sc.frame_set(f)
        fp = FLY.root.matrix_world.translation.copy()
        cam.location = fp + Vector((0.9, -1.2, 0.35))
        cam.rotation_euler = (fp - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(OUT, "work", "fly_f%03d.png" % f)
        bpy.ops.render.render(write_still=True)
        log("fly still", f)
elif MODE == "stills":
    for f in STILLS:
        sc.frame_set(f)
        sc.render.filepath = os.path.join(OUT, "demo_still_f%03d.png" % f)
        bpy.ops.render.render(write_still=True)
        log("still", f)
elif MODE == "anim":
    d = os.path.join(OUT, "demo_frames")
    os.makedirs(d, exist_ok=True)
    sc.render.filepath = os.path.join(d, "f_")
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
    log("anim rendered", d)
log("DONE")
