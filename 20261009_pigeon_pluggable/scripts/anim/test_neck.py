"""Neck deform test (owner F1): extreme head poses on the hero rig, a full proxy and head-only proxies with tier neck
extensions; gap metrics (head rim and body rim vs the neck surface, deform on vs off), head-into-neck penetration;
stills grid; optional 20-bird scheduled clip.

scripts/bslot.sh -b --factory-startup --python scripts/anim/test_neck.py -- stills | crowd
Outputs under outputs/assets/characters/neck_test/ (sheet + mp4 kept, frames deleted by the caller).
"""
import math
import os
import sys
import time

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import behaviors as BH  # noqa: E402
import keys as KY  # noqa: E402
import pgeom as G  # noqa: E402
import pigeon_rig as PR  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT_DIR, "outputs", "assets", "characters", "neck_test")
os.makedirs(OUT, exist_ok=True)
MODE = (sys.argv[sys.argv.index("--") + 1:] or ["stills"])[0] if "--" in sys.argv else "stills"
D2R = math.pi / 180
T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 16
sc.render.resolution_x = sc.render.resolution_y = 540
sc.view_settings.view_transform = "AgX"
w = bpy.data.worlds.new("w")
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.26, 0.28, 1)
sc.world = w
for nm, rot, e in (("sun", (0.7, 0.2, 0.6), 3.0), ("sun2", (1.2, 0.0, 3.6), 1.2)):
    ld = bpy.data.lights.new(nm, "SUN")
    ld.energy = e
    lo = bpy.data.objects.new(nm, ld)
    lo.rotation_euler = rot
    sc.collection.objects.link(lo)
cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(cam)
sc.camera = cam
cd.clip_start = 0.005
lib = PR.proxy_library(log=log)
log("proxy parts", sorted(lib))

POSES = [("rest", 0, 0, 0, 0), ("yaw+110", 110, 0, 0, 0), ("yaw-110", -110, 0, 0, 0), ("yaw+150_roll+35", 150, 35, 0, 0),
         ("yaw-150_roll-35", -150, -35, 0, 0), ("roll+60_nod+20", 0, 60, 20, 0), ("yaw+100_roll-40_bob", 100, -40, -10, 0.004)]


def rim_sets(r, head_objs, body_objs, neck_objs, thr=0.0025):
    """Rest vertex sets (world) of head / body surfaces within thr (hero m) of the neck surface."""
    sc.frame_set(1)
    dg = bpy.context.evaluated_depsgraph_get()
    s = r.root.scale[0]

    def pts(objs):
        out = []
        for o in objs:
            ev = o.evaluated_get(dg)
            m = ev.to_mesh()
            out += [(o, i, ev.matrix_world @ v.co) for i, v in enumerate(m.vertices)]
            ev.to_mesh_clear()
        return out
    neck = pts(neck_objs)
    kd = KDTree(len(neck))
    for i, (_, _, p) in enumerate(neck):
        kd.insert(p, i)
    kd.balance()
    res = {}
    for key, objs in (("head", head_objs), ("body", body_objs)):
        res[key] = [(o, i) for o, i, p in pts(objs) if kd.find(p)[2] < thr * s]
    return res


def gap(r, rims, neck_objs, f, deform=True):
    """Max distance (hero mm) from the rest-contact rim vertices to the posed neck surface, per rim."""
    mods = [m for o in neck_objs for m in o.modifiers if m.type == "ARMATURE"]
    for m in mods:
        m.show_viewport = deform
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    s = r.root.scale[0]
    npts = []
    for o in neck_objs:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        npts += [ev.matrix_world @ v.co for v in m.vertices]
        ev.to_mesh_clear()
    kd = KDTree(len(npts))
    for i, p in enumerate(npts):
        kd.insert(p, i)
    kd.balance()
    out = {}
    for key, lst in rims.items():
        byo = {}
        for o, i in lst:
            byo.setdefault(o, []).append(i)
        dmax = 0.0
        for o, idx in byo.items():
            ev = o.evaluated_get(dg)
            m = ev.to_mesh()
            for i in idx:
                dmax = max(dmax, kd.find(ev.matrix_world @ m.vertices[i].co)[2])
            ev.to_mesh_clear()
        out[key] = round(dmax / s * 1000, 2)
    # head-into-neck penetration: posed lower-neck vertices (rest t < 0.85) inside the posed head box (head frame)
    hinv = r.head.matrix_world.inverted()
    pen = 0
    for o in neck_objs:
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        rest = o.data.vertices
        for v, v0 in zip(m.vertices, rest):
            if G.neck_t(tuple(_bird(r, o, v0.co))) < 0.85:
                q = hinv @ (ev.matrix_world @ v.co)
                hp = Vector(G.HEAD_PIVOT)
                if G.head_dist(tuple(q + hp)) < -0.0008:
                    pen += 1
        ev.to_mesh_clear()
    out["pen"] = pen
    for m in mods:
        m.show_viewport = True
    return out


def _bird(r, o, co):
    """Rest bird-frame position (hero m) of a mesh vertex (rig root at its rest transform)."""
    M = r.root.matrix_world.inverted() @ o.matrix_world
    return M @ co


def pose_keys(r, f0):
    for i, (nm, y, ro, nd, b) in enumerate(POSES):
        PR.head_pose(r, f0 + i, y * D2R, ro * D2R, nd * D2R, b, "CONSTANT")


if MODE == "stills":
    hero = PR.build_hero_rig("HERO", loc=(0, 0, 0), yaw=0.0, log=log)
    pfull = PR.make_proxy_rig("PF", lib, (0.8, 0, 0), 0.0)
    pext = PR.make_proxy_rig("PX", lib, (1.6, 0, 0), 0.0, mode="head", neck_ext=0.126)
    pmid = PR.make_proxy_rig("PM", lib, (2.4, 0, 0), 0.0, mode="head", neck_ext=0.046)
    rigs = [("hero", hero), ("proxy", pfull), ("proxy_ext126", pext), ("proxy_ext046", pmid)]
    for _, r in rigs:
        PR.rest_pose(r, 1)
        pose_keys(r, 1)
    KY.flush()
    sc.frame_start, sc.frame_end = 1, len(POSES)
    report = {}
    for nm, r in rigs:
        if nm == "hero":
            neck_objs = [o for o in r.glb if o.name.startswith(PR.NECK_SKIN)]
            head_objs = [o for o in r.glb if o.name.startswith(("skin_head_cranium", "skin_head_jaw"))]
            body_objs = [o for o in r.glb if o.name.startswith(("skin_back_body", "skin_belly_body"))]
        else:
            kids = [o for o in bpy.data.objects if o.name.startswith(r.name + "_m_")]
            neck_objs = [o for o in kids if o.name.endswith("_m_neck")]
            head_objs = [o for o in kids if o.name.endswith("_m_head")]
            body_objs = [o for o in kids if o.name.endswith(("_m_body", "_m_rest"))]
        rims = rim_sets(r, head_objs, body_objs, neck_objs)
        rows = []
        for i, (pn, *_rest) in enumerate(POSES):
            new = gap(r, rims, neck_objs, 1 + i, True)
            old = gap(r, rims, neck_objs, 1 + i, False)
            rows.append((pn, new, old))
        report[nm] = rows
        log("GAP %s rim sizes head %d body %d" % (nm, len(rims["head"]), len(rims["body"])))
        for pn, new, old in rows:
            log("GAP %-14s %-20s deform: head %5.2f body %5.2f pen %3d | static neck: head %5.2f body %5.2f pen %3d"
                % (nm, pn, new["head"], new["body"], new["pen"], old["head"], old["body"], old["pen"]))
    # stills: tight neck close-ups (back and right side), deform on, then off for comparison
    shots = [1, 2, 4, 6]
    views = (("back", Vector((-0.20, 0.0, 0.07))), ("right", Vector((0.02, -0.21, 0.04))), ("left", Vector((0.02, 0.21, 0.04))))

    def snap(tag):
        for nm, r in rigs[:3]:
            s_ = r.root.scale[0]
            for f in shots:
                sc.frame_set(f)
                hc = r.root.matrix_world @ Vector((0.034, 0.0, 0.066)) + Vector((0.0, 0.0, 0.0))
                if nm == "proxy_ext126":
                    hc = r.root.matrix_world @ Vector((0.05, 0.0, 0.066))
                for vn, off in views:
                    cam.location = hc + off * (s_ / 2.6)
                    cam.rotation_euler = (hc - cam.location).to_track_quat("-Z", "Y").to_euler()
                    cd.lens = 50
                    sc.render.filepath = os.path.join(OUT, "nk_%s_%s_f%d_%s.png" % (tag, nm, f, vn))
                    bpy.ops.render.render(write_still=True)
    snap("on")
    for o in bpy.data.objects:
        for m in o.modifiers:
            if m.type == "ARMATURE" and m.name == "neck":
                m.show_render = m.show_viewport = False
    snap("off")
    log("DONE")

elif MODE == "crowd":
    # 20 head-only birds on 4 tiers with the rack neck extensions, scheduled for 90 frames; gap sampled every 6 frames
    rigs, specs = {}, []
    exts = (0.0, 0.046, 0.126, 0.0)
    for row in range(4):
        for col in range(5):
            bid = "c%d%d" % (row, col)
            pos = (0.0 - 0.08 * row, -0.6 + 0.3 * col + (0.15 if row % 2 else 0.0), 0.2 + 0.25 * row)
            r = PR.make_proxy_rig(bid, lib, pos, 0.0, mode="head", neck_ext=exts[row])
            PR.rest_pose(r, 1)
            rigs[bid] = r
            specs.append(BH.BirdSpec(bid, pos, 0.0, group="rack", neck_ext=exts[row], yaw_lim=100 * D2R, roll_lim=40 * D2R))
    cam.location = (1.05, -0.35, 0.72)
    cam.rotation_euler = (Vector((-0.05, -0.12, 0.62)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cd.lens = 32
    subj = BH.Track([(1, (2.0, -1.5, 1.5)), (90, (2.0, 1.5, 1.5))])
    sch = BH.schedule(specs, 1, 90, seed=5, subject=subj, params=BH.Params(observe_every=(40, 80), p_link=0.5))
    BH.apply(sch, rigs)
    log("schedule", sch.stats())
    sc.frame_start, sc.frame_end = 1, 90
    worst = {"head": 0.0, "body": 0.0, "pen": 0}
    for bid in ("c00", "c11", "c22", "c33", "c24"):
        r = rigs[bid]
        kids = [o for o in bpy.data.objects if o.name.startswith(bid + "_m_")]
        neck_objs = [o for o in kids if o.name.endswith("_m_neck")]
        head_objs = [o for o in kids if o.name.endswith("_m_head")]
        body_objs = [o for o in kids if o.name.endswith("_m_rest")]
        rims = rim_sets(r, head_objs, body_objs, neck_objs)
        for f in range(1, 91, 6):
            g = gap(r, rims, neck_objs, f, True)
            for k in worst:
                worst[k] = max(worst[k], g[k])
    log("CROWD max over 5 birds x 15 frames: head rim gap %.2f mm, body rim gap %.2f mm, penetrating neck verts %d (hero mm)"
        % (worst["head"], worst["body"], worst["pen"]))
    t = time.time()
    for f in range(1, 91, 10):
        sc.frame_set(f)
    log("frame_set %.3f s/frame (20 birds)" % ((time.time() - t) / 9))
    sc.render.filepath = os.path.join(OUT, "crowd_frames", "f_")
    bpy.ops.render.render(animation=True)
    log("DONE")
