"""Feet contact check (owner v3): proxy birds on every crowd slot of R's hall (hall_layout.json), toe-sole vertices ray-cast
down onto the rack geometry. Reports per bird the lowest-toe gap (+ floating / - sunk) and the share of sole vertices
with floor under them (overhang). Mode 'fix' applies pigeon_rig.place_on_floor first. Optional close-up renders.

scripts/bslot.sh -b --factory-startup --python scripts/anim/test_feet.py -- measure|fix [render]
"""
import json
import math
import os
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pigeon_rig as PR  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
DC = os.path.join(ROOT_DIR, "assets", "components", "datacenter")
OUT = os.path.join(ROOT_DIR, "outputs", "assets", "characters", "feet_test")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE = argv[0] if argv else "measure"
RENDER = "render" in argv
T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
L = json.load(open(os.path.join(DC, "hall_layout.json")))
cols = {}
for tname, td in L["rack_types"].items():
    if tname != "bird_q3400":
        continue
    with bpy.data.libraries.load(os.path.join(ROOT_DIR, td["asset"]), link=False) as (src, dst):
        dst.collections = [td["collection"]]
    cols[tname] = dst.collections[0]
racks = []
for r in L["racks"]:
    if r["type"] not in cols:
        continue
    o = bpy.data.objects.new("rack_" + r["id"], None)
    o.instance_type = "COLLECTION"
    o.instance_collection = cols[r["type"]]
    o.location = r["location"]
    o.rotation_euler = (0, 0, math.radians(r["rotation_z_deg"]))
    sc.collection.objects.link(o)
    racks.append(o)
lib = PR.proxy_library(log=log)
cells = [c for c in L["cells"]["list"] if c["occupant"] == "crowd"]
rigs = []
for c in cells:
    bid = c["rack"] + "_" + c["bird_hook"]
    r = PR.make_proxy_rig(bid, lib, tuple(c["feet_origin"]), math.radians(c["bird_yaw_deg"]), mode="head", lids=False,
                          neck_ext=c["neck_extension_m"])
    PR.rest_pose(r, 1)
    rigs.append((c, r))
bpy.context.view_layer.update()
if MODE == "fix":
    bvh = PR.surfaces_bvh(racks)
    shifts = []
    for c, r in rigs:
        res_ = PR.place_on_floor(r, 1, bvh=bvh)
        shifts.append((c["tier"], res_["shift"], res_["dz"], res_["support"]))
    for tier in (0, 1, 2):
        sub = [x for x in shifts if x[0] == tier]
        log("FIX tier %d: back-shift %.0f-%.0f mm, dz %.2f..%.2f mm, support min %.0f %%" % (tier, min(x[1] for x in sub) * 1e3,
            max(x[1] for x in sub) * 1e3, min(x[2] for x in sub) * 1e3, max(x[2] for x in sub) * 1e3, 100 * min(x[3] for x in sub)))
    bpy.context.view_layer.update()
sc.frame_set(1)
dg = bpy.context.evaluated_depsgraph_get()
# toe-sole vertices (bird frame z < 1.2 mm hero, toes ahead of the leg) of the shared rest mesh
sole = PR.sole_points(rigs[0][1])
log("sole verts per bird", len(sole))
samples = []
for c, r in rigs:
    M = r.root.matrix_world
    samples.append([M @ v for v in sole])
for _, r in rigs:
    for o in r.objects:
        if o.type in ("MESH", "ARMATURE"):
            o.hide_viewport = True
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
res = []
for (c, r), pts in zip(rigs, samples):
    gaps, over = [], 0
    for p in pts:
        hit, loc, nrm, idx, ob, mat = sc.ray_cast(dg, p + Vector((0, 0, 0.03)), Vector((0, 0, -1)), distance=0.08)
        if not hit:
            over += 1
            continue
        gaps.append(p.z - loc.z)
    res.append((c, min(gaps) if gaps else None, max(gaps) if gaps else None, over / len(pts)))
lo = [x[1] for x in res if x[1] is not None]
log("BIRDS", len(res))
log("lowest-toe gap (mm, + floating / - sunk): min %.2f  max %.2f  mean %.2f" % (min(lo) * 1e3, max(lo) * 1e3, sum(lo) / len(lo) * 1e3))
for tier in (0, 1, 2):
    sub = [x for x in res if x[0]["tier"] == tier]
    g = [x[1] for x in sub if x[1] is not None]
    log("tier %d: n %d gap min %.2f max %.2f mm; sole share without floor: mean %.0f %% max %.0f %%"
        % (tier, len(sub), min(g) * 1e3, max(g) * 1e3, 100 * sum(x[3] for x in sub) / len(sub), 100 * max(x[3] for x in sub)))
bad = [x for x in res if x[1] is None or abs(x[1]) > 0.0015 or x[3] > 0.02]
log("birds failing (|gap| > 1.5 mm or > 2 %% of the sole without floor):", len(bad))
for x in bad[:6]:
    log("  e.g.", x[0]["rack"], x[0]["bird_hook"], "tier", x[0]["tier"], "gap", None if x[1] is None else round(x[1] * 1e3, 2), "no-floor %.0f %%" % (100 * x[3]))
if RENDER:
    os.makedirs(OUT, exist_ok=True)
    for _, r in rigs:
        for o in r.objects:
            if o.type in ("MESH",):
                o.hide_viewport = False
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.eevee.taa_render_samples = 16
    sc.render.resolution_x, sc.render.resolution_y = 540, 674
    sc.view_settings.view_transform = "AgX"
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.2, 0.21, 0.23, 1)
    sc.world = w
    for nm, rot, e in (("sun", (0.6, 0.3, 1.2), 3.0), ("sun2", (1.1, 0.0, -1.9), 1.0)):
        ld = bpy.data.lights.new(nm, "SUN")
        ld.energy = e
        lo_ = bpy.data.objects.new(nm, ld)
        lo_.rotation_euler = rot
        sc.collection.objects.link(lo_)
    cd = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cd.lens = 40
    cd.clip_start = 0.01
    pick = {}
    for c, r in rigs:
        if c["rack"] == "L01" and c["col"] in (0, 1) and (c["tier"], c["unit"]) not in pick:
            pick[(c["tier"], c["unit"])] = (c, r)
    for (tier, unit), (c, r) in sorted(pick.items())[:4]:
        p = Vector(c["feet_origin"])
        y = math.radians(c["bird_yaw_deg"])
        fwd = Vector((math.cos(y), math.sin(y), 0))
        side = Vector((-math.sin(y), math.cos(y), 0))
        tgt = p + fwd * 0.03 + Vector((0, 0, 0.03))
        cam.location = tgt + fwd * 0.42 + side * 0.30 + Vector((0, 0, 0.10))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(OUT, "feet_%s_t%d_u%d.png" % (MODE, tier, unit))
        bpy.ops.render.render(write_still=True)
    log("renders", OUT)
log("DONE")
