"""Scale test: 200 head-only rack birds + 20 full street birds, 62 s schedule, apply, timing and key counts (no render).
scripts/bslot.sh -b --factory-startup --python scripts/anim/test_crowd_scale.py
"""
import math
import os
import random
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behaviors as BH  # noqa: E402
import pigeon_rig as PR  # noqa: E402

D2R = math.pi / 180
t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
lib = PR.proxy_library()
rng = random.Random(3)
rigs, specs = {}, []
for side, x, yaw in (("L", -1.16, 0.0), ("R", 1.16, math.pi)):
    for k in range(10):
        for row in range(10):
            bid = "%s%02d_%d" % (side, k, row)
            pos = (x, 1.4 + 0.8 * k + 0.24 * (row % 3 - 1), 0.2 + 0.21 * row)
            rigs[bid] = PR.make_proxy_rig(bid, lib, pos, yaw, mode="head")
            specs.append(BH.BirdSpec(bid, pos, yaw, group="rack"))
for k in range(20):
    bid = "st%02d" % k
    pos = (rng.uniform(-4, 4), 18 + rng.uniform(-3, 3), rng.uniform(0, 5))
    y = rng.uniform(-3, 3)
    r = PR.make_proxy_rig(bid, lib, pos, y)
    PR.rest_pose(r, 1)
    PR.place(r, 1, pos, y)
    rigs[bid] = r
    specs.append(BH.BirdSpec(bid, pos, y, kind="street", yaw_lim=110 * D2R, roll_lim=50 * D2R, group="street"))
t1 = time.time()
subj = BH.Track([(1, (0.0, 0.0, 1.55)), (900, (0.0, 12.0, 1.55)), (1860, (0.0, 22.0, 1.55))])
sc = BH.schedule(specs, 1, 1861, seed=7, subject=subj, force_observe=[(1650, 1720)])
t2 = time.time()
BH.apply(sc, rigs, breathe=True)
t3 = time.time()
nkeys = sum(len(fc.keyframe_points) for a in bpy.data.actions for fc in a.fcurves)
nobj = len(bpy.data.objects)
bpy.context.scene.frame_set(900)
t4 = time.time()
for f in range(900, 930):
    bpy.context.scene.frame_set(f)
t5 = time.time()
r0 = PR.PigeonRig.from_root(rigs["st00"].root)
print("SCALE rigs %d objects %d build %.1fs schedule %.1fs apply %.1fs keys %d frame_set %.3fs/frame from_root %s %s"
      % (len(rigs), nobj, t1 - t0, t2 - t1, t3 - t2, nkeys, (t5 - t4) / 30, r0.head.name, sorted(r0.wopen)), sc.stats())
