"""Baseline: foot slide of the v1 gary walk/run (read-only on gary.blend). Blender -b gary.blend --python m2_measure_v1.py"""
import json, os, sys
import bpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2_measure as MS
root = next(o for o in bpy.data.objects if o.name.startswith("ROOT_"))
arm = bpy.data.objects[root["asset_id"] + "_rig"]
res = {}
for n in ("walk", "run"):
    act = bpy.data.actions["ACT_%s_%s" % (root["asset_id"], n)]
    N = int(act.frame_range[1])
    sp = float(act["root_speed_mps"])
    res[n] = MS.measure_gait(root, arm, act, N, sp)
    print("V1SLIDE", n, json.dumps(res[n]))
