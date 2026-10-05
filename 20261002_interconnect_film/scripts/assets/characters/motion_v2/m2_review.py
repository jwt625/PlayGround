"""Review renders: bake actions in the open character blend (not saved) and render contact strips.

Blender -b <rig>.blend --python m2_review.py -- <names|all> <out_dir> [rig_tag]
"""
import math
import os
import shutil
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motion_v2 as M  # noqa: E402
import m2_engine as E  # noqa: E402
import m2_lib as LIB  # noqa: E402
import m2_render as R  # noqa: E402
import m2_measure as MS  # noqa: E402
import json

argv = sys.argv[sys.argv.index("--") + 1:]
SETS = dict(
    manager="walk,walk_brisk,run,stomp_walk,turn_left_90,idle_breathe_tense,idle_arms_cross,point,point_accuse,shout_rant,anger_outburst,manager_slow_burn,gun_ready,gun_aim_hold,gun_fire,gun_raise_aim_fire,whiteboard_write,whiteboard_underline,thinking,haymaker,shove,collar_grab_shake,fall_back,shot_hit_fall,get_up,react_head_hit,startle,flinch".split(","),
    npc="walk,run,idle_breathe,haymaker,react_head_hit,fall_back_brawl,shove,chest_bump,hug_give,flail_windmill".split(","))
if argv[0] == "base":
    names = [n for n in LIB.ORDER if not n.endswith(("_L", "_upper")) and "_right_" not in n]
elif argv[0] in SETS:
    names = SETS[argv[0]]
else:
    names = LIB.ORDER if argv[0] == "all" else argv[0].split(",")
out = argv[1]
os.makedirs(out, exist_ok=True)
root = M.find_root()
aid = root["asset_id"]
tmp = os.path.join(os.environ.get("M2_TMP", "/tmp"), "m2_tiles_%s" % aid)
os.makedirs(tmp, exist_ok=True)
for o in bpy.data.objects:
    if o.type == "MESH":
        o.hide_viewport = True
rig = E.Rig2(root)
cam = R.setup_scene()
VIEW_BY = {k: "prof" for k in ("haymaker", "hook", "uppercut", "slap", "head_butt", "shove", "chest_bump", "collar_grab_shake", "whiff_overbalance", "grapple", "point", "point_accuse",
           "gun_aim_hold", "gun_fire", "gun_raise_aim_fire", "gun_ready", "typing", "whiteboard_write", "whiteboard_underline", "plug_connector", "react_head_hit", "react_gut_hit", "react_shoved", "react_slapped",
           "react_headbutt", "react_chest_bump", "duck", "kneel_down", "tie_fibers_kneel", "kneel_up")}
for k in ("fall_back", "fall_back_brawl", "shot_hit_fall", "get_up", "ground_twitch", "whipped"):
    VIEW_BY[k] = "wide34"
SHIFT_BY = {k: (0.0, 0.7) for k in ("fall_back", "fall_back_brawl", "shot_hit_fall", "get_up", "ground_twitch")}
acts = {}
MEAS = {}
for n in names:
    spec0 = M.build_spec(rig, n)
    act, info = M.bake_one(rig, n)
    acts[n] = (act, info, spec0)
# show meshes again for rendering
for o in bpy.data.objects:
    if o.type == "MESH" and not (o.name.startswith("PV_") or o.name.startswith("Plane")):
        o.hide_viewport = False
for n in names:
    act, info, spec = acts[n]
    rv = dict(spec.get("review", {}))
    N = info["frames"]
    loop = info["loop"]
    if loop:
        cnt = max(6, min(8, int(round(N / 4))))
        frames = [int(round(k * N / cnt)) for k in range(cnt)]
    else:
        step = rv.get("step", max(3, min(6, int(math.ceil(N / 16)))))
        cnt = rv.get("count", min(16, int(N / step) + 1))
        f0 = rv.get("start", 0)
        frames = [min(N, f0 + k * step) for k in range(cnt)]
    if n in ("fall_back", "fall_back_brawl"):
        rv["frames"] = [0, 3, 6, 9, 12, 16, 20, 24, 28, 34, 40, 50, 60, 72, 86, 104]
    if n == "shot_hit_fall":
        rv["frames"] = [0, 3, 5, 7, 9, 12, 15, 18, 21, 24, 28, 34, 44, 56, 70, 92]
    if n == "manager_slow_burn":
        rv["frames"] = [0, 30, 60, 90, 108, 118, 122, 126, 130, 134, 140, 149]
    if "frames" in rv:
        frames = rv["frames"]
    speed = float(info["meta"].get("root_speed_mps", 0.0))
    view = rv.get("view", "side" if speed else VIEW_BY.get(n.replace("_L", "").replace("_upper", ""), "front34"))
    shift = rv.get("shift", SHIFT_BY.get(n, (0.0, 0.0)))
    origin = (0.0, speed * (N / 30.0) / 2 if speed else 0.0)
    ax = (0, -1, 0)
    paths = R.render_frames(root, rig.arm, act, frames, os.path.join(tmp, n), view=view, focus_z=rv.get("focus_z", 0.9), speed=speed,
                            follow=rv.get("follow", not speed), origin=origin, travel_axis=ax, lens=rv.get("lens"), shift=shift, face=bpy.data.actions.get(info["face_action"]) if info["face_action"] else None)
    if speed and loop and rv.get("measure", True):
        res = MS.measure_gait(root, rig.arm, act, N, speed)
        print("SLIDE", n, json.dumps(res))
        MEAS[n] = res
    sz = M.compose_sheet(paths, os.path.join(out, "%s_%s.png" % (aid, n)), cols=4)
    print("SHEET", n, sz // 1000, "KB", frames)
    for p in paths:
        os.remove(p)
shutil.rmtree(tmp, ignore_errors=True)

mp = os.path.join(out, 'measure_%s.json' % aid)
old = json.load(open(mp)) if os.path.exists(mp) else {}
old.update(MEAS)
json.dump(old, open(mp, 'w'), indent=1)
