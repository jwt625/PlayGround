"""Gun consistency test (run on manager.blend or npc.blend): attach the shotgun prop like S1/S2 and measure/ render the muzzle line.
Blender -b manager.blend --python m2_guntest.py -- <out_dir>
"""
import json, math, os, sys
import bpy
import numpy as np
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motion_v2 as M
import m2_engine as E
import m2_render as R

out = sys.argv[sys.argv.index("--") + 1]
os.makedirs(out, exist_ok=True)
root = M.find_root()
aid = root["asset_id"]
for o in bpy.data.objects:
    if o.type == "MESH":
        o.hide_viewport = True
rig = E.Rig2(root)
names = ["gun_ready", "gun_aim_hold", "gun_fire", "gun_raise_aim_fire"]
acts = {n: M.bake_one(rig, n) for n in names}
# shotgun prop
p = os.path.join(M.COMP, "props", "shotgun.blend")
with bpy.data.libraries.load(p, link=False) as (src, dst):
    dst.collections = [c for c in src.collections if c.startswith("ASSET_")][:1]
gc = dst.collections[0]
bpy.context.scene.collection.children.link(gc)
objs = []
def walk(c):
    objs.extend(c.objects)
    for ch in c.children:
        walk(ch)
walk(gc)
groot = next(o for o in objs if o.name.startswith("ROOT_"))
for ch in gc.children_recursive:
    if ch.name.startswith("VARIANT_wood"):
        ch.hide_render = True; ch.hide_viewport = True
hook = bpy.data.objects["HOOK_gun_grip_R"]
groot.parent = hook
groot.matrix_parent_inverse.identity()
groot.location = (0, 0, 0)
groot.rotation_euler = (0, 0, math.pi)
muz = [o for o in objs if o.name.startswith("HOOK_muzzle_L")][0]
cam = R.setup_scene()
for o in bpy.data.objects:
    if o.type == "MESH" and not o.name.startswith(("PV_", "Plane")):
        o.hide_viewport = False
scn = bpy.context.scene
rep = {}
for n in names:
    act, info = acts[n]
    rig.arm.animation_data_create().action = act
    dirs, pos = [], []
    for f in range(0, info["frames"] + 1):
        scn.frame_set(f)
        bpy.context.view_layer.update()
        m = muz.matrix_world
        d = np.array(m.to_3x3() @ Vector((0, 0, 1)))
        # character frame: root rotation is identity here
        dirs.append(d); pos.append(np.array(m.translation))
    dirs = np.array(dirs)
    pitch = np.degrees(np.arcsin(dirs[:, 2]))
    yaw = np.degrees(np.arctan2(dirs[:, 0], -dirs[:, 1]))
    rep[n] = dict(pitch_deg_min=float(pitch.min()), pitch_deg_max=float(pitch.max()), yaw_deg_min=float(yaw.min()), yaw_deg_max=float(yaw.max()),
                  pitch_deg_first=float(pitch[0]), pitch_deg_last=float(pitch[-1]), muzzle_height_m=[float(np.min(np.array(pos)[:, 2])), float(np.max(np.array(pos)[:, 2]))],
                  pitch_series=[round(float(x), 1) for x in pitch[::2]], yaw_series=[round(float(x), 1) for x in yaw[::2]])
    print("GUN", n, json.dumps({k: v for k, v in rep[n].items() if "series" not in k}))
json.dump(rep, open(os.path.join(out, "gun_muzzle_%s.json" % aid), "w"), indent=1)
# strips
tmp = os.path.join(out, "_t")
os.makedirs(tmp, exist_ok=True)
for n, frames in (("gun_raise_aim_fire", [0, 6, 12, 16, 20, 40, 44, 46, 48, 50, 54, 62]), ("gun_fire", [0, 4, 6, 7, 8, 10, 14, 20])):
    act, info = acts[n]
    paths = R.render_frames(root, rig.arm, act, frames, os.path.join(tmp, n), view="prof", focus_z=1.0, follow=True)
    M.compose_sheet(paths, os.path.join(out, "%s_%s_withgun.png" % (aid, n)), cols=4)
    for q in paths:
        os.remove(q)
os.rmdir(tmp)
