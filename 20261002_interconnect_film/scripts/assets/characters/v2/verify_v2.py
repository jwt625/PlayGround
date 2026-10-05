"""Numeric compatibility and budget checks of a v2 character against its v1 asset.

Blender -b --python verify_v2.py -- <gary|manager> [json_out]

Checks: bones, hooks, root props, shape-key objects and key names, hole props, action name mapping, v1 action retarget
(v1 action played on the v2 armature: bone world positions vs the v1 rig), evaluated triangles (render subdivision),
head height ratios, squash/jiggle drivers.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "scripts", "film_v1"))
import asm  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
CID = args[0]
JOUT = args[1] if len(args) > 1 else None
COMP = os.path.join(PROJ, "assets", "components", "characters")
res = {}

asm.new_scene(preset="draft", w=64, h=80)
a1 = asm.append("characters/%s" % CID, actions=True)
a2 = asm.append("characters/%s_v2" % CID, actions=True)
j1 = json.load(open(os.path.join(COMP, CID + ".json")))
j2 = json.load(open(os.path.join(COMP, CID + "_v2.json")))

# ---- names
b1, b2 = set(j1["bones"]), set(j2["bones"])
res["bones_missing_in_v2"] = sorted(b1 - b2)
res["bones_added_in_v2"] = sorted(b2 - b1)
h1 = {h["name"] for h in j1["hooks"]}
h2 = {h["name"] for h in j2["hooks"]}
res["hooks_missing_in_v2"] = sorted(h1 - h2)
res["hooks_added_in_v2"] = sorted(h2 - h1)
p1, p2 = set(j1["root_custom_properties"]), set(j2["root_custom_properties"])
res["root_props_missing_in_v2"] = sorted(p1 - p2)
res["root_props_added_in_v2"] = sorted(p2 - p1)
res["shape_key_objects_equal"] = sorted(j1["shape_key_objects"]) == sorted(j2["shape_key_objects"]) or sorted(set(j2["shape_key_objects"]) - set(j1["shape_key_objects"]))
res["shape_keys_equal"] = j1["shape_keys_on_face_objects"] == j2["shape_keys_on_face_objects"]
res["holes_v1"] = [(h["name"], h["root_property"]) for h in j1["holes"]]
res["holes_v2"] = [(h["name"], h["root_property"]) for h in j2["holes"]]
n1 = {a["name"][len("ACT_%s_" % CID):] for a in j1["actions"]}
n2 = {a["name"][len("ACT_%s_v2_" % CID):] for a in j2["actions"]}
res["actions_missing_in_v2"] = sorted(n1 - n2)
res["actions_count_v1_v2"] = [len(n1), len(n2)]

# ---- bone hierarchy equality (parent names of every v1 bone)
arm1, arm2 = a1.armature, a2.armature
par1 = {b.name: (b.parent.name if b.parent else None) for b in arm1.data.bones}
par2 = {b.name: (b.parent.name if b.parent else None) for b in arm2.data.bones}
res["hierarchy_diff"] = sorted(k for k in par1 if par2.get(k, "MISSING") != par1[k])

# ---- rest-pose bone head/tail differences (world, m)
rest_diff = {}
for b in arm1.data.bones:
    if b.name in arm2.data.bones:
        c = arm2.data.bones[b.name]
        dh = (b.head_local - c.head_local).length
        dt = (b.tail_local - c.tail_local).length
        if dh > 1e-4 or dt > 1e-4:
            rest_diff[b.name] = [round(dh, 4), round(dt, 4)]
res["rest_pose_bones_moved(head,tail m)"] = rest_diff

# ---- v1 actions played on the v2 armature: bone world positions vs the v1 rig (same root transform at the origin)
names = ["idle", "walk", "run", "point", "aim_gun", "shout_loop", "topple_back", "jolt_hit", "pull_cable", "hug_leg", "punch_loop", "slap", "kneel_and_tie"]
probe = ["hand_L", "hand_R", "foot_L", "foot_R", "forearm_L", "forearm_R", "shin_L", "shin_R", "chest"]
dev = {}
worst = 0.0
for nm in names:
    act1 = bpy.data.actions.get("ACT_%s_%s" % (CID, nm))
    if act1 is None:
        continue
    arm1.animation_data_create().action = act1
    arm2.animation_data_create().action = act1          # v1 action on the v2 rig (retarget by bone name)
    f0, f1 = act1.frame_range
    mx = 0.0
    for f in np.linspace(f0, f1, 6):
        bpy.context.scene.frame_set(int(round(f)))
        bpy.context.view_layer.update()
        for b in probe:
            p = arm1.pose.bones[b].head
            q = arm2.pose.bones[b].head
            P = arm1.matrix_world @ p
            Q = arm2.matrix_world @ q
            d = (P - Q).length
            mx = max(mx, d)
        # head bone: compare tails (the head pivot was deliberately moved down)
        P = arm1.matrix_world @ arm1.pose.bones["head"].tail
        Q = arm2.matrix_world @ arm2.pose.bones["head"].tail
        mx = max(mx, (P - Q).length)
    dev[nm] = round(mx, 4)
    worst = max(worst, mx)
res["v1_action_on_v2_max_bone_head_deviation_m"] = dev
res["v1_action_on_v2_worst_m"] = round(worst, 4)
arm1.animation_data.action = None
arm2.animation_data.action = None
for arm_ in (arm1, arm2):
    for pb in arm_.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()

# ---- evaluated triangles with all modifiers at render subdivision
dg = bpy.context.evaluated_depsgraph_get()
def tris(coll_asset):
    n = 0
    for o in coll_asset.objs:
        if o.type != "MESH":
            continue
        for m in o.modifiers:
            if m.type == "SUBSURF":
                m.levels = m.render_levels
        bpy.context.view_layer.update()
        e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = e.to_mesh()
        n += sum(len(p.vertices) - 2 for p in me.polygons)
        e.to_mesh_clear()
    return n
res["evaluated_tris_v1"] = tris(a1)
res["evaluated_tris_v2"] = tris(a2)

# ---- head ratio (evaluated head mesh + hair, no hat) and stature
def zrange(asset, names_):
    zs = []
    for o in asset.objs:
        if o.type == "MESH" and any(o.name.endswith("_" + n) for n in names_):
            e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = e.to_mesh()
            for v in me.vertices:
                zs.append((o.matrix_world @ v.co).z)
            e.to_mesh_clear()
    return (min(zs), max(zs)) if zs else None
for tag, a in (("v1", a1), ("v2", a2)):
    hz = zrange(a, ["head"])
    allz = []
    for o in a.objs:
        if o.type == "MESH" and "_rim" not in o.name and not o.name.endswith("_hardhat"):
            e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = e.to_mesh()
            for v in me.vertices:
                allz.append((o.matrix_world @ v.co).z)
            e.to_mesh_clear()
    top = max(allz)
    res["head_%s" % tag] = dict(head_object_z_min=round(hz[0], 4), head_object_z_max=round(hz[1], 4), head_height_m=round(hz[1] - hz[0], 4),
                                 crown_without_hat_m=round(top, 4), head_height_over_crown=round((hz[1] - hz[0]) / top, 4),
                                 one_over=round(top / (hz[1] - hz[0]), 2))
# head height excluding the lowest chin: use skull crown from root stature property
res["stature_prop_v2"] = a2.root.get("p_stature_m")

# ---- squash drivers
r = a2.root
base_head = zrange(a2, ["head"])
r["p_squash_head"] = 0.5
r.update_tag()
bpy.context.view_layer.update()
sq = zrange(a2, ["head"])
r["p_squash_head"] = 0.0
r["p_squash"] = -0.4
r.update_tag()
bpy.context.view_layer.update()
allz = []
for o in a2.objs:
    if o.type == "MESH" and "_rim" not in o.name:
        e = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = e.to_mesh()
        allz += [(o.matrix_world @ v.co).z for v in me.vertices]
        e.to_mesh_clear()
res["squash_head_0.5_head_height_m"] = [round(base_head[1] - base_head[0], 4), round(sq[1] - sq[0], 4)]
res["squash_-0.4_top_m"] = round(max(allz), 4)
r["p_squash"] = 0.0
print("VERIFY", json.dumps(res, indent=1))
if JOUT:
    json.dump(res, open(JOUT, "w"), indent=1)
