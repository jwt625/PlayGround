"""Compatibility test of the v2 characters inside the film assembly framework (scripts/film_v1/asm.py).

Blender -b --python test_v2_asm.py -- <out_dir> <gary|manager> <scene|compare|sheet>

scene   : v2 asset appended with actions, NLA sequence (walk, idle, point, hit reaction, fall) like the scene scripts do,
          shotgun on HOOK_gun_grip_R (manager) or HOOK_hand_R (gary); stills at 540x675, EEVEE 16 samples.
compare : v1 (left) and v2 (right) in the same scene, same pose list, one still per pose (540x675); the stills are tiled
          into <cid>_compare_NN.png sheets.
Writes small PNGs only to <out_dir>; temp tiles are removed.
"""
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
OUT, CID, MODE = args[0], args[1], args[2]
os.makedirs(OUT, exist_ok=True)
W, H = 540, 675
PI = math.pi


def base_scene():
    scn = asm.new_scene(preset="standard", w=W, h=H, frames=300)
    scn.eevee.taa_render_samples = 16
    asm.rig("daylight")
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    fl = bpy.context.active_object
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.42, 0.45, 0.40, 1)
    fl.data.materials.append(m)
    return scn


def cam_to(loc, tgt, lens=40):
    cam = bpy.data.objects["CAM"]
    t = bpy.data.objects["CAM_TARGET"]
    cam.data.lens = lens
    cam.location = loc
    t.location = tgt
    bpy.context.view_layer.update()


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def tile(paths, cols, out):
    rows = (len(paths) + cols - 1) // cols
    arr = np.zeros((rows * H, cols * W, 4), np.float32)
    arr[..., 3] = 1
    for k, p in enumerate(paths):
        im = bpy.data.images.load(p)
        px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
        r, c = divmod(k, cols)
        y0 = (rows - 1 - r) * H
        arr[y0:y0 + H, c * W:(c + 1) * W] = px
        bpy.data.images.remove(im)
    img = bpy.data.images.new("sheet", cols * W, rows * H, alpha=False)
    img.pixels = arr.ravel().tolist()
    img.filepath_raw = out
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def scene_test():
    base_scene()
    a = asm.append("characters/%s_v2" % CID, actions=True)
    asm.place(a.root, (0, 0, 0), yaw=0.0)
    gun = None
    if CID == "manager":
        gun = asm.append("props/shotgun")
        gun.variant("clay", True)
        asm.attach(gun.root, a.hook("gun_grip_R"), rot=(0, 0, PI))
    else:
        gun = asm.append("props/shotgun")
        gun.variant("clay", True)
        asm.attach(gun.root, a.hook("hand_R"), rot=(0, 0, PI))
    # NLA sequence: walk 0..2.0 s along -y... (the root walks), idle, point, aim/shot, hit reaction, fall
    t = asm.walk(a, 0.0, [(-1.6, 1.2, 0), (0.0, 1.2, 0)], gait="walk")      # ends at x = 0
    asm.play(a, "idle", t, hold=False, repeat=1)
    asm.play(a, "point", t + 2.0)
    asm.play(a, "aim_gun", t + 3.8)
    asm.play(a, "jolt_hit", t + 6.0)
    asm.play(a, "topple_back", t + 7.2)
    times = [("walk", 1.0), ("point", t + 2.0 + 0.5), ("aim", t + 3.8 + 1.2), ("shot", t + 3.8 + 1.2 + 0.0), ("jolt", t + 6.0 + 0.2), ("fall", t + 7.2 + 1.9)]
    paths = []
    for lab, tt in times:
        fr = asm.F(tt)
        bpy.context.scene.frame_set(fr)
        ctr = a.root.location
        cam_to((ctr.x + 1.3, ctr.y - 4.3, 1.1), (ctr.x, ctr.y, 0.95), 40)
        p = os.path.join(OUT, "_tmp_%s_%s.png" % (CID, lab))
        render(p)
        paths.append(p)
    tile(paths, 3, os.path.join(OUT, "%s_v2_scene_test.png" % CID))
    for p in paths:
        os.remove(p)


COMPARE_POSES = {
    "gary": [("idle", 15), ("walk", 8), ("point", 16), ("aim_gun", 36), ("shout_loop", 8), ("topple_back", 32),
             ("jolt_hit", 10), ("pull_cable", 18), ("hug_leg", 18), ("throw_up", 11), ("thinking", 30), ("hold_plate", 15)],
    "manager": [("idle", 15), ("walk", 8), ("point", 16), ("aim_gun", 36), ("shout_loop", 8), ("topple_back", 32),
                ("punch_loop", 6), ("slap", 11), ("shove", 9), ("arm_whip", 19), ("hold_printout", 15), ("stagger", 9)],
}


def compare_test():
    base_scene()
    a1 = asm.append("characters/%s" % CID, actions=True)
    a2 = asm.append("characters/%s_v2" % CID, actions=True)
    asm.place(a1.root, (-0.85, 0, 0))
    asm.place(a2.root, (0.85, 0, 0))
    paths = []
    sheets = 0
    k = 0
    for nm, fr in COMPARE_POSES[CID]:
        a1.armature.animation_data_create().action = bpy.data.actions["ACT_%s_%s" % (CID, nm)]
        a2.armature.animation_data_create().action = bpy.data.actions["ACT_%s_v2_%s" % (CID, nm)]
        bpy.context.scene.frame_set(fr)
        if nm in ("topple_back",):
            cam_to((0.0, -5.2, 0.9), (0.0, 0.0, 0.75), 40)
        else:
            cam_to((0.0, -3.9, 1.1), (0.0, 0.0, 0.92), 40)
        p = os.path.join(OUT, "_tmp_cmp_%s_%s.png" % (CID, nm))
        render(p)
        paths.append(p)
        if len(paths) == 4:
            tile(paths, 2, os.path.join(OUT, "%s_compare_%02d.png" % (CID, sheets)))
            for q in paths:
                os.remove(q)
            paths = []
            sheets += 1
    if paths:
        tile(paths, 2, os.path.join(OUT, "%s_compare_%02d.png" % (CID, sheets)))
        for q in paths:
            os.remove(q)


if MODE == "scene":
    scene_test()
elif MODE == "compare":
    compare_test()
