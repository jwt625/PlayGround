"""Previews and assembly tests of the NPC v2 family through scripts/film_v1/asm.py (every preview appends the asset
the way a scene does). EEVEE, 16 samples; tiles rendered at 540x675.

Blender -b --python preview_npc_v2.py -- <out_dir> <mode> [variants comma list]
modes:
  views    front and three-quarter of each variant (<id>_front.png, <id>_three_quarter.png, 540x675)
  lineup   all variants side by side at the same scale, daylight floor and a dark data-hall floor (npc_v2_lineup*.png)
  compare  v1 (left) and v2 (right) of each variant, one tile each, tiled into npc_v2_compare.png
  asm      per variant: asm.walk, then idle, punch_loop (fight), topple_back (fall) as NLA strips; stills at
           walk / idle / punch / fall tiled into <id>_asm_test.png (tiles downsampled 2x for size)
Temp tiles go to PV_TMP (env, default <out_dir>/_tmp) and are removed.
"""
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "scripts", "film_v1"))
import asm  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
OUT, MODE = args[0], args[1]
ALL = ["npc", "npc_molexx", "npc_nubiss", "npc_terahop", "npc_ayarr", "npc_nvydia", "npc_openay"]
VARS = args[2].split(",") if len(args) > 2 and args[2] else ALL
os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get("PV_TMP", os.path.join(OUT, "_tmp"))
os.makedirs(TMP, exist_ok=True)
W, H = 540, 675


def base_scene(w=W, h=H, dark=False):
    scn = asm.new_scene(preset="standard", w=w, h=h, frames=300, world=None if dark else "WORLD_cartoon_sky")
    scn.eevee.taa_render_samples = 16
    scn.render.image_settings.compression = 90
    if dark:
        wd = bpy.data.worlds.new("dark")
        wd.use_nodes = True
        wd.node_tree.nodes["Background"].inputs[0].default_value = (0.010, 0.012, 0.016, 1)
        scn.world = wd
        asm.rig("data_hall")
        col = (0.035, 0.037, 0.042, 1)
    else:
        asm.rig("daylight")
        col = (0.42, 0.45, 0.40, 1)
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    fl = bpy.context.active_object
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = col
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
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


def tile(paths, cols, out, tw=W, th=H, down=1):
    rows = (len(paths) + cols - 1) // cols
    ow, oh = tw // down, th // down
    arr = np.zeros((rows * oh, cols * ow, 4), np.float32)
    arr[..., 3] = 1
    for k, p in enumerate(paths):
        im = bpy.data.images.load(p)
        px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
        if down > 1:
            px = px[:oh * down, :ow * down].reshape(oh, down, ow, down, 4).mean(axis=(1, 3))
        r, c = divmod(k, cols)
        y0 = (rows - 1 - r) * oh
        arr[y0:y0 + oh, c * ow:(c + 1) * ow] = px
        bpy.data.images.remove(im)
    img = bpy.data.images.new("sheet", cols * ow, rows * oh, alpha=False)
    img.pixels = arr.ravel().tolist()
    img.filepath_raw = out
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    for p in paths:
        os.remove(p)


def stature(a):
    return float(a.root.get("p_stature_m", 1.75))


if MODE == "views":
    for var in VARS:
        base_scene()
        a = asm.append("characters/%s_v2" % var, actions=False)
        asm.place(a.root, (0, 0, 0))
        s = stature(a)
        zc = s * 0.52
        d = s * 2.05
        for nm, v in (("front", (0.0, -1.0, 0.08)), ("three_quarter", (0.72, -0.90, 0.20))):
            v = np.array(v) / np.linalg.norm(v)
            cam_to((v[0] * d, v[1] * d, zc + v[2] * d), (0, 0, zc), 50)
            render(os.path.join(OUT, "%s_v2_%s.png" % (var, nm)))
        if os.environ.get("PV_CLOSE"):
            fz = float(a.root.get("pv_face_z", s * 0.9))
            fd = float(a.root.get("pv_face_dist", 1.2))
            cam_to((0.35 * fd, -1.25 * fd, fz + 0.02), (0, 0, fz - 0.06), 50)
            render(os.path.join(TMP, "%s_v2_close.png" % var))
            cam_to((-1.3 * fd, 0.25 * fd, fz + 0.02), (0, 0, fz - 0.06), 50)
            render(os.path.join(TMP, "%s_v2_side.png" % var))
            cam_to((0.5 * d, 0.85 * d, zc + 0.2 * d), (0, 0, zc), 50)
            render(os.path.join(TMP, "%s_v2_back.png" % var))
            # (close/side/back stay in TMP for inspection; TMP must be a scratch directory)

elif MODE == "lineup":
    for dark in (False, True):
        base_scene(1620, 675, dark=dark)
        xs = np.linspace(-3.0, 3.0, len(VARS))
        for x, var in zip(xs, VARS):
            a = asm.append("characters/%s_v2" % var, actions=False)
            asm.place(a.root, (float(x), 0, 0), yaw=0.18 if x < 0 else -0.18)
        cam_to((0.0, -9.0, 1.25), (0.0, 0.0, 0.92), 38)
        render(os.path.join(OUT, "npc_v2_lineup%s.png" % ("_dark" if dark else "")))

elif MODE == "compare":
    paths = []
    for var in VARS:
        base_scene()
        a1 = asm.append("characters/%s" % var, actions=False)
        a2 = asm.append("characters/%s_v2" % var, actions=False)
        asm.place(a1.root, (-0.62, 0, 0))
        asm.place(a2.root, (0.62, 0, 0))
        s = max(stature(a1), 1.6)
        cam_to((0.0, -3.3 * s / 1.75, 1.0 * s / 1.75), (0.0, 0.0, 0.95 * s / 1.75), 40)
        p = os.path.join(TMP, "cmp_%s.png" % var)
        render(p)
        paths.append(p)
    tile(paths, 4, os.path.join(OUT, "npc_v2_compare.png"), down=1)

elif MODE == "asm":
    for var in VARS:
        base_scene()
        a = asm.append("characters/%s_v2" % var, actions=True)
        asm.place(a.root, (0, 0, 0))
        t = asm.walk(a, 0.0, [(-1.4, 0.0, 0), (0.0, 0.0, 0)], gait="walk")
        a.root.rotation_euler = (0, 0, 0)
        asm.play(a, "idle", t, hold=False)
        asm.play(a, "punch_loop", t + 1.2, hold=False, repeat=2)
        asm.play(a, "topple_back", t + 3.2)
        # face the camera after the walk
        asm.key_loc(a.root, t + 0.05, (0, 0, 0), yaw=0.0)
        tl = asm.action_len(a, "topple_back")
        times = [("walk", t * 0.5), ("idle", t + 0.6), ("punch", t + 1.2 + 0.20), ("fall", t + 3.2 + tl + 0.2)]
        s = stature(a)
        paths = []
        for lab, tt in times:
            bpy.context.scene.frame_set(asm.F(tt))
            c = a.root.matrix_world.translation
            if lab == "fall":
                cam_to((c.x + 2.6 * s / 1.75, c.y - 3.2 * s / 1.75, 1.3 * s / 1.75), (c.x, c.y + 0.6 * s / 1.75, 0.3), 40)
            else:
                cam_to((c.x + 1.4 * s / 1.75, c.y - 4.0 * s / 1.75, 1.05 * s / 1.75), (c.x, c.y, 0.92 * s / 1.75), 40)
            p = os.path.join(TMP, "asm_%s_%s.png" % (var, lab))
            render(p)
            paths.append(p)
        tile(paths, 4, os.path.join(OUT, "%s_v2_asm_test.png" % var), down=2)
