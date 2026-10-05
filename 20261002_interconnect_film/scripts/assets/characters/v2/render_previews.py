"""Preview renderer for character blends.

Usage: Blender -b <asset>.blend --python render_previews.py -- <asset_id> <out_dir> <modes comma list> [action_names comma list]
modes: views, faces, holes, poses
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
AID, OUT, MODES = argv[0], argv[1], argv[2].split(",")
os.makedirs(OUT, exist_ok=True)
scn = bpy.context.scene
root = bpy.data.objects["ROOT_" + AID]
arm = bpy.data.objects[AID + "_rig"]
SC = root.get("p_scale", 1.0)


def setup():
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = 16
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.compression = 100
    scn.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("PV_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.62, 0.70, 0.80, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    scn.world = w
    sun = bpy.data.lights.new("PV_sun", "SUN")
    sun.energy = 3.0
    so = bpy.data.objects.new("PV_sun", sun)
    so.rotation_euler = (0.9, 0.15, 0.55)
    scn.collection.objects.link(so)
    fill = bpy.data.lights.new("PV_fill", "AREA")
    fill.energy = 250
    fill.size = 3
    fo = bpy.data.objects.new("PV_fill", fill)
    fo.location = (-2.5, -3, 2)
    fo.rotation_euler = (1.2, 0, -0.6)
    scn.collection.objects.link(fo)
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "PV_floor"
    m = bpy.data.materials.new("PV_floor")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.45, 0.47, 0.44, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    fl.data.materials.append(m)
    cam = bpy.data.objects.new("PV_cam", bpy.data.cameras.new("PV_cam"))
    scn.collection.objects.link(cam)
    scn.camera = cam
    return cam, fo, so


cam, fill_o, sun_o = setup()


def shot(path, loc, tgt, lens=50, res=(900, 675)):
    cam.location = loc
    cam.data.lens = lens
    d = Vector(tgt) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)


def compose(tiles, cols, rows, tw, th, path):
    arr = np.zeros((rows * th, cols * tw, 4), np.float32)
    arr[..., 3] = 1
    for k, t in enumerate(tiles):
        img = bpy.data.images.load(t)
        px = np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], 4)
        r, c = divmod(k, cols)
        y0 = (rows - 1 - r) * th
        arr[y0:y0 + th, c * tw:(c + 1) * tw] = px[:th, :tw]
        bpy.data.images.remove(img)
    out = bpy.data.images.new("sheet", cols * tw, rows * th, alpha=False)
    out.pixels = arr.ravel().tolist()
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()
    bpy.data.images.remove(out)


bb = [(arm.matrix_world @ Vector(v)) for v in arm.bound_box]
stature = float(root.get("p_stature_m", 1.75))
zmid = stature * 0.5
tmp = os.path.join(os.environ.get("PV_TMP", "/tmp"), "pv_" + AID)
os.makedirs(tmp, exist_ok=True)
headz = float(root.get("pv_face_z", stature - 0.11 * stature / 1.75))
FD = float(root.get("pv_face_dist", 0.78 * stature / 1.75))

if "views" in MODES:
    d = stature * 2.15
    for name, v in (("front", (0, -1, 0.1)), ("three_quarter", (0.75, -0.9, 0.25)), ("back", (0, 1, 0.1))):
        v = Vector(v).normalized()
        shot(os.path.join(OUT, "%s_%s.png" % (AID, name)), (v.x * d, v.y * d, zmid + v.z * d), (0, 0, zmid), 50)
    # head and shoulders close-up (three-quarter)
    v = Vector((0.5, -1, 0.15)).normalized()
    shot(os.path.join(OUT, "%s_closeup_head.png" % AID), (v.x * 1.3 * FD, v.y * 1.3 * FD, headz + 0.05), (0, 0, headz - 0.02), 62)

if "faces" in MODES:
    import chars_head_keys  # noqa: F401  (list of expression names)
    tiles = []
    names = chars_head_keys.KEYS
    for e in names:
        if e != "neutral":
            root["p_expr_" + e] = 1.0
        p = os.path.join(tmp, "f_%s.png" % e)
        shot(p, (0.12, -FD, headz), (0, 0, headz - 0.01), 62, (300, 337))
        tiles.append(p)
        if e != "neutral":
            root["p_expr_" + e] = 0.0
    root["p_anger"] = 1.0
    root["p_flush"] = 1.0
    p = os.path.join(tmp, "f_anger_flush.png")
    shot(p, (0.12, -FD, headz), (0, 0, headz - 0.01), 62, (300, 337))
    tiles.append(p)
    root["p_anger"] = 0.0
    root["p_flush"] = 0.0
    compose(tiles, 5, 3, 300, 337, os.path.join(OUT, "%s_faces.png" % AID))

if "hand" in MODES:
    hk = bpy.data.objects["HOOK_hand_R"]
    pbs = arm.pose.bones
    tiles = []
    for tag, cur in (("open", dict(index=0.12, middle=0.12, ring=0.12, pinky=0.12, thumb=0.12)),
                     ("fist", dict(index=1.0, middle=1.0, ring=1.0, pinky=1.0, thumb=0.8)),
                     ("point", dict(index=0.0, middle=1.0, ring=1.0, pinky=1.0, thumb=0.6)),
                     ("grip", dict(index=0.55, middle=0.8, ring=0.8, pinky=0.8, thumb=0.5))):
        for f, v in cur.items():
            pbs["ctl_hand_R"]["curl_" + f] = v
        bpy.context.view_layer.update()
        t = hk.matrix_world.translation
        p = os.path.join(tmp, "h_%s.png" % tag)
        shot(p, (t.x - 0.28, t.y - 0.42, t.z + 0.12), (t.x, t.y, t.z), 70, (450, 450))
        tiles.append(p)
    compose(tiles, 4, 1, 450, 450, os.path.join(OUT, "%s_hand_closeup.png" % AID))
    for f in ("index", "middle", "ring", "pinky", "thumb"):
        pbs["ctl_hand_R"]["curl_" + f] = 0.12

if "squash" in MODES:
    tiles = []
    cfgs = [dict(), dict(p_squash=-0.35), dict(p_squash=0.35), dict(p_squash_head=0.45), dict(p_squash_head=-0.3, p_jiggle_belly=0.04, p_jiggle_hat=0.35, p_tie_swing=0.7)]
    for i, c in enumerate(cfgs):
        for k in ("p_squash", "p_squash_head", "p_jiggle_belly", "p_jiggle_hat", "p_tie_swing"):
            if k in root.keys():
                root[k] = c.get(k, 0.0)
        bpy.context.view_layer.update()
        p = os.path.join(tmp, "s_%d.png" % i)
        d = stature * 1.95
        shot(p, (0.55 * d, -0.85 * d, stature * 0.55), (0, 0, stature * 0.5), 50, (300, 337))
        tiles.append(p)
    for k in ("p_squash", "p_squash_head", "p_jiggle_belly", "p_jiggle_hat", "p_tie_swing"):
        if k in root.keys():
            root[k] = 0.0
    compose(tiles, 5, 1, 300, 337, os.path.join(OUT, "%s_squash_jiggle.png" % AID))

if "poseboard" in MODES:
    BOARD = [("idle", 15), ("walk", 8), ("run", 5), ("point", 16), ("aim_gun", 36), ("shout_loop", 8), ("punch_loop", 6), ("topple_back", 32),
             ("stagger", 9), ("pull_cable", 18), ("hug_leg", 18), ("jolt_hit", 10)]
    SIDEV = {"topple_back": (1, -0.25, 0.15), "walk": (1, -0.5, 0.2), "run": (1, -0.5, 0.2), "aim_gun": (0.8, -0.8, 0.2), "pull_cable": (1, -0.5, 0.2)}
    tiles = []
    for nm, fr in BOARD:
        act = bpy.data.actions.get("ACT_%s_%s" % (AID.replace("_holes30", ""), nm)) or bpy.data.actions.get("ACT_%s_%s" % (AID, nm))
        arm.animation_data.action = act
        scn.frame_set(fr)
        v = Vector(SIDEV.get(nm, (0.65, -1, 0.15))).normalized()
        d = stature * 1.85
        p = os.path.join(tmp, "b_%s.png" % nm)
        if nm == "topple_back":
            shot(p, (stature * 1.9, stature * 0.55, 0.9), (0, stature * 0.55, 0.2), 50, (300, 337))
        else:
            shot(p, (v.x * d, v.y * d, stature * 0.52 + v.z * d), (0, 0, stature * 0.5), 50, (300, 337))
        tiles.append(p)
    compose(tiles, 4, 3, 300, 337, os.path.join(OUT, "%s_poseboard.png" % AID))
    arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
    scn.frame_set(1)

if "holes" in MODES:
    props = [k for k in root.keys() if k.startswith("p_hole_") or k == "p_head_hole_radius"]
    he = bpy.data.objects.get("HOLE_hole_1")
    hh = bpy.data.objects.get("HOLE_headhole")

    def hshots(tag):
        if he is not None:
            t = he.matrix_world.translation
            shot(os.path.join(OUT, "%s_hole_straight%s.png" % (AID, tag)), (t.x, -1.2 * stature / 1.75, t.z), (t.x, 0, t.z), 60, (900, 675))
            shot(os.path.join(OUT, "%s_holes_closeup%s.png" % (AID, tag)), (0.45, -1.2 * stature / 1.75, stature * 0.64), (0, 0, stature * 0.64), 50)
        if hh is not None:
            t = hh.matrix_world.translation
            shot(os.path.join(OUT, "%s_headhole%s.png" % (AID, tag)), (1.0 * stature / 1.75, t.y, t.z), (0, t.y, t.z), 60)
    for k in props:
        root[k] = 1.0
    hshots("")
    for k in props:
        root[k] = 0.3
    hshots("_30")

if "poses" in MODES:
    acts = sorted([a for a in bpy.data.actions if a.name.startswith("ACT_%s_" % AID)], key=lambda a: a.name)
    sel = argv[3].split(",") if len(argv) > 3 else None
    KEYF = {"topple_back": [3, 18, 32, 60], "kneel_and_tie": [0, 24, 39], "aim_gun": [12, 36, 40], "slap": [8, 11, 14], "shove": [0, 9, 22], "throw_up": [5, 11, 16],
            "arm_whip": [10, 19, 24], "jolt_hit": [3, 10, 24], "stagger": [0, 9, 18], "hand_over_envelope": [18, 30, 40], "point": [10, 16, 34],
            "walk": [0, 8, 15], "run": [0, 5, 9], "punch_loop": [6, 12, 18], "pull_cable": [0, 18, 36], "idle": [0, 15, 44]}
    SIDE = {"topple_back": (1, -0.25, 0.15), "kneel_and_tie": (1, -0.5, 0.2), "walk": (1, -0.5, 0.2), "run": (1, -0.5, 0.2), "aim_gun": (0.8, -0.8, 0.2), "pull_cable": (1, -0.5, 0.2)}
    tiles = []
    k = 0
    sheet = 0
    for a in acts:
        nm = a.name[len("ACT_%s_" % AID):]
        if sel and nm not in sel:
            continue
        arm.animation_data.action = a
        fr = KEYF.get(nm)
        if fr is None:
            f0, f1 = a.frame_range
            fr = [int(f0 + (f1 - f0) * t) for t in (0.2, 0.55, 0.9)]
        fr = (fr + [fr[-1]] * 3)[:3] if len(fr) < 4 else fr[:4]
        for f in fr[:3]:
            scn.frame_set(f)
            v = Vector(SIDE.get(nm, (0.65, -1, 0.15))).normalized()
            d = stature * 1.85
            p = os.path.join(tmp, "p_%s_%d.png" % (nm, f))
            shot(p, (v.x * d, v.y * d, stature * 0.52 + v.z * d), (0, 0, stature * 0.5), 50, (300, 337))
            tiles.append(p)
            k += 1
        if len(tiles) >= 6:
            compose(tiles[:6], 3, 2, 300, 337, os.path.join(OUT, "%s_poses_%02d.png" % (AID, sheet)))
            tiles = tiles[6:]
            sheet += 1
    if tiles:
        while len(tiles) < 6:
            tiles.append(tiles[-1])
        compose(tiles[:6], 3, 2, 300, 337, os.path.join(OUT, "%s_poses_%02d.png" % (AID, sheet)))

if "anger" in MODES:
    tiles = []
    for a, fl in ((0.0, 0.0), (0.4, 0.3), (0.7, 0.7), (1.0, 1.0)):
        root["p_anger"] = a
        root["p_flush"] = fl
        p = os.path.join(tmp, "a_%.1f.png" % a)
        shot(p, (0.12, -FD, headz), (0, 0, headz - 0.01), 62, (300, 337))
        tiles.append(p)
    root["p_anger"] = 1.0
    root["p_flush"] = 1.0
    p = os.path.join(tmp, "a_side.png")
    shot(p, (0.5, -0.9 * FD, headz), (0, 0, headz - 0.01), 62, (300, 337))
    tiles.append(p)
    shot(os.path.join(tmp, "a_full.png"), (0.4, -3.2, stature * 0.55), (0, 0, stature * 0.52), 50, (300, 337))
    tiles.append(os.path.join(tmp, "a_full.png"))
    compose(tiles, 3, 2, 300, 337, os.path.join(OUT, "%s_anger.png" % AID))
    root["p_anger"] = 0.0
    root["p_flush"] = 0.0
