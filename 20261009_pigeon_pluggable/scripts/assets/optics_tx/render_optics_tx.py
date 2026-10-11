"""Verification renders of the TX optics subassembly (workstream T, DevLog-008). Blender 4.2 EEVEE, headless.

Run from the project folder:
  scripts/bslot.sh -b assets/components/optics_tx/optics_tx.blend --python scripts/assets/optics_tx/render_optics_tx.py \
      -- outputs/assets/optics_tx [views]
views: comma list of overview,exploded,pic_top,pic_zoom,pic_bonds,laser,mapping (default: all).
Raw PNGs + label/notes JSON go to <out>/work/; label_overlay_tx.py composes the final PNGs.
"""
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["outputs/assets/optics_tx"]
OUT = ARGS[0]
ALL = "overview,exploded,pic_top,pic_zoom,pic_bonds,laser,mapping,mapping_side"
VIEWS = (ARGS[1] if len(ARGS) > 1 else ALL).split(",")
WORK = os.path.join(OUT, "work")
os.makedirs(WORK, exist_ok=True)
MM = 0.001
SCN = bpy.context.scene

SCN.render.engine = "BLENDER_EEVEE_NEXT"
SCN.eevee.taa_render_samples = 48
SCN.render.resolution_percentage = 100
SCN.render.film_transparent = False
SCN.view_settings.view_transform = "AgX"
SCN.view_settings.look = "AgX - Punchy"
SCN.render.image_settings.file_format = "PNG"
SCN.render.image_settings.color_mode = "RGB"
world = bpy.data.worlds.new("dark")
SCN.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.035, 0.040, 0.052, 1)
bg.inputs["Strength"].default_value = 1.0

SCN.use_nodes = True
nt = SCN.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
gl = nt.nodes.new("CompositorNodeGlare")
gl.glare_type = "FOG_GLOW"
gl.quality = "HIGH"
gl.threshold = 1.5
gl.size = 6
gl.mix = -0.6
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
nt.links.new(gl.outputs["Image"], comp.inputs["Image"])

C0 = Vector((-1.5, -4.7, 0.0)) * MM
LIGHTS = []


def light(name, loc, power, color, size, target):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.color = color
    ld.size = size
    ob = bpy.data.objects.new(name, ld)
    SCN.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).normalized().to_track_quat("-Z", "Y").to_euler()
    LIGHTS.append(ob)
    return ob


def setup_lights(target, scale, gain=1.0):
    for o in LIGHTS:
        bpy.data.objects.remove(o, do_unlink=True)
    LIGHTS.clear()
    t = Vector(target)
    s = scale
    light("key", t + Vector((0.9, 0.5, 0.8)) * s, gain * 34 * s * s, (1.0, 0.96, 0.9), 0.6 * s, t)
    light("rim_cyan", t + Vector((-0.4, 0.9, 0.5)) * s, gain * 26 * s * s, (0.4, 0.8, 1.0), 0.8 * s, t)
    light("fill", t + Vector((0.6, -0.9, 0.2)) * s, gain * 12 * s * s, (0.85, 0.88, 1.0), 0.8 * s, t)
    light("top", t + Vector((1.2, 0.0, 0.0)) * s, gain * 10 * s * s, (0.95, 0.95, 1.0), 1.0 * s, t)


cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
SCN.collection.objects.link(cam)
SCN.camera = cam
cam_data.clip_start = 0.0001
cam_data.clip_end = 1.0


def look(loc, target, up=(0, 0, 1)):
    cam.location = loc
    d = (Vector(target) - Vector(loc)).normalized()
    cam.rotation_mode = "QUATERNION"
    q = d.to_track_quat("-Z", "Y")
    # roll so that camera up matches `up`
    cam.rotation_quaternion = q
    bpy.context.view_layer.update()
    cu = cam.matrix_world.to_3x3() @ Vector((0, 1, 0))
    u = Vector(up) - d * Vector(up).dot(d)
    u.normalize()
    ang = math.atan2(d.dot(cu.cross(u)), cu.dot(u))
    from mathutils import Quaternion
    cam.rotation_quaternion = Quaternion(d, ang) @ q
    bpy.context.view_layer.update()


OBJS = [o for o in bpy.data.objects if o.type == "MESH"]
BASE_LOC = {o.name: o.location.copy() for o in OBJS}


def set_explode(on):
    for o in OBJS:
        e = Vector(o.get("explode", (0, 0, 0)))
        o.location = BASE_LOC[o.name] + (e if on else Vector((0, 0, 0)))
    bpy.context.view_layer.update()


def show_beams(on):
    c = bpy.data.collections.get("tx_beams_viz")
    if c:
        c.hide_render = not on


def render(name, res):
    SCN.render.resolution_x, SCN.render.resolution_y = res
    SCN.render.filepath = os.path.join(WORK, name + "_raw.png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", name)


def labels_json(name, res, items):
    rows = []
    for text, names in items:
        obs = [bpy.data.objects[n] for n in names if n in bpy.data.objects]
        if not obs:
            continue
        pts = []
        for o in obs:
            for c in o.bound_box:
                pts.append(o.matrix_world @ Vector(c))
        ctr = sum(pts, Vector()) / len(pts)
        v = world_to_camera_view(SCN, cam, ctr)
        rows.append(dict(text=text, anchor=[v.x * res[0], (1 - v.y) * res[1]]))
    with open(os.path.join(WORK, name + "_labels.json"), "w") as fh:
        json.dump(dict(res=list(res), labels=rows), fh, indent=1)


def lane_tags(name, res):
    rows = []
    for k in range(8):
        o = bpy.data.objects.get("tx_beam_core_TX%d" % (k + 1))
        if o is None:
            continue
        me = o.data
        top = max((o.matrix_world @ v.co for v in me.vertices), key=lambda p: p.x)
        v = world_to_camera_view(SCN, cam, top)
        rows.append(dict(text="TX%d" % (k + 1), anchor=[v.x * res[0], (1 - v.y) * res[1]]))
    with open(os.path.join(WORK, name + "_tags.json"), "w") as fh:
        json.dump(dict(res=list(res), labels=rows), fh, indent=1)


def persp(lens):
    cam_data.type = "PERSP"
    cam_data.lens = lens
    cam_data.sensor_width = 36


def ortho(scale):
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = scale * MM


X_W = -1.70
Y_EDGE = -2.25


def fit(target, d, up, margin=1.08, objs=None):
    """Place the camera along direction d from target so all mesh bboxes fit the frame (perspective)."""
    obs = [o for o in (objs or OBJS) if not o.hide_render and not (o.users_collection and o.users_collection[0].hide_render)]
    pts = [o.matrix_world @ Vector(c) for o in obs for c in o.bound_box]
    dist = 0.05
    for _ in range(6):
        look(Vector(target) + Vector(d).normalized() * dist, target, up)
        mi = cam.matrix_world.inverted()
        fx = 1.0 / math.tan(cam_data.angle_x / 2)
        ar = SCN.render.resolution_x / SCN.render.resolution_y
        need = 0.0
        for p in pts:
            q = mi @ p
            z = -q.z
            if z <= 1e-6:
                continue
            need = max(need, abs(q.x) * fx / z, abs(q.y) * fx * ar / z)
        dist *= need * margin if need > 0 else 1.0
    look(Vector(target) + Vector(d).normalized() * dist, target, up)
    # re-centre: shift camera and target so the projected bbox is centred
    for _ in range(2):
        mi = cam.matrix_world.inverted()
        fx = 1.0 / math.tan(cam_data.angle_x / 2)
        ar = SCN.render.resolution_x / SCN.render.resolution_y
        xs, ys_, zs = [], [], []
        for p in pts:
            q = mi @ p
            z = -q.z
            if z > 1e-6:
                xs.append(q.x * fx / z)
                ys_.append(q.y * fx * ar / z)
                zs.append(z)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys_) + max(ys_)) / 2
        zm = sum(zs) / len(zs)
        R = cam.matrix_world.to_3x3()
        shift = R @ Vector((cx * zm / fx, cy * zm / (fx * ar), 0))
        target = Vector(target) + shift
        look(cam.location + shift, target, up)
    return dist


OUTLINES = []


def outlines(on, names_prefix=("tx_staircase", "tx_row_shift", "tx_fau_", "tx_rhomboid", "tx_eye_plate_2x4", "tx_pic_edge_lid",
                               "tx_si_lens", "tx_isolator")):
    """Render-only technical outlines (wireframe copies) for glass / small optics; not part of the asset."""
    global OUTLINES
    for o in OUTLINES:
        bpy.data.objects.remove(o, do_unlink=True)
    OUTLINES = []
    if not on:
        return
    m = bpy.data.materials.get("viz_outline") or bpy.data.materials.new("viz_outline")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.75, 0.92, 1.0, 1)
    b.inputs["Emission Color"].default_value = (0.55, 0.85, 1.0, 1)
    b.inputs["Emission Strength"].default_value = 0.6
    for o in OBJS:
        if not o.name.startswith(names_prefix):
            continue
        if o.name == "tx_eye_plate_2x4":                       # boolean mesh: outline its box only
            bb = [o.matrix_world @ Vector(c_) for c_ in o.bound_box]
            me = bpy.data.meshes.new("eye_box_outline")
            me.from_pydata(bb, [], [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            c = bpy.data.objects.new("eye_box_outline", me)
            me.materials.append(m)
            SCN.collection.objects.link(c)
            md = c.modifiers.new("wf", "WIREFRAME")
            md.thickness = 0.000004
            md.use_relative_offset = False
            OUTLINES.append(c)
            continue
        c = o.copy()
        c.data = o.data.copy()
        c.data.materials.clear()
        c.data.materials.append(m)
        SCN.collection.objects.link(c)
        md = c.modifiers.new("wf", "WIREFRAME")
        md.thickness = 0.000004
        md.use_relative_offset = False
        md.use_even_offset = True
        OUTLINES.append(c)
    bpy.context.view_layer.update()


if "overview" in VIEWS:
    outlines(False)
    set_explode(False)
    show_beams(False)
    persp(70)
    SCN.render.resolution_x, SCN.render.resolution_y = (1600, 1200)
    fit(C0, (1.0, 0.62, 0.78), (0, 0, 1), margin=1.06)
    setup_lights(C0, 0.03)
    render("optics_tx_overview", (1600, 1200))

if "exploded" in VIEWS:
    outlines(False)
    set_explode(True)
    show_beams(False)
    persp(45)
    tgt = Vector((4.8, -3.2, 0.0)) * MM
    res = (1800, 1350)
    SCN.render.resolution_x, SCN.render.resolution_y = res
    pts = [o.matrix_world @ Vector(c) for o in OBJS if not o.users_collection[0].name == "tx_beams_viz" for c in o.bound_box]
    tgt = sum(pts, Vector()) / len(pts)
    fit(tgt, (0.35, 0.20, 0.92), (1, 0, 0), margin=1.30,
        objs=[o for o in OBJS if o.users_collection[0].name != "tx_beams_viz"])
    setup_lights(tgt, 0.05, gain=2.2)
    render("optics_tx_exploded", res)
    labels_json("optics_tx_exploded", res, [
        ("carrier + RF pair / DC routing", ["tx_carrier"]),
        ("SiPh PIC: 8 TW-MZM (GSSG), heaters, MPDs", ["tx_pic_siph_die"]),
        ("HS differential bonds (GSSG)", ["tx_wirebonds_hs_diff_gssg"]),
        ("DC bonds (heaters, MPD)", ["tx_wirebonds_dc_ls"]),
        ("AlN optical bench", ["tx_laser_bench_aln"]),
        ("CW DFB on AlN CoS (x2)", ["tx_dfb_laser_die_LD2"]),
        ("Si lens 1, collimator (x2)", ["tx_si_lens1_collimator_LD2"]),
        ("isolator, optional (x2)", ["tx_isolator_garnet_LD2"]),
        ("Si lens 2, focuser (x2)", ["tx_si_lens2_focuser_LD2"]),
        ("FAU-attach collimator + 1x12 lens array", ["tx_fau_collimator_block"]),
        ("PIC edge lid", ["tx_pic_edge_lid"]),
        ("45 deg staircase TIR prism", ["tx_staircase_tir_prism"]),
        ("row-shift rhomboids x6", ["tx_row_shift_rhomboids"]),
        ("2x4 eye plate (Galilean x3.5)", ["tx_eye_plate_2x4"]),
    ])
    set_explode(False)

if "pic_top" in VIEWS:
    outlines(False)
    show_beams(False)
    ortho(8.8)
    tgt = Vector((X_W, -6.05, 0.0)) * MM
    look(tgt + Vector((0.02, 0, 0)), tgt, up=(0, 1, 0))
    setup_lights(tgt, 0.03)
    render("optics_tx_pic_top", (1800, 1800))

if "pic_zoom" in VIEWS:
    outlines(False)
    show_beams(False)
    ortho(2.6)
    tgt = Vector((X_W, Y_EDGE - 3.75, -2.35)) * MM
    look(tgt + Vector((0.02, 0, 0)), tgt, up=(0, 1, 0))
    setup_lights(tgt, 0.02)
    render("optics_tx_pic_zoom_mzm", (1600, 1600))

if "pic_bonds" in VIEWS:
    outlines(False)
    show_beams(False)
    persp(60)
    tgt = Vector((X_W - 0.3, Y_EDGE - 7.45, -1.6)) * MM
    d = Vector((0.75, -0.55, 0.45)).normalized()
    look(tgt + d * 0.0075, tgt)
    setup_lights(tgt, 0.012)
    render("optics_tx_pic_bonds", (1600, 1200))

if "laser" in VIEWS:
    show_beams(True)
    persp(60)
    tgt = Vector((X_W - 0.1, -0.95, 2.6)) * MM
    d = Vector((0.82, 0.14, 0.55)).normalized()
    look(tgt + d * 0.0066, tgt)
    setup_lights(tgt, 0.012)
    outlines(True)
    render("optics_tx_laser_coupling", (1600, 1200))

if "mapping" in VIEWS:
    show_beams(True)
    persp(50)
    tgt = Vector((X_W + 0.55, -0.35, 0.0)) * MM
    d = Vector((0.08, 0.86, 0.42)).normalized()
    look(tgt + d * 0.0085, tgt, up=(1, 0, 0))
    setup_lights(tgt, 0.014)
    outlines(True)
    render("optics_tx_mapping_1x8_2x4", (1600, 1200))
    lane_tags("optics_tx_mapping_1x8_2x4", (1600, 1200))

if "mapping_side" in VIEWS:
    show_beams(True)
    persp(50)
    tgt = Vector((X_W + 0.7, -0.2, 0.0)) * MM
    d = Vector((0.55, 0.62, 0.56)).normalized()
    look(tgt + d * 0.0085, tgt, up=(1, 0, 0))
    setup_lights(tgt, 0.014)
    outlines(True)
    render("optics_tx_mapping_oblique", (1600, 1200))
    lane_tags("optics_tx_mapping_oblique", (1600, 1200))

outlines(False)
show_beams(True)
print("OPTICS_TX_RENDER_DONE", VIEWS)
