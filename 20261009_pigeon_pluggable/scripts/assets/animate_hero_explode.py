"""12 s orbit animation: half-transparent pigeon -> eased explosion -> billboard labels fade in (Blender 4.2 EEVEE).

Run from the project folder (frames -> outputs/hero/frames_explode/, encode with ffmpeg afterwards):
  scripts/bslot.sh -b assets/hero/pigeon_osfp_dr8.blend --python scripts/assets/animate_hero_explode.py -- outputs/hero [fstart fend]
Seamless 15 s loop (v2, 2026-10-10; the 12 s v1 is kept as outputs/hero/pigeon_osfp_dr8_explode_orbit_v1_12s.mp4).
Timeline (30 fps, 450-frame cycle, frame 1 = t 0, frame 451 == frame 1): 1-31 assembled hold, 31-151 explode (eased,
camera pulls out), 151-169 labels fade in, 169-259 hold, 259-277 fade out, 277-397 un-explode (camera dollies back),
397-451 assembled hold. One full orbit at constant angular speed; distance, elevation and pivot return to the start.
Flex ribbons and the sternum keep explode = 0 so they stay in place and keep linking the organs.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["outputs/hero"]
OUT = ARGS[0]
F0, F1 = (int(ARGS[1]), int(ARGS[2])) if len(ARGS) > 2 else (1, 450)
N_CYCLE = 450
K_EX0, K_EX1, K_EX2, K_EX3 = 31, 151, 277, 397   # explode start/end, un-explode start/end
L_IN, L_OUT = 151, 259                             # label fade-in start, fade-out start
FR = os.path.join(OUT, "frames_explode")
os.makedirs(FR, exist_ok=True)
SCN = bpy.context.scene
SCN.render.engine = "BLENDER_EEVEE_NEXT"
SCN.eevee.taa_render_samples = 12
SCN.render.resolution_x = SCN.render.resolution_y = 1080
SCN.render.resolution_percentage = 100
SCN.render.fps = 30
SCN.frame_start, SCN.frame_end = F0, F1
SCN.view_settings.view_transform = "AgX"
SCN.view_settings.look = "AgX - Punchy"
SCN.render.image_settings.file_format = "PNG"
SCN.render.image_settings.color_mode = "RGB"
SCN.render.filepath = os.path.abspath(os.path.join(FR, "frame_"))
world = SCN.world or bpy.data.worlds.new("dark")
SCN.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.008, 0.010, 0.015, 1)
SCN.use_nodes = True
nt = SCN.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new("CompositorNodeRLayers")
gl = nt.nodes.new("CompositorNodeGlare")
gl.glare_type = "FOG_GLOW"
gl.quality = "MEDIUM"
gl.threshold = 1.2
gl.size = 6
gl.mix = -0.6
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
nt.links.new(gl.outputs["Image"], comp.inputs["Image"])

C = Vector((0.0, 0.006, 0.056))


def light(name, loc, power, color, size):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.color, ld.size = power, color, size
    ob = bpy.data.objects.new(name, ld)
    SCN.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (C - Vector(loc)).normalized().to_track_quat("-Z", "Y").to_euler()
    return ob


# lights ride with the orbit so the lit side stays toward the camera
pivot = bpy.data.objects.new("orbit_pivot", None)
SCN.collection.objects.link(pivot)
pivot.location = C
for nm, loc, pw, col, sz in (("key", (-0.22, -0.22, 0.30), 9.0, (1.0, 0.96, 0.92), 0.16),
                             ("rim_cyan", (0.24, 0.20, 0.16), 6.0, (0.35, 0.8, 1.0), 0.20),
                             ("rim_magenta", (-0.20, 0.25, 0.10), 1.5, (1.0, 0.35, 0.8), 0.20),
                             ("fill", (0.20, -0.25, 0.08), 2.0, (0.8, 0.85, 1.0), 0.20),
                             ("cam_fill", (-0.20, -0.08, 0.12), 3.0, (0.95, 0.95, 1.0), 0.15)):
    lo = light(nm, loc, pw, col, sz)
    lo.parent = pivot
    lo.location = Vector(loc) - C

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam_data.clip_start = 0.002
cam = bpy.data.objects.new("cam", cam_data)
SCN.collection.objects.link(cam)
SCN.camera = cam
cam.parent = pivot
tc = cam.constraints.new("TRACK_TO")
tc.target = pivot
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"
# camera: tight on the assembled bird (~61% of frame height), then dolly out in step with the explosion (eased,
# frames 60-240) so the exploded extent just fits at 8 s; the pivot also drifts from the bird centre to the
# exploded-extent centre. Orbit (pivot Z rotation) stays linear for the full 12 s.
C_ASM = Vector((0.0, 0.011, 0.047))
for f, rh, z, piv in ((1, 0.224, 0.056, C_ASM), (K_EX0, 0.224, 0.058, C_ASM), (K_EX1, 0.425, 0.120, C),
                      (K_EX2, 0.425, 0.130, C), (K_EX3, 0.224, 0.058, C_ASM), (N_CYCLE + 1, 0.224, 0.056, C_ASM)):
    cam.location = (-rh * 0.87, -rh * 0.50, z)
    cam.keyframe_insert("location", frame=f)
    pivot.location = piv
    pivot.keyframe_insert("location", frame=f)
for fc in cam.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = "BEZIER"
        kp.easing = "EASE_IN_OUT"
        kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
pivot.rotation_euler = (0, 0, 0)
pivot.keyframe_insert("rotation_euler", index=2, frame=1)
pivot.rotation_euler = (0, 0, 2 * math.pi)
pivot.keyframe_insert("rotation_euler", index=2, frame=N_CYCLE + 1)
for fc in pivot.animation_data.action.fcurves:
    for kp in (fc.keyframe_points if fc.data_path == "rotation_euler" else []):
        kp.interpolation = "LINEAR"

# half-transparent skin
skin = []
for cname in ("shell_top", "shell_bottom"):
    stack = [bpy.data.collections[cname]]
    while stack:
        c = stack.pop()
        skin += [o for o in c.objects if o.type == "MESH"]
        stack += list(c.children)
cache = {}
for ob in skin:
    m0 = ob.data.materials[0]
    if m0.name not in cache:
        m = m0.copy()
        m.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.5
        m.surface_render_method = "BLENDED"
        m.use_backface_culling = True
        cache[m0.name] = m
    ob.data.materials[0] = cache[m0.name]
    ob.visible_shadow = False


# explosion keyframes (eased)
def ex_of(o):
    return Vector(o.get("explode", (0, 0, 0)))


def anc_ex(o):
    p = o.parent
    while p is not None and "explode" not in p:
        p = p.parent
    return ex_of(p) if p is not None else Vector((0, 0, 0))


MOVES = []
for ob in bpy.data.objects:
    if "explode" in ob and ob.parent is not None:
        d = ex_of(ob) - anc_ex(ob)
        if d.length > 0:
            MOVES.append((ob, ob.location.copy(), d))
for ob, base, d in MOVES:
    for fk, dd_ in ((K_EX0, 0.0), (K_EX1, 1.0), (K_EX2, 1.0), (K_EX3, 0.0)):
        ob.location = base + d * dd_
        ob.keyframe_insert("location", frame=fk)
    for fc in ob.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.easing = "EASE_IN_OUT"
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"

# label anchors in the exploded state
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_hero_pigeon.py")).read()
LABELS = eval(src[src.index("LABELS = [") + len("LABELS = "):src.index("]\n\nif \"exploded\"") + 1])
SCN.frame_set(K_EX1)
bpy.context.view_layer.update()
anchors = []
for text, names in LABELS:
    pts = []
    for nm in names:
        o = bpy.data.objects[nm]
        pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    anchors.append((text, sum(pts, Vector()) / len(pts)))
SCN.frame_set(1)


# labels: screen-space anchors per frame (drawn afterwards as camera-facing 2D callouts by anim_label_overlay.py)
import json
from bpy_extras.object_utils import world_to_camera_view

from anchor_vis import visible_anchor

label_objs = [(text, [bpy.data.objects[nm] for nm in names]) for text, names in LABELS]
see_through = {o.name for o in skin} | {o.name for o in bpy.data.objects if o.type == "MESH" and o.data.materials
                                        and o.data.materials[0].name.startswith("glass")}
per_frame = {}
n_hidden = 0
for f in range(max(F0, L_IN), min(F1, K_EX2) + 1):
    SCN.frame_set(f)
    rows = []
    for text, obs in label_objs:
        a, vis = visible_anchor(SCN, cam, obs, transparent=see_through, per_obj=120, max_tests=120)
        n_hidden += 0 if vis else 1
        v = world_to_camera_view(SCN, cam, a)
        rows.append([text, v.x * 1080, (1 - v.y) * 1080])
    per_frame[f] = rows
print("ANCHORS_HIDDEN_FALLBACKS", n_hidden)
SCN.frame_set(F0)
os.makedirs(os.path.join(OUT, "work"), exist_ok=True)
with open(os.path.join(OUT, "work", "anim_label_anchors_%04d_%04d.json" % (F0, F1)), "w") as fh:
    json.dump(per_frame, fh)

bpy.ops.render.render(animation=True)
print("ANIM_DONE frames %d-%d -> %s" % (F0, F1, FR))
