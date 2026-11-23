"""Presenter cam loops: short seamless animations of Pat Reynolds for the host frame.

Same character, set, lights, camera and pose as shot_host_pat.py (the stills host_pat_present.jpg and
host_pat_empathy.jpg), so the still works as the poster of the video. Every clip starts and ends on the still's
neutral pose and expression with zero velocity in every channel, so the engine can chain clips in any order.

Clips (24 fps):
  idle            4.0 s  breathing, two blinks, tiny head drift
  talk            3.0 s  mouth open / close cycles, small nods, one brow lift, a blink
  gesture         5.0 s  talking plus one open-hand gesture (right hand rises into frame and returns)
  empathy_idle    4.0 s  hand-on-heart pose: breathing, blinks, head drift
  empathy_talk    3.0 s  hand-on-heart pose: talking

Motion: face shape keys (the kit's expression system, with extra keys for mouth open, rounded mouth and brow lift),
lid bones (rigid rotation, so a blink never cuts into the eyeball), head and spine bones, and the arm IK target for
the gesture. The silver-bob groove texture reads a per-vertex copy of the neutral-pose positions, so it stays on
the hair when the head moves (the kit's clay materials already use the undeformed rest positions).

Run (through the Blender slot wrapper, one clip per process):
  bslot.sh -b --factory-startup --python anim_host_pat.py -- clip=idle out_dir=<frames dir>
  options: frames=0,40,95 (subset), view=main|face|side (face / side = close-up and profile checks),
           samples=48 noise=0.03 res=800 (3:4 frame height in px), threads=4, teeth=1 (show the upper teeth; off by default:
           in both closed expressions they poke through the skin below the lower lip)
Frames are written as <out_dir>/<clip>_NNNN.png; encode_host_cam.sh turns them into the MP4 loops.
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "charkit"))
import numpy as np  # noqa: E402
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402
import chars_head as HD  # noqa: E402
import chars_geo as G  # noqa: E402
import shot_host_pat as S  # noqa: E402  (character, set, lights, pose)

A = C.args(clip="idle", out_dir="/tmp/pat_cam", frames="", view="main", samples=48, noise=0.03, res=800, threads=4,
           teeth=0)

FPS = 24
CLIPS = {"idle": 96, "talk": 72, "gesture": 120, "empathy_idle": 96, "empathy_talk": 72}

# extra expression keys (deltas from the kit's DEFAULT; the kit's gains apply). Shape keys add linearly on top of
# the pose expression (pat_smile or pat_concern = 1).
HD.EXPR["pat_open"] = dict(gap=0.020)
HD.EXPR["pat_round"] = dict(gap=0.013, width=0.88)
HD.EXPR["pat_brow"] = dict(brow_in=0.0045, brow_out=0.0040)
for _k in ("pat_open", "pat_round", "pat_brow"):
    if _k not in HD.KEYS:
        HD.KEYS.append(_k)


# ------------------------------------------------------------------------------------------------ curves
def win(t, t0, t1):
    """Smooth bump 0 -> 1 -> 0 on [t0, t1] (sin^2), zero value and slope at both ends, 0 outside."""
    if t <= t0 or t >= t1:
        return 0.0
    return math.sin(math.pi * (t - t0) / (t1 - t0)) ** 2


def ease_keys(keys, t):
    """Cosine interpolation through (t, value) keys (zero slope at every key); constant outside."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, va), (tb, vb) in zip(keys, keys[1:]):
        if t <= tb:
            u = (t - ta) / (tb - ta)
            return va + (vb - va) * (0.5 - 0.5 * math.cos(math.pi * u))
    return keys[-1][1]


def blink(t, tc):
    """Blink centred at frame tc: closes in 2 frames, holds 1, opens in 3 (0 = open, 1 = closed)."""
    return ease_keys([(tc - 2.5, 0.0), (tc, 1.0), (tc + 1.0, 1.0), (tc + 4.0, 0.0)], t)


def talk_track(n, seed, t_start=3, t_end_pad=5):
    """Syllable keys for mouth open and rounding: alternating open peaks and partial closures, a word gap now and
    then, fully closed for the first t_start and last t_end_pad frames."""
    rng = np.random.RandomState(seed)
    ko, kr = [(0.0, 0.0), (float(t_start), 0.0)], [(0.0, 0.0), (float(t_start), 0.0)]
    t = float(t_start)
    end = n - 1 - t_end_pad
    while True:
        dur = rng.uniform(4.5, 7.0)
        if t + dur > end:
            break
        peak = t + dur * rng.uniform(0.40, 0.55)
        ko.append((peak, rng.uniform(0.45, 1.0)))
        kr.append((peak, rng.uniform(0.0, 0.9) if rng.rand() < 0.4 else 0.0))
        t += dur
        gap = rng.rand() < 0.18
        ko.append((t, 0.0 if gap or rng.rand() < 0.35 else rng.uniform(0.08, 0.22)))
        kr.append((t, 0.0))
        if gap:
            t += rng.uniform(2.0, 4.0)
            ko.append((t, 0.0))
            kr.append((t, 0.0))
    ko.append((float(n - 1 - t_end_pad + 2), 0.0))
    kr.append((float(n - 1 - t_end_pad + 2), 0.0))
    ko.append((float(n), 0.0))
    kr.append((float(n), 0.0))
    return ko, kr


def state(clip, t):
    """Per-frame deltas from the neutral pose. t = frame index (0 .. n-1); every channel is 0 at t = 0 and t = n."""
    n = CLIPS[clip]
    s = dict(open=0.0, round=0.0, brow=0.0, blink=0.0, turn=0.0, nod=0.0, tilt=0.0, breath=0.0, gesture=0.0,
             beat=0.0)
    talk = clip in ("talk", "gesture", "empathy_talk")
    # breathing: whole cycles over the clip (about 4 s per breath at rest, faster while talking)
    cycles = {"idle": 1, "empathy_idle": 1, "talk": 1, "empathy_talk": 1, "gesture": 2}[clip]
    s["breath"] = math.sin(math.pi * cycles * t / n) ** 2
    if clip in ("idle", "empathy_idle"):
        s["blink"] = max(blink(t, 30.0), blink(t, 74.0))
        s["turn"] = math.radians(1.6) * win(t, 0, n) * math.sin(2 * math.pi * t / n + 0.6) * 1.15
        s["nod"] = math.radians(0.9) * win(t, 8, 70) - math.radians(0.5) * win(t, 60, 92)
        s["tilt"] = math.radians(1.1) * win(t, 0, n) * math.sin(math.pi * t / n)
    if talk:
        ko, kr = talk_track(n, {"talk": 11, "gesture": 23, "empathy_talk": 31}[clip])
        s["open"] = max(0.0, ease_keys(ko, t))
        s["round"] = max(0.0, ease_keys(kr, t))
        # nods on stressed beats, one brow lift, slight turn
        nods = {"talk": (10, 38), "gesture": (14, 50, 86), "empathy_talk": (14, 44)}[clip]
        s["nod"] = sum(math.radians(1.7) * win(t, c - 7, c + 9) for c in nods)
        brow_c = {"talk": 20, "gesture": 46, "empathy_talk": 26}[clip]
        s["brow"] = win(t, brow_c - 9, brow_c + 13)
        s["turn"] = math.radians(1.3) * win(t, 2, n - 2) * math.sin(2 * math.pi * (t - 2) / (n - 4))
        s["tilt"] = math.radians(0.8) * win(t, 4, n - 4)
        s["blink"] = blink(t, {"talk": 52.0, "gesture": 100.0, "empathy_talk": 56.0}[clip])
    if clip == "gesture":
        # rise 8 -> 36, hold with two small beats, return 82 -> 112
        s["gesture"] = ease_keys([(8.0, 0.0), (36.0, 1.0), (82.0, 1.0), (112.0, 0.0)], t)
        s["beat"] = win(t, 44, 58) - 0.7 * win(t, 62, 78)
    return s


# ------------------------------------------------------------------------------------------------ rig helpers
def mesh_keys(ch):
    out = []
    for ob in ch.objs.values():
        if ob.type == "MESH" and ob.data.shape_keys is not None:
            out.append(ob.data.shape_keys.key_blocks)
    return out


def pose0_attribute(ch, names=("hair",)):
    """Store the neutral-pose (armature-deformed, before solidify / subdivision) positions as the point attribute
    pat_pose0 and point the hair groove texture at it instead of the object coordinates."""
    for nm in names:
        ob = ch.objs[nm]
        saved = [(md, md.show_viewport) for md in ob.modifiers if md.type != "ARMATURE"]
        for md, _ in saved:
            md.show_viewport = False
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.zeros(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        for md, v in saved:
            md.show_viewport = v
        assert len(co) == len(ob.data.vertices) * 3
        at = ob.data.attributes.new("pat_pose0", "FLOAT_VECTOR", "POINT")
        at.data.foreach_set("vector", co)
        for slot in ob.material_slots:
            nt = slot.material.node_tree
            for node in list(nt.nodes):
                if node.type != "TEX_COORD":
                    continue
                links = [lk for lk in nt.links if lk.from_node == node and lk.from_socket.name == "Object"]
                if not links:
                    continue
                an = nt.nodes.new("ShaderNodeAttribute")
                an.attribute_type = "GEOMETRY"
                an.attribute_name = "pat_pose0"
                for lk in links:
                    nt.links.new(an.outputs["Vector"], lk.to_socket)
    bpy.context.view_layer.update()


def capture(pb):
    pb.rotation_mode = "QUATERNION"
    return pb.location.copy(), pb.rotation_quaternion.copy()


def apply_local(pb, loc, q):
    pb.location = loc
    pb.rotation_quaternion = q


def gesture_targets(ch, cam_loc):
    """Local channels of ik_hand_R and ik_elbow_R for the rest pose, an arc midpoint and the gesture peak:
    open hand at mid-chest height, in front of the blazer, palm toward the camera, fingers up and slightly out."""
    arm, K = ch.arm, ch.K
    pb = arm.pose.bones
    hb, eb = pb["ik_hand_R"], pb["ik_elbow_R"]
    rest_h, rest_e = capture(hb), capture(eb)
    f0, n0, _ = ch.hand_frames["R"]
    out = {"rest": (rest_h, rest_e)}
    el = ch.J["el"] * np.array([-1, 1, 1])
    for key, wr, f1, n1, pole in (
        ("mid", np.array([-0.205, -0.245, 1.000]), np.array([-0.15, -0.55, 0.60]), np.array([0.55, -0.60, 0.20]),
         el + np.array([-0.30, 0.25, -0.20])),
        ("peak", np.array([-0.150, -0.275, 1.135]), np.array([-0.25, -0.35, 0.90]), np.array([0.25, -0.85, 0.35]),
         el + np.array([-0.32, 0.22, -0.25])),
    ):
        R = S.frame_rot(f0, n0, G.nrm(f1), n1)
        S.set_pose_matrix(arm, "ik_hand_R", Vector(wr) * K, R)
        S.set_pose_matrix(arm, "ik_elbow_R", Vector(pole) * K)
        out[key] = (capture(hb), capture(eb))
    apply_local(hb, *rest_h)
    apply_local(eb, *rest_e)
    bpy.context.view_layer.update()
    return out


def bez(p0, c, p1, u):
    return p0 * (1 - u) ** 2 + c * 2 * u * (1 - u) + p1 * u ** 2


# ------------------------------------------------------------------------------------------------ main
def main():
    clip = A["clip"]
    n = CLIPS[clip]
    kind = "empathy" if clip.startswith("empathy") else "present"
    t_build = time.time()
    ch = S.build_pat()
    K = ch.K
    eye_z = float(np.mean([c[2] for c in ch.eye_centres])) * K
    S.subdivide_all(ch)
    S.build_jewellery(ch)
    S.lapel_pin(ch)
    S.subdivide_all(ch, skip=("lenses",))
    res = A["res"]
    C.setup_render(samples=A["samples"], res=(res * 3 // 4, res), noise=A["noise"], look=S.A["look"],
                   exposure=S.A["exposure"])
    sc = bpy.context.scene
    sc.render.threads_mode = "FIXED"
    sc.render.threads = A["threads"]
    sc.render.use_persistent_data = True
    sc.cycles.seed = 0
    sc.cycles.use_animated_seed = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.image_settings.compression = 15
    C.world_hdri("studio_small_09", strength=0.10, rot_z_deg=120, bg_color=(0.05, 0.045, 0.04))
    S.studio_set()
    S.lights(K)
    # 3:4 host frame camera, identical to shot_host_pat.py view=main
    lens = 85.0
    dist = 0.60 * lens / 36.0
    cam_loc = Vector((0.0, -0.12 - dist, eye_z))
    cam = C.camera(cam_loc, cam_loc + Vector((0, 1, 0)), lens=lens, sensor=36.0)
    frame_h = 0.60 * 4.0 / 3.0
    z_top = (1.75 * 1.0 + 0.04) * K + 0.03
    cam.data.shift_y = ((z_top - frame_h / 2) - eye_z) / frame_h
    cam.data.shift_x = -0.03
    C.set_aperture(cam, (0, -0.12 - 0.06, eye_z), pupil_mm=lens * 1e-3 / 3.2)
    S.pose(ch, kind, cam.location)
    if A["view"] == "face":
        sc.render.resolution_x = sc.render.resolution_y = res
        v = C.camera((0.0, -1.10, eye_z), Vector((0, 0, eye_z - 0.03)), lens=85.0, sensor=36.0)
        v.data.dof.use_dof = False
    elif A["view"] == "side":
        sc.render.resolution_x = sc.render.resolution_y = res
        az = math.radians(-65.0)
        c0 = Vector((0, 0, 1.05))
        v = C.camera(c0 + Vector((math.sin(az), -math.cos(az), 0.10)) * 1.5, c0, lens=50.0)
        v.data.dof.use_dof = False
    if A["teeth"]:
        ch.objs["teeth_up"].hide_render = False
    C.compositor(vignette=0.12)
    pose0_attribute(ch)

    arm = ch.arm
    pb = arm.pose.bones
    expr = "pat_concern" if kind == "empathy" else "pat_smile"
    kbs = mesh_keys(ch)
    base = {}
    for nm in ("head", "neck", "spine_2", "chest"):
        pb[nm].rotation_mode = "XYZ"
        base[nm] = (pb[nm].rotation_euler.copy(), pb[nm].location.copy())
    for nm in ("lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R"):
        pb[nm].rotation_mode = "XYZ"
    gt = gesture_targets(ch, cam.location) if clip == "gesture" else None
    curl_rest = {"index": 0.18, "middle": 0.24, "ring": 0.28, "pinky": 0.32, "thumb": 0.10}
    curl_open = {"index": 0.04, "middle": 0.05, "ring": 0.08, "pinky": 0.12, "thumb": 0.05}

    def apply(t):
        s = state(clip, t)
        w = {expr: 1.0, "pat_open": s["open"] * (1 - 0.5 * s["round"]), "pat_round": s["round"] * s["open"],
             "pat_brow": s["brow"]}
        for kb in kbs:
            for name, val in w.items():
                if name in kb:
                    kb[name].value = val
        # lids: upper lid rotates down, lower lid up (rigid rotation about the eyeball centre)
        for side in ("L", "R"):
            pb["lid_up_" + side].rotation_euler = (1.05 * s["blink"], 0.0, 0.0)
            pb["lid_lo_" + side].rotation_euler = (-0.55 * s["blink"], 0.0, 0.0)
        r0, _ = base["head"]
        pb["head"].rotation_euler = (r0[0] + s["nod"] * 0.7, r0[1] + s["turn"], r0[2] + s["tilt"])
        r0, _ = base["neck"]
        pb["neck"].rotation_euler = (r0[0] + s["nod"] * 0.3, r0[1], r0[2])
        r0, l0 = base["chest"]
        pb["chest"].location = l0 + Vector((0.0, 0.0022 * K * s["breath"], 0.0))
        r0, _ = base["spine_2"]
        pb["spine_2"].rotation_euler = (r0[0] - math.radians(0.35) * s["breath"], r0[1], r0[2])
        if gt is not None:
            g = s["gesture"]
            (h0, e0), (hm, em), (h1, e1) = gt["rest"], gt["mid"], gt["peak"]
            # quadratic Bezier through the arc midpoint at g = 0.5
            ch_ = hm[0] * 2 - (h0[0] + h1[0]) * 0.5
            ce_ = em[0] * 2 - (e0[0] + e1[0]) * 0.5
            hl = bez(h0[0], ch_, h1[0], g)
            q = h0[1].slerp(hm[1], min(1.0, 2 * g)) if g < 0.5 else hm[1].slerp(h1[1], 2 * g - 1)
            # emphasis beats at the peak: small wrist lift and rotation (pose-space: local y is up the chest bone)
            hl = hl + Vector((0.0, 0.016 * K * s["beat"], -0.006 * K * s["beat"]))
            apply_local(pb["ik_hand_R"], hl, q)
            el_ = bez(e0[0], ce_, e1[0], g)
            pb["ik_elbow_R"].location = el_
            curls = {k: curl_rest[k] + (curl_open[k] - curl_rest[k]) * min(1.0, g * 1.4) for k in curl_rest}
            S.set_curl(ch, "R", curls)
        bpy.context.view_layer.update()

    print("BUILD_SECONDS %.1f" % (time.time() - t_build))
    frames = [int(x) for x in A["frames"].split(",") if x != ""] or list(range(n))
    os.makedirs(A["out_dir"], exist_ok=True)
    tag = clip if A["view"] == "main" else "%s_%s" % (clip, A["view"])
    t0 = time.time()
    for t in frames:
        apply(t)
        out = os.path.join(A["out_dir"], "%s_%04d.png" % (tag, t))
        C.render(out)
    dt = time.time() - t0
    print("CLIP_SECONDS %s frames=%d total=%.1f per_frame=%.2f" % (clip, len(frames), dt, dt / max(1, len(frames))))


if __name__ == "__main__":
    main()
