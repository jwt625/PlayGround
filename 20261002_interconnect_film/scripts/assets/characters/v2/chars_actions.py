"""Pose engine and action library for the character rigs.

Poses are authored in baseline character space (metres at stature 1.75; x = character left, y = back, z = up) and
scaled by the rig's K. IK targets (hands, feet, look-at) are given as absolute positions; FK bones as angles:
flex (+ = bend forward), twist (+ = turn to the character's left), lean (+ = toward the character's left).
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

FPS = 30
FK_BONES = ["hips", "spine_1", "spine_2", "chest", "neck", "head", "jaw", "lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R", "root"]
IK_BONES = ["ik_hand_L", "ik_hand_R", "ik_foot_L", "ik_foot_R", "ik_elbow_L", "ik_elbow_R", "ik_knee_L", "ik_knee_R", "ctl_look"]
CURLS = ["index", "middle", "ring", "pinky", "thumb"]
V = lambda *a: np.array(a, float)  # noqa: E731


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def nrm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Rig:
    def __init__(self, arm, K, J, frames_hand):
        self.arm, self.K, self.J = arm, K, J
        self.pb = arm.pose.bones
        self.frames_hand = frames_hand  # {'L': (f, n, c), 'R': ...}
        self.rest_rot = {b.name: np.array(b.matrix_local.to_3x3()) for b in arm.data.bones}
        for n in FK_BONES:
            self.pb[n].rotation_mode = "XYZ"
        self.rest = dict(
            hand_L=J["wr"].copy(), hand_R=J["wr"] * V(-1, 1, 1),
            foot_L=J["ankle"].copy(), foot_R=J["ankle"] * V(-1, 1, 1))

    def clear(self):
        for pb in self.pb:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
        for sn in ("L", "R"):
            for f in CURLS:
                self.pb["ctl_hand_" + sn]["curl_" + f] = 0.12
        bpy.context.view_layer.update()

    # -------------------------------------------------------------------- setting a pose
    def set_world(self, name, pos, R=None):
        pb = self.pb[name]
        R = self.rest_rot[name] if R is None else R
        M = Matrix.Translation(Vector(np.asarray(pos) * self.K)) @ Matrix(R.tolist()).to_4x4()
        pb.matrix = M

    def hand_R_world(self, sn, f, n):
        f0, n0, _ = self.frames_hand[sn]
        F0 = np.stack([f0, n0, np.cross(f0, n0)], axis=1)
        n = np.asarray(n, float)
        f = nrm(f)
        n = nrm(n - f * np.dot(n, f))
        F1 = np.stack([f, n, np.cross(f, n)], axis=1)
        return F1 @ F0.T @ self.rest_rot["ik_hand_" + sn]

    def pose(self, P):
        """Apply pose dict P (everything absent = rest)."""
        K = self.K
        pb = self.pb
        self.clear()
        for name in ("root", "hips"):
            loc = P.get(name + "_loc")
            if loc is not None:
                pb[name].location = (loc[0] * K, loc[2] * K, -loc[1] * K)
        rot = P.get("root_rot")
        if rot is not None:
            pb["root"].rotation_euler = (rot[0], rot[1], -rot[2])
        rot = P.get("hips_rot")
        if rot is not None:
            pb["hips"].rotation_euler = (rot[0], rot[1], -rot[2])
        sp = P.get("spine")
        if sp is not None:
            for n, w in (("spine_1", 0.25), ("spine_2", 0.35), ("chest", 0.40)):
                pb[n].rotation_euler = (sp[0] * w, sp[1] * w, -sp[2] * w)
        for n in ("neck", "head"):
            r = P.get(n)
            if r is not None:
                pb[n].rotation_euler = (r[0], r[1], -r[2])
        pb["jaw"].rotation_euler = (P.get("jaw", 0.0), 0, 0)
        bl = P.get("blink", 0.0)
        for sn in ("L", "R"):
            pb["lid_up_" + sn].rotation_euler = (0.95 * bl + P.get("lid_up", 0.0), 0, 0)
            pb["lid_lo_" + sn].rotation_euler = (-0.12 * bl - P.get("lid_lo", 0.0), 0, 0)
        bpy.context.view_layer.update()
        for sn, side in (("L", 1), ("R", -1)):
            h = P.get("hand_" + sn)
            if h is not None:
                pos = np.asarray(h["pos"], float)
                R = self.hand_R_world(sn, h.get("f", self.frames_hand[sn][0]), h.get("n", self.frames_hand[sn][1]))
                self.set_world("ik_hand_" + sn, pos, R)
            e = P.get("elbow_" + sn)
            if e is not None:
                self.set_world("ik_elbow_" + sn, e)
            ft = P.get("foot_" + sn)
            if ft is not None:
                pos = np.asarray(ft["pos"], float)
                R = rz(ft.get("yaw", 0.0)) @ rx(-ft.get("pitch", 0.0)) @ self.rest_rot["ik_foot_" + sn]
                self.set_world("ik_foot_" + sn, pos, R)
            k = P.get("knee_" + sn)
            if k is not None:
                self.set_world("ik_knee_" + sn, k)
            c = P.get("curl_" + sn)
            if c is not None:
                if isinstance(c, (int, float)):
                    c = {f: c for f in CURLS}
                for f, v in c.items():
                    pb["ctl_hand_" + sn]["curl_" + f] = float(v)
        lk = P.get("look")
        if lk is not None:
            self.set_world("ctl_look", lk)
        bpy.context.view_layer.update()

    def snapshot(self):
        pb = self.pb
        st = {}
        for n in FK_BONES:
            st[(n, "rotation_euler")] = tuple(pb[n].rotation_euler)
            if n in ("root", "hips"):
                st[(n, "location")] = tuple(pb[n].location)
        for n in IK_BONES:
            st[(n, "location")] = tuple(pb[n].location)
            st[(n, "rotation_quaternion")] = tuple(pb[n].rotation_quaternion)
        for sn in ("L", "R"):
            for f in CURLS:
                st[("ctl_hand_" + sn, '["curl_%s"]' % f)] = float(pb["ctl_hand_" + sn]["curl_%s" % f])
        return st

    def write_key(self, st, frame):
        pb = self.pb
        for (n, attr), v in st.items():
            if attr.startswith("["):
                pb[n][attr[2:-2]] = v
            else:
                setattr(pb[n], attr, v)
            pb[n].keyframe_insert(attr, frame=frame)

    def make_action(self, aid, name, keys, loop=False, meta=None):
        """keys: list of (frame, pose dict). Poses are solved with no action assigned, then written as keyframes."""
        arm = self.arm
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = None
        snaps = []
        for f, P in keys:
            self.pose(P)
            snaps.append((f, self.snapshot()))
        act = bpy.data.actions.new("ACT_%s_%s" % (aid, name))
        act.use_fake_user = True
        arm.animation_data.action = act
        for f, st in snaps:
            self.write_key(st, f)
        frs = [k[0] for k in keys]
        act.use_frame_range = True
        act.frame_start, act.frame_end = min(frs), max(frs)
        act.use_cyclic = bool(loop)
        act["loop"] = bool(loop)
        for k_, v in (meta or {}).items():
            act[k_] = v
        for fc in act.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
            if loop:
                fc.modifiers.new("CYCLES")
        return act


# ---------------------------------------------------------------------------------------------- pose vocabulary
FWD = V(0, -1, 0)
UP = V(0, 0, 1)
DOWN = V(0, 0, -1)


def H(side, x, y, z, f=None, n=None):
    d = dict(pos=V(side * x, y, z))
    if f is not None:
        d["f"] = V(*f)
    if n is not None:
        d["n"] = V(*n)
    return d


def medial(side):
    return V(-side, 0, 0)


def foot_ankle(y, contact="flat", pitch=0.0, z_lift=0.0, x=0.10, side=1):
    """Ankle target for a foot whose ground-contact pivot is at y (contact: flat, heel, ball). pitch>0 toe up."""
    az = 0.0707
    if contact == "flat":
        return dict(pos=V(side * x, y, az + z_lift), pitch=pitch)
    if contact == "heel":
        v = V(0, -0.075, az)  # heel -> ankle
        th = -pitch  # rotation about +X, positive = toe down
        yy = v[1] * math.cos(th) - v[2] * math.sin(th)
        zz = v[1] * math.sin(th) + v[2] * math.cos(th)
        return dict(pos=V(side * x, y + yy, zz + z_lift), pitch=pitch)
    v = V(0, 0.145, az - 0.03)  # ball -> ankle
    th = -pitch
    yy = v[1] * math.cos(th) - v[2] * math.sin(th)
    zz = v[1] * math.sin(th) + v[2] * math.cos(th)
    return dict(pos=V(side * x, y + yy, zz + 0.0 + z_lift), pitch=pitch)


def stand_feet(dy=0.0, spread=0.10):
    return dict(foot_L=foot_ankle(dy, "flat", x=spread, side=1), foot_R=foot_ankle(dy, "flat", x=spread, side=-1))


def merge(*ds):
    out = {}
    for d in ds:
        out.update(d)
    return out


def library(K):
    """Return {action_name: (keys, loop, meta)} authored in baseline space."""
    L = {}
    rest_hand_y = -0.08

    # ------------------------------------------------------------------ idle
    def idle_pose(t, blink=0.0):
        s = math.sin(2 * math.pi * t)
        return dict(spine=(0.02 * s, 0.0, 0.0), head=(-0.02 * s + 0.02, 0.05 * math.sin(2 * math.pi * t + 1), 0.0),
                    hips_loc=(0.012 * math.sin(2 * math.pi * t), 0, 0.0), hips_rot=(0, 0, 0.02 * math.sin(2 * math.pi * t)),
                    blink=blink, curl_L=0.2, curl_R=0.2)
    L["idle"] = ([(0, idle_pose(0)), (15, idle_pose(0.25)), (36, idle_pose(0.6, 0.0)), (40, idle_pose(0.66, 1.0)), (44, idle_pose(0.73, 0.0)),
                  (60, idle_pose(1.0))], True, dict(note="breathing, weight shift, one blink"))

    # ------------------------------------------------------------------ walk / run (in place; assembler moves the root)
    def cycle(n_frames, step, lift, bob, arm_sw, lean, spd_stance, name):
        keys = []
        stance = 0.6
        for i in range(n_frames + 1):
            ph = (i / n_frames) % 1.0
            P = {}
            for side, off in ((1, 0.0), (-1, 0.5)):
                p = (ph + off) % 1.0
                yc = None
                if p < stance:
                    yb = step / 2 - p / stance * step  # y of ball/ankle moves from front (-) to back (+): front is negative y
                    y = -step / 2 + (p / stance) * step
                    z = 0.0
                    if p < 0.12:
                        ft = foot_ankle(y + 0.075, "heel", pitch=0.35 * (1 - p / 0.12), side=side)
                    elif p < 0.45:
                        ft = foot_ankle(y, "flat", side=side)
                    else:
                        q = (p - 0.45) / (stance - 0.45)
                        ft = foot_ankle(y - 0.145, "ball", pitch=-0.65 * q, side=side)
                else:
                    q = (p - stance) / (1 - stance)
                    y = step / 2 - q * step
                    z = lift * math.sin(math.pi * q)
                    ft = foot_ankle(y, "flat", pitch=-0.4 * (1 - q) + 0.35 * q * q, z_lift=z, side=side)
                P["foot_" + ("L" if side == 1 else "R")] = ft
                P["knee_" + ("L" if side == 1 else "R")] = V(side * 0.10, -0.8, 0.5)
            sw = math.sin(2 * math.pi * ph)
            P["hips_loc"] = (0.02 * math.sin(2 * math.pi * ph), 0.0, bob * math.cos(4 * math.pi * ph) - bob)
            P["hips_rot"] = (lean * 0.3, 0.14 * sw * (step / 0.7), 0.04 * math.sin(2 * math.pi * ph))
            P["spine"] = (lean, -0.18 * sw * (step / 0.7), 0.0)
            P["head"] = (-lean * 0.7, 0.05 * sw, 0.0)
            for side, sn, sgn in ((1, "L", -1), (-1, "R", 1)):
                sg = sgn * sw
                y = -0.08 - arm_sw * sg
                z = 0.86 + 0.06 * max(0.0, -sg) + (0.07 if name == "run" else 0.0)
                bend = 0.12 if name == "walk" else 0.0
                if name == "run":
                    P["hand_" + sn] = H(side, 0.23, y * 1.3 - 0.12, z + 0.2 + 0.05 * (-sg), f=(0, -0.7, -0.7), n=(-side, 0, 0.2))
                    P["curl_" + sn] = 0.8
                else:
                    P["hand_" + sn] = H(side, 0.28, y, z)
                    P["curl_" + sn] = 0.25
            keys.append((i, P))
        return keys
    L["walk"] = (cycle(30, 0.70, 0.11, 0.022, 0.20, 0.03, 1.17, "walk"), True, dict(root_speed_mps=1.17, cycle_frames=30, step_m=0.70))
    L["run"] = (cycle(18, 1.05, 0.22, 0.045, 0.28, 0.20, 3.0, "run"), True, dict(root_speed_mps=3.0, cycle_frames=18, step_m=1.05))

    # ------------------------------------------------------------------ kneel_and_tie
    kn = dict(foot_L=foot_ankle(-0.45, "flat", x=0.12, side=1), foot_R=dict(pos=V(-0.10, 0.38, 0.065), pitch=-1.25),
              knee_L=V(0.12, -0.9, 0.7), knee_R=V(-0.10, -0.6, 0.0))
    kpose = merge(kn, dict(hips_loc=(0, -0.12, -0.45), hips_rot=(0.0, 0.0, 0.0), spine=(0.75, 0.0, 0.0), neck=(0.15, 0, 0), head=(0.2, 0.0, 0.0),
                           look=V(0, -0.65, 0.12), curl_L=0.5, curl_R=0.5))

    def tie(t):
        a = math.sin(2 * math.pi * t)
        b = math.sin(2 * math.pi * t + math.pi)
        P = dict(kpose)
        P["hand_L"] = H(1, 0.09 - 0.05 * a, -0.52 + 0.06 * a, 0.14 + 0.03 * max(a, 0), f=(0, -0.6, -0.8), n=(-1, 0, 0.2))
        P["hand_R"] = H(-1, 0.09 - 0.05 * b, -0.52 + 0.06 * b, 0.14 + 0.03 * max(b, 0), f=(0, -0.6, -0.8), n=(1, 0, 0.2))
        P["curl_L"] = 0.6 + 0.2 * a
        P["curl_R"] = 0.6 + 0.2 * b
        P["head"] = (0.25 + 0.03 * a, 0.08 * a, 0.0)
        return P
    stand = dict(idle_pose(0))
    keys = [(0, stand), (10, merge(stand, dict(hips_loc=(0, -0.05, -0.25), spine=(0.4, 0, 0), foot_R=dict(pos=V(-0.10, 0.15, 0.2), pitch=-0.5), foot_L=foot_ankle(-0.3, "flat", x=0.12, side=1), knee_L=V(0.12, -0.9, 0.6), knee_R=V(-0.1, -0.5, 0.2)))),
            (24, tie(0.0))]
    for i, t in enumerate((0.25, 0.5, 0.75, 1.0)):
        keys.append((24 + (i + 1) * 15, tie(t)))
    L["kneel_and_tie"] = (keys, False, dict(note="frames 0-24 kneel down; frames 24-84 tying loop (cyclic section)", loop_start=24, loop_end=84))

    # ------------------------------------------------------------------ topple_back
    def fall(rot, z=0.0, spine=(0, 0, 0), arms=0.0, head=(0, 0, 0), jaw=0.0, legs=0.0):
        P = dict(stand_feet(), root_rot=(rot, 0, 0), root_loc=(0, 0, z), spine=spine, head=head, jaw=jaw)
        up = arms
        P["hand_L"] = H(1, 0.30 + 0.25 * up, -0.08 - 0.25 * up, 0.84 + 0.45 * up)
        P["hand_R"] = H(-1, 0.30 + 0.25 * up, -0.08 - 0.25 * up, 0.84 + 0.45 * up)
        P["curl_L"] = P["curl_R"] = 0.1 + 0.4 * (1 - up)
        P["knee_L"] = V(0.10, -0.8 - legs, 0.5)
        P["knee_R"] = V(-0.10, -0.8 - legs, 0.5)
        P["foot_L"] = dict(pos=V(0.10, 0.0, 0.0707 + 0.12 * legs), pitch=0.2 * legs)
        P["foot_R"] = dict(pos=V(-0.10, 0.0, 0.0707 + 0.05 * legs), pitch=0.0)
        return P
    topple = [(0, fall(0)), (3, fall(-0.06, spine=(-0.45, 0, 0), arms=1.0, head=(-0.3, 0, 0), jaw=0.35)),
              (8, fall(0.05, spine=(0.2, 0, 0), arms=0.7, head=(0.1, 0, 0), jaw=0.3)),
              (13, fall(-0.10, spine=(-0.15, 0, 0), arms=0.8, head=(-0.1, 0, 0), jaw=0.25)),
              (18, fall(-0.12, spine=(-0.1, 0, 0), arms=0.9, head=(0, 0, 0), jaw=0.3)),
              (32, fall(-1.42, z=0.05, spine=(-0.05, 0, 0), arms=1.0, head=(-0.25, 0, 0), jaw=0.2, legs=0.5)),
              (38, fall(-1.30, z=0.08, spine=(0.0, 0, 0), arms=0.5, head=(-0.2, 0, 0), jaw=0.1, legs=0.3)),
              (44, fall(-1.46, z=0.045, spine=(0, 0, 0), arms=0.35, head=(-0.15, 0.2, 0), jaw=0.15, legs=0.1)),
              (60, fall(-1.46, z=0.045, spine=(0, 0, 0), arms=0.3, head=(-0.15, 0.2, 0), jaw=0.0, legs=0.0))]
    L["topple_back"] = (topple, False, dict(note="hit in the chest, wobble, fall backward about the heels, stay down; body ends lying on its back with head at +Y"))

    # ------------------------------------------------------------------ stagger
    def stg(x, lean, lf, rf, arms, head):
        P = dict(foot_L=foot_ankle(lf, "flat", x=0.12 + 0.03 * abs(x), side=1), foot_R=foot_ankle(rf, "flat", x=0.12 + 0.03 * abs(x), side=-1),
                 hips_loc=(x * 0.05, 0.05 * lean * 2, -0.02), hips_rot=(-0.1 * lean, 0, -x * 0.1), spine=(-lean, x * 0.3, -x * 0.25), head=(head, -x * 0.3, x * 0.2), jaw=0.3)
        P["hand_L"] = H(1, 0.30 + 0.2 * arms, -0.1 + 0.15 * x, 1.0 + 0.3 * arms, f=(0.3, 0, -1))
        P["hand_R"] = H(-1, 0.30 + 0.2 * arms, -0.1 - 0.15 * x, 1.0 + 0.3 * arms, f=(-0.3, 0, -1))
        P["knee_L"] = V(0.12, -0.8, 0.5)
        P["knee_R"] = V(-0.12, -0.8, 0.5)
        P["curl_L"] = P["curl_R"] = 0.0
        return P
    L["stagger"] = ([(0, stg(-0.5, 0.3, -0.05, 0.12, 0.8, -0.15)), (9, stg(0.6, 0.1, 0.18, -0.12, 1.0, 0.15)), (18, stg(-0.6, 0.25, -0.12, 0.2, 0.9, -0.1)),
                     (27, stg(0.4, 0.05, 0.1, -0.1, 0.8, 0.1)), (36, stg(-0.5, 0.3, -0.05, 0.12, 0.8, -0.15))], True, dict(note="off-balance lurch, flailing arms"))

    # ------------------------------------------------------------------ shout_loop
    def shout(t):
        s = math.sin(2 * math.pi * t)
        o = 0.5 + 0.45 * math.sin(2 * math.pi * t * 2 - 1.2)
        P = dict(stand_feet(0, 0.14), spine=(0.12 + 0.05 * s, 0, 0), head=(0.12 * s - 0.05, 0.1 * s, 0), jaw=0.55 * o,
                 hips_loc=(0, 0, -0.02))
        P["hand_L"] = H(1, 0.30, -0.18, 1.12 + 0.05 * s, f=(0, -0.5, -0.8), n=(-1, 0, 0))
        P["hand_R"] = H(-1, 0.22, -0.30 - 0.05 * s, 1.45 + 0.12 * s, f=(0, -0.2, 1), n=(1, -0.2, 0))  # raised fist
        P["curl_L"] = P["curl_R"] = 0.95
        P["foot_L"] = foot_ankle(0.0, "flat", x=0.15, side=1)
        P["foot_R"] = foot_ankle(-0.05, "flat", x=0.15, side=-1)
        return P
    L["shout_loop"] = ([(0, shout(0)), (8, shout(0.25)), (15, shout(0.5)), (23, shout(0.75)), (30, shout(1.0))], True, dict(note="mouth flap via jaw bone; set p_expr_shouting for the face"))

    # ------------------------------------------------------------------ punch_loop
    def punch(t):
        a = math.sin(2 * math.pi * t)  # +: right jab out
        ext_r = max(0.0, a)
        ext_l = max(0.0, -a)
        P = dict(foot_L=foot_ankle(-0.22, "flat", x=0.17, side=1), foot_R=foot_ankle(0.18, "flat", x=0.17, side=-1), hips_loc=(0, -0.05, -0.08),
                 hips_rot=(0.0, 0.25 * a, 0), spine=(0.18, 0.3 * a, 0), head=(0.05, -0.15 * a, 0), jaw=0.15,
                 knee_L=V(0.2, -0.9, 0.5), knee_R=V(-0.2, -0.6, 0.5))
        P["hand_R"] = H(-1, 0.15 - 0.06 * ext_r, -0.18 - 0.36 * ext_r, 1.38 - 0.02 * ext_r, f=(0, -1, 0.1), n=(1, 0, 0))
        P["hand_L"] = H(1, 0.15 - 0.06 * ext_l, -0.18 - 0.36 * ext_l, 1.38 - 0.02 * ext_l, f=(0, -1, 0.1), n=(-1, 0, 0))
        P["curl_L"] = P["curl_R"] = 1.0
        return P
    L["punch_loop"] = ([(0, punch(0)), (6, punch(0.25)), (12, punch(0.5)), (18, punch(0.75)), (24, punch(1.0))], True, dict(note="alternating jabs"))

    # ------------------------------------------------------------------ shove
    def shv(ext, lunge):
        P = dict(foot_L=foot_ankle(-0.20 * lunge, "flat", x=0.14, side=1), foot_R=foot_ankle(0.12 * lunge, "flat", x=0.14, side=-1),
                 hips_loc=(0, -0.20 * lunge, -0.06 * lunge), spine=(0.12 + 0.25 * lunge, 0, 0), head=(0.0, 0, 0), jaw=0.3,
                 knee_L=V(0.14, -0.9, 0.5), knee_R=V(-0.14, -0.7, 0.5))
        y = -0.22 - 0.30 * ext - 0.0
        P["hand_L"] = H(1, 0.14, y, 1.30, f=(0, 0.1, 1), n=(0, -1, 0))
        P["hand_R"] = H(-1, 0.14, y, 1.30, f=(0, 0.1, 1), n=(0, -1, 0))
        P["curl_L"] = P["curl_R"] = 0.0
        return P
    L["shove"] = ([(0, shv(0, 0)), (5, shv(0.0, 0.1)), (9, shv(1, 1)), (13, shv(1, 1)), (22, shv(0.3, 0.3)), (30, shv(0, 0))], False, dict(note="two-handed shove; contact at frame 9"))

    # ------------------------------------------------------------------ aim_gun
    def aim(raise_, recoil=0.0):
        r = raise_
        P = dict(stand_feet(0, 0.14), spine=(0.05 * r, -0.22 * r + 0.0 * recoil, 0.04 * r), head=(0.0, -0.10 * r, 0.12 * r), neck=(0.0, 0, 0), jaw=0.0,
                 hips_rot=(0, 0.12 * r, 0), hips_loc=(0.0, 0.0, -0.01 * r + 0.0))
        P["foot_L"] = foot_ankle(-0.12 * r, "flat", x=0.15, side=1)
        P["foot_R"] = foot_ankle(0.06 * r, "flat", x=0.15, side=-1)
        P["hand_R"] = H(-1, 0.0, 0, 0)
        hr = V(-0.14, -0.30 + 0.07 * recoil, 1.31 + 0.03 * recoil)
        hl = V(-0.07, -0.54 + 0.07 * recoil, 1.31 + 0.05 * recoil)
        rest_r, rest_l = V(-0.296, -0.08, 0.84), V(0.296, -0.08, 0.84)
        P["hand_R"] = dict(pos=rest_r + (hr - rest_r) * r, f=V(0, -0.2, -0.3) if r < 0.5 else V(0, -0.6, -0.1), n=V(1, 0, 0))
        P["hand_L"] = dict(pos=rest_l + (hl - rest_l) * r, f=V(0, -1, 0), n=V(0, 0, 1))
        P["elbow_R"] = V(-0.30, 0.2, 1.25) if r > 0.3 else None
        P["elbow_L"] = V(0.28, 0.1, 1.0) if r > 0.3 else None
        for k in ("elbow_R", "elbow_L"):
            if P[k] is None:
                del P[k]
        P["curl_R"] = dict(index=0.35 * r + 0.1, middle=0.8 * r + 0.1, ring=0.8 * r + 0.1, pinky=0.8 * r + 0.1, thumb=0.5 * r + 0.1)
        P["curl_L"] = 0.5 * r + 0.15
        P["look"] = V(0, -3.0, 1.55) if r > 0 else V(0, -0.9, 1.63)
        if r > 0.5:
            P["blink"] = 0.0
            P["lid_up"] = 0.25  # squint
        return P
    L["aim_gun"] = ([(0, aim(0)), (12, aim(1.0)), (24, aim(1.0)), (34, aim(1.0)), (36, aim(1.0, 1.0)), (40, aim(1.0, 0.6)), (46, aim(1.0, 0.0))], False,
                    dict(note="raise shotgun to the shoulder, hold, shot at frame 36 (recoil), settle; right hand = HOOK_gun_grip_R, left hand = HOOK_gun_support_L / HOOK_hand_L"))

    # ------------------------------------------------------------------ hold_plate
    def plate(t):
        s = math.sin(2 * math.pi * t)
        P = dict(stand_feet(0, 0.11), spine=(-0.06, 0, 0), head=(0.05, 0, 0), hips_loc=(0, 0, 0.0))
        P["hand_L"] = H(1, 0.13, -0.40, 1.08 + 0.01 * s, f=(0, -1, 0), n=(0, 0, 1))
        P["hand_R"] = H(-1, 0.13, -0.40, 1.08 + 0.01 * s, f=(0, -1, 0), n=(0, 0, 1))
        P["elbow_L"] = V(0.28, 0.2, 1.0)
        P["elbow_R"] = V(-0.28, 0.2, 1.0)
        P["curl_L"] = P["curl_R"] = 0.25
        return P
    L["hold_plate"] = ([(0, plate(0)), (15, plate(0.5)), (30, plate(1.0))], True, dict(note="plate carried palms up in front of the waist; plate hook at the midpoint of HOOK_hand_L / HOOK_hand_R"))

    # ------------------------------------------------------------------ throw_up
    def toss(k, jump=0.0, bend=0.0):
        P = dict(stand_feet(0, 0.12))
        h = 1.08 + k * 0.72
        yy = -0.40 + k * 0.28
        P["hand_L"] = H(1, 0.13 + 0.10 * k, yy, h, f=(0, -1, 0.2 * k * 4), n=(0, 0, 1) if k < 0.5 else (0, -0.3, 1))
        P["hand_R"] = H(-1, 0.13 + 0.10 * k, yy, h, f=(0, -1, 0.2 * k * 4), n=(0, 0, 1) if k < 0.5 else (0, -0.3, 1))
        P["spine"] = (-0.06 - 0.25 * k + bend, 0, 0)
        P["head"] = (-0.15 * k, 0, 0)
        P["hips_loc"] = (0, 0, jump * 0.06 - 0.04 * bend)
        P["foot_L"] = foot_ankle(0, "flat", x=0.12, side=1, z_lift=jump * 0.05)
        P["foot_R"] = foot_ankle(0, "flat", x=0.12, side=-1, z_lift=jump * 0.05)
        P["curl_L"] = P["curl_R"] = 0.25 * (1 - k)
        P["jaw"] = 0.2 * k
        return P
    L["throw_up"] = ([(0, toss(0)), (5, toss(0.0, 0, 0.25)), (9, toss(0.7, 0.5)), (11, toss(1.0, 1.0)), (16, toss(1.0, 0.3)), (24, toss(0.85, 0.0))], False,
                     dict(note="dip, heave up, release at frame 11, arms stay raised"))

    # ------------------------------------------------------------------ point
    def pnt(r):
        P = dict(stand_feet(0, 0.12), spine=(0.03, 0.2 * r, 0), head=(0, 0.1 * r, 0), hips_rot=(0, 0.1 * r, 0))
        rest = V(-0.296, -0.08, 0.84)
        tgt = V(-0.24, -0.50, 1.38)
        P["hand_R"] = dict(pos=rest + (tgt - rest) * r, f=V(-0.12, -1, 0.08) if r > 0.1 else V(0, 0, -1), n=V(0.6, 0, -0.8))
        P["curl_R"] = dict(index=0.0, middle=0.95 * r + 0.1, ring=0.95 * r + 0.1, pinky=0.95 * r + 0.1, thumb=0.6 * r + 0.1)
        P["look"] = V(-0.4, -3, 1.5)
        P["jaw"] = 0.0
        return P
    L["point"] = ([(0, pnt(0)), (10, pnt(1.0)), (16, merge(pnt(1.0), dict(hand_R=dict(pos=V(-0.24, -0.50, 1.40), f=V(-0.12, -1, 0.12), n=V(0.6, 0, -0.8))))), (22, pnt(1.0)), (34, pnt(1.0)), (44, pnt(0))],
                  False, dict(note="right arm points forward-right with the index finger; jab at frame 16"))

    # ------------------------------------------------------------------ slap
    def slp(ph):
        # ph: 0 rest, 1 wound up (hand across the body on the left), 2 contact (far right), 3 follow-through
        P = dict(stand_feet(0, 0.14))
        if ph == 0:
            P["hand_R"] = H(-1, 0.296, -0.08, 0.84)
            return P
        hands = {1: (V(0.25, -0.22, 1.35), -0.5, 0.45), 2: (V(-0.50, -0.38, 1.38), 0.55, -0.55), 3: (V(-0.45, -0.15, 1.30), 0.65, -0.65)}
        pos, tw, hip = hands[ph]
        P["hand_R"] = dict(pos=pos, f=V(0.0, -1, 0.15) if ph != 2 else V(-0.2, -1, 0.0), n=V(1, -0.2, 0) if ph == 1 else V(-1, -0.2, 0))
        P["spine"] = (0.05, -tw, 0)
        P["hips_rot"] = (0, hip * 0.5, 0)
        P["head"] = (0, tw * 0.4, 0)
        P["curl_R"] = 0.0
        P["foot_L"] = foot_ankle(0.0, "flat", x=0.15, side=1)
        P["foot_R"] = foot_ankle(0.05, "flat", x=0.15, side=-1)
        P["hand_L"] = H(1, 0.3, -0.1, 0.9)
        return P
    L["slap"] = ([(0, slp(0)), (8, slp(1)), (11, slp(2)), (14, slp(3)), (24, slp(0))], False, dict(note="right-hand horizontal slap from the left across to the right; contact at frame 11, target centre about 0.5 m to the character's right, 0.4 m ahead"))

    # ------------------------------------------------------------------ hug_leg
    def hug(t):
        s = math.sin(2 * math.pi * t)
        P = dict(stand_feet(0, 0.10), hips_loc=(0, -0.08, -0.05), spine=(0.30 + 0.02 * s, 0, 0), head=(0.35, 0.0, 0.1), neck=(0.1, 0, 0))
        P["hand_L"] = H(1, 0.05 - 0.01 * s, -0.36, 0.86, f=(-0.6, -0.6, -0.5), n=(-0.3, 0, 0.9))
        P["hand_R"] = H(-1, 0.05 - 0.01 * s, -0.36, 0.86, f=(0.6, -0.6, -0.5), n=(0.3, 0, 0.9))
        P["elbow_L"] = V(0.34, -0.1, 1.0)
        P["elbow_R"] = V(-0.34, -0.1, 1.0)
        P["curl_L"] = P["curl_R"] = 0.8
        P["knee_L"] = V(0.10, -0.9, 0.5)
        P["knee_R"] = V(-0.10, -0.9, 0.5)
        P["look"] = V(0, -0.5, 0.5)
        return P
    L["hug_leg"] = ([(0, hug(0)), (18, hug(0.5)), (36, hug(1.0))], True, dict(note="clinging to a leg that sits at about 0.28 m in front of the character axis"))

    # ------------------------------------------------------------------ hand_over_envelope
    def env(ph):
        P = dict(stand_feet(0, 0.12), spine=(0.04 * ph, -0.1 * ph, 0), head=(0, 0, 0))
        rest = V(-0.296, -0.08, 0.84)
        tgt = V(-0.13, -0.52, 1.14)
        pos = rest + (tgt - rest) * ph
        P["hand_R"] = dict(pos=pos, f=V(0, -1, 0) if ph > 0 else V(0, 0, -1), n=V(0.7, 0, 0.7) if ph > 0 else V(1, 0, 0))
        P["curl_R"] = dict(index=0.4, middle=0.4, ring=0.4, pinky=0.4, thumb=0.35)
        P["hand_L"] = H(1, 0.296, -0.08, 0.84)
        return P
    open_ = merge(env(1.0), dict(curl_R=0.0))
    L["hand_over_envelope"] = ([(0, env(0)), (10, env(0.6)), (18, env(1.0)), (26, env(1.0)), (30, open_), (40, merge(open_, dict(hand_R=dict(pos=V(-0.16, -0.50, 1.12), f=V(0, -1, 0), n=V(0.7, 0, 0.7))))), (52, env(0))],
                               False, dict(note="offers an envelope with the right hand, releases at frame 30"))

    # ------------------------------------------------------------------ thinking
    def thk(t):
        s = math.sin(2 * math.pi * t)
        P = dict(stand_feet(0, 0.11), spine=(0.03, 0, 0.05), head=(0.0 + 0.02 * s, 0.25, 0.12), hips_rot=(0, 0, 0.03))
        P["hand_R"] = dict(pos=V(-0.07, -0.20, 1.44), f=V(0.3, -0.2, 1), n=V(0.8, -0.5, 0))
        P["curl_R"] = dict(index=0.2, middle=0.8, ring=0.8, pinky=0.8, thumb=0.5)
        P["hand_L"] = dict(pos=V(-0.02, -0.26, 1.16), f=V(-1, -0.2, 0), n=V(0, 0, 1))
        P["elbow_R"] = V(-0.26, -0.1, 1.15)
        P["curl_L"] = 0.3
        P["look"] = V(0.8, -2.5, 2.0)
        return P
    L["thinking"] = ([(0, thk(0)), (30, thk(0.5)), (60, thk(1.0))], True, dict(note="chin-stroking pose"))

    # ------------------------------------------------------------------ arm_whip (left hand)
    def whp(ph):
        P = dict(stand_feet(0, 0.14))
        if ph == "rest":
            return P
        k = {"wind": (V(0.26, 0.18, 1.80), -0.25, -0.35, 0.0), "lash": (V(0.12, -0.52, 1.28), 0.45, 0.45, -0.02), "follow": (V(0.18, -0.30, 0.90), 0.65, 0.5, -0.05)}[ph]
        pos, flex, tw, dz = k
        P["hand_L"] = dict(pos=pos, f=V(0.2, 0.0, 1.0) if ph == "wind" else V(0, -1, -0.3), n=V(-1, 0, 0))
        P["spine"] = (flex, tw, 0)
        P["hips_rot"] = (0, tw * 0.5, 0)
        P["head"] = (-flex * 0.3, tw * 0.3, 0)
        P["foot_L"] = foot_ankle(-0.18 if ph != "wind" else 0.0, "flat", x=0.15, side=1)
        P["foot_R"] = foot_ankle(0.12 if ph != "wind" else 0.1, "flat", x=0.15, side=-1)
        P["jaw"] = 0.3 if ph != "wind" else 0.1
        P["curl_L"] = 0.9
        P["hand_R"] = H(-1, 0.3, -0.15, 1.1)
        return P
    L["arm_whip"] = ([(0, whp("rest")), (10, whp("wind")), (16, whp("wind")), (19, whp("lash")), (24, whp("follow")), (36, whp("follow")), (46, whp("rest"))], False,
                     dict(note="overhand whip with the left hand (HOOK_whip_grip_L): wind-up, crack at frame 19, follow-through"))

    # ------------------------------------------------------------------ jolt_hit
    def jolt(a, b=0.0):
        P = dict(stand_feet(0, 0.12), spine=(-0.5 * a + 0.25 * b, 0.2 * a, 0.0), head=(-0.5 * a, 0.3 * a, 0), hips_loc=(0, 0.04 * a, 0.02 * a), hips_rot=(0.1 * a, 0, 0), jaw=0.45 * a)
        P["hand_L"] = H(1, 0.30 + 0.30 * a, -0.08 - 0.1 * a, 0.84 + 0.45 * a, f=(0.3, 0, -1))
        P["hand_R"] = H(-1, 0.30 + 0.30 * a, -0.08 - 0.1 * a, 0.84 + 0.45 * a, f=(-0.3, 0, -1))
        P["curl_L"] = P["curl_R"] = 0.0
        P["foot_L"] = foot_ankle(0.05 * a, "flat", x=0.12, side=1, z_lift=0.06 * a)
        P["blink"] = 1.0 if a > 0.9 else 0.0
        return P
    L["jolt_hit"] = ([(0, jolt(0)), (3, jolt(1.0)), (6, jolt(0.7, 0.2)), (10, jolt(0.9)), (16, jolt(0.4, 0.8)), (24, jolt(0.15, 1.0)), (32, jolt(0.1, 1.0))], False,
                     dict(note="whip hit: arch back, arms fling out, rebound into a slump"))

    # ------------------------------------------------------------------ pull_cable
    def pull(t):
        a = 0.5 - 0.5 * math.cos(2 * math.pi * t)  # 0 reach, 1 heave
        P = dict(foot_L=foot_ankle(-0.30, "flat", x=0.17, side=1), foot_R=foot_ankle(0.28, "flat", x=0.17, side=-1), hips_loc=(0, 0.08 + 0.10 * a, -0.10 - 0.02 * a),
                 spine=(0.22 - 0.50 * a, 0, 0), head=(0.05, 0, 0), jaw=0.2 * a, knee_L=V(0.2, -0.9, 0.5), knee_R=V(-0.2, -0.9, 0.5), hips_rot=(0, 0, 0))
        y = -0.55 + 0.38 * a
        P["hand_L"] = H(1, 0.10, y, 1.00 + 0.05 * a, f=(0, -1, -0.2), n=(-1, 0, 0))
        P["hand_R"] = H(-1, 0.10, y, 1.00 + 0.05 * a, f=(0, -1, -0.2), n=(1, 0, 0))
        P["curl_L"] = P["curl_R"] = 0.9
        P["elbow_L"] = V(0.3, 0.25, 1.0)
        P["elbow_R"] = V(-0.3, 0.25, 1.0)
        return P
    L["pull_cable"] = ([(0, pull(0)), (9, pull(0.25)), (18, pull(0.5)), (27, pull(0.75)), (36, pull(1.0))], True, dict(note="two-handed heave on a cable held 0.5 m ahead at belt height, braced stance"))

    # ------------------------------------------------------------------ hold_printout
    def prt(t):
        s = math.sin(2 * math.pi * t)
        P = dict(stand_feet(0, 0.12), spine=(0.0, -0.1, 0), head=(0.0, 0.1, 0), hips_loc=(0, 0, 0))
        P["hand_R"] = dict(pos=V(-0.16, -0.36 + 0.005 * s, 1.30), f=V(0, -0.3, 1), n=V(0.3, 0.9, 0))
        P["hand_L"] = dict(pos=V(0.04, -0.34, 1.12), f=V(-0.8, -0.4, 0), n=V(0, 0, 1))
        P["curl_R"] = dict(index=0.5, middle=0.5, ring=0.5, pinky=0.5, thumb=0.4)
        P["elbow_R"] = V(-0.30, 0.2, 1.15)
        return P
    L["hold_printout"] = ([(0, prt(0)), (30, prt(1))], True, dict(note="holds a printout up in the right hand (HOOK_paper_R) at chest height"))
    return L


def make_all(arm, K, J, frames_hand, aid, only=None):
    rig = Rig(arm, K, J, frames_hand)
    lib = library(K)
    names = []
    for name, (keys, loop, meta) in lib.items():
        if only and name not in only:
            continue
        rig.make_action(aid, name, keys, loop, meta)
        names.append("ACT_%s_%s" % (aid, name))
    arm.animation_data.action = None
    rig.clear()
    return names


def describe(names):
    out = []
    for n in names:
        a = bpy.data.actions[n]
        d = dict(name=n, frame_start=int(a.frame_range[0]), frame_end=int(a.frame_range[1]), fps=FPS, loop=bool(a.get("loop", False)))
        for k in a.keys():
            if k not in ("loop",):
                v = a[k]
                d[k] = v if isinstance(v, (int, float, str, bool)) else str(v)
        out.append(d)
    return out
