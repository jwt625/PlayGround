"""Motion library v2: pose engine, timeline sampler, filters, baker.

Works BY BONE NAME on the existing character skeleton (root, hips, spine_1, spine_2, chest, neck, head, jaw, lids,
clavicle_L/R, IK targets ik_hand/ik_foot/ik_elbow/ik_knee, ctl_look, ctl_hand finger curls).  Added bones are ignored, so the
same code bakes for gary, manager, npc and the later v2 meshes.  Rest dimensions are read from the armature (no mesh needed).

Pose language (baseline character space = metres at stature 1.75: x = character left, y = back, z = up; forward = -y).
All authoring is deterministic; every action is sampled per frame at 30 fps and written as LINEAR dense keys.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

FPS = 30
CURLS = ["index", "middle", "ring", "pinky", "thumb"]
V = lambda *a: np.array(a, float)  # noqa: E731


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def nrm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def ortho(v, f):
    v = np.asarray(v, float)
    return nrm(v - f * np.dot(v, f))


# ---------------------------------------------------------------------------------------------------- easing
def ease(name, u):
    u = min(max(u, 0.0), 1.0)
    if name == "lin":
        return u
    if name == "smooth":
        return u * u * (3 - 2 * u)
    if name == "in":      # accelerate into the key (strike)
        return u ** 2.4
    if name == "in2":
        return u ** 1.6
    if name == "out":     # decelerate into the key (settle)
        return 1 - (1 - u) ** 2.4
    if name == "out2":
        return 1 - (1 - u) ** 1.7
    if name == "snap":    # fast arrival with overshoot (easeOutBack)
        c1 = 2.2
        return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2
    if name == "back":    # mild overshoot
        c1 = 1.2
        return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2
    if name == "wind":    # reverse a little before moving (anticipation inside the segment)
        c1 = 1.7
        return (c1 + 1) * u ** 3 - c1 * u ** 2
    if name == "elastic":  # fast arrival, damped oscillation around the key
        return 1 - (1 - u) ** 2 * math.cos(2 * math.pi * 1.25 * u)
    if name == "cut":     # jump at the end of the segment
        return 0.0 if u < 1.0 else 1.0
    if name == "cut0":    # jump at the start of the segment
        return 1.0 if u > 0.0 else 0.0
    raise KeyError(name)


def spring_filter(x, dt, freq=5.0, zeta=0.32):
    """Underdamped follow of a sampled signal (overlap / follow-through); x: (N, C)."""
    w = 2 * math.pi * freq
    y = x[0].astype(float).copy()
    vel = np.zeros_like(y)
    out = np.empty_like(x, dtype=float)
    sub = 4
    h = dt / sub
    for i in range(len(x)):
        for _ in range(sub):
            acc = w * w * (x[i] - y) - 2 * zeta * w * vel
            vel += acc * h
            y = y + vel * h
        out[i] = y
    return out


# ---------------------------------------------------------------------------------------------------- pose container
KEYSPEC = dict(
    root_loc=3, root_rot=3, hips_loc=3, hips_rot=3, spine=3, neck=3, head=3, jaw=1, blink=1, lid_up=1, lid_lo=1,
    clav_L=2, clav_R=2,                  # (shrug up, shoulder forward) radians
    hand_L=9, hand_R=9,                  # pos(3), finger dir f(3), palm normal n(3); baseline character space
    hfol_L=1, hfol_R=1,                  # 0 = absolute, 1 = hand target rigidly follows the attach bone (chest/head/hips/root)
    ffol_L=1, ffol_R=1,                  # 0 = foot planted (absolute), 1 = foot rigidly follows the root bone transform (falls)
    elbow_L=3, elbow_R=3,                # pole offset from the shoulder joint; x positive = outward (side relative)
    curl_L=5, curl_R=5,
    foot_L=5, foot_R=5,                  # ankle target: x (absolute, side sign applied by the engine), y, z, pitch (+ toe up), yaw
    knee_L=3, knee_R=3,                  # pole position relative to the hip joint (x outward positive)
    look=3, look_mix=1, gaze=2,          # abs gaze target (baseline), mix with auto head-follow gaze; gaze = (yaw, pitch) offset
    sq_c=3, sq_h=3, sq_s=3, sq_b=3, sq_n=3,  # squash/stretch scale on chest, head, spine_1/2, hips, neck
    ground=1,                            # 1 = clamp the body above the floor with the capsule model (falls / lying)
)
CONT_KEYS = [k for k in KEYSPEC if k != "ground"]


def neutral():
    P = {k: np.zeros(n) for k, n in KEYSPEC.items()}
    P["sq_c"][:] = P["sq_h"][:] = P["sq_s"][:] = P["sq_b"][:] = P["sq_n"][:] = 1.0
    P["curl_L"][:] = P["curl_R"][:] = 0.18
    P["hfol_L"][:] = P["hfol_R"][:] = 0.0
    P["hand_L"] = np.concatenate([V(0.296, -0.08, 0.84), nrm(V(0.04, -0.22, -0.97)), nrm(V(-1, 0, 0))])
    P["hand_R"] = np.concatenate([V(-0.296, -0.08, 0.84), nrm(V(-0.04, -0.22, -0.97)), nrm(V(1, 0, 0))])
    P["elbow_L"] = V(0.16, 0.38, -0.31)
    P["elbow_R"] = V(0.16, 0.38, -0.31)
    P["foot_L"] = V(0.095, 0.0, 0.0707, 0.0, 0.0)
    P["foot_R"] = V(0.095, 0.0, 0.0707, 0.0, 0.0)
    P["knee_L"] = V(0.03, -0.55, 0.0)
    P["knee_R"] = V(0.03, -0.55, 0.0)
    P["look"] = V(0, -0.9, 1.6)
    return P


def copy_pose(P):
    return {k: np.array(v, float) for k, v in P.items()}


def lerp_pose(A, B, e):
    return {k: A[k] + (B[k] - A[k]) * e for k in A}


def mirror_pose(P):
    """Swap L/R and mirror x: a left-lead pose becomes right-lead."""
    Q = copy_pose(P)
    for a in ("hand", "hfol", "ffol", "elbow", "curl", "foot", "knee", "clav"):
        Q[a + "_L"], Q[a + "_R"] = P[a + "_R"].copy(), P[a + "_L"].copy()
    for k in ("hand_L", "hand_R"):
        Q[k][[0, 3, 6]] *= -1
    # elbow and knee x are outward-relative, foot x is a magnitude: unchanged
    for k in ("hips_loc", "root_loc"):
        Q[k][0] *= -1
    for k in ("hips_rot", "spine", "neck", "head", "root_rot"):
        Q[k][1] *= -1
        Q[k][2] *= -1
    for k in ("foot_L", "foot_R"):
        Q[k][4] *= -1
    Q["look"][0] *= -1
    Q["gaze"][0] *= -1
    return Q


def hand(pos, f=(0, -1, 0.1), n=(-1, 0, 0), side=1):
    """Left hand 9-vector (side=1); the palm normal n is given for the LEFT hand (medial = -x). side=-1 mirrors for the right hand."""
    pos, f, n = V(*pos), nrm(V(*f)), nrm(V(*n))
    if side < 0:
        pos, f, n = pos * V(-1, 1, 1), f * V(-1, 1, 1), n * V(-1, 1, 1)
    return np.concatenate([pos, f, n])


# ---------------------------------------------------------------------------------------------------- foot model (baseline m)
FOOT_HEEL = 0.10      # ankle -> back of sole (measured on the v1 boot mesh: 0.096)
FOOT_BALL = 0.19      # ankle -> pivot ahead (toe tip 0.233; pivot slightly behind to limit toe dip)
FOOT_TOE = 0.23
AZ = 0.0707           # ankle height above the sole


def ankle_from_pivot(piv_xy, pivot, pitch, x, yaw=0.0, lift=0.0, az=AZ):
    """Ankle target for a foot that rotates by `pitch` (+ toe up) about a sole pivot fixed at (x, piv_y). pivot: 'heel', 'ball', 'flat'.
    flat: piv_y is the ankle y (foot flat on the ground)."""
    if pivot == "flat":
        return V(x, piv_xy, az + lift, pitch, yaw)
    # local sole points relative to the ankle: (y, z)
    py = FOOT_HEEL if pivot == "heel" else -FOOT_BALL
    pz = -az
    c, s = math.cos(-pitch), math.sin(-pitch)   # positive pitch = toe up = rotate so that +y(back)... use rotation about X by -pitch
    # rotate (y, z) of the pivot vector by angle th about X: y' = y c - z s ; z' = y s + z c  with th = -pitch (toe up raises -y)
    yy = py * c - pz * s
    zz = py * s + pz * c
    # ankle = pivot_world - rotated_pivot_local ; pivot world z = lift (0 on the ground)
    return V(x, piv_xy - yy, lift - zz, pitch, yaw)


# ---------------------------------------------------------------------------------------------------- the rig
class Rig2:
    FK = ["hips", "spine_1", "spine_2", "chest", "neck", "head", "jaw", "lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R", "root"]
    SCALE_BONES = ["hips", "spine_1", "spine_2", "chest", "neck", "head"]
    IK = ["ik_hand_L", "ik_hand_R", "ik_foot_L", "ik_foot_R", "ik_elbow_L", "ik_elbow_R", "ik_knee_L", "ik_knee_R", "ctl_look"]
    CLAV = ["clavicle_L", "clavicle_R"]
    # capsule model radii (baseline m) for the floor clamp: bone -> (radius)
    CAPS = dict(head=0.105, neck=0.06, chest=0.125, spine_2=0.12, spine_1=0.12, hips=0.12, upper_arm_L=0.05, upper_arm_R=0.05,
                forearm_L=0.04, forearm_R=0.04, hand_L=0.04, hand_R=0.04, thigh_L=0.085, thigh_R=0.085, shin_L=0.06, shin_R=0.06)

    def __init__(self, root):
        self.root_obj = root
        aid = root["asset_id"]
        self.aid = aid
        self.arm = bpy.data.objects[aid + "_rig"]
        arm = self.arm
        self.pb = arm.pose.bones
        bones = arm.data.bones
        # K from the rest hip height (baseline hip height 0.925 m); the p_stature_m property is the hat-on height, not K
        self.K = float(bones["hips"].head_local[2]) / 0.925
        K = self.K
        self.rest_rot = {b.name: np.array(b.matrix_local.to_3x3()) for b in bones}
        self.rest_mat = {b.name: b.matrix_local.copy() for b in bones}
        hd = lambda n: np.array(bones[n].head_local) / K  # noqa: E731
        tl = lambda n: np.array(bones[n].tail_local) / K  # noqa: E731
        self.dims = dict(
            hip=hd("thigh_L"), knee=hd("shin_L"), ankle=hd("foot_L"), ball=tl("foot_L"), sh=hd("upper_arm_L"), el=hd("forearm_L"),
            wr=hd("hand_L"), clav=hd("clavicle_L"), hips=hd("hips"), head=hd("head"))
        d = self.dims
        self.Lthigh = float(np.linalg.norm(d["knee"] - d["hip"]))
        self.Lshin = float(np.linalg.norm(d["ankle"] - d["knee"]))
        self.Lup = float(np.linalg.norm(d["el"] - d["sh"]))
        self.Lfore = float(np.linalg.norm(d["wr"] - d["el"]))
        self.az = float(d["ankle"][2])
        # hand frames per side (finger direction, palm normal at rest)
        self.frames_hand = {}
        for sn, side in (("L", 1), ("R", -1)):
            f = nrm(self.rest_rot["hand_" + sn][:, 1])
            n = ortho(V(-side, 0, 0), f)
            self.frames_hand[sn] = (f, n)
        for n in self.FK:
            self.pb[n].rotation_mode = "XYZ"
        for n in self.SCALE_BONES + ["root"]:
            self.pb[n].rotation_mode = "XYZ" if n in self.FK else self.pb[n].rotation_mode
        self.meshes = [o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
        self.warn = []
        self._channels()

    # ------------------------------------------------------------ channel list
    def _channels(self):
        ch = []
        for n in self.FK:
            ch += [(n, "rotation_euler", i) for i in range(3)]
        for n in ("root", "hips"):
            ch += [(n, "location", i) for i in range(3)]
        for n in self.SCALE_BONES:
            ch += [(n, "scale", i) for i in range(3)]
        for n in self.CLAV:
            ch += [(n, "rotation_quaternion", i) for i in range(4)]
        for n in self.IK:
            ch += [(n, "location", i) for i in range(3)]
            if n.startswith(("ik_hand", "ik_foot")):
                ch += [(n, "rotation_quaternion", i) for i in range(4)]
        for sn in ("L", "R"):
            for f in CURLS:
                ch.append(("ctl_hand_" + sn, '["curl_%s"]' % f, -1))
        self.channels = ch

    def read(self):
        out = np.empty(len(self.channels))
        pb = self.pb
        for k, (n, attr, i) in enumerate(self.channels):
            if i < 0:
                out[k] = float(pb[n][attr[2:-2]])
            else:
                out[k] = getattr(pb[n], attr)[i]
        return out

    # ------------------------------------------------------------ pose solving
    def clear(self):
        for pb in self.pb:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.scale = (1, 1, 1)
        for sn in ("L", "R"):
            for f in CURLS:
                self.pb["ctl_hand_" + sn]["curl_" + f] = 0.12
        bpy.context.view_layer.update()

    def set_world(self, name, pos, R=None, scale_pos=True):
        pb = self.pb[name]
        R = self.rest_rot[name] if R is None else R
        p = np.asarray(pos, float) * (self.K if scale_pos else 1.0)
        pb.matrix = Matrix.Translation(Vector(p)) @ Matrix(R.tolist()).to_4x4()

    def delta_matrix(self, bone):
        """Pose-minus-rest transform of a bone in armature space (K-scaled coordinates)."""
        return self.pb[bone].matrix @ self.rest_mat[bone].inverted()

    def hand_R_world(self, sn, f, n):
        f0, n0 = self.frames_hand[sn]
        F0 = np.stack([f0, n0, np.cross(f0, n0)], axis=1)
        f = nrm(f)
        n = nrm(n - f * np.dot(n, f))
        F1 = np.stack([f, n, np.cross(f, n)], axis=1)
        return F1 @ F0.T @ self.rest_rot["ik_hand_" + sn]

    def pose(self, P, attach=None):
        """Apply a full pose dict P to the pose bones (solves IK targets by matrix assignment)."""
        K = self.K
        pb = self.pb
        attach = attach or {"L": "chest", "R": "chest"}
        self.clear()
        for name in ("root", "hips"):
            loc = P[name + "_loc"]
            pb[name].location = (loc[0] * K, loc[2] * K, -loc[1] * K)
        for name in ("root", "hips"):
            r = P[name + "_rot"]
            pb[name].rotation_euler = (r[0], r[1], -r[2])
        sp = P["spine"]
        for n, w in (("spine_1", 0.25), ("spine_2", 0.35), ("chest", 0.40)):
            pb[n].rotation_euler = (sp[0] * w, sp[1] * w, -sp[2] * w)
        for n in ("neck", "head"):
            r = P[n]
            pb[n].rotation_euler = (r[0], r[1], -r[2])
        pb["jaw"].rotation_euler = (float(P["jaw"][0]), 0, 0)
        bl = float(P["blink"][0])
        for sn in ("L", "R"):
            pb["lid_up_" + sn].rotation_euler = (0.95 * bl + float(P["lid_up"][0]), 0, 0)
            pb["lid_lo_" + sn].rotation_euler = (-0.12 * bl - float(P["lid_lo"][0]), 0, 0)
        for n, k in (("chest", "sq_c"), ("head", "sq_h"), ("spine_1", "sq_s"), ("spine_2", "sq_s"), ("hips", "sq_b"), ("neck", "sq_n")):
            pb[n].scale = tuple(P[k])
        bpy.context.view_layer.update()
        # clavicles (rotated about their head in armature space)
        for sn, side in (("L", 1), ("R", -1)):
            c = P["clav_" + sn]
            if abs(c[0]) > 1e-6 or abs(c[1]) > 1e-6:
                b = pb["clavicle_" + sn]
                head = Vector(b.head)
                Rw = rz(-side * c[1]) @ ry(-side * c[0])
                b.matrix = Matrix.Translation(head) @ Matrix(Rw.tolist()).to_4x4() @ Matrix.Translation(-head) @ b.matrix
        bpy.context.view_layer.update()
        # legs first (hips location is final): reach clamp, then IK targets
        for sn, side in (("L", 1), ("R", -1)):
            fo = P["foot_" + sn]
            pos = V(side * fo[0], fo[1], fo[2])
            R = rz(fo[4]) @ rx(-fo[3]) @ self.rest_rot["ik_foot_" + sn]
            ff = float(P["ffol_" + sn][0])
            if ff > 1e-6:
                Md = self.delta_matrix("root")
                A = np.array(Md.to_3x3())
                pw = (A @ (pos * K) + np.array(Md.translation)) / K
                q0 = Matrix(R.tolist()).to_quaternion()
                q1 = Matrix((A @ R).tolist()).to_quaternion()
                R = np.array(q0.slerp(q1, ff).to_matrix())
                pos = pos * (1 - ff) + pw * ff
            hip = np.array(pb["thigh_" + sn].head) / K
            Lmax = (self.Lthigh + self.Lshin) * 0.995
            dv = pos - hip
            dist = np.linalg.norm(dv)
            if dist > Lmax:
                self.warn.append(("reach_foot_" + sn, float(dist - Lmax)))
                pos = hip + dv / dist * Lmax
            self.set_world("ik_foot_" + sn, pos, R)
            kn = P["knee_" + sn]
            self.set_world("ik_knee_" + sn, hip + V(side * kn[0], kn[1], kn[2]))
        # eyes: auto gaze follows the head, blended with an absolute target
        hm = self.delta_matrix("head")
        fwd = np.array((hm.to_3x3() @ Vector((0, -1, 0))))
        gz = P["gaze"]
        fwd = rz(-gz[0]) @ rx(gz[1]) @ fwd if (abs(gz[0]) + abs(gz[1])) > 1e-9 else fwd
        hpos = np.array(pb["head"].head) / K + V(0, 0, 0.05)
        auto = hpos + nrm(fwd) * 0.9
        mix = float(P["look_mix"][0])
        self.set_world("ctl_look", auto * (1 - mix) + P["look"] * mix)
        # arms
        for sn, side in (("L", 1), ("R", -1)):
            h = P["hand_" + sn]
            pos, f, n = h[0:3].copy(), h[3:6].copy(), h[6:9].copy()
            fol = float(P["hfol_" + sn][0])
            if fol > 1e-6:
                Md = self.delta_matrix(attach[sn])
                A = np.array(Md.to_3x3())
                t = np.array(Md.translation)
                pw = A @ (pos * K) + t
                pos = pos * (1 - fol) + (pw / K) * fol
                f = nrm(f * (1 - fol) + (A @ f) * fol)
                n = nrm(n * (1 - fol) + (A @ n) * fol)
            sh = np.array(pb["upper_arm_" + sn].head) / K
            Lmax = (self.Lup + self.Lfore) * 0.995
            dv = pos - sh
            dist = np.linalg.norm(dv)
            if dist > Lmax:
                self.warn.append(("reach_hand_" + sn, float(dist - Lmax)))
                pos = sh + dv / dist * Lmax
            R = self.hand_R_world(sn, f, n)
            self.set_world("ik_hand_" + sn, pos, R)
            e = P["elbow_" + sn]
            self.set_world("ik_elbow_" + sn, sh + V(side * e[0], e[1], e[2]))
            for i, fn in enumerate(CURLS):
                pb["ctl_hand_" + sn]["curl_" + fn] = float(P["curl_" + sn][i])
        bpy.context.view_layer.update()

    # ------------------------------------------------------------ floor clamp (capsule model)
    def lowest_z(self):
        """Lowest point (baseline m) of the body using capsule radii around the bones (no mesh evaluation)."""
        K = self.K
        zs = []
        for n, r in self.CAPS.items():
            b = self.pb[n]
            zs.append(min(b.head[2], b.tail[2]) / K - r)
        # boots: sole contact approximated by ankle - az (feet may be lifted)
        for sn in ("L", "R"):
            b = self.pb["foot_" + sn]
            zs.append(min(b.head[2], b.tail[2]) / K - self.az)
        return min(zs)


# ---------------------------------------------------------------------------------------------------- timeline
class Timeline:
    """Pose keys at integer frames. Unspecified channels carry over from the previous key. Foot channels always ease smoothly
    and get an automatic lift arc when the foot travels; set lift=<m> on a key to control the arc height (0 disables)."""

    def __init__(self, base=None):
        self.keys = []   # (frame, pose, ease_into_key, lift)
        self.base = copy_pose(base) if base is not None else neutral()

    def key(self, f, ease_="smooth", lift=None, **over):
        P = copy_pose(self.keys[-1][1] if self.keys else self.base)
        for k, v in over.items():
            P[k] = np.array(v, float).reshape(P[k].shape) if np.size(v) == P[k].size else np.full(P[k].shape, float(np.asarray(v).ravel()[0]))
        if self.keys and f <= self.keys[-1][0]:
            raise ValueError("timeline key %s not after %s" % (f, self.keys[-1][0]))
        self.keys.append((f, P, ease_, lift))
        return P

    def hold(self, f):
        """Hold the previous pose until frame f (hit-stop)."""
        self.key(f, "cut")

    def last(self):
        return self.keys[-1][1]

    def eval(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for (t0, A, _, _), (t1, B, es, lf) in zip(ks[:-1], ks[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0)
                e = ease(es, u)
                P = lerp_pose(A, B, e)
                if es == "cut":
                    P = dict(A) if u < 1.0 else dict(B)
                P["ground"] = B["ground"] if u >= 0.5 else A["ground"]
                es_ = ease("smooth", u)
                for sn in ("L", "R"):
                    a, b = A["foot_" + sn], B["foot_" + sn]
                    if es != "cut":
                        P["foot_" + sn] = a + (b - a) * es_
                        dist = float(np.linalg.norm((b - a)[:3]))
                        if dist > 0.03 and lf != 0.0:
                            h = lf if lf is not None else min(0.06 + 0.25 * dist, 0.16)
                            P["foot_" + sn][2] += h * math.sin(math.pi * u)
                return P
        return ks[-1][1]


# ---------------------------------------------------------------------------------------------------- sampling and baking
def stack(frames_poses):
    """list of pose dicts -> dict key -> (N, size) array."""
    return {k: np.stack([p[k] for p in frames_poses]) for k in frames_poses[0]}


def unstack(S, i):
    return {k: S[k][i] for k in S}


def apply_overlap(S, overlap, dt=1.0 / FPS):
    """overlap: {key: (freq, zeta)} spring lag on whole channels (hand vectors are re-normalised)."""
    for k, (fr, zt) in (overlap or {}).items():
        S[k] = spring_filter(S[k], dt, fr, zt)
        if k.startswith("hand_"):
            for a in (3, 6):
                S[k][:, a:a + 3] /= np.maximum(np.linalg.norm(S[k][:, a:a + 3], axis=1, keepdims=True), 1e-9)
    return S


def sample_poses(fn, n_frames, loop, overlap=None, mirror=False):
    """fn(frame_float) -> pose. Returns list of n_frames+1 poses (frame 0..n). Loops are filtered over three cycles."""
    if loop:
        fr = list(range(-n_frames, 2 * n_frames + 1))
    else:
        fr = list(range(0, n_frames + 1))
    poses = [fn(float(f)) for f in fr]
    S = stack(poses)
    S = apply_overlap(S, overlap)
    if loop:
        S = {k: v[n_frames:2 * n_frames + 1] for k, v in S.items()}
        for k in S:
            S[k][-1] = S[k][0]   # exact loop closure
    out = [unstack(S, i) for i in range(len(next(iter(S.values()))))]
    if mirror:
        out = [mirror_pose(p) for p in out]
    return out


def solve_frames(rig, poses, attach=None, ground_pad=0.0):
    """Solve every pose with the rig; returns (N, nch) channel array. 'ground' poses are clamped above the floor."""
    out = []
    for P in poses:
        rig.pose(P, attach)
        if P["ground"][0] > 0.5:
            lz = rig.lowest_z()
            dz = ground_pad - lz
            if abs(dz) > 1e-4:
                Q = copy_pose(P)
                Q["root_loc"] = Q["root_loc"] + V(0, 0, dz)
                rig.pose(Q, attach)
        out.append(rig.read())
    return np.array(out)


# channel groups (for layered actions)
GROUPS = dict(
    lower=lambda n: n in ("root", "hips", "ik_foot_L", "ik_foot_R", "ik_knee_L", "ik_knee_R"),
    upper=lambda n: not (n in ("root", "hips", "ik_foot_L", "ik_foot_R", "ik_knee_L", "ik_knee_R")),
    arms=lambda n: n.startswith(("ik_hand", "ik_elbow", "ctl_hand", "clavicle")),
    arm_L=lambda n: n.endswith("_L") and n.startswith(("ik_hand", "ik_elbow", "ctl_hand", "clavicle")),
    arm_R=lambda n: n.endswith("_R") and n.startswith(("ik_hand", "ik_elbow", "ctl_hand", "clavicle")),
    head=lambda n: n in ("neck", "head", "jaw", "lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R", "ctl_look"),
    full=lambda n: True,
)


def write_action(rig, name, data, n_frames, loop, group="full", meta=None, quat_fix=True, eps=1e-5):
    """data: (N, nch). Creates ACT_<aid>_v2_<name> with direct fcurves (LINEAR). Constant channels get a single key."""
    arm = rig.arm
    full = "ACT_%s_v2_%s" % (rig.aid, name)
    if full in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[full])
    act = bpy.data.actions.new(full)
    act.use_fake_user = True
    sel = GROUPS[group]
    chans = rig.channels
    data = data.copy()
    if quat_fix:   # sign continuity of quaternions
        idx = {}
        for k, (n, attr, i) in enumerate(chans):
            if attr == "rotation_quaternion":
                idx.setdefault(n, []).append(k)
        for n, ks in idx.items():
            for r in range(1, len(data)):
                if np.dot(data[r, ks], data[r - 1, ks]) < 0:
                    data[r, ks] = -data[r, ks]
    frames = np.arange(len(data), dtype=float)
    nkeys = 0
    for k, (n, attr, i) in enumerate(chans):
        if not sel(n):
            continue
        col = data[:, k]
        const = (col.max() - col.min()) < eps
        if attr.startswith("["):
            path = 'pose.bones["%s"]%s' % (n, attr)
            idxn = 0
        else:
            path = 'pose.bones["%s"].%s' % (n, attr)
            idxn = i
        fc = act.fcurves.new(data_path=path, index=idxn, action_group=n)
        if const:
            fc.keyframe_points.add(1)
            fc.keyframe_points[0].co = (0.0, float(col[0]))
            fc.keyframe_points[0].interpolation = "LINEAR"
            nkeys += 1
        else:
            m = len(col)
            fc.keyframe_points.add(m)
            co = np.empty(2 * m)
            co[0::2] = frames
            co[1::2] = col
            fc.keyframe_points.foreach_set("co", co)
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            nkeys += m
        if loop:
            fc.modifiers.new("CYCLES")
        fc.update()
    act.use_frame_range = True
    act.frame_start, act.frame_end = 0.0, float(n_frames)
    act.use_cyclic = bool(loop)
    act["loop"] = bool(loop)
    act["layer_group"] = group
    for k_, v in (meta or {}).items():
        act[k_] = v
    return act, nkeys
