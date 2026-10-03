"""S3 fight engine (v1.1): procedural brawl actions authored on the existing NPC armatures, baked per frame.

Poses are written in a canonical 'lead-left' frame (the partner is on the character's left, lead hand = L) and mirrored for
fighters whose partner is on the right.  Coordinates are baseline character space (metres at 1.75 m stature, x = left,
y = back, z = up) exactly as scripts/assets/characters/chars_actions.py, which supplies the IK pose engine (Rig).
Nothing in assets/ is modified: new actions are named ACT_<asset>_s03_<name> and live in the scene blend.

Features: anticipation / overshoot easing, hit-stop holds, squash and stretch (chest, head, hips, root scale), belly and
shirt jiggle (damped oscillation on spine/chest scale after impacts), head lag spring, procedural foot planting and
stepping, eye look-at on the partner.  Deterministic (no random numbers except fixed-seed hashes).
"""
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJ, "scripts", "assets", "characters"))
import chars_actions as CA  # noqa: E402

FPS = 30
AZ = 0.0707  # ankle height above the ground (baseline metres), chars_actions.foot_ankle


def v(*a):
    return np.array(a, float)


def nrm(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else x


def ortho(x, f):
    x = np.asarray(x, float)
    return nrm(x - f * np.dot(x, f))


# ------------------------------------------------------------------------------------------------ easing
def ease(name, u):
    u = min(max(u, 0.0), 1.0)
    if name == "lin":
        return u
    if name == "smooth":
        return u * u * (3 - 2 * u)
    if name == "in":      # accelerate into the key (strike)
        return u ** 2.4
    if name == "out":     # decelerate into the key (settle)
        return 1 - (1 - u) ** 2.4
    if name == "snap":    # fast arrival with overshoot (easeOutBack)
        c1 = 1.9
        return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2
    if name == "wind":    # slight reverse before moving (easeInBack): anticipation built into the segment
        c1 = 1.5
        return (c1 + 1) * u ** 3 - c1 * u ** 2
    if name == "hold":
        return 0.0 if u < 1.0 else 1.0
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ rig access
def rig_for(asset):
    arm = asset.armature
    K = float(asset.root["p_stature_m"]) / 1.75
    b = arm.data.bones
    J = {"wr": np.array(b["hand_L"].head_local) / K, "ankle": np.array(b["foot_L"].head_local) / K}
    dh = nrm((0.04, -0.22, -0.97))
    hf = {}
    for sn, side in (("L", 1), ("R", -1)):
        f = nrm(dh * np.array([side, 1, 1]))
        n = ortho((-side, 0, 0), f)
        c = ortho((0, 1, 0), f)
        hf[sn] = (f, n, c)
    return CA.Rig(arm, K, J, hf), K, hf


def neutral(hf):
    """Canonical neutral (standing, arms hanging); all keys present so poses can be lerped."""
    def hand(sn, side):
        f, n, _ = hf[sn]
        return np.concatenate([v(side * 0.296, -0.08, 0.84), f, n])
    return dict(
        hips_loc=v(0, 0, 0), hips_rot=v(0, 0, 0), spine=v(0, 0, 0), neck=v(0, 0, 0), head=v(0, 0, 0), jaw=0.0,
        hand_L=hand("L", 1), hand_R=hand("R", -1), curl_L=np.full(5, 0.2), curl_R=np.full(5, 0.2),
        elbow_L=v(0.36, 0.40, 1.10), elbow_R=v(-0.36, 0.40, 1.10),
        foot_L=v(0.12, 0.0, 0.0, 0.0), foot_R=v(0.12, 0.0, 0.0, 0.0),  # (x magnitude, y, lift (world m), pitch)
        knee_L=v(0.10, -0.8, 0.5), knee_R=v(-0.10, -0.8, 0.5),
        sqc=v(1, 1, 1), sqh=v(1, 1, 1), sqb=v(1, 1, 1), sqr=v(1, 1, 1), hop=0.0, yo=0.0, blink=0.0,
    )


def hv(pos, f=(0, -1, 0.1), n=(-1, 0, 0)):
    """Hand 9-vector: position, finger direction, palm normal (canonical frame; lead hand = L, palm medial)."""
    return np.concatenate([v(*pos), nrm(f), nrm(n)])


def mirror(P):
    """Swap L/R and flip x: canonical lead-left -> lead-right."""
    Q = {}
    for k, x in P.items():
        Q[k] = x
    sw = {"hand_L": "hand_R", "hand_R": "hand_L", "curl_L": "curl_R", "curl_R": "curl_L", "elbow_L": "elbow_R", "elbow_R": "elbow_L",
          "foot_L": "foot_R", "foot_R": "foot_L", "knee_L": "knee_R", "knee_R": "knee_L"}
    for a, b in sw.items():
        Q[a] = P[b]
    for k in ("hand_L", "hand_R"):
        h = Q[k].copy()
        h[[0, 3, 6]] *= -1
        Q[k] = h
    for k in ("elbow_L", "elbow_R", "knee_L", "knee_R", "hips_loc"):
        h = Q[k].copy()
        h[0] *= -1
        Q[k] = h
    for k in ("hips_rot", "spine", "neck", "head"):
        h = Q[k].copy()
        h[1] *= -1
        h[2] *= -1
        Q[k] = h
    Q["yo"] = -P["yo"]
    return Q


def lerp_pose(A, B, e):
    return {k: A[k] + (B[k] - A[k]) * e for k in A}


class Timeline:
    """Pose keys (t, canonical pose, ease into the key). Missing time = hold the previous key until the next."""

    def __init__(self, base):
        self.keys = []
        self.base = base

    def add(self, t, pose=None, ease_="smooth"):
        P = dict(self.base)
        if pose:
            for k, x in pose.items():
                P[k] = np.asarray(x, float) if isinstance(x, (list, tuple, np.ndarray)) else x
        if self.keys and t <= self.keys[-1][0] + 1e-6:
            print("WARN timeline key %.3f not after %.3f (%s): nudged" % (t, self.keys[-1][0], getattr(self, "name", "?")))
            t = self.keys[-1][0] + 0.034
        self.keys.append((t, P, ease_))
        return P

    def last(self):
        return self.keys[-1][1]

    def hold(self, t_end, jitter=None):
        """Hold the last pose until t_end (hit-stop)."""
        self.add(t_end, dict(self.last()), "hold" if jitter is None else "hold")

    def eval(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for (t0, A, _), (t1, B, es) in zip(ks[:-1], ks[1:]):
            if t0 <= t <= t1:
                e = ease(es, (t - t0) / (t1 - t0))
                return lerp_pose(A, B, e)
        return ks[-1][1]


class PathTL:
    """World xy path of the root: keys (t, (x, y), ease)."""

    def __init__(self, p0):
        self.keys = [(-1.0, np.array(p0, float), "lin")]

    def add(self, t, p, ease_="smooth"):
        if t <= self.keys[-1][0]:
            print("WARN path key %.3f not after %.3f: nudged" % (t, self.keys[-1][0]))
            t = self.keys[-1][0] + 0.034
        self.keys.append((t, np.array(p, float), ease_))

    def last(self):
        return self.keys[-1][1]

    def eval(self, t):
        ks = self.keys
        if t >= ks[-1][0]:
            return ks[-1][1]
        for (t0, A, _), (t1, B, es) in zip(ks[:-1], ks[1:]):
            if t0 <= t <= t1:
                return A + (B - A) * ease(es, (t - t0) / (t1 - t0))
        return ks[0][1]


class Fighter:
    def __init__(self, fid, asset, d, base_xy, yaw_fight):
        self.id = fid
        self.asset = asset
        self.d = d
        self.rig, self.K, self.hf = rig_for(asset)
        self.neutral = neutral(self.hf)
        self.tl = Timeline(self.neutral)
        self.tl.name = fid
        self.path = PathTL(base_xy)
        self.base_xy = np.array(base_xy, float)
        self.yaw_fight = yaw_fight
        self.yaw_keys = []      # (t, yaw) base yaw (without pose yaw offset)
        self.jiggles = []       # (t, amplitude)
        self.partner = None
        self.expr = []          # (t, prop, value, interp) face pulses
        # foot stepper state
        self.plant = {}
        self.step = {}
        self.out = {}

    # world geometry helpers -------------------------------------------------------------------------
    def yaw_base(self, t):
        ks = self.yaw_keys
        if t <= ks[0][0]:
            return ks[0][1]
        for (t0, a), (t1, b) in zip(ks[:-1], ks[1:]):
            if t0 <= t <= t1:
                return a + (b - a) * (t - t0) / (t1 - t0)
        return ks[-1][1]

    def pos(self, t):
        return self.path.eval(t)

    def head_world(self, t, h=1.60):
        p = self.pos(t)
        return np.array([p[0], p[1], h * self.K])

    def to_local(self, t, world_xyz, yaw=None):
        """World point -> own baseline coordinates (x left, y back, z up); uses root position and base yaw."""
        p = self.pos(t)
        th = self.yaw_base(t) if yaw is None else yaw
        dx, dy = world_xyz[0] - p[0], world_xyz[1] - p[1]
        xl = dx * math.cos(th) + dy * math.sin(th)
        yb = -dx * math.sin(th) + dy * math.cos(th)
        return np.array([xl, yb, world_xyz[2]]) / self.K

    def to_canon(self, t, world_xyz):
        l = self.to_local(t, world_xyz)
        l[0] *= self.d
        return l


def spring_filter(x, dt, freq=5.0, zeta=0.32):
    """Underdamped follow of a sampled signal (head/neck lag and overshoot); x: (N, C)."""
    w = 2 * math.pi * freq
    y = x[0].copy()
    vel = np.zeros_like(y)
    out = np.empty_like(x)
    sub = 4
    h = dt / sub
    for i in range(len(x)):
        for _ in range(sub):
            acc = w * w * (x[i] - y) - 2 * zeta * w * vel
            vel += acc * h
            y = y + vel * h
        out[i] = y
    return out


# ------------------------------------------------------------------------------------------------ baking
def bake_fighter(fg, t0, t1, aid_suffix="s03_brawl", log=None):
    """Solve the pose at every frame in [t0, t1], then write the action. Returns (action, per-frame root keys)."""
    rig, K, d = fg.rig, fg.K, fg.d
    arm = rig.arm
    pb = rig.pb
    dt = 1.0 / FPS
    n = int(round((t1 - t0) * FPS)) + 1
    states, roots = [], []
    fg.plant, fg.step = {}, {}
    prev_pos = None
    heads, necks, jaws = [], [], []
    for i in range(n):
        t = t0 + i * dt
        Pc = fg.tl.eval(t)
        P = mirror(Pc) if d < 0 else Pc
        pos = fg.pos(t)
        yaw = fg.yaw_base(t) + P["yo"]
        z = P["hop"]
        vel = (pos - prev_pos) / dt if prev_pos is not None else np.zeros(2)
        prev_pos = pos
        # ---- foot planting / stepping (world space, then converted to local baseline)
        left = np.array([math.cos(yaw), math.sin(yaw)])
        back = np.array([-math.sin(yaw), math.cos(yaw)])
        feet = {}
        for sn, side in (("L", 1), ("R", -1)):
            fx, fy, fl, fp = P["foot_" + sn]
            ideal = pos + K * (side * fx * left + fy * back)
            st = fg.step.get(sn)
            pl = fg.plant.get(sn)
            if pl is None:
                fg.plant[sn] = pl = ideal.copy()
            lift = 0.0
            other = "R" if sn == "L" else "L"
            if st is None:
                err = np.linalg.norm(pl - ideal)
                if err > 0.20 * K and fg.step.get(other) is None or err > 0.38 * K:
                    tgt = ideal + vel * 0.10
                    dur = 0.12 + 0.25 * min(np.linalg.norm(tgt - pl), 0.6)
                    fg.step[sn] = st = (t, pl.copy(), tgt, dur)
            if st is not None:
                ts, a, b, dur = st
                u = (t - ts) / dur
                if u >= 1.0:
                    fg.plant[sn] = pl = b.copy()
                    fg.step[sn] = None
                    cur = pl
                else:
                    cur = a + (b - a) * ease("smooth", u)
                    lift = 0.10 * K * math.sin(math.pi * u)
            else:
                cur = pl
            dx, dy = cur[0] - pos[0], cur[1] - pos[1]
            xl = dx * math.cos(yaw) + dy * math.sin(yaw)
            yb = -dx * math.sin(yaw) + dy * math.cos(yaw)
            zl = AZ + (fl + lift - z) / K
            feet[sn] = dict(pos=v(xl / K, yb / K, zl), pitch=fp - 0.5 * lift / K)
        # ---- engine pose dict
        E = {}
        E["hips_loc"] = tuple(P["hips_loc"])
        E["hips_rot"] = tuple(P["hips_rot"])
        E["spine"] = tuple(P["spine"])
        E["neck"] = tuple(P["neck"])
        E["head"] = tuple(P["head"])
        E["jaw"] = float(P["jaw"])
        E["blink"] = float(P["blink"])
        for sn in ("L", "R"):
            h = P["hand_" + sn]
            E["hand_" + sn] = dict(pos=h[0:3], f=h[3:6], n=h[6:9])
            E["curl_" + sn] = dict(zip(CA.CURLS, [float(x) for x in P["curl_" + sn]]))
            E["elbow_" + sn] = P["elbow_" + sn]
            E["knee_" + sn] = P["knee_" + sn]
            E["foot_" + sn] = feet[sn]
        # eyes on the partner
        if fg.partner is not None:
            pw = fg.partner.head_world(t)
            lk = fg.to_local(t, pw, yaw=yaw)
            lk = lk * 1.0
            E["look"] = v(lk[0], lk[1], lk[2])
        # ---- scale channels (squash / stretch + jiggle)
        sqc, sqh, sqb = P["sqc"].copy(), P["sqh"].copy(), P["sqb"].copy()
        jig = np.zeros(3)
        for tj, amp in fg.jiggles:
            if t >= tj:
                s = (t - tj)
                g = amp * math.exp(-s / 0.20) * math.sin(2 * math.pi * 6.5 * s)
                jig += v(-0.5 * g, g, -0.5 * g)
        for nm, sq in (("chest", sqc), ("head", sqh), ("hips", sqb)):
            pb[nm].scale = tuple(sq if nm != "chest" else sq + jig * 0.6)
        pb["spine_1"].scale = tuple(np.ones(3) + jig)
        pb["spine_2"].scale = tuple(np.ones(3) + jig * 0.5)
        rig.pose(E)
        st = rig.snapshot()
        for nm in ("chest", "head", "hips", "spine_1", "spine_2"):
            st[(nm, "scale")] = tuple(pb[nm].scale)
        states.append(st)
        heads.append(st[("head", "rotation_euler")])
        necks.append(st[("neck", "rotation_euler")])
        jaws.append(st[("jaw", "rotation_euler")])
        roots.append((t, pos.copy(), yaw, z, tuple(P["sqr"])))
    # ---- secondary motion: head / neck lag with overshoot
    for key, arr in (("head", np.array(heads)), ("neck", np.array(necks))):
        arr2 = spring_filter(arr, dt, freq=6.0 if key == "head" else 7.5, zeta=0.30 if key == "head" else 0.4)
        for i, st in enumerate(states):
            st[(key, "rotation_euler")] = tuple(arr2[i])
    # ---- write the action
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = None
    act = bpy.data.actions.new("ACT_%s_%s" % (fg.asset.root["asset_id"], aid_suffix))
    act.use_fake_user = True
    arm.animation_data.action = act
    for i, st in enumerate(states):
        rig.write_key(st, i)
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    arm.animation_data.action = None
    rig.clear()
    for nm in ("chest", "head", "hips", "spine_1", "spine_2"):
        pb[nm].scale = (1, 1, 1)
    if log:
        log("baked %s: %d frames" % (act.name, n))
    return act, roots


def key_root(fg, roots, replace_before=None):
    """Write dense root location / yaw / scale keys from bake_fighter output."""
    r = fg.asset.root
    for (t, pos, yaw, z, sq) in roots:
        f = 1 + int(math.floor(t * FPS + 0.5))
        r.location = (pos[0], pos[1], z)
        r.rotation_euler = (0, 0, yaw)
        r.scale = tuple(sq)
        r.keyframe_insert("location", frame=f)
        r.keyframe_insert("rotation_euler", frame=f)
        r.keyframe_insert("scale", frame=f)
    ad = r.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path in ("location", "rotation_euler", "scale"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
