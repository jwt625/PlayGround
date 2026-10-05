"""S6 cable-whip physics: deterministic position-based (verlet) chain with bending stiffness, baked to bone rotations of the aoc_whip rig.

No random numbers. Fixed substeps. Functions are called from s06_fiber.py at build time:
  inputs per frame (evaluated from the scene): world matrix of ROOT_aoc_whip (follows the Manager's hand hook), Gary's bone capsules.
  outputs: per-frame chain points (world), per-bone quaternions keyed into an action on aoc_whip_rig, contact and crack events.
Model (documented in DevLog/v1/DevLog-004-s06-new-assets.md):
  - 24 segments of 0.125 m (the rig's bones), 25 points; points 0..4 are the rigid handle (pinned to the hand frame).
  - point masses taper from 1.0 at the handle to TIP_MASS at the tip (momentum transfer makes the travelling wave speed up: whip crack).
  - distance constraints (24 Gauss-Seidel passes x 8 substeps per frame), bending stiffness by second-neighbour distance constraints whose strength
    falls along the chain as exp(-(i-3)/7) (the rig's stiffness profile), gravity, light air damping plus quadratic drag, floor with friction,
    capsule colliders from Gary's rig bones (inelastic, friction), hit-stop (frozen frames) after the first torso contact.
"""
import math

import numpy as np
from mathutils import Matrix, Quaternion, Vector

N_SEG = 24
L_SEG = 0.125
N_PT = N_SEG + 1
PINNED = 5                    # points 0..4 follow the hand rigidly
SUB = 8
ITERS = 24
G = 9.81
TIP_MASS = 0.07
DAMP = 0.9996                 # per substep velocity retention
DRAG = 0.55                   # quadratic drag coefficient (1/m) scaled by 1/mass
FLOOR_Z = 0.03
R_CABLE = 0.03                # collision radius of the (thickened) bundle
BEND_K = 0.9                  # bending constraint strength at the handle end
GARY_R = {"hips": 0.22, "spine_1": 0.22, "spine_2": 0.22, "chest": 0.22, "neck": 0.08, "head": 0.19, "thigh_L": 0.11, "thigh_R": 0.11,
          "shin_L": 0.075, "shin_R": 0.075, "upper_arm_L": 0.06, "upper_arm_R": 0.06}
TORSO = ("hips", "spine_1", "spine_2", "chest")
BACKSIDE = ("hips", "thigh_L", "thigh_R")   # v1.2: contacts recorded on all capsules (capsule index in the record); the backside set is used for the hit


def masses():
    t = np.arange(N_PT) / (N_PT - 1.0)
    m = 1.0 - (1.0 - TIP_MASS) * t ** 1.15
    m[:PINNED] = 1e9
    return m


def rest_points():
    """Chain points in whip-root / armature coordinates at rest: along -Y."""
    p = np.zeros((N_PT, 3))
    p[:, 1] = -L_SEG * np.arange(N_PT)
    return p


def _closest_on_seg(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return a + t * ab


def simulate(handle_pts, caps, hold=None):
    """handle_pts: (F, PINNED, 3) world positions of the pinned handle points per frame; caps: list (F) of capsule lists [(a, b, r, is_torso)];
    hold: set of frame indices (0-based) in which the chain is frozen (hit-stop). Returns P (F, N_PT, 3), events dict."""
    F = len(handle_pts)
    hold = hold or set()
    m = masses()
    w = 1.0 / m
    P = np.zeros((F, N_PT, 3))
    # initial chain: hang from the handle along gravity direction (down), then relax on the floor with the handle fixed
    x = np.zeros((N_PT, 3))
    x[:PINNED] = handle_pts[0]
    dirv = handle_pts[0][-1] - handle_pts[0][-2]
    dirv = dirv / max(np.linalg.norm(dirv), 1e-9)
    for i in range(PINNED, N_PT):
        x[i] = x[i - 1] + dirv * L_SEG
    xp = x.copy()
    kb = BEND_K * np.exp(-(np.arange(N_PT) - 3.0) / 7.0)
    kb = np.clip(kb, 0.0, 1.0)
    contacts = []

    def step(x, xp, hp0, hp1, cap, record=None, f_idx=None):
        for s in range(SUB):
            a = (s + 1.0) / SUB
            tgt = hp0 + (hp1 - hp0) * a
            CA = np.array([c[0] for c in cap]).reshape(-1, 3)
            CB = np.array([c[1] for c in cap]).reshape(-1, 3)
            CR = np.array([c[2] for c in cap])
            CT = [c[3] for c in cap]
            dt = 1.0 / (30.0 * SUB)
            v = (x - xp)
            sp = np.linalg.norm(v, axis=1, keepdims=True) / dt
            v = v * DAMP - v * np.minimum(DRAG * sp * dt * w[:, None] * 0.3, 0.2)
            xn = x + v
            xn[:, 2] -= G * dt * dt
            xn[:PINNED] = tgt
            for it in range(ITERS):
                for i in range(N_PT - 1):
                    d = xn[i + 1] - xn[i]
                    dist = np.linalg.norm(d)
                    if dist < 1e-9:
                        continue
                    c = (dist - L_SEG) / dist * d
                    ws = w[i] + w[i + 1]
                    xn[i] += w[i] / ws * c
                    xn[i + 1] -= w[i + 1] / ws * c
                for i in range(2, N_PT):
                    d = xn[i] - xn[i - 2]
                    dist = np.linalg.norm(d)
                    if dist < 1e-9:
                        continue
                    c = (dist - 2.0 * L_SEG) / dist * d * kb[i] * 0.5
                    ws = w[i] + w[i - 2]
                    xn[i - 2] += w[i - 2] / ws * c
                    xn[i] -= w[i] / ws * c
                xn[:PINNED] = tgt
                np.maximum(xn[PINNED:, 2], FLOOR_Z, out=xn[PINNED:, 2])
                if len(cap):
                    X = xn[PINNED:]
                    AB = CB - CA
                    tt = np.clip(np.einsum("nck,ck->nc", X[:, None, :] - CA[None], AB) / np.maximum(np.einsum("ck,ck->c", AB, AB), 1e-12), 0.0, 1.0)
                    Q = CA[None] + tt[..., None] * AB[None]
                    D = X[:, None, :] - Q
                    dl = np.linalg.norm(D, axis=2)
                    pen = dl < (CR + R_CABLE)[None]
                    if pen.any():
                        for ni, ci in zip(*np.nonzero(pen)):
                            i = PINNED + ni
                            Dv = xn[i] - Q[ni, ci]
                            dd = np.linalg.norm(Dv)
                            nrm = Dv / dd if dd > 1e-9 else np.array([0.0, 0.0, 1.0])
                            xn[i] = Q[ni, ci] + nrm * (CR[ci] + R_CABLE)
                            if record is not None:
                                record.append((f_idx, i, float(np.linalg.norm(xn[i] - x[i]) / dt), int(ci)))
            # floor and collider friction: damp tangential velocity of touching points
            for i in range(PINNED, N_PT):
                if xn[i, 2] <= FLOOR_Z + 1e-6:
                    xn[i, :2] = x[i, :2] + (xn[i, :2] - x[i, :2]) * 0.82
            xp[:] = x
            x[:] = xn
        return x, xp

    # prewarm on the first frame's pose (60 frames of simulated time, handle fixed)
    for _ in range(60):
        step(x, xp, handle_pts[0], handle_pts[0], caps[0])
    P[0] = x
    rec = []
    for f in range(1, F):
        if f in hold:
            P[f] = P[f - 1]
            xp[:] = x
            continue
        x, xp = step(x, xp, handle_pts[f - 1], handle_pts[f], caps[f], rec, f)
        P[f] = x
    return P, rec


def tip_speed(P):
    v = np.linalg.norm(np.diff(P[:, -1], axis=0), axis=1) * 30.0
    return np.concatenate([[0.0], v])


def bone_quats(P, W_inv3):
    """Per-frame pose quaternions for whip_01..whip_24 from world chain points. W_inv3: inverse of the root's world 4x4 (to armature space)."""
    F = len(P)
    rest3 = Matrix.Diagonal((1.0, -1.0, -1.0))          # bone matrix_local rotation part (all bones identical): bone y = -Y armature
    out = np.zeros((F, N_SEG, 4))
    for f in range(F):
        pts = [W_inv3[f] @ Vector(p) for p in P[f]]
        Qprev = Quaternion((1, 0, 0, 0))       # rotation of the previous bone frame relative to rest, in armature space
        dprev = Vector((0, -1, 0))
        Pm_par = rest3.copy()
        for i in range(N_SEG):
            d = (pts[i + 1] - pts[i])
            if i < PINNED - 1 or d.length < 1e-9:
                Qi = Quaternion((1, 0, 0, 0))
                dcur = Vector((0, -1, 0))
            else:
                dcur = d.normalized()
                Qi = dprev.rotation_difference(dcur) @ Qprev
            Pm = Qi.to_matrix() @ rest3
            Rb = (Pm_par.inverted() @ Pm)
            out[f, i] = Rb.to_quaternion()[:]
            Qprev, dprev, Pm_par = Qi, dcur, Pm
    return out


def key_quats(arm, quats, f0, name="ACT_s06_whip_baked"):
    """Create an action with quaternion F-curves for whip_01..24 and assign it to the armature (pose bones set to QUATERNION mode)."""
    import bpy
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    act = bpy.data.actions.new(name)
    F = quats.shape[0]
    frames = np.arange(F) + f0
    for i in range(N_SEG):
        bn = "whip_%02d" % (i + 1)
        for k in range(4):
            fc = act.fcurves.new('pose.bones["%s"].rotation_quaternion' % bn, index=k, action_group=bn)
            fc.keyframe_points.add(F)
            co = np.empty(2 * F)
            co[0::2] = frames
            co[1::2] = quats[:, i, k]
            fc.keyframe_points.foreach_set("co", co)
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
    ad = arm.animation_data_create()
    ad.action = act
    return act
