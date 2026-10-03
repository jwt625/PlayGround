"""S3 camera (v1.1): eased shots, handheld noise, impact shake. Baked to dense per-frame keys on blender_lib's CAM / CAM_TARGET.

Replaces asm.shot (linear moves, hard cuts): each shot is eased in/out; a deterministic sum-of-sines handheld drift and
decaying shake pulses on impacts (angular, so close shots shake as much on screen as wide ones) are added.
Captions are parented to the camera, so they do not shake.
"""
import math

import numpy as np

import blender_lib as L
from asm import F

FPS = 30


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _drift(t, seed, amp):
    out = np.zeros(3)
    for ax in range(3):
        for k, (fr, w) in enumerate(((0.37, 1.0), (0.83, 0.6), (1.9, 0.3))):
            ph = 1.7 * (seed * 3 + ax) + 2.3 * k
            out[ax] += w * math.sin(2 * math.pi * fr * t + ph)
    return out * amp / 1.9


def bake(shots, events, total_t=10.0, pos_amp=0.010, ang_amp=0.0016, shake_ang=0.014):
    """shots: list of dicts(t0, t1, p0, a0, p1, a1, lens0, lens1, ease). events: [(t, strength)] impact shake."""
    cam, tgt, hold = L.CAM, L.TGT, L.HOLD
    n = int(round(total_t * FPS))
    for fi in range(n):
        t = fi / FPS + 1e-9
        f = 1 + fi
        sh = None
        for s in shots:
            if s["t0"] - 1e-6 <= t < s["t1"] - 1e-6:
                sh = s
        if sh is None:
            sh = shots[-1]
        u = smooth((t - sh["t0"]) / max(sh["t1"] - sh["t0"], 1e-6)) if sh.get("ease", True) else (t - sh["t0"]) / (sh["t1"] - sh["t0"])
        p = np.array(sh["p0"]) + (np.array(sh["p1"]) - np.array(sh["p0"])) * u
        a = np.array(sh["a0"]) + (np.array(sh["a1"]) - np.array(sh["a0"])) * u
        lens = sh["lens0"] + (sh["lens1"] - sh["lens0"]) * u
        # handheld drift (metres on the camera, angular on the target)
        p = p + _drift(t, 1, pos_amp)
        dist = max(np.linalg.norm(a - p), 0.5)
        fwd = (a - p) / dist
        right = np.cross(fwd, np.array([0, 0, 1.0]))
        right = right / max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, fwd)
        d3 = _drift(t, 2, ang_amp * dist)
        # impact shake: decaying 17 Hz pulses; the first 0.04 s ramp in
        ox = oy = 0.0
        for te, st in events:
            dt = t - te
            if 0.0 <= dt < 0.5:
                env = st * math.exp(-dt / 0.11) * min(1.0, dt / 0.03 + 0.2)
                ox += env * math.sin(2 * math.pi * 17 * dt + 0.6)
                oy += env * math.sin(2 * math.pi * 13 * dt + 2.1)
        a2 = a + right * (d3[0] + ox * shake_ang * dist) + up * (d3[1] + oy * shake_ang * dist)
        p2 = p + right * ox * shake_ang * 0.25 * dist + up * oy * shake_ang * 0.25 * dist
        cam.location = tuple(p2)
        cam.keyframe_insert("location", frame=f)
        tgt.location = tuple(a2)
        tgt.keyframe_insert("location", frame=f)
        cam.data.lens = lens
        cam.data.keyframe_insert("lens", frame=f)
        s_ = L.HOLD_S0 * L.LENS0 / lens
        hold.scale = (s_, s_, s_)
        hold.keyframe_insert("scale", frame=f)
    for ob in (cam, tgt, hold, cam.data):
        ad = ob.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
