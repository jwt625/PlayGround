"""Action registry and shared helpers for motion v2."""
import math

import numpy as np

import m2_engine as E
from m2_engine import V, hand, neutral, Timeline, ease  # noqa: F401

REG = {}   # name -> dict(builder, acting, group, ...)
ORDER = []


def action(name, group="full", acting=None, use="", note="", mirror_name=None, also_mirror=None):
    """Register a builder b(rig) -> spec dict(fn|timeline, n, loop, overlap, events, attach, meta, face).
    acting='R' means the builder is authored with the LEFT hand/side as the acting one (left coordinates) and the exported action is
    its mirror (right hand acts); '<name>_L' is registered as the unmirrored variant."""
    def deco(b):
        REG[name] = dict(builder=b, group=group, mirror=(acting == "R"), use=use, note=note, variant_of=None)
        ORDER.append(name)
        if acting == "R":
            nm = mirror_name or name + "_L"
            REG[nm] = dict(builder=b, group=group, mirror=False, use=use, note=note + " (left-hand variant)", variant_of=name)
            ORDER.append(nm)
        if also_mirror:
            REG[also_mirror] = dict(builder=b, group=group, mirror=True, use=use, note=note + " (mirrored)", variant_of=name)
            ORDER.append(also_mirror)
        return b
    return deco


def ss(a):
    """Volume-preserving squash/stretch scale triple for a bone with local Y along its length. a>0 stretch, a<0 squash."""
    s = 1.0 + a
    return V(1.0 / math.sqrt(s), s, 1.0 / math.sqrt(s))


def blink_curve(i, n, times, dur=4):
    """0..1 blink value at frame i for blink onset frames `times` (closing 1 frame, hold, 2-frame open)."""
    v = 0.0
    for t0 in times:
        d = (i - t0) % n if n else i - t0
        if 0 <= d < dur:
            v = max(v, [0.6, 1.0, 1.0, 0.45][d] if dur == 4 else 1.0)
    return v


def stand_feet(x=0.10, y=0.0, yaw=0.06, az=E.AZ):
    return dict(foot_L=V(x, y, az, 0, yaw), foot_R=V(x, y, az, 0, -yaw))


def smooth_loop(arr, k=2):
    """Circular box smoothing of an (N, C) array (cycle)."""
    out = np.zeros_like(arr)
    for s in range(-k, k + 1):
        out += np.roll(arr, s, axis=0)
    return out / (2 * k + 1)


def finalize_spec(spec):
    spec.setdefault("loop", False)
    spec.setdefault("overlap", {})
    spec.setdefault("events", [])
    spec.setdefault("meta", {})
    spec.setdefault("face", None)
    spec.setdefault("attach", None)
    spec.setdefault("group", None)
    return spec


def add_upper_variants(names, suffix="_upper"):
    """Register upper-body-only (layerable) copies of existing actions: same poses, only upper-body channels are keyed."""
    for n in names:
        if n in REG and n + suffix not in REG:
            e = dict(REG[n])
            e["group"] = "upper"
            e["note"] = e["note"] + " [upper-body layer: legs, hips and root are not keyed]"
            e["variant_of"] = n
            REG[n + suffix] = e
            ORDER.append(n + suffix)
