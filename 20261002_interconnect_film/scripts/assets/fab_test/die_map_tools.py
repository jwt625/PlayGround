"""Assembler helpers for the wafer_300mm_siph die map (run inside Blender).

Usage inside a scene that has the asset appended / linked:
    import sys; sys.path.append("<repo>/scripts/assets/fab_test")
    import die_map_tools as D
    dies = D.get_dies()                      # die objects sorted by probe order
    D.set_state_all("idle")
    D.assign_pass_fail(dies, seed=7)         # exactly round(N/10) dies are marked pass
    D.apply_state(dies[k], "probing_glow", frame=F0); D.apply_state(dies[k], "pass", frame=F1)  # keyframed colour
"""
import random

import bpy

STATES = {"idle": (0.10, 0.14, 0.26, 0.0), "probing_glow": (1.0, 1.0, 1.0, 1.0),
          "pass": (0.10, 0.90, 0.25, 0.8), "fail": (0.80, 0.08, 0.08, 0.6)}


def get_dies(prefix="wafer_300mm_siph_die_r"):
    ds = [o for o in bpy.data.objects if o.name.startswith(prefix) and "die_probe_order" in o]
    return sorted(ds, key=lambda o: o["die_probe_order"])


def apply_state(obj, state, frame=None):
    obj.color = STATES[state]
    if frame is not None:
        obj.keyframe_insert("color", frame=frame)


def set_state_all(state, dies=None):
    for o in (dies or get_dies()):
        o.color = STATES[state]


def pass_set_exact_tenth(dies, seed=7):
    """Exactly round(N/10) dies, spread uniformly at random (seeded)."""
    n = len(dies)
    k = round(n / 10)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return set(idx[:k])


def assign_pass_fail(dies=None, seed=7, apply=True):
    dies = dies or get_dies()
    ps = pass_set_exact_tenth(dies, seed)
    res = {}
    for i, o in enumerate(dies):
        res[o.name] = "pass" if i in ps else "fail"
        if apply:
            o.color = STATES[res[o.name]]
    return res


def assign_pass_fail_by_order(dies=None, every=10, phase=5, apply=True):
    """Every 10th die in probe order passes (deterministic, as in the crude film)."""
    dies = dies or get_dies()
    res = {}
    for i, o in enumerate(dies):
        res[o.name] = "pass" if (i % every) == phase else "fail"
        if apply:
            o.color = STATES[res[o.name]]
    return res
