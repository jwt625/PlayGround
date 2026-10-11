"""Assembly hatchery v3 integration helpers for the scene build (workstream M, DevLog-019). Blender 4.2; bpy only.

    import sys; sys.path.insert(0, "<repo>/scripts/assembly"); import assembly_api as AA
    root, objs = AA.append_assembly(AA.DEFAULT_BLEND, loc=(600, 0, 0), rot_z=0.0)   # whole ENV_assembly collection
    AA.key_assembly(objs, f0=1446, speed=1.0)          # sequence starts at film frame f0 (speed > 1 = faster, e.g. 1.36 -> 66 frames)
    for f, loc, tgt, lens, cut in AA.camera_keys(root, f0=1446, speed=1.0):  # suggested camera (world space)
        ...                                            # cut=True: hold the previous key until f-1, then jump (or cover with a fill)
    AA.event_frame("S5_eyes_full", f0=1446)             # film frame of an event (SFX / VO sync); names in the JSON sidecar

The sequence is one-shot (not a loop): all keys live in object / material / curve / light actions, frames F0..F0+DUR-1
(F0 = 1, DUR = 90 at build time; hop 2 of the finished bird runs to F0+100). Before the first key every station shows
its start state, after the last key its end state.
"""
import json
import os

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_BLEND = os.path.join(ROOT, "assets", "components", "assembly", "assembly_hatchery.blend")
SIDECAR = os.path.splitext(DEFAULT_BLEND)[0] + ".json"


def _side():
    with open(SIDECAR) as fh:
        return json.load(fh)


def append_assembly(path=DEFAULT_BLEND, loc=(0.0, 0.0, 0.0), rot_z=0.0, collection=None):
    """Append ENV_assembly (root empty `assembly_root` + ASM_* children) and place the root. Returns (root, objects)."""
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = ["ENV_assembly"]
    top = dst.collections[0]
    (collection or bpy.context.scene.collection).children.link(top)
    objs = list(top.all_objects)
    root = next(o for o in objs if o.name.startswith("assembly_root"))
    root.location = loc
    root.rotation_euler = (0.0, 0.0, rot_z)
    bpy.context.view_layer.update()
    return root, objs


def _ids(objs):
    """Objects + the data-blocks that carry assembly animation (materials' node trees, curves, lights)."""
    out, seen = [], set()
    for o in objs:
        out.append(o)
        d = o.data
        if d is not None and d.name not in seen and getattr(d, "animation_data", None) and d.animation_data.action:
            seen.add(d.name)
            out.append(d)
        mats = getattr(d, "materials", None) if d is not None else None
        for m in (mats or []):
            if m and m.node_tree and m.node_tree.animation_data and m.node_tree.animation_data.action and m.name not in seen:
                seen.add(m.name)
                out.append(m.node_tree)
    return out


def key_assembly(objs, f0, speed=1.0, f_build=1):
    """Re-time every assembly key: frame x -> f0 + (x - f_build) / speed. Call once after append."""
    n = 0
    for idb in _ids(objs):
        ad = idb.animation_data
        if not ad or not ad.action:
            continue
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                for attr in ("co", "handle_left", "handle_right"):
                    p = getattr(kp, attr)
                    p.x = f0 + (p.x - f_build) / speed
                n += 1
            fc.update()
    return n


def event_frame(name, f0, speed=1.0):
    return f0 + _side()["events"][name] / speed


def camera_keys(root, f0, speed=1.0, shots=("A_align_macro", "B_bond_macro", "C_press_to_test", "D_test_stand")):
    """Suggested camera keys in world space: [(frame, loc, target, lens, cut)] (root transform applied)."""
    side = _side()
    bpy.context.view_layer.update()
    M = root.matrix_world.copy()
    out = []
    for sh in side["camera_plan"]:
        if sh["name"] not in shots:
            continue
        for k, (t, loc, tgt, lens) in enumerate(sh["keys"]):
            out.append((f0 + t / speed, tuple(M @ Vector(loc)), tuple(M @ Vector(tgt)), lens, k == 0 and sh.get("cut_in", False)))
    return out


def apply_camera(cam, keys):
    """Key a camera object from camera_keys(): location, rotation (track to target) and lens; cuts are CONSTANT holds."""
    prev = None
    for f, loc, tgt, lens, cut in keys:
        if cut and prev is not None:
            _cam_key(cam, f - 1, prev[1], prev[2], prev[3], "CONSTANT")
        _cam_key(cam, f, loc, tgt, lens, "BEZIER")
        prev = (f, loc, tgt, lens)


def _cam_key(cam, f, loc, tgt, lens, interp):
    cam.location = loc
    cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam.data.keyframe_insert("lens", frame=f)
    for idb in (cam, cam.data):
        for fc in idb.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                if abs(kp.co.x - f) < 0.01:
                    kp.interpolation = interp
