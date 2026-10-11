"""Hatchery integration helpers for the scene build (workstream E, DevLog-013 section 7). Blender 4.2; bpy only.

The hatchery loop lives in object actions (+ the belt material's node-tree action) with F-curve Cycles modifiers, so it
survives a plain append (verified in scenes/film_v1.blend, 2026-10-10: robot joints and pallets move at the S5 frames).
    import sys; sys.path.insert(0, "<repo>/scripts/env"); import hatchery_api as HA
    root, objs = HA.append_hatchery(path, loc=(600, 0, 0))            # append (copies actions; loop keeps cycling)
    HA.shift_loop(objs, HA.phase_offset("hop1", film_frame))           # optional: event lands on a given film frame
    HA.bake_hatchery_loop(objs, f0, f1)                                # optional: explicit keys (no modifiers) for f0..f1
"""
import bpy

LOOP = 60
F0 = 1
EVENTS = {  # frames within the loop (DevLog-013 section 7)
    "robot_down": 6, "grip": 8, "place": 25, "release": 26, "eyes_on": 16, "eyes_full": 22, "qa_pass": 22,
    "hop1": 4, "ledge": 13, "hop2": 19, "floor": 28, "hop3": 33, "hop4": 45, "hopper_gone": 54,
}


def _ids(objs):
    """Objects plus the material / node-tree datablocks that carry hatchery animation."""
    out = list(objs)
    seen = set()
    for o in objs:
        if o.type != "MESH":
            continue
        for m in o.data.materials:
            if m and m.node_tree and m.node_tree.animation_data and m.node_tree.animation_data.action and m.name not in seen:
                seen.add(m.name)
                out.append(m.node_tree)
    return out


def append_hatchery(path, loc=(0.0, 0.0, 0.0), rot_z=0.0, collection=None):
    """Append the whole set (collection ENV_hatchery: root empty + HATCH_* children) and place hatchery_root.
    Returns (root, objects). Everything is already parented to the root; nothing to reparent."""
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = ["ENV_hatchery"]
    top = dst.collections[0]
    (collection or bpy.context.scene.collection).children.link(top)
    objs = list(top.all_objects)
    root = next(o for o in objs if o.name.startswith("hatchery_root"))
    root.location = loc
    root.rotation_euler = (0.0, 0.0, rot_z)
    return root, objs


def phase_offset(event, film_frame, loop=LOOP, f0=F0):
    """Frame offset for shift_loop so that `event` (EVENTS key) happens at film_frame."""
    return (film_frame - (f0 + EVENTS[event])) % loop


def shift_loop(objs, offset):
    """Shift every hatchery key by `offset` frames (cyclic, so any integer works)."""
    for idb in _ids(objs):
        ad = idb.animation_data
        if not ad or not ad.action:
            continue
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.co.x += offset
                kp.handle_left.x += offset
                kp.handle_right.x += offset
            fc.update()


def bake_hatchery_loop(objs, f0, f1, step=1, keep_paths=None):
    """Replace cyclic F-curves by explicit per-frame keys over [f0, f1] (sampled with the Cycles modifiers, then the
    modifiers removed). Use when a pipeline drops F-curve modifiers or when retiming with NLA. Visibility jumps stay exact
    at integer frames. keep_paths: data paths to leave untouched (e.g. the scene's hide_render keys)."""
    keep = set(keep_paths or ("hide_render", "hide_viewport"))
    n = 0
    for idb in _ids(objs):
        ad = idb.animation_data
        if not ad or not ad.action:
            continue
        for fc in ad.action.fcurves:
            if fc.data_path in keep or not any(m.type == "CYCLES" for m in fc.modifiers):
                continue
            frames = list(range(f0, f1 + 1, step))
            vals = [fc.evaluate(fr) for fr in frames]
            for m in list(fc.modifiers):
                fc.modifiers.remove(m)
            fc.keyframe_points.clear()
            fc.keyframe_points.add(len(frames))
            for kp, fr, v in zip(fc.keyframe_points, frames, vals):
                kp.co = (fr, v)
                kp.interpolation = "LINEAR"
            fc.update()
            n += 1
    return n
