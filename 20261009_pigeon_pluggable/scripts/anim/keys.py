"""Fast keyframe helpers shared by the anim modules (bpy only, headless-safe).

Frames are absolute scene frames (S uses blender_lib.F(idx, t) = 1 + idx*300 + round(t*30)).
Keys go straight into the object's action through the F-curve API (keyframe_points.insert with FAST),
so thousands of birds can be keyed quickly; call flush() (or flush_obj) once at the end to recalc handles.
"""
import bpy

FPS = 30
_DIRTY = set()


def frame(t, offset=1, fps=FPS):
    """Seconds -> frame (round half up), frame 1 = t 0 by default."""
    return offset + int(t * fps + 0.5)


def _action(obj):
    ad = obj.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(obj.name + "_act")
    return ad.action


def fcurve(obj, path, index):
    act = _action(obj)
    fc = act.fcurves.find(path, index=index)
    if fc is None:
        fc = act.fcurves.new(path, index=index, action_group=obj.name)
    return fc


def key(obj, path, f, values, interp="BEZIER", indices=None):
    """Key obj.<path>[i] = values[i] at frame f (float allowed). values: sequence or scalar (index 0)."""
    if not hasattr(values, "__len__"):
        values = (values,)
        indices = indices or (0,)
    idx = indices or range(len(values))
    for i, v in zip(idx, values):
        if v is None:
            continue
        fc = fcurve(obj, path, i)
        kp = fc.keyframe_points.insert(float(f), float(v), options={"FAST", "REPLACE"})
        kp.interpolation = interp
        _DIRTY.add(fc)


def value(obj, path, index, f, default=None):
    """Animated value of obj.<path>[index] at frame f (F-curve if keyed, else the static property)."""
    ad = obj.animation_data
    if ad is not None and ad.action is not None:
        fc = ad.action.fcurves.find(path, index=index)
        if fc is not None and len(fc.keyframe_points):
            return fc.evaluate(float(f))
    if default is not None:
        return default
    return getattr(obj, path)[index]


def values(obj, path, f, n=3):
    return tuple(value(obj, path, i, f) for i in range(n))


def last_key_frame(obj, path=None):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return None
    fr = [kp.co[0] for fc in ad.action.fcurves if path is None or fc.data_path == path for kp in fc.keyframe_points]
    return max(fr) if fr else None


def flush():
    """Recalculate handles of every F-curve touched since the last flush (required after FAST inserts)."""
    for fc in list(_DIRTY):
        try:
            fc.update()
        except ReferenceError:
            pass
    _DIRTY.clear()


def cyclic_noise(obj, path, index, amp, scale_frames, seed, offset=0.0):
    """Add a Noise F-modifier (deterministic phase from seed) to a channel; base value = offset. Not baked; used only on request."""
    fc = fcurve(obj, path, index)
    if not len(fc.keyframe_points):
        fc.keyframe_points.insert(1.0, offset, options={"FAST"})
    m = fc.modifiers.new("NOISE")
    m.strength = amp * 2.0
    m.scale = scale_frames
    m.phase = (seed * 7.31) % 1000.0
    m.blend_type = "ADD"
    _DIRTY.add(fc)
    return m
