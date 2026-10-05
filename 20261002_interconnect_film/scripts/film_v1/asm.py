"""Assembly framework for the v1.0 film: one .blend per scene, built from the asset library.

Each scene module (scripts/film_v1/sNN_*.py) calls asm.new_scene(), appends assets with asm.append(), places and
animates them (asm.place / asm.key_prop / asm.play / asm.walk), writes camera shots and captions (scene-local time in
seconds; frame = 1 + round(t * 30); the scene is 300 frames, S7 is 60), then asm.finalize(out_blend).

Reuses scripts/blender_lib.py (camera shots, overlay text, visibility windows, keyframe helpers) with scene index 0.
Asset conventions: assets/ASSET_SPEC.md; asset index: assets/INDEX.md.
"""
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
PROJ = os.path.dirname(SCRIPTS)
COMP = os.path.join(PROJ, "assets", "components")
for p in (SCRIPTS, os.path.join(SCRIPTS, "assets", "materials_vfx")):
    if p not in sys.path:
        sys.path.insert(0, p)
import blender_lib as L  # noqa: E402
import render_presets as RP  # noqa: E402

PI = math.pi
FPS = 30
HUD = os.environ.get("FILM_HUD", "1") != "0"  # draft HUD (FX notes, timecode, source footnotes) rendered in Blender
# With FILM_HUD=0 the render is clean: FX notes, timecodes and source footnote cards are not rendered but recorded in
# OVERLAYS and written next to the blend as <blend>.overlays.json, for burning in during post (post_overlays.py).
OVERLAYS = []


def F(t):
    return 1 + int(math.floor(t * FPS + 0.5))  # round half up (Python round() is banker's rounding)


# ------------------------------------------------------------------ scene setup
def new_scene(preset="standard", w=1080, h=1350, frames=300, world="WORLD_cartoon_sky", hold_z=None):
    """Empty scene with render preset, camera (+ overlay holder), and an optional world from the materials library."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    RP.apply_render_preset(scn, preset, w=w, h=h)
    scn.frame_start, scn.frame_end = 1, frames
    L._VIS.clear()
    L.HITS.clear()
    if hold_z is not None:  # distance of the caption plane from the camera (default 0.8 m; use ~0.06 for macro scenes)
        L.HOLD_Z = hold_z
        L.HOLD_S0 = hold_z / 3.0
    L.setup_camera()
    if world:
        with bpy.data.libraries.load(os.path.join(COMP, "materials_vfx", "lighting_and_world.blend"), link=False) as (src, dst):
            dst.worlds = [world] if world in src.worlds else []
        if dst.worlds:
            scn.world = dst.worlds[0]
    return scn


def rig(name="daylight", scale=1.0, loc=(0, 0, 0)):
    """Append a lighting rig ASSET_light_<name> (daylight, macro_studio, data_hall, lab, hot_chip_rim)."""
    a = append("materials_vfx/lighting_and_world", only=["ASSET_light_" + name])
    r = a.root
    r.scale = (scale,) * 3
    r.location = loc
    return a


# ------------------------------------------------------------------ appending assets
class Asset:
    def __init__(self, coll, root, objs):
        self.coll = coll
        self.root = root
        self.objs = objs

    def hook(self, name):
        n = "HOOK_" + name
        for o in self.objs:
            if o.name == n or o.name.startswith(n + ".0"):
                return o
        raise KeyError(n)

    def hooks(self):
        return [o.name for o in self.objs if o.name.startswith("HOOK_")]

    @property
    def armature(self):
        for o in self.objs:
            if o.type == "ARMATURE":
                return o
        return None

    def variant(self, name, show=True):
        """Show or hide a nested VARIANT_<name> collection (render and viewport)."""
        def walk(c):
            for ch in c.children:
                if ch.name == "VARIANT_" + name or ch.name.startswith("VARIANT_" + name + "."):
                    ch.hide_render = not show
                    ch.hide_viewport = not show
                    for lc in _layer_colls(bpy.context.view_layer.layer_collection):
                        if lc.collection is ch:
                            lc.exclude = False
                    return True
                if walk(ch):
                    return True
            return False
        if not walk(self.coll):
            raise KeyError("VARIANT_" + name)

    def prop(self, name, value):
        set_prop(self.root, name, value)


def _layer_colls(lc):
    yield lc
    for ch in lc.children:
        yield from _layer_colls(ch)


def _all_objs(coll):
    out = list(coll.objects)
    for ch in coll.children:
        out += _all_objs(ch)
    return out


def append(rel, only=None, actions=False, link=True):
    """Append ASSET_ collections from assets/components/<rel>.blend; returns the first as an Asset (others in .others)."""
    path = os.path.join(COMP, rel + ".blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        names = only or [c for c in src.collections if c.startswith("ASSET_")][:1]
        dst.collections = [n for n in names if n in src.collections]
        if actions:
            dst.actions = list(src.actions)
    assets = []
    for c in dst.collections:
        bpy.context.scene.collection.children.link(c)
        objs = _all_objs(c)
        root = next((o for o in objs if o.name.startswith("ROOT_")), None)
        assets.append(Asset(c, root, objs))
    a = assets[0]
    a.others = assets[1:]
    return a


def clone(asset, name=None):
    """Linked duplicate of an appended asset: copies the objects (sharing mesh data) into a new collection."""
    coll = bpy.data.collections.new(name or asset.coll.name + "_copy")
    bpy.context.scene.collection.children.link(coll)
    mp = {}
    for o in asset.objs:
        c = o.copy()
        coll.objects.link(c)
        mp[o] = c
    for o, c in mp.items():
        if c.animation_data is not None:
            c.animation_data_clear()  # copies share the action: clear so clones animate independently
        if o.parent in mp:
            c.parent = mp[o.parent]
            c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
    root = mp.get(asset.root)
    return Asset(coll, root, list(mp.values()))


def unique_material(obj, slot=0, rename=None):
    """Give obj its own copy of the material in a slot (object-linked) so it can be colored independently."""
    m = obj.material_slots[slot].material.copy()
    if rename:
        m.name = rename
    obj.material_slots[slot].link = "OBJECT"
    obj.material_slots[slot].material = m
    return m


# ------------------------------------------------------------------ placement, properties, animation
def place(obj, pos=(0, 0, 0), yaw=0.0, scale=None, rot=None):
    obj.location = pos
    obj.rotation_euler = rot if rot is not None else (0, 0, yaw)
    if scale is not None:
        obj.scale = (scale,) * 3 if isinstance(scale, (int, float)) else scale
    return obj


def set_prop(obj, name, value):
    obj[name] = value
    obj.update_tag()
    bpy.context.view_layer.update()


def key_prop(obj, name, t, value, interp="LINEAR"):
    """Keyframe a custom property (scene-local seconds)."""
    obj[name] = value
    obj.update_tag()
    obj.keyframe_insert('["%s"]' % name, frame=F(t))
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path == '["%s"]' % name:
                for kp in fc.keyframe_points:
                    if int(round(kp.co[0])) == F(t):
                        kp.interpolation = interp


def key_loc(obj, t, pos, yaw=None, interp="LINEAR", rot=None):
    obj.location = pos
    obj.keyframe_insert("location", frame=F(t))
    if yaw is not None or rot is not None:
        obj.rotation_euler = rot if rot is not None else (0, 0, yaw)
        obj.keyframe_insert("rotation_euler", frame=F(t))
    L._interp(obj, "location", F(t), interp)
    if yaw is not None or rot is not None:
        L._interp(obj, "rotation_euler", F(t), interp)


def play(asset, action, t, speed=1.0, hold=True, repeat=1.0, blend_in=0):
    """Schedule an action on the asset's armature as an NLA strip starting at scene time t.

    action: full name or the short name (e.g. 'walk' -> ACT_<asset_id>_walk).
    """
    arm = asset.armature
    if isinstance(action, str):
        nm = action if action.startswith("ACT_") else "ACT_%s_%s" % (asset.root["asset_id"], action)
        act = bpy.data.actions[nm]
    else:
        act = action
    ad = arm.animation_data_create()
    ad.action = None
    tr = ad.nla_tracks.new()
    st = tr.strips.new(act.name, F(t), act)
    st.blend_type = "REPLACE"
    st.scale = 1.0 / speed
    st.repeat = repeat
    st.extrapolation = "HOLD_FORWARD" if hold else "NOTHING"
    if blend_in:
        st.blend_in = blend_in
    return st


def action_len(asset, action):
    nm = action if action.startswith("ACT_") else "ACT_%s_%s" % (asset.root["asset_id"], action)
    a = bpy.data.actions[nm]
    return (a.frame_range[1] - a.frame_range[0]) / FPS


def attach(obj, parent_obj, offset=(0, 0, 0), rot=(0, 0, 0)):
    """Parent obj to another object (typically a HOOK_ empty on a bone) with a local offset."""
    obj.parent = parent_obj
    obj.matrix_parent_inverse.identity()
    obj.location = offset
    obj.rotation_euler = rot


def walk(asset, t0, pts, speed_mps=None, gait="walk"):
    """Walk the character root along pts [(x,y,z), ...] starting at t0 (scene-local seconds); returns the end time.

    Loops the in-place gait action and keys the root position at the action's stored speed; yaw faces travel.
    """
    arm = asset.armature
    act = bpy.data.actions["ACT_%s_%s" % (asset.root["asset_id"], gait)]
    sp = speed_mps or float(act.get("root_speed_mps", 1.2))
    t = t0
    total = 0.0
    key_loc(asset.root, t, pts[0], yaw=_yaw(pts[0], pts[1]), interp="LINEAR")
    for a, b in zip(pts[:-1], pts[1:]):
        d = (Vector(b) - Vector(a)).length
        t += d / sp
        key_loc(asset.root, t, b, yaw=_yaw(a, b), interp="LINEAR")
        total += d
    dur = total / sp
    play(asset, gait, t0, repeat=max(dur * FPS / max(act.frame_range[1] - act.frame_range[0], 1), 1.0), hold=False)
    return t


def _yaw(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.atan2(dx, -dy) if (dx or dy) else 0.0


def yaw_to(src, dst):
    return _yaw(src, dst)


# ------------------------------------------------------------------ effects from the materials_vfx library
def fx(name, t, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, parent=None, intensity=1.0, dur=1.5, show_pad=0.05):
    """Append ASSET_fx_<name> and trigger it at scene time t: p_t is keyed 0 -> dur seconds; visible only in the window."""
    a = append("materials_vfx/vfx_particles_and_effects", only=["ASSET_fx_" + name])
    r = a.root
    if parent is not None:
        r.parent = parent
        r.matrix_parent_inverse.identity()
    r.location, r.rotation_euler, r.scale = loc, rot, (scale,) * 3
    if "p_auto" in r.keys():
        r["p_auto"] = 0
    if "p_intensity" in r.keys():
        r["p_intensity"] = intensity
    key_prop(r, "p_t", t, 0.0)
    key_prop(r, "p_t", t + dur, dur)
    show(a, t - show_pad, t + dur + show_pad)
    return a


# ------------------------------------------------------------------ visibility of whole assets (collection windows)
_COLL_WINDOWS = {}


def show(asset_or_coll, t0, t1):
    c = asset_or_coll.coll if isinstance(asset_or_coll, Asset) else asset_or_coll
    _COLL_WINDOWS.setdefault(c, []).append((F(t0), F(t1)))


def _finalize_collections(total):
    """Collection.hide_render is not animatable in Blender 4.2: key the objects of each collection instead."""
    for c, wins in _COLL_WINDOWS.items():
        for o in _all_objs(c):
            for f0, f1 in wins:
                L.VA(o, f0, f1)


# ------------------------------------------------------------------ captions (scene-local)
def cap(t0, t1, s): L.ovt("CAP", s, 0, t0, t1)
def big(t0, t1, s): L.ovt("BIG", s, 0, t0, t1)
def lab(t0, t1, s): L.ovt("LAB", s, 0, t0, t1)
def _post(kind, t0, t1, s):
    OVERLAYS.append({"kind": kind, "t0": round(t0, 3), "t1": round(t1, 3), "text": s})


def card(t0, t1, s):
    _post("CARD", t0, t1, s)
    if HUD:
        L.ovt("CARD", s, 0, t0, t1)


def fxn(t0, t1, s):
    _post("FX", t0, t1, s)
    if HUD:
        L.ovt("FX", s, 0, t0, t1)


def narr(segs): L.narr(0, segs)
def wl(body, pos, t0, t1, size=0.3, color=(1.0, 1.0, 1.0)): return L.wl(body, pos, 0, t0, t1, size=size, color=color)


def narr_vo(scene_no):
    """Subtitles for scene `scene_no` from scripts/audio/narration.json (same source as the voice track)."""
    import json
    d = json.load(open(os.path.join(SCRIPTS, "audio", "narration.json")))
    off = 10.0 * (scene_no - 1)
    segs = [(round(x["t0"] - off, 3), round(x["t1"] - off, 3), x["show"]) for x in d["lines"] if off <= x["t0"] < off + 10.0]
    narr(segs)
    return segs


def timecode(scene_no, seconds=10):
    for s in range(seconds):
        _post("TC", s, s + 1, "S%d  0:%02d" % (scene_no, s))
    if not HUD:
        return
    for s in range(seconds):
        L.ovt("TC", "S%d  0:%02d" % (scene_no, s), 0, s, s + 1)


def shot(t0, t1, p0, a0, p1=None, a1=None, lens=28.0, lens1=None, ease="LINEAR"):
    L.shot(0, t0, t1, p0, a0, p1, a1, lens=lens, lens1=lens1, ease=ease)


def finalize(out_blend, frames=300):
    _finalize_collections(frames)
    L.finalize_visibility(frames)
    scn = bpy.context.scene
    scn.frame_end = frames
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    import json
    with open(out_blend + ".overlays.json", "w") as fh:
        json.dump({"fps": FPS, "frames": frames, "hud_rendered": HUD, "overlays": OVERLAYS}, fh, indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print("SAVED", out_blend)
