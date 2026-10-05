"""Motion library v2: public API.

Bake (offline, one rig blend at a time; the rig blend is only read):
    Blender -b assets/components/characters/gary.blend --python scripts/assets/characters/motion_v2/motion_v2.py -- bake
  or from Python inside Blender with a character blend open:
    import motion_v2 as M2;  M2.bake_all(out_blend=".../gary_motion_v2.blend")

Use in a scene script (mirrors asm.play):
    import motion_v2 as M2
    M2.apply(gary, "walk", t0)            # asset = asm.Asset; short name -> ACT_<asset_id>_v2_<name>
    M2.walk(gary, t0, pts)                 # root path + gait strip, like asm.walk
    M2.apply(gary, "shot_hit_fall", t_hit)

See DevLog/v1/DevLog-005-motion-v2.md (usage guide) and assets/components/characters/motion_v2/*_manifest.json.
"""
import json
import math
import os
import subprocess
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
COMP = os.path.join(PROJ, "assets", "components")
CHARS = os.path.join(COMP, "characters")
LIBDIR = os.path.join(CHARS, "motion_v2")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import m2_engine as E  # noqa: E402
import m2_lib as LIB  # noqa: E402
import m2_loco  # noqa: E402,F401
for _m in ("m2_idle", "m2_fight", "m2_react", "m2_acts"):
    try:
        __import__(_m)
    except ImportError as _e:  # module not written yet
        if _m not in str(_e):
            raise

UPPER_VARIANTS = ["thinking", "point", "point_accuse", "shout_rant", "shrug", "facepalm", "flinch", "whiteboard_write", "whiteboard_underline", "typing", "plug_connector", "idle_arms_cross", "idle_look_around"]
LIB.add_upper_variants(UPPER_VARIANTS)


# proposed per-scene use (scene agents decide; S4/S5 were not inspected beyond the v1 scripts' action calls)
SCENE_USE = {
    "S1_copper": {"Gary": ["idle_breathe", "idle_fidget", "walk", "pull_cable", "flinch", "shot_hit_fall", "ground_twitch"],
                  "Manager": ["stomp_walk", "idle_breathe_tense", "manager_slow_burn", "gun_raise_aim_fire", "gun_fire"]},
    "S2_retimers": {"Gary": ["idle_look_around", "idle_fidget", "startle", "shot_hit_fall"], "Manager": ["thinking", "idle_arms_cross", "manager_slow_burn", "gun_raise_aim_fire"]},
    "S3_npo": {"Gary": ["idle_breathe", "shrug", "facepalm", "startle", "flinch"],
               "vendors (npc rigs)": ["fight_idle", "chest_bump", "shove", "react_shoved", "haymaker", "react_head_hit", "hook", "uppercut", "react_gut_hit", "slap", "react_slapped", "head_butt", "react_headbutt",
                                      "collar_grab_shake", "collar_shaken", "duck", "whiff_overbalance", "flail_windmill", "grapple", "fall_back_brawl", "get_up", "shout_rant", "point_accuse"],
               "Manager": ["walk_brisk", "gun_raise_aim_fire"]},
    "S4_cpo": {"Gary": ["idle_breathe", "panic_wiggle", "flinch", "shot_hit_fall"], "Manager": ["gun_raise_aim_fire", "shout_rant"]},
    "S5_wafer_test": {"Gary": ["thinking", "plug_connector", "typing", "shrug", "shot_hit_fall"], "Manager": ["idle_arms_cross", "whiteboard_write", "whiteboard_underline", "gun_raise_aim_fire"]},
    "S6_fiber": {"Gary": ["kneel_down", "tie_fibers_kneel", "tie_fibers_stand", "kneel_up", "plug_connector", "whipped", "get_up", "hug_squashed", "panic_wiggle", "facepalm"],
                 "Manager": ["stomp_walk", "anger_outburst", "point_accuse", "shout_rant"], "customers (npc rigs)": ["walk", "idle_arms_cross", "hug_give", "shrug"]},
}

FPS = 30
VERSION = "motion_v2 2026-10-02"


# ---------------------------------------------------------------------------------------------------- baking
def find_root():
    return next(o for o in bpy.data.objects if o.name.startswith("ROOT_"))


def build_spec(rig, name):
    ent = LIB.REG[name]
    spec = LIB.finalize_spec(ent["builder"](rig))
    spec["mirror"] = ent["mirror"]
    if spec["mirror"]:
        for k in ("root_yaw_delta_rad",):
            if k in spec["meta"]:
                spec["meta"][k] = -spec["meta"][k]
    if spec.get("group") is None:
        spec["group"] = ent["group"]
    return spec


def sample_spec(rig, spec):
    """Spec -> list of poses (frame 0..n)."""
    n, loop = spec["n"], spec["loop"]
    if "timeline" in spec:
        T = spec["timeline"]
        fn = lambda f: T.eval(f)  # noqa: E731
    else:
        fn = spec["fn"]
    return E.sample_poses(fn, n, loop, spec.get("overlap"), mirror=spec["mirror"])


def bake_one(rig, name, log=print):
    spec = build_spec(rig, name)
    n = spec["n"]
    rig.warn.clear()
    poses = sample_spec(rig, spec)
    data = E.solve_frames(rig, poses, spec.get("attach"))
    ent = LIB.REG[name]
    meta = dict(spec["meta"])
    meta.update(note=ent["note"], use=ent["use"])
    act, nk = E.write_action(rig, name, data, n, spec["loop"], spec["group"], meta={k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))})
    face_name = None
    if spec.get("face"):
        face_name = write_face_action(rig, name, spec["face"], n, spec["loop"])
    reach = {}
    rf = {}
    for k, d, fr in rig.warn:
        if d > reach.get(k, 0.0):
            rf[k] = fr
        reach[k] = max(reach.get(k, 0.0), d)
    info = dict(
        name=act.name, short=name, frame_start=0, frame_end=n, frames=n, loop=bool(spec["loop"]), fps=FPS, layer_group=spec["group"],
        events=[dict(name=e[0], frame=int(e[1])) for e in spec["events"]],
        hitstop=[list(map(int, h)) for h in spec.get("hitstop", [])],
        hooks=spec.get("hooks", []), face_action=face_name, requires_props=spec.get("props", []),
        use=ent["use"], note=ent["note"], mirrored_from=ent["variant_of"], mirrored=bool(spec["mirror"]),
        keys=nk, reach_violations_m={k: round(v * rig.K, 4) for k, v in reach.items()}, reach_worst_frame=rf, meta=spec["meta"],
    )
    for k, v in info["meta"].items():
        if isinstance(v, np.generic):
            info["meta"][k] = float(v)
    return act, info


def write_face_action(rig, name, face, n, loop):
    """face: list of (frame, {prop: value}); linear interpolation between keys; keys the root empty custom properties."""
    root = rig.root_obj
    full = "ACT_%s_v2_%s_face" % (rig.aid, name)
    if full in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[full])
    act = bpy.data.actions.new(full)
    act.use_fake_user = True
    props = sorted({p for _, d in face for p in d})
    for p in props:
        pts = [(f, d[p]) for f, d in face if p in d]
        if pts[0][0] > 0:
            pts = [(0, 0.0)] + pts
        fc = act.fcurves.new(data_path='["%s"]' % p, index=0, action_group="face")
        fc.keyframe_points.add(len(pts))
        for kp, (f, v) in zip(fc.keyframe_points, pts):
            kp.co = (float(f), float(v))
            kp.interpolation = "LINEAR"
        fc.update()
    act.use_frame_range = True
    act.frame_start, act.frame_end = 0.0, float(n)
    act["loop"] = bool(loop)
    return act.name


def bake_all(rig_blend=None, out_blend=None, only=None, manifest=None, log=print):
    """Bake every registered action on the CURRENTLY OPEN character blend (or open rig_blend first) and write the action library.

    out_blend: actions-only library .blend (fake users). manifest: JSON path (default next to out_blend)."""
    if rig_blend:
        bpy.ops.wm.open_mainfile(filepath=rig_blend)
    root = find_root()
    for o in bpy.data.objects:   # no mesh evaluation while solving poses
        if o.type == "MESH":
            o.hide_viewport = True
    rig = E.Rig2(root)
    arm = rig.arm
    names = [n for n in LIB.ORDER if (not only or n in only)]
    infos = []
    for n in names:
        try:
            act, info = bake_one(rig, n, log)
        except Exception as ex:  # keep going, report
            import traceback
            traceback.print_exc()
            log("FAILED %s: %r" % (n, ex))
            continue
        infos.append(info)
        log("baked %-28s %4d frames  keys %6d  reach %s @%s" % (info["name"], info["frames"], info["keys"], info["reach_violations_m"], info["reach_worst_frame"]))
    rig.clear()
    if arm.animation_data:
        arm.animation_data.action = None
    acts = {a for a in bpy.data.actions if a.name.startswith("ACT_%s_v2_" % rig.aid)}
    if out_blend:
        os.makedirs(os.path.dirname(out_blend), exist_ok=True)
        bpy.data.libraries.write(out_blend, acts, fake_user=True)
        log("WROTE %s (%.1f MB, %d actions)" % (out_blend, os.path.getsize(out_blend) / 1e6, len(acts)))
        mp = manifest or out_blend.replace(".blend", "_manifest.json")
        man = dict(
            library=VERSION, asset_id=rig.aid, rig_blend=os.path.basename(bpy.data.filepath) if bpy.data.filepath else None,
            K=rig.K, fps=FPS, frame_convention="action frame 0 = first pose; loops: last frame equals first (strip repeat does not double the seam frame)",
            coordinate_convention="baseline character space: x left, y back, z up; forward = -y; multiply metres by K for this rig",
            scene_use=SCENE_USE, naming="ACT_<asset>_v2_<name>; face actions ACT_<asset>_v2_<name>_face key the ROOT empty p_ properties; <name>_L = left-hand variant, <name>_upper = upper-body layer",
            actions=sorted(infos, key=lambda d: d["name"]))
        with open(mp, "w") as fh:
            json.dump(man, fh, indent=1, default=float)
        log("WROTE %s" % mp)
    return infos


# ---------------------------------------------------------------------------------------------------- using the library in scenes
def lib_path(asset_id):
    return os.path.join(LIBDIR, "%s_motion_v2.blend" % asset_id)


def action_name(asset, name):
    aid = asset.root["asset_id"] if hasattr(asset, "root") else asset
    return name if name.startswith("ACT_") else "ACT_%s_v2_%s" % (aid, name)


def ensure_action(asset, name, path=None):
    """Return the bpy Action, appending it (and its face action) from the library blend when absent."""
    full = action_name(asset, name)
    if full in bpy.data.actions:
        return bpy.data.actions[full]
    aid = asset if isinstance(asset, str) else asset.root["asset_id"]
    p = path or lib_path(aid)
    if not os.path.exists(p):
        raise FileNotFoundError("no motion_v2 library for %s: %s (run bake_all on the %s blend)" % (aid, p, aid))
    with bpy.data.libraries.load(p, link=False) as (src, dst):
        want = [n for n in (full, full + "_face") if n in src.actions]
        dst.actions = want
    for a in dst.actions:
        a.use_fake_user = True
    if full not in bpy.data.actions:
        raise KeyError(full)
    return bpy.data.actions[full]


def F(t):
    return 1 + int(math.floor(t * FPS + 0.5))


def _new_track(ad, name):
    tr = ad.nla_tracks.new()
    tr.name = name
    return tr


def apply(asset, name, t0, speed=1.0, hold=True, repeat=1.0, blend_in=0, blend_out=0, layer=None, blend="REPLACE", influence=1.0,
          start_frame=None, end_frame=None, face=True, key_root_yaw=True):
    """Schedule action `name` on the asset armature as an NLA strip at scene time t0 (seconds; frame = 1 + round(t*30)).

    layer: None = default track per call (later calls override earlier ones, like asm.play); a string names the track (strips on
    one named track must not overlap). Layered actions (layer_group upper/lower/arms/head) only key their own bones, so they combine
    with a full-body idle or walk strip underneath. blend: REPLACE | COMBINE | ADD (additive on top of lower tracks).
    start_frame / end_frame trim the action (e.g. play a hit up to the impact frame, hold, resume).
    Returns the strip (strip.frame_end is the end frame)."""
    arm = asset.armature
    act = ensure_action(asset, name)
    ad = arm.animation_data_create()
    ad.action = None
    tname = "v2_" + (layer or name)
    tr = None
    if layer:
        for t in ad.nla_tracks:
            if t.name == tname:
                tr = t
    if tr is None:
        tr = _new_track(ad, tname)
    st = tr.strips.new(act.name, F(t0), act)
    if start_frame is not None:
        st.action_frame_start = float(start_frame)
    if end_frame is not None:
        st.action_frame_end = float(end_frame)
    st.blend_type = blend
    st.influence = influence
    st.use_animated_influence = False
    st.scale = 1.0 / speed
    st.repeat = repeat
    st.extrapolation = "HOLD_FORWARD" if hold else "NOTHING"
    if blend_in:
        st.blend_in = blend_in
    if blend_out:
        st.blend_out = blend_out
    fa = bpy.data.actions.get(act.name + "_face")
    if face and fa is None and (act.name + "_face") not in bpy.data.actions:
        try:
            fa = ensure_action(asset, name + "_face")
        except KeyError:
            fa = None
    if face and fa is not None and asset.root is not None:
        rad = asset.root.animation_data_create()
        ftr = _new_track(rad, "v2_face_" + name)
        fs = ftr.strips.new(fa.name, F(t0), fa)
        if start_frame is not None:
            fs.action_frame_start = float(start_frame)
        if end_frame is not None:
            fs.action_frame_end = float(end_frame)
        fs.scale = 1.0 / speed
        fs.repeat = repeat
        fs.extrapolation = "HOLD_FORWARD" if hold else "NOTHING"
        fs.blend_type = "REPLACE"
    dur = (st.frame_end - st.frame_start) / FPS
    # root yaw hand-off for turn actions (only meaningful when the strip is not held)
    yaw = float(act.get("root_yaw_delta_rad", 0.0))
    if key_root_yaw and yaw and not hold:
        r = asset.root
        yaw0 = r.rotation_euler[2]
        if r.animation_data and r.animation_data.action:       # evaluated yaw at t0 when the root is keyed
            for fc in r.animation_data.action.fcurves:
                if fc.data_path == "rotation_euler" and fc.array_index == 2:
                    yaw0 = fc.evaluate(F(t0))
        full_len = max(act.frame_range[1] - act.frame_range[0], 1.0)
        frac = (st.action_frame_end - st.action_frame_start) / full_len
        fe = int(round(st.frame_end))
        r.rotation_euler[2] = yaw0
        r.keyframe_insert("rotation_euler", frame=F(t0), index=2)
        r.keyframe_insert("rotation_euler", frame=fe, index=2)
        r.rotation_euler[2] = yaw0 + yaw * frac * repeat
        r.keyframe_insert("rotation_euler", frame=fe + 1, index=2)
        if r.animation_data and r.animation_data.action:
            for fc in r.animation_data.action.fcurves:
                if fc.data_path == "rotation_euler" and fc.array_index == 2:
                    for kp in fc.keyframe_points:
                        if abs(kp.co[0] - fe) < 0.5:
                            kp.interpolation = "CONSTANT"
    return st


def duration(asset, name, speed=1.0):
    a = ensure_action(asset, name)
    return (a.frame_range[1] - a.frame_range[0]) / FPS / speed


def walk(asset, t0, pts, gait="walk", speed_mps=None):
    """Move the asset root along pts [(x,y,z), ...] starting at scene time t0 with a v2 gait (walk / walk_brisk / run / stomp_walk).
    Root speed is the action's stored root_speed_mps (foot-lock speed); returns the end time. yaw faces travel."""
    if len(pts) < 2:
        raise ValueError("walk needs at least two points")
    sys.path.insert(0, os.path.join(PROJ, "scripts", "film_v1"))
    import asm  # noqa: E402
    act = ensure_action(asset, gait)
    sp = float(speed_mps or act["root_speed_mps"])
    # speed scaling of the strip keeps the feet locked: strip speed = requested / baked speed
    sstrip = sp / float(act["root_speed_mps"])
    t = t0
    total = 0.0
    asm.key_loc(asset.root, t, pts[0], yaw=asm.yaw_to(pts[0], pts[1]), interp="LINEAR")
    for a, b in zip(pts[:-1], pts[1:]):
        d = math.dist(a, b)
        t += d / sp
        asm.key_loc(asset.root, t, b, yaw=asm.yaw_to(a, b), interp="LINEAR")
        total += d
    dur = total / sp
    cyc = (act.frame_range[1] - act.frame_range[0]) / FPS / sstrip
    apply(asset, gait, t0, speed=sstrip, hold=False, repeat=max(dur / cyc, 1.0))
    return t


# ---------------------------------------------------------------------------------------------------- review sheets
def compose_sheet(tiles, out_png, cols=4, tw=270, max_kb=380):
    """ffmpeg tile of PNG tiles (already rendered at 540x675) into one contact sheet, quantised if needed to stay under max_kb."""
    d = os.path.dirname(tiles[0])
    lst = os.path.join(d, "_list.txt")
    with open(lst, "w") as fh:
        for t in tiles:
            fh.write("file '%s'\nduration 1\n" % t)
    rows = int(math.ceil(len(tiles) / cols))
    vf = "scale=%d:-1,tile=%dx%d:padding=2:color=0x20242a" % (tw, cols, rows)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", vf, "-frames:v", "1", out_png], check=True)
    pal = out_png.replace(".png", "_q.png")   # 128-colour palette (flat clay shading) keeps sheets small
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out_png, "-vf", "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=5", pal], check=True)
    os.replace(pal, out_png)
    os.remove(lst)
    return os.path.getsize(out_png)


if __name__ == "__main__" or "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if argv and argv[0] == "bake":
        root = find_root()
        aid = root["asset_id"]
        only = argv[1].split(",") if len(argv) > 1 and argv[1] != "all" else None
        bake_all(out_blend=os.path.join(LIBDIR, "%s_motion_v2.blend" % aid), only=only)
