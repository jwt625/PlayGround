"""Scene S5 (film 40-50 s): CPO yield, back through the factory to wafer test. Built from the v1 asset library.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s05_wafer_test.py -- scenes/v1/s05_wafer_test.blend
Timing, camera moves, captions, narration and FX notes follow the crude s5 in scripts/build_crude_film.py.
Scene-local time in seconds, frame = 1 + round(t * 30); 300 frames; 1080x1350.

Scale policy (see DevLog/v1/DevLog-003-scene-s05.md): stations, wafers, card, screens, characters are real size (1 unit = 1 m).
Only the hero die + engine (the object that is yanked back through the line) is scaled up by HERO_S so it reads next to
2 m machines; the probe card is shown sub-scale (PROBE_CARD_S) because the real 410 mm card would hide the 300 mm wafer.
"""
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asm  # noqa: E402
from asm import F  # noqa: E402

PROJ = asm.PROJ
sys.path.insert(0, os.path.join(PROJ, "scripts", "assets", "fab_test"))
import tex_gen as T  # noqa: E402
import blender_lib as L  # noqa: E402
import die_map_tools as D  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(PROJ, "scenes", "v1", "s05_wafer_test.blend")
TEX = os.path.join(PROJ, "assets", "generated_textures", "v1", "s05")
PI = math.pi

# ------------------------------------------------------------------ layout and scale constants
HERO_S = 90.0          # hero die and engine scale-up (die 7 x 9 mm -> 0.63 x 0.81 m)
WAFER_LIFT = 0.075     # wafer shown raised above the platen during probing (real wafer sits 43 mm below the platen top and is hidden)
PROBE_CARD_S = 0.22    # probe card shown sub-scale (real card 410 mm would cover the whole wafer from above); v1.2 0.15 -> 0.22
MOTION_V2 = os.path.join(os.path.dirname(HERE), "assets", "characters", "motion_v2")
sys.path.insert(0, MOTION_V2)
import motion_v2 as M2  # noqa: E402
import s05_tex  # noqa: E402
# v1.1 layout: stations side by side (gaps 0.35-0.5 m), fronts aligned at about y = -0.75; per-station y offsets align the fronts
X_ENG = -4.6                                  # engine with the hero die (start of the shot)
X_OVEN, X_FAU, X_BOND, X_SAW, X_CM = 0.0, 4.0, 5.55, 7.8, 9.6
Y_OVEN, Y_FAU, Y_BOND, Y_SAW, Y_CM = -0.05, -0.40, -0.23, 0.0, -0.10
DIE_Y, DIE_Z = -1.15, 1.30                    # hero die path: in front of the machines
T_END = 10.0
MOTION_SHUTTER = 0.5
HERO_LINE_S = 36.0     # hero die scale while it dwells at the line stations

CLAY = {"white": (0.95, 0.95, 0.95), "blue": (0.15, 0.3, 0.8), "yellow": (0.95, 0.8, 0.15), "red": (0.85, 0.15, 0.12)}


# ------------------------------------------------------------------ small helpers
def kf_hide(coll, t, hidden, hold_prev=True):
    """Keyframe a collection's render/viewport hide at scene time t (CONSTANT)."""
    f = F(t)
    coll.hide_render = hidden
    coll.hide_viewport = hidden
    coll.keyframe_insert("hide_render", frame=f)
    coll.keyframe_insert("hide_viewport", frame=f)


def _intersect(a, b):
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if lo < hi:
                out.append((lo, hi))
    return sorted(out) or [(10 ** 6, 10 ** 6 + 1)]


def collection_windows_to_objects():
    """Framework workaround: Collection.hide_render is not animatable in Blender 4.2, so asm.show windows are converted
    to per-object visibility windows (intersected with any object-level windows already registered)."""
    for c, wins in asm._COLL_WINDOWS.items():
        for o in asm._all_objs(c):
            cur = L._VIS.get(o)
            L._VIS[o] = list(wins) if cur is None else _intersect(cur, wins)
    asm._COLL_WINDOWS.clear()


def constant_fcurves(id_data):
    ad = id_data.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def find_coll(root_coll, prefix):
    for ch in root_coll.children_recursive:
        if ch.name == prefix or ch.name.startswith(prefix + "."):
            return ch
    raise KeyError(prefix)


def world_of(obj):
    bpy.context.view_layer.update()
    return obj.matrix_world.copy()


def hole_value(T_now, T_shot):
    """Hole radius after the storyboard rule: x0.9 every 1.5 s after the hit, floor 0.3 (step function)."""
    return max(0.3, 0.9 ** math.floor((T_now - T_shot) / 1.5))


def key_hole(root, prop, T_shot, t_end=10.0, t_film0=40.0):
    """Key a bullet-hole radius on a character root for the whole scene (CONSTANT steps)."""
    v0 = hole_value(t_film0, T_shot)
    asm.key_prop(root, prop, 0.0, v0, "CONSTANT")
    k = math.floor((t_film0 - T_shot) / 1.5) + 1
    while True:
        t_step = T_shot + 1.5 * k - t_film0
        if t_step >= t_end:
            break
        asm.key_prop(root, prop, t_step, hole_value(t_film0 + t_step, T_shot), "CONSTANT")
        k += 1

# ------------------------------------------------------------------ easing, baking helpers (v1.1)
def sstep(u):
    """smootherstep: zero velocity and acceleration at both ends."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def lerp3(a, b, u):
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(len(a)))


def eval_stops(stops, t):
    """stops: [(t, vec), ...] sorted; piecewise smootherstep between consecutive stops (dwell = zero speed at every stop)."""
    if t <= stops[0][0]:
        return tuple(stops[0][1])
    for (t0, v0), (t1, v1) in zip(stops[:-1], stops[1:]):
        if t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            return lerp3(v0, v1, sstep(u))
    return tuple(stops[-1][1])


def kframe(obj, f, loc=None, rot=None, scale=None):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert("location", frame=f)
    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert("rotation_euler", frame=f)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert("scale", frame=f)


def linear_all(obj, paths=("location", "rotation_euler", "scale")):
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path in paths:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


def add_noise(obj, strength, scale_frames=9.0, seed=1, paths=("location",)):
    """Handheld noise: additive NOISE modifiers on the baked F-curves (deterministic for a given phase)."""
    for fc in obj.animation_data.action.fcurves:
        if fc.data_path in paths:
            m = fc.modifiers.new("NOISE")
            m.scale = scale_frames
            m.strength = strength
            m.phase = seed * 7.3 + fc.array_index * 3.1
            m.depth = 1


# ------------------------------------------------------------------ environment: fab bay
def floor_material(fl):
    """v1.2: tiled floor (0.6 m raised-floor tiles, darker than the walls) so the floor reads as a floor under the fallen Gary."""
    m = bpy.data.materials.new("MAT_s05_floor_tiles")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.45
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.0
    br.squash_frequency = 1
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Brick Width"].default_value = 0.6
    br.inputs["Row Height"].default_value = 0.6
    br.inputs["Mortar Size"].default_value = 0.008
    br.inputs["Mortar Smooth"].default_value = 0.3
    br.inputs["Color1"].default_value = (0.40, 0.42, 0.44, 1.0)
    br.inputs["Color2"].default_value = (0.45, 0.47, 0.49, 1.0)
    br.inputs["Mortar"].default_value = (0.22, 0.23, 0.25, 1.0)
    nt.links.new(geo.outputs["Position"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    fl.data.materials.clear()
    fl.data.materials.append(m)


def build_env():
    # cleanroom floor (grey tiles, darker than the walls), long enough for the whole line
    fl = L.box("floor", (3.0, 0, -0.05), (40.0, 30.0, 0.1), (0.45, 0.47, 0.49), rough=0.45)
    floor_material(fl)
    # aisle markings (yellow) along the line, 1.9 m in front of the machine fronts
    for y in (-2.6, -2.75):
        L.box("aisle_line", (3.0, y, 0.002), (40.0, 0.05, 0.004), (0.55, 0.45, 0.06), rough=0.6)
    # back wall of the fab bay (white wall panels) and wall base strip
    L.box("fab_wall", (3.0, 4.0, 3.0), (40.0, 0.2, 6.0), (0.70, 0.72, 0.75), rough=0.8)
    L.box("fab_wall_base", (3.0, 3.88, 0.15), (40.0, 0.05, 0.3), (0.30, 0.40, 0.55), rough=0.5)
    # ceiling light panels (emissive; v1.1: emission 6 -> 1.5; v1.2: 0.8, below the bloom trigger at draft)
    for ix in range(-3, 6):
        for y in (-1.5, 1.5):
            L.box("ceil_panel", (ix * 3.2, y, 4.7), (2.4, 0.5, 0.06), (1.0, 0.97, 0.92), emit=0.8)


def lighting():
    """Lab lighting rig (ASSET_light_lab) scaled 1.6, one per zone; visible only in its window (windows overlap to avoid pops)."""
    zones = [  # (x, y, windows)
        (-3.0, -0.5, [(0.0, 0.9)]), (1.0, -0.5, [(0.3, 1.5)]),
        (X_FAU, -0.5, [(1.0, 2.0)]), (X_BOND, -0.5, [(1.6, 2.7)]), (X_SAW, -0.5, [(2.3, 3.2)]),
        (X_CM - 1.2, -2.6, [(3.0, 10.0)]),
    ]
    rigs = []
    # broad soft fill from the aisle side (camera side) so the machine fronts are not in shadow
    ld = bpy.data.lights.new("s05_fill", "AREA")
    ld.energy = 180.0
    ld.size = 9.0
    ld.size_y = 3.0
    fo = bpy.data.objects.new("s05_fill", ld)
    bpy.context.scene.collection.objects.link(fo)
    fo.location = (4.5, -4.5, 3.4)
    fo.rotation_euler = (math.radians(65), 0, 0)
    # v1.2: warm soft key on the human set at the prober (contact shadows under the characters, warm clay look)
    kd = bpy.data.lights.new("s05_key_people", "AREA")
    kd.energy = 240.0
    kd.size = 1.8
    kd.color = (1.0, 0.88, 0.74)
    ko = bpy.data.objects.new("s05_key_people", kd)
    bpy.context.scene.collection.objects.link(ko)
    ko.location = (8.6, -3.4, 3.6)
    d = Vector((10.2, -0.6, 0.8)) - ko.location
    ko.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    L.V(ko, 0, 6.6, 10.0)
    for x, y, wins in zones:
        r = asm.rig("lab", scale=1.6, loc=(x, y, 0.0))
        for o in r.objs:
            if o.type == "LIGHT" and o.data.type == "SPOT":
                o.data.energy = 0.0
        if x == X_CM - 1.2 and "p_energy" in r.root.keys():
            r.root["p_energy"] = 0.6          # v1.2: prober zone was blown out (deck, paper, wall) under the compositor bloom
        for w in wins:
            asm.show(r, w[0], w[1])
        rigs.append(r)
    return rigs


# ------------------------------------------------------------------ hero die + engine
def build_hero():
    eng = asm.append("photonics/oe_module_cpo")
    try:
        eng.variant("open", False)
    except KeyError:
        pass
    asm.place(eng.root, (X_ENG, 0, 0.62), scale=HERO_S)
    asm.show(eng, 0.0, 0.85)
    pic = asm.append("photonics/pic_die")
    eic = asm.append("photonics/eic_die")
    grp = bpy.data.objects.new("HERO_DIE", None)
    bpy.context.scene.collection.objects.link(grp)
    for a in (pic, eic):
        a.root.parent = grp
        a.root.matrix_parent_inverse.identity()
    # EIC bonded face down on the PIC site: rotate 180 deg about X, then land its bond-face hook on the PIC site hook
    eic.root.rotation_euler = (PI, 0, 0)
    bpy.context.view_layer.update()
    tgt = world_of(pic.hook("eic_site_center")).translation
    cur = world_of(eic.hook("bond_face_center")).translation
    eic.root.location = eic.root.location + (tgt - cur)
    bpy.context.view_layer.update()
    return eng, pic, eic, grp


# station dwell definitions: (name, x_station, y_station, work point, camera offset from the work point, lens, die offset)
def dwell_table():
    return {
        "oven": dict(w=(X_OVEN + 0.75, Y_OVEN, 0.95), cam=(0.05, -1.75, 0.80), lens=32.0, die=(-0.22, -0.60, -0.06)),
        "fau": dict(w=(X_FAU + 0.0, Y_FAU, 0.95), cam=(0.10, -1.45, 0.65), lens=34.0, die=(-0.22, -0.45, -0.06)),
        "bond": dict(w=(X_BOND + 0.15, Y_BOND, 1.08), cam=(0.10, -1.55, 0.50), lens=34.0, die=(-0.22, -0.50, -0.08)),
        "saw": dict(w=(X_SAW, Y_SAW + 0.08, 1.08), cam=(0.10, -1.60, 0.50), lens=34.0, die=(-0.22, -0.55, -0.08)),
    }


# dwell timings (arrive, leave) in scene seconds
DW = {"oven": (0.95, 1.20), "fau": (1.45, 1.80), "bond": (2.10, 2.45), "saw": (2.75, 3.05)}


def hero_stops(z_rest):
    """Hero die x/y/z stops. Dwell beside each work point, eased between (no linear motion)."""
    tab = dwell_table()
    stops = [(0.0, (X_ENG, 0.0, z_rest)), (0.3, (X_ENG, 0.0, z_rest))]
    for k in ("oven", "fau", "bond", "saw"):
        d = tab[k]
        wx, wy, wz = d["w"]
        dx, dy, dz = d["die"]
        p = (wx + dx, wy + dy, wz + dz)
        stops.append((DW[k][0], p))
        stops.append((DW[k][1], (p[0] + 0.06, p[1], p[2])))
    stops.append((3.3, (X_CM - 0.4, DIE_Y, DIE_Z)))
    return stops


def animate_hero(grp, pic, eic):
    """Die on the engine 0-0.3, then yanked +X through the line (reverse order), eased dwell at every tool, gone at 3.3 s."""
    z_rest = 0.62 + 3.3e-3 * HERO_S + 0.004  # on the engine lid
    stops = hero_stops(z_rest)
    nf = F(3.3)
    for f in range(1, nf + 1):
        t = (f - 1) / 30.0
        x, y, z = eval_stops([(a, b) for a, b in stops], t)
        # progress along travel for the yaw wobble and tilt (die faces the camera at dwell, banks during travel)
        vx = eval_stops(stops, t + 1 / 30.0)[0] - x
        speed = abs(vx) * 30.0
        tilt = 0.0 if t <= 0.3 else 1.1
        yaw = 0.0 if t <= 0.3 else min(0.7, 0.05 * speed) + 0.25 * math.sin(t * 9.0)
        # hero scale: x90 on the engine, eased down to x50 for the dwell shots so the die does not hide the tool interior
        sc_h = HERO_S + (HERO_LINE_S - HERO_S) * sstep((t - 0.3) / 0.35)
        kframe(grp, f, loc=(x, y, z), rot=(tilt, 0.0, yaw), scale=(sc_h,) * 3)
    linear_all(grp)
    asm.show(pic, 0.0, 3.3)
    asm.show(eic, 0.0, 3.3)
    tr = asm.append("materials_vfx/vfx_particles_and_effects", only=["ASSET_fx_glow_trail"])
    tr.root.parent = grp
    tr.root.matrix_parent_inverse.identity()
    tr.root.location = (0, 0, 0.002)
    tr.root.rotation_euler = (0, 0, PI / 2)
    for k, v in (("p_length", 0.014), ("p_width", 0.004)):
        if k in tr.root.keys():
            tr.root[k] = v
    tr.root.scale = (0.12,) * 3          # v1.2: smaller trail (bloomed cyan blob under the subtitles)
    asm.show(tr, 0.3, 3.3)
    return grp


# ------------------------------------------------------------------ line stations
def build_line(hero_grp):
    st = {}
    oven = asm.append("fab_test/reflow_oven_line")
    asm.place(oven.root, (X_OVEN, Y_OVEN, 0))
    st["oven"] = oven
    asm.show(oven, 0.0, 3.6)
    fau = asm.append("fab_test/fau_attach_station")
    asm.place(fau.root, (X_FAU, Y_FAU, 0))
    asm.show(fau, 0.0, 3.6)
    bond = asm.append("fab_test/die_bonder_station")
    asm.place(bond.root, (X_BOND, Y_BOND, 0))
    try:
        bond.variant("enclosure_glass", False)
    except KeyError:
        pass
    asm.show(bond, 0.0, 3.6)
    saw = asm.append("fab_test/dicing_saw_station")
    asm.place(saw.root, (X_SAW, Y_SAW, 0))
    try:
        saw.variant("hood_glass", False)
    except KeyError:
        pass
    asm.show(saw, 0.0, 3.6)
    tape = asm.append("fab_test/wafer_tape_frame")
    tape.root.parent = saw.hook("wafer_slot")
    tape.root.matrix_parent_inverse.identity()
    tape.root.location = (0, 0, 0)
    asm.show(tape, 0.0, 3.6)
    close = asm.append("fab_test/eic_pic_bonding_close")
    asm.show(close, 1.9, 2.8)
    st.update(fau=fau, bond=bond, saw=saw, tape=tape, close=close)

    # ---- oven cutaway: lids of zones 7-9 hidden, trench cut into the body front/top over x 0.17..1.33 (zones 7-9) to show belt + board
    body = [o for o in oven.objs if o.name == "reflow_oven_line_body"][0]
    cutter = L.box("oven_cutter", (X_OVEN + 0.75, Y_OVEN + 0.0, 1.0), (1.16, 1.12, 0.40), (0.1, 0.1, 0.1), rough=0.9)
    cutter.hide_render = True
    cutter.display_type = "WIRE"
    # interior material of the trench: dark steel with a faint heat glow
    cm_ = bpy.data.materials.new("MAT_s05_oven_inside")
    cm_.use_nodes = True
    nt = cm_.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.10, 0.09, 0.09, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.35, 0.08, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.35
    cutter.data.materials.clear()
    cutter.data.materials.append(cm_)
    mod = body.modifiers.new("S05_CUTAWAY", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    mod.solver = "EXACT"
    try:
        mod.material_mode = "TRANSFER"
    except Exception:
        pass
    st["cut_objs"] = []
    for z in (7, 8, 9):
        for o in oven.objs:
            if o.name.startswith("reflow_oven_line_zone_%02d_" % z):
                st["cut_objs"].append(o)
    # heater elements visible in the trench: two emissive bars along X at the back wall and below the belt
    L.box("oven_heater_back", (X_OVEN + 0.75, Y_OVEN + 0.36, 0.99), (1.12, 0.02, 0.05), (1.0, 0.4, 0.1), emit=1.3)
    L.box("oven_heater_floor", (X_OVEN + 0.75, Y_OVEN - 0.02, 0.83), (1.12, 0.05, 0.015), (1.0, 0.4, 0.1), emit=1.1)

    # ---- reverse-order motion (we run the line backwards), timed to the dwell windows
    asm.key_prop(oven.root, "p_board_x", 0.3, 5000.0)
    asm.key_prop(oven.root, "p_board_x", 1.25, 3300.0)
    asm.key_prop(oven.root, "p_board_x", 1.9, 0.0)
    asm.key_prop(oven.root, "p_glow", 0.0, 1.0)
    asm.key_prop(fau.root, "p_fau_x", 1.4, 0.0)
    asm.key_prop(fau.root, "p_fau_x", 2.0, -30.0)
    asm.key_prop(fau.root, "p_dispense", 1.4, 40.0)
    asm.key_prop(fau.root, "p_dispense", 1.75, 0.0)
    asm.key_prop(fau.root, "p_uv", 1.4, 1.0)
    asm.key_prop(fau.root, "p_uv", 1.8, 0.0)
    asm.key_prop(bond.root, "p_head_x", 2.4, 0.0)
    asm.key_prop(bond.root, "p_head_x", 2.9, -480.0)
    asm.key_prop(bond.root, "p_head_z", 2.05, 45.0)
    asm.key_prop(bond.root, "p_head_z", 2.4, 0.0)
    asm.key_prop(bond.root, "p_heat", 2.05, 1.0)
    asm.key_prop(bond.root, "p_heat", 2.6, 0.0)
    asm.key_prop(close.root, "p_gap", 2.05, 0.0)
    asm.key_prop(close.root, "p_gap", 2.5, 3.0)
    asm.key_prop(close.root, "p_heat", 2.05, 1.0)
    asm.key_prop(close.root, "p_heat", 2.5, 0.0)
    asm.key_prop(saw.root, "p_spin", 2.4, 0.0)
    asm.key_prop(saw.root, "p_spin", 3.4, 360.0 * 40)
    for k, (t, xmm) in enumerate([(2.4, -150.0), (2.7, 150.0), (2.95, -150.0), (3.2, 150.0), (3.45, -150.0)]):
        asm.key_prop(saw.root, "p_table_x", t, xmm)
    # reverse of dicing: diced tiles join back into the whole wafer on tape as the die passes (2.95 s)
    whole = find_coll(tape.coll, "VARIANT_whole")
    diced = find_coll(tape.coll, "VARIANT_diced")
    for o in asm._all_objs(diced):
        L.VA(o, 1, F(2.95))
    for o in asm._all_objs(whole):
        L.VA(o, F(2.95), 10 ** 6)
    # macro inset of the bonding close-up hovering above-right of the bonder chuck
    close.root.location = (X_BOND + 0.62, Y_BOND - 0.15, 1.72)
    close.root.scale = (4.0,) * 3
    close.root.rotation_euler = (0.0, 0.0, 0.0)
    close.root.parent = None
    return st


def finish_cutaway(st):
    """Called after collection windows are converted: lids of zones 7-9 never visible."""
    for o in st["cut_objs"]:
        L._VIS[o] = [(10 ** 6, 10 ** 6 + 1)]


# ------------------------------------------------------------------ CM300-style station: wafer-level test (v1.2)
# Per wafer w (t0 = 3.3 + 0.5 w, the wafer_slide SFX cues): slides in over the deck from the front-right (t0-0.08 .. t0+0.12),
# lands on the (lifted) chuck, 8 stage hops one per frame from t0+0.17 (probe_tick cues), stage returns at t0+0.437
# (stage_clunk cue), slides out to the front-left (t0+0.44 .. t0+0.62) while the next wafer comes in. The drawer stays closed.
W_PERIOD = 0.5
HOPS = 8
W_IN = (0.56, -0.40, WAFER_LIFT + 0.03)        # slot frame (m): entry point, front-right above the deck
W_OUT = (-0.56, -0.42, WAFER_LIFT + 0.04)      # exit point, front-left
DIE_STATE = {"idle": (0.30, 0.32, 0.36, 0.0), "probe": (0.85, 0.85, 0.75, 0.12),
             "pass": (0.10, 0.90, 0.25, 0.35), "fail": (0.80, 0.08, 0.08, 0.12)}   # 4th = glow (library values 0.6-1.0 bloomed)


def wafer_times(w):
    t0 = 3.3 + W_PERIOD * w
    return dict(t0=t0, in0=t0 - 0.08, in1=t0 + 0.12, p0=t0 + 0.17, p1=t0 + 0.17 + HOPS / 30.0, out0=t0 + 0.44, out1=t0 + 0.62)


def bake_slide(obj, t_a, t_b, p_a, p_b, rz_a, rz_b, ease_out):
    fa, fb = F(t_a), F(t_b)
    n = max(fb - fa, 1)
    for i in range(n + 1):
        u = i / n
        e = 1.0 - (1.0 - u) ** 3 if ease_out else u ** 2.2
        pos = lerp3(p_a, p_b, e)
        tilt = 0.12 * (1.0 - e) if ease_out else 0.12 * e
        kframe(obj, fa + i, loc=pos, rot=(tilt, 0.0, rz_a + (rz_b - rz_a) * e))


def build_cm(st):
    cm = asm.append("fab_test/probe_station_cm300_style")
    asm.place(cm.root, (X_CM, Y_CM, 0))
    asm.show(cm, 2.2, 10.0)
    st["cm"] = cm
    # v1.2: the four positioners and the microscope barrel sit in the plane of the lifted wafer: hidden (see devlog)
    st["cm_hidden"] = [o for o in cm.objs if any(k in o.name for k in ("_pos1_", "_pos2_", "_pos3_", "_pos4_", "scope_barrel", "scope_lens",
                                                                       "scope_ring_light", "scope_flange"))]
    asm.key_prop(cm.root, "p_drawer", 0.0, 0.0, "CONSTANT")
    pc = asm.append("fab_test/probe_card")
    try:
        pc.variant("full_needles", False)
    except KeyError:
        pass
    bpy.context.view_layer.update()
    probe = world_of(cm.hook("probe_center")).translation + Vector((0, 0, WAFER_LIFT))
    pc.root.location = probe + Vector((0, 0, 0.02))
    pc.root.scale = (PROBE_CARD_S,) * 3
    asm.show(pc, 3.0, 5.0)
    st["pc"] = pc

    base = asm.append("fab_test/wafer_300mm_siph")
    wafers = [base, asm.clone(base, "wafer_siph_2"), asm.clone(base, "wafer_siph_3")]
    slot = cm.hook("wafer_slot")
    hop_frames = []
    for w, a in enumerate(wafers):
        tt = wafer_times(w)
        a.root.parent = slot
        a.root.matrix_parent_inverse.identity()
        asm.show(a, tt["in0"] - 0.04, tt["out1"] + 0.04)
        spin = 1.4 * (-1) ** w
        bake_slide(a.root, tt["in0"], tt["in1"], W_IN, (0.0, 0.0, WAFER_LIFT), spin, 0.0, True)
        kframe(a.root, F(tt["out0"]), loc=(0.0, 0.0, WAFER_LIFT), rot=(0.0, 0.0, 0.0))
        bake_slide(a.root, tt["out0"], tt["out1"], (0.0, 0.0, WAFER_LIFT), W_OUT, 0.0, -spin, False)
        linear_all(a.root)
        # die map: serpentine bands coloured as the stage steps; exactly 1 in 10 dies pass per wafer
        dies = sorted([o for o in a.objs if "die_probe_order" in o], key=lambda o: o["die_probe_order"])
        green = D.pass_set_exact_tenth(dies, seed=7 + 13 * w)
        is_green = {o.name: (i in green) for i, o in enumerate(dies)}
        print("WAFER", w, "dies", len(dies), "green", len(green))
        rows = sorted({o["die_row"] for o in dies})
        nb = HOPS // 2
        band = {r: min(nb - 1, int(nb * i / len(rows))) for i, r in enumerate(rows)}
        groups = [[] for _ in range(HOPS)]
        for o in dies:
            b = band[o["die_row"]]
            left = o["die_x_mm"] < 0.0
            first = left if b % 2 == 0 else (not left)
            groups[2 * b + (0 if first else 1)].append(o)
        for o in dies:
            o.color = DIE_STATE["idle"]
            o.keyframe_insert("color", frame=F(tt["in0"] - 0.05))
        stage_xy = []
        for g, grp_dies in enumerate(groups):
            f = F(tt["p0"]) + g
            hop_frames.append(f)
            cx = sum(o["die_x_mm"] for o in grp_dies) / max(len(grp_dies), 1)
            cy = sum(o["die_y_mm"] for o in grp_dies) / max(len(grp_dies), 1)
            stage_xy.append((-0.7 * cx, -0.7 * cy))
            for o in grp_dies:
                o.color = DIE_STATE["probe"]
                o.keyframe_insert("color", frame=f)
                o.color = DIE_STATE["pass"] if is_green[o.name] else DIE_STATE["fail"]
                o.keyframe_insert("color", frame=f + 1)
        constant_fcurves_for(dies)
        asm.key_prop(cm.root, "p_stage_x", tt["in0"], 0.0, "CONSTANT")
        asm.key_prop(cm.root, "p_stage_y", tt["in0"], 0.0, "CONSTANT")
        for g in range(HOPS):
            t = tt["p0"] + g / 30.0
            asm.key_prop(cm.root, "p_stage_x", t, max(-100.0, min(100.0, stage_xy[g][0])), "CONSTANT")
            asm.key_prop(cm.root, "p_stage_y", t, max(-120.0, min(120.0, stage_xy[g][1])), "CONSTANT")
        asm.key_prop(cm.root, "p_stage_x", tt["p1"], 0.0, "CONSTANT")
        asm.key_prop(cm.root, "p_stage_y", tt["p1"], 0.0, "CONSTANT")
        # probe card: touches down for the hops, lifts 20 mm while wafers are exchanged
        asm.key_loc(pc.root, tt["p0"] - 2 / 30.0, tuple(probe + Vector((0, 0, 0.02))), interp="LINEAR")
        asm.key_loc(pc.root, tt["p0"], tuple(probe), interp="LINEAR")
        asm.key_loc(pc.root, tt["p1"], tuple(probe), interp="LINEAR")
        asm.key_loc(pc.root, tt["p1"] + 2 / 30.0, tuple(probe + Vector((0, 0, 0.02))), interp="LINEAR")
    st["wafers"] = wafers
    st["hop_frames"] = hop_frames
    return cm, probe


def constant_fcurves_for(objs):
    for o in objs:
        constant_fcurves(o)


# ------------------------------------------------------------------ monitor: 10 rings measured one per 0.2 s, die strip with 1 green
MON_T0, MON_T1 = 4.8, 8.2


def build_spectrum(cm):
    f0 = F(MON_T0)
    nsp = F(MON_T1) - f0
    sdir = os.path.join(TEX, "mon_s5")
    s05_tex.monitor_sequence(sdir, "mon_s5", nsp)
    first = os.path.join(sdir, "mon_s5_0001.png")
    scr = [o for o in cm.objs if o.name == "probe_station_cm300_style_screen"][0]
    m = scr.material_slots[0].material
    nt = m.node_tree
    img = bpy.data.images.load(first)
    img.source = "SEQUENCE"
    n = nt.nodes["IMG_screen"]
    n.image = img
    n.image_user.frame_duration = nsp
    n.image_user.frame_start = f0
    n.image_user.frame_offset = 0
    n.image_user.use_auto_refresh = True
    n.interpolation = "Closest"
    # v1.1 fix kept: the screen mesh UV has u along local +Z and v along local -X; image u' = 1 - v, v' = u (dips point down)
    tc = nt.nodes["Texture Coordinate"]
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], inv.inputs[1])
    nt.links.new(inv.outputs[0], comb.inputs["X"])
    nt.links.new(sep.outputs["X"], comb.inputs["Y"])
    nt.links.new(comb.outputs["Vector"], n.inputs["Vector"])
    mix = nt.nodes["MIX_use_image"].outputs[0]
    mix.default_value = 0.0
    mix.keyframe_insert("default_value", frame=1)
    mix.default_value = 1.0
    mix.keyframe_insert("default_value", frame=f0)
    constant_fcurves(nt)
    if "Emission" in nt.nodes:
        nt.nodes["Emission"].inputs["Strength"].default_value = 1.1
    return scr


# ------------------------------------------------------------------ human scale: Gary breaks the news at the prober, Manager shoots
GARY_POS = (X_CM + 1.35, -1.00)      # right of the prober (clear of the shelf and cabinet when he falls back)
MGR_POS = (X_CM - 0.45, -1.35)
GARY_FACE_WORK = PI                  # facing +Y (keyboard), back to the camera
GARY_FACE_NEWS = 2 * PI - 0.20       # facing the camera, 11 deg toward the Manager (hole 5 axis within ~15 deg of the view: see-through)
T_SHOT = 9.1                         # shotgun SFX cue 49.1 film
GUN_SPEED = 1.25                     # gun_raise_aim_fire: shot at action frame 44 -> start = T_SHOT - 44/30/GUN_SPEED
FALL_SPEED = 1.5                     # shot_hit_fall after the 4-frame hit-stop (ground contact at about 9.63 s)


def build_cast():
    g = asm.append("characters/gary_v2", actions=True)
    m = asm.append("characters/manager_v2", actions=True)
    asm.place(g.root, (GARY_POS[0], GARY_POS[1], 0.0), yaw=GARY_FACE_WORK)
    yaw_m = asm.yaw_to((MGR_POS[0], MGR_POS[1], 0), (GARY_POS[0], GARY_POS[1], 0))
    asm.place(m.root, (MGR_POS[0], MGR_POS[1], 0.0), yaw=yaw_m)
    for a in (g, m):
        asm.show(a, 6.6, 10.0)
    gp = (GARY_POS[0], GARY_POS[1], 0.0)
    # Gary: thinks at the monitor, turns left 180 to face the Manager and the camera (motion_v2 turn with yaw hand-off),
    # holds the printout up, is shot
    asm.key_loc(g.root, 0.0, gp, yaw=GARY_FACE_WORK, interp="LINEAR")
    M2.apply(g, "thinking", 6.6, hold=False, repeat=1.0, face=False)
    st_turn = M2.apply(g, "turn_left_180", 7.2, speed=1.4, hold=False, face=False)
    fe = int(round(st_turn.frame_end))
    # settle 11 deg back toward the Manager (eased object yaw after the hand-off)
    g.root.rotation_euler = (0, 0, 2.0 * PI)
    g.root.keyframe_insert("rotation_euler", frame=fe + 2, index=2)
    g.root.rotation_euler = (0, 0, GARY_FACE_NEWS)
    g.root.keyframe_insert("rotation_euler", frame=fe + 13, index=2)
    st_hp = asm.play(g, "hold_printout", (fe + 1 - 1) / 30.0, hold=True)
    st_hp.blend_in = 3
    M2.apply(g, "shot_hit_fall", T_SHOT, hold=False, end_frame=4, face=False, layer="fallA")
    M2.apply(g, "shot_hit_fall", T_SHOT + 4 / 30.0, speed=FALL_SPEED, start_frame=4, hold=True, face=False, layer="fallB")
    # Manager: low-ready with the gun while Gary talks, then raise, aim, fire at T_SHOT (action frame 44)
    M2.apply(m, "gun_ready", 6.6, hold=False, repeat=2.0, face=False)
    M2.apply(m, "gun_raise_aim_fire", T_SHOT - 44 / 30.0 / GUN_SPEED, speed=GUN_SPEED, hold=True, face=False)
    return g, m


def ramp(root, prop, keys):
    """keys: [(t, value, interp)]"""
    for t, v, it in keys:
        asm.key_prop(root, prop, t, v, it)


def drop_copy(pr, t0, land, name):
    """At t0 a linked copy of the printout leaves the hand pose and flutters to the floor at `land` (scene s)."""
    scn = bpy.context.scene
    scn.frame_set(F(t0))
    bpy.context.view_layer.update()
    W = pr.root.matrix_world.copy()
    scn.frame_set(1)
    cp = asm.clone(pr, name)
    cp.root.parent = None
    cp.root.constraints.clear()
    cp.root.matrix_parent_inverse.identity()
    t = L._text("1 / 10", 0.05, (0.1, 0.1, 0.1), (0, -0.118, 0.001), cp.root, "CENTER", "CENTER")
    loc, eul = W.translation, W.to_euler()
    asm.key_loc(cp.root, t0, (loc.x, loc.y, loc.z), rot=tuple(eul), interp="BEZIER")
    asm.key_loc(cp.root, t0 + 0.2, (loc.x - 0.25, loc.y - 0.2, loc.z * 0.6), rot=(eul.x + 0.6, eul.y, eul.z + 0.8), interp="BEZIER")
    asm.key_loc(cp.root, land, (loc.x - 0.45, loc.y - 0.35, 0.004), rot=(0.0, 0.0, 0.9), interp="CONSTANT")
    asm.show(cp, t0, 10.0)
    L.V(t, 0, t0, 10.0)


def build_props_and_fx(g, m):
    pr = asm.append("props/office_gags", only=["ASSET_wafer_map_printout"])
    rot_pr = Matrix(((0, 0, 1), (0, 1, 0), (-1, 0, 0))).to_euler()
    # v1.2: the sheet follows Gary's right hand and turns its face (local +Z, text below the map) to the camera
    pr.root.parent = None
    pr.root.location = (0.0, 0.0, 0.10)
    pr.root.rotation_euler = rot_pr
    c = pr.root.constraints.new("COPY_LOCATION")
    c.target = g.hook("hand_R")
    c.use_offset = True
    c = pr.root.constraints.new("TRACK_TO")
    c.target = L.CAM
    c.track_axis = "TRACK_Z"
    c.up_axis = "UP_Y"
    txt = L._text("1 / 10", 0.05, (0.1, 0.1, 0.1), (0, -0.118, 0.001), pr.root, "CENTER", "CENTER")
    for o in pr.objs:   # v1.2: paper white 1.0 bloomed under the key light; scene copy of the material toned to 0.8
        for sl in getattr(o, "material_slots", []):
            mt = sl.material
            if mt and mt.use_nodes and "Principled BSDF" in mt.node_tree.nodes:
                bc = mt.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
                if not bc.is_linked and min(bc.default_value[:3]) > 0.85:
                    bc.default_value = tuple(v * 0.8 for v in bc.default_value[:3]) + (1.0,)
    asm.show(pr, 6.6, T_SHOT)
    L.V(txt, 0, 6.6, T_SHOT)
    drop_copy(pr, T_SHOT, T_SHOT + 0.45, "wafer_map_printout_dropped")
    sg = asm.append("props/shotgun")
    for v, on in (("clay", True), ("wood", False)):
        try:
            sg.variant(v, on)
        except KeyError:
            pass
    asm.attach(sg.root, m.hook("gun_grip_R"), offset=(0, 0, 0), rot=(0, 0, PI))   # motion_v2 convention (S1/S2)
    asm.show(sg, 6.6, 10.0)
    # holes: earlier holes at their healed radii, hole 5 opens at the bang
    key_hole(g.root, "p_hole_1_radius", 9.2)
    key_hole(g.root, "p_hole_2_radius", 19.2)
    key_hole(g.root, "p_head_hole_radius", 28.95)
    key_hole(g.root, "p_hole_4_radius", 38.9)
    asm.key_prop(g.root, "p_hole_5_radius", 0.0, 0.0, "CONSTANT")
    asm.key_prop(g.root, "p_hole_5_radius", T_SHOT, 1.0, "CONSTANT")
    # faces (face actions off; keyed here)
    ramp(g.root, "p_expr_worried", [(0.0, 0.0, "LINEAR"), (7.3, 0.0, "LINEAR"), (7.7, 1.0, "LINEAR"), (T_SHOT - 1 / 30, 1.0, "CONSTANT"), (T_SHOT, 0.0, "CONSTANT")])
    ramp(g.root, "p_expr_sweating", [(0.0, 0.0, "LINEAR"), (7.6, 0.0, "LINEAR"), (8.2, 1.0, "LINEAR"), (T_SHOT - 1 / 30, 1.0, "CONSTANT"), (T_SHOT, 0.0, "CONSTANT")])
    ramp(g.root, "p_expr_dread", [(0.0, 0.0, "LINEAR"), (8.35, 0.0, "LINEAR"), (8.75, 1.0, "LINEAR"), (T_SHOT - 1 / 30, 1.0, "CONSTANT"), (T_SHOT, 0.0, "CONSTANT")])
    ramp(g.root, "p_expr_shock", [(0.0, 0.0, "CONSTANT"), (T_SHOT, 1.0, "LINEAR"), (9.4, 1.0, "LINEAR"), (9.55, 0.0, "LINEAR")])
    ramp(g.root, "p_expr_dead_eyed", [(0.0, 0.0, "CONSTANT"), (9.4, 0.0, "LINEAR"), (9.55, 1.0, "LINEAR")])
    ramp(m.root, "p_anger", [(6.6, 0.25, "LINEAR"), (7.9, 0.4, "LINEAR"), (8.5, 0.9, "LINEAR"), (T_SHOT, 1.0, "LINEAR")])
    ramp(m.root, "p_flush", [(0.0, 0.0, "LINEAR"), (7.9, 0.0, "LINEAR"), (8.6, 0.8, "LINEAR")])
    for side in ("L", "R"):
        asm.fx("muzzle_flash", T_SHOT, parent=sg.hook("muzzle_" + side), rot=(-PI / 2, 0, 0), scale=0.8, dur=0.12)
    asm.fx("smoke_ring", T_SHOT + 0.05, parent=sg.hook("muzzle_R"), rot=(-PI / 2, 0, 0), scale=0.5, dur=0.9)
    return sg, pr


# ------------------------------------------------------------------ effects: dust motes, shimmer, sparks, optical test beams, probe tip
def rod(name, p0, p1, radius, color, emit):
    a, b = Vector(p0), Vector(p1)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=radius, depth=d.length, location=(a + b) / 2)
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o.name = name
    o.data.materials.append(L.mat(color, emit=emit))
    return o


def build_effects(st, probe):
    rnd = random.Random(55)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.0045, location=(0, 0, -50))
    proto = bpy.context.active_object
    proto.data.materials.append(L.mat((0.95, 0.97, 1.0), emit=0.5))   # v1.2: 1.2 -> 0.5, 90 -> 45 motes (speckle)
    mesh = proto.data
    bpy.data.objects.remove(proto)
    for i in range(45):
        o = bpy.data.objects.new("mote_%02d" % i, mesh)
        bpy.context.scene.collection.objects.link(o)
        p0 = Vector((rnd.uniform(-4.0, 11.5), rnd.uniform(-2.4, 0.2), rnd.uniform(0.5, 2.4)))
        v = Vector((rnd.uniform(-0.08, 0.08), rnd.uniform(-0.04, 0.04), rnd.uniform(-0.03, 0.06)))
        for t in (0.0, 5.0, 10.0):
            w = 0.03 * math.sin(t * 1.3 + i)
            kframe(o, F(t), loc=tuple(p0 + v * t + Vector((w, 0, w * 0.5))))
        linear_all(o)
        L.V(o, 0, 0.0, 6.9)
    asm.fx("heat_shimmer", 0.6, loc=(X_OVEN + 0.75, Y_OVEN - 0.85, 1.12), rot=(0, 0, 0), scale=0.8, dur=0.9)
    # sparks at the bonder contact and the dicing blade (SFX cues 42.12, 42.78, 43.0 kept)
    asm.fx("sparks_burst", 2.12, loc=(X_BOND + 0.15, Y_BOND, 1.08), scale=0.25, dur=0.5, intensity=0.5)
    asm.fx("sparks_burst", 2.78, loc=(X_SAW, Y_SAW + 0.06, 1.09), scale=0.25, dur=0.5, intensity=0.5)
    asm.fx("sparks_burst", 3.0, loc=(X_SAW, Y_SAW + 0.06, 1.09), scale=0.25, dur=0.5, intensity=0.5)
    # optical test: two cyan fibers from above to the probe point (laser_zap cue 43.4); small tip flash on every stage hop
    px, py, pz = probe
    for sx in (-1, 1):
        o = rod("test_fiber_%d" % sx, (px + 0.16 * sx, py + 0.05, pz + 0.22), (px + 0.004 * sx, py, pz + 0.008), 0.0012, (0.2, 0.9, 1.0), emit=1.6)
        L.V(o, 0, 3.4, 4.8)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.005, location=(px, py, pz + 0.003))
    tip = bpy.context.active_object
    tip.name = "probe_tip_flash"
    tip.data.materials.append(L.mat((1.0, 0.95, 0.8), emit=2.0))
    L.V(tip, 0, 3.4, 4.8)
    kframe(tip, 1, scale=(0.0, 0.0, 0.0))
    for f in st["hop_frames"]:
        kframe(tip, f, scale=(1.0, 1.0, 1.0))
        kframe(tip, f + 1, scale=(0.0, 0.0, 0.0))
    constant_fcurves(tip)


# ------------------------------------------------------------------ cameras, labels, captions
def camera_stops(probe, scr):
    tab = dwell_table()
    stops = []  # (t, cam, tgt, lens)
    # 0.0-0.5 calm (transition T4 window): engine shot with a slow push; the die is yanked out of frame at 0.3
    stops.append((0.0, (X_ENG, -4.6, 1.5), (X_ENG, 0.0, 0.85), 28.0))
    stops.append((0.5, (X_ENG + 0.05, -4.35, 1.47), (X_ENG + 0.05, 0.0, 0.85), 28.0))
    for k in ("oven", "fau", "bond", "saw"):
        d = tab[k]
        w = Vector(d["w"])
        c = w + Vector(d["cam"])
        c2 = w + Vector(d["cam"]) * 0.9 + Vector((0.07, 0.0, -0.02))
        ta, tl = DW[k]
        stops.append((ta, tuple(c), tuple(w), d["lens"]))
        stops.append((tl, tuple(c2), tuple(w), d["lens"]))
    P = Vector(probe)
    # prober: front-above, wafers enter from the front-right and leave front-left
    stops.append((3.4, tuple(P + Vector((0.08, -0.80, 1.10))), tuple(P + Vector((0.0, -0.24, 0.0))), 32.0))
    stops.append((4.8, tuple(P + Vector((0.06, -0.74, 1.02))), tuple(P + Vector((0.0, -0.24, 0.0))), 32.0))
    bpy.context.view_layer.update()
    vs = [scr.matrix_world @ v.co for v in scr.data.vertices]
    sc = sum(vs, Vector((0, 0, 0))) / len(vs)
    stops.append((5.4, tuple(sc + Vector((0.22, -0.295, 0.02))), tuple(sc + Vector((0.0, 0.0, -0.035))), 22.0))
    stops.append((7.0, tuple(sc + Vector((0.20, -0.27, 0.02))), tuple(sc + Vector((0.0, 0.0, -0.035))), 22.0))
    stops.append((7.35, (sc.x - 0.05, sc.y - 0.95, sc.z + 0.55), (sc.x - 0.10, sc.y - 0.4, sc.z - 0.2), 24.0))
    # two-shot: Manager left, Gary right, about 45 percent of frame height, feet and floor in frame
    mx = (MGR_POS[0] + GARY_POS[0]) / 2.0
    stops.append((7.9, (mx + 0.10, -4.35, 1.45), (mx + 0.10, -1.0, 1.02), 30.0))
    stops.append((9.08, (mx + 0.15, -4.10, 1.45), (mx + 0.15, -1.0, 1.02), 30.0))
    # after the hit-stop: crane up and right onto Gary on the floor (hole 5 seen from above), calm from 9.55
    stops.append((9.16, (mx + 0.15, -4.10, 1.45), (mx + 0.15, -1.0, 1.02), 30.0))
    gx, gy = GARY_POS
    # near-overhead on the lying body: hole 5 axis is vertical, the floor shows through it (no floating body)
    stops.append((9.55, (gx + 0.16, gy + 1.28, 2.62), (gx + 0.26, gy + 1.84, 0.13), 30.0))
    stops.append((10.0, (gx + 0.17, gy + 1.33, 2.55), (gx + 0.26, gy + 1.84, 0.13), 30.0))
    return stops, sc


def build_cameras_and_text(probe, scr, sg):
    stops, sc = camera_stops(probe, scr)
    cam, tgt = L.CAM, L.TGT
    k = 0.4
    L.HOLD.location = (0, 0, -L.HOLD_Z * k)
    for f in range(1, F(T_END) + 1):
        t = (f - 1) / 30.0
        cpos = eval_stops([(s[0], s[1]) for s in stops], t)
        tpos = eval_stops([(s[0], s[2]) for s in stops], t)
        lens = eval_stops([(s[0], (s[3],)) for s in stops], t)[0]
        kframe(cam, f, loc=cpos)
        kframe(tgt, f, loc=tpos)
        cam.data.lens = lens
        cam.data.keyframe_insert("lens", frame=f)
        L.HOLD.scale = (L.HOLD_S0 * L.LENS0 / lens * k,) * 3
        L.HOLD.keyframe_insert("scale", frame=f)
    linear_all(cam, ("location",))
    linear_all(tgt, ("location",))
    for obj in (cam.data, L.HOLD):
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    add_noise(cam, 0.006, 11.0, seed=1)
    add_noise(tgt, 0.005, 13.0, seed=2)
    print("SCREEN_WORLD", tuple(round(v, 3) for v in sc), "PROBE", tuple(round(v, 3) for v in probe))

    # world labels: one per station, above the work point (upper third, clear of the subtitles); v1.2 sizes about 0.7x
    tab = dwell_table()
    for key, body, t0, t1 in (("oven", "REFLOW ON SUBSTRATE", 0.95, 1.25), ("fau", "FAU ATTACH", 1.5, 1.85),
                              ("bond", "EIC / PIC BONDING", 2.15, 2.5), ("saw", "DICING + TAPE", 2.8, 3.1)):
        w = Vector(tab[key]["w"])
        asm.wl(body, tuple(w + Vector((0.0, 0.05, 0.42))), t0, t1, size=0.045)
    P = Vector(probe)
    asm.wl("RING TRANSMISSION (LORENTZIAN DIPS)", (sc.x - 0.04, sc.y - 0.02, sc.z + 0.17), 5.3, 7.0, size=0.015)
    # captions, narration (narration.json), cards, fx notes
    asm.narr_vo(5)
    asm.lab(0.3, 3.3, "BACK THROUGH THE LINE")
    asm.lab(3.3, 4.8, "WAFER-LEVEL TEST (CM300-STYLE PROBER)")
    asm.lab(4.8, 7.0, "RING RESONANCES vs SPEC")
    asm.big(7.2, 7.85, "1 IN 10")
    # BANG as a world label above the muzzle (the BIG overlay sat on Gary's torso and hid hole 5)
    scn = bpy.context.scene
    scn.frame_set(F(T_SHOT))
    bpy.context.view_layer.update()
    mz = sg.hook("muzzle_R").matrix_world.translation.copy()
    scn.frame_set(1)
    asm.wl("BANG", (mz.x + 0.15, mz.y, mz.z + 0.42), T_SHOT, T_SHOT + 0.2, size=0.30, color=(1.0, 0.9, 0.25))   # gone before the crane (9.16)
    asm.card(4.8, 7.0, "Illustrative Lorentzian ring transmission; 1 of 10 rings inside the spec window (by construction)")
    asm.fxn(0.3, 3.3, "[FX: motion-blur shuffle back through the line, eased dwell at each tool]")
    asm.fxn(3.3, 4.8, "[FX: wafers slide in and out; stage steps; probe touchdown; die map]")
    asm.fxn(4.8, 7.0, "[FX: 10 rings measured, 1 passes, ping]")
    asm.fxn(T_SHOT, 9.7, "[FX: muzzle flash, smoke ring, hole 5]")
    asm.timecode(5)


def finish_hidden(st):
    for o in st.get("cm_hidden", []):
        L._VIS[o] = [(10 ** 6, 10 ** 6 + 1)]


def main():
    scn = asm.new_scene(preset="standard", world="WORLD_lab")
    L.CAM.data.clip_start = 0.1   # dies sit 20 um above the wafer: keep the depth range tight
    L.CAM.data.clip_end = 300.0
    scn.render.use_motion_blur = True            # motion blur on (shutter 0.5 frames); render_scene.py keeps the scene value
    scn.render.motion_blur_shutter = MOTION_SHUTTER
    build_env()
    lighting()
    eng, pic, eic, grp = build_hero()
    animate_hero(grp, pic, eic)
    st = build_line(grp)
    cm, probe = build_cm(st)
    scr = build_spectrum(cm)
    g, m = build_cast()
    sg, pr = build_props_and_fx(g, m)
    build_effects(st, probe)
    build_cameras_and_text(probe, scr, sg)
    collection_windows_to_objects()
    finish_cutaway(st)
    finish_hidden(st)
    asm.finalize(OUT)


if __name__ == "__main__":
    main()
