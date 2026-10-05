"""Scene S1 'Copper: the stretch' (film 0-10 s), assembled from the v1 asset library (v1.2: v2 characters, motion_v2).

Run: FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s01_copper.py -- scenes/v1/s01_copper.blend
Optional extra args after the output path: --no-eye (skip regenerating the eye PNG sequence if it already exists).

World frame (1 unit = 1 m, real size): x along the hall (rack pair on the left, bench on the right), +y away from the
camera (cameras sit near y = -4), z up. Everything of the action sits on the line y = Y_ACT in front of the cartridge wall.
Storyboard: DevLog/DevLog-001 Section 4 (S1), narration Section 5; crude reference: scripts/build_crude_film.py s1.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asm  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import tex_gen as T  # noqa: E402  (scripts/ is on sys.path via asm)
import s01_cam_fx as FX  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "assets", "characters", "motion_v2"))
import motion_v2 as M2  # noqa: E402

PI = math.pi
TAU = 2.0 * math.pi
F = asm.F
PROJ = asm.PROJ
COMP = asm.COMP
TEX = os.path.join(PROJ, "assets", "generated_textures", "v1", "s01")
LIB = os.path.join(COMP, "materials_vfx", "materials_vfx_library.blend")
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = ARGS[0] if ARGS else os.path.join(PROJ, "scenes", "v1", "s01_copper.blend")

# ------------------------------------------------------------------ layout (metres)
Y_ACT = 2.7            # centre line of racks / bench / people
X_A = 0.0              # rack A (fixed); rack B at X_A + gap
Y_WALL = 4.0           # cartridge wall front plane (hall wall at y = 4.25)
GAP_KEYS = [(0.0, 5.0), (2.2, 5.0), (3.0, 2.0), (3.6, 1.0)]   # constant jumps (storyboard times)
X_BENCH = 8.0          # bench centre
X_MGR_STARE = 8.45     # manager stands here beside the scope to read it (v1.1: walks all the way to the bench)
X_MGR_START = 11.4     # manager enters from the hall end (walk starts 4.1 s so he arrives at about 6.7 s)
HIT_STOP = 3           # frames of hit-stop at the bang
FAN_SLACK, FAN_TAUT = 9.0, 5.0   # mid-span fan multiplier of the 14-cable bundle (cross-section pitch x this)
FAN_END = 2.6          # fan multiplier at the end blocks
CABLE_THICK = 2.0      # visual exaggeration of the cable radius (asset: 3.75 mm radius)
T_BANG = 9.2

# ------------------------------------------------------------------ small helpers


def ramp(t, ta, tb, va, vb):
    if t <= ta:
        return va
    if t >= tb:
        return vb
    return va + (vb - va) * (t - ta) / (tb - ta)


def lib_mats(names):
    with bpy.data.libraries.load(LIB, link=False) as (src, dst):
        dst.materials = [n for n in names if n in src.materials]
    return {m.name: m for m in dst.materials}


def wpos(obj):
    return obj.matrix_world.translation.copy()


def update():
    bpy.context.view_layer.update()


def hide_coll(c, hidden=True):
    c.hide_render = hidden
    c.hide_viewport = hidden


def set_interp(obj, path, interp, index=None):
    ad = obj.animation_data
    if not (ad and ad.action):
        return
    for fc in ad.action.fcurves:
        if fc.data_path == path and (index is None or fc.array_index == index):
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def clone_objs(objs, name):
    """Linked duplicate of an object subtree (shares mesh data); returns (collection, copy of the first top object)."""
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    mp = {}
    for o in objs:
        c = o.copy()
        coll.objects.link(c)
        mp[o] = c
    top = None
    for o, c in mp.items():
        if o.parent in mp:
            c.parent = mp[o.parent]
            c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
        else:
            c.parent = None
            top = c
    return coll, top


# ------------------------------------------------------------------ scene
scn = asm.new_scene(preset="standard", world="WORLD_data_hall")
asm.timecode(1)

# ---- data hall (shell only: racks, containment and cable trays hidden so the staging area is free)
hall = asm.append("datacenter/datahall_environment")
for ch in hall.coll.children:
    if ch.name.split(".")[0] in ("RACKS", "CONTAINMENT", "OVERHEAD"):
        hide_coll(ch)
hall.root.scale = (1.0, 1.0, 1.28)     # ceiling 3.6 -> 4.6 m so the 4.1 m cartridge wall fits (see devlog)

# ---- lighting: data_hall rig re-laid as strips along the hall (rig built for a 6 m aisle)
lr = asm.rig("data_hall", scale=2.2)
lr.root.rotation_euler = (0, 0, PI / 2)
lr.root.location = (1.5, Y_ACT - 1.0, 0)
for o in lr.objs:
    if o.type == "LIGHT" and "strip" in o.name:
        o.location.z = 4.2 / 2.2
update()

# front fill so the dark wall, racks and people read (added; the rig strips only light the ceiling zone)
for nm, loc, size, watt in (("fill_wall", (4.0, -1.5, 3.4), (22.0, 3.0), 400.0), ("fill_action", (6.0, -3.5, 3.0), (14.0, 3.0), 260.0)):
    ld = bpy.data.lights.new(nm, "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = size
    ld.energy = watt
    ld.color = (1.0, 0.94, 0.86)      # v1.2: warm clay key (was cool white)
    lo = bpy.data.objects.new(nm, ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = loc
    tgt = Vector((loc[0], Y_ACT + 1.2, 1.6))
    lo.rotation_euler = (tgt - Vector(loc)).to_track_quat("-Z", "Y").to_euler()

# ---- cartridge wall (12x5 variant) + 14-cable bundle, from one append
wall = asm.append("interconnect/nvl72_copper_cartridge_wall")
wall.variant("cartridge_detail", False)
wall.variant("wall_12x5", True)
wall.variant("bundle_14", True)
wall_src = [c for c in wall.coll.children if c.name.startswith("VARIANT_wall_12x5")][0]
wall_objs = asm._all_objs(wall_src)
WALL_W = 1.92
N_WALL = 9
WALL_X0 = -9.0 + WALL_W / 2
wall.root.location = (0, 0, 0)
wall_root_src = [o for o in wall_objs if o.name == "wall_root"][0]
wall_root_src.location = (WALL_X0, Y_WALL, 0)
for k in range(1, N_WALL):
    coll, top = clone_objs(wall_objs, "WALL_copy_%d" % k)
    top.location = (WALL_X0 + k * WALL_W, Y_WALL, 0)

# ---- crude-style copper conduit rows in front of the wall (Pulse runs on the z = 1.4 one); not part of the asset
mats = lib_mats(["MAT_vfx_copper", "MAT_vfx_clay_shirt_yellow", "MAT_vfx_clay_eye_white", "MAT_vfx_clay_pupil_black"])
X_CON0, X_CON1 = -9.0, N_WALL * WALL_W - 9.0 + 0.0
for row, z in enumerate((0.4, 1.0, 1.4, 2.2, 3.0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.035, depth=X_CON1 - X_CON0,
                                        location=((X_CON0 + X_CON1) / 2, Y_WALL - 0.22, z), rotation=(0, PI / 2, 0))
    c = bpy.context.object
    c.name = "wall_conduit_%d" % row
    c.data.materials.append(mats["MAT_vfx_copper"])
    for p in c.data.polygons:
        p.use_smooth = True
PULSE_Z = 1.4
PULSE_Y = Y_WALL - 0.22

# ---- rack pair (A fixed at X_A, B at X_A + gap). v1.2: p_gap baked per frame: constant jumps at the storyboard cuts, then
# the heave-by-heave stretch 4.4-6.0 driven by Gary's hands on a grip bar (see haul())
racks = asm.append("datacenter/rack_pair_for_cable_gag")
asm.show(racks, 0.0, 10.0)
CUTS = [2.2, 3.0, 3.6, 4.2, 6.0, 6.4]          # hard cuts (camera and the gap jumps); motion-blur shutter 0 on these frames
CUT_FRAMES = [F(t) for t in CUTS]
STRETCH_T0, STRETCH_T1 = 4.4, 6.0
N_HEAVE = 3
P_CYC = (STRETCH_T1 - STRETCH_T0) / (N_HEAVE - 0.5)       # 0.64 s per pull_cable cycle (heave + reach)
K_G = 0.9657                    # gary_v2 motion_v2 K (manifest)
H0, DH = 0.52 * K_G, 0.36 * K_G  # pull_cable: hands 0.52 K ahead at the reach, stroke 0.36 K (m2_acts.b_pull_cable)
HANDLE_DX = 0.33                # grip bar beyond rack B's centre (rack half width 0.30 + 0.03 stand-off)
HANDLE_Z = 1.0 * K_G            # pull_cable hand height
GAP0 = 1.0
R0 = GAP0 + HANDLE_DX + H0      # Gary root x at the first reach


def haul(t):
    """(gap, gary_root_x, a, hop) of the stretch. Heave: Gary's root is planted and rack B follows his hands; reach: the
    rack stays and Gary scoots back with a small hop. a = pull_cable phase (0 reach, 1 heave peak)."""
    if t >= STRETCH_T1:
        g = GAP0 + N_HEAVE * DH
        return g, g + HANDLE_DX + H0 - DH, 1.0, 0.0
    u = max(0.0, (t - STRETCH_T0) / P_CYC)
    k = int(math.floor(u))
    ph = u - k
    a = 0.5 - 0.5 * math.cos(TAU * ph)
    if ph < 0.5:
        return GAP0 + k * DH + DH * a, R0 + k * DH, a, 0.0
    return GAP0 + (k + 1) * DH, R0 + (k + 1) * DH - DH * a, a, 0.05 * math.sin(math.pi * (ph - 0.5) / 0.5)


def gap_at(t):
    g = 5.0
    for tk, gv in GAP_KEYS:
        if t >= tk - 1e-6:
            g = gv
    if t >= STRETCH_T0 - 1e-6:
        g = haul(t)[0]
    return g


for _f in range(1, 301):
    racks.root["p_gap"] = float(gap_at((_f - 1) / 30.0))
    racks.root.keyframe_insert('["p_gap"]', frame=_f)
FX.bake_obj(racks.root, lambda t: (X_A + gap_at(t) / 2.0, Y_ACT, 0.0, 0.0), jumps=CUT_FRAMES, rot=False)
for fc in racks.root.animation_data.action.fcurves:
    if fc.data_path == '["p_gap"]':
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if int(round(kp.co[0])) + 1 in CUT_FRAMES else "LINEAR"
racks.root.update_tag()
update()

# grip bar on rack B's outer face (Gary hauls on it); keyed with the rack
_hm = lib_mats(["MAT_vfx_clay_shirt_yellow"])
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.02, depth=0.36, location=(0, 0, 0), rotation=(PI / 2, 0, 0))
handle = bpy.context.object
handle.name = "s01_rack_grip_bar"
handle.data.materials.append(list(_hm.values())[0] if _hm else bpy.data.materials.new("grip"))
for _sy in (-0.16, 0.16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.012, depth=0.05, location=(-0.025, _sy, 0), rotation=(0, PI / 2, 0))
    _po = bpy.context.object
    _po.name = "s01_rack_grip_post"
    _po.data.materials.append(handle.data.materials[0])
    _po.parent = handle
for _o in [handle] + list(handle.children):
    for _p in _o.data.polygons:
        _p.use_smooth = True

# ---- bundle end hooks follow the rack port hooks (world-space drivers; hook rotation is ignored by the bundle)
for nm, port in (("HOOK_bundle_a", "HOOK_port_A"), ("HOOK_bundle_b", "HOOK_port_B")):
    h = wall.hook(nm.replace("HOOK_", ""))
    target = racks.hook(port.replace("HOOK_", ""))
    for idx, ch in enumerate(("LOC_X", "LOC_Y", "LOC_Z")):
        fc = h.driver_add("location", idx)
        d = fc.driver
        d.type = "AVERAGE"
        v = d.variables.new()
        v.name = "p"
        v.type = "TRANSFORMS"
        v.targets[0].id = target
        v.targets[0].transform_type = ch
        v.targets[0].transform_space = "WORLD_SPACE"
# bundle visibility window: only while the racks exist (VARIANT_bundle_14 is a nested collection of the wall asset)
bundle_coll = [c for c in wall.coll.children if c.name.startswith("VARIANT_bundle_14")][0]
asm.show(bundle_coll, 0.0, 10.0)
# sag: loose at 5 / 2 / 1 m, pulled taut by the stretch
for t, s in ((0.0, 0.25), (2.2, 0.25), (3.0, 0.16), (3.6, 0.16), (4.4, 0.16)):
    asm.key_prop(wall.root, "p_sag_m", t, s, "CONSTANT")
for _f in range(F(4.4) + 1, F(6.0) + 1):          # v1.2: pulled taut heave by heave, with a creaking tremble
    _t = (_f - 1) / 30.0
    wall.root["p_sag_m"] = max(0.0, 0.16 * (1.0 - (haul(_t)[0] - GAP0) / (N_HEAVE * DH)) + 0.010 * math.sin(TAU * 9.0 * _t))
    wall.root.keyframe_insert('["p_sag_m"]', frame=_f)

# ---- v1.1: fan the 14 cables out (asset: all cables converge to one point and read as a single thin wire)
import re  # noqa: E402

cables = sorted([o for o in wall.objs if o.name.startswith("bundle14_cable_")], key=lambda o: o.name)
TS = [0.0, None, 0.2, 0.35, 0.5, 0.65, 0.8, None, 1.0]
_pitch = 7.5 * 1.02
OFFS = []
for _row, _cnt in enumerate((4, 5, 5)):
    for _i in range(_cnt):
        OFFS.append(((_i - (_cnt - 1) / 2) * _pitch, (_row - 1) * _pitch * 0.87))
for t, v, ip in ((0.0, FAN_SLACK, "CONSTANT"), (4.4, FAN_SLACK, "LINEAR"), (6.0, FAN_TAUT, "LINEAR")):
    asm.key_prop(wall.root, "p_fan", t, v, ip)
for ci, cab in enumerate(cables):
    cu = cab.data
    cu.bevel_depth *= CABLE_THICK
    cu.bevel_resolution = 4
    oy, oz = OFFS[ci]
    for fc in cu.animation_data.drivers:
        m = re.match(r"splines\[0\]\.points\[(\d+)\]\.co", fc.data_path)
        pi_, k = int(m.group(1)), fc.array_index
        tt = TS[pi_]
        d = fc.driver
        vf = d.variables.new()
        vf.name = "f"
        vf.type = "SINGLE_PROP"
        vf.targets[0].id = wall.root
        vf.targets[0].data_path = '["p_fan"]'
        off = (0.0, oy * 0.001, oz * 0.001)[k]
        w = 1.0 + 0.3 * math.sin(ci * 2.1)     # per-cable sag variation so the bundle opens up
        if tt is None:
            base = "a+0.08" if pi_ == 1 else "b-0.08"
            d.expression = "%s+%r" % (base, off * FAN_END)
        else:
            mm = "(%r+(f-%r)*%r)" % (FAN_END, FAN_END, math.sin(math.pi * tt))
            sag = "-4*s*%r*%r" % (tt * (1 - tt), w) if k == 2 else ""
            d.expression = "a+%r*(b-a)+%r*%s%s" % (tt, off, mm, sag)
for nm in ("bundle14_end_block_a", "bundle14_end_block_b"):
    bpy.data.objects[nm].scale = (1.0, FAN_END, FAN_END)
# stylised jacket colours (three tones) so the individual cables read against the dark wall
JACKET = [(0.55, 0.57, 0.60), (0.20, 0.21, 0.23), (0.85, 0.45, 0.12)]
for ci, cab in enumerate(cables[1:], start=1):
    mj = bpy.data.materials.new("s01_jacket_%d" % ci)
    mj.use_nodes = True
    bsdf = mj.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*JACKET[ci % 3], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    cab.data.materials.clear()
    cab.data.materials.append(mj)


def tunnel_material(name, t_opaque0, t_opaque1, alpha0=0.28):
    """Cable 00 jacket: translucent shell seen from outside (electron pockets visible), glowing copper ring tunnel from
    inside (back faces); the shell becomes opaque between t_opaque0 and t_opaque1."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.surface_render_method = "BLENDED"
    try:
        m.use_transparency_overlap = False
    except Exception:
        pass
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mixb = nt.nodes.new("ShaderNodeMixShader")      # fac = backfacing: front shell / inside tunnel
    mixa = nt.nodes.new("ShaderNodeMixShader")      # fac = alpha: transparent / shell
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    shell = nt.nodes.new("ShaderNodeBsdfPrincipled")
    shell.inputs["Base Color"].default_value = (0.05, 0.05, 0.06, 1.0)
    shell.inputs["Roughness"].default_value = 0.35
    em = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 90.0
    wave.inputs["Distortion"].default_value = 0.0
    ramp_n = nt.nodes.new("ShaderNodeValToRGB")
    ramp_n.color_ramp.elements[0].position = 0.62
    ramp_n.color_ramp.elements[0].color = (0.004, 0.002, 0.002, 1.0)
    ramp_n.color_ramp.elements[1].position = 0.75
    ramp_n.color_ramp.elements[1].color = (1.0, 0.42, 0.12, 1.0)
    nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], ramp_n.inputs["Fac"])
    nt.links.new(ramp_n.outputs["Color"], em.inputs["Color"])
    em.inputs["Strength"].default_value = 0.45
    nt.links.new(tr.outputs[0], mixa.inputs[1])
    nt.links.new(shell.outputs[0], mixa.inputs[2])
    nt.links.new(geo.outputs["Backfacing"], mixb.inputs["Fac"])
    nt.links.new(mixa.outputs[0], mixb.inputs[1])
    nt.links.new(em.outputs[0], mixb.inputs[2])
    nt.links.new(mixb.outputs[0], out.inputs["Surface"])
    mixa.inputs["Fac"].default_value = alpha0
    mixa.inputs["Fac"].keyframe_insert("default_value", frame=1)
    mixa.inputs["Fac"].default_value = alpha0
    mixa.inputs["Fac"].keyframe_insert("default_value", frame=F(t_opaque0))
    mixa.inputs["Fac"].default_value = 1.0
    mixa.inputs["Fac"].keyframe_insert("default_value", frame=F(t_opaque1))
    return m


# taut cable twang after the pull stops (damped, non-negative sag; baked per frame)
for fr in range(F(6.0) + 1, F(6.0) + 22):
    dt = (fr - F(6.0)) / 30.0
    wall.root["p_sag_m"] = 0.05 * math.exp(-dt / 0.18) * (0.5 + 0.5 * math.cos(TAU * 5.0 * dt))
    wall.root.keyframe_insert('["p_sag_m"]', frame=fr)
set_interp(wall.root, '["p_sag_m"]', "LINEAR")
for fc in wall.root.animation_data.action.fcurves:
    if fc.data_path == '["p_sag_m"]':
        for kp in fc.keyframe_points:
            if kp.co[0] < F(4.4):
                kp.interpolation = "CONSTANT"
cable0 = cables[0]
cable0.data.materials.clear()
cable0.data.materials.append(tunnel_material("s01_cable00_tunnel", 2.0, 2.3))

# ---- Gary (v2 asset, motion_v2 strips; root baked per frame)
gary = asm.append("characters/gary_v2")
GARY_DX = 0.8     # stands this far to the right of rack B's centre line before the stretch
GARY_Y = Y_ACT - 0.35
GARY_SPOT = (6.3, 2.3)            # left of the bench end: two-shot Gary | scope | Manager
GARY_YAW_SPOT = PI / 2 - 0.55     # faces the Manager, turned 3/4 to the camera so hole 1 (left chest) reads
T_TURN = 3.62
T_RUN0 = 6.1
RUN_MPS = 2.2
GARY_HAUL_END = haul(STRETCH_T1)[1]
T_RUN1 = T_RUN0 + math.dist((GARY_HAUL_END, GARY_Y), GARY_SPOT) / RUN_MPS
T_STARTLE = 8.12
GARY_FALL_SPEED = 1.35            # shot_hit_fall: tip at about 9.5 s, ground hit at about 9.74 s

M2.apply(gary, "idle_breathe", 0.0, hold=True, repeat=4)
st_turn = M2.apply(gary, "turn_right_90", T_TURN, speed=1.25, hold=False, key_root_yaw=False)
F_TURN_END = int(math.floor(st_turn.frame_end))
M2.apply(gary, "pull_cable", STRETCH_T0, speed=36.0 / (P_CYC * 30.0), repeat=N_HEAVE - 0.5, hold=False, face=False, blend_in=3)
_run = M2.ensure_action(gary, "run")
_run_sp = RUN_MPS / float(_run["root_speed_mps"])
_run_cyc = (_run.frame_range[1] - _run.frame_range[0]) / _run_sp
M2.apply(gary, "run", T_RUN0, speed=_run_sp, repeat=(T_RUN1 - T_RUN0) * 30.0 / _run_cyc, hold=False, blend_in=3)
M2.apply(gary, "idle_breathe", T_RUN1, hold=True, repeat=2, blend_in=5)
M2.apply(gary, "startle", T_STARTLE, hold=False, blend_in=2)
M2.apply(gary, "shot_hit_fall", T_BANG, speed=GARY_FALL_SPEED, hold=True)
_run_yaw = asm.yaw_to((GARY_HAUL_END, GARY_Y), GARY_SPOT)


def gary_state(t):
    f = F(t)
    if t < STRETCH_T0 - 1e-6:
        x = X_A + gap_at(t) + GARY_DX
        if t >= 4.25:
            x = ramp(t, 4.25, STRETCH_T0, x, R0)          # settles onto the grip bar
        return x, GARY_Y, 0.0, (0.0 if f <= F_TURN_END else -PI / 2)
    if t <= STRETCH_T1 + 1e-6:
        _g, x, _a, hop = haul(t)
        return x, GARY_Y, hop, -PI / 2
    if t < T_RUN0:                                         # lets go and turns round (off camera)
        return GARY_HAUL_END, GARY_Y, 0.0, -PI / 2 + (PI / 2 + _run_yaw) * FX.smooth3((t - STRETCH_T1) / (T_RUN0 - STRETCH_T1))
    if t < T_RUN1:
        k = (t - T_RUN0) / (T_RUN1 - T_RUN0)
        return GARY_HAUL_END + (GARY_SPOT[0] - GARY_HAUL_END) * k, GARY_Y + (GARY_SPOT[1] - GARY_Y) * k, 0.0, _run_yaw
    return GARY_SPOT[0], GARY_SPOT[1], 0.0, _run_yaw + (GARY_YAW_SPOT - _run_yaw) * FX.smooth3((t - T_RUN1 + 0.1) / 0.4)


FX.bake_obj(gary.root, gary_state, jumps=CUT_FRAMES + [F_TURN_END + 1])
FX.bake_obj(handle, lambda t: (X_A + gap_at(t) + HANDLE_DX, GARY_Y, HANDLE_Z, 0.0), jumps=CUT_FRAMES)
# faces: strain while hauling (gritted, sweating, red pulses on each heave), hopeful at the scope, scared at the gun
asm.key_prop(gary.root, "p_expr_sweating", 0.0, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 4.3, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 4.55, 1.0)
for t, v in ((0.0, 0.0), (4.3, 0.0), (4.5, 0.55), (6.0, 0.55), (6.3, 0.0)):
    asm.key_prop(gary.root, "p_expr_angry", t, v)
asm.key_prop(gary.root, "p_flush", 0.0, 0.0)
for _f in range(F(4.4), F(6.0) + 1, 2):
    asm.key_prop(gary.root, "p_flush", (_f - 1) / 30.0, 0.2 + 0.5 * haul((_f - 1) / 30.0)[2])
asm.key_prop(gary.root, "p_flush", 6.4, 0.0)
for t, v in ((0.0, 0.0), (T_RUN1 - 0.1, 0.0), (T_RUN1 + 0.15, 0.7), (T_STARTLE, 0.7), (T_STARTLE + 0.05, 0.0)):
    asm.key_prop(gary.root, "p_expr_happy", t, v)
for t, v in ((0.0, 0.0), (T_STARTLE, 0.0), (T_STARTLE + 0.12, 1.0), (T_BANG, 1.0), (T_BANG + 0.1, 0.0)):
    asm.key_prop(gary.root, "p_expr_scared", t, v)
# hole 1: opens at the bang, over-sized cartoon pop peaking at the hole_pop cue (9.45) so it reads at phone size, settles
# at the storyboard radius 1.0 by 9.75 (S2 continuity unchanged). A pale disc at the hole centre (inside the cut cylinder,
# so only seen through the hole) gives the dark-wall background contrast: the classic see-through gag.
asm.key_prop(gary.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
for t, v in ((T_BANG, 1.0), (T_BANG + 0.07, 1.6), (9.42, 1.6), (9.47, 1.9), (9.75, 1.0)):
    asm.key_prop(gary.root, "p_hole_1_radius", t, v)
_he = bpy.data.objects["HOLE_hole_1"] if "HOLE_hole_1" in bpy.data.objects else [o for o in gary.objs if o.name.startswith("HOLE_hole_1")][0]
_dm = bpy.data.meshes.new("s01_hole_light")
import bmesh  # noqa: E402
_bm = bmesh.new()
bmesh.ops.create_circle(_bm, cap_ends=True, segments=24, radius=0.0435 * 0.85)
_bm.to_mesh(_dm)
_bm.free()
hole_light = bpy.data.objects.new("s01_hole_light", _dm)
bpy.context.scene.collection.objects.link(hole_light)
_hlm = bpy.data.materials.new("s01_hole_light")
_hlm.use_nodes = True
_hb = _hlm.node_tree.nodes["Principled BSDF"]
_hb.inputs["Base Color"].default_value = (0.95, 0.88, 0.78, 1.0)
_hb.inputs["Emission Color"].default_value = (0.95, 0.88, 0.78, 1.0)
_hb.inputs["Emission Strength"].default_value = 1.1
_dm.materials.append(_hlm)
hole_light.parent = _he
hole_light.matrix_parent_inverse.identity()
hole_light.visible_shadow = False
for _i in range(3):
    _fc = hole_light.driver_add("scale", _i)
    _v = _fc.driver.variables.new()
    _v.name = "v"
    _v.targets[0].id = gary.root
    _v.targets[0].data_path = '["p_hole_1_radius"]'
    _fc.driver.expression = "max(v,0.0001)" if _i < 2 else "1.0"
asm.L.V(hole_light, 0, T_BANG, 10.0)
update()

# ---- Pulse (clay courier): hook run on the wall conduit, then three runs along the bundle
R0 = 0.14


def make_pulse(name):
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = (-900, 0, 0)

    def part(nm, loc, r, mat, scale=(1, 1, 1)):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=r, location=(0, 0, 0))
        o = bpy.context.object
        o.name = nm
        o.scale = scale
        o.data.materials.append(mat)
        for p in o.data.polygons:
            p.use_smooth = True
        o.parent = root
        o.location = loc
        return o
    part(name + "_body", (0, 0, R0), R0, mats["MAT_vfx_clay_shirt_yellow"])
    for sx in (-1, 1):
        part(name + "_eye", (sx * 0.055, -R0 * 0.80, R0 * 1.18), 0.05, mats["MAT_vfx_clay_eye_white"])
        part(name + "_pupil", (sx * 0.055, -R0 * 0.80 - 0.035, R0 * 1.18), 0.026, mats["MAT_vfx_clay_pupil_black"])
    return root


p_hook = make_pulse("Pulse_hook")
for _f in range(F(0.0), F(2.1) + 1):
    _t = (_f - 1) / 30.0
    _w = abs(math.sin(TAU * 4.0 * _t))                      # hop phase
    p_hook.location = (-8.5 + 20.0 * _t / 2.1, PULSE_Y, PULSE_Z + 0.06 * _w)
    p_hook.scale = (1.0 + 0.22 * (1.0 - _w), 1.0 - 0.10 * (1.0 - _w), 1.0 - 0.10 * (1.0 - _w))   # stretch in flight, squash on landing
    p_hook.keyframe_insert("location", frame=_f)
    p_hook.keyframe_insert("scale", frame=_f)
set_interp(p_hook, "location", "LINEAR")
set_interp(p_hook, "scale", "LINEAR")
for ch in [p_hook] + list(p_hook.children):
    asm.L.V(ch, 0, 0.0, 2.15)
RUNS = [(2.2, 2.93, 5.0), (3.0, 3.55, 2.0), (3.6, 4.15, 1.0)]
cable0.data.use_path = True
p_run = make_pulse("Pulse_run")
con = p_run.constraints.new("FOLLOW_PATH")
con.target = cable0
con.use_curve_follow = False
con.use_fixed_location = True
p_run.location = (0, 0, 0)
con.offset_factor = 0.0
p_run.scale = (1, 1, 1)
con.keyframe_insert("offset_factor", frame=1)
for ta, tb, ln in RUNS:
    con.offset_factor = 0.0
    con.keyframe_insert("offset_factor", frame=F(ta))
    con.offset_factor = 1.0
    con.keyframe_insert("offset_factor", frame=F(tb))
    a = math.exp(-0.32 * ln)
    for _f in range(F(ta), F(tb) + 1):          # decay envelope with a springy squash/stretch wobble (illustrative)
        _t = (_f - 1) / 30.0
        _k = (_t - ta) / (tb - ta)
        _a = 1.0 + (a - 1.0) * _k
        _w = math.sin(TAU * 6.0 * (_t - ta)) * 0.12
        p_run.scale = (_a * (1.0 + _w), _a * (1.0 - 0.5 * _w), _a * (1.0 - 0.5 * _w))
        p_run.keyframe_insert("scale", frame=_f)
set_interp(p_run, "scale", "LINEAR")
for fc in con.id_data.animation_data.action.fcurves:
    if "offset_factor" in fc.data_path:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
for ch in [p_run] + list(p_run.children):
    asm.L.V(ch, 0, 2.2, 4.2)

# ---- bench + scope with the eye sequence
bench = asm.append("lab_office/lab_bench")
asm.place(bench.root, (X_BENCH, Y_ACT, 0), yaw=0)
scope = asm.append("lab_office/bench_oscilloscope")
update()
slot = wpos(bench.hook("scope_slot"))
asm.place(scope.root, tuple(slot), yaw=0)
update()

eye_dir = os.path.join(TEX, "eye_s1")
eye_params = []
for f in range(1, 301):
    t = (f - 1) / 30.0
    eye_params.append(dict(sigma_g=ramp(t, 6.0, 7.4, 0.38, 0.45), noise=ramp(t, 6.0, 7.4, 0.05, 0.30),
                           jitter=ramp(t, 6.0, 7.4, 0.03, 0.12)))
first_png = os.path.join(eye_dir, "eye_s1_0001.png")
if "--no-eye" not in ARGS or not os.path.exists(first_png):
    bers = T.eye_sequence(eye_dir, "eye_s1", eye_params, seed=11)
    print("EYE eye_s1 BER first %.2e last %.2e min %.2e max %.2e" % (bers[0], bers[-1], min(bers), max(bers)))
screen_obj = [o for o in scope.objs if o.name == "bench_oscilloscope_screen"][0]
smat = screen_obj.material_slots[0].material
img = bpy.data.images.load(first_png)
img.source = "SEQUENCE"
node = smat.node_tree.nodes["SCREEN_IMAGE"]
node.image = img
iu = node.image_user
iu.frame_duration = 300
iu.frame_start = 1
iu.frame_offset = 0
iu.use_auto_refresh = True
smat.node_tree.nodes["SCREEN_FAC"].outputs[0].default_value = 1.0

# ---- Manager (v2): stomps in from the hall end, stops right of the scope, slow burn facing the screen, shotgun pops into
# his hands ("Shotgun time"), raise, aim with an anger tremor, bang at T_BANG (3-frame hit-stop). He stands 1.46 m right
# of the screen so the raised gun stays right of / above the scope in every camera (v1.1 gun crossed the scope).
mgr = asm.append("characters/manager_v2")
update()
_sc_obj = [o for o in scope.objs if o.name == "bench_oscilloscope_screen"][0]
_bb = [_sc_obj.matrix_world @ Vector(c) for c in _sc_obj.bound_box]
scr = sum(_bb, Vector()) / 8.0        # true centre of the screen mesh (the asset hook is offset from it)
MGR_Y = 1.95
X_MGR_START, X_MGR_STAND = 11.4, 9.1
MGR_MPS = 1.0                          # stomp_walk strip sped up from its baked 0.749 m/s (feet stay locked)
T_M_ARR = 7.15
T_M0 = T_M_ARR - (X_MGR_START - X_MGR_STAND) / MGR_MPS
T_R1 = T_BANG - 24 / 30.0              # gun_raise_aim_fire aim_start (action frame 20); shot = frame 44 at T_BANG
T_R0 = T_R1 - 10 / 30.0               # frames 0-20 (port arms to aim) played at 2x
STAND = (X_MGR_STAND, MGR_Y)
YAW_READ = -PI / 2 - 0.2      # faces the scope side (screen 25 deg to his right), 3/4 face to the camera
YAW_AIM = asm.yaw_to(STAND, GARY_SPOT)
M2.apply(mgr, "idle_breathe_tense", 0.0, hold=True, repeat=5)
_st = M2.ensure_action(mgr, "stomp_walk")
_st_sp = MGR_MPS / float(_st["root_speed_mps"])
M2.apply(mgr, "stomp_walk", T_M0, speed=_st_sp, repeat=(T_M_ARR - T_M0) * 30.0 / (36.0 / _st_sp), hold=False)
M2.apply(mgr, "manager_slow_burn", T_M_ARR, speed=1.8, start_frame=76, end_frame=128, hold=True, blend_in=5, face=False)
M2.apply(mgr, "gun_raise_aim_fire", T_R0, speed=2.0, start_frame=0, end_frame=20, hold=True, blend_in=3, face=False)
M2.apply(mgr, "gun_raise_aim_fire", T_R1, start_frame=20, end_frame=44, hold=True, face=False)
M2.apply(mgr, "gun_raise_aim_fire", T_BANG + HIT_STOP / 30.0, start_frame=44, hold=True, face=False)


def mgr_state(t):
    if t < T_M0:
        return X_MGR_START, MGR_Y, 0.0, -PI / 2
    if t < T_M_ARR:
        return X_MGR_START - MGR_MPS * (t - T_M0), MGR_Y, 0.0, -PI / 2
    y = -PI / 2 + (YAW_READ + PI / 2) * FX.smooth3((t - T_M_ARR) / 0.35)
    y = y + (YAW_AIM - YAW_READ) * FX.smooth3((t - (T_R0 - 0.1)) / 0.4)
    return X_MGR_STAND, MGR_Y, 0.0, y


FX.bake_obj(mgr.root, mgr_state, jumps=CUT_FRAMES)
for t, v in ((0.0, 0.0), (7.2, 0.0), (8.15, 1.0)):
    asm.key_prop(mgr.root, "p_anger", t, v)
for t, v in ((0.0, 0.0), (7.3, 0.0), (8.2, 1.0)):
    asm.key_prop(mgr.root, "p_flush", t, v)

# ---- shotgun on the Manager's right hand (rot z = pi on the grip hook); hammerspace pop-in at T_R0
gun = asm.append("props/shotgun")
gun.variant("clay", True)
asm.attach(gun.root, mgr.hook("gun_grip_R"), rot=(0, 0, PI))
asm.show(gun, T_R0 - 1 / 30.0, 10.0)
_f0 = F(T_R0)
for df, sc in ((-1, 0.05), (0, 0.3), (2, 1.25), (4, 0.92), (6, 1.0)):
    gun.root.scale = (sc, sc, sc)
    gun.root.keyframe_insert("scale", frame=_f0 + df)
_tn = FX.Noise(13, channels=2)
for fr in range(F(T_R1), F(T_BANG) + 1):        # anger tremor while aiming (6-8 Hz plus slow wander), zero at the shot
    _t = (fr - 1) / 30.0
    w = FX.smooth3((_t - T_R1) / 0.2) * (1.0 - FX.smooth3((_t - (T_BANG - 0.12)) / 0.12))
    gun.root.rotation_euler = (w * (0.014 * math.sin(TAU * 7.0 * _t) + 0.008 * _tn(_t, 0)), w * 0.010 * math.sin(TAU * 5.3 * _t + 1.0), PI)
    gun.root.keyframe_insert("rotation_euler", frame=fr)
set_interp(gun.root, "scale", "LINEAR")
set_interp(gun.root, "rotation_euler", "LINEAR")
# darker clay gun so it reads against the pale wall (scene copies of the materials; asset file untouched)
for o in gun.objs:
    if o.type != "MESH":
        continue
    for sl in o.material_slots:
        m = sl.material
        if m and m.use_nodes and not m.get("s01_dark"):
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED" and not n.inputs["Base Color"].is_linked:
                    c = n.inputs["Base Color"].default_value
                    if 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.25:
                        n.inputs["Base Color"].default_value = (c[0] * 0.35, c[1] * 0.35, c[2] * 0.35, 1.0)
            m["s01_dark"] = 1
asm.fx("dust_cloud", T_R0, parent=mgr.hook("gun_grip_R"), scale=0.22, dur=0.5)

# ---- effects: ear steam (puffs from about 7.55 s, steam_hiss cue), muzzle flash + smoke ring at the muzzles
steam = asm.fx("ear_steam", 7.35, dur=2.2)
for side in ("L", "R"):
    eh = steam.hook("ear_" + side)
    eh.parent = mgr.hook("steam_" + side)
    eh.matrix_parent_inverse.identity()
    eh.location = (0, 0, 0)
    eh.rotation_euler = (0, 0, 0)
    eh.scale = (0.6, 0.6, 0.6)       # smaller puffs (v1.1 popcorn covered his head)
steam.root.parent = None
for side in ("L", "R"):
    asm.fx("muzzle_flash", T_BANG - 1 / 30.0, rot=(-PI / 2, 0, 0), parent=gun.hook("muzzle_" + side), dur=0.15)   # flash grows from 0: visible ON the bang frame
    asm.fx("smoke_ring", T_BANG + 0.07, rot=(-PI / 2, 0, 0), parent=gun.hook("muzzle_" + side), dur=0.35)   # gone before the push (it drifted across the lens)
asm.key_prop(gun.root, "p_trigger", T_BANG - 0.05, 0.0)
asm.key_prop(gun.root, "p_trigger", T_BANG, 1.0)
asm.key_prop(gun.root, "p_trigger", T_BANG + 0.3, 0.0)

# ---- hit effects at Gary's chest: clay crumbs blown out of his BACK (no star puff: it covered the hole in v1.1)
scn.frame_set(F(T_BANG) + 1)
update()
garm = gary.armature
chest_w = garm.matrix_world @ garm.pose.bones["chest"].head
back = Vector((-math.sin(GARY_YAW_SPOT), math.cos(GARY_YAW_SPOT), 0.0))
hit_pt = chest_w + back * 0.18
print("HIT point", tuple(round(c, 3) for c in hit_pt))
crumb_mat = mats["MAT_vfx_clay_shirt_yellow"]
FX.add_crumbs(tuple(hit_pt), tuple(back), T_BANG, crumb_mat, n=16, seed=5)
scn.frame_set(1)
update()

# ---- popped copper strands at the strand_pop cues (sfx_cues.json 4.95, 5.35, 5.65, 5.95), sparks on each
for i, (tp, ci, off) in enumerate(((4.95, 3, 0.42), (5.35, 8, 0.56), (5.65, 11, 0.47), (5.95, 5, 0.62))):
    anc, _ = FX.add_strand_pop(cables[ci], off, tp, mats["MAT_vfx_copper"], t_end=6.0 + 1 / 30.0, seed=21 + i, n=5, length=(0.10, 0.18), radius=0.006)
    asm.fx("impact_stars", tp, parent=anc, scale=0.14, dur=0.25)

# ------------------------------------------------------------------ camera director (baked, eased, handheld, impact shakes)
# electron pockets: sample the evaluated centreline of cable 00 (frame 1: gap 5 m, sag 0.25 m)
scn.frame_set(1)
update()
probe = bpy.data.objects.new("probe_path", None)
bpy.context.scene.collection.objects.link(probe)
pcon = probe.constraints.new("FOLLOW_PATH")
pcon.target = cable0
pcon.use_curve_follow = False
pcon.use_fixed_location = True
CPTS = []
for i in range(81):
    pcon.offset_factor = i / 80.0
    update()
    CPTS.append(tuple(probe.matrix_world.translation))
bpy.data.objects.remove(probe)
line = FX.Polyline(CPTS)
print("CABLE00 length %.3f m, start %s end %s" % (line.length, tuple(round(c, 3) for c in CPTS[0]), tuple(round(c, 3) for c in CPTS[-1])))

CY = -4.0
D = FX.Director(T_BANG, seed=7, shake_win=0.45)
for tp, amp in ((4.95, 0.22), (5.35, 0.28), (5.65, 0.28), (5.95, 0.32)):
    D.add_hit(tp, amp, 0.3)
S0 = 0.46 * line.length
P0, T0 = line.at(S0)
P0 = tuple(P0)
_n1 = T0.cross(Vector((0, 0, 1))).normalized()
A0 = tuple(Vector(P0) + T0 * 0.6 + _n1 * 0.12)
P_END, A_END = (2.9, CY + 0.1, 1.9), (2.9, Y_ACT, 1.4)
K_ZOOM = 9.0


def shot_zoom_out(u, t):
    e = FX.smooth5(u)
    g = (math.exp(K_ZOOM * e) - 1.0) / (math.exp(K_ZOOM) - 1.0)      # log-like pull-back: slow inside the cable, fast outside
    pos = FX.lerp3(P0, P_END, g)
    if e < 0.3:
        tgt = FX.lerp3(A0, P0, FX.smooth3(e / 0.3))                          # look down the tube, then across at the cable
    else:
        tgt = FX.lerp3(P0, A_END, FX.smooth3((e - 0.3) / 0.7))
    return pos, tgt, 22.0 + (21.0 - 22.0) * e


D.add(0.0, 2.2, shot_zoom_out, hh=1.0)
D.simple(2.2, 3.0, (2.9, CY + 0.1, 1.9), (2.9, Y_ACT, 1.4), (2.9, CY + 0.6, 1.9), (2.9, Y_ACT, 1.4), lens=21)
D.simple(3.0, 3.6, (2.0, -2.2, 1.7), (1.4, Y_ACT, 1.4), (1.4, -1.6, 1.7), (1.4, Y_ACT, 1.4), lens=26)
D.simple(3.6, 4.2, (0.5, -0.4, 1.6), (0.9, Y_ACT, 1.4), (0.6, 0.4, 1.5), (0.9, Y_ACT, 1.4), lens=20)
# the stretch: 3/4 front on the cable gap, rack B and Gary (face visible), tracking the rack as it is hauled out
D.simple(4.2, 6.0, (1.45, -0.8, 1.3), (1.5, Y_ACT - 0.2, 1.05), (1.55, -0.7, 1.3), (1.6, Y_ACT - 0.2, 1.05), lens=32)   # near-static: rack B visibly slides away
D.simple(6.0, 6.4, (scr.x + 0.06, scr.y - 0.6, scr.z + 0.04), tuple(scr), (scr.x + 0.05, scr.y - 0.5, scr.z + 0.035), tuple(scr), lens=28, hh=0.6)
# 6.4-10.0 one continuous shot: Manager stomps in and burns beside the scope (E), eases back to the two-shot
# Gary | scope | Manager (F), holds through the bang, hit and tip, then an eased-out push onto the scope screen (G)
E_P0, E_A0, E_P1, E_A1 = (8.15, -0.75, 1.35), (8.9, 2.1, 1.2), (8.05, -0.4, 1.35), (8.75, 2.1, 1.25)
F_P0, F_A0, F_P1, F_A1 = (7.8, -0.75, 1.3), (7.8, 2.2, 1.0), (7.78, -0.63, 1.28), (7.76, 2.2, 1.0)
Z_P, Z_A = (scr.x + 0.03, scr.y - 0.8, scr.z + 0.03), tuple(scr)
T_EF0, T_EF1, T_PUSH = 7.95, 8.45, 9.55
T_LAST = 10.0 - 1.0 / 30.0


def shot_tail(u, t):
    if t < T_EF0:
        k = FX.smooth3((t - 6.4) / (T_EF0 - 6.4))
        return FX.lerp3(E_P0, E_P1, k), FX.lerp3(E_A0, E_A1, k), 28.0
    if t < T_EF1:
        k = FX.smooth5((t - T_EF0) / (T_EF1 - T_EF0))
        return FX.lerp3(E_P1, F_P0, k), FX.lerp3(E_A1, F_A0, k), 28.0 + (26.0 - 28.0) * k
    if t < T_PUSH:
        k = FX.smooth3((t - T_EF1) / (T_PUSH - T_EF1))
        return FX.lerp3(F_P0, F_P1, k), FX.lerp3(F_A0, F_A1, k), 26.0
    k = FX.ease_out3((t - T_PUSH) / (T_LAST - T_PUSH))
    k = FX.smooth3(min(1.0, k * 1.0)) * 0.25 + k * 0.75        # soft start, decelerating to rest on the last frame
    return FX.lerp3(F_P1, Z_P, k), FX.lerp3(F_A1, Z_A, k), 26.0 + (34.0 - 26.0) * k, 1.0 - k


D.add(6.4, 10.0, shot_tail, hh=1.0)
D.bake(cuts=set(CUT_FRAMES))
FX.shutter_cuts(scn, CUT_FRAMES, 0.5)


# electron pockets in the cable (cyan: moving right, amber: moving left), baked from the camera distance
def glow(name, col, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*col, 1.0)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


FX.add_electrons(D, line, glow("s01_electron_fwd", (0.25, 0.85, 1.0), 2.5), glow("s01_electron_bwd", (1.0, 0.62, 0.15), 2.5), t_end=2.2, n=22, seed=3, center=S0)


def big_at(t0, t1, text, loc, size):
    """BIG overlay at a custom frame position (blender_lib OVL is shared; restored after the call)."""
    d = asm.L.OVL["BIG"]
    old = (d["loc"], d["size"], d.get("outline", 0.0))
    d["loc"], d["size"], d["outline"] = loc, size, old[2] * size / old[1]
    try:
        asm.big(t0, t1, text)
    finally:
        d["loc"], d["size"], d["outline"] = old


# ---- captions: subtitles from narration.json (same source as the VO); one source card per shot; HUD gated by FILM_HUD
asm.narr_vo(1)
asm.big(0.2, 1.0, "COPPER")
asm.big(2.2, 3.0, "5 m")
asm.big(3.0, 3.6, "2 m")
asm.big(3.6, 4.2, "1 m")
big_at(4.4, 5.2, "STRETCH", (0.0, 1.85), 0.5)        # top band, off Gary's head (critique S1 4.4-6.0)
big_at(T_BANG, T_PUSH, "BANG", (0.0, 1.8), 0.55)      # top band, off Gary and the hole
asm.card(0.0, 2.2, "NVL72 backplane: ~5,000 copper cables, 2 miles (SemiAnalysis)")
asm.card(2.2, 3.0, "4x25G passive twinax: >= 5 m (IEEE 802.3bj)")
asm.card(3.0, 3.6, "100G/lane passive twinax: >= 2 m (IEEE 802.3ck)")
asm.card(3.6, 4.2, "200G/lane passive twinax: >= 1 m, objective (IEEE 802.3dj)")
asm.card(6.0, 8.0, "BER = 0.5 erfc(Q/sqrt2), Q from simulated trace samples (illustrative eye model)")
asm.fxn(0.0, 2.2, "[FX: fast fly-along, clay cable bundles, Pulse sprint trail]")
asm.fxn(2.2, 4.2, "[FX: Pulse fades as it travels (illustrative exp. decay)]")
asm.fxn(4.4, 6.0, "[FX: cables creak, strands pop, glow dims]")
asm.fxn(7.4, 8.5, "[FX: face turns red, vein pops, steam blast]")
asm.fxn(T_BANG, 9.8, "[FX: muzzle flash, smoke ring, hole decal]")
asm.wl("NVL72 BACKPLANE\nCOPPER CARTRIDGES", (-4.5, Y_WALL - 0.6, 3.5), 0.0, 2.2, size=0.4)
asm.wl("PULSE DECAYS", (X_A + 1.2, Y_ACT - 1.1, 2.45), 3.0, 4.2, size=0.24)    # not in the 5 m shot (hidden under "5 m")

# ---- look: tame blown-out emitters (LED strips 12 -> 3, bench light panel 6 -> 2), keep the warm clay grade
EMIT_CAP = {"MAT_datacenter_led_strip": 3.0, "MAT_lab_office_light_panel": 2.0, "MAT_datacenter_sign_exit_emit": 2.0,
            "MAT_datacenter_sign_text_emit": 1.5}
for m in bpy.data.materials:
    base = m.name.split(".")[0]
    if base in EMIT_CAP and m.use_nodes:
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED" and "Emission Strength" in n.inputs:
                n.inputs["Emission Strength"].default_value = min(n.inputs["Emission Strength"].default_value, EMIT_CAP[base])

# motion blur on (shutter 0.5, animated to 0 on hard-cut frames); render_scene.py keeps the scene's own blur through presets
scn.render.use_motion_blur = True
asm.finalize(OUT)
