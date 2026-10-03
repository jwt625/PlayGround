"""Scene S1 'Copper: the stretch' (film 0-10 s), assembled from the v1 asset library.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s01_copper.py -- scenes/v1/s01_copper.blend
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


def _show_objects(asset_or_coll, t0, t1):
    """Framework workaround: Collection.hide_render is not animatable in Blender 4.2, so key the objects instead."""
    c = asset_or_coll.coll if isinstance(asset_or_coll, asm.Asset) else asset_or_coll
    for o in asm._all_objs(c):
        asm.L._VIS.setdefault(o, []).append((F(t0), F(t1)))


asm.show = _show_objects     # asm.fx() resolves show() through the module namespace

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
    ld.color = (0.95, 0.97, 1.0)
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

# ---- rack pair (A fixed at X_A, B at X_A + gap), p_gap constant-keyed
racks = asm.append("datacenter/rack_pair_for_cable_gag")
asm.show(racks, 0.0, 10.0)


def key_gap(t, gap, interp="CONSTANT", lin_from=None):
    racks.root["p_gap"] = float(gap)
    racks.root.update_tag()
    racks.root.keyframe_insert('["p_gap"]', frame=F(t))
    racks.root.location = (X_A + gap / 2.0, Y_ACT, 0)
    racks.root.keyframe_insert("location", frame=F(t))


for t, g in GAP_KEYS:
    key_gap(t, g)
STRETCH_T0, STRETCH_T1 = 4.4, 6.0
key_gap(STRETCH_T0, 1.0)
key_gap(STRETCH_T1, 2.0)
set_interp(racks.root, '["p_gap"]', "CONSTANT")
set_interp(racks.root, "location", "CONSTANT")
# the stretch (4.4-6.0) is a continuous pull: linear keys on the last segment
for path in ('["p_gap"]', "location"):
    ad = racks.root.animation_data
    for fc in ad.action.fcurves:
        if fc.data_path == path and (path != "location" or fc.array_index == 0):
            kps = sorted(fc.keyframe_points, key=lambda k: k.co[0])
            kps[-2].interpolation = "LINEAR"
update()


def gap_at(t):
    g = 5.0
    for tk, gv in GAP_KEYS:
        if t >= tk:
            g = gv
    if t >= STRETCH_T0:
        g = ramp(t, STRETCH_T0, STRETCH_T1, 1.0, 2.0)
    return g


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
for t, s in ((0.0, 0.25), (2.2, 0.25), (3.0, 0.16), (3.6, 0.16), (4.4, 0.16), (6.0, 0.0)):
    asm.key_prop(wall.root, "p_sag_m", t, s, "CONSTANT" if t < 4.4 else "LINEAR")

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
cable0 = cables[0]
cable0.data.materials.clear()
cable0.data.materials.append(tunnel_material("s01_cable00_tunnel", 2.0, 2.3))

# ---- Gary
gary = asm.append("characters/gary", actions=True)
GARY_DX = 0.8     # stands this far to the right of rack B's centre line (hands reach the side of rack B)
GARY_Y = Y_ACT - 0.35


def gary_x(t):
    return X_A + gap_at(t) + GARY_DX


asm.place(gary.root, (gary_x(0.0), GARY_Y, 0), yaw=0)
asm.play(gary, "idle", 0.0, hold=True, repeat=3)
act_pull = bpy.data.actions["ACT_gary_pull_cable"]
n_pull = int(math.ceil((STRETCH_T1 - 4.2) * 30 / (act_pull.frame_range[1] - act_pull.frame_range[0])))
asm.play(gary, "pull_cable", 4.2, hold=True, repeat=n_pull)
asm.play(gary, "idle", STRETCH_T1, hold=True, repeat=3)
asm.play(gary, "topple_back", T_BANG + HIT_STOP / 30.0, hold=True)   # hit-stop: the fall starts 3 frames after the bang
# root keys: constant jumps with the rack, a linear drag during the stretch, then stand
for t, gx in ((0.0, gary_x(0.0)), (2.2, gary_x(2.2)), (3.0, gary_x(3.0)), (3.6, gary_x(3.6))):
    asm.key_loc(gary.root, t, (gx, GARY_Y, 0), yaw=0.0, interp="CONSTANT")
asm.key_loc(gary.root, 4.2, (gary_x(3.6), GARY_Y, 0), yaw=-PI / 2, interp="CONSTANT")
asm.key_loc(gary.root, STRETCH_T0, (gary_x(3.6), GARY_Y, 0), yaw=-PI / 2, interp="LINEAR")
asm.key_loc(gary.root, STRETCH_T1, (gary_x(STRETCH_T1), GARY_Y, 0), yaw=-PI / 2, interp="CONSTANT")
# after the pull Gary walks 1.6 m clear of the racks (so he does not topple into rack B), then stands facing the Manager
GARY_END_X = gary_x(STRETCH_T1) + 1.6
asm.walk(gary, 6.2, [(gary_x(STRETCH_T1), GARY_Y, 0), (GARY_END_X, GARY_Y, 0)])
# expressions
for t, v in ((0.0, 0.0), (4.2, 0.0), (6.0, 1.0)):
    asm.key_prop(gary.root, "p_expr_sweating", t, v)
for t, v in ((0.0, 0.0), (6.6, 0.0), (7.4, 1.0), (8.5, 0.0)):
    asm.key_prop(gary.root, "p_expr_dread", t, v)
for t, v in ((0.0, 0.0), (8.5, 0.0), (8.9, 1.0), (T_BANG, 1.0), (T_BANG + 0.1, 0.0)):
    asm.key_prop(gary.root, "p_expr_scared", t, v)
asm.key_prop(gary.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_hole_1_radius", T_BANG, 1.0, "CONSTANT")

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

# ---- Manager: walks all the way to the bench and reads the scope from beside it (v1.1)
mgr = asm.append("characters/manager", actions=True)
MGR_Y = Y_ACT - 0.85
update()
_sc_obj = [o for o in scope.objs if o.name == "bench_oscilloscope_screen"][0]
_bb = [_sc_obj.matrix_world @ Vector(c) for c in _sc_obj.bound_box]
scr = sum(_bb, Vector()) / 8.0        # true centre of the screen mesh (the asset hook is offset from it)
STAND = (X_MGR_STARE, MGR_Y, 0)
t_arr = asm.walk(mgr, 4.1, [(X_MGR_START, MGR_Y, 0), STAND])
asm.place(mgr.root, (X_MGR_START, MGR_Y, 0), yaw=-PI / 2)
t_aim0 = T_BANG - 36 / 30.0
asm.play(mgr, "idle", t_arr, hold=True, repeat=max(1.0, (t_aim0 - t_arr) / 2.0))
yaw_scope = asm.yaw_to(STAND, (scr.x, scr.y, 0))
aim_yaw = asm.yaw_to(STAND, (GARY_END_X, GARY_Y, 0))
asm.key_loc(mgr.root, t_arr, STAND, yaw=-PI / 2, interp="LINEAR")
asm.key_loc(mgr.root, t_arr + 0.5, STAND, yaw=yaw_scope, interp="LINEAR")
asm.key_loc(mgr.root, 7.7, STAND, yaw=yaw_scope, interp="LINEAR")
asm.key_loc(mgr.root, 8.35, STAND, yaw=aim_yaw, interp="LINEAR")
asm.key_loc(mgr.root, T_BANG, STAND, yaw=aim_yaw, interp="LINEAR")
for t, v in ((0.0, 0.0), (7.0, 0.0), (8.2, 1.0)):
    asm.key_prop(mgr.root, "p_anger", t, v)
for t, v in ((0.0, 0.0), (7.2, 0.0), (8.2, 1.0)):
    asm.key_prop(mgr.root, "p_flush", t, v)
# aim_gun: frames 0..36 up to the shot pose (bang at T_BANG), 3-frame hit-stop, then frames 36..end
act_aim = bpy.data.actions["ACT_manager_aim_gun"]
ad_m = mgr.armature.animation_data
tr_a = ad_m.nla_tracks.new()
st_a = tr_a.strips.new("aim_to_shot", F(t_aim0), act_aim)
st_a.action_frame_end = 36.0
st_a.extrapolation = "HOLD_FORWARD"
tr_b = ad_m.nla_tracks.new()
st_b = tr_b.strips.new("aim_after_shot", F(T_BANG) + HIT_STOP, act_aim)
st_b.action_frame_start = 36.0
st_b.action_frame_end = act_aim.frame_range[1]
st_b.frame_start = F(T_BANG) + HIT_STOP
st_b.extrapolation = "HOLD_FORWARD"
print("AIM strips", st_a.frame_start, st_a.frame_end, st_b.frame_start, st_b.frame_end)
# recoil: damped pitch of the whole body about the feet (springy), baked per frame
for fr in range(F(T_BANG), F(T_BANG) + 22):
    dt = (fr - F(T_BANG)) / 30.0
    mgr.root.rotation_euler = (FX.spring(dt, -0.05, 6.5, 0.16), 0.0, aim_yaw)
    mgr.root.keyframe_insert("rotation_euler", frame=fr)
set_interp(mgr.root, "rotation_euler", "LINEAR")

# ---- shotgun on the Manager's right hand, mounted like S2/S4 (rot z = pi on the grip hook)
gun = asm.append("props/shotgun")
gun.variant("clay", True)
ghook = mgr.hook("gun_grip_R")
asm.attach(gun.root, ghook, rot=(0, 0, PI))
for fr in range(F(T_BANG), F(T_BANG) + 20):     # recoil: slides back along the barrel axis and muzzle flips up, damped
    dt = (fr - F(T_BANG)) / 30.0
    back = max(0.0, 0.07 * math.exp(-dt / 0.10) * math.cos(TAU * 7.0 * dt))
    flip = FX.spring(dt, -0.12, 6.0, 0.14)
    gun.root.location = (0, -back, 0)
    gun.root.rotation_euler = (flip, 0, PI)
    gun.root.keyframe_insert("location", frame=fr)
    gun.root.keyframe_insert("rotation_euler", frame=fr)
set_interp(gun.root, "location", "LINEAR")
set_interp(gun.root, "rotation_euler", "LINEAR")

# ---- effects: ear steam at the Manager's ear hooks, muzzle flash + smoke ring at the muzzles
steam = asm.fx("ear_steam", 7.5, dur=2.2)
for side in ("L", "R"):
    eh = steam.hook("ear_" + side)
    eh.parent = mgr.hook("steam_" + side)
    eh.matrix_parent_inverse.identity()
    eh.location = (0, 0, 0)
    eh.rotation_euler = (0, 0, 0)
steam.root.parent = None
for side in ("L", "R"):
    asm.fx("muzzle_flash", T_BANG, rot=(-PI / 2, 0, 0), parent=gun.hook("muzzle_" + side), dur=0.15)
    asm.fx("smoke_ring", T_BANG + 0.02, rot=(-PI / 2, 0, 0), parent=gun.hook("muzzle_" + side), dur=0.9)
asm.key_prop(gun.root, "p_trigger", T_BANG - 0.05, 0.0)
asm.key_prop(gun.root, "p_trigger", T_BANG, 1.0)
asm.key_prop(gun.root, "p_trigger", T_BANG + 0.3, 0.0)

# ---- hit effects on Gary's chest at the bang: squash/stretch spring, clay crumbs, dust puff
scn.frame_set(F(T_BANG))
update()
garm = gary.armature
chest_w = garm.matrix_world @ garm.pose.bones["chest"].head
hit_pt = Vector((chest_w.x + 0.10, chest_w.y, chest_w.z + 0.05))
print("HIT point", tuple(round(c, 3) for c in hit_pt))
for fr in range(F(T_BANG), F(T_BANG) + 20):
    dt = (fr - F(T_BANG)) / 30.0
    sz = 1.0 + FX.spring(dt, -0.07, 7.0, 0.14)
    sxy = 1.0 / math.sqrt(sz)
    gary.root.scale = (sxy, sxy, sz)
    gary.root.keyframe_insert("scale", frame=fr)
set_interp(gary.root, "scale", "LINEAR")
crumb_mat = mats["MAT_vfx_clay_shirt_yellow"]
FX.add_crumbs(tuple(hit_pt), (-1.0, 0.0, 0.0), T_BANG, crumb_mat, n=16, seed=5)
asm.fx("dust_cloud", T_BANG, loc=tuple(hit_pt), scale=0.5, dur=1.2)
asm.fx("impact_stars", T_BANG, loc=tuple(hit_pt), scale=0.3, dur=0.5)
scn.frame_set(1)
update()

# ------------------------------------------------------------------ camera director (baked, eased, handheld, impact shake)
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
D = FX.Director(T_BANG, seed=7)
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
D.simple(4.2, 6.0, (1.9, -0.6, 1.4), (1.6, Y_ACT, 1.3), (2.9, -0.2, 1.3), (2.6, Y_ACT, 1.2), lens=26)
D.simple(6.0, 6.8, (scr.x + 0.06, scr.y - 0.6, scr.z + 0.04), tuple(scr), (scr.x + 0.04, scr.y - 0.42, scr.z + 0.03), tuple(scr), lens=28, hh=0.6)
# reading + turning angry: scope screen and the Manager together in frame
D.simple(6.8, 8.5, (7.1, 0.35, 1.3), (7.95, 2.3, 1.3), (7.4, 0.9, 1.3), (8.2, 2.3, 1.4), lens=36, lens1=42)
D.simple(8.5, 9.2, (6.9, 0.4, 1.45), (7.75, 1.9, 1.4), (7.2, 0.75, 1.45), (7.65, 1.9, 1.4), lens=28)
# bang: wide hold (Gary and scope both in frame) then a fast eased zoom onto the scope screen
W_P, W_A = (6.4, -2.0, 1.5), (6.5, Y_ACT, 1.2)
Z_P, Z_A = (scr.x + 0.05, scr.y - 0.8, scr.z + 0.03), tuple(scr)
T_ZOOM0 = 9.45


def shot_final(u, t):
    k = FX.smooth5((t - T_ZOOM0) / (10.0 - 1.0 / 30.0 - T_ZOOM0))
    return FX.lerp3(W_P, Z_P, k), FX.lerp3(W_A, Z_A, k), 24.0 + (34.0 - 24.0) * k


D.add(9.2, 10.0, shot_final, hh=1.0)
D.bake()

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

asm.narr([(0.2, 2.2, "Gary wants faster data over copper."), (2.2, 4.2, "But faster means a shorter wire."),
          (4.2, 6.4, "So Gary tries stretching it one more meter."), (6.5, 7.5, "Bad idea.")])
asm.big(0.2, 1.0, "COPPER")
asm.big(2.2, 3.0, "5 m")
asm.big(3.0, 3.6, "2 m")
asm.big(3.6, 4.2, "1 m")
asm.big(4.4, 5.4, "STRETCH")
asm.big(T_BANG, T_BANG + 0.6, "BANG")
asm.card(0.0, 2.2, "NVL72 backplane: ~5,000 copper cables, 2 miles (SemiAnalysis)")
asm.card(2.2, 3.0, "4x25G passive twinax: >= 5 m (IEEE 802.3bj)")
asm.card(3.0, 3.6, "100G/lane passive twinax: >= 2 m (IEEE 802.3ck)")
asm.card(3.6, 4.2, "200G/lane passive twinax: >= 1 m, objective (IEEE 802.3dj)")
asm.card(6.0, 8.5, "BER = 0.5 erfc(Q/sqrt2), Q from simulated trace samples (illustrative eye model)")
asm.lab(6.0, 7.4, "EYE NEARLY CLOSED   BER RISING")
asm.lab(7.4, 8.5, "MANAGER READS THE SCOPE")
asm.fxn(0.0, 2.2, "[FX: fast fly-along, clay cable bundles, Pulse sprint trail]")
asm.fxn(2.2, 4.2, "[FX: Pulse fades as it travels (illustrative exp. decay)]")
asm.fxn(4.4, 6.0, "[FX: cables creak, strands pop, glow dims]")
asm.fxn(7.4, 8.5, "[FX: face turns red, vein pops, steam blast]")
asm.fxn(T_BANG, 9.8, "[FX: muzzle flash, smoke ring, hole decal]")
asm.wl("NVL72 BACKPLANE\nCOPPER CARTRIDGES", (-4.5, Y_WALL - 0.6, 3.5), 0.0, 2.2, size=0.4)
asm.wl("PULSE DECAYS", (X_A + 2.5, Y_ACT - 0.2, 2.9), 2.2, 4.2, size=0.3)
asm.wl("SCOPE", (slot.x - 0.1, slot.y - 0.2, 1.65), 6.0, 6.8, size=0.2)

# motion blur ready (the coordinator render preset decides whether it is on: draft off, hero on)
scn.render.use_motion_blur = True
scn.render.motion_blur_shutter = 0.5
asm.finalize(OUT)
