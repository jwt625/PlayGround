"""Scene S4 "CPO: keep it scorching hot" (film 30-40 s, scene-local 0-10 s), v1.1 assembly from the asset library.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s04_cpo.py -- scenes/v1/s04_cpo.blend

v1.1 changes (DevLog/v1/DevLog-003-scene-s04.md): OE lift-around-frame-and-descend path; PIC shot is a camera-attached rig that
cross-dissolves in and out (no black); three heat pulses; eggs are tossed in on ballistic arcs, bounce on the OE lid and splat
(yolk spring); overcooked eggs; lab bench + whiteboard (graph going back down) for the finale with all 16 eggs thrown
(head, near shoulder, chest, rest on the floor); eased camera moves with handheld noise; no monkeypatches of asm.

Scale decision (unchanged from v1.0):
  * Hardware group (HGX S4 board + XPU package + 16 optical engines) is scaled up by M = 6 about the origin (empty HW_GROUP).
  * Optical engines are scaled 0.5 (carrier 24 x 20 -> 12 x 10 mm hardware units = 72 x 60 mm world).
  * Eggs and all motion are computed in WORLD metres (not parented to HW_GROUP).
  * Humans stay at real size in the lab zone at x = +30 m.
"""
import math
import os
import random
import sys

import bpy
from mathutils import Euler, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm  # noqa: E402
import tex_gen as T  # noqa: E402

L = asm.L
F = asm.F
COMP = asm.COMP
PROJ = asm.PROJ
LIB = os.path.join(COMP, "materials_vfx", "materials_vfx_library.blend")
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "scenes/v1/s04_cpo.blend"
TEXDIR = os.path.join(PROJ, "assets", "generated_textures", "v1", "s04")
S2TEX = os.path.join(PROJ, "assets", "generated_textures", "v1", "s02")

PI = math.pi
GACC = 9.81
M = 6.0                       # hardware group scale
KC = 0.0659                   # crude unit -> metres
OE_SCALE = 0.5
HX = 30.0                     # lab zone origin (x)
EGG_OE = 0.5                  # fried egg scale on the OEs (the 'spread' shape key widens the 51 mm prop to about 100 mm; lid is 55 x 48 mm)
FLY_EXAG = 3.5                # eggs thrown in the lab grow to 3.5x so they read from 4.5 m (cartoon exaggeration)
T_BANG = 8.9
OVERCOOK_BROWN = 0.72         # Brown Extent: at p_cook 1 only a small crescent of white stays next to the yolk (default 0.42)

rnd = random.Random(21)


# ----------------------------------------------------------------------------- small helpers
def clamp01(x):
    return min(1.0, max(0.0, x))


def smooth(a, b, x):
    u = clamp01((x - a) / (b - a))
    return u * u * u * (u * (u * 6 - 15) + 10)       # smootherstep


def cw(p, k=KC):
    return (p[0] * k, p[1] * k, p[2] * k)


def frames_between(t0, t1):
    return range(F(t0), F(t1) + 1)


def T_of(f):
    return (f - 1) / 30.0


def collect_tree(coll):
    out = list(coll.objects)
    for ch in coll.children:
        out += collect_tree(ch)
    return out


def driver_multi(owner_id, path, vars_, expr, index=-1):
    """vars_ = {name: (root_object, custom_prop)}."""
    fc = owner_id.driver_add(path) if index < 0 else owner_id.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    for nm, (root, prop) in vars_.items():
        v = d.variables.new()
        v.name = nm
        v.type = "SINGLE_PROP"
        v.targets[0].id = root
        v.targets[0].data_path = '["%s"]' % prop
    d.expression = expr
    return fc


def group_node(m):
    return next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeGroup")


def load_mats(names):
    with bpy.data.libraries.load(LIB, link=False) as (src, dst):
        dst.materials = [n for n in names if n in src.materials]
    return {n: bpy.data.materials[n] for n in names}


def key_fc_interp(obj_or_id, path, interp):
    ad = obj_or_id.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path == path:
                for kp in fc.keyframe_points:
                    kp.interpolation = interp


def hide_always(asset):
    for o in asset.objs:
        o.hide_render = True
        o.hide_viewport = True


# ----------------------------------------------------------------------------- scene, world, rigs
asm.new_scene(preset="standard", hold_z=0.06)   # macro scene: caption plane 0.06 m from the camera
scn = bpy.context.scene
L.CAM.data.clip_start = 0.004
L.CAM.data.clip_end = 400


def floor(name, center, size, tile, z, color1, color2):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(center[0], center[1], z))
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0], size[1], 1)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = size[0] / tile / 2.0     # tile = side of one square
    chk.inputs["Color1"].default_value = (*color1, 1)
    chk.inputs["Color2"].default_value = (*color2, 1)
    uv = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(uv.outputs["UV"], chk.inputs["Vector"])
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Roughness"].default_value = 0.95
    nt.links.new(chk.outputs["Color"], bs.inputs["Base Color"])
    o.data.materials.append(m)
    return o, m


fl_hw, _ = floor("floor_hw", (0, 0), (60, 60), 1.0, -0.09, (0.52, 0.54, 0.46), (0.46, 0.49, 0.41))
L.V(fl_hw, 0, 0, 8.2)

rig_hw = asm.rig("macro_studio", scale=5.0)
rig_hw.root.location = (0, 0, 0)
rig_hw.root["p_energy"] = 0.45
asm.show(rig_hw, 0.0, 8.2)

# ----------------------------------------------------------------------------- hardware group
G = L.empty("HW_GROUP", (0, 0, 0))
G.scale = (M, M, M)

hgx = asm.append("packaging/hgx_baseboard", only=["ASSET_hgx_baseboard"])
hgx_root = bpy.data.objects["ROOT_hgx_baseboard"]
xpu_root = bpy.data.objects["ROOT_xpu_package_rubin_style"]
hgx_root.parent = G
for v, on in (("hgx_board", False), ("hgx_oam_modules", False), ("s4_board", True), ("s4_socket_pads", True),
              ("s4_xpu_installed", True)):
    hgx.variant(v, on)
asm.show(hgx, 0.0, 8.2)
bpy.context.view_layer.update()

# bloom over-glow at draft (v1.0 issue): cap the emission of every heat-glow material on the board/XPU package
_seen = set()
for _o in collect_tree(hgx.coll):
    if _o.type != "MESH":
        continue
    for _m in _o.data.materials:
        if _m is None or _m in _seen or not _m.use_nodes:
            continue
        _seen.add(_m)
        for _n in _m.node_tree.nodes:
            if _n.bl_idname == "ShaderNodeGroup" and "Strength Max" in _n.inputs:
                _n.inputs["Strength Max"].default_value = _n.inputs["Strength Max"].default_value * 0.4
                print("XPU glow strength capped:", _m.name, _n.inputs["Strength Max"].default_value)


for _m in bpy.data.materials:                       # die emission driver is v*4 (v = p_heat): cap at v*1.2 (blown-out bloom at draft)
    if _m.name.startswith("MAT_packaging_xpu_die_") and _m.node_tree.animation_data:
        for _d in _m.node_tree.animation_data.drivers:
            if "Emission Strength" in _d.data_path and _d.driver.expression == "v*4":
                _d.driver.expression = "v*1.2"


_steel = bpy.data.materials.get("MAT_packaging_steel")      # glossy retention frame reflects the key light as a huge white blob
if _steel is not None:
    for _n in _steel.node_tree.nodes:
        if _n.bl_idname == "ShaderNodeBsdfPrincipled":
            _n.inputs["Roughness"].default_value = 0.6
            _n.inputs["Metallic"].default_value = 0.6


def to_G(obj):
    return G.matrix_world.inverted() @ obj.matrix_world


# ----------------------------------------------------------------------------- optical engines (16)
oe_src = asm.append("photonics/oe_module_cpo", only=["ASSET_oe_module_cpo"])
oe_src.variant("closed", True)
for ch in list(oe_src.coll.children):
    if ch.name.startswith("VARIANT_open"):
        for o in collect_tree(ch):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(ch)
oe_src.objs = collect_tree(oe_src.coll)
oe_src.root = bpy.data.objects["ROOT_oe_module_cpo"]

MATS = load_mats(["MAT_vfx_egg_white", "MAT_vfx_egg_yolk", "MAT_vfx_heat_glow", "MAT_vfx_clay_egg_white"])


def lid_of(asset):
    return next(o for o in asset.objs if o.name.startswith("oe_module_cpo_lid_body"))


hooks_oe = {}
for side, sname in ((-1, "L"), (1, "R")):
    for r in range(8):
        hooks_oe[(side, r)] = bpy.data.objects["HOOK_oe_%s_%d" % (sname, r)]

OE_FLY = 1.0       # seconds of flight
OE_SETTLE = 0.4    # seconds of settle after touchdown
oes = []           # (side, r, asset, pos_G, yaw)
for n, ((side, r), hook) in enumerate(sorted(hooks_oe.items(), key=lambda kv: (kv[0][0], kv[0][1]))):
    a = oe_src if n == 0 else asm.clone(oe_src, "ASSET_oe_module_cpo_%d" % n)
    asm.show(a, 0.0, 8.2)
    mg = to_G(hook)
    pos = mg.translation.copy()
    yaw = mg.to_euler().z
    a.root.parent = G
    a.root.matrix_parent_inverse.identity()
    a.root.scale = (OE_SCALE,) * 3
    a.root.rotation_euler = (0, 0, yaw)
    a.root["p_heat"] = 0.0
    lid = lid_of(a)
    m = MATS["MAT_vfx_heat_glow"].copy()
    m.name = "MAT_oe_heat_%02d" % n
    gn = group_node(m)
    gn.inputs["Base Color"].default_value = (0.35, 0.37, 0.42, 1)
    gn.inputs["Metallic"].default_value = 0.85
    gn.inputs["Strength Max"].default_value = 1.5      # v1.0: 1.3 (OE heat faint); at 2.0+ the comp bloom washes the eggs on the lid
    gn.inputs["Roughness"].default_value = 0.3
    driver_multi(m.node_tree, 'nodes["%s"].inputs["p_heat"].default_value' % gn.name, {"p": (a.root, "p_heat")}, "p")
    lid.data = lid.data.copy()
    lid.data.materials.clear()
    lid.data.materials.append(m)
    oes.append((side, r, a, pos, yaw))
bpy.context.view_layer.update()


def oe_path(side, r, pos, yaw, s):
    """OE pose at normalised flight time s in 0..1 (+ settle for s > 1). Returns (loc in G units, euler).
    Lift straight up, swing around the stiffener frame (outward arc in y), descend onto the hook with a yaw/roll rock and a damped bounce."""
    pw = Vector(pos) * M
    s_ = clamp01(s)
    lift = 0.30                                       # world metres of clearance above the stiffener frame
    ph = smooth(0.0, 0.30, s_) * (1.0 - smooth(0.62, 1.0, s_))
    z = pw.z + lift * ph
    adv = smooth(0.14, 0.88, s_)                      # travel along x: starts after the lift, finishes during the descent
    x = pw.x + side * 0.50 * (1.0 - adv)
    swing = (1.0 if r % 2 == 0 else -1.0) * (0.14 + 0.015 * r)
    y = pw.y + swing * math.sin(PI * adv) * (1.0 - 0.3 * adv)
    yaw_o = yaw + side * 0.55 * math.sin(PI * adv) * (1.0 - adv)
    roll = 0.30 * math.sin(2 * PI * adv) * (1.0 - adv) * (1.0 if side < 0 else -1.0)
    pitch = 0.22 * math.sin(PI * adv) * (1.0 - adv)
    if s > 1.0:                                       # settle after touchdown
        u = (s - 1.0) * OE_FLY
        z += 0.012 * abs(math.cos(2 * PI * 3.0 * u)) * math.exp(-u / 0.12)
        roll += 0.05 * math.sin(2 * PI * 4.5 * u) * math.exp(-u / 0.14)
        pitch += 0.04 * math.sin(2 * PI * 3.5 * u + 1.0) * math.exp(-u / 0.14)
    return (x / M, y / M, z / M), (pitch, roll, yaw_o)


for (side, r, a, pos, yaw) in oes:
    n = sorted(hooks_oe).index((side, r))
    t_in = 0.20 + 0.035 * r + (0.0 if side < 0 else 0.018)
    t_end = t_in + OE_FLY + OE_SETTLE
    s0_loc, s0_rot = oe_path(side, r, pos, yaw, 0.0)
    L.K(a.root, 1, loc=s0_loc, rot=s0_rot, interp="LINEAR")
    for f in range(F(t_in), F(t_end) + 1):
        loc, rot = oe_path(side, r, pos, yaw, (T_of(f) - t_in) / OE_FLY)
        L.K(a.root, f, loc=loc, rot=rot, interp="LINEAR")
    # heating (cyan -> orange -> red) after the pull-back starts, staggered 0.025 s
    th = 4.75 + 0.025 * n
    asm.key_prop(a.root, "p_heat", 0.0, 0.0)
    asm.key_prop(a.root, "p_heat", th, 0.0)
    asm.key_prop(a.root, "p_heat", th + 0.35, 0.6)
    asm.key_prop(a.root, "p_heat", th + 0.75, 1.0)

# ----------------------------------------------------------------------------- XPU heat (p_heat on the package root)
bars_h = [rnd.uniform(0.3, 3.0) for _ in range(18)]
asm.key_prop(xpu_root, "p_heat", 0.0, 0.0)
asm.key_prop(xpu_root, "p_heat", 1.4, 0.25)
for q, h in enumerate(bars_h):
    asm.key_prop(xpu_root, "p_heat", 1.4 + q * (1.2 / 17), 0.35 + 0.65 * (h / 3.0))
asm.key_prop(xpu_root, "p_heat", 2.7, 0.9)
asm.key_prop(xpu_root, "p_heat", 8.2, 1.0)

pw_ = L.box("powerbar", cw((-0.8, -7.0, 0.5)), (KC * 0.5, KC * 0.5, KC), (0.95, 0.8, 0.15), emit=1.0)
tb = L.box("tempbar", cw((0.8, -7.0, 0.5)), (KC * 0.5, KC * 0.5, KC), (0.15, 0.3, 0.8), emit=1.0)
tr = L.box("tempred", cw((0.8, -7.0, 0.5)), (KC * 0.52, KC * 0.52, KC), (0.85, 0.15, 0.12), emit=3.0)
for o in (pw_, tb, tr):
    L.V(o, 0, 1.4, 2.6)
for q, h in enumerate(bars_h):
    t = 1.4 + q * (1.2 / 17)
    f = F(t)
    L.K(pw_, f, loc=cw((-0.8, -7.0, h / 2)), scale=(KC * 0.5, KC * 0.5, KC * h), interp="LINEAR")
    L.K(tb, f, loc=cw((0.8, -7.0, h / 2)), scale=(KC * 0.5, KC * 0.5, KC * h), interp="LINEAR")
    hr = max(h - 1.6, 0.02)
    L.K(tr, f, loc=cw((0.8, -7.0, hr / 2)), scale=(KC * 0.52, KC * 0.52, KC * hr), interp="LINEAR")
asm.wl("XPU POWER", cw((-0.8, -7.0, 3.4)), 1.4, 2.6, size=0.28 * KC)
asm.wl("TEMPERATURE", cw((0.8, -7.0, 3.8)), 1.4, 2.6, size=0.28 * KC)
asm.wl("XPU: 4 GPU DIES + 16 HBM", cw((0, 0, 1.4)), 1.0, 3.0, size=0.5 * KC)

for q, (dx, dy) in enumerate([(-0.15, 0.05), (0.12, -0.04), (0.0, 0.0), (-0.3, -0.1), (0.3, 0.08)]):
    asm.fx("smoke_puff", 1.5 + 0.1 * q, loc=(dx, dy, 0.03), rot=(0, 0, 0), scale=0.5, intensity=1.0, dur=1.0)
asm.fx("heat_shimmer", 1.4, loc=(0, -0.28, 0.0), rot=(0, 0, 0), scale=1.4, dur=1.3)

# ----------------------------------------------------------------------------- PIC close-up: camera-attached rig, cross-dissolved
# PIC_RIG is a child of the camera (so the dissolve needs no second camera and no black quad); PIC_ORBIT carries the slow oblique
# drift and the heat-shimmer wobble; the chip, its floor and its macro_studio lights live under it. Every PIC material is wrapped in
# Mix(Transparent, original) driven by PIC_RIG["p_alpha"] (keyed): a true alpha cross-dissolve against the OE zoom behind it.
PIC_D = 0.12                  # rig distance in front of the camera (m); lens 85 -> field width 0.051 m
PIC_S = 0.0062                # chip scale: 7.5 asset units -> 0.046 m
LENS_PIC = 85.0
PIC_E = 0.012                 # PIC key-light energy factor (p_energy), calibrated by test renders
rig_root = L.empty("PIC_RIG", (0, 0, -PIC_D))
rig_root.parent = L.CAM
rig_root.matrix_parent_inverse.identity()
rig_root["p_alpha"] = 0.0
orbit = L.empty("PIC_ORBIT", (0, 0, 0))
orbit.parent = rig_root
orbit.matrix_parent_inverse.identity()

pic = asm.append("photonics/microring_array_closeup", only=["ASSET_microring_array_closeup"])
pic.variant("real_size", False)
pic.variant("scaled_x15000", True)
pic.root.parent = orbit
pic.root.matrix_parent_inverse.identity()
pic.root.location = (0, 0, 0)
pic.root.scale = (PIC_S,) * 3
asm.show(pic, 3.0, 5.0)

pic_floor, pic_floor_mat = floor("floor_pic", (0, 0), (4.0, 4.0), 0.05, -0.0004, (0.52, 0.54, 0.46), (0.46, 0.49, 0.41))
pic_floor.parent = orbit
pic_floor.matrix_parent_inverse.identity()
pic_floor.location = (0, 0, -0.0004)
L.V(pic_floor, 0, 3.0, 5.0)

rig_pic = asm.rig("macro_studio", scale=0.2)
rig_pic.root.parent = rig_root
rig_pic.root.matrix_parent_inverse.identity()
rig_pic.root.location = (0, 0, 0.02)
rig_pic.root["p_energy"] = PIC_E
asm.show(rig_pic, 3.0, 5.0)
pic_lights = [o for o in rig_pic.objs if o.type == "LIGHT"]
for _o in list(bpy.data.objects):          # cyclorama backdrops show their curved edge; not wanted
    if "backdrop" in _o.name:
        bpy.data.objects.remove(_o, do_unlink=True)

# light linking: the PIC lights only light the PIC (EEVEE Next supports light linking in 4.2)
pic_recv = bpy.data.collections.new("PIC_RECEIVERS")
scn.collection.children.link(pic_recv)
for _o in collect_tree(pic.coll) + [pic_floor]:
    if _o.type == "MESH" and _o.name not in pic_recv.objects:
        pic_recv.objects.link(_o)
for _o in pic_lights:
    try:
        _o.light_linking.receiver_collection = pic_recv
    except Exception as e:                      # noqa: BLE001
        print("light linking unavailable:", e)


def alpha_wrap(mat):
    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output), None)
    if out is None or not out.inputs["Surface"].links:
        return
    src = out.inputs["Surface"].links[0].from_socket
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr_ = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(tr_.outputs[0], mix.inputs[1])
    nt.links.new(src, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    driver_multi(nt, 'nodes["%s"].inputs[0].default_value' % mix.name, {"a": (rig_root, "p_alpha")}, "a")


_wrapped = set()
for _o in collect_tree(pic.coll) + [pic_floor]:
    if _o.type != "MESH":
        continue
    for _m in _o.data.materials:
        if _m is not None and _m.use_nodes and _m not in _wrapped:
            _wrapped.add(_m)
            alpha_wrap(_m)

# alpha: dissolve in 3.15-3.5, hold, out 4.5-4.9 (lights follow alpha**2 so the board is not relit by the PIC lights during the dissolve)
for t, a_ in ((0.0, 0.0), (3.15, 0.0), (3.5, 1.0), (4.52, 1.0), (4.9, 0.0)):
    asm.key_prop(rig_root, "p_alpha", t, a_, "BEZIER")
    asm.key_prop(rig_pic.root, "p_energy", t, PIC_E * a_ * a_ + 1e-6, "BEZIER")

# three heat pulses (3.52-3.78, 3.88-4.14, 4.24-4.50): the wave sweeps across rings and substrate slabs left to right, a flat gap
# follows each one. Distortion: the rig wobbles (rotation + scale ripple) while a front crosses; no global light change.
PULSES = [(3.52, 3.78), (3.88, 4.14), (4.24, 4.50)]
asm.key_prop(pic.root, "p_wave_pos", 0.0, -0.3)
for (ta, tb_) in PULSES:
    asm.key_prop(pic.root, "p_wave_pos", ta - 0.05, -0.3, "LINEAR")
    asm.key_prop(pic.root, "p_wave_pos", ta, -0.3, "LINEAR")
    asm.key_prop(pic.root, "p_wave_pos", tb_, 1.3, "LINEAR")
asm.key_prop(pic.root, "p_wave_pos", 4.55, -0.3, "LINEAR")

orbit.rotation_mode = "XYZ"
for f in frames_between(3.0, 5.0):
    t = T_of(f)
    k = smooth(3.0, 4.9, t)
    rx = math.radians(-34.0 + 10.0 * k)
    rz = math.radians(-8.0 + 16.0 * k)
    env = 0.0
    for (ta, tb_) in PULSES:
        if ta <= t <= tb_ + 0.12:
            env = max(env, math.sin(PI * clamp01((t - ta) / (tb_ - ta + 0.12))) ** 2)
    wob = env * math.sin(2 * PI * 9.0 * t)
    wob2 = env * math.sin(2 * PI * 7.0 * t + 1.3)
    orbit.rotation_euler = (rx + math.radians(0.7) * wob, math.radians(0.6) * wob2, rz + math.radians(0.8) * wob2)
    sc_ = 1.0 + 0.012 * wob
    orbit.scale = (sc_, 1.0 / sc_, 1.0)
    orbit.keyframe_insert("rotation_euler", frame=f)
    orbit.keyframe_insert("scale", frame=f)
for dp in ("rotation_euler", "scale"):
    key_fc_interp(orbit, dp, "LINEAR")

pl1 = asm.wl("PIC: MICRORING MODULATORS", (0, 0, 0), 3.5, 4.5, size=0.0016)
pl2 = asm.wl("HEAT WAVE ->", (0, 0, 0), 3.5, 4.5, size=0.0016, color=(1.0, 0.6, 0.3))
for _o, yy in ((pl1, 0.0235), (pl2, -0.0235)):
    _o.parent = orbit
    _o.matrix_parent_inverse.identity()
    _o.location = (0, yy, 0.004)

# ----------------------------------------------------------------------------- eggs
egg_src = asm.append("props/eggs_plate", only=["ASSET_fried_egg"])
egg_obj_src = next(o for o in egg_src.objs if o.name == "fried_egg")
hide_always(egg_src)      # the library copy stays hidden; every egg below is its own copy


def make_egg(name, cook=None, splat=None, parent=None, scale=1.0, brown=OVERCOOK_BROWN):
    """Fried egg with its own mesh/material copies. Root empty carries p_cook, p_splat (shape 'spread', 0..1.15) and p_wob (yolk spring).
    cook/splat None -> animated by key_splat (initial 0); a number -> constant (already fried)."""
    root = L.empty(name, (0, 0, 0))
    root.scale = (scale,) * 3
    root.rotation_mode = "XYZ"
    root["p_cook"] = 0.0 if cook is None else float(cook)
    root["p_splat"] = 0.0 if splat is None else float(splat)
    root["p_wob"] = 0.0
    mesh = egg_obj_src.copy()
    mesh.data = egg_obj_src.data.copy()
    mesh.name = name + "_mesh"
    mesh.hide_render = False
    mesh.hide_viewport = False
    scn.collection.objects.link(mesh)
    mesh.parent = root
    mesh.matrix_parent_inverse.identity()
    mesh.location = (0, 0, 0)
    mesh.rotation_euler = (0, 0, 0)
    mesh.scale = (1, 1, 1)
    if parent is not None:
        root.parent = parent
        root.matrix_parent_inverse.identity()
    for slot, key in zip(mesh.material_slots, ("MAT_vfx_egg_white", "MAT_vfx_egg_yolk")):
        m = MATS[key].copy()
        m.name = "%s_%s" % (key, name)
        slot.material = m
        gn = group_node(m)
        gn.inputs["Brown Extent"].default_value = brown
        driver_multi(m.node_tree, 'nodes["%s"].inputs["p_cook"].default_value' % gn.name, {"p": (root, "p_cook")}, "p")
    sk = mesh.data.shape_keys
    sk.key_blocks["spread"].slider_max = 1.3
    sk.key_blocks["yolk_dome"].slider_max = 1.6
    exprs = (("spread", "min(1.15,max(0,sp))"),
             ("edge_crisp", "min(1,max(0,(p-0.3)/0.6))"),
             ("yolk_dome", "min(1,max(0,(p-0.15)/0.6))+0.45*w"),
             ("bubbles", "min(1,max(0,(p-0.12)/0.2))*(1-0.5*min(1,max(0,(p-0.7)/0.3)))"))
    for nm, ex in exprs:
        driver_multi(sk, 'key_blocks["%s"].value' % nm, {"p": (root, "p_cook"), "sp": (root, "p_splat"), "w": (root, "p_wob")}, ex)
    return root, mesh


def key_splat(root, t_splat, fresh=True, cook_dur=0.8, wob_amp=1.0):
    """Yolk wobble spring on p_wob from t_splat; if fresh also the splat spring on p_splat (overshoot then settle) and the
    cook ramp 0 -> 1 -> overcook."""
    asm.key_prop(root, "p_wob", 0.0, 0.0)
    for f in range(F(t_splat), F(t_splat) + 30):
        u = (f - F(t_splat)) / 30.0
        root["p_wob"] = wob_amp * math.exp(-u / 0.22) * math.cos(2 * PI * 6.0 * u)
        root.keyframe_insert('["p_wob"]', frame=f)
    asm.key_prop(root, "p_wob", t_splat + 1.1, 0.0)
    key_fc_interp(root, '["p_wob"]', "LINEAR")
    if not fresh:
        return
    asm.key_prop(root, "p_splat", 0.0, 0.0)
    asm.key_prop(root, "p_splat", t_splat - 1 / 30.0, 0.0)
    for f in range(F(t_splat), F(t_splat) + 14):
        u = (f - F(t_splat)) / 30.0
        root["p_splat"] = min(1.15, 1.0 - math.exp(-u / 0.06) * math.cos(2 * PI * 4.5 * u))
        root.keyframe_insert('["p_splat"]', frame=f)
    key_fc_interp(root, '["p_splat"]', "LINEAR")
    asm.key_prop(root, "p_cook", 0.0, 0.0)
    asm.key_prop(root, "p_cook", t_splat, 0.0)
    asm.key_prop(root, "p_cook", t_splat + cook_dur, 1.0)


def ballistic(p0, v0, t):
    return Vector((p0.x + v0.x * t, p0.y + v0.y * t, p0.z + v0.z * t - 0.5 * GACC * t * t))


# raw (shell) egg proxy: clay ovoid that flies and bounces once, replaced by the splat on the second contact
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=20, ring_count=12, location=(0, 0, -50))
ovoid_proto = bpy.context.active_object
ovoid_proto.name = "ovoid_proto"
for p_ in ovoid_proto.data.polygons:
    p_.use_smooth = True
_om = bpy.data.materials.new("MAT_s04_shell")       # smooth cream clay (the library clay bump is sized for human scale and reads as cauliflower here)
_om.use_nodes = True
_ob = _om.node_tree.nodes["Principled BSDF"]
_ob.inputs["Base Color"].default_value = (0.93, 0.86, 0.70, 1)
_ob.inputs["Roughness"].default_value = 0.55
ovoid_proto.data.materials.append(_om)
ovoid_proto.hide_render = True
ovoid_proto.hide_viewport = True

# settled OE poses (needs the OE keyframes): lid tops in world coordinates
scn.frame_set(F(2.2))
oe_top = {}
for (side, r, a, pos, yaw) in oes:
    hk = next(o for o in a.objs if o.name.startswith("HOOK_lid_top"))
    oe_top[(side, r)] = hk.matrix_world.translation.copy()
scn.frame_set(1)
print("OE lid tops (world):", {k: tuple(round(x, 3) for x in v) for k, v in list(oe_top.items())[:3]})

OV_A, OV_B = 0.021, 0.015       # ovoid semi-axes (m)
egg_oe = []
for n, (side, r, a, pos, yaw) in enumerate(oes):
    top = oe_top[(side, r)]
    t_land = (5.62 + 0.15 * r) if side < 0 else (6.85 + 0.10 * r)
    Tfl = 0.44
    L0 = top + Vector((0, 0, OV_B))
    P0 = L0 + Vector((side * 0.10, 0.75 * (1.0 if r < 4 else -1.0) + 0.0 * r, 0.62))   # tossed in from the far end of the column, above the camera line
    v0 = (L0 - P0) / Tfl + Vector((0, 0, 0.5 * GACC * Tfl))
    t_launch = t_land - Tfl
    vimp = v0 + Vector((0, 0, -GACC * Tfl))
    vh = Vector((vimp.x * 0.05, vimp.y * 0.05, 0.0))
    vz2 = -0.30 * vimp.z
    th = 2 * vz2 / GACC
    t_splat = t_land + th
    ov = ovoid_proto.copy()
    ov.name = "ovoid_%02d" % n
    scn.collection.objects.link(ov)
    ov.hide_render = False
    ov.hide_viewport = False
    ov.scale = (OV_A, OV_B, OV_B)
    ov.rotation_mode = "XYZ"
    w = Vector((rnd.uniform(-14, 14), rnd.uniform(-14, 14), rnd.uniform(-14, 14)))
    r0 = Euler((rnd.uniform(0, 6), rnd.uniform(0, 6), rnd.uniform(0, 6)))
    for f in range(F(t_launch), F(t_splat) + 1):
        t = T_of(f)
        p = ballistic(P0, v0, t - t_launch) if t <= t_land else ballistic(L0, Vector((vh.x, vh.y, vz2)), t - t_land)
        k = clamp01((t - t_land) / max(th, 1e-3))
        ov.location = p
        ov.rotation_euler = (r0.x + w.x * (t - t_launch) * (1 - 0.8 * k), r0.y + w.y * (t - t_launch) * (1 - 0.8 * k),
                             r0.z + w.z * (t - t_launch) * (1 - 0.8 * k))
        ov.keyframe_insert("location", frame=f)
        ov.keyframe_insert("rotation_euler", frame=f)
    for dp in ("location", "rotation_euler"):
        key_fc_interp(ov, dp, "LINEAR")
    L.V(ov, 0, t_launch, t_splat)
    root, mesh = make_egg("egg_%02d" % n, scale=EGG_OE)
    ps = L0 + vh * th
    root.location = (ps.x, ps.y, top.z + 0.001)
    root.rotation_euler = (0, 0, rnd.uniform(0, 2 * PI))
    key_splat(root, t_splat, cook_dur=0.6)
    L.V(mesh, 0, t_splat, 8.2)
    # sizzle steam: one puff per egg and a second on every 2nd egg (stateless GN particles, 1.4 s each; render cost)
    asm.fx("steam", t_splat + 0.35, loc=(ps.x, ps.y, top.z + 0.012), scale=0.06, dur=1.4)
    if n % 2 == 0:
        asm.fx("steam", t_splat + 1.6, loc=(ps.x, ps.y, top.z + 0.012), scale=0.06, dur=1.4)
    egg_oe.append((root, mesh, t_splat))

# ----------------------------------------------------------------------------- lab zone (8.2-10 s): bench, scope, whiteboard, Gary, Manager
BX, BY = HX + 2.7, -0.8
WBX, WBY, WB_Z0 = HX + 1.4, 1.5, 0.9
G_POS = (HX + 1.3, -1.5, 0.0)
M_POS = (HX - 0.4, 0.8, 0.0)

fl_lab, _ = floor("floor_lab", (HX + 1.0, 0.0), (40, 40), 1.0, 0.0, (0.72, 0.74, 0.78), (0.55, 0.58, 0.64))
L.V(fl_lab, 0, 8.2, 10)
rig_lab = asm.rig("daylight", scale=1.6, loc=(HX + 1.5, 0.0, 0.0))
asm.show(rig_lab, 8.2, 10.0)

bench = asm.append("lab_office/lab_bench")
asm.place(bench.root, (BX, BY, 0.0), yaw=0.0)
scope = asm.append("lab_office/bench_oscilloscope")
bpy.context.view_layer.update()
slot = bench.hook("scope_slot").matrix_world.translation
asm.place(scope.root, (slot.x, slot.y, slot.z), yaw=0.0)
wb = asm.append("lab_office/whiteboard_big")
asm.place(wb.root, (WBX, WBY, WB_Z0), yaw=0.0)
for a_ in (bench, scope, wb):
    asm.show(a_, 8.2, 10.0)


def plug_image(mat, fac_node, first_png, n=1, start=1, sequence=True):
    nt = mat.node_tree
    tex = nt.nodes["SCREEN_IMAGE"]
    img = bpy.data.images.load(first_png)
    if sequence:
        img.source = "SEQUENCE"
    img.filepath = bpy.path.relpath(first_png, start=os.path.dirname(os.path.abspath(OUT)))
    tex.image = img
    tex.interpolation = "Closest"
    if sequence:
        iu = tex.image_user
        iu.frame_duration = n
        iu.frame_start = start
        iu.frame_offset = 0
        iu.use_auto_refresh = True
    nt.nodes[fac_node].outputs[0].default_value = 1.0


# scope: the recovered eye from S2 (last frame of the S2 sequence, read only)
scr = next(o for o in scope.objs if o.name == "bench_oscilloscope_screen")
plug_image(scr.material_slots[0].material, "SCREEN_FAC", os.path.join(S2TEX, "eye_s2", "eye_s2_0300.png"), sequence=False)

# whiteboard graph: starts at the S2 end state (p = 1) and runs back DOWN along the same curves (tips retreat to p = 0.2)
G_F0 = F(8.2)
GN = 300 - G_F0 + 1


def GPROG(t):
    return 1.0 - 0.8 * smooth(8.45, 9.85, t)


gdir = os.path.join(TEXDIR, "graph_s4")
if not os.path.exists(os.path.join(gdir, "graph_s4_%04d.png" % GN)):
    T.graph_sequence(gdir, "graph_s4", [GPROG(T_of(f)) for f in range(G_F0, 301)])
wbs = next(o for o in wb.objs if o.name == "whiteboard_big_screen")
plug_image(wbs.material_slots[0].material, "BOARD_FAC", os.path.join(gdir, "graph_s4_0001.png"), n=GN, start=G_F0)

gary = asm.append("characters/gary", actions=True)
mgr = asm.append("characters/manager", actions=True)
asm.show(gary, 8.2, 10)
asm.show(mgr, 8.2, 10)
gy = asm.yaw_to(G_POS, M_POS)
my = asm.yaw_to(M_POS, G_POS)
asm.place(gary.root, G_POS, yaw=gy)
asm.place(mgr.root, M_POS, yaw=my)

T_REL = T_BANG + 11.0 / 30.0 / 1.5
asm.play(gary, "hold_plate", 7.0, hold=True, repeat=1)
asm.play(gary, "throw_up", T_BANG, speed=1.5, blend_in=2)
asm.play(gary, "topple_back", T_BANG + 0.3, speed=2.2, blend_in=3)
ASPEED = 0.9
asm.play(mgr, "aim_gun", T_BANG - 36.0 / 30.0 / ASPEED, speed=ASPEED)
asm.key_prop(mgr.root, "p_anger", 0.0, 0.2)
asm.key_prop(mgr.root, "p_anger", 8.2, 0.4)
asm.key_prop(mgr.root, "p_anger", T_BANG, 1.0)
asm.key_prop(gary.root, "p_expr_scared", 0.0, 0.0)
asm.key_prop(gary.root, "p_expr_scared", T_BANG, 1.0)

# holes: schedule from the brief, r = max(0.3, 0.9 ** ((T_now - T_shot) / 1.5)), stepped every 1.5 s; hole 4 at the bang
HOLE_FILM = {"p_hole_1_radius": 9.2, "p_hole_2_radius": 19.2, "p_head_hole_radius": 28.95}
T0_FILM = 30.0
for prop, shot in HOLE_FILM.items():
    asm.key_prop(gary.root, prop, 0.0, max(0.3, 0.9 ** ((T0_FILM - shot) / 1.5)), "CONSTANT")
    k = math.floor((T0_FILM - shot) / 1.5) + 1
    while True:
        ts = shot + k * 1.5 - T0_FILM
        if ts > 10.0:
            break
        asm.key_prop(gary.root, prop, ts, max(0.3, 0.9 ** k), "CONSTANT")
        k += 1
asm.key_prop(gary.root, "p_hole_4_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_hole_4_radius", T_BANG, 1.0, "CONSTANT")

gun = asm.append("props/shotgun")
asm.show(gun, 8.2, 10)
grip = next(o for o in mgr.objs if o.name.startswith("HOOK_gun_grip_R"))
asm.attach(gun.root, grip, rot=(0, 0, PI))
asm.fx("muzzle_flash", T_BANG, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), scale=1.0, parent=next(o for o in gun.objs if o.name.startswith("HOOK_muzzle_L")), dur=0.12)
asm.fx("smoke_ring", T_BANG + 0.05, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), scale=1.0, parent=next(o for o in gun.objs if o.name.startswith("HOOK_muzzle_R")), dur=0.9)

# plate carried at the midpoint of Gary's two hand hooks, with a stack of 16 overcooked fried eggs
plate = asm.append("props/eggs_plate", only=["ASSET_plate_with_eggs"])
asm.show(plate, 8.2, 10)
for o in [o for o in plate.objs if o.name.startswith("plate_with_eggs_egg_")]:
    bpy.data.objects.remove(o, do_unlink=True)
mid = L.empty("plate_mount", (0, 0, 0))
for tgt_name, infl in (("HOOK_hand_L", 1.0), ("HOOK_hand_R", 0.5)):
    tgt = next(o for o in gary.objs if o.name.startswith(tgt_name))
    c = mid.constraints.new("COPY_LOCATION")
    c.target = tgt
    c.influence = infl
mid.rotation_euler = (0, 0, gy)
plate.root.parent = mid
plate.root.matrix_parent_inverse.identity()
plate.root.location = (0, 0, 0)
plate.root.rotation_euler = (0, 0, 0)

N_EGG = 16
stack = []
for k in range(N_EGG):
    layer, idx = divmod(k, 3)
    ang = 2 * PI * idx / 3 + 0.9 * layer
    rr = 0.045 if layer < 5 else 0.0
    lp = Vector((rr * math.cos(ang) + rnd.uniform(-0.006, 0.006), rr * math.sin(ang) + rnd.uniform(-0.006, 0.006), 0.014 + 0.013 * layer))
    root, mesh = make_egg("plate_egg_%02d" % k, cook=1.0, splat=1.0, parent=plate.root, brown=0.5)
    root.location = lp
    root.rotation_euler = (0, 0, rnd.uniform(0, 2 * PI))
    L.V(mesh, 0, 8.2, T_REL + 0.012 * k)
    stack.append(lp)

# Manager target points (head, near shoulder, chest), evaluated from the rig at the frame set before the call
arm = mgr.armature
head_hook = next(o for o in mgr.objs if o.name.startswith("HOOK_head_top"))
CAMPOS = Vector((HX - 2.45, -2.50, 1.45))      # lab camera (side view: Gary near/right, Manager far/left)
FACE = Vector((G_POS[0] - M_POS[0], G_POS[1] - M_POS[1], 0.0)).normalized()


def mgr_points():
    ae = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mw = ae.matrix_world

    def pbw(name):
        return mw @ ae.pose.bones[name].head
    sh_l, sh_r = pbw("upper_arm_L"), pbw("upper_arm_R")
    near = sh_l if (sh_l - CAMPOS).length < (sh_r - CAMPOS).length else sh_r
    return {"head": head_hook.matrix_world.translation.copy(),
            "shoulder": near + Vector((0, 0, 0.07)),
            "chest": pbw("chest") + FACE * 0.15 + Vector((0, 0, 0.12))}


T_HIT = {"head": 9.56, "shoulder": 9.64, "chest": 9.72}
scn.frame_set(F(T_REL))
plate_m = plate.root.matrix_world.copy()
launch = [plate_m @ lp for lp in stack]
pts_hit = {}
for kind, th_ in T_HIT.items():
    scn.frame_set(F(th_))
    pts_hit[kind] = mgr_points()[kind]
scn.frame_set(1)
print("Manager hit points:", {k: tuple(round(x, 2) for x in v) for k, v in pts_hit.items()})

floor_targets = []
n_floor = N_EGG - 3
for i in range(n_floor):
    ang = 2 * PI * (i + rnd.uniform(-0.2, 0.2)) / n_floor + 0.3
    rad = 0.45 + 0.55 * ((i * 5) % 7) / 6.0 + rnd.uniform(0, 0.15)
    floor_targets.append(Vector((M_POS[0] + rad * math.cos(ang), M_POS[1] + 0.8 * rad * math.sin(ang), 0.0)))

order = list(range(N_EGG))
rnd.shuffle(order)
kinds = ["head", "shoulder", "chest"] + ["floor"] * n_floor
assign = {order[i]: kinds[i] for i in range(N_EGG)}
fi = 0
for k in range(N_EGG):
    kind = assign[k]
    t_rel = T_REL + 0.012 * k
    P0 = launch[k]
    root, mesh = make_egg("fly_egg_%02d" % k, cook=1.0, splat=1.0, brown=0.5)
    spin = Vector((rnd.uniform(-9, 9), rnd.uniform(-9, 9), rnd.uniform(-9, 9)))
    q0 = Euler((rnd.uniform(0, 6), rnd.uniform(0, 6), rnd.uniform(0, 6))).to_quaternion()
    root.rotation_mode = "QUATERNION"

    def scale_f(t, t_rel=t_rel):
        return 1.0 + (FLY_EXAG - 1.0) * smooth(t_rel, t_rel + 0.16, t)

    if kind == "floor":
        Lg = floor_targets[fi]
        fi += 1
        q_flat = Quaternion((0, 0, 1), rnd.uniform(0, 2 * PI))
        Tfl = 0.46 + 0.04 * (k % 4)
        t_land = t_rel + Tfl
        v0 = (Lg - P0) / Tfl + Vector((0, 0, 0.5 * GACC * Tfl))
        vimp = v0 + Vector((0, 0, -GACC * Tfl))
        e1 = 0.34
        v1 = Vector((vimp.x * 0.35, vimp.y * 0.35, -e1 * vimp.z))
        t1 = 2 * v1.z / GACC
        v2 = Vector((v1.x * 0.5, v1.y * 0.5, e1 * v1.z))
        t2 = 2 * v2.z / GACC
        t_rest = t_land + t1 + t2
        for f in range(F(t_rel), min(F(t_rest) + 1, 301)):
            t = T_of(f)
            if t <= t_land:
                p = ballistic(P0, v0, t - t_rel)
                u = 0.0
            elif t <= t_land + t1:
                p = ballistic(Lg, v1, t - t_land)
                u = 0.5 * (t - t_land) / t1
            else:
                p = ballistic(Lg + v1 * t1, v2, t - t_land - t1)
                u = 0.5 + 0.5 * (t - t_land - t1) / max(t2, 1e-3)
            sc = scale_f(t)
            if t > t_land:
                p.z += 0.003 * sc
            root.location = p
            qspin = Quaternion(spin.normalized(), spin.length * (t - t_rel) * (1.0 - 0.85 * smooth(0, 1, u)))
            root.rotation_quaternion = (qspin @ q0).slerp(q_flat, smooth(0.0, 1.0, u))
            root.scale = (sc, sc, sc)
            for dp in ("location", "rotation_quaternion", "scale"):
                root.keyframe_insert(dp, frame=f)
        key_splat(root, t_rest, fresh=False, wob_amp=0.8)
    else:
        t_hit = T_HIT[kind]
        Lg = pts_hit[kind]
        Tfl = t_hit - t_rel
        v0 = (Lg - P0) / Tfl + Vector((0, 0, 0.5 * GACC * Tfl))
        base_q = Euler((rnd.uniform(-0.35, 0.35), rnd.uniform(-0.35, 0.35), rnd.uniform(0, 6))).to_quaternion()
        for f in range(F(t_rel), 301):
            t = T_of(f)
            sc = scale_f(t)
            if t <= t_hit:
                p = ballistic(P0, v0, t - t_rel)
                u = clamp01((t - t_rel) / Tfl)
                qspin = Quaternion(spin.normalized(), spin.length * (t - t_rel) * (1.0 - u))
                q = (qspin @ q0).slerp(base_q, smooth(0.5, 1.0, u))
            else:
                scn.frame_set(f)
                p = mgr_points()[kind].copy()
                if kind == "chest":
                    p.z -= 0.16 * smooth(0.0, 1.0, (t - t_hit) / 0.45)     # slides down the shirt
                p.z += 0.004 * sc
                q = base_q
            root.location = p
            root.rotation_quaternion = q
            root.scale = (sc, sc, sc)
            for dp in ("location", "rotation_quaternion", "scale"):
                root.keyframe_insert(dp, frame=f)
        scn.frame_set(1)
        key_splat(root, t_hit, fresh=False, wob_amp=1.0)
    for dp in ("location", "rotation_quaternion", "scale"):
        key_fc_interp(root, dp, "LINEAR")
    L.V(mesh, 0, t_rel, 10.0)

# ----------------------------------------------------------------------------- cameras
col = {-1: [], 1: []}
for (side, r, a, pos, yaw) in oes:
    col[side].append(oe_top[(side, r)])
CXL = sum(v.x for v in col[-1]) / 8.0
CXR = sum(v.x for v in col[1]) / 8.0
COLY = [col[-1][r].y for r in range(8)]
DIR = 1.0 if COLY[7] > COLY[0] else -1.0
print("OE columns x L %.3f R %.3f, y %s, direction %+.0f" % (CXL, CXR, [round(y, 3) for y in COLY], DIR))

# A: eased oblique push-in over the board while the OEs fly in (oblique so the lift/arc is visible)
asm.shot(0.0, 1.4, (0.0, -0.95, 1.20), (0.0, -0.02, 0.0), (0.0, -0.72, 1.08), (0.0, -0.02, 0.0), lens=28.0, ease="BEZIER")
# B: dolly-zoom onto the XPU (bars lurch)
asm.shot(1.4, 2.6, cw((0, -16, 7.0)), (0, 0, 0.5 * KC), cw((0, -9.0, 3.0)), (0, 0, 0.5 * KC), lens=20, lens1=55, ease="BEZIER")
tgt_oe = oe_top[(1, 4)].copy()
CZ0 = (tgt_oe.x + 0.13, tgt_oe.y - 0.14, tgt_oe.z + 0.10)     # outside the column: the XPU lid stays in the background
# C: continuous zoom onto the right-column OE R_4, then hold (the PIC rig dissolves in over the hold; camera still during the pulses)
asm.shot(2.6, 3.25, cw((0, -9.0, 3.0)), (0, 0, 0.5 * KC), CZ0, tuple(tgt_oe), lens=55, lens1=LENS_PIC, ease="BEZIER")
asm.shot(3.25, 4.5, CZ0, tuple(tgt_oe), CZ0, tuple(tgt_oe), lens=LENS_PIC)
# E: pull back to the high board view while the PIC dissolves out; engines glow
asm.shot(4.5, 5.4, CZ0, tuple(tgt_oe), (0.0, -1.18, 0.66), (0.0, 0.0, 0.03), lens=LENS_PIC, lens1=28.0, ease="BEZIER")
# F: low, close along the left column, then the right: target walks along the OE column with the egg landings
yS, yN = COLY[0], COLY[7]
asm.shot(5.4, 6.85, (CXL - 0.17, yS - DIR * 0.16, 0.25), (CXL, yS + DIR * 0.03, 0.05),
         (CXL - 0.17, yN - DIR * 0.16, 0.26), (CXL, yN - DIR * 0.03, 0.05), lens=38.0, ease="BEZIER")
yM = 0.5 * (yS + yN)
asm.shot(6.85, 7.65, (CXR + 0.17, yS - DIR * 0.16, 0.25), (CXR, yS + DIR * 0.03, 0.05),
         (CXR + 0.17, yN - DIR * 0.16, 0.26), (CXR, yN - DIR * 0.03, 0.05), lens=38.0, ease="BEZIER")
# pull back along the column so the first (fully overcooked) eggs and the last ones are in frame together
asm.shot(7.65, 8.2, (CXR + 0.17, yN - DIR * 0.16, 0.26), (CXR, yN - DIR * 0.03, 0.05),
         (CXR + 0.24, yM - DIR * 0.20, 0.32), (CXR, yM, 0.04), lens=38.0, lens1=32.0, ease="BEZIER")
# lab finale: one wide, stable position (no jumps), slow push
LC0 = (HX - 2.45, -2.50, 1.45)
LC1 = (HX - 2.20, -2.30, 1.45)
LT = (HX + 0.40, -0.30, 1.35)
asm.shot(8.2, 10.0, LC0, LT, LC1, LT, lens=30.0, ease="BEZIER")


def add_noise(fc, f0, f1, strength, scale=14.0, phase=0.0, blend=4):
    m = fc.modifiers.new("NOISE")
    m.blend_type = "REPLACE"
    m.scale = scale
    m.strength = strength
    m.phase = phase
    m.depth = 1
    m.use_restricted_range = True
    m.frame_start, m.frame_end = f0, f1
    m.blend_in = blend
    m.blend_out = blend


# handheld noise per shot (metres of camera position; target gets 0.6x), deterministic phases; none while the PIC pulses run
SHAKE = [((0.0, 1.4), 0.006, 26.0),
         ((1.4, 2.6), 0.060, 6.0),          # dolly-zoom spikes: strong and fast
         ((2.6, 3.25), 0.003, 12.0),
         ((4.5, 5.4), 0.004, 14.0),
         ((5.4, 8.2), 0.0035, 14.0),
         ((8.2, 10.0), 0.010, 20.0)]
for obj, tag in ((L.CAM, 0.0), (L.TGT, 100.0)):
    fcs = [fc for fc in obj.animation_data.action.fcurves if fc.data_path == "location"]
    for (ta, tb_), amp, sc in SHAKE:
        for fc in fcs:
            add_noise(fc, F(ta), F(tb_), amp * (0.6 if obj is L.TGT else 1.0), scale=sc, phase=tag + fc.array_index * 17.3 + ta * 3.1)

# overlay holder and PIC rig follow the lens exactly (blender_lib keys the HOLD scale linearly between shot ends, wrong while the
# lens eases): re-key per frame. The PIC rig keeps a constant apparent size while the lens pulls back during the dissolve-out.
_lens_fc = next(fc for fc in L.CAM.data.animation_data.action.fcurves if fc.data_path == "lens")
for fc_ in list(L.HOLD.animation_data.action.fcurves):
    if fc_.data_path == "scale":
        L.HOLD.animation_data.action.fcurves.remove(fc_)
for f in range(1, 301):
    lens = _lens_fc.evaluate(f)
    s = L.HOLD_S0 * L.LENS0 / lens
    L.HOLD.scale = (s, s, s)
    L.HOLD.keyframe_insert("scale", frame=f)
    s2 = LENS_PIC / lens
    rig_root.scale = (s2, s2, s2)
    rig_root.keyframe_insert("scale", frame=f)
for o_ in (L.HOLD, rig_root):
    key_fc_interp(o_, "scale", "LINEAR")

# ----------------------------------------------------------------------------- captions (unchanged from v1.0 except the FX notes)
asm.narr([(0.2, 2.4, "Gary glues the optics onto the chip."),
          (2.4, 5.2, "The chip's heat swings wildly, but optics need it steady."),
          (5.4, 8.0, "So: scorching hot, always.")])
asm.lab(0.0, 1.4, "NPO -> CPO   (8 + 8 OEs)")
asm.lab(1.4, 2.6, "XPU POWER   TEMPERATURE")
asm.lab(2.6, 3.4, "ZOOM ON AN OE")
asm.lab(3.5, 4.6, "RING RESONANCES: HEAT WAVE")
asm.lab(4.6, 5.4, "HEATERS ON")
asm.big(6.2, 8.0, "BREAKFAST")
asm.big(T_BANG, T_BANG + 0.6, "BANG")
asm.fxn(1.4, 2.6, "[FX: dolly-zoom, shake on spikes, smoke + heat shimmer]")
asm.fxn(2.6, 3.4, "[FX: continuous zoom, then cross-dissolve to the PIC]")
asm.fxn(3.5, 4.6, "[FX: three heat pulses over rings and substrate, left to right; shimmer wobble]")
asm.fxn(4.6, 5.4, "[FX: dissolve back to the board; engines glow red]")
asm.fxn(5.4, 8.2, "[FX: eggs tossed onto the OEs: bounce, splat, yolk wobble, overcook, sizzle steam]")
asm.fxn(8.9, 9.8, "[FX: 16 eggs tossed up: head, shoulder, chest, floor; smoke ring, muzzle flash]")
asm.timecode(4)

# whiteboard curve labels riding the tips (graph pixel coords -> board plane), as in S2
bw, bh = 2.94, 1.654
bz = wbs.matrix_world.translation.z
wy = WBY - 0.03


def to_world(px, py):
    return (WBX + (px / 640.0 - 0.5) * bw, wy, bz + (0.5 - py / 360.0) * bh)


le = asm.wl("ENERGY / BIT", (0, 0, 0), 8.2, 10.0, size=0.12, color=(1.0, 0.45, 0.4))
ll = asm.wl("LATENCY", (0, 0, 0), 8.2, 10.0, size=0.12, color=(0.5, 0.65, 1.0))
t = 8.2
while t <= 10.0001:
    (ex_, ey_), (lx_, ly_) = T.graph_tip(GPROG(t))
    wex, wey, wez = to_world(ex_, ey_)
    wlx, wly, wlz = to_world(lx_, ly_)
    le.location = (wex, wey, wez + 0.17)
    le.keyframe_insert("location", frame=F(t))
    ll.location = (wlx, wly, wlz - 0.17)
    ll.keyframe_insert("location", frame=F(t))
    t += 0.1
for o_ in (le, ll):
    key_fc_interp(o_, "location", "LINEAR")

asm.finalize(OUT)
