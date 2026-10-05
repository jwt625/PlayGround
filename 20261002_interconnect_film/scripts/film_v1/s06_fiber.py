"""S6 "Fiber: the tray" (film 50-60 s, scene-local 0-10 s), v1.2 (v2 characters, motion_v2, kick finale, warm hall).

Run: FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s06_fiber.py -- scenes/v1/s06_fiber.blend

Coordinates (metres, world): hall aisle along +x, rack row b fronts at y = 0.6 (facing -y), row a cleared over the working bay.
Characters face -y at yaw 0. v1.2 layout: Gary faces the tray (+y) at G_POS, so his backside faces -y; the Manager stands behind
him on -y (whip lands on the backside); the customers come from -y behind the Manager and kick him toward Gary.
Timeline (scene-local s): 0-1.5 MPO macro; 1.5-3.3 Gary heaves a fibre bundle, tray slides out, fibres spill; 3.3-6.9 Gary kneels
and ties, Manager stomps in and points, push-in on Gary (facepalm, "ETA: NEXT TUESDAY" box, fibre through hole 2); 6.9-8.6 whip
(crack 8.067, hit 8.30, SFX markers); 8.6-10 customers, kick contact 9.147, Manager launched forward, done by 9.6.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asm  # noqa: E402
from asm import F, PI  # noqa: E402
import s06_whip_sim as WS  # noqa: E402  (verlet whip chain)
import s06_kick  # noqa: E402  (kick_butt / react_kicked_butt motion_v2 actions, baked in the scene)
import motion_v2 as M2  # noqa: E402  (path added by s06_kick)

L = asm.L


def show_objs(asset_or_coll, t0, t1):
    """Visibility window per object, skipping objects hidden on purpose (asm.show would key them visible)."""
    objs = asset_or_coll.objs if isinstance(asset_or_coll, asm.Asset) else asm._all_objs(asset_or_coll)
    for o in objs:
        if o.hide_render:
            continue
        L.V(o, 0, t0, t1)


asm.show = show_objs   # asm.fx() resolves show() at call time
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(asm.PROJ, "scenes", "v1", "s06_fiber.blend")

# ------------------------------------------------------------------ layout constants (documented in the devlog)
X_T = 0.3                      # tray rack centre x (replaces hall rack b16)
Y_RACK_FRONT = 0.6
Y_STUB = Y_RACK_FRONT + 0.5
G_POS = (0.3, -0.62, 0.0)      # Gary (pulls, then kneels in place), yaw pi = faces the tray
G_YAW = PI
M_DIST0 = 1.80                 # Manager distance behind Gary (tuned by the whip search below)
M_DX = -0.62                   # lateral offset: the whip arc runs about 0.6 m to the +x side of the Manager's root (measured), so it lands on Gary's centreline
HIGH = os.environ.get("S06_LOD", "low") == "high"
PILE_RADIUS_M = 0.0025 if HIGH else 0.004
PILE_STRANDS = 9072 if HIGH else 600
Z_STRETCH = 0.744 / 0.35
T_KICK = 9.147                 # kick contact (= v1.1 slap cue 59.147)
HS = 3                         # hit-stop frames
F_HIT_WANT, F_CRACK_WANT = 250, 243   # SFX cues 58.30 / 58.067 (markers kept)


def log(*a):
    print("[s06]", *a, flush=True)


# ------------------------------------------------------------------ scene, light, hall
scn = asm.new_scene(preset="standard", world="WORLD_data_hall", hold_z=0.2)   # overlay plane 0.2 m: the MPO macro camera is 0.35 m from the plug
lr = asm.rig("data_hall")
lr.root.rotation_euler = (0, 0, PI / 2)
lr.root.location = (1.4, -0.9, 0.0)

hall = asm.append("datacenter/datahall_environment")
asm.place(hall.root, (0, 0, 0))
RACK_HIDE_A = (-3.6, 4.8)
RACK_CULL = (-6.0, 7.5)
for o in hall.objs:
    n = o.name
    for row in ("a", "b"):
        key = "datahall_environment_rack_%s" % row
        if n.startswith(key):
            root = o if o.type == "EMPTY" else o.parent
            if root is None:
                continue
            x = root.location.x
            if row == "a" and RACK_HIDE_A[0] <= x <= RACK_HIDE_A[1]:
                o.hide_render = o.hide_viewport = True
            if row == "b" and abs(x - X_T) < 0.01:
                o.hide_render = o.hide_viewport = True
            if x < RACK_CULL[0] or x > RACK_CULL[1]:
                o.hide_render = o.hide_viewport = True
    if any(k in n for k in ("pedestal", "stringer", "subfloor", "signage", "containment_roof", "underfloor")):
        o.hide_render = o.hide_viewport = True


def warm_hall():
    """v1.2 look: readable warm data hall. World lifted and warmed, ceiling LED emission cut (blown discs), hall strip lights made
    large and soft, warm key over the bay, cool rims for the characters, perforated alpha removed (dither speckle)."""
    g = next(n for n in scn.world.node_tree.nodes if n.type == "GROUP")
    g.inputs["Top Color"].default_value = (0.20, 0.17, 0.14, 1.0)
    g.inputs["Bottom Color"].default_value = (0.07, 0.06, 0.05, 1.0)
    g.inputs["Strength"].default_value = 1.0
    _lights()


def tweak_materials():
    """Material look fixes; called after every asset is appended (the MPO and fibre materials do not exist earlier)."""
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for nd in m.node_tree.nodes:
            if nd.type != "BSDF_PRINCIPLED":
                continue
            if m.name == "MAT_datacenter_led_strip":
                nd.inputs["Emission Strength"].default_value = 2.2
                nd.inputs["Emission Color"].default_value = (1.0, 0.88, 0.72, 1.0)
            if m.name in ("MAT_datacenter_perforated_door", "MAT_datacenter_perforated_tile", "MAT_datacenter_ru_rail_holes"):
                a = nd.inputs["Alpha"]
                for lk in list(a.links):
                    m.node_tree.links.remove(lk)
                a.default_value = 1.0
            if m.name.startswith("MAT_interconnect_fiber_core"):        # ferrule fibre ends: visible dots, no bloom
                nd.inputs["Emission Strength"].default_value = 0.6
            if m.name.startswith("MAT_interconnect_plastic_green"):      # APC green read as a glowing blob under the warm key
                nd.inputs["Base Color"].default_value = (0.02, 0.17, 0.04, 1.0)
            if m.name.startswith("MAT_interconnect_mt_ferrule"):         # ferrule face was blown white under the key
                nd.inputs["Base Color"].default_value = (0.55, 0.52, 0.42, 1.0)
            if m.name == "MAT_datacenter_floor_tile":
                c = nd.inputs["Base Color"]
                if not c.is_linked:
                    c.default_value = (0.30, 0.28, 0.26, 1.0)
                else:                                  # textured tile: multiply down (white floor washed out the subtitles)
                    src = c.links[0].from_socket
                    mx = m.node_tree.nodes.new("ShaderNodeMix")
                    mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
                    mx.inputs[0].default_value = 1.0
                    m.node_tree.links.new(src, mx.inputs[6])
                    mx.inputs[7].default_value = (0.55, 0.52, 0.48, 1.0)
                    m.node_tree.links.new(mx.outputs[2], c)


def _lights():
    for o in bpy.data.objects:
        if o.type == "LIGHT" and o.name.startswith("light_hall_strip"):
            o.data.size = 1.2
            o.data.energy = 60.0
            o.data.use_shadow = False      # render cost: only the bay key casts shadows
            o.data.color = (1.0, 0.86, 0.70)
        if o.type == "LIGHT" and o.name == "light_hall_floor_bounce":
            o.data.energy = 8.0
            o.data.use_shadow = False
            o.data.color = (1.0, 0.85, 0.7)

    def area(name, loc, aim, energy, color, size, spot=False, shadow=False):
        d = bpy.data.lights.new(name, "SPOT" if spot else "AREA")
        d.energy, d.color = energy, color
        if spot:
            d.spot_size, d.spot_blend, d.shadow_soft_size = 0.9, 0.8, 0.3
        else:
            d.size = size
        d.use_shadow = shadow
        o = bpy.data.objects.new(name, d)
        bpy.context.scene.collection.objects.link(o)
        o.location = loc
        v = Vector(aim) - Vector(loc)
        o.rotation_euler = v.to_track_quat("-Z", "Y").to_euler()
        return o
    area("key_warm_bay", (1.6, -1.6, 3.3), (0.4, -1.6, 0.6), 160.0, (1.0, 0.80, 0.60), 3.0, shadow=True)
    area("fill_warm_front", (3.8, -1.4, 1.6), (0.3, -1.5, 1.0), 60.0, (1.0, 0.85, 0.70), 2.5)
    area("rim_cool_back", (-2.4, -1.4, 2.6), (0.4, -1.6, 1.1), 120.0, (0.62, 0.78, 1.0), 1.5)
    area("rim_cool_tray", (-2.2, 0.3, 2.8), (0.3, -0.6, 0.9), 45.0, (0.62, 0.78, 1.0), 1.0)
    area("tray_fill", (X_T + 1.8, 0.0, 2.6), (X_T, 0.4, 0.75), 30.0, (1.0, 0.9, 0.78), 1.0)   # far enough from Gary's head (no hot spot)
    area("rim_warm_cust", (-1.6, -4.4, 2.4), (0.2, -2.6, 1.2), 110.0, (1.0, 0.72, 0.45), 1.5)


warm_hall()

# ------------------------------------------------------------------ tray (p_slide 0 -> 1, 1.5-3.2 s, bezier)
tray = asm.append("datacenter/tray_fiber_pullout")
asm.place(tray.root, (X_T, Y_STUB, 0.0), yaw=0.0)
asm.key_prop(tray.root, "p_slide", 0.0, 0.0, "CONSTANT")
asm.key_prop(tray.root, "p_slide", 1.5, 0.0, "BEZIER")
asm.key_prop(tray.root, "p_slide", 3.2, 1.0, "BEZIER")

# ------------------------------------------------------------------ fiber spaghetti (600 strand LOD, growth by visible fraction)
sp = asm.append("interconnect/fiber_spaghetti_generator")
for o in sp.objs:
    if o.name.endswith("_tray") or o.name.endswith("_slide_rails"):
        o.hide_render = o.hide_viewport = True
for ch in sp.coll.children:
    ch.hide_render = ch.hide_viewport = ch.name != ("LOD_high" if HIGH else "LOD_low")
    for lc in asm._layer_colls(bpy.context.view_layer.layer_collection):
        if lc.collection == ch:
            lc.exclude = False
pile = next(o for o in sp.objs if o.name.endswith("pile_9072" if HIGH else "pile_low_600"))
ng = bpy.data.node_groups["NG_fiber_spaghetti"]


def patch_spaghetti_nodes(ng):
    """Add inputs Fill (0..1 growth), Strands, Z Stretch and Spread; delete points beyond each strand's grown length, then stretch z."""
    it = ng.interface
    s_fill = it.new_socket("Fill", in_out="INPUT", socket_type="NodeSocketFloat")
    s_str = it.new_socket("Strands", in_out="INPUT", socket_type="NodeSocketFloat")
    s_zs = it.new_socket("Z Stretch", in_out="INPUT", socket_type="NodeSocketFloat")
    s_spr = it.new_socket("Spread", in_out="INPUT", socket_type="NodeSocketFloat")
    s_fill.default_value, s_str.default_value, s_zs.default_value, s_spr.default_value = 1.0, 600.0, 1.0, 3.0
    N = ng.nodes
    gi = N["Group Input"]
    sp_node = N["Set Position"]
    old = next(lk for lk in ng.links if lk.to_node.name == sp_node.name and lk.to_socket.name == "Geometry")
    ng.links.remove(old)

    def math_(op, a=None, b=None, c=None, x=0, y=0):
        n = N.new("ShaderNodeMath")
        n.operation = op
        n.location = (x, y)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                ng.links.new(v, n.inputs[i])
        return n

    idx = N.new("GeometryNodeInputIndex")
    strand = math_("FLOOR", math_("DIVIDE", idx.outputs[0], 32.0).outputs[0])
    local = math_("SUBTRACT", idx.outputs[0], math_("MULTIPLY", strand.outputs[0], 32.0).outputs[0])
    rank = math_("DIVIDE", strand.outputs[0], math_("SUBTRACT", gi.outputs["Strands"], 1.0).outputs[0])
    fs = math_("MULTIPLY", gi.outputs["Fill"], math_("ADD", gi.outputs["Spread"], 1.0).outputs[0])
    rs = math_("MULTIPLY", rank.outputs[0], gi.outputs["Spread"])
    prog = math_("SUBTRACT", fs.outputs[0], rs.outputs[0])
    prog = math_("MINIMUM", math_("MAXIMUM", prog.outputs[0], 0.0).outputs[0], 1.0)
    lim = math_("MULTIPLY", prog.outputs[0], 31.0)
    gt = math_("GREATER_THAN", local.outputs[0], math_("ADD", lim.outputs[0], 0.001).outputs[0])
    dele = N.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    ng.links.new(gi.outputs["Geometry"], dele.inputs["Geometry"])
    ng.links.new(gt.outputs[0], dele.inputs["Selection"])
    tr = N.new("GeometryNodeTransform")
    comb = N.new("ShaderNodeCombineXYZ")
    comb.inputs[0].default_value = 1.0
    comb.inputs[1].default_value = 1.0
    ng.links.new(gi.outputs["Z Stretch"], comb.inputs[2])
    ng.links.new(dele.outputs["Geometry"], tr.inputs["Geometry"])
    ng.links.new(comb.outputs[0], tr.inputs["Scale"])
    ng.links.new(tr.outputs["Geometry"], sp_node.inputs["Geometry"])
    return {s.name: s.identifier for s in (s_fill, s_str, s_zs, s_spr)}


SOCK = patch_spaghetti_nodes(ng)
mod = pile.modifiers["fiber_gn"]
mod[SOCK["Fill"]] = 0.0
mod[SOCK["Strands"]] = float(PILE_STRANDS)
mod[SOCK["Z Stretch"]] = Z_STRETCH
mod[SOCK["Spread"]] = 3.0
mod["Socket_2"] = PILE_RADIUS_M
pile.update_tag()


def key_mod(mod_name, sock, t, v, interp="LINEAR"):
    pile.modifiers[mod_name][sock] = v
    pile.update_tag()
    pile.keyframe_insert('modifiers["%s"]["%s"]' % (mod_name, sock), frame=F(t))
    for fc in pile.animation_data.action.fcurves:
        if fc.data_path == 'modifiers["%s"]["%s"]' % (mod_name, sock):
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) == F(t):
                    kp.interpolation = interp


key_mod("fiber_gn", SOCK["Fill"], 0.0, 0.0)
key_mod("fiber_gn", SOCK["Fill"], 2.0, 0.0)
key_mod("fiber_gn", SOCK["Fill"], 6.0, 1.0)
asm.attach(sp.root, tray.hook("slide"), offset=(0, 0, -0.7))
asm.key_prop(sp.root, "p_squirm", 0.0, 0.0)
asm.key_prop(sp.root, "p_squirm", 2.0, 0.1)
asm.key_prop(sp.root, "p_squirm", 4.4, 0.8)
asm.key_prop(sp.root, "p_squirm", 8.0, 1.0)
asm.set_prop(sp.root, "p_squirm_speed", 3.0)

# ------------------------------------------------------------------ characters (v2) and scene-local kick actions (baked before any NLA)
gary = asm.append("characters/gary_v2", actions=True)
mgr = asm.append("characters/manager_v2", actions=True)
nv = asm.append("characters/npc_nvydia_v2", actions=True)      # leather jacket built in
op = asm.append("characters/npc_openay_v2", actions=True)
for _a in (nv, op):
    if "p_accessory" in _a.root.keys():
        _a.root["p_accessory"] = 0
s06_kick.bake(nv, "kick_butt")
s06_kick.bake(mgr, "react_kicked_butt")
log("kick actions baked")

asm.place(gary.root, G_POS, yaw=G_YAW)
asm.show(gary, 1.45, 10.0)

# Gary NLA (creation order = priority): idle base, pull fibres, kneel, tie loop (ends at the hit), facepalm layer; whipped added after the whip pass
M2.apply(gary, "idle_breathe", 0.0, hold=True, repeat=2.0, face=False)
T_PULL0, T_KNEEL = 1.4, 3.3
M2.apply(gary, "pull_cable", T_PULL0, hold=False, repeat=(T_KNEEL - T_PULL0) * 30.0 / 36.0 + 0.05, blend_out=3)
st_kneel = M2.apply(gary, "kneel_down", T_KNEEL, hold=False, blend_in=4)
T_TIE0 = (st_kneel.frame_end - 1) / 30.0
st_tie = M2.apply(gary, "tie_fibers_kneel", T_TIE0, hold=True, repeat=(10.0 - T_TIE0) * 30.0 / 36.0, blend_in=2)
M2.apply(gary, "facepalm_upper", 5.6, layer="gesture", hold=False, end_frame=58, blend_in=5, blend_out=8)

# bullet holes: r = max(0.3, 0.9 ** floor((T - T_shot) / 1.5)), film time T = 50 + t, CONSTANT keys every 1.5 s
SHOTS = {"p_hole_1_radius": 9.2, "p_hole_2_radius": 19.2, "p_head_hole_radius": 28.95, "p_hole_4_radius": 38.9, "p_hole_5_radius": 49.1}
for prop, ts in SHOTS.items():
    t = 0.0
    while t < 10.0 + 1e-6:
        steps = math.floor((50.0 + t - ts) / 1.5)
        r = max(0.3, 0.9 ** max(steps, 0))
        asm.key_prop(gary.root, prop, t, r, "CONSTANT")
        nxt = ts + (steps + 1) * 1.5 - 50.0
        if nxt > 10.0 or r <= 0.3 + 1e-9:
            break
        t = max(nxt, t + 1e-3)
asm.key_prop(gary.root, "p_expr_sweating", 3.3, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 5.0, 0.7)

# ------------------------------------------------------------------ Manager + AOC whip
whip = asm.append("props/aoc_whip", actions=True)
WHIP_ROT = Euler((0, 0, PI))
_grip = Vector((0, -0.325, 0))
WHIP_THICK, PLUG_THICK = 7.0, 3.5
_seen = set()
for _o in whip.objs:
    if _o.type == "MESH" and _o.data.name not in _seen:
        _seen.add(_o.data.name)
        _k = PLUG_THICK if "_plug" in _o.name else WHIP_THICK
        _o.data.transform(Matrix.Diagonal((_k, 1.0, _k, 1.0)))
asm.attach(whip.root, mgr.hook("whip_grip_L"), offset=tuple(-(WHIP_ROT.to_matrix() @ _grip)), rot=tuple(WHIP_ROT))
whip_arm = whip.armature
ARM_T0, ARM_SPEED = 7.2, 19.0 / 24.0
ARM_ACT = bpy.data.actions["ACT_manager_v2_arm_whip"]
T_MWALK = 0.9


def m_pos(dist):
    return (G_POS[0] + M_DX, G_POS[1] - dist, 0.0)


def build_manager(dist, af_c=None):
    """(Re)build the Manager's root keys and NLA for a distance `dist` behind Gary. af_c: split arm_whip at that action frame (hit-stop)."""
    arm = mgr.armature
    if arm.animation_data:
        for tr in list(arm.animation_data.nla_tracks):
            arm.animation_data.nla_tracks.remove(tr)
    if mgr.root.animation_data:
        for tr in list(mgr.root.animation_data.nla_tracks):
            mgr.root.animation_data.nla_tracks.remove(tr)
        if mgr.root.animation_data.action:
            bpy.data.actions.remove(mgr.root.animation_data.action)
    mp = m_pos(dist)
    myaw = PI                    # faces +y (Gary's backside); the whip plane is offset by M_DX
    M2.apply(mgr, "idle_breathe_tense", 0.0, hold=True, repeat=5.0, face=False)
    pts = [(mp[0] - 2.4, mp[1] - 1.75, 0.0), (mp[0] - 0.40, mp[1] - 0.62, 0.0), mp]
    t_arr = M2.walk(mgr, T_MWALK, pts, gait="stomp_walk", speed_mps=1.1)
    asm.key_loc(mgr.root, t_arr, mp, yaw=asm.yaw_to(pts[1], pts[2]))
    asm.key_loc(mgr.root, t_arr + 0.25, mp, yaw=myaw, interp="BEZIER")
    M2.apply(mgr, "point_accuse", t_arr, hold=False, blend_in=4, face=False)
    t_x = t_arr + 40 / 30.0
    M2.apply(mgr, "idle_arms_cross", t_x - 0.1, hold=False, repeat=(ARM_T0 - t_x + 0.2) * 30.0 / 72.0, blend_in=5, face=False)
    tr = arm.animation_data.nla_tracks.new()
    tr.name = "arm_whip"
    sc = 1.0 / ARM_SPEED
    f0 = float(F(ARM_T0))
    end = ARM_ACT.frame_range[1]

    def strip(name, start, a0, a1, scale):
        s = tr.strips.new(name, int(round(start)), ARM_ACT)
        s.action_frame_start, s.action_frame_end = a0, a1
        s.scale = scale
        s.blend_type, s.extrapolation = "REPLACE", "NOTHING"
        return s
    if af_c is None:
        s = strip("arm_whip", f0, 0.0, end, sc)
        s.blend_in = 4
    else:
        a = strip("arm_whip_A", f0, 0.0, af_c, sc)
        a.blend_in = 4
        c = strip("arm_whip_freeze", a.frame_end, af_c, af_c + 0.5, HS / 0.5)
        b = strip("arm_whip_B", c.frame_end, af_c, end, sc)
        b.blend_out = 6
    M2.apply(mgr, "react_kicked_butt", T_KICK, hold=True, blend_in=2, face=True)
    for tt, v in ((t_arr - 0.3, 0.0), (t_arr + 0.4, 0.6), (ARM_T0, 0.75), (8.0, 1.0), (T_KICK - 0.03, 1.0), (T_KICK, 0.0)):
        asm.key_prop(mgr.root, "p_anger", tt, v, "CONSTANT" if tt == T_KICK - 0.03 else "LINEAR")
    asm.key_prop(mgr.root, "p_expr_shouting", t_arr, 0.0)
    asm.key_prop(mgr.root, "p_expr_shouting", t_arr + 0.3, 0.8)
    asm.key_prop(mgr.root, "p_expr_shouting", t_x + 0.2, 0.0)
    asm.key_prop(mgr.root, "p_expr_angry", t_x, 0.0)
    asm.key_prop(mgr.root, "p_expr_angry", t_x + 0.4, 0.8)
    asm.key_prop(mgr.root, "p_expr_angry", T_KICK - 0.03, 0.8, "CONSTANT")
    asm.key_prop(mgr.root, "p_expr_angry", T_KICK, 0.0)
    return mp, myaw, t_arr


def whip_inputs(f0, f1):
    """Per frame: whip handle points (world), Gary bone capsules. Heavy GN modifiers are muted while stepping."""
    muted = []
    for o in bpy.data.objects:
        for md in o.modifiers:
            if md.type == "NODES" and md.show_viewport:
                md.show_viewport = False
                muted.append(md)
    H, W, caps = [], [], []
    garm = gary.armature
    names = list(WS.GARY_R)
    for f in range(f0, f1 + 1):
        scn.frame_set(f)
        Wm = whip.root.matrix_world.copy()
        W.append(Wm)
        H.append([tuple(Wm @ Vector((0, -WS.L_SEG * i, 0))) for i in range(WS.PINNED)])
        cl = []
        for nm in names:
            pb = garm.pose.bones[nm]
            cl.append((np.array(garm.matrix_world @ pb.head), np.array(garm.matrix_world @ pb.tail), WS.GARY_R[nm], nm in WS.BACKSIDE))
        caps.append(cl)
    for md in muted:
        md.show_viewport = True
    return np.array(H), W, caps, names


def first_contact(rec, names, f_base):
    if not rec:
        return None
    fi, pt, spd, ci = rec[0]
    return f_base + fi, names[ci], pt, spd


# pass 0: search the Manager distance so the first contact is on the backside at the hit cue frame
M_POS, M_YAW, T_ARR = build_manager(M_DIST0)
SF0, SF1 = F(6.5), F(8.9)
_H, _W, _caps, _names = whip_inputs(SF0, SF1)
_fwd = np.array([0.0, 1.0, 0.0])          # Manager -> Gary direction (+y)
best = None
for dd in np.arange(-0.30, 0.31, 0.05):
    _P, _rec = WS.simulate(_H + _fwd[None, None, :] * dd, _caps)
    fc = first_contact(_rec, _names, SF0)
    log("search dist %.3f: first contact %s" % (M_DIST0 - dd, fc))
    if os.environ.get("S06_DEBUG"):
        bs = [r for r in _rec if _names[r[3]] in WS.BACKSIDE]
        hi = _names.index("hips")
        clear = []
        for fi in range(len(_P)):
            a, b, rr, _ = _caps[fi][hi]
            dmin = min(np.linalg.norm(p - WS._closest_on_seg(p, a, b)) - rr for p in _P[fi, WS.PINNED:])
            clear.append((SF0 + fi, round(float(dmin), 2), tuple(np.round(_P[fi, -1], 2))))
        log("   first backside contact %s; hips clearance/tip by frame %s" % (bs[0] if bs else None, [c for c in clear if F(7.6) <= c[0] <= F(8.6)][::2]))
    if fc is None:
        continue
    score = abs(fc[0] - F_HIT_WANT) + (0 if fc[1] in WS.BACKSIDE else 20)
    if best is None or score < best[0]:
        best = (score, M_DIST0 - dd, fc)
M_DIST = best[1] if best else M_DIST0
log("chosen Manager distance %.3f m (score %s)" % (M_DIST, best))
M_POS, M_YAW, T_ARR = build_manager(M_DIST)

# pass 1: full range, no reaction yet: first contact and crack
SIM_F0, SIM_F1 = F(T_MWALK), 300
scn.frame_set(SIM_F0)
_H, _W, _caps, _names = whip_inputs(SIM_F0, SIM_F1)
_P, _rec = WS.simulate(_H, _caps)
fc = first_contact([r for r in _rec if r[0] >= F(7.4) - SIM_F0], _names, SIM_F0)
assert fc, "whip never reached Gary"
F_HIT, HIT_BONE, _pt, _spd = fc
_ts = WS.tip_speed(_P)
_lo = F(7.4) - SIM_F0
F_CRACK = SIM_F0 + _lo + int(np.argmax(_ts[_lo:F_HIT - SIM_F0 + 1]))
if abs(F_CRACK - F_CRACK_WANT) <= 1:     # keep the crack FX/marker on the SFX cue frame (tip speed is near its peak there too)
    log("crack frame %d (tip %.1f m/s) set to cue frame %d (tip %.1f m/s)" % (F_CRACK, _ts[F_CRACK - SIM_F0], F_CRACK_WANT, _ts[F_CRACK_WANT - SIM_F0]))
    F_CRACK = F_CRACK_WANT
log("whip pass 1: crack frame %d (tip %.1f m/s), first contact frame %d on %s (point %d, %.1f m/s)" % (F_CRACK, _ts[F_CRACK - SIM_F0], F_HIT, HIT_BONE, _pt, _spd))
T_HIT, T_CRACK = (F_HIT - 1) / 30.0, (F_CRACK - 1) / 30.0

# pass 2: hit-stop (arm frozen, chain frozen), Gary's whipped reaction after the stop
build_manager(M_DIST, af_c=(F_HIT - F(ARM_T0)) * ARM_SPEED)
st_tie.repeat = max((F_HIT - st_tie.frame_start) / 36.0, 0.1)     # tie loop ends (and holds) at the contact pose
M2.apply(gary, "whipped", T_HIT + HS / 30.0, start_frame=HS, blend_in=3, hold=True)
scn.frame_set(SIM_F0)
_H, _W, _caps, _names = whip_inputs(SIM_F0, SIM_F1)
_hold = set(range(F_HIT - SIM_F0 + 1, F_HIT - SIM_F0 + HS + 1))
_P, _rec = WS.simulate(_H, _caps, hold=_hold)
_Winv = [w.inverted() for w in _W]
_Q = WS.bone_quats(_P, _Winv)
WS.key_quats(whip_arm, _Q, SIM_F0)
_ts = WS.tip_speed(_P)
log("whip baked: %d frames, max tip speed %.1f m/s at frame %d" % (len(_P), _ts.max(), SIM_F0 + int(_ts.argmax())))
TIP_AT_CRACK = tuple(_P[F_CRACK - SIM_F0, -1])
HIT_POINT = tuple(_P[F_HIT - SIM_F0, _pt])
scn.timeline_markers.new("SFX_whip_crack", frame=F_CRACK)
scn.timeline_markers.new("SFX_whip_hit_thump", frame=F_HIT)
scn.timeline_markers.new("HITSTOP_start", frame=F_HIT)
scn.timeline_markers.new("HITSTOP_end", frame=F_HIT + HS)
for _fr, _v in ((F_HIT, 0.0), (F_HIT + 1, -0.16), (F_HIT + HS, -0.16), (F_HIT + HS + 3, 0.08), (F_HIT + HS + 9, 0.0)):
    gary.root["p_squash"] = _v
    gary.root.keyframe_insert('["p_squash"]', frame=_fr)
asm.show(mgr, T_MWALK, 10.0)
asm.show(whip, T_MWALK, 10.0)

# ------------------------------------------------------------------ customers: NVYDIA kicks the Manager's backside, OPENAY watches
F_KICK = F(T_KICK)
scn.frame_set(F_KICK)
bpy.context.view_layer.update()
_marm = mgr.armature
_mh = _marm.matrix_world @ _marm.pose.bones["hips"].head
_mf = Vector((math.sin(M_YAW), -math.cos(M_YAW), 0.0))
BUTT = _mh - _mf * 0.17 + Vector((0, 0, -0.04))
NV_YAW = M_YAW - 0.30                                  # turned a little toward the camera (+x)
_nf = Vector((math.sin(NV_YAW), -math.cos(NV_YAW), 0.0))
T_KS = T_KICK - s06_kick.KICK_CONTACT / 30.0
nv_tmp = BUTT - _nf * 0.9
asm.key_loc(nv.root, 0.0, (nv_tmp.x, nv_tmp.y, 0.0), yaw=NV_YAW, interp="CONSTANT")
M2.apply(nv, "kick_butt", T_KS, hold=True, blend_in=3)
scn.frame_set(F_KICK)
bpy.context.view_layer.update()
_narm = nv.armature
_toe = _narm.matrix_world @ _narm.pose.bones["foot_R"].tail
_d = BUTT - _toe
NV_POS = (nv_tmp.x + _d.x, nv_tmp.y + _d.y, 0.0)
log("kick: butt %s toe %s height diff %.3f -> NVYDIA at %s" % (tuple(round(v, 3) for v in BUTT), tuple(round(v, 3) for v in _toe), _d.z, tuple(round(v, 3) for v in NV_POS)))
bpy.data.actions.remove(nv.root.animation_data.action)
nv_start = (NV_POS[0] - _nf.x * 1.5, NV_POS[1] - _nf.y * 1.5, 0.0)
M2.apply(nv, "idle_breathe", 0.0, hold=True, repeat=4.0, face=False)
t_nv_end = M2.walk(nv, T_KS - 1.5 / 0.95 - 0.05, [nv_start, NV_POS])
asm.key_loc(nv.root, T_KS, NV_POS, yaw=NV_YAW, interp="CONSTANT")
asm.show(nv, T_KS - 1.7, 10.0)
log("NVYDIA walk ends %.2f, kick starts %.2f" % (t_nv_end, T_KS))
# the walk strip was created after the kick strip (higher track): re-add the kick on top so it wins from T_KS
M2.apply(nv, "kick_butt", T_KS, hold=True, blend_in=3)

OP_POS = (NV_POS[0] + 0.55, NV_POS[1] - 0.75, 0.0)
OP_YAW = PI - 0.85
op_start = (OP_POS[0] - 0.2, OP_POS[1] - 1.4, 0.0)
M2.apply(op, "idle_breathe", 0.0, hold=True, repeat=4.0, face=False)
t_op = M2.walk(op, T_KS - 1.6, [op_start, OP_POS])
asm.key_loc(op.root, t_op + 0.2, OP_POS, yaw=OP_YAW, interp="BEZIER")
M2.apply(op, "idle_arms_cross", t_op, hold=True, repeat=2.0, blend_in=6, face=False)
asm.key_prop(op.root, "p_expr_smug", t_op, 0.0)
asm.key_prop(op.root, "p_expr_smug", t_op + 0.4, 0.9)
asm.show(op, T_KS - 1.7, 10.0)

# ------------------------------------------------------------------ intro: MPO patch cord jams crooked into an MPO receptacle (macro)
MACRO_S = 8.0
mac = asm.append("interconnect/fiber_cables_and_trays")   # only the source of bundle_cords / cable_tie meshes (variants hidden)
for ch in mac.coll.children:
    ch.hide_render = ch.hide_viewport = True
    for lc in asm._layer_colls(bpy.context.view_layer.layer_collection):
        if lc.collection == ch:
            lc.exclude = False
plate = asm.append("interconnect/mpo_connector_and_adapter", only=["ASSET_mpo_adapter_plate4", "ASSET_mpo_plug"])
plug = plate.others[0]
MOUTH_T = Vector((1.78, -0.30, 1.41))
asm.place(plate.root, (0, 0, 0), yaw=-PI / 2, scale=MACRO_S)
bpy.context.view_layer.update()
_hp = plate.hook("port_2").matrix_world.translation.copy()
plate.root.location = tuple(MOUTH_T - _hp)
bpy.context.view_layer.update()
MOUTH = plate.hook("port_2").matrix_world.translation.copy()
asm.show(plate, 0.0, 1.55)
asm.show(plug, 0.0, 1.55)
cable_mat = L.mat((0.20, 0.78, 0.80), rough=0.55, emit=0.35)    # aqua trunk jacket (v1.1 neon tube read as a light)
for o in plug.objs:
    if o.type == "MESH" and o.name.endswith("_cable"):
        o.data.materials.clear()
        o.data.materials.append(cable_mat)
plug.root.scale = (MACRO_S,) * 3
for o in plug.objs:     # the asset carries three ferrule variants at the same place: keep MPO-12 male (2 guide pins)
    if "mpo12_female" in o.name or "mpo16" in o.name:
        o.hide_render = o.hide_viewport = True
PLUG_FAR = 1.0


def plug_pose(t, x, yaw_err=0.0, roll=0.0, dy=0.0, dz=0.0):
    plug.root.location = (MOUTH.x - x, MOUTH.y + dy, MOUTH.z + dz)
    plug.root.rotation_euler = (0.0, roll, PI / 2 + yaw_err)
    plug.root.keyframe_insert("location", frame=F(t))
    plug.root.keyframe_insert("rotation_euler", frame=F(t))
    L._interp(plug.root, "location", F(t), "LINEAR")
    L._interp(plug.root, "rotation_euler", F(t), "LINEAR")


for tt, xx in ((0.0, PLUG_FAR), (0.2, 0.84), (0.45, 0.45), (0.7, 0.16), (0.85, 0.045)):
    plug_pose(tt, xx)
plug_pose(0.95, 0.03, yaw_err=0.20, roll=0.28, dz=0.004)
for k in range(9):
    tt = 1.0 + 0.06 * k
    sg = 1 if k % 2 == 0 else -1
    plug_pose(tt, 0.03 + 0.006 * sg, yaw_err=0.20 + 0.05 * sg, roll=0.28 + 0.04 * sg, dz=0.004 * sg)
_cu = bpy.data.curves.new("mpo_cord", "CURVE")
_cu.dimensions = "3D"
_cu.bevel_depth = 0.0015 * MACRO_S
_cu.bevel_resolution = 4
_sp = _cu.splines.new("NURBS")
_sp.order_u = 4
_sp.use_endpoint_u = True
_sp.points.add(3)
scn.frame_set(1)
bpy.context.view_layer.update()
_end = plug.hook("cable_end").matrix_world.translation.copy()
_cp = [Vector((-2.2, -1.5, 0.95)), Vector((-1.2, -1.0, 0.8)), _end + Vector((-0.45, 0.0, -0.1)), _end]
for pt, w in zip(_sp.points, _cp):
    pt.co = (w.x, w.y, w.z, 1.0)
mpo_cord = bpy.data.objects.new("mpo_cord", _cu)
bpy.context.scene.collection.objects.link(mpo_cord)
_cu.materials.append(cable_mat)
_hm = mpo_cord.modifiers.new("hk_end", "HOOK")
_hm.object = plug.hook("cable_end")
_hm.vertex_indices_set([2, 3])
_hm.matrix_inverse = plug.hook("cable_end").matrix_world.inverted()
L.V(mpo_cord, 0, 0.0, 1.55)
JAM_AT = MOUTH + Vector((-0.04, 0.0, 0.0))
# macro fill (0-1.5 s only): lifts the dark rack background of the MPO shot (luma jump into the bright hall at the 1.5 s cut)
_mfd = bpy.data.lights.new("macro_fill", "AREA")
_mfd.size, _mfd.color, _mfd.use_shadow = 1.2, (1.0, 0.88, 0.75), False
_mf = bpy.data.objects.new("macro_fill", _mfd)
bpy.context.scene.collection.objects.link(_mf)
_mf.location = (MOUTH.x - 0.5, MOUTH.y - 0.9, MOUTH.z + 0.5)
_mf.rotation_euler = (MOUTH + Vector((-0.6, 0.4, -0.3)) - _mf.location).to_track_quat("-Z", "Y").to_euler()
for _t, _e in ((0.0, 30.0), (1.5 - 1 / 30.0, 30.0), (1.5, 0.0)):
    _mfd.energy = _e
    _mfd.keyframe_insert("energy", frame=F(_t))
for _fc in _mfd.animation_data.action.fcurves:
    for _kp in _fc.keyframe_points:
        _kp.interpolation = "CONSTANT"


def _empty(name, parent=None, loc=(0, 0, 0)):
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    if parent is not None:
        e.parent = parent
        e.matrix_parent_inverse.identity()
    e.location = loc
    return e


def hooked_curve(name, pts, groups, radius, mat_, frame):
    """NURBS curve through world points pts; groups = [(obj, [point indices])] hook modifiers set at `frame`."""
    scn.frame_set(frame)
    bpy.context.view_layer.update()
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    s = cu.splines.new("NURBS")
    s.order_u = 4
    s.use_endpoint_u = True
    s.points.add(len(pts) - 1)
    for p, w in zip(s.points, pts):
        p.co = (w[0], w[1], w[2], 1.0)
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    cu.materials.append(mat_)
    for k, (hobj, idxs) in enumerate(groups):
        hm = ob.modifiers.new("hk%d" % k, "HOOK")
        hm.object = hobj
        hm.vertex_indices_set(idxs)
        hm.matrix_inverse = hobj.matrix_world.inverted()
    return ob


# ------------------------------------------------------------------ fibre bundle Gary heaves on (hands -> drawer), 1.4-3.3 s
scn.frame_set(F(1.5))
bpy.context.view_layer.update()
_hl = gary.hook("hand_L").matrix_world.translation.copy()
_hr = gary.hook("hand_R").matrix_world.translation.copy()
_sl = tray.hook("slide").matrix_world.translation.copy()
_dr = _empty("pull_bundle_drawer", tray.hook("slide"), loc=(0, 0, 0))
_mid = (_hl + _hr) * 0.5
_in = Vector((_sl.x, _sl.y - 0.05, 0.80))
bundle_mat = L.mat((0.95, 0.80, 0.15), rough=0.5, emit=0.15)
pull_bundle = hooked_curve("pull_bundle", [_hr, _hl, _mid + (_in - _mid) * 0.5 + Vector((0, 0, -0.08)), _in],
                           [(gary.hook("hand_R"), [0]), (gary.hook("hand_L"), [1]), (_dr, [2, 3])], 0.018, bundle_mat, F(1.5))
L.V(pull_bundle, 0, 1.45, 3.3)

# ------------------------------------------------------------------ the fiber that threads through Gary's hole 2 (healed radius 0.3 of base)
hole2 = next(o for o in gary.objs if o.name.startswith("HOLE_hole_2"))
scn.frame_set(F(6.6))
bpy.context.view_layer.update()
_Mw = hole2.matrix_world.copy()
_H2 = _Mw.translation.copy()
_ax = (_Mw.to_3x3() @ Vector((0, 0, 1))).normalized()
_axh = Vector((_ax.x, _ax.y, 0.0)).normalized()
_SV = [-0.9, -0.5, -0.12, 0.0, 0.12, 0.5, 0.9]
_PW = []
for sv in _SV:
    if abs(sv) <= 0.12:
        w = _H2 + _ax * sv
    else:
        w = _H2 + _axh * sv
        w.z = max(0.03, _H2.z * 0.35) if abs(sv) < 0.7 else 0.03
    _PW.append(w)
e_front = _empty("thread_end_front", loc=tuple(_PW[0]))
e_back = _empty("thread_end_back", loc=tuple(_PW[-1]))
e_hole = _empty("thread_hole_anchor", hole2)
asm.key_loc(e_hole, 5.4, (0, 0, 0.6), interp="BEZIER")
asm.key_loc(e_hole, 6.6, (0, 0, 0.0), interp="BEZIER")
thread = hooked_curve("thread_fiber", _PW, [(e_front, [0, 1]), (e_hole, [2, 3, 4]), (e_back, [5, 6])], 0.006, L.mat((1.0, 0.45, 0.05), emit=0.6), F(6.6))
L.V(thread, 0, 5.4, T_HIT)

# ------------------------------------------------------------------ props: tie gun, tie box, bundles, ETA box
tg = asm.append("props/cable_ties_tools", only=["ASSET_cable_tie_gun"])
_R = Euler((0, 0, PI))
_gg = Vector((0.001, 0.056, 0.04))
asm.attach(tg.root, gary.hook("hand_R"), offset=tuple(-(_R.to_matrix() @ _gg)), rot=tuple(_R))
asm.show(tg, T_KNEEL + 0.4, 10.0)
tb = asm.append("props/cable_ties_tools", only=["ASSET_cable_tie_box"])
asm.place(tb.root, (-0.25, -0.55, 0.0), yaw=0.5, scale=1.5)
asm.show(tb, 1.4, 10.0)
src_b = next(o for o in mac.objs if o.name.endswith("bundle_cords"))
src_t = next(o for o in mac.objs if o.name == "cable_tie")
BUNDLES = [((0.55, -0.12, 0.04), 0.35, 4.6), ((0.05, -0.18, 0.04), -0.5, 5.8), ((0.75, -0.40, 0.04), 1.2, 7.0)]
for pos, yaw, tt in BUNDLES:
    grp = _empty("bundle_grp")
    asm.place(grp, pos, yaw=yaw, scale=8.0)
    for so in (src_b, src_t):
        c = so.copy()
        bpy.context.scene.collection.objects.link(c)
        c.parent = grp
        c.matrix_parent_inverse.identity()
        c.location = (0, 0.02, 0) if so is src_t else (0, 0, 0)
        c.rotation_euler = (0, 0, 0)
        c.hide_render = c.hide_viewport = False
        L.V(c, 0, tt, 10.0)

# cardboard box with the gag label (no number on it): right-length fibre, ETA next Tuesday
ETA_POS = (0.95, -0.25, 0.0)
def flat_box(name, loc, size, color, rough=0.85, parent=None):
    o = L.box(name, (0, 0, 0), size, color, rough=rough)
    o.data.transform(Matrix.Diagonal((size[0], size[1], size[2], 1.0)))
    o.scale = (1, 1, 1)
    if parent is not None:
        o.parent = parent
        o.matrix_parent_inverse.identity()
    o.location = loc
    return o


eta = flat_box("eta_box", (ETA_POS[0], ETA_POS[1], 0.19), (0.50, 0.40, 0.38), (0.55, 0.38, 0.22), rough=0.9)
eta.rotation_euler = (0, 0, -0.67)     # label face (+x) turned toward the 3.3-6.9 s camera
_lab = flat_box("eta_label", (0.253, 0.0, 0.0), (0.004, 0.37, 0.29), (0.92, 0.90, 0.84), rough=0.8, parent=eta)


def label_text(body, size, loc, parent, color=(0.08, 0.06, 0.05)):
    cu = bpy.data.curves.new("eta_txt", "FONT")
    cu.body = body
    cu.size = size
    cu.font = L.font("Arial Black")
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    o = bpy.data.objects.new("eta_txt", cu)
    bpy.context.scene.collection.objects.link(o)
    o.data.materials.append(L.mat(color, rough=0.7))
    o.parent = parent
    o.location = loc
    o.rotation_euler = (PI / 2, 0, PI / 2)     # text faces +x (the camera side of the box)
    return o


label_text("RIGHT-LENGTH\nFIBER", 0.052, (0.259, 0.0, 0.07), eta)
label_text("ETA: NEXT\nTUESDAY", 0.050, (0.259, 0.0, -0.065), eta, color=(0.75, 0.05, 0.03))
for o in [eta, _lab] + [c for c in eta.children]:
    L.V(o, 0, 3.3, 10.0)        # not in the 1.5-3.3 pull shot (it sat in the foreground there)

# ------------------------------------------------------------------ effects
asm.fx("shockwave", T_CRACK, loc=TIP_AT_CRACK, scale=0.9, dur=0.45, intensity=0.35)
asm.fx("impact_stars", T_HIT, loc=HIT_POINT, scale=0.7, dur=0.8, intensity=0.35)
asm.fx("shockwave", T_HIT, loc=HIT_POINT, scale=0.6, dur=0.4, intensity=0.35)
KICK_PT = tuple(BUTT)
asm.fx("impact_stars", T_KICK, loc=KICK_PT, scale=0.6, dur=0.6, intensity=0.45)
asm.fx("shockwave", T_KICK, loc=KICK_PT, scale=0.6, dur=0.4, intensity=0.4)
_rng = np.random.RandomState(6)
_crumb_mat = L.mat((0.16, 0.30, 0.70), emit=0.0)       # overalls-blue clay crumbs off the backside
for _k in range(14):
    _dv = _rng.normal(size=3)
    _dv[2] = abs(_dv[2]) * 0.8 + 0.2
    _dv /= np.linalg.norm(_dv)
    _v = _dv * _rng.uniform(0.9, 2.2)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=_rng.uniform(0.012, 0.024), location=HIT_POINT)
    _c = bpy.context.active_object
    _c.name = "crumb_%02d" % _k
    _c.data.materials.append(_crumb_mat)
    for _j in range(0, 13, 2):
        _tt = _j / 30.0
        _c.location = (HIT_POINT[0] + _v[0] * _tt, HIT_POINT[1] + _v[1] * _tt, max(0.02, HIT_POINT[2] + _v[2] * _tt - 0.5 * 9.81 * _tt * _tt))
        _c.keyframe_insert("location", frame=F_HIT + _j)
    L.V(_c, 0, T_HIT - 0.01, T_HIT + 12 / 30.0)

# ------------------------------------------------------------------ captions: VO subtitles, cards, gag words (corner layers, off the contact)
L.OVL["BIG_TL"] = dict(L.OVL["BIG"], loc=(-0.75, 1.75), size=0.5)
L.OVL["BIG_TR"] = dict(L.OVL["BIG"], loc=(0.85, 1.75), size=0.5)
L.WRAP["BIG_TL"] = L.WRAP["BIG_TR"] = 10
asm.timecode(6)
asm.narr_vo(6)
asm.card(0.0, 1.5, "SMF-28: <= 0.18 dB/km at 1550 nm (Corning); 0.4 dB per connector adds up")
asm.card(1.5, 3.3, "9,072 fibers per rack: hypothetical NVL576-style, 200G per fiber (own estimate)")
asm.lab(8.6, 10.0, "THE CUSTOMER")
asm.wl("CONNECTOR JAM", (JAM_AT.x - 0.02, JAM_AT.y + 0.05, JAM_AT.z + 0.16), 0.9, 1.5, size=0.03)
asm.fx("impact_stars", 0.95, loc=tuple(JAM_AT), scale=0.15, dur=0.6, intensity=0.6)
asm.fxn(1.5, 3.2, "[FX: Gary heaves, tray slides out, fibers spill like noodles]")
asm.fxn(4.4, 6.9, "[FX: fibers squirm, one threads through Gary's bullet hole]")
asm.fxn(6.9, 8.6, "[FX: AOC bundle whip (verlet), crack, hit on the backside]")
asm.fxn(8.6, 9.8, "[FX: customer kicks the Manager's backside, launch squash]")
L.ovt("BIG_TL", "CRACK", 0, T_CRACK, min(T_CRACK + 0.5, 8.55))
L.ovt("BIG_TR", "KICK", 0, T_KICK, T_KICK + 0.43)

# ------------------------------------------------------------------ camera (v1.2: 4 cuts; eased moves; slow first and last 0.5 s)
EZ = "BEZIER"
M = MOUTH
asm.shot(0.0, 0.85, (M.x - 0.42, M.y - 0.15, M.z + 0.05), (M.x - 0.98, M.y, M.z), (M.x - 0.30, M.y - 0.40, M.z + 0.10), (M.x - 0.05, M.y, M.z + 0.01), lens=40, ease=EZ)
asm.shot(0.85, 1.5, (M.x - 0.30, M.y - 0.40, M.z + 0.10), (M.x - 0.05, M.y, M.z + 0.01), (M.x - 0.25, M.y - 0.37, M.z + 0.09), (M.x - 0.04, M.y, M.z + 0.01), lens=40, ease=EZ)
asm.shot(1.5, 3.3, (1.95, 0.28, 1.30), (0.22, -0.40, 0.95), (1.85, 0.20, 1.25), (0.28, -0.32, 0.90), lens=22, ease=EZ)
asm.shot(3.3, 5.0, (2.85, -1.30, 1.30), (0.30, -1.45, 0.90), (2.75, -1.32, 1.28), (0.32, -1.40, 0.88), lens=26, ease=EZ)
asm.shot(5.0, 6.9, (2.75, -1.32, 1.28), (0.32, -1.40, 0.88), (1.95, -1.05, 1.05), (0.55, -0.42, 0.55), lens=26, lens1=30, ease=EZ)
asm.shot(6.9, 8.6, (3.30, -1.55, 1.40), (0.35, -1.50, 1.35), (3.10, -1.45, 1.35), (0.35, -1.25, 1.15), lens=25, ease=EZ)
asm.shot(8.6, 10.0, (2.45, -1.85, 1.20), (-0.25, -2.75, 0.95), (2.42, -1.88, 1.20), (-0.25, -2.70, 0.95), lens=30, ease=EZ)


def noise_mods(obj, strength, scale, seed, frame_range=None, blend=6):
    ad = obj.animation_data
    for fc in ad.action.fcurves:
        if fc.data_path != "location":
            continue
        m = fc.modifiers.new("NOISE")
        m.strength, m.scale, m.phase = strength, scale, seed * 7.31 + fc.array_index * 3.17
        m.depth = 1
        if frame_range:
            m.use_restricted_range = True
            m.frame_start, m.frame_end = frame_range
            m.blend_in, m.blend_out = 1.0, float(blend)


noise_mods(L.CAM, 0.006, 24.0, 1, frame_range=(F(1.5), F(9.4)), blend=10)   # handheld (off in the first/last 0.5 s and in the macro)
noise_mods(L.CAM, 0.08, 2.5, 3, frame_range=(F_HIT, F_HIT + 14))
noise_mods(L.CAM, 0.04, 2.5, 5, frame_range=(F_CRACK, F_CRACK + 8), blend=4)
noise_mods(L.CAM, 0.05, 2.5, 7, frame_range=(F_KICK, F_KICK + 10))
scn.render.use_motion_blur = True
scn.render.motion_blur_shutter = 0.5
scn.render.motion_blur_position = "START"     # shutter opens on the frame: no blur smear across the hard cuts
log("T_HIT %.3f (frame %d, want %d), T_CRACK %.3f (frame %d, want %d), Manager at %s, arrival %.2f" % (T_HIT, F_HIT, F_HIT_WANT, T_CRACK, F_CRACK, F_CRACK_WANT, M_POS, T_ARR))

tweak_materials()
asm.finalize(OUT)
