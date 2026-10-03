"""S6 "Fiber: the tray" (film 50-60 s, scene-local 0-10 s), v1.0 assembly from the accurate asset library.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s06_fiber.py -- scenes/v1/s06_fiber.blend

Coordinates (metres, world): hall aisle along +x, cold aisle y in [-0.6, 0.6], row b racks at y > 0.6 with fronts at y = 0.6
(facing -y). Characters face -y at yaw 0. Timeline and captions copy build_crude_film.s6 (scene-local seconds).
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
import s06_whip_sim as WS  # noqa: E402  (verlet whip chain, v1.1)

L = asm.L


def show_objs(asset_or_coll, t0, t1):
    """Visibility window per object (Collection.hide_render is not animatable in Blender 4.2, so asm.show fails at finalize;
    needed framework change: asm.show should key the objects, see devlog)."""
    objs = asset_or_coll.objs if isinstance(asset_or_coll, asm.Asset) else asm._all_objs(asset_or_coll)
    for o in objs:
        if o.hide_render:      # objects hidden on purpose stay hidden
            continue
        L.V(o, 0, t0, t1)


asm.show = show_objs   # asm.fx() resolves show() at call time
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(asm.PROJ, "scenes", "v1", "s06_fiber.blend")

# ------------------------------------------------------------------ layout constants (documented in the devlog)
X_T = 0.3                      # tray rack centre x (replaces hall rack b16)
Y_RACK_FRONT = 0.6             # hall rack front plane
Y_STUB = Y_RACK_FRONT + 0.5    # stub centre (stub is 1.0 m deep, front at -y)
GARY_POS = (0.3, -1.15, 0.0)
CORD_SCALE = 8.0               # patch cord / connector scale-up for the intro macro (real: 2.0 mm jacket)
HIGH = os.environ.get("S06_LOD", "low") == "high"   # 9,072-strand variant (cost test); default is the 600-strand LOD
PILE_RADIUS_M = 0.0025 if HIGH else 0.004   # strand radius in the pile (real cord 1 mm radius): x2.5 high, x4 low LOD
PILE_STRANDS = 9072 if HIGH else 600
Z_STRETCH = 0.744 / 0.35       # spaghetti tray-lip height 0.35 m -> drawer top 0.744 m


def log(*a):
    print("[s06]", *a, flush=True)


# ------------------------------------------------------------------ scene, light, hall
scn = asm.new_scene(preset="standard", world="WORLD_data_hall")
lr = asm.rig("data_hall")
lr.root.rotation_euler = (0, 0, PI / 2)
lr.root.location = (1.4, -0.9, 0.0)
lr.root.scale = (1.0, 1.0, 1.0)

hall = asm.append("datacenter/datahall_environment")
asm.place(hall.root, (0, 0, 0))
RACK_HIDE_A = (-3.6, 4.8)      # row a racks cleared over the working bay (the aisle is only 1.2 m wide)
RACK_CULL = (-6.0, 7.5)        # racks outside this x range are never in frame: hidden for render cost
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
            if row == "b" and abs(x - X_T) < 0.01:      # rack b16 is replaced by the pull-out tray rack
                o.hide_render = o.hide_viewport = True
            if x < RACK_CULL[0] or x > RACK_CULL[1]:
                o.hide_render = o.hide_viewport = True
    if any(k in n for k in ("pedestal", "stringer", "subfloor", "signage", "containment_roof")):
        o.hide_render = o.hide_viewport = True   # under-floor, signs, containment roof (roof would shadow the staging area)

# extra soft light over the tray position (the dark tray against dark racks otherwise reads poorly)
_ld = bpy.data.lights.new("tray_fill", "SPOT")
_ld.energy, _ld.spot_size, _ld.spot_blend, _ld.shadow_soft_size = 40.0, 0.55, 0.8, 0.2
_ld.color = (0.9, 0.95, 1.0)
_lo = bpy.data.objects.new("tray_fill", _ld)
bpy.context.scene.collection.objects.link(_lo)
_lo.location = (X_T, -0.9, 1.8)
_lo.rotation_euler = (math.radians(180 - 62), 0, 0)   # points down and toward +y (the drawer)

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
    old = next(l for l in ng.links if l.to_node.name == sp_node.name and l.to_socket.name == "Geometry")
    ng.links.remove(old)

    def math(op, a=None, b=None, c=None, x=0, y=0):
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
    idx.location = (-1400, -300)
    strand = math("FLOOR", math("DIVIDE", idx.outputs[0], 32.0, x=-1300, y=-300).outputs[0], x=-1150, y=-300)
    local = math("SUBTRACT", idx.outputs[0], math("MULTIPLY", strand.outputs[0], 32.0, x=-1150, y=-450).outputs[0], x=-1000, y=-400)
    rank = math("DIVIDE", strand.outputs[0], math("SUBTRACT", gi.outputs["Strands"], 1.0, x=-1150, y=-600).outputs[0], x=-1000, y=-600)
    fs = math("MULTIPLY", gi.outputs["Fill"], math("ADD", gi.outputs["Spread"], 1.0, x=-1000, y=-750).outputs[0], x=-850, y=-700)
    rs = math("MULTIPLY", rank.outputs[0], gi.outputs["Spread"], x=-850, y=-600)
    prog = math("SUBTRACT", fs.outputs[0], rs.outputs[0], x=-700, y=-650)
    prog = math("MINIMUM", math("MAXIMUM", prog.outputs[0], 0.0, x=-600, y=-650).outputs[0], 1.0, x=-500, y=-650)
    lim = math("MULTIPLY", prog.outputs[0], 31.0, x=-350, y=-650)
    gt = math("GREATER_THAN", local.outputs[0], math("ADD", lim.outputs[0], 0.001, x=-250, y=-750).outputs[0], x=-100, y=-600)
    dele = N.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    dele.location = (0, -300)
    ng.links.new(gi.outputs["Geometry"], dele.inputs["Geometry"])
    ng.links.new(gt.outputs[0], dele.inputs["Selection"])
    tr = N.new("GeometryNodeTransform")
    tr.location = (200, -300)
    comb = N.new("ShaderNodeCombineXYZ")
    comb.location = (0, -800)
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

# ------------------------------------------------------------------ Gary
gary = asm.append("characters/gary", actions=True)
asm.place(gary.root, GARY_POS, yaw=PI)
asm.show(gary, 3.2, 10.0)


def play_part(asset, action, t, f0, f1, repeat=1.0, speed=1.0, hold=False):
    """NLA strip of an action sub-range [f0, f1] starting at scene time t."""
    arm = asset.armature
    nm = action if action.startswith("ACT_") else "ACT_%s_%s" % (asset.root["asset_id"], action)
    act = bpy.data.actions[nm]
    ad = arm.animation_data_create()
    ad.action = None
    tr = ad.nla_tracks.new()
    st = tr.strips.new(nm + "_p", F(t), act)
    st.action_frame_start = f0
    st.action_frame_end = f1
    st.scale = 1.0 / speed
    st.repeat = repeat
    st.blend_type = "REPLACE"
    st.extrapolation = "HOLD_FORWARD" if hold else "NOTHING"
    return st


# kneel (frames 0-24) then the tying loop (24-84) until the whip hit
play_part(gary, "kneel_and_tie", 3.2, 0, 24)
play_part(gary, "kneel_and_tie", 3.2 + 24 / 30.0, 24, 84, repeat=math.ceil((10.0 - 4.0) / 2.0))   # v1.1: loop runs to the end; the jolt strip (placed at the whip contact) overrides it

# bullet holes: healed radius schedule r = max(0.3, 0.9 ** floor((T - T_shot) / 1.5)), film time T = 50 + t, key every 1.5 s
SHOTS = {"p_hole_1_radius": 9.2, "p_hole_2_radius": 19.2, "p_head_hole_radius": 28.95, "p_hole_4_radius": 38.9, "p_hole_5_radius": 49.1}
for prop, ts in SHOTS.items():
    t = 0.0
    while t < 10.0 + 1e-6:
        steps = math.floor((50.0 + t - ts) / 1.5)
        r = max(0.3, 0.9 ** max(steps, 0))
        asm.key_prop(gary.root, prop, t, r, "CONSTANT")
        # next change at the next 1.5 s boundary
        nxt = ts + (steps + 1) * 1.5 - 50.0
        if nxt > 10.0 or r <= 0.3 + 1e-9:
            break
        t = max(nxt, t + 1e-3)
# faces
asm.key_prop(gary.root, "p_expr_worried", 3.2, 0.0)
asm.key_prop(gary.root, "p_expr_worried", 5.0, 0.8)
asm.key_prop(gary.root, "p_expr_sweating", 4.4, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 7.0, 0.8)

# ------------------------------------------------------------------ Manager + AOC whip (v1.1: baked verlet chain, hit-stop, squash)
mgr = asm.append("characters/manager", actions=True)
M_POS = (2.05, -1.55, 0.0)     # v1.0: x = 2.55; moved 0.5 m closer so the simulated whip tip reaches Gary's torso
asm.place(mgr.root, M_POS, yaw=0)
M_YAW = asm.yaw_to(M_POS, GARY_POS)
asm.show(mgr, 3.0, 10.0)       # v1.1: Manager is in the scene from the start of his walk-in (cutaway 6.7-7.4 shows him)

# walk in along the aisle (x decreasing) 3.2 -> 7.2
mgr_start = (M_POS[0] + 5.0, M_POS[1], 0.0)
asm.walk(mgr, 3.2, [mgr_start, (M_POS[0], M_POS[1], 0.0)])
asm.key_loc(mgr.root, 7.2, M_POS, yaw=asm.yaw_to(mgr_start, M_POS), interp="CONSTANT")
asm.key_loc(mgr.root, 7.3, M_POS, yaw=M_YAW, interp="LINEAR")
asm.key_prop(mgr.root, "p_anger", 6.0, 0.0)
asm.key_prop(mgr.root, "p_anger", 7.2, 0.5)
asm.key_prop(mgr.root, "p_anger", 8.0, 1.0)

whip = asm.append("props/aoc_whip", actions=True)
asm.show(whip, 3.0, 10.0)
WHIP_ROT = Euler((0, 0, PI))
_grip = Vector((0, -0.325, 0))   # HOOK_grip in whip root coordinates
# v1.1 thickness: scale of the bundle cross-section (x and z of every whip mesh about the whip axis), root scale stays 1 so the bake is rigid-body clean
WHIP_THICK, PLUG_THICK = 7.0, 3.5          # bundle (cables, ties, tape) x7, QSFP-DD-style plugs x3.5 (cable 3 mm real -> 21 mm; bundle about 6 cm)
_seen = set()
for _o in whip.objs:
    if _o.type == "MESH" and _o.data.name not in _seen:
        _seen.add(_o.data.name)
        _k = PLUG_THICK if "_plug" in _o.name else WHIP_THICK
        _o.data.transform(Matrix.Diagonal((_k, 1.0, _k, 1.0)))
asm.attach(whip.root, mgr.hook("whip_grip_L"), offset=tuple(-(WHIP_ROT.to_matrix() @ _grip)), rot=tuple(WHIP_ROT))
whip_arm = whip.armature
ARM_T0, ARM_SPEED = 7.2, 19.0 / 24.0
ARM_ACT = bpy.data.actions["ACT_manager_arm_whip"]
HS = 3                                                       # hit-stop frames


def mgr_arm_strips(af_c=None):
    """Manager arm_whip NLA: plain strip, or (af_c given) split at action frame af_c with a HS-frame freeze (hit-stop)."""
    ad = mgr.armature.animation_data_create()
    ad.action = None
    tr = ad.nla_tracks.new()
    sc = 1.0 / ARM_SPEED
    f0 = float(F(ARM_T0))
    end = ARM_ACT.frame_range[1]

    def strip(name, start, a0, a1, scale):
        st = tr.strips.new(name, int(round(start)), ARM_ACT)
        st.action_frame_start, st.action_frame_end = a0, a1
        st.scale = scale
        st.blend_type, st.extrapolation = "REPLACE", "NOTHING"
        return st
    if af_c is None:
        st = strip("arm_whip", f0, 0.0, end, sc)
        return tr
    a = strip("arm_whip_A", f0, 0.0, af_c, sc)
    c = strip("arm_whip_freeze", a.frame_end, af_c, af_c + 0.5, HS / 0.5)
    strip("arm_whip_B", c.frame_end, af_c, end, sc)
    return tr


def whip_inputs(f0, f1):
    """Evaluate per frame: whip root world matrix (handle points), Gary bone capsules. Heavy GN modifiers are muted while stepping."""
    muted = []
    for o in bpy.data.objects:
        for md in o.modifiers:
            if md.type == "NODES" and md.show_viewport:
                md.show_viewport = False
                muted.append(md)
    H, W, caps = [], [], []
    garm = gary.armature
    for f in range(f0, f1 + 1):
        scn.frame_set(f)
        Wm = whip.root.matrix_world.copy()
        W.append(Wm)
        H.append([tuple(Wm @ Vector((0, -WS.L_SEG * i, 0))) for i in range(WS.PINNED)])
        cl = []
        for nm, r in WS.GARY_R.items():
            pb = garm.pose.bones[nm]
            cl.append((np.array(garm.matrix_world @ pb.head), np.array(garm.matrix_world @ pb.tail), r, nm in WS.TORSO))
        caps.append(cl)
    for md in muted:
        md.show_viewport = True
    return np.array(H), W, caps


mgr_arm_strips()
SIM_F0, SIM_F1 = F(3.2), 300
scn.frame_set(SIM_F0)
# pass 1: Gary keeps kneeling (no jolt yet); find the first torso contact
_H, _W, _caps = whip_inputs(SIM_F0, SIM_F1)
_P, _rec = WS.simulate(_H, _caps)
_hits = [r for r in _rec if r[2] > 1.5]
_dm = []
for _i in range(len(_P)):
    _best = 9.0
    for (_a, _b, _r, _t) in _caps[_i]:
        if _t:
            for _pp in _P[_i, WS.PINNED:]:
                _best = min(_best, np.linalg.norm(_pp - WS._closest_on_seg(_pp, _a, _b)) - _r - WS.R_CABLE)
    _dm.append(_best)
log("whip pass 1 clearance to torso (m) by frame: min %.3f at frame %d; hips@min %s" % (min(_dm), SIM_F0 + int(np.argmin(_dm)), [round(x, 2) for x in _caps[int(np.argmin(_dm))][0][0]]))
assert _hits, "whip never reached Gary: tune M_POS"
_fi, _pt, _spd = _hits[0]
F_HIT = SIM_F0 + _fi
_ts = WS.tip_speed(_P)
_lo = F(7.4) - SIM_F0
F_CRACK = SIM_F0 + _lo + int(np.argmax(_ts[_lo:_fi + 1]))
T_HIT, T_CRACK = (F_HIT - 1) / 30.0, (F_CRACK - 1) / 30.0
log("whip pass 1: crack frame %d (t=%.3f, tip %.1f m/s), first torso contact frame %d (t=%.3f, point %d, %.1f m/s)" % (F_CRACK, T_CRACK, _ts[F_CRACK - SIM_F0], F_HIT, T_HIT, _pt, _spd))
# pass 2: hit-stop (Manager arm frozen HS frames, whip chain frozen) and Gary's reaction after the stop
for _tr in list(mgr.armature.animation_data.nla_tracks):
    mgr.armature.animation_data.nla_tracks.remove(_tr)
mgr.armature.animation_data.action = None
asm.walk(mgr, 3.2, [mgr_start, (M_POS[0], M_POS[1], 0.0)])    # re-create the walk strip (tracks were cleared)
mgr_arm_strips(af_c=(F_HIT - F(ARM_T0)) * ARM_SPEED)
asm.play(gary, "jolt_hit", T_HIT + HS / 30.0, blend_in=2)
scn.frame_set(SIM_F0)
_H, _W, _caps = whip_inputs(SIM_F0, SIM_F1)
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
# Gary squash reaction on contact (clay body: z compresses, x/y bulge; springs back)
for _fr, _sc in ((F_HIT, (1, 1, 1)), (F_HIT + 2, (1.07, 1.07, 0.88)), (F_HIT + HS + 3, (0.97, 0.97, 1.04)), (F_HIT + HS + 9, (1, 1, 1))):
    gary.root.scale = _sc
    gary.root.keyframe_insert("scale", frame=_fr)
asm.key_prop(gary.root, "p_expr_dread", T_HIT - 0.1, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_expr_dread", T_HIT, 1.0, "CONSTANT")
scn.frame_set(1)

# ------------------------------------------------------------------ customers: NVYDIA slaps the Manager, OPENAY / ANTHROPY hugs his leg
def walk_speed(asset, gait="walk"):
    return float(bpy.data.actions["ACT_%s_%s" % (asset.root["asset_id"], gait)].get("root_speed_mps", 1.2))


def rz(v, th):
    return (v[0] * math.cos(th) - v[1] * math.sin(th), v[0] * math.sin(th) + v[1] * math.cos(th))


nv = asm.append("characters/npc_nvydia_leather", actions=True)   # v1.1: black leather jacket variant (new asset, base asset untouched)
NV_YAW = -PI / 2                                           # faces -x
_off = rz((-0.5, -0.4), NV_YAW)                            # slap target: 0.5 m to his right, 0.4 m ahead (slap action note)
NV_POS = (M_POS[0] - _off[0], M_POS[1] - _off[1], 0.0)
T_SLAP_START = 9.147 - 11 / 30.0                           # contact at action frame 11 -> 9.147 s
nv_start = (NV_POS[0] + 1.8, NV_POS[1], 0.0)
t_nv0 = T_SLAP_START - 1.8 / walk_speed(nv)
asm.walk(nv, t_nv0, [nv_start, NV_POS])
asm.key_loc(nv.root, T_SLAP_START, NV_POS, yaw=NV_YAW, interp="CONSTANT")
asm.play(nv, "slap", T_SLAP_START, hold=True)
asm.show(nv, t_nv0 - 0.05, 10.0)

op = asm.append("characters/npc_openay", actions=True)
_o = rz((0.40, -0.10), NV_YAW)
OP_POS = (NV_POS[0] + _o[0], NV_POS[1] + _o[1], 0.0)
OP_YAW = asm.yaw_to(OP_POS, NV_POS)
op_start = (OP_POS[0] + 2.0, OP_POS[1] - 0.2, 0.0)
T_HUG = 9.0
t_op0 = T_HUG - 2.0 / walk_speed(op)
asm.walk(op, t_op0, [op_start, OP_POS])
asm.key_loc(op.root, T_HUG, OP_POS, yaw=OP_YAW, interp="CONSTANT")
asm.play(op, "hug_leg", T_HUG, hold=True)
asm.show(op, t_op0 - 0.05, 10.0)

# Manager reaction to the slap: stagger, shock face, head snap
asm.play(mgr, "stagger", 9.147, hold=True, repeat=1.0)
asm.key_prop(mgr.root, "p_expr_shock", 9.1, 0.0, "CONSTANT")
asm.key_prop(mgr.root, "p_expr_shock", 9.147, 1.0, "CONSTANT")
asm.key_loc(mgr.root, 9.147, M_POS, yaw=M_YAW, interp="CONSTANT")
asm.key_loc(mgr.root, 9.19, M_POS, yaw=M_YAW - 0.7, interp="LINEAR")

# ------------------------------------------------------------------ intro (v1.1): glowing MPO patch cord jams crooked into an MPO receptacle (macro station)
MACRO_S = 8.0                  # MPO plug / adapter plate scale-up for the macro (real plug 12.4 x 8 mm); documented in the devlog
mac = asm.append("interconnect/fiber_cables_and_trays")   # kept only as the source of bundle_cords / cable_tie meshes (all variants hidden)
for ch in mac.coll.children:
    ch.hide_render = ch.hide_viewport = True
    for lc in asm._layer_colls(bpy.context.view_layer.layer_collection):
        if lc.collection == ch:
            lc.exclude = False
plate = asm.append("interconnect/mpo_connector_and_adapter", only=["ASSET_mpo_adapter_plate4", "ASSET_mpo_plug"])
plug = plate.others[0]
MOUTH_T = Vector((1.78, -0.30, 1.41))        # where the jam happens (the camera aims here)
asm.place(plate.root, (0, 0, 0), yaw=-PI / 2, scale=MACRO_S)
bpy.context.view_layer.update()
_hp = plate.hook("port_2").matrix_world.translation.copy()
plate.root.location = tuple(MOUTH_T - _hp)
bpy.context.view_layer.update()
MOUTH = plate.hook("port_2").matrix_world.translation.copy()
asm.show(plate, 0.0, 1.55)
asm.show(plug, 0.0, 1.55)
glow = L.mat((0.15, 0.75, 1.0), emit=0.9)     # v1.0 used 2.2 (bloom over-glow at draft)
for o in plug.objs:
    if o.type == "MESH" and o.name.endswith("_cable"):
        o.data.materials.clear()
        o.data.materials.append(glow)
plug.root.scale = (MACRO_S,) * 3
PLUG_FAR = 1.0                                # start 1.0 m before the mouth (world x)


def plug_pose(t, x, yaw_err=0.0, roll=0.0, dy=0.0, dz=0.0):
    plug.root.location = (MOUTH.x - x, MOUTH.y + dy, MOUTH.z + dz)
    plug.root.rotation_euler = (0.0, roll, PI / 2 + yaw_err)    # Euler XYZ: roll about the plug axis (asset Y), then yaw about Z; plug face looks +x
    plug.root.keyframe_insert("location", frame=F(t))
    plug.root.keyframe_insert("rotation_euler", frame=F(t))
    L._interp(plug.root, "location", F(t), "LINEAR")
    L._interp(plug.root, "rotation_euler", F(t), "LINEAR")


# approach (ease-in, 0 -> 0.85 s), touch at 0.85-0.95 s (crooked: key rolled 0.28 rad, yaw 0.2 rad), push back, then rattle until 1.55 s
for tt, xx in ((0.0, PLUG_FAR), (0.2, 0.84), (0.45, 0.45), (0.7, 0.16), (0.85, 0.045)):
    plug_pose(tt, xx, yaw_err=0.0, roll=0.0)
plug_pose(0.95, 0.03, yaw_err=0.20, roll=0.28, dz=0.004)
for k in range(9):
    tt = 1.0 + 0.06 * k
    sg = 1 if k % 2 == 0 else -1
    plug_pose(tt, 0.03 + 0.006 * sg, yaw_err=0.20 + 0.05 * sg, roll=0.28 + 0.04 * sg, dz=0.004 * sg)
# cord: 4-point NURBS, front two points fixed, rear two follow the plug cable end
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
_cu.materials.append(glow)
_hm = mpo_cord.modifiers.new("hk_end", "HOOK")
_hm.object = plug.hook("cable_end")
_hm.vertex_indices_set([2, 3])
_hm.matrix_inverse = plug.hook("cable_end").matrix_world.inverted()
L.V(mpo_cord, 0, 0.0, 1.55)
JAM_AT = MOUTH + Vector((-0.04, 0.0, 0.0))

# ------------------------------------------------------------------ the fiber that threads through Gary's hole 2 (healed radius 0.3 of base)
hole2 = next(o for o in gary.objs if o.name == "HOLE_hole_2")
scn.frame_set(F(6.6))
bpy.context.view_layer.update()
_h2 = hole2.evaluated_get(bpy.context.evaluated_depsgraph_get())
_Mw = _h2.matrix_world.copy()
_H = _Mw.translation.copy()
_ax = (_Mw.to_3x3() @ Vector((0, 0, 1))).normalized()
log("hole2 world", [round(x, 3) for x in _H], "axis", [round(x, 3) for x in _ax])
THREAD_R = 0.006
cu = bpy.data.curves.new("thread_fiber", "CURVE")
cu.dimensions = "3D"
cu.bevel_depth = THREAD_R
cu.bevel_resolution = 3
sp_ = cu.splines.new("NURBS")
sp_.order_u = 4
sp_.use_endpoint_u = True
_SV = [-0.9, -0.5, -0.12, 0.0, 0.12, 0.5, 0.9]
_axh = Vector((_ax.x, _ax.y, 0.0)).normalized()
sp_.points.add(len(_SV) - 1)
_PW = []
for pt, sv in zip(sp_.points, _SV):
    if abs(sv) <= 0.12:
        w = _H + _ax * sv                 # through the (tilted) hole along its axis
    else:
        w = _H + _axh * sv                # outside: horizontal run, sagging to the pile / floor
        w.z = max(0.03, _H.z * 0.35) if abs(sv) < 0.7 else 0.03
    _PW.append(w)
    pt.co = (w.x, w.y, w.z, 1.0)
thread = bpy.data.objects.new("thread_fiber", cu)
bpy.context.scene.collection.objects.link(thread)
cu.materials.append(L.mat((1.0, 0.45, 0.05), emit=0.8))   # v1.0: 1.5


def _empty(name, parent=None):
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    if parent is not None:
        e.parent = parent
        e.matrix_parent_inverse.identity()
    return e


e_front = _empty("thread_end_front")
e_front.location = tuple(_PW[0])
e_back = _empty("thread_end_back")
e_back.location = tuple(_PW[-1])
e_hole = _empty("thread_hole_anchor", hole2)       # moves with Gary's spine; slides along the hole axis
asm.key_loc(e_hole, 5.4, (0, 0, 0.6), interp="BEZIER")
asm.key_loc(e_hole, 6.6, (0, 0, 0.0), interp="BEZIER")
scn.frame_set(F(6.6))
bpy.context.view_layer.update()
for nm, em, idxs, wm in (("hk_front", e_front, [0, 1], Matrix.Translation(e_front.location)),
                         ("hk_hole", e_hole, [2, 3, 4], _Mw),
                         ("hk_back", e_back, [5, 6], Matrix.Translation(e_back.location))):
    hm = thread.modifiers.new(nm, "HOOK")
    hm.object = em
    hm.vertex_indices_set(idxs)
    hm.matrix_inverse = wm.inverted()
L.V(thread, 0, 5.4, 10.0)
scn.frame_set(1)

# ------------------------------------------------------------------ props: tie gun in Gary's right hand, tie box, bundled fibers with ties
tg = asm.append("props/cable_ties_tools", only=["ASSET_cable_tie_gun"])
_R = Euler((0, 0, PI))
_gg = Vector((0.001, 0.056, 0.04))
asm.attach(tg.root, gary.hook("hand_R"), offset=tuple(-(_R.to_matrix() @ _gg)), rot=tuple(_R))
asm.show(tg, 3.2, 10.0)
tb = asm.append("props/cable_ties_tools", only=["ASSET_cable_tie_box"])
asm.place(tb.root, (0.95, -1.05, 0.0), yaw=0.5, scale=1.5)
asm.show(tb, 3.2, 10.0)

# bundles: bundle_cords + cable_tie meshes of fiber_cables_and_trays (ties_and_velcro variant) cloned and scaled x8 like the intro cord
src_b = next(o for o in mac.objs if o.name.endswith("bundle_cords"))
src_t = next(o for o in mac.objs if o.name == "cable_tie")
BUNDLES = [((0.62, -0.78, 0.04), 0.35, 4.6), ((-0.05, -0.9, 0.04), -0.5, 5.8), ((0.55, -1.5, 0.04), 1.2, 7.0)]
for pos, yaw, tt in BUNDLES:
    grp = bpy.data.objects.new("bundle_grp", None)
    bpy.context.scene.collection.objects.link(grp)
    asm.place(grp, pos, yaw=yaw, scale=8.0)
    for so in (src_b, src_t):
        c = so.copy()
        bpy.context.scene.collection.objects.link(c)
        c.parent = grp
        c.matrix_parent_inverse.identity()
        c.location = (0, 0, 0)
        c.rotation_euler = (0, 0, 0)
        c.hide_render = c.hide_viewport = False
        L.V(c, 0, tt, 10.0)
        if so is src_t:
            c.location = (0, 0.02, 0)

# ------------------------------------------------------------------ effects (crack at the tip's peak speed, thwack on contact, slap)
asm.fx("shockwave", T_CRACK, loc=TIP_AT_CRACK, scale=0.9, dur=0.45, intensity=0.4)
asm.fx("impact_stars", T_HIT, loc=HIT_POINT, scale=0.8, dur=0.9, intensity=0.4)
asm.fx("shockwave", T_HIT, loc=HIT_POINT, scale=0.7, dur=0.45, intensity=0.4)
asm.fx("impact_stars", 9.147, loc=(M_POS[0] - 0.1, M_POS[1] - 0.25, 1.7), scale=0.7, dur=0.9, intensity=0.8)
asm.fx("shockwave", 9.147, loc=(M_POS[0] - 0.1, M_POS[1] - 0.25, 1.7), scale=0.8, dur=0.45, intensity=0.6)
# clay-crumb puffs on the hit (small clay-coloured chips: tiny low-poly icospheres with ballistic keyed paths, fixed seed)
_rng = np.random.RandomState(6)
_crumb_mat = L.mat((0.12, 0.12, 0.13), emit=0.0)
for _k in range(14):
    _d = _rng.normal(size=3)
    _d[2] = abs(_d[2]) * 0.8 + 0.2
    _d /= np.linalg.norm(_d)
    _v = _d * _rng.uniform(0.9, 2.2)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=_rng.uniform(0.012, 0.024), location=HIT_POINT)
    _c = bpy.context.active_object
    _c.name = "crumb_%02d" % _k
    _c.data.materials.append(_crumb_mat)
    for _j in range(0, 13, 2):
        _tt = _j / 30.0
        _c.location = (HIT_POINT[0] + _v[0] * _tt, HIT_POINT[1] + _v[1] * _tt, max(0.02, HIT_POINT[2] + _v[2] * _tt - 0.5 * 9.81 * _tt * _tt))
        _c.keyframe_insert("location", frame=F_HIT + _j)
    L.V(_c, 0, T_HIT - 0.01, T_HIT + 12 / 30.0)

# ------------------------------------------------------------------ captions (copied from the crude s6)
asm.timecode(6)
asm.narr([(0.2, 3.4, "Last job: Gary pulls out the tray and ties up the fibers."), (3.4, 4.6, "Thousands of them."),
          (5.4, 8.0, "Manager wants it done yesterday.")])
asm.lab(0.0, 1.5, "FIBER")
asm.lab(8.6, 10.0, "THE CUSTOMER")
asm.card(0.0, 1.5, "SMF-28: <= 0.18 dB/km at 1550 nm (Corning); 0.4 dB per connector adds up")
asm.card(3.2, 8.0, "9,072 fibers per rack: hypothetical NVL576-style, 200G per fiber (own estimate)")
asm.wl("CONNECTOR JAM", (JAM_AT.x - 0.02, JAM_AT.y + 0.05, JAM_AT.z + 0.22), 0.9, 1.5, size=0.035)
asm.wl("CUSTOMER", (NV_POS[0], NV_POS[1], 2.25), 8.9, 10.0, size=0.1)
asm.fx("impact_stars", 0.95, loc=tuple(JAM_AT), scale=0.15, dur=0.6, intensity=0.7)
asm.fxn(1.5, 3.2, "[FX: tray slides out, fibers spill like noodles]")
asm.fxn(4.4, 7.4, "[FX: fibers squirm, one threads through Gary's bullet hole]")
asm.fxn(7.4, 8.7, "[FX: bundle of active optical cables as the whip; crack shockwave]")
asm.fxn(9.0, 9.8, "[FX: slap shockwave, Manager's head snaps; customer hug squash]")
asm.big(T_CRACK, T_CRACK + 0.6, "CRACK")
asm.big(9.1, 9.7, "SLAP")

# ------------------------------------------------------------------ camera (crude sequence adapted to the real hall; v1.1: eased moves, Manager cutaway, handheld noise, hit shake)
EZ = "BEZIER"
asm.shot(0.0, 0.9, (-1.9, -2.0, 1.3), (0.2, -0.25, 1.35), (-0.3, -1.9, 1.3), (1.1, -0.25, 1.38), lens=35, ease=EZ)
asm.shot(0.9, 1.5, (MOUTH.x - 0.8, MOUTH.y - 0.8, MOUTH.z + 0.1), tuple(MOUTH), (MOUTH.x - 0.55, MOUTH.y - 0.65, MOUTH.z + 0.06), tuple(MOUTH), lens=50, ease=EZ)
asm.shot(1.5, 3.2, (-1.3, -2.7, 1.25), (0.2, 0.1, 0.85), (0.3, -2.3, 1.0), (0.3, 0.0, 0.75), lens=30, ease=EZ)
asm.shot(3.2, 5.6, (1.9, -2.9, 1.3), (0.3, -0.4, 0.5), (1.1, -2.6, 1.1), (0.3, -0.4, 0.5), ease=EZ)
asm.shot(5.6, 6.7, (-1.0, -2.4, 0.85), (0.38, -1.05, 0.65), (-0.1, -1.9, 0.8), (0.38, -1.05, 0.65), lens=35, lens1=50, ease=EZ)
# v1.1 Manager cutaway: he walks in with the whip trailing on the floor (visibility fix: full body in frame)
asm.shot(6.7, 7.4, (0.6, -3.6, 1.0), (3.0, -1.6, 1.0), (0.9, -3.5, 1.05), (2.6, -1.6, 1.05), lens=32, ease=EZ)
# wind-up and swing: wide enough for the whip arc overhead (whip tip reaches z = 3 m)
asm.shot(7.4, T_HIT - 0.02, (-0.1, -3.95, 1.5), (1.25, -1.55, 1.55), (0.1, -3.85, 1.5), (1.2, -1.5, 1.5), lens=22, lens1=24, ease=EZ)   # y >= -4.05: beyond that the camera is inside the row-a racks
asm.shot(T_HIT - 0.02, 8.6, (1.9, -2.9, 1.0), (0.45, -1.15, 0.8), (1.7, -2.7, 1.05), (0.4, -1.15, 0.85), lens=32, ease=EZ)
asm.shot(8.6, 10.0, (1.9, -4.05, 1.8), (1.9, -1.4, 0.95), (1.9, -3.85, 1.7), (1.9, -1.4, 0.95), lens=21, ease=EZ)


def noise_mods(obj, strength, scale, seed, frame_range=None, blend=6):
    """Deterministic F-curve noise on location (handheld, or a hit shake restricted to a frame range)."""
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


noise_mods(L.CAM, 0.012, 24.0, 1)                                   # handheld
noise_mods(L.CAM, 0.10, 2.5, 3, frame_range=(F_HIT, F_HIT + 14))    # hit shake
noise_mods(L.CAM, 0.05, 2.5, 5, frame_range=(F_CRACK, F_CRACK + 8), blend=4)
noise_mods(L.TGT, 0.006, 30.0, 2)
scn.render.motion_blur_shutter = 0.5                                # motion blur is on in the hero preset

asm.finalize(OUT)
