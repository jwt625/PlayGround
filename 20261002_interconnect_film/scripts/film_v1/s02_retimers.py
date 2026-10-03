"""Scene S2 "Retimers: inside the rack, row after row" (film 10-20 s, scene-local 0-10 s), v1 assembly from the asset library.

Build:  /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s02_retimers.py -- scenes/v1/s02_retimers.blend
Timing, captions, cards and FX notes are copied from the crude s2 (scripts/build_crude_film.py); camera shots are adapted to real-size assets.
Stage: rack interior at x = RX (stylised tray stack, detail scale 4, pitch 0.7 m); lab (bench, scope, whiteboard, Gary, Manager) around x = 2.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asm  # noqa: E402
import blender_lib as L  # noqa: E402
import tex_gen as T  # noqa: E402

PROJ = asm.PROJ
TEXDIR = os.path.join(PROJ, "assets", "generated_textures", "v1", "s02")
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(PROJ, "scenes", "v1", "s02_retimers.blend")
PI = math.pi
F = asm.F

# ------------------------------------------------------------------ layout
RX = -6.0                      # rack stage x
BX, BY = 2.7, -0.8             # bench origin (x centre, y centre; z = 0 floor)
WBX, WBY = 1.4, 1.5            # whiteboard bottom-centre on the wall plane
WB_Z0 = 0.9                    # board bottom edge height (asset suggestion)
G_POS = (1.3, -1.5, 0.0)      # Gary
M_POS = (-0.4, 0.8, 0.0)        # Manager
NT = 8
GUN_ROT = (0.0, 0.0, PI)       # shotgun root (barrels -Y) onto the hand hook frame (Y along fingers)
GUN_OFF = (0.0, 0.0, 0.0)
S_DETAIL = 4.0                 # stylised rack interior: hardware detail scale (rack JSON p_detail_scale)
CHIP_S = 9.0                   # v1.1: retimer chips drawn at 9x (135 mm; was 4x = 60 mm, too small)
COL_DX = 0.12                  # v1.1: two retimer columns per connector at +-COL_DX
RACK_LAB = (4.85, BY, 0.0)     # v1.1: NVL72-style rack next to the bench (lab shots)
# camera hops inside the rack: arrival time of the camera at tray j (seconds); hop duration, dwell height above the board
T_ARR = [0.8, 2.2, 2.6, 3.0, 3.4, 3.85, 4.35, 4.85]
HOP = 0.20
CAM_H = 0.55
LAYOUT_SEQ = ["gb300_compute", "gb300_compute", "nvlink_switch", "gb300_compute", "helios_compute", "nvlink_switch", "gb300_compute", "helios_compute"]
SHAKES = []                    # (t_hit, amplitude rad, decay 1/s, freq Hz) camera shake events (baked into the target noise)


def ramp(t, ta, tb, va, vb):
    if t <= ta:
        return va
    if t >= tb:
        return vb
    return va + (vb - va) * (t - ta) / (tb - ta)


def yaw_to(a, b):
    return asm._yaw(a, b)


def show(asset_or_coll, t0, t1):
    """Visibility window for a whole asset/collection. asm.show keys Collection.hide_render, which is not animatable in
    Blender 4.2 (TypeError at finalize), so key every object of the collection through blender_lib.V instead."""
    c = asset_or_coll.coll if isinstance(asset_or_coll, asm.Asset) else asset_or_coll
    for o in asm._all_objs(c):
        L.V(o, 0, t0, t1)


def fx(name, t, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, parent=None, intensity=1.0, dur=1.5, show_pad=0.05):
    """asm.fx with the object-level visibility window (see show)."""
    a = asm.append("materials_vfx/vfx_particles_and_effects", only=["ASSET_fx_" + name])
    r = a.root
    if parent is not None:
        r.parent = parent
        r.matrix_parent_inverse.identity()
    r.location, r.rotation_euler, r.scale = loc, rot, (scale,) * 3
    if "p_auto" in r.keys():
        r["p_auto"] = 0
    if "p_intensity" in r.keys():
        r["p_intensity"] = intensity
    asm.key_prop(r, "p_t", t, 0.0)
    asm.key_prop(r, "p_t", t + dur, dur)
    show(a, t - show_pad, t + dur + show_pad)
    return a


def link_to(coll, objs):
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)


# ------------------------------------------------------------------ v1.1 camera rig: eased baked moves, handheld noise, shake
def smoother(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


def spring(tau, w=22.0, z=7.0):
    """Damped spring response to a unit kick (0 for tau < 0): exp(-z tau) cos(w tau)."""
    return 0.0 if tau < 0 else math.exp(-z * tau) * math.cos(w * tau)


def _lerp(a, b, e):
    return tuple(x + (y - x) * e for x, y in zip(a, b))


def noise3(t, seed):
    """Smooth deterministic noise in [-1, 1] (sum of sines with fixed phases)."""
    return (math.sin(2 * PI * 0.37 * t + seed * 1.7) * 0.5 + math.sin(2 * PI * 0.91 * t + seed * 3.1) * 0.3
            + math.sin(2 * PI * 1.73 * t + seed * 5.3) * 0.2)


class CamRig:
    def __init__(self):
        cam, tgt, hold = L.CAM, L.TGT, L.HOLD
        self.cam, self.tgt, self.hold = cam, tgt, hold
        self.tn = L.empty("CAM_TARGET_NOISY")
        for c in cam.constraints:
            if c.type == "TRACK_TO":
                c.target = self.tn
        # overlay holder must not shake: follow a clean camera (same position, aimed at the clean target)
        camc = L.empty("CAM_CLEAN")
        c1 = camc.constraints.new("COPY_LOCATION")
        c1.target = cam
        c2 = camc.constraints.new("TRACK_TO")
        c2.target = tgt
        c2.track_axis = "TRACK_NEGATIVE_Z"
        c2.up_axis = "UP_Y"
        hold.parent = camc
        hold.matrix_parent_inverse.identity()

    def key(self, f, pos, aim, lens, hh=0.0025, t=None):
        """Key one frame: camera position, clean aim, lens, overlay scale, noisy aim (handheld hh rad + shake events)."""
        t = (f - 1) / 30.0 if t is None else t
        cam, tgt, tn, hold = self.cam, self.tgt, self.tn, self.hold
        cam.location = pos
        cam.keyframe_insert("location", frame=f)
        tgt.location = aim
        tgt.keyframe_insert("location", frame=f)
        cam.data.lens = lens
        cam.data.keyframe_insert("lens", frame=f)
        hold.scale = (L.HOLD_S0 * L.LENS0 / lens,) * 3
        hold.keyframe_insert("scale", frame=f)
        d = Vector(aim) - Vector(pos)
        dist = d.length
        d.normalize()
        right = d.cross(Vector((0, 0, 1)))
        right.normalize()
        up = right.cross(d)
        ay = hh * noise3(t, 1.0)
        ap = hh * noise3(t, 2.0)
        for (th, amp, dec, fr) in SHAKES:
            if t >= th:
                e = amp * math.exp(-dec * (t - th))
                ay += e * math.sin(2 * PI * fr * (t - th) + 0.6)
                ap += e * math.cos(2 * PI * fr * 1.13 * (t - th))
        tn.location = Vector(aim) + (right * ay + up * ap) * dist
        tn.keyframe_insert("location", frame=f)

    def shot(self, t0, t1, p0, a0, p1=None, a1=None, lens=28.0, lens1=None, ease=smoother, hh=0.0025):
        p1 = p0 if p1 is None else p1
        a1 = a0 if a1 is None else a1
        lens1 = lens if lens1 is None else lens1
        f0, f1 = F(t0), F(t1) - 1
        for f in range(f0, f1 + 1):
            u = (f - f0) / max(f1 - f0, 1)
            e = ease(u)
            self.key(f, _lerp(p0, p1, e), _lerp(a0, a1, e), lens + (lens1 - lens) * e, hh)

    def finish(self):
        for o in (self.cam, self.tgt, self.tn, self.hold, self.cam.data):
            ad = o.animation_data
            if ad and ad.action:
                for fc in ad.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"


def rack_cam_z(t, tray_z):
    """Camera height at time t: dwells at tray j (arrival T_ARR[j]) and hops to the next tray in HOP seconds (smoother-step)."""
    for j in range(len(T_ARR) - 1, 0, -1):
        if t >= T_ARR[j] - HOP:
            u = (t - (T_ARR[j] - HOP)) / HOP
            return (tray_z[j - 1] + CAM_H) + (tray_z[j] - tray_z[j - 1]) * smoother(u)
    return tray_z[0] + CAM_H


# ------------------------------------------------------------------ textures (tex_gen, same parameters as the crude s2)
def make_eye_sequence():
    d = os.path.join(TEXDIR, "eye_s2")
    params = []
    for f in range(1, 301):
        t = (f - 1) / 30.0
        params.append(dict(sigma_g=ramp(t, 5.4, 7.4, 0.45, 0.38), noise=ramp(t, 5.4, 7.4, 0.30, 0.05), jitter=ramp(t, 5.4, 7.4, 0.12, 0.03)))
    if os.path.exists(os.path.join(d, "eye_s2_0300.png")):
        print("EYE sequence exists, skipped")
    else:
        bers = T.eye_sequence(d, "eye_s2", params, seed=22)
        print("EYE eye_s2 BER first %.2e last %.2e min %.2e max %.2e" % (bers[0], bers[-1], min(bers), max(bers)))
    return os.path.join(d, "eye_s2_0001.png"), 300, 1


GP = lambda t: ramp(t, 6.4, 8.6, 0.0, 1.0)  # noqa: E731
G_F0 = 1


def make_graph_sequence():
    d = os.path.join(TEXDIR, "graph_s2")
    n = 300 - G_F0 + 1
    if os.path.exists(os.path.join(d, "graph_s2_%04d.png" % n)):
        print("GRAPH sequence exists, skipped")
    else:
        T.graph_sequence(d, "graph_s2", [GP((f - 1) / 30.0) for f in range(G_F0, 301)])
    return os.path.join(d, "graph_s2_0001.png"), n, G_F0


def plug_sequence(mat, fac_node, first_png, n, start):
    """Plug an image sequence into a lab_office screen material (SCREEN_IMAGE, SCREEN_FAC / BOARD_FAC)."""
    nt = mat.node_tree
    tex = nt.nodes["SCREEN_IMAGE"]
    img = bpy.data.images.load(first_png)
    img.source = "SEQUENCE"
    img.filepath = bpy.path.relpath(first_png, start=os.path.dirname(os.path.abspath(OUT)))
    tex.image = img
    tex.interpolation = "Closest"
    iu = tex.image_user
    iu.frame_duration = n
    iu.frame_start = start
    iu.frame_offset = 0
    iu.use_auto_refresh = True
    nt.nodes[fac_node].outputs[0].default_value = 1.0


# ------------------------------------------------------------------ materials for the chips
def prep_chip_materials(tmpl_objs):
    """Body materials: multiply by object alpha (Object Info) so each chip fades in; glow material: p_glow read from the object."""
    body_mats = set()
    for o in tmpl_objs:
        if o.type == "MESH" and o.name != "glow":
            for m in o.data.materials:
                if m:
                    body_mats.add(m)
    for m in body_mats:
        nt = m.node_tree
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
        link = out.inputs["Surface"].links[0]
        src = link.from_socket
        mix = nt.nodes.new("ShaderNodeMixShader")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        nt.links.new(oi.outputs["Alpha"], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(src, mix.inputs[2])
        nt.links.remove(link)
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        m.surface_render_method = "DITHERED"
        m.use_transparent_shadow = True


def prep_glow_material(m):
    """Glow quad: remove the driver on the asset's mix factor (it points at the first root) and read p_glow from the quad itself.

    p_glow semantics kept: 0 = off, 1 = peak flash; settled orange = 0.55. Colour goes white-hot above 0.55 and strength rises with it.
    """
    nt = m.node_tree
    if nt.animation_data:
        for d in list(nt.animation_data.drivers):
            nt.animation_data.drivers.remove(d)
    mix = next(n for n in nt.nodes if n.type == "MIX_SHADER")
    em = next(n for n in nt.nodes if n.type == "EMISSION")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = "p_glow"
    # fac = clamp(p_glow); hot = clamp((p - 0.55) / 0.45)
    hot = nt.nodes.new("ShaderNodeMapRange")
    hot.inputs["From Min"].default_value = 0.55
    hot.inputs["From Max"].default_value = 1.0
    hot.clamp = True
    nt.links.new(at.outputs["Fac"], hot.inputs["Value"])
    cmix = nt.nodes.new("ShaderNodeMix")
    cmix.data_type = "RGBA"
    base = em.inputs["Color"].default_value
    cmix.inputs["A"].default_value = (1.0, 0.42, 0.08, 1.0)  # settled orange (crude: 1.0, 0.45, 0.1)
    cmix.inputs["B"].default_value = (1.0, 0.85, 0.55, 1.0)
    nt.links.new(hot.outputs["Result"], cmix.inputs["Factor"])
    nt.links.new(cmix.outputs["Result"], em.inputs["Color"])
    sm = nt.nodes.new("ShaderNodeMath")
    sm.operation = "MULTIPLY_ADD"
    sm.inputs[1].default_value = 9.0
    sm.inputs[2].default_value = 5.0
    nt.links.new(hot.outputs["Result"], sm.inputs[0])
    nt.links.new(sm.outputs[0], em.inputs["Strength"])
    cl = nt.nodes.new("ShaderNodeMath")
    cl.operation = "MULTIPLY"
    cl.use_clamp = True
    cl.inputs[1].default_value = 1.0
    nt.links.new(at.outputs["Fac"], cl.inputs[0])
    # soft round falloff: the 18 mm quad would otherwise flash as a hard white square
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.0, 2.0, 2.0)
    mp.inputs["Location"].default_value = (-1.0, -1.0, 0.0)
    gr = nt.nodes.new("ShaderNodeTexGradient")
    gr.gradient_type = "QUADRATIC_SPHERE"
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], gr.inputs["Vector"])
    fall = nt.nodes.new("ShaderNodeMath")
    fall.operation = "MULTIPLY"
    fall.use_clamp = True
    nt.links.new(cl.outputs[0], fall.inputs[0])
    nt.links.new(gr.outputs["Fac"], fall.inputs[1])
    for l in list(mix.inputs[0].links):
        nt.links.remove(l)
    nt.links.new(fall.outputs[0], mix.inputs[0])
    m.surface_render_method = "BLENDED"


def build_chip_template():
    a = asm.append("packaging/narrowcom_retimer_chip")
    a.variant("mounted", False)
    a.variant("heat_spreader", False)
    keep, glow = [], None
    for o in a.objs:
        n = o.name
        if o.type != "MESH":
            continue
        if n.endswith("_glow"):
            glow = o
        elif n.endswith(("_substrate", "_mold", "_valley_mark", "_wordmark", "_row_0", "_row_1", "_row_2", "_row_3", "_row_qr",
                         "_pin1_dimple", "_pin1_underside")):
            keep.append(o)
    # join the visible BASE parts into one mesh (single object per chip; the chips share it as linked duplicates)
    bpy.ops.object.select_all(action="DESELECT")
    for o in keep:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = keep[0]
    with bpy.context.temp_override(active_object=keep[0], selected_objects=keep, selected_editable_objects=keep):
        bpy.ops.object.join()
    body = keep[0]
    body.name = "chip_body"
    glow.name = "glow"
    prep_chip_materials([body])
    prep_glow_material(glow.data.materials[0])
    # everything else in the asset is not needed in the chip copies (balls, passives, mounted/spreader variants, source meshes, hooks)
    for o in list(bpy.data.objects):
        if o.name in a.coll.all_objects and o not in (a.root, body, glow):
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(a.coll.children):
        bpy.data.collections.remove(c)
    return a, body, glow


def make_chip(tmpl_root, body, glow, coll, name):
    r = tmpl_root.copy()
    r.name = "chip_root_" + name
    b = body.copy()
    b.name = "chip_body_" + name
    g = glow.copy()
    g.name = "chip_glow_" + name
    for o in (r, b, g):
        coll.objects.link(o)
    b.parent = r
    g.parent = r
    r["p_glow"] = 0.0
    g["p_glow"] = 0.0
    # glow quad reads p_glow from itself: drive it from the chip root (the asset's control property)
    fc = g.driver_add('["p_glow"]')
    dv = fc.driver
    dv.type = "AVERAGE"
    var = dv.variables.new()
    var.name = "v"
    var.type = "SINGLE_PROP"
    var.targets[0].id = r
    var.targets[0].data_path = '["p_glow"]'
    return r, b, g


# ------------------------------------------------------------------ main build
def build():
    scn = asm.new_scene(preset="standard", world="WORLD_cartoon_sky")
    asm.rig("daylight", scale=1.6, loc=(1.5, 0.0, 0.0))
    cam_rig = CamRig()
    scn.render.motion_blur_shutter = 0.5   # motion blur ready (preset enables it for hero; draft/standard leave it off)
    eye_png, eye_n, eye_start = make_eye_sequence()
    gr_png, gr_n, gr_start = make_graph_sequence()
    TRACKS = []

    # ---------------- floor (checker, 1 m tiles x2: storyboard cross-scene rule)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(-2.0, 0.0, 0.0))
    fl = bpy.context.object
    fl.name = "floor"
    fm = bpy.data.materials.new("floor_checker")
    fm.use_nodes = True
    nt = fm.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    ck = nt.nodes.new("ShaderNodeTexChecker")
    ck.inputs["Scale"].default_value = 20.0  # 40 m plane: 20 checks per axis pair = 2 m period (1 m tiles)
    ck.inputs["Color1"].default_value = (0.72, 0.74, 0.78, 1)
    ck.inputs["Color2"].default_value = (0.55, 0.58, 0.64, 1)
    nt.links.new(ck.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    fl.data.materials.append(fm)
    fl.visible_shadow = True
    L.V(fl, 0, 0.0, 0.8)
    L.V(fl, 0, 5.3, 10.0)
    bpy.ops.mesh.primitive_plane_add(size=12, location=(RX, 0.0, -0.002))
    sf = bpy.context.object
    sf.name = "rack_stage_floor"
    sm_ = bpy.data.materials.new("S2_stage_floor")
    sm_.use_nodes = True
    sm_.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.012, 0.013, 0.016, 1)
    sm_.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    sf.data.materials.append(sm_)
    L.V(sf, 0, 0.8, 5.4)
    # the plane is z = 0; the rack stage floor is the same plane

    # ---------------- lab: bench + scope + whiteboard
    bench = asm.append("lab_office/lab_bench")
    asm.place(bench.root, (BX, BY, 0.0), yaw=0.0)
    scope = asm.append("lab_office/bench_oscilloscope")
    bpy.context.view_layer.update()
    slot = bench.hook("scope_slot").matrix_world.translation
    asm.place(scope.root, (slot.x, slot.y, slot.z), yaw=0.0)
    wb = asm.append("lab_office/whiteboard_big")
    asm.place(wb.root, (WBX, WBY, WB_Z0), yaw=0.0)
    def scope_hook(nm):
        return next(o for o in scope.objs if o.name.startswith("HOOK_" + nm)).matrix_world.translation.copy()

    scr = next(o for o in scope.objs if o.name == "bench_oscilloscope_screen")
    plug_sequence(scr.material_slots[0].material, "SCREEN_FAC", eye_png, eye_n, eye_start)
    wbs = next(o for o in wb.objs if o.name == "whiteboard_big_screen")
    plug_sequence(wbs.material_slots[0].material, "BOARD_FAC", gr_png, gr_n, gr_start)

    # ---------------- characters
    gary = asm.append("characters/gary", actions=True)
    mgr = asm.append("characters/manager", actions=True)
    gyaw0 = yaw_to(G_POS, (slot.x - 0.06, slot.y - 0.08))
    myaw0 = yaw_to(M_POS, (WBX, WBY))  # faces the whiteboard (beside its left end)
    gyaw1 = yaw_to(G_POS, M_POS)
    myaw1 = yaw_to(M_POS, G_POS)
    if myaw1 - myaw0 > PI:
        myaw1 -= 2 * PI
    elif myaw1 - myaw0 < -PI:
        myaw1 += 2 * PI
    asm.place(gary.root, G_POS, yaw=gyaw0)
    asm.place(mgr.root, M_POS, yaw=myaw0)
    # Gary: idle stare, turns to the Manager at 8.5-8.8, hit at 9.2, topples
    asm.play(gary, "idle", 0.0, hold=False, repeat=9.4 / 2.0)
    asm.play(gary, "topple_back", 9.2 + 0.1, speed=2.4)   # 3-frame hit-stop before the fall (camera shake + puffs cover it)
    asm.key_loc(gary.root, 0.0, G_POS, yaw=gyaw0, interp="LINEAR")
    asm.key_loc(gary.root, 8.5, G_POS, yaw=gyaw0, interp="BEZIER")
    asm.key_loc(gary.root, 8.85, G_POS, yaw=gyaw1, interp="LINEAR")
    asm.key_loc(gary.root, 10.0, G_POS, yaw=gyaw1, interp="LINEAR")
    # Gary face: worried, relieved when the eye recovers, shocked when the Manager turns
    asm.key_prop(gary.root, "p_expr_worried", 0.0, 0.9)
    asm.key_prop(gary.root, "p_expr_worried", 6.4, 0.9)
    asm.key_prop(gary.root, "p_expr_worried", 7.4, 0.0)
    asm.key_prop(gary.root, "p_expr_happy", 6.4, 0.0)
    asm.key_prop(gary.root, "p_expr_happy", 7.4, 0.7)
    asm.key_prop(gary.root, "p_expr_happy", 8.5, 0.7)
    asm.key_prop(gary.root, "p_expr_happy", 8.85, 0.0)
    asm.key_prop(gary.root, "p_expr_scared", 8.5, 0.0)
    asm.key_prop(gary.root, "p_expr_scared", 8.85, 1.0)
    # holes: hole 1 from S1 (shot at film T = 9.2 s, steps of x0.9 every 1.5 s, floor 0.3), hole 2 at 9.2 s (this scene)
    T_SHOT1 = 9.2
    n_step, tt = 0, 0.0
    asm.key_prop(gary.root, "p_hole_1_radius", 0.0, max(0.3, 0.9 ** math.floor((10.0 - T_SHOT1) / 1.5)), "CONSTANT")
    k = 1
    while True:
        t_local = T_SHOT1 + 1.5 * k - 10.0
        if t_local > 10.0:
            break
        if t_local > 0.0:
            asm.key_prop(gary.root, "p_hole_1_radius", t_local, max(0.3, 0.9 ** k), "CONSTANT")
        k += 1
    asm.key_prop(gary.root, "p_hole_2_radius", 0.0, 0.0, "CONSTANT")
    asm.key_prop(gary.root, "p_hole_2_radius", 9.2, 1.0, "CONSTANT")

    # Manager: thinking pose at the whiteboard, turns at 8.6, shoots at 9.2
    asm.play(mgr, "thinking", 0.0, hold=False, repeat=9.0 / 2.0)
    asm.play(mgr, "aim_gun", 8.6, speed=36.0 / 18.0, blend_in=4)
    # v1.1: the Manager turns to face the camera and turns angry at t = 7.6-8.4 (stays in frame), then turns to Gary and shoots at 9.2
    def unwrap(prev, a):
        while a - prev > PI:
            a -= 2 * PI
        while a - prev < -PI:
            a += 2 * PI
        return a
    myawc = unwrap(myaw0, yaw_to(M_POS, (-0.45, -2.7)))
    myaw1 = unwrap(myawc, yaw_to(M_POS, G_POS))
    back = Vector((M_POS[0] - G_POS[0], M_POS[1] - G_POS[1], 0.0)).normalized()
    ms = lambda u: u * u * (3 - 2 * u)  # noqa: E731

    def mgr_state(tt):
        if tt < 7.5:
            yw = myaw0
        elif tt < 8.1:
            yw = myaw0 + (myawc - myaw0) * smoother((tt - 7.5) / 0.6)
        elif tt < 8.5:
            yw = myawc
        else:
            yw = myawc + (myaw1 - myawc) * smoother((tt - 8.5) / 0.5)
        pos = Vector(M_POS)
        if 8.15 < tt < 8.62:   # anger tremble
            a_ = 0.012 * math.sin(2 * PI * 13.0 * tt) * math.sin(PI * (tt - 8.15) / 0.47)
            pos += Vector((a_, 0.6 * a_, 0))
        if tt >= 9.2:          # recoil, springy settle
            tau = tt - 9.2
            pos += back * 0.11 * math.exp(-6.0 * tau) * math.sin(16.0 * tau)
        return tuple(pos), yw
    asm.key_loc(mgr.root, 0.0, M_POS, yaw=myaw0, interp="LINEAR")
    for f in range(F(7.4), F(10.0) + 1):
        pos, yw = mgr_state((f - 1) / 30.0)
        asm.key_loc(mgr.root, (f - 1) / 30.0, pos, yaw=yw, interp="LINEAR")
    asm.key_prop(mgr.root, "p_anger", 7.6, 0.0)
    asm.key_prop(mgr.root, "p_anger", 8.2, 1.0)
    asm.key_prop(mgr.root, "p_flush", 7.6, 0.0)
    asm.key_prop(mgr.root, "p_flush", 8.4, 1.0)
    # anger steam from both ears 8.1-9.2 (ear hooks of the effect parented to the Manager's steam hooks)
    es = fx("ear_steam", 8.1, dur=1.1)
    for side in ("L", "R"):
        eh = next(o for o in es.objs if o.name.startswith("HOOK_ear_" + side))
        sh = next(o for o in mgr.objs if o.name.startswith("HOOK_steam_" + side))
        eh.parent = sh
        eh.matrix_parent_inverse.identity()
        eh.location = (0, 0, 0)
        eh.rotation_euler = (0, 0, 0)

    gun = asm.append("props/shotgun")
    ghook = next(o for o in mgr.objs if o.name.startswith("HOOK_gun_grip_R"))
    asm.attach(gun.root, ghook, rot=GUN_ROT, offset=GUN_OFF)
    show(gun, 8.55, 10.0)
    bpy.context.view_layer.update()
    for side in ("L", "R"):
        mh = next(o for o in gun.objs if o.name.startswith("HOOK_muzzle_" + side))
        fx("muzzle_flash", 9.2, rot=(-PI / 2, 0, 0), scale=1.0, parent=mh, dur=0.14)
    mh = next(o for o in gun.objs if o.name.startswith("HOOK_muzzle_R"))
    fx("smoke_ring", 9.25, rot=(-PI / 2, 0, 0), scale=1.0, parent=mh, dur=0.9)
    # clay-crumb puffs + impact stars at Gary's chest on the hit (the hit-stop frames are covered by these and the camera shake)
    dv = Vector((M_POS[0] - G_POS[0], M_POS[1] - G_POS[1], 0.0)).normalized()
    hitp = (G_POS[0] + dv.x * 0.17, G_POS[1] + dv.y * 0.17, 1.12)
    fx("dust_cloud", 9.2, loc=hitp, scale=0.35, dur=1.0)
    fx("impact_stars", 9.2, loc=hitp, scale=0.4, dur=0.5)

    # ---------------- rack interior (stylised tray stack) at x = RX, with the retimer chips
    rack = asm.append("datacenter/rack_interior_tray_stack")
    asm.place(rack.root, (RX, 0.0, 0.0))
    bpy.context.view_layer.update()
    # v1.1: the plain silkscreen footprints (one chip per connector) no longer match the 2-column chips: remove them
    for o in [o for o in rack.objs if o.name.endswith("_silk")]:
        bpy.data.objects.remove(o, do_unlink=True)
    rack.objs = asm._all_objs(rack.coll)
    tmpl, body, glow = build_chip_template()
    tmpl.coll.hide_render = True
    tmpl.coll.hide_viewport = True
    chips = bpy.data.collections.new("S2_retimer_chips")
    scn.collection.children.link(chips)
    hooks = {o.name: o for o in rack.objs if o.name.startswith("HOOK_")}
    tray_z = [hooks["HOOK_tray%d_surface" % j].matrix_world.translation.z for j in range(NT)]
    Yb = 0.8

    # ---- v1.1 dressing: dummy components (3 layouts), NVL72 backplane cartridge bands
    dress = bpy.data.collections.new("S2_rack_dressing")
    scn.collection.children.link(dress)
    lay = {}
    for nm in ("gb300_compute", "nvlink_switch", "helios_compute"):
        a_ = asm.append("datacenter/tray_dummy_layouts", only=["ASSET_tray_dummy_" + nm])
        lay[nm] = next(o for o in a_.objs if o.type == "MESH")
        lay[nm].parent = None
        lay[nm].hide_render = True
        a_.root.hide_render = True
        a_.root.hide_viewport = True
        lay[nm].hide_viewport = True
        link_to(dress, [lay[nm]])
        bpy.data.collections.remove(a_.coll)
    cb = asm.append("datacenter/nvl72_backplane_cartridges")
    band = next(o for o in cb.objs if o.type == "MESH")
    band.parent = None
    link_to(dress, [band])
    bpy.data.collections.remove(cb.coll)
    for m_ in band.data.materials:
        if m_ and "label_white" in m_.name:
            m_.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.32, 0.3, 1)
    n_dress = 0
    for j in range(NT):
        src = lay[LAYOUT_SEQ[j]]
        o = src.copy()
        o.name = "dress_%s_t%d" % (LAYOUT_SEQ[j], j)
        o.hide_render = False
        o.hide_viewport = False
        dress.objects.link(o)
        o.location = (RX, 0.0, tray_z[j])
        o.scale = (S_DETAIL,) * 3
        bn = band.copy()
        bn.name = "cartridge_band_t%d" % j
        dress.objects.link(bn)
        bn.location = (RX, Yb, tray_z[j] - 0.0304 + 0.19)
        bn.scale = (S_DETAIL,) * 3
        n_dress += 1
    for nm in lay:
        dress.objects.unlink(lay[nm])
    dress.objects.unlink(band)
    show(dress, 0.8, 5.9)
    print("DRESSING trays", n_dress)

    # ---- camera z over time (baked later) and per-tray reveal times
    cam_z = lambda tt: rack_cam_z(tt, tray_z)  # noqa: E731
    n_chip = 0
    slide = Vector((0.0, -0.35, 0.35))
    sparks_t = []
    chip_info = {}     # (j, kk, col) -> (root, [glow keys], x, rows)
    for j in range(NT):
        rows = 1 + (1 if j >= 2 else 0) + (1 if j >= 5 else 0)
        for r in range(rows):
            t_row = (0.95 if j == 0 else T_ARR[j] - 0.22) + 0.08 * r
            sparks_t.append((j, r, t_row))
            for kk in range(4):
                for col in (0, 1):
                    h = hooks["HOOK_retimer_row%d_tray%d_%d" % (r, j, kk)]
                    p = h.matrix_world.translation.copy()
                    p.x += (-COL_DX if col == 0 else COL_DX)
                    tk = t_row + 0.05 * kk + 0.025 * col
                    root, b, g = make_chip(tmpl.root, body, glow, chips, "r%d_t%d_%d%s" % (r, j, kk, "ab"[col]))
                    root.scale = (CHIP_S,) * 3
                    g.location = (0, 0, 0.0007)  # lift the glow quad off the board (z-fight)
                    final = Vector((p.x, p.y, p.z))
                    # springy landing: overshoot along the slide direction, damped; squash at contact
                    tl = tk + 0.22
                    for f in range(F(tk), F(tk + 0.7) + 1):
                        tt = (f - 1) / 30.0
                        if tt < tl:
                            u = (tt - tk) / 0.22
                            e = 1 - (1 - smoother(u))
                            off = slide * (1 - e)
                            sq = 0.0
                        else:
                            tau = tt - tl
                            off = Vector((0, -0.012 * math.exp(-8 * tau) * math.sin(20 * tau), 0.0))
                            sq = spring(tau, 24.0, 8.0)
                        root.location = tuple(final + off)
                        root.keyframe_insert("location", frame=f)
                        root.scale = (CHIP_S * (1 + 0.07 * sq), CHIP_S * (1 + 0.07 * sq), CHIP_S * (1 - 0.14 * sq))
                        root.keyframe_insert("scale", frame=f)
                    for o in (b, g):
                        L.V(o, 0, tk, 5.9)
                    b.color = (1, 1, 1, 0.0)
                    b.keyframe_insert("color", index=3, frame=F(tk))
                    b.color = (1, 1, 1, 1.0)
                    b.keyframe_insert("color", index=3, frame=F(tk + 0.22))
                    gk = [(tk, 0.0), (tk + 0.2, 1.0), (tk + 0.9, 0.55)]
                    g.scale = (1.0, 1.0, 1.0)
                    g.keyframe_insert("scale", frame=F(tk))
                    g.scale = (2.3, 2.3, 1.0)
                    g.keyframe_insert("scale", frame=F(tk + 0.2))
                    g.scale = (1.45, 1.45, 1.0)
                    g.keyframe_insert("scale", frame=F(tk + 0.9))
                    chip_info[(j, r, kk, col)] = (root, gk, final.x, final.y, tk)
                    n_chip += 1
    print("CHIPS", n_chip)

    # ---- signal-path pulses: a glowing bead runs from each connector forward along its column through the rows, each chip it
    # passes gets a glow bump. Deterministic, baked.
    bead_me = bpy.data.meshes.new("bead")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.5)
    bm.to_mesh(bead_me)
    bm.free()
    bead_mat = bpy.data.materials.new("S2_bead_glow")
    bead_mat.use_nodes = True
    nt_ = bead_mat.node_tree
    nt_.nodes.clear()
    eo = nt_.nodes.new("ShaderNodeOutputMaterial")
    em = nt_.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.75, 0.35, 1.0)
    em.inputs["Strength"].default_value = 18.0
    nt_.links.new(em.outputs[0], eo.inputs[0])
    bead_me.materials.append(bead_mat)
    beads = bpy.data.collections.new("S2_signal_beads")
    scn.collection.children.link(beads)
    SPD = 1.3
    bump = {}
    for j in range(NT):
        rows = 1 + (1 if j >= 2 else 0) + (1 if j >= 5 else 0)
        t_end = (T_ARR[j + 1] + 0.3) if j + 1 < NT else 5.45
        for kk in range(4):
            for col in (0, 1):
                bd = bpy.data.objects.new("bead_t%d_%d%s" % (j, kk, "ab"[col]), bead_me)
                beads.objects.link(bd)
                hx = hooks["HOOK_retimer_row0_tray%d_%d" % (j, kk)].matrix_world.translation
                x = hx.x + (-COL_DX if col == 0 else COL_DX)
                y0 = Yb - 0.205
                z0 = tray_z[j] + 0.040 + 0.075
                y_last = hooks["HOOK_retimer_row%d_tray%d_%d" % (rows - 1, j, kk)].matrix_world.translation.y - 0.10
                dur = (y0 - y_last) / SPD
                t0 = (0.95 if j == 0 else T_ARR[j] - 0.22) + 0.30 + 0.07 * kk + 0.04 * col
                first = True
                while t0 + dur <= t_end and t0 < 5.8:
                    for tt, yy, s in ((t0, y0, 0.0), (t0 + 0.05, y0 - SPD * 0.05, 1.0), (t0 + dur - 0.05, y_last + SPD * 0.05, 1.0), (t0 + dur, y_last, 0.0)):
                        bd.location = (x, yy, z0)
                        bd.scale = (0.08 * s, 0.20 * s, 0.08 * s)
                        bd.keyframe_insert("location", frame=F(tt))
                        bd.keyframe_insert("scale", frame=F(tt))
                    # glow bump on the chips of this column when the bead passes their row
                    for r in range(rows):
                        yr = chip_info[(j, r, kk, col)][3]
                        tp = t0 + (y0 - yr) / SPD
                        bump.setdefault((j, r, kk, col), []).append(tp)
                    first = False
                    t0 += 0.7
                if bd.animation_data is None:
                    bpy.data.objects.remove(bd, do_unlink=True)
                else:
                    L.V(bd, 0, 0.9, 5.9)
    for key, (root, gk, cx, cy, tk) in chip_info.items():
        ks = list(gk)
        for tp in bump.get(key, []):
            if tp > tk + 1.0:
                ks += [(tp - 0.08, 0.55), (tp, 0.82), (tp + 0.2, 0.55)]
        ks.sort()
        last_f = None
        for tt, v in ks:
            f = F(tt)
            if last_f is not None and f <= last_f:
                continue
            asm.key_prop(root, "p_glow", tt, v)
            last_f = f
    show(rack, 0.8, 5.9)

    # ---- landing sparks: one burst per tray at its first row (appended fx, hidden outside its window)
    for (j, r, tr_) in sparks_t:
        if r == 0:
            xs_ = hooks["HOOK_retimer_row0_tray%d_0" % j].matrix_world.translation
            fx("sparks_burst", tr_ + 0.25, loc=(RX, xs_.y, xs_.z + 0.05), scale=0.3, dur=0.5, intensity=0.5)

    # NARROWCOM sign on the backplane: the chip's own valley mark and wordmark meshes scaled up (parody mark, gold laser material)
    sign_src = asm.append("packaging/narrowcom_retimer_chip")
    sign_src.variant("mounted", False)
    sign_src.variant("heat_spreader", False)
    sc = bpy.data.collections.new("S2_sign")
    scn.collection.children.link(sc)
    vm = next(o for o in sign_src.objs if "_valley_mark" in o.name)
    wm = next(o for o in sign_src.objs if "_wordmark" in o.name)
    for o in list(sign_src.objs):
        if o not in (vm, wm) and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(sign_src.coll.children):
        bpy.data.collections.remove(c)
    for o in (vm, wm):
        o.parent = None
    link_to(sc, [vm, wm])
    bpy.data.collections.remove(sign_src.coll)
    sign_y = Yb - 0.10
    sign_z = tray_z[0] + 0.40
    sg = 70.0
    vm.scale = (sg, sg, sg)
    vm.rotation_euler = (PI / 2, 0, 0)
    vm.location = (RX, sign_y, sign_z + 0.05)
    wm.scale = (sg, sg, sg)
    wm.rotation_euler = (PI / 2, 0, 0)
    wm.location = (RX, sign_y, sign_z - 0.08)
    for o in (vm, wm):
        o.visible_shadow = False
    show(sc, 0.8, 3.0)
    sm = vm.data.materials[0].copy()
    sm.name = "S2_sign_gold"
    nt = sm.node_tree
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Emission Color"].default_value = (0.95, 0.8, 0.35, 1.0)
            n.inputs["Emission Strength"].default_value = 1.2
    for o in (vm, wm):
        o.data.materials[0] = sm

    # rack headlamp: area light following the camera height (baked), energy only in the rack window
    ld = bpy.data.lights.new("rack_lamp", "AREA")
    ld.size = 2.2
    ld.size_y = 0.5
    ld.color = (0.9, 0.95, 1.0)
    lamp = bpy.data.objects.new("rack_lamp", ld)
    scn.collection.objects.link(lamp)
    lamp.rotation_euler = (1.25, 0, 0)
    for f in range(F(0.8), F(5.5) + 1):
        lamp.location = (RX, -1.30, cam_z((f - 1) / 30.0) + 0.12)
        lamp.keyframe_insert("location", frame=f)
    ld.energy = 0.0
    for t, e in ((0.0, 0.0), (0.79, 0.0), (0.8, 150.0), (5.5, 150.0), (5.6, 0.0)):
        ld.energy = e
        ld.keyframe_insert("energy", frame=F(t))
    for fc in list(ld.animation_data.action.fcurves) + list(lamp.animation_data.action.fcurves):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if fc.data_path == "energy" else "LINEAR"

    # ---------------- lab: NVL72-style rack next to the bench and scope, cables rack -> scope (v1.1)
    lrack = asm.append("datacenter/rack_nvl72_style")
    asm.place(lrack.root, RACK_LAB, yaw=0.0)
    bpy.context.view_layer.update()
    ports = []
    for nm in ("HOOK_tray_front_ou14", "HOOK_tray_front_ou15", "HOOK_tray_front_ou23"):
        h = next(o for o in lrack.objs if o.name.startswith(nm))
        ports.append(h.matrix_world.translation.copy())
    scope_pts = [scope_hook("ch1_bnc"), scope_hook("probe_tip"), scope_hook("trigger_level_knob")]
    cable_mat = bpy.data.materials.new("S2_cable_jacket")
    cable_mat.use_nodes = True
    cable_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.035, 0.05, 1)
    cable_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
    cables = []
    NPT = 22
    for ci in range(3):
        a_, b_ = ports[ci], scope_pts[ci]
        cu = bpy.data.curves.new("cable%d" % ci, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.0075
        cu.bevel_resolution = 2
        sp_ = cu.splines.new("POLY")
        sp_.points.add(NPT - 1)
        cu.materials.append(cable_mat)
        co = bpy.data.objects.new("cable_rack_scope_%d" % ci, cu)
        scn.collection.objects.link(co)
        cables.append((co, sp_, a_, b_, 0.55 + 0.12 * ci))

    def cable_pts(a_, b_, sag, ci):
        out = []
        for i in range(NPT):
            s = i / (NPT - 1)
            x = a_.x + (b_.x - a_.x) * s
            y = a_.y + (b_.y - a_.y) * s - 0.55 * math.sin(PI * min(1.0, s * 1.15)) * (1 - s) ** 0.5 - 0.18 * ci * math.sin(PI * s)
            z = a_.z + (b_.z - a_.z) * s - sag * 4 * s * (1 - s)
            z = max(z, 0.012 + 0.0075 if 0.2 < s < 0.8 else z)
            out.append((x, y, z, 1.0))
        return out
    for ci, (co, sp_, a_, b_, sag) in enumerate(cables):
        for f in range(F(5.3), F(6.6) + 1):
            tt = (f - 1) / 30.0
            k = 1.0 + 0.35 * spring(tt - 5.4, 14.0, 5.0)     # springy settle of the cable sag
            for i, q in enumerate(cable_pts(a_, b_, sag * k, ci)):
                sp_.points[i].co = q
                sp_.points[i].keyframe_insert("co", frame=f)
        for i, q in enumerate(cable_pts(a_, b_, sag, ci)):
            sp_.points[i].co = q
        for fc in co.data.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        L.V(co, 0, 5.3, 10.0)
        L.V(co, 0, 0.0, 0.8)
    show(lrack, 5.3, 10.0)
    show(lrack, 0.0, 0.8)

    # ---------------- visibility of the lab/rack windows
    # lab stuff is visible 0-0.8 and 5.4-10 (the camera is inside the rack in between); the rack stage is hidden outside 0.8-5.9
    for a in (bench, scope, wb, gary, mgr):
        show(a, 0.0, 0.8)
        show(a, 5.3, 10.0)

    # ---------------- camera (baked: eased moves, handheld noise, shake on the BANG)
    SHAKES.append((9.2, 0.030, 5.5, 17.0))
    SHAKES.append((9.2 + 0.1, 0.012, 4.0, 11.0))
    # flash cut 0-0.8
    cam_rig.shot(0.0, 0.8, (2.5, -2.6, 1.3), (1.75, -1.0, 1.1), (2.6, -2.4, 1.3), (1.75, -1.0, 1.1), lens=28, ease=lambda u: u)
    # rack: one baked move 0.8-5.4 (dwell + hop at each tray), slow lateral drift, 23 deg downward view
    for f in range(F(0.8), F(5.4)):
        tt = (f - 1) / 30.0
        zc = cam_z(tt)
        u = (tt - 0.8) / 4.6
        cx = RX + 0.30 * math.cos(1.6 * PI * u)
        cy = -1.30 + 0.35 * smoother((tt - 1.0) / 1.2)
        cam_rig.key(f, (cx, cy, zc), (RX + 0.05, 0.30, zc - 0.56), 29.0, hh=0.003)
    # out of the rack: crane down to the lab, rack + bench + scope + cables
    cam_rig.shot(5.4, 6.4, (-0.5, -7.6, 5.6), (3.1, -0.6, 0.9), (3.4, -4.6, 2.2), (3.4, -0.9, 1.05), lens=28)
    # whiteboard: push-in (t=6.4 wide, t=8 close, Manager in frame)
    cam_rig.shot(6.4, 8.6, (0.4, -4.4, 2.2), (1.2, 1.0, 1.3), (-0.45, -2.7, 1.75), (0.75, 1.3, 1.4), lens=26, lens1=30)
    cam_rig.shot(8.6, 9.2, (3.0, 0.2, 1.5), (-0.3, 0.7, 1.3), (2.2, -0.1, 1.45), (-0.4, 0.6, 1.3), lens=28)
    cam_rig.shot(9.2, 10.0, (1.2, -4.4, 1.7), (0.7, -0.4, 1.1), (1.2, -4.2, 1.7), (0.7, -0.4, 1.1), lens=26, hh=0.003)
    cam_rig.finish()

    # ---------------- captions, labels, cards, FX notes (copied from the crude s2)
    asm.narr([(0.2, 1.0, "Signal fading?"), (1.0, 3.8, "Gary puts a chip on every connector."),
              (3.8, 8.0, "Now it takes forever to arrive and cranks up the electricity bill.")])
    asm.lab(0.0, 0.8, "EYE CLOSING")
    asm.lab(0.8, 1.6, "RETIMER ROW ADDED BEHIND EACH CONNECTOR")
    asm.lab(1.6, 5.4, "TRAY AFTER TRAY: MORE ROWS")
    asm.lab(5.4, 6.4, "OUT OF THE RACK")
    asm.lab(6.4, 8.6, "EYE RECOVERS   BER FALLS")
    asm.card(6.4, 8.6, "Passive Cu 0 pJ/b (Arista) -> retimed ~15.6 pJ/b = 25 W / 1.6T (Ciena) [derived]. "
                       "Curves: no units. Eye and BER: illustrative model.")
    asm.fxn(0.8, 1.6, "[FX: glow flash + sparks as each retimer slides in and fades in]")
    asm.fxn(1.6, 5.4, "[FX: particles + glow trails as rows slide in tray by tray]")
    asm.fxn(6.4, 8.6, "[FX: marker squeak as the curves draw]")
    asm.fxn(9.2, 9.8, "[FX: muzzle flash, smoke ring, hole decal]")
    asm.big(9.2, 9.8, "BANG")
    asm.timecode(2)

    # world labels riding next to the newest point of each curve (graph pixel coords -> board plane)
    bw, bh = 2.94, 1.654
    bz = wbs.matrix_world.translation.z
    wy = WBY - 0.03

    def to_world(px, py):
        return (WBX + (px / 640.0 - 0.5) * bw, wy, bz + (0.5 - py / 360.0) * bh)

    le = asm.wl("ENERGY / BIT", (0, 0, 0), 6.4, 10.0, size=0.12, color=(1.0, 0.45, 0.4))
    ll = asm.wl("LATENCY", (0, 0, 0), 6.4, 10.0, size=0.12, color=(0.5, 0.65, 1.0))
    t = 6.4
    while t <= 8.7:
        (ex, ey), (lx, ly) = T.graph_tip(GP(t))
        wex, wey, wez = to_world(ex, ey)
        wlx, wly, wlz = to_world(lx, ly)
        le.location = (wex, wey, wez + 0.17)
        le.keyframe_insert("location", frame=F(t))
        ll.location = (wlx, wly, wlz - 0.17)
        ll.keyframe_insert("location", frame=F(t))
        t += 0.1
    for o in (le, ll):
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    return scn


if __name__ == "__main__":
    build()
    asm.finalize(OUT)
