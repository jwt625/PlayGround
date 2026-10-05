"""Scene S2 "Retimers: inside the rack, row after row" (film 10-20 s, scene-local 0-10 s), v1.2.

Build:  FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s02_retimers.py -- scenes/v1/s02_retimers.blend
v1.2 (2026-10-03): one continuous world. The lab (bench, scope, whiteboard, Gary, Manager) stands in the S1 data-hall shell; the
stylised tray stack is the lab rack itself (uniform scale RACK_S next to the bench), so the camera rides up inside the rack that is
seen from outside and leaves it with a crane move (no hard cut). One baked camera path for the whole scene. v2 characters and
motion_v2 actions. Subtitles from scripts/audio/narration.json (asm.narr_vo). SFX-cued times kept (see DevLog-003-scene-s02.md).
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

sys.path.insert(0, os.path.join(asm.PROJ, "scripts", "assets", "characters", "motion_v2"))
import motion_v2 as M2  # noqa: E402

PROJ = asm.PROJ
TEXDIR = os.path.join(PROJ, "assets", "generated_textures", "v1", "s02")
OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(PROJ, "scenes", "v1", "s02_retimers.blend")
PI = math.pi
F = asm.F
GARY_ID, MGR_ID = "gary_v2", "manager_v2"

# ------------------------------------------------------------------ layout (1 unit = 1 m; z = 0 floor)
WALL_Y = 1.60                  # inner face of the hall's +y wall (the hall is shifted so this holds)
WBX, WB_Z0 = 0.70, 0.90        # whiteboard bottom-centre x, bottom edge height (board on the wall)
BX, BY = 3.50, -0.10           # bench centre
RACK_O = (5.25, -0.05, 0.0)    # stylised tray stack origin in the lab (front at -y)
RACK_S = 0.42                  # stack scale: 5.29 m -> 2.22 m tall, 2.3 m -> 0.97 m wide (see devlog)
M_POS = (0.20, 1.07, 0.0)      # Manager at the board (whiteboard_write: board 0.45 m x K ahead)
G0 = (2.72, -0.70, 0.0)        # Gary at the scope
G1 = (1.65, -0.30, 0.0)        # Gary after walking over to the board
NT = 8
GUN_ROT = (0.0, 0.0, PI)       # shotgun root onto HOOK_gun_grip_R (barrels along the fingers)
S_DETAIL = 4.0
CHIP_S = 7.0                   # retimer chips at 7x (105 mm) in rack-local units (v1.1: 9x; smaller so the column pairs separate)
COL_DX = 0.075                 # two retimer columns per connector at +-COL_DX (v1.1: 0.12; pairs now read per connector)
T_ARR = [0.8, 2.2, 2.6, 3.0, 3.4, 3.85, 4.35, 4.85]   # tray arrival times (chip-click SFX are cued on these)
CAM_H = 0.55                   # camera height above the tray board (rack-local)
LAYOUT_SEQ = ["gb300_compute", "gb300_compute", "nvlink_switch", "gb300_compute", "helios_compute", "nvlink_switch", "gb300_compute", "helios_compute"]
SHAKES = []                    # (t_hit, amplitude rad, decay 1/s, freq Hz)
T_SHOT = 9.2


def ramp(t, ta, tb, va, vb):
    if t <= ta:
        return va
    if t >= tb:
        return vb
    return va + (vb - va) * (t - ta) / (tb - ta)


def yaw_to(a, b):
    return asm._yaw(a, b)


def unwrap(prev, a):
    while a - prev > PI:
        a -= 2 * PI
    while a - prev < -PI:
        a += 2 * PI
    return a


def show(asset_or_coll, t0, t1):
    """Object-level visibility window for a whole asset/collection (Collection.hide_render is not animatable in 4.2)."""
    c = asset_or_coll.coll if isinstance(asset_or_coll, asm.Asset) else asset_or_coll
    for o in asm._all_objs(c):
        L.V(o, 0, t0, t1)


def fx(name, t, loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0, parent=None, intensity=1.0, dur=1.5, show_pad=0.05):
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


def smoother(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


def spring(tau, w=22.0, z=7.0):
    return 0.0 if tau < 0 else math.exp(-z * tau) * math.cos(w * tau)


def noise3(t, seed):
    return (math.sin(2 * PI * 0.37 * t + seed * 1.7) * 0.5 + math.sin(2 * PI * 0.91 * t + seed * 3.1) * 0.3
            + math.sin(2 * PI * 1.73 * t + seed * 5.3) * 0.2)


def W(p):
    """Rack-local (unscaled stack coordinates, stack origin at 0) -> world."""
    return Vector((RACK_O[0] + RACK_S * p[0], RACK_O[1] + RACK_S * p[1], RACK_O[2] + RACK_S * p[2]))


# ------------------------------------------------------------------ curves for the camera path
def herm(p0, v0, p1, v1, u, T):
    """Cubic Hermite between p0 and p1 (vectors) with end velocities v0, v1 (units per second) over duration T."""
    u2, u3 = u * u, u * u * u
    h00, h10, h01, h11 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2
    return p0 * h00 + v0 * (h10 * T) + p1 * h01 + v1 * (h11 * T)


def pchip_eval(knots, t, slow=0.3):
    """Monotone eased 1D curve through knots [(t, z)], slope at interior knots = slow * mean secant (dwell = slow, not stop)."""
    n = len(knots)
    if t <= knots[0][0]:
        return knots[0][1], 0.0
    if t >= knots[-1][0]:
        return knots[-1][1], 0.0
    sec = [(knots[i + 1][1] - knots[i][1]) / (knots[i + 1][0] - knots[i][0]) for i in range(n - 1)]
    m = [0.0] * n
    for i in range(1, n - 1):
        m[i] = slow * 0.5 * (sec[i - 1] + sec[i])
    m[-1] = slow * sec[-1]
    for i in range(n - 1):
        t0, z0 = knots[i]
        t1, z1 = knots[i + 1]
        if t0 <= t <= t1:
            T_ = t1 - t0
            u = (t - t0) / T_
            u2, u3 = u * u, u * u * u
            z = (2 * u3 - 3 * u2 + 1) * z0 + (u3 - 2 * u2 + u) * T_ * m[i] + (-2 * u3 + 3 * u2) * z1 + (u3 - u2) * T_ * m[i + 1]
            dz = ((6 * u2 - 6 * u) * z0 + (3 * u2 - 4 * u + 1) * T_ * m[i] + (-6 * u2 + 6 * u) * z1 + (3 * u2 - 2 * u) * T_ * m[i + 1]) / T_
            return z, dz
    return knots[-1][1], 0.0


class CamRig:
    """Baked camera: position, clean aim, lens, overlay scale per frame; TRACK_TO follows a noisy aim (handheld + shakes)."""

    def __init__(self):
        cam, tgt, hold = L.CAM, L.TGT, L.HOLD
        self.cam, self.tgt, self.hold = cam, tgt, hold
        self.tn = L.empty("CAM_TARGET_NOISY")
        for c in cam.constraints:
            if c.type == "TRACK_TO":
                c.target = self.tn
        camc = L.empty("CAM_CLEAN")
        c1 = camc.constraints.new("COPY_LOCATION")
        c1.target = cam
        c2 = camc.constraints.new("TRACK_TO")
        c2.target = tgt
        c2.track_axis = "TRACK_NEGATIVE_Z"
        c2.up_axis = "UP_Y"
        hold.parent = camc
        hold.matrix_parent_inverse.identity()

    def key(self, f, pos, aim, lens, hh=0.0025):
        t = (f - 1) / 30.0
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

    def finish(self):
        for o in (self.cam, self.tgt, self.tn, self.hold, self.cam.data):
            ad = o.animation_data
            if ad and ad.action:
                for fc in ad.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"


# ------------------------------------------------------------------ textures
def make_eye_sequence():
    d = os.path.join(TEXDIR, "eye_s2")
    params = []
    for f in range(1, 301):
        t = (f - 1) / 30.0
        params.append(dict(sigma_g=ramp(t, 5.4, 7.4, 0.45, 0.38), noise=ramp(t, 5.4, 7.4, 0.30, 0.05), jitter=ramp(t, 5.4, 7.4, 0.12, 0.03)))
    if os.path.exists(os.path.join(d, "eye_s2_0300.png")):
        print("EYE sequence exists, skipped")
    else:
        T.eye_sequence(d, "eye_s2", params, seed=22)
    return os.path.join(d, "eye_s2_0001.png"), 300, 1


GP = lambda t: ramp(t, 6.4, 8.6, 0.0, 1.0)  # noqa: E731  curve progress (curve_rise / marker_squeak SFX follow this, linear)


def graph_frame_thick(p, W_=640, H_=360):
    """Same axes and curve shapes as tex_gen.graph_frame (tex_gen._curves), strokes 2.7x thicker so they read at phone size."""
    import numpy as np
    img = np.full((H_, W_, 3), 246, dtype=np.uint8)
    L_, R_, T_, B_ = T.GRAPH_L, T.GRAPH_R, T.GRAPH_T, T.GRAPH_B
    img[T_:B_ + 1, L_ - 5:L_ + 1] = (20, 20, 20)
    img[B_:B_ + 6, L_ - 5:R_] = (20, 20, 20)
    xs, ys = T._curves(p)
    n = int(len(xs) * max(p, 0.0))
    for color, y in zip(((200, 30, 30), (30, 70, 200)), ys):
        for k in range(n):
            T._disk(img, int(L_ + (R_ - L_) * xs[k]), int(B_ - (B_ - T_) * y[k]), 8, color)
        if n > 0:
            T._disk(img, int(L_ + (R_ - L_) * xs[n - 1]), int(B_ - (B_ - T_) * y[n - 1]), 14, color)
    return img


def make_graph_sequence():
    d = os.path.join(TEXDIR, "graph_s2_v12")
    if os.path.exists(os.path.join(d, "graph_s2_v12_0300.png")):
        print("GRAPH sequence exists, skipped")
    else:
        os.makedirs(d, exist_ok=True)
        for f in range(1, 301):
            T.write_png(os.path.join(d, "graph_s2_v12_%04d.png" % f), graph_frame_thick(GP((f - 1) / 30.0)))
    return os.path.join(d, "graph_s2_v12_0001.png"), 300, 1


def plug_sequence(mat, fac_node, first_png, n, start):
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


# ------------------------------------------------------------------ retimer chips
def prep_glow_material(m):
    """Glow halo read from the quad's p_glow (0 off, 1 flash peak, 0.55 settled). v1.2: capped strength and a ring falloff so the
    halo surrounds the chip instead of a white blob (v1.1 peak strength 14 bloomed into glare)."""
    nt = m.node_tree
    if nt.animation_data:
        for d in list(nt.animation_data.drivers):
            nt.animation_data.drivers.remove(d)
    mix = next(n for n in nt.nodes if n.type == "MIX_SHADER")
    em = next(n for n in nt.nodes if n.type == "EMISSION")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = "p_glow"
    hot = nt.nodes.new("ShaderNodeMapRange")
    hot.inputs["From Min"].default_value = 0.55
    hot.inputs["From Max"].default_value = 1.0
    hot.clamp = True
    nt.links.new(at.outputs["Fac"], hot.inputs["Value"])
    cmix = nt.nodes.new("ShaderNodeMix")
    cmix.data_type = "RGBA"
    cmix.inputs["A"].default_value = (1.0, 0.42, 0.08, 1.0)
    cmix.inputs["B"].default_value = (1.0, 0.78, 0.45, 1.0)
    nt.links.new(hot.outputs["Result"], cmix.inputs["Factor"])
    nt.links.new(cmix.outputs["Result"], em.inputs["Color"])
    sm = nt.nodes.new("ShaderNodeMath")
    sm.operation = "MULTIPLY_ADD"
    sm.inputs[1].default_value = 1.6      # peak strength 1.1 + 1.6 = 2.7 (v1.1: 14)
    sm.inputs[2].default_value = 1.5      # settled strength (v1.1: 5)
    nt.links.new(hot.outputs["Result"], sm.inputs[0])
    nt.links.new(sm.outputs[0], em.inputs["Strength"])
    cl = nt.nodes.new("ShaderNodeMath")
    cl.operation = "MULTIPLY"
    cl.use_clamp = True
    cl.inputs[1].default_value = 1.0
    nt.links.new(at.outputs["Fac"], cl.inputs[0])
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
    fall2 = nt.nodes.new("ShaderNodeMath")
    fall2.operation = "MULTIPLY"
    fall2.use_clamp = True
    fall2.inputs[1].default_value = 0.85  # halo opacity cap
    nt.links.new(fall.outputs[0], fall2.inputs[0])
    for lk in list(mix.inputs[0].links):
        nt.links.remove(lk)
    nt.links.new(fall2.outputs[0], mix.inputs[0])
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
    print("CHIP materials", [sl.material.name for sl in body.material_slots if sl.material])
    for sl in body.material_slots:          # black mold -> dark slate so the chips separate from connectors and board
        m = sl.material
        if m and m.use_nodes and "mold" in m.name:
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    n.inputs["Base Color"].default_value = (0.11, 0.12, 0.15, 1)
    prep_glow_material(glow.data.materials[0])
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
    fc = g.driver_add('["p_glow"]')
    dv = fc.driver
    dv.type = "AVERAGE"
    var = dv.variables.new()
    var.name = "v"
    var.type = "SINGLE_PROP"
    var.targets[0].id = r
    var.targets[0].data_path = '["p_glow"]'
    return r, b, g


def emission_scale(mat_name_part, factor):
    """Scale emission strengths of every material whose name contains mat_name_part (lamp de-glare)."""
    for m in bpy.data.materials:
        if mat_name_part in m.name and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "EMISSION":
                    n.inputs["Strength"].default_value *= factor
                elif n.type == "BSDF_PRINCIPLED":
                    n.inputs["Emission Strength"].default_value *= factor


def area_light(name, loc, rot, size, size_y, energy, color):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = size, size_y
    ld.energy = energy
    ld.color = color
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.location, o.rotation_euler = loc, rot
    return o


def aim_rot(loc, target):
    d = Vector(target) - Vector(loc)
    return d.to_track_quat("-Z", "Y").to_euler()


# ------------------------------------------------------------------ main build
def build():
    scn = asm.new_scene(preset="standard", world="WORLD_data_hall")
    scn.render.use_motion_blur = True       # kept by render_scene.py through the presets
    scn.render.motion_blur_shutter = 0.5
    cam_rig = CamRig()
    eye_png, eye_n, eye_start = make_eye_sequence()
    gr_png, gr_n, gr_start = make_graph_sequence()

    # ---------------- data-hall shell (same set as S1): racks, containment, overhead hidden
    hall = asm.append("datacenter/datahall_environment")
    for ch in hall.coll.children:
        if ch.name.split(".")[0] in ("RACKS", "CONTAINMENT", "OVERHEAD"):
            for o in asm._all_objs(ch):
                o.hide_render = True
                o.hide_viewport = True
    walls = next(o for o in hall.objs if o.type == "MESH" and any(c.name.startswith("WALLS") for c in o.users_collection))
    ys = sorted({round(v.co.y, 3) for v in walls.data.vertices if v.co.y > 2.0})
    inner = ys[0]
    asm.place(hall.root, (1.5, WALL_Y - inner, 0.0))
    print("HALL wall inner y (asset) %.3f -> root y %.3f" % (inner, WALL_Y - inner))
    # perforated tiles / doors: dithered alpha (speckle at 8 spp) -> plain floor tile
    tile = bpy.data.materials.get("MAT_datacenter_floor_tile")
    for o in hall.objs:
        if o.type == "MESH":
            for sl in o.material_slots:
                if sl.material and "perforated" in sl.material.name and tile:
                    sl.material = tile
    emission_scale("led_strip", 0.35)      # ceiling LED strips: tame the blown-out lamps
    emission_scale("sign_", 0.6)
    emission_scale("led_green", 0.3)     # rack status LEDs bloomed into green glare in the crane shot

    # ---------------- lighting: warm key, cool fill, overhead soft, rack front light (static: no moving headlamp)
    area_light("L_key", (-1.2, -4.0, 3.2), aim_rot((-1.2, -4.0, 3.2), (1.8, 0.2, 1.0)), 4.0, 3.0, 110.0, (1.0, 0.9, 0.78))
    area_light("L_fill", (6.0, -3.5, 2.4), aim_rot((6.0, -3.5, 2.4), (2.5, 0.3, 1.0)), 3.0, 2.0, 45.0, (0.82, 0.9, 1.0))
    area_light("L_top", (2.2, -0.6, 3.45), (0, 0, 0), 7.0, 3.5, 90.0, (1.0, 0.96, 0.9))
    rack_front = Vector((RACK_O[0], RACK_O[1] - 1.9, 1.15))
    area_light("L_rack", tuple(rack_front), aim_rot(rack_front, (RACK_O[0], RACK_O[1], 1.0)), 1.6, 2.6, 30.0, (1.0, 0.95, 0.88))

    # ---------------- lab: bench + scope + whiteboard
    bench = asm.append("lab_office/lab_bench")
    asm.place(bench.root, (BX, BY, 0.0), yaw=0.0)
    scope = asm.append("lab_office/bench_oscilloscope")
    bpy.context.view_layer.update()
    slot = bench.hook("scope_slot").matrix_world.translation.copy()
    asm.place(scope.root, (slot.x, slot.y, slot.z), yaw=0.0)
    wb = asm.append("lab_office/whiteboard_big")
    asm.place(wb.root, (WBX, WALL_Y - 0.05, WB_Z0), yaw=0.0)
    bpy.context.view_layer.update()

    def scope_hook(nm):
        return next(o for o in scope.objs if o.name.startswith("HOOK_" + nm)).matrix_world.translation.copy()

    scr = next(o for o in scope.objs if o.name == "bench_oscilloscope_screen")
    plug_sequence(scr.material_slots[0].material, "SCREEN_FAC", eye_png, eye_n, eye_start)
    wbs = next(o for o in wb.objs if o.name == "whiteboard_big_screen")
    plug_sequence(wbs.material_slots[0].material, "BOARD_FAC", gr_png, gr_n, gr_start)
    bpy.context.view_layer.update()
    scr_c = sum((scr.matrix_world @ Vector(c) for c in scr.bound_box), Vector()) / 8.0
    wbs_c = sum((wbs.matrix_world @ Vector(c) for c in wbs.bound_box), Vector()) / 8.0
    wb_front = min((wbs.matrix_world @ Vector(c)).y for c in wbs.bound_box)
    print("SCOPE screen centre", tuple(round(x, 3) for x in scr_c), "BOARD centre", tuple(round(x, 3) for x in wbs_c), "front y %.3f" % wb_front)

    # ---------------- characters (v2) and motion (motion_v2)
    gary = asm.append("characters/" + GARY_ID, actions=True)
    mgr = asm.append("characters/" + MGR_ID, actions=True)
    m_pos = (M_POS[0], wb_front - 0.48, 0.0)
    g_yaw0 = yaw_to(G0, (scr_c.x, scr_c.y))
    g_yaw_walk = unwrap(g_yaw0, yaw_to(G0, G1))
    g_yaw_board = unwrap(g_yaw_walk, yaw_to(G1, (G1[0] - 0.15, WALL_Y)))
    g_yaw_cam = unwrap(g_yaw_board, yaw_to(G1, (G1[0] - 0.55, G1[1] - 1.0)))   # faces the camera, a little toward the Manager
    asm.place(gary.root, G0, yaw=g_yaw0)
    asm.place(mgr.root, m_pos, yaw=0.0)

    # Gary: tense idle at the scope, walks over to the board 5.9-7.25, looks at the curves, startles at the scream, shot at 9.2
    T_WALK0 = 5.9
    M2.apply(gary, "idle_breathe_tense", -0.5, hold=False, repeat=(T_WALK0 + 0.5) * 30 / 72 + 0.1)   # starts before frame 1 (motion blur)
    asm.key_loc(gary.root, 0.0, G0, yaw=g_yaw0)
    asm.key_loc(gary.root, T_WALK0 - 0.25, G0, yaw=g_yaw0, interp="BEZIER")
    asm.key_loc(gary.root, T_WALK0, G0, yaw=g_yaw_walk)
    t_arr = M2.walk(gary, T_WALK0, [G0, G1])
    # M2.walk re-keys the yaw at the path points: re-key the turns around it
    asm.key_loc(gary.root, t_arr, G1, yaw=g_yaw_walk, interp="BEZIER")
    asm.key_loc(gary.root, t_arr + 0.3, G1, yaw=g_yaw_board)
    M2.apply(gary, "idle_breathe_tense", t_arr, hold=False, repeat=2.0, blend_in=4)
    T_STARTLE = 8.22
    asm.key_loc(gary.root, T_STARTLE, G1, yaw=g_yaw_board, interp="BEZIER")
    asm.key_loc(gary.root, T_STARTLE + 0.35, G1, yaw=g_yaw_cam)
    M2.apply(gary, "startle", T_STARTLE, hold=True, blend_in=3)
    M2.apply(gary, "idle_breathe_tense", T_STARTLE + 36 / 30.0, hold=False, repeat=1.0, blend_in=5)
    M2.apply(gary, "shot_hit_fall", T_SHOT, speed=1.25, hold=True)
    asm.key_loc(gary.root, 10.0, G1, yaw=g_yaw_cam)
    # Gary face (root keys win over the NLA face strips for these channels)
    asm.key_prop(gary.root, "p_expr_worried", 0.0, 0.9)
    asm.key_prop(gary.root, "p_expr_worried", 6.6, 0.9)
    asm.key_prop(gary.root, "p_expr_worried", 7.2, 0.0)
    asm.key_prop(gary.root, "p_expr_happy", 6.6, 0.0)
    asm.key_prop(gary.root, "p_expr_happy", 7.2, 0.7)
    asm.key_prop(gary.root, "p_expr_happy", 8.1, 0.7)
    asm.key_prop(gary.root, "p_expr_happy", 8.3, 0.0)
    asm.key_prop(gary.root, "p_expr_scared", 8.2, 0.0)
    asm.key_prop(gary.root, "p_expr_scared", 8.4, 1.0)
    asm.key_prop(gary.root, "p_expr_scared", T_SHOT, 1.0)
    asm.key_prop(gary.root, "p_expr_scared", T_SHOT + 0.1, 0.0)
    # holes: hole 1 from S1 (film T 9.2), stepped x0.9 every 1.5 s (floor 0.3); hole 2 at the bang, small pop at 9.45 (hole_pop SFX)
    T_SHOT1 = 9.2
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
    asm.key_prop(gary.root, "p_hole_2_radius", T_SHOT, 1.3)      # cartoon-large at the hit so it reads at phone size
    asm.key_prop(gary.root, "p_hole_2_radius", 9.4, 1.4)
    asm.key_prop(gary.root, "p_hole_2_radius", 9.47, 1.75)    # hole_pop SFX 19.45
    asm.key_prop(gary.root, "p_hole_2_radius", 9.6, 1.5)

    # Manager: thinking at the board, writes 6.4-7.55 (curves draw), turns to camera and boils over (steam 8.1, red pop 8.2),
    # turns to Gary while raising the shotgun, fires at 9.2 (gun_raise_aim_fire shot frame 44)
    M2.apply(mgr, "thinking", -0.5, hold=False, repeat=6.95 * 30 / 60, face=False)
    M2.apply(mgr, "whiteboard_write", 6.4, hold=False, repeat=1.2 * 30 / 48, blend_in=5, face=False)
    # anger beat: tense idle (straight legs; anger_outburst's knee bend read as a squat from the front) + shouting upper body
    M2.apply(mgr, "idle_breathe_tense", 7.55, hold=True, blend_in=5, face=False)
    M2.apply(mgr, "shout_rant_upper", 7.95, hold=False, repeat=0.45 * 30 / 36 + 0.2, blend_in=3, blend_out=3, face=False)
    GSPD = 1.4
    t_gun = T_SHOT - (44 - 6) / (30.0 * GSPD)
    M2.apply(mgr, "gun_raise_aim_fire", t_gun, speed=GSPD, start_frame=6, hold=True, blend_in=3, face=False)
    myaw_b = 0.0                                                     # facing the board (+y)
    myaw_c = unwrap(myaw_b, yaw_to(m_pos, (0.9, -3.5)))             # toward the camera
    myaw_g = unwrap(myaw_c, yaw_to(m_pos, G1))                        # toward Gary
    back = Vector((m_pos[0] - G1[0], m_pos[1] - G1[1], 0.0)).normalized()

    def mgr_state(tt):
        if tt < 7.55:
            yw = myaw_b
        elif tt < 8.05:
            yw = myaw_b + (myaw_c - myaw_b) * smoother((tt - 7.55) / 0.5)
        elif tt < t_gun + 0.05:
            yw = myaw_c
        else:
            yw = myaw_c + (myaw_g - myaw_c) * smoother((tt - t_gun - 0.05) / 0.45)
        pos = Vector(m_pos)
        if 8.1 < tt < 8.5:
            a_ = 0.010 * math.sin(2 * PI * 13.0 * tt) * math.sin(PI * (tt - 8.1) / 0.4)
            pos += Vector((a_, 0.6 * a_, 0))
        if tt >= T_SHOT:
            tau = tt - T_SHOT
            pos += back * 0.09 * math.exp(-6.0 * tau) * math.sin(16.0 * tau)
        return tuple(pos), yw
    for f in range(1, F(10.0) + 1, 1):
        tt = (f - 1) / 30.0
        if tt < 7.4 and f > 1:
            continue
        pos, yw = mgr_state(tt)
        asm.key_loc(mgr.root, tt, pos, yaw=yw)
    asm.key_prop(mgr.root, "p_expr_worried", 0.0, 0.3)
    asm.key_prop(mgr.root, "p_expr_worried", 7.5, 0.0)
    asm.key_prop(mgr.root, "p_anger", 7.6, 0.0)
    asm.key_prop(mgr.root, "p_anger", 8.2, 1.0)
    asm.key_prop(mgr.root, "p_flush", 7.95, 0.0)
    asm.key_prop(mgr.root, "p_flush", 8.2, 1.0)
    asm.key_prop(mgr.root, "p_expr_yell", 7.95, 0.0)
    asm.key_prop(mgr.root, "p_expr_yell", 8.1, 1.0)
    asm.key_prop(mgr.root, "p_expr_yell", 8.6, 0.25)
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
    asm.attach(gun.root, ghook, rot=GUN_ROT, offset=(0.0, 0.0, 0.0))
    show(gun, t_gun, 10.0)
    bpy.context.view_layer.update()
    for side in ("L", "R"):
        mh = next(o for o in gun.objs if o.name.startswith("HOOK_muzzle_" + side))
        fx("muzzle_flash", T_SHOT, rot=(-PI / 2, 0, 0), scale=0.8, parent=mh, dur=0.12, intensity=0.7)
    mh = next(o for o in gun.objs if o.name.startswith("HOOK_muzzle_R"))
    fx("smoke_ring", T_SHOT + 0.05, rot=(-PI / 2, 0, 0), scale=0.55, parent=mh, dur=0.7)
    # hit FX beside the hole (not on it, so the new hole reads), on Gary's left toward the Manager
    hole2 = next(o for o in gary.objs if o.name.startswith("HOLE_hole_2"))
    dv = Vector((m_pos[0] - G1[0], m_pos[1] - G1[1], 0.0)).normalized()
    hitp = (G1[0] + dv.x * 0.30, G1[1] + dv.y * 0.30, 1.05)
    fx("dust_cloud", T_SHOT, loc=hitp, scale=0.25, dur=0.8, intensity=0.6)
    fx("impact_stars", T_SHOT, loc=(hitp[0], hitp[1], 1.35), scale=0.25, dur=0.35, intensity=0.6)

    # ---------------- stylised rack (tray stack) built in rack-local units, then scaled into the lab next to the bench
    grp = bpy.data.objects.new("S2_RACK_GRP", None)
    scn.collection.objects.link(grp)
    rack_top = []
    rack = asm.append("datacenter/rack_interior_tray_stack")
    asm.place(rack.root, (0.0, 0.0, 0.0))
    rack_top.append(rack.root)
    bpy.context.view_layer.update()
    for o in [o for o in rack.objs if o.name.endswith("_silk")]:
        bpy.data.objects.remove(o, do_unlink=True)
    rack.objs = asm._all_objs(rack.coll)
    # pans: mid-grey instead of white (their front lips sweep through the frame during the rise)
    for o in rack.objs:
        if o.type == "MESH" and o.name.endswith("_pan"):
            for sl in o.material_slots:
                if sl.material:
                    m2 = sl.material.copy()
                    m2.name = "S2_pan_" + sl.material.name
                    for n in m2.node_tree.nodes:
                        if n.type == "BSDF_PRINCIPLED":
                            c = n.inputs["Base Color"].default_value
                            n.inputs["Base Color"].default_value = (c[0] * 0.45, c[1] * 0.45, c[2] * 0.47, 1)
                    sl.link = "OBJECT"
                    sl.material = m2
    tmpl, body, glow = build_chip_template()
    tmpl.coll.hide_render = True
    tmpl.coll.hide_viewport = True
    chips = bpy.data.collections.new("S2_retimer_chips")
    scn.collection.children.link(chips)
    hooks = {o.name: o for o in rack.objs if o.name.startswith("HOOK_")}
    tray_z = [hooks["HOOK_tray%d_surface" % j].matrix_world.translation.z for j in range(NT)]
    Yb = 0.8

    dress = bpy.data.collections.new("S2_rack_dressing")
    scn.collection.children.link(dress)
    lay = {}
    for nm in ("gb300_compute", "nvlink_switch", "helios_compute"):
        a_ = asm.append("datacenter/tray_dummy_layouts", only=["ASSET_tray_dummy_" + nm])
        lay[nm] = next(o for o in a_.objs if o.type == "MESH")
        lay[nm].parent = None
        a_.root.hide_render = True
        a_.root.hide_viewport = True
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
    for j in range(NT):
        o = lay[LAYOUT_SEQ[j]].copy()
        o.name = "dress_%s_t%d" % (LAYOUT_SEQ[j], j)
        dress.objects.link(o)
        o.location = (0.0, 0.0, tray_z[j])
        o.scale = (S_DETAIL,) * 3
        bn = band.copy()
        bn.name = "cartridge_band_t%d" % j
        dress.objects.link(bn)
        bn.location = (0.0, Yb, tray_z[j] - 0.0304 + 0.19)
        bn.scale = (S_DETAIL,) * 3
        rack_top += [o, bn]
    for nm in lay:
        dress.objects.unlink(lay[nm])
    dress.objects.unlink(band)

    # chips: rows appear on the v1.1 schedule (chip_click SFX), slide in from front/above with a scale-in, springy landing
    slide = Vector((0.0, -0.35, 0.35))
    sparks_t = []
    chip_info = {}
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
                    rack_top.append(root)
                    g.location = (0, 0, 0.0007)
                    final = Vector((p.x, p.y, p.z))
                    tl = tk + 0.22
                    for f in range(F(tk), F(tk + 0.7) + 1):
                        tt = (f - 1) / 30.0
                        if tt < tl:
                            u = (tt - tk) / 0.22
                            e = smoother(u)
                            off = slide * (1 - e)
                            sc = 0.55 + 0.45 * e
                            sq = 0.0
                        else:
                            tau = tt - tl
                            off = Vector((0, -0.012 * math.exp(-8 * tau) * math.sin(20 * tau), 0.0))
                            sc = 1.0
                            sq = spring(tau, 24.0, 8.0)
                        root.location = tuple(final + off)
                        root.keyframe_insert("location", frame=f)
                        root.scale = (CHIP_S * sc * (1 + 0.07 * sq), CHIP_S * sc * (1 + 0.07 * sq), CHIP_S * sc * (1 - 0.14 * sq))
                        root.keyframe_insert("scale", frame=f)
                    for o in (b, g):
                        L.V(o, 0, tk, 10.0)
                    gk = [(tk, 0.0), (tk + 0.2, 1.0), (tk + 0.9, 0.55)]
                    g.scale = (1.0, 1.0, 1.0)
                    g.keyframe_insert("scale", frame=F(tk))
                    g.scale = (1.8, 1.8, 1.0)
                    g.keyframe_insert("scale", frame=F(tk + 0.2))
                    g.scale = (1.6, 1.6, 1.0)
                    g.keyframe_insert("scale", frame=F(tk + 0.9))
                    chip_info[(j, r, kk, col)] = (root, gk, final.x, final.y, tk)
    print("CHIPS", len(chip_info))

    # signal beads: run connector -> rows; each chip glows a little as a bead passes (dimmed in v1.2)
    import bmesh
    bead_me = bpy.data.meshes.new("bead")
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
    em.inputs["Color"].default_value = (1.0, 0.8, 0.45, 1.0)
    em.inputs["Strength"].default_value = 3.0
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
                while t0 + dur <= t_end and t0 < 5.8:
                    for tt, yy, s in ((t0, y0, 0.0), (t0 + 0.05, y0 - SPD * 0.05, 1.0), (t0 + dur - 0.05, y_last + SPD * 0.05, 1.0), (t0 + dur, y_last, 0.0)):
                        bd.location = (x, yy, z0)
                        bd.scale = (0.06 * s, 0.16 * s, 0.06 * s)
                        bd.keyframe_insert("location", frame=F(tt))
                        bd.keyframe_insert("scale", frame=F(tt))
                    for r in range(rows):
                        tp = t0 + (y0 - chip_info[(j, r, kk, col)][3]) / SPD
                        bump.setdefault((j, r, kk, col), []).append(tp)
                    t0 += 0.7
                if bd.animation_data is None:
                    bpy.data.objects.remove(bd, do_unlink=True)
                else:
                    L.V(bd, 0, 0.9, 5.9)
                    rack_top.append(bd)
    for key, (root, gk, cx, cy, tk) in chip_info.items():
        ks = list(gk)
        for tp in bump.get(key, []):
            if tp > tk + 1.0:
                ks += [(tp - 0.08, 0.55), (tp, 0.72), (tp + 0.2, 0.55)]
        ks.sort()
        last_f = None
        for tt, v in ks:
            f = F(tt)
            if last_f is not None and f <= last_f:
                continue
            asm.key_prop(root, "p_glow", tt, v)
            last_f = f

    # landing sparks: one small burst per tray at its first row
    for (j, r, tr_) in sparks_t:
        if r == 0:
            xs_ = hooks["HOOK_retimer_row0_tray%d_0" % j].matrix_world.translation
            a_ = fx("sparks_burst", tr_ + 0.25, loc=(0.0, xs_.y, xs_.z + 0.05), scale=0.3, dur=0.5, intensity=0.35)
            rack_top.append(a_.root)

    # NARROWCOM sign on the backplane (chip valley mark + wordmark meshes scaled up, parody mark), tray-0 gap
    sign_src = asm.append("packaging/narrowcom_retimer_chip")
    sign_src.variant("mounted", False)
    sign_src.variant("heat_spreader", False)
    sc_ = bpy.data.collections.new("S2_sign")
    scn.collection.children.link(sc_)
    vm = next(o for o in sign_src.objs if "_valley_mark" in o.name)
    wm = next(o for o in sign_src.objs if "_wordmark" in o.name)
    for o in list(sign_src.objs):
        if o not in (vm, wm) and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    for c in list(sign_src.coll.children):
        bpy.data.collections.remove(c)
    for o in (vm, wm):
        o.parent = None
    link_to(sc_, [vm, wm])
    bpy.data.collections.remove(sign_src.coll)
    sign_y, sign_z, sg = Yb - 0.10, tray_z[0] + 0.40, 70.0
    vm.scale = wm.scale = (sg, sg, sg)
    vm.rotation_euler = wm.rotation_euler = (PI / 2, 0, 0)
    vm.location = (0.0, sign_y, sign_z + 0.05)
    wm.location = (0.0, sign_y, sign_z - 0.08)
    sm = vm.data.materials[0].copy()
    sm.name = "S2_sign_gold"
    for n in sm.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Emission Color"].default_value = (0.95, 0.8, 0.35, 1.0)
            n.inputs["Emission Strength"].default_value = 0.9
    for o in (vm, wm):
        o.visible_shadow = False
        o.data.materials[0] = sm
        rack_top.append(o)
    show(sc_, 0.0, 3.0)

    # rack interior fill: one soft light per tray slot under the pan above (static, low, warm): no moving lamp, no hot spots
    for j in range(NT):
        lo = area_light("L_slot_%d" % j, (0.0, 0.50, tray_z[j] + 0.60), (0, 0, 0), 1.9, 0.45, 9.0, (1.0, 0.93, 0.82))
        lo.data.energy = 6.0   # scaled below with the stack (power ~ scale^2); over the retimer rows, not the dressing
        rack_top.append(lo)

    # parent the whole stack to the group, then scale it into the lab
    for o in rack_top:
        if o.parent is None:
            o.parent = grp
            o.matrix_parent_inverse.identity()
    grp.location = RACK_O
    grp.scale = (RACK_S,) * 3
    for o in rack_top:
        if o.type == "LIGHT":
            o.data.energy *= RACK_S ** 2 * 0.8
    bpy.context.view_layer.update()

    # ---------------- cables rack -> scope (tray fronts of trays 1-3)
    ports = [W((-0.98, -0.70, tray_z[1 + ci] + 0.09)) for ci in range(3)]   # left front corner: out of the rising camera's view
    scope_pts = [scope_hook("ch1_bnc"), scope_hook("probe_tip"), scope_hook("trigger_level_knob")]
    cable_mat = bpy.data.materials.new("S2_cable_jacket")
    cable_mat.use_nodes = True
    cable_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.035, 0.05, 1)
    cable_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
    NPT = 22

    def cable_pts(a_, b_, sag, ci):
        out = []
        for i in range(NPT):
            s = i / (NPT - 1)
            x = a_.x + (b_.x - a_.x) * s
            y = a_.y + (b_.y - a_.y) * s - 0.30 * math.sin(PI * s) - 0.10 * ci * math.sin(PI * s)
            z = a_.z + (b_.z - a_.z) * s - sag * 4 * s * (1 - s)
            z = max(z, 0.02)
            out.append((x, y, z, 1.0))
        return out
    for ci in range(3):
        cu = bpy.data.curves.new("cable%d" % ci, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.0075
        cu.bevel_resolution = 2
        sp_ = cu.splines.new("POLY")
        sp_.points.add(NPT - 1)
        cu.materials.append(cable_mat)
        co = bpy.data.objects.new("cable_rack_scope_%d" % ci, cu)
        scn.collection.objects.link(co)
        for i, q in enumerate(cable_pts(ports[ci], scope_pts[ci], 0.55 + 0.12 * ci, ci)):
            sp_.points[i].co = q

    # ---------------- camera: one baked path (C1-continuous Hermite segments)
    SHAKES.append((T_SHOT, 0.020, 7.0, 17.0))
    SHAKES.append((T_SHOT + 0.1, 0.008, 6.0, 11.0))
    V3 = lambda *a: Vector(a)  # noqa: E731

    def rack_pose(t):
        """In-rack camera (rack-local -> world): continuous eased rise, slow at each tray arrival, slight lateral drift."""
        knots = [(1.15, tray_z[0] + CAM_H), (1.70, tray_z[0] + CAM_H + 0.06)] + [(T_ARR[j], tray_z[j] + CAM_H) for j in range(1, NT)] + [(5.4, tray_z[7] + CAM_H + 0.30)]
        z, dz = pchip_eval(knots, t, slow=0.15)
        if t >= 5.4:
            dz = 0.30 / 0.55 * 0.15
        u = (t - 1.15) / 4.25
        x = 0.25 * math.sin(1.4 * PI * u)
        y = -1.20 + 0.15 * smoother((t - 1.15) / 1.2)   # stay 0.4-0.5 m in front of the pan lips (they sweep by as thin bands)
        pos = W((x, y, z))
        aim = W((0.05 * x, 0.45, z - 0.75))
        vel = V3(0, 0, dz * RACK_S)
        return pos, aim, vel

    A0 = V3(scr_c.x + 0.12, scr_c.y - 1.62, scr_c.z + 0.50)    # over Gary's right shoulder, scope screen centre-right
    A1 = V3(scr_c.x + 0.18, scr_c.y - 1.80, scr_c.z + 0.54)
    A_aim0 = V3(scr_c.x - 0.20, scr_c.y, scr_c.z - 0.06)
    A_aim1 = V3(scr_c.x - 0.22, scr_c.y, scr_c.z - 0.08)
    C_pos, C_aim = V3(2.9, -4.1, 2.05), V3(3.0, 0.2, 0.95)      # crane end: bench, rack, cables, scope, Gary; board at left
    D_pos, D_aim = V3(0.85, -2.6, 1.6), V3(0.75, 1.0, 1.45)     # whiteboard push-in end: board large, Manager left, Gary right
    E_pos, E_aim = V3(0.45, -2.45, 1.30), V3(0.95, 0.40, 1.02)   # two-shot

    def cam_at(t):
        if t < 0.55:
            u = t / 0.55
            return A0.lerp(A1, u), A_aim0.lerp(A_aim1, u), 38.0, 0.0012
        if t < 1.15:
            # swoop along the cables to the rack: via a waypoint in front of the bench (keeps clear of the bench corner)
            p1, a1, v1 = rack_pose(1.15)
            vA = (A1 - A0) / 0.55
            Pm = V3(RACK_O[0] - 0.55, -1.75, 0.75)
            vm = (p1 - A1) / 0.6 * 1.2
            am = V3(RACK_O[0] - 0.2, RACK_O[1], 0.5)
            vam = (a1 - A_aim1) / 0.6
            lens = 38.0
            if t < 0.85:
                u = (t - 0.55) / 0.3
                return herm(A1, vA, Pm, vm, u, 0.3), herm(A_aim1, (A_aim1 - A_aim0) / 0.55, am, vam, u, 0.3), lens, 0.002
            u = (t - 0.85) / 0.3
            return herm(Pm, vm, p1, v1, u, 0.3), herm(am, vam, a1, V3(0, 0, 0), u, 0.3), lens, 0.002
        if t < 5.4:
            p, a, _ = rack_pose(t)
            return p, a, 38.0, 0.002
        if t < 6.4:
            # crane: up out of the open rack top first (waypoint above the stack), then pull back and down to the bench wide
            p0, a0, v0 = rack_pose(5.4)
            vC = (D_pos - C_pos) / 2.2 * 0.6
            vaC = (D_aim - C_aim) / 2.2 * 0.6
            Pc = W((0.0, -1.3, tray_z[7] + CAM_H + 1.1))
            Ac = W((0.0, 0.2, tray_z[7] - 0.4))
            vPc = (C_pos - p0) / 1.0 * 0.9
            vAc = (C_aim - a0) / 1.0 * 0.9
            lens = 38.0 + (26.0 - 38.0) * smoother((t - 5.4) / 1.0)
            if t < 5.8:
                u = (t - 5.4) / 0.4
                return herm(p0, v0, Pc, vPc, u, 0.4), herm(a0, V3(0, 0, 0), Ac, vAc, u, 0.4), lens, 0.002
            u = (t - 5.8) / 0.6
            return herm(Pc, vPc, C_pos, vC, u, 0.6), herm(Ac, vAc, C_aim, vaC, u, 0.6), lens, 0.002
        if t < 8.6:
            u = (t - 6.4) / 2.2
            vC = (D_pos - C_pos) / 2.2 * 0.6
            return herm(C_pos, vC, D_pos, V3(0, 0, 0), u, 2.2), herm(C_aim, (D_aim - C_aim) / 2.2 * 0.6, D_aim, V3(0, 0, 0), u, 2.2), 26.0 + (30.0 - 26.0) * smoother(u), 0.0025
        if t < 9.1:
            e = smoother((t - 8.6) / 0.5)
            return D_pos.lerp(E_pos, e), D_aim.lerp(E_aim, e), 30.0 + (28.0 - 30.0) * e, 0.0025
        u = (t - 9.1) / 0.9
        return E_pos + V3(0.0, 0.08, 0.0) * u, E_aim, 28.0, 0.002

    for f in range(1, 301):
        p, a, lens, hh = cam_at((f - 1) / 30.0)
        cam_rig.key(f, tuple(p), tuple(a), lens, hh)
    cam_rig.finish()

    # ---------------- captions (narration.json), labels, card, FX notes (HUD-gated)
    asm.narr_vo(2)
    asm.lab(0.0, 0.8, "EYE CLOSING")
    asm.lab(1.2, 2.0, "RETIMER ROW ADDED BEHIND EACH CONNECTOR")
    asm.lab(2.0, 5.4, "TRAY AFTER TRAY: MORE ROWS")
    asm.lab(5.6, 6.4, "EYE RECOVERS   BER FALLS")
    asm.card(6.4, 8.6, "Passive Cu 0 pJ/b (Arista) -> retimed ~15.6 pJ/b = 25 W / 1.6T (Ciena) [derived]. "
                       "Curves: no units. Eye and BER: illustrative model.")
    asm.fxn(0.8, 1.6, "[FX: glow flash + sparks as each retimer slides in]")
    asm.fxn(1.6, 5.4, "[FX: particles + glow trails as rows slide in tray by tray]")
    asm.fxn(6.4, 8.6, "[FX: marker squeak as the curves draw]")
    asm.fxn(9.2, 9.8, "[FX: muzzle flash, smoke ring, hole decal]")
    asm.big(T_SHOT, T_SHOT + 0.23, "BANG")   # short: the caption sits between the two and must clear the hole and aim quickly
    asm.timecode(2)

    bw, bh = 2.94, 1.654
    bz = wbs_c.z
    wy = wb_front - 0.25        # billboard labels turn toward the camera: keep them clear of the board plane

    def to_world(px, py):
        return (WBX + (px / 640.0 - 0.5) * bw, wy, bz + (0.5 - py / 360.0) * bh)

    le = asm.wl("ENERGY / BIT", (0, 0, 0), 6.4, 10.0, size=0.16, color=(1.0, 0.35, 0.3))
    ll = asm.wl("LATENCY", (0, 0, 0), 6.4, 10.0, size=0.16, color=(0.35, 0.5, 1.0))
    t = 6.4
    while t <= 8.7:
        (ex, ey), (lx, ly) = T.graph_tip(GP(t))
        wex, wey, wez = to_world(ex, ey)
        wlx, wly, wlz = to_world(lx, ly)
        le.location = (min(wex, WBX + 0.95), wey, wez + 0.17)
        le.keyframe_insert("location", frame=F(t))
        ll.location = (min(wlx, WBX + 1.05), wly, wlz - 0.20)
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
