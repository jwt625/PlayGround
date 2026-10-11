"""OUTNOLIGHT HATCHERY assembly line (workstream E, DevLog-013): conveyor, 5 SCARA pick-and-place robots, inspection
camera, stations adding chassis (legs/feet), PCB, DSP, optics, shell + feathers; eyes light up at inspection.
Stage meshes come from the hero GLB (workstream H) split by part-name prefix and decimated, so the line assembles the
same bird as the film. Seamless loop of LOOP frames (default 60 = 2 s at 30 fps) via F-curve Cycles modifiers.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/env/build_hatchery.py -- [out.blend] [loop_frames] [frame_start]
Local frame (root empty `hatchery_root`): conveyor along +X, belt top z = 0.85, robots on +Y, open front (camera) on -Y.
"""
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_lib as E  # noqa: E402
import street_kit as SK  # noqa: E402
from env_lib import MB, PI  # noqa: E402

import vfx  # noqa: E402

HERO_GLB = os.path.join(E.ROOT, "assets", "hero", "pigeon_osfp_dr8.glb")
S_BIRD = 2.6                     # hall scale, same as build_crude_film.py
BELT_Z = 0.85
PITCH = 0.75
X0 = -2.25                       # position k at x = X0 + k * PITCH; k = -1 (in tunnel) .. 7 (out tunnel)
PALLET_H = 0.04
ROBOT_Y = 0.55
L1, L2 = 0.32, 0.3
ARM_Z = 1.3
STAGES = ["chassis", "pcb", "dsp", "optics", "shell"]
STATION_LABEL = ["CHASSIS", "PCB", "DSP", "OPTICS", "SHELL + FEATHERS", "EYES ON / QA"]
PREFIX = {
    "chassis": ("leg_", "foot_"),
    "pcb": ("pcb_", "elec_", "flex_", "silk_", "conn_", "mark_"),
    "dsp": ("dsp_", "thermal_"),
    "optics": ("tx_", "rx_", "clip_"),
    "shell": ("skin_", "wing_", "tail_fan", "beak_", "leash_", "eye_tx_bezel", "eye_rx_bezel", "eye_tx_iris_ring", "eye_rx_iris_ring"),
    "eyes_tx": ("tx_eye_lenslets_out_2x4", "eye_tx_iris_glow_ring"),
    "eyes_rx": ("rx_lenslet_array_2x4", "eye_rx_iris_glow_ring"),
}
DROP = ("tx_beam_", "rx_viz_", "tx_dfb_facet_glow", "skin_panel_seams", "skin_screws")
TARGET_FACES = {"chassis": 4000, "pcb": 9000, "dsp": 2500, "optics": 9000, "shell": 14000, "eyes_tx": 1500, "eyes_rx": 1500}


def stage_of(name):
    if name.startswith(DROP):
        return None
    for st in ("eyes_tx", "eyes_rx"):
        if name.startswith(PREFIX[st]):
            return st
    for st in STAGES:
        if name.startswith(PREFIX[st]):
            return st
    return "pcb"


def stage_meshes():
    """Import the hero GLB, split into stage meshes in the bird frame (beak +X, feet origin, metres at S_BIRD)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=HERO_GLB)
    new = [o for o in bpy.data.objects if o not in before]
    root = next(o for o in new if o.name.startswith("HERO_ROOT"))
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0, 0, PI / 2)
    root.scale = (S_BIRD,) * 3
    bpy.context.view_layer.update()
    groups = {}
    for o in new:
        if o.type != "MESH":
            continue
        st = stage_of(o.name.split(".")[0])
        if st:
            groups.setdefault(st, []).append(o)
    meshes = {}
    for st, objs in groups.items():
        parts = []
        for o in objs:
            me = o.data.copy()
            me.transform(o.matrix_world)
            ob = bpy.data.objects.new(st + "_part", me)
            bpy.context.scene.collection.objects.link(ob)
            parts.append(ob)
        for ob in bpy.context.view_layer.objects:
            ob.select_set(False)
        for ob in parts:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        if len(parts) > 1:
            bpy.ops.object.join()
        res = bpy.context.view_layer.objects.active
        n0 = len(res.data.polygons)
        ratio = min(1.0, TARGET_FACES[st] / max(n0, 1))
        if ratio < 0.98:
            mod = res.modifiers.new("dec", "DECIMATE")
            mod.ratio = ratio
            bpy.ops.object.modifier_apply(modifier=mod.name)
        print("STAGE", st, "objects", len(objs), "faces", n0, "->", len(res.data.polygons))
        res.data.name = "HATCH_stage_" + st
        meshes[st] = res.data
        bpy.data.objects.remove(res, do_unlink=True)
    for o in new:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    tx = vfx.eye_glow_material("HATCH_eye_tx", vfx.TX_RED, strength=4.0)
    rx = vfx.eye_glow_material("HATCH_eye_rx", vfx.RX_CYAN, strength=4.0)
    for st, m in (("eyes_tx", tx), ("eyes_rx", rx)):
        meshes[st].materials.clear()
        meshes[st].materials.append(m)
    return meshes


def centroid(me):
    n = len(me.vertices)
    s = Vector()
    for v in me.vertices:
        s += v.co
    return s / max(n, 1)


def ik(tx, ty):
    r2 = tx * tx + ty * ty
    c2 = max(-1.0, min(1.0, (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    j2 = math.acos(c2)
    j1 = math.atan2(ty, tx) - math.atan2(L2 * math.sin(j2), L1 + L2 * math.cos(j2))
    return j1, j2


def key(o, path, frame, value, idx=None, interp="LINEAR"):
    if idx is None:
        setattr(o, path, value)
        o.keyframe_insert(path, frame=frame)
    else:
        getattr(o, path)[idx] = value
        o.keyframe_insert(path, index=idx, frame=frame)
    for fc in o.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            if int(round(kp.co[0])) == frame:
                kp.interpolation = interp


def cycle_all(ids):
    for idb in ids:
        ad = getattr(idb, "animation_data", None)
        if not ad or not ad.action:
            continue
        for fc in ad.action.fcurves:
            if not any(m.type == "CYCLES" for m in fc.modifiers):
                m = fc.modifiers.new("CYCLES")
                m.mode_before = m.mode_after = "REPEAT"


def build(out_blend, LOOP=60, F0=1):
    E.reset()
    E.HOOKS.clear()
    scn = bpy.context.scene
    scn.frame_start, scn.frame_end = F0, F0 + LOOP - 1
    scn.render.fps = 30
    top = E.coll("ENV_hatchery")
    cset = E.coll("HATCH_set", top)
    cline = E.coll("HATCH_line", top)
    cl = E.coll("HATCH_lights", top)
    hc = E.coll("HATCH_hooks", top)
    root = E.empty("hatchery_root", top, (0, 0, 0), (0, 0, 0), 0.5)
    # timing (frames within the loop)
    T_DOWN, T_GRIP, T_UP, T_OVER, T_PLACE, T_REL, T_CLR, T_HOME = 6, 8, 11, 21, 25, 26, 29, 50
    T_MOVE0, T_MOVE1 = 0, LOOP          # v2: continuous conveyor (one pitch per loop at constant speed); robots track the pallets
    f = lambda t: F0 + t

    # ---------------------------------------------------------------- set: floor, wall, sign, lights
    mb = MB()
    mb.box((0, 0.6, -0.05), (10.0, 6.0, 0.1), m=0)
    for y in (-1.1, -1.25):
        mb.box((0, y, 0.002), (9.6, 0.06, 0.004), m=1)
    mb.box((0, 2.4, 2.0), (10.0, 0.2, 4.0), m=2)
    for x in range(-4, 5):
        mb.box((x * 1.1, 2.29, 2.0), (0.04, 0.03, 4.0), m=3)
    mb.box((0, 2.28, 0.6), (10.0, 0.04, 0.06), m=4)
    for x in (-3.0, 0.0, 3.0):
        mb.box((x, 0.6, 3.6), (2.6, 0.12, 0.06), m=5)
    o = E.obj("hatch_room", mb, [vfx.wet_ground_material("HATCH_floor", (0.05, 0.055, 0.06), wetness=0.2, coverage=0.0,
                                                          rough_dry=0.35, rough_wet=0.2, grain=0.05),
                                  "paint_yellow", "cladding_light", "metal_dark", "led_cyan", "led_warm"], cset, parent=root)
    SK.neon_sign(cset, (0, 2.28, 3.0), 0.0, "OUTNOLIGHT HATCHERY", "neon_mag", 0.34, name="sign_hatchery",
                 light=((1.0, 0.1, 0.7), 120.0, cl))
    E.hook("hatchery_sign", hc, (0, 2.25, 3.0), -PI / 2, "landmark", parent=root)
    for x in (-3.0, 0.0, 3.0):
        E.light("hatch_strip_%d" % int(x), "AREA", (x, 0.6, 3.5), 180, (0.85, 0.9, 1.0), cl, size=2.4, rot=(0, 0, 0))
    E.light("hatch_front_fill", "AREA", (0, -3.5, 2.6), 120, (0.7, 0.75, 0.9), cl, size=6, target=(0, 0.5, 1.0))
    # conveyor frame, belt, tunnels, header beam with station labels
    mb = MB()
    mb.box((0, 0, BELT_Z - 0.08), (7.4, 0.5, 0.14), m=0)
    for x in [-3.5 + 0.7 * i for i in range(11)]:
        for s in (-0.22, 0.22):
            mb.box((x, s, (BELT_Z - 0.15) / 2), (0.05, 0.05, BELT_Z - 0.15), m=0)
    for s in (-0.26, 0.26):
        mb.box((0, s, BELT_Z + 0.02), (7.4, 0.02, 0.08), m=3)
    mb.box((0, -0.272, BELT_Z - 0.04), (7.4, 0.004, 0.012), m=1)
    for sx in (-1, 1):
        x = sx * 3.2
        mb.box((x, 0, BELT_Z + 0.45), (1.1, 0.7, 0.04), m=0)
        for s in (-0.35, 0.35):
            mb.box((x, s, BELT_Z + 0.22), (1.1, 0.02, 0.46), m=0)
        mb.box((x + sx * 0.55, 0, BELT_Z + 0.45), (0.02, 0.7, 0.04), m=1)
        for k in range(9):
            mb.box((sx * 2.65, -0.3 + 0.075 * k, BELT_Z + 0.22), (0.006, 0.065, 0.42), m=2)
    mb.box((0, -0.15, 2.2), (6.2, 0.1, 0.1), m=0)
    for x in (-3.0, 3.0):
        mb.box((x, -0.15, 1.2), (0.1, 0.1, 2.4), m=0)
    belt_m, b, out = vfx.nodes.new_material("HATCH_belt")
    off = b.node("ShaderNodeValue")
    off.name = off.label = "belt_offset"
    uv = b.sep(b.texcoord("Object"))
    sx_ = b.math("FRACT", b.math("DIVIDE", b.sub(uv[0], off.outputs[0]), PITCH / 6))
    st = b.smooth(sx_, 0.85, 0.9)
    p = b.principled(Base_Color=b.cmix((0.03, 0.03, 0.032, 1), (0.08, 0.08, 0.085, 1), st), Roughness=0.6)
    b.link(p.outputs[0], out.inputs["Surface"])
    E.obj("conveyor", mb, ["metal_paint", "led_cyan", E.principled("HATCH_curtain", (0.2, 0.22, 0.25), 0, 0.3), "galv"], cline, parent=root)
    bb = MB()
    bb.box((0, 0, BELT_Z - 0.005), (7.3, 0.46, 0.01), m=0)
    E.obj("belt", bb, [belt_m], cline, parent=root)
    off.outputs[0].default_value = 0.0
    off.outputs[0].keyframe_insert("default_value", frame=f(0))
    off.outputs[0].default_value = PITCH
    off.outputs[0].keyframe_insert("default_value", frame=f(LOOP))
    for fc in belt_m.node_tree.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    lab_m = E.principled("HATCH_label", (0.9, 0.95, 1.0), 0, 0.4, (0.85, 0.95, 1.0), 1.5)
    for k, lab in enumerate(STATION_LABEL):
        x = X0 + k * PITCH
        pb = MB()
        pb.box((x, -0.22, 2.05), (0.7, 0.02, 0.2), m=0)
        E.obj("station_plate_%d" % k, pb, ["sign_back"], cset, parent=root)
        t = E.text("station_lbl_%d" % k, lab, 0.075 if len(lab) < 10 else 0.05, cset, (x, -0.232, 2.05), (PI / 2, 0, 0), mat=lab_m,
                   fnt="Arial Bold", parent=root)

    # ---------------------------------------------------------------- stage meshes + slots
    meshes = stage_meshes()
    cen = {st: centroid(me) for st, me in meshes.items()}
    pal = MB()
    pal.box((0, 0, PALLET_H / 2), (0.46, 0.36, PALLET_H), m=0)
    for s in (-1, 1):
        pal.box((0, s * 0.17, PALLET_H + 0.01), (0.46, 0.02, 0.02), m=1)
    pal.box((0.2, 0, PALLET_H + 0.002), (0.03, 0.3, 0.004), m=2)
    slots = []
    for k in range(-1, 8):
        x = X0 + k * PITCH
        s = E.empty("slot_%d" % k, cline, (x, 0, BELT_Z), (0, 0, 0), 0.2, parent=root)
        key(s, "location", f(0), x, 0)
        key(s, "location", f(LOOP), x + PITCH, 0)
        E.obj("pallet_%d" % k, pal, ["metal_dark", "paint_yellow", "metal_paint"], cline, parent=s, key="pallet")
        parts = {}
        for j, st in enumerate(STAGES + ["eyes_tx", "eyes_rx"]):
            o = bpy.data.objects.new("slot_%d_%s" % (k, st), meshes[st])
            cline.objects.link(o)
            o.parent = s
            o.location = (0, 0, PALLET_H)
            jj = STAGES.index("optics") if st.startswith("eyes") else j
            vis0 = 1.0 if (0 <= k <= 5 and jj <= k - 1) else 0.0      # k >= 6: the finished bird is the hopper
            added = (0 <= k <= 4 and jj == k)
            o.scale = (vis0,) * 3
            key(o, "scale", f(0), (vis0,) * 3, interp="CONSTANT")
            if added:
                key(o, "scale", f(T_REL), (1.0,) * 3, interp="CONSTANT")
            if st in ("pcb", "dsp", "optics"):        # internals hidden once the shell is on (not visible, saves draw cost)
                if k >= 5:
                    key(o, "scale", f(0), (0.0,) * 3, interp="CONSTANT")
                elif k == 4:
                    key(o, "scale", f(T_REL + 1), (0.0,) * 3, interp="CONSTANT")
            if st.startswith("eyes"):
                lit = 1.0 if k >= 6 else 0.0
                vfx.key_obj_alpha(o, f(0), lit, "CONSTANT")
                if k == 5:
                    vfx.key_obj_alpha(o, f(16), 0.0, "LINEAR")
                    vfx.key_obj_alpha(o, f(22), 1.0, "CONSTANT")
            parts[st] = o
        slots.append((k, s, parts))

    # ---------------------------------------------------------------- robots
    robm = [E.principled("HATCH_robot_orange", (0.85, 0.32, 0.04), 0.2, 0.35, None, 0.0, coat=0.6), "metal_dark", "grille", "led_cyan",
            "rubber"]
    for k, st in enumerate(STAGES):
        x = X0 + k * PITCH
        xb = x + PITCH * 25 / LOOP           # robot sits where its pallet is at the place frame (continuous conveyor)
        base = E.empty("robot_%d_base" % k, cline, (xb, ROBOT_Y, 0), (0, 0, 0), 0.2, parent=root)
        bm = MB()
        bm.box((0, 0, 0.03), (0.5, 0.5, 0.06), m=1)
        bm.cyl((0, 0, (ARM_Z - 0.06) / 2 + 0.03), 0.08, ARM_Z - 0.12, seg=20, m=0)
        bm.cyl((0, 0, ARM_Z - 0.03), 0.1, 0.06, seg=20, m=1)
        bm.box((0.085, 0, 0.6), (0.02, 0.09, 0.16), m=2)
        bm.ellipsoid((0.09, 0, 0.75), (0.01, 0.02, 0.01), 8, 4, m=3)
        E.obj("robot_%d_column" % k, bm, robm, cline, parent=base, key="robot_column")
        j1o = E.empty("robot_%d_j1" % k, cline, (0, 0, ARM_Z), (0, 0, 0), 0.1, parent=base)
        a1 = MB()
        a1.cyl((0, 0, 0.06), 0.09, 0.12, seg=20, m=0)
        a1.box((L1 / 2, 0, 0.06), (L1, 0.1, 0.08), m=0)
        a1.cyl((L1, 0, 0.06), 0.065, 0.12, seg=18, m=0)
        E.obj("robot_%d_link1" % k, a1, robm, cline, parent=j1o, key="robot_link1")
        j2o = E.empty("robot_%d_j2" % k, cline, (L1, 0, 0.12), (0, 0, 0), 0.1, parent=j1o)
        a2 = MB()
        a2.cyl((0, 0, 0.05), 0.06, 0.1, seg=18, m=0)
        a2.box((L2 / 2, 0, 0.05), (L2, 0.08, 0.07), m=0)
        a2.cyl((L2, 0, 0.05), 0.05, 0.14, seg=16, m=1)
        a2.box((L2 * 0.4, 0, 0.11), (0.1, 0.06, 0.05), m=2)
        E.obj("robot_%d_link2" % k, a2, robm, cline, parent=j2o, key="robot_link2")
        q = E.empty("robot_%d_quill" % k, cline, (L2, 0, 0), (0, 0, 0), 0.1, parent=j2o)
        qm = MB()
        qm.cyl((0, 0, -0.25), 0.022, 0.5, seg=12, m=1)
        qm.box((0, 0, -0.52), (0.1, 0.1, 0.04), m=1)
        for s in (-1, 1):
            qm.box((s * 0.04, 0, -0.57), (0.012, 0.05, 0.07), m=4)
        E.obj("robot_%d_tool" % k, qm, robm, cline, parent=q, key="robot_tool")
        # targets relative to the base axis: place = part centroid over the slot at position k; feeder behind the robot
        c = cen[st]
        z_tip_place = BELT_Z + PALLET_H + c.z
        place = Vector((x + c.x, c.y, z_tip_place))
        feed = Vector((xb + 0.28 + c.x * 0.2, ROBOT_Y + 0.42, z_tip_place))
        tool_len = 0.6
        z_up = (place.z + 0.22) - ARM_Z - 0.12 + tool_len        # quill z offset when raised (tip 0.22 above the target)
        z_dn = place.z - ARM_Z - 0.12 + tool_len
        jf = ik(feed.x - xb, feed.y - ROBOT_Y)
        def track(t):          # place point rides with the pallet (continuous conveyor)
            return ik(place.x + PITCH * t / LOOP - xb, place.y - ROBOT_Y)
        poses = [(0, jf, z_up, "BEZIER"), (T_DOWN, jf, z_dn, "BEZIER"), (T_GRIP, jf, z_dn, "BEZIER"), (T_UP, jf, z_up, "BEZIER")]
        for t in range(T_OVER, T_CLR + 1):
            u = (t - T_OVER) / (T_PLACE - T_OVER)
            qz = z_up + (z_dn - z_up) * min(1.0, u) if t <= T_REL else z_dn + (z_up - z_dn) * (t - T_REL) / (T_CLR - T_REL)
            poses.append((t, track(t), qz, "LINEAR"))
        poses += [(T_HOME, jf, z_up, "BEZIER"), (LOOP, jf, z_up, "BEZIER")]
        for t, (j1, j2), qz, ip in poses:
            key(j1o, "rotation_euler", f(t), j1, 2, ip)
            key(j2o, "rotation_euler", f(t), j2, 2, ip)
            key(q, "rotation_euler", f(t), -(j1 + j2), 2, ip)
            key(q, "location", f(t), qz, 2, ip)
        # carried copy on the tool tip (tip = quill origin - tool_len in z); part centroid on the tip
        cobj = bpy.data.objects.new("robot_%d_carry_%s" % (k, st), meshes[st])
        cline.objects.link(cobj)
        cobj.parent = q
        cobj.location = (-c.x, -c.y, -tool_len - c.z)
        key(cobj, "scale", f(0), (0.0,) * 3, interp="CONSTANT")
        key(cobj, "scale", f(T_DOWN), (1.0,) * 3, interp="CONSTANT")
        key(cobj, "scale", f(T_REL), (0.0,) * 3, interp="CONSTANT")
        if st == "optics":
            for es in ("eyes_tx", "eyes_rx"):
                e = bpy.data.objects.new("robot_%d_carry_%s" % (k, es), meshes[es])
                cline.objects.link(e)
                e.parent = q
                e.location = cobj.location
                vfx.key_obj_alpha(e, f(0), 0.0, "CONSTANT")
                key(e, "scale", f(0), (0.0,) * 3, interp="CONSTANT")
                key(e, "scale", f(T_DOWN), (1.0,) * 3, interp="CONSTANT")
                key(e, "scale", f(T_REL), (1.0 * 0,) * 3, interp="CONSTANT")
        # feeder table with a resting part
        fb = MB()
        fb.box((0, 0, (BELT_Z) / 2), (0.36, 0.34, BELT_Z), m=0)
        fb.box((0, 0, BELT_Z + 0.01), (0.4, 0.38, 0.02), m=1)
        E.obj("feeder_%d" % k, fb, ["metal_paint", "metal_dark"], cline, loc=(feed.x, feed.y, 0), parent=root, key="feeder")
        fo = bpy.data.objects.new("feeder_%d_%s" % (k, st), meshes[st])
        cline.objects.link(fo)
        fo.parent = root
        fo.location = (feed.x - c.x, feed.y - c.y, BELT_Z + PALLET_H)
        E.hook("station_%s" % st, hc, (x, 0.0, BELT_Z + PALLET_H), PI / 2, "landmark", parent=root)

    # ---------------------------------------------------------------- inspection (position 5): gantry, camera, ring light, laser
    x5 = X0 + 5 * PITCH + 0.23 * PITCH        # gantry where the moving bird is during the eyes-on window (f16-22)
    gm = MB()
    for s in (-0.42, 0.42):
        gm.box((x5, s, 0.95), (0.08, 0.08, 1.9), m=0)
    gm.box((x5, 0, 1.92), (0.12, 0.96, 0.1), m=0)
    gm.box((x5, -0.05, 1.72), (0.16, 0.14, 0.26), m=1)
    gm.cyl((x5, -0.05, 1.56), 0.05, 0.08, seg=16, m=2)
    gm.ring((x5, -0.05, 1.5), 0.12, 0.09, 0.02, seg=24, m=3)
    gm.box((x5, -0.53, 1.95), (0.08, 0.14, 0.04), m=0)                 # QA display hangs from the gantry (v2: off the hop path)
    gm.box((x5, -0.62, 1.84), (0.02, 0.02, 0.2), m=0)
    gm.box((x5, -0.62, 1.64), (0.5, 0.04, 0.24), m=1)
    gm.box((x5, -0.642, 1.64), (0.46, 0.004, 0.2), m=4)
    gm.box((x5, -0.25, 1.95), (0.06, 0.5, 0.04), m=0)
    E.obj("inspection_gantry", gm, ["metal_paint", "camera_white", "lens_black", "led_cyan", "glass_dark"], cline, parent=root)
    E.hook("inspection_camera", hc, (x5, -0.05, 1.5), PI / 2, "camera", parent=root, rot=(0, PI / 2, 0))
    laser_m = E.principled("HATCH_laser", (1.0, 0.05, 0.03), 0, 0.3, (1.0, 0.05, 0.03), 25.0)
    lm = MB()
    lm.box((0, 0, 0), (0.004, 0.4, 0.004), m=0)
    las = E.obj("inspection_laser", lm, [laser_m], cline, loc=(x5 - 0.22, 0, BELT_Z + 0.28), parent=root, shadow=False)
    key(las, "location", f(0), x5 - 0.22, 0)
    key(las, "location", f(4), x5 - 0.22, 0)
    key(las, "location", f(18), x5 + 0.22, 0)
    key(las, "location", f(26), x5 - 0.22, 0)
    key(las, "location", f(LOOP), x5 - 0.22, 0)
    key(las, "scale", f(0), (0.0,) * 3, interp="CONSTANT")
    key(las, "scale", f(4), (1.0,) * 3, interp="CONSTANT")
    key(las, "scale", f(26), (0.0,) * 3, interp="CONSTANT")
    ok = E.text("qa_pass", "PASS", 0.1, cline, (x5, -0.646, 1.64), (PI / 2, 0, 0), mat="neon_green", fnt="Arial Black", parent=root)
    ok.color = (1, 1, 1, 0)
    vfx.key_obj_alpha(ok, f(0), 0.0, "CONSTANT")
    vfx.key_obj_alpha(ok, f(22), 1.0, "CONSTANT")
    vfx.key_obj_alpha(ok, f(56), 0.0, "CONSTANT")
    E.light("inspection_ring", "SPOT", (x5, -0.05, 1.48), 60, (0.85, 0.95, 1.0), cl, spot_deg=50, blend=0.5, rot=(0, 0, 0), radius=0.1)
    head5 = Vector((x5 + 0.044 * S_BIRD * 1.0, 0, BELT_Z + PALLET_H + 0.0815 * S_BIRD))
    # ---------------------------------------------------------------- finished bird hops off the end of the line (v2)
    x6 = X0 + 6 * PITCH
    dm = MB()
    dm.box((2.5, -0.64, 0.6), (0.62, 0.38, 0.04), m=0)
    for sx in (2.24, 2.76):
        for sy in (-0.8, -0.48):
            dm.box((sx, sy, 0.29), (0.04, 0.04, 0.58), m=0)
    dm.box((2.5, -0.832, 0.6), (0.62, 0.006, 0.04), m=1)
    E.obj("dispatch_ledge", dm, ["metal_paint", "hazard"], cset, parent=root)
    E.text("dispatch_lbl", "DISPATCH", 0.045, cset, (2.5, -0.836, 0.6), (PI / 2, 0, 0), mat=lab_m, fnt="Arial Bold", parent=root)
    hop = E.empty("hopper", cline, (x6, 0, BELT_Z + PALLET_H), (0, 0, 0), 0.15, parent=root)
    for st in ("chassis", "shell", "eyes_tx", "eyes_rx"):
        ho = bpy.data.objects.new("hopper_" + st, meshes[st])
        cline.objects.link(ho)
        ho.parent = hop
    z0 = BELT_Z + PALLET_H
    p0 = Vector((x6, 0, z0))
    path = [(4, 13, Vector((x6 + PITCH * 4 / LOOP, 0, z0)), Vector((2.5, -0.62, 0.62)), 0.22),
            (19, 28, Vector((2.5, -0.62, 0.62)), Vector((2.75, -1.3, 0.0)), 0.16),
            (33, 41, Vector((2.75, -1.3, 0.0)), Vector((3.1, -1.9, 0.0)), 0.18),
            (45, 53, Vector((3.1, -1.9, 0.0)), Vector((3.6, -2.5, 0.0)), 0.18)]
    hop.rotation_mode = "XYZ"
    key(hop, "location", f(0), tuple(p0))
    key(hop, "location", f(3), (x6 + PITCH * 3 / LOOP, 0, z0))
    key(hop, "rotation_euler", f(0), (0, 0, 0))
    key(hop, "scale", f(0), (1, 1, 1), interp="LINEAR")
    for ta, tb, pa, pb, peak in path:
        yaw = math.atan2(pb.y - pa.y, pb.x - pa.x)
        key(hop, "rotation_euler", f(ta - 2), (0, 0, yaw))
        key(hop, "scale", f(ta - 1), (1.04, 1.04, 0.86))
        key(hop, "scale", f(ta), (1, 1, 1))
        for t in range(ta, tb + 1):
            u = (t - ta) / (tb - ta)
            pos = pa.lerp(pb, u) + Vector((0, 0, peak * 4 * u * (1 - u) + max(0.0, pa.z - pb.z) * 0.0))
            key(hop, "location", f(t), tuple(pos))
            key(hop, "rotation_euler", f(t), (0, -0.25 if u < 0.5 else 0.2, yaw))
        key(hop, "rotation_euler", f(tb + 1), (0, 0, yaw))
        key(hop, "scale", f(tb + 1), (1.04, 1.04, 0.88))
        key(hop, "scale", f(tb + 2), (1, 1, 1))
    key(hop, "scale", f(54), (0, 0, 0), interp="CONSTANT")
    key(hop, "location", f(LOOP), tuple(p0))
    key(hop, "rotation_euler", f(LOOP), (0, 0, 0))
    key(hop, "scale", f(LOOP), (1, 1, 1), interp="CONSTANT")
    E.hook("hop_start", hc, tuple(p0), 0.0, "landmark", parent=root, note="finished bird leaves the pallet f4")
    E.hook("dispatch_ledge", hc, (2.5, -0.62, 0.62), -PI / 2, "landmark", parent=root, note="hop 1 lands f13, leaves f19")
    E.hook("cam_hop", hc, (1.45, -2.7, 1.35), PI / 2, "camera", parent=root, target=[2.55, -0.7, 0.65],
           note="suggested camera for the hop-off (f0-30)")
    E.hook("eyes_on", hc, tuple(head5), PI / 2, "landmark", parent=root, note="bird head at the inspection station (eyes light f16-22)")
    E.hook("exit", hc, (X0 + 6 * PITCH, 0, BELT_Z + PALLET_H), 0.0, "landmark", parent=root)
    E.hook("cam_overview", hc, (0.2, -3.6, 1.75), PI / 2, "camera", parent=root, target=[0.2, 0.3, 1.1])

    cycle_all(list(bpy.data.objects) + [belt_m.node_tree] + [m.node_tree for m in bpy.data.materials if m.node_tree])
    vfx.night_sky_world(scn, strength=0.0)
    vfx.world_haze(scn, density=0.004, color=(0.6, 0.65, 0.8))
    vfx.setup_eevee(scn, "draft", view_transform="AgX", look="AgX - Punchy")
    objs = list(top.all_objects)
    print("HATCHERY objects", len(objs), "tris", E.count_tris(objs))
    E.save_blend(out_blend)
    E.write_json(os.path.splitext(out_blend)[0] + ".json", {
        "asset_id": "hatchery", "date": "2026-10-10", "units": "m", "author": "workstream E (DevLog-013)", "blend": E.rel(out_blend),
        "root": "hatchery_root (empty; move/rotate it to place the set)", "loop_frames": LOOP, "frame_start": F0,
        "loop": "all F-curves carry Cycles modifiers (REPEAT); belt stripe period divides the pitch, so the offset reset is invisible",
        "bird_scale": S_BIRD, "pitch_m": PITCH, "belt_top_z": BELT_Z,
        "stations": {str(k): lab for k, lab in enumerate(STATION_LABEL)},
        "timing_frames": {"robot_down": T_DOWN, "grip": T_GRIP, "place": T_PLACE, "release": T_REL,
                          "conveyor": "continuous, one pitch per loop", "eyes_on": [16, 22], "qa_pass": [22, 56],
                          "hop": [[4, 13], [19, 28], [33, 41], [45, 53]], "hopper_hidden": [54, LOOP]},
        "append": "see scripts/env/hatchery_api.py: append_hatchery(), shift_loop(), bake_hatchery_loop(); DevLog-013 section 7",
        "stage_source": "hero GLB parts by prefix: " + "; ".join("%s=%s" % (k, ",".join(v)) for k, v in PREFIX.items()),
        "hooks": E.hooks_json(), "objects": len(objs), "triangles": E.count_tris(objs)})


if __name__ == "__main__":
    a = E.argv_after_dashes()
    out = a[0] if a else os.path.join(E.OUT_ASSETS, "hatchery.blend")
    loop = int(a[1]) if len(a) > 1 else 60
    f0 = int(a[2]) if len(a) > 2 else 1
    build(out, loop, f0)
    print("SAVED", out)
