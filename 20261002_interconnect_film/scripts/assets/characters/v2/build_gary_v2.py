"""Gary v2: chunky clay hardware technician. Run:
Blender -b --python build_gary_v2.py -- <out_dir> [no_actions]

Builds gary_v2 (holes default 0) and gary_v2_holes30 (holes default 0.3). Same skeleton / hooks / properties as v1
(plus jiggle, squash and hat bones). Design values (not measured): crown 1.69 m, hat top 1.753 m.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy
import chars_geo as G
import chars_mat as M
import chars_body as B
import chars_outfit as O
import chars_actions as A
import common as C
import meta_v2 as MV

ARGS = C.argv_after_dashes()
OUT = ARGS[0]
NO_ACTIONS = "no_actions" in ARGS

HR = 0.045
GARY_HOLES = [
    dict(name="hole_1", prop="p_hole_1_radius", bone="chest", pos=(0.07, 0.0, 1.24), axis="Y", radius=HR, half_len=0.30),
    dict(name="hole_2", prop="p_hole_2_radius", bone="spine_2", pos=(-0.08, 0.0, 1.12), axis="Y", radius=HR, half_len=0.30),
    dict(name="hole_3", prop="p_hole_3_radius", bone="spine_1", pos=(0.10, 0.0, 1.05), axis="Y", radius=HR, half_len=0.30),
    dict(name="hole_4", prop="p_hole_4_radius", bone="chest", pos=(-0.05, 0.0, 1.32), axis="Y", radius=HR, half_len=0.30),
    dict(name="hole_5", prop="p_hole_5_radius", bone="spine_1", pos=(-0.12, 0.0, 1.04), axis="Y", radius=HR, half_len=0.30),
    dict(name="headhole", prop="p_head_hole_radius", bone="head", pos=(0.0, 0.017, 1.650), axis="X", radius=0.036, half_len=0.26),
]

ZB_HAT = 1.652      # hat base height (baseline m)
HAT_TOP = 1.799
HAT_R = 0.148       # dome base radius (x)


def z_at_x(spec, x, off):
    zs = np.linspace(1.30, 1.47, 400)
    rx, ry, cy = B.torso_at(spec, zs, off)
    k = int(np.argmin(np.abs(rx - abs(x))))
    return float(zs[k])


def head_part(ch, part, name, mats, **kw):
    p = part.copy()
    p.v = ch.T(p.v)
    return ch.add(p, name, mats, **kw)


def build(cid, hole_default):
    spec = B.default_spec()
    spec["H"] = 1.69
    spec["build"] = 1.0
    spec["belly"] = 0.45
    spec["wide"] = 1.06
    spec["extra_bones"] = [dict(name="hat", head=(0.0, 0.0, 1.66), tail=(0.0, 0.0, 1.80), parent="head")]
    ch = B.Character(cid, spec)
    K = ch.K
    ch.jiggles.append(("belly", (0.0, -0.13, 1.04), (0.20, 0.14, 0.13), 0.9))
    ch.root_props()
    for h in GARY_HOLES:
        h["default"] = hole_default
    hg = ch.setup_holes(GARY_HOLES)
    pre = "MAT_characters_%s_" % cid
    O.face_mats(ch, "#e2a57a", brow_hex="#3a2412", hole_group=hg, lip_hex="#c9705f")
    FOLD_PANTS = [(0.50, 0.10, 17.0, 0.5), (0.93, 0.08, 15.0, 0.4), (0.22, 0.07, 24.0, 0.4)]
    ch.mat("hat", M.clay(pre + "hardhat", "#ff7a12", rough=0.42, bump=0.16, holes=hg))
    ch.mat("overalls", M.clay(pre + "overalls", "#2c5db5", rough=0.72, bump=0.28, holes=hg, sheen=0.2, folds=FOLD_PANTS))
    ch.mat("shirt", M.clay(pre + "shirt", "#d9d1bd", rough=0.74, bump=0.28, holes=hg, sheen=0.2,
                           folds=[(1.30, 0.06, 26.0, 0.5), (1.09, 0.10, 18.0, 0.5)]))
    ch.mat("boot", M.clay(pre + "boots", "#7b4a25", rough=0.55, bump=0.3, holes=hg))
    ch.mat("sole", M.clay(pre + "sole", "#2a2a2c", rough=0.7, bump=0.2, holes=hg))
    ch.mat("belt", M.clay(pre + "belt", "#5b3a1f", rough=0.5, bump=0.25, holes=hg))
    ch.mat("tool_red", M.clay(pre + "tool_red", "#c32a20", rough=0.55, bump=0.2, holes=hg))
    ch.mat("tool_blue", M.clay(pre + "tool_blue", "#3aa0e0", rough=0.55, bump=0.2, holes=hg))
    ch.mat("metal", M.metal(pre + "metal", "#c3c7cf", 0.3))
    ch.mat("brass", M.metal(pre + "brass", "#d9b04a", 0.3))
    ch.mat("hair", M.clay(pre + "hair", "#3a2412", rough=0.75, bump=0.3, holes=hg))
    spec = ch.spec
    J = ch.J
    surf = O.surf_pt

    # ---------------------------------------------------------------- bare skin
    ch.add(ch.skin_parts(), "body", ["skin"], jig=True)
    ch.build_face()

    # ---------------------------------------------------------------- t-shirt: torso + short sleeves with hems + neck band
    tp = ch.torso_part(off=0.010, z0=1.00, z1=1.385, cap1=None)
    parts = [tp]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el = J["sh"] * sx, J["el"] * sx
        e2 = sh + (el - sh) * 0.50
        rr = ch.ARM_UA[0] + 0.016
        sl = G.tube(np.array([sh, (sh + e2) / 2, e2]), np.array([rr[0], rr[0] * 0.98, rr[1] * 0.99]), np.array([rr[0], rr[0] * 0.98, rr[1] * 0.99]),
                    u=(0, 1, 0), nseg=24, cap0="round", cap1=None)
        sl.set_w("upper_arm_" + sn, 1.0)
        parts.append(sl)
        d = (e2 - sh) / np.linalg.norm(e2 - sh)
        hem = O.ring_tube(e2, d, rr[1] * 0.99, rr[1] * 0.99, 0.0105, u_hint=(0, 1, 0))
        hem.set_w("upper_arm_" + sn, 1.0)
        parts.append(hem)
    rx_, ry_, cy_ = B.torso_at(spec, np.array([1.385]), 0.010)
    nb = O.ring_tube((0, cy_[0], 1.385), (0, 0, 1), rx_[0] * 1.0, ry_[0] * 1.0, 0.0125)
    nb.set_w("neck", 1.0)
    parts.append(nb)
    shirt = G.merge(parts)
    shirt.closed = False
    ch.add(shirt, "shirt", ["shirt"], jig=True)

    # ---------------------------------------------------------------- overalls: pelvis + legs + waist band, rolled cuffs, folds
    TAPER = (0.80, 1.0, 0.42)
    band = ch.torso_part(off=0.016, z0=0.80, z1=1.22, cap1=None, taper=TAPER, dz=0.02, nseg=48)
    parts = [band]
    creases = []
    for side, sn in ((1, "L"), (-1, "R")):
        lt, end = O.leg_tubes(ch, sn, side, 0.015, ztop_shin=0.215, cuff=0.0, flare=0.022)
        parts += lt
        # rolled cuff sits on the open hem
        c, rx0, ry0, bone = O.leg_point(ch, sn, side, 0.215, 0.015)
        cuff = O.ring_tube(c, (0, 0, 1), ry0 + 0.022, rx0 + 0.022, 0.0165, u_hint=(1, 0, 0), n=32, ns=10)
        cuff.set_w("shin_" + sn, 1.0)
        parts.append(cuff)
        # knee creases (front arcs), two chunky folds
        for z, sag, r_, a, tl in ((0.545, 0.010, 0.0080, 1.15, 0.010), (0.490, 0.015, 0.0092, 1.30, -0.012)):
            creases.append(O.fold_arc(ch, sn, side, z, 0.0165, -a, a, r=r_, sag=sag, tilt=tl))
    ov = G.merge(parts)
    ov.closed = False
    ch.add(ov, "overalls", ["overalls"], jig=True, solid=0.0)
    for z, a, tl, sg in ((1.050, 1.05, 0.008, 0.008), (1.082, 0.85, -0.010, 0.010), (1.112, 0.60, 0.006, 0.008)):
        creases.append(O.torso_arc(ch, z, 0.0165, -a, a, r=0.0085, sag=sg, tilt=tl))
    cr = G.merge(creases)
    ch.add(cr, "creases", ["overalls"], jig=True)

    # bib: structured patch on the torso front, slightly proud, thickness inward
    def bib_pt(i, j, nz=16, nx=21, z0=1.17, z1=1.355):
        z = z0 + (z1 - z0) * i / (nz - 1)
        u = -1 + 2 * j / (nx - 1)
        xb = 0.118 * (1 - 0.30 * G.smoothstep(1.28, 1.36, z) ** 1.6)
        return surf(spec, u * xb, z, True, 0.0215)
    bib = O.grid_patch(bib_pt, 21, 16)
    ch.torso_weights(bib)
    ch.add(bib, "bib", ["overalls"], solid=0.007, jig=True, bevel=0.0015)
    # bib pocket (flattened superellipsoid pasted on the bib) with a pen
    pz = 1.262
    py = surf(spec, 0.0, pz, True, 0.0215)[1]
    pocket = G.ellipsoid(np.array([0.0, py - 0.001, pz]), (0.060, 0.0115, 0.046), nseg=20, nrings=12, e=0.42)
    ch.torso_weights(pocket)
    pen = G.capsule(np.array([0.030, py - 0.008, pz + 0.028]), np.array([0.036, py - 0.014, pz + 0.095]), 0.0082, 0.0074, nseg=10)
    ch.torso_weights(pen)
    ch.add(G.merge([pocket]), "pocket", ["overalls"], jig=True)
    ch.add(pen, "pen", ["tool_red"], jig=False)
    # buckles at the strap/bib junction and rivets
    fr, bt, st_ = [], [], []
    for side in (1, -1):
        cb = surf(spec, side * 0.088, 1.335, True, 0.034)
        fr.append(O.rect_frame(cb, 0.040, 0.032, r=0.0042))
        bar = G.capsule(cb + np.array([-0.019, -0.0, 0.0]), cb + np.array([0.019, -0.0, 0.0]), 0.0035, 0.0035, nseg=8)
        fr.append(bar)
        for xx in (side * 0.056,):
            pp = surf(spec, xx, 1.298, True, 0.0235)
            rv = G.ellipsoid(pp, (0.0075, 0.0045, 0.0075), nseg=10, nrings=6)
            bt.append(rv)
        # strap: bib top corner over the shoulder to the back
        zt = z_at_x(spec, side * 0.126, 0.030)
        pts = [surf(spec, side * 0.088, 1.33, True, 0.026), surf(spec, side * 0.108, 1.385, True, 0.030),
               np.array([side * 0.126, 0.0 + 0.0, zt + 0.0]), surf(spec, side * 0.116, 1.385, False, 0.030),
               surf(spec, side * 0.098, 1.30, False, 0.026), surf(spec, side * 0.090, 1.23, False, 0.024)]
        stp = G.tube(np.array(pts), 0.022, 0.0065, u=(1, 0, 0), nseg=10, cap0="flat", cap1="flat")
        stp.set_w("chest", 1.0)
        st_.append(stp)
    for q in fr + bt:
        ch.torso_weights(q)
    ch.add(G.merge(st_), "straps", ["overalls"])
    ch.add(G.merge(fr), "bib_buckles", ["brass"], bevel=0.001)
    ch.add(G.merge(bt), "buttons", ["brass"])

    # ---------------------------------------------------------------- tool belt, buckle, pouches, pliers, cable-tie roll
    zb = 1.000
    th = np.linspace(0, 2 * math.pi, 56, endpoint=False)
    rx, ry, cy = B.torso_at(spec, np.array([zb]), 0.0)
    path = np.array([[(rx[0] + 0.034) * math.cos(t), cy[0] + (ry[0] + 0.034) * math.sin(t), zb] for t in th])
    belt = G.tube(path, 0.023, 0.0085, u=(0, 0, 1), nseg=10, closed_path=True)
    belt.set_w("spine_1", 0.5)
    belt.add_w("hips", 0.5)
    ch.add(belt, "belt", ["belt"], jig=True)
    bc = np.array([0.0, cy[0] - (ry[0] + 0.045), zb])
    bk = [O.rect_frame(bc, 0.062, 0.050, r=0.0065), G.capsule(bc + np.array([-0.024, 0, 0]), bc + np.array([0.0, 0, 0]), 0.0045, 0.0045, nseg=8)]
    for q in bk:
        q.set_w("hips", 1.0)
    ch.add(G.merge(bk), "buckle", ["brass"], bevel=0.001)
    pouches = []

    def pouch(c, rad, flap=True, rot=None):
        pp = G.ellipsoid(np.array(c), rad, nseg=18, nrings=10, e=0.58)
        pp.set_w("hips", 1.0)
        pouches.append(pp)
        if flap:
            fl = G.ellipsoid(np.array(c) + np.array([0, 0, rad[2] * 0.62]), (rad[0] * 1.06, rad[1] * 1.10, rad[2] * 0.42), nseg=16, nrings=8, e=0.55)
            fl.set_w("hips", 1.0)
            pouches.append(fl)
    pouch((-0.226, -0.030, 0.935), (0.034, 0.058, 0.070), flap=False)     # pliers holster (character right)
    pouch((0.226, -0.020, 0.940), (0.034, 0.056, 0.066))                  # left hip pouch with flap
    pouch((-0.075, 0.140, 0.945), (0.062, 0.034, 0.058))                  # back pouch
    ch.add(G.merge(pouches), "pouches", ["belt"], jig=False)
    tools = []
    for dx, dy in ((-0.013, 0.0), (0.013, -0.010)):
        hnd = G.capsule(np.array([-0.226 + dx * 0.5, -0.030 + dy, 0.985]), np.array([-0.226 + dx * 1.7, -0.030 + dy * 2.2, 1.098]), 0.0112, 0.0125, nseg=12)
        hnd.set_w("hips", 1.0)
        tools.append(hnd)
    ch.add(G.merge(tools), "pliers_handles", ["tool_red"])
    jaw = G.capsule(np.array([-0.226, -0.030, 0.968]), np.array([-0.226, -0.030, 1.010]), 0.0095, 0.008, nseg=10)
    jaw.set_w("hips", 1.0)
    ch.add(jaw, "pliers_jaw", ["metal"])
    ring = G.tube(np.array([[0.125 + 0.060 * math.cos(t), 0.135 + 0.0, 0.955 + 0.060 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 22, endpoint=False)]),
                  0.0125, 0.0125, u=(0, 1, 0), nseg=8, closed_path=True)
    ring.set_w("hips", 1.0)
    ch.add(ring, "cable_tie_roll", ["tool_blue"])

    # ---------------------------------------------------------------- boots
    for side, sn in ((1, "L"), (-1, "R")):
        bt_ = O.boot(ch, sn, side)
        ch.add(G.merge([bt_[0], bt_[2], bt_[3]]), "boot_" + sn, ["boot"])
        ch.add(bt_[1], "sole_" + sn, ["sole"], bevel=0.002)

    # ---------------------------------------------------------------- hair (v1 head units, then T) and hard hat
    hr = O.hair_band(ch, 1.575, 1.70, 0.007, back_only=True, front_cut=0.028)
    head_part(ch, hr, "hair", ["hair"])
    cy0 = -0.006
    ZT = HAT_TOP
    Hh = ZT - ZB_HAT
    prof = [(HAT_R + 0.0, ZB_HAT - 0.014)]
    for a in np.linspace(0.0, math.pi / 2, 16):
        e_ = 2.35
        rr = HAT_R * (math.cos(a) ** (2.0 / e_))
        zz = ZB_HAT + Hh * (math.sin(a) ** (2.0 / e_))
        prof.append((max(rr, 0.0), zz))
    prof[-1] = (0.0, ZT)
    dome = G.lathe(prof, center=(0, cy0, 0), axis=(0, 0, 1), nseg=44)
    dome.v[:, 1] = cy0 + (dome.v[:, 1] - cy0) * 1.10
    brim_prof = [(HAT_R - 0.006, ZB_HAT - 0.010), (HAT_R + 0.024, ZB_HAT - 0.016), (HAT_R + 0.032, ZB_HAT - 0.007),
                 (HAT_R + 0.026, ZB_HAT + 0.006), (HAT_R + 0.006, ZB_HAT + 0.012)]
    brim = G.lathe(brim_prof, center=(0, cy0, 0), axis=(0, 0, 1), nseg=44)
    d = brim.v[:, 1] - cy0
    brim.v[:, 1] = cy0 + np.where(d < 0, d * 1.50, d * 1.10)
    ang = np.linspace(-1.38, 1.38, 22)
    yy = HAT_R * 0.985 * np.sin(ang)
    rr = np.minimum(np.abs(yy) / HAT_R, 0.999)
    ridge_path = np.stack([np.zeros_like(ang), cy0 + 1.10 * yy, ZB_HAT + Hh * (1 - rr ** 2.35) ** (1 / 2.35) + 0.006], axis=1)
    ridge = G.tube(ridge_path, 0.0165, 0.0115, u=(1, 0, 0), nseg=10)
    hat = G.merge([dome, brim, ridge])
    R_tilt = G.rot_about((1, 0, 0), -0.07)
    c_t = np.array([0.0, cy0, 1.70])
    hat.v = (hat.v - c_t) @ R_tilt.T + c_t
    hat.set_w("hat", 1.0)
    ch.add(hat, "hardhat", ["hat"], bevel=0.0015)

    # ---------------------------------------------------------------- hooks (names and meaning as in v1)
    ch.hook("head_top", "head", (0.0, 0.005, 1.815))
    ch.hook("hip", "hips", (-0.19, -0.0, 0.95))
    ch.hook("muzzle_self", "head", (-0.21, 0.017, 1.650), rotm=np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]))
    O.hook_hand(ch, "L", 1)
    O.hook_hand(ch, "R", -1)
    ch.build_rims(ch.m["rim"])
    ch.root["p_stature_m"] = 1.75
    ch.root["p_scale"] = 1.0
    ch.root["pv_face_z"] = float(K * (spec["chin_post"] + 0.45 * 0.233 * spec["headS"][2]))
    ch.root["pv_face_dist"] = float(3.0 * K * 0.233 * spec["headS"][2])
    return ch


def meta(ch):
    m = B.meta_common(ch)
    m.update(dict(
        description="Gary v2 (chunky clay): hardware technician with oversized hard hat, t-shirt, overalls with bib pocket, tool belt, boots, mitten hands. Same rig, hooks, properties and shape keys as gary (v1).",
        sources=[dict(what="Joint table and K identical to v1 (Drillis and Contini fractions as used in v1; unverified)", used_for="skeleton", access="2026-10-02", level="C"),
                 dict(what="Design values chosen for the cartoon look (head about 1/5 of crown height, thicker limbs, bigger hat and boots)", used_for="v2 proportions", access="2026-10-02", level="C")],
    ))
    return m


if __name__ == "__main__":
    for cid, hd in (("gary_v2", 0.0), ("gary_v2_holes30", 0.3)):
        ch = build(cid, hd)
        m = MV.extend(meta(ch), ch, "gary", [
            "Gary extras: bone 'hat' (child of head) carries the hard hat (jiggle via p_jiggle_hat); the hat is tilted back 4 degrees and its top is at 1.753 m (HOOK_head_top).",
            "gary_v2_holes30 is the same asset with all hole properties defaulting to 0.3 (as gary_holes30 in v1). Hole k is created in scene k (see SCENE_BRIEF); schedule r = max(0.3, 0.9 ** ((T_now - T_shot) / 1.5)).",
            "HOOK_muzzle_self is at the same place as v1 (0.21 m to the character's right of the head axis at head-hole height); the bigger head leaves about 0.06 m between the head side and the hook."])
        if not NO_ACTIONS:
            names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, cid)
            m["actions"] = A.describe(names)
        bpy.context.view_layer.update()
        C.finish(cid, os.path.join(OUT, cid + ".blend"), ch.coll, m, preview_dir=None)
        if NO_ACTIONS:
            break
