"""Manager v2: chunky clay manager (suit with lapels and collars, white shirt, red tie with a swinging blade, thick shoes).

Run: Blender -b --python build_manager_v2.py -- <out_dir> [no_actions]
Same skeleton, hooks, properties and shape keys as manager (v1); plus belly, tie_1..3 jiggle bones, p_squash, p_squash_head.
Design values (not measured): stature 1.85 m.
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

Z_NECK = 1.395      # top of the clothing (open neckline)
Z_BTN = 1.00        # jacket button point (V opening ends)
HW_TOP = 0.128      # half-width of the V opening at the neckline


def hw(z):
    t = np.clip((z - Z_BTN) / (Z_NECK - Z_BTN), 0.0, 1.0)
    return HW_TOP * t ** 0.62 if z > Z_BTN else 0.0


def head_part(ch, part, name, mats, **kw):
    p = part.copy()
    p.v = ch.T(p.v)
    return ch.add(p, name, mats, **kw)


def build(cid="manager_v2"):
    spec = B.default_spec()
    spec["H"] = 1.85
    spec["build"] = 1.10
    spec["belly"] = 0.30
    spec["shoulders"] = 1.08
    spec["arm_r"] = 1.06
    spec["leg_r"] = 1.05
    spec["wide"] = 1.04
    spec["head"].update(dict(w0=0.082, jaw=0.05, chin=0.085, brow_t=0.0088, brow_h=0.0112, ear_s=0.92))
    spec["extra_bones"] = [
        dict(name="tie_1", head=(0.0, -0.112, 1.345), tail=(0.0, -0.112, 1.250), parent="chest"),
        dict(name="tie_2", head=(0.0, -0.112, 1.250), tail=(0.0, -0.112, 1.150), parent="tie_1"),
        dict(name="tie_3", head=(0.0, -0.112, 1.150), tail=(0.0, -0.112, 1.040), parent="tie_2"),
    ]
    ch = B.Character(cid, spec)
    K = ch.K
    ch.jiggles.append(("belly", (0.0, -0.14, 1.04), (0.20, 0.14, 0.13), 0.9))
    ch.root_props()
    pre = "MAT_characters_%s_" % cid
    O.face_mats(ch, "#e89b86", brow_hex="#3d2616", vein=(0, 0.115, -0.085, 1.655, 1.712), flush_color="#e0301f", lip_hex="#cc6a62")
    ch.mat("jacket", M.clay(pre + "suit", "#2a2f4a", rough=0.75, bump=0.3, sheen=0.25,
                            folds=[(0.50, 0.09, 15.0, 0.30), (0.20, 0.06, 20.0, 0.30)]))
    ch.mat("shirt", M.clay(pre + "shirt", "#eeeae0", rough=0.7, bump=0.28, sheen=0.2))
    ch.mat("tie", M.clay(pre + "tie", "#d4121c", rough=0.5, bump=0.22, sheen=0.3))
    ch.mat("shoe", M.gloss(pre + "shoes", "#161618", rough=0.3, spec=0.6))
    ch.mat("sole", M.clay(pre + "sole", "#0c0c0d", rough=0.7, bump=0.0))
    ch.mat("hair", M.clay(pre + "hair", "#4a2f1c", rough=0.7, bump=0.3))
    ch.mat("brass", M.metal(pre + "brass", "#d9b04a", 0.3))
    J = ch.J
    surf = O.surf_pt
    TAPER = (0.80, 1.0, 0.42)

    ch.add(ch.skin_parts(), "body", ["skin"], jig=True)
    ch.build_face()

    # ---------------------------------------------------------------- white shirt: torso, long sleeves with cuffs, collar band and points
    tp = ch.torso_part(off=0.010, z0=0.96, z1=Z_NECK, cap1=None)
    parts = [tp]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.04
        ar = spec["arm_r"]
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), ch.ARM_UA[0] * ar + 0.010, ch.ARM_UA[1] * ar + 0.010, u=(0, 1, 0), nseg=24)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), ch.ARM_FA[0] * ar + 0.010, ch.ARM_FA[1] * ar + 0.010, u=(0, 1, 0), nseg=24, cap1="flat")
        fa.set_w("forearm_" + sn, 1.0)
        d = (end - el) / np.linalg.norm(end - el)
        cuff = O.ring_tube(end - d * 0.02, d, ch.ARM_FA[0][2] * ar + 0.016, ch.ARM_FA[0][2] * ar + 0.016, 0.011, u_hint=(0, 1, 0))
        cuff.set_w("forearm_" + sn, 1.0)
        parts += [ua, fa, cuff]
    rx_, ry_, cy_ = B.torso_at(spec, np.array([Z_NECK]), 0.010)
    band = O.ring_tube((0, cy_[0], Z_NECK), (0, 0, 1), rx_[0] * 0.98, ry_[0] * 0.98, 0.014)
    band.set_w("neck", 1.0)
    parts.append(band)
    # collar points: two wedge ribbons from the centre front down and out over the chest
    for side in (1, -1):
        pts = [surf(spec, side * 0.030, 1.372, True, 0.018), surf(spec, side * 0.060, 1.338, True, 0.021), surf(spec, side * 0.088, 1.296, True, 0.023)]
        cp = G.tube(np.array(pts), [0.020, 0.026, 0.010], [0.0055, 0.0055, 0.0045], u=(0.55, -0.1, 0.6), nseg=8, cap0="round", cap1="round", ncap=2)
        cp.set_w("chest", 1.0)
        parts.append(cp)
    shirt = G.merge(parts)
    shirt.closed = False
    ch.add(shirt, "shirt", ["shirt"], jig=True)

    # ---------------------------------------------------------------- trousers with a break over the shoes
    tr = [ch.torso_part(off=0.014, z0=0.80, z1=1.12, cap1=None, taper=TAPER, dz=0.02, nseg=48)]
    creases = []
    for side, sn in ((1, "L"), (-1, "R")):
        lt, end = O.leg_tubes(ch, sn, side, 0.014, ztop_shin=0.12, cuff=0.0, flare=0.012)
        tr += lt
        c, rx0, ry0, bone = O.leg_point(ch, sn, side, 0.12, 0.014)
        cuff = O.ring_tube(c, (0, 0, 1), ry0 + 0.012, rx0 + 0.012, 0.0125, u_hint=(1, 0, 0), n=32, ns=8)
        cuff.set_w("shin_" + sn, 1.0)
        tr.append(cuff)
        for z, sag, r_, a, tl in ((0.215, 0.012, 0.0070, 1.2, 0.012), (0.175, 0.016, 0.0080, 1.35, -0.010)):
            creases.append(O.fold_arc(ch, sn, side, z, 0.0150, -a, a, r=r_, sag=sag, tilt=tl))
        for z, sag, r_, a, tl in ((0.545, 0.008, 0.0070, 1.1, 0.008),):
            creases.append(O.fold_arc(ch, sn, side, z, 0.0150, -a, a, r=r_, sag=sag, tilt=tl))
    trs = G.merge(tr)
    trs.closed = False
    ch.add(trs, "trousers", ["jacket"], jig=True)
    ch.add(G.merge(creases), "creases", ["jacket"])

    # ---------------------------------------------------------------- jacket: open-front shell with V, hem, sleeves
    jk = O.jacket_shell(ch, 0.026, 0.78, Z_NECK, hw, n=60, dz=0.02, flare=lambda z: 0.016 * float(G.smoothstep(0.98, 0.78, z)))
    sleeves = []
    ar = spec["arm_r"]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.12
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), ch.ARM_UA[0] * ar + 0.014, ch.ARM_UA[1] * ar + 0.014, u=(0, 1, 0), nseg=26)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), ch.ARM_FA[0] * ar + 0.014, ch.ARM_FA[1] * ar + 0.014, u=(0, 1, 0), nseg=26, cap1=None)
        fa.set_w("forearm_" + sn, 1.0)
        d = (end - el) / np.linalg.norm(end - el)
        hem = O.ring_tube(end, d, ch.ARM_FA[0][2] * ar + 0.014, ch.ARM_FA[0][2] * ar + 0.014, 0.0095, u_hint=(0, 1, 0))
        hem.set_w("forearm_" + sn, 1.0)
        sleeves += [ua, fa, hem]
    # hem bead along the jacket bottom edge and the front opening edge
    zb = 0.78
    rx, ry, cy = B.torso_at(spec, np.array([zb]), 0.026)
    th = np.linspace(0, 2 * math.pi, 56, endpoint=False)
    hem_ring = G.tube(np.array([[(rx[0] + 0.018) * math.cos(t), cy[0] + (ry[0] + 0.018) * math.sin(t), zb] for t in th]), 0.0095, 0.0095, u=(0, 0, 1), nseg=8, closed_path=True)
    ch.torso_weights(hem_ring)
    jacket = G.merge([jk, hem_ring] + sleeves)
    jacket.closed = False
    ch.add(jacket, "jacket", ["jacket"], solid=0.008, jig=True)
    # lapels: ribbons along the V edge, widest at mid height, peaked at the neck
    lap = []
    for side in (1, -1):
        zs = np.linspace(Z_NECK - 0.004, Z_BTN + 0.012, 14)
        pts, rr, rt = [], [], []
        for z in zs:
            t = (Z_NECK - z) / (Z_NECK - Z_BTN)
            w = 0.026 + 0.040 * math.sin(math.pi * min(t * 1.08, 1.0)) ** 0.9
            xe = float(hw(z))
            pts.append(surf(spec, side * (xe + w * 0.50), z, True, 0.026 + 0.010))
            rr.append(w * 0.5)
            rt.append(0.0085)
        lp = G.tube(np.array(pts), np.array(rr), np.array(rt), u=(1, -0.12, 0), nseg=10, cap0="round", cap1="round", ncap=2)
        ch.torso_weights(lp)
        lap.append(lp)
    ch.add(G.merge(lap), "lapels", ["jacket"], jig=True, bevel=0.0015)
    # jacket collar: stand-up ribbon around the back of the neck, joined to the lapel tops
    ang = np.linspace(-2.55, 2.55, 22)
    rxc, ryc, cyc = B.torso_at(spec, np.array([Z_NECK]), 0.0)
    cpath = np.array([[(0.100 + 0.0) * math.sin(a), cyc[0] + 0.012 + 0.092 * math.cos(a), Z_NECK + 0.012] for a in ang])
    col = G.tube(cpath, 0.030, 0.0105, u=(0, 0, 1), nseg=10, cap0="round", cap1="round", ncap=2)
    col.set_w("chest", 0.5)
    col.add_w("neck", 0.5)
    ch.add(col, "collar", ["jacket"])
    # jacket details: buttons, hip pocket flaps, breast pocket with a pocket square
    det, btn = [], []
    for z in (1.02, 0.90):
        pp = surf(spec, 0.0, z, True, 0.0275)
        btn.append(G.ellipsoid(pp + np.array([0, 0.0, 0]), (0.0115, 0.0055, 0.0115), nseg=12, nrings=8))
    for side in (1, -1):
        pp = surf(spec, side * 0.115, 0.865, True, 0.026)
        det.append(G.ellipsoid(pp, (0.050, 0.0085, 0.0155), nseg=14, nrings=8, e=0.5))
    pp = surf(spec, 0.095, 1.185, True, 0.026)
    det.append(G.ellipsoid(pp, (0.030, 0.0075, 0.0095), nseg=12, nrings=8, e=0.5))
    for q in det + btn:
        ch.torso_weights(q)
    ch.add(G.merge(det), "jacket_details", ["jacket"], jig=True)
    ch.add(G.merge(btn), "jacket_buttons", ["shoe"])
    sq = G.ellipsoid(surf(spec, 0.095, 1.205, True, 0.0295), (0.020, 0.0055, 0.014), nseg=10, nrings=6, e=0.6)
    ch.torso_weights(sq)
    ch.add(sq, "pocket_square", ["shirt"])

    # ---------------------------------------------------------------- tie: knot + wide blade on a jiggle chain
    zs = np.linspace(1.345, 1.020, 16)
    ty = [surf(spec, 0.0, z, True, 0.019) for z in zs]
    wid = np.interp(zs, [1.020, 1.06, 1.13, 1.22, 1.30, 1.345], [0.014, 0.046, 0.056, 0.044, 0.026, 0.016])  # xp must ascend
    blade = G.tube(np.array(ty), wid, np.full(len(zs), 0.0085), u=(1, 0, 0), nseg=12, cap0="flat", cap1="flat")
    knot = G.ellipsoid(surf(spec, 0.0, 1.352, True, 0.024), (0.028, 0.018, 0.024), nseg=16, nrings=10)
    for q in (blade, knot):
        zq = q.v[:, 2]
        w = B.lerp_w(zq, [1.352, 1.30, 1.20, 1.10], ["chest", "tie_1", "tie_2", "tie_3"])
        q.only_w(w)
    ch.add(G.merge([blade, knot]), "tie", ["tie"])

    # ---------------------------------------------------------------- shoes (thick, with soles)
    for side, sn in ((1, "L"), (-1, "R")):
        sh_ = O.boot(ch, sn, side, shaft_h=0.05, scale=1.12, toe_len=0.25)
        ch.add(G.merge([sh_[0], sh_[2], sh_[3]]), "shoe_" + sn, ["shoe"])
        ch.add(sh_[1], "sole_" + sn, ["sole"], bevel=0.002)

    # ---------------------------------------------------------------- hair (v1 head units then T): cap with a parted fringe and a quiff
    hc = O.hair_cap(ch, 1.650, off=0.007, front_z=1.718)
    q = G.tube(np.array([(-0.050, -0.050, 1.713), (-0.008, -0.072, 1.731), (0.038, -0.084, 1.731), (0.072, -0.072, 1.714)]),
               [0.012, 0.020, 0.019, 0.010], [0.012, 0.019, 0.018, 0.010], u=(0, 0, 1), nseg=12, ncap=2)
    q.set_w("head", 1.0)
    tmp_ = []
    for side in (1, -1):
        tt = G.ellipsoid(np.array([side * 0.066, -0.040, 1.700]), (0.026, 0.034, 0.030), nseg=12, nrings=8)
        tt.set_w("head", 1.0)
        tmp_.append(tt)
    q2 = G.ellipsoid(np.array([0.0, 0.052, 1.655]), (0.080, 0.046, 0.070), nseg=16, nrings=10)
    q2.set_w("head", 1.0)
    nape = O.hair_band(ch, 1.560, 1.665, 0.012, back_only=True, front_cut=0.034)
    head_part(ch, G.merge([hc, q, q2, nape] + tmp_), "hair", ["hair"])

    # ---------------------------------------------------------------- hooks (v1 names)
    ear_x = 0.178
    ch.hook("steam_L", "head", (ear_x, 0.0, 1.526), rot=(0, math.pi / 2, 0))
    ch.hook("steam_R", "head", (-ear_x, 0.0, 1.526), rot=(0, -math.pi / 2, 0))
    O.hook_hand(ch, "L", 1)
    O.hook_hand(ch, "R", -1)
    for nm, sn, side in (("gun_grip_R", "R", -1), ("gun_support_L", "L", 1), ("whip_grip_L", "L", 1), ("paper_R", "R", -1)):
        f, n, c = ch.hand_frames[sn]
        wr = ch.J["wr"] * np.array([side, 1, 1])
        pos = wr + 0.060 * f + n * 0.020
        ch.hook(nm, "hand_" + sn, pos, rotm=np.stack([np.cross(f, n), f, n], axis=1))
    ch.hook("head_top", "head", (0.0, 0.005, 1.78))
    ch.root["p_stature_m"] = 1.85
    ch.root["p_scale"] = 1.0
    ch.root["pv_face_z"] = float(K * (spec["chin_post"] + 0.45 * 0.233 * spec["headS"][2]))
    ch.root["pv_face_dist"] = float(3.0 * K * 0.233 * spec["headS"][2])
    return ch


if __name__ == "__main__":
    ch = build("manager_v2")
    m = B.meta_common(ch)
    m.update(description="Manager v2 (chunky clay): suit jacket with lapels and collar, white shirt with collar points and cuffs, red tie with a swinging blade (tie_1..3), trousers with a break over thick shoes. Same rig, hooks, properties, shape keys as manager (v1).",
             sources=[dict(what="Joint table and K identical to v1", used_for="skeleton", access="2026-10-02", level="C"),
                      dict(what="Design values for the cartoon look (head about 1/4.5 of stature, thicker limbs)", used_for="v2 proportions", access="2026-10-02", level="C")])
    m = MV.extend(m, ch, "manager", [
        "Manager extras: bones tie_1..3 (tie blade chain, driven by p_tie_swing) and belly. HOOK_steam_L/R now sit at the ears of the bigger head (+-0.178 m baseline from the head axis, ear height); v1 had them at 0.092 m. HOOK_gun_grip_R, HOOK_gun_support_L, HOOK_whip_grip_L, HOOK_paper_R and HOOK_hand_L/R are unchanged (same positions and frames as v1).",
        "p_anger drives brows, lids, snarl, yell mouth, nostril flare and forehead veins; veins live on the forehead strip between the brow ridge and the hairline; p_flush mixes the red flush into the skin above the neck."])
    if not NO_ACTIONS:
        names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, "manager_v2")
        m["actions"] = A.describe(names)
    bpy.context.view_layer.update()
    C.finish("manager_v2", os.path.join(OUT, "manager_v2.blend"), ch.coll, m, preview_dir=None)
