"""Manager: clay manager (suit, white shirt, red tie, shoes). Run: Blender -b --python build_manager.py -- <out_dir>"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bpy
import chars_geo as G
import chars_mat as M
import chars_body as B
import chars_outfit as O
import chars_actions as A
import common as C

OUT = C.argv_after_dashes()[0]


def build(cid="manager"):
    spec = B.default_spec()
    spec["H"] = 1.85
    spec["build"] = 1.10
    spec["belly"] = 0.30
    spec["shoulders"] = 1.08
    spec["arm_r"] = 1.06
    spec["leg_r"] = 1.05
    spec["head"]["w0"] = 0.081
    spec["head"]["jaw"] = 0.08
    ch = B.Character(cid, spec)
    ch.root_props()
    pre = "MAT_characters_%s_" % cid
    O.face_mats(ch, "#e89b86", brow_hex="#3d2616", vein=(0, 0.075, -0.06, 1.625, 1.705), flush_color="#e0301f")
    ch.mat("jacket", M.clay(pre + "suit", "#2a2f4a", rough=0.75, bump=0.3, sheen=0.25))
    ch.mat("shirt", M.clay(pre + "shirt", "#eeeae0", rough=0.7, bump=0.28, sheen=0.2))
    ch.mat("tie", M.clay(pre + "tie", "#d4121c", rough=0.5, bump=0.22, sheen=0.3))
    ch.mat("shoe", M.gloss(pre + "shoes", "#161618", rough=0.3, spec=0.6))
    ch.mat("sole", M.clay(pre + "sole", "#0c0c0d", rough=0.7, bump=0.0))
    ch.mat("hair", M.clay(pre + "hair", "#4a2f1c", rough=0.7, bump=0.3))
    ch.mat("brass", M.metal(pre + "brass", "#d9b04a", 0.3))
    spec = ch.spec
    J = ch.J
    ch.add(ch.skin_parts(), "body", ["skin"])
    ch.build_face()
    # white shirt: torso + sleeves to the wrist + collar
    tp = ch.torso_part(off=0.007, z0=0.90, z1=1.458)
    parts = [tp]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), np.array([0.047, 0.043, 0.039]) * spec["arm_r"] + 0.007, u=(0, 1, 0), nseg=18)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + wr) / 2, wr - (wr - el) * 0.03]), np.array([0.038, 0.034, 0.029]) * spec["arm_r"] + 0.007, u=(0, 1, 0), nseg=18, cap1="flat")
        fa.set_w("forearm_" + sn, 1.0)
        parts += [ua, fa]
    th = np.linspace(0, 2 * math.pi, 28, endpoint=False)
    col = G.tube(np.array([[0.056 * math.cos(t), 0.006 + 0.058 * math.sin(t), 1.455] for t in th]), 0.009, 0.006, u=(0, 0, 1), nseg=8, closed_path=True)
    col.set_w("neck", 1.0)
    parts.append(col)
    shirt = G.merge(parts)
    shirt.closed = False
    ch.add(shirt, "shirt", ["shirt"])
    # trousers
    tr = [ch.torso_part(off=0.012, z0=0.84, z1=1.06)]
    for side, sn in ((1, "L"), (-1, "R")):
        tr += O.leg_tubes(ch, sn, side, 0.010, ztop_shin=0.10, cuff=0.004)
    ch.add(G.merge(tr), "trousers", ["jacket"])
    # jacket with V opening
    jk = ch.torso_part(off=0.020, z0=0.80, z1=1.46)
    jk.closed = False
    jk.fm = [0] * len(jk.f)

    def vcut(c):
        z = c[:, 2]
        half = np.clip(0.085 * (1.43 - z) / 0.30, 0.0, 0.085) + 0.012
        return (c[:, 1] < -0.02) & (np.abs(c[:, 0]) < half) & (z > 1.09)
    jk.drop_faces(vcut)
    sleeves = []
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.10
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), np.array([0.047, 0.043, 0.039]) * spec["arm_r"] + 0.016, u=(0, 1, 0), nseg=18)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), np.array([0.038, 0.034, 0.030]) * spec["arm_r"] + 0.016, u=(0, 1, 0), nseg=18, cap1="flat")
        fa.set_w("forearm_" + sn, 1.0)
        sleeves += [ua, fa]
    jacket = G.merge([jk] + sleeves)
    jacket.closed = False
    ch.add(jacket, "jacket", ["jacket"], solid=0.005)
    # lapels: two flat wedges along the V edge
    lap = []
    for side in (1, -1):
        pts = []
        for z in np.linspace(1.40, 1.12, 8):
            half = 0.085 * (1.43 - z) / 0.30 + 0.012
            pts.append(O.surf_pt(spec, side * (half + 0.012), z, True, 0.026))
        lp = G.tube(np.array(pts), 0.016, 0.005, u=(1, 0, 0), nseg=8, cap0="round", cap1="round")
        lp.set_w("chest", 0.6)
        lp.add_w("spine_2", 0.4)
        lap.append(lp)
    ch.add(G.merge(lap), "lapels", ["jacket"])
    # tie: knot + blade
    zs = np.linspace(1.425, 1.02, 10)
    pts = [O.surf_pt(spec, 0.0, z, True, 0.016) for z in zs]
    wd = np.linspace(0.016, 0.040, 10)
    wd[-1] = 0.03
    tie = G.tube(np.array(pts), wd, 0.004, u=(1, 0, 0), nseg=10, cap0="flat", cap1="flat")
    tie.set_w("chest", 0.5)
    tie.add_w("spine_2", 0.5)
    tie.add_w("spine_1", 0.0)
    knot = G.ellipsoid(O.surf_pt(spec, 0.0, 1.418, True, 0.020), (0.020, 0.012, 0.016), nseg=14, nrings=8)
    knot.set_w("chest", 1.0)
    ch.add(G.merge([tie, knot]), "tie", ["tie"])
    for side, sn in ((1, "L"), (-1, "R")):
        sh_ = O.boot(ch, sn, side, shaft_h=0.02)
        ch.add(G.merge([sh_[0], sh_[2], sh_[3]]), "shoe_" + sn, ["shoe"])
        ch.add(sh_[1], "sole_" + sn, ["sole"])
    # hair: cap with a side part and a quiff
    hc = O.hair_cap(ch, 1.655, off=0.008, front_z=1.700)
    q = G.ellipsoid(np.array([0.022, -0.062, 1.722]), (0.050, 0.034, 0.016), nseg=14, nrings=8)
    q.set_w("head", 1.0)
    ch.add(G.merge([hc, q]), "hair", ["hair"])
    # hooks
    ch.hook("steam_L", "head", (0.092, 0.017, 1.640), rot=(0, math.pi / 2, 0))
    ch.hook("steam_R", "head", (-0.092, 0.017, 1.640), rot=(0, -math.pi / 2, 0))
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
    return ch


if __name__ == "__main__":
    ch = build("manager")
    names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, "manager")
    meta = B.meta_common(ch)
    meta.update(description="Manager: clay suit, white shirt, red tie, shoes; p_anger drives brows, mouth, nostril flare and vein bump; p_flush the red flush",
                actions=A.describe(names))
    bpy.context.view_layer.update()
    C.finish("manager", os.path.join(OUT, "manager.blend"), ch.coll, meta, preview_dir=None)
