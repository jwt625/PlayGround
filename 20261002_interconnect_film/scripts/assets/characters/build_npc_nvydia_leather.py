"""NVYDIA customer in a black clay leather (moto-style) jacket: NEW asset npc_nvydia_leather (npc_nvydia.blend is untouched).

Run: Blender -b --python build_npc_nvydia_leather.py -- <out_dir>
Same body, rig, hooks, holes, expressions and actions as npc_nvydia (same build code), plus a jacket: closed torso shell with
hem band, long sleeves with cuffs, standing collar with two snapped lapels, a metal front zipper with a pull tab and two
slanted hip-pocket zips. The jacket meshes are weighted at build time with the same bone weights as the shirt / suit shells of
the other characters (torso: hips/spine_1/spine_2/chest blend; sleeves: upper_arm/forearm), i.e. it deforms with the rig.
Stylisation only: generic black moto jacket, no logo, no likeness of a real person. The shirt under it is plain dark grey (no wordmark).
Action names are ACT_npc_nvydia_leather_<name> (same set as npc_nvydia: asm.play resolves them from root['asset_id']).
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

ARGS = C.argv_after_dashes()
OUT = ARGS[0]
CID = "npc_nvydia_leather"
V = dict(H=1.82, build=1.10, belly=0.15, skin="#f0c4a0", hair="buzz", hair_col="#8a8a8a", shirt="#2a2c33", jacket="#101012",
         trousers="#1d1d22", shoe="#0f0f10")


def build():
    v = V
    spec = B.default_spec()
    spec["H"], spec["build"], spec["belly"] = v["H"], v["build"], v["belly"]
    spec["head"]["nose_s"] = 1.0
    spec["head"]["ear_s"] = 1.0
    ch = B.Character(CID, spec)
    ch.root_props()
    holes = [dict(name="hole_1", prop="p_hole_1_radius", bone="chest", pos=(0.04, 0.0, 1.20), axis="Y", radius=0.042, half_len=0.14, default=0.0)]
    hg = ch.setup_holes(holes)
    pre = "MAT_characters_%s_" % CID
    O.face_mats(ch, v["skin"], brow_hex=v["hair_col"], hole_group=hg)
    ch.mat("shirt", M.clay(pre + "shirt", v["shirt"], rough=0.72, bump=0.3, holes=hg, sheen=0.2))
    ch.mat("leather", M.gloss(pre + "leather", v["jacket"], rough=0.38, spec=0.7, holes=hg, coat=0.2))
    ch.mat("zip", M.metal(pre + "zip", "#b9bcc2", 0.28, holes=hg))
    ch.mat("trousers", M.clay(pre + "trousers", v["trousers"], rough=0.75, bump=0.3, holes=hg))
    ch.mat("shoe", M.clay(pre + "shoes", v["shoe"], rough=0.55, bump=0.25, holes=hg))
    ch.mat("sole", M.clay(pre + "sole", "#e9e6dc", rough=0.6, bump=0.1, holes=hg))
    ch.mat("hair", M.clay(pre + "hair", v["hair_col"], rough=0.7, bump=0.3, holes=hg))
    J = ch.J
    ch.add(ch.skin_parts(), "body", ["skin"])
    ch.build_face()
    # dark shirt (crew collar, short sleeves) under the jacket, no logo
    tp = ch.torso_part(off=0.008, z0=0.93, z1=1.456)
    parts = [tp]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el = J["sh"] * sx, J["el"] * sx
        e2 = sh + (el - sh) * 0.5
        sl = G.tube(np.array([sh, (sh + e2) / 2, e2]), np.array([0.047, 0.045, 0.043]) + 0.010, u=(0, 1, 0), nseg=18, cap0="round", cap1=None)
        sl.set_w("upper_arm_" + sn, 1.0)
        parts.append(sl)
    th = np.linspace(0, 2 * math.pi, 28, endpoint=False)
    col = G.tube(np.array([[0.056 * math.cos(t), 0.006 + 0.058 * math.sin(t), 1.452] for t in th]), 0.009, 0.007, u=(0, 0, 1), nseg=8, closed_path=True)
    col.set_w("neck", 1.0)
    parts.append(col)
    shirt = G.merge(parts)
    shirt.closed = False
    ch.add(shirt, "shirt", ["shirt"])

    # ---- leather jacket
    JOFF = 0.024
    jk = ch.torso_part(off=JOFF, z0=0.88, z1=1.458)
    jk.closed = False
    sleeves, cuffs = [], []
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.10
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), np.array([0.047, 0.043, 0.039]) * spec["arm_r"] + 0.014, u=(0, 1, 0), nseg=20)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), np.array([0.038, 0.034, 0.030]) * spec["arm_r"] + 0.014, u=(0, 1, 0), nseg=20, cap1="flat")
        fa.set_w("forearm_" + sn, 1.0)
        sleeves += [ua, fa]
        c0 = end - (end - el) * 0.16
        cf = G.tube(np.array([c0, (c0 + end) / 2, end + (end - el) * 0.02]), np.array([0.032, 0.031, 0.030]) * spec["arm_r"] + 0.019, u=(0, 1, 0), nseg=20, cap0=None, cap1="flat")
        cf.set_w("forearm_" + sn, 1.0)
        cuffs.append(cf)
    jacket = G.merge([jk] + sleeves)
    jacket.closed = False
    ch.add(jacket, "jacket", ["leather"], solid=0.005)
    # hem band and cuffs (ribbed look: slightly thicker rings)
    hem = ch.torso_part(off=JOFF + 0.006, z0=0.88, z1=0.935)
    hem.closed = False
    ch.add(G.merge([hem] + cuffs), "jacket_bands", ["leather"], solid=0.006)
    # standing collar + two snapped lapels
    th = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    ring = G.tube(np.array([[0.068 * math.cos(t), 0.007 + 0.072 * math.sin(t), 1.478] for t in th]), 0.024, 0.011, u=(0, 0, 1), nseg=10, closed_path=True)
    ring.set_w("chest", 1.0)
    lap = []
    for side in (1, -1):
        pts = [O.surf_pt(spec, side * x, z, True, JOFF + 0.014) for x, z in ((0.050, 1.452), (0.075, 1.405), (0.098, 1.335), (0.108, 1.275))]
        lp = G.tube(np.array(pts), np.array([0.026, 0.030, 0.026, 0.016]), 0.006, u=(1, 0, 0), nseg=10, cap0="round", cap1="round")
        lp.set_w("chest", 0.6)
        lp.add_w("spine_2", 0.4)
        lap.append(lp)
    ch.add(G.merge([ring] + lap), "jacket_collar", ["leather"])
    # front zipper (metal strip + pull tab) and two slanted hip-pocket zips
    zs = np.linspace(0.90, 1.47, 12)
    zp = G.tube(np.array([O.surf_pt(spec, 0.0, z, True, JOFF + 0.004) for z in zs]), 0.0045, 0.0022, u=(1, 0, 0), nseg=8, cap0="flat", cap1="flat")
    ch.torso_weights(zp)
    pull = G.ellipsoid(O.surf_pt(spec, 0.0, 1.425, True, JOFF + 0.011), (0.007, 0.0035, 0.017), nseg=12, nrings=8)
    pull.set_w("chest", 1.0)
    pz = []
    for side in (1, -1):
        pp = np.array([O.surf_pt(spec, side * x, z, True, JOFF + 0.003) for x, z in ((0.075, 1.04), (0.125, 1.00), (0.170, 0.97))])
        q = G.tube(pp, 0.0035, 0.002, u=(0, 0, 1), nseg=8, cap0="round", cap1="round")
        q.set_w("spine_1", 0.5)
        q.add_w("hips", 0.5)
        pz.append(q)
    ch.add(G.merge([zp, pull] + pz), "jacket_zip", ["zip"])

    # trousers, shoes, hair, hooks: identical to npc_nvydia
    tr = [ch.torso_part(off=0.013, z0=0.84, z1=1.02)]
    for side, sn in ((1, "L"), (-1, "R")):
        tr += O.leg_tubes(ch, sn, side, 0.010, ztop_shin=0.095, cuff=0.003)
    ch.add(G.merge(tr), "trousers", ["trousers"])
    for side, sn in ((1, "L"), (-1, "R")):
        sh_ = O.boot(ch, sn, side, shaft_h=0.03)
        ch.add(G.merge([sh_[0], sh_[2], sh_[3]]), "shoe_" + sn, ["shoe"])
        ch.add(sh_[1], "sole_" + sn, ["sole"])
    hp = O.hair_cap(ch, 1.64, off=0.003, front_z=1.69)
    ch.add(hp, "hair", ["hair"])
    O.hook_hand(ch, "L", 1)
    O.hook_hand(ch, "R", -1)
    for nm, sn, side in (("gun_grip_R", "R", -1), ("gun_support_L", "L", 1)):
        f, n, c = ch.hand_frames[sn]
        wr = ch.J["wr"] * np.array([side, 1, 1])
        ch.hook(nm, "hand_" + sn, wr + 0.060 * f + n * 0.020, rotm=np.stack([np.cross(f, n), f, n], axis=1))
    ch.hook("head_top", "head", (0.0, 0.005, 1.76))
    ch.hook("hug_target", "root", (0.0, -0.30, 0.65))
    ch.build_rims(ch.m["rim"])
    ch.root["p_stature_m"] = ch.spec["H"]
    ch.root["p_scale"] = 1.0
    return ch


if __name__ == "__main__":
    ch = build()
    names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, CID)
    meta = B.meta_common(ch)
    meta.update(description="NVYDIA customer variant wearing a black clay leather moto-style jacket (zipper, hem band, cuffs, standing collar with lapels, hip-pocket zips), no logo; "
                            "same body, rig, hooks, holes, expressions and action set as npc_nvydia (actions named ACT_npc_nvydia_leather_<name>)",
                variant=V, actions=A.describe(names),
                jacket_notes=["Jacket meshes npc_nvydia_leather_jacket / _jacket_bands / _jacket_collar / _jacket_zip are skinned to the same armature with build-time vertex weights (torso: hips, spine_1, spine_2, chest; sleeves: upper_arm, forearm); materials MAT_characters_npc_nvydia_leather_leather (glossy black, coat 0.2) and _zip (metal) carry the bullet-hole cutout group",
                              "Swap procedure for a scene: asm.append('characters/npc_nvydia_leather', actions=True) instead of 'characters/npc_nvydia'; every property, hook and action keeps its name apart from the asset id inside action names (use asm.play(asset, 'slap') which resolves ACT_<asset_id>_<name>)",
                              "Stylisation only: generic black moto jacket, no logo, no likeness of a real person; accuracy level C"])
    bpy.context.view_layer.update()
    C.finish(CID, os.path.join(OUT, CID + ".blend"), ch.coll, meta, preview_dir=None)
