"""Generic person rig with parody-wordmark shirt variants.
Run: Blender -b --python build_npc.py -- <out_dir> <variant>   (variant: npc | npc_molexx | npc_nubiss | npc_terahop | npc_ayarr | npc_nvydia | npc_openay)
"""
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

ARGS = C.argv_after_dashes()
OUT = ARGS[0]
VAR = ARGS[1] if len(ARGS) > 1 else "npc"

VARIANTS = {
    "npc": dict(H=1.75, build=1.0, belly=0.1, skin="#d8a27c", hair="short", hair_col="#3b2a1c", shirt="#8d939c", logo=None, trousers="#3a3f4a", shoe="#d8d8d8"),
    "npc_molexx": dict(H=1.80, build=1.05, belly=0.1, skin="#d9a07a", hair="short", hair_col="#2a1c12", shirt="#d1222a", logo=["MOLEXX"], logo_col="#ffffff", trousers="#2c3140", shoe="#222226"),
    "npc_nubiss": dict(H=1.70, build=1.22, belly=0.45, skin="#c98f66", hair="bald", hair_col="#4a3322", shirt="#2f9e4f", logo=["NUBISS"], logo_col="#ffffff", trousers="#4b4f5a", shoe="#3a2a20", nose=1.15),
    "npc_terahop": dict(H=1.88, build=0.92, belly=0.0, skin="#e8b894", hair="long", hair_col="#6a4020", shirt="#6a3fb0", logo=["TERAHOP"], logo_col="#ffffff", trousers="#25303f", shoe="#1a1a1c", ear=0.9),
    "npc_ayarr": dict(H=1.62, build=0.95, belly=0.05, skin="#8a5a3c", hair="curly", hair_col="#15100c", shirt="#f2c230", logo=["AYARR"], logo_col="#111111", trousers="#3a3a42", shoe="#e8e8e8"),
    "npc_nvydia": dict(H=1.82, build=1.10, belly=0.15, skin="#f0c4a0", hair="buzz", hair_col="#8a8a8a", shirt="#76c21a", logo=["NVYDIA"], logo_col="#101010", trousers="#1d1d22", shoe="#0f0f10"),
    "npc_openay": dict(H=1.75 * 0.8, build=0.95, belly=0.0, skin="#e0b08a", hair="bob", hair_col="#2a2a30", shirt="#1c1f26", logo=["OPENAY", "ANTHROPY"], logo_col="#ffffff", trousers="#2a2d38", shoe="#c8c8c8"),
}


def build(cid):
    v = VARIANTS[cid]
    spec = B.default_spec()
    spec["H"] = v["H"]
    spec["build"] = v["build"]
    spec["belly"] = v["belly"]
    spec["head"]["nose_s"] = v.get("nose", 1.0)
    spec["head"]["ear_s"] = v.get("ear", 1.0)
    ch = B.Character(cid, spec)
    ch.root_props()
    holes = [dict(name="hole_1", prop="p_hole_1_radius", bone="chest", pos=(0.04, 0.0, 1.20), axis="Y", radius=0.042, half_len=0.14, default=0.0)]
    hg = ch.setup_holes(holes)
    pre = "MAT_characters_%s_" % cid
    O.face_mats(ch, v["skin"], brow_hex=v["hair_col"], hole_group=hg)
    ch.mat("shirt", M.clay(pre + "shirt", v["shirt"], rough=0.72, bump=0.3, holes=hg, sheen=0.2))
    ch.mat("logo", M.clay(pre + "logo", v.get("logo_col", "#ffffff"), rough=0.5, bump=0.1, holes=hg))
    ch.mat("trousers", M.clay(pre + "trousers", v["trousers"], rough=0.75, bump=0.3, holes=hg))
    ch.mat("shoe", M.clay(pre + "shoes", v["shoe"], rough=0.55, bump=0.25, holes=hg))
    ch.mat("sole", M.clay(pre + "sole", "#e9e6dc", rough=0.6, bump=0.1, holes=hg))
    ch.mat("hair", M.clay(pre + "hair", v["hair_col"], rough=0.7, bump=0.3, holes=hg))
    spec = ch.spec
    J = ch.J
    ch.add(ch.skin_parts(), "body", ["skin"])
    ch.build_face()
    # shirt: torso, short sleeves, crew collar
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
    if v["logo"]:
        lg = O.logo_part(ch, v["logo"], 1.215, 0.27 if len(v["logo"]) == 1 else 0.22, 0)
        ch.add(lg, "logo", ["logo"])
    tr = [ch.torso_part(off=0.013, z0=0.84, z1=1.02)]
    for side, sn in ((1, "L"), (-1, "R")):
        tr += O.leg_tubes(ch, sn, side, 0.010, ztop_shin=0.095, cuff=0.003)
    ch.add(G.merge(tr), "trousers", ["trousers"])
    for side, sn in ((1, "L"), (-1, "R")):
        sh_ = O.boot(ch, sn, side, shaft_h=0.03)
        ch.add(G.merge([sh_[0], sh_[2], sh_[3]]), "shoe_" + sn, ["shoe"])
        ch.add(sh_[1], "sole_" + sn, ["sole"])
    hs = v["hair"]
    if hs == "short":
        hp = O.hair_cap(ch, 1.655, off=0.007, front_z=1.70)
    elif hs == "buzz":
        hp = O.hair_cap(ch, 1.64, off=0.003, front_z=1.69)
    elif hs == "bald":
        hp = O.hair_band(ch, 1.56, 1.64, 0.005)
    elif hs == "long":
        hp = O.hair_long(ch)
    elif hs == "curly":
        hp = O.hair_curly(ch)
    else:
        hp = O.hair_bob(ch)
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
    ch = build(VAR)
    names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, VAR)
    meta = B.meta_common(ch)
    v = VARIANTS[VAR]
    meta.update(description="Generic clay person (%s): shirt colour %s, wordmark %s (parody), one body hole (hole_1)" % (VAR, v["shirt"], v["logo"]),
                variant=v, actions=A.describe(names))
    bpy.context.view_layer.update()
    C.finish(VAR, os.path.join(OUT, VAR + ".blend"), ch.coll, meta, preview_dir=None)
