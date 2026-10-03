"""Gary: clay hardware technician. Run: Blender -b --python build_gary.py -- <out_dir>"""
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
import common as C
import chars_actions as A

OUT = C.argv_after_dashes()[0]

GARY_HOLES = [
    dict(name="hole_1", prop="p_hole_1_radius", bone="chest", pos=(0.07, 0.0, 1.24), axis="Y", radius=0.042, half_len=0.14),
    dict(name="hole_2", prop="p_hole_2_radius", bone="spine_2", pos=(-0.08, 0.0, 1.12), axis="Y", radius=0.042, half_len=0.14),
    dict(name="hole_3", prop="p_hole_3_radius", bone="spine_1", pos=(0.10, 0.0, 1.05), axis="Y", radius=0.042, half_len=0.14),
    dict(name="hole_4", prop="p_hole_4_radius", bone="chest", pos=(-0.05, 0.0, 1.32), axis="Y", radius=0.042, half_len=0.14),
    dict(name="hole_5", prop="p_hole_5_radius", bone="spine_1", pos=(-0.12, 0.0, 1.04), axis="Y", radius=0.042, half_len=0.14),
    dict(name="headhole", prop="p_head_hole_radius", bone="head", pos=(0.0, 0.017, 1.667), axis="X", radius=0.032, half_len=0.14),
]


def build(cid, hole_default):
    spec = B.default_spec()
    spec["H"] = 1.69 * 1.0
    spec["build"] = 1.0
    spec["belly"] = 0.35
    ch = B.Character(cid, spec)
    K = ch.K
    ch.root_props()
    for h in GARY_HOLES:
        h["default"] = hole_default
    hg = ch.setup_holes(GARY_HOLES)
    pre = "MAT_characters_%s_" % cid
    O.face_mats(ch, "#e2a57a", brow_hex="#3a2412", hole_group=hg)
    ch.mat("hat", M.clay(pre + "hardhat", "#ff7a12", rough=0.45, bump=0.18, holes=hg))
    ch.mat("overalls", M.clay(pre + "overalls", "#2c5db5", rough=0.7, bump=0.3, holes=hg, sheen=0.2))
    ch.mat("shirt", M.clay(pre + "shirt", "#d6cfbd", rough=0.72, bump=0.3, holes=hg, sheen=0.2))
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
    # bare skin body
    ch.add(ch.skin_parts(), "body", ["skin"])
    ch.build_face()
    # t-shirt: torso + short sleeves + collar
    tp = ch.torso_part(off=0.006, z0=0.90, z1=1.455)
    tp.v  # open bottom
    parts = [tp]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el = J["sh"] * sx, J["el"] * sx
        e2 = sh + (el - sh) * 0.46
        sl = G.tube(np.array([sh, (sh + e2) / 2, e2]), [0.055, 0.052, 0.050], u=(0, 1, 0), nseg=18, cap0="round", cap1=None)
        sl.set_w("upper_arm_" + sn, 1.0)
        parts.append(sl)
    shirt = G.merge(parts)
    shirt.closed = False
    ch.add(shirt, "shirt", ["shirt"])
    # overalls: trousers (waist to ankle), bib, straps
    ov = ch.torso_part(off=0.013, z0=0.84, z1=1.20)
    ov.set_mat(0)
    # open top: replace the flat cap by dropping nothing (cap hidden under the bib/shirt)
    parts = [ov]
    for side, sn in ((1, "L"), (-1, "R")):
        parts += O.leg_tubes(ch, sn, side, 0.011, ztop_shin=0.135, cuff=0.006)
    ch.add(G.merge(parts), "overalls", ["overalls"])
    bib = ch.torso_part(off=0.0135, z0=1.17, z1=1.345)
    bib.drop_faces(lambda c: (c[:, 1] > -0.02) | (np.abs(c[:, 0]) > 0.115))
    bib.closed = False
    ch.add(bib, "bib", ["overalls"], solid=0.006)
    straps = []
    for side in (1, -1):
        pts = [O.surf_pt(spec, side * 0.075, 1.34, True, 0.016), O.surf_pt(spec, side * 0.080, 1.40, True, 0.016),
               np.array([side * 0.088, 0.003, 1.455]), np.array([side * 0.086, 0.060, 1.43]), O.surf_pt(spec, side * 0.080, 1.30, False, 0.016),
               O.surf_pt(spec, side * 0.075, 1.22, False, 0.016)]
        st = G.tube(np.array(pts), 0.017, 0.004, u=(1, 0, 0), nseg=8, cap0="flat", cap1="flat")
        ch_ = st
        ch_.set_w("chest", 1.0)
        straps.append(ch_)
        btn = G.ellipsoid(O.surf_pt(spec, side * 0.075, 1.33, True, 0.021), (0.011, 0.005, 0.011), nseg=12, nrings=8)
        btn.set_w("chest", 1.0)
        straps.append(btn)
    ch.add(G.merge(straps[0::2]), "straps", ["overalls"])
    ch.add(G.merge(straps[1::2]), "buttons", ["brass"])
    # belt and tools
    zb = 1.00
    th = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    rx, ry, cy = B.torso_at(spec, np.array([zb]), 0.0)
    path = np.array([[(rx[0] + 0.020) * math.cos(t), cy[0] + (ry[0] + 0.020) * math.sin(t), zb] for t in th])
    belt = G.tube(path, 0.017, 0.0065, u=(0, 0, 1), nseg=8, closed_path=True)
    belt.set_w("spine_1", 0.5)
    belt.add_w("hips", 0.5)
    ch.add(belt, "belt", ["belt"])
    buckle = G.ellipsoid(np.array([0.0, cy[0] - (ry[0] + 0.0255), zb]), (0.022, 0.004, 0.017), nseg=14, nrings=8)
    buckle.set_w("hips", 1.0)
    ch.add(buckle, "buckle", ["brass"])
    pouches = []
    for (x, y, z, sx_, sy_, sz_) in [(-0.172, -0.01, 0.945, 0.020, 0.040, 0.050), (0.150, 0.085, 0.95, 0.040, 0.020, 0.045),
                                     (-0.07, 0.115, 0.95, 0.045, 0.020, 0.045)]:
        pp = G.ellipsoid(np.array([x, y, z]), (sx_, sy_, sz_), nseg=14, nrings=8, e=0.55)
        pp.set_w("hips", 1.0)
        pouches.append(pp)
    ch.add(G.merge(pouches), "pouches", ["belt"])
    tools = []
    for dx in (-0.012, 0.012):
        hnd = G.capsule(np.array([-0.186 + dx * 0.4, -0.03 + dx, 0.935]), np.array([-0.190, -0.05 + dx * 2, 0.80]), 0.007, 0.0075)
        hnd.set_w("hips", 1.0)
        tools.append(hnd)
    ch.add(G.merge(tools), "pliers_handles", ["tool_red"])
    jaw = G.capsule(np.array([-0.186, -0.02, 0.972]), np.array([-0.186, -0.02, 1.015]), 0.0055, 0.0045)
    jaw.set_w("hips", 1.0)
    ch.add(jaw, "pliers_jaw", ["metal"])
    ring = G.tube(np.array([[0.150 + 0.05 * math.cos(t), 0.085, 0.95 + 0.001 + 0.05 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 18, endpoint=False)]), 0.007, 0.007, u=(0, 1, 0), nseg=8, closed_path=True)
    ring.set_w("hips", 1.0)
    ch.add(ring, "cable_tie_roll", ["tool_blue"])
    # boots
    for side, sn in ((1, "L"), (-1, "R")):
        bt = O.boot(ch, sn, side)
        ch.add(G.merge([bt[0], bt[2], bt[3]]), "boot_" + sn, ["boot"])
        ch.add(bt[1], "sole_" + sn, ["sole"])
    # hair under the hat and hard hat
    ch.add(O.hair_band(ch, 1.58, 1.70, 0.006), "hair", ["hair"], subsurf=0)
    zb_ = 1.682
    prof = []
    for k in range(0, 13):
        a = (math.pi / 2) * k / 12
        prof.append((0.094 * math.cos(a), zb_ + 0.128 * math.sin(a)))
    prof = [(max(r, 0.0), z) for r, z in prof]
    prof = [(0.094, zb_ - 0.004)] + prof
    dome = G.lathe(prof, center=(0, 0.005, 0), axis=(0, 0, 1), nseg=32)
    dome.v[:, 1] = 0.005 + (dome.v[:, 1] - 0.005) * 1.20
    dome.set_w("head", 1.0)
    brim_prof = [(0.088, zb_ - 0.004), (0.118, zb_ - 0.007), (0.126, zb_ - 0.001), (0.118, zb_ + 0.005), (0.092, zb_ + 0.006)]
    brim = G.lathe(brim_prof, center=(0, 0.005, 0), axis=(0, 0, 1), nseg=32)
    d = brim.v[:, 1] - 0.005
    brim.v[:, 1] = 0.005 + np.where(d < 0, d * 1.75, d * 1.15)
    brim.set_w("head", 1.0)
    ridge_path = np.array([[0.0, 0.005 + 1.2 * 0.094 * math.cos(a) * 0.9, zb_ + 0.128 * math.sin(a) + 0.006] for a in np.linspace(-1.35, 1.35, 16)])
    ridge_path[:, 1] = 0.005 + ridge_path[:, 1] * 0 + 0.108 * np.sin(np.linspace(-1.35, 1.35, 16)) * 1.0
    ridge_path[:, 2] = zb_ + 0.128 * np.cos(np.linspace(-1.35, 1.35, 16)) + 0.002
    ridge = G.tube(ridge_path, 0.011, 0.010, u=(1, 0, 0), nseg=10)
    ridge.set_w("head", 1.0)
    ch.add(G.merge([dome, brim, ridge]), "hardhat", ["hat"])
    # hooks
    ch.hook("head_top", "head", (0.0, 0.005, 1.815))
    ch.hook("hip", "hips", (-0.19, -0.0, 0.95))
    ch.hook("muzzle_self", "head", (-0.21, 0.017, 1.667), rotm=np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]) * 0 + np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]))
    O.hook_hand(ch, "L", 1)
    O.hook_hand(ch, "R", -1)
    ch.build_rims(ch.m["rim"])
    ch.root['p_stature_m'] = 1.75
    ch.root['p_scale'] = 1.0
    return ch


def finalize(ch, cid):
    meta = B.meta_common(ch)
    meta.update(dict(
        description="Gary, clay hardware technician (orange hard hat, blue overalls, tool belt, boots, bare hands).",
        sources=[dict(what="Body proportions: Drillis and Contini segment fractions of stature (limb lengths, joint heights)",
                      used_for="skeleton and limb lengths", access="2026-10-02 (from memory of the standard table; verify)", level="C")],
        dimensions=[dict(item="stature without hat (m)", value=round(ch.spec["H"], 3), provenance="design: Gary about 1.75 m with hat", accuracy="C"),
                    dict(item="stature with hard hat", value=None, provenance="from bbox", accuracy="C")],
    ))
    return meta


if __name__ == "__main__":
    for cid, hd in (("gary", 0.0), ("gary_holes30", 0.3)):
        ch = build(cid, hd)
        names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, cid)
        meta = finalize(ch, cid)
        meta['actions'] = A.describe(names)
        bpy.context.view_layer.update()
        C.finish(cid, os.path.join(OUT, cid + ".blend"), ch.coll, meta, preview_dir=None)
