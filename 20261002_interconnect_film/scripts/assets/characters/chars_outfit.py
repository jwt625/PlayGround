"""Shared outfit/material helpers for the character build scripts."""
import math

import numpy as np

import chars_geo as G
import chars_mat as M
import bpy
import chars_body as B


def face_mats(ch, skin_hex, brow_hex="#2a1a10", vein=None, hole_group=None, flush_color="#d8402f"):
    hg = hole_group
    pre = "MAT_characters_%s_" % ch.id
    zmin = 1.40 * ch.K
    if vein:
        vein = tuple(v * ch.K for v in vein)
        sk = M.skin(pre + "skin", skin_hex, ch.root, "p_flush", "p_anger", flush_color, holes=hg, vein_region=vein, flush_zmin=zmin)
    else:
        sk = M.skin(pre + "skin", skin_hex, ch.root, "p_flush", None, flush_color, holes=hg, flush_zmin=zmin)
    ch.mat("skin", sk)
    ch.mat("mouth", M.clay(pre + "mouth", "#5a1418", rough=0.5, bump=0.0))
    ch.mat("eye_white", M.gloss(pre + "eye_white", "#f4f1e6", rough=0.18, spec=0.5))
    ch.mat("pupil", M.gloss(pre + "pupil", "#08080a", rough=0.08, spec=0.8))
    ch.mat("brow", M.clay(pre + "brow", brow_hex, rough=0.7, bump=0.2, holes=hg))
    ch.mat("nostril", M.clay(pre + "nostril", "#3a1810", rough=0.6, bump=0.0))
    ch.mat("teeth", M.clay(pre + "teeth", "#f0ead8", rough=0.35, bump=0.0))
    ch.mat("tongue", M.clay(pre + "tongue", "#c8505a", rough=0.5, bump=0.0))
    ch.mat("drop", M.gloss(pre + "drop", "#8fd0f0", rough=0.1, spec=0.7))
    ch.mat("rim", M.clay(pre + "hole_rim", "#cfae8a", rough=0.7, bump=0.3))


# ------------------------------------------------------------------ clothing / accessories (baseline coords)
def surf_pt(spec, x, z, front=True, off=0.0):
    rx, ry, cy = B.torso_at(spec, np.array([z]), off)
    rx, ry, cy = rx[0], ry[0], cy[0]
    t = max(0.0, 1 - (x / rx) ** 2)
    return np.array([x, cy + (-1 if front else 1) * ry * math.sqrt(t), z])


def leg_tubes(ch, sn, side, off, ztop_shin=None, cuff=0.0, mat=0):
    """Offset leg tubes (thigh + shin) for trousers; shin may stop above the ankle (ztop_shin = z of the cut)."""
    spec, J = ch.spec, ch.J
    lr = spec["leg_r"]
    sx = np.array([side, 1, 1])
    hip, kn, an = J["hip"] * sx, J["knee"] * sx, J["ankle"] * sx
    th = G.tube(np.array([hip, hip + (kn - hip) * 0.5, kn]), (np.array([0.084, 0.072, 0.056]) * lr) + off, u=(0, 1, 0), nseg=20, mat=mat)
    th.set_w("thigh_" + sn, 1.0)
    end = an
    if ztop_shin is not None:
        t = (kn[2] - ztop_shin) / (kn[2] - an[2])
        end = kn + (an - kn) * t
    mid = kn + (end - kn) * 0.35
    rs = np.array([0.056, 0.059, 0.043 if ztop_shin else 0.036]) * lr + off
    sh = G.tube(np.array([kn, mid, end]), rs, u=(0, 1, 0), nseg=20, cap1="flat" if ztop_shin else "round", mat=mat)
    sh.set_w("shin_" + sn, 1.0)
    parts = [th, sh]
    if cuff:
        ring = G.tube(np.array([end + np.array([0, 0, 0.012]), end - np.array([0, 0, 0.012])]), [rs[-1] + cuff] * 2, u=(0, 1, 0), nseg=20, cap0="round", cap1="round", ncap=1, mat=mat)
        ring.set_w("shin_" + sn, 1.0)
        parts.append(ring)
    return parts


def boot(ch, sn, side, shaft_h=0.115, sole_mat=1, mat=0):
    J = ch.J
    sx = np.array([side, 1, 1])
    an = J["ankle"] * sx
    ax, ay = an[0], an[1]
    ys = ay + np.array([0.072, 0.0, -0.075, -0.145, -0.20])
    hw = np.array([0.040, 0.047, 0.050, 0.049, 0.038])
    hh = np.array([0.046, 0.043, 0.036, 0.031, 0.024])
    cz = np.array([0.050, 0.047, 0.040, 0.034, 0.027])
    path = np.stack([np.full(5, ax), ys, cz], axis=1)
    toe = G.tube(path, hw, hh, u=(1, 0, 0), nseg=20, mat=mat)
    toe.set_w("foot_" + sn, 1.0)
    cz2 = np.array([0.011] * 5)
    path2 = np.stack([np.full(5, ax), ys - np.array([0.0, 0.0, 0.0, 0.004, 0.006]), cz2], axis=1)
    sole = G.tube(path2, hw + 0.005, np.array([0.011] * 5), u=(1, 0, 0), nseg=20, mat=sole_mat)
    sole.set_w("foot_" + sn, 1.0)
    shaft = G.tube(np.array([[ax, ay + 0.008, 0.045], [ax, ay + 0.008, 0.045 + shaft_h]]), [0.052, 0.050], [0.050, 0.048], u=(1, 0, 0), nseg=20, cap0="flat", cap1="flat", mat=mat)
    shaft.set_w("foot_" + sn, 1.0)
    cuff = G.tube(np.array([[ax, ay + 0.008, 0.045 + shaft_h - 0.012], [ax, ay + 0.008, 0.045 + shaft_h + 0.010]]), [0.057, 0.057], [0.055, 0.055], u=(1, 0, 0), nseg=20, cap0="round", cap1="round", ncap=1, mat=mat)
    cuff.set_w("foot_" + sn, 1.0)
    return [toe, sole, shaft, cuff]


def hair_band(ch, z0, z1, off=0.007, back_only=True, mat=0, front_cut=-0.035):
    """Shell of the head surface between heights z0..z1 (baseline), optionally without the face side."""
    head = ch.head
    V = head.V0.copy()
    cen = np.array([0, head.cy, (z0 + z1) / 2])
    d = V - cen
    d[:, 2] *= 0.3
    V = V + off * d / np.linalg.norm(d, axis=1, keepdims=True)
    p = G.Part(V, head.faces_full, mat, closed=False)
    zc = V[:, 2]
    def pred(c):
        bad = (c[:, 2] < z0) | (c[:, 2] > z1)
        if back_only:
            bad |= (c[:, 1] < head.cy + front_cut)
        return bad
    p.drop_faces(pred)
    p.set_w("head", 1.0)
    return p


def hair_cap(ch, z0, off=0.008, front_z=None, mat=0):
    """Full hair shell above height z0 (front lowered to front_z for a fringe)."""
    head = ch.head
    V = head.V0.copy()
    cen = np.array([0, head.cy, 1.62])
    d = V - cen
    V = V + off * d / np.linalg.norm(d, axis=1, keepdims=True)
    p = G.Part(V, head.faces_full, mat, closed=False)
    fz = front_z if front_z is not None else z0

    def pred(c):
        lim = np.where(c[:, 1] < head.cy - 0.04, fz, z0)
        return c[:, 2] < lim
    p.drop_faces(pred)
    p.set_w("head", 1.0)
    return p


def hook_hand(ch, sn, side):
    f, n, c = ch.hand_frames[sn]
    wr = ch.J["wr"] * np.array([side, 1, 1])
    pos = wr + 0.060 * f + n * 0.020
    X = np.cross(f, n)
    R = np.stack([X, f, n], axis=1)
    ch.hook("hand_" + sn, "hand_" + sn, pos, rotm=R)


FONT_CANDIDATES = ["/System/Library/Fonts/Supplemental/Arial Black.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf",
                   "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]


def text_part(body, font_path=None, extrude=0.06):
    """Convert a text curve to mesh arrays (x, y, z_ext) normalised to height 1 baseline units."""
    import os
    cu = bpy.data.curves.new("tmp_text", "FONT")
    cu.body = body
    cu.size = 1.0
    cu.extrude = extrude
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.resolution_u = 4
    for fp in ([font_path] if font_path else FONT_CANDIDATES):
        if fp and os.path.exists(fp):
            cu.font = bpy.data.fonts.load(fp)
            break
    ob = bpy.data.objects.new("tmp_text", cu)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    V = np.array([v.co[:] for v in me.vertices])
    F = [tuple(p.vertices) for p in me.polygons]
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    bpy.data.meshes.remove(me)
    return V, F


def logo_part(ch, lines, zc, width, color_mat=0, line_gap=1.25):
    """Raised wordmark geometry on the shirt front. lines: list of strings, stacked; width = target width of the widest line."""
    spec = ch.spec
    parts = []
    widths = []
    pieces = []
    for ln in lines:
        V, F = text_part(ln)
        w = V[:, 0].max() - V[:, 0].min()
        pieces.append((V, F, w))
        widths.append(w)
    sc = width / max(widths)
    h = sc * 1.0
    n = len(lines)
    for i, (V, F, w) in enumerate(pieces):
        V = V.copy()
        V[:, 0] -= (V[:, 0].max() + V[:, 0].min()) / 2
        V[:, 1] -= (V[:, 1].max() + V[:, 1].min()) / 2
        ext = V[:, 2] - V[:, 2].min()
        xs, zs_ = V[:, 0] * sc, V[:, 1] * sc + zc + (n - 1) / 2 * h * line_gap - i * h * line_gap
        out = np.zeros_like(V)
        for k in range(len(V)):
            sp = surf_pt(spec, float(xs[k]), float(zs_[k]), True, 0.0075)
            out[k] = (xs[k], sp[1] - ext[k] * sc * 0.9 + 0.0004, zs_[k])
        parts.append(G.Part(out, F, color_mat, closed=False))
    p = G.merge(parts)
    p.closed = False
    ch.torso_weights(p)
    return p


def hair_curly(ch, seed=3, r=0.024, n=34, z0=1.64):
    rng = np.random.RandomState(seed)
    head = ch.head
    parts = []
    for _ in range(n):
        th = rng.uniform(0, 2 * math.pi)
        ph = rng.uniform(0.0, 1.15)
        d = np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th) * 1.05, math.cos(ph)])
        c = np.array([0, head.cy, 1.630]) + d * np.array([0.082, 0.098, 0.115])
        if c[2] < z0 or (c[1] < head.cy - 0.05 and c[2] < z0 + 0.04):
            continue
        s = ellipsoid_ = G.ellipsoid(c, (r, r, r), nseg=10, nrings=6)
        s.set_w("head", 1.0)
        parts.append(s)
    return G.merge(parts)


def hair_long(ch, color_mat=0):
    cap = hair_cap(ch, 1.64, off=0.009, front_z=1.70)
    back = G.ellipsoid(np.array([0, ch.head.cy + 0.060, 1.58]), (0.080, 0.050, 0.115), nseg=16, nrings=10)
    back.set_w("head", 1.0)
    tail = G.capsule(np.array([0, ch.head.cy + 0.095, 1.52]), np.array([0, ch.head.cy + 0.10, 1.34]), 0.030, 0.018, nseg=12)
    tail.set_w("neck", 0.5)
    tail.add_w("head", 0.5)
    return G.merge([cap, back, tail])


def hair_bob(ch):
    cap = hair_cap(ch, 1.63, off=0.010, front_z=1.69)
    band = hair_band(ch, 1.47, 1.66, 0.016, back_only=True, front_cut=-0.01)
    return G.merge([cap, band])
