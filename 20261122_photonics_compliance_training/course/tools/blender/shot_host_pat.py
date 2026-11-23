"""Host portrait: Pat Reynolds, Director, Office of Photonic Conduct (Five Sigma Photonics).

Chunky clay character built with the same kit as Gary and his manager (charkit/: numpy loft geometry, clay
materials with thumbprint bump, rigged body, shape-key face), at a higher resolution for a close medium shot:
every character mesh gets render subdivision 2 (head 2), hair carries strand grooves, glasses carry lenses.
Look follows the earlier illustrated avatar: silver bob, rectangular brown glasses, pearl necklace, cream blouse,
indigo blazer. Corporate studio set: warm wall, walnut slat panel, a small
Five Sigma Photonics logo sign (true-scale Gaussian with the +5 sigma line and one outlier, as in the org avatar).

Run (through the Blender slot wrapper):
  bslot.sh -b --factory-startup --python shot_host_pat.py -- out=course/media/render/host_pat_present.jpg pose=present
  pose = present (arms relaxed, gentle smile) | empathy (right hand on the heart, concerned brows)
  view = main (3:4 host frame) | face (close-up check) | wide (set check)
  avatar=<path> also writes a 256 px square crop centred on the face (same frame as the other avatars).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "charkit"))
import numpy as np  # noqa: E402
import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

import common as C  # noqa: E402  (course render helpers)
import chars_geo as G  # noqa: E402
import chars_mat as M  # noqa: E402
import chars_body as B  # noqa: E402
import chars_outfit as O  # noqa: E402
import chars_head as HD  # noqa: E402
import npc_kit_v2 as NK  # noqa: E402

A = C.args(out="/tmp/host_pat.jpg", pose="present", view="main", samples=256, res=1600, noise=0.01,
           look="AgX - Medium High Contrast", avatar="", exposure=0.0, light=0.55, az=70.0, threads=4)

H = 1.66                 # crown height (m); design value
Z_NECK = 1.395           # clothing neckline (baseline coordinates, crown 1.75)
Z_BTN = 1.05             # blazer button point (bottom of the V)
HW_TOP = 0.120           # blazer V half-width at the neckline
Z_BV = 1.275             # blouse V bottom
HW_BTOP = 0.080          # blouse V half-width at the neckline


# ------------------------------------------------------------------------------------------------ body shape
_torso_table = B.torso_table


def _torso_table_pat(spec):
    """Kit torso with a softer waist and a modest bust (the blazer and blouse follow it)."""
    z, rx, ry, cy = _torso_table(spec)
    z = z.copy()
    rx, ry, cy = rx.copy(), ry.copy(), cy.copy()
    bust = np.exp(-((z - 1.25) / 0.07) ** 2)
    ry = ry * (1 + 0.10 * bust)
    cy = cy - 0.010 * bust
    waist = np.exp(-((z - 1.06) / 0.07) ** 2)
    rx = rx * (1 - 0.05 * waist)
    return z, rx, ry, cy


B.torso_table = _torso_table_pat


def hw_jacket(z):
    t = np.clip((z - Z_BTN) / (Z_NECK - Z_BTN), 0.0, 1.0)
    return HW_TOP * t ** 0.62 if z > Z_BTN else 0.0


def hw_blouse(z):
    t = np.clip((z - Z_BV) / (Z_NECK - Z_BV), 0.0, 1.0)
    return HW_BTOP * t ** 0.70 if z > Z_BV else 0.0


# ------------------------------------------------------------------------------------------------ materials
def hair_material(name, light="#cdc8c0", dark="#8f8982"):
    """Clay hair with strand grooves: stripes around the vertical axis (angle from atan2), wobbled by noise."""
    m = M.clay(name, light, rough=0.55, bump=0.18, sheen=0.35)
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sx.inputs[0])
    at = nt.nodes.new("ShaderNodeMath")
    at.operation = "ARCTAN2"
    nt.links.new(sx.outputs["X"], at.inputs[0])
    nt.links.new(sx.outputs["Y"], at.inputs[1])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 9.0
    nz.inputs["Detail"].default_value = 3.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    mad = nt.nodes.new("ShaderNodeMath")
    mad.operation = "MULTIPLY_ADD"
    mad.inputs[1].default_value = 0.12
    nt.links.new(nz.outputs["Fac"], mad.inputs[0])
    nt.links.new(at.outputs[0], mad.inputs[2])
    k = nt.nodes.new("ShaderNodeMath")
    k.operation = "MULTIPLY"
    k.inputs[1].default_value = 220.0
    nt.links.new(mad.outputs[0], k.inputs[0])
    s = nt.nodes.new("ShaderNodeMath")
    s.operation = "SINE"
    nt.links.new(k.outputs[0], s.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -1.0
    nt.links.new(s.outputs[0], mr.inputs["Value"])
    ramp = nt.nodes.new("ShaderNodeMix")
    ramp.data_type = "RGBA"
    ramp.inputs["A"].default_value = (*M.srgb(dark), 1.0)
    ramp.inputs["B"].default_value = (*M.srgb(light), 1.0)
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 0.6
    nt.links.new(mr.outputs["Result"], pw.inputs[0])
    nt.links.new(pw.outputs[0], ramp.inputs["Factor"])
    nt.links.new(ramp.outputs["Result"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0015
    nt.links.new(mr.outputs["Result"], bump.inputs["Height"])
    old = [lk for lk in nt.links if lk.to_socket == bsdf.inputs["Normal"]]
    if old:
        nt.links.new(old[0].from_socket, bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def pearl_material(name):
    m = M.gloss(name, "#efe9dc", rough=0.16, spec=0.6)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Coat Weight"].default_value = 0.6
    b.inputs["Coat Roughness"].default_value = 0.08
    b.inputs["Sheen Weight"].default_value = 0.4
    b.inputs["Sheen Tint"].default_value = (1.0, 0.86, 0.88, 1.0)
    return m


def lens_material(name):
    """Thin spectacle lens: transparent (so shadow rays pass and the eyes stay lit) plus a Fresnel glossy sheen."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.03
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.12
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Fresnel"], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


# ------------------------------------------------------------------------------------------------ character
def bob_hair(ch):
    """Silver chin-length bob: head-grid shell, hanging straight below the widest ring, face opening with a
    side-swept fringe (part on the character's left), slight A-line (front longer), ends turned under; covers the
    ears."""
    head = ch.head
    V = head.V0.copy()
    NS = head.NS
    gi, gj = head.gi, head.gj
    cen = np.array([0, head.cy, 1.62])
    d = G.nrm(V - cen)
    z = V[:, 2]
    off = 0.013 + 0.009 * G.smoothstep(1.71, 1.63, z)
    V = V + d * off[:, None]
    # below z_ref the hair falls straight (no jaw follow), flaring slightly, then turning under at the ends
    z_ref = 1.628
    ring_ref = int(np.argmin(np.abs(np.array([V[(gi == i)][0, 2] if np.any(gi == i) else 9 for i in range(head.NR + 1)]) - z_ref)))
    r_ref = {}
    for j in range(NS):
        idx = np.nonzero((gi == ring_ref) & (gj == j))[0][0]
        r_ref[j] = V[idx, :2] - np.array([0, head.cy])
    for k in range(len(V)):
        if gi[k] in (0, head.NR) or V[k, 2] >= z_ref:
            continue
        base = r_ref[gj[k]]
        t = float(G.smoothstep(z_ref, 1.545, V[k, 2]))
        tuck = float(G.smoothstep(1.575, 1.538, V[k, 2]))
        f = 1.0 + 0.045 * t - 0.050 * tuck
        p = V[k, :2] - np.array([0, head.cy])
        if np.linalg.norm(p) < np.linalg.norm(base) * f:
            V[k, :2] = np.array([0, head.cy]) + base * f
    def pred(c):
        x, y, zz = c[:, 0], c[:, 1], c[:, 2]
        ang = np.arctan2(x, -(y - head.cy))           # 0 = straight front, +- pi = back
        back = np.abs(ang) / math.pi
        z_bot = 1.552 - 0.012 * G.smoothstep(0.45, 0.95, back)    # A-line: back shorter
        # fringe: hairline 1.716 at the part (x = +0.030), sweeping down to 1.694 at the right temple
        fr = 1.706 - 0.014 * G.smoothstep(0.03, -0.075, x)
        fr = np.where(x > 0.03, 1.706 + 0.006 * G.smoothstep(0.03, 0.07, x), fr)
        a_open = np.radians(57 + 4 * G.smoothstep(1.66, 1.58, zz))
        face = (np.abs(ang) < a_open) & (zz < fr)
        return (zz < z_bot) | face
    # closed volume: outer shell, inner shell on the scalp (3 mm under the skin), rim quads along the open edges,
    # so the bob meets the face and the neck instead of showing a hollow hood
    F = head.faces_full
    cents = np.array([V[list(q)].mean(axis=0) for q in F])
    Fk = [q for q, k in zip(F, ~pred(cents)) if k]
    cnt = {}
    for q in Fk:
        for a, b in zip(q, q[1:] + q[:1]):
            e = (min(a, b), max(a, b))
            cnt[e] = cnt.get(e, 0) + 1
    N = len(V)
    Vin = head.V0 - G.nrm(head.V0 - cen) * 0.003
    faces = list(Fk) + [tuple(reversed([i + N for i in q])) for q in Fk]
    for q in Fk:
        for a, b in zip(q, q[1:] + q[:1]):
            if cnt[(min(a, b), max(a, b))] == 1:
                faces.append((b, a, a + N, b + N))
    part = G.Part(np.concatenate([V, Vin]), faces, 0, closed=True)
    part.drop_faces(lambda c: np.zeros(len(c), bool))   # prune unused vertices
    part.set_w("head", 1.0)
    return part


def lens_part(center, w, h, n=40):
    """Flat rounded-rectangle lens (same superellipse as O.rect_frame), fan-triangulated, facing -y."""
    hw, hh = w / 2, h / 2
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        c, s_ = math.cos(t), math.sin(t)
        e = 6.0
        pts.append(np.asarray(center, float) + np.array([hw * np.sign(c) * abs(c) ** (2 / e), 0, hh * np.sign(s_) * abs(s_) ** (2 / e)]))
    V = np.array([np.asarray(center, float)] + pts)
    F = [(0, 1 + (k + 1) % n, 1 + k) for k in range(n)]
    p = G.Part(V, F, 0, closed=False)
    p.set_w("head", 1.0)
    return p


# custom closed-mouth expressions (same parameter set as the kit's 14 expressions; gains applied by HD.E)
HD.EXPR["pat_smile"] = dict(corner=0.0052, gap=0.0030, width=1.02, cheek=0.0035, lid_lo=0.07, brow_in=0.002, brow_out=0.003)
HD.EXPR["pat_concern"] = dict(brow_in=0.0065, brow_out=-0.003, lid_up=0.10, lid_tilt=-0.14, corner=-0.0015, gap=0.0025,
                              width=0.97, pupil=1.05)
for _k in ("pat_smile", "pat_concern"):
    if _k not in HD.KEYS:
        HD.KEYS.append(_k)


def build_pat():
    spec = B.default_spec()
    spec.update(H=H, build=0.92, belly=0.0, shoulders=0.93, arm_r=0.90, leg_r=0.95, wide=0.95)
    spec["head"].update(w0=0.078, jaw=0.035, chin=0.055, brow_t=0.0048, brow_h=0.0064, brow_w=1.04,
                        nose_s=0.78, nose_len=0.019, eye_r=0.0245, ear_s=0.55, lip_r=0.0064, lip_f=0.0056)
    ch = B.Character("pat", spec)
    ch.root_props()
    pre = "MAT_pat_"
    O.face_mats(ch, "#e2b48e", brow_hex="#77716a", lip_hex="#b4605c", flush_color="#d0806e")
    ch.mat("jacket", M.clay(pre + "blazer", "#1f2448", rough=0.72, bump=0.22, sheen=0.12,
                            folds=[(0.50, 0.07, 15.0, 0.25), (0.20, 0.05, 20.0, 0.25)]))
    ch.mat("blouse", M.clay(pre + "blouse", "#f3efe6", rough=0.55, bump=0.16, sheen=0.35))
    ch.mat("hair", hair_material(pre + "hair"))
    ch.mat("frame", M.gloss(pre + "glasses", "#4a3628", rough=0.28, spec=0.5, coat=0.25))
    ch.mat("lens", lens_material(pre + "lens"))
    ch.mat("pearl", pearl_material(pre + "pearl"))
    ch.mat("gold", M.metal(pre + "gold", "#d9b45a", 0.28))
    ch.mat("btn", M.gloss(pre + "button", "#1c1f38", rough=0.25, spec=0.6))
    ch.mat("pin_navy", M.gloss(pre + "pin_navy", "#0b2545", rough=0.2, spec=0.6))
    ch.mat("pin_amber", M.gloss(pre + "pin_amber", "#f2a541", rough=0.25, spec=0.5))
    J = ch.J
    surf = O.surf_pt

    ch.add(ch.skin_parts(), "body", ["skin"])
    ch.build_face()

    # ------------------------------------------------ blouse: V neck shell, long sleeves with cuffs, soft collar
    bl = O.jacket_shell(ch, 0.010, 0.96, Z_NECK, hw_blouse, n=72, dz=0.015)
    parts = [bl]
    ar = spec["arm_r"]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.04
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), ch.ARM_UA[0] * ar + 0.010, ch.ARM_UA[1] * ar + 0.010, u=(0, 1, 0), nseg=28)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), ch.ARM_FA[0] * ar + 0.010, ch.ARM_FA[1] * ar + 0.010, u=(0, 1, 0), nseg=28, cap1="flat")
        fa.set_w("forearm_" + sn, 1.0)
        dd = (end - el) / np.linalg.norm(end - el)
        cuff = O.ring_tube(end - dd * 0.02, dd, ch.ARM_FA[0][2] * ar + 0.016, ch.ARM_FA[0][2] * ar + 0.016, 0.010, u_hint=(0, 1, 0))
        cuff.set_w("forearm_" + sn, 1.0)
        parts += [ua, fa, cuff]
    blouse = G.merge(parts)
    blouse.closed = False
    ch.add(blouse, "blouse", ["blouse"], solid=0.004)
    col = []
    for side in (1, -1):
        zs = np.linspace(Z_NECK + 0.004, Z_BV + 0.010, 14)
        pts, rr = [], []
        for zz in zs:
            t = (Z_NECK - zz) / (Z_NECK - Z_BV)
            w = 0.016 + 0.010 * math.sin(math.pi * min(max(t, 0.0), 1.0)) ** 0.8
            pts.append(surf(spec, side * (float(hw_blouse(zz)) + w * 0.45), zz, True, 0.010 + 0.0095))
            rr.append(w * 0.5)
        cp = G.tube(np.array(pts), np.array(rr), np.full(len(zs), 0.0042), u=(1, -0.15, 0), nseg=10, cap0="round", cap1="round", ncap=2)
        ch.torso_weights(cp)
        col.append(cp)
    rx_, ry_, cy_ = B.torso_at(spec, np.array([Z_NECK]), 0.010)
    ang = np.linspace(-2.45, 2.45, 26)
    band = G.tube(np.array([[(rx_[0] * 0.80) * math.sin(a), cy_[0] + ry_[0] * 1.08 * math.cos(a), Z_NECK + 0.006] for a in ang]),
                  0.012, 0.0045, u=(0, 0, 1), nseg=10, cap0="round", cap1="round", ncap=2)
    band.set_w("neck", 0.5)
    band.add_w("chest", 0.5)
    col.append(band)
    ch.add(G.merge(col), "blouse_collar", ["blouse"])

    # ------------------------------------------------ blazer: open-front shell, sleeves, hem bead, lapels, collar
    jk = O.jacket_shell(ch, 0.026, 0.78, Z_NECK, hw_jacket, n=72, dz=0.015, flare=lambda z: 0.014 * float(G.smoothstep(0.98, 0.78, z)))
    sleeves = []
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.12
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), ch.ARM_UA[0] * ar + 0.016, ch.ARM_UA[1] * ar + 0.016, u=(0, 1, 0), nseg=30)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), ch.ARM_FA[0] * ar + 0.016, ch.ARM_FA[1] * ar + 0.016, u=(0, 1, 0), nseg=30, cap1=None)
        fa.set_w("forearm_" + sn, 1.0)
        dd = (end - el) / np.linalg.norm(end - el)
        hem = O.ring_tube(end, dd, ch.ARM_FA[0][2] * ar + 0.016, ch.ARM_FA[0][2] * ar + 0.016, 0.0085, u_hint=(0, 1, 0))
        hem.set_w("forearm_" + sn, 1.0)
        sleeves += [ua, fa, hem]
    zb = 0.78
    rx, ry, cy = B.torso_at(spec, np.array([zb]), 0.026)
    th = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    ring_pts = np.array([[(rx[0] + 0.016) * math.cos(t), cy[0] + (ry[0] + 0.016) * math.sin(t), zb] for t in th])
    hem_ring = G.tube(ring_pts, 0.0085, 0.0085, u=(0, 0, 1), nseg=8, closed_path=True)
    ch.torso_weights(hem_ring)
    jacket = G.merge([jk, hem_ring] + sleeves)
    jacket.closed = False
    ch.add(jacket, "blazer", ["jacket"], solid=0.007)
    lap = []
    for side in (1, -1):
        zs = np.linspace(Z_NECK - 0.004, Z_BTN + 0.012, 18)
        pts, rr = [], []
        for zz in zs:
            t = (Z_NECK - zz) / (Z_NECK - Z_BTN)
            w = 0.022 + 0.034 * math.sin(math.pi * min(max(t, 0.0) * 1.08, 1.0)) ** 0.9
            pts.append(surf(spec, side * (float(hw_jacket(zz)) + w * 0.50), zz, True, 0.026 + 0.0095))
            rr.append(w * 0.5)
        lp = G.tube(np.array(pts), np.array(rr), np.full(len(zs), 0.0078), u=(1, -0.12, 0), nseg=12, cap0="round", cap1="round", ncap=2)
        ch.torso_weights(lp)
        lap.append(lp)
    ch.add(G.merge(lap), "lapels", ["jacket"], bevel=0.0015)
    rxc, ryc, cyc = B.torso_at(spec, np.array([Z_NECK]), 0.0)
    ang = np.linspace(-2.55, 2.55, 26)
    cpath = np.array([[0.098 * math.sin(a), cyc[0] + 0.014 + 0.090 * math.cos(a), Z_NECK + 0.014] for a in ang])
    jc = G.tube(cpath, 0.028, 0.0098, u=(0, 0, 1), nseg=10, cap0="round", cap1="round", ncap=2)
    jc.set_w("chest", 0.5)
    jc.add_w("neck", 0.5)
    ch.add(jc, "blazer_collar", ["jacket"])
    det = []
    for side in (1, -1):
        det.append(G.ellipsoid(surf(spec, side * 0.112, 0.87, True, 0.026), (0.048, 0.0080, 0.0145), nseg=14, nrings=8, e=0.5))
    for q in det:
        ch.torso_weights(q)
    ch.add(G.merge(det), "pocket_flaps", ["jacket"])
    btn = G.ellipsoid(surf(spec, 0.0, Z_BTN - 0.006, True, 0.0272), (0.0115, 0.0055, 0.0115), nseg=16, nrings=8)
    ch.torso_weights(btn)
    ch.add(btn, "button", ["btn"])

    # ------------------------------------------------ hair, glasses with lenses
    NK.head_part(ch, bob_hair(ch), "hair", ["hair"])
    gp = NK.glasses(ch, "rect", rx=0.052, rz=0.036, fwd=0.050, tube_r=0.0060)
    ch.add(gp, "glasses", ["frame"])
    lenses = []
    for c in ch.eye_centres:
        cc = c + np.array([0, -0.050, 0.004])
        lenses.append(lens_part(cc + np.array([0, 0.0008, 0]), 2 * 0.052 - 0.004, 2 * 0.036 - 0.004))
    ch.add(G.merge(lenses), "lenses", ["lens"], solid=0.0016)
    return ch


def build_jewellery(ch):
    """Pearl strand resting on the skin / blouse / blouse collar in the V (ray cast onto the evaluated meshes), pearls
    placed at equal arc length. The bob covers the ears, so no earrings are modelled."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    K = ch.K
    bvhs = [BVHTree.FromObject(ch.objs[n], dg) for n in ("body", "blouse", "blouse_collar")]
    r = 0.0072
    dense = []
    for x in np.linspace(-0.124, 0.124, 440):
        zz = 1.318 + 0.075 * (abs(x) / 0.112) ** 1.8
        best = None
        for bv in bvhs:
            hit = bv.ray_cast(Vector((x * K, -0.6, zz * K)), Vector((0, 1, 0)))
            if hit[0] is not None and (best is None or hit[0].y < best[0].y):
                best = hit
        if best is None:
            continue
        nrm = Vector(best[1]).normalized()
        if nrm.y > 0:
            nrm = -nrm
        dense.append(np.array(best[0]) / K + np.array(nrm) * (r + 0.0010))
    dense = np.array(dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    sarc = np.concatenate([[0.0], np.cumsum(seg)])
    step = 2 * r * 1.04
    n = int(sarc[-1] / step)
    s0 = (sarc[-1] - n * step) / 2
    pearls = []
    last = None
    for k in range(n + 1):
        sk = s0 + k * step
        c = np.array([np.interp(sk, sarc, dense[:, i]) for i in range(3)])
        if last is not None and np.linalg.norm(c - last) < 2 * r * 0.98:
            continue   # path jumps where the ray moves from skin to collar: never let pearls overlap
        last = c
        p = G.ellipsoid(c, (r, r, r), nseg=18, nrings=10)
        p.set_w("chest", 1.0)
        pearls.append(p)
    ch.add(G.merge(pearls), "pearls", ["pearl"])
    return {"necklace_pearls": len(pearls), "max_step_m": round(float(seg.max()), 4)}


def lapel_pin(ch):
    """Five Sigma Photonics lapel pin on the wearer's left lapel: navy enamel disc, gold rim, amber outlier dot."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    K = ch.K
    bv = BVHTree.FromObject(ch.objs["lapels"], dg)
    x, z = 0.118, 1.250
    hit = bv.ray_cast(Vector((x * K, -0.6, z * K)), Vector((0, 1, 0)))
    if hit[0] is None:
        print("PIN: no lapel hit")
        return
    p = np.array(hit[0]) / K
    nrm = np.array(Vector(hit[1]).normalized())
    if nrm[1] > 0:
        nrm = -nrm
    disc = G.lathe(np.array([[0.0, 0.0], [0.0105, 0.0], [0.0110, 0.0018], [0.0100, 0.0032], [0.0, 0.0034]]), center=p, axis=nrm, nseg=32)
    rim = O.ring_tube(p + nrm * 0.0028, nrm, 0.0102, 0.0102, 0.0011, u_hint=(1, 0, 0), n=36, ns=6)
    dot = G.ellipsoid(p + nrm * 0.0036 + np.array([0.004, 0, -0.003]), (0.0024, 0.0012, 0.0024), nseg=12, nrings=6)
    for q in (disc, rim, dot):
        ch.torso_weights(q)
    ch.add(disc, "pin", ["pin_navy"])
    ch.add(rim, "pin_rim", ["gold"])
    ch.add(dot, "pin_dot", ["pin_amber"])


def subdivide_all(ch, skip=("lenses",)):
    for name, ob in ch.objs.items():
        if ob.type != "MESH" or name in skip:
            continue
        sd = ob.modifiers.get("Subsurf") or ob.modifiers.new("Subsurf", "SUBSURF")
        sd.levels = 1
        sd.render_levels = 2


# ------------------------------------------------------------------------------------------------ pose
def set_pose_matrix(arm, bone, loc_world, R=None):
    rest = arm.data.bones[bone].matrix_local.copy()
    rot = rest.to_3x3()
    if R is not None:
        rot = Matrix(np.asarray(R).tolist()) @ rot
    mtx = rot.to_4x4()
    mtx.translation = Vector(loc_world)
    arm.pose.bones[bone].matrix = arm.matrix_world.inverted() @ mtx
    bpy.context.view_layer.update()


def frame_rot(f0, n0, f1, n1):
    def basis(f, n):
        f = G.nrm(np.asarray(f, float))
        n = np.asarray(n, float) - f * np.dot(n, f)
        n = G.nrm(n)
        return np.stack([f, n, np.cross(f, n)], axis=1)
    return basis(f1, n1) @ basis(f0, n0).T


def set_expression(ch, weights):
    """Set the face shape keys directly (the kit drives them from root properties; drivers are replaced by fixed
    values here so the still does not depend on driver evaluation in background mode)."""
    for ob in ch.objs.values():
        keys = ob.data.shape_keys if ob.type == "MESH" else None
        if keys is None:
            continue
        for kb in keys.key_blocks[1:]:
            try:
                keys.driver_remove('key_blocks["%s"].value' % kb.name)
            except Exception:
                pass
            kb.value = float(weights.get(kb.name, 0.0))
    # closed-mouth expressions only: teeth and tongue would poke through the closed lips
    for n in ("teeth_up", "teeth_lo", "tongue", "drops"):
        if n in ch.objs:
            ch.objs[n].hide_render = True


def set_curl(ch, sn, curls):
    """Finger curl without drivers: rotation about the bone X axis, same angles as the kit's curl drivers."""
    pb = ch.arm.pose.bones
    for f, v in curls.items():
        for k_ in range(3):
            b = pb["%s_%d_%s" % (f, k_ + 1, sn)]
            try:
                b.driver_remove("rotation_euler", 0)
            except Exception:
                pass
            b.rotation_mode = "XYZ"
            b.rotation_euler[0] = v * B.CURL_ANGLES[f][k_]


def pose(ch, kind, cam_loc):
    arm = ch.arm
    K = ch.K
    pb = arm.pose.bones
    set_pose_matrix(arm, "ctl_look", cam_loc)
    if kind == "present":
        for sn, side in (("L", 1), ("R", -1)):
            set_pose_matrix(arm, "ik_hand_" + sn, Vector((side * 0.235, -0.115, 0.845)) * K)
        set_expression(ch, {"pat_smile": 1.0})
        for sn in ("L", "R"):
            set_curl(ch, sn, {"index": 0.18, "middle": 0.24, "ring": 0.28, "pinky": 0.32, "thumb": 0.10})
        pb["head"].rotation_mode = "XYZ"
        pb["head"].rotation_euler = (0.0, math.radians(-2.5), 0.0)
    else:
        # right hand flat on the heart (wearer's left chest), palm facing the body, fingers toward +x and up
        set_pose_matrix(arm, "ik_hand_L", Vector((0.235, -0.115, 0.845)) * K)
        f0, n0, c0 = ch.hand_frames["R"]
        f1 = G.nrm(np.array([0.62, -0.08, 0.78]))
        n1 = np.array([0.0, 1.0, 0.0])   # kit hand frame: n is the palm normal (palm onto the chest)
        R = frame_rot(f0, n0, f1, n1)
        palm_target = np.array([0.075, 0.0, 1.215])
        # put the palm on the blazer: ray cast from the front at the palm footprint
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        bvs = [BVHTree.FromObject(ch.objs[n], dg) for n in ("blazer", "lapels", "blouse", "pin", "pin_rim") if n in ch.objs]
        ys = []
        cvec = np.cross(f1, n1)
        for a in np.linspace(-0.05, 0.06, 6):
            for b in np.linspace(-0.04, 0.04, 5):
                q = palm_target + f1 * a + cvec * b
                for bv in bvs:
                    hit = bv.ray_cast(Vector((q[0] * K, -0.6, q[2] * K)), Vector((0, 1, 0)))
                    if hit[0] is not None:
                        ys.append(hit[0].y / K)
        y_surf = min(ys)
        palm_c = np.array([palm_target[0], y_surf - 0.0265 * 0.92, palm_target[2]])
        wr = palm_c - 0.056 * f1
        set_pose_matrix(arm, "ik_hand_R", Vector(wr) * K, R)
        # elbow down and slightly out
        el = ch.J["el"] * np.array([-1, 1, 1])
        set_pose_matrix(arm, "ik_elbow_R", Vector(el + np.array([-0.25, 0.20, -0.10])) * K)
        set_curl(ch, "R", {"index": 0.03, "middle": 0.03, "ring": 0.04, "pinky": 0.05, "thumb": 0.35})
        set_curl(ch, "L", {"index": 0.18, "middle": 0.24, "ring": 0.28, "pinky": 0.32, "thumb": 0.10})
        set_expression(ch, {"pat_concern": 1.0})
        pb["head"].rotation_mode = "XYZ"
        pb["head"].rotation_euler = (math.radians(-2.0), math.radians(-3.0), math.radians(-4.0))
    bpy.context.view_layer.update()


# ------------------------------------------------------------------------------------------------ set
WALL_Y = 2.30


def logo_sign(cx, cz, size):
    """Five Sigma Photonics sign: navy panel on four steel standoffs; Gaussian (computed), +5 sigma dashed
    line and one amber outlier, as in the org avatar (512 px design grid)."""
    navy = C.principled("sign_navy", color=M.srgb("#0b2545"), rough=0.35, coat=0.5)
    light = C.principled("sign_light", color=M.srgb("#e7ecef"), rough=0.4, emission=M.srgb("#e7ecef"), emission_strength=0.6)
    amber = C.principled("sign_amber", color=M.srgb("#f2a541"), rough=0.4, emission=M.srgb("#f2a541"), emission_strength=0.8)
    fillc = tuple(0.84 * a + 0.16 * b for a, b in zip(M.srgb("#0b2545"), M.srgb("#e7ecef")))
    fill = C.principled("sign_fill", color=fillc, rough=0.4)
    steel = C.principled("standoff", color=(0.62, 0.62, 0.64), metallic=1.0, rough=0.3)
    s = size / 512.0
    t_panel, l_std = 0.012, 0.030
    y_front = WALL_Y - l_std - t_panel
    C.box("sign_panel", (size, t_panel, size), (cx, y_front + t_panel / 2, cz), navy, bevel=0.003)
    for ix in (-1, 1):
        for iz in (-1, 1):
            loc = (cx + ix * (size / 2 - 0.03), WALL_Y - (l_std + t_panel) / 2 + 0.002, cz + iz * (size / 2 - 0.03))
            C.cylinder("standoff", 0.009, l_std + t_panel + 0.004, loc,
                       steel, verts=24, rot=(math.pi / 2, 0, 0), bevel=0.0015)

    def P(xp, yp, dy=0.0):
        return (cx + (xp - 256) * s, y_front - dy, cz + (256 - yp) * s)
    x0, x1, base, peak = 48, 464, 384, 120
    lo, hi = -3.3, 6.2
    def px(zz):
        return x0 + (zz - lo) / (hi - lo) * (x1 - x0)
    curve = [(px(lo + i * (hi - lo) / 200), base - (base - peak) * math.exp(-0.5 * (lo + i * (hi - lo) / 200) ** 2)) for i in range(201)]
    pts = np.array([P(x, y, 0.004) for x, y in curve])
    tb = G.tube(pts, 11 * s, 3 * s, u=(0, 0, 1), nseg=10)
    G.to_object(tb, "sign_curve", [light], bpy.context.scene.collection)
    V, F = [], []
    for i, (x, y) in enumerate(curve):
        V.append(P(x, y, 0.001))
        V.append(P(x, base, 0.001))
        if i:
            k = 2 * i
            F.append((k - 2, k, k + 1, k - 1))
    C.mesh_obj("sign_fill", V, F, fill)
    x5 = px(5.0)
    yy = 104
    while yy < base:
        y2 = min(yy + 24, base)
        seg = G.capsule(np.array(P(x5, yy, 0.003)), np.array(P(x5, y2, 0.003)), 6 * s, 6 * s, nseg=10)
        G.to_object(seg, "sign_dash", [amber], bpy.context.scene.collection)
        yy += 40
    dot = G.ellipsoid(np.array(P(px(5.75), base - 40, 0.004)), (20 * s, 4 * s, 20 * s), nseg=24, nrings=10)
    G.to_object(dot, "sign_dot", [amber], bpy.context.scene.collection)
    brushed = C.principled("letters", color=(0.80, 0.80, 0.80), metallic=1.0, rough=0.32)
    t = C.text_mesh("wordmark", "FIVE SIGMA PHOTONICS", size * 0.115, (0, 0, 0), h=0.006, mat=brushed,
                    rot=(math.pi / 2, 0, 0))
    t.location = (cx, WALL_Y - 0.004, cz - size / 2 - 0.075)
    t.data.space_character = 1.15
    try:
        t.data.font = bpy.data.fonts.load("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    except Exception:
        pass


def studio_set():
    wall = C.principled("wall", color=M.srgb("#8b8379"), rough=0.85, bump=0.15, bump_scale=60.0, bump_dist=0.0005)
    C.box("wall", (8.0, 0.05, 4.0), (0, WALL_Y + 0.025, 1.8), wall)
    floor = C.principled("floor", color=M.srgb("#3d332b"), rough=0.6)
    C.box("floor", (8.0, 6.0, 0.02), (0, 0, -0.01), floor)
    walnut = C.principled("walnut", color=M.srgb("#5a3d2a"), rough=0.5, color_var=0.25, color_scale=8.0)
    # vertical walnut slats on a dark backer, behind the subject's right shoulder (image left)
    backer = C.principled("backer", color=(0.02, 0.02, 0.02), rough=0.8)
    x0, x1 = -1.9, -0.18
    C.box("slat_backer", (x1 - x0, 0.012, 2.6), ((x0 + x1) / 2, WALL_Y - 0.006, 1.3), backer)
    x = x0 + 0.02
    while x < x1 - 0.02:
        C.box("slat", (0.034, 0.022, 2.6), (x, WALL_Y - 0.012 - 0.011, 1.3), walnut, bevel=0.002)
        x += 0.054
    logo_sign(0.60, 1.96, 0.30)
    # plant in a ceramic pot, far left, out of focus
    pot = C.principled("pot", color=M.srgb("#d8d2c8"), rough=0.4)
    C.cylinder("pot", 0.17, 0.50, (-0.95, 1.55, 0.25), pot, verts=48, r2=0.14, bevel=0.01)
    leaf = C.principled("leaf", color=M.srgb("#3f6b3a"), rough=0.45)
    rng = np.random.RandomState(7)
    for k in range(26):
        a = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(0.15, 0.75)
        L = rng.uniform(0.45, 0.95)
        base = np.array([-0.95, 1.55, 0.48])
        dirv = np.array([math.cos(a) * math.sin(tilt), math.sin(a) * math.sin(tilt), math.cos(tilt)])
        tip = base + dirv * L
        mid = base + dirv * L * 0.55 + np.array([0, 0, 0.05])
        w = rng.uniform(0.03, 0.05)
        lp = G.tube(np.array([base + dirv * 0.05, mid, tip]), [w * 0.4, w, w * 0.2], [0.004, 0.006, 0.003],
                    u=(math.cos(a + 1.57), math.sin(a + 1.57), 0), nseg=10)
        G.to_object(lp, "leaf", [leaf], bpy.context.scene.collection)


def lights(K):
    face = Vector((0, -0.10, 1.50))
    s = A["light"]
    C.area_light("key", (-1.10, -1.45, 2.05), face, 1.2, 130.0 * s, color=(1.0, 0.96, 0.91), size_y=1.4)
    C.area_light("fill", (1.50, -1.30, 1.45), face, 1.6, 40.0 * s, color=(0.92, 0.95, 1.0))
    C.area_light("rim_L", (-0.95, 0.85, 2.15), Vector((0, 0.05, 1.50)), 0.5, 45.0 * s, color=(1.0, 0.97, 0.92))
    C.area_light("rim_R", (0.95, 0.90, 2.10), Vector((0, 0.05, 1.50)), 0.5, 35.0 * s, color=(0.92, 0.96, 1.0))
    C.area_light("wall_wash", (0.4, 1.2, 3.1), Vector((0.3, WALL_Y, 1.5)), 1.5, 90.0 * s, color=(1.0, 0.93, 0.84))
    C.area_light("top", (0, -0.4, 3.0), Vector((0, 0, 1.2)), 2.0, 25.0 * s)


# ------------------------------------------------------------------------------------------------ main
def main():
    ch = build_pat()
    K = ch.K
    eye_z = float(np.mean([c[2] for c in ch.eye_centres])) * K
    subdivide_all(ch)
    info = build_jewellery(ch)
    lapel_pin(ch)
    subdivide_all(ch, skip=("lenses",))
    C.setup_render(samples=A["samples"], res=(A["res"] * 3 // 4, A["res"]), noise=A["noise"], look=A["look"], exposure=A["exposure"])
    sc = bpy.context.scene
    sc.render.image_settings.quality = 85
    sc.render.threads_mode = "FIXED"   # keep the machine responsive (CPU side of the GPU render)
    sc.render.threads = A["threads"]
    C.world_hdri("studio_small_09", strength=0.10, rot_z_deg=120, bg_color=(0.05, 0.045, 0.04))
    studio_set()
    lights(K)
    if A["view"] == "face":
        sc.render.resolution_x, sc.render.resolution_y = A["res"], A["res"]
        tgt = Vector((0, 0, eye_z - 0.03))
        cam = C.camera((0.0, -1.10, eye_z), tgt, lens=85.0, sensor=36.0)
        cam.data.dof.use_dof = False
    elif A["view"] == "side":
        sc.render.resolution_x, sc.render.resolution_y = A["res"], A["res"]
        az = math.radians(A["az"])
        c0 = Vector((0, 0, 1.15))
        cam = C.camera(c0 + Vector((math.sin(az), -math.cos(az), 0.08)) * 1.6, c0, lens=50.0)
    elif A["view"] == "wide":
        cam = C.camera((0.9, -3.6, 1.6), Vector((0, 0.8, 1.2)), lens=35.0)
    else:
        # 3:4 host frame: subject width about 0.60 m at the face plane, eye level camera, frame shifted down
        lens = 85.0
        dist = 0.60 * lens / 36.0
        cam_loc = Vector((0.0, -0.12 - dist, eye_z))
        cam = C.camera(cam_loc, cam_loc + Vector((0, 1, 0)), lens=lens, sensor=36.0)
        frame_h = 0.60 * 4.0 / 3.0
        z_top = (1.75 * 1.0 + 0.04) * K + 0.03
        z_c = z_top - frame_h / 2
        cam.data.shift_y = (z_c - eye_z) / frame_h
        cam.data.shift_x = -0.03
        C.set_aperture(cam, (0, -0.12 - 0.06, eye_z), pupil_mm=lens * 1e-3 / 3.2)
    pose(ch, A["pose"], cam.location)
    C.compositor(vignette=0.12)
    sc.frame_set(sc.frame_current)
    bpy.context.view_layer.update()
    lips = ch.objs["lips"]
    print("KEYS smile=%.3f concern=%.3f" % (lips.data.shape_keys.key_blocks["pat_smile"].value, lips.data.shape_keys.key_blocks["pat_concern"].value))
    # report image positions of the eyes and the frame edges of the subject
    for nm, p in (("eyeL", Vector(ch.eye_centres[0]) * K), ("eyeR", Vector(ch.eye_centres[1]) * K)):
        uv = world_to_camera_view(sc, cam, p)
        print("IMG %s x=%.3f y=%.3f" % (nm, uv.x, 1 - uv.y))
    print("INFO", info, "K", round(K, 4), "eye_z", round(eye_z, 4))
    if A["view"] == "main" and A["avatar"]:
        # avatar crop: centred between the eyes, top just above the hair
        hair = ch.objs["hair"]
        top = max((hair.matrix_world @ Vector(c)).z for c in hair.bound_box)
        uv = world_to_camera_view(sc, cam, Vector((0, -0.1, eye_z)))
        uv_top = world_to_camera_view(sc, cam, Vector((0, 0.0, top)))
        print("AVATAR x=%.4f top=%.4f" % (uv.x, 1 - uv_top.y))
    C.render(A["out"])
    if A["view"] == "main" and A["avatar"]:
        make_avatar(A["out"], A["avatar"], uv.x, 1 - uv_top.y)


def make_avatar(src, dst, u, v_top, size=256, frac=0.80, margin=0.05):
    """Square crop of the render (side = frac of the image width), centred on u, top at the hair top minus a margin
    (fraction of the side), scaled to size x size. Saved with the Standard view transform so the already tone-mapped
    pixels are written unchanged."""
    img = bpy.data.images.load(src)
    W, Hh = img.size
    side = int(W * frac)
    x0 = max(0, min(W - side, int(u * W) - side // 2))
    y0 = max(0, min(Hh - side, int(v_top * Hh - margin * side)))
    px = np.array(img.pixels[:], dtype=np.float32).reshape(Hh, W, 4)[::-1]
    crop = px[y0:y0 + side, x0:x0 + side]
    out = bpy.data.images.new("avatar", side, side)
    out.pixels = crop[::-1].ravel().tolist()
    out.scale(size, size)
    sc = bpy.context.scene
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 88
    out.save_render(dst, scene=sc)
    print("AVATAR", dst, "crop", x0, y0, side)


if __name__ == "__main__":
    main()
