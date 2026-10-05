"""NPC v2 helpers (agent C2, 2026-10-03): clothing, hair styles, face hair that follows the expression shape keys,
glasses, accessories (lanyard badge, cap, headset, backpack) and raised parody wordmarks for the v2 NPC family.

Imports the v2 kit in scripts/assets/characters/v2 (chars_geo, chars_body, chars_head, chars_outfit, chars_mat) and
does not modify it. Coordinates: baseline (crown 1.75 m) unless a function says "v1 head units" (then the head
transform ch.T maps them onto the v2 head). The caller's Character.add scales by K.
"""
import math

import numpy as np

import chars_geo as G
import chars_body as B
import chars_head as HD
import chars_mat as M
import chars_outfit as O


# ------------------------------------------------------------------------------------------------ small utils
def head_part(ch, part, name, mats, **kw):
    """Map a Part from v1 head units through the head transform T, then add it."""
    p = part.copy()
    p.v = ch.T(p.v)
    return ch.add(p, name, mats, **kw)


def grid_normals(head):
    """Outward vertex normals of the intact head grid (v1 head units)."""
    V = head.V0
    N = np.zeros_like(V)
    for f in head.faces_full:
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        n = np.cross(b - a, c - a)
        for i in f:
            N[i] += n
    cen = np.array([0.0, head.cy, head.zc + head.hh * 0.5])
    rad = V - cen
    s = np.sign(np.sum(N * rad, axis=1))
    s[s == 0] = 1.0
    N = N * s[:, None]
    return G.nrm(N)


def _orient_out(V, F, cen):
    """Flip faces whose normal points toward cen (for open shells that are not normal-recalculated)."""
    out = []
    for f in F:
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        n = np.cross(b - a, c - a)
        m = V[list(f)].mean(axis=0)
        out.append(tuple(reversed(f)) if np.dot(n, m - cen) < 0 else tuple(f))
    return out


def ear_centres(ch):
    """Ear centres (baseline), [left(+x), right(-x)]."""
    ears = HD.make_ears(ch.head, ch.spec["head"])
    v = ch.T(ears.v)
    return [v[v[:, 0] > 0].mean(axis=0), v[v[:, 0] < 0].mean(axis=0)]


def head_halfwidth(ch, z_v1):
    """Half-width of the head (baseline) at v1 head height z_v1."""
    V = ch.head.V0
    sel = np.abs(V[:, 2] - z_v1) < 0.008
    return float(np.max(np.abs(V[sel, 0]))) * ch.spec["headS"][0]


def accessory_toggle(ch, objs):
    """Root property p_accessory (1 = shown, 0 = hidden) drives hide_render / hide_viewport of the accessory objects."""
    r = ch.root
    if "p_accessory" not in r.keys():
        r["p_accessory"] = 1.0
        r.id_properties_ui("p_accessory").update(min=0.0, max=1.0, description="1 = vendor accessory visible (lanyard badge / cap / headset / backpack), 0 = hidden")
    for ob in objs:
        M._driver(ob, "hide_render", r, "p_accessory", "v<0.5")
        M._driver(ob, "hide_viewport", r, "p_accessory", "v<0.5")


# ------------------------------------------------------------------------------------------------ clothing
def tee(ch, z0=0.90, z1=1.385, off_top=0.012, off_low=0.008, flare=0.018, sleeve_frac=0.50, hem=True, sleeve_pad=0.016, sleeve_hem=True):
    """Untucked t-shirt: torso shell (loose below the belly, flared at the hips so it drapes over the trouser
    waist and the thigh tops), short sleeves with rolled hems, neck band. Returns a merged Part (open shell)."""
    spec, J = ch.spec, ch.J
    zs = np.arange(z0, z1 + 1e-6, 0.02)
    if zs[-1] < z1 - 1e-6:
        zs = np.append(zs, z1)
    off = off_top + off_low * G.smoothstep(1.10, 0.95, zs) + flare * G.smoothstep(1.00, z0 - 0.02, zs)
    rx, ry, cy = B.torso_at(spec, zs, 0.0)
    rx, ry = rx + off, ry + off
    path = np.stack([np.zeros_like(zs), cy, zs], axis=1)
    body = G.tube(path, rx, ry, u=(1, 0, 0), nseg=48, cap0=None, cap1=None)
    ch.torso_weights(body)
    parts = [body]
    if hem:
        h = O.ring_tube((0, cy[0], z0 + 0.004), (0, 0, 1), rx[0] - 0.002, ry[0] - 0.002, 0.0105, n=48)
        ch.torso_weights(h)
        parts.append(h)
    ar = spec["arm_r"]
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el = J["sh"] * sx, J["el"] * sx
        e2 = sh + (el - sh) * sleeve_frac
        rr = ch.ARM_UA[0] * ar + sleeve_pad
        sl = G.tube(np.array([sh, (sh + e2) / 2, e2]), np.array([rr[0], rr[0] * 0.98, rr[1] * 0.99]), np.array([rr[0], rr[0] * 0.98, rr[1] * 0.99]),
                    u=(0, 1, 0), nseg=24, cap0="round", cap1=None)
        sl.set_w("upper_arm_" + sn, 1.0)
        d = (e2 - sh) / np.linalg.norm(e2 - sh)
        hm = O.ring_tube(e2, d, rr[1] * 0.99, rr[1] * 0.99, 0.0105, u_hint=(0, 1, 0))
        hm.set_w("upper_arm_" + sn, 1.0)
        parts += [sl, hm] if sleeve_hem else [sl]
    rx_, ry_, cy_ = B.torso_at(spec, np.array([z1]), off_top)
    nb = O.ring_tube((0, cy_[0], z1), (0, 0, 1), rx_[0], ry_[0], 0.0125)
    nb.set_w("neck", 1.0)
    parts.append(nb)
    p = G.merge(parts)
    p.closed = False
    return p


def trousers(ch, off=0.012, z1=1.02, hem_z=0.12, flare=0.012):
    """Trousers: pelvis tapered into the legs (no crotch bulge / diaper seam), legs with a hem cuff over the shoe, knee
    creases. Returns (trousers Part, creases Part)."""
    TAPER = (0.80, 1.0, 0.42)
    tr = [ch.torso_part(off=off, z0=0.80, z1=z1, cap1=None, taper=TAPER, dz=0.02, nseg=48)]
    creases = []
    for side, sn in ((1, "L"), (-1, "R")):
        lt, end = O.leg_tubes(ch, sn, side, off, ztop_shin=hem_z, cuff=0.0, flare=flare)
        tr += lt
        c, rx0, ry0, bone = O.leg_point(ch, sn, side, hem_z, off)
        cuff = O.ring_tube(c, (0, 0, 1), ry0 + flare, rx0 + flare, 0.0115, u_hint=(1, 0, 0), n=32, ns=8)
        cuff.set_w("shin_" + sn, 1.0)
        tr.append(cuff)
        for z, sag, r_, a, tl in ((0.215, 0.012, 0.0065, 1.2, 0.012), (0.545, 0.008, 0.0065, 1.1, 0.008), (0.505, 0.012, 0.0070, 1.25, -0.010)):
            creases.append(O.fold_arc(ch, sn, side, z, off + 0.002, -a, a, r=r_, sag=sag, tilt=tl))
    p = G.merge(tr)
    p.closed = False
    return p, G.merge(creases)


def wordmark(ch, lines, zc, width, off, line_gap=1.28, depth=1.4):
    """Raised parody wordmark (text outlines converted to mesh, no font file needed at render time) projected on the
    torso front at surface offset `off`; depth scales the v1 extrusion."""
    spec = ch.spec
    pieces = []
    for ln in lines:
        V, F = O.text_part(ln)
        pieces.append((V, F, V[:, 0].max() - V[:, 0].min()))
    sc = width / max(w for _, _, w in pieces)
    h = sc
    n = len(lines)
    parts = []
    for i, (V, F, w) in enumerate(pieces):
        V = V.copy()
        V[:, 0] -= (V[:, 0].max() + V[:, 0].min()) / 2
        V[:, 1] -= (V[:, 1].max() + V[:, 1].min()) / 2
        ext = V[:, 2] - V[:, 2].min()
        xs = V[:, 0] * sc
        zs = V[:, 1] * sc + zc + (n - 1) / 2 * h * line_gap - i * h * line_gap
        out = np.zeros_like(V)
        for k in range(len(V)):
            sp = O.surf_pt(spec, float(xs[k]), float(zs[k]), True, off)
            out[k] = (xs[k], sp[1] - ext[k] * sc * 0.9 * depth + 0.0006, zs[k])
        parts.append(G.Part(out, F, 0, closed=False))
    p = G.merge(parts)
    p.closed = False
    ch.torso_weights(p)
    return p


# ------------------------------------------------------------------------------------------------ hair (v1 head units)
def hair_shell(ch, z_back, z_front, off_fn, zmax=None, back_only_below=None):
    """Hair shell over the head grid: faces below a height limit that rises from z_back (back) to z_front (front) are
    dropped; off_fn(z) gives the offset along the radial direction; zmax flattens the top (flat-top cut)."""
    head = ch.head
    V = head.V0.copy()
    cen = np.array([0, head.cy, 1.62])
    d = G.nrm(V - cen)
    V = V + d * np.asarray(off_fn(V[:, 2]), float)[:, None]
    if zmax is not None:
        V[:, 2] = np.minimum(V[:, 2], zmax)
    p = G.Part(V, head.faces_full, 0, closed=False)

    def pred(c):
        t = G.smoothstep(head.cy + 0.02, head.cy - 0.06, c[:, 1])
        lim = z_back + (z_front - z_back) * t
        return c[:, 2] < lim
    p.drop_faces(pred)
    p.f = _orient_out(p.v, p.f, cen)
    p.set_w("head", 1.0)
    return p


def afro_blobs(ch, r=0.040, off=0.020, z_min=1.655):
    """Curly hair: clay blobs on a regular grid of the upper head (baseline radius r), mapped through T at their
    centres only (so the blobs stay round on the wider v2 head). Returns a Part in BASELINE coordinates."""
    head = ch.head
    NR, NS = head.NR, head.NS
    cen = np.array([0, head.cy, 1.62])
    parts = []
    for i in range(19, NR, 2):
        step = 3 if i < 33 else 5
        for j in range(0, NS, step):
            jj = (j + (i // 2) % 2) % NS
            v = head.V0[head.vid(i, jj)]
            if v[2] < z_min:
                continue
            if v[1] < head.cy - 0.03 and v[2] < 1.695:
                continue
            c = ch.T(v + G.nrm(v - cen) * 0.006)
            dv = G.nrm(c - ch.T(cen))
            s = G.ellipsoid(c + dv * off, (r, r, r * 0.92), nseg=10, nrings=6)
            parts.append(s)
    top = ch.T(np.array([0.0, head.cy, head.zc + head.hh]))
    parts.append(G.ellipsoid(top + np.array([0, 0.005, off * 0.6]), (r * 1.3, r * 1.3, r), nseg=12, nrings=7))
    p = G.merge(parts)
    p.set_w("head", 1.0)
    return p


def ponytail(ch):
    """Ponytail (baseline coordinates) from the back of the head."""
    head, hs = ch.head, ch.spec["head"]
    P = ch.T(np.array([0.0, head.cy + hs["b_back"] * 0.90, head.zc + 0.52 * head.hh]))
    path = np.array([P + np.array([0, -0.012, 0.0]), P + np.array([0, 0.022, -0.050]), P + np.array([0, 0.036, -0.140]),
                     P + np.array([0, 0.036, -0.235])])
    t = G.tube(path, [0.032, 0.036, 0.030, 0.013], [0.030, 0.034, 0.028, 0.012], u=(1, 0, 0), nseg=14, ncap=2)
    band = O.ring_tube(path[0] + np.array([0, 0.012, -0.012]), path[1] - path[0], 0.030, 0.028, 0.008, u_hint=(1, 0, 0), n=16, ns=6)
    t.set_w("head", 1.0)
    band.set_w("head", 1.0)
    return t, band


# ------------------------------------------------------------------------------------------------ face hair (follows the expressions)
def face_shell(ch, sel, name, mats, off=0.0085, solid=0.006):
    """Beard / goatee: a shell over the selected head-grid vertices, offset along the normals. Its 14 expression shape
    keys are generated from the same head deformation as the face (mouth, jaw and cheeks), and it is weighted to the
    head/jaw bones exactly like the head mesh, so it follows every expression and the jaw."""
    head = ch.head
    n = head.n_grid
    N = grid_normals(head)
    faces = [f for f in head.faces if max(f) < n and all(sel[i] for i in f)]
    idx = sorted({i for f in faces for i in f})
    remap = {o: k for k, o in enumerate(idx)}
    F = [tuple(remap[i] for i in f) for f in faces]
    idx = np.array(idx)
    offv = N[idx] * off

    def fn(e):
        return head.deformed(e)[:n][idx] + offv
    V = fn(HD.E("neutral"))
    cen = np.array([0.0, head.cy, head.zc + head.hh * 0.5])
    p = G.Part(V, _orient_out(V, F, cen), 0, closed=False)
    wj = head.jaw_weights()[:n][idx]
    p.only_w({"head": 1.0 - wj, "jaw": wj})
    p.v = ch.T(p.v)
    ob = ch.add(p, name, mats, register=False, solid=solid, subsurf=1)
    HD.add_expression_keys(ob, lambda e: ch.T(fn(e)), ch.root, ch.K, p.v)
    return ob


def beard_select(ch, kind):
    head = ch.head
    n = head.n_grid
    di, adj = head.di[:n], np.abs(head.dj[:n])
    if kind == "full":
        return (adj <= 15) & ((di <= -3) | ((adj >= 7) & (di <= 5)))
    if kind == "goatee":
        return (adj <= 5) & (di <= -3) & (di >= -11)
    raise ValueError(kind)


def moustache(ch, name, mats, off=0.0095, thick=1.0):
    """Moustache tube anchored to head-grid vertices above the upper lip; re-lofted per expression (follows snarl,
    width and corner keys)."""
    head = ch.head
    N = grid_normals(head)
    IM, j0 = head.IM, head.j0
    spec_pts = [(-6, IM - 1), (-5, IM + 1), (-3, IM + 3), (-1, IM + 3), (1, IM + 3), (3, IM + 3), (5, IM + 1), (6, IM - 1)]
    vids = np.array([head.vid(r, j0 + c) for c, r in spec_pts])
    rx = np.array([0.0035, 0.0062, 0.0080, 0.0076, 0.0076, 0.0080, 0.0062, 0.0035]) * thick
    ry = np.array([0.0030, 0.0050, 0.0062, 0.0060, 0.0060, 0.0062, 0.0050, 0.0030]) * thick

    def fn(e):
        P = head.deformed(e)[vids] + N[vids] * off
        return G.tube(P, rx, ry, u=(0, 0, 1), nseg=12, ncap=2).v
    t = G.tube(head.deformed(HD.E("neutral"))[vids] + N[vids] * off, rx, ry, u=(0, 0, 1), nseg=12, ncap=2)
    # weights: jaw share of the nearest anchor vertex (the tips sit on rows the head mesh partly weights to the jaw)
    wj_a = head.jaw_weights()[vids]
    P0 = head.deformed(HD.E("neutral"))[vids]
    near = np.argmin(np.linalg.norm(t.v[:, None, :] - P0[None, :, :], axis=2), axis=1)
    wj = wj_a[near]
    t.only_w({"head": 1.0 - wj, "jaw": wj})
    t.v = ch.T(t.v)
    ob = ch.add(t, name, mats, register=False, subsurf=1)
    HD.add_expression_keys(ob, lambda e: ch.T(fn(e)), ch.root, ch.K, t.v)
    return ob


# ------------------------------------------------------------------------------------------------ glasses / accessories (baseline)
def glasses(ch, shape="round", rx=0.056, rz=0.048, fwd=0.050, tube_r=0.0068):
    cL, cR = ch.eye_centres
    hs = ch.spec["head"]
    zc_v1 = hs["z_chin"] + hs["eye_z"]
    hw = head_halfwidth(ch, zc_v1)
    ears = ear_centres(ch)
    parts = []
    inner = []
    for c, side, ear in ((cL, 1, ears[0]), (cR, -1, ears[1])):
        cc = c + np.array([0, -fwd, 0.004])
        if shape == "round":
            ring = O.ring_tube(cc, (0, -1, 0), rx, rz, tube_r, u_hint=(1, 0, 0), n=28, ns=8)
        else:
            ring = O.rect_frame(cc, 2 * rx, 2 * rz, r=tube_r, plane="xz", n=10)
        parts.append(ring)
        inner.append(cc + np.array([-side * rx, 0, 0.004]))
        p0 = cc + np.array([side * rx, 0, 0.004])
        tp = np.array([p0, np.array([side * (hw + 0.010), cc[1] + 0.045, cc[2] + 0.006]), np.array([side * (hw + 0.008), ear[1] - 0.010, cc[2] - 0.004]),
                       np.array([side * (hw + 0.004), ear[1] + 0.012, cc[2] - 0.030])])
        parts.append(G.tube(tp, tube_r * 0.75, tube_r * 0.75, u=(0, 0, 1), nseg=8, ncap=2))
    mid = (inner[0] + inner[1]) / 2 + np.array([0, -0.004, 0.012])
    parts.append(G.tube(np.array([inner[0], mid, inner[1]]), tube_r * 0.8, tube_r * 0.8, u=(0, 0, 1), nseg=8, ncap=2))
    p = G.merge(parts)
    p.set_w("head", 1.0)
    return p


def headset(ch, side=1):
    """Call-centre headset: band over the crown, ear cups, mic boom to the mouth corner. Returns (dark Part, mic Part)."""
    head = ch.head
    ears = ear_centres(ch)
    N = grid_normals(head)
    NR, NS = head.NR, head.NS
    i_ear = 19
    idx = [head.vid(i, 0) for i in range(i_ear, NR)] + [len(head.V0) - 1] + [head.vid(i, NS // 2) for i in range(NR - 1, i_ear - 1, -1)]
    P = head.V0[idx] + N[idx] * 0.016
    P = ch.T(P)
    band = G.tube(P, 0.013, 0.0065, u=(0, 1, 0), nseg=10, cap0="round", cap1="round", ncap=2)
    cups = []
    for e, s in ((ears[0], 1), (ears[1], -1)):
        cups.append(G.ellipsoid(e + np.array([s * 0.020, 0.0, 0.0]), (0.024, 0.044, 0.050), nseg=16, nrings=10))
    dark = G.merge([band] + cups)
    dark.set_w("head", 1.0)
    # mic boom from the cup front to the mouth corner
    e = ears[0] if side == 1 else ears[1]
    rim = head.V0[head.rim_idx]
    corner_v1 = rim[np.argmax(side * rim[:, 0])]
    corner = ch.T(corner_v1)
    start = e + np.array([side * 0.040, -0.030, -0.030])
    end = corner + np.array([side * 0.022, -0.050, -0.012])
    mid = (start + end) / 2 + np.array([side * 0.030, -0.010, -0.008])
    boom = G.tube(np.array([start, mid, end]), 0.0060, 0.0060, u=(0, 0, 1), nseg=8, ncap=2)
    tip = G.ellipsoid(end + np.array([-side * 0.006, -0.004, 0]), (0.016, 0.014, 0.013), nseg=12, nrings=8)
    boom.set_w("head", 1.0)
    tip.set_w("head", 1.0)
    return dark, boom, tip


def cap(ch, z_back=1.660, z_front=1.700, off=0.021, bone="hat"):
    """Baseball cap: crown shell following the head (v1 units -> T), button, curved brim (baseline). Returns
    (crown Part, brim Part, button Part), all weighted to `bone`."""
    head = ch.head
    crown = hair_shell(ch, z_back, z_front, lambda z: np.full_like(z, off))
    crown.v = ch.T(crown.v)
    crown.set_w(bone, 1.0)
    crown.w = {bone: crown.w[bone]}
    yf = head.surface_y(0.0, z_front)
    Pf = ch.T(np.array([0.0, (yf if yf is not None else head.cy - 0.09) - off, z_front]))
    # brim: half superellipse plate, slightly curved down at the sides and tilted up
    nu, nv = 17, 7
    V = []
    for i in range(nv):
        r = i / (nv - 1)
        for j in range(nu):
            a = math.pi * j / (nu - 1)
            x = -math.cos(a) * 0.118 * (0.62 + 0.38 * r)
            y = -math.sin(a) * 0.160 * r
            z = -0.018 * (abs(x) / 0.118) ** 2
            V.append((x, y, z))
    V = np.array(V)
    F = [(i * nu + j, i * nu + j + 1, (i + 1) * nu + j + 1, (i + 1) * nu + j) for i in range(nv - 1) for j in range(nu - 1)]
    R = G.rot_about((1, 0, 0), 0.06)
    V = V @ R.T + Pf + np.array([0, 0.012, 0.002])
    brim = G.Part(V, F, 0, closed=False)
    brim.set_w(bone, 1.0)
    topz = float(crown.v[:, 2].max())
    ctop = crown.v[np.argmax(crown.v[:, 2])]
    btn = G.ellipsoid(np.array([ctop[0], ctop[1], topz + 0.004]), (0.014, 0.014, 0.008), nseg=10, nrings=6)
    btn.set_w(bone, 1.0)
    return crown, brim, btn


def lanyard(ch, z_badge=1.07, off=0.030, x_out=0.165):
    """Lanyard cords from the back of the neck over the shoulders, down the chest edges (outside the wordmark) to a
    badge on the belly. Returns (cord Part, badge Part, clip Part)."""
    spec = ch.spec
    surf = O.surf_pt
    cords = []
    for side in (1, -1):
        pts = [surf(spec, side * 0.050, 1.395, False, 0.022), np.array([side * 0.095, 0.020, 1.405]),
               surf(spec, side * 0.120, 1.370, True, off - 0.004), surf(spec, side * x_out, 1.28, True, off),
               surf(spec, side * (x_out - 0.010), 1.215, True, off), surf(spec, side * 0.050, z_badge + 0.075, True, off),
               surf(spec, side * 0.012, z_badge + 0.052, True, off)]
        cords.append(G.tube(np.array(pts), 0.0075, 0.0030, u=(0.0, 0.0, 1.0), nseg=8, cap0="round", cap1="round", ncap=2))
    back = G.tube(np.array([surf(spec, 0.050, 1.395, False, 0.022), surf(spec, 0.0, 1.392, False, 0.023), surf(spec, -0.050, 1.395, False, 0.022)]),
                  0.0075, 0.0030, u=(0, 0, 1), nseg=8, ncap=2)
    cords.append(back)
    cord = G.merge(cords)
    ch.torso_weights(cord)
    c = surf(spec, 0.0, z_badge, True, off + 0.004)
    badge = G.ellipsoid(c, (0.040, 0.0045, 0.052), nseg=16, nrings=8, e=0.30)
    stripe = G.ellipsoid(c + np.array([0, -0.0035, 0.030]), (0.036, 0.0030, 0.012), nseg=12, nrings=6, e=0.30)
    clip = G.ellipsoid(surf(spec, 0.0, z_badge + 0.056, True, off + 0.002), (0.012, 0.005, 0.008), nseg=10, nrings=6, e=0.5)
    for q in (badge, stripe, clip):
        ch.torso_weights(q)
    return cord, badge, stripe, clip


def backpack(ch):
    """Backpack (pack body, front pocket, top handle, two shoulder straps). Returns (pack Part, strap Part)."""
    spec = ch.spec
    surf = O.surf_pt
    b = surf(spec, 0.0, 1.17, False, 0.02)
    w = 0.150 * spec.get("wide", 1.0)
    pack = G.ellipsoid(b + np.array([0, 0.080, 0]), (w, 0.080, 0.190), nseg=20, nrings=12, e=0.45)
    pocket = G.ellipsoid(b + np.array([0, 0.150, -0.070]), (w * 0.70, 0.030, 0.085), nseg=16, nrings=8, e=0.45)
    handle = G.tube(np.array([b + np.array([-0.035, 0.050, 0.180]), b + np.array([0, 0.060, 0.215]), b + np.array([0.035, 0.050, 0.180])]),
                    0.008, 0.008, u=(0, 1, 0), nseg=8, ncap=2)
    straps = []
    for side in (1, -1):
        zs = np.linspace(1.30, 1.47, 300)
        rx, ry, cy = B.torso_at(spec, zs, 0.030)
        zt = float(zs[int(np.argmin(np.abs(rx - 0.125)))])
        pts = [b + np.array([side * 0.085, 0.020, 0.160]), surf(spec, side * 0.115, 1.385, False, 0.032), np.array([side * 0.125, 0.0, zt + 0.004]),
               surf(spec, side * 0.130, 1.360, True, 0.030), surf(spec, side * 0.145, 1.24, True, 0.028), surf(spec, side * 0.165, 1.12, True, 0.026),
               surf(spec, side * 0.180, 1.06, True, 0.024)]
        straps.append(G.tube(np.array(pts), 0.019, 0.0055, u=(1, 0, 0), nseg=10, cap0="round", cap1="round", ncap=2))
    for q in [pack, pocket, handle] + straps:
        ch.torso_weights(q)
    return G.merge([pack, pocket, handle]), G.merge(straps)


# ------------------------------------------------------------------------------------------------ leather moto jacket (NVYDIA)
def leather_jacket(ch, z0=0.88, z_neck=1.395, z_open=1.10, hw_top=0.112, joff=0.030):
    """Black clay leather moto jacket on the v2 body: shell open above z_open (V shows the shirt), wide lapels with
    snap studs, fold-over collar, belted hem band, long sleeves with zip cuffs, centre zipper (closed part + teeth along
    the V edges + pull tab), slanted hip-pocket zips. Returns dict of Parts (leather, collar, metal)."""
    spec, J = ch.spec, ch.J
    surf = O.surf_pt

    def hw(z):
        t = np.clip((z - z_open) / (z_neck - z_open), 0.0, 1.0)
        return hw_top * t ** 0.70 if z > z_open else 0.0
    shell = O.jacket_shell(ch, joff, z0, z_neck, hw, n=60, dz=0.02, flare=lambda z: 0.010 * float(G.smoothstep(0.98, 0.88, z)))
    ar = spec["arm_r"]
    sleeves = []
    for side, sn in ((1, "L"), (-1, "R")):
        sx = np.array([side, 1, 1])
        sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
        end = wr - (wr - el) * 0.10
        ua = G.tube(np.array([sh, (sh + el) / 2, el]), ch.ARM_UA[0] * ar + np.array([0.012, 0.018, 0.020]), ch.ARM_UA[1] * ar + np.array([0.012, 0.018, 0.020]), u=(0, 1, 0), nseg=26)
        ua.set_w("upper_arm_" + sn, 1.0)
        fa = G.tube(np.array([el, (el + end) / 2, end]), ch.ARM_FA[0] * ar + 0.020, ch.ARM_FA[1] * ar + 0.020, u=(0, 1, 0), nseg=26, cap1=None)
        fa.set_w("forearm_" + sn, 1.0)
        d = (end - el) / np.linalg.norm(end - el)
        cf = O.ring_tube(end - d * 0.012, d, ch.ARM_FA[0][2] * ar + 0.020, ch.ARM_FA[0][2] * ar + 0.020, 0.011, u_hint=(0, 1, 0))
        cf.set_w("forearm_" + sn, 1.0)
        sleeves += [ua, fa, cf]
    rx, ry, cy = B.torso_at(spec, np.array([z0]), joff)
    th = np.linspace(0, 2 * math.pi, 60, endpoint=False)
    hem = G.tube(np.array([[(rx[0] + 0.012) * math.cos(t), cy[0] + (ry[0] + 0.012) * math.sin(t), z0 + 0.016] for t in th]), 0.022, 0.011,
                 u=(0, 0, 1), nseg=10, closed_path=True)
    ch.torso_weights(hem)
    leather = G.merge([shell, hem] + sleeves)
    leather.closed = False
    # lapels: wide ribbons along the V edges (moto: broad, pointed at the bottom), snap studs at the tips
    lap, metal = [], []
    for side in (1, -1):
        zs = np.linspace(z_neck - 0.004, z_open + 0.020, 12)
        pts, rr = [], []
        for z in zs:
            t = (z_neck - z) / (z_neck - z_open)
            w = 0.030 + 0.050 * math.sin(math.pi * min(t * 1.05, 1.0)) ** 0.8
            pts.append(surf(spec, side * (float(hw(z)) + w * 0.5), z, True, joff + 0.012))
            rr.append(w * 0.5)
        lp = G.tube(np.array(pts), np.array(rr), np.full(len(zs), 0.0085), u=(1, -0.15, 0), nseg=10, cap0="round", cap1="round", ncap=2)
        ch.torso_weights(lp)
        lap.append(lp)
        zt = z_open + 0.13
        metal.append(G.ellipsoid(surf(spec, side * (float(hw(zt)) + 0.062), zt, True, joff + 0.022), (0.0075, 0.0045, 0.0075), nseg=10, nrings=6))
        # zipper teeth along the V edge
        ze = np.linspace(z_open, z_neck - 0.01, 10)
        tp = np.array([surf(spec, side * (float(hw(z)) + 0.004), z, True, joff + 0.006) for z in ze])
        metal.append(G.tube(tp, 0.0042, 0.0024, u=(1, 0, 0), nseg=8, cap0="flat", cap1="flat"))
        # slanted hip-pocket zip
        pp = np.array([surf(spec, side * x, z, True, joff + 0.004) for x, z in ((0.080, 1.05), (0.130, 1.00), (0.175, 0.965))])
        metal.append(G.tube(pp, 0.0040, 0.0022, u=(0, 0, 1), nseg=8, cap0="round", cap1="round"))
    zc = np.linspace(z0 + 0.02, z_open, 8)
    metal.append(G.tube(np.array([surf(spec, 0.0, z, True, joff + 0.005) for z in zc]), 0.0050, 0.0025, u=(1, 0, 0), nseg=8, cap0="flat", cap1="flat"))
    metal.append(G.ellipsoid(surf(spec, 0.0, z_open - 0.024, True, joff + 0.012), (0.0080, 0.0040, 0.019), nseg=12, nrings=8))
    for q in metal:
        ch.torso_weights(q)
    # fold-over collar around the back of the neck, joined to the lapel tops
    ang = np.linspace(-2.45, 2.45, 22)
    rxc, ryc, cyc = B.torso_at(spec, np.array([z_neck]), 0.0)
    cpath = np.array([[0.108 * math.sin(a), cyc[0] + 0.012 + 0.098 * math.cos(a), z_neck - 0.002] for a in ang])
    col = G.tube(cpath, 0.024, 0.010, u=(0, 0, 1), nseg=10, cap0="round", cap1="round", ncap=2)
    col.set_w("chest", 0.5)
    col.add_w("neck", 0.5)
    return dict(leather=leather, lapels=G.merge(lap), collar=col, metal=G.merge(metal), hw=hw)
