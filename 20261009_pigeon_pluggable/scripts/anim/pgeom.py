"""Pigeon geometry and eye-aim math (pure Python, no bpy; usable in Blender and in `uv run python`).

Bird frame (same as the film builds, DevLog-006): origin on the ground between the feet, +X = beak (forward),
+Y = bird LEFT (RX eye, cyan), -Y = bird RIGHT (TX eye, red), +Z up. Units: metres at HERO scale
(the GLB is true size, 131.9 mm beak to tail); world size = hero metres x `scale` (film hall scale S_BIRD = 2.6).

Head rotation (pivot HEAD_PIVOT, Blender euler XYZ on the head empty): R = Rz(yaw) Ry(nod) Rx(roll).
  yaw  > 0 turns the beak toward the bird's left (+Y);
  nod  > 0 tips the beak down;
  roll > 0 tilts the head so the RX (left) eye looks up and the TX (right) eye looks down.
Eye axes are purely lateral (hero GLB hooks), so a lateral eye is aimed by yaw (bearing) + roll (elevation);
nod does not change where an eye points (it rotates about the eye axis) and is free for expression.
World yaw of a bird: beak direction = (cos yaw, sin yaw, 0).
"""
import math

HEAD_PIVOT = (0.036, 0.0, 0.070)       # neck/head yaw pivot (same as DevLog-006 P_H)
BODY_PIVOT = (0.010, 0.0, 0.016)       # hip line: body pitch/crouch pivot (legs counter-rotate here)
BREATH_PIVOT = (0.0, 0.0, 0.040)       # chest centre for breathing scale
WING_PIVOT = {"L": (0.026, 0.019, 0.050), "R": (0.026, -0.019, 0.050)}   # shoulders
TAIL_PIVOT = (-0.038, 0.0, 0.020)
LEG_PIVOT = {"L": (0.010, 0.0073, 0.016), "R": (0.010, -0.0073, 0.016)}  # hips (top of tarsus); L = RX side, R = TX side
EYE = {"tx": (0.049, -0.01021, 0.0815), "rx": (0.049, 0.01021, 0.0815)}  # hero GLB HOOK_eye_tx / HOOK_eye_rx (2026-10-10 02:54 build)
EYE_SIGN = {"tx": -1.0, "rx": 1.0}     # eye axis = (0, sign, 0) in the head frame
EYE_R = 0.0047                          # bezel outer radius (lid size)
SIDE_OF_EYE = {"tx": "R", "rx": "L"}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rotx(a, v):
    c, s = math.cos(a), math.sin(a)
    return (v[0], c * v[1] - s * v[2], s * v[1] + c * v[2])


def roty(a, v):
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


def rotz(a, v):
    c, s = math.cos(a), math.sin(a)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def head_rot(yaw, nod, roll, v):
    return rotz(yaw, roty(nod, rotx(roll, v)))


def eye_local(k, yaw=0.0, roll=0.0, nod=0.0, bob=0.0, eye=None, pivot=HEAD_PIVOT):
    """Eye centre in the bird frame (hero metres) for a head pose."""
    e = eye or EYE[k]
    return add(add(pivot, (bob, 0.0, 0.0)), head_rot(yaw, nod, roll, sub(e, pivot)))


def eye_world(pos, body_yaw, scale, k, yaw=0.0, roll=0.0, nod=0.0, bob=0.0, eye=None, pivot=HEAD_PIVOT):
    return add(pos, rotz(body_yaw, mul(eye_local(k, yaw, roll, nod, bob, eye, pivot), scale)))


def axis_world(body_yaw, k, yaw=0.0, roll=0.0, nod=0.0):
    return rotz(body_yaw, head_rot(yaw, nod, roll, (0.0, EYE_SIGN[k], 0.0)))


def aim(pos, body_yaw, scale, k, target, yaw=0.0, roll=0.0, iters=6, eye=None, pivot=HEAD_PIVOT):
    """Head (yaw, roll) that points lateral eye k exactly at target (fixed-point: the eye moves with the head). Unclamped."""
    s = EYE_SIGN[k]
    for _ in range(iters):
        d = sub(target, eye_world(pos, body_yaw, scale, k, yaw, roll, 0.0, 0.0, eye, pivot))
        n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1e-9
        bearing = math.atan2(d[1], d[0])
        yaw = wrap(bearing - body_yaw - s * math.pi / 2)
        roll = s * math.asin(max(-1.0, min(1.0, d[2] / n)))
    return yaw, roll


def aim_error(pos, body_yaw, scale, k, target, yaw, roll, eye=None, pivot=HEAD_PIVOT):
    """Angle (rad) between eye k's axis and the direction to target."""
    e = eye_world(pos, body_yaw, scale, k, yaw, roll, 0.0, 0.0, eye, pivot)
    d = sub(target, e)
    n = math.sqrt(sum(x * x for x in d)) or 1e-9
    a = axis_world(body_yaw, k, yaw, roll)
    c = sum(x * y for x, y in zip(a, d)) / n
    return math.acos(max(-1.0, min(1.0, c)))


def clamp(v, lim):
    return max(-lim, min(lim, v))


def stare(pos, body_yaw, scale, target, yaw_lim, roll_lim, prefer=None, eyes=None, pivot=HEAD_PIVOT):
    """One-eyed stare at target: the eye that needs the smaller head yaw (or `prefer`); clamped. Returns (eye, yaw, roll, exact).
    eyes: optional {'tx': xyz, 'rx': xyz} (bird frame, hero m) for a shifted head (neck extension)."""
    best = None
    for k in ((prefer,) if prefer else ("tx", "rx")):
        y, r = aim(pos, body_yaw, scale, k, target, eye=(eyes or {}).get(k), pivot=pivot)
        exact = abs(y) <= yaw_lim and abs(r) <= roll_lim
        cand = (abs(y) + (0 if exact else 10.0), k, clamp(y, yaw_lim), clamp(r, roll_lim), exact)
        if best is None or cand[0] < best[0]:
            best = cand
    return best[1], best[2], best[3], best[4]


def solve_link(a, b, iters=5):
    """a, b: dicts with pos, yaw (body), scale, optional eye_tx / eye_rx / pivot (bird frame, hero m).
    Bird a's TX eye -> bird b's RX eye. Returns (ya, ra, yb, rb)."""
    ya = ra = yb = rb = 0.0
    pa, pb = a.get("pivot") or HEAD_PIVOT, b.get("pivot") or HEAD_PIVOT
    for _ in range(iters):
        tb = eye_world(b["pos"], b["yaw"], b["scale"], "rx", yb, rb, eye=b.get("eye_rx"), pivot=pb)
        ya, ra = aim(a["pos"], a["yaw"], a["scale"], "tx", tb, ya, ra, eye=a.get("eye_tx"), pivot=pa)
        ta = eye_world(a["pos"], a["yaw"], a["scale"], "tx", ya, ra, eye=a.get("eye_tx"), pivot=pa)
        yb, rb = aim(b["pos"], b["yaw"], b["scale"], "rx", ta, yb, rb, eye=b.get("eye_rx"), pivot=pb)
    return ya, ra, yb, rb


def neck_shift(ext_world, scale):
    """Head pivot and eyes (bird frame, hero m) for a forward neck extension of ext_world metres (world)."""
    d = ext_world / scale
    return (add(HEAD_PIVOT, (d, 0.0, 0.0)), {k: add(EYE[k], (d, 0.0, 0.0)) for k in EYE})


# ---------------------------------------------------------------------------------------------- curves
def catmull(pts, n_per=24):
    """Centripetal-ish uniform Catmull-Rom through pts (list of 3-tuples); returns dense polyline."""
    if len(pts) < 2:
        return list(pts)
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for j in range(n_per):
            t = j / n_per
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[c]) + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2
                                     + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3) for c in range(3)))
    out.append(tuple(pts[-1]))
    return out


def arclen(poly):
    s = [0.0]
    for a, b in zip(poly[:-1], poly[1:]):
        s.append(s[-1] + dist(a, b))
    return s


def sample_at(poly, s_tab, s):
    """Point and unit tangent at arc length s."""
    if s <= 0:
        i = 0
    elif s >= s_tab[-1]:
        i = len(poly) - 2
    else:
        lo, hi = 0, len(s_tab) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if s_tab[m] <= s:
                lo = m
            else:
                hi = m
        i = lo
    a, b = poly[i], poly[i + 1]
    seg = (s_tab[i + 1] - s_tab[i]) or 1e-9
    u = max(0.0, min(1.0, (s - s_tab[i]) / seg))
    p = tuple(a[c] + (b[c] - a[c]) * u for c in range(3))
    d = sub(b, a)
    n = math.sqrt(sum(x * x for x in d)) or 1e-9
    return p, mul(d, 1.0 / n)


# ---------------------------------------------------------------------------------------------- neck deform (owner F1)
# The neck is the band between the body shell and the head (bird frame, hero m). d0 = distance outside the body shell
# profile (measured on the hero GLB, side profile at y ~ 0, polar radius about (0, 40 mm)); d1 = distance outside the
# head (rounded box around the cranium + jaw). t = d0 / (d0 + d1): 0 on/under the body shell, 1 on/inside the head.
# Neck bones all rotate about the head pivot (the head stays exactly where the existing API puts it) by a fraction of
# the head rotation that grows with t (pigeon-like: most rotation near the head); translations (bob, tier neck
# extension) are distributed linearly along t (the neck stretches).
def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


_BODY_C = (0.0, 0.040)
_BODY_PROF = sorted((math.atan2(z - 40.0, x), math.hypot(x, z - 40.0) / 1000.0) for x, z in (
    (33.1, 20.8), (39.0, 25.5), (40.0, 27.2), (40.7, 39.1), (41.9, 47.1), (40.5, 53.5), (40.1, 55.2), (34.6, 60.1),
    (27.0, 64.8), (22.2, 66.7), (13.2, 66.1), (8.3, 64.7), (2.8, 62.4), (-1.1, 60.5), (-4.8, 58.3), (-6.6, 57.0),
    (-10.2, 54.3), (-11.9, 52.9), (-15.4, 49.9), (-17.1, 48.4)))   # hero GLB 2026-10-10, side profile (mm)
_HEAD_C, _HEAD_HALF, _HEAD_R = (0.044, 0.0815), (0.0143, 0.0123), 0.006


def body_dist(p):
    """Radial distance of p outside the body shell side profile (negative inside)."""
    dx, dz = p[0] - _BODY_C[0], p[2] - _BODY_C[1]
    a = math.atan2(dz, dx)
    r = math.hypot(dx, dz)
    pr = _BODY_PROF
    if a <= pr[0][0]:
        rb = pr[0][1]
    elif a >= pr[-1][0]:
        rb = pr[-1][1]
    else:
        for (a0, r0), (a1, r1) in zip(pr[:-1], pr[1:]):
            if a0 <= a <= a1:
                rb = r0 + (r1 - r0) * (a - a0) / ((a1 - a0) or 1e-9)
                break
    return r - rb


def head_dist(p):
    """Distance of p outside the head (rounded box in xz around the cranium + jaw; negative inside)."""
    qx = abs(p[0] - _HEAD_C[0]) - (_HEAD_HALF[0] - _HEAD_R)
    qz = abs(p[2] - _HEAD_C[1]) - (_HEAD_HALF[1] - _HEAD_R)
    out = math.hypot(max(qx, 0.0), max(qz, 0.0))
    return out + min(max(qx, qz), 0.0) - _HEAD_R


# (bone, t centre, rotation fraction, translation fraction)
NECK_BONES = (("neck_root", 0.0, 0.0, 0.0), ("neck_1", 0.25, 0.095, 0.25), ("neck_2", 0.5, 0.31, 0.5),
              ("neck_3", 0.75, 0.61, 0.75), ("neck_head", 1.0, 1.0, 1.0))


def neck_t(p, head_pivot=None):
    """Position along the neck: 0 at/under the body shell, 1 at/inside the head. head_pivot (bird frame, hero m) shifts
    the head box for a neck extension."""
    d0 = body_dist(p)
    hp = p if head_pivot is None else (p[0] - (head_pivot[0] - HEAD_PIVOT[0]), p[1], p[2] - (head_pivot[2] - HEAD_PIVOT[2]))
    d1 = head_dist(hp)
    if d0 <= 0.0:
        return 0.0
    if d1 <= 0.0:
        return 1.0
    return d0 / (d0 + d1)


def _smooth(e0, e1, x):
    u = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return u * u * (3 - 2 * u)


def neck_gate(p):
    """1 inside the neck/head column, 0 outside (smooth), for meshes that are not pure neck skins (flexes, straps, seams)."""
    return (1.0 - _smooth(0.020, 0.027, abs(p[1]))) * _smooth(0.034, 0.040, p[2])


def neck_weights(p, pure_neck=True):
    """{bone: weight} for a bird-frame point. pure_neck: the vertex belongs to a neck skin (no column gate)."""
    t = neck_t(p)
    g = 1.0 if pure_neck else neck_gate(p)
    w = {}
    cs = NECK_BONES
    for (na, ta, _, _), (nb, tb, _, _) in zip(cs[:-1], cs[1:]):
        if ta <= t <= tb:
            u = (t - ta) / (tb - ta)
            w[na] = (1 - u) * g
            w[nb] = u * g
            break
    w["neck_root"] = w.get("neck_root", 0.0) + (1.0 - g)
    return {k: v for k, v in w.items() if v > 1e-4}
