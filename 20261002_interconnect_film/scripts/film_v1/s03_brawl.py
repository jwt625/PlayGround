"""S3 brawl choreography (v1.1): two pairs, TERAHOP vs MOLEXX (left) and NUBISS vs AYARR (right), scene-local 1.7-7.3 s.

Beats write canonical (lead-left) pose keys into each fighter's Timeline, move the root paths (world x only; y = 1.4 m)
and register hit events (time, kind, point) used for dust / crumb particles, flying props and camera shake.
Geometry: partners stand on the x axis, fighters face them at 0.9 rad, so the unit vector toward the partner in the canonical
frame is (sin 0.9, -cos 0.9)... computed per beat from the actual root positions.
"""
import math

import numpy as np

from s03_fight import Fighter, hv, nrm, v

Y_ROW = 1.4
FIGHT_YAW = 0.9
EVENTS = []   # dicts: t, kind, att, vic, s (strength 0..1), pt (world xyz), dir (+1/-1 along x, away from the attacker)

FIST, OPEN, POINT = np.full(5, 1.0), np.full(5, 0.08), np.array([0.0, 0.95, 0.95, 0.95, 0.6])


def cu(x):
    return np.full(5, float(x))


# ------------------------------------------------------------------------------------------------ poses
def G(**o):
    """Fighting guard (canonical lead-left)."""
    p = dict(hips_loc=v(0, -0.02, -0.07), spine=v(0.14, 0.22, 0), head=v(0.08, 0, 0), jaw=0.15,
             hand_L=hv((0.22, -0.30, 1.36), (0, -0.8, 0.6), (-1, 0, 0)), hand_R=hv((-0.08, -0.26, 1.32), (0, -0.8, 0.6), (1, 0, 0)),
             curl_L=FIST, curl_R=FIST, foot_L=v(0.17, -0.22, 0, 0), foot_R=v(0.17, 0.16, 0, 0))
    p.update(o)
    return p


def HANG(**o):
    """Spent, panting stance (end of the brawl)."""
    p = dict(hips_loc=v(0, 0.02, -0.04), spine=v(0.22, 0.1, 0), head=v(0.18, 0, 0), jaw=0.25,
             hand_L=hv((0.30, -0.10, 0.80), (0.05, -0.2, -1), (-1, 0, 0)), hand_R=hv((-0.30, -0.10, 0.80), (-0.05, -0.2, -1), (1, 0, 0)),
             curl_L=cu(0.5), curl_R=cu(0.5), foot_L=v(0.16, -0.10, 0, 0), foot_R=v(0.16, 0.08, 0, 0))
    p.update(o)
    return p


def sg(a, b, t):
    return 1.0 if b.pos(t)[0] > a.pos(t)[0] else -1.0


def shoulder(hand, hips, flex, twist):
    s = 0.19 if hand == "L" else -0.19
    c, sn = math.cos(twist), math.sin(twist)
    x = s * c
    y = s * sn
    return np.array([hips[0] + x, hips[1] + y - 0.45 * flex, 1.405 + hips[2]])


def contact(a, b, t, hz, so=0.10, hand="L", hips=(0, -0.2, -0.1), flex=0.35, twist=0.3, lateral=0.0, reach=0.62):
    """Canonical hand target for a strike from a onto b at height hz (baseline of b), stopping 'so' metres short of b's axis."""
    p = b.pos(t)
    c = a.to_canon(t, np.array([p[0], p[1], hz * b.K]))
    dirn = nrm(c[:2])
    c[:2] = c[:2] - dirn * so / a.K
    c[:2] = c[:2] + np.array([-dirn[1], dirn[0]]) * lateral  # perpendicular offset (positive = toward the character's left)
    sh = shoulder(hand, hips, flex, twist)
    r = c - sh
    L = np.linalg.norm(r)
    if L > reach:
        c = sh + r / L * reach
    return c, dirn


def event(t, kind, a, b, s, hz=1.3, dirn=None):
    p = b.pos(t)
    d = 1.0 if p[0] > a.pos(t)[0] else -1.0
    EVENTS.append(dict(t=t, kind=kind, att=a.id, vic=b.id, s=s, pt=(p[0] - d * 0.10, p[1] - 0.10, hz * b.K), dir=d))
    b.jiggles.append((t, 0.10 * s + 0.03))
    a.jiggles.append((t, 0.03 * s))
    b.expr.append((t, "p_expr_shock", 0.8 * min(1.0, s + 0.3)))
    b.expr.append((t + 0.22, "p_expr_shock", 0.0))


def go(path, t0, t1, p, e="smooth"):
    """Hold the path until t0, then move to p by t1."""
    last_t = path.keys[-1][0]
    if t0 > last_t + 1e-6:
        path.add(t0, path.last(), "lin")
    path.add(t1, p, e)


# ------------------------------------------------------------------------------------------------ beats
def setup():
    EVENTS.clear()


def start_pose(a, t):
    a.tl.add(t, dict(foot_L=v(0.10, 0, 0, 0), foot_R=v(0.10, 0, 0, 0)), "smooth")


def argue(a, b, t0, t1, off=0.0, per=0.24, hz=1.28, hand_up=True):
    """Shouting match: head thrusts and finger jabs at the partner's chest on a 0.24 s rhythm."""
    t = t0 + off
    k = 0
    while t + per <= t1 + 1e-6:
        pull = G(hips_loc=v(0, 0.03, -0.09), spine=v(0.05, 0.15, 0), head=v(-0.06, 0.05, 0), jaw=0.55,
                 hand_L=hv((0.24, -0.20, 1.30), (0, -0.5, 0.8), (-1, 0, 0)), curl_L=FIST,
                 hand_R=hv((-0.22, -0.20, 1.52), (0, -0.2, 1), (1, 0, 0)), curl_R=FIST)
        c, dn = contact(a, b, t + per * 0.5, hz, so=0.07, hand="L", hips=(0, -0.10, -0.04), flex=0.30, twist=0.3)
        fdir = v(dn[0], dn[1], -0.15)
        jab = G(hips_loc=v(0, -0.11, -0.04), spine=v(0.30, 0.30, 0), head=v(0.24, 0.1, 0), jaw=0.15,
                hand_L=hv(c, fdir, (-0.6, 0, -0.8)), curl_L=POINT,
                hand_R=hv((-0.18, -0.34, 1.58), (0, -0.2, 1), (1, 0, 0)), curl_R=FIST)
        a.tl.add(t, pull, "smooth")
        a.tl.add(t + per * 0.5, jab, "snap")
        t += per
        k += 1
    return t


def bump(a, b, t, xa, xb):
    """Chest bump at t: both step in, chests squash, both recoil with 'what?!' hands."""
    for f in (a, b):
        go(f.path, t - 0.35, t, (xa if f is a else xb, Y_ROW), "in")
    for f in (a, b):
        f.tl.add(t - 0.14, G(hips_loc=v(0, 0.06, -0.10), spine=v(0.02, 0.3, 0), head=v(-0.1, 0, 0), jaw=0.4,
                             hand_L=hv((0.26, -0.10, 1.12), (0, -0.4, 0.6), (-1, 0, 0)), hand_R=hv((-0.26, -0.10, 1.12), (0, -0.4, 0.6), (1, 0, 0))), "smooth")
        f.tl.add(t, G(hips_loc=v(0, -0.20, -0.08), spine=v(0.34, 0.2, 0), head=v(0.2, 0.0, 0), jaw=0.3,
                      hand_L=hv((0.34, -0.12, 1.15), (0, -0.4, 0.6), (-1, 0, 0)), hand_R=hv((-0.34, -0.12, 1.15), (0, -0.4, 0.6), (1, 0, 0)),
                      sqc=v(1.12, 0.88, 1.10), sqh=v(1.05, 0.95, 1.05)), "in")
        f.tl.hold(t + 0.10)
    s = sg(a, b, t)
    go(a.path, t + 0.10, t + 0.40, (xa - s * 0.22, Y_ROW), "out")
    go(b.path, t + 0.10, t + 0.40, (xb + s * 0.22, Y_ROW), "out")
    for f in (a, b):
        f.tl.add(t + 0.32, G(hips_loc=v(0, 0.10, -0.06), spine=v(-0.08, 0.1, 0), head=v(-0.1, 0.2, 0), jaw=0.5,
                             hand_L=hv((0.40, -0.25, 1.10), (0, -0.6, 0.4), (0, 0, 1)), hand_R=hv((-0.40, -0.25, 1.10), (0, -0.6, 0.4), (0, 0, 1)),
                             curl_L=OPEN, curl_R=OPEN, sqc=v(0.97, 1.04, 0.97)), "out")
    event(t, "bump", a, b, 0.6)
    event(t, "bump", b, a, 0.6)


def headbutt(a, b, t, xa, xb):
    """a (small) charges head-first into b's belly; b doubles over and is pushed back."""
    s = sg(a, b, t)
    go(a.path, t - 0.30, t, (xa, Y_ROW), "in")
    go(b.path, t, t + 0.12, (xb, Y_ROW), "lin")
    go(b.path, t + 0.10, t + 0.40, (xb + s * 0.18, Y_ROW), "out")
    a.tl.add(t - 0.22, G(hips_loc=v(0, 0.10, -0.14), spine=v(-0.15, 0.2, 0), head=v(0.1, 0, 0),
                         hand_L=hv((0.26, 0.0, 1.10), (0, -0.4, 0.6), (-1, 0, 0)), hand_R=hv((-0.26, 0.0, 1.10), (0, -0.4, 0.6), (1, 0, 0))), "smooth")
    a.tl.add(t, G(hips_loc=v(0, -0.28, -0.22), spine=v(0.75, 0.15, 0), head=v(0.35, 0, 0), jaw=0.0,
                  hand_L=hv((0.34, -0.20, 1.02), (0, -0.4, 0.4), (-1, 0, 0)), hand_R=hv((-0.34, -0.20, 1.02), (0, -0.4, 0.4), (1, 0, 0)),
                  sqc=v(0.95, 1.06, 0.95)), "in")
    a.tl.hold(t + 0.10)
    a.tl.add(t + 0.35, G(hips_loc=v(0, -0.12, -0.12), spine=v(0.45, 0.2, 0), head=v(0.0, 0, 0)), "out")
    b.tl.add(t - 0.05, G(), "smooth")
    b.tl.add(t, G(hips_loc=v(0, 0.12, -0.10), spine=v(0.55, 0.0, 0), head=v(0.3, 0, 0), jaw=0.5,
                  hand_L=hv((0.28, -0.25, 1.00), (0, -0.4, 0.2), (0, 0, 1)), hand_R=hv((-0.28, -0.25, 1.00), (0, -0.4, 0.2), (0, 0, 1)),
                  curl_L=OPEN, curl_R=OPEN, sqb=v(1.10, 0.86, 1.14), sqc=v(1.05, 0.93, 1.05)), "in")
    b.tl.hold(t + 0.10)
    b.tl.add(t + 0.40, G(hips_loc=v(0, 0.14, -0.08), spine=v(0.25, -0.1, 0), head=v(0.1, 0, 0), jaw=0.45), "out")
    event(t, "headbutt", a, b, 0.7, hz=1.05)


def shove(a, b, t, xa, xb, slide=0.42, hz=1.28, strength=1.0):
    s = sg(a, b, t)
    go(a.path, t - 0.30, t, (xa, Y_ROW), "in")
    go(b.path, t, t + 0.10, (xb, Y_ROW), "lin")
    go(b.path, t + 0.10, t + 0.46, (xb + s * slide, Y_ROW), "out")
    go(a.path, t + 0.10, t + 0.40, (xa + s * 0.10, Y_ROW), "out")
    hips_s, flex_s, tw = (0, -0.22, -0.14), 0.42, 0.40
    cL, dn = contact(a, b, t, hz, so=0.12, hand="L", hips=hips_s, flex=flex_s, twist=tw, lateral=0.11)
    cR, _ = contact(a, b, t, hz, so=0.12, hand="R", hips=hips_s, flex=flex_s, twist=tw, lateral=-0.11)
    wnd = G(hips_loc=v(0, 0.09, -0.11), spine=v(-0.10, 0.10, 0), head=v(0.22, 0, 0), jaw=0.3,
            hand_L=hv((0.16, -0.10, 1.24), (0, -0.4, 0.9), (0, -1, 0)), hand_R=hv((-0.16, -0.10, 1.24), (0, -0.4, 0.9), (0, -1, 0)), curl_L=OPEN, curl_R=OPEN)
    a.tl.add(t - 0.28, wnd, "wind")
    a.tl.add(t, G(hips_loc=v(*hips_s), spine=v(flex_s, tw, 0), head=v(0.28, 0, 0), jaw=0.25, hand_L=hv(cL, (dn[0], dn[1], 0.8), (0, -1, 0)),
                  hand_R=hv(cR, (dn[0], dn[1], 0.8), (0, -1, 0)), curl_L=OPEN, curl_R=OPEN, sqc=v(0.96, 1.05, 0.96)), "in")
    a.tl.hold(t + 0.10)
    a.tl.add(t + 0.26, G(hips_loc=v(0, -0.30, -0.17), spine=v(0.5, 0.35, 0), head=v(0.3, 0, 0), jaw=0.25,
                         hand_L=hv(cL + np.array([dn[0], dn[1], 0]) * 0.06, (dn[0], dn[1], 0.6), (0, -1, 0)),
                         hand_R=hv(cR + np.array([dn[0], dn[1], 0]) * 0.06, (dn[0], dn[1], 0.6), (0, -1, 0)), curl_L=OPEN, curl_R=OPEN), "out")
    b.tl.add(t - 0.05, G(jaw=0.5), "smooth")
    b.tl.add(t, G(hips_loc=v(0, 0.08, -0.05), spine=v(-0.22, -0.10, 0), head=v(-0.28, 0.2, 0), jaw=0.45,
                  hand_L=hv((0.30, -0.05, 1.20), (0.3, -0.2, 1), (-0.3, 0, 0.9)), hand_R=hv((-0.30, -0.05, 1.20), (-0.3, -0.2, 1), (0.3, 0, 0.9)),
                  curl_L=OPEN, curl_R=OPEN, sqc=v(1.09, 0.90, 1.09), sqh=v(1.05, 0.95, 1.05)), "in")
    b.tl.hold(t + 0.10)
    b.tl.add(t + 0.30, G(hips_loc=v(0, 0.22, -0.06), spine=v(-0.42, -0.30, 0.10), head=v(-0.42, 0.45, 0), jaw=0.55,
                         hand_L=hv((0.55, -0.02, 1.52), (0.3, -0.2, 1), (-0.3, 0, 0.9)), hand_R=hv((-0.55, -0.02, 1.52), (-0.3, -0.2, 1), (0.3, 0, 0.9)),
                         curl_L=OPEN, curl_R=OPEN, sqc=v(0.97, 1.05, 0.97),
                         foot_L=v(0.20, 0.0, 0, 0), foot_R=v(0.20, 0.12, 0, 0)), "out")
    b.tl.add(t + 0.50, G(hips_loc=v(0, 0.08, -0.05), spine=v(-0.10, 0.10, 0.05), head=v(-0.05, 0, 0), jaw=0.5,
                         hand_L=hv((0.50, -0.05, 1.05), (0.3, -0.2, 0.4), (-0.3, 0, 1)), hand_R=hv((-0.50, -0.05, 1.05), (-0.3, -0.2, 0.4), (0.3, 0, 1)),
                         curl_L=cu(0.4), curl_R=cu(0.4)), "smooth")
    event(t, "shove", a, b, strength, hz=hz)


def hook(a, b, t, xa, xb, hand="R", hz=1.60, open_=False, push=0.12, spin=0.0, strength=0.8, windup=0.20, hold=0.10, rec=0.0):
    """Hook / cross with the given canonical hand; hit-stop; victim's head snaps to the side of the punch."""
    s = sg(a, b, t)
    go(a.path, t - windup - 0.04, t, (xa, Y_ROW), "in")
    go(b.path, t, t + 0.06, (xb, Y_ROW), "lin")
    go(b.path, t + hold, t + hold + 0.30, (xb + s * push, Y_ROW), "out")
    hs = -1.0 if hand == "R" else 1.0   # canonical sign of the swing side (R hand: wind-up toward the right/back)
    hips_h, flex_h, tw_h = (0, -0.16, -0.10), 0.20, 0.50
    c, dn = contact(a, b, t, hz, so=0.07, hand=hand, hips=hips_h, flex=flex_h, twist=tw_h)
    curl = OPEN if open_ else FIST
    far = hv((hs * 0.40, 0.12, 1.38), (hs * 0.3, -0.3, 0.9), (-hs, 0, 0))
    key = "hand_" + hand
    oth = "hand_L" if hand == "R" else "hand_R"
    oth_g = G()[oth]
    elb_out = v(hs * 0.62, 0.15, 1.45)
    wind = G(hips_loc=v(0, 0.05, -0.11), spine=v(0.12, 0.22 - 0.70 * 1.0, 0), head=v(0.15, -0.2, 0), jaw=0.1,
             **{key: far, "curl_" + hand: curl, "elbow_" + hand: elb_out})
    a.tl.add(t - windup, wind, "wind")
    hit = G(hips_loc=v(*hips_h), spine=v(flex_h, tw_h, 0), head=v(0.22, 0.2, 0), jaw=0.15, yo=0.0,
            **{key: hv(c, (dn[0] + hs * -0.4, dn[1], 0.1), (-hs, 0, 0)), "curl_" + hand: curl, "elbow_" + hand: v(hs * 0.50, -0.02, 1.52)})
    hit["sqc"] = v(0.97, 1.04, 0.97)
    a.tl.add(t, hit, "in")
    a.tl.hold(t + hold)
    fol = dict(hit)
    fol["hips_loc"] = v(0, -0.24, -0.14)
    fol["spine"] = v(0.28, 0.80, 0)
    fol[key] = hv(c + np.array([dn[0], dn[1], 0.0]) * 0.05 + np.array([-hs * 0.12, 0, 0]), (dn[0] - hs * 0.8, dn[1], 0.1), (-hs, 0, 0))
    fol["yo"] = spin
    a.tl.add(t + hold + 0.14, fol, "out")
    # victim
    tw_snap = -0.85 * hs * 1.0
    b.tl.add(t - 0.04, G(jaw=0.4), "smooth")
    b.tl.add(t, G(hips_loc=v(0, 0.05, -0.08), spine=v(-0.10, tw_snap * 0.3, 0), head=v(-0.2, tw_snap, 0.12 * hs), jaw=0.55,
                  sqh=v(1.10, 0.88, 1.10), sqc=v(1.04, 0.96, 1.04)), "in")
    b.tl.hold(t + hold)
    b.tl.add(t + hold + 0.18, G(hips_loc=v(0, 0.14, -0.07), spine=v(-0.30, tw_snap * 0.7, 0.1 * hs), head=v(-0.30, tw_snap * 1.25, 0.2 * hs), jaw=0.5,
                                hand_L=hv((0.40, -0.12, 1.30), (0.2, -0.2, 1), (-0.3, 0, 0.9)), curl_L=OPEN, sqh=v(0.95, 1.07, 0.95), sqc=v(0.98, 1.03, 0.98)), "out")
    event(t, "hook", a, b, strength, hz=hz)


def uppercut(a, b, t, xa, xb, hand="R", strength=1.0, hold=0.12, push=0.55):
    s = sg(a, b, t)
    go(a.path, t - 0.30, t, (xa, Y_ROW), "in")
    go(b.path, t, t + 0.05, (xb, Y_ROW), "lin")
    go(b.path, t + hold, t + hold + 0.45, (xb + s * push, Y_ROW), "out")
    hs = -1.0 if hand == "R" else 1.0
    hips_u, flex_u, tw_u = (0, -0.18, -0.06), 0.28, 0.55
    c, dn = contact(a, b, t, 1.50, so=0.08, hand=hand, hips=hips_u, flex=flex_u, twist=tw_u)
    key = "hand_" + hand
    low = hv((hs * 0.14, -0.18, 0.78), (0, -0.5, 0.6), (-hs, 0, 0))
    a.tl.add(t - 0.26, G(hips_loc=v(0, 0.02, -0.26), spine=v(0.34, -0.25 * hs * -1, 0), head=v(0.2, 0, 0), **{key: low, "elbow_" + hand: v(hs * 0.3, 0.3, 0.9)}), "wind")
    hit = G(hips_loc=v(*hips_u), spine=v(flex_u, tw_u, 0), head=v(0.15, 0, 0), jaw=0.2, sqc=v(0.96, 1.06, 0.96),
            **{key: hv(c, (0.2 * -hs, -0.4, 0.9), (-hs, 0, 0)), "elbow_" + hand: v(hs * 0.28, 0.05, 1.15)})
    a.tl.add(t, hit, "in")
    a.tl.hold(t + hold)
    fol = dict(hit)
    fol["hips_loc"] = v(0, -0.24, -0.02)
    fol["hop"] = 0.05
    fol["foot_L"] = v(0.17, -0.22, 0.05, 0)
    fol[key] = hv(c + v(0, 0, 0.14), (0.2 * -hs, -0.2, 1.0), (-hs, 0, 0))
    a.tl.add(t + hold + 0.16, fol, "out")
    b.tl.add(t - 0.04, G(), "smooth")
    b.tl.add(t, G(hips_loc=v(0, 0.06, -0.04), spine=v(-0.20, 0, 0), head=v(-0.55, 0, 0), jaw=0.6, sqh=v(0.92, 1.12, 0.92), sqc=v(1.0, 1.0, 1.0)), "in")
    b.tl.hold(t + hold)
    air = G(hips_loc=v(0, 0.24, 0.0), spine=v(-0.48, 0.0, 0), head=v(-0.70, 0, 0), jaw=0.6, hop=0.10,
            hand_L=hv((0.55, 0.05, 1.50), (0.3, -0.2, 1), (-0.3, 0, 0.9)), hand_R=hv((-0.55, 0.05, 1.50), (-0.3, -0.2, 1), (0.3, 0, 0.9)),
            curl_L=OPEN, curl_R=OPEN, sqr=v(0.96, 0.96, 1.07), foot_L=v(0.16, 0.10, 0.14, -0.4), foot_R=v(0.16, 0.18, 0.10, -0.4))
    b.tl.add(t + hold + 0.18, air, "out")
    land = G(hips_loc=v(0, 0.12, -0.14), spine=v(-0.12, 0, 0), head=v(0.0, 0, 0), jaw=0.5, hop=0.0, sqr=v(1.08, 1.08, 0.88),
             hand_L=hv((0.45, -0.05, 1.10), (0.3, -0.2, 0.5), (-0.3, 0, 1)), hand_R=hv((-0.45, -0.05, 1.10), (-0.3, -0.2, 0.5), (0.3, 0, 1)))
    b.tl.add(t + hold + 0.32, land, "in")
    b.tl.add(t + hold + 0.46, G(jaw=0.5), "out")
    event(t, "uppercut", a, b, strength, hz=1.50)


def haymaker(a, b, t, xa, xb, hand="R", windup=0.34):
    """a swings wildly, b ducks; a's momentum spins him off balance."""
    s = sg(a, b, t)
    go(a.path, t - windup - 0.01, t, (xa, Y_ROW), "in")
    go(b.path, t - 0.30, t - 0.10, (xb, Y_ROW), "lin")
    go(a.path, t, t + 0.35, (xa + s * 0.22, Y_ROW), "out")
    hs = -1.0 if hand == "R" else 1.0
    key = "hand_" + hand
    big = hv((hs * 0.50, 0.35, 1.78), (hs * 0.4, 0.3, 1.0), (-hs, 0, 0))
    a.tl.add(t - windup, G(hips_loc=v(0, 0.12, -0.10), spine=v(-0.22, -0.70, 0.18 * hs * -1), head=v(0.1, -0.2, 0), jaw=0.5,
                         **{key: big, "elbow_" + hand: v(hs * 0.7, 0.5, 1.6)}), "wind")
    a.tl.add(t - 0.12, G(hips_loc=v(0, 0.12, -0.10), spine=v(-0.24, -0.78, 0.2 * hs * -1), head=v(0.1, -0.2, 0), jaw=0.5,
                         **{key: big + v(hs * 0.04, 0.06, 0.04, 0, 0, 0, 0, 0, 0), "elbow_" + hand: v(hs * 0.72, 0.55, 1.62)}), "lin")
    swing_end = hv((-hs * 0.62, -0.30, 1.50), (-hs * 0.3, -1, 0), (-hs, 0, 0))
    a.tl.add(t, G(hips_loc=v(0, -0.22, -0.12), spine=v(0.30, 0.95, 0), head=v(0.1, 0.5, 0), jaw=0.6, yo=0.0,
                  **{key: swing_end, "elbow_" + hand: v(hs * 0.3, -0.25, 1.45)}), "in")
    fp = G(hips_loc=v(0, -0.30, -0.18), spine=v(0.45, 1.10, 0.2), head=v(0.3, 0.6, 0), jaw=0.6, yo=0.75,
           hand_L=hv((0.50, -0.20, 1.10), (0.3, -0.4, 0.5), (-0.5, 0, 1)), hand_R=hv((-0.50, -0.20, 1.10), (-0.3, -0.4, 0.5), (0.5, 0, 1)),
           curl_L=OPEN, curl_R=OPEN)
    fp[key] = hv((hs * -0.60, -0.35, 1.45), (-hs * 0.5, -0.8, 0.1), (-hs, 0, 0))
    a.tl.add(t + 0.20, fp, "out")
    a.tl.add(t + 0.52, G(hips_loc=v(0, -0.10, -0.12), spine=v(0.3, 0.4, 0), head=v(0.2, 0.1, 0), jaw=0.5, yo=0.20,
                         foot_L=v(0.22, -0.18, 0, 0), foot_R=v(0.22, 0.10, 0, 0)), "smooth")
    # victim ducks
    duck = G(hips_loc=v(0, 0.02, -0.36), spine=v(0.60, 0.1, 0), head=v(0.35, 0, 0), jaw=0.3,
             hand_L=hv((0.22, -0.28, 1.10), (0, -0.5, 0.8), (-1, 0, 0)), hand_R=hv((-0.12, -0.26, 1.08), (0, -0.5, 0.8), (1, 0, 0)),
             foot_L=v(0.20, -0.20, 0, 0), foot_R=v(0.20, 0.16, 0, 0), sqb=v(1.06, 0.94, 1.06), knee_L=v(0.2, -1.0, 0.4), knee_R=v(-0.2, -1.0, 0.4))
    b.tl.add(t - 0.30, G(jaw=0.4), "smooth")
    b.tl.add(t - 0.10, duck, "snap")
    b.tl.add(t + 0.20, duck, "lin")
    b.tl.add(t + 0.52, G(jaw=0.5, sqb=v(0.98, 1.03, 0.98)), "out")
    EVENTS.append(dict(t=t, kind="whoosh", att=a.id, vic=b.id, s=0.5, pt=(a.pos(t)[0] + s * 0.5, Y_ROW - 0.3, 1.6), dir=s))


def grab_tug(a, b, t, t_end, xa, xb, cycles=2, hz=1.40):
    """a grabs b's collar with both hands and shakes him; b pries at a's wrists. Ends with a shove at t_end."""
    s = sg(a, b, t)
    go(a.path, t - 0.30, t, (xa, Y_ROW), "in")
    go(b.path, t - 0.30, t, (xb, Y_ROW), "in")
    hips_g, flex_g, tw_g = (0, -0.14, -0.10), 0.30, 0.35
    cL, dn = contact(a, b, t, hz, so=0.10, hand="L", hips=hips_g, flex=flex_g, twist=tw_g, lateral=0.07)
    cR, _ = contact(a, b, t, hz, so=0.10, hand="R", hips=hips_g, flex=flex_g, twist=tw_g, lateral=-0.07)
    fdir = v(dn[0], dn[1], -0.7)
    a.tl.add(t - 0.22, G(hips_loc=v(0, 0.06, -0.10), spine=v(0.05, 0.1, 0), head=v(0.2, 0, 0), jaw=0.5,
                         hand_L=hv((0.26, -0.24, 1.20), fdir, (-0.4, 0, -0.9)), hand_R=hv((-0.10, -0.24, 1.20), fdir, (0.4, 0, -0.9)), curl_L=cu(0.5), curl_R=cu(0.5)), "wind")
    grip = G(hips_loc=v(*hips_g), spine=v(flex_g, tw_g, 0), head=v(0.25, 0, 0), jaw=0.35, sqc=v(0.97, 1.04, 0.97),
             hand_L=hv(cL, fdir, (-0.4, 0, -0.9)), hand_R=hv(cR, fdir, (0.4, 0, -0.9)), curl_L=cu(0.95), curl_R=cu(0.95))
    a.tl.add(t, grip, "in")
    pry = G(hips_loc=v(0, 0.06, -0.08), spine=v(-0.12, -0.15, 0.04), head=v(-0.20, 0.1, 0), jaw=0.5, sqc=v(1.05, 0.94, 1.05),
            hand_L=hv((0.12, -0.34, 1.30), (0, -0.5, 0.2), (-0.6, 0, 0.6)), hand_R=hv((-0.10, -0.34, 1.26), (0, -0.5, 0.2), (0.6, 0, 0.6)), curl_L=cu(0.9), curl_R=cu(0.9))
    b.tl.add(t - 0.05, G(jaw=0.5), "smooth")
    b.tl.add(t + 0.04, pry, "in")
    per = (t_end - t - 0.10) / (cycles * 2)
    tt = t + 0.10
    for i in range(cycles * 2):
        pull = (i % 2 == 0)
        a_pose = dict(grip)
        b_pose = dict(pry)
        if pull:    # a heaves back, drags b with him
            a_pose["hips_loc"] = v(0, 0.14, -0.14)
            a_pose["spine"] = v(-0.15, 0.25, 0)
            a_pose["head"] = v(0.1, 0.1, 0)
            a_pose["hand_L"] = hv(cL + v(0.05, 0.26, -0.04), fdir, (-0.4, 0, -0.9))
            a_pose["hand_R"] = hv(cR + v(0.05, 0.26, -0.04), fdir, (0.4, 0, -0.9))
            b_pose["hips_loc"] = v(0, -0.20, -0.10)
            b_pose["spine"] = v(0.40, 0.1, 0.1)
            b_pose["head"] = v(0.3, 0, 0)
            go(a.path, tt - per, tt, (xa - s * 0.10, Y_ROW), "smooth")
            go(b.path, tt - per, tt, (xb - s * 0.12, Y_ROW), "smooth")
        else:       # b wrenches back, a hangs on
            a_pose["hips_loc"] = v(0, -0.22, -0.12)
            a_pose["spine"] = v(0.42, 0.35, 0)
            b_pose["hips_loc"] = v(0, 0.20, -0.08)
            b_pose["spine"] = v(-0.30, -0.1, -0.12)
            b_pose["head"] = v(-0.25, 0.3, 0)
            go(a.path, tt - per, tt, (xa + s * 0.02, Y_ROW), "smooth")
            go(b.path, tt - per, tt, (xb + s * 0.10, Y_ROW), "smooth")
        a.tl.add(tt, a_pose, "smooth")
        b.tl.add(tt, b_pose, "smooth")
        EVENTS.append(dict(t=tt, kind="tug", att=a.id, vic=b.id, s=0.35, pt=(b.pos(tt)[0], Y_ROW - 0.1, 0.05), dir=s))
        a.jiggles.append((tt, 0.05))
        b.jiggles.append((tt, 0.07))
        tt += per
    return t_end


def flurry(a, b, t0, n, per, hz=1.05, strength=0.4, xa=None, xb=None, hit_hz=None, covered=False):
    """a throws n fast alternating punches at b (belly or head); b flinches on each."""
    s = sg(a, b, t0)
    if xa is not None:
        go(a.path, t0 - 0.35, t0, (xa, Y_ROW), "in")
    if xb is not None:
        go(b.path, t0 - 0.35, t0, (xb, Y_ROW), "in")
    hips_f, flex_f, tw_f = (0, -0.20, -0.14), 0.42, 0.4
    a.tl.add(t0 - 0.18, G(hips_loc=v(0, 0.05, -0.16), spine=v(0.45, 0.2, 0), head=v(0.2, 0, 0), jaw=0.5), "smooth")
    for i in range(n):
        t = t0 + i * per
        hand = "R" if i % 2 == 0 else "L"
        hs = -1.0 if hand == "R" else 1.0
        c, dn = contact(a, b, t, hz, so=0.07, hand=hand, hips=hips_f, flex=flex_f, twist=tw_f)
        key = "hand_" + hand
        oth_h = "hand_L" if hand == "R" else "hand_R"
        hit = G(hips_loc=v(0, -0.20 - 0.02 * (i % 2), -0.17), spine=v(flex_f, tw_f * (1.0 if hand == "L" else 0.4), 0), head=v(0.25, 0.1 * hs, 0), jaw=0.5,
                sqc=v(0.97, 1.04, 0.97), **{key: hv(c, (dn[0], dn[1], 0.1), (-hs, 0, 0)), "elbow_" + hand: v(hs * 0.42, 0.05, 1.20),
                                            oth_h: hv((-hs * 0.18, -0.30, 1.30), (0, -0.8, 0.6), (hs, 0, 0))})
        a.tl.add(t, hit, "snap" if i else "in")
        if i < n - 1:
            ret = dict(hit)
            ret[key] = hv(c + v(-dn[0], -dn[1], 0) * 0.20 + v(hs * 0.12, 0, 0), (0, -0.8, 0.5), (-hs, 0, 0))
            ret["hips_loc"] = v(0, -0.10, -0.15)
            a.tl.add(t + per * 0.55, ret, "smooth")
        # victim flinch
        flex_v = 0.35 + 0.06 * i
        b.tl.add(t - 0.02 if i == 0 else t + 0.02, G(hips_loc=v(0, 0.02, -0.07), spine=v(0.10, 0.2, 0), head=v(0.1, 0, 0), jaw=0.4) if i == 0 else
                 G(hips_loc=v(0, 0.06, -0.08), spine=v(flex_v, 0.1, 0), head=v(0.15, 0.2 * hs, 0), jaw=0.5,
                   hand_L=hv((0.20, -0.30, 1.50), (0, -0.4, 0.9), (-1, 0, 0)), hand_R=hv((-0.12, -0.30, 1.48), (0, -0.4, 0.9), (1, 0, 0)),
                   sqb=v(1.08, 0.88, 1.10), sqc=v(1.04, 0.95, 1.04)), "in")
        event(t, "punch", a, b, strength, hz=hz)
    a.tl.add(t0 + (n - 1) * per + per * 0.8, G(hips_loc=v(0, -0.15, -0.14), spine=v(0.4, 0.3, 0), head=v(0.2, 0, 0), jaw=0.5), "out")
    return t0 + (n - 1) * per + per * 0.8


def break_apart(fa, fb, t0, t1, xa, xb):
    """Both stagger back to their row slots, panting; then hold the spent stance."""
    s = sg(fa, fb, t0)
    go(fa.path, t0, t1, (xa, Y_ROW), "out")
    go(fb.path, t0, t1, (xb, Y_ROW), "out")
    for f in (fa, fb):
        f.tl.add(t0 + 0.30, G(hips_loc=v(0, 0.10, -0.08), spine=v(0.3, 0.15, 0), head=v(0.15, 0, 0), jaw=0.5, foot_L=v(0.20, -0.10, 0, 0), foot_R=v(0.20, 0.10, 0, 0)), "out")
        f.tl.add(t1, HANG(), "smooth")
        f.tl.add(t1 + 0.15, HANG(spine=v(0.26, 0.1, 0), sqc=v(1.03, 1.02, 1.03)), "smooth")
        f.tl.add(t1 + 0.4, HANG(spine=v(0.22, 0.1, 0), sqc=v(0.99, 1.0, 0.99)), "smooth")


# ------------------------------------------------------------------------------------------------ composition
def compose(T, M, N, Y):
    """Full 1.7-7.6 s brawl. T/M/N/Y are Fighter objects (TERAHOP, MOLEXX, NUBISS, AYARR)."""
    setup()
    for f in (T, M, N, Y):
        start_pose(f, 1.70)
    # ---------------- pair A: TERAHOP (left, lead L) vs MOLEXX
    go(T.path, 1.75, 2.30, (-2.28, Y_ROW))
    go(M.path, 1.75, 2.30, (-1.52, Y_ROW))
    argue(T, M, 1.78, 2.40, off=0.0)
    argue(M, T, 1.78, 2.40, off=0.12)
    bump(T, M, 2.52, -2.14, -1.66)
    shove(T, M, 3.18, -2.02, -1.44, slide=0.44)
    grab_tug(M, T, 3.98, 4.50, -1.56, -2.04, cycles=2)
    hook(M, T, 4.84, -1.62, -2.12, hand="R", push=0.10, strength=0.9)
    hook(T, M, 5.36, -2.08, -1.56, hand="R", push=0.10, strength=0.9)
    uppercut(M, T, 5.98, -1.62, -2.10, hand="R", push=0.45)
    haymaker(T, M, 6.82, -2.34, -1.52, windup=0.26)
    go(T.path, 7.10, 7.48, (-2.4, Y_ROW), "smooth")
    go(M.path, 7.10, 7.48, (-1.4, Y_ROW), "smooth")
    T.tl.add(7.46, HANG(), "smooth")
    M.tl.add(7.46, HANG(), "smooth")
    # ---------------- pair B: NUBISS (lead L) vs AYARR
    go(N.path, 1.75, 2.30, (1.52, Y_ROW))
    go(Y.path, 1.75, 2.30, (2.28, Y_ROW))
    argue(N, Y, 1.84, 2.50, off=0.0, hz=1.30)
    argue(Y, N, 1.84, 2.50, off=0.06, hz=1.28)
    headbutt(Y, N, 2.72, 2.02, 1.66)
    shove(N, Y, 3.46, 1.52, 2.02, slide=0.48)
    flurry(Y, N, 4.18, 3, 0.17, hz=1.05, xa=2.00, xb=1.50)
    hook(N, Y, 4.90, 1.52, 1.98, hand="R", open_=True, push=0.38, spin=0.5, strength=1.0, hz=1.45)
    flurry(Y, N, 5.40, 4, 0.16, hz=1.40, xa=2.04, xb=1.56)
    hook(N, Y, 6.22, 1.54, 2.00, hand="R", push=0.12)
    grab_tug(N, Y, 6.70, 7.05, 1.55, 1.95, cycles=1)
    break_apart(N, Y, 7.06, 7.42, 1.4, 2.4)
