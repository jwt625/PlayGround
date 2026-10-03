"""Build the crude storyboard film v0.2 (DevLog-001 rev 5) as a single .blend.

Usage: blender -b --python scripts/build_crude_film.py -- <out.blend>
Crude on purpose: primitives, camera moves, captions, and in-scene [FX: ...] labels where effects will go.
Scope eyes, the whiteboard graph and the ring spectrum are procedural image sequences (scripts/tex_gen.py).
"""
import math
import os
import random
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_lib as L  # noqa: E402
import tex_gen as T  # noqa: E402
from blender_lib import (F, LP, K, V, VA, at, sc, rt, box, cyl, sph, torus, empty, mat, shot, wl, narr, ovt,  # noqa: E402,F401
                         gary_at, mgr_at, park, shoot, self_shoot, face, make_bench_scope, make_person, make_cast,
                         seq_material, key_emit, key_alpha, set_blend, plane, yaw_to, _text, fade_quad, valley_mark,
                         person_at, shout, brawl, person_shoot, person_fall, shoot_person, ear_steam)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(PROJ, "assets", "generated_textures")
random.seed(7)
PI = math.pi
CLAY = {
    "gray": (0.28, 0.29, 0.33), "dark": (0.12, 0.12, 0.14), "copper": (0.8, 0.42, 0.14),
    "cyan": (0.2, 0.7, 0.8), "tan": (0.82, 0.68, 0.45), "green": (0.2, 0.5, 0.3), "red": (0.85, 0.15, 0.12),
    "white": (0.95, 0.95, 0.95), "yellow": (0.95, 0.8, 0.15), "blue": (0.15, 0.3, 0.8),
}


def cap(i, t0, t1, s): ovt("CAP", s, i, t0, t1)
def big(i, t0, t1, s): ovt("BIG", s, i, t0, t1)
def lab(i, t0, t1, s): ovt("LAB", s, i, t0, t1)
def card(i, t0, t1, s): ovt("CARD", s, i, t0, t1)
def fx(i, t0, t1, s): ovt("FX", s, i, t0, t1)


def ramp(t, ta, tb, va, vb):
    if t <= ta:
        return va
    if t >= tb:
        return vb
    return va + (vb - va) * (t - ta) / (tb - ta)


def lerp3(a, b, u):
    return tuple(a[k] + (b[k] - a[k]) * u for k in range(3))


def heat_color(h):
    """0 -> blue, 0.33 -> green, 0.66 -> yellow, 1 -> red."""
    stops = [(0.1, 0.3, 1.0), (0.1, 0.9, 0.3), (1.0, 0.85, 0.1), (1.0, 0.12, 0.05)]
    h = min(max(h, 0.0), 1.0) * 3.0
    k = min(int(h), 2)
    return lerp3(stops[k], stops[k + 1], h - k)


def eye_material(key, i, t0, t1, param_fn, seed):
    """Eye sequence covering scene-i time [t0, t1); returns a material (frame t0 shows image 1)."""
    f0, f1 = F(i, t0), F(i, t1)
    d = os.path.join(TEX, key)
    params = [param_fn((f - F(i, 0)) / 30.0) for f in range(f0, f1)]
    bers = T.eye_sequence(d, key, params, seed=seed)
    print("EYE", key, "BER first %.2e last %.2e min %.2e max %.2e" % (bers[0], bers[-1], min(bers), max(bers)))
    return seq_material(os.path.join(d, "%s_0001.png" % key), f1 - f0, f0)


def rack(name, loc, i, t0=0, t1=10):
    root = empty(name, LP(i, loc))
    b = box(name + "_body", (0, 0, 0), (0.6, 0.9, 2.0), CLAY["gray"], root)
    led = box(name + "_led", (0, -0.46, 0.5), (0.4, 0.02, 1.2), (0.3, 0.8, 1.0), root, emit=1.5)
    for o in (b, led):
        V(o, i, t0, t1)
    return root


def pulse_obj(name, i, t0, t1):
    p = empty(name, LP(i, (-900, 0, 0)))
    sph(name + "_b", (0, 0, 0), 0.13, CLAY["yellow"], p)
    for sx in (-1, 1):
        sph(name + "_e", (sx * 0.05, -0.1, 0.04), 0.03, (0, 0, 0), p)
    for o in p.children:
        V(o, i, t0, t1)
    return p


# ------------------------------------------------------------------ S1
def s1(c):
    i = 0
    park(c, i)
    for col in range(12):
        for row in range(5):
            x, z = -5.5 + col, 0.4 + 0.6 * row
            V(box("cart", LP(i, (x, 3.5, z)), (0.85, 0.4, 0.5), (0.85, 0.45, 0.2)), i)
            V(box("cart_port", LP(i, (x, 3.28, z)), (0.5, 0.03, 0.18), (0.15, 0.1, 0.08)), i)
            V(box("cart_handle", LP(i, (x, 3.27, z + 0.2)), (0.3, 0.03, 0.03), (0.75, 0.75, 0.78)), i)
    for row in range(5):
        V(cyl("wallcable", LP(i, (0, 3.1, 0.4 + 0.6 * row)), 0.05, 12, CLAY["copper"], axis="X", emit=0.6), i)
    wl("NVL72 BACKPLANE COPPER CARTRIDGES", (0, 3.0, 3.5), i, 0, 2.2, size=0.32)
    rack("rackA", (-1.5, 0, 1.0), i, 2.2, 10)
    rB = rack("rackB", (3.5, 0, 1.0), i, 2.2, 10)
    bpts = [(2.2, 3.5), (2.85, 3.5), (3.0, 0.5), (3.5, 0.5), (3.6, -0.5), (4.2, -0.5), (6.0, 0.5)]
    at(rB, i, [(t, (x, 0, 1.0)) for t, x in bpts])
    for zi, z in enumerate((1.35, 1.7)):
        for yi in range(7):
            y = -0.36 + 0.12 * yi
            shade = 0.85 + 0.15 * ((yi + zi) % 3) / 2
            cab = cyl("cable", LP(i, (1, y, z)), 0.035, 1, (0.8 * shade, 0.42 * shade, 0.14), axis="X", emit=1.0)
            V(cab, i, 2.2, 10)
            for t, x in bpts:
                ln = abs(x + 1.5) - 0.6
                K(cab, F(i, t), loc=LP(i, ((x - 1.5) / 2, y, z)), scale=(1, 1, max(ln, 0.1)))
    p = pulse_obj("pulse", i, 0, 4.2)
    K(p, F(i, 0.0), loc=LP(i, (-6, 3.0, 1.4)), scale=(1, 1, 1), interp="LINEAR")
    K(p, F(i, 2.1), loc=LP(i, (6, 3.0, 1.4)), scale=(1, 1, 1), interp="CONSTANT")
    for (ta, tb, ln) in ((2.2, 2.93, 5.0), (3.0, 3.55, 2.0), (3.6, 4.15, 1.0)):
        endx = -1.5 + ln
        K(p, F(i, ta), loc=LP(i, (-1.5, 0, 1.55)), scale=(1, 1, 1), interp="LINEAR")
        a = math.exp(-0.32 * ln)
        K(p, F(i, tb), loc=LP(i, (endx, 0, 1.55)), scale=(a, a, a), interp="CONSTANT")
    wl("PULSE DECAYS", (1.0, 0, 2.4), i, 2.2, 4.2, size=0.22)
    gary_at(c, i, 0.0, (4.0, -1.0, 0), 0.0)
    gary_at(c, i, 2.2, (4.4, -0.8, 0), 0.0)
    gary_at(c, i, 3.0, (1.4, -0.8, 0), 0.0)
    gary_at(c, i, 3.6, (0.4, -0.8, 0), 0.0)
    gary_at(c, i, 4.2, (0.4, -0.8, 0), -PI / 2, tilt=0.0, interp="LINEAR")
    gary_at(c, i, 6.0, (1.4, -0.8, 0), -PI / 2, tilt=-0.35, interp="CONSTANT")
    # bench scope: ONE set of eye traces; the eye closes by noise and jitter (not fully), BER climbs
    em = eye_material("eye_s1", i, 0.0, 10.0, lambda t: dict(
        sigma_g=ramp(t, 6.0, 7.4, 0.38, 0.45), noise=ramp(t, 6.0, 7.4, 0.05, 0.30), jitter=ramp(t, 6.0, 7.4, 0.03, 0.12)), seed=11)
    make_bench_scope(i, (5.8, -0.5), (0, 10), em)
    wl("SCOPE", (5.68, -0.5, 2.15), i, 6.0, 6.8, size=0.2)
    V(c.head_red, i, 7.9, 10)
    face(c, i, 7.0, 0.0)
    face(c, i, 8.2, 1.0)
    ear_steam(c, i, 7.5, 8.5)
    shot(i, 0.0, 2.2, (-6, -3, 1.6), (-6, 3.5, 1.6), (3.2, -3, 1.4), (4.0, -1.0, 1.0))
    shot(i, 2.2, 3.0, (1, -6.8, 1.9), (1, 0, 1.4), (1, -6.2, 1.9), (1, 0, 1.4))
    shot(i, 3.0, 3.6, (-0.5, -5.5, 1.8), (-0.5, 0, 1.4), (-0.5, -5.1, 1.8), (-0.5, 0, 1.4))
    shot(i, 3.6, 4.2, (-0.8, -4.2, 1.7), (-0.5, 0, 1.4), (-0.8, -3.4, 1.6), (-0.5, 0, 1.4), lens=20)
    shot(i, 4.2, 6.0, (0.4, -3.6, 1.4), (-0.2, 0, 1.2), (1.2, -3.0, 1.2), (0.7, 0, 1.2))
    shot(i, 6.0, 6.8, (5.68, -1.6, 1.5), (5.68, -0.5, 1.36), (5.68, -1.4, 1.45), (5.68, -0.5, 1.36))
    shot(i, 6.8, 8.5, (6.9, -6.0, 1.7), (6.8, -0.6, 1.3), (6.9, -5.0, 1.6), (6.8, -0.6, 1.3))
    shoot(c, i, (7.9, -0.8, 0), (1.4, -0.8, 0), 8.5, 9.2, 1, walk=(6.4, (11.0, -0.8, 0), 7.2), stare_yaw=-PI / 2)
    shot(i, 8.5, 9.2, (5.8, -2.6, 1.3), (7.1, -1.0, 1.1), (6.1, -2.0, 1.25), (7.1, -1.0, 1.1))
    shot(i, 9.2, 10.0, (3.5, -8.5, 2.2), (3.5, 0, 1.0), (3.5, -8.0, 2.2), (3.5, 0, 1.0))
    narr(i, [(0.2, 2.2, "Gary wants faster data over copper."), (2.2, 4.2, "But faster means a shorter wire."),
             (4.2, 6.4, "So Gary tries stretching it one more meter."), (6.5, 7.5, "Bad idea.")])
    big(i, 0.2, 1.0, "COPPER")
    big(i, 2.2, 3.0, "5 m"); big(i, 3.0, 3.6, "2 m"); big(i, 3.6, 4.2, "1 m"); big(i, 4.4, 5.4, "STRETCH")
    card(i, 0.0, 2.2, "NVL72 backplane: ~5,000 copper cables, 2 miles (SemiAnalysis)")
    card(i, 2.2, 3.0, "4x25G passive twinax: >= 5 m (IEEE 802.3bj)")
    card(i, 3.0, 3.6, "100G/lane passive twinax: >= 2 m (IEEE 802.3ck)")
    card(i, 3.6, 4.2, "200G/lane passive twinax: >= 1 m, objective (IEEE 802.3dj)")
    card(i, 6.0, 8.5, "BER = 0.5 erfc(Q/sqrt2), Q from simulated trace samples (illustrative eye model)")
    lab(i, 6.0, 7.4, "EYE NEARLY CLOSED   BER RISING")
    lab(i, 7.4, 8.5, "MANAGER READS THE SCOPE")
    fx(i, 0.0, 2.2, "[FX: fast fly-along, clay cable bundles, Pulse sprint trail]")
    fx(i, 2.2, 4.2, "[FX: Pulse fades as it travels (illustrative exp. decay)]")
    fx(i, 4.4, 6.0, "[FX: cables creak, strands pop, glow dims]")
    fx(i, 7.4, 8.5, "[FX: face turns red, vein pops, steam blast]")
    fx(i, 9.2, 9.8, "[FX: muzzle flash, smoke ring, hole decal]")


# ------------------------------------------------------------------ S2
def s2(c):
    i = 1
    park(c, i)
    RX = -6.0
    NT, Z0, DZ = 8, 0.6, 0.7
    zs = [Z0 + DZ * j for j in range(NT)]
    ZT = zs[-1] + 0.9
    # rack interior only: backplane + side walls (no front posts, so the camera sits fully inside)
    V(box("backplane", LP(i, (RX, 1.0, ZT / 2)), (2.2, 0.08, ZT), (0.2, 0.2, 0.24)), i)
    for sx in (-1, 1):
        V(box("sidewall", LP(i, (RX + sx * 1.05, 0.1, ZT / 2)), (0.05, 1.8, ZT), (0.25, 0.27, 0.32)), i)
    z_cam0, z_cam1, t_cam0, t_cam1 = zs[0] + 0.3, zs[-1] + 0.3, 1.6, 5.4

    def t_arrive(z):
        return t_cam0 + (z + 0.3 - z_cam0) / (z_cam1 - z_cam0) * (t_cam1 - t_cam0)
    for j, z in enumerate(zs):
        V(box("tray", LP(i, (RX, 0.1, z)), (1.9, 1.5, 0.05), (0.15, 0.4, 0.25)), i)
        rows = 1 + (1 if j >= 2 else 0) + (1 if j >= 5 else 0)
        for k in range(4):
            x = RX - 0.75 + 0.5 * k
            V(box("conn", LP(i, (x, 0.7, z + 0.12)), (0.32, 0.22, 0.16), CLAY["blue"]), i)
            V(box("conn_pins", LP(i, (x, 0.58, z + 0.1)), (0.26, 0.02, 0.1), (0.8, 0.7, 0.2)), i)
        for r in range(rows):
            t_row = (0.9 if j == 0 else t_arrive(z) - 1.2) + 0.45 * r
            y = 0.42 - 0.27 * r
            for k in range(4):
                x = RX - 0.75 + 0.5 * k
                tk = t_row + 0.1 * k
                m_chip = mat((0.06, 0.06, 0.08), 0.4, 0.0, unique=True)
                m_glow = mat((1.0, 0.45, 0.1), 0.5, 4.0, unique=True)
                set_blend(m_chip); set_blend(m_glow)
                key_alpha(m_chip, F(i, tk), 0.0); key_alpha(m_chip, F(i, tk + 0.45), 1.0)
                key_emit(m_glow, F(i, tk), (1.0, 0.8, 0.5), 14.0, "LINEAR")
                key_emit(m_glow, F(i, tk + 0.9), (1.0, 0.45, 0.1), 4.0, "LINEAR")
                key_alpha(m_glow, F(i, tk), 0.0); key_alpha(m_glow, F(i, tk + 0.45), 1.0)
                ch = box("chip", LP(i, (x, y, z + 0.08)), (0.28, 0.2, 0.1), (0.06, 0.06, 0.08))
                ch.data.materials.clear(); ch.data.materials.append(m_chip)
                gl = box("chipglow", LP(i, (x, y, z + 0.027)), (0.33, 0.25, 0.02), (1.0, 0.45, 0.1))
                gl.data.materials.clear(); gl.data.materials.append(m_glow)
                for o, zo in ((ch, z + 0.08), (gl, z + 0.027)):
                    V(o, i, tk, 10)
                    K(o, F(i, tk), loc=LP(i, (x, y + 0.45, zo + 0.35)), interp="BEZIER")
                    K(o, F(i, tk + 0.5), loc=LP(i, (x, y, zo)), interp="BEZIER")
                tx = _text("NARROWCOM", 0.034, (0.9, 0.8, 0.45), (OX_(i) + x, y, z + 0.135), None, "CENTER", "CENTER")
                V(tx, i, tk + 0.5, 10)
    # parody logo sign on the backplane (valley mark with the tip flipped, over a wave line)
    for o in valley_mark("sign", LP(i, (RX, 0.93, 1.45)), 0.35, (0.95, 0.85, 0.45), emit=0.6):
        V(o, i, 0.8, 3.0)
    V(_text("NARROWCOM", 0.17, (0.95, 0.85, 0.45), LP(i, (RX, 0.93, 1.1)), None, "CENTER", "CENTER", rot=(PI / 2, 0, 0)), i, 0.8, 3.0)
    # human area: bench scope with the recovering eye, whiteboard with climbing curves
    em = eye_material("eye_s2", i, 0.0, 10.0, lambda t: dict(
        sigma_g=ramp(t, 5.4, 7.4, 0.45, 0.38), noise=ramp(t, 5.4, 7.4, 0.30, 0.05), jitter=ramp(t, 5.4, 7.4, 0.12, 0.03)), seed=22)
    make_bench_scope(i, (1.8, -1.2), (0, 10), em)
    wl("SCOPE", (1.68, -1.2, 2.2), i, 6.4, 8.4, size=0.2)
    V(box("whiteboard", LP(i, (2.6, 2.6, 1.9)), (5.2, 0.1, 2.8), (0.9, 0.9, 0.92)), i, 0, 10)
    V(box("wb_tray", LP(i, (2.6, 2.45, 0.45)), (4.6, 0.12, 0.05), (0.7, 0.7, 0.72)), i, 0, 10)
    n_g = F(i, 10.0) - F(i, 5.0)
    gp = lambda t: ramp(t, 6.4, 8.6, 0.0, 1.0)
    gdir = os.path.join(TEX, "graph_s2")
    T.graph_sequence(gdir, "graph_s2", [gp((f - F(i, 0)) / 30.0) for f in range(F(i, 5.0), F(i, 10.0))])
    gm = seq_material(os.path.join(gdir, "graph_s2_0001.png"), n_g, F(i, 5.0))
    V(plane("graph", LP(i, (2.6, 2.54, 1.9)), (4.8, 2.7), gm), i, 5.0, 10)
    # latency / energy labels ride next to the newest point of each curve
    le = wl("ENERGY / BIT", (0, 0, 0), i, 6.4, 10, size=0.2, color=(1.0, 0.45, 0.4))
    ll = wl("LATENCY", (0, 0, 0), i, 6.4, 10, size=0.2, color=(0.5, 0.65, 1.0))
    t = 6.4
    while t <= 8.7:
        (ex, ey), (lx, ly) = T.graph_tip(gp(t))
        wx = lambda px: 2.6 + (px / 640.0 - 0.5) * 4.8
        wz = lambda py: 1.9 + (0.5 - py / 360.0) * 2.7
        at(le, i, [(t, (wx(ex), 2.45, wz(ey) + 0.28))], "LINEAR")
        at(ll, i, [(t, (wx(lx), 2.45, wz(ly) - 0.28))], "LINEAR")
        t += 0.1
    at(le, i, [(10.0, (wx(ex), 2.45, wz(ey) + 0.28))]); at(ll, i, [(10.0, (wx(lx), 2.45, wz(ly) - 0.28))])
    gary_at(c, i, 0.0, (-0.7, -1.7, 0), yaw_to((-0.7, -1.7), (1.4, -1.2)))
    shot(i, 0.0, 0.8, (1.8, -3.2, 1.5), (1.7, -1.2, 1.4), (1.8, -2.9, 1.5), (1.7, -1.2, 1.4))
    shot(i, 0.8, 1.6, (RX + 0.4, -0.9, 0.95), (RX, 0.6, 0.8), (RX + 0.2, -0.9, 0.95), (RX, 0.6, 0.8), lens=26)
    shot(i, 1.6, 5.4, (RX, -0.9, z_cam0), (RX, 0.6, z_cam0 - 0.12), (RX, -0.9, z_cam1), (RX, 0.6, z_cam1 - 0.12), lens=26)
    shot(i, 5.4, 6.4, (-3.0, -8.0, 8.5), (1.5, 0, 1.2), (1.5, -6.5, 3.0), (1.5, 0, 1.2))
    shot(i, 6.4, 8.6, (2.0, -6.0, 2.0), (2.2, 1.2, 1.6), (2.0, -5.2, 1.9), (2.2, 1.2, 1.6), lens=24)
    mgr_at(c, i, 5.4, (3.9, 1.4, 0), PI)
    shoot(c, i, (3.9, 1.4, 0), (-0.7, -1.7, 0), 8.6, 9.2, 2)
    shot(i, 8.6, 9.2, (2.5, -3.6, 1.5), (3.5, 1.0, 1.1), (3.0, -3.0, 1.4), (3.7, 1.1, 1.1))
    shot(i, 9.2, 10.0, (1.8, -7.0, 2.2), (1.8, 0.5, 1.0), (1.8, -6.6, 2.2), (1.8, 0.5, 1.0), lens=24)
    narr(i, [(0.2, 1.0, "Signal fading?"), (1.0, 3.8, "Gary puts a chip on every connector."),
             (3.8, 8.0, "Now it takes forever to arrive and cranks up the electricity bill.")])
    lab(i, 0.0, 0.8, "EYE CLOSING"); lab(i, 0.8, 1.6, "RETIMER ROW ADDED BEHIND EACH CONNECTOR")
    lab(i, 1.6, 5.4, "TRAY AFTER TRAY: MORE ROWS"); lab(i, 5.4, 6.4, "OUT OF THE RACK"); lab(i, 6.4, 8.6, "EYE RECOVERS   BER FALLS")
    card(i, 6.4, 8.6, "Passive Cu 0 pJ/b (Arista) -> retimed ~15.6 pJ/b = 25 W / 1.6T (Ciena) [derived]. "
                      "Curves: no units. Eye and BER: illustrative model.")
    fx(i, 0.8, 1.6, "[FX: glow flash + sparks as each retimer slides in and fades in]")
    fx(i, 1.6, 5.4, "[FX: particles + glow trails as rows slide in tray by tray]")
    fx(i, 6.4, 8.6, "[FX: marker squeak as the curves draw]")
    fx(i, 9.2, 9.8, "[FX: muzzle flash, smoke ring, hole decal]")


def OX_(i):
    return L.OX(i)


# ------------------------------------------------------------------ S3
def pkg(name, loc, size, n, kind, sub_color, i, t0, t1):
    root = empty(name, LP(i, loc))
    root.rotation_euler = (-PI / 2, 0, 0)  # underside faces the camera (-Y)
    parts = [box(name + "_sub", (0, 0, 0), (size, size, 0.04), sub_color, root),
             box(name + "_die", (0, 0, 0.035), (size * 0.5, size * 0.5, 0.03), (0.08, 0.08, 0.1), root)]
    pitch = size * 0.85 / (n - 1)
    for a in range(n):
        for b in range(n):
            px, py = -size * 0.425 + a * pitch, -size * 0.425 + b * pitch
            if kind == "BGA":
                parts.append(sph(name + "_ball", (px, py, -0.035), size * 0.028, (0.8, 0.8, 0.85), root, seg=8, rings=6))
            else:
                parts.append(box(name + "_pad", (px, py, -0.022), (size * 0.04, size * 0.04, 0.006), (0.85, 0.7, 0.2), root))
    for o in parts:
        V(o, i, t0, t1)
    return root


def s3(c):
    i = 2
    park(c, i)
    V(box("table", LP(i, (0, 0, 0.74)), (5.6, 2.2, 0.12), CLAY["tan"]), i)
    for lx in (-2.5, 2.5):
        for ly in (-0.9, 0.9):
            V(box("leg", LP(i, (lx, ly, 0.34)), (0.12, 0.12, 0.68), CLAY["tan"]), i)
    V(box("asic", LP(i, (0.3, 0, 0.84)), (1.0, 1.0, 0.1), CLAY["dark"]), i)
    mod = box("npo_module", LP(i, (2.0, 0, 0.95)), (0.45, 0.45, 0.15), CLAY["cyan"], emit=0.4)
    V(mod, i)
    at(mod, i, [(0.0, (2.0, 0, 0.95)), (0.2, (2.0, 0, 0.95)), (1.0, (0.9, 0, 0.95))])
    wl("NPO MODULE", (1.3, 0, 1.5), i, 0.2, 1.8, size=0.2)
    vend = [("MOLEXX", (0.75, 0.12, 0.12), (1, 1, 1), (0.5, 0.05, 0.05)),
            ("NUBISS", (0.1, 0.55, 0.25), (1, 1, 1), (0.05, 0.3, 0.12)),
            ("TERAHOP", (0.45, 0.2, 0.65), (1, 1, 1), (0.25, 0.1, 0.4)),
            ("AYARR", (0.9, 0.7, 0.1), (0.05, 0.05, 0.05), (0.6, 0.45, 0.05))]
    V_ = []
    vpos = [(-3.0, 1.7), (-1.7, 1.7), (-0.4, 1.7), (0.9, 1.7)]
    for k, (logo, col, lc, badge) in enumerate(vend):
        V_.append(make_person("vendor%d" % k, i, (vpos[k][0], vpos[k][1], 0), col, logo, lc, logo_size=0.085, badge=badge))
    A, B, C, D = V_
    V(box("laser_in", LP(i, (0.3, 0, 0.95)), (0.3, 0.3, 0.12), CLAY["copper"], emit=1.0), i, 1.8, 3.0)
    V(box("laser_ext", LP(i, (-1.2, 0.6, 1.0)), (0.8, 0.5, 0.4), CLAY["copper"], emit=1.0), i, 2.0, 3.0)
    wl("LASER IN PACKAGE", (0.3, 0, 1.4), i, 1.8, 3.0, size=0.17)
    wl("LASER EXTERNAL", (-1.2, 0.6, 1.5), i, 2.0, 3.0, size=0.17)
    pkg("pkgA", (-2.6, 0.3, 1.2), 0.42, 9, "BGA", (0.15, 0.4, 0.25), i, 3.0, 4.2)
    pkg("pkgB", (-1.5, 0.3, 1.2), 0.56, 12, "LGA", (0.7, 0.5, 0.25), i, 3.1, 4.2)
    pkg("pkgC", (-0.4, 0.3, 1.2), 0.32, 7, "BGA", (0.2, 0.25, 0.5), i, 3.2, 4.2)
    wl("BGA", (-2.6, 0.3, 1.65), i, 3.0, 4.2, size=0.2); wl("LGA", (-1.5, 0.3, 1.65), i, 3.1, 4.2, size=0.2)
    wl("BGA (FINE)", (-0.4, 0.3, 1.65), i, 3.2, 4.2, size=0.16)
    sizes = [(0.3, 0.3), (0.5, 0.35), (0.35, 0.6), (0.6, 0.6)]
    for k, (sx_, sy_) in enumerate(sizes):
        d = box("die", LP(i, (-2.0 + 0.8 * k, 0.1, 0.9)), (sx_, sy_, 0.08), CLAY["white"])
        V(d, i, 4.2, 7.0)
        at(d, i, [(5.2, (-2.0 + 0.8 * k, 0.1, 0.9)), (6.2, (-0.6, -0.2, 0.9 + 0.14 * k)), (6.9, (-0.6, -0.2, 0.9 + 0.14 * k))])
        rt(d, i, [(6.2, (0, 0, 0)), (6.5, (0, 0.05 * (k % 2 * 2 - 1), 0.08 * k)), (6.9, (0, 0, 0))])
    wl("DIE SIZE?", (-1.0, 0.1, 1.5), i, 4.2, 5.2, size=0.18)
    wl("2D", (-1.2, 0.1, 1.4), i, 5.2, 6.0, size=0.2); wl("3D", (-0.6, -0.2, 1.7), i, 6.2, 7.0, size=0.2)
    cal = box("calendar", LP(i, (-2.6, -0.5, 1.3)), (0.5, 0.02, 0.6), (1, 1, 1))
    V(cal, i, 5.4, 7.0)
    at(cal, i, [(5.4, (-2.6, -0.8, 1.3)), (7.0, (-2.6, -0.8, 2.6))])
    ct = wl("+3 MONTHS", (-2.6, -0.8, 1.3), i, 5.4, 7.0, size=0.22, color=(1, 0.4, 0.3))
    at(ct, i, [(5.4, (-2.6, -0.9, 1.3)), (7.0, (-2.6, -0.9, 2.6))])
    tag = sph("pricetag", LP(i, (1.8, -0.8, 1.4)), 0.25, CLAY["red"])
    V(tag, i, 5.4, 8.3)
    sc(tag, i, [(5.4, 0.2), (7.0, 1.6), (8.3, 1.6)])
    wl("$$$", (1.8, -0.9, 1.4), i, 5.4, 8.3, size=0.3, color=(1, 1, 1))
    env = box("envelope", LP(i, (3.0, -0.2, 1.2)), (0.4, 0.25, 0.02), CLAY["white"])
    V(env, i, 7.0, 8.3)
    at(env, i, [(7.0, (3.0, -0.2, 1.2)), (7.6, (0.9, 1.0, 1.2)), (8.3, (0.9, 1.0, 1.2))])
    et = wl("YEAR-END BONUS", (3.0, -0.2, 1.5), i, 7.0, 8.3, size=0.14, color=(0.4, 1.0, 0.4))
    at(et, i, [(7.0, (3.0, -0.2, 1.5)), (7.6, (0.9, 1.0, 1.5)), (8.3, (0.9, 1.0, 1.5))])
    gary_at(c, i, 0.0, (3.2, -0.2, 0), -PI / 2)
    # vendors shout, shove and beat each other from the laser fight on; then two shoot the other two
    for p in V_:
        shout(p, 1.8, 8.3)
    brawl(A, 3.0, 7.9, (-1.7, 1.7), seed=1)
    brawl(B, 3.0, 7.9, (-3.0, 1.7), seed=2)
    brawl(C, 3.0, 8.25, (0.9, 1.7), seed=3)
    brawl(D, 3.0, 8.2, (-0.4, 1.7), seed=4)
    for k, (p, t0) in enumerate(((A, 2.2), (B, 3.6), (C, 5.0), (D, 6.4))):
        wl("@#$%!", (vpos[k][0], 1.7, 2.0), i, t0, t0 + 0.9, size=0.3, color=(1.0, 0.35, 0.3))
        wl("!!", (vpos[k][0] + 0.2, 1.5, 2.1), i, t0 + 2.0, t0 + 2.9, size=0.4, color=(1.0, 0.8, 0.2))
    person_shoot(A, 7.9, 8.3, yaw_to(vpos[0], vpos[3]), (vpos[0][0], vpos[0][1], 0))
    person_shoot(B, 7.9, 8.4, yaw_to(vpos[1], vpos[2]), (vpos[1][0], vpos[1][1], 0))
    person_fall(D, 8.3, (vpos[3][0], vpos[3][1], 0), yaw_to(vpos[3], vpos[0]))
    person_fall(C, 8.4, (vpos[2][0], vpos[2][1], 0), yaw_to(vpos[2], vpos[1]))
    big(i, 8.3, 8.9, "BANG BANG")
    # Manager shoots one remaining vendor (B) while Gary shoots himself
    mpos = (-3.2, -1.2, 0)
    myaw = yaw_to(mpos[:2], (3.2, -0.2))
    K(c.m, F(i, 7.2), loc=LP(i, (-6.5, -1.2, 0)), rot=(0, 0, myaw), interp="LINEAR")
    K(c.m, F(i, 7.8), loc=LP(i, mpos), rot=(0, 0, myaw), interp="CONSTANT")
    face(c, i, 7.8, 0.5); face(c, i, 8.9, 1.0)
    shoot_person(c, i, mpos, (vpos[1][0], vpos[1][1], 0), 8.5, 8.95)
    person_fall(B, 8.95, (vpos[1][0], vpos[1][1], 0), yaw_to(vpos[1], mpos), dur=0.8)
    self_shoot(c, i, (3.2, -0.2, 0), -PI / 2, 8.3, 8.95, caption=False)
    big(i, 8.95, 9.55, "BANG")
    wl("GARY'S GUN", (3.6, -1.4, 2.0), i, 8.3, 8.95, size=0.2)
    shot(i, 0.0, 1.8, (2.8, -4.6, 2.2), (0.8, 0, 1.0), (1.8, -3.6, 1.8), (0.8, 0, 1.0))
    shot(i, 1.8, 3.0, (-4.5, -4.5, 2.0), (-1.5, 0.5, 1.0), (-1.0, -4.5, 2.0), (-0.3, 0.5, 1.0))
    shot(i, 3.0, 4.2, (-1.5, -3.2, 1.5), (-1.5, 0.3, 1.2), (-1.5, -2.7, 1.4), (-1.5, 0.3, 1.2))
    shot(i, 4.2, 5.2, (-1.0, -3.5, 1.5), (-1.2, 0.4, 0.95), (-0.5, -3.2, 1.4), (-1.2, 0.4, 0.95))
    shot(i, 5.2, 7.0, (0.5, -3.5, 1.6), (-0.5, 0, 1.0), (0.8, -7.5, 3.0), (-0.5, 0, 1.0))
    shot(i, 7.0, 8.3, (2.5, -4.5, 1.8), (1.6, 0.4, 1.1), (0.5, -4.8, 1.8), (1.6, 0.4, 1.1))
    shot(i, 8.3, 10.0, (0.0, -6.0, 1.9), (0, 0.6, 1.0), (0.2, -5.4, 1.7), (0, 0.6, 1.0))
    narr(i, [(0.2, 3.0, "Gary moves the optics closer to the logic."), (3.0, 5.4, "Every vendor wants something different."),
             (5.6, 8.2, "Now he waits three more months and pays his bonus.")])
    lab(i, 0.0, 1.8, "NPO"); lab(i, 1.8, 3.0, "LASER: IN PACKAGE OR EXTERNAL?"); lab(i, 3.0, 4.2, "BGA vs LGA vs PITCH vs SIZE")
    lab(i, 4.2, 5.2, "DIE SIZE"); lab(i, 5.2, 7.0, "2D vs 3D   LEAD TIME   COST"); lab(i, 7.0, 8.3, "THE VENDORS LOSE IT")
    card(i, 0.0, 1.8, "NPO: module on package substrate / HDI beside the ASIC (Cheng 2025)")
    fx(i, 1.8, 3.0, "[FX: vendors shout, shove, swing; comic dust clouds]")
    fx(i, 3.0, 4.2, "[FX: packages flip to show underside ball/land arrays]")
    fx(i, 5.2, 7.0, "[FX: die tower wobble, calendar pages fly, price tag inflates]")
    fx(i, 8.3, 9.5, "[FX: muzzle flashes, smoke rings, hole decals]")


# ------------------------------------------------------------------ S4
def s4(c):
    i = 3
    park(c, i)
    rnd = random.Random(21)
    OEX = 7.0
    V(box("pcb", LP(i, (0, 0, -0.1)), (21, 16, 0.2), (0.12, 0.35, 0.18)), i)
    V(box("substrate", LP(i, (0, 0, 0.125)), (16.4, 10.4, 0.25), (0.25, 0.45, 0.3)), i)
    # Rubin-Ultra-style package: gold lid frame, a wide central interposer band with 4 reticle-size GPU dies,
    # 8 HBM stacks above and 8 below
    gold = (0.8, 0.65, 0.3)
    for sy in (-1, 1):
        V(box("lid_h", LP(i, (0, sy * 4.45, 0.45)), (12.4, 0.3, 0.4), gold, rough=0.3), i)
    for sx in (-1, 1):
        V(box("lid_v", LP(i, (sx * 6.05, 0, 0.45)), (0.3, 8.6, 0.4), gold, rough=0.3), i)
    V(box("lid_floor", LP(i, (0, 0, 0.27)), (11.8, 8.6, 0.04), (0.1, 0.1, 0.12)), i)
    V(box("interposer", LP(i, (0, 0, 0.34)), (10.0, 3.0, 0.1), (0.2, 0.5, 0.5)), i)
    for k, gx in enumerate((-3.75, -1.25, 1.25, 3.75)):
        V(box("gpu", LP(i, (gx, 0, 0.47)), (2.2, 2.4, 0.16), (0.62, 0.56, 0.4)), i)
        for a in range(3):
            for b in range(4):
                V(cyl("bump", LP(i, (gx - 0.6 + 0.6 * a, -0.75 + 0.5 * b, 0.56)), 0.1, 0.02, (0.85, 0.8, 0.7), verts=12), i)
    for sy in (-1, 1):
        for hx in (-4.45, -3.3, -2.15, -1.0, 1.0, 2.15, 3.3, 4.45):
            V(box("hbm", LP(i, (hx, sy * 2.9, 0.41)), (1.0, 1.5, 0.24), (0.78, 0.68, 0.42), rough=0.4), i)
    wl("XPU: 4 GPU DIES + 16 HBM", (0, 0, 1.4), i, 1.0, 3.0, size=0.5)
    # board mock-up
    def free(x, y):
        return abs(x) > 8.5 or abs(y) > 5.5

    def place(n, fn):
        k = 0
        while k < n:
            x, y = rnd.uniform(-10, 10), rnd.uniform(-7.6, 7.6)
            if free(x, y):
                fn(x, y)
                k += 1
    place(28, lambda x, y: [V(box("vreg", LP(i, (x, y, 0.1)), (0.5, 0.5, 0.2), (0.05, 0.05, 0.06)), i),
                            V(sph("vreg_dot", LP(i, (x - 0.15, y - 0.15, 0.21)), 0.04, (0.9, 0.9, 0.9), seg=8, rings=6), i)])
    place(22, lambda x, y: [V(box("ind", LP(i, (x, y, 0.18)), (0.55, 0.55, 0.36), (0.25, 0.25, 0.28)), i),
                            V(box("ind_top", LP(i, (x, y, 0.37)), (0.5, 0.5, 0.03), (0.7, 0.7, 0.72)), i)])
    place(8, lambda x, y: V(cyl("xfmr", LP(i, (x, y, 0.3)), 0.35, 0.6, (0.1, 0.1, 0.12)), i))
    place(10, lambda x, y: V(cyl("coil", LP(i, (x, y, 0.15)), 0.18, 0.3, (0.7, 0.4, 0.15), verts=12), i))
    for n in range(18):
        cx, cy = rnd.choice([-1, 1]) * rnd.uniform(8.7, 9.8), rnd.uniform(-7.2, 7.2)
        if rnd.random() < 0.5:
            cx, cy = rnd.uniform(-9.0, 9.0), rnd.choice([-1, 1]) * rnd.uniform(5.7, 7.4)
        for a in range(14):
            V(box("mlcc", LP(i, (cx + 0.18 * (a % 7), cy + 0.12 * (a // 7), 0.04)), (0.14, 0.08, 0.06), (0.8, 0.65, 0.4)), i)
    for k in range(4):
        V(box("conn", LP(i, (-8.6 + 5.6 * k, -7.7, 0.15)), (1.8, 0.35, 0.3), (0.1, 0.1, 0.14)), i)
    for sx in (-1, 1):
        for sy in (-1, 1):
            V(cyl("bolt", LP(i, (sx * 10.0, sy * 7.4, 0.2)), 0.2, 0.4, (0.75, 0.75, 0.78)), i)
    # 8 + 8 optical engines (16 total) on the left and right edges of the package
    oes = []
    for side in (-1, 1):
        for r in range(8):
            y = -3.5 + r * 1.0
            root = empty("oe", LP(i, (side * 11.6, y, 0.0)))
            at(root, i, [(0.2 + 0.03 * r, (side * 11.6, y, 0)), (1.2, (side * OEX, y, 0))])
            m = mat((0.2, 0.7, 0.8), 0.6, 0.4, unique=True)
            b = box("oe_body", (0, 0, 0.55), (0.7, 0.8, 0.22), (0.2, 0.7, 0.8), root)
            b.data.materials.clear(); b.data.materials.append(m)
            V(b, i, 0, 10)
            oes.append((side, r, root, m))
    for n, (side, r, root, m) in enumerate(oes):
        key_emit(m, F(i, 0.0), (0.2, 0.7, 0.8), 0.4, "LINEAR")
        key_emit(m, F(i, 4.6 + 0.03 * n), (0.2, 0.7, 0.8), 0.4, "LINEAR")
        key_emit(m, F(i, 5.0 + 0.03 * n), (1.0, 0.45, 0.1), 3.0, "LINEAR")
        key_emit(m, F(i, 5.4 + 0.03 * n), (1.0, 0.1, 0.05), 6.0, "LINEAR")
    # workload / temperature bars
    pw = box("powerbar", LP(i, (-0.8, -7.0, 0.5)), (0.5, 0.5, 1.0), CLAY["yellow"], emit=1.0)
    tb = box("tempbar", LP(i, (0.8, -7.0, 0.5)), (0.5, 0.5, 1.0), CLAY["blue"], emit=1.0)
    tr = box("tempred", LP(i, (0.8, -7.0, 0.5)), (0.52, 0.52, 1.0), CLAY["red"], emit=3.0)
    for o in (pw, tb, tr):
        V(o, i, 1.4, 2.6)
    for n in range(18):
        h = rnd.uniform(0.3, 3.0)
        t = 1.4 + n * (1.2 / 17)
        K(pw, F(i, t), loc=LP(i, (-0.8, -7.0, h / 2)), scale=(0.5, 0.5, h), interp="LINEAR")
        K(tb, F(i, t), loc=LP(i, (0.8, -7.0, h / 2)), scale=(0.5, 0.5, h), interp="LINEAR")
        hr = max(h - 1.6, 0.02)
        K(tr, F(i, t), loc=LP(i, (0.8, -7.0, hr / 2)), scale=(0.52, 0.52, hr), interp="LINEAR")
    wl("XPU POWER", (-0.8, -7.0, 3.4), i, 1.4, 2.6, size=0.28); wl("TEMPERATURE", (0.8, -7.0, 3.8), i, 1.4, 2.6, size=0.28)
    for n in range(12):
        s_ = sph("smoke", LP(i, (rnd.uniform(-1.5, 1.5), rnd.uniform(-1.0, 1.0), 0.7)), 0.4, (0.55, 0.55, 0.55))
        t0 = 1.5 + n * 0.08
        V(s_, i, t0, t0 + 1.2)
        at(s_, i, [(t0, (rnd.uniform(-2.0, 2.0), rnd.uniform(-1.0, 1.0), 0.7)), (t0 + 1.1, (rnd.uniform(-3.0, 3.0), rnd.uniform(-2.0, 2.0), 4.5))])
        sc(s_, i, [(t0, 0.4), (t0 + 1.1, 2.4)])
    # zoomed-in PIC (reached by a fast fade): heat arrives as a left-to-right wave over rings AND the substrate
    px = 30.0
    t_a, t_b = 3.5, 4.6
    base = (0.08, 0.28, 0.4)
    NS = 28
    for k in range(NS):
        m = mat(base, 0.6, 0.5, unique=True)
        sx_ = px - 3.7 + (k + 0.5) * 7.4 / NS
        slab = box("pic_slab", LP(i, (sx_, 0, 0.1)), (7.4 / NS, 4.2, 0.2), base)
        slab.data.materials.clear(); slab.data.materials.append(m)
        V(slab, i, t_a, t_b)
        for f in range(F(i, t_a), F(i, t_b) + 1, 3):
            tt = (f - F(i, 0)) / 30.0
            hval = 0.5 + 0.5 * math.sin(2 * PI * (0.9 * tt - 0.9 * (k / NS) * 1.6))
            key_emit(m, f, lerp3(base, heat_color(hval), 0.08 + 0.3 * hval), 0.35 + 0.35 * hval, "LINEAR")
    for by in (-1.3, 0.0, 1.3):
        V(box("wg", LP(i, (px, by, 0.22)), (6.8, 0.07, 0.04), (0.5, 0.95, 1.0), emit=3.0), i, t_a, t_b)
        for k in range(8):
            ry = by + (0.3 if k % 2 == 0 else -0.3)
            rx = px - 3.0 + 0.85 * k
            m = mat((0.2, 0.8, 1.0), 0.5, 1.2, unique=True)
            tz = torus("ring", LP(i, (rx, ry, 0.24)), 0.24, 0.035, (0.2, 0.8, 1.0), material=m)
            V(tz, i, t_a, t_b)
            xn = (rx - (px - 3.7)) / 7.4
            for f in range(F(i, t_a), F(i, t_b) + 1, 3):
                tt = (f - F(i, 0)) / 30.0
                hval = 0.5 + 0.5 * math.sin(2 * PI * (0.9 * tt - 0.9 * xn * 1.6))
                key_emit(m, f, heat_color(hval), 1.5, "LINEAR")
    wl("PIC: MICRORING MODULATORS", (px, 0, 1.6), i, t_a, t_b, size=0.34)
    wl("HEAT WAVE ->", (px, -2.4, 0.6), i, t_a, t_b, size=0.3, color=(1.0, 0.6, 0.3))
    fade_quad(i, [(3.1, 0.0), (3.45, 1.0), (3.5, 1.0), (3.8, 0.0), (4.3, 0.0), (4.6, 1.0), (4.65, 1.0), (4.9, 0.0)])
    # eggs: one per OE
    for n, (side, r, root, m) in enumerate(oes):
        ex, ey = side * OEX, -3.5 + r * 1.0
        t0 = 5.4 + 0.06 * n
        egg = empty("egg", LP(i, (ex, ey, 3.0)))
        at(egg, i, [(t0, (ex, ey, 3.0)), (t0 + 0.25, (ex, ey, 0.68))])
        glass = cyl("eggglass", (0, 0, 0), 0.3, 0.025, (0.75, 0.85, 0.95), parent=egg, rough=0.1, verts=16)
        wh = cyl("eggwhite", (0, 0, 0.01), 0.3, 0.03, (1, 1, 1), parent=egg, verts=16)
        brown = torus("eggbrown", (0, 0, 0.02), 0.285, 0.025, (0.45, 0.25, 0.08), parent=egg, segs=16)
        yolk = sph("eggyolk", (0, 0, 0.05), 0.12, (1.0, 0.7, 0.05), egg, scale=(1, 1, 0.6), seg=12, rings=8)
        bub = [sph("bubble", (rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2), 0.04), 0.03, (1, 0.97, 0.9), egg, seg=8, rings=6) for _ in range(3)]
        for o, a in ((glass, t0 + 0.25), (wh, t0 + 0.25), (brown, t0 + 1.4), (yolk, t0 + 0.25)):
            V(o, i, a, 8.2)
        for q, o in enumerate(bub):
            V(o, i, t0 + 0.9 + 0.3 * q, t0 + 1.4 + 0.3 * q)
        sc(wh, i, [(t0 + 0.25, (0.2, 0.2, 1)), (t0 + 1.3, (1, 1, 1))])
        if n % 4 == 0:
            for s in range(3):
                st = sph("steam", LP(i, (ex, ey, 0.9)), 0.12, (0.95, 0.95, 0.95), seg=10, rings=6)
                ts = t0 + 0.8 + s * 0.35
                V(st, i, ts, ts + 0.9)
                at(st, i, [(ts, (ex + rnd.uniform(-0.2, 0.2), ey, 0.9)), (ts + 0.85, (ex + rnd.uniform(-0.4, 0.4), ey, 2.4))])
    wl("[EGG FRY SHADER]", (-6.0, -4.6, 1.8), i, 5.6, 8.2, size=0.3, color=(1, 0.85, 0.2))
    # human area
    hx0 = 50.0
    gary_at(c, i, 0.0, (hx0, -4.0, 0), 0.0)
    plate = cyl("plate", LP(i, (hx0 + 0.45, -4.55, 0.85)), 0.28, 0.03, CLAY["white"])
    V(plate, i, 8.2, 10)
    mhead = (hx0 + 3.6, -4.0)
    for k in range(3):
        ex0 = hx0 + 0.45 + 0.12 * (k - 1)
        er = empty("tegg", LP(i, (ex0, -4.55, 0.92)))
        wd = cyl("tegg_w", (0, 0, 0), 0.15, 0.025, (1, 1, 1), parent=er, verts=12)
        yk = sph("tegg_y", (0, 0, 0.03), 0.06, (1.0, 0.7, 0.05), er, seg=10, rings=6)
        V(wd, i, 8.2, 10); V(yk, i, 8.2, 10)
        at(er, i, [(8.2, (ex0, -4.55, 0.92)), (8.9, (ex0, -4.55, 0.92)), (9.25, (ex0 + 0.3 * k, -4.4, 14.0)),
                   (9.75, (mhead[0] - 0.05 + 0.05 * k, mhead[1], 1.7 + 0.03 * k))], interp="BEZIER")
    shoot(c, i, (hx0 + 3.6, -4.0, 0), (hx0, -4.0, 0), 8.3, 8.9, 4)
    # camera
    shot(i, 0.0, 1.4, (0, -1.5, 19), (0, 0, 0))
    shot(i, 1.4, 2.6, (0, -16, 7.0), (0, 0, 0.5), (0, -9.0, 3.0), (0, 0, 0.5), lens=20, lens1=55, ease="BEZIER")
    shot(i, 2.6, 3.45, (0, -9.0, 3.0), (0, 0, 0.5), (5.8, -3.8, 1.5), (7.0, 0.0, 0.7), lens=55, lens1=85, ease="BEZIER")
    shot(i, 3.5, 4.6, (px - 3.0, -3.8, 2.4), (px - 1.0, 0, 0.3), (px + 2.0, -3.4, 2.0), (px + 2.5, 0, 0.3))
    shot(i, 4.6, 5.4, (0, -16, 8.5), (0, 0, 0.4), (0, -19, 9.5), (0, 0, 0.4))
    shot(i, 5.4, 6.8, (-12.5, -7.0, 3.2), (-7.0, 0.0, 0.6), (-12.8, -3.5, 3.4), (-7.0, 0.0, 0.6))
    shot(i, 6.8, 8.2, (12.5, -7.0, 3.2), (7.0, 0.0, 0.6), (12.8, -3.5, 3.4), (7.0, 0.0, 0.6))
    shot(i, 8.2, 10.0, (hx0 + 1.8, -10.0, 3.2), (hx0 + 1.8, -4.0, 4.0), (hx0 + 1.8, -9.6, 3.2), (hx0 + 1.8, -4.0, 4.0))
    narr(i, [(0.2, 2.4, "Gary glues the optics onto the chip."), (2.4, 5.2, "The chip's heat swings wildly, but optics need it steady."),
             (5.4, 8.0, "So: scorching hot, always.")])
    lab(i, 0.0, 1.4, "NPO -> CPO   (8 + 8 OEs)"); lab(i, 1.4, 2.6, "XPU POWER   TEMPERATURE"); lab(i, 2.6, 3.4, "ZOOM ON AN OE")
    lab(i, 3.5, 4.6, "RING RESONANCES: HEAT WAVE"); lab(i, 4.6, 5.4, "HEATERS ON"); big(i, 6.2, 8.0, "BREAKFAST")
    fx(i, 1.4, 2.6, "[FX: dolly-zoom, shake on spikes, smoke + heat shimmer]")
    fx(i, 2.6, 3.4, "[FX: continuous zoom, then quick fade to the PIC]")
    fx(i, 3.5, 4.6, "[FX: heat wave over rings and substrate, left to right]")
    fx(i, 4.6, 5.4, "[FX: engines glow red, heater coils pulse]")
    fx(i, 5.4, 8.2, "[FX: egg: glassy -> opaque white, edges brown, bubbles, yolk sets, steam]")
    fx(i, 8.9, 9.8, "[FX: eggs tossed out of frame, land on Manager; smoke ring, muzzle flash]")


# ------------------------------------------------------------------ S5
def cm300(i, cx, t0, t1):
    """FormFactor CM300-style probe station (crude): black cabinet, silver plate, microscope column, 4 positioners."""
    def b(name, loc, size, color, rough=0.5):
        return V(box(name, LP(i, loc), size, color, rough=rough), i, t0, t1)
    b("cm_cab", (cx, 0, 0.75), (4.2, 3.0, 1.5), (0.04, 0.04, 0.06))
    b("cm_plate", (cx, 0, 1.54), (4.0, 2.8, 0.08), (0.7, 0.72, 0.76), rough=0.25)
    V(cyl("cm_ring", LP(i, (cx, 0, 1.59)), 1.15, 0.05, (0.12, 0.12, 0.14)), i, t0, t1)
    V(cyl("cm_chuck", LP(i, (cx, 0, 1.64)), 1.0, 0.12, (0.75, 0.75, 0.78), rough=0.25), i, t0, t1)
    b("cm_column", (cx, 0.5, 2.7), (0.6, 0.6, 2.2), (0.05, 0.05, 0.07))
    V(cyl("cm_lens", LP(i, (cx, 0.5, 1.9)), 0.22, 0.6, (0.1, 0.1, 0.12)), i, t0, t1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b("cm_pos", (cx + sx * 1.55, sy * 0.75, 1.75), (0.45, 0.6, 0.35), (0.06, 0.06, 0.08))
            b("cm_posS", (cx + sx * 1.2, sy * 0.75, 1.72), (0.25, 0.25, 0.2), (0.75, 0.75, 0.78), rough=0.25)
            V(cyl("cm_cable", LP(i, (cx + sx * 1.7, sy * 0.75, 2.2)), 0.03, 0.9, (0.05, 0.05, 0.05), verts=8), i, t0, t1)
    b("cm_front", (cx, -1.4, 0.8), (3.4, 0.3, 0.9), (0.55, 0.57, 0.6), rough=0.3)
    V(cyl("cm_drawerchuck", LP(i, (cx, -1.7, 1.0)), 0.8, 0.1, (0.8, 0.8, 0.82), rough=0.25), i, t0, t1)
    b("cm_monitor", (cx + 1.6, 0.9, 2.5), (1.0, 0.08, 0.65), (0.05, 0.06, 0.1))


def s5(c):
    i = 4
    park(c, i)
    rnd = random.Random(33)
    eng = box("engine", LP(i, (0, 0, 0.5)), (1.4, 1.4, 0.3), CLAY["cyan"], emit=0.4)
    V(eng, i, 0, 0.8)
    die = box("die", LP(i, (0, 0, 0.7)), (0.5, 0.5, 0.1), CLAY["white"])
    V(die, i, 0, 3.3)
    stops = [(0.0, 0), (0.3, 0), (1.0, 8), (1.55, 15), (2.05, 22), (2.55, 29), (3.2, 40)]
    at(die, i, [(t, (x, 0, 0.7 if x < 38 else 1.8)) for t, x in stops])
    V(box("oven", LP(i, (8, 0, 0.7)), (3.2, 1.8, 1.4), (0.2, 0.2, 0.22)), i, 0.5, 3.4)
    V(box("oven_glow", LP(i, (8, -0.91, 0.7)), (2.8, 0.02, 0.5), (1.0, 0.4, 0.1), emit=5.0), i, 0.5, 3.4)
    wl("REFLOW ON SUBSTRATE", (8, 0, 2.2), i, 0.5, 3.4, size=0.3)
    V(box("fau_base", LP(i, (15, 0, 0.3)), (1.6, 1.4, 0.6), CLAY["gray"]), i, 0.5, 3.4)
    V(cyl("fau_arm", LP(i, (15, 0, 1.2)), 0.12, 1.2, (0.7, 0.7, 0.75)), i, 0.5, 3.4)
    V(box("fau_head", LP(i, (15, 0, 1.9)), (0.7, 0.4, 0.25), (0.2, 0.2, 0.25)), i, 0.5, 3.4)
    for k in range(6):
        V(cyl("fau_fiber", LP(i, (15 - 0.25 + 0.1 * k, -0.2, 1.65)), 0.02, 0.5, (1.0, 0.8, 0.2), verts=8, emit=1.0), i, 0.5, 3.4)
    wl("FAU ATTACH", (15, 0, 2.6), i, 0.5, 3.4, size=0.3)
    V(box("bond_col", LP(i, (22, 0.7, 1.1)), (0.5, 0.5, 2.2), CLAY["gray"]), i, 0.5, 3.4)
    V(box("bond_head", LP(i, (22, 0.1, 1.9)), (0.9, 0.9, 0.5), (0.2, 0.2, 0.25)), i, 0.5, 3.4)
    V(box("bond_stage", LP(i, (22, 0, 0.3)), (1.6, 1.4, 0.6), CLAY["gray"]), i, 0.5, 3.4)
    wl("EIC / PIC BONDING", (22, 0, 2.8), i, 0.5, 3.4, size=0.3)
    V(cyl("saw_blade", LP(i, (29, -0.7, 1.2)), 0.6, 0.05, CLAY["red"], axis="Y"), i, 0.5, 3.4)
    V(box("saw_base", LP(i, (29, 0, 0.4)), (2.0, 1.6, 0.8), CLAY["gray"]), i, 0.5, 3.4)
    V(cyl("tape_roll", LP(i, (31, 0.5, 1.0)), 0.5, 0.7, (0.9, 0.8, 0.2), axis="Y"), i, 0.5, 3.4)
    wl("DICING + TAPE", (29.5, 0, 2.4), i, 0.5, 3.4, size=0.3)
    for n in range(14):
        z = rnd.uniform(0.3, 2.5)
        col = rnd.choice([CLAY["blue"], CLAY["yellow"], CLAY["red"], CLAY["white"]])
        sk = box("streak", LP(i, (40, rnd.uniform(-3, -7), z)), (7, 0.05, 0.05), col, emit=1.5)
        V(sk, i, 0.3, 3.2)
        t0 = 0.3 + rnd.uniform(0, 1.8)
        sy = -rnd.uniform(3, 7)
        at(sk, i, [(t0, (40, sy, z)), (t0 + 0.5, (-2, sy, z))])
    # wafer-level test on a CM300-style station: the probe head is fixed; wafers slide in, the stage steps the
    # wafer under the probe, wafers slide out quickly
    cx = 40.0
    cm300(i, cx, 2.9, 4.9)
    pc = empty("probe", LP(i, (cx, 0, 2.0)))
    V(box("pc_body", (0, 0, 0.0), (0.35, 0.35, 0.06), CLAY["gray"], pc), i, 2.9, 4.9)
    for nx in (-0.1, 0.1):
        for ny in (-0.1, 0.1):
            V(cyl("needle", (nx, ny, -0.1), 0.01, 0.2, (0.9, 0.8, 0.2), parent=pc, verts=8), i, 2.9, 4.9)
    pitch = 0.2
    pts = [(gx * pitch, gy * pitch) for gy in range(-4, 5) for gx in range(-4, 5) if (gx * pitch) ** 2 + (gy * pitch) ** 2 <= 0.8 ** 2]
    rows = {}
    for px_, py_ in pts:
        rows.setdefault(round(py_, 3), []).append(px_)
    order_all = []
    for ri, py_ in enumerate(sorted(rows)):
        order_all += [(x_, py_) for x_ in sorted(rows[py_], reverse=(ri % 2 == 1))]
    wafers = [(3.3, 3.45, 3.85, 4.0), (3.95, 4.05, 4.35, 4.5), (4.45, 4.55, 4.8, 4.9)]
    for w, (t_in, t_t0, t_t1, t_out) in enumerate(wafers):
        z = 1.72
        root = empty("wafer", LP(i, (cx - 4.0, -3.4, z)))
        V(cyl("wafer_disc", (0, 0, 0), 0.95, 0.04, (0.78, 0.8, 0.86), parent=root, rough=0.3), i, t_in, t_out + 0.12)
        n_probe = 14
        start = (w * 7) % max(len(order_all) - n_probe, 1)
        probed = order_all[start:start + n_probe]
        probed_set = {pt for pt in probed}
        for (px_, py_) in order_all:
            if (px_, py_) in probed_set:
                kk = probed.index((px_, py_))
                tprobe = t_t0 + (t_t1 - t_t0) * kk / n_probe
                colr = CLAY["green"] if kk % 10 == 5 else (0.7, 0.1, 0.1)
                V(box("die_g", (px_, py_, 0.03), (0.17, 0.17, 0.02), (0.55, 0.55, 0.58), root), i, t_in, tprobe)
                V(box("die_c", (px_, py_, 0.03), (0.17, 0.17, 0.02), colr, root, emit=1.0), i, tprobe, t_out + 0.12)
            else:
                V(box("die_g", (px_, py_, 0.03), (0.17, 0.17, 0.02), (0.55, 0.55, 0.58), root), i, t_in, t_out + 0.12)
        p0 = probed[0]
        K(root, F(i, t_in), loc=LP(i, (cx + 3.5, -3.0, z)), interp="BEZIER")
        K(root, F(i, t_t0), loc=LP(i, (cx - p0[0], -p0[1], z)), interp="CONSTANT")
        for kk, (px_, py_) in enumerate(probed):
            tprobe = t_t0 + (t_t1 - t_t0) * kk / n_probe
            K(root, F(i, tprobe), loc=LP(i, (cx - px_, -py_, z)), interp="CONSTANT")
        pl = probed[-1]
        K(root, F(i, t_t1), loc=LP(i, (cx - pl[0], -pl[1], z)), interp="BEZIER")
        K(root, F(i, t_out), loc=LP(i, (cx - 3.5, -3.0, z)), interp="CONSTANT")
    wl("CM300-STYLE WAFER PROBE STATION", (cx, 0, 4.6), i, 3.0, 4.8, size=0.22)
    wl("WAFERS IN / OUT, STAGE STEPS", (cx, -1.0, 4.0), i, 3.0, 4.8, size=0.18)
    sx0 = 47.0
    nsp = F(i, 10.0) - F(i, 4.8)
    sdir = os.path.join(TEX, "spec_s5")
    T.spectrum_sequence(sdir, "spec_s5", nsp, n_traces=30, hold_from=0.72, seed=5)
    spm = seq_material(os.path.join(sdir, "spec_s5_0001.png"), nsp, F(i, 4.8))
    V(box("spec_frame", LP(i, (sx0, -3, 1.4)), (3.5, 0.1, 2.0), CLAY["dark"]), i, 4.8, 10)
    V(plane("spec", LP(i, (sx0, -3.07, 1.4)), (3.2, 1.8), spm), i, 4.8, 10)
    wl("RING TRANSMISSION (LORENTZIAN DIPS)", (sx0, -3.1, 2.8), i, 4.8, 7.0, size=0.24)
    wl("SPEC WINDOW", (sx0, -3.1, 0.15), i, 4.8, 7.0, size=0.22, color=(0.4, 1.0, 0.5))
    hx = 58.0
    gary_at(c, i, 0.0, (hx, -3.5, 0), 0.0)
    mgr_at(c, i, 7.0, (hx + 3.5, -3.5, 0), -PI / 2)
    K(c.arm, F(i, 7.0), rot=(0.55, 0, 0), interp="CONSTANT")
    V(c.printout, i, 7.0, 8.4)
    pt = _text("1 / 10", 0.25, (0.1, 0.1, 0.1), (0, -0.62, 0), c.arm, "CENTER", "CENTER", rot=(PI / 2, 0, 0))
    V(pt, i, 7.0, 8.4)
    shot(i, 0.0, 0.3, (0, -3.5, 1.4), (0, 0, 0.7))
    shot(i, 0.3, 1.0, (-3.5, -5, 1.7), (0, 0, 0.8), (4.5, -5, 1.7), (8, 0, 0.8))
    shot(i, 1.0, 1.55, (4.5, -5, 1.7), (8, 0, 0.9), (11.5, -5, 1.7), (15, 0, 0.9))
    shot(i, 1.55, 2.05, (11.5, -5, 1.7), (15, 0, 0.9), (18.5, -5, 1.7), (22, 0, 0.9))
    shot(i, 2.05, 2.55, (18.5, -5, 1.7), (22, 0, 0.9), (25.5, -5, 1.7), (29, 0, 0.9))
    shot(i, 2.55, 3.3, (25.5, -5, 1.7), (29, 0, 0.9), (cx - 3.5, -6.5, 3.2), (cx, 0, 1.6))
    shot(i, 3.3, 4.8, (cx, -4.2, 3.3), (cx, -0.3, 1.7), (cx + 0.2, -3.8, 3.1), (cx, -0.3, 1.7))
    shot(i, 4.8, 7.0, (sx0, -6.2, 1.6), (sx0, -3, 1.4), (sx0, -5.2, 1.6), (sx0, -3, 1.4))
    shot(i, 7.0, 8.4, (hx + 1.8, -10.0, 2.2), (hx + 1.9, -3.5, 1.0), (hx + 1.8, -9.5, 2.2), (hx + 1.9, -3.5, 1.0))
    shoot(c, i, (hx + 3.5, -3.5, 0), (hx, -3.5, 0), 8.4, 9.1, 5, stare_yaw=-PI / 2)
    shot(i, 8.4, 10.0, (hx + 1.8, -9.5, 2.2), (hx + 1.9, -3.5, 1.0), (hx + 1.8, -9.0, 2.2), (hx + 1.9, -3.5, 1.0))
    narr(i, [(0.2, 3.0, "Gary tests every optical chip at the factory."), (3.2, 5.6, "The rings all come out slightly different."),
             (5.8, 8.2, "Only one out of ten rings works.")])
    lab(i, 0.3, 3.3, "BACK THROUGH THE LINE"); lab(i, 3.3, 4.8, "WAFER-LEVEL TEST"); lab(i, 4.8, 7.0, "RING RESONANCES vs SPEC")
    big(i, 7.2, 8.4, "1 IN 10")
    card(i, 4.8, 7.0, "Illustrative Lorentzian ring transmission; 1 in 10 traces inside the spec window (by construction)")
    fx(i, 0.3, 3.3, "[FX: heavy motion-blur shuffle back through the line, speed ramp]")
    fx(i, 3.3, 4.8, "[FX: wafers fly in and out; stage steps; probe taps; die map glows]")
    fx(i, 4.8, 7.0, "[FX: new trace flashes white as it is measured]")
    fx(i, 9.1, 9.7, "[FX: muzzle flash, smoke ring, hole decal]")


# ------------------------------------------------------------------ S6
def s6(c):
    i = 5
    park(c, i)
    rnd = random.Random(44)
    V(box("rack6", LP(i, (0, 3.2, 1.1)), (1.4, 1.2, 2.2), CLAY["gray"]), i)
    tray = box("tray", LP(i, (0, 3.2, 0.8)), (1.1, 1.6, 0.22), CLAY["dark"])
    V(tray, i, 1.5, 10)
    at(tray, i, [(1.5, (0, 3.2, 0.8)), (3.2, (0, 1.2, 0.8))], interp="BEZIER")
    V(cyl("fiber_main", LP(i, (0, 1.5, 1.3)), 0.03, 24, (0.3, 0.9, 1.0), axis="X", emit=4.0), i, 0, 1.5)
    V(box("faceplate", LP(i, (2.5, 1.5, 1.3)), (0.35, 0.35, 0.35), CLAY["gray"]), i, 0.5, 1.5)
    for n in range(8):
        V(cyl("endface", LP(i, (2.5 - 0.12 + 0.08 * (n % 4), 1.32, 1.3 - 0.06 + 0.12 * (n // 4))), 0.025, 0.04,
              (rnd.random(), rnd.random(), rnd.random()), axis="Y"), i, 0.9, 1.5)
    wl("CONNECTOR JAM", (2.5, 1.5, 1.9), i, 0.9, 1.5, size=0.25)
    pal = [(1.0, 0.5, 0.1), (0.3, 0.9, 1.0), (1.0, 0.9, 0.2), (1.0, 0.4, 0.7), (0.3, 1.0, 0.4), (0.8, 0.8, 1.0)]
    fibers = []
    for n in range(120):
        t0 = 2.0 + 4.0 * n / 120
        ln = rnd.uniform(1.0, 2.4)
        pos = (rnd.uniform(-0.8, 0.8), rnd.uniform(0.4, 2.0), rnd.uniform(0.2, 0.7))
        f = cyl("fiber", LP(i, pos), 0.022, ln, rnd.choice(pal), emit=1.2)
        f.rotation_euler = (rnd.uniform(0, PI), rnd.uniform(0, PI), rnd.uniform(0, PI))
        V(f, i, t0, 10)
        fibers.append((f, pos))
    for f, pos in fibers[:45]:
        for s in range(14):
            K(f, F(i, 4.4 + s * 0.3), rot=(rnd.uniform(0, PI), rnd.uniform(0, PI), rnd.uniform(0, PI)), interp="BEZIER")
    gary_at(c, i, 3.2, (0.4, -0.3, 0), PI, tilt=0.7)
    K(c.g, F(i, 8.1), loc=LP(i, (0.4, -0.3, 0)), rot=(0, 0, PI), interp="CONSTANT")
    K(c.g, F(i, 8.15), loc=LP(i, (0.4, 0.05, 0)), rot=(0, 0, PI), interp="CONSTANT")
    K(c.gt, F(i, 8.1), rot=(0.7, 0, 0), interp="LINEAR")
    K(c.gt, F(i, 8.25), rot=(-0.45, 0, 0), interp="BEZIER")
    for w in c.whip_parts:
        V(w, i, 0, 10)
    thr = cyl("fiber_thru", (-0.1, 0, 0.5), 0.022, 1.3, (1.0, 0.5, 0.1), axis="Y", parent=c.gt, emit=1.5)
    V(thr, i, 5.6, 10)
    mpos = (1.7, -0.5)
    myaw = yaw_to(mpos, (0.4, -0.3))
    K(c.m, F(i, 6.4), loc=LP(i, (4.6, -0.5, 0)), rot=(0, 0, myaw), interp="LINEAR")
    K(c.m, F(i, 7.2), loc=LP(i, (mpos[0], mpos[1], 0)), rot=(0, 0, myaw), interp="CONSTANT")
    K(c.larm, F(i, 7.2), rot=(1.45, 0, 0), interp="BEZIER")
    K(c.larm, F(i, 7.8), rot=(-0.9, 0, 0), interp="LINEAR")
    K(c.larm, F(i, 8.0), rot=(-0.9, 0, 0), interp="LINEAR")
    K(c.larm, F(i, 8.13), rot=(0.9, 0, 0), interp="BEZIER")
    K(c.larm, F(i, 9.5), rot=(0.9, 0, 0), interp="CONSTANT")
    face(c, i, 7.2, 0.5); face(c, i, 8.0, 1.0)
    big(i, 8.1, 8.7, "CRACK")
    nv = make_person("nvda", i, (7.0, -1.8, 0), (0.46, 0.73, 0.0), "NVYDIA", (0, 0, 0), win=(8.2, 10), logo_size=0.1,
                     badge=(0.25, 0.45, 0.0))
    K(nv.root, F(i, 8.2), loc=LP(i, (7.0, -1.8, 0)), rot=(0, 0, -PI / 2), interp="LINEAR")
    K(nv.root, F(i, 8.8), loc=LP(i, (2.2, -1.9, 0)), rot=(0, 0, -PI / 2), interp="CONSTANT")
    K(nv.root, F(i, 8.85), loc=LP(i, (2.2, -1.9, 0)), rot=(0, 0, PI), interp="CONSTANT")
    K(nv.arm_r, F(i, 8.2), rot=(0, 0, 0), interp="CONSTANT")
    K(nv.arm_r, F(i, 8.85), rot=(0, 0, 0), interp="LINEAR")
    K(nv.arm_r, F(i, 9.0), rot=(-1.57, 0, -0.9), interp="LINEAR")
    K(nv.arm_r, F(i, 9.15), rot=(-1.57, 0, 0.9), interp="CONSTANT")
    big(i, 9.1, 9.7, "SLAP")
    K(c.m, F(i, 9.15), loc=LP(i, (mpos[0], mpos[1], 0)), rot=(0, 0, myaw + 0.9), interp="CONSTANT")
    hg = make_person("llm", i, (6.0, -1.8, 0), (0.15, 0.15, 0.18), "OPENAY\nANTHROPY", (1, 1, 1), scale=0.8,
                     win=(8.4, 10), logo_size=0.07, badge=(0.35, 0.2, 0.1))
    K(hg.root, F(i, 8.4), loc=LP(i, (6.0, -1.9, 0)), rot=(0, 0, -PI / 2), interp="LINEAR")
    K(hg.root, F(i, 9.0), loc=LP(i, (2.55, -1.9, 0)), rot=(0, 0, -PI / 2), interp="CONSTANT")
    K(hg.arm_r, F(i, 9.0), rot=(-1.4, 0, 0), interp="CONSTANT")
    wl("CUSTOMER", (2.2, -1.9, 2.1), i, 8.9, 10, size=0.2)
    shot(i, 0.0, 0.9, (-3, -1.5, 1.3), (0, 1.5, 1.3), (1.5, -1.0, 1.3), (3, 1.5, 1.3))
    shot(i, 0.9, 1.5, (2.0, 0.4, 1.4), (2.5, 1.5, 1.3), (2.3, 0.6, 1.35), (2.5, 1.5, 1.3), lens=40)
    shot(i, 1.5, 3.2, (-2.5, -6, 1.6), (0, 3, 1.0), (0.5, -5, 1.6), (0, 2.5, 0.8))
    shot(i, 3.2, 5.6, (1.8, -3.0, 1.2), (0, 0.8, 0.5), (0.9, -2.2, 1.0), (0, 0.8, 0.5))
    shot(i, 5.6, 7.4, (-1.4, -2.6, 0.9), (0.4, -0.3, 0.6), (-0.6, -2.2, 0.8), (0.4, -0.3, 0.6))
    shot(i, 7.4, 8.0, (4.5, -3.0, 1.6), (1.7, -0.5, 1.3), (3.2, -2.6, 1.5), (1.7, -0.5, 1.3))
    shot(i, 8.0, 8.6, (1.6, -2.4, 1.1), (0.6, -0.2, 0.9), (1.4, -2.2, 1.1), (0.6, -0.2, 0.9))
    shot(i, 8.6, 10.0, (1.5, -6.8, 1.9), (1.5, -0.8, 0.9), (1.5, -6.3, 1.8), (1.5, -0.8, 0.9), lens=26)
    narr(i, [(0.2, 3.4, "Last job: Gary pulls out the tray and ties up the fibers."), (3.4, 4.6, "Thousands of them."),
             (5.4, 8.0, "Manager wants it done yesterday.")])
    lab(i, 0.0, 1.5, "FIBER"); lab(i, 8.6, 10.0, "THE CUSTOMER")
    card(i, 0.0, 1.5, "SMF-28: <= 0.18 dB/km at 1550 nm (Corning); 0.4 dB per connector adds up")
    card(i, 3.2, 8.0, "9,072 fibers per rack: hypothetical NVL576-style, 200G per fiber (own estimate)")
    fx(i, 1.5, 3.2, "[FX: tray slides out, fibers spill like noodles]")
    fx(i, 4.4, 8.0, "[FX: fibers squirm, one threads through Gary's bullet hole]")
    fx(i, 7.4, 8.7, "[FX: bundle of active optical cables as the whip; crack shockwave]")
    fx(i, 9.0, 9.8, "[FX: slap shockwave, Manager's head snaps; customer hug squash]")


# ------------------------------------------------------------------ S7
def s7():
    i = 6
    V(box("black", LP(i, (0, 6, 2)), (14, 0.1, 18), (0, 0, 0), emit=100), i, 0, 2)
    shot(i, 0.0, 2.0, (0, 0, 2), (0, 6, 2))
    ovt("DISC", "DISCLAIMER:\nGary is fictional.\nResults not typical.\nFigures vary by standard,\nvendor and test condition.\n"
                "Parody; not affiliated\nwith any company.", i, 0, 2.0)
    src = ("SOURCES\n"
           "IEEE 802.3bj-2014, 802.3ck, 802.3dj task-force documents: copper reach objectives\n"
           "SemiAnalysis, 'Nvidia's Optical Boogeyman': NVL72 copper backplane\n"
           "Arista and Ciena slides, OFC 2026; Broadcom 800G slides, 2023: power per bit\n"
           "Cheng et al., Opt. Express 33, 24190 (2025): pluggable vs NPO vs CPO\n"
           "Broadcom TH5-51.2T Bailly CPO deck; NVIDIA Developer Blog, Scaling AI Factories with CPO\n"
           "Corning SMF-28 spec PI-1424; own hypothetical NVL576-style fiber count\n"
           "Package layout after NVIDIA's public Rubin Ultra slide; probe station after FormFactor CM300 photo\n"
           "Eye, BER, ring-spectrum, heat-wave and decay animations: illustrative models, not measurements\n"
           "Style reference: Zack D. Films")
    ovt("SRC", src, i, 0, 2.0)
    ovt("FX", "[VO ~9 words/s: Gary is fictional. Results not typical. Figures vary by standard, vendor and mood. "
              "Void where copper is cheaper.]", i, 0, 2.0)


# ------------------------------------------------------------------ main
def build(out_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = int(os.environ.get("CRUDE_SAMPLES", "12"))
    scn.render.resolution_x = int(os.environ.get("CRUDE_W", "720"))
    scn.render.resolution_y = int(os.environ.get("CRUDE_H", "900"))
    scn.render.fps = 30
    scn.render.image_settings.file_format = "PNG"
    scn.view_settings.view_transform = "Standard"
    scn.frame_start = 1
    scn.frame_end = 1860
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.62, 0.76, 0.95, 1)
    bg.inputs["Strength"].default_value = 0.85
    scn.world = w
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.8
    sun.angle = 0.2
    so = bpy.data.objects.new("sun", sun)
    scn.collection.objects.link(so)
    so.rotation_euler = (0.9, 0.25, 0.6)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(300, 0, 0))
    fl = bpy.context.active_object
    fl.scale = (2000, 2000, 1)
    fl.name = "floor"
    fm = bpy.data.materials.new("floor")
    fm.use_nodes = True
    nt = fm.node_tree
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.inputs["Scale"].default_value = 1000.0
    chk.inputs["Color1"].default_value = (0.52, 0.54, 0.46, 1)
    chk.inputs["Color2"].default_value = (0.46, 0.49, 0.41, 1)
    uv = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(uv.outputs["UV"], chk.inputs["Vector"])
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Roughness"].default_value = 0.95
    nt.links.new(chk.outputs["Color"], bs.inputs["Base Color"])
    fl.data.materials.append(fm)
    VA(fl, 1, F(6, 0))
    L.setup_camera()
    c = make_cast()
    s1(c); s2(c); s3(c); s4(c); s5(c); s6(c); s7()
    for i in range(6):
        for s in range(10):
            L.ovt("TC", "S%d  %d:%02d" % (i + 1, 0, s), i, s, s + 1)
    for s in range(2):
        L.ovt("TC", "S7  0:%02d" % s, 6, s, s + 1)
    L.finalize_visibility(1860)
    L.schedule_holes()
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("SAVED", out_path)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    build(argv[0] if argv else "crude_v0_2.blend")
