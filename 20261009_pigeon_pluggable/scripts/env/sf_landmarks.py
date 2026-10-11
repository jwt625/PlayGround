"""San Francisco backdrop for the S6 street (workstream E, DevLog-013 section 9; owner v3 feedback, DevLog-018 V5).

Generic low-poly silhouettes with window emission, placed in the vista at the end of the street so they read from S's
S6 cameras: a steep hill street rising straight ahead with Victorian row houses and a cable car, a frontal "painted
ladies" row at the hilltop, and behind it a downtown cluster with a pyramid tower and a tall rounded tower with a lit
crown, a hilltop fluted tower, a red-orange suspension bridge in fog on the left and a grey suspension bridge with
cable lights on the right. No logos, no text marks. Distances are compressed (200-650 m) so the landmarks rise above the
hill line (about 17-20 deg up from the street) at 540 x 674.
"""
import math
import random

import bpy
from mathutils import Vector

import env_lib as E
import street_kit as SK
from env_lib import MB, PI

import vfx

VIEW = Vector((-3.0, 14.0, 1.6))          # S6 street eye point used for the placement checks
HILL_Y0, HILL_Y1, HILL_Z1, PLATEAU_Y = 37.5, 118.0, 18.0, 170.0
LADIES_Y = 124.0
PASTEL = [(0.42, 0.24, 0.32), (0.22, 0.32, 0.46), (0.48, 0.42, 0.2), (0.24, 0.38, 0.3), (0.36, 0.3, 0.46), (0.5, 0.48, 0.44),
          (0.5, 0.28, 0.2)]
LANDMARKS = {  # name: base point (x, y) and height (m), for the JSON / hooks
    "pyramid": ((-7.0, 530.0), 260.0), "round_tower": ((125.0, 600.0), 326.0), "hill_tower": ((28.0, 340.0), 159.0),
    "gg_tower_1": ((-56.0, 590.0), 284.0), "gg_tower_2": ((-110.0, 800.0), 284.0),
    "bay_tower_1": ((210.0, 590.0), 250.0), "bay_tower_2": ((320.0, 780.0), 250.0),
}


def hill_z(y):
    if y <= HILL_Y0:
        return 0.0
    if y >= HILL_Y1:
        return HILL_Z1
    return HILL_Z1 * (y - HILL_Y0) / (HILL_Y1 - HILL_Y0)


SLOPE = math.atan2(HILL_Z1, HILL_Y1 - HILL_Y0)


def _mats():
    m = {"trim": E.principled("SF_trim", (0.62, 0.6, 0.56), 0.0, 0.6), "roof": E.principled("SF_roof", (0.05, 0.05, 0.055), 0.0, 0.7)}
    for i, c in enumerate(PASTEL):
        m["paint_%d" % i] = E.principled("SF_paint_%d" % i, c, 0.0, 0.65, c, 0.16)
    return m


# ================================================================== hill street + cable car
def hill_street(c, lc, hc):
    mb = MB()
    y0, y1, z1 = HILL_Y0, HILL_Y1, HILL_Z1
    mb.quad((-4.0, y0, 0.0), (4.0, y0, 0.0), (4.0, y1, z1), (-4.0, y1, z1), m=0, uv=False)
    mb.quad((-4.0, y1, z1), (4.0, y1, z1), (4.0, PLATEAU_Y, z1), (-4.0, PLATEAU_Y, z1), m=0, uv=False)
    for s in (-1, 1):
        for (ya, za, yb, zb) in ((y0, 0.0, y1, z1), (y1, z1, PLATEAU_Y, z1)):
            xa, xb = s * 4.0, s * 5.6
            mb.quad((min(xa, xb), ya, za + 0.15), (max(xa, xb), ya, za + 0.15), (max(xa, xb), yb, zb + 0.15), (min(xa, xb), yb, zb + 0.15),
                    m=1, uv=False)
    for x, w in ((-0.55, 0.08), (0.55, 0.08), (0.0, 0.05)):
        for (ya, za, yb, zb) in ((y0, 0.0, y1, z1), (y1, z1, PLATEAU_Y, z1)):
            mb.quad((x - w / 2, ya, za + 0.01), (x + w / 2, ya, za + 0.01), (x + w / 2, yb, zb + 0.01), (x - w / 2, yb, zb + 0.01), m=2,
                    uv=False)
    E.obj("sf_hill_street", mb, ["asphalt", "sidewalk", "metal_dark"], c)
    for k, (s, y) in enumerate(((-1, 48.0), (1, 64.0), (-1, 80.0), (1, 96.0), (-1, 112.0))):
        lp = SK.lamp_post(c, (s * 5.0, y, hill_z(y) + 0.15), 0.0 if s < 0 else PI, "sf_lamp_%d" % k)
        E.light("sf_lamp_%d_spot" % k, "SPOT", tuple(lp["head"]), 420, (1.0, 0.76, 0.5), lc, spot_deg=120, blend=0.8, rot=(0, 0, 0),
                radius=0.1, shadow=False)


def cable_car(c, y=70.0, name="sf_cable_car"):
    """Classic two-section cable car (8.5 m) on the hill street centre line, pitched to the slope; generic, no marks."""
    root = E.empty(name + "_root", c, (0.0, y, hill_z(y)), (SLOPE, 0, 0), 0.5)
    mb = MB()
    mb.box((0, 0, 0.75), (2.4, 8.5, 0.9), m=0)
    mb.box((0, 0, 1.25), (2.45, 8.55, 0.08), m=1)
    mb.box((0, 0.9, 2.05), (2.4, 4.6, 1.5), m=1)
    for s in (-1, 1):
        for k in range(5):
            yy = -1.0 + 0.9 * k
            mb.quad((s * 1.205, yy - 0.35, 1.45), (s * 1.205, yy + 0.35, 1.45), (s * 1.205, yy + 0.35, 2.55), (s * 1.205, yy - 0.35, 2.55),
                    m=2)
        for k in range(4):
            mb.box((s * 1.15, -2.0 - 0.6 * k, 2.05), (0.06, 0.06, 1.55), m=3)
            mb.box((s * 1.15, 3.3 + 0.0 * k, 2.05), (0.06, 0.06, 1.55), m=3)
    mb.box((0, 0, 2.92), (2.6, 8.8, 0.12), m=1)
    mb.box((0, 0.9, 3.05), (1.2, 4.0, 0.18), m=1)
    mb.box((0, -2.2, 1.7), (2.2, 3.0, 0.08), m=3)
    for yy in (-2.8, 2.8):
        mb.box((0, yy, 0.2), (1.8, 1.4, 0.35), m=4)
    mb.ellipsoid((0, 4.3, 1.0), (0.16, 0.05, 0.16), 12, 6, m=5)
    mb.ellipsoid((0, -4.3, 1.0), (0.12, 0.05, 0.12), 10, 6, m=6)
    mb.box((0, 4.26, 2.75), (1.6, 0.04, 0.1), m=5)
    o = E.obj(name + "_body", mb, [E.principled("SF_car_maroon", (0.32, 0.04, 0.05), 0.2, 0.4, (0.32, 0.04, 0.05), 0.08),
                                     E.principled("SF_car_cream", (0.62, 0.55, 0.4), 0.0, 0.5, (0.62, 0.55, 0.4), 0.05), "win_warm",
                                     "metal_dark", "rubber", "lamp_lens", E.principled("SF_car_tail", (0.6, 0.02, 0.02), 0, 0.3,
                                                                                     (1.0, 0.05, 0.03), 4.0)], c)
    o.parent = root
    hl = E.light(name + "_headlight", "SPOT", (0, 4.4, 1.0), 300, (1.0, 0.92, 0.8), c, spot_deg=50, blend=0.5, rot=(-PI / 2, 0, 0),
                 radius=0.1, shadow=False)
    hl.parent = root
    il = E.light(name + "_interior", "POINT", (0, 0.5, 2.1), 60, (1.0, 0.75, 0.5), c, radius=0.6, shadow=False)
    il.parent = root
    return root


def cable_car_path(u):
    """World point of the car base on the centre line at path fraction u (0 = y 42 near the cross street, 1 = y 114)."""
    y = 42.0 + (114.0 - 42.0) * u
    return Vector((0.0, y, hill_z(y)))


# ================================================================== Victorian row houses
def _house(mb, m, idx, rng, gable=True, bay=True, h=10.0, w=6.0, d=12.0, plinth=0.0):
    """Local frame: facade plane y = 0 facing +y, width along x, body from z = -plinth (garage) to h."""
    pm = m["paint_%d" % (idx % len(PASTEL))]
    mb.box((0, -d / 2, (h - plinth) / 2), (w, d, h + plinth), pm)
    if plinth > 0.2:
        mb.quad((-w / 2 + 0.6, 0.005, -plinth + 0.05), (-0.2, 0.005, -plinth + 0.05), (-0.2, 0.005, -0.1), (-w / 2 + 0.6, 0.005, -0.1),
                m["dark"])
    mb.box((0, 0.2, h - 0.25), (w + 0.1, 0.5, 0.5), m["trim"])
    mb.box((0, 0.1, 3.3), (w, 0.25, 0.15), m["trim"])
    if bay:
        for z0 in (0.6, 3.8):
            mb.box((-1.4, 0.45, z0 + 1.35), (2.4, 0.9, 2.9), pm)
            mb.box((-1.4, 0.95, z0 + 2.85), (2.6, 0.12, 0.12), m["trim"])
            for xx in (-2.2, -1.4, -0.6):
                kind = rng.choice(("win_warm", "win_warm", "win_cool", "glass_dark", "win_pink"))
                mb.quad((xx - 0.3, 0.905, z0 + 0.3), (xx + 0.3, 0.905, z0 + 0.3), (xx + 0.3, 0.905, z0 + 2.4), (xx - 0.3, 0.905, z0 + 2.4),
                        m[kind])
    for z0 in (0.6, 3.8, 6.6):
        if z0 < h - 1.5:
            kind = rng.choice(("win_warm", "glass_dark", "win_cool", "win_warm"))
            mb.quad((1.0, 0.01, z0 + 0.3), (2.2, 0.01, z0 + 0.3), (2.2, 0.01, z0 + 2.2), (1.0, 0.01, z0 + 2.2), m[kind])
            mb.box((1.6, 0.06, z0 + 2.35), (1.5, 0.12, 0.15), m["trim"])
    mb.quad((0.2, 0.012, 0.0), (0.9, 0.012, 0.0), (0.9, 0.012, 2.3), (0.2, 0.012, 2.3), m["dark"])
    for k in range(3):
        mb.box((0.55, 0.2 + 0.25 * k, -0.15 - 0.17 * k), (0.9, 0.3, 0.17), m["trim"])
    if gable:
        top = h + 2.6
        v = [(-w / 2 - 0.1, 0.3, h), (w / 2 + 0.1, 0.3, h), (0, 0.3, top), (-w / 2 - 0.1, -d, h), (w / 2 + 0.1, -d, h), (0, -d, top)]
        mb._add(v, [(0, 1, 2), (3, 5, 4)], m["trim"])
        mb._add(v, [(0, 2, 5, 3), (1, 4, 5, 2)], m["roof"])
    else:
        mb.box((0, 0.3, h + 0.35), (w + 0.3, 0.7, 0.3), m["trim"])


def _house_mats():
    names = ["dark", "win_warm", "win_cool", "glass_dark", "win_pink", "trim", "roof"] + ["paint_%d" % i for i in range(len(PASTEL))]
    sm = _mats()
    mats = []
    for n in names:
        if n == "dark":
            mats.append(E.M("plastic_dark"))
        elif n in sm:
            mats.append(sm[n])
        else:
            mats.append(n)
    return mats, {n: i for i, n in enumerate(names)}


def victorians(c, seed=5):
    rng = random.Random(seed)
    mats, m = _house_mats()
    mb = MB()
    k = 0
    for s in (-1, 1):
        rz = -PI / 2 if s < 0 else PI / 2              # local +y (facade normal) toward the street centre
        for i in range(12):
            y = 50.0 + 6.2 * i
            if y + 3.1 > HILL_Y1 + 2:
                break
            zl, zh = hill_z(y - 3.0), hill_z(y + 3.0)
            with mb.at((s * 5.9, y, zh), rz):
                _house(mb, m, k, rng, gable=(k % 3 != 1), plinth=zh - zl + 0.4, h=9.5 + 1.5 * (k % 2))
            k += 1
    for i in range(7):                                  # frontal "painted ladies" row at the hilltop, facing the street
        x = -18.0 + 6.0 * i
        with mb.at((x, LADIES_Y, HILL_Z1), PI):
            _house(mb, m, i + 3, rng, gable=True, plinth=0.4, h=10.5)
    o = E.obj("sf_victorians", mb, mats, c)
    gm = MB()
    gm.box((0.0, (LADIES_Y + PLATEAU_Y) / 2 + 8, HILL_Z1 / 2), (80.0, PLATEAU_Y - LADIES_Y + 16, HILL_Z1), m=0)
    for s in (-1, 1):
        gm._add([(s * 5.6, HILL_Y0, 0.0), (s * 40.0, HILL_Y0, 0.0), (s * 40.0, HILL_Y1, HILL_Z1), (s * 5.6, HILL_Y1, HILL_Z1)],
                [(0, 1, 2, 3)], 0)
        gm._add([(s * 5.6, HILL_Y1, HILL_Z1), (s * 40.0, HILL_Y1, HILL_Z1), (s * 40.0, LADIES_Y, HILL_Z1), (s * 5.6, LADIES_Y, HILL_Z1)],
                [(0, 1, 2, 3)], 0)
    E.obj("sf_hill_mass", gm, [E.principled("SF_hill", (0.02, 0.022, 0.025), 0.0, 0.9)], c)
    return o


# ================================================================== downtown, landmarks
def _prism(mb, cx, cy, z0, z1, r0, r1, n, mi, rot=0.0, squircle=0.0):
    """n-gon prism from z0 (radius r0) to z1 (radius r1); squircle > 0 rounds a square section."""
    def ring(z, r):
        pts = []
        for i in range(n):
            a = rot + 2 * PI * i / n
            ca, sa = math.cos(a), math.sin(a)
            if squircle:
                k = 1.0 / max(abs(ca), abs(sa)) ** (1.0 - squircle)
                pts.append((cx + r * ca * k, cy + r * sa * k, z))
            else:
                pts.append((cx + r * ca, cy + r * sa, z))
        return pts
    v = ring(z0, r0) + ring(z1, r1)
    f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    mb._add(v, f, mi)


def downtown(c, seed=3):
    rng = random.Random(seed)
    far = SK.window_grid_material("SF_window_far", seed=1.0, cell=(4.0, 7.0), lit=0.32, strength=(0.4, 1.3))
    light = SK.window_grid_material("SF_window_light", seed=2.0, cell=(3.0, 4.0), lit=0.5, strength=(0.6, 1.5), base=(0.55, 0.55, 0.58),
                                    glass=(0.1, 0.1, 0.11), base_emit=0.6)
    mb = MB()
    spots = [(42, 560, 150), (77, 505, 120), (-42, 590, 180), (168, 660, 200), (14, 660, 210), (224, 545, 130), (-70, 660, 140),
             (98, 730, 230), (-28, 755, 170), (154, 810, 190), (56, 840, 160), (-84, 530, 110), (266, 660, 150), (196, 900, 200)]
    for x, y, h in spots:
        w, d = rng.uniform(26, 40), rng.uniform(26, 40)
        mb.box((x, y, h / 2), (w, d, h), m=0)
        if rng.random() < 0.4:
            mb.box((x, y, h + 6), (w * 0.6, d * 0.6, 12), m=0)
    o = E.obj("sf_downtown", mb, [far], c)
    # pyramid tower with two side wings and a spire
    (px, py), ph = LANDMARKS["pyramid"]
    pm = MB()
    _prism(pm, px, py, 0.0, 212.0, 37.0, 1.5, 4, 0, rot=PI / 4)
    _prism(pm, px, py, 212.0, ph, 1.5, 0.15, 4, 1, rot=PI / 4)
    for s in (-1, 1):
        pm._add([(px + s * 22, py - 6, 20), (px + s * 22, py + 6, 20), (px + s * 9, py + 6, 160), (px + s * 9, py - 6, 160),
                 (px + s * 26, py - 6, 20), (px + s * 26, py + 6, 20), (px + s * 13, py + 6, 160), (px + s * 13, py - 6, 160)],
                [(0, 1, 5, 4), (4, 5, 6, 7), (1, 2, 6, 5), (3, 0, 4, 7), (2, 3, 7, 6)], 0)
    pm.ellipsoid((px, py, 212.0), (2.2, 2.2, 2.2), 10, 6, m=2)
    E.obj("sf_pyramid", pm, [light, "galv", E.principled("SF_beacon_white", (1, 1, 1), 0, 0.3, (1.0, 0.95, 0.9), 30.0)], c)
    # tall rounded tower with a glowing crown
    (sx, sy), sh = LANDMARKS["round_tower"]
    sm = MB()
    _prism(sm, sx, sy, 0.0, 290.0, 27.0, 21.0, 24, 0, squircle=0.55)
    _prism(sm, sx, sy, 290.0, sh - 8, 21.0, 15.0, 24, 1, squircle=0.55)
    _prism(sm, sx, sy, sh - 8, sh, 15.0, 7.0, 24, 1, squircle=0.55)
    E.obj("sf_round_tower", sm, [SK.window_grid_material("SF_window_round", seed=4.0, cell=(3.0, 5.0), lit=0.3, strength=(0.4, 1.2),
                                                           base=(0.12, 0.13, 0.15)),
                                  E.principled("SF_crown", (0.8, 0.85, 0.95), 0, 0.3, (0.75, 0.82, 1.0), 3.0)], c)
    # hilltop fluted tower on a lit hill
    (cx, cy), ch = LANDMARKS["hill_tower"]
    hm = MB()
    hm.ellipsoid((cx, cy, -20.0), (120.0, 90.0, 115.0), 24, 12, m=0)
    _prism(hm, cx, cy, 90.0, 151.0, 5.5, 5.0, 16, 1)
    _prism(hm, cx, cy, 151.0, ch, 6.0, 5.6, 16, 2)
    _prism(hm, cx, cy, 86.0, 92.0, 12.0, 12.0, 16, 2)
    E.obj("sf_hill_tower", hm, [SK.window_grid_material("SF_hill_lights", seed=5.0, cell=(6.0, 6.0), lit=0.18, strength=(0.4, 1.0),
                                                          base=(0.02, 0.025, 0.02), glass=(0.02, 0.025, 0.02), win=(0.38, 0.38, 0.38)),
                                 E.principled("SF_flood_warm", (0.75, 0.62, 0.45), 0, 0.6, (1.0, 0.82, 0.6), 1.1),
                                 E.principled("SF_flood_warm_dim", (0.5, 0.42, 0.32), 0, 0.6, (1.0, 0.82, 0.6), 0.5)], c)
    return o


def _suspension(c, name, t1, t2, h, deck_z, half_w, mats, n_struts=4, xbrace=False, led=False, ext=90.0):
    """Two-tower suspension bridge between tower bases t1, t2 (x, y). mats: [tower, cable, deck, light, beacon]."""
    p1, p2 = Vector((t1[0], t1[1], 0)), Vector((t2[0], t2[1], 0))
    d = (p2 - p1).normalized()
    nrm = Vector((d.y, -d.x, 0))
    mb = MB()
    for p in (p1, p2):
        for s in (-1, 1):
            leg = p + nrm * s * half_w
            mb.rod(tuple(leg), tuple(leg + Vector((0, 0, h))), 4.0, 4, m=0, r2=3.0)
        for k in range(n_struts):
            z = deck_z + (h - deck_z) * (k + 0.6) / n_struts
            mb.rod(tuple(p - nrm * half_w + Vector((0, 0, z))), tuple(p + nrm * half_w + Vector((0, 0, z))), 2.2, 4, m=0)
        if xbrace:
            for k in range(3):
                za = deck_z + (h - deck_z) * k / 3
                zb = deck_z + (h - deck_z) * (k + 1) / 3
                mb.rod(tuple(p - nrm * half_w + Vector((0, 0, za))), tuple(p + nrm * half_w + Vector((0, 0, zb))), 1.0, 4, m=0)
                mb.rod(tuple(p + nrm * half_w + Vector((0, 0, za))), tuple(p - nrm * half_w + Vector((0, 0, zb))), 1.0, 4, m=0)
        mb.ellipsoid(tuple(p + Vector((0, 0, h + 2))), (2.0, 2.0, 2.0), 8, 4, m=4)
    a1, a2 = p1 - d * ext, p2 + d * ext
    mb.seg(tuple(a1 + Vector((0, 0, deck_z))), tuple(a2 + Vector((0, 0, deck_z))), 2 * half_w + 4, 6.0, m=2)
    lights = MB()
    for s in (-1, 1):
        off = nrm * s * half_w
        t1t, t2t = p1 + off + Vector((0, 0, h)), p2 + off + Vector((0, 0, h))
        main = E.catenary(t1t, t2t, h - deck_z - 6.0, 30)
        mb.tube(main, 0.9, 6, m=1)
        mb.tube(E.catenary(a1 + off + Vector((0, 0, deck_z + 3)), t1t, 12.0, 12), 0.9, 6, m=1)
        mb.tube(E.catenary(t2t, a2 + off + Vector((0, 0, deck_z + 3)), 12.0, 12), 0.9, 6, m=1)
        for q in main[1:-1]:
            mb.rod(tuple(q), (q.x, q.y, deck_z + 3.0), 0.25, 4, m=1)
            if led:
                for j in range(5):
                    zz = deck_z + 3.0 + (q.z - deck_z - 3.0) * (j + 0.5) / 5
                    lights.ellipsoid((q.x, q.y, zz), (0.9, 0.9, 0.9), 6, 4, m=0)
        for k in range(40):
            u = k / 39
            lp = a1.lerp(a2, u) + off * 1.1 + Vector((0, 0, deck_z + 3.5))
            lights.ellipsoid(tuple(lp), (1.0, 1.0, 1.0), 6, 4, m=1)
    o = E.obj(name, mb, mats[:3] + [mats[3], mats[4]], c)
    lo = E.obj(name + "_lights", lights, [E.principled(name + "_led", (1, 1, 1), 0, 0.3, (0.9, 0.95, 1.0), 8.0), mats[3]], c, shadow=False)
    return o, lo


def bridges(c):
    beacon = E.principled("SF_beacon_red", (1, 0.05, 0.03), 0, 0.3, (1.0, 0.05, 0.03), 20.0)
    deck_light = E.principled("SF_deck_light", (1, 0.8, 0.5), 0, 0.3, (1.0, 0.75, 0.45), 10.0)
    orange = E.principled("SF_bridge_orange", (0.6, 0.14, 0.05), 0.2, 0.5, (0.85, 0.22, 0.08), 0.55)
    grey = E.principled("SF_bridge_grey", (0.3, 0.32, 0.35), 0.5, 0.4, (0.5, 0.55, 0.62), 0.15)
    gg = _suspension(c, "sf_red_bridge", LANDMARKS["gg_tower_1"][0], LANDMARKS["gg_tower_2"][0], 284.0, 84.0, 14.0,
                     [orange, orange, orange, deck_light, beacon], n_struts=4)
    bay = _suspension(c, "sf_grey_bridge", LANDMARKS["bay_tower_1"][0], LANDMARKS["bay_tower_2"][0], 250.0, 75.0, 12.0,
                      [grey, grey, grey, deck_light, beacon], n_struts=2, xbrace=True, led=True)
    return gg, bay


def fog_and_hills(c):
    vfx.haze_cards("sf_fog_bridge", (-85.0, 700.0, 80.0), (180.0, 300.0, 150.0), n=5, color=(0.62, 0.62, 0.68), density=0.12,
                   normal_axis="Y", noise_scale=0.012, drift=1.5, height_falloff=1.8, collection=c, seed=7.0)
    vfx.haze_cards("sf_fog_bay", (80.0, 470.0, 35.0), (600.0, 80.0, 90.0), n=3, color=(0.55, 0.55, 0.62), density=0.07,
                   normal_axis="Y", noise_scale=0.01, drift=1.0, height_falloff=2.0, collection=c, seed=9.0)
    mb = MB()
    mb.ellipsoid((-320.0, 950.0, -120.0), (420.0, 160.0, 300.0), 24, 10, m=0)
    mb.ellipsoid((420.0, 1000.0, -150.0), (380.0, 160.0, 260.0), 24, 10, m=0)
    E.obj("sf_far_hills", mb, [E.principled("SF_far_hill", (0.006, 0.007, 0.01), 0.0, 0.9)], c)


def build(c, lc, hc, car_collection=None):
    """Whole SF backdrop into collection c (cable car into car_collection so S can animate its instancer)."""
    hill_street(c, lc, hc)
    victorians(c)
    downtown(c)
    bridges(c)
    fog_and_hills(c)
    car = cable_car(car_collection or c)
    p0, p1 = cable_car_path(0.0), cable_car_path(1.0)
    E.hook("cable_car", hc, tuple(car.location), PI / 2, "landmark", path_start=list(p0), path_end=list(p1),
           note="ENV_cable_car instancer: key location (path point - this location) to move the car up/down the hill")
    for nm, ((x, y), h) in LANDMARKS.items():
        E.hook("sf_" + nm, hc, (x, y, h), PI / 2, "landmark")
    E.hook("sf_ladies", hc, (0.0, LADIES_Y, HILL_Z1), -PI / 2, "landmark")
    return car
