"""Street kit generators (workstream E, DevLog-013): true-scale props for the night street (S1 opening, S6 ending).

Every prop builds in a local frame and is placed with (loc, yaw); each returns the main object (one multi-material mesh
per prop where possible) plus named local points in world space (for hooks / lights). Parody signage only.
"""
import math
import random

import bpy
from mathutils import Matrix, Vector

import env_lib as E
from env_lib import MB, PI

import vfx


def Rz(yaw):
    return Matrix.Rotation(yaw, 3, "Z")


def W(loc, yaw, p):
    return Vector(loc) + Rz(yaw) @ Vector(p)


def _root(name, c, loc, yaw):
    r = E.empty(name, c, loc, (0, 0, yaw), 0.3)
    bpy.context.view_layer.update()
    return r


def _child(o, root):
    """Parent o (created with local coordinates as its world transform) to root, keeping the local transform."""
    mw = o.matrix_world.copy()
    o.parent = root
    o.matrix_world = root.matrix_world @ mw
    return o


# ==================================================================== lamp post
def lamp_post(c, loc, yaw, name="lamp", H=4.6, arm=1.05, street_sign=None):
    """Post at loc, arm toward local +X. Top of post z = H (flat cap); arm tube top z = H - 0.02; head lens at the arm end."""
    mb = MB()
    mb.cyl((0, 0, 0.25), 0.15, 0.5, seg=16, m=1)
    mb.cyl((0, 0, 0.52), 0.1, 0.04, seg=16, m=1)
    mb.box((0.15, 0, 0.25), (0.012, 0.12, 0.22), m=1)
    mb.cyl((0, 0, (0.54 + H) / 2), 0.075, H - 0.54, seg=16, m=0, r2=0.055)
    mb.cyl((0, 0, H - 0.01), 0.065, 0.02, seg=16, m=0)
    mb.box((0.085, 0, 2.6), (0.05, 0.14, 0.22), m=1)
    mb.rod((0.04, 0, H - 0.05), (arm - 0.12, 0, H - 0.05), 0.03, 12, m=0)
    pts = [(0.05 + 0.62 * t, 0, (H - 0.6) + 0.52 * math.sin(t * PI / 2)) for t in [i / 10 for i in range(11)]]
    mb.tube(pts, 0.014, 8, m=0)
    mb.ring((0.32, 0, H - 0.27), 0.09, 0.075, 0.02, axis="y", seg=20, m=0)
    mb.ellipsoid((arm, 0, H - 0.07), (0.27, 0.13, 0.065), 18, 8, m=0)
    mb.ellipsoid((arm + 0.02, 0, H - 0.128), (0.2, 0.095, 0.012), 18, 4, m=2)
    o = E.obj(name, mb, ["metal_paint", "metal_dark", "lamp_lens"], c, loc=loc, rot=(0, 0, yaw), smooth=False)
    out = {"obj": o, "head": W(loc, yaw, (arm + 0.02, 0, H - 0.15)), "top": W(loc, yaw, (0, 0, H)),
           "arm": lambda d: W(loc, yaw, (d, 0, H - 0.02))}
    if street_sign:
        street_sign_plate(c, W(loc, yaw, (0, 0, 2.95)), yaw, street_sign, name + "_streetsign")
    return out


def street_sign_plate(c, loc, yaw, text, name):
    root = _root(name, c, loc, yaw + PI / 2)
    mb = MB()
    mb.box((0, 0, 0), (0.78, 0.025, 0.17), m=0)
    mb.box((0, 0, 0.1), (0.04, 0.04, 0.04), m=1)
    o = E.obj(name + "_plate", mb, [E.principled("ENV_streetsign_green", (0.02, 0.12, 0.06), 0.3, 0.4), "metal_dark"], c)
    _child(o, root)
    for s in (1, -1):
        t = E.text(name + "_txt", text, 0.09, c, (0, -0.0135 * s, 0), (PI / 2, 0, 0 if s > 0 else PI),
                   mat=E.principled("ENV_streetsign_white", (0.85, 0.85, 0.85), 0, 0.4, (0.85, 0.85, 0.85), 0.4), fnt="Arial Bold")
        _child(t, root)
    return root


# ==================================================================== utility pole + wires
def utility_pole(c, loc, yaw=0.0, name="pole", H=6.4, arm_z=5.85, transformer=False):
    """Wood pole; crossarm along local X with 3 pin insulators at x = -0.5, 0, 0.5 (wire attach z = arm_z + 0.1)."""
    mb = MB()
    mb.cyl((0, 0, H / 2), 0.13, H, seg=14, m=0, r2=0.1)
    mb.box((0, 0, arm_z), (1.4, 0.09, 0.09), m=0)
    for s in (-1, 1):
        mb.seg((s * 0.45, 0.06, arm_z - 0.045), (s * 0.06, 0.06, arm_z - 0.45), 0.04, 0.01, m=1)
    for dx in (-0.5, 0.0, 0.5):
        mb.cyl((dx, 0, arm_z + 0.0725), 0.02, 0.055, seg=10, m=2)
        for zz in (arm_z + 0.06, arm_z + 0.085):
            mb.cyl((dx, 0, zz), 0.036, 0.008, seg=12, m=2)
    mb.box((0, 0.13, 4.9), (0.06, 0.06, 0.12), m=1)
    for k, zz in enumerate([2.4 + 0.4 * i for i in range(8)]):
        s = 1 if k % 2 else -1
        mb.rod((0, s * 0.1, zz), (0, s * 0.2, zz + 0.02), 0.009, 6, m=1)
    mb.box((0, -0.12, 2.0), (0.09, 0.006, 0.13), m=4)
    if transformer:
        mb.box((0.27, 0, 4.75), (0.16, 0.08, 0.06), m=1)
        mb.box((0.27, 0, 4.25), (0.16, 0.08, 0.06), m=1)
        mb.cyl((0.42, 0, 4.45), 0.25, 0.8, seg=18, m=3)
        mb.cyl((0.42, 0, 4.87), 0.26, 0.04, seg=18, m=3)
        for dx in (-0.1, 0.1):
            mb.cyl((0.42 + dx, 0, 4.95), 0.025, 0.12, seg=8, m=2)
        mb.tube([(0.32, 0, 5.0), (0.2, 0, 5.3), (-0.0, 0, arm_z + 0.06)], 0.006, 6, m=5)
        mb.tube([(0.52, 0, 5.0), (0.55, 0, 5.4), (0.5, 0, arm_z + 0.06)], 0.006, 6, m=5)
    o = E.obj(name, mb, ["wood", "galv", "insulator", "transformer", "paint_yellow", "cable"], c, loc=loc, rot=(0, 0, yaw))
    att = {dx: W(loc, yaw, (dx, 0, arm_z + 0.1)) for dx in (-0.5, 0.0, 0.5)}
    att["telecom"] = W(loc, yaw, (0, 0.17, 4.9))
    return {"obj": o, "attach": att, "arm_top": lambda dx: W(loc, yaw, (dx, 0, arm_z + 0.045))}


def wires(c, spans, name="wires"):
    """spans: list of (p0, p1, sag, radius). Catenary tubes, one object."""
    mb = MB()
    for p0, p1, sag, r in spans:
        n = max(12, int((Vector(p1) - Vector(p0)).length * 2.5))
        mb.tube(E.catenary(p0, p1, sag, n), r, 6, m=0)
    return E.obj(name, mb, ["cable"], c, shadow=True)


# ==================================================================== railing
def railing(c, x, y0, y1, H=1.0, name="railing"):
    """Pedestrian guard rail along +y at fixed x; top-rail tube top at z = H + 0.025."""
    mb = MB()
    mb.rod((x, y0 - 0.03, H), (x, y1 + 0.03, H), 0.025, 12, m=0)
    for zz in (0.55, 0.13):
        mb.rod((x, y0, zz), (x, y1, zz), 0.016, 8, m=0)
    n = int(round(y1 - y0))
    for i in range(n + 1):
        y = y0 + (y1 - y0) * i / n
        mb.rod((x, y, 0.0), (x, y, H), 0.024, 10, m=0)
        mb.box((x, y, 0.006), (0.12, 0.12, 0.012), m=1)
        if i < n:
            for j in range(1, 8):
                yy = y + (y1 - y0) / n * j / 8
                mb.rod((x, yy, 0.13), (x, yy, H - 0.02), 0.008, 6, m=0)
    for y in (y0 - 0.03, y1 + 0.03):
        mb.ellipsoid((x, y, H), (0.026, 0.026, 0.026), 10, 6, m=0)
    return E.obj(name, mb, ["rail", "metal_dark"], c)


# ==================================================================== AC units
def ac_wall(c, center, yaw, name="ac"):
    """Wall-mounted outdoor AC unit 0.55 (out) x 0.8 x 0.55 m centred at `center`, front toward local +X; top z = cz + 0.275."""
    mb = MB()
    mb.box((0, 0, 0), (0.55, 0.8, 0.55), m=0)
    mb.cyl((0.274, 0.1, 0), 0.2, 0.004, axis="x", seg=24, m=1)
    for r in (0.205, 0.15, 0.1, 0.05):
        mb.ring((0.279, 0.1, 0), r, r - 0.008, 0.008, axis="x", seg=24, m=1)
    for a in (0, PI / 2):
        mb.box((0.28, 0.1, 0), (0.006, 0.41 * abs(math.cos(a)) + 0.008, 0.41 * abs(math.sin(a)) + 0.008), m=1)
    mb.cyl((0.28, 0.1, 0), 0.03, 0.012, axis="x", seg=12, m=1)
    for k in range(7):
        mb.box((0.278, -0.27, -0.2 + 0.065 * k), (0.008, 0.18, 0.02), m=1)
    mb.box((0.276, -0.27, -0.24), (0.003, 0.1, 0.035), m=4)
    for s in (-0.3, 0.3):
        mb.box((0.005, s, -0.3), (0.56, 0.04, 0.04), m=2)
        mb.seg((-0.27, s, -0.62), (0.2, s, -0.31), 0.03, 0.03, m=2)
    mb.rod((-0.1, -0.4, -0.15), (-0.27, -0.46, -0.15), 0.022, 8, m=3)
    mb.rod((-0.1, -0.4, -0.08), (-0.27, -0.46, -0.08), 0.012, 8, m=2)
    mb.rod((0.0, 0.32, -0.275), (0.0, 0.32, -0.95), 0.009, 6, m=3)
    o = E.obj(name, mb, ["ac_white", "grille", "metal_dark", "plastic_dark", "plastic_white"], c, loc=center, rot=(0, 0, yaw))
    return {"obj": o, "top": W(center, yaw, (0, 0, 0.275))}


def ac_roof(mb, p, rz=0.0, mi=None):
    """Rooftop condenser into an existing MB (local frame of the caller). Top at p.z + 0.86."""
    m = mi
    with mb.at(p, rz):
        for s in (-0.4, 0.4):
            mb.box((0, s, 0.03), (1.05, 0.08, 0.06), m["metal_dark"])
        mb.box((0, 0, 0.46), (1.0, 0.85, 0.8), m["ac_white"])
        for s in (-1, 1):
            mb.box((s * 0.502, 0, 0.42), (0.004, 0.75, 0.62), m["grille"])
        mb.box((0, 0.427, 0.42), (0.9, 0.004, 0.62), m["grille"])
        mb.cyl((0, 0, 0.862), 0.33, 0.004, seg=24, m=m["grille"])
        for r in (0.34, 0.24, 0.14):
            mb.ring((0, 0, 0.868), r, r - 0.012, 0.012, seg=24, m=m["grille"])
        mb.box((0, 0, 0.87), (0.68, 0.02, 0.012), m["grille"])
        mb.box((0, 0, 0.87), (0.02, 0.68, 0.012), m["grille"])


def antenna(mb, p, H, m, kind="yagi", rz=0.0):
    with mb.at(p, rz):
        mb.box((0, 0, 0.01), (0.3, 0.3, 0.02), m["metal_dark"])
        mb.rod((0, 0, 0), (0, 0, H), 0.022, 8, m=m["galv"])
        if kind == "yagi":
            mb.rod((-0.6, 0, H - 0.15), (0.6, 0, H - 0.15), 0.01, 6, m=m["galv"])
            for k in range(7):
                L = 0.5 - 0.04 * k
                xx = -0.55 + 0.18 * k
                mb.rod((xx, -L / 2, H - 0.15), (xx, L / 2, H - 0.15), 0.005, 4, m=m["galv"])
        elif kind == "whip":
            mb.rod((0, 0, H), (0, 0, H + 1.4), 0.006, 4, m=m["galv"])
            mb.ellipsoid((0, 0, H + 1.42), (0.03, 0.03, 0.03), 8, 4, m=m["led_red"])
        for a in (0.3, 2.4, 4.5):
            mb.rod((0, 0, H * 0.7), (math.cos(a) * H * 0.55, math.sin(a) * H * 0.55, 0.0), 0.003, 4, m=m["cable"])


def dish_small(mb, p, m, az=0.0, el=0.6, r=0.38):
    d = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    with mb.at(p):
        mb.rod((0, 0, 0), (0, 0, 0.6), 0.03, 8, m=m["galv"])
        c = Vector((0, 0, 0.65))
        mb.dish(tuple(c), r, r * 0.3, tuple(d), 20, 4, m=m["dish"])
        mb.rod(tuple(c + d * 0.02), tuple(c + d * 0.42), 0.008, 4, m=m["galv"])
        mb.cyl(tuple(c + d * 0.44), 0.03, 0.06, seg=8, m=m["metal_dark"])


# ==================================================================== bin, hydrant, cone, vending machine
def bin_prop(c, loc, name="bin"):
    """Slatted street litter bin; lid top z = 0.955 (crude perch height)."""
    mb = MB()
    mb.cyl((0, 0, 0.03), 0.3, 0.06, seg=24, m=1)
    for i in range(18):
        a = 2 * PI * i / 18
        with mb.at((0, 0, 0), a):
            mb.box((0.275, 0, 0.47), (0.02, 0.07, 0.8), m=0)
    mb.ring((0, 0, 0.86), 0.3, 0.265, 0.045, seg=24, m=1)
    mb.ring((0, 0, 0.1), 0.3, 0.265, 0.04, seg=24, m=1)
    mb.cyl((0, 0, 0.47), 0.25, 0.8, seg=20, m=2)
    mb.cyl((0, 0, 0.92), 0.3, 0.07, seg=24, m=0)
    mb.box((0.27, 0, 0.915), (0.07, 0.24, 0.045), m=2)
    o = E.obj(name, mb, ["bin_green", "metal_dark", "plastic_dark"], c, loc=loc)
    return {"obj": o, "top": Vector(loc) + Vector((0, 0, 0.955))}


def hydrant(c, loc, name="hydrant"):
    mb = MB()
    mb.cyl((0, 0, 0.04), 0.14, 0.08, seg=16, m=0)
    mb.cyl((0, 0, 0.36), 0.1, 0.56, seg=16, m=0)
    mb.cyl((0, 0, 0.62), 0.12, 0.04, seg=16, m=0)
    mb.ellipsoid((0, 0, 0.66), (0.1, 0.1, 0.08), 16, 8, m=0)
    mb.cyl((0, 0, 0.75), 0.025, 0.04, seg=6, m=1)
    for a in (0, PI):
        with mb.at((0, 0, 0.45), a):
            mb.cyl((0.13, 0, 0), 0.045, 0.08, axis="x", seg=12, m=0)
            mb.cyl((0.18, 0, 0), 0.05, 0.03, axis="x", seg=6, m=1)
    o = E.obj(name, mb, ["paint_yellow", "metal_dark"], c, loc=loc)
    return {"obj": o, "top": Vector(loc) + Vector((0, 0, 0.77))}


def cone_prop(mb, p, m):
    with mb.at(p):
        mb.box((0, 0, 0.015), (0.36, 0.36, 0.03), m["rubber"])
        mb.cyl((0, 0, 0.36), 0.14, 0.66, seg=16, m=m["cone"], r2=0.025)
        mb.cyl((0, 0, 0.42), 0.1, 0.1, seg=16, m=m["plastic_white"], r2=0.085)


def vending_machine(c, loc, yaw, name="vending", label="DRINKS"):
    """0.9 x 0.75 x 1.85 m machine against a wall, front toward local -Y (yaw rotates it), lit display."""
    root = _root(name, c, loc, yaw)
    mb = MB()
    mb.box((0, 0.375, 0.925), (0.9, 0.75, 1.85), m=0)
    mb.quad((-0.4, -0.004, 0.75), (0.15, -0.004, 0.75), (0.15, -0.004, 1.7), (-0.4, -0.004, 1.7), m=1)
    for r in range(4):
        for k in range(5):
            mb.box((-0.36 + 0.1 * k, -0.02, 0.86 + 0.22 * r), (0.055, 0.03, 0.12), m=2 + (r + k) % 3)
    mb.box((0.28, -0.01, 1.2), (0.16, 0.02, 0.5), m=5)
    mb.box((0.28, -0.022, 1.32), (0.05, 0.004, 0.08), m=6)
    mb.box((-0.12, -0.02, 0.35), (0.6, 0.04, 0.18), m=5)
    o = E.obj(name + "_body", mb, [E.principled("ENV_vend_body", (0.6, 0.62, 0.66), 0.2, 0.35), "shopfront",
                                     E.principled("ENV_can_a", (0.6, 0.05, 0.05), 0.6, 0.3),
                                     E.principled("ENV_can_b", (0.05, 0.25, 0.6), 0.6, 0.3),
                                     E.principled("ENV_can_c", (0.9, 0.7, 0.1), 0.6, 0.3), "plastic_dark", "led_green"], c)
    _child(o, root)
    t = E.text(name + "_label", label, 0.12, c, (-0.12, -0.01, 1.78), (PI / 2, 0, 0), mat="neon_cyan", tube=True, bevel=0.005)
    _child(t, root)
    return {"root": root, "top": W(loc, yaw, (0, 0.375, 1.85)), "front": W(loc, yaw, (0, -0.5, 1.2))}


# ==================================================================== scooter
def scooter(c, loc, yaw, name="scooter", plate="OSFP 800"):
    """Parked step-through scooter, front toward local -Y; seat top z = 0.71 at local y = 0.15."""
    root = _root(name, c, loc, yaw)
    mb = MB()
    for y in (-0.45, 0.45):
        mb.ring((0, y, 0.2), 0.2, 0.125, 0.095, axis="x", seg=24, m=1)
        mb.cyl((0, y, 0.2), 0.126, 0.07, axis="x", seg=20, m=2)
        mb.cyl((0, y, 0.2), 0.04, 0.12, axis="x", seg=10, m=4)
    mb.box((0, -0.08, 0.3), (0.32, 0.52, 0.05), m=0)
    mb.box((0, -0.08, 0.27), (0.26, 0.5, 0.03), m=4)
    mb.ellipsoid((0, 0.3, 0.45), (0.2, 0.38, 0.2), 20, 10, m=0)
    mb.ellipsoid((0, 0.15, 0.66), (0.15, 0.33, 0.05), 18, 8, m=3)
    with mb.at((0, -0.38, 0.62), rx=-0.25):
        mb.box((0, 0, 0), (0.36, 0.05, 0.62), m=0)
    mb.ellipsoid((0, -0.47, 0.34), (0.09, 0.21, 0.08), 14, 8, m=0)
    mb.rod((0, -0.42, 0.42), (0, -0.4, 1.0), 0.03, 10, m=4)
    for s in (-0.06, 0.06):
        mb.rod((s, -0.45, 0.2), (s, -0.43, 0.48), 0.018, 8, m=4)
    mb.rod((-0.32, -0.4, 1.05), (0.32, -0.4, 1.05), 0.016, 8, m=4)
    for s in (-1, 1):
        mb.rod((s * 0.24, -0.4, 1.05), (s * 0.34, -0.4, 1.05), 0.02, 8, m=1)
        mb.rod((s * 0.22, -0.4, 1.06), (s * 0.27, -0.42, 1.24), 0.006, 6, m=5)
        mb.ellipsoid((s * 0.28, -0.42, 1.26), (0.05, 0.012, 0.035), 10, 6, m=5)
    mb.ellipsoid((0, -0.42, 1.02), (0.12, 0.09, 0.07), 14, 8, m=0)
    mb.cyl((0, -0.5, 1.0), 0.06, 0.04, axis="y", seg=16, m=5)
    mb.cyl((0, -0.522, 1.0), 0.05, 0.004, axis="y", seg=16, m=5)
    mb.box((0, 0.68, 0.52), (0.12, 0.02, 0.04), m=6)
    mb.box((0, 0.7, 0.4), (0.17, 0.006, 0.1), m=8)
    mb.rod((0.1, 0.0, 0.26), (0.19, 0.06, 0.0), 0.012, 6, m=4)
    o = E.obj(name + "_body", mb, ["red_body", "rubber", "galv", "seat", "metal_dark", "chrome",
                                     E.principled("ENV_tail_red", (0.5, 0.02, 0.02), 0, 0.3, (1.0, 0.05, 0.03), 1.5),
                                     "lamp_lens_cool", "plastic_white"], c, smooth=False)
    _child(o, root)
    t = E.text(name + "_plate", plate, 0.035, c, (0, 0.704, 0.4), (PI / 2, 0, PI),
               mat=E.principled("ENV_plate_ink", (0.01, 0.01, 0.01), 0, 0.5), fnt="Arial Black")
    _child(t, root)
    return {"root": root, "seat": W(loc, yaw, (0, 0.15, 0.71)), "bar": W(loc, yaw, (0.3, -0.4, 1.066))}


# ==================================================================== hanging sign
def hanging_sign(c, center, yaw, lines, neon="neon_orange", name="hanging_sign", wall_off=0.6):
    """Double-sided lightbox sign 0.9 x 0.08 x 0.6 m hung from a wall bracket; local +X away from the wall.
    Sign top z = center.z + 0.3 (crude perch height); wall at local x = -wall_off."""
    root = _root(name, c, center, yaw)
    mb = MB()
    wx = -wall_off
    mb.box((wx + 0.01, 0, 0.42), (0.02, 0.16, 0.22), m=1)
    mb.box(((wx + 0.5) / 2, 0, 0.42), (0.5 - wx, 0.04, 0.04), m=1)
    mb.seg((wx + 0.02, 0, 0.12), (-0.05, 0, 0.41), 0.03, 0.02, m=1)
    for s in (-0.35, 0.35):
        mb.rod((s, 0, 0.3), (s, 0, 0.405), 0.006, 6, m=1)
        mb.ring((s, 0, 0.405), 0.018, 0.01, 0.006, axis="y", seg=10, m=1)
    mb.box((0, 0, 0), (0.9, 0.08, 0.6), m=0)
    pts = [(-0.42, 0, -0.27), (0.42, 0, -0.27), (0.42, 0, 0.27), (-0.42, 0, 0.27)]
    for s in (-1, 1):
        mb.tube([(p[0], s * 0.048, p[2]) for p in pts], 0.007, 6, m=2, closed=True)
    o = E.obj(name + "_box", mb, ["sign_back", "metal_dark", neon], c)
    _child(o, root)
    for s in (-1, 1):
        for i, ln in enumerate(lines):
            zz = 0.1 - 0.22 * i if len(lines) > 1 else 0.0
            t = E.text("%s_txt" % name, ln, 0.19, c, (0, s * 0.047, zz), (PI / 2, 0, 0 if s < 0 else PI), mat=neon,
                       tube=True, bevel=0.007)
            _child(t, root)
    return {"root": root, "top": W(center, yaw, (0, 0, 0.3))}


# ==================================================================== CCTV
def cctv(c, center, yaw, mount="pole", name="cctv", tilt=0.0):
    """Bullet camera housing 0.4 x 0.2 x 0.18 m centred at `center`, lens toward local +X; sunshield top z = cz + 0.1.
    mount 'pole': pole 0.35 m behind the camera down to z = 0; 'wall': wall plate 0.38 m behind the camera."""
    mb = MB()
    with mb.at((0, 0, 0), ry=tilt):
        mb.box((0, 0, 0), (0.4, 0.2, 0.18), m=0)
        mb.box((0.03, 0, 0.095), (0.48, 0.25, 0.01), m=0)
        mb.cyl((0.21, 0, 0), 0.07, 0.03, axis="x", seg=20, m=0)
        mb.cyl((0.226, 0, 0), 0.058, 0.004, axis="x", seg=20, m=1)
        mb.ring((0.229, 0, 0), 0.066, 0.06, 0.004, axis="x", seg=20, m=4)
        mb.cyl((-0.21, 0, -0.03), 0.025, 0.03, axis="x", seg=8, m=2)
        mb.ellipsoid((-0.15, 0.07, 0.1), (0.012, 0.012, 0.008), 8, 4, m=3)
    mb.cyl((-0.1, 0, -0.11), 0.03, 0.04, seg=10, m=2)
    mb.rod((-0.1, 0, -0.13), (-0.1, 0, -0.22), 0.018, 8, m=2)
    if mount == "pole":
        mb.rod((-0.1, 0, -0.22), (-0.35, 0, -0.22), 0.018, 8, m=2)
        mb.rod((-0.35, 0, -0.1), (-0.35, 0, -center[2]), 0.04, 12, m=2)
        mb.box((-0.35, 0, -center[2] + 0.006), (0.16, 0.16, 0.012), m=2)
        mb.box((-0.35, 0.045, -center[2] + 1.0), (0.1, 0.06, 0.25), m=2)
    else:
        mb.rod((-0.1, 0, -0.22), (-0.37, 0, -0.22), 0.018, 8, m=2)
        mb.box((-0.375, 0, -0.22), (0.02, 0.12, 0.18), m=2)
    ir = E.principled("ENV_cctv_ir", (0.2, 0.02, 0.02), 0, 0.3, (0.6, 0.02, 0.02), 0.6)
    o = E.obj(name, mb, ["camera_white", "lens_black", "metal_dark", "led_red", ir], c, loc=center, rot=(0, 0, yaw))
    lens_dir = Rz(yaw) @ Vector((math.cos(tilt), 0, -math.sin(tilt)))
    return {"obj": o, "lens": W(center, yaw, (0.232 * math.cos(tilt), 0, -0.232 * math.sin(tilt))), "lens_dir": lens_dir,
            "top": W(center, yaw, (0, 0, 0.1))}


# ==================================================================== satellite
def satellite(c, loc, name="satellite", scale=1.0, yaw=0.0):
    """Stylised comms satellite (bus 0.9 m, wingspan 6.3 m at scale 1, nadir dish facing -Z); root empty animatable.
    scale enlarges the whole prop (and its key light)."""
    root = _root(name, c, loc, yaw)
    root.scale = (scale,) * 3
    mb = MB()
    mb.box((0, 0, 0), (0.9, 0.9, 1.0), m=0)
    mb.box((0, 0, 0.52), (0.95, 0.95, 0.04), m=3)
    mb.box((0, 0, -0.52), (0.95, 0.95, 0.04), m=3)
    for s in (-1, 1):
        mb.rod((s * 0.45, 0, 0), (s * 0.85, 0, 0), 0.03, 8, m=3)
        for j in range(3):
            x0 = s * (0.85 + 0.78 * j)
            x1 = s * (0.85 + 0.78 * j + 0.72)
            a, b = min(x0, x1), max(x0, x1)
            mb.box(((a + b) / 2, 0, 0), (b - a + 0.02, 0.98, 0.03), m=3)
            mb.quad((a, -0.46, -0.017), (b, -0.46, -0.017), (b, 0.46, -0.017), (a, 0.46, -0.017), m=1)
            mb.quad((a, 0.46, 0.017), (b, 0.46, 0.017), (b, -0.46, 0.017), (a, -0.46, 0.017), m=1)
    mb.dish((0, 0, -0.56), 0.5, 0.14, (0, 0, -1), 24, 5, m=2)
    for a in (0, 2.1, 4.2):
        mb.rod((0.42 * math.cos(a), 0.42 * math.sin(a), -0.68), (0, 0, -0.95), 0.008, 4, m=3)
    mb.cyl((0, 0, -0.97), 0.04, 0.08, seg=10, m=3)
    mb.rod((0.3, 0.3, 0.54), (0.3, 0.3, 1.1), 0.008, 4, m=3)
    mb.box((-0.3, 0.3, 0.6), (0.12, 0.12, 0.12), m=3)
    mb.ellipsoid((0, -0.46, 0.35), (0.07, 0.04, 0.07), 10, 6, m=4)
    o = E.obj(name + "_body", mb, ["foil", "solar", "dish", "galv", "led_red"], c)
    _child(o, root)
    k = E.light(name + "_key", "POINT", (2.5, -3.0, -2.0), 900 * scale * scale, (0.75, 0.82, 1.0), c, radius=1.0, shadow=False)
    _child(k, root)
    return {"root": root, "dish": W(loc, yaw, (0, 0, -0.97 * scale))}


# ==================================================================== hologram projector (S1 tree base)
def holo_projector(c, loc, name="holo_projector"):
    mb = MB()
    mb.cyl((0, 0, 0.03), 0.45, 0.06, seg=40, m=0)
    mb.cyl((0, 0, 0.065), 0.3, 0.01, seg=40, m=2)
    mb.ring((0, 0, 0.066), 0.38, 0.34, 0.012, seg=40, m=1)
    for i in range(8):
        a = 2 * PI * i / 8
        mb.ellipsoid((0.4 * math.cos(a), 0.4 * math.sin(a), 0.062), (0.025, 0.025, 0.012), 8, 4, m=2)
    mb.box((1.6, 0, 0.012), (2.4, 0.08, 0.024), m=0)
    o = E.obj(name, mb, ["metal_dark", "led_cyan", "lens_black"], c, loc=loc)
    cone = vfx.light_cone(name + "_cone", tuple(Vector(loc) + Vector((0, 0, 0.07))), (0, 0, 1), 1.2, 50, vfx.HOLO_CYAN, 0.12, c)
    return {"obj": o, "cone": cone, "top": Vector(loc) + Vector((0, 0, 0.07))}


# ==================================================================== neon signs
def neon_sign(c, loc, yaw, text, neon="neon_mag", size=0.42, name="neon", backer=True, border=True, light=None, fnt="Arial Black"):
    """Flat wall sign: neon tube text on a dark backer. Text faces local -Y; yaw rotates the facing
    (facing +x: yaw = pi/2, -x: -pi/2, +y: pi). Returns dict with root and size."""
    root = _root(name, c, loc, yaw)
    t = E.text(name + "_txt", text, size, c, (0, -0.0, 0), (PI / 2, 0, 0), mat=neon, tube=True, bevel=size * 0.03, fnt=fnt)
    d = E.text_dims(t)
    _child(t, root)
    w, h = d.x + 0.3, size + 0.25
    if backer:
        mb = MB()
        mb.box((0, 0.04, 0), (w, 0.05, h), m=0)
        for s in (-1, 1):
            mb.box((s * w * 0.3, 0.075, h / 2 + 0.05), (0.03, 0.03, 0.12), m=1)
        if border:
            q = [(-w / 2 + 0.05, 0.005, -h / 2 + 0.05), (w / 2 - 0.05, 0.005, -h / 2 + 0.05), (w / 2 - 0.05, 0.005, h / 2 - 0.05),
                 (-w / 2 + 0.05, 0.005, h / 2 - 0.05)]
            mb.tube(q, size * 0.022, 6, m=2, closed=True)
        o = E.obj(name + "_backer", mb, ["sign_back", "metal_dark", neon], c)
        _child(o, root)
    out = {"root": root, "w": w, "h": h, "center": Vector(loc)}
    if light is not None:
        col, energy, lc = light
        face = Rz(yaw) @ Vector((0, -1, 0))
        p = Vector(loc) + face * 0.7
        E.light(name + "_spill", "AREA", tuple(p), energy, col, lc, size=max(w, 0.5), target=tuple(p + face))
    return out


def blade_sign(c, loc, yaw, text, neon="neon_cyan", size=0.32, name="blade"):
    """Vertical projecting sign (letters stacked), perpendicular to the wall; local +X = away from the wall; readable
    from both directions along the wall (local +-Y)."""
    root = _root(name, c, loc, yaw)
    n = len(text)
    hgt = n * size * 1.05 + 0.3
    mb = MB()
    mb.box((0.45, 0, 0), (0.62, 0.1, hgt), m=0)
    for zz in (hgt / 2 - 0.15, -hgt / 2 + 0.15):
        mb.box((0.07, 0, zz), (0.14, 0.05, 0.05), m=1)
    for s in (-1, 1):
        q = [(0.18, s * 0.056, -hgt / 2 + 0.06), (0.72, s * 0.056, -hgt / 2 + 0.06), (0.72, s * 0.056, hgt / 2 - 0.06),
             (0.18, s * 0.056, hgt / 2 - 0.06)]
        mb.tube(q, 0.008, 6, m=2, closed=True)
    o = E.obj(name + "_box", mb, ["sign_back", "metal_dark", neon], c)
    _child(o, root)
    for s in (-1, 1):
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            zz = hgt / 2 - 0.15 - size * 0.55 - i * size * 1.05
            t = E.text("%s_ch" % name, ch, size, c, (0.45, s * 0.056, zz), (PI / 2, 0, PI if s > 0 else 0), mat=neon,
                       tube=True, bevel=size * 0.03)
            _child(t, root)
    return {"root": root, "top": W(loc, yaw, (0.45, 0, hgt / 2))}


# ==================================================================== puddles, manholes, drains
def puddles(c, specs, name="puddles", seed=3):
    """specs: list of (x, y, r, z). Irregular blobs (single n-gon) 4 mm above the given ground z."""
    rng = random.Random(seed)
    mb = MB()
    for x, y, r, z in specs:
        n = 28
        ph = [rng.uniform(0, 2 * PI) for _ in range(3)]
        sx, sy = rng.uniform(0.7, 1.3), rng.uniform(0.6, 1.1)
        v = []
        for i in range(n):
            a = 2 * PI * i / n
            rr = r * (1 + 0.22 * math.sin(2 * a + ph[0]) + 0.12 * math.sin(3 * a + ph[1]) + 0.07 * math.sin(5 * a + ph[2]))
            v.append((x + rr * sx * math.cos(a), y + rr * sy * math.sin(a), z + 0.004))
        mb._add(v, [tuple(range(n))], 0)
    return E.obj(name, mb, ["puddle"], c, shadow=False)


# ==================================================================== buildings
WIN_KINDS = [("glass_dark", 0.42), ("win_warm", 0.25), ("win_cool", 0.14), ("win_tv", 0.09), ("win_pink", 0.1)]
BMATS = ["wall", "concrete_light", "metal_dark", "glass_dark", "win_warm", "win_cool", "win_tv", "win_pink", "shopfront",
         "shutter", "sign_back", "rust", "ac_white", "grille", "galv", "water_tank", "dish", "fabric_awning", "led_red",
         "cable", "rubber", "plastic_white", "cone"]


def _bm(wall):
    mats = []
    for nm in BMATS:
        if nm == "wall":
            mats.append(wall)
        elif nm == "cone":
            mats.append(E.principled("ENV_cone_orange", (0.9, 0.25, 0.02), 0, 0.5))
        else:
            mats.append(nm)
    return mats, {nm: i for i, nm in enumerate(BMATS)}


def _window(mb, m, x, z0, w=1.0, h=1.5, kind="glass_dark", ac=False):
    """Window on a facade plane y = 0 (outward +y): frame, mullion, sill, glass quad with 0..1 UV."""
    x0, x1, z1 = x - w / 2, x + w / 2, z0 + h
    mb.box((x0 - 0.03, 0.03, (z0 + z1) / 2), (0.06, 0.06, h + 0.06), m["metal_dark"])
    mb.box((x1 + 0.03, 0.03, (z0 + z1) / 2), (0.06, 0.06, h + 0.06), m["metal_dark"])
    mb.box((x, 0.03, z1 + 0.03), (w + 0.12, 0.06, 0.06), m["metal_dark"])
    mb.box((x, 0.05, z0 - 0.03), (w + 0.2, 0.12, 0.05), m["concrete_light"])
    mb.box((x, 0.02, (z0 + z1) / 2), (0.035, 0.04, h), m["metal_dark"])
    mb.box((x, 0.02, z0 + h * 0.62), (w, 0.03, 0.03), m["metal_dark"])
    mb.quad((x0, 0.004, z0), (x1, 0.004, z0), (x1, 0.004, z1), (x0, 0.004, z1), m[kind])
    mb.box((x, -0.02, (z0 + z1) / 2), (w + 0.12, 0.04, h + 0.12), m["metal_dark"])
    if ac:
        mb.box((x + 0.15, 0.24, z0 - 0.32), (0.62, 0.45, 0.4), m["ac_white"])
        mb.box((x + 0.15, 0.467, z0 - 0.32), (0.5, 0.004, 0.3), m["grille"])
        mb.box((x + 0.15, 0.24, z0 - 0.54), (0.5, 0.42, 0.03), m["metal_dark"])
        return Vector((x + 0.15, 0.24, z0 - 0.12))
    return None


def _fire_escape(mb, m, xa, xb, floors, depth=0.85):
    """Fire escape on facade y = 0 between local x = xa..xb: a platform per upper floor (z = 3n), railings, stairs, ladder.
    Returns local perch points on the top rails."""
    perches = []
    for n in range(1, floors):
        z = 3.0 * n - 0.05
        mb.box(((xa + xb) / 2, depth / 2, z), (xb - xa, depth, 0.05), m["rust"])
        for k in range(10):
            yy = 0.06 + (depth - 0.12) * k / 9
            mb.box(((xa + xb) / 2, yy, z + 0.03), (xb - xa - 0.06, 0.03, 0.012), m["rust"])
        H = z + 1.0
        mb.box(((xa + xb) / 2, depth, H), (xb - xa, 0.035, 0.035), m["rust"])
        for xx in (xa, xb):
            mb.box((xx, depth / 2, H), (0.035, depth, 0.035), m["rust"])
            mb.box((xx, depth, (z + H) / 2), (0.035, 0.035, H - z), m["rust"])
        nb = int((xb - xa) / 0.16)
        for k in range(1, nb):
            xx = xa + (xb - xa) * k / nb
            mb.box((xx, depth, (z + H) / 2), (0.014, 0.014, H - z), m["rust"])
        for k in range(1, 5):
            yy = depth * k / 5
            for xx in (xa, xb):
                mb.box((xx, yy, (z + H) / 2), (0.014, 0.014, H - z), m["rust"])
        for s in (0.3, 0.7):
            mb.seg((xa, 0.0, z - 0.02), (xa + 0.02, 0.0, z - 0.9), 0.04, 0.04, m["rust"])
            mb.seg((xa + (xb - xa) * s, depth - 0.05, z - 0.03), (xa + (xb - xa) * s, 0.02, z - 0.6), 0.03, 0.03, m["rust"])
        perches.append(Vector(((xa + xb) / 2 - 0.3, depth, H + 0.018)))
        perches.append(Vector((xb, depth * 0.5, H + 0.018)))
        if n >= 2:      # stair from this platform down to the one below
            zt, zb = z, z - 3.0
            x_top, x_bot = xa + 0.25, xb - 0.35
            for yy in (0.18, depth - 0.12):
                mb.seg((x_top, yy, zt), (x_bot, yy, zb + 0.03), 0.03, 0.12, m["rust"])
            for k in range(1, 12):
                u = k / 12
                mb.box((x_top + (x_bot - x_top) * u, (0.18 + depth - 0.12) / 2, zt + (zb - zt) * u), (0.2, depth - 0.3, 0.02),
                       m["rust"])
            mb.seg((x_top, depth - 0.12, zt + 0.9), (x_bot, depth - 0.12, zb + 0.93), 0.03, 0.03, m["rust"])
    zl = 3.0 - 0.05
    for xx in (xb - 0.5, xb - 0.15):
        mb.box((xx, depth - 0.1, zl - 0.9), (0.03, 0.03, 1.8), m["rust"])
    for k in range(6):
        mb.box((xb - 0.325, depth - 0.1, zl - 1.7 + 0.3 * k), (0.35, 0.02, 0.02), m["rust"])
    return perches


def building(c, base, yaw, width, depth, h, wall, rng, name, shop="lit", fire_escape=None, roof=(), awning=False,
             skip_windows=(), win_ac=0.25, sidewalk_z=0.12):
    """Facade building. Local frame: origin at the facade base centre, facade plane y = 0 facing +y (outward),
    width along x, depth into -y. Parapet coping top at z = h + 0.1 on the facade line (crude roof-edge perch height).
    Returns {'obj', 'perches': [(name, world point, world yaw)], 'roof': world z}."""
    mats, m = _bm(wall)
    mb = MB()
    hw = width / 2
    top = h - 0.45
    mb.box((0, -depth / 2, top / 2), (width, depth, top), m["wall"])
    mb.box((0, -0.1, (top + h + 0.04) / 2), (width, 0.2, h + 0.04 - top), m["wall"])
    for s in (-1, 1):
        mb.box((s * (hw - 0.1), -depth / 2, (top + h + 0.04) / 2), (0.2, depth, h + 0.04 - top), m["wall"])
    mb.box((0, -depth + 0.1, (top + h + 0.04) / 2), (width, 0.2, h + 0.04 - top), m["wall"])
    mb.box((0, -0.075, h + 0.07), (width + 0.04, 0.29, 0.06), m["concrete_light"])
    floors = int(round(h / 3.0))
    for n in range(1, floors):
        mb.box((0, 0.04, 3.0 * n), (width, 0.08, 0.12), m["concrete_light"])
    perches = []
    # ground floor
    z0 = sidewalk_z
    if shop == "shutter":
        mb.quad((-hw + 0.15, 0.006, z0), (hw - 0.15, 0.006, z0), (hw - 0.15, 0.006, 2.65), (-hw + 0.15, 0.006, 2.65), m["shutter"])
        mb.box((0, 0.11, 2.78), (width - 0.2, 0.22, 0.26), m["metal_dark"])
        for s in (-1, 1):
            mb.box((s * (hw - 0.12), 0.03, 1.35), (0.06, 0.06, 2.7), m["metal_dark"])
    else:
        x_door = hw - 0.55
        mb.box((-0.35, 0.02, z0 + 0.17), (width - 1.5, 0.04, 0.34), m["metal_dark"])
        mb.quad((-hw + 0.1, 0.004, z0 + 0.34), (hw - 1.0, 0.004, z0 + 0.34), (hw - 1.0, 0.004, 2.6), (-hw + 0.1, 0.004, 2.6),
                m["shopfront" if shop == "lit" else "glass_dark"])
        for xx in (-hw + 0.1, (-hw + 0.1 + hw - 1.0) / 2, hw - 1.0):
            mb.box((xx, 0.03, (z0 + 2.6) / 2 + 0.05), (0.06, 0.06, 2.6 - z0), m["metal_dark"])
        mb.box((x_door, 0.004, (z0 + 2.3) / 2), (0.8, 0.008, 2.3 - z0), m["glass_dark"])
        for xx in (x_door - 0.42, x_door + 0.42):
            mb.box((xx, 0.03, (z0 + 2.35) / 2), (0.05, 0.06, 2.35 - z0), m["metal_dark"])
        mb.box((x_door, 0.03, 2.36), (0.9, 0.06, 0.05), m["metal_dark"])
        mb.box((0, 0.04, 2.75), (width, 0.08, 0.25), m["sign_back"])
    if awning:
        with mb.at((-0.3, 0, 2.62), rx=-0.35):
            mb.box((0, 0.45, 0), (width - 1.2, 0.95, 0.03), m["fabric_awning"])
        mb.box((-0.3, 0.92, 2.42), (width - 1.2, 0.02, 0.18), m["fabric_awning"])
    # upper floors
    for n in range(1, floors):
        for xi, x in enumerate((-0.85, 0.85)):
            if (n, xi) in skip_windows:
                continue
            r = rng.random()
            acc, kind = 0.0, "glass_dark"
            for kk, p in WIN_KINDS:
                acc += p
                if r < acc:
                    kind = kk
                    break
            ac = rng.random() < win_ac and (fire_escape is None or not (fire_escape[0] <= x <= fire_escape[1]))
            p = _window(mb, m, x, 3.0 * n + 0.9, 1.0, 1.5, kind, ac)
            if p is not None:
                perches.append(("winac_%d_%d" % (n, xi), p + Vector((0, 0, 0.0)), PI / 2))
    if fire_escape:
        for k, p in enumerate(_fire_escape(mb, m, fire_escape[0], fire_escape[1], floors)):
            perches.append(("fire_escape_%d" % k, p, PI / 2))
    # roof clutter (local points on the roof slab top = h - 0.45)
    for kind, x, y, extra in roof:
        p = (x, y, top)
        if kind == "ac":
            ac_roof(mb, p, extra or 0.0, m)
            perches.append(("roof_ac", Vector((x, y, top + 0.875)), PI / 2))
        elif kind == "yagi":
            antenna(mb, p, extra or 2.6, m, "yagi")
        elif kind == "whip":
            antenna(mb, p, extra or 2.0, m, "whip")
        elif kind == "dish":
            dish_small(mb, p, m, az=extra or 0.0)
        elif kind == "tank":
            with mb.at(p):
                for sx in (-0.6, 0.6):
                    for sy in (-0.6, 0.6):
                        mb.box((sx, sy, 0.6), (0.08, 0.08, 1.2), m["rust"])
                mb.cyl((0, 0, 2.1), 0.95, 1.8, seg=24, m=m["water_tank"])
                mb.cyl((0, 0, 3.15), 1.0, 0.3, seg=24, m=m["water_tank"], r2=0.1)
                for zz in (1.5, 2.1, 2.7):
                    mb.ring((0, 0, zz), 0.97, 0.94, 0.04, seg=24, m=m["rust"])
            perches.append(("tank_top", Vector((x, y, top + 3.31)), PI / 2))
        elif kind == "vent":
            with mb.at(p):
                mb.cyl((0, 0, 0.5), 0.12, 1.0, seg=12, m=m["galv"])
                mb.cyl((0, 0, 1.05), 0.2, 0.1, seg=12, m=m["galv"], r2=0.05)
    o = E.obj(name, mb, mats, c, loc=base, rot=(0, 0, yaw))
    perches.append(("parapet", Vector((rng.uniform(-0.8, 0.8), 0.04, h + 0.1)), PI / 2))
    out = []
    for nm, p, yl in perches:
        out.append((nm, W(base, yaw, p), yaw + yl))
    return {"obj": o, "perches": out, "roof": top}


def window_grid_material(name="ENV_window_grid", seed=0.0, cell=(1.5, 3.0), lit=0.2, strength=(0.3, 0.9),
                         base=(0.05, 0.052, 0.062), glass=(0.01, 0.012, 0.016), win=(0.2, 0.32, 0.15), base_emit=0.0):
    """Procedural window grid on box-UV metres (cell = (w, h) m), fraction `lit` of windows lit warm/cool with emission in
    `strength`. win = (side margin, bottom margin, top margin) as cell fractions. Use bigger cells for far towers
    (e.g. (4, 8)) so the grid does not shimmer below a pixel."""
    m, b, out = vfx.nodes.new_material(name)
    uv = b.sep(b.texcoord("UV"))
    u, v = uv[0], uv[1]
    cu, cv = b.math("DIVIDE", u, cell[0]), b.math("DIVIDE", v, cell[1])
    fu, fv = b.math("FRACT", cu), b.math("FRACT", cv)
    sm, bm_, tm = win
    w = b.mul(b.mul(b.smooth(fu, sm, sm + 0.03), b.smooth(b.one_minus(fu), sm, sm + 0.03)),
              b.mul(b.smooth(fv, bm_, bm_ + 0.03), b.smooth(b.one_minus(fv), tm, tm + 0.03)))
    c = b.comb(b.math("FLOOR", cu), b.math("FLOOR", cv), b.add(seed, b.mul(b.objinfo("Random"), 100.0)))
    r = b.white(c, "3D", "Value")
    r2 = b.white(b.vmath("ADD", c, (3.1, 7.7, 1.3)), "3D", "Value")
    on = b.math("LESS_THAN", r, lit)
    col = b.cmix((1.0, 0.62, 0.32, 1), (0.55, 0.75, 1.0, 1), b.math("GREATER_THAN", r2, 0.6))
    em = b.emission(col, b.mul(b.mul(w, on), b.lerp(strength[0], strength[1], r2)))
    if base_emit:                                   # floodlit facade (reads at night without scene lights)
        em = b.add_shader(em, b.emission((*base, 1), b.mul(b.one_minus(w), base_emit)))
    bc = b.cmix((*base, 1), (*glass, 1), w)
    p = b.principled(Base_Color=bc, Roughness=b.lerp(0.8, 0.1, w))
    b.link(b.add_shader(p.outputs[0], em), out.inputs["Surface"])
    m.diffuse_color = (*base, 1)
    return m
