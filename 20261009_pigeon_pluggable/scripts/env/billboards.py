"""Parody AI-era San Francisco billboards, a bus-shelter ad and a robotaxi for the S6 street (workstream E, DevLog-013
section 10; owner add-on W8, DevLog-020).

Flat billboard art only: emissive panel (lit at night), bold system type (Arial Black / Arial Bold), one accent colour,
at most one simple shape. Parody names and taglines riffing on the 2023-2026 SF 101 billboards (refs in DevLog-020 W8);
no real logos, fonts or trade dress. Text is converted to mesh (cheap to render).
"""
import math

import bpy
from mathutils import Matrix, Vector

import env_lib as E
import street_kit as SK
from env_lib import MB, PI

import vfx

INK = (0.015, 0.015, 0.018)
DESIGNS = {
    # bg colour, bg emission, lines [(text, colour, emissive)], brand (text, colour), accent (kind, colour)
    "artisand": dict(bg=(0.92, 0.9, 0.84), glow=1.4, lines=[("STOP HIRING HUMANS.", INK, False), ("HIRE BIRDS.", (0.85, 0.08, 0.06), False)],
                     brand=("ARTISAND", INK), accent=("bar", (0.85, 0.08, 0.06))),
    "sigmoid": dict(bg=(0.03, 0.2, 0.12), glow=1.0, lines=[("WELCOME TO", (0.95, 0.95, 0.9), True), ("AI COUNTRY", (0.95, 0.95, 0.9), True)],
                    brand=("SIGMOID", (1.0, 0.82, 0.25)), accent=("disc", (1.0, 0.82, 0.25))),
    "nonlinear": dict(bg=(0.03, 0.03, 0.07), glow=1.0, lines=[("AGENTS.", (0.95, 0.95, 1.0), True), ("AT YOUR COMMAND.", (0.95, 0.95, 1.0), True)],
                      brand=("NONLINEAR", (0.6, 0.45, 1.0)), accent=("bar", (0.6, 0.45, 1.0))),
    "snooks": dict(bg=(1.0, 0.82, 0.12), glow=1.3, lines=[("AI WON'T TAKE YOUR JOB.", INK, False), ("A BIRD USING SNOOKS WILL.", INK, False)],
                   brand=("SNOOKS", INK), accent=("triangle", INK)),
    "snooks_portrait": dict(bg=(1.0, 0.82, 0.12), glow=1.4, lines=[("AI WON'T", INK, False), ("TAKE YOUR JOB.", INK, False),
                                                                ("A BIRD USING", INK, False), ("SNOOKS WILL.", INK, False)],
                            brand=("SNOOKS", INK), accent=None),
    "agents": dict(bg=(0.02, 0.02, 0.02), glow=1.0, lines=[("AGENTS.", (0.97, 0.97, 0.97), True)], brand=None, accent=None),
    "vantanot": dict(bg=(0.94, 0.94, 0.96), glow=1.6, lines=[("COMPLIANCE", INK, False), ("THAT DOESN'T", INK, False), ("SOC 2", (0.45, 0.2, 0.85), False),
                                                            ("MUCH", INK, False)],
                     brand=("VANTANOT", (0.45, 0.2, 0.85)), accent=("bar", (0.45, 0.2, 0.85))),
}


def _yaw_for(facing):
    """Yaw so that local -Y (panel front, E.text default) points along `facing` (x, y)."""
    return math.atan2(facing[0], -facing[1])


def _fit_text(c, body, max_w, max_h, mat, loc, fnt="Arial Black", name="bb_txt"):
    t = E.text(name, body, 1.0, c, loc, (PI / 2, 0, 0), mat=mat, fnt=fnt)
    d = E.text_dims(t)
    s = min(max_w / max(d.x, 1e-6), max_h / max(d.y, 1e-6))      # text curve lies in its local XY plane
    t.scale = (s, s, s)
    return t, s


def _art(c, root, w, h, design, name, lc):
    """Panel art on the plane y = 0 of `root` (front toward local -Y). Returns created objects."""
    D = DESIGNS[design]
    bg = E.principled("BB_bg_%s" % design, D["bg"], 0.0, 0.5, D["bg"], D["glow"])
    objs = []
    mb = MB()
    mb.box((0, 0.03, 0), (w, 0.06, h), m=0)
    mb.box((0, 0.065, 0), (w + 0.2, 0.05, h + 0.2), m=1)
    o = E.obj(name + "_panel", mb, [bg, "metal_dark"], c)
    objs.append(o)
    margin = 0.08 * h
    lines = D["lines"]
    brand_h = 0.14 * h if D.get("brand") else 0.0
    acc = D.get("accent")
    x0 = -w / 2 + margin + (0.06 * w if acc and acc[0] == "bar" else 0.0)
    avail_w = w / 2 - margin - x0 - (0.3 * h if acc and acc[0] in ("disc", "triangle") else 0.0)
    avail_h = h - 2 * margin - brand_h - (0.04 * h if brand_h else 0)
    lh = avail_h / len(lines)
    ztop = h / 2 - margin
    for i, (txt, col, emi) in enumerate(lines):
        mat = E.principled("BB_ink_%s_%d" % (design, i), col, 0.0, 0.5, col if emi else None, 2.2 if emi else 0.0)
        t, s = _fit_text(c, txt, avail_w, lh * 0.82, mat, (0, 0, 0), name="%s_l%d" % (name, i))
        d = E.text_dims(t) * s
        t.location = (x0 + d.x / 2, -0.012, ztop - lh * (i + 0.5))
        objs.append(t)
    if D.get("brand"):
        bt, bcol = D["brand"]
        mat = E.principled("BB_brand_%s" % design, bcol, 0.0, 0.5, bcol if sum(D["bg"]) < 0.6 else None, 2.0 if sum(D["bg"]) < 0.6 else 0.0)
        t, s = _fit_text(c, bt, w * 0.35, brand_h, mat, (0, 0, 0), fnt="Arial Bold", name=name + "_brand")
        d = E.text_dims(t) * s
        t.location = (w / 2 - margin - d.x / 2, -0.012, -h / 2 + margin + brand_h / 2)
        objs.append(t)
    if acc:
        kind, col = acc
        am = E.principled("BB_acc_%s" % design, col, 0.0, 0.5, col if sum(D["bg"]) < 0.6 else None, 2.0 if sum(D["bg"]) < 0.6 else 0.0)
        mb = MB()
        if kind == "bar":
            mb.box((-w / 2 + margin + 0.02 * w, -0.012, 0), (0.035 * w, 0.01, h - 2 * margin), m=0)
        elif kind == "disc":
            mb.cyl((w / 2 - margin - 0.12 * h, -0.012, h / 2 - margin - 0.12 * h), 0.12 * h, 0.01, axis="y", seg=32, m=0)
        elif kind == "triangle":
            cx, cz, r = w / 2 - margin - 0.1 * h, h / 2 - margin - 0.1 * h, 0.1 * h
            mb._add([(cx - r, -0.012, cz - r * 0.8), (cx + r, -0.012, cz - r * 0.8), (cx, -0.012, cz + r)], [(0, 1, 2)], 0)
        objs.append(E.obj(name + "_accent", mb, [am], c))
    bpy.context.view_layer.update()            # matrices of the last-placed text are stale otherwise
    for o in objs:
        o.visible_shadow = False
        mw = o.matrix_world.copy()
        o.parent = root
        o.matrix_world = root.matrix_world @ mw
    return objs


def billboard(c, center, facing, w, h, design, lc, name, mount="roof", base_z=0.0, light=True):
    """Billboard panel w x h (m) centred at `center`, front facing the (x, y) direction `facing`.
    mount 'roof': two posts + catwalk down to base_z; 'pole': single freeway column down to base_z; 'wall': bracket only."""
    yaw = _yaw_for(facing)
    root = E.empty(name, c, center, (0, 0, yaw), 0.5)
    bpy.context.view_layer.update()
    objs = _art(c, root, w, h, design, name, lc)
    mb = MB()
    zb = base_z - center[2]
    if mount == "roof":
        for sx in (-0.35, 0.35):
            mb.box((sx * w, 0.35, (zb + (-h / 2)) / 2), (0.16, 0.16, -h / 2 - zb), m=0)
            mb.seg((sx * w, 0.35, zb + 0.3), (sx * w, 1.2, zb), 0.1, 0.1, m=0)
        mb.box((0, -0.35, -h / 2 - 0.12), (w, 0.6, 0.05), m=0)          # catwalk (no railing: it crossed the art)
    elif mount == "pole":
        mb.cyl((0, 0.6, (zb - h / 2) / 2), 0.45, -h / 2 - zb, seg=16, m=0)
        mb.box((0, 0.35, -h / 2 - 0.3), (w * 0.8, 0.6, 0.5), m=0)
        mb.box((0, -0.35, -h / 2 - 0.12), (w, 0.6, 0.05), m=0)
    else:
        for sx in (-0.4, 0.4):
            mb.box((sx * w, 0.25, 0), (0.08, 0.35, 0.08), m=0)
    for sx in (-0.33, 0.0, 0.33):                                      # uplight heads on arms in front of the panel
        mb.seg((sx * w, -0.05, -h / 2 - 0.1), (sx * w, -0.9, -h / 2 - 0.25), 0.05, 0.05, m=0)
        mb.box((sx * w, -0.95, -h / 2 - 0.3), (0.3, 0.2, 0.14), m=1)
    st = E.obj(name + "_structure", mb, ["metal_dark", "lamp_lens"], c)
    st.parent = root
    st.matrix_parent_inverse = Matrix.Identity(4)
    if light:
        f = Vector((facing[0], facing[1], 0)).normalized()
        p = Vector(center) + f * 2.5 + Vector((0, 0, -h * 0.2))
        E.light(name + "_wash", "AREA", tuple(p), 18.0 * w, (1.0, 0.95, 0.85), lc, size=w * 0.8, target=tuple(Vector(center)), shadow=False)
    return root, objs


def bus_shelter(c, loc, yaw, design, lc, name="bus_shelter"):
    """2.4 m glass shelter along local +Y with a lit portrait ad box at its near end (front facing local -Y)."""
    root = E.empty(name, c, loc, (0, 0, yaw), 0.4)
    bpy.context.view_layer.update()
    mb = MB()
    for x in (-0.6, 0.6):
        for y in (0.0, 2.4):
            mb.box((x, y, 1.15), (0.06, 0.06, 2.3), m=0)
    mb.box((0, 1.2, 2.33), (1.5, 2.6, 0.08), m=0)
    mb.box((0.6, 1.2, 1.2), (0.02, 2.4, 1.9), m=1)
    mb.box((0.2, 1.2, 0.45), (0.7, 2.0, 0.06), m=0)
    mb.box((0, 0.0, 1.15), (1.25, 0.24, 2.0), m=0)
    st = E.obj(name + "_frame", mb, ["metal_dark", E.principled("BB_shelter_glass", (0.15, 0.17, 0.2), 0.0, 0.05)], c)
    st.parent = root
    st.matrix_parent_inverse = Matrix.Identity(4)
    art_root = E.empty(name + "_ad", c, (0, -0.125, 1.2), (0, 0, 0), 0.2)
    art_root.parent = root
    bpy.context.view_layer.update()
    objs = _art(c, art_root, 1.05, 1.75, design, name + "_ad", lc)
    p = root.matrix_world @ Vector((0, -1.2, 1.3))
    E.light(name + "_glow", "AREA", tuple(p), 25.0, (1.0, 0.97, 0.92), lc, size=1.0,
            target=tuple(root.matrix_world @ Vector((0, -3.0, 0.6))), shadow=False)
    return root, objs


def robotaxi(c, loc, yaw, name="robotaxi", label="WAYMORE"):
    """Generic white robotaxi (front toward local -Y) with a roof lidar dome (spinning lit band) and corner sensors."""
    root = E.empty(name, c, loc, (0, 0, yaw), 0.5)
    bpy.context.view_layer.update()
    mb = MB()
    mb.box((0, 0, 0.72), (1.95, 4.8, 0.78), m=0)
    mb.ellipsoid((0, -2.2, 0.75), (0.97, 0.45, 0.38), 16, 8, m=0)
    mb.ellipsoid((0, 2.25, 0.8), (0.97, 0.4, 0.4), 16, 8, m=0)
    mb.box((0, 0.25, 1.42), (1.75, 2.7, 0.62), m=1)
    for y in (-1.1, 0.25, 1.6):
        for s in (-1, 1):
            mb.box((s * 0.875, y, 1.42), (0.05, 0.12, 0.62), m=0)
    mb.box((0, 0.25, 1.76), (1.75, 2.75, 0.07), m=0)
    mb.cyl((0, -0.35, 1.86), 0.3, 0.14, seg=24, m=0)
    mb.cyl((0, -0.35, 2.05), 0.26, 0.24, seg=24, m=2)
    mb.ellipsoid((0, -0.35, 2.17), (0.26, 0.26, 0.08), 16, 6, m=0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            mb.box((sx * 0.98, sy * 2.15, 1.05), (0.12, 0.25, 0.16), m=3)
            mb.cyl((sx * 0.92, sy * 1.5, 0.36), 0.36, 0.24, axis="x", seg=20, m=4)
        mb.box((sx * 0.6, -2.62, 0.78), (0.42, 0.04, 0.12), m=5)
        mb.box((sx * 0.6, 2.62, 0.82), (0.42, 0.04, 0.1), m=6)
    o = E.obj(name + "_body", mb, [E.principled("BB_taxi_white", (0.78, 0.79, 0.8), 0.1, 0.25, coat=1.0), "glass_dark",
                                     E.principled("BB_lidar_glass", (0.05, 0.06, 0.07), 0.0, 0.08, coat=1.0), "lens_black", "rubber",
                                     "lamp_lens_cool", E.principled("BB_taxi_tail", (0.5, 0.02, 0.02), 0, 0.3, (1.0, 0.05, 0.03), 3.0)], c)
    o.parent = root
    o.matrix_parent_inverse = Matrix.Identity(4)
    band = MB()
    band.ring((0, 0, 0), 0.265, 0.25, 0.03, seg=32)
    band.box((0.26, 0, 0), (0.03, 0.08, 0.05))
    b = E.obj(name + "_lidar_band", band, [E.principled("BB_lidar_band", (0.1, 0.9, 0.8), 0, 0.3, (0.1, 0.9, 0.8), 6.0)], c,
              loc=(0, -0.35, 2.02), shadow=False)
    b.parent = root
    b.matrix_parent_inverse = Matrix.Identity(4)
    b.rotation_euler = (0, 0, 0)
    b.keyframe_insert("rotation_euler", index=2, frame=1)
    b.rotation_euler = (0, 0, 2 * PI)
    b.keyframe_insert("rotation_euler", index=2, frame=11)
    for fc in b.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.modifiers.new("CYCLES")
    lm = E.principled("BB_taxi_label", (0.25, 0.27, 0.3), 0.0, 0.4)
    for s in (-1, 1):
        t = E.text(name + "_label", label, 0.16, c, (s * 0.977, -0.3, 0.85), (PI / 2, 0, s * PI / 2), mat=lm, fnt="Arial Bold")
        t.parent = root
        t.matrix_parent_inverse = Matrix.Identity(4)
    return root


# ================================================================== placement for the S6 street
def build(c, lc, hc, ctaxi=None, layout="ending"):
    """Billboards placed for S's S6 framing (cameras near x = -3, looking +y down the street): rooftop boards on R2
    (AGENTS.), R3 (ARTISAND, low) and R4 (NONLINEAR, high, staggered so they do not overlap) and L3 (SIGMOID), all
    angled toward the camera side; bus-shelter ads on both sidewalks (VANTANOT left, SNOOKS right); the robotaxi parked at
    the right curb. All outside the landmark vista (azimuth ~69-100 deg from the S6 camera positions, above the hill
    line) and above the caption band. The end-of-street flanks were rejected: the side buildings hide them from x = -3."""
    out = {}
    out["artisand"] = billboard(c, (7.6, 27.0, 10.9), (-0.55, -0.83), 6.0, 2.4, "artisand", lc, "bb_artisand", "roof", base_z=8.55)
    out["nonlinear"] = billboard(c, (7.8, 31.0, 15.4), (-0.5, -0.87), 7.0, 2.8, "nonlinear", lc, "bb_nonlinear", "roof", base_z=11.55)
    out["sigmoid"] = billboard(c, (-7.4, 27.4, 7.4), (0.55, -0.83), 5.6, 2.6, "sigmoid", lc, "bb_sigmoid", "roof", base_z=5.55)
    out["agents"] = billboard(c, (7.0, 24.2, 6.7), (-0.6, -0.8), 3.0, 1.5, "agents", lc, "bb_agents", "roof", base_z=5.55)
    out["vantanot"] = bus_shelter(c, (-4.75, 28.2, 0.12), 0.0, "vantanot", lc, "bus_shelter_L")
    out["snooks"] = bus_shelter(c, (4.75, 28.6, 0.12), 0.0, "snooks_portrait", lc, "bus_shelter_R")
    for nm in ("artisand", "sigmoid", "nonlinear", "agents"):
        root = out[nm][0]
        E.hook("billboard_" + nm, hc, tuple(root.location), math.atan2(*reversed(_facing_xy(root))), "landmark")
    for nm in ("vantanot", "snooks"):
        E.hook("billboard_" + nm, hc, tuple(out[nm][0].matrix_world @ Vector((0, -0.125, 1.2))), -PI / 2, "landmark")
    if ctaxi is not None:
        tx = robotaxi(ctaxi, (2.35, 35.2, 0.0), 0.0, "robotaxi", "WAYMORE")        # waiting behind the crosswalk
        E.hook("robotaxi", hc, tuple(tx.location), -PI / 2, "landmark", path_start=[2.35, 35.2, 0.0], path_end=[2.35, 26.5, 0.0],
               note="ENV_robotaxi instancer: key location (path point - this location) to drive slowly toward -y; lidar band spins")
        out["robotaxi"] = tx
    return out


def _facing_xy(root):
    f = root.matrix_world.to_3x3() @ Vector((0, -1, 0))
    return (f.x, f.y)
