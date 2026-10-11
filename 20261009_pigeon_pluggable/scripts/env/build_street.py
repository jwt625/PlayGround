"""Build the night-street sets (workstream E, DevLog-013): street_opening.blend, street_ending.blend, street_kit.blend.

Usage: scripts/bslot.sh -b --factory-startup --python scripts/env/build_street.py -- opening|ending|kit [out.blend]
World frame = scripts/build_crude_film.py: DC facade (hall end wall) at y = 13 (12.94 .. 13.06), street along +y, road
|x| < 3.5, sidewalks to |x| = 6, building fronts at |x| = 6. The hall side (y < 12.94, z < 8) is left empty for S / R.
Writes <out>.blend and <out>.json (hooks with location + facing, counts).
"""
import math
import os
import random
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_lib as E  # noqa: E402
import street_kit as SK  # noqa: E402
from env_lib import MB, PI  # noqa: E402

import vfx  # noqa: E402

# ------------------------------------------------------------------ crude-film anchors (build_crude_film.py v0.2.3)
FACADE_Y = 13.0
FLAP = Vector((0.0, FACADE_Y, 3.25))
TREE_C = Vector((0.0, 17.0, 0.0))
LAMPS = [Vector((-3.7, 16.2, 0)), Vector((3.7, 21.0, 0)), Vector((-3.7, 26.0, 0))]
LAMP_H = 4.6
POLES = [Vector((4.9, 15.4, 0)), Vector((4.9, 28.0, 0))]
ARM_Z = 5.85                       # crossarm centre; pin-insulator wire attach z = 5.95 (= crude wire end height)
WIRE_SAG = 0.55
CCTV = [Vector((3.6, 24.5, 3.45)), Vector((-5.95, 17.0, 3.8)), Vector((5.95, 27.0, 4.4))]
RAIL_X, RAIL_Z = -3.35, 1.0
SAT0, SAT1 = Vector((4, 38, 26)), Vector((9, 38, 27))
SIDEWALK_Z = 0.12
SAT_SCALE = 1.6
SAT_PASS = (Vector((-5.0, 27.0, 17.0)), Vector((1.0, 28.8, 19.0)))   # v2: through S's S6 6.5 s roof-camera view
TREE_ALTS = {"crude": Vector((0.0, 17.0, 0.0)), "y18p5": Vector((0.0, 18.5, 0.0)), "y19p0": Vector((0.0, 19.0, 0.0)),
             "xm1p5": Vector((-1.5, 17.0, 0.0)), "xp1p5": Vector((1.5, 17.0, 0.0)), "xm1p5_y18p5": Vector((-1.5, 18.5, 0.0))}
ZONE_HALF = (1.3, 1.6, 4.6)          # tree footprint half-size x, y and height (tree width +-0.62 m + labels + hero)


def tree_alternatives(hc, top):
    """HOOK_tree_base_<alt> for alternative tree positions, each with a clear-zone check against every set mesh (AABB);
    results go to the JSON (tree_zones)."""
    bpy.context.view_layer.update()
    skip = ("road", "sidewalks", "road_markings", "puddles", "ENV_sky_dome", "street_haze", "wires", "wire_hardware", "skyline",
            "holo_projector", "bldg_backfill")
    boxes = []
    for o in top.all_objects:
        if o.type != "MESH" or o.name.startswith(skip) or "cone" in o.name:
            continue
        pts = [o.matrix_world @ Vector(v) for v in o.bound_box]
        boxes.append((o.name, Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
                      Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))))
    out = {}
    for nm, b in TREE_ALTS.items():
        lo = Vector((b.x - ZONE_HALF[0], b.y - ZONE_HALF[1], 0.05))
        hi = Vector((b.x + ZONE_HALF[0], b.y + ZONE_HALF[1], ZONE_HALF[2]))
        hits = sorted({n for n, a, c in boxes if a.x < hi.x and c.x > lo.x and a.y < hi.y and c.y > lo.y and a.z < hi.z and c.z > lo.z})
        if nm != "crude":
            E.hook("tree_base_" + nm, hc, (b.x, b.y, 0.07), PI / 2, "landmark", clear_hits=",".join(hits) or "none",
                   offset=list(b - TREE_ALTS["crude"]), note="instance ENV_tree_projector at this offset")
        out[nm] = {"base": list(b), "zone_min": list(lo), "zone_max": list(hi), "hits": hits}
        print("TREE ZONE", nm, hits or "clear")
    return out
DC_DIR = os.path.join(E.ROOT, "assets", "components", "datacenter")


def bldg_h(sx, k):
    return 6 + 3 * ((k * 7 + (sx > 0)) % 3)


# ================================================================== ground
def ground(c):
    mb = MB()
    mb.box((0, 25.0, -0.05), (7.0, 23.9, 0.1), m=0)                     # road y 13.06 .. 36.96
    mb.box((0, 35.25, -0.05), (34.0, 4.5, 0.1), m=0)                    # cross street y 33 .. 37.5
    o_road = E.obj("road", mb, ["asphalt"], c)
    mb = MB()
    for sx in (-1, 1):
        mb.box((sx * 4.75, 23.03, SIDEWALK_Z / 2), (2.5, 19.94, SIDEWALK_Z), m=0)      # y 13.06 .. 33
        mb.box((sx * 11.0, 23.03, SIDEWALK_Z / 2), (10.0, 19.94, SIDEWALK_Z), m=0)     # under the buildings / behind
        mb.box((sx * 3.55, 23.03, 0.07), (0.15, 19.94, 0.14), m=1)                    # curb
        for y in (17.0, 25.0, 32.3):
            mb.box((sx * 3.3, y, 0.003), (0.35, 0.8, 0.006), m=2)                      # storm drain grate
        for k in range(20):
            mb.box((sx * 4.75, 13.6 + k, SIDEWALK_Z + 0.001), (2.48, 0.012, 0.002), m=3)    # paving joints
    mb.box((0, 38.5, SIDEWALK_Z / 2), (34.0, 2.0, SIDEWALK_Z), m=0)                  # far sidewalk (end block)
    mb.box((0, 13.63, 0.01), (7.0, 1.14, 0.02), m=4)                                 # DC forecourt apron
    mb.box((0, 14.2, 0.021), (7.0, 0.12, 0.004), m=2)                                # trench drain
    o_side = E.obj("sidewalks", mb, ["sidewalk", "curb", "grille", "metal_dark", "concrete_light"], c)
    mb = MB()
    for y in (20.5, 23.5, 26.5, 29.5):
        mb.box((0, y, 0.004), (0.12, 1.4, 0.006), m=0)
    for k in range(7):                                                                 # zebra crossing
        mb.box((-3.0 + k, 31.6, 0.004), (0.5, 2.0, 0.006), m=0)
    mb.box((0, 30.3, 0.004), (7.0, 0.3, 0.006), m=0)
    for x, y in ((1.5, 22.5), (-1.2, 29.0)):
        mb.cyl((x, y, 0.004), 0.34, 0.008, seg=24, m=1)
        mb.ring((x, y, 0.009), 0.3, 0.26, 0.004, seg=24, m=2)
        mb.ring((x, y, 0.009), 0.18, 0.15, 0.004, seg=24, m=2)
    o_mark = E.obj("road_markings", mb, ["paint_white", "metal_dark", "grille"], c)
    pud = SK.puddles(c, [(-1.8, 16.0, 0.45, 0.0), (2.1, 19.8, 0.55, 0.0), (-0.9, 24.2, 0.4, 0.0), (1.5, 27.6, 0.6, 0.0),
                         (-2.5, 30.4, 0.5, 0.0), (4.8, 19.0, 0.3, SIDEWALK_Z), (-4.6, 21.6, 0.35, SIDEWALK_Z),
                         (4.5, 30.0, 0.3, SIDEWALK_Z), (1.2, 14.6, 0.3, 0.0), (-2.0, 34.5, 0.8, 0.0)])
    return [o_road, o_side, o_mark, pud]


# ================================================================== DC facade
def _wall_strips(mb, xs, holes, z0, z1, y0, y1, m):
    """Wall boxes between x breakpoints xs, from z0 to z1, minus rectangular holes (xa, xb, za, zb) aligned to xs."""
    for xa, xb in zip(xs[:-1], xs[1:]):
        xm = (xa + xb) / 2
        cuts = sorted((za, zb) for (ha, hb, za, zb) in holes if ha <= xm <= hb)
        z = z0
        for za, zb in cuts:
            if za > z:
                mb.box2((xa, y0, z), (xb, y1, za), m)
            z = max(z, zb)
        if z < z1:
            mb.box2((xa, y0, z), (xb, y1, z1), m)


def dc_facade(c, lc, hc):
    """EQUONOX end wall seen from the street. Openings: door 1.4 x 2.5 at x = 0, bird flap 0.7 x 0.5 centred at
    (0, 3.25), loading door 2.8 x 3.2 at x = 2.6 .. 5.4. R's bird_flap_vent is appended under HOOK_bird_flap_vent."""
    fy0, fy1 = FACADE_Y - 0.06, FACADE_Y + 0.06
    door = (-0.7, 0.7, 0.0, 2.5)
    flap = (-0.35, 0.35, 3.0, 3.5)
    dock = (2.6, 5.4, 0.0, 3.2)
    mats = ["cladding", "concrete", "metal_dark", "glass_dark", "led_cyan", "hazard", "shutter", "rubber", "paint_yellow",
            "grille", "led_warm", "camera_white", "lens_black", "ac_white", "galv", "led_red", "concrete_light", "win_cool"]
    m = {n: i for i, n in enumerate(mats)}
    mb = MB()
    xs = [-6.0, -0.7, -0.35, 0.35, 0.7, 2.6, 5.4, 6.0]
    _wall_strips(mb, xs, [door, flap, dock], 0.0, 8.0, fy0, fy1, m["cladding"])
    for sx in (-1, 1):                                                   # returns to the side buildings
        mb.box2((sx * 6.0, fy1, 0.0), (sx * 6.25, 13.25, 8.6), m["cladding"])
    _wall_strips(mb, [-6.0, -0.7, 0.7, 2.6, 5.4, 6.0], [door, dock], 0.0, 0.6, fy1, fy1 + 0.1, m["concrete"])   # plinth
    # parapet + roof slab (above the hall ceiling; nothing below z = 7.85 on the hall side)
    mb.box2((-6.0, fy0, 8.0), (6.0, fy1 + 0.04, 8.6), m["cladding"])
    mb.box2((-6.1, fy0 - 0.05, 8.6), (6.1, fy1 + 0.1, 8.66), m["concrete_light"])
    mb.box2((-6.0, 10.4, 7.85), (6.0, fy0, 8.0), m["concrete"])
    # slit windows (control room) and LED band
    for k in range(9):
        x = -5.0 + 1.25 * k
        if abs(x) < 0.4:
            continue
        mb.quad((x - 0.45, fy1 + 0.004, 6.5), (x + 0.45, fy1 + 0.004, 6.5), (x + 0.45, fy1 + 0.004, 6.78), (x - 0.45, fy1 + 0.004, 6.78),
                m["win_cool"])
        mb.box((x, fy1 + 0.02, 6.47), (0.95, 0.04, 0.03), m["metal_dark"])
    mb.box((0, fy1 + 0.03, 7.6), (11.8, 0.03, 0.04), m["led_cyan"])
    # door frame + canopy + threshold + reader
    for sx in (-1, 1):
        mb.box((sx * 0.73, fy1 + 0.03, 1.29), (0.06, 0.06, 2.58), m["metal_dark"])
    mb.box((0, fy1 + 0.03, 2.54), (1.52, 0.06, 0.08), m["metal_dark"])
    mb.box((0, fy1 + 0.6, 2.7), (3.0, 1.2, 0.1), m["metal_dark"])
    mb.box((0, fy1 + 1.18, 2.65), (3.0, 0.03, 0.02), m["led_cyan"])
    for sx in (-1, 1):
        mb.seg((sx * 1.3, fy1, 3.4), (sx * 1.3, fy1 + 1.1, 2.76), 0.03, 0.03, m["galv"])
    mb.box((0, fy1 + 0.25, 0.005), (1.6, 0.5, 0.01), m["metal_dark"])
    mb.box((1.05, 13.32, 0.65), (0.12, 0.08, 1.3), m["metal_dark"])
    mb.box((1.05, 13.275, 1.15), (0.08, 0.004, 0.12), m["glass_dark"])
    mb.ellipsoid((1.05, 13.272, 1.27), (0.008, 0.004, 0.008), 6, 4, m["led_cyan"])
    mb.ellipsoid((0, fy1 + 0.95, 2.6), (0.08, 0.08, 0.05), 12, 6, m["lens_black"])
    mb.cyl((0, fy1 + 0.95, 2.64), 0.09, 0.02, seg=14, m=m["camera_white"])
    # flap exterior: hood, guide LEDs (R's vent frame provides the trim and the outside ledge)
    with mb.at((0, fy1, 3.66), rx=0.25):
        mb.box((0, 0.17, 0), (1.0, 0.34, 0.03), m["metal_dark"])
    for sx in (-1, 1):
        mb.box((sx * 0.47, fy1 + 0.012, 3.25), (0.025, 0.02, 0.5), m["led_cyan"])
        for k in range(3):
            zz = 2.95 - 0.12 * k
            mb.box((sx * 0.47, fy1 + 0.012, zz), (0.025, 0.02, 0.05), m["led_cyan"])
    # loading door: shutter, frame, hood, bollards, wall pack, keep-clear box
    mb.quad((2.6, fy1 - 0.05, 0.0), (5.4, fy1 - 0.05, 0.0), (5.4, fy1 - 0.05, 3.2), (2.6, fy1 - 0.05, 3.2), m["shutter"])
    mb.box((4.0, fy1 - 0.03, 0.03), (2.8, 0.05, 0.06), m["rubber"])
    for xa in (2.5, 5.4):
        mb.box2((xa, fy1, 0.0), (xa + 0.1, fy1 + 0.1, 3.25), m["hazard"])
    mb.box2((2.45, fy1, 3.2), (5.55, fy1 + 0.32, 3.55), m["metal_dark"])
    for x in (2.3, 5.7):
        mb.cyl((x, 13.55, 0.5), 0.1, 1.0, seg=14, m=m["paint_yellow"])
        mb.cyl((x, 13.55, 1.0), 0.1, 0.04, seg=14, m=m["metal_dark"], r2=0.06)
    mb.box((4.0, fy1 + 0.08, 3.8), (0.3, 0.16, 0.18), m["metal_dark"])
    mb.box((4.0, fy1 + 0.08, 3.705), (0.24, 0.12, 0.01), m["led_warm"])
    for (xa, xb, ya, yb) in ((2.6, 5.4, 13.2, 13.28), (2.6, 5.4, 14.92, 15.0), (2.6, 2.68, 13.2, 15.0), (5.32, 5.4, 13.2, 15.0)):
        mb.box2((xa, ya, 0.0), (xb, yb, 0.026), m["paint_yellow"])
    for k in range(6):
        y = 13.35 + 0.28 * k
        mb.seg((2.7, y, 0.023), (3.3, y + 0.25, 0.023), 0.05, 0.004, m["paint_yellow"])
    # exhaust louvre bank (viewer's right of the door)
    mb.box2((-5.6, fy1, 3.8), (-2.6, fy1 + 0.08, 5.45), m["metal_dark"])
    for k in range(11):
        zz = 3.88 + 0.14 * k
        with mb.at((-4.1, fy1 + 0.1, zz), rx=-0.6):
            mb.box((0, 0, 0), (2.9, 0.12, 0.012), m["grille"])
    # roof: dry coolers + mast
    for x in (-3.6, 0.0, 3.6):
        mb.box((x, 11.7, 8.65), (2.4, 1.2, 1.3), m["ac_white"])
        for dx in (-0.6, 0.6):
            mb.cyl((x + dx, 11.7, 9.31), 0.48, 0.01, seg=24, m=m["grille"])
            mb.ring((x + dx, 11.7, 9.32), 0.5, 0.46, 0.03, seg=24, m=m["metal_dark"])
        for sx in (-1, 1):
            mb.box((x + sx * 1.2, 11.7, 8.6), (0.006, 1.1, 1.1), m["grille"])
    mb.rod((5.2, 12.2, 8.0), (5.2, 12.2, 11.2), 0.035, 8, m=m["galv"])
    mb.ellipsoid((5.2, 12.2, 11.25), (0.05, 0.05, 0.05), 8, 4, m["led_red"])
    for zz in (9.2, 10.3):
        mb.rod((4.7, 12.2, zz), (5.7, 12.2, zz), 0.012, 6, m=m["galv"])
    o = E.obj("dc_facade", mb, mats, c)
    out = {"obj": o}
    # sliding door leaves (open), separate so S can animate them (origin = leaf centre)
    for sx, nm in ((-1, "L"), (1, "R")):
        lb = MB()
        lb.box((0, 0, 0), (0.74, 0.03, 2.46), m=1)
        lb.box((0, 0, 0), (0.7, 0.02, 2.4), m=0)
        lb.box((-sx * 0.33, 0.02, 0.0), (0.03, 0.02, 0.9), m=1)
        lo = E.obj("dc_door_leaf_" + nm, lb, ["glass_dark", "metal_dark"], c, loc=(sx * 1.07, fy1 + 0.1, 1.25))
        lo["closed_x"] = sx * 0.36
        lo["open_x"] = sx * 1.07
        out["leaf_" + nm] = lo
    # sign and plates
    SK.neon_sign(c, (0, fy1 + 0.07, 4.55), PI, "EQUONOX", "neon_cyan", 0.55, name="sign_equonox",
                 light=((0.1, 0.85, 1.0), 90.0, lc))
    plate_m = E.principled("ENV_plate_white", (0.8, 0.8, 0.8), 0, 0.4, (0.8, 0.8, 0.8), 0.25)
    for txt, x, z, sz in (("DC-01", -1.15, 1.8, 0.09), ("AUTHORIZED BIRDS ONLY", 1.2, 3.25, 0.055),
                          ("DOCK 1", 4.0, 4.2, 0.2), ("KEEP CLEAR", 4.0, 1.9, 0.0)):
        if sz == 0:
            continue
        pb = MB()
        w = max(0.3, len(txt) * sz * 0.72 + 0.12)
        pb.box((x, fy1 + 0.012, z), (w, 0.02, sz * 1.9), m=0)
        E.obj("plate_" + txt.replace(" ", "_"), pb, ["sign_back"], c)
        E.text("plate_txt_" + txt.replace(" ", "_"), txt, sz, c, (x, fy1 + 0.0225, z), (PI / 2, 0, PI), mat=plate_m, fnt="Arial Bold")
    t = E.text("keep_clear", "KEEP CLEAR", 0.32, c, (4.0, 14.1, 0.028), (0, 0, PI), mat="paint_yellow", fnt="Arial Black")
    # cameras
    for nm, x, yaw in (("dc_L", -5.4, PI / 2 - 0.35), ("dc_R", 5.4, PI / 2 + 0.35)):
        cc = SK.cctv(c, (x + 0.0, 13.06 + 0.37, 6.2), yaw, mount="wall", name="cctv_" + nm, tilt=0.3)
        E.hook("cctv_%s_lens" % nm, hc, tuple(cc["lens"]), math.atan2(cc["lens_dir"].y, cc["lens_dir"].x), "camera")
    # traffic cones by the dock
    cb = MB()
    _, mi = SK._bm("concrete")
    for p in ((3.05, 14.35, 0), (3.55, 14.7, 0), (5.0, 14.25, 0)):
        SK.cone_prop(cb, p, mi)
    E.obj("cones", cb, SK._bm("concrete")[0], c)
    # lights
    for sx in (-1, 1):
        E.light("dc_canopy_down_%d" % (sx > 0), "SPOT", (sx * 0.8, fy1 + 0.9, 2.62), 140, (0.8, 0.9, 1.0), lc, spot_deg=110,
                blend=0.8, rot=(0, 0, 0), radius=0.05)
    E.light("dc_dock_pack", "SPOT", (4.0, fy1 + 0.25, 3.68), 260, (1.0, 0.85, 0.65), lc, spot_deg=120, blend=0.7, target=(4.0, 15.0, 0.0),
            radius=0.08)
    E.light("dc_flap_glow", "POINT", (0, fy1 + 0.35, 3.25), 12, (0.1, 0.85, 1.0), lc, radius=0.1, shadow=False)
    E.light("dc_wash", "AREA", (0, 17.5, 7.5), 220, (0.65, 0.75, 1.0), lc, size=8, target=(0, 13.0, 4.0))
    # hooks
    E.hook("bird_flap_vent", hc, (FLAP.x, FACADE_Y - 0.06, FLAP.z), -PI / 2, "mount",
           note="R's ASSET_bird_flap_vent: origin = opening centre on the inner wall face, +X into the hall")
    E.hook("flap_outside", hc, (FLAP.x, fy1 + 0.1, FLAP.z), PI / 2, "landmark")
    E.hook("door_dc", hc, (0.0, fy1, 0.0), PI / 2, "landmark", note="door opening 1.4 x 2.5 m; +X = out to the street")
    E.hook("door_dc_inside", hc, (0.0, fy0, 0.0), -PI / 2, "landmark")
    E.hook("loading_door", hc, (4.0, fy1, 0.0), PI / 2, "landmark")
    E.hook("dc_sign", hc, (0.0, fy1 + 0.07, 4.55), PI / 2, "landmark")
    return out


def append_flap_vent(c, hook):
    path = os.path.join(DC_DIR, "bird_flap_vent.blend")
    if not os.path.exists(path):
        print("WARN no R flap vent at", path)
        return None
    vc = E.append_collection(path, "ASSET_bird_flap_vent")
    if vc is None:
        print("WARN collection ASSET_bird_flap_vent missing")
        return None
    c.children.link(vc)
    for o in vc.all_objects:
        if o.parent is None:
            o.parent = hook
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(v) for o in vc.all_objects if o.type == "MESH" for v in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    print("R flap vent world bbox", tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi))
    return vc


# ================================================================== buildings
SIGNS = {  # (side, k): flat neon over the shop (text, neon, z, size) ; crude panel colours kept per building
    (-1, 0): ("BIT ERROR BAR", "neon_mag", 3.05, 0.3), (-1, 2): ("CLOSEDAI", "neon_white", 3.05, 0.36),
    (-1, 4): ("PERPLEXED", "neon_cyan", 3.05, 0.34), (1, 0): ("PACKET INN", "neon_cyan", 3.05, 0.32),
    (1, 3): ("MISANTHROPIC", "neon_orange", 3.05, 0.28), (1, 4): ("OPEN 24H", "neon_orange", 3.05, 0.36),
}
BIG = {(-1, 3): ("LINK UP", "neon_mag", 3.45, 0.6), (1, 2): ("NO SIGNAL", "neon_cyan_fl", 3.9, 0.55)}   # crude channel letters
BLADES = [((1, 1), "HOTEL", "neon_orange", 1.5, 6.0), ((-1, 1), "BAR", "neon_mag", -1.1, 5.2), ((-1, 2), "PHO", "neon_cyan", 1.4, 7.6),
          ((1, 4), "LIVE", "neon_violet", -1.3, 7.5)]
WALLS = ["plaster_a", "plaster_b", "concrete", "plaster_c", "tile_dark"]
BCFG = {  # shop, fire escape (local x range), roof items, awning, skip windows
    (-1, 0): dict(shop="lit", roof=[("ac", 0.6, -1.6, 0.0), ("whip", -1.2, -2.2, 1.6)], awning=True),
    (-1, 1): dict(shop="shutter", roof=[("ac", 0.3, -1.8, 0.4), ("yagi", -1.0, -2.3, 2.4)]),
    (-1, 2): dict(shop="lit", roof=[("tank", 0.2, -1.6, 0), ("yagi", 1.3, -2.5, 2.8), ("vent", -1.3, -0.9, 0)]),
    (-1, 3): dict(shop="shutter", roof=[("dish", 0.5, -1.4, 0.8), ("vent", -1.0, -2.2, 0)], skip=[(1, 0), (1, 1)]),
    (-1, 4): dict(shop="lit", fire=(-1.6, 0.1), roof=[("ac", 0.8, -1.8, 0.0), ("dish", -0.9, -1.2, -0.6)]),
    (1, 0): dict(shop="lit", roof=[("ac", -0.3, -1.7, 0.0), ("vent", 1.0, -2.2, 0)]),
    (1, 1): dict(shop="shutter", roof=[("ac", 0.5, -1.5, 0.3), ("ac", -0.8, -2.0, 0.0), ("whip", 1.3, -2.4, 2.0)]),
    (1, 2): dict(shop="lit", roof=[("yagi", 0.0, -2.0, 2.2)], skip=[(1, 0), (1, 1)], awning=True),
    (1, 3): dict(shop="lit", roof=[("dish", 0.3, -1.5, 2.4), ("ac", -1.0, -2.1, 0.0)]),
    (1, 4): dict(shop="lit", fire=(-0.1, 1.6), roof=[("whip", 1.2, -2.5, 2.4)]),          # v4: tank removed (NONLINEAR board)
}


def buildings(c, sc, lc, rng):
    perches = []
    for sx in (-1, 1):
        for k in range(5):
            h = bldg_h(sx, k)
            yc = 15 + 4 * k
            yaw = -PI / 2 if sx < 0 else PI / 2
            cfg = BCFG[(sx, k)]
            wall = E.M(WALLS[(k * 2 + (sx > 0)) % len(WALLS)])
            b = SK.building(c, (sx * 6.0, yc, 0.0), yaw, 3.6, 3.0, h, wall, rng, "bldg_%s%d" % ("L" if sx < 0 else "R", k),
                            shop=cfg["shop"], fire_escape=cfg.get("fire"), roof=cfg["roof"], awning=cfg.get("awning", False),
                            skip_windows=cfg.get("skip", ()), win_ac=0.3)
            for nm, p, yw in b["perches"]:
                perches.append(("%s%d_%s" % ("L" if sx < 0 else "R", k, nm), p, yw))
            face = PI / 2 if sx < 0 else -PI / 2                                  # neon text facing the street
            col = {"neon_mag": (1.0, 0.1, 0.7), "neon_cyan": (0.1, 0.85, 1.0), "neon_orange": (1.0, 0.45, 0.05),
                   "neon_orange_fl": (1.0, 0.45, 0.05), "neon_cyan_fl": (0.1, 0.85, 1.0), "neon_violet": (0.55, 0.2, 1.0),
                   "neon_white": (0.9, 0.95, 1.0)}
            if (sx, k) in SIGNS:
                txt, neon, z, size = SIGNS[(sx, k)]
                SK.neon_sign(sc, (sx * (6.0 - 0.09), yc, z), face, txt, neon, size, name="sign_%s" % txt.replace(" ", "_"),
                             light=(col[neon], 70.0, lc))
            if (sx, k) in BIG:
                txt, neon, z, size = BIG[(sx, k)]
                SK.neon_sign(sc, (sx * (6.0 - 0.04), yc, z), face, txt, neon, size, name="sign_%s" % txt.replace(" ", "_"),
                             backer=False, light=(col[neon], 150.0, lc))
    for (sx, k), txt, neon, dl, z in BLADES:
        yc = 15 + 4 * k + dl
        bl = SK.blade_sign(sc, (sx * 6.0, yc, z), 0.0 if sx < 0 else PI, txt, neon, 0.3, name="blade_%s" % txt)
        col = {"neon_mag": (1.0, 0.1, 0.7), "neon_cyan": (0.1, 0.85, 1.0), "neon_orange": (1.0, 0.45, 0.05),
               "neon_violet": (0.55, 0.2, 1.0)}[neon]
        E.light("blade_%s_spill" % txt, "POINT", (sx * 5.2, yc, z), 60, col, lc, radius=0.4, shadow=False)
        perches.append(("blade_%s_top" % txt, bl["top"] + Vector((0, 0, 0.0)), PI / 2 if sx < 0 else -PI / 2))
    # back walls behind the building row (close the 0.4 m gaps)
    mb = MB()
    for sx in (-1, 1):
        mb.box((sx * 9.2, 23.0, 3.0), (0.3, 20.0, 6.0), m=0)            # v4: 14 -> 6 m (it rose behind the roofs and hid the billboards)
        for k in range(4):
            mb.box((sx * 7.5, 17 + 4 * k, 3.0), (3.0, 0.4, 6.0), m=0)
    E.obj("bldg_backfill", mb, ["concrete"], c)
    # string course ledge (crude ledge perches at z 6.06)
    mb = MB()
    mb.box((-5.87, 21.5, 6.0), (0.26, 8.0, 0.12), m=0)
    mb.box((-5.76, 21.5, 5.925), (0.04, 8.0, 0.03), m=0)
    E.obj("ledge", mb, ["concrete_light"], c)
    return perches


# ================================================================== skyline (end of the street + far towers)
def skyline(c, lc, rng, hc=None, top=None):
    """v3: corner blocks at the cross street (y 37.5 .. 46.5) framing a steep hill street with Victorian houses and a cable
    car, and the San Francisco backdrop behind it (scripts/env/sf_landmarks.py)."""
    import sf_landmarks as SF
    gm = SK.window_grid_material()
    mb = MB()
    for sx in (-1, 1):
        mb.box((sx * 10.3, 42.0, 5.0), (9.4, 9.0, 10.0), m=0)
        xa, xb = (-14.6, -6.0) if sx < 0 else (6.0, 14.6)
        mb.quad((xa, 37.49, 0.4 + SIDEWALK_Z), (xb, 37.49, 0.4 + SIDEWALK_Z), (xb, 37.49, 3.0), (xa, 37.49, 3.0), m=1)
        mb.box((sx * 10.3, 37.45, 3.3), (9.2, 0.1, 0.4), m=2)
    for sx in (-1, 1):
        for k in range(3):
            x = sx * (17 + 7 * k)
            h = rng.uniform(10, 18)
            mb.box((x, 30 - 6 * k, h / 2), (6, 5, h), m=0)
    o = E.obj("skyline_corners", mb, [gm, "shopfront", "sign_back"], c)
    SK.neon_sign(c, (-10.3, 37.42, 7.0), 0.0, "OUNOLIGHT", "neon_mag", 0.9, name="sign_OUNOLIGHT", light=((1.0, 0.1, 0.7), 400.0, lc))
    SK.neon_sign(c, (10.3, 37.42, 6.5), 0.0, "LATENCY LOUNGE", "neon_cyan", 0.42, name="sign_LATENCY_LOUNGE",
                 light=((0.1, 0.85, 1.0), 200.0, lc))
    if top is not None:
        csf = E.coll("ENV_sf_skyline", top)
        ccar = E.coll("ENV_cable_car", top)
        SF.build(csf, lc, hc, ccar)
    return o


# ================================================================== street furniture
def furniture(c, lc, hc, layout):
    P = {}            # perch name -> (point, yaw, crude_name)
    lamps = []
    for i, p in enumerate(LAMPS):
        yaw = 0.0 if p.x < 0 else PI
        lp = SK.lamp_post(c, tuple(p), yaw, "lamp_%d" % i, LAMP_H, 1.05, street_sign="PIGEON ST" if i == 0 else None)
        lamps.append(lp)
        hd = lp["head"]
        E.light("lamp_%d_spot" % i, "SPOT", tuple(hd), 650, (1.0, 0.76, 0.5), lc, spot_deg=125, blend=0.85, rot=(0, 0, 0), radius=0.12)
        vfx.light_cone("lamp_%d_cone" % i, tuple(hd + Vector((0, 0, 0.01))), (0, 0, -1), 4.3, 78, (1.0, 0.78, 0.55), 0.06, c)
        E.hook("lamp_%d_head" % i, hc, tuple(hd), yaw, "light")
    poles = [SK.utility_pole(c, tuple(p), 0.0, "util_pole_%d" % i, 6.4, ARM_Z, transformer=(i == 0)) for i, p in enumerate(POLES)]
    a0, a1 = poles[0]["attach"], poles[1]["attach"]
    spans = [(a0[dx], a1[dx], WIRE_SAG, 0.008) for dx in (-0.5, 0.0, 0.5)]
    spans.append((a0["telecom"], a1["telecom"], 0.45, 0.014))
    spans.append((a0[-0.5], Vector((-6.0, 18.0, 5.9)), 0.45, 0.008))
    spans += [(a1[0.5], Vector((6.0, 27.2, 6.6)), 0.3, 0.007), (a1[0.5], Vector((6.0, 30.0, 7.0)), 0.35, 0.007),
              (a0[0.5], Vector((6.0, 14.2, 6.3)), 0.2, 0.007), (a0[0.0], Vector((5.8, 13.08, 6.4)), 0.25, 0.008),
              (a1[0.0], Vector((4.9, 40.0, 6.2)), 0.6, 0.008), (a1[-0.5], Vector((4.4, 40.0, 6.2)), 0.6, 0.008),
              (a1["telecom"], Vector((4.9, 40.0, 5.0)), 0.5, 0.014)]
    festoon = (Vector((-6.0, 24.6, 7.8)), Vector((6.0, 26.0, 7.4)), 0.8)
    spans.append((festoon[0], festoon[1], festoon[2], 0.005))
    SK.wires(c, spans, "wires")
    mb = MB()
    for pt in (Vector((-6.0, 18.0, 5.9)), Vector((6.0, 27.2, 6.6)), Vector((6.0, 30.0, 7.0)), Vector((6.0, 14.2, 6.3)),
               festoon[0], festoon[1]):
        mb.box(tuple(pt - Vector((math.copysign(0.03, pt.x), 0, 0))), (0.06, 0.08, 0.08), m=0)
    fpts = E.catenary(festoon[0], festoon[1], festoon[2], 15)
    for p in fpts[1:-1]:
        mb.rod(tuple(p), tuple(p - Vector((0, 0, 0.06))), 0.003, 4, m=0)
        mb.ellipsoid(tuple(p - Vector((0, 0, 0.1))), (0.045, 0.045, 0.06), 10, 6, m=1)
    E.obj("wire_hardware", mb, ["galv", "led_warm"], c, shadow=False)
    SK.railing(c, RAIL_X, 18.0, 23.0, RAIL_Z, "railing")
    acs = [SK.ac_wall(c, (5.72, 19.0, 3.0), PI, "ac_1"), SK.ac_wall(c, (5.72, 23.5, 4.8), PI, "ac_2"),
           SK.ac_wall(c, (-5.72, 22.0, 3.6), 0.0, "ac_3")]
    hs = SK.hanging_sign(c, (-5.4, 17.5, 3.4), 0.0, ["RAMEN", "24H"], "neon_orange", "hanging_sign_ramen")
    E.light("hanging_sign_spill", "POINT", (-5.0, 17.5, 3.2), 45, (1.0, 0.45, 0.05), lc, radius=0.3, shadow=False)
    bn = SK.bin_prop(c, (3.4, 18.0, 0.0), "bin")
    sc = SK.scooter(c, (-2.9, 15.6, 0.0), 0.0, "scooter")
    hy = SK.hydrant(c, (4.4, 22.6, SIDEWALK_Z), "hydrant")
    vm = SK.vending_machine(c, (-5.25, 24.7, SIDEWALK_Z), PI / 2, "vending")
    E.light("vending_glow", "AREA", tuple(vm["front"] + Vector((0.3, 0, 0))), 35, (0.8, 0.9, 1.0), lc, size=0.8,
            target=tuple(vm["front"] + Vector((3.0, 0, -0.5))))
    cams = [SK.cctv(c, tuple(CCTV[0]), PI, "pole", "cctv_0"),
            SK.cctv(c, (-6.0 + 0.385, CCTV[1].y, CCTV[1].z), 0.0, "wall", "cctv_1", tilt=0.15),
            SK.cctv(c, (6.0 - 0.385, CCTV[2].y, CCTV[2].z), PI, "wall", "cctv_2", tilt=0.15)]
    for i, cc in enumerate(cams):
        E.hook("cctv_%d_lens" % i, hc, tuple(cc["lens"]), math.atan2(cc["lens_dir"].y, cc["lens_dir"].x), "camera",
               crude_pos=list(CCTV[i]))
    sat = None
    E.hook("street_vista", hc, (0.0, 33.0, 1.6), PI / 2, "landmark")
    # ---------------------------------------------------------- crude perch spots (same coordinates as build_crude_film)
    def wire_z(y, dx=0.0):
        p0, p1 = a0[dx], a1[dx]
        return E.catenary_z(p0, p1, WIRE_SAG, (y - p0.y) / (p1.y - p0.y))
    crude = [("lamp1_top", LAMPS[0] + Vector((0, 0, LAMP_H + 0.02)), 0.3), ("lamp1_arm", LAMPS[0] + Vector((0.65, 0, LAMP_H - 0.02)), -1.3),
             ("lamp2_arm", LAMPS[1] + Vector((-0.5, 0, LAMP_H - 0.02)), 2.4)]
    for y, yaw in ((18.2, PI), (19.0, PI - 0.2), (20.7, 0.1)):
        crude.append(("wire_%.1f" % y, Vector((POLES[0].x, y, wire_z(y) + 0.01)), yaw))
    crude += [("rail_a", Vector((RAIL_X, 18.6, RAIL_Z + 0.025)), 0.35), ("rail_b", Vector((RAIL_X, 22.1, RAIL_Z + 0.025)), -0.25),
              ("ac_1", Vector((5.72, 19.0, 3.275)), PI), ("ac_2", Vector((5.72, 23.5, 5.075)), PI - 0.4),
              ("ledge_a", Vector((-5.85, 23.0, 6.06)), 0.1), ("ledge_b", Vector((-5.85, 24.1, 6.06)), -0.35),
              ("sign", Vector((-5.4, 17.5, 3.7)), 0.5), ("bin", Vector((3.4, 17.8, 0.955)), 2.1),
              ("scooter", Vector((-2.9, 15.8, 0.71)), 0.2), ("roof_L", Vector((-6.04, 19.6, bldg_h(-1, 1) + 0.1)), 0.0),
              ("roof_R", Vector((6.04, 15.6, bldg_h(1, 0) + 0.1)), PI),
              ("cctv_pole", Vector((CCTV[0].x, CCTV[0].y + 0.05, CCTV[0].z + 0.1)), PI + 0.5),
              ("hero", Vector((RAIL_X, 20.1, RAIL_Z + 0.025)), 0.15), ("comp0", LAMPS[2] + Vector((0, 0, LAMP_H + 0.02)), 0.4),
              ("comp1", Vector((POLES[0].x, 23.0, wire_z(23.0) + 0.01)), PI + 0.15)]
    for nm, p, yaw in crude:
        P[nm] = (p, yaw, nm)
    extras = [("lamp_1_top", lamps[1]["top"] + Vector((0, 0, 0.02)), PI + 0.4), ("lamp_2_arm", lamps[2]["arm"](0.55), 0.6),
              ("lamp_0_arm_tip", lamps[0]["arm"](0.85), 0.9),
              ("pole_0_arm_end", poles[0]["arm_top"](-0.66), PI - 0.3), ("pole_1_arm", poles[1]["arm_top"](-0.25), PI),
              ("pole_1_arm_b", poles[1]["arm_top"](0.25), PI + 0.4),
              ("wire_W_16.6", Vector((POLES[0].x - 0.5, 16.6, wire_z(16.6, -0.5) + 0.01)), PI - 0.1),
              ("wire_E_21.8", Vector((POLES[0].x + 0.5, 21.8, wire_z(21.8, 0.5) + 0.01)), PI + 0.2),
              ("wire_W_25.4", Vector((POLES[0].x - 0.5, 25.4, wire_z(25.4, -0.5) + 0.01)), PI - 0.4),
              ("wire_M_26.6", Vector((POLES[0].x, 26.6, wire_z(26.6) + 0.01)), 0.3),
              ("ac_3", acs[2]["top"], 0.2), ("hydrant", hy["top"], 1.2), ("vending", vm["top"] + Vector((0.0, 0.1, 0)), 0.0),
              ("cctv_2", cams[2]["top"] + Vector((0, 0, 0.0)), PI + 0.3), ("scooter_bar", sc["bar"], PI / 2 + 0.6),
              ("dc_canopy", Vector((0.9, FACADE_Y + 1.2, 2.75)), PI / 2), ("dc_dock_hood", Vector((4.6, FACADE_Y + 0.25, 3.55)), PI / 2 + 0.4),
              ("rail_post_23", Vector((RAIL_X, 23.03, RAIL_Z + 0.04)), -0.5)]
    for nm, p, yaw in extras:
        P["x_" + nm] = (p, yaw, None)
    return P, sat, poles


# ================================================================== layouts
def build(layout, out_blend):
    E.reset()
    E.HOOKS.clear()
    scn = bpy.context.scene
    rng = random.Random(20261010)
    top = E.coll("ENV_street_%s" % layout)
    cg = E.coll("ENV_ground", top)
    cb = E.coll("ENV_buildings", top)
    cdc = E.coll("ENV_dc_facade", top)
    cs = E.coll("ENV_signs", top)
    cf = E.coll("ENV_furniture", top)
    ck = E.coll("ENV_skyline", top)
    csky = E.coll("ENV_sky", top)
    cl = E.coll("ENV_lights", top)
    cv = E.coll("ENV_vfx", top)
    hc = E.coll("ENV_hooks", top)
    ground(cg)
    dc_facade(cdc, cl, hc)
    flap_hook = next(h for h in E.HOOKS if h.name == "HOOK_bird_flap_vent")
    rc = E.coll("EXT_R_bird_flap_vent", top)
    append_flap_vent(rc, flap_hook)
    bperches = buildings(cb, cs, cl, rng)
    skyline(ck, cl, rng, hc, top)
    P, sat, poles = furniture(cf, cl, hc, layout)
    # perches: crude spots always (street birds are visible in S1 and S6); natural extras in the ending layout
    for nm, (p, yaw, crude) in P.items():
        if nm.startswith("x_") and layout != "ending":
            continue
        E.hook("perch_" + nm.replace(".", "p"), hc, tuple(p), yaw, "perch", size=0.12, crude_name=crude or "",
               source="crude" if crude else "natural")
    if layout == "ending":
        keep = {"L4_fire_escape_1", "L4_fire_escape_2", "R4_fire_escape_0", "R4_fire_escape_3", "L2_tank_top", "R1_roof_ac",
                "L0_parapet", "R3_parapet", "L3_parapet", "R2_parapet", "blade_HOTEL_top", "blade_PHO_top", "R0_roof_ac"}
        for nm, p, yaw in bperches:
            if nm in keep or nm.split("_")[1] == "winac":
                E.hook("perch_x_" + nm, hc, tuple(p), yaw, "perch", size=0.12, crude_name="", source="natural")
    # v4 (DevLog-020 W8): parody AI-era SF billboards, bus-shelter ad, robotaxi (own RNG: perch hooks unchanged)
    import billboards as BB
    cbb = E.coll("ENV_billboards", top)
    ctx = E.coll("ENV_robotaxi", top)
    BB.build(cbb, cl, hc, ctx, layout)
    # opening: hologram tree space + projector (own collection so S can instance it with an offset to any tree base)
    tree_zones = {}
    if layout == "opening":
        ctp = E.coll("ENV_tree_projector", top)
        pr = SK.holo_projector(ctp, tuple(TREE_C), "holo_projector")
        E.hook("tree_base", hc, tuple(pr["top"]), PI / 2, "landmark", note="crude TREE_C; hologram tree trunk origin")
        clr = E.empty("CLEAR_tree_zone", hc, (0.0, 16.8, 2.3), (0, 0, 0), 1.0, "CUBE")
        clr.scale = (1.3, 3.7, 2.3)
        clr["note"] = "keep free: |x| < 1.3, 13.1 < y < 20.5, z < 4.6 (S1 camera + tree + subject path)"
        tree_zones = tree_alternatives(hc, top)
    # ending: satellite (v2: 1.6x, on a close pass over the street that reads from the S6 roof camera)
    if layout == "ending":
        csat = E.coll("ENV_satellite", top)
        sat = SK.satellite(csat, tuple(SAT_PASS[0]), "satellite", scale=SAT_SCALE, yaw=0.3)
        E.hook("satellite", hc, tuple(SAT_PASS[0]), 0.0, "landmark", pass_end=list(SAT_PASS[1]), crude_path=list(SAT0) + list(SAT1),
               scale=SAT_SCALE, note="v2 close pass: animate the ENV_satellite instancer by (pass_end - location) over the shot")
        E.hook("satellite_pass_end", hc, tuple(SAT_PASS[1]), 0.0, "landmark")
        E.hook("satellite_dish", hc, tuple(sat["dish"]), 0.0, "camera", rot=(0, PI / 2, 0), note="at the pass start")
        E.hook("satellite_far", hc, tuple(SAT0), 0.0, "landmark", note="v1 far position (reads ~20 px at 540 px); kit prop scale 1")
    # atmosphere
    vfx.sky_dome("ENV_sky_dome", (0, 25, 0), 1200.0, csky, stars=1.0)   # v3: encloses the SF backdrop (camera clip_end >= 1300)
    vfx.night_sky_world(scn)
    vfx.world_haze(scn, density=0.0035, color=(0.55, 0.6, 0.78), anisotropy=0.3)
    vfx.haze_cards("street_haze", (0.0, 27.0, 4.0), (12.0, 18.0, 8.0), n=4, color=(0.5, 0.55, 0.75), density=0.035,
                   normal_axis="Y", collection=cv, seed=3.0)
    E.light("moon", "SUN", (0, 20, 30), 0.25, (0.6, 0.7, 0.95), cl, rot=(0.55, 0.15, 2.6))
    E.light("street_fill", "AREA", (0, 22, 11), 160, (0.55, 0.62, 0.85), cl, size=12, rot=(0, 0, 0))
    vfx.setup_eevee(scn, "draft", view_transform="AgX", look="AgX - Punchy")
    scn.render.fps = 30
    # report
    objs = [o for o in top.all_objects]
    tris = E.count_tris(objs)
    print("LAYOUT", layout, "objects", len(objs), "tris", tris, "hooks", len(E.HOOKS))
    E.save_blend(out_blend)
    d = {"asset_id": "street_%s" % layout, "date": "2026-10-10", "units": "m", "author": "workstream E (DevLog-013)",
         "blend": E.rel(out_blend), "root_collection": top.name,
         "frame": "world = build_crude_film.py frame: DC facade at y = 13 (wall 12.94..13.06), street along +y, road |x| < 3.5, "
                  "sidewalk top z = 0.12, building fronts |x| = 6; hall side y < 12.94 empty",
         "hook_convention": "HOOK_*: location = feet origin (perch) or landmark point; facing = local +X (yaw_deg about Z)",
         "objects": len(objs), "triangles": tris, "hooks": E.hooks_json(), "tree_zones": tree_zones,
         "notes": ["perch hooks named after the crude street_perches() / FLYER_PERCHES keep their coordinates "
                   "('.' -> 'p' in names; crude_name holds the original)",
                   "wire perches sit 1 cm above the wire centre (wire radius 8 mm), as in the crude film",
                   "EXT_R_bird_flap_vent is R's ASSET_bird_flap_vent appended under HOOK_bird_flap_vent (not rebuilt)",
                   "sky: ENV_sky_dome (radius 1200 m, camera clip_end >= 1300) and the world WORLD_night_street; world volume haze "
                   "density 0.0035"]}
    E.write_json(os.path.splitext(out_blend)[0] + ".json", d)
    return scn


def build_kit(out_blend):
    """Each prop at the origin in its own ASSET_env_<prop> collection (for reuse elsewhere)."""
    E.reset()
    E.HOOKS.clear()
    rng = random.Random(7)
    items = []

    def asset(name):
        cc = E.coll("ASSET_env_" + name)
        items.append(name)
        return cc
    c = asset("lamp_post")
    SK.lamp_post(c, (0, 0, 0), 0.0, "lamp", LAMP_H)
    c = asset("utility_pole")
    SK.utility_pole(c, (0, 0, 0), 0.0, "pole", transformer=True)
    c = asset("railing_5m")
    SK.railing(c, 0.0, 0.0, 5.0, 1.0)
    c = asset("ac_wall")
    SK.ac_wall(c, (0, 0, 0), 0.0)
    c = asset("bin")
    SK.bin_prop(c, (0, 0, 0))
    c = asset("hydrant")
    SK.hydrant(c, (0, 0, 0))
    c = asset("scooter")
    SK.scooter(c, (0, 0, 0), 0.0)
    c = asset("hanging_sign")
    SK.hanging_sign(c, (0, 0, 0), 0.0, ["RAMEN", "24H"])
    c = asset("cctv_pole")
    SK.cctv(c, (0, 0, 3.45), 0.0, "pole")
    c = asset("cctv_wall")
    SK.cctv(c, (0, 0, 0), 0.0, "wall")
    c = asset("satellite")
    SK.satellite(c, (0, 0, 0))
    c = asset("holo_projector")
    SK.holo_projector(c, (0, 0, 0))
    c = asset("vending_machine")
    SK.vending_machine(c, (0, 0, 0), 0.0)
    c = asset("building_9m")
    SK.building(c, (0, 0, 0), 0.0, 3.6, 3.0, 9.0, E.M("plaster_a"), rng, "building", fire_escape=(-1.6, 0.1),
                roof=[("ac", 0.6, -1.6, 0.0), ("yagi", -1.0, -2.3, 2.4)])
    c = asset("neon_sign")
    SK.neon_sign(c, (0, 0, 0), 0.0, "RAMEN 24H", "neon_orange", 0.4)
    E.save_blend(out_blend)
    E.write_json(os.path.splitext(out_blend)[0] + ".json",
                 {"asset_id": "street_kit", "date": "2026-10-10", "units": "m", "collections": ["ASSET_env_" + n for n in items],
                  "note": "each prop at the origin, local frames as documented in scripts/env/street_kit.py"})


if __name__ == "__main__":
    a = E.argv_after_dashes()
    layout = a[0] if a else "opening"
    out = a[1] if len(a) > 1 else os.path.join(E.OUT_ASSETS, "street_%s.blend" % layout)
    if layout == "kit":
        build_kit(out)
    else:
        build(layout, out)
    print("SAVED", out)
