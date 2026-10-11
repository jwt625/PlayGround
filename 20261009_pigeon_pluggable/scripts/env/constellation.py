"""Generic LEO constellation "train" (workstream E, DevLog-013 section 8; owner item F6, DevLog-015).

Flat-panel broadband satellites with a single solar wing, flying in a line shortly after deployment (the familiar
"train" of dots). Generic design, no logos or markings. Sizes stylised from the publicly described flat-panel class
(bus about 2.8 x 1.4 m, one long wing); the default scale is enlarged for readability at 540 px.

    import constellation as CN
    sats = CN.constellation_train(coll, n=10)                               # v3 default scale 0.5       # root = sats[0].parent
    CN.constellation_pass(sats[0].parent, CN.S6_PASS_V3[0], CN.S6_PASS_V3[1], f0, f1)
    vfx.beams.beam_track(beam, eye, sats[2:6], f0, f1)                       # objects as targets
"""
import math
import random

import bpy
from mathutils import Vector

import env_lib as E
from env_lib import MB

# Pass across S's S6 roof-camera view ((-4.3, 18.3, 8.3) -> (-2, 26, 16), lens 22): perpendicular to the view axis about
# 16 m out, left -> right on screen, so satellites enter the frame one after another.
S6_PASS = (Vector((-16.0, 33.0, 25.5)), Vector((14.0, 24.0, 27.0)))
# v3 (owner, DevLog-018 V4): satellites 2x smaller (scale 0.5) and 2x faster. S6_PASS_V3 doubles the length of S's v2.1 pass
# ((-15, 27, 13.5) -> (13, 24, 15)) about its centre, so over the same window (S6 3.6-8.2 s) the train moves 2x faster.
V3_SCALE = 0.5
S6_PASS_V3 = (Vector((-29.0, 28.5, 12.75)), Vector((27.0, 22.5, 15.75)))


def scale_pass(p_start, p_end, factor):
    """Lengthen (factor > 1) or shorten a pass about its midpoint: same window -> `factor` x the speed."""
    a, b = Vector(p_start), Vector(p_end)
    mid = (a + b) / 2
    return mid + (a - mid) * factor, mid + (b - mid) * factor


def _sat_mesh(name="CN_sat_mesh"):
    me = bpy.data.meshes.get(name)
    if me:
        return me
    mb = MB()
    mb.box((0, 0, 0), (2.8, 1.4, 0.16), m=0)                          # flat bus (local +X along the train)
    for i in range(6):                                                # phased-array tiles on the nadir face
        for j in range(3):
            mb.box((-1.15 + 0.46 * i, -0.46 + 0.46 * j, -0.085), (0.4, 0.4, 0.012), m=1)
    mb.box((1.2, 0, 0.12), (0.18, 0.18, 0.1), m=2)
    mb.rod((0.0, 0.7, 0.0), (0.0, 1.5, 0.25), 0.04, 8, m=2)            # boom to the single wing (sideways, +Y)
    with mb.at((0.0, 1.5, 0.25), rx=0.35):
        mb.box((0, 4.2, 0), (1.3, 8.4, 0.04), m=2)
        mb.quad((-0.64, 0.0, 0.022), (0.64, 0.0, 0.022), (0.64, 8.4, 0.022), (-0.64, 8.4, 0.022), m=3)
        mb.quad((-0.64, 8.4, -0.022), (0.64, 8.4, -0.022), (0.64, 0.0, -0.022), (-0.64, 0.0, -0.022), m=3)
    mb.ellipsoid((0.0, 0.0, 0.1), (0.3, 0.3, 0.08), 12, 4, m=4)          # sunlit glint (reads as the classic bright dot)
    me = mb.mesh(name)
    # "sunlit at night" look (the train is visible because it is still in sunlight): faintly emissive bus and wing
    me.materials.append(E.principled("CN_bus", (0.55, 0.56, 0.6), 0.4, 0.35, (0.75, 0.78, 0.85), 0.5))
    me.materials.append(E.M("lens_black"))
    me.materials.append(E.principled("CN_frame", (0.5, 0.5, 0.52), 0.8, 0.3, (0.7, 0.72, 0.78), 0.3))
    me.materials.append(E.principled("CN_wing", (0.04, 0.07, 0.2), 0.3, 0.15, (0.25, 0.35, 0.8), 0.45))
    me.materials.append(E.principled("CN_glint", (1.0, 0.97, 0.9), 0.0, 0.2, (1.0, 0.97, 0.9), 25.0))
    return me


def constellation_train(c, n=10, spacing=4.5, scale=V3_SCALE, name="cn_train", seed=7, jitter=0.0):
    """n satellites in a line along the root's +X (spacing in metres, before `scale`). Default jitter 0: identical attitude
    (bodies and solar arrays aligned, owner 2026-10-10); jitter > 0 adds seeded position/attitude scatter. Returns the satellite objects (sats[0].parent is the root empty)."""
    rng = random.Random(seed)
    root = E.empty(name + "_root", c, (0, 0, 0), (0, 0, 0), 1.0)
    root.scale = (scale,) * 3
    me = _sat_mesh()
    sats = []
    for i in range(n):
        o = bpy.data.objects.new("%s_sat_%02d" % (name, i), me)
        c.objects.link(o)
        o.parent = root
        o.location = (-i * spacing, rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter) * 0.6)
        k = jitter / 0.25
        o.rotation_euler = (k * rng.uniform(-0.15, 0.15), k * rng.uniform(-0.1, 0.1), k * rng.uniform(-0.2, 0.2))
        sats.append(o)
    return sats


def constellation_pass(root, p_start, p_end, f0, f1, roll=0.0):
    """Fly the train root from p_start to p_end (linear, constant speed) between f0 and f1; the train axis (+X) is
    aligned with the direction of travel. Satellites trail behind the root (lead satellite = sats[0])."""
    p0, p1 = Vector(p_start), Vector(p_end)
    d = (p1 - p0).normalized()
    yaw = math.atan2(d.y, d.x)
    pitch = -math.asin(max(-1.0, min(1.0, d.z)))
    root.rotation_euler = (roll, pitch, yaw)
    for f, p in ((f0, p0), (f1, p1)):
        root.location = p
        root.keyframe_insert("location", frame=f)
    for fc in root.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.extrapolation = "LINEAR"


def sat_position(root_start, root_end, f0, f1, index, spacing=4.5, scale=V3_SCALE):
    """Analytic world position of satellite `index` (ignores jitter) as a callable f(frame) -> Vector, for beam helpers
    when the train is instanced (no object access)."""
    p0, p1 = Vector(root_start), Vector(root_end)
    d = (p1 - p0).normalized()

    def pos(f):
        u = (f - f0) / max(1, f1 - f0)
        return p0.lerp(p1, u) - d * (index * spacing * scale)
    return pos
