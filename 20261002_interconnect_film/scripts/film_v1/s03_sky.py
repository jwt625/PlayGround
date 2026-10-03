"""S3 exterior day/night time-lapse (v1.1): world sky with sun and moon discs, stars, dusk glow; sun and moon lamps; skyline.

The back wall of the conference room (+Y) gets its sill and blinds hidden so the exterior shows through three tall windows.
During the '+3 MONTHS' card window the sky runs 3 full day cycles (sunrise, noon, sunset, moon, stars); outside the window
it holds late morning (cycle phase 0.25), so the cut-in and cut-out are continuous.  Deterministic: every animated value is
keyed per frame from a closed-form function of time.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

import blender_lib as L
from asm import F

PHASE_HOLD = 0.25   # noon-ish morning (sun elevation about 27 degrees, azimuth toward the windows)


def smoothstep(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def sky_state(t, t0, t1, cycles=3.0):
    """Closed-form cycle state at scene time t."""
    w = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    ph = PHASE_HOLD + cycles * (w * w * (3 - 2 * w) * 0.35 + w * 0.65)   # eased start/stop, mostly linear
    phi = 2 * math.pi * ph
    sun = np.array([0.9 * math.cos(phi), 1.0, 0.5 * math.sin(phi)])
    sun = sun / np.linalg.norm(sun)
    moon = np.array([-0.9 * math.cos(phi), 1.0, -0.5 * math.sin(phi)])
    moon = moon / np.linalg.norm(moon)
    day = smoothstep(-0.06, 0.28, sun[2])
    glow = math.exp(-((sun[2] - 0.02) / 0.14) ** 2) * (1.0 if sun[2] > -0.25 else 0.0)
    stars = 1.0 - smoothstep(-0.12, 0.10, sun[2])
    return dict(sun=sun, moon=moon, day=day, glow=glow, stars=stars, phase=ph)


def _val(nodes, name, v):
    n = nodes.new("ShaderNodeValue")
    n.name = n.label = name
    n.outputs[0].default_value = v
    return n


def _math(nodes, op, a=None, b=None, clamp=False):
    n = nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    if a is not None and not isinstance(a, tuple):
        n.inputs[0].default_value = a
    if b is not None and not isinstance(b, tuple):
        n.inputs[1].default_value = b
    return n


def build_world(scn):
    w = bpy.data.worlds.new("WORLD_s03_cycle")
    w.use_nodes = True
    nt = w.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    tc = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tc.outputs["Generated"], sep.inputs[0])
    V = {k: _val(nodes, k, 0.0) for k in ("DAY", "GLOW", "STARS", "DISC", "SX", "SY", "SZ", "MX", "MY", "MZ")}
    sun = nodes.new("ShaderNodeCombineXYZ")
    moon = nodes.new("ShaderNodeCombineXYZ")
    for i, k in enumerate("XYZ"):
        links.new(V["S" + k].outputs[0], sun.inputs[i])
        links.new(V["M" + k].outputs[0], moon.inputs[i])

    def vdot(a_out, b_out):
        d = nodes.new("ShaderNodeVectorMath")
        d.operation = "DOT_PRODUCT"
        links.new(a_out, d.inputs[0])
        links.new(b_out, d.inputs[1])
        return d.outputs["Value"]

    ds = vdot(tc.outputs["Generated"], sun.outputs[0])
    dm = vdot(tc.outputs["Generated"], moon.outputs[0])

    def maprange(inp, a, b, c=0.0, d=1.0):
        m = nodes.new("ShaderNodeMapRange")
        m.clamp = True
        m.inputs["From Min"].default_value = a
        m.inputs["From Max"].default_value = b
        m.inputs["To Min"].default_value = c
        m.inputs["To Max"].default_value = d
        links.new(inp, m.inputs["Value"])
        return m.outputs[0]

    def rgb(c):
        n = nodes.new("ShaderNodeRGB")
        n.outputs[0].default_value = (*c, 1)
        return n.outputs[0]

    def mix(fac, a, b, blend="MIX"):
        m = nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        m.blend_type = blend
        if isinstance(fac, float):
            m.inputs[0].default_value = fac
        else:
            links.new(fac, m.inputs[0])
        links.new(a, m.inputs[6])
        links.new(b, m.inputs[7])
        return m.outputs[2]

    def scale_col(col, fac_out):
        m = nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        m.blend_type = "MULTIPLY"
        m.inputs[0].default_value = 1.0
        links.new(col, m.inputs[6])
        c = nodes.new("ShaderNodeCombineColor")
        for i in range(3):
            links.new(fac_out, c.inputs[i])
        links.new(c.outputs[0], m.inputs[7])
        return m.outputs[2]

    # vertical gradient
    zc = _math(nodes, "POWER", None, 0.55, clamp=True)
    zpos = _math(nodes, "MAXIMUM", None, 0.0)
    links.new(sep.outputs["Z"], zpos.inputs[0])
    zpos.inputs[1].default_value = 0.0
    links.new(zpos.outputs[0], zc.inputs[0])
    day_col = mix(zc.outputs[0], rgb((0.80, 0.90, 1.0)), rgb((0.28, 0.52, 0.95)))
    night_col = mix(zc.outputs[0], rgb((0.05, 0.07, 0.16)), rgb((0.008, 0.012, 0.05)))
    base = mix(V["DAY"].outputs[0], night_col, day_col)
    ground_col = mix(V["DAY"].outputs[0], rgb((0.02, 0.025, 0.04)), rgb((0.22, 0.27, 0.45)))
    base = mix(maprange(sep.outputs["Z"], 0.0, -0.02), base, ground_col)
    # dusk / dawn glow
    gl = _math(nodes, "POWER", None, 3.0)
    links.new(maprange(ds, 0.2, 1.0), gl.inputs[0])
    gmul = _math(nodes, "MULTIPLY")
    links.new(gl.outputs[0], gmul.inputs[0])
    links.new(V["GLOW"].outputs[0], gmul.inputs[1])
    gmul2 = _math(nodes, "MULTIPLY", None, 1.6)
    links.new(gmul.outputs[0], gmul2.inputs[0])
    glow_col = scale_col(rgb((1.0, 0.50, 0.22)), gmul2.outputs[0])
    col = mix(1.0, base, glow_col, "ADD")
    # discs
    above = maprange(sep.outputs["Z"], -0.03, 0.0)
    sd = _math(nodes, "MULTIPLY", None, 7.0)
    links.new(maprange(ds, 0.9925, 0.9955), sd.inputs[0])
    sdd = _math(nodes, "MULTIPLY")
    links.new(sd.outputs[0], sdd.inputs[0])
    links.new(V["DISC"].outputs[0], sdd.inputs[1])
    sun_col = scale_col(rgb((1.0, 0.90, 0.62)), sdd.outputs[0])
    col = mix(1.0, col, sun_col, "ADD")
    md = _math(nodes, "MULTIPLY", None, 3.5)
    links.new(maprange(dm, 0.9965, 0.9980), md.inputs[0])
    mdd = _math(nodes, "MULTIPLY")
    links.new(md.outputs[0], mdd.inputs[0])
    links.new(V["DISC"].outputs[0], mdd.inputs[1])
    moon_col = scale_col(rgb((0.85, 0.90, 1.0)), mdd.outputs[0])
    col = mix(1.0, col, moon_col, "ADD")
    # stars
    vor = nodes.new("ShaderNodeTexVoronoi")
    vor.voronoi_dimensions = "3D"
    vor.feature = "F1"
    vor.inputs["Scale"].default_value = 22.0
    links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    star = _math(nodes, "SUBTRACT", 1.0, None, clamp=True)
    links.new(maprange(vor.outputs["Distance"], 0.05, 0.14), star.inputs[1])
    rsep = nodes.new("ShaderNodeSeparateColor")
    links.new(vor.outputs["Color"], rsep.inputs[0])
    keep = maprange(rsep.outputs[0], 0.45, 0.55)
    s2 = _math(nodes, "MULTIPLY")
    links.new(star.outputs[0], s2.inputs[0])
    links.new(keep, s2.inputs[1])
    s3 = _math(nodes, "MULTIPLY")
    links.new(s2.outputs[0], s3.inputs[0])
    links.new(V["STARS"].outputs[0], s3.inputs[1])
    s4 = _math(nodes, "MULTIPLY", None, 6.0)
    links.new(s3.outputs[0], s4.inputs[0])
    s5 = _math(nodes, "MULTIPLY")
    links.new(s4.outputs[0], s5.inputs[0])
    links.new(above, s5.inputs[1])
    star_col = scale_col(rgb((1.0, 1.0, 0.92)), s5.outputs[0])
    col = mix(1.0, col, star_col, "ADD")
    links.new(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    links.new(bg.outputs[0], out.inputs[0])
    scn.world = w
    return w, V


def lamp(name, kind="SUN", energy=3.0, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    ld.angle = 0.02
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    return o


def key_cycle(world_nodes, sun_l, moon_l, t0, t1, cycles=3.0, pre=0.0):
    """Key sky values, lamp directions, energies and colours every frame from t0-0.1 to t1+0.1 and hold outside."""
    w = world_nodes
    names = ("DAY", "GLOW", "STARS", "DISC", "SX", "SY", "SZ", "MX", "MY", "MZ")
    f_a, f_b = F(t0) - 3, F(t1) + 3
    frames = [1] + list(range(F(t0) - 12, F(t1) + 13)) + [301]
    for f in frames:
        t = (f - 1) / 30.0
        st = sky_state(t, t0, t1, cycles)
        env = smoothstep(t0 - 0.30, t0 - 0.05, t) * (1.0 - smoothstep(t1 + 0.05, t1 + 0.30, t))   # discs only during the card window
        vals = dict(DAY=st["day"], GLOW=st["glow"], STARS=st["stars"], DISC=env, SX=st["sun"][0], SY=st["sun"][1], SZ=st["sun"][2],
                    MX=st["moon"][0], MY=st["moon"][1], MZ=st["moon"][2])
        for k in names:
            n = w[k]
            n.outputs[0].default_value = vals[k]
            n.outputs[0].keyframe_insert("default_value", frame=f)
        for lo, d, up_energy, col in ((sun_l, st["sun"], 4.0 * st["day"], None), (moon_l, st["moon"], 0.45 * (1 - st["day"]), None)):
            q = Vector(-d).to_track_quat("-Z", "Y")
            lo.rotation_euler = q.to_euler()
            lo.keyframe_insert("rotation_euler", frame=f)
            lo.data.energy = up_energy * (1.0 if d[2] > -0.02 else 0.0)
            lo.data.keyframe_insert("energy", frame=f)
        warm = st["glow"]
        sun_l.data.color = (1.0, 0.92 - 0.35 * warm, 0.80 - 0.5 * warm)
        sun_l.data.keyframe_insert("color", frame=f)
        moon_l.data.color = (0.65, 0.75, 1.0)
        moon_l.data.keyframe_insert("color", frame=f)
    for ob in (sun_l, moon_l, sun_l.data, moon_l.data):
        ad = ob.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    wad = bpy.context.scene.world.node_tree.animation_data
    if wad and wad.action:
        for fc in wad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def exterior(seed=7):
    """Ground plane and a clay skyline beyond the +Y wall (distance 14-60 m) plus a few trees."""
    rng = np.random.default_rng(seed)
    coll = bpy.data.collections.new("EXTERIOR_s03")
    bpy.context.scene.collection.children.link(coll)

    def box(name, cx, cy, sx, sy, sz, mat):
        me = bpy.data.meshes.new(name)
        vs = [(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0), (-sx, -sy, 2 * sz), (sx, -sy, 2 * sz), (sx, sy, 2 * sz), (-sx, sy, 2 * sz)]
        me.from_pydata(vs, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        me.update()
        o = bpy.data.objects.new(name, me)
        coll.objects.link(o)
        o.data.materials.append(mat)
        o.location = (cx, cy, -0.02)
        return o

    def mat(name, col, rough=0.9):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*col, 1)
        m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = rough
        return m
    grass = mat("MAT_s03_grass", (0.30, 0.50, 0.25))
    g = box("ext_ground", 0, 40, 80, 40, 0.01, grass)
    g.location = (0, 43.2, -0.03)
    bmat = mat("MAT_s03_bldg", (0.60, 0.64, 0.72))
    # emissive window grid at night: checker on object coords multiplied by a keyed strength is not needed; flat clay silhouettes
    x = -48.0
    while x < 48.0:
        w = 2.5 + 4.0 * rng.random()
        h = 4.0 + 16.0 * rng.random() ** 1.5
        y = 14.0 + 30.0 * rng.random()
        box("ext_bldg", x + w, y, w, 2.5 + 3 * rng.random(), h / 2, bmat)
        x += 2 * w + 0.5 + 2.0 * rng.random()
    tmat = mat("MAT_s03_trunk", (0.40, 0.28, 0.18))
    lmat = mat("MAT_s03_leaf", (0.22, 0.46, 0.20))
    import bmesh
    for i in range(9):
        tx = -14 + 3.4 * i + rng.random()
        ty = 6.5 + 4.0 * rng.random()
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        me = bpy.data.meshes.new("tree_leaf")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new("ext_tree", me)
        coll.objects.link(o)
        o.data.materials.append(lmat)
        r = 1.1 + 0.6 * rng.random()
        o.location = (tx, ty, 2.6 + r * 0.3)
        o.scale = (r, r, r * 1.2)
        for p in me.polygons:
            p.use_smooth = True
        tr = box("ext_trunk", tx, ty, 0.12, 0.12, 1.4, tmat)
    return coll
