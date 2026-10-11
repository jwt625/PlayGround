"""Atmosphere (workstream E, DevLog-013): night sky (world shader or dome mesh), world-volume haze, haze cards, fake light
cones. Blender 4.2 EEVEE Next. Cheap options first: haze cards and light cones are additive surfaces (no volume cost);
world_haze is a true (global) volume, tuned with post.setup_eevee(volumetrics=...).
"""
import math

import bpy
from mathutils import Vector

from .nodes import NB, group_node, new_group, new_material, set_blended


def _rgba(c):
    return (*c[:3], 1.0) if len(c) == 3 else tuple(c)


# ==================================================================== night sky
def sky_group():
    """Direction-based night sky (input Vector = unit view direction, Z up): zenith/horizon gradient with warm city glow,
    stars (voronoi points, random brightness, slow twinkle), optional moon disc."""
    ng, b, gi, go = new_group("VFX_NightSky", [
        ("Vector", "NodeSocketVector", (0.0, 0.0, 1.0)),
        ("Zenith", "NodeSocketColor", (0.0015, 0.002, 0.006, 1.0)),
        ("Horizon", "NodeSocketColor", (0.012, 0.011, 0.022, 1.0)),
        ("City Glow", "NodeSocketColor", (0.05, 0.022, 0.03, 1.0)),
        ("Glow Height", "NodeSocketFloat", 0.12, 0.001, 1.0),
        ("Stars", "NodeSocketFloat", 1.0, 0.0, 10.0),
        ("Star Density", "NodeSocketFloat", 120.0, 1.0, 5000.0),
        ("Star Size", "NodeSocketFloat", 0.16, 0.0, 0.5),
        ("Twinkle", "NodeSocketFloat", 0.35, 0.0, 1.0),
        ("Moon", "NodeSocketFloat", 0.0, 0.0, 100.0),
        ("Moon Dir", "NodeSocketVector", (0.3, 0.8, 0.5)),
    ], [("Color", "NodeSocketColor")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    d = b.vmath("NORMALIZE", I["Vector"])
    z = b.sep(d)[2]
    zc = b.math("MAXIMUM", z, 0.0)
    grad = b.cmix(I["Horizon"], I["Zenith"], b.pow(zc, 0.45))
    glow = b.math("EXPONENT", b.mul(b.math("DIVIDE", zc, I["Glow Height"]), -1.0))
    col = b.cmix(grad, I["City Glow"], b.mul(glow, 0.9), "ADD")
    # stars
    vp = b.vmath("SCALE", d, scale=I["Star Density"])
    vn = b.node("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1")
    b.link(vp, vn.inputs["Vector"])
    vn.inputs["Scale"].default_value = 1.0
    vn.inputs["Randomness"].default_value = 1.0
    dist, cell = vn.outputs["Distance"], vn.outputs["Color"]
    r = b.sep(cell)
    keep = b.math("GREATER_THAN", r[0], 0.55)                      # ~45 % of cells hold a star
    bright = b.pow(b.mrange(r[1], 0.0, 1.0, 0.0, 1.0), 3.0)
    size = b.mul(I["Star Size"], b.add(0.5, bright))
    star = b.mul(b.smooth(b.sub(size, dist), 0.0, b.mul(size, 0.8)), keep)
    tw = b.noise(w=b.add(b.mul(t, 1.7), b.mul(r[2], 100.0)), scale=1.0, detail=0.0, dims="1D")
    tw = b.sub(1.0, b.mul(I["Twinkle"], b.smooth(tw, 0.35, 0.65)))
    horizon_fade = b.smooth(z, 0.02, 0.18)
    sk = b.mul(b.mul(b.mul(star, b.add(0.15, bright)), b.mul(tw, horizon_fade)), b.mul(I["Stars"], 1.6))
    col = b.cmix(col, b.comb(sk, sk, b.mul(sk, 1.08)), 1.0, "ADD")
    # moon
    md = b.vmath("NORMALIZE", I["Moon Dir"])
    cosang = b.vmath("DOT_PRODUCT", d, md)
    disc = b.smooth(cosang, 0.99993, 0.99996)
    halo = b.mul(b.pow(b.math("MAXIMUM", cosang, 0.0), 400.0), 0.02)
    mk = b.mul(b.add(disc, halo), I["Moon"])
    col = b.cmix(col, b.comb(mk, b.mul(mk, 0.97), b.mul(mk, 0.9)), 1.0, "ADD")
    b.link(col, go.inputs["Color"])
    return ng


def night_sky_world(scene=None, name="WORLD_night_street", strength=1.0, stars=1.0, moon=0.0, city_glow=(0.05, 0.022, 0.03),
                    zenith=(0.0015, 0.002, 0.006), horizon=(0.012, 0.011, 0.022)):
    """Set (and return) a world with the night sky. Very dark on purpose: lighting comes from the set lights."""
    w = bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    b = NB(nt)
    tc = b.node("ShaderNodeTexCoord")
    g = group_node(b, sky_group(), Stars=stars, Moon=moon, City_Glow=_rgba(city_glow), Zenith=_rgba(zenith),
                   Horizon=_rgba(horizon))
    b.link(tc.outputs["Generated"], g.inputs["Vector"])
    bg = b.node("ShaderNodeBackground")
    b.link(g.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    out = b.node("ShaderNodeOutputWorld")
    b.link(bg.outputs[0], out.inputs["Surface"])
    w["vfx_sky"] = True
    if scene is not None:
        scene.world = w
    return w


def sky_dome(name="ENV_sky_dome", center=(0, 25, 0), radius=240.0, collection=None, stars=1.0, moon=0.0, strength=1.0,
             city_glow=(0.05, 0.022, 0.03)):
    """Inward-facing emissive hemisphere-plus (sphere) with the night sky; for a single-scene film whose world must stay
    neutral (visibility-windowed like any set object). Camera clip_end must exceed radius + distance to center."""
    seg, rings = 48, 24
    v, f = [], []
    for j in range(rings + 1):
        th = -0.25 * math.pi + 0.75 * math.pi * j / rings       # from -45 deg (below horizon) to the zenith
        for i in range(seg):
            a = 2 * math.pi * i / seg
            v.append((radius * math.cos(th) * math.cos(a), radius * math.cos(th) * math.sin(a), radius * math.sin(th)))
    for j in range(rings):
        for i in range(seg):
            k = (i + 1) % seg
            f.append((j * seg + i, (j + 1) * seg + i, (j + 1) * seg + k, j * seg + k))   # inward winding
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], f)
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(o)
    o.location = center
    m, b, out = new_material("VFX_sky_dome")
    g = group_node(b, sky_group(), Stars=stars, Moon=moon, City_Glow=_rgba(city_glow))
    b.link(b.texcoord("Object"), g.inputs["Vector"])
    b.link(b.emission(g.outputs["Color"], strength), out.inputs["Surface"])
    me.materials.append(m)
    o.visible_shadow = False
    for attr in ("visible_diffuse", "visible_glossy", "visible_volume_scatter"):
        try:
            setattr(o, attr, attr == "visible_glossy")
        except AttributeError:
            pass
    return o


# ==================================================================== world-volume haze
def world_haze(scene, density=0.004, color=(0.55, 0.6, 0.75), anisotropy=0.25, absorption=0.0):
    """Global participating medium in the world volume (EEVEE Next fills the view frustum up to volumetric_end).
    Density per metre; 0.003-0.008 reads as light street haze around lamps. Returns the volume node."""
    w = scene.world
    if w is None:
        w = bpy.data.worlds.new("WORLD_haze")
        scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")
    vol = nt.nodes.get("VFX_world_haze") or nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.name = "VFX_world_haze"
    vol.inputs["Color"].default_value = _rgba(color)
    vol.inputs["Density"].default_value = density
    vol.inputs["Anisotropy"].default_value = anisotropy
    vol.inputs["Absorption Color"].default_value = (0, 0, 0, 1) if absorption == 0 else _rgba((1 - absorption,) * 3)
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    return vol


# ==================================================================== haze cards (cheap)
def haze_card_group():
    ng, b, gi, go = new_group("VFX_HazeCard", [
        ("Color", "NodeSocketColor", (0.5, 0.55, 0.7, 1.0)),
        ("Density", "NodeSocketFloat", 0.05, 0.0, 10.0),
        ("Noise Scale", "NodeSocketFloat", 0.6, 0.0, 100.0),
        ("Drift", "NodeSocketFloat", 0.08, -10.0, 10.0),
        ("Height Falloff", "NodeSocketFloat", 1.2, 0.0, 10.0),
        ("Edge", "NodeSocketFloat", 0.25, 0.001, 0.5),
        ("Seed", "NodeSocketFloat", 0.0),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    uv = b.sep(b.texcoord("UV"))
    u, v = uv[0], uv[1]
    e = I["Edge"]
    edge = b.mul(b.mul(b.smooth(u, 0.0, e), b.smooth(b.one_minus(u), 0.0, e)),
                 b.mul(b.smooth(v, 0.0, b.mul(e, 0.5)), b.smooth(b.one_minus(v), 0.0, e)))
    height = b.math("EXPONENT", b.mul(b.mul(v, I["Height Falloff"]), -1.0))
    wp = b.vmath("ADD", b.geometry("Position"), b.comb(b.mul(t, I["Drift"]), b.add(I["Seed"], b.mul(b.objinfo("Random"), 50.0)),
                                                        b.mul(t, b.mul(I["Drift"], 0.3))))
    n = b.noise(b.vmath("SCALE", wp, scale=I["Noise Scale"]), scale=1.0, detail=3.0, rough=0.55)
    cloud = b.smooth(n, 0.35, 0.75)
    a = b.mul(b.mul(edge, height), b.mul(cloud, b.mul(I["Density"], b.objinfo("Alpha"))))
    # view-facing fade: cards seen edge-on vanish (no visible planes)
    face = b.one_minus(b.pow(b.layer_weight(0.5, "Facing"), 2.0))
    sh = b.add_shader(b.transparent(), b.emission(I["Color"], b.mul(a, face)))
    b.link(sh, go.inputs["Shader"])
    return ng


def haze_cards(name, center, size, n=6, color=(0.5, 0.55, 0.7), density=0.05, normal_axis="Y", noise_scale=0.6, drift=0.08,
               height_falloff=1.2, collection=None, seed=0.0):
    """n additive mist planes inside a box (center, size=(sx, sy, sz)), stacked along normal_axis ('X' or 'Y'), facing it.
    Pick the axis closest to the camera view direction. density ~0.03-0.1 per card. Returns the list of objects."""
    mat = bpy.data.materials.get("VFX_haze_card_%s" % name)
    if mat is None:
        mat, b, out = new_material("VFX_haze_card_%s" % name)
        g = group_node(b, haze_card_group(), Color=_rgba(color), Density=density, Noise_Scale=noise_scale, Drift=drift,
                       Height_Falloff=height_falloff, Seed=seed)
        b.link(g.outputs["Shader"], out.inputs["Surface"])
        set_blended(mat)
    cx, cy, cz = center
    sx, sy, sz = size
    objs = []
    for k in range(n):
        u = (k + 0.5) / n - 0.5
        if normal_axis == "Y":
            w, y0 = sx, cy + u * sy
            verts = [(cx - w / 2, y0, cz - sz / 2), (cx + w / 2, y0, cz - sz / 2), (cx + w / 2, y0, cz + sz / 2), (cx - w / 2, y0, cz + sz / 2)]
        else:
            w, x0 = sy, cx + u * sx
            verts = [(x0, cy - w / 2, cz - sz / 2), (x0, cy + w / 2, cz - sz / 2), (x0, cy + w / 2, cz + sz / 2), (x0, cy - w / 2, cz + sz / 2)]
        me = bpy.data.meshes.new("%s_%d" % (name, k))
        me.from_pydata(verts, [], [(0, 1, 2, 3)])
        uvl = me.uv_layers.new(name="UVMap")
        for li, uvv in zip(range(4), ((0, 0), (1, 0), (1, 1), (0, 1))):
            uvl.data[li].uv = uvv
        me.materials.append(mat)
        o = bpy.data.objects.new("%s_%d" % (name, k), me)
        (collection or bpy.context.scene.collection).objects.link(o)
        o.visible_shadow = False
        objs.append(o)
    return objs


# ==================================================================== fake light cones (lamp shafts)
def light_cone_group():
    ng, b, gi, go = new_group("VFX_LightCone", [
        ("Color", "NodeSocketColor", (1.0, 0.78, 0.55, 1.0)),
        ("Strength", "NodeSocketFloat", 0.25, 0.0, 100.0),
        ("Length", "NodeSocketFloat", 4.0, 0.001, 1000.0),
        ("Dust", "NodeSocketFloat", 0.35, 0.0, 1.0),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    zloc = b.mul(b.sep(b.texcoord("Object"))[2], -1.0)               # distance from the apex along local -Z
    along = b.pow(b.one_minus(b.math("DIVIDE", zloc, I["Length"])), 1.6)
    near = b.smooth(zloc, 0.0, 0.15)
    core = b.pow(b.one_minus(b.layer_weight(0.5, "Facing")), 2.5)
    wp = b.vmath("ADD", b.geometry("Position"), b.comb(b.mul(t, 0.05), 0.0, b.mul(t, -0.03)))
    dust = b.lerp(1.0, b.mul(b.noise(b.vmath("SCALE", wp, scale=3.0), scale=1.0, detail=2.0), 1.6), I["Dust"])
    k = b.mul(b.mul(b.mul(along, near), b.mul(core, dust)), b.mul(I["Strength"], b.objinfo("Alpha")))
    b.link(b.add_shader(b.transparent(), b.emission(I["Color"], k)), go.inputs["Shader"])
    return ng


def light_cone(name, apex, direction=(0, 0, -1), length=4.0, angle_deg=55.0, color=(1.0, 0.78, 0.55), strength=0.25,
               collection=None, seg=24, dust=0.35):
    """Open cone (apex at `apex`, axis along `direction`, full angle angle_deg) with an additive falloff shader: a fake
    volumetric lamp shaft for a few hundred triangles. Returns the object."""
    r = length * math.tan(math.radians(angle_deg) / 2)
    v = [(0.0, 0.0, 0.0)]
    rings = 6
    for j in range(1, rings + 1):
        zz = -length * j / rings
        rr = r * j / rings
        for i in range(seg):
            a = 2 * math.pi * i / seg
            v.append((rr * math.cos(a), rr * math.sin(a), zz))
    f = [(0, 1 + (i + 1) % seg, 1 + i) for i in range(seg)]
    for j in range(rings - 1):
        a0, a1 = 1 + j * seg, 1 + (j + 1) * seg
        for i in range(seg):
            k = (i + 1) % seg
            f.append((a0 + i, a0 + k, a1 + k, a1 + i))
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], f)
    for p in me.polygons:
        p.use_smooth = True
    key = "VFX_light_cone_%02x%02x%02x_%.2f_%.2f" % (tuple(int(255 * c) for c in color[:3]) + (strength, length))
    mat = bpy.data.materials.get(key)
    if mat is None:
        mat, b, out = new_material(key)
        g = group_node(b, light_cone_group(), Color=_rgba(color), Strength=strength, Length=length, Dust=dust)
        b.link(g.outputs["Shader"], out.inputs["Surface"])
        set_blended(mat)
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(o)
    o.location = apex
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = (-Vector(direction)).normalized().to_track_quat("Z", "Y")
    o.visible_shadow = False
    return o
