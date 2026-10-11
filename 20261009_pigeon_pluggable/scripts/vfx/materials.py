"""VFX materials (workstream E, DevLog-013): hologram, beam/link, neon, wet ground, puddle, eye glow. Blender 4.2 EEVEE Next.

Every effect is a shader node group `VFX_<Name>` (created once per .blend, shared) plus a material factory that wires the
group with per-material defaults. Per-object control uses the object colour alpha (Object Info > Alpha): one material can
drive many objects with different fades / eye powers, keyed with `key_obj_alpha`. Time comes from a `frame/30` driver.
"""
import math

import bpy
from mathutils import Vector

from .nodes import NB, group_node, new_group, new_material, set_blended

TX_RED = (1.0, 0.08, 0.05)        # crude-film TX colour (C["red"])
RX_CYAN = (0.1, 0.85, 1.0)        # crude-film RX colour (C["cyan"])
HOLO_CYAN = (0.2, 0.9, 1.0)       # crude-film hologram colour (C["holo"])
BEAM_NOTE = "* beams invisible (1311 nm); rendered for viewing"   # the scene adds it on screen (DevLog-010 section 4)


def _rgba(c):
    return (*c[:3], 1.0) if len(c) == 3 else tuple(c)


# ==================================================================== hologram
def hologram_group():
    ng, b, gi, go = new_group("VFX_Hologram", [
        ("Color", "NodeSocketColor", _rgba(HOLO_CYAN)),
        ("Strength", "NodeSocketFloat", 3.0, 0.0, 1000.0),
        ("Alpha", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Base", "NodeSocketFloat", 0.3, 0.0, 1.0),
        ("Rim Power", "NodeSocketFloat", 2.0, 0.1, 10.0),
        ("Scan Density", "NodeSocketFloat", 45.0, 0.0, 1000.0),
        ("Scan Speed", "NodeSocketFloat", 0.35, -50.0, 50.0),
        ("Scan Mix", "NodeSocketFloat", 0.45, 0.0, 1.0),
        ("Flicker", "NodeSocketFloat", 0.25, 0.0, 1.0),
        ("Flicker Rate", "NodeSocketFloat", 5.0, 0.0, 100.0),
        ("Glitch", "NodeSocketFloat", 0.4, 0.0, 4.0),
        ("Seed", "NodeSocketFloat", 0.0),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    seed = b.add(I["Seed"], b.mul(b.objinfo("Random"), 17.0))
    facing = b.layer_weight(0.5, "Facing")
    rim = b.pow(facing, I["Rim Power"])
    shape = b.lerp(I["Base"], 1.0, rim)
    z = b.sep(b.geometry("Position"))[2]
    ph = b.mul(b.sub(b.mul(z, I["Scan Density"]), b.mul(t, b.mul(I["Scan Speed"], I["Scan Density"]))), 2 * math.pi)
    s = b.add(b.mul(b.math("SINE", ph), 0.5), 0.5)
    scan = b.sub(1.0, b.mul(I["Scan Mix"], b.one_minus(b.pow(s, 0.6))))
    n1 = b.noise(w=b.add(b.mul(t, I["Flicker Rate"]), seed), scale=1.0, detail=1.0, dims="1D")
    fl = b.sub(1.0, b.mul(I["Flicker"], b.smooth(n1, 0.52, 0.72)))
    jit = b.white(b.add(b.math("FLOOR", b.mul(t, 24.0)), seed), dims="1D")
    fl = b.mul(fl, b.sub(1.0, b.mul(b.mul(I["Flicker"], 0.35), jit)))
    # glitch: rare moments where a horizontal band flares
    gm = b.smooth(b.noise(w=b.add(b.mul(t, 1.3), b.add(seed, 7.0)), scale=1.0, detail=0.0, dims="1D"), 0.64, 0.7)
    band = b.smooth(b.math("FRACT", b.add(b.mul(z, 2.5), b.mul(b.math("FLOOR", b.mul(t, 8.0)), 0.37))), 0.9, 0.97)
    gl = b.add(1.0, b.mul(b.mul(I["Glitch"], gm), band))
    alpha = b.mul(I["Alpha"], b.objinfo("Alpha"))
    k = b.mul(b.mul(b.mul(shape, scan), b.mul(fl, gl)), b.mul(I["Strength"], alpha))
    sh = b.add_shader(b.transparent(), b.emission(I["Color"], k))
    b.link(sh, go.inputs["Shader"])
    return ng


def hologram_material(name="VFX_holo", color=HOLO_CYAN, strength=3.0, base=0.3, rim_power=2.0, scan_density=45.0,
                      scan_speed=0.35, scan_mix=0.45, flicker=0.25, flicker_rate=5.0, glitch=0.4, alpha=1.0, seed=0.0):
    """Additive hologram: fresnel rim + world-Z scanlines + flicker + rare glitch bands. Use base=1, rim_power irrelevant, for
    flat text/labels. Fade with the material input 'Alpha' (key via key_input) or per object with key_obj_alpha."""
    m, b, out = new_material(name)
    g = group_node(b, hologram_group(), Color=_rgba(color), Strength=strength, Base=base, Rim_Power=rim_power,
                   Scan_Density=scan_density, Scan_Speed=scan_speed, Scan_Mix=scan_mix, Flicker=flicker,
                   Flicker_Rate=flicker_rate, Glitch=glitch, Alpha=alpha, Seed=seed)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    set_blended(m)
    m.diffuse_color = _rgba(color)
    return m


# ==================================================================== beam / link
def beam_group():
    ng, b, gi, go = new_group("VFX_Beam", [
        ("Color", "NodeSocketColor", _rgba(TX_RED)),
        ("Strength", "NodeSocketFloat", 8.0, 0.0, 1000.0),
        ("Fade", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Core Power", "NodeSocketFloat", 2.0, 0.1, 10.0),
        ("Flow Mix", "NodeSocketFloat", 0.4, 0.0, 1.0),
        ("Flow Density", "NodeSocketFloat", 4.0, 0.0, 1000.0),
        ("Flow Speed", "NodeSocketFloat", 2.5, -100.0, 100.0),
        ("Seed", "NodeSocketFloat", 0.0),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    facing = b.layer_weight(0.5, "Facing")
    core = b.pow(b.one_minus(facing), I["Core Power"])
    shape = b.lerp(0.3, 1.0, core)
    along = b.vmath("LENGTH", b.vmath("SUBTRACT", b.geometry("Position"), b.objinfo("Location")))   # metres from the source
    q = b.math("FRACT", b.add(b.sub(b.mul(along, I["Flow Density"]), b.mul(t, b.mul(I["Flow Speed"], I["Flow Density"]))),
                              b.add(I["Seed"], b.mul(b.objinfo("Random"), 5.0))))
    pulse = b.pow(b.one_minus(q), 5.0)
    flow = b.add(b.one_minus(I["Flow Mix"]), b.mul(I["Flow Mix"], b.mul(pulse, 2.2)))
    fade = b.mul(I["Fade"], b.objinfo("Alpha"))
    k = b.mul(b.mul(shape, flow), b.mul(I["Strength"], fade))
    whiteish = b.cmix(I["Color"], (1, 1, 1, 1), b.mul(core, 0.35))
    sh = b.add_shader(b.transparent(), b.emission(whiteish, k))
    b.link(sh, go.inputs["Shader"])
    return ng


def beam_material(name="VFX_beam_tx", color=TX_RED, strength=8.0, core_power=2.0, flow_mix=0.4, flow_density=4.0,
                  flow_speed=2.5, fade=1.0, seed=0.0):
    """Thin glowing link: bright core (view-facing), travelling pulses along the beam (source -> target), additive.
    Beam meshes from make_beam run along local +Z from the source. Fade: material 'Fade' input or per-object alpha."""
    m, b, out = new_material(name)
    g = group_node(b, beam_group(), Color=_rgba(color), Strength=strength, Core_Power=core_power, Flow_Mix=flow_mix,
                   Flow_Density=flow_density, Flow_Speed=flow_speed, Fade=fade, Seed=seed)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    set_blended(m)
    m.diffuse_color = _rgba(color)
    return m


def _beam_mesh(length=1.0, radius=0.004, seg=8):
    key = "VFX_beam_mesh_%d_%.4f_%.3f" % (seg, radius, length)
    me = bpy.data.meshes.get(key)
    if me:
        return me
    v, f = [], []
    for zc in (0.0, length):
        for i in range(seg):
            a = 2 * math.pi * i / seg
            v.append((radius * math.cos(a), radius * math.sin(a), zc))
    for i in range(seg):
        j = (i + 1) % seg
        f.append((i, j, seg + j, seg + i))
    me = bpy.data.meshes.new(key)
    me.from_pydata(v, [], f)
    for p in me.polygons:
        p.use_smooth = True
    me.update()
    return me


def make_beam(name, p0, p1, radius=0.004, material=None, color=TX_RED, collection=None, seg=8):
    """Beam object from p0 (origin) to p1: unit-length mesh along local +Z scaled to the distance (shared mesh per radius).
    Returns the object; object colour alpha = per-object fade (1 = on). Not shadow casting."""
    me = _beam_mesh(1.0, radius, seg)
    o = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(o)
    if material is None:
        material = beam_material("VFX_beam_%02x%02x%02x" % tuple(int(255 * c) for c in color[:3]), color)
    if not me.materials:
        me.materials.append(material)
    o.material_slots[0].link = "OBJECT"
    o.material_slots[0].material = material
    o.visible_shadow = False
    o.rotation_mode = "QUATERNION"
    set_beam(o, p0, p1)
    return o


def set_beam(o, p0, p1):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    o.location = p0
    o.rotation_quaternion = d.normalized().to_track_quat("Z", "Y") if d.length > 1e-9 else (1, 0, 0, 0)
    o.scale = (1.0, 1.0, max(d.length, 1e-4))


def key_beam_endpoints(o, frame, p0, p1, interp="LINEAR"):
    """Key a beam to run p0 -> p1 at `frame` (follow moving birds by keying every few frames)."""
    set_beam(o, p0, p1)
    for path in ("location", "rotation_quaternion", "scale"):
        o.keyframe_insert(path, frame=frame)
    _set_interp(o, frame, interp)


def key_obj_alpha(o, frame, value, interp="LINEAR"):
    """Key the object-colour alpha read by every VFX group (beam fade, hologram alpha, eye power, haze)."""
    c = list(o.color)
    c[3] = value
    o.color = c
    o.keyframe_insert("color", index=3, frame=frame)
    _set_interp(o, frame, interp)


def key_beam(o, f_on, f_off, ramp=6, grow=True, length=None):
    """Fade in at f_on (over `ramp` frames), hold, fade out to f_off. grow=True also extends the beam from the source
    (scale Z from 0 to its length) during the fade-in. Uses the current endpoints (set_beam) for the length."""
    L = length if length is not None else o.scale.z
    key_obj_alpha(o, f_on, 0.0)
    key_obj_alpha(o, f_on + ramp, 1.0)
    key_obj_alpha(o, max(f_on + ramp, f_off - ramp), 1.0)
    key_obj_alpha(o, f_off, 0.0)
    if grow:
        o.scale.z = 1e-4
        o.keyframe_insert("scale", index=2, frame=f_on)
        o.scale.z = L
        o.keyframe_insert("scale", index=2, frame=f_on + ramp)
        _set_interp(o, f_on, "BEZIER")


def _set_interp(o, frame, interp):
    ad = o.animation_data
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        for kp in fc.keyframe_points:
            if int(round(kp.co[0])) == int(frame):
                kp.interpolation = interp


def key_input(m, group_input, frame, value, interp="LINEAR"):
    """Key an input of the material's VFX group node (e.g. key_input(holo, 'Alpha', 150, 0.0))."""
    nd = next(n for n in m.node_tree.nodes if n.type == "GROUP")
    s = nd.inputs[group_input]
    s.default_value = value
    s.keyframe_insert("default_value", frame=frame)
    ad = m.node_tree.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) == int(frame):
                    kp.interpolation = interp


# ==================================================================== neon
def neon_group():
    ng, b, gi, go = new_group("VFX_Neon", [
        ("Color", "NodeSocketColor", (1.0, 0.1, 0.7, 1.0)),
        ("Strength", "NodeSocketFloat", 12.0, 0.0, 1000.0),
        ("On", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Flicker", "NodeSocketFloat", 0.0, 0.0, 1.0),
        ("Flicker Rate", "NodeSocketFloat", 2.5, 0.0, 100.0),
        ("Buzz", "NodeSocketFloat", 0.06, 0.0, 1.0),
        ("Core", "NodeSocketFloat", 0.5, 0.0, 1.0),
        ("Seed", "NodeSocketFloat", 0.0),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    seed = b.add(I["Seed"], b.mul(b.objinfo("Random"), 37.0))
    n = b.noise(w=b.add(b.mul(t, I["Flicker Rate"]), seed), scale=1.0, detail=0.0, dims="1D")
    off = b.smooth(n, 0.6, 0.64)
    level = b.sub(1.0, b.mul(I["Flicker"], off))
    buzz = b.noise(w=b.add(b.mul(t, 47.0), b.mul(seed, 3.1)), scale=1.0, detail=0.0, dims="1D")
    level = b.mul(level, b.sub(1.0, b.mul(I["Buzz"], buzz)))
    level = b.mul(level, b.mul(I["On"], b.objinfo("Alpha")))
    core = b.pow(b.one_minus(b.layer_weight(0.5, "Facing")), 2.0)
    col = b.cmix(I["Color"], (1, 1, 1, 1), b.mul(b.mul(core, I["Core"]), 0.55))
    em = b.emission(col, b.mul(I["Strength"], level))
    tube = b.principled(Base_Color=b.cscale(I["Color"], 0.06), Roughness=0.25)
    b.link(b.add_shader(tube.outputs[0], em), go.inputs["Shader"])
    return ng


def neon_material(name, color, strength=12.0, flicker=0.0, flicker_rate=2.5, buzz=0.06, core=0.5, on=1.0, seed=0.0):
    """Neon tube emission: white-hot core toward the viewer, optional dropout flicker (flicker 0..1: fraction of time
    the tube cuts out, ~0.3 = a failing sign), mains buzz; per-object variation via Object Info Random."""
    m, b, out = new_material(name)
    g = group_node(b, neon_group(), Color=_rgba(color), Strength=strength, Flicker=flicker, Flicker_Rate=flicker_rate,
                   Buzz=buzz, Core=core, On=on, Seed=seed)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    m.diffuse_color = _rgba(color)
    return m


# ==================================================================== wet ground + puddles
REFL_A = (1.0, 0.1, 0.7)       # fake-reflection colour toward -x (magenta signs on the left side of the street)
REFL_B = (0.1, 0.85, 1.0)      # toward +x (cyan signs)
REFL_C = (1.0, 0.7, 0.4)       # warm lamp pools (noise patches along y)
_FAST_INPUTS = [("Fast", "NodeSocketFloat", 1.0, 0.0, 1.0), ("Refl A", "NodeSocketColor", (*REFL_A, 1.0)),
                ("Refl B", "NodeSocketColor", (*REFL_B, 1.0)), ("Refl C", "NodeSocketColor", (*REFL_C, 1.0)),
                ("Refl Strength", "NodeSocketFloat", 0.35, 0.0, 100.0), ("Refl Split X", "NodeSocketFloat", 0.0),
                ("Refl Width", "NodeSocketFloat", 7.0, 0.01, 1000.0)]


def _fake_reflection(b, I, pos):
    """Probe-free wet reflection for the fast tier (no ray tracing): colour graded across x (A -> B) with warm patches,
    streaks stretched along y (the street axis), fresnel (grazing views reflect more), slow ripple. Returns (colour, strength)."""
    t = b.time()
    sp = b.sep(pos)
    gx = b.smooth(sp[0], b.sub(I["Refl Split X"], b.mul(I["Refl Width"], 0.5)), b.add(I["Refl Split X"], b.mul(I["Refl Width"], 0.5)))
    col = b.cmix(I["Refl A"], I["Refl B"], gx)
    warm = b.smooth(b.noise(b.comb(0.0, b.mul(sp[1], 0.25), 0.0), scale=1.0, detail=1.0), 0.55, 0.7)
    col = b.cmix(col, I["Refl C"], b.mul(warm, 0.6))
    streak = b.smooth(b.noise(b.comb(b.mul(sp[0], 4.0), b.add(b.mul(sp[1], 0.35), b.mul(t, 0.05)), 0.0), scale=1.0, detail=2.0),
                      0.42, 0.72)
    fres = b.pow(b.layer_weight(0.2, "Facing"), 1.2)
    k = b.mul(b.mul(I["Refl Strength"], b.add(0.15, b.mul(fres, 0.85))), b.add(0.35, b.mul(streak, 0.65)))
    return col, k


def wet_ground_group():
    ng, b, gi, go = new_group("VFX_WetGround", [
        ("Base Color", "NodeSocketColor", (0.03, 0.032, 0.04, 1.0)),
        ("Wetness", "NodeSocketFloat", 0.6, 0.0, 1.0),
        ("Puddle Coverage", "NodeSocketFloat", 0.3, 0.0, 1.0),
        ("Puddle Scale", "NodeSocketFloat", 0.35, 0.0, 100.0),
        ("Rough Dry", "NodeSocketFloat", 0.8, 0.0, 1.0),
        ("Rough Wet", "NodeSocketFloat", 0.3, 0.0, 1.0),
        ("Rough Puddle", "NodeSocketFloat", 0.03, 0.0, 1.0),
        ("Grain", "NodeSocketFloat", 0.25, 0.0, 1.0),
        ("Grain Scale", "NodeSocketFloat", 40.0, 0.0, 1000.0),
        ("Seed", "NodeSocketFloat", 0.0),
    ] + _FAST_INPUTS, [("BSDF", "NodeSocketShader"), ("Puddle Mask", "NodeSocketFloat")])
    if b is None:
        return ng
    I = gi.outputs
    pos0 = b.geometry("Position")
    pos = b.vmath("ADD", pos0, b.comb(I["Seed"], b.mul(I["Seed"], 1.7), 0.0))
    n = b.noise(b.vmath("SCALE", pos, scale=I["Puddle Scale"]), scale=1.0, detail=3.0, rough=0.55)
    thr = b.lerp(0.66, 0.42, I["Puddle Coverage"])
    mask = b.smooth(n, thr, b.add(thr, 0.025))
    n2 = b.noise(b.vmath("SCALE", pos, scale=1.6), scale=1.0, detail=2.0)
    damp = b.mul(I["Wetness"], b.smooth(n2, 0.32, 0.62))
    rough = b.lerp(I["Rough Dry"], I["Rough Wet"], damp)
    rough = b.lerp(rough, I["Rough Puddle"], mask)
    dark = b.sub(1.0, b.add(b.mul(damp, 0.4), b.mul(mask, 0.3)))
    base = b.cscale(I["Base Color"], dark)
    grain = b.noise(b.vmath("SCALE", pos, scale=I["Grain Scale"]), scale=1.0, detail=4.0, rough=0.7)
    bump = b.node("ShaderNodeBump")
    b.put(bump.inputs["Strength"], b.mul(I["Grain"], b.one_minus(mask)))
    b.put(bump.inputs["Distance"], 0.002)
    b.put(bump.inputs["Height"], grain)
    p = b.principled(Base_Color=base, Roughness=rough)
    p.inputs["Specular IOR Level"].default_value = 0.5
    b.link(bump.outputs["Normal"], p.inputs["Normal"])
    # fast tier: glossy but not mirror (no area-light specular blobs), plus the fake reflection on puddles / damp patches
    pf = b.principled(Base_Color=base, Roughness=b.math("MAXIMUM", rough, 0.3))
    pf.inputs["Specular IOR Level"].default_value = 0.2
    b.link(bump.outputs["Normal"], pf.inputs["Normal"])
    col, k = _fake_reflection(b, I, pos0)
    fast = b.add_shader(pf.outputs[0], b.emission(col, b.mul(k, b.add(mask, b.mul(damp, 0.2)))))
    b.link(b.mix_shader(I["Fast"], p.outputs[0], fast), go.inputs["BSDF"])
    b.link(mask, go.inputs["Puddle Mask"])
    return ng


def wet_ground_material(name="VFX_wet_asphalt", base_color=(0.03, 0.032, 0.04), wetness=0.6, coverage=0.3, scale=0.35,
                        rough_dry=0.8, rough_wet=0.3, rough_puddle=0.03, grain=0.25, grain_scale=40.0, seed=0.0, fast=True,
                        refl_strength=0.35, refl_a=REFL_A, refl_b=REFL_B, refl_c=REFL_C):
    """Wet asphalt/concrete in world space (any ground mesh, no UVs): damp patches (darker, glossier) and puddles from a noise
    mask (coverage 0..1). fast=True (default, fast tier without ray tracing): probe-free fake neon reflection on the puddles;
    fast=False: true glossy puddles for ray-traced tiers. Switch later with set_fast_reflections()."""
    m, b, out = new_material(name)
    g = group_node(b, wet_ground_group(), Base_Color=_rgba(base_color), Wetness=wetness, Puddle_Coverage=coverage,
                   Puddle_Scale=scale, Rough_Dry=rough_dry, Rough_Wet=rough_wet, Rough_Puddle=rough_puddle, Grain=grain,
                   Grain_Scale=grain_scale, Seed=seed, Fast=1.0 if fast else 0.0, Refl_Strength=refl_strength,
                   Refl_A=_rgba(refl_a), Refl_B=_rgba(refl_b), Refl_C=_rgba(refl_c))
    b.link(g.outputs["BSDF"], out.inputs["Surface"])
    m.diffuse_color = _rgba(base_color)
    return m


def puddle_group():
    ng, b, gi, go = new_group("VFX_Puddle", [
        ("Tint", "NodeSocketColor", (0.012, 0.013, 0.016, 1.0)),
        ("Roughness", "NodeSocketFloat", 0.02, 0.0, 1.0),
    ] + _FAST_INPUTS, [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    p = b.principled(Base_Color=I["Tint"], Roughness=I["Roughness"])
    p.inputs["Specular IOR Level"].default_value = 0.6
    pos = b.geometry("Position")
    pf = b.principled(Base_Color=b.cscale(I["Tint"], 0.8), Roughness=0.32)
    pf.inputs["Specular IOR Level"].default_value = 0.15
    rip = b.node("ShaderNodeBump")
    rip.inputs["Strength"].default_value = 0.15
    rip.inputs["Distance"].default_value = 0.002
    b.link(b.noise(b.vmath("SCALE", pos, scale=6.0), w=None, scale=1.0, detail=2.0), rip.inputs["Height"])
    b.link(rip.outputs["Normal"], pf.inputs["Normal"])
    col, k = _fake_reflection(b, I, pos)
    fast = b.add_shader(pf.outputs[0], b.emission(col, b.mul(k, 1.4)))
    b.link(b.mix_shader(I["Fast"], p.outputs[0], fast), go.inputs["Shader"])
    return ng


def puddle_material(name="VFX_puddle", tint=(0.012, 0.013, 0.016), roughness=0.02, fast=True, refl_strength=0.35,
                    refl_a=REFL_A, refl_b=REFL_B, refl_c=REFL_C, split_x=0.0, width=7.0):
    """Standing-water decal (puddle meshes 2-5 mm above the ground). fast=True: probe-free wet look for the fast tier
    (darker gloss, roughness 0.32, faint neon reflection graded A (-x) -> B (+x) around split_x, streaks along y, fresnel);
    fast=False: near-black mirror for ray-traced tiers. Switch later with set_fast_reflections()."""
    m, b, out = new_material(name)
    g = group_node(b, puddle_group(), Tint=_rgba(tint), Roughness=roughness, Fast=1.0 if fast else 0.0, Refl_Strength=refl_strength,
                   Refl_A=_rgba(refl_a), Refl_B=_rgba(refl_b), Refl_C=_rgba(refl_c), Refl_Split_X=split_x, Refl_Width=width)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    m.diffuse_color = _rgba(tint)
    return m


def set_fast_reflections(on=True, materials=None):
    """Switch every VFX wet-ground / puddle material (or the given ones) between the fast fake reflection (on=True, no ray
    tracing) and true glossy reflections (on=False, use with setup_eevee(raytracing=True)). Returns the number switched."""
    n = 0
    for m in (materials or bpy.data.materials):
        if not m.node_tree:
            continue
        for nd in m.node_tree.nodes:
            if nd.type == "GROUP" and nd.node_tree and "Fast" in nd.inputs and nd.node_tree.name.startswith(("VFX_WetGround", "VFX_Puddle")):
                nd.inputs["Fast"].default_value = 1.0 if on else 0.0
                n += 1
    return n


# ==================================================================== eye glow
def eye_glow_group():
    ng, b, gi, go = new_group("VFX_EyeGlow", [
        ("Color", "NodeSocketColor", _rgba(TX_RED)),
        ("Power", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Strength", "NodeSocketFloat", 10.0, 0.0, 1000.0),
        ("Rim", "NodeSocketFloat", 0.5, 0.0, 4.0),
        ("Pulse", "NodeSocketFloat", 0.0, 0.0, 1.0),
        ("Pulse Rate", "NodeSocketFloat", 1.5, 0.0, 50.0),
        ("Glass Color", "NodeSocketColor", (0.01, 0.012, 0.016, 1.0)),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    facing = b.layer_weight(0.4, "Facing")
    core = b.pow(b.one_minus(facing), 1.5)
    rim = b.mul(b.pow(facing, 3.0), I["Rim"])
    s = b.add(b.mul(b.math("SINE", b.mul(t, b.mul(I["Pulse Rate"], 2 * math.pi))), 0.5), 0.5)
    pulse = b.sub(1.0, b.mul(I["Pulse"], b.one_minus(s)))
    power = b.mul(I["Power"], b.objinfo("Alpha"))
    k = b.mul(b.mul(b.add(core, rim), pulse), b.mul(I["Strength"], power))
    em = b.emission(b.cmix(I["Color"], (1, 1, 1, 1), b.mul(core, 0.1)), k)
    glass = b.principled(Base_Color=I["Glass Color"], Roughness=0.05)
    glass.inputs["Coat Weight"].default_value = 1.0
    b.link(b.add_shader(glass.outputs[0], em), go.inputs["Shader"])
    return ng


def eye_glow_material(name, color=TX_RED, strength=5.0, power=1.0, rim=0.5, pulse=0.0, pulse_rate=1.5):
    """Lenslet / iris glow: dark coated glass when off, emissive core + rim when on. power 0..1 (material) x object alpha
    (per bird: key_eye_power). TX_RED / RX_CYAN match the crude film."""
    m, b, out = new_material(name)
    g = group_node(b, eye_glow_group(), Color=_rgba(color), Strength=strength, Power=power, Rim=rim, Pulse=pulse,
                   Pulse_Rate=pulse_rate)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    m.diffuse_color = _rgba(color)
    return m


def key_eye_power(o, frame, power, interp="LINEAR"):
    """Per-object eye power (0 dark .. 1 lit) via object-colour alpha; e.g. key 0 at f, 1 at f+4 for a power-on."""
    key_obj_alpha(o, frame, power, interp)
