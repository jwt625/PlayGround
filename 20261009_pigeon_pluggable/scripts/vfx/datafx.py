"""High-speed data look (workstream E, DevLog-013 section 8; owner item F7, DevLog-015).

- Data emitter: lenslets / laser facets blink fast and randomly per lane with PAM4-like levels (0, 1/3, 2/3, 1). Lanes are
  cells of a regular grid in object space (pitch + offset) so one mesh holding all 8 lenslets still blinks per lenslet;
  per-object variation via Object Info Random; key_blink() adds seeded per-object on/off keys for separate emitter objects.
- Data beam: pulse train of PAM4 symbols scrolling from the source along the beam, with dark gaps between symbols and
  random fast blinking; same mesh convention as materials.make_beam (origin at the source).
Time from the `frame/30` driver; fully shader-driven, no per-frame keys needed.
"""
import random

import bpy

from .materials import TX_RED, _beam_mesh, key_obj_alpha, set_beam
from .nodes import group_node, new_group, new_material, set_blended


def _rgba(c):
    return (*c[:3], 1.0)


def _white4(b, vec, w):
    nd = b.node("ShaderNodeTexWhiteNoise", noise_dimensions="4D")
    b.put(nd.inputs["Vector"], vec)
    b.put(nd.inputs["W"], w)
    return nd.outputs["Value"]


def data_emitter_group():
    ng, b, gi, go = new_group("VFX_DataEmitter", [
        ("Color", "NodeSocketColor", _rgba(TX_RED)),
        ("Strength", "NodeSocketFloat", 6.0, 0.0, 1000.0),
        ("Power", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Rate", "NodeSocketFloat", 20.0, 0.0, 1000.0),
        ("Levels", "NodeSocketFloat", 4.0, 2.0, 16.0),
        ("Base", "NodeSocketFloat", 0.08, 0.0, 1.0),
        ("Pitch", "NodeSocketFloat", 1.0, 1e-6, 1e6),
        ("Offset", "NodeSocketVector", (0.0, 0.0, 0.0)),
        ("Seed", "NodeSocketFloat", 0.0),
        ("Glass Color", "NodeSocketColor", (0.01, 0.012, 0.016, 1.0)),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    p = b.vmath("ADD", b.texcoord("Object"), I["Offset"])
    cell = b.vmath("FLOOR", b.vmath("DIVIDE", p, b.comb(I["Pitch"], I["Pitch"], I["Pitch"])))
    cell = b.vmath("ADD", cell, b.comb(b.mul(b.objinfo("Random"), 113.0), I["Seed"], 0.0))
    sym = b.math("FLOOR", b.mul(t, I["Rate"]))
    r = _white4(b, cell, sym)
    lv = b.math("DIVIDE", b.math("FLOOR", b.mul(r, b.sub(I["Levels"], 0.001))), b.sub(I["Levels"], 1.0))
    amp = b.lerp(I["Base"], 1.0, lv)
    k = b.mul(b.mul(amp, I["Strength"]), b.mul(I["Power"], b.objinfo("Alpha")))
    core = b.pow(b.one_minus(b.layer_weight(0.4, "Facing")), 1.5)
    em = b.emission(b.cmix(I["Color"], (1, 1, 1, 1), b.mul(b.mul(core, lv), 0.15)), k)
    glass = b.principled(Base_Color=I["Glass Color"], Roughness=0.05)
    b.link(b.add_shader(glass.outputs[0], em), go.inputs["Shader"])
    return ng


def data_emitter_material(name, color=TX_RED, strength=6.0, rate=20.0, levels=4, base=0.08, pitch=1.0, offset=(0.0, 0.0, 0.0),
                          seed=0.0, power=1.0):
    """Fast random per-lane blinking emitter. pitch/offset: lane grid in the object's local units (e.g. the hero lenslet
    pitch 0.0023 m in hero-GLB units; one lane per lenslet when the grid is aligned to the lenslet centres +- pitch/2).
    rate: symbols per second (keep <= fps/1.5 to read as blinking; 20 at 30 fps). levels 4 = PAM4."""
    m, b, out = new_material(name)
    g = group_node(b, data_emitter_group(), Color=_rgba(color), Strength=strength, Rate=rate, Levels=float(levels), Base=base,
                   Pitch=pitch, Offset=offset, Seed=seed, Power=power)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    m.diffuse_color = _rgba(color)
    return m


def data_beam_group():
    ng, b, gi, go = new_group("VFX_DataBeam", [
        ("Color", "NodeSocketColor", _rgba(TX_RED)),
        ("Strength", "NodeSocketFloat", 9.0, 0.0, 1000.0),
        ("Fade", "NodeSocketFloat", 1.0, 0.0, 1.0),
        ("Symbol Length", "NodeSocketFloat", 0.06, 1e-6, 1e6),
        ("Speed", "NodeSocketFloat", 3.0, -1e6, 1e6),
        ("Levels", "NodeSocketFloat", 4.0, 2.0, 16.0),
        ("Gap", "NodeSocketFloat", 0.3, 0.0, 0.95),
        ("Blink Rate", "NodeSocketFloat", 15.0, 0.0, 1000.0),
        ("Blink Depth", "NodeSocketFloat", 0.35, 0.0, 1.0),
        ("Core Power", "NodeSocketFloat", 2.0, 0.1, 10.0),
        ("Seed", "NodeSocketFloat", 0.0),
    ], [("Shader", "NodeSocketShader")])
    if b is None:
        return ng
    I = gi.outputs
    t = b.time()
    seed = b.add(I["Seed"], b.mul(b.objinfo("Random"), 71.0))
    d = b.vmath("LENGTH", b.vmath("SUBTRACT", b.geometry("Position"), b.objinfo("Location")))
    ph = b.math("DIVIDE", b.sub(d, b.mul(t, I["Speed"])), I["Symbol Length"])
    sym = b.math("FLOOR", ph)
    fr = b.math("FRACT", ph)
    r = b.white(b.add(sym, b.mul(seed, 13.0)), dims="1D")
    lv = b.math("DIVIDE", b.math("FLOOR", b.mul(r, b.sub(I["Levels"], 0.001))), b.sub(I["Levels"], 1.0))
    mask = b.smooth(b.sub(b.one_minus(I["Gap"]), fr), 0.0, 0.04)
    amp = b.add(0.06, b.mul(lv, mask))
    bl = b.math("GREATER_THAN", b.white(b.add(b.math("FLOOR", b.mul(t, I["Blink Rate"])), seed), dims="1D"), 0.5)
    blink = b.sub(1.0, b.mul(I["Blink Depth"], bl))
    core = b.pow(b.one_minus(b.layer_weight(0.5, "Facing")), I["Core Power"])
    shape = b.lerp(0.3, 1.0, core)
    k = b.mul(b.mul(b.mul(shape, amp), b.mul(blink, I["Strength"])), b.mul(I["Fade"], b.objinfo("Alpha")))
    col = b.cmix(I["Color"], (1, 1, 1, 1), b.mul(b.mul(core, lv), 0.3))
    b.link(b.add_shader(b.transparent(), b.emission(col, k)), go.inputs["Shader"])
    return ng


def data_beam_material(name, color=TX_RED, strength=9.0, symbol_length=0.06, speed=3.0, levels=4, gap=0.3, blink_rate=15.0,
                       blink_depth=0.35, core_power=2.0, fade=1.0, seed=0.0):
    """Pulse-train beam: PAM4 symbols (symbol_length in metres along the beam) scrolling source -> target at `speed` m/s,
    dark gaps between symbols, random fast blinking. For the x1000 macro sets use symbol_length ~0.05-0.15 and speed 2-6;
    for hall-scale links 0.01-0.03 and 0.5-1.5. Per-object fade via key_obj_alpha / key_beam like make_beam."""
    m, b, out = new_material(name)
    g = group_node(b, data_beam_group(), Color=_rgba(color), Strength=strength, Symbol_Length=symbol_length, Speed=speed,
                   Levels=float(levels), Gap=gap, Blink_Rate=blink_rate, Blink_Depth=blink_depth, Core_Power=core_power, Fade=fade,
                   Seed=seed)
    b.link(g.outputs["Shader"], out.inputs["Surface"])
    set_blended(m)
    m.diffuse_color = _rgba(color)
    return m


def make_data_beam(name, p0, p1, radius=0.004, material=None, color=TX_RED, collection=None, seg=8, **kw):
    """Beam object (same mesh / keying conventions as materials.make_beam) with a data-beam material."""
    me = _beam_mesh(1.0, radius, seg)
    o = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(o)
    if material is None:
        material = data_beam_material("VFX_databeam_%02x%02x%02x" % tuple(int(255 * c) for c in color[:3]), color, **kw)
    if not me.materials:
        me.materials.append(material)
    o.material_slots[0].link = "OBJECT"
    o.material_slots[0].material = material
    o.visible_shadow = False
    o.rotation_mode = "QUATERNION"
    set_beam(o, p0, p1)
    return o


def apply_data_look(objs, material):
    """Assign `material` to every slot of the given objects (object-linked slots, shared meshes untouched)."""
    for o in objs:
        for s in o.material_slots:
            s.link = "OBJECT"
            s.material = material


def key_blink(objs, f0, f1, rate=12.0, duty=0.6, seed=0, fps=30, levels=(0.0, 0.33, 0.66, 1.0)):
    """Seeded random per-object blinking for separate emitter objects (laser facets, LEDs): keys object alpha (read by
    every VFX group) with CONSTANT steps at `rate` symbols/s between f0 and f1; levels drawn from `levels` with
    probability `duty` of a non-zero level. Ends at alpha 1."""
    rng = random.Random(seed)
    stepf = max(1, round(fps / rate))
    for o in objs:
        f = f0
        while f < f1:
            v = rng.choice(levels[1:]) if rng.random() < duty else levels[0]
            key_obj_alpha(o, f, v, "CONSTANT")
            f += stepf
        key_obj_alpha(o, f1, 1.0, "CONSTANT")
