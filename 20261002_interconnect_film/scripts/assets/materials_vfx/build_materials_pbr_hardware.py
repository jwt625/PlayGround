"""Build materials_pbr_hardware: NG_pbr_surface + MAT_vfx_<name> hardware materials + previews.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_materials_pbr_hardware.py -- assets/components/materials_vfx
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bpy
import common as C
import vfx_lib as V

OUT = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/materials_vfx")
PREV = os.path.join(OUT, "previews")
os.makedirs(PREV, exist_ok=True)
BLEND = os.path.join(OUT, "materials_pbr_hardware.blend")

# Typical reflectance (F0, linear) of common metals, art-directed values after the Filament PBR reflectance table
# (Google Filament, "Physically based rendering" docs, accessed from memory, flagged as typical values not measurements).
F0 = {
    "gold": (1.000, 0.766, 0.336),
    "copper": (0.955, 0.638, 0.538),
    "aluminum": (0.913, 0.922, 0.924),
    "nickel": (0.660, 0.609, 0.526),
    "steel": (0.562, 0.565, 0.578),
    "tin": (0.80, 0.80, 0.78),
    "silver": (0.972, 0.960, 0.915),
}

SURF_IN = [
    ("Base Color", "Color", (0.8, 0.8, 0.8, 1.0), None, None, "Base colour (or metal F0 when Metallic = 1)"),
    ("Metallic", "Float", 0.0, 0.0, 1.0, ""),
    ("Roughness", "Float", 0.4, 0.0, 1.0, ""),
    ("Roughness Variation", "Float", 0.15, 0.0, 1.0, "Spatial variation of roughness (smudges, micro-texture)"),
    ("Specular", "Float", 0.5, 0.0, 1.0, "Specular IOR level for dielectrics"),
    ("IOR", "Float", 1.45, 1.0, 3.0, "Dielectric IOR"),
    ("Anisotropic", "Float", 0.0, 0.0, 1.0, "Anisotropy (brushed metals)"),
    ("Brushed", "Float", 0.0, 0.0, 1.0, "Brush streak bump + roughness streaks along object X"),
    ("Grain Strength", "Float", 0.1, 0.0, 2.0, "Micro bump (bead blast, orange peel, mold grain)"),
    ("Grain Scale", "Float", 800.0, 1.0, 20000.0, "Micro bump frequency per metre of object space"),
    ("Coat Weight", "Float", 0.0, 0.0, 1.0, "Clear coat (lacquer, plating gloss)"),
    ("Coat Roughness", "Float", 0.05, 0.0, 1.0, ""),
    ("Sheen Weight", "Float", 0.0, 0.0, 1.0, "Fabric/rubber sheen"),
    ("Emission Color", "Color", (0.0, 0.0, 0.0, 1.0), None, None, ""),
    ("Emission Strength", "Float", 0.0, 0.0, 1000.0, ""),
]


def build_ng_surface():
    tree, gi, go = V.new_group("NG_pbr_surface", "Shader", SURF_IN, [("BSDF", "Shader")])
    g = lambda n: V.X(gi.outputs[n])
    tc = V.N("ShaderNodeTexCoord")
    pos = V.wrapv(tc.outputs["Object"])
    # grain noise
    ng_ = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=2.0, in_Roughness=0.6)
    V.L(pos, ng_.inputs["Vector"])
    V.L(g("Grain Scale"), ng_.inputs["Scale"])
    # brushed streaks: stretch noise along X
    sc = V.N("ShaderNodeMapping", vector_type="POINT")
    V.L(pos, sc.inputs["Vector"])
    sc.inputs["Scale"].default_value = (1.0, 220.0, 220.0)
    br = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=3.0, in_Roughness=0.7)
    V.L(sc.outputs[0], br.inputs["Vector"])
    br.inputs["Scale"].default_value = 60.0
    # smudge noise for roughness variation (large scale)
    sm = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=4.0, in_Roughness=0.6)
    V.L(pos, sm.inputs["Vector"])
    sm.inputs["Scale"].default_value = 18.0

    h = V.X(ng_.outputs["Fac"]) * g("Grain Strength") + V.X(br.outputs["Fac"]) * g("Brushed") * 0.6
    bump = V.N("ShaderNodeBump", in_Distance=0.0005, in_Strength=1.0)
    V.L(h, bump.inputs["Height"])
    rough = V.clamp01(g("Roughness") * (1.0 + (V.X(sm.outputs["Fac"]) - 0.5) * 2.0 * g("Roughness Variation")
                                         + (V.X(br.outputs["Fac"]) - 0.5) * g("Brushed") * 0.8))
    pb = V.N("ShaderNodeBsdfPrincipled", distribution="MULTI_GGX")
    V.L(gi.outputs["Base Color"], pb.inputs["Base Color"])
    V.L(g("Metallic"), pb.inputs["Metallic"])
    V.L(rough, pb.inputs["Roughness"])
    V.L(g("Specular"), pb.inputs["Specular IOR Level"])
    V.L(g("IOR"), pb.inputs["IOR"])
    V.L(g("Anisotropic"), pb.inputs["Anisotropic"])
    V.L(g("Coat Weight"), pb.inputs["Coat Weight"])
    V.L(g("Coat Roughness"), pb.inputs["Coat Roughness"])
    V.L(g("Sheen Weight"), pb.inputs["Sheen Weight"])
    V.L(bump.outputs["Normal"], pb.inputs["Normal"])
    V.L(gi.outputs["Emission Color"], pb.inputs["Emission Color"])
    V.L(gi.outputs["Emission Strength"], pb.inputs["Emission Strength"])
    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    V.auto_layout(tree)
    return tree


def build_ng_silicon():
    """Silicon die: dark polished mirror with a faint view-angle rainbow (thin-film / diffraction look).
    The rainbow is an angle-based colour ramp, EEVEE-cheap and independent of Principled thin-film support."""
    ins = [
        ("Base Color", "Color", (0.075, 0.08, 0.10, 1.0), None, None, "Bare silicon tone (dark blue-gray)"),
        ("Roughness", "Float", 0.12, 0.0, 1.0, ""),
        ("Diffraction", "Float", 0.35, 0.0, 1.0, "Amount of rainbow tint (0 = plain polished silicon)"),
        ("Diffraction Scale", "Float", 1.0, 0.1, 10.0, "Rainbow band density across the viewing angle"),
        ("Pattern Scale", "Float", 3000.0, 1.0, 100000.0, "Spatial frequency of the faint circuit-grating pattern per metre"),
        ("Pattern Strength", "Float", 0.0, 0.0, 1.0, "Faint grating/pad pattern contrast (0 = off)"),
        ("Metallic", "Float", 0.85, 0.0, 1.0, ""),
    ]
    tree, gi, go = V.new_group("NG_pbr_silicon", "Shader", ins, [("BSDF", "Shader"), ("Color", "Color")])
    g = lambda n: V.X(gi.outputs[n])
    geo = V.N("ShaderNodeNewGeometry")
    facing = V.absx(V.vdot(V.wrapv(geo.outputs["Normal"]), V.wrapv(geo.outputs["Incoming"])))
    ang = (1.0 - facing) * g("Diffraction Scale")
    ramp = V.N("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "LINEAR"
    cols = [(0.0, (0.55, 0.60, 0.9, 1)), (0.18, (0.5, 0.75, 0.55, 1)), (0.38, (0.95, 0.75, 0.35, 1)),
            (0.58, (0.85, 0.35, 0.55, 1)), (0.8, (0.3, 0.45, 0.95, 1)), (1.0, (0.4, 0.8, 0.7, 1))]
    cr.elements[0].position, cr.elements[0].color = cols[0]
    cr.elements[1].position, cr.elements[1].color = cols[1]
    for p, c in cols[2:]:
        e = cr.elements.new(p)
        e.color = c
    V.L(V.fract(ang * 1.7), ramp.inputs["Fac"])
    mix = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    V.L(g("Diffraction") * 0.55, mix.inputs["Factor"])
    V.L(gi.outputs["Base Color"], mix.inputs[6])
    V.L(ramp.outputs["Color"], mix.inputs[7])
    # faint circuit grating: checker-like pattern from two stretched waves
    tc = V.N("ShaderNodeTexCoord")
    pos = V.wrapv(tc.outputs["Object"])
    wv = V.N("ShaderNodeTexChecker")
    V.L(pos, wv.inputs["Vector"])
    V.L(g("Pattern Scale"), wv.inputs["Scale"])
    wv.inputs["Color1"].default_value = (1, 1, 1, 1)
    wv.inputs["Color2"].default_value = (0.0, 0.0, 0.0, 1)
    mix2 = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    V.L(g("Pattern Strength") * 0.5, mix2.inputs["Factor"])
    V.L(mix.outputs[2], mix2.inputs[6])
    V.L(wv.outputs["Color"], mix2.inputs[7])
    pb = V.N("ShaderNodeBsdfPrincipled", distribution="MULTI_GGX")
    V.L(mix2.outputs[2], pb.inputs["Base Color"])
    V.L(g("Metallic"), pb.inputs["Metallic"])
    V.L(g("Roughness"), pb.inputs["Roughness"])
    # spatially varying diffraction tint also on the specular
    V.L(mix2.outputs[2], pb.inputs["Specular Tint"])
    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    tree.links.new(mix2.outputs[2], go.inputs["Color"])
    V.auto_layout(tree)
    return tree


def build_ng_led():
    ins = [("Color", "Color", (0.1, 1.0, 0.2, 1.0), None, None, "LED emission colour"),
           ("Strength", "Float", 1.8, 0.0, 1000.0, "Emission strength (keyframe for blinking; comp bloom adds the glow)"),
           ("Body Color", "Color", (0.02, 0.02, 0.02, 1.0), None, None, "Housing tint when off")]
    tree, gi, go = V.new_group("NG_pbr_led", "Shader", ins, [("BSDF", "Shader")])
    pb = V.N("ShaderNodeBsdfPrincipled")
    V.L(gi.outputs["Body Color"], pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 0.25
    V.L(gi.outputs["Color"], pb.inputs["Emission Color"])
    V.L(gi.outputs["Strength"], pb.inputs["Emission Strength"])
    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    V.auto_layout(tree)
    return tree


def build_ng_fiber_core():
    """Glowing fibre core with optional travelling pulses along the object's local Z axis."""
    ins = [("Color", "Color", V.srgb("#FFB347"), None, None, "Core glow colour (comic: amber)"),
           ("Strength", "Float", 3.5, 0.0, 1000.0, "Base glow strength"),
           ("Pulse Amount", "Float", 0.0, 0.0, 1.0, "0 = steady glow, 1 = isolated travelling pulses"),
           ("Phase", "Float", 0.0, -1e6, 1e6, "Pulse phase in cycles (keyframe: increases by Speed per second)"),
           ("Pulse Frequency", "Float", 3.0, 0.01, 1000.0, "Pulses per metre of object Z"),
           ("Pulse Width", "Float", 0.25, 0.02, 1.0, "Fraction of the period lit")]
    tree, gi, go = V.new_group("NG_pbr_fiber_core", "Shader", ins, [("BSDF", "Shader"), ("Pulse", "Float")])
    g = lambda n: V.X(gi.outputs[n])
    tc = V.N("ShaderNodeTexCoord")
    z = V.wrapv(tc.outputs["Object"]).z
    ph = V.fract(z * g("Pulse Frequency") - g("Phase"))
    pulse = 1.0 - V.smoothstep(0.0, g("Pulse Width"), ph)
    lev = V.lerp(1.0, pulse, g("Pulse Amount"))
    em = V.N("ShaderNodeEmission")
    V.L(gi.outputs["Color"], em.inputs["Color"])
    V.L(g("Strength") * lev, em.inputs["Strength"])
    tree.links.new(em.outputs[0], go.inputs["BSDF"])
    V.L(lev, go.inputs["Pulse"])
    V.auto_layout(tree)
    return tree


# ----------------------------------------------------------------------------- material factories
def mat_group(name, group, set_in=None, viewport=(0.5, 0.5, 0.5, 1.0)):
    m = V.new_material("MAT_vfx_" + name)
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    gn = V.group_node(t, group)
    for k, v in (set_in or {}).items():
        gn.inputs[k.replace("__", " ")].default_value = v
    o = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(gn.outputs["BSDF"], o.inputs["Surface"])
    gn.label = group.name
    V.auto_layout(t)
    m.diffuse_color = viewport
    return m


def S(hexcol):
    return V.srgb(hexcol)


def build_all(g_surf, g_si, g_led, g_fib):
    mats = []

    def A(name, group, **kw):
        vp = kw.pop("_vp", None)
        base = kw.get("Base__Color", (0.5, 0.5, 0.5, 1))
        m = mat_group(name, group, kw, viewport=vp or (tuple(base) if len(base) == 4 else (*base, 1)))
        mats.append(m)
        return m

    def metal(name, f0, rough, **kw):
        return A(name, g_surf, Base__Color=(*f0, 1.0), Metallic=1.0, Roughness=rough, **kw)

    # --- semiconductor
    A("silicon", g_si)
    m = A("silicon_pattern", g_si, Pattern__Strength=0.45, Diffraction=0.5, Pattern__Scale=2500.0)
    # --- metals
    metal("gold", F0["gold"], 0.22, Grain__Strength=0.02)
    metal("gold_brushed", F0["gold"], 0.32, Brushed=1.0, Anisotropic=0.6, Grain__Strength=0.02)
    metal("gold_pad", F0["gold"], 0.28, Grain__Strength=0.08, Grain__Scale=4000.0)
    metal("copper", F0["copper"], 0.28, Grain__Strength=0.05)
    metal("copper_brushed", F0["copper"], 0.35, Brushed=1.0, Anisotropic=0.5)
    metal("solder_tin", F0["tin"], 0.3, Grain__Strength=0.1, Grain__Scale=3000.0)   # SAC305 balls, solder joints
    metal("solder_sac_ball", F0["tin"], 0.18, Grain__Strength=0.02)
    metal("nickel_plated", (0.70, 0.67, 0.62), 0.22, Grain__Strength=0.03, Roughness__Variation=0.25)  # package lid
    metal("nickel_satin", F0["nickel"], 0.4, Brushed=0.3, Grain__Strength=0.15)
    metal("aluminum_raw", F0["aluminum"], 0.38, Grain__Strength=0.25, Grain__Scale=1200.0)
    metal("aluminum_brushed", F0["aluminum"], 0.3, Brushed=1.0, Anisotropic=0.7)
    metal("steel_brushed", F0["steel"], 0.33, Brushed=1.0, Anisotropic=0.8, Roughness__Variation=0.25)
    metal("steel_polished", (0.66, 0.66, 0.68), 0.08, Roughness__Variation=0.1)
    metal("steel_zinc", (0.62, 0.64, 0.66), 0.4, Grain__Strength=0.3, Roughness__Variation=0.4)  # rack, screws
    A("anodized_black", g_surf, Base__Color=S("#0E0E10"), Metallic=0.85, Roughness=0.42, Grain__Strength=0.5,
      Grain__Scale=1500.0, Roughness__Variation=0.1, Specular=0.5)
    A("anodized_blue", g_surf, Base__Color=S("#1E4FB5"), Metallic=0.85, Roughness=0.4, Grain__Strength=0.5,
      Grain__Scale=1500.0)
    A("anodized_red", g_surf, Base__Color=S("#B5262A"), Metallic=0.85, Roughness=0.4, Grain__Strength=0.5,
      Grain__Scale=1500.0)
    A("heatsink_dark", g_surf, Base__Color=S("#2A2C31"), Metallic=0.8, Roughness=0.5, Grain__Strength=0.4,
      Grain__Scale=1200.0)
    A("powder_coat_black", g_surf, Base__Color=S("#151518"), Metallic=0.0, Roughness=0.5, Grain__Strength=0.5,
      Grain__Scale=700.0, Specular=0.4)
    A("powder_coat_gray", g_surf, Base__Color=S("#8C8F96"), Metallic=0.0, Roughness=0.55, Grain__Strength=0.5,
      Grain__Scale=700.0)
    # --- PCB
    for nm, hx in (("green", "#0B6B2E"), ("blue", "#103A8C"), ("black", "#0B0B0D"), ("red", "#9A1A1C"), ("purple", "#3B1568")):
        A("soldermask_" + nm, g_surf, Base__Color=S(hx), Roughness=0.34, Specular=0.55, Grain__Strength=0.05, Grain__Scale=2500.0, Roughness__Variation=0.2, Sheen__Weight=0.0)
    A("silkscreen_white", g_surf, Base__Color=S("#E8E8E2"), Roughness=0.6)
    A("fr4_edge", g_surf, Base__Color=S("#8A7A3C"), Roughness=0.62, Grain__Strength=0.6, Grain__Scale=900.0,
      Roughness__Variation=0.35)
    A("ceramic_alumina", g_surf, Base__Color=S("#E7E4DC"), Roughness=0.5, Grain__Strength=0.3, Grain__Scale=3000.0)
    A("organic_substrate", g_surf, Base__Color=S("#2D3B2A"), Roughness=0.4, Specular=0.5, Grain__Strength=0.04)
    A("mold_compound", g_surf, Base__Color=S("#111113"), Roughness=0.55, Specular=0.4, Grain__Strength=0.7,
      Grain__Scale=3500.0, Roughness__Variation=0.2)
    A("laser_etch_gold", g_surf, Base__Color=(0.75, 0.55, 0.2, 1.0), Metallic=0.7, Roughness=0.45, Grain__Strength=0.1)
    # --- plastics, rubber
    A("abs_black", g_surf, Base__Color=S("#17171A"), Roughness=0.38, Specular=0.5, Grain__Strength=0.18, Grain__Scale=1500.0)
    A("abs_white", g_surf, Base__Color=S("#E9E9E4"), Roughness=0.4, Specular=0.5, Grain__Strength=0.18, Grain__Scale=1500.0)
    A("abs_gray", g_surf, Base__Color=S("#8B8E94"), Roughness=0.4, Specular=0.5, Grain__Strength=0.18, Grain__Scale=1500.0)
    A("plastic_glossy_black", g_surf, Base__Color=S("#0D0D0F"), Roughness=0.12, Specular=0.6)
    A("rubber_black", g_surf, Base__Color=S("#161618"), Roughness=0.8, Specular=0.3, Grain__Strength=0.4, Grain__Scale=900.0,
      Sheen__Weight=0.1)
    A("thermal_pad", g_surf, Base__Color=S("#B9B3B8"), Roughness=0.7, Grain__Strength=0.3, Grain__Scale=1200.0, Sheen__Weight=0.2)
    A("thermal_paste", g_surf, Base__Color=S("#8E9094"), Roughness=0.3, Specular=0.6, Grain__Strength=0.5,
      Grain__Scale=250.0, Roughness__Variation=0.4, Metallic=0.15)
    # --- cable jackets (slightly glossy LSZH/PVC)
    for nm, hx in (("yellow", "#F5C400"), ("orange", "#F26A12"), ("aqua", "#18B7B7"), ("blue", "#1F5FD0"),
                   ("black", "#141416"), ("white", "#EDEDE8"), ("red", "#C8202A"), ("green", "#2AA84A"),
                   ("purple", "#7447B8"), ("gray", "#7A7D84")):
        A("cable_" + nm, g_surf, Base__Color=S(hx), Roughness=0.42, Specular=0.5, Grain__Strength=0.1,
          Grain__Scale=1800.0, Sheen__Weight=0.0, Roughness__Variation=0.15)
    # --- LEDs
    for nm, hx, s in (("green", "#1FE040", 1.8), ("red", "#FF1A10", 1.8), ("amber", "#FF8A00", 1.8),
                      ("blue", "#1F55FF", 1.8), ("white", "#FFFFFF", 1.4)):
        A("led_" + nm, g_led, Color=S(hx), Strength=s)
    # --- fibre
    A("fiber_jacket_yellow", g_surf, Base__Color=S("#F5C400"), Roughness=0.42, Specular=0.5, Grain__Strength=0.1,
      Grain__Scale=1800.0)
    A("fiber_buffer_white", g_surf, Base__Color=S("#F2F2EE"), Roughness=0.45, Specular=0.5)
    A("fiber_core_glow", g_fib)
    return mats


def special_materials(mats):
    """Glass, frosted glass, fast glass (alpha), fiber bare glass thread: Principled-based, no group."""
    def prin(name, **vals):
        m = V.new_material("MAT_vfx_" + name)
        m.use_fake_user = True
        b = m.node_tree.nodes["Principled BSDF"]
        for k, v in vals.items():
            b.inputs[k.replace("__", " ")].default_value = v
        mats.append(m)
        return m
    g = prin("glass", Base__Color=(1, 1, 1, 1), Roughness=0.03, IOR=1.5, Transmission__Weight=1.0, Specular__IOR__Level=0.5)
    g.use_raytrace_refraction = False
    g.diffuse_color = (0.8, 0.9, 1.0, 0.3)
    gf = prin("glass_frosted", Base__Color=(0.95, 0.97, 1.0, 1), Roughness=0.35, IOR=1.5, Transmission__Weight=1.0)
    gx = prin("glass_fast", Base__Color=(0.85, 0.93, 1.0, 1), Roughness=0.02, IOR=1.5, Alpha=0.16, Specular__IOR__Level=0.8)
    V.set_render_method(gx, "BLENDED")
    gt = prin("glass_tinted_amber", Base__Color=S("#C86A1A"), Roughness=0.03, IOR=1.5, Transmission__Weight=1.0)
    fb = prin("fiber_bare_glass", Base__Color=(0.9, 0.95, 1.0, 1), Roughness=0.02, IOR=1.46, Alpha=0.35, Specular__IOR__Level=0.8)
    V.set_render_method(fb, "BLENDED")
    return mats


# ----------------------------------------------------------------------------- previews
def card_demo(mats_by_name):
    """Hardware vignette: substrate, die, lid, balls, gold pads, heatsink fins, cable, LED."""
    V.reopen(BLEND)
    V.prep_scene(samples=48)
    scn = bpy.context.scene
    for o in list(scn.objects):
        o.hide_render = True
        o.hide_viewport = True
    M = lambda n: bpy.data.materials["MAT_vfx_" + n]

    def box(name, size, loc, mat, bev=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        o = bpy.context.active_object
        o.name = name
        o.scale = size
        bpy.ops.object.transform_apply(scale=True)
        o.data.materials.append(mat)
        if bev:
            md = o.modifiers.new("b", "BEVEL")
            md.width = bev
            md.segments = 3
        return o
    mm = C.mm
    # PCB 110 x 64 x 1.6 mm with soldermask top and FR4 edge slot
    pcb = box("pcb", (mm(110), mm(64), mm(1.6)), (0, 0, 0), M("soldermask_green"), bev=mm(0.1))
    pcb.data.materials.append(M("fr4_edge"))
    for p in pcb.data.polygons:
        p.material_index = 0 if abs(p.normal.z) > 0.5 else 1
    top = mm(0.8)
    # package A: substrate + nickel lid;  package B: substrate + bare silicon die + ring of gold pads
    box("subA", (mm(26), mm(26), mm(1.2)), (mm(-34), mm(-4), top + mm(0.6)), M("organic_substrate"), bev=mm(0.1))
    box("lid", (mm(24), mm(24), mm(2.6)), (mm(-34), mm(-4), top + mm(1.2 + 1.3)), M("nickel_plated"), bev=mm(0.4))
    box("subB", (mm(26), mm(26), mm(1.2)), (mm(-3), mm(-4), top + mm(0.6)), M("organic_substrate"), bev=mm(0.1))
    box("die", (mm(12), mm(12), mm(0.75)), (mm(-3), mm(-4), top + mm(1.2 + 0.375)), M("silicon_pattern"), bev=mm(0.05))
    for k in range(10):
        box("pad", (mm(1.4), mm(1.4), mm(0.05)), (mm(-14 + k * 2.4), mm(5.2), top + mm(1.22)), M("gold_pad"))
    for i in range(7):
        for j in range(5):
            box("pad", (mm(1.6), mm(1.6), mm(0.05)), (mm(30 + i * 2.8), mm(-12 + j * 2.8), top + mm(0.03)), M("gold_pad"))
    for i in range(8):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=mm(0.9), segments=24, ring_count=12, location=(mm(-45 + i * 3.4), mm(-24), mm(-0.9)))
        o = bpy.context.active_object
        V.smooth(o)
        o.data.materials.append(M("solder_sac_ball"))
    for i in range(7):
        box("fin", (mm(1.2), mm(22), mm(14)), (mm(32 + i * 3.0), mm(18), top + mm(9.0)), M("anodized_black"), bev=mm(0.1))
    box("hs_base", (mm(22), mm(22), mm(2)), (mm(38), mm(18), top + mm(1.0)), M("anodized_black"), bev=mm(0.2))
    box("chip", (mm(9), mm(9), mm(1.2)), (mm(-30), mm(22), top + mm(0.6)), M("mold_compound"), bev=mm(0.1))
    box("chipb", (mm(9), mm(9), mm(1.2)), (mm(-18), mm(22), top + mm(0.6)), M("ceramic_alumina"), bev=mm(0.1))
    for k, nm in enumerate(("led_green", "led_amber", "led_red")):
        box("led", (mm(1.6), mm(0.8), mm(0.6)), (mm(-8 + k * 3), mm(24), top + mm(0.3)), M(nm))
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=mm(1.5), depth=mm(60), location=(mm(0), mm(-36), mm(1.0)), rotation=(0, math.pi / 2, 0))
    cab = bpy.context.active_object
    V.smooth(cab)
    cab.data.materials.append(M("cable_yellow"))
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=mm(0.45), depth=mm(40), location=(mm(10), mm(-30), mm(2.3)), rotation=(0, math.pi / 2, 0))
    cc = bpy.context.active_object
    V.smooth(cc)
    cc.data.materials.append(M("fiber_core_glow"))
    # floor (matte gray) under it
    box("table", (0.4, 0.4, mm(2)), (0, 0, mm(-2.8)), M("powder_coat_gray"))
    V.studio_lights(center=(0, 0, 0), scale=0.1, key=1.0, world=0.7)
    V.make_camera((mm(30), mm(-150), mm(110)), (mm(0), mm(-2), mm(3)), lens=70)
    scn.eevee.taa_render_samples = 48
    p = os.path.join(PREV, "materials_pbr_hardware_vignette.png")
    V.render_still(p)
    return p


def sheets():
    names = [n for n in bpy.data.materials.keys() if n.startswith("MAT_vfx_")]
    paths = []
    per = 24
    for i in range(0, len(names), per):
        V.reopen(BLEND)
        V.prep_scene(samples=32)
        ms = [bpy.data.materials[n] for n in names[i:i + per]]
        p = os.path.join(PREV, "materials_pbr_hardware_swatches_%d.png" % (i // per + 1))
        V.material_sheet(ms, p, cols=6, pitch=0.025, radius=0.01, label_size=0.0017, samples=32)
        paths.append(p)
    return paths


def main():
    t0 = time.time()
    C.reset()
    g_surf = build_ng_surface()
    g_si = build_ng_silicon()
    g_led = build_ng_led()
    g_fib = build_ng_fiber_core()
    for g in (g_surf, g_si, g_led, g_fib):
        g.use_fake_user = True
        V.mark_asset(g, ("hardware",), "PBR hardware node group")
    mats = build_all(g_surf, g_si, g_led, g_fib)
    special_materials(mats)
    for m in mats:
        V.mark_asset(m, ("hardware",), "Hardware PBR material")
    coll, root = C.new_asset("materials_pbr_hardware", accuracy="C")
    for i, m in enumerate(mats):
        o = C.sphere("materials_pbr_hardware_demo_%02d" % i, 0.004, loc=((i % 12) * 0.01, 0, (i // 12) * 0.01 + 0.004), mat=m)
        C.add(o, coll, root)
    meta = {
        "category": "materials_vfx",
        "description": "Realistic hardware materials: node groups NG_pbr_surface, NG_pbr_silicon, NG_pbr_led, NG_pbr_fiber_core and "
                       "ready MAT_vfx_<name> materials (silicon, gold, copper, solder, nickel lid, anodized aluminium, steel, solder "
                       "masks, FR4 edge, mold compound, plastics, rubber, cable jackets, LEDs, fibre, glass, thermal paste).",
        "sources": [
            {"what": "Metal F0 reflectance table values (gold, copper, aluminum, steel, nickel, silver) used as base colours",
             "source": "Google Filament 'Physically based rendering' documentation, reflectance table (typical values; recalled, not re-fetched; treat as art-directed)",
             "accuracy": "C"},
            {"what": "Solder mask, FR4, mold compound colours", "source": "general appearance of production boards and packages; and the Broadcom chip photo "
             "supplied in the storyboard chat (matte black mold compound, shallow gold laser etch); art-directed", "accuracy": "C"},
        ],
        "node_groups": {
            "NG_pbr_surface": {"inputs": [i[0] for i in SURF_IN], "outputs": ["BSDF"]},
            "NG_pbr_silicon": {"inputs": ["Base Color", "Roughness", "Diffraction", "Diffraction Scale", "Pattern Scale", "Pattern Strength", "Metallic"], "outputs": ["BSDF", "Color"]},
            "NG_pbr_led": {"inputs": ["Color", "Strength", "Body Color"], "outputs": ["BSDF"]},
            "NG_pbr_fiber_core": {"inputs": ["Color", "Strength", "Pulse Amount", "Phase", "Pulse Frequency", "Pulse Width"], "outputs": ["BSDF", "Pulse"]},
        },
        "materials": [m.name for m in mats],
        "custom_properties": {},
        "hooks": [],
        "usage": [
            "Append materials by name (they pull their node group in). Names are MAT_vfx_<name>; hardware agents may swap them in for their own.",
            "Micro-texture (grain, brush streaks) uses Object coordinates: set the object scale to identity (spec) and the frequencies are per metre. "
            "For a very large or small object tune 'Grain Scale'. Brushed streaks run along object X.",
            "LED blink: keyframe the 'Strength' input of the NG_pbr_led group node (0..4 typical). Fibre pulses: keyframe 'Phase' (cycles) and 'Pulse Amount' on NG_pbr_fiber_core; 'Pulse' output gives the 0..1 level.",
            "Glass uses Principled transmission; with ray tracing off EEVEE Next shows it as a darkened tinted surface: use MAT_vfx_glass_fast (dithered alpha) in cheap shots, or turn on ray tracing for hero close-ups.",
            "Silicon 'Diffraction' is a view-angle rainbow (cheap, no Principled thin film needed); raise 'Pattern Strength' for a faint circuit-grating look on close-ups.",
        ],
        "accuracy_notes": "Colours and roughness are art-directed (level C); metals use typical F0 values.",
        "build_seconds": round(time.time() - t0, 1),
    }
    V.finish_lib("materials_pbr_hardware", BLEND, coll, meta)
    sheets()
    card_demo(None)


main()
