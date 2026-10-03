"""Build shader_heat_wave: NG_heat_wave (travelling heat colour) and NG_heat_glow (emission ramp), demo + contact sheets.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_shader_heat_wave.py -- assets/components/materials_vfx
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
SCR = os.environ.get("VFX_SCRATCH", "/tmp")
os.makedirs(PREV, exist_ok=True)
BLEND = os.path.join(OUT, "shader_heat_wave.blend")

HEAT_STOPS = [(0.0, (0.05, 0.15, 1.0)), (0.2, (0.0, 0.55, 1.0)), (0.4, (0.1, 0.85, 0.3)), (0.62, (1.0, 0.9, 0.1)),
              (0.82, (1.0, 0.4, 0.05)), (1.0, (1.0, 0.05, 0.03))]
GLOW_STOPS = [(0.0, (0.1, 0.8, 1.0)), (0.5, (1.0, 0.45, 0.05)), (1.0, (1.0, 0.04, 0.02))]


def ramp_node(stops):
    r = V.N("ShaderNodeValToRGB")
    cr = r.color_ramp
    cr.elements[0].position, c0 = stops[0][0], stops[0][1]
    cr.elements[0].color = (*c0, 1)
    cr.elements[1].position, c1 = stops[1][0], stops[1][1]
    cr.elements[1].color = (*c1, 1)
    for p, c in stops[2:]:
        e = cr.elements.new(p)
        e.color = (*c, 1)
    return r


def build_ng_heat_wave():
    ins = [
        ("Base Color", "Color", (0.55, 0.56, 0.6, 1.0), None, None, "Surface colour where heat colour is blended in"),
        ("p_wave_phase", "Float", 0.0, -1e6, 1e6, "Wave phase in cycles. Keyframe linearly (e.g. 0 -> N) for a travelling wave"),
        ("p_wave_speed", "Float", 0.0, -100.0, 100.0, "Automatic phase speed in cycles per second (uses the Time input, driven by frame/fps)"),
        ("p_wave_k", "Float", 6.0, 0.0, 10000.0, "Spatial frequency in cycles per metre of object space along Axis (wavelength = 1 / k)"),
        ("Time", "Float", 0.0, -1e6, 1e6, "Scene time in seconds (helper adds driver frame/fps)"),
        ("Axis", "Vector", (1.0, 0.0, 0.0), None, None, "Direction of travel in object space"),
        ("Offset", "Float", 0.0, -1e6, 1e6, "Extra spatial phase in cycles; per-slab / per-ring materials set this to k * position"),
        ("Use Position", "Float", 1.0, 0.0, 1.0, "1 = phase from object-space position along Axis; 0 = only Offset (one value per object)"),
        ("Amplitude", "Float", 1.0, 0.0, 2.0, "Swing of the heat value"),
        ("Bias", "Float", 0.0, -1.0, 1.0, "Constant added to the wave (global heating)"),
        ("Contrast", "Float", 1.0, 0.2, 6.0, "Power applied to the 0..1 wave (higher = narrow hot crests)"),
        ("Color Blend", "Float", 0.75, 0.0, 1.0, "How much of the heat colour replaces Base Color"),
        ("Emission Strength", "Float", 1.2, 0.0, 100.0, "Emission of the heat colour"),
        ("Roughness", "Float", 0.4, 0.0, 1.0, ""),
        ("Metallic", "Float", 0.0, 0.0, 1.0, ""),
    ]
    tree, gi, go = V.new_group("NG_heat_wave", "Shader", ins, [("BSDF", "Shader"), ("Heat", "Float"), ("Heat Color", "Color")])
    g = lambda n: V.X(gi.outputs[n])
    tc = V.N("ShaderNodeTexCoord")
    pos = V.wrapv(tc.outputs["Object"])
    s = V.vdot(pos, V.vnorm(gi.outputs["Axis"]))
    phase = s * g("p_wave_k") * g("Use Position") + g("Offset") - g("p_wave_phase") - g("p_wave_speed") * g("Time")
    w = 0.5 + 0.5 * V.sin(phase * (2.0 * math.pi))
    w = V.mathop("POWER", V.clamp01(w), g("Contrast"))
    heat = V.clamp01(g("Bias") + w * g("Amplitude"))
    r = ramp_node(HEAT_STOPS)
    V.L(heat, r.inputs["Fac"])
    mix = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    V.L(g("Color Blend"), mix.inputs["Factor"])
    V.L(gi.outputs["Base Color"], mix.inputs[6])
    V.L(r.outputs["Color"], mix.inputs[7])
    pb = V.N("ShaderNodeBsdfPrincipled")
    V.L(mix.outputs[2], pb.inputs["Base Color"])
    V.L(g("Roughness"), pb.inputs["Roughness"])
    V.L(g("Metallic"), pb.inputs["Metallic"])
    V.L(r.outputs["Color"], pb.inputs["Emission Color"])
    V.L(g("Emission Strength") * heat, pb.inputs["Emission Strength"])
    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    V.L(heat, go.inputs["Heat"])
    tree.links.new(r.outputs["Color"], go.inputs["Heat Color"])
    V.auto_layout(tree)
    return tree


def build_ng_heat_glow():
    ins = [
        ("p_heat", "Float", 0.0, 0.0, 1.0, "Heat 0 (cold, cyan) .. 1 (hot, red). Keyframe this"),
        ("Base Color", "Color", (0.12, 0.12, 0.14, 1.0), None, None, "Lid body colour"),
        ("Strength Max", "Float", 5.0, 0.0, 200.0, "Emission strength at p_heat = 1"),
        ("Strength Min", "Float", 0.15, 0.0, 50.0, "Emission strength at p_heat = 0"),
        ("Curve", "Float", 1.5, 0.2, 5.0, "Emission growth exponent"),
        ("Color Shift", "Float", 0.0, -0.5, 0.5, "Shifts the cyan-orange-red ramp (negative = stay cooler)"),
        ("Roughness", "Float", 0.35, 0.0, 1.0, ""),
        ("Metallic", "Float", 0.6, 0.0, 1.0, ""),
    ]
    tree, gi, go = V.new_group("NG_heat_glow", "Shader", ins, [("BSDF", "Shader"), ("Glow Color", "Color"), ("Strength", "Float")])
    g = lambda n: V.X(gi.outputs[n])
    h = V.clamp01(g("p_heat") + g("Color Shift"))
    r = ramp_node(GLOW_STOPS)
    V.L(h, r.inputs["Fac"])
    st = V.lerp(g("Strength Min"), g("Strength Max"), V.mathop("POWER", V.clamp01(g("p_heat")), g("Curve")))
    pb = V.N("ShaderNodeBsdfPrincipled")
    V.L(gi.outputs["Base Color"], pb.inputs["Base Color"])
    V.L(g("Roughness"), pb.inputs["Roughness"])
    V.L(g("Metallic"), pb.inputs["Metallic"])
    V.L(r.outputs["Color"], pb.inputs["Emission Color"])
    V.L(st, pb.inputs["Emission Strength"])
    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    tree.links.new(r.outputs["Color"], go.inputs["Glow Color"])
    V.L(st, go.inputs["Strength"])
    V.auto_layout(tree)
    return tree


def mat_with(name, group, **inputs):
    m = V.new_material(name)
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    gn = V.group_node(t, group)
    for k, v in inputs.items():
        gn.inputs[k.replace("__", " ")].default_value = v
    o = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(gn.outputs["BSDF"], o.inputs["Surface"])
    V.auto_layout(t)
    return m, gn


def main():
    t0 = time.time()
    C.reset()
    gw = build_ng_heat_wave()
    gg = build_ng_heat_glow()
    for g in (gw, gg):
        g.use_fake_user = True
        V.mark_asset(g, ("heat", "shader"), "Heat shader")
    m_w, n_w = mat_with("MAT_vfx_heat_wave", gw, p_wave_k=6.0, p_wave_speed=0.5)
    V.drive_time_input(m_w, n_w)
    m_g, n_g = mat_with("MAT_vfx_heat_glow", gg)
    for m in (m_w, m_g):
        V.mark_asset(m, ("heat",), "Ready heat material")
    coll, root = C.new_asset("shader_heat_wave", accuracy="C")
    root["p_wave_phase"] = 0.0
    root["p_heat"] = 0.0
    o = C.cube("shader_heat_wave_demo_bar", (0.3, 0.05, 0.02), loc=(0, 0, 0.01), mat=m_w)
    C.add(o, coll, root)
    o2 = C.cube("shader_heat_wave_demo_lid", (0.05, 0.05, 0.01), loc=(0.4, 0, 0.005), mat=m_g)
    C.add(o2, coll, root)
    C.hook("demo_none", coll, root)
    meta = {
        "category": "materials_vfx",
        "description": "Heat shaders: NG_heat_wave (travelling blue-green-yellow-red heat along an axis, usable on one object or "
                       "per-slab/per-ring materials) and NG_heat_glow (cyan -> orange -> red emission driven by p_heat) for the optical-engine lids.",
        "sources": [],
        "node_groups": {
            "NG_heat_wave": {"inputs": ["Base Color", "p_wave_phase", "p_wave_speed", "p_wave_k", "Time", "Axis", "Offset", "Use Position",
                                        "Amplitude", "Bias", "Contrast", "Color Blend", "Emission Strength", "Roughness", "Metallic"],
                             "outputs": ["BSDF", "Heat", "Heat Color"]},
            "NG_heat_glow": {"inputs": ["p_heat", "Base Color", "Strength Max", "Strength Min", "Curve", "Color Shift", "Roughness", "Metallic"],
                             "outputs": ["BSDF", "Glow Color", "Strength"]},
        },
        "materials": ["MAT_vfx_heat_wave", "MAT_vfx_heat_glow"],
        "custom_properties": {"p_wave_phase": "group input of NG_heat_wave (cycles; keyframe)", "p_wave_speed": "group input (cycles/s)",
                              "p_wave_k": "group input (cycles/m)", "p_heat": "group input of NG_heat_glow (0..1; keyframe)"},
        "hooks": [],
        "usage": [
            "One object: assign MAT_vfx_heat_wave (Use Position = 1); the wave runs along object-space Axis (default +X). "
            "Set p_wave_k = cycles per metre of object space (wavelength = 1 / k); for a 3 cm slab row with a ~1 cm wavelength use k = 100.",
            "Drive the motion either by keyframing p_wave_phase (cycles) or by setting p_wave_speed (cycles per second): the MAT_vfx_heat_wave "
            "material carries a driver Time = frame / 30 on the group's Time input; for copies call vfx_lib.drive_time_input(mat, node).",
            "Per-slab / per-ring materials: duplicate the material per object, set Use Position = 0 and Offset = k * x_object (cycles, "
            "x_object = the object's position along the travel axis in the same coordinate system), then all objects share one coherent wave "
            "driven by the same p_wave_phase. The build script's contact sheet shows 28 slabs + 24 rings this way.",
            "NG_heat_glow: keyframe p_heat 0 -> 1 (cyan -> orange -> red glow). 'Glow Color' and 'Strength' outputs can feed halos or light-emitting meshes.",
            "Emission is HDR: enable the comp glow (lighting_and_world NG_comp_post) for bloom.",
        ],
        "accuracy_notes": "Artistic colour ramps (blue-green-yellow-red heat map; cyan-orange-red glow), not thermal physics.",
        "build_seconds": round(time.time() - t0, 1),
    }
    V.finish_lib("shader_heat_wave", BLEND, coll, meta)
    previews()


def previews():
    import numpy as np
    V.reopen(BLEND)
    V.prep_scene(res=(600, 340), samples=24)
    scn = bpy.context.scene
    for o in scn.objects:
        o.hide_render = True
    gw = bpy.data.node_groups["NG_heat_wave"]
    mm = C.mm
    # PIC-like layout: 28 substrate slabs (per-slab Offset) and 24 rings on top (per-ring Offset)
    k = 19.0  # cycles per metre (about one wave across the 54 mm substrate)
    L = 0.054
    slabs = []
    ncell = 28
    sl_w = L / ncell
    mats = []
    for i in range(ncell):
        x = -L / 2 + (i + 0.5) * sl_w
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, 0))
        o = bpy.context.active_object
        o.scale = (sl_w * 0.94, 0.02, 0.003)
        o.name = "slab_%d" % i
        m, n = mat_with("TMP_slab_%d" % i, gw, Use__Position=0.0, Offset=k * x, p_wave_k=k, Base__Color=(0.35, 0.37, 0.42, 1), Color__Blend=0.55, Emission__Strength=0.8)
        o.data.materials.append(m)
        mats.append((m, n))
    for i in range(24):
        x = -L / 2 + (i + 0.5) * L / 24
        bpy.ops.mesh.primitive_torus_add(location=(x, 0, 0.0035), major_radius=0.0009, minor_radius=0.00035, major_segments=24, minor_segments=8)
        o = bpy.context.active_object
        V.smooth(o)
        o.name = "ring_%d" % i
        m, n = mat_with("TMP_ring_%d" % i, gw, Use__Position=0.0, Offset=k * x, p_wave_k=k, Color__Blend=0.9, Emission__Strength=3.0, Contrast=1.5)
        o.data.materials.append(m)
        mats.append((m, n))
    # a continuous bar below (Use Position = 1)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.032, 0))
    bar = bpy.context.active_object
    bar.scale = (L, 0.01, 0.003)
    bpy.ops.object.transform_apply(scale=True)
    bm, bn = mat_with("TMP_bar", gw, Use__Position=1.0, p_wave_k=k, Color__Blend=0.9, Emission__Strength=1.5)
    bar.data.materials.append(bm)
    mats.append((bm, bn))
    # heat glow lids
    gg = bpy.data.node_groups["NG_heat_glow"]
    lids = []
    for i in range(6):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.0225 + i * 0.009, 0.03, 0.0))
        o = bpy.context.active_object
        o.scale = (0.0075, 0.0075, 0.003)
        m, n = mat_with("TMP_lid_%d" % i, gg, p_heat=i / 5.0)
        o.data.materials.append(m)
    V.studio_lights(center=(0, 0, 0), scale=0.05, key=1.1, world=0.5)
    bpy.ops.mesh.primitive_plane_add(size=0.3, location=(0, 0, -0.002))
    fl = bpy.context.active_object
    fm = V.new_material("TMP_floor")
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.04, 0.04, 0.05, 1)
    fl.data.materials.append(fm)
    cam, tg = V.make_camera((0.0, -0.12, 0.085), (0, -0.002, 0), lens=55)
    scn.render.resolution_x, scn.render.resolution_y = 600, 340
    paths = []
    ph = [0.0, 0.17, 0.33, 0.5, 0.67, 0.83]
    for i, p in enumerate(ph):
        for m, n in mats:
            n.inputs["p_wave_phase"].default_value = p
        pth = os.path.join(SCR, "heatwave_f%d.png" % i)
        V.render_still(pth)
        paths.append(pth)
    sheet = os.path.join(PREV, "shader_heat_wave_contact_sheet.png")
    V.tile_pngs(paths, sheet, 2)
    for p in paths:
        os.remove(p)
    print("sheet", sheet)


main()
