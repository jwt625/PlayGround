"""Build shader_egg_fry: NG_egg_fry (fried-egg cooking shader, one parameter p_cook 0..1), a test fried egg mesh and a 6-frame contact sheet.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_shader_egg_fry.py -- assets/components/materials_vfx
"""
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bmesh
import bpy
import common as C
import vfx_lib as V

OUT = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/materials_vfx")
PREV = os.path.join(OUT, "previews")
SCR = os.environ.get("VFX_SCRATCH", "/tmp")
os.makedirs(PREV, exist_ok=True)
BLEND = os.path.join(OUT, "shader_egg_fry.blend")

INS = [
    ("p_cook", "Float", 0.0, 0.0, 1.0, "Cooking progress 0 (raw) .. 1 (fully fried). Keyframe this"),
    ("Is Yolk", "Float", 0.0, 0.0, 1.0, "0 = egg white, 1 = yolk (same group on both materials)"),
    ("Rim", "Float", 0.0, 0.0, 1.0, "0 at the egg centre .. 1 at the white's outer edge. Wire the 'egg_rim' mesh attribute (or radial fallback)"),
    ("Noise Scale", "Float", 5.0, 0.5, 100.0, "Frequency of the browning-front noise over the bounding box (scale independent)"),
    ("Brown Extent", "Float", 0.42, 0.0, 1.0, "How far browning reaches inward from the rim at p_cook = 1"),
    ("Bubble Density", "Float", 0.4, 0.0, 1.0, "Share of cells that blister"),
    ("Bubble Scale", "Float", 7.0, 1.0, 100.0, "Blister cell frequency over the bounding box"),
    ("Bubble Height", "Float", 1.0, 0.0, 4.0, "Blister bump strength"),
    ("Raw Alpha", "Float", 0.22, 0.0, 1.0, "White opacity when raw (glassy)"),
    ("Oil Sheen", "Float", 0.6, 0.0, 1.0, "Oil sheen (clear coat) weight"),
    ("Frill", "Float", 0.12, 0.0, 1.0, "Lacy crisp-edge alpha erosion at the rim when cooked (0 = off)"),
    ("White Cooked Color", "Color", V.srgb("#F4F1E6"), None, None, ""),
    ("Yolk Raw Color", "Color", V.srgb("#F07A00"), None, None, ""),
    ("Yolk Cooked Color", "Color", V.srgb("#F2B53C"), None, None, ""),
]

BROWN_STOPS = [(0.0, V.srgb("#F4F1E6")), (0.12, V.srgb("#F0D890")), (0.3, V.srgb("#D8A040")), (0.58, V.srgb("#8A4C14")),
               (1.0, V.srgb("#2E1606"))]


def build_ng():
    tree, gi, go = V.new_group("NG_egg_fry", "Shader", INS, [("BSDF", "Shader"), ("Browning", "Float"), ("Opacity", "Float"), ("Steam", "Float")])
    g = lambda n: V.X(gi.outputs[n])
    c = g("p_cook")
    tc = V.N("ShaderNodeTexCoord")
    gen = V.wrapv(tc.outputs["Generated"])
    rim = V.clamp01(g("Rim"))

    def noise(scale, detail=3.0, rough=0.55, off=(0, 0, 0), dist=0.0):
        n = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=detail, in_Roughness=rough, in_Distortion=dist)
        V.L(gen + off if off != (0, 0, 0) else gen, n.inputs["Vector"])
        V.L(scale, n.inputs["Scale"])
        return V.X(n.outputs["Fac"])

    n1 = noise(g("Noise Scale"))
    n2 = noise(g("Noise Scale") * 8.0, detail=2.0, off=(3.1, 1.7, 0.4))
    npatch = noise(g("Noise Scale") * 0.6, off=(9.0, 2.0, 5.0))

    # --- white: opacity front (rim cooks first), browning band from the rim inward
    t_op = 0.08 + (1.0 - rim) * 0.34 + (n1 - 0.5) * 0.14
    op = V.smoothstep(t_op - 0.07, t_op + 0.07, c)
    w_ = V.clamp01((c - 0.3) / 0.7) * g("Brown Extent")
    d = rim + (n1 - 0.5) * 0.16 + (n2 - 0.5) * 0.05
    brown_m = V.smoothstep(1.0 - w_ - 0.05, 1.0 - w_ + 0.05, d) * V.smoothstep(0.0, 0.05, w_)
    depth = V.clamp01((d - (1.0 - w_)) / V.maxx(w_, 0.05))
    bt = brown_m * (0.25 + 0.75 * depth) * (0.45 + 0.55 * c)
    ramp = V.N("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = BROWN_STOPS[0][0], BROWN_STOPS[0][1]
    cr.elements[1].position, cr.elements[1].color = BROWN_STOPS[1][0], BROWN_STOPS[1][1]
    for p, col in BROWN_STOPS[2:]:
        cr.elements.new(p).color = col
    V.L(bt, ramp.inputs["Fac"])
    # blend raw tint -> cooked white -> brown
    raw_tint = (0.86, 0.93, 0.80, 1.0)
    mraw = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    V.L(op, mraw.inputs["Factor"])
    mraw.inputs[6].default_value = raw_tint
    V.L(gi.outputs["White Cooked Color"], mraw.inputs[7])
    mbr = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    V.L(V.smoothstep(0.0, 0.18, bt), mbr.inputs["Factor"])
    V.L(mraw.outputs[2], mbr.inputs[6])
    V.L(ramp.outputs["Color"], mbr.inputs[7])
    white_col = mbr.outputs[2]
    # frills: erode alpha at the very edge when cooked
    frill = V.smoothstep(0.93, 1.02, rim + (n2 - 0.5) * 0.25 * g("Frill") * 2.0) * g("Frill") * V.clamp01((c - 0.5) * 2.0)
    white_alpha = V.maxx(V.lerp(g("Raw Alpha"), 1.0, op), V.smoothstep(0.02, 0.2, bt)) * (1.0 - frill)
    white_rough = V.lerp(0.10, 0.42, op) + brown_m * 0.3
    # blisters
    vor = V.N("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1", in_Randomness=1.0)
    V.L(gen + V.comb(4.4, 1.3, 7.7), vor.inputs["Vector"])
    V.L(g("Bubble Scale"), vor.inputs["Scale"])
    cell = V.sepxyz(V.wrapv(vor.outputs["Color"]))[0]
    pick = V.smoothstep(1.0 - g("Bubble Density") - 0.05, 1.0 - g("Bubble Density") + 0.05, cell)
    dome = 1.0 - V.smoothstep(0.0, 0.4, V.X(vor.outputs["Distance"]))
    bub = dome * pick * V.smoothstep(0.12, 0.5, c) * (1.0 - V.smoothstep(0.92, 1.0, rim))
    wr = V.X(V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=4.0, in_Roughness=0.7).outputs["Fac"])
    h_white = bub * g("Bubble Height") + brown_m * (n2 * 0.5) + (n2 - 0.5) * 0.1 * op
    # oil sheen: patches of coat
    oil = g("Oil Sheen") * V.lerp(0.35, 1.0, V.smoothstep(0.35, 0.65, npatch)) * (1.0 - 0.45 * op)

    # --- yolk
    firm = V.smoothstep(0.35, 0.9, c)
    ymix = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    V.L(firm, ymix.inputs["Factor"])
    V.L(gi.outputs["Yolk Raw Color"], ymix.inputs[6])
    V.L(gi.outputs["Yolk Cooked Color"], ymix.inputs[7])
    yolk_col = ymix.outputs[2]
    yolk_rough = V.lerp(0.05, 0.5, firm)
    yolk_coat = V.lerp(1.0, 0.15, firm)
    yolk_sss = V.lerp(0.85, 0.2, firm)
    yolk_h = n2 * 0.25 * firm

    # --- select white / yolk
    y = V.clamp01(g("Is Yolk"))
    col = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
    V.L(y, col.inputs["Factor"])
    V.L(white_col, col.inputs[6])
    V.L(yolk_col, col.inputs[7])
    alpha = V.lerp(white_alpha, 1.0, y)
    rough = V.lerp(white_rough, yolk_rough, y)
    coat = V.lerp(oil, yolk_coat, y)
    sss = V.lerp(0.25 * op, yolk_sss, y)
    height = V.lerp(h_white, yolk_h, y)
    bump = V.N("ShaderNodeBump", in_Distance=0.0012)
    bump.inputs["Strength"].default_value = 0.8
    V.L(height, bump.inputs["Height"])

    pb = V.N("ShaderNodeBsdfPrincipled", subsurface_method="RANDOM_WALK")
    V.L(col.outputs[2], pb.inputs["Base Color"])
    V.L(rough, pb.inputs["Roughness"])
    V.L(alpha, pb.inputs["Alpha"])
    V.L(coat, pb.inputs["Coat Weight"])
    pb.inputs["Coat Roughness"].default_value = 0.06
    pb.inputs["Specular IOR Level"].default_value = 0.6
    V.L(sss, pb.inputs["Subsurface Weight"])
    pb.inputs["Subsurface Scale"].default_value = 0.004
    pb.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.12)
    V.L(bump.outputs["Normal"], pb.inputs["Normal"])
    # raw yolk glows slightly (juicy translucent look)
    V.L(yolk_col, pb.inputs["Emission Color"])
    V.L(y * (1.0 - firm) * 0.12, pb.inputs["Emission Strength"])
    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    V.L(bt, go.inputs["Browning"])
    V.L(op, go.inputs["Opacity"])
    V.L(V.smoothstep(0.08, 0.35, c) * (1.0 - 0.6 * V.smoothstep(0.85, 1.0, c)), go.inputs["Steam"])
    V.auto_layout(tree)
    return tree


def egg_material(name, group, yolk):
    m = V.new_material(name)
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    gn = V.group_node(t, group)
    gn.inputs["Is Yolk"].default_value = 1.0 if yolk else 0.0
    if not yolk:
        at = t.nodes.new("ShaderNodeAttribute")
        at.attribute_name = "egg_rim"
        gen = t.nodes.new("ShaderNodeTexCoord")
        rad = V.vlen(V.vop("SUBTRACT", V.wrapv(gen.outputs["Generated"]), (0.5, 0.5, 0.0)) * (2.0 * 1.0, 2.0, 0.0)) if False else None
        # radial fallback from generated coordinates (xy), used where the attribute is zero
        sep = t.nodes.new("ShaderNodeSeparateXYZ")
        t.links.new(gen.outputs["Generated"], sep.inputs[0])
        fx = V.X(sep.outputs[0]) - 0.5
        fy = V.X(sep.outputs[1]) - 0.5
        radial = V.clamp01(V.sqrt(fx * fx + fy * fy) * 2.0)
        rimv = V.where(V.X(at.outputs["Fac"]) > 0.0001, V.X(at.outputs["Fac"]), radial)
        V.L(rimv, gn.inputs["Rim"])
    o = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(gn.outputs["BSDF"], o.inputs["Surface"])
    gn.label = "NG_egg_fry"
    V.auto_layout(t)
    if not yolk:
        V.set_render_method(m, "BLENDED")
        m.use_backface_culling = False
        m.show_transparent_back = False
    return m, gn


# ----------------------------------------------------------------------------- the test mesh
def egg_white_mesh(name, R=0.042, hc=0.0065, seed=3, nr=28, ns=96):
    rng = random.Random(seed)
    ph = [rng.uniform(0, 6.28) for _ in range(6)]

    def outline(t):
        return R * (1 + 0.10 * math.sin(2 * t + ph[0]) + 0.07 * math.sin(3 * t + ph[1]) + 0.04 * math.sin(5 * t + ph[2]) + 0.02 * math.sin(7 * t + ph[3]))

    def height(rho, t):
        h = 0.0009 + (hc - 0.0009) * (1 - rho ** 2) ** 1.25
        h += 0.00045 * math.sin(3 * t + ph[4] + rho * 4) * rho * (1 - rho)
        return h
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    rings = []
    center = bm.verts.new((0, 0, height(0, 0)))
    for i in range(1, nr + 1):
        rho = i / nr
        row = []
        for j in range(ns):
            t = 2 * math.pi * j / ns
            r = rho * outline(t)
            row.append(bm.verts.new((r * math.cos(t), r * math.sin(t), height(rho, t))))
        rings.append(row)
    for j in range(ns):
        bm.faces.new((center, rings[0][(j + 1) % ns], rings[0][j]))
    for i in range(nr - 1):
        for j in range(ns):
            bm.faces.new((rings[i][j], rings[i][(j + 1) % ns], rings[i + 1][(j + 1) % ns], rings[i + 1][j]))
    # side wall + bottom
    bot = []
    for j in range(ns):
        v = rings[-1][j]
        bot.append(bm.verts.new((v.co.x * 0.985, v.co.y * 0.985, 0.0)))
    for j in range(ns):
        bm.faces.new((rings[-1][j], bot[j], bot[(j + 1) % ns], rings[-1][(j + 1) % ns]))
    bc = bm.verts.new((0, 0, 0))
    for j in range(ns):
        bm.faces.new((bc, bot[j], bot[(j + 1) % ns]))
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    # egg_rim attribute: radial 0..1, wall + bottom get 1.0 at the rim, centre 0 on the bottom as well
    at = me.attributes.new("egg_rim", "FLOAT", "POINT")
    for k, v in enumerate(me.vertices):
        rr = math.hypot(v.co.x, v.co.y)
        t = math.atan2(v.co.y, v.co.x)
        at.data[k].value = min(1.0, rr / (outline(t) * (0.985 if v.co.z < 1e-6 else 1.0)))
    for p in me.polygons:
        p.use_smooth = True
    return me


def yolk_mesh(name, r=0.0145, sq=0.62):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=r)
    bm.to_mesh(bpy.data.meshes.new(name))
    me = bpy.data.meshes[name]
    bm.free()
    for v in me.vertices:
        v.co.z *= sq
    for p in me.polygons:
        p.use_smooth = True
    return me


def build_egg(coll, root, mats, prefix="shader_egg_fry"):
    me = egg_white_mesh(prefix + "_white")
    white = bpy.data.objects.new(prefix + "_white", me)
    coll.objects.link(white)
    white.parent = root
    white.data.materials.append(mats[0])
    # shape key: cooked = rim curls up slightly, centre flattens
    white.shape_key_add(name="Basis")
    sk = white.shape_key_add(name="cooked")
    for k, v in enumerate(me.vertices):
        rho = me.attributes["egg_rim"].data[k].value
        if v.co.z > 1e-6:
            sk.data[k].co.z = v.co.z + 0.0012 * rho ** 4 - 0.0007 * (1 - rho ** 2)
            sk.data[k].co.x = v.co.x * (1 + 0.02 * rho)
            sk.data[k].co.y = v.co.y * (1 + 0.02 * rho)
    sk.slider_min = 0.0
    sk.slider_max = 1.0
    ye = bpy.data.objects.new(prefix + "_yolk", yolk_mesh(prefix + "_yolk"))
    coll.objects.link(ye)
    ye.parent = root
    ye.location = (0.004, -0.003, 0.0105)
    ye.data.materials.append(mats[1])
    # yolk sits on the white dome
    return white, ye


def bind_egg(root, objs):
    """Make materials unique per egg and drive their p_cook input (and the 'cooked' shape key) from root['p_cook']."""
    for o in objs:
        for slot in o.material_slots:
            m = slot.material
            if m is None or m.node_tree is None:
                continue
            gn = next((n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeGroup" and n.node_tree and n.node_tree.name == "NG_egg_fry"), None)
            if gn is None:
                continue
            m2 = m.copy()
            m2.name = "%s_%s" % (m.name, root.name)
            slot.material = m2
            gn2 = next(n for n in m2.node_tree.nodes if n.bl_idname == "ShaderNodeGroup")
            path = 'nodes["%s"].inputs["p_cook"].default_value' % gn2.name
            fc = m2.node_tree.driver_add(path)
            d = fc.driver
            d.type = "SCRIPTED"
            v = d.variables.new()
            v.name = "p"
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["p_cook"]'
            d.expression = "p"
        if o.type == "MESH" and o.data.shape_keys:
            fc = o.data.shape_keys.key_blocks["cooked"].driver_add("value")
            d = fc.driver
            d.type = "SCRIPTED"
            v = d.variables.new()
            v.name = "p"
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["p_cook"]'
            d.expression = "clamp((p - 0.25) / 0.6, 0, 1)"


def main():
    t0 = time.time()
    C.reset()
    ng = build_ng()
    ng.use_fake_user = True
    V.mark_asset(ng, ("egg", "shader"), "Fried egg cooking shader driven by p_cook")
    mw, nw = egg_material("MAT_vfx_egg_white", ng, False)
    my, ny = egg_material("MAT_vfx_egg_yolk", ng, True)
    for m in (mw, my):
        V.mark_asset(m, ("egg",), "Egg material")
    coll, root = C.new_asset("shader_egg_fry", accuracy="C")
    root["p_cook"] = 0.0
    root.id_properties_ui("p_cook").update(min=0.0, max=1.0, description="Cooking progress 0 raw .. 1 fully fried")
    white, yolk = build_egg(coll, root, (mw, my))
    h_steam = C.hook("steam", coll, root, loc=(0, 0, 0.02))
    h_att = C.hook("attach", coll, root, loc=(0, 0, 0.0))
    h_top = C.hook("yolk_top", coll, root, loc=(0.004, -0.003, 0.0185))
    bind_egg(root, [white, yolk])
    # keyframed sequence on the root property
    root["p_cook"] = 0.0
    root.keyframe_insert('["p_cook"]', frame=1)
    root["p_cook"] = 1.0
    root.keyframe_insert('["p_cook"]', frame=151)
    meta = {
        "category": "materials_vfx",
        "description": "Fried-egg cooking shader NG_egg_fry driven by one parameter p_cook (0..1) plus a test fried egg (white mesh with 'cooked' shape key and a squashed-sphere yolk).",
        "sources": [{"what": "Visual behaviour (white from glassy to opaque, edges golden-brown and crisp, blisters, yolk firming, oil sheen)",
                     "source": "general knowledge of pan-fried eggs; no footage was consulted; timing proposals are art-directed", "accuracy": "C"}],
        "node_groups": {"NG_egg_fry": {"inputs": [i[0] for i in INS], "outputs": ["BSDF", "Browning", "Opacity", "Steam"]}},
        "materials": ["MAT_vfx_egg_white", "MAT_vfx_egg_yolk"],
        "custom_properties": {"p_cook": "root property 0..1 (keyframe it); drives both per-egg material copies and the white's 'cooked' shape key through drivers"},
        "hooks": ["HOOK_steam (above the yolk, parent steam FX here)", "HOOK_attach (underside centre, mount point on the engine lid)", "HOOK_yolk_top"],
        "dimensions": {"white_diameter_mm": "about 84 (average of an irregular outline; estimate, typical fried egg 80-100 mm)",
                       "yolk_diameter_mm": 29, "white_centre_height_mm": 6.5,
                       "note": "Dimensions are plausible generic values (level C) and the props agent may scale the whole egg to fit an optical engine"},
        "usage": [
            "Props/assembler: for the real egg meshes use two materials from the library, MAT_vfx_egg_white (BLENDED alpha) on the white and "
            "MAT_vfx_egg_yolk on the yolk. The white needs a float POINT attribute named 'egg_rim' (0 at the centre, 1 at the outer edge of the white; "
            "see build_shader_egg_fry.egg_white_mesh); without it the shader falls back to a radial estimate from generated coordinates.",
            "Per egg state: call build_shader_egg_fry.bind_egg(root, [white, yolk]) (copy of the function in this script) to give every egg its own material copies "
            "driven by root['p_cook']; then keyframe root['p_cook'] 0 -> 1 (the test sequence uses frames 1..151). Stagger p_cook keyframes for 16 eggs.",
            "Stages (art-directed): 0.0 raw glassy; 0.10-0.45 white turns opaque from the rim inward; 0.30-1.0 golden then deep brown crisp edge advancing inward "
            "(Brown Extent); blisters from 0.12; yolk firms 0.35-0.9; oil sheen throughout, fading with opacity. Outputs 'Steam' (0..1) for the steam FX rate.",
            "Dimensions: shader is scale-independent (noise on generated coordinates); 'Noise Scale' and 'Bubble Scale' set pattern size relative to the egg.",
            "Alpha: the white uses BLENDED render method (sorted per object). Keep the yolk a separate object so it is opaque.",
        ],
        "accuracy_notes": "Level C (art-directed look).",
        "build_seconds": round(time.time() - t0, 1),
    }
    V.finish_lib("shader_egg_fry", BLEND, coll, meta)
    previews()


def previews():
    V.reopen(BLEND)
    V.prep_scene(res=(480, 360), samples=48)
    scn = bpy.context.scene
    root = bpy.data.objects["ROOT_shader_egg_fry"]
    for o in scn.objects:
        if o.type not in {"LIGHT", "CAMERA"}:
            pass
    mm = C.mm
    # pan: dark cast-iron disc with oil film
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.075, depth=0.012, location=(0, 0, -0.006))
    pan = bpy.context.active_object
    pm = V.new_material("TMP_pan")
    b = pm.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.03, 0.03, 0.032, 1)
    b.inputs["Roughness"].default_value = 0.28
    b.inputs["Metallic"].default_value = 0.6
    pan.data.materials.append(pm)
    bpy.ops.mesh.primitive_plane_add(size=0.6, location=(0, 0, -0.012))
    fl = bpy.context.active_object
    fm = V.new_material("TMP_floor")
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.25, 0.27, 0.3, 1)
    fl.data.materials.append(fm)
    V.studio_lights(center=(0, 0, 0), scale=0.12, key=1.1, world=0.6)
    V.make_camera((0.0, -0.17, 0.12), (0.0, 0.0, 0.004), lens=60)
    paths = []
    for i, p in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        scn.frame_set(1 + int(round(150 * p)))
        pth = os.path.join(SCR, "egg_f%d.png" % i)
        V.render_still(pth)
        paths.append(pth)
    sheet = os.path.join(PREV, "shader_egg_fry_contact_sheet.png")
    V.tile_pngs(paths, sheet, 3)
    for p in paths:
        os.remove(p)
    # one hero close-up at p_cook = 0.7
    scn.frame_set(1 + 105)
    scn.render.resolution_x, scn.render.resolution_y = 900, 675
    bpy.data.objects["CAM"].location = (0.0, -0.1, 0.07)
    V.render_still(os.path.join(PREV, "shader_egg_fry_closeup_p07.png"))


main()
