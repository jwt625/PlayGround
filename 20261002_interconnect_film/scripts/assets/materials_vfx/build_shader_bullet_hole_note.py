"""Build shader_bullet_hole_note: NG_cutout_tube, a small generic alpha-cutout node group (cylinder in object space, soft rim).
The characters agent owns the real bullet-hole shader; this is only a reusable building block.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_shader_bullet_hole_note.py -- assets/components/materials_vfx
"""
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
BLEND = os.path.join(OUT, "shader_bullet_hole_note.blend")


def build_ng():
    ins = [
        ("Shader", "Shader", None, None, None, "Surface shader to cut (plug the clay / any shader here)"),
        ("Center", "Vector", (0.0, 0.0, 0.0), None, None, "Cylinder axis point in object space"),
        ("Axis", "Vector", (0.0, 1.0, 0.0), None, None, "Cylinder axis direction in object space (default +Y: through Gary's chest front to back)"),
        ("Radius", "Float", 0.065, 0.0, 100.0, "Cylinder radius in object-space units (metres at identity scale); animate for the slow shrink"),
        ("Softness", "Float", 0.004, 0.0, 10.0, "Soft rim half-width (same units); with dithered alpha this gives a fuzzy edge"),
        ("Half Length", "Float", 100.0, 0.0, 1000.0, "Cut only within this distance along the axis from Center (short = pit, large = through hole)"),
        ("Invert", "Float", 0.0, 0.0, 1.0, "0 = remove inside the tube (hole); 1 = keep only inside the tube"),
        ("Rim Width", "Float", 0.012, 0.0, 10.0, "Width of the Rim output band just outside the hole"),
    ]
    tree, gi, go = V.new_group("NG_cutout_tube", "Shader", ins, [("Shader", "Shader"), ("Mask", "Float"), ("Rim", "Float")])
    g = lambda n: V.X(gi.outputs[n])
    tc = V.N("ShaderNodeTexCoord")
    p = V.wrapv(tc.outputs["Object"]) - V.wrapv(gi.outputs["Center"])
    a = V.vnorm(gi.outputs["Axis"])
    along = V.vdot(p, a)
    radial = p - V.vscale(a, along)
    r = V.vlen(radial)
    inside_r = 1.0 - V.smoothstep(g("Radius") - g("Softness"), g("Radius") + g("Softness"), r)
    inside_l = 1.0 - V.smoothstep(g("Half Length") - g("Softness"), g("Half Length") + g("Softness"), V.absx(along))
    inside = inside_r * inside_l
    keep = V.lerp(1.0 - inside, inside, g("Invert"))
    band = V.smoothstep(g("Radius") + g("Rim Width") + g("Softness"), g("Radius") + g("Softness"), r) * inside_l
    tr = V.N("ShaderNodeBsdfTransparent")
    mx = V.N("ShaderNodeMixShader")
    V.L(keep, mx.inputs["Fac"])
    tree.links.new(tr.outputs[0], mx.inputs[1])
    tree.links.new(gi.outputs["Shader"], mx.inputs[2])
    tree.links.new(mx.outputs[0], go.inputs["Shader"])
    V.L(keep, go.inputs["Mask"])
    V.L(band * (1.0 - inside), go.inputs["Rim"])
    V.auto_layout(tree)
    return tree


def main():
    t0 = time.time()
    C.reset()
    ng = build_ng()
    ng.use_fake_user = True
    V.mark_asset(ng, ("cutout",), "Alpha cutout by object-space cylinder with soft rim")
    # demo material: clay-ish diffuse -> NG_cutout_tube
    m = V.new_material("MAT_vfx_cutout_tube_demo")
    m.use_fake_user = True
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    bs = t.nodes.new("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = V.srgb("#EDB98F")
    bs.inputs["Roughness"].default_value = 0.8
    gn = V.group_node(t, ng)
    t.links.new(bs.outputs["BSDF"], gn.inputs["Shader"])
    o = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(gn.outputs["Shader"], o.inputs["Surface"])
    V.auto_layout(t)
    V.set_render_method(m, "DITHERED")
    coll, root = C.new_asset("shader_bullet_hole_note", accuracy="C")
    root["p_radius_note"] = "animate NG_cutout_tube input Radius"
    d = C.cube("shader_bullet_hole_note_demo_block", (0.2, 0.1, 0.2), loc=(0, 0, 0.1), mat=m)
    C.add(d, coll, root)
    meta = {
        "category": "materials_vfx",
        "description": "Small generic alpha-cutout node group NG_cutout_tube (object-space cylinder, soft rim) for others to reuse. "
                       "The characters agent owns the actual bullet-hole shader.",
        "sources": [],
        "node_groups": {"NG_cutout_tube": {"inputs": [i[0] for i in
                                                      [("Shader",), ("Center",), ("Axis",), ("Radius",), ("Softness",), ("Half Length",), ("Invert",), ("Rim Width",)]],
                                           "outputs": ["Shader", "Mask", "Rim"]}},
        "materials": ["MAT_vfx_cutout_tube_demo"],
        "custom_properties": {},
        "hooks": [],
        "usage": [
            "Wire any shader into the group's 'Shader' input and the group's 'Shader' output to Material Output. Set the material "
            "surface_render_method to DITHERED (V.set_render_method) so the transparent part cuts the surface and its shadow "
            "(use_transparent_shadow = True).",
            "Centre/Axis/Radius are in OBJECT space, so the hole stays glued to the body when it moves. Animate 'Radius' for appearance (0 -> r at the hit "
            "frame) and the slow shrink; 'Half Length' < the body thickness makes a blind pit instead of a through hole.",
            "'Rim' output (0..1 band just outside the cut edge) can darken or tint the edge; 'Mask' is 1 where the surface is kept.",
            "Multiple holes: chain several NG_cutout_tube groups (feed one group's Shader output into the next group's Shader input).",
            "The clay thumbprint bump of NG_clay is unaffected. The hole has no wall geometry: the back of the body shows through; for an "
            "interior wall the characters agent can model a thin tube or use a backface-visible dark material.",
        ],
        "accuracy_notes": "Utility shader, no physical dimensions. Defaults assume Gary-size units (0.065 m radius).",
        "build_seconds": round(time.time() - t0, 1),
    }
    V.finish_lib("shader_bullet_hole_note", BLEND, coll, meta)
    previews()


def previews():
    V.reopen(BLEND)
    V.prep_scene(res=(900, 675), samples=48)
    scn = bpy.context.scene
    for o in scn.objects:
        o.hide_render = True
    ng = bpy.data.node_groups["NG_cutout_tube"]
    # three blocks: hole, soft hole, partial pit; clay-like block with sphere ends
    base = bpy.data.materials["MAT_vfx_cutout_tube_demo"]
    variants = [("through", {}), ("soft", {"Softness": 0.03}), ("pit", {"Half Length": 0.025, "Center": (0.0, -0.2, 0.0), "Radius": 0.09}), ("shrunk", {"Radius": 0.03})]
    for i, (nm, ov) in enumerate(variants):
        m = base.copy()
        m.name = "TMP_" + nm
        for n in m.node_tree.nodes:
            if n.bl_idname == "ShaderNodeGroup":
                for k, v in ov.items():
                    n.inputs[k].default_value = v
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(-0.75 + i * 0.5, 0, 0.2), segments=48, ring_count=24)
        o = bpy.context.active_object
        V.smooth(o)
        o.data.materials.append(m)
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0.6, 0.2), rotation=(1.5708, 0, 0))
    bk = bpy.context.active_object
    bm = V.new_material("TMP_bk")
    bm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.35, 0.8, 1)
    bk.data.materials.append(bm)
    bk.visible_shadow = False
    V.studio_lights(center=(0, 0, 0.2), scale=1.0, key=1.0, world=0.6)
    V.make_camera((0, -2.6, 0.4), (0, 0, 0.2), lens=50)
    V.render_still(os.path.join(PREV, "shader_bullet_hole_note_cutout.png"))


main()
