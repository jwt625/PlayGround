"""Build materials_clay: NG_clay node group + MAT_vfx_clay_<color> materials + previews.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_materials_clay.py -- assets/components/materials_vfx
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bpy
import common as C
import vfx_lib as V

OUT = C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/materials_vfx"
OUT = os.path.abspath(OUT)
PREV = os.path.join(OUT, "previews")
os.makedirs(PREV, exist_ok=True)

# ----------------------------------------------------------------------------- the node group
INPUTS = [
    ("Base Color", "Color", V.srgb("#E8B48E"), None, None, "Clay colour (sRGB picked, stored linear)"),
    ("Roughness", "Float", 0.78, 0.0, 1.0, "Matte clay: 0.7-0.9"),
    ("Specular", "Float", 0.30, 0.0, 1.0, "Specular IOR level; clay is slightly waxy"),
    ("Thumbprint Strength", "Float", 0.5, 0.0, 3.0, "Bump strength of broad dimples and pokes (0 = smooth)"),
    ("Thumbprint Scale", "Float", 14.0, 0.5, 400.0, "Dimple frequency per metre of object space; raise for small objects"),
    ("Detail Scale", "Float", 70.0, 1.0, 1500.0, "Fine surface unevenness frequency per metre"),
    ("Fingerprint Ridges", "Float", 0.12, 0.0, 1.0, "Strength of fingerprint ridge patches (0 = off)"),
    ("Fingerprint Scale", "Float", 70.0, 5.0, 2000.0, "Ridge frequency per metre of object space"),
    ("Color Variation", "Float", 0.05, 0.0, 0.5, "Subtle mottling of the clay colour"),
    ("SSS Weight", "Float", 0.12, 0.0, 1.0, "Subsurface scattering weight (soft clay look)"),
    ("SSS Scale", "Float", 0.02, 0.0, 1.0, "Subsurface scale in metres"),
    ("Sheen Weight", "Float", 0.20, 0.0, 1.0, "Soft velvet sheen at grazing angles"),
    ("Rim Weight", "Float", 0.0, 0.0, 20.0, "Toon rim light strength (0 = off); emissive, EEVEE-cheap"),
    ("Rim Color", "Color", (1.0, 0.85, 0.65, 1.0), None, None, "Toon rim colour"),
    ("Rim Edge", "Float", 0.62, 0.0, 1.0, "Rim threshold in 1 - facing ratio; lower = wider rim"),
    ("Rim Softness", "Float", 0.08, 0.001, 0.5, "Rim transition width (small = hard toon edge)"),
    ("Seed", "Float", 0.0, 0.0, 1000.0, "Offsets the noise so two clay objects do not share the same thumbprints"),
]


def build_ng_clay():
    tree, gi, go = V.new_group("NG_clay", "Shader", INPUTS, [("BSDF", "Shader"), ("Color", "Color"), ("Height", "Float")])
    g = lambda n: V.X(gi.outputs[n])
    # coordinates: object space (so thumbprints stay glued to the object), offset by Seed
    tc = V.N("ShaderNodeTexCoord")
    seedoff = V.vscale(V.comb(1.0, 2.3, 3.7), g("Seed"))
    pos = V.wrapv(tc.outputs["Object"]) + seedoff

    # broad dimples: noise + smooth voronoi (poke marks)
    n1 = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=3.0, in_Roughness=0.55, in_Distortion=0.3)
    V.L(pos, n1.inputs["Vector"])
    V.L(g("Thumbprint Scale"), n1.inputs["Scale"])
    vor = V.N("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="SMOOTH_F1", in_Randomness=1.0, in_Smoothness=0.7)
    V.L(pos, vor.inputs["Vector"])
    V.L(g("Thumbprint Scale") * 0.6, vor.inputs["Scale"])
    pokes = 1.0 - V.clamp01(V.X(vor.outputs["Distance"]) * 1.15)
    broad = V.X(n1.outputs["Fac"]) * 0.55 + pokes * 0.45

    # fine unevenness
    n2 = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=5.0, in_Roughness=0.6)
    V.L(pos, n2.inputs["Vector"])
    V.L(g("Detail Scale"), n2.inputs["Scale"])
    fine = V.X(n2.outputs["Fac"])

    # fingerprint ridges: distorted wave rings, masked into patches by low-frequency noise
    warp = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=2.0, in_Roughness=0.5)
    V.L(pos, warp.inputs["Vector"])
    V.L(g("Fingerprint Scale") * 0.04, warp.inputs["Scale"])
    wv = V.N("ShaderNodeTexWave", wave_type="RINGS", rings_direction="SPHERICAL", wave_profile="SIN", in_Distortion=3.0,
             in_Detail=0.0, in_Detail__Scale=1.0, in_Detail__Roughness=0.5)
    V.L(pos, wv.inputs["Vector"])
    V.L(g("Fingerprint Scale"), wv.inputs["Scale"])
    mk = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=1.0, in_Roughness=0.5)
    V.L(pos + V.comb(7.0, 3.0, 1.0), mk.inputs["Vector"])
    V.L(g("Thumbprint Scale") * 0.5, mk.inputs["Scale"])
    patch = V.smoothstep(0.50, 0.62, V.X(mk.outputs["Fac"]))
    ridges = V.X(wv.outputs["Color"]) * patch * g("Fingerprint Ridges")
    # warp rings a bit (shift wave phase by noise)
    V.L(V.X(warp.outputs["Fac"]) * 3.0, wv.inputs["Phase Offset"])

    height = broad * 1.0 + fine * 0.06 + ridges * 0.3
    bump = V.N("ShaderNodeBump", in_Distance=0.01)
    V.L(g("Thumbprint Strength"), bump.inputs["Strength"])
    V.L(height, bump.inputs["Height"])

    # colour mottling: multiply by 1 + (noise - 0.5) * variation
    mn = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=2.0, in_Roughness=0.5)
    V.L(pos + V.comb(11.0, 5.0, 2.0), mn.inputs["Vector"])
    mn.inputs["Scale"].default_value = 9.0
    mott = 1.0 + (V.X(mn.outputs["Fac"]) - 0.5) * g("Color Variation") * 2.0
    mix = V.N("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = 1.0
    V.L(gi.outputs["Base Color"], mix.inputs[6])
    V.L(V.comb(mott, mott, mott), mix.inputs[7])
    color = mix.outputs[2]

    # toon rim: (1 - |N.I|) thresholded
    geo = V.N("ShaderNodeNewGeometry")
    fac = V.absx(V.vdot(V.wrapv(geo.outputs["Normal"]), V.wrapv(geo.outputs["Incoming"])))
    rim = V.smoothstep(g("Rim Edge") - g("Rim Softness"), g("Rim Edge") + g("Rim Softness"), 1.0 - fac)

    pb = V.N("ShaderNodeBsdfPrincipled", subsurface_method="RANDOM_WALK")
    V.L(color, pb.inputs["Base Color"])
    V.L(g("Roughness"), pb.inputs["Roughness"])
    V.L(g("Specular"), pb.inputs["Specular IOR Level"])
    V.L(g("SSS Weight"), pb.inputs["Subsurface Weight"])
    V.L(g("SSS Scale"), pb.inputs["Subsurface Scale"])
    pb.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.25)
    V.L(g("Sheen Weight"), pb.inputs["Sheen Weight"])
    pb.inputs["Sheen Roughness"].default_value = 0.5
    pb.inputs["Sheen Tint"].default_value = (1.0, 0.95, 0.9, 1.0)
    V.L(bump.outputs["Normal"], pb.inputs["Normal"])
    V.L(gi.outputs["Rim Color"], pb.inputs["Emission Color"])
    V.L(rim * g("Rim Weight"), pb.inputs["Emission Strength"])

    tree.links.new(pb.outputs["BSDF"], go.inputs["BSDF"])
    tree.links.new(color, go.inputs["Color"])
    V.L(height, go.inputs["Height"])
    V.auto_layout(tree)
    return tree


# ----------------------------------------------------------------------------- the ready materials
# name: (sRGB hex, overrides)
CLAY = {
    # skin
    "skin_pale": ("#F4CBAE", {}),
    "skin_light": ("#EDB98F", {}),
    "skin_tan": ("#C98F63", {}),
    "skin_brown": ("#8D5A3B", {}),
    "skin_dark": ("#5C3A27", {}),
    "skin_pink": ("#F29C8C", {}),          # Manager base
    "skin_red": ("#E5352A", {}),           # Manager angry (keyframe Base Color between skin_pink and this)
    # shirts / clothing
    "shirt_blue": ("#2E62D9", {}),
    "shirt_red": ("#D93A32", {}),
    "shirt_green": ("#5FAE2E", {}),
    "shirt_yellow": ("#F2C230", {}),
    "shirt_purple": ("#7B4FC9", {}),
    "shirt_orange": ("#F28A2A", {}),
    "shirt_teal": ("#1FA7A0", {}),
    "shirt_pink": ("#E86FA0", {}),
    "shirt_dark": ("#2B2D3A", {}),
    "shirt_white": ("#EDEDE8", {}),
    "trousers_navy": ("#27345E", {}),
    "trousers_khaki": ("#B59A6A", {}),
    "suit_gray": ("#6B7080", {}),
    "suit_dark": ("#33364A", {}),
    "tie_red": ("#C71F2B", {}),
    "shoe_black": ("#1E1C20", {"Roughness": 0.55, "Specular": 0.5}),
    # headgear
    "hardhat_orange": ("#FF7A1A", {"Roughness": 0.5, "Specular": 0.5, "Thumbprint Strength": 0.2}),
    "hardhat_yellow": ("#F7D21E", {"Roughness": 0.5, "Specular": 0.5, "Thumbprint Strength": 0.2}),
    "hardhat_white": ("#ECECE6", {"Roughness": 0.5, "Specular": 0.5, "Thumbprint Strength": 0.2}),
    # hair
    "hair_black": ("#1D1713", {"Roughness": 0.65}),
    "hair_brown": ("#5A3A22", {"Roughness": 0.65}),
    "hair_blond": ("#D5A94E", {"Roughness": 0.65}),
    "hair_gray": ("#9A9A9E", {"Roughness": 0.65}),
    # face parts
    "eye_white": ("#F4F1EA", {"Roughness": 0.35, "Specular": 0.6, "Thumbprint Strength": 0.0, "Fingerprint Ridges": 0.0}),
    "pupil_black": ("#09090B", {"Roughness": 0.18, "Specular": 0.8, "Thumbprint Strength": 0.0, "Fingerprint Ridges": 0.0,
                                 "SSS Weight": 0.0, "Sheen Weight": 0.0}),
    "mouth_dark": ("#3A0D12", {"Roughness": 0.6, "Thumbprint Strength": 0.1}),
    "tongue_pink": ("#C9485B", {"Roughness": 0.5, "Thumbprint Strength": 0.1}),
    "brow_dark": ("#241812", {}),
    # props
    "wood_stock": ("#8A5A2B", {"Roughness": 0.7}),
    "gun_metal": ("#7C7F88", {"Roughness": 0.55, "Specular": 0.5, "Thumbprint Strength": 0.25}),
    "bench_wood": ("#B9854F", {"Roughness": 0.75}),
    "egg_white": ("#F3F0E4", {}),
    "egg_yolk": ("#F4B323", {"Roughness": 0.6}),
    "paper_white": ("#F5F3EC", {"Thumbprint Strength": 0.0, "Fingerprint Ridges": 0.0, "Roughness": 0.9, "SSS Weight": 0.0}),
    "smoke_gray": ("#D9D9DC", {"Roughness": 0.9, "Thumbprint Strength": 0.15, "Sheen Weight": 0.0}),
    "gray_mid": ("#8A8A90", {}),
    "white": ("#F2F2F0", {}),
    "black": ("#17171A", {}),
}


def build_materials(ng):
    mats = []
    for name, (hx, ov) in CLAY.items():
        m = V.new_material("MAT_vfx_clay_" + name)
        m.use_fake_user = True
        t = m.node_tree
        V.clear_nodes(t)
        V.use(t)
        gn = V.group_node(t, ng)
        gn.inputs["Base Color"].default_value = V.srgb(hx)
        for k, v in ov.items():
            gn.inputs[k].default_value = v
        # per-material seed so no two share identical thumbprints
        gn.inputs["Seed"].default_value = float(len(mats) * 7 % 97)
        o = t.nodes.new("ShaderNodeOutputMaterial")
        t.links.new(gn.outputs["BSDF"], o.inputs["Surface"])
        gn.label = "NG_clay"
        V.auto_layout(t)
        mats.append(m)
        m.diffuse_color = V.srgb(hx)  # viewport colour
    return mats


BLEND = os.path.join(OUT, "materials_clay.blend")


def sheets():
    paths = []
    names_all = [n for n in bpy.data.materials.keys() if n.startswith("MAT_vfx_clay_") and "demo" not in n]
    groups = [names_all[:24], names_all[24:]]
    for gi_, grp in enumerate(groups):
        V.reopen(BLEND)
        V.prep_scene(samples=24)
        ms = [bpy.data.materials[n] for n in grp]
        p = os.path.join(PREV, "materials_clay_swatches_%d.png" % (gi_ + 1))
        V.material_sheet(ms, p, cols=6, pitch=1.25, radius=0.5, label_size=0.085)
        paths.append(p)
    return paths


def thumb_closeup():
    """Variants of thumbprint parameters on large spheres, for judging the bump."""
    V.reopen(BLEND)
    V.prep_scene(samples=32)
    ng = bpy.data.node_groups["NG_clay"]
    variants = [
        ("default", {}),
        ("strong", {"Thumbprint Strength": 1.6, "Fingerprint Ridges": 0.4}),
        ("ridges", {"Thumbprint Strength": 0.35, "Fingerprint Ridges": 1.0}),
        ("rim toon", {"Rim Weight": 2.5, "Rim Color": (1.0, 0.8, 0.5, 1)}),
        ("smooth", {"Thumbprint Strength": 0.0, "Fingerprint Ridges": 0.0, "Color Variation": 0.0}),
        ("sss high", {"SSS Weight": 0.6, "SSS Scale": 0.1}),
    ]
    mats = []
    for nm, ov in variants:
        m = V.new_material("MAT_vfx_clay_demo_" + nm.replace(" ", "_"))
        t = m.node_tree
        V.clear_nodes(t)
        gn = V.group_node(t, ng)
        gn.inputs["Base Color"].default_value = V.srgb("#F29C8C")
        for k, v in ov.items():
            gn.inputs[k].default_value = v
        o = t.nodes.new("ShaderNodeOutputMaterial")
        t.links.new(gn.outputs["BSDF"], o.inputs["Surface"])
        mats.append(m)
    p = os.path.join(PREV, "materials_clay_thumbprint.png")
    V.material_sheet(mats, p, cols=3, pitch=1.25, radius=0.5, label_size=0.08, fit=1.0)
    # raking-light close-up of the default material (bump check)
    V.reopen(BLEND)
    V.prep_scene(samples=48)
    V.simple_world((0.5, 0.55, 0.65), 0.25)
    V.area_light("rake", 260, 1.0, (-2.5, -1.2, 0.9), target=(0, 0, 0))
    V.area_light("fillx", 40, 2.0, (2.5, -3.0, 0.5), target=(0, 0, 0))
    mm = []
    for nm, ov in (("default", {}), ("ridges", {"Fingerprint Ridges": 1.0, "Fingerprint Scale": 70.0})):
        m = V.new_material("MAT_vfx_clay_demo_close_" + nm)
        t = m.node_tree
        V.clear_nodes(t)
        gn = V.group_node(t, bpy.data.node_groups["NG_clay"])
        gn.inputs["Base Color"].default_value = V.srgb("#EDB98F")
        for k, v in ov.items():
            gn.inputs[k].default_value = v
        o = t.nodes.new("ShaderNodeOutputMaterial")
        t.links.new(gn.outputs["BSDF"], o.inputs["Surface"])
        mm.append(m)
    V.material_sheet(mm, os.path.join(PREV, "materials_clay_closeup_raking.png"), cols=2, pitch=1.05, radius=0.5,
                     labels=False, lights=False, samples=48)
    return p


def main():
    t0 = time.time()
    C.reset()
    ng = build_ng_clay()
    ng.use_fake_user = True
    V.mark_asset(ng, ("clay",), "Clay shader: thumbprint bump, SSS, optional toon rim")
    mats = build_materials(ng)
    for m in mats:
        V.mark_asset(m, ("clay",), "Ready clay material")
    coll, root = C.new_asset("materials_clay", accuracy="C")
    root["p_note"] = "library asset: materials and node group only; spheres are a demo row"
    for i, m in enumerate(mats):
        o = C.sphere("materials_clay_demo_%02d" % i, 0.05, loc=((i % 12) * 0.12, 0, (i // 12) * 0.12 + 0.05), mat=m)
        C.add(o, coll, root)
    meta = {
        "category": "materials_vfx",
        "description": "Clay shading for characters and gag props: node group NG_clay and ready MAT_vfx_clay_<color> materials.",
        "sources": [],
        "node_groups": {"NG_clay": {"kind": "shader", "inputs": [i[0] for i in INPUTS], "outputs": ["BSDF", "Color", "Height"]}},
        "materials": [m.name for m in mats],
        "custom_properties": {},
        "hooks": [],
        "usage": [
            "Append: bpy.data.libraries.load(path, link=False) with node_groups=['NG_clay'] and the wanted materials (materials pull in NG_clay).",
            "Per-object tweaks: duplicate the material, then set the Group node inputs (Base Color, Thumbprint Scale, Rim Weight, ...). "
            "Keyframe 'Base Color' on the group node input to animate colour, e.g. the Manager turning from MAT_vfx_clay_skin_pink to skin_red.",
            "Thumbprint noise uses Object coordinates: it scales with the object and stays glued while the object moves. Objects of very "
            "different size need a different 'Thumbprint Scale' (default 14 per metre suits a roughly 1 m character; use ~140 for 10 cm props) "
            "and 'Fingerprint Scale'.",
            "Toon rim: set 'Rim Weight' > 0 (2-3 typical). It is emissive and does not depend on lights.",
            "Bullet-hole cutouts: put the hole shader (characters agent) after the group; NG_clay outputs a plain Shader (see NG_cutout_tube).",
        ],
        "accuracy_notes": "Stylised look, no physical dimension claims. Clay colours are art-directed sRGB hex values chosen for this film.",
        "build_seconds": round(time.time() - t0, 1),
    }
    V.finish_lib("materials_clay", BLEND, coll, meta)
    sheets()
    thumb_closeup()


main()
