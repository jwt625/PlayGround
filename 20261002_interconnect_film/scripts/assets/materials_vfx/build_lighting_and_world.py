"""Build lighting_and_world: lighting rigs (collections), world datablocks (cartoon sky + others), compositor post group NG_comp_post, DOF helper rig.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/materials_vfx/build_lighting_and_world.py -- assets/components/materials_vfx
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import bmesh
import bpy
import common as C
import vfx_lib as V
from mathutils import Vector

PI = math.pi
OUT = os.path.abspath(C.argv_after_dashes()[0] if C.argv_after_dashes() else "assets/components/materials_vfx")
PREV = os.path.join(OUT, "previews")
SCR = os.environ.get("VFX_SCRATCH", "/tmp")
BLEND = os.path.join(OUT, "lighting_and_world.blend")
CLAY_BLEND = os.path.join(OUT, "materials_clay.blend")
PBR_BLEND = os.path.join(OUT, "materials_pbr_hardware.blend")
os.makedirs(PREV, exist_ok=True)
RIGS = {}


# ----------------------------------------------------------------------------- world node group
def build_world_group():
    ins = [
        ("Zenith Color", "Color", V.srgb("#2A78E8"), None, None, "Sky colour overhead (saturated cartoon blue)"),
        ("Horizon Color", "Color", V.srgb("#BFE6FF"), None, None, "Pale sky colour at the horizon"),
        ("Ground Color", "Color", V.srgb("#8C9A78"), None, None, "Below-horizon bounce colour"),
        ("Horizon Sharpness", "Float", 0.6, 0.05, 3.0, "Gradient exponent (lower = wider pale band)"),
        ("Cloud Amount", "Float", 0.3, 0.0, 1.0, "Cartoon cloud coverage (0 = clear)"),
        ("Cloud Scale", "Float", 3.0, 0.2, 20.0, "Cloud size"),
        ("Cloud Softness", "Float", 0.06, 0.005, 0.5, "Cloud edge softness (small = toon)"),
        ("Cloud Offset", "Float", 0.0, -1000.0, 1000.0, "Scrolls the clouds (keyframe for drifting)"),
        ("Strength", "Float", 1.0, 0.0, 20.0, "World light strength"),
    ]
    tree, gi, go = V.new_group("NG_world_cartoon_sky", "Shader", ins, [("Shader", "Shader")])
    g = lambda n: V.X(gi.outputs[n])
    tc = V.N("ShaderNodeTexCoord")
    gen = V.wrapv(tc.outputs["Generated"])
    gx, gy, gz = V.sepxyz(gen)
    t = gz                                      # raw world direction z: -1 (down) .. 1 (up)
    up = V.mathop("POWER", V.clamp01(t), g("Horizon Sharpness"))
    sky = V.N("ShaderNodeMix", data_type="RGBA")
    V.L(up, sky.inputs["Factor"])
    V.L(gi.outputs["Horizon Color"], sky.inputs[6])
    V.L(gi.outputs["Zenith Color"], sky.inputs[7])
    gr = V.N("ShaderNodeMix", data_type="RGBA")
    V.L(V.smoothstep(0.0, -0.25, t), gr.inputs["Factor"])
    V.L(sky.outputs[2], gr.inputs[6])
    V.L(gi.outputs["Ground Color"], gr.inputs[7])
    # clouds on a flat projection of the upper hemisphere
    den = V.maxx(t, 0.0) + 0.6
    nz = V.N("ShaderNodeTexNoise", noise_dimensions="3D", in_Detail=4.0, in_Roughness=0.55)
    V.L(V.comb(gx / den + g("Cloud Offset"), gy / den, 0.0), nz.inputs["Vector"])
    V.L(g("Cloud Scale"), nz.inputs["Scale"])
    thr = 1.0 - g("Cloud Amount")
    cl = V.smoothstep(thr, thr + g("Cloud Softness"), V.X(nz.outputs["Fac"])) * V.smoothstep(0.05, 0.4, t)
    cm = V.N("ShaderNodeMix", data_type="RGBA")
    V.L(cl, cm.inputs["Factor"])
    V.L(gr.outputs[2], cm.inputs[6])
    cm.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
    bg = V.N("ShaderNodeBackground")
    V.L(cm.outputs[2], bg.inputs["Color"])
    V.L(g("Strength"), bg.inputs["Strength"])
    tree.links.new(bg.outputs[0], go.inputs["Shader"])
    V.auto_layout(tree)
    return tree


def make_world(name, group, **vals):
    w = bpy.data.worlds.new(name)
    w.use_nodes = True
    nt = w.node_tree
    V.clear_nodes(nt)
    V.use(nt)
    gn = V.group_node(nt, group)
    for k, v in vals.items():
        gn.inputs[k.replace("__", " ")].default_value = v
    o = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(gn.outputs[0], o.inputs[0])
    w.use_fake_user = True
    return w, gn


def make_flat_world(name, color_top, color_bottom, strength):
    """Studio / interior world: simple vertical gradient (reused: group NG_world_gradient)."""
    g = bpy.data.node_groups.get("NG_world_gradient")
    if g is None:
        ins = [("Top Color", "Color", (0.5, 0.5, 0.5, 1), None, None, ""), ("Bottom Color", "Color", (0.1, 0.1, 0.1, 1), None, None, ""),
               ("Strength", "Float", 1.0, 0.0, 20.0, "")]
        g, gi, go = V.new_group("NG_world_gradient", "Shader", ins, [("Shader", "Shader")])
        tc = V.N("ShaderNodeTexCoord")
        _, _, gz = V.sepxyz(V.wrapv(tc.outputs["Generated"]))
        m = V.N("ShaderNodeMix", data_type="RGBA")
        V.L(V.smoothstep(0.35, 0.75, gz * 0.5 + 0.5), m.inputs["Factor"])
        V.L(gi.outputs["Bottom Color"], m.inputs[6])
        V.L(gi.outputs["Top Color"], m.inputs[7])
        bg = V.N("ShaderNodeBackground")
        V.L(m.outputs[2], bg.inputs["Color"])
        V.L(V.X(gi.outputs["Strength"]), bg.inputs["Strength"])
        g.links.new(bg.outputs[0], go.inputs["Shader"])
        V.auto_layout(g)
        g.use_fake_user = True
    return make_world(name, g, Top__Color=color_top, Bottom__Color=color_bottom, Strength=strength)


# ----------------------------------------------------------------------------- rigs
def new_rig(name, desc, scale_note):
    coll, root = C.new_asset("light_" + name, accuracy="C")
    root["p_energy"] = 1.0
    root["p_note"] = scale_note
    RIGS[name] = {"collection": coll.name, "root": root.name, "description": desc, "scale_note": scale_note, "lights": [], "world": None}
    return coll, root


def add_area(coll, root, rigname, name, energy_irr, size, pos, target=(0, 0, 0), size_y=None, color=(1, 1, 1), spread=None):
    """Area light placed at pos (rig-local), aimed at target; power set from the irradiance at the target: P = E * pi * d^2; scales with root scale^2 and p_energy."""
    d = (Vector(pos) - Vector(target)).length
    P = energy_irr * PI * d * d
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = P
    ld.shape = "RECTANGLE" if size_y else "SQUARE"
    ld.size = size
    if size_y:
        ld.size_y = size_y
    ld.color = color
    if spread:
        ld.spread = spread
    ld.use_shadow = True
    ld.shadow_soft_size = 0.0
    o = bpy.data.objects.new(name, ld)
    coll.objects.link(o)
    o.parent = root
    o.location = pos
    o.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    drive_energy(ld, root, P)
    RIGS[rigname]["lights"].append({"name": name, "type": "AREA", "power_W_at_scale1": round(P, 1), "size_m": size if not size_y else [size, size_y],
                                    "color": list(color), "pos_m": list(pos), "target_irradiance_estimate": energy_irr})
    return o


def drive_energy(ld, root, base):
    fc = ld.driver_add("energy")
    d = fc.driver
    d.type = "SCRIPTED"
    v = d.variables.new()
    v.name = "s"
    v.type = "TRANSFORMS"
    v.targets[0].id = root
    v.targets[0].transform_type = "SCALE_X"
    v.targets[0].transform_space = "WORLD_SPACE"
    v2 = d.variables.new()
    v2.name = "m"
    v2.type = "SINGLE_PROP"
    v2.targets[0].id = root
    v2.targets[0].data_path = '["p_energy"]'
    d.expression = "%f * s * s * m" % base


def add_sun(coll, root, rigname, name, strength, elev_deg, azim_deg, angle=0.12, color=(1.0, 0.96, 0.88)):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = strength
    ld.angle = angle
    ld.color = color
    ld.use_shadow = True
    o = bpy.data.objects.new(name, ld)
    coll.objects.link(o)
    o.parent = root
    # sun shines along -Z of the object: tilt from straight down
    o.rotation_euler = (math.radians(90 - elev_deg), 0, math.radians(azim_deg))
    fc = ld.driver_add("energy")
    d = fc.driver
    d.type = "SCRIPTED"
    v2 = d.variables.new()
    v2.name = "m"
    v2.type = "SINGLE_PROP"
    v2.targets[0].id = root
    v2.targets[0].data_path = '["p_energy"]'
    d.expression = "%f * m" % strength
    RIGS[rigname]["lights"].append({"name": name, "type": "SUN", "strength_W_m2": strength, "elevation_deg": elev_deg, "azimuth_deg": azim_deg, "angle_rad": angle})
    return o


def cyclorama(coll, root, name, w=4.0, depth=3.0, height=2.5, radius=0.6, top=(0.35, 0.4, 0.5), bottom=(0.06, 0.07, 0.09)):
    """Floor + curved wall (cyclorama) with a vertical gradient material."""
    bm = bmesh.new()
    prof = [(-depth, 0.0), (0.0, 0.0)]
    n = 10
    for i in range(1, n + 1):
        a = PI / 2 * i / n
        prof.append((radius * math.sin(a), radius * (1 - math.cos(a))))
    prof.append((radius, height))
    # profile in (y, z); extrude along x
    rows = []
    for (py, pz) in prof:
        rows.append([bm.verts.new((-w / 2, py + 0.0, pz)), bm.verts.new((w / 2, py, pz))])
    for i in range(len(rows) - 1):
        bm.faces.new((rows[i][0], rows[i][1], rows[i + 1][1], rows[i + 1][0]))
    for f in bm.faces:
        f.normal_flip()
        f.smooth = True
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = root
    o.location = (0, 1.2, 0)
    o.visible_shadow = False
    m = V.new_material("MAT_vfx_backdrop_gradient")
    t = m.node_tree
    V.clear_nodes(t)
    V.use(t)
    tc = t.nodes.new("ShaderNodeTexCoord")
    z = V.wrapv(tc.outputs["Object"]).z
    mx = t.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    V.L(V.smoothstep(0.0, height, z), mx.inputs["Factor"])
    mx.inputs[6].default_value = (*top, 1)
    mx.inputs[7].default_value = (*bottom, 1)
    pb = t.nodes.new("ShaderNodeBsdfPrincipled")
    t.links.new(mx.outputs[2], pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 0.9
    o2 = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(pb.outputs[0], o2.inputs[0])
    V.auto_layout(t)
    me.materials.append(m)
    return o


def build_rigs():
    # --- daylight: bright clay-friendly
    coll, root = new_rig("daylight", "Bright clay-friendly daylight: soft sun, big sky fill, warm front fill. Pair with WORLD_cartoon_sky. Built for a ~1 m subject.",
                         "Subject ~1-3 m. Scale ROOT uniformly for bigger sets; area light power follows scale^2, the sun is unchanged.")
    add_sun(coll, root, "daylight", "light_daylight_sun", 2.4, 52.0, -35.0, angle=0.14)
    add_area(coll, root, "daylight", "light_daylight_sky", 0.9, 8.0, (0, 0, 6.0), size_y=8.0, color=(0.82, 0.9, 1.0))
    add_area(coll, root, "daylight", "light_daylight_fill", 0.5, 3.0, (-4.0, -5.0, 1.8), color=(1.0, 0.93, 0.85))
    RIGS["daylight"]["world"] = "WORLD_cartoon_sky"
    # --- macro hardware studio
    coll, root = new_rig("macro_studio", "Macro hardware studio: large key softbox, cool fill, rim strip, top light and a gradient cyclorama backdrop (MAT_vfx_backdrop_gradient). Built for a ~0.3 m subject.",
                         "Built for a subject of ~0.3 m: scale ROOT to the subject size (e.g. 0.1 for a 3 cm chip); keep the hardware at real scale or scale it up as the assembler wishes.")
    add_area(coll, root, "macro_studio", "light_macro_key", 3.0, 0.9, (-0.9, -1.0, 0.9), target=(0, 0, 0.05), color=(1.0, 0.97, 0.93))
    add_area(coll, root, "macro_studio", "light_macro_fill", 0.7, 1.4, (1.2, -1.0, 0.4), target=(0, 0, 0.05), color=(0.85, 0.92, 1.0))
    add_area(coll, root, "macro_studio", "light_macro_rim", 3.5, 0.25, (0.9, 1.0, 0.7), target=(0, 0, 0.05), size_y=1.2, color=(1.0, 0.95, 0.9))
    add_area(coll, root, "macro_studio", "light_macro_top", 0.9, 0.9, (0, 0, 1.4), target=(0, 0, 0), color=(1, 1, 1))
    cyclorama(coll, root, "light_macro_studio_backdrop")
    RIGS["macro_studio"]["world"] = "WORLD_studio"
    # --- data hall
    coll, root = new_rig("data_hall", "Data-hall interior: four long overhead LED strips (cool white) over an aisle plus dim ambient; for rack-aisle scenes. Built for a ~6 m aisle.",
                         "Aisle length ~6 m along Y; scale ROOT for longer halls (strip lengths scale too).")
    for i, x in enumerate((-1.8, -0.6, 0.6, 1.8)):
        add_area(coll, root, "data_hall", "light_hall_strip_%d" % i, 1.2, 0.18, (x, 0, 3.0), target=(x, 0, 0.0), size_y=6.0, color=(0.88, 0.94, 1.0))
    add_area(coll, root, "data_hall", "light_hall_floor_bounce", 0.25, 5.0, (0, -1.0, 0.1), target=(0, -1.0, 2.0), color=(0.9, 0.95, 1.0))
    RIGS["data_hall"]["world"] = "WORLD_data_hall"
    # --- lab
    coll, root = new_rig("lab", "Lab lighting: two large cool-white ceiling panels and a warm desk lamp spot on the bench; flat, readable, instrument-friendly. Built for a ~4 m room.",
                         "Room ~4 m; scale ROOT for other sizes.")
    add_area(coll, root, "lab", "light_lab_panel_a", 1.6, 1.4, (-1.0, 0.0, 2.7), target=(-0.6, 0, 0.8), color=(0.92, 0.96, 1.0))
    add_area(coll, root, "lab", "light_lab_panel_b", 1.6, 1.4, (1.2, 0.0, 2.7), target=(0.8, 0, 0.8), color=(0.92, 0.96, 1.0))
    ld = bpy.data.lights.new("light_lab_desk_lamp", "SPOT")
    P = 1.5 * PI * 1.0
    ld.energy = P * 4
    ld.spot_size = 1.0
    ld.spot_blend = 0.6
    ld.color = (1.0, 0.85, 0.65)
    o = bpy.data.objects.new("light_lab_desk_lamp", ld)
    coll.objects.link(o)
    o.parent = root
    o.location = (0.5, -0.4, 1.7)
    o.rotation_euler = (Vector((0.2, 0.2, 0.9)) - Vector((0.5, -0.4, 1.7))).to_track_quat("-Z", "Y").to_euler()
    drive_energy(ld, root, P * 4)
    RIGS["lab"]["lights"].append({"name": "light_lab_desk_lamp", "type": "SPOT", "power_W_at_scale1": round(P * 4, 1), "color": [1.0, 0.85, 0.65]})
    RIGS["lab"]["world"] = "WORLD_lab"
    # --- hot chip rim (orange)
    coll, root = new_rig("hot_chip_rim", "Dramatic hot-chip rim light: two orange strips from behind, dim cool fill from the front and a red underglow; for heated optical engines / GPU close-ups. Built for a ~0.3 m subject.",
                         "Subject ~0.3 m; scale ROOT to the subject size. Power scales with scale^2.")
    add_area(coll, root, "hot_chip_rim", "light_hot_rim_left", 2.5, 0.2, (-0.8, 0.8, 0.5), target=(0, 0, 0.05), size_y=1.0, color=(1.0, 0.42, 0.08))
    add_area(coll, root, "hot_chip_rim", "light_hot_rim_right", 2.5, 0.2, (0.8, 0.8, 0.5), target=(0, 0, 0.05), size_y=1.0, color=(1.0, 0.35, 0.06))
    add_area(coll, root, "hot_chip_rim", "light_hot_fill", 0.5, 1.2, (0.2, -1.4, 0.6), target=(0, 0, 0.05), color=(0.35, 0.5, 1.0))
    add_area(coll, root, "hot_chip_rim", "light_hot_underglow", 0.4, 0.6, (0, -0.2, -0.15), target=(0, 0, 0.1), color=(1.0, 0.12, 0.04))
    RIGS["hot_chip_rim"]["world"] = "WORLD_hot_chip"


# ----------------------------------------------------------------------------- compositor group
def build_comp_group():
    ins = [
        ("Image", "Color", (0, 0, 0, 1), None, None, ""),
        ("Bloom Amount", "Float", 0.5, 0.0, 4.0, "Added glow (0 = off)"),
        ("Bloom Threshold", "Float", 1.0, 0.0, 10.0, "Brightness above which pixels glow (HDR: emission > 1)"),
        ("Vignette Amount", "Float", 0.35, 0.0, 1.0, "Edge darkening (0 = off)"),
        ("Vignette Softness", "Float", 0.35, 0.0, 1.0, "Relative blur size of the vignette mask"),
        ("CA Amount", "Float", 0.004, 0.0, 0.05, "Chromatic aberration (dispersion)"),
        ("Grain Amount", "Float", 0.12, 0.0, 1.0, "Film grain strength (0 = off)"),
        ("Grain Seed", "Float", 0.0, -1e5, 1e5, "Grain pattern offset; driven by the scene frame in the helper"),
    ]
    tree, gi, go = V.new_group("NG_comp_post", "Compositor", ins, [("Image", "Color")])
    N = tree.nodes
    L = tree.links
    gl = N.new("CompositorNodeGlare")
    gl.glare_type = "BLOOM"
    gl.threshold = 1.0
    gl.mix = 1.0
    gl.size = 7
    gl.quality = "MEDIUM"
    L.new(gi.outputs["Image"], gl.inputs["Image"])
    add = N.new("CompositorNodeMixRGB")
    add.blend_type = "ADD"
    L.new(gi.outputs["Image"], add.inputs[1])
    L.new(gl.outputs[0], add.inputs[2])
    L.new(gi.outputs["Bloom Amount"], add.inputs[0])
    # chromatic aberration
    ld = N.new("CompositorNodeLensdist")
    ld.use_fit = False
    L.new(add.outputs[0], ld.inputs["Image"])
    L.new(gi.outputs["CA Amount"], ld.inputs["Dispersion"])
    ld.inputs["Distortion"].default_value = 0.0
    # vignette: ellipse mask -> blur -> lerp(1, mask, amount) -> multiply
    em = N.new("CompositorNodeEllipseMask")
    em.mask_type = "ADD"
    em.x, em.y = 0.5, 0.5
    em.mask_width, em.mask_height = 0.92, 0.92
    em.inputs["Mask"].default_value = 0.0
    em.inputs["Value"].default_value = 1.0
    bl = N.new("CompositorNodeBlur")
    bl.filter_type = "GAUSS"
    bl.use_relative = True
    bl.aspect_correction = "NONE"
    bl.factor_x = 14.0
    bl.factor_y = 14.0
    L.new(em.outputs[0], bl.inputs["Image"])
    vm = N.new("CompositorNodeMixRGB")        # fac = amount; in1 = white, in2 = blurred mask -> factor image
    vm.blend_type = "MIX"
    vm.inputs[1].default_value = (1, 1, 1, 1)
    L.new(gi.outputs["Vignette Amount"], vm.inputs[0])
    L.new(bl.outputs[0], vm.inputs[2])
    mul = N.new("CompositorNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    L.new(ld.outputs[0], mul.inputs[1])
    L.new(vm.outputs[0], mul.inputs[2])
    # grain: noise texture, offset animated by Grain Seed
    tex = bpy.data.textures.get("TEX_vfx_grain") or bpy.data.textures.new("TEX_vfx_grain", "NOISE")
    tn = N.new("CompositorNodeTexture")
    tn.texture = tex
    tn.node_output = 1
    tn.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    L.new(gi.outputs["Grain Seed"], tn.inputs["Offset"])
    gmix = N.new("CompositorNodeMixRGB")
    gmix.blend_type = "SOFT_LIGHT"
    L.new(gi.outputs["Grain Amount"], gmix.inputs[0])
    L.new(mul.outputs[0], gmix.inputs[1])
    L.new(tn.outputs["Color"], gmix.inputs[2])
    L.new(gmix.outputs[0], go.inputs["Image"])
    V.auto_layout(tree)
    tree.use_fake_user = True
    return tree


def comp_presets():
    return {
        "off": dict(bloom=0.0, vignette=0.0, ca=0.0, grain=0.0),
        "light": dict(bloom=0.25, vignette=0.2, ca=0.0015, grain=0.04),
        "cartoon": dict(bloom=0.6, vignette=0.3, ca=0.003, grain=0.08),
        "hero_glow": dict(bloom=1.2, vignette=0.4, ca=0.005, grain=0.12),
    }


def setup_comp(scn, preset="cartoon"):
    """Assembler helper (also used by the previews): enable the compositor and put NG_comp_post between Render Layers and Composite."""
    scn.use_nodes = True
    nt = scn.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    g = nt.nodes.new("CompositorNodeGroup")
    g.node_tree = bpy.data.node_groups["NG_comp_post"]
    cp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], g.inputs["Image"])
    nt.links.new(g.outputs["Image"], cp.inputs["Image"])
    p = comp_presets()[preset]
    g.inputs["Bloom Amount"].default_value = p["bloom"]
    g.inputs["Vignette Amount"].default_value = p["vignette"]
    g.inputs["CA Amount"].default_value = p["ca"]
    g.inputs["Grain Amount"].default_value = p["grain"]
    return g


# ----------------------------------------------------------------------------- DOF rig
def build_dof_rig():
    coll, root = C.new_asset("light_dof_rig", accuracy="C")
    root["p_fstop"] = 2.8
    root["p_focus_distance"] = 3.0
    root["p_dof_on"] = 1
    f = bpy.data.objects.new("light_dof_focus_target", None)
    f.empty_display_type = "SPHERE"
    f.empty_display_size = 0.05
    coll.objects.link(f)
    f.parent = root
    f.location = (0, -3.0, 1.0)
    C.hook("focus", coll, f)
    cd = bpy.data.cameras.new("light_dof_demo_cam")
    cd.lens = 50
    cd.dof.use_dof = True
    cd.dof.focus_object = f
    cam = bpy.data.objects.new("light_dof_demo_cam", cd)
    coll.objects.link(cam)
    cam.parent = root
    cam.location = (0, -6.0, 1.2)
    cam.rotation_euler = (PI / 2, 0, 0)
    for nm, expr in (("aperture_fstop", "f if on > 0.5 else 64"),):
        fc = cd.dof.driver_add(nm)
        d = fc.driver
        d.type = "SCRIPTED"
        for vn, pn in (("f", "p_fstop"), ("on", "p_dof_on")):
            v = d.variables.new()
            v.name = vn
            v.type = "SINGLE_PROP"
            v.targets[0].id = root
            v.targets[0].data_path = '["%s"]' % pn
        d.expression = expr
    RIGS["dof_rig"] = {"collection": coll.name, "root": root.name, "description": "Depth-of-field helper: focus empty (HOOK_focus; animate it for rack focus) and a demo camera whose f-stop is driven by root p_fstop (p_dof_on = 0 disables DOF by f/64). Copy the camera settings or call render_presets.attach_dof(camera, focus_object).",
                       "lights": [], "world": None}


# ----------------------------------------------------------------------------- demo / previews
def subject(kind="full"):
    """Clay man + hardware board ('full') or the board alone centred at the origin ('hardware'), using the appended library materials."""
    M = lambda n: bpy.data.materials[n]
    r = bpy.data.objects.new("DEMO_subject", None)
    bpy.context.scene.collection.objects.link(r)

    def prim(kind_, loc, size, mat, rot=(0, 0, 0)):
        if kind_ == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16, location=loc)
        elif kind_ == "cube":
            bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=1, location=loc)
        o = bpy.context.active_object
        o.scale = size
        o.rotation_euler = rot
        if kind_ != "cube":
            V.smooth(o)
        o.data.materials.append(mat)
        o.parent = r
        o.name = "DEMO_" + o.name
        return o
    if kind == "full":
        prim("sphere", (0, 0, 0.55), (0.3, 0.24, 0.42), M("MAT_vfx_clay_shirt_blue"))
        prim("sphere", (0, 0, 1.08), (0.2, 0.2, 0.2), M("MAT_vfx_clay_skin_light"))
        prim("sphere", (0, 0, 1.2), (0.22, 0.22, 0.12), M("MAT_vfx_clay_hardhat_orange"))
        prim("sphere", (0, -0.17, 1.1), (0.05, 0.05, 0.05), M("MAT_vfx_clay_eye_white"))
        for sx in (-1, 1):
            prim("cyl", (sx * 0.1, 0, 0.17), (0.075, 0.075, 0.36), M("MAT_vfx_clay_trousers_navy"))
            prim("cyl", (sx * 0.36, 0, 0.62), (0.055, 0.055, 0.42), M("MAT_vfx_clay_skin_light"))
        ox, oy = 0.9, -0.1
        prim("cube", (-0.9, -0.2, 0.1), (0.22, 0.22, 0.2), M("MAT_vfx_clay_gray_mid"))
    else:
        ox, oy = 0.0, 0.0
    prim("cube", (ox, oy, 0.03), (0.5, 0.35, 0.03), M("MAT_vfx_soldermask_green"))
    prim("cube", (ox - 0.02, oy, 0.08), (0.2, 0.2, 0.07), M("MAT_vfx_nickel_plated"))
    prim("cube", (ox + 0.15, oy + 0.1, 0.1), (0.1, 0.12, 0.1), M("MAT_vfx_anodized_black"))
    prim("sphere", (ox - 0.15, oy - 0.18, 0.085), (0.05, 0.05, 0.05), M("MAT_vfx_gold"))
    prim("cube", (ox + 0.22, oy - 0.15, 0.065), (0.04, 0.02, 0.02), M("MAT_vfx_led_green"))
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "DEMO_floor"
    fl.data.materials.append(M("MAT_vfx_clay_gray_mid") if kind == "full" else M("MAT_vfx_clay_gray_mid"))
    fl.parent = r
    return r


def previews():
    configs = [("daylight", 1.0, ((1.0, -4.6, 1.5), (0.2, 0, 0.55), 40)),
               ("macro_studio", 0.3, ((0.2, -0.42, 0.2), (0.0, 0, 0.02), 50)),
               ("data_hall", 1.0, ((0.2, -3.8, 1.3), (0.2, 0, 0.6), 40)),
               ("lab", 1.0, ((0.5, -4.2, 1.4), (0.0, 0, 0.6), 40)),
               ("hot_chip_rim", 0.3, ((0.2, -0.42, 0.2), (0.0, 0, 0.02), 50))]
    paths = []
    for name, sc, (cl, ct, lens) in configs:
        V.reopen(BLEND)
        V.prep_scene(res=(450, 338), samples=16)
        scn = bpy.context.scene
        V.append_library(CLAY_BLEND, materials=["MAT_vfx_clay_shirt_blue", "MAT_vfx_clay_skin_light", "MAT_vfx_clay_hardhat_orange",
                                                "MAT_vfx_clay_eye_white", "MAT_vfx_clay_trousers_navy", "MAT_vfx_clay_gray_mid", "MAT_vfx_clay_white"])
        V.append_library(PBR_BLEND, materials=["MAT_vfx_soldermask_green", "MAT_vfx_nickel_plated", "MAT_vfx_anodized_black", "MAT_vfx_gold", "MAT_vfx_led_green"])
        # hide all rigs except one
        for rn, info in RIGS.items():
            if rn == "dof_rig":
                continue
            c = bpy.data.collections[info["collection"]]
            c.hide_render = rn != name
            c.hide_viewport = rn != name
        scn.world = bpy.data.worlds[RIGS[name]["world"]]
        root = bpy.data.objects[RIGS[name]["root"]]
        root.scale = (sc, sc, sc)
        if name in ("macro_studio", "hot_chip_rim"):
            sub = subject("hardware")
            sub.scale = (0.3, 0.3, 0.3)
        else:
            sub = subject("full")
        if name == "data_hall":
            for o in bpy.data.objects:
                if o.name == "DEMO_floor":
                    o.hide_render = False
        V.make_camera(cl, ct, lens=lens)
        scn.eevee.shadow_ray_count = 2
        scn.eevee.shadow_step_count = 8
        p = os.path.join(SCR, "light_%s.png" % name)
        V.render_still(p)
        paths.append(p)
    # comp demonstration: daylight rig + cartoon_preset on a glowing object
    V.reopen(BLEND)
    V.prep_scene(res=(450, 338), samples=16)
    scn = bpy.context.scene
    V.append_library(CLAY_BLEND, materials=["MAT_vfx_clay_shirt_blue", "MAT_vfx_clay_skin_light", "MAT_vfx_clay_hardhat_orange",
                                            "MAT_vfx_clay_eye_white", "MAT_vfx_clay_trousers_navy", "MAT_vfx_clay_gray_mid", "MAT_vfx_clay_white"])
    V.append_library(PBR_BLEND, materials=["MAT_vfx_soldermask_green", "MAT_vfx_nickel_plated", "MAT_vfx_anodized_black", "MAT_vfx_gold", "MAT_vfx_led_green"])
    for rn, info in RIGS.items():
        if rn == "dof_rig":
            continue
        c = bpy.data.collections[info["collection"]]
        c.hide_render = rn != "daylight"
    scn.world = bpy.data.worlds["WORLD_cartoon_sky"]
    subject("full")
    em = V.new_material("TMP_emit")
    t = em.node_tree
    V.clear_nodes(t)
    e = t.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (1.0, 0.4, 0.05, 1)
    e.inputs["Strength"].default_value = 8.0
    oo = t.nodes.new("ShaderNodeOutputMaterial")
    t.links.new(e.outputs[0], oo.inputs[0])
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0.45, -0.5, 0.9))
    s_ = bpy.context.active_object
    s_.data.materials.append(em)
    V.make_camera((1.0, -4.6, 1.5), (0.2, 0, 0.55), lens=40)
    setup_comp(scn, "cartoon")
    scn.eevee.shadow_ray_count = 2
    p = os.path.join(SCR, "light_comp.png")
    V.render_still(p)
    paths.append(p)
    sheet = os.path.join(PREV, "lighting_and_world_contact_sheet.png")
    V.tile_pngs(paths, sheet, 3)
    for p in paths:
        os.remove(p)
    # sky world alone (wide)
    V.reopen(BLEND)
    V.prep_scene(res=(900, 450), samples=16)
    scn = bpy.context.scene
    scn.world = bpy.data.worlds["WORLD_cartoon_sky"]
    for c in scn.collection.children:
        c.hide_render = True
    V.make_camera((0, 0, 0), (0, 5, 2.5), lens=24)
    scn.world.node_tree.nodes  # noqa
    scn.render.film_transparent = False
    V.render_still(os.path.join(PREV, "lighting_and_world_sky.png"))


def main():
    t0 = time.time()
    C.reset()
    wg = build_world_group()
    wg.use_fake_user = True
    w1, _ = make_world("WORLD_cartoon_sky", wg)
    w2, _ = make_flat_world("WORLD_studio", (0.55, 0.58, 0.64, 1), (0.04, 0.045, 0.055, 1), 0.7)
    w3, _ = make_flat_world("WORLD_data_hall", (0.12, 0.14, 0.18, 1), (0.03, 0.035, 0.045, 1), 0.7)
    w4, _ = make_flat_world("WORLD_lab", (0.65, 0.7, 0.78, 1), (0.28, 0.3, 0.34, 1), 0.7)
    w5, _ = make_flat_world("WORLD_hot_chip", (0.02, 0.025, 0.04, 1), (0.01, 0.01, 0.012, 1), 0.5)
    for w in (w1, w2, w3, w4, w5):
        V.mark_asset(w, ("world",), "World")
    bpy.context.scene.world = w1
    build_rigs()
    build_dof_rig()
    cg = build_comp_group()
    V.mark_asset(cg, ("compositor",), "Post: bloom, vignette, chromatic aberration, grain")
    # asset-level hooks: save the root registry
    C.save(BLEND)
    meta = {
        "category": "materials_vfx",
        "description": "Lighting rigs (collections ASSET_light_<name>), world datablocks, compositor post group NG_comp_post and DOF helper rig.",
        "sources": [],
        "rigs": RIGS,
        "worlds": {"WORLD_cartoon_sky": "NG_world_cartoon_sky (gradient + toon clouds)", "WORLD_studio": "NG_world_gradient", "WORLD_data_hall": "NG_world_gradient",
                   "WORLD_lab": "NG_world_gradient", "WORLD_hot_chip": "NG_world_gradient"},
        "node_groups": {"NG_world_cartoon_sky": [i[0] for i in []], "NG_comp_post": {"inputs": ["Image", "Bloom Amount", "Bloom Threshold", "Vignette Amount", "Vignette Softness",
                                                                                           "CA Amount", "Grain Amount", "Grain Seed"], "outputs": ["Image"]}},
        "comp_presets": comp_presets(),
        "custom_properties": {"p_energy": "on each rig ROOT: multiplies all light energies", "p_fstop / p_focus_distance / p_dof_on": "on ROOT_light_dof_rig"},
        "usage": [
            "Append a rig collection ASSET_light_<name> and set the scene world to WORLD_<...> listed in its entry (append the world and its node groups too). Scale the ROOT empty uniformly to the subject size: area light power follows scale^2 through drivers, the sun does not.",
            "Compositor: call lighting_and_world helper setup_comp(scene, 'cartoon') (copy of the function in build_lighting_and_world.py, also exposed in render_presets.py) or place the NG_comp_post group node between Render Layers and Composite; bloom threshold is fixed at 1.0 inside the group (emission > 1 glows); presets are in 'comp_presets'. "
            "Grain is the legacy Noise texture, scrolled with Grain Seed; to animate set a driver Grain Seed = frame on the group node input.",
            "Shadows / AO: area light shadows use 2-3 shadow rays and 8 steps (render_presets sets this); EEVEE Next AO comes from 'Fast GI' (off in the draft preset, on in 'hero'). Sun soft shadow comes from the sun angle (0.14 rad).",
            "DOF: ROOT_light_dof_rig carries a focus empty (HOOK_focus; animate for rack focus) and a camera with f-stop driven by p_fstop; or call render_presets.attach_dof(camera, focus_object, fstop).",
        ],
        "accuracy_notes": "Artistic rigs; power values are irradiance targets (W/m2) at the subject converted to area-light watts, not photometric measurements.",
        "build_seconds": round(time.time() - t0, 1),
    }
    C.write_meta(os.path.splitext(BLEND)[0] + ".json", meta)
    print("SAVED", BLEND)
    previews()


main()
