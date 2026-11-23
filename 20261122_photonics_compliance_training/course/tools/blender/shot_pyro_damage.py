"""Shot 4: pyroelectric discharge damage on interdigitated (comb) gold electrodes on x-cut LNOI,
as seen in a bright-field reflected-light microscope (coaxial illumination, top-down).

Units in this scene: 1 Blender unit = 1 um. Orthographic camera, field 200 um wide.
Electrodes: two bus bars with interleaved fingers, 4 um wide, 4 um gaps (8 um pitch per
electrode pair half-period), 200 nm Au. Damage: one main arc site where three fingers were
partly vaporized (beaded, resolidified ends, a molten bridge, splatter, darkened halo on the
substrate) and one minor site (single notched finger tip with a small halo).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402

A = C.args(out="/tmp/pyro.jpg", samples=128, res=1920, noise=0.015, look="AgX - Medium High Contrast")
H_AU = 0.2
FW, GAP = 4.0, 4.0
rnd = random.Random(7)


def mats():
    m = {}
    m["sub"] = C.principled("lnoi", color=(0.26, 0.409, 0.323), metallic=1.0, rough=0.06, rough_var=0.02,
                            rough_scale=0.05)
    m["au"] = C.principled("au", color=(1.0, 0.76, 0.30), metallic=1.0, rough=0.05, rough_var=0.02,
                           rough_scale=0.8)
    m["au_smooth"] = C.principled("au_smooth", color=(1.0, 0.76, 0.30), metallic=1.0, rough=0.06)
    m["au_melt"] = C.principled("au_melt", color=(0.95, 0.62, 0.25), metallic=1.0, rough=0.22, rough_var=0.08,
                                rough_scale=1.5, bump=0.6, bump_scale=2.0, bump_dist=0.08)
    m["bar"] = C.principled("scalebar", color=(0.95, 0.95, 0.95), rough=0.8,
                            emission=(1, 1, 1), emission_strength=0.6)
    return m


def scorch_material(name, center, r_in, r_out, strength=1.0, seed=0.0):
    """Darkened halo on the substrate: redeposited / reduced material, brown-black, irregular."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    dk = nt.nodes.new("ShaderNodeBsdfPrincipled")
    dk.inputs["Base Color"].default_value = (0.09, 0.05, 0.025, 1)
    dk.inputs["Roughness"].default_value = 0.75
    mix = nt.nodes.new("ShaderNodeMixShader")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (center[0], center[1], 0)
    nt.links.new(tc.outputs["Object"], sub.inputs[0])
    # warp the radius with noise for an irregular outline
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.08
    nz.inputs["Detail"].default_value = 8
    nz.noise_dimensions = "4D"
    nz.inputs["W"].default_value = seed
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(sub.outputs[0], ln.inputs[0])
    warp = nt.nodes.new("ShaderNodeMath")
    warp.operation = "MULTIPLY_ADD"
    warp.inputs[1].default_value = 0.9 * r_out
    nt.links.new(nz.outputs["Fac"], warp.inputs[0])
    nt.links.new(ln.outputs["Value"], warp.inputs[2])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = r_in + 0.45 * r_out
    mr.inputs["From Max"].default_value = r_out + 0.45 * r_out + 0.25 * r_out
    mr.inputs["To Min"].default_value = strength
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(warp.outputs[0], mr.inputs["Value"])
    # mottled density
    nz2 = nt.nodes.new("ShaderNodeTexNoise")
    nz2.inputs["Scale"].default_value = 0.6
    nz2.inputs["Detail"].default_value = 6
    nt.links.new(tc.outputs["Object"], nz2.inputs["Vector"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mr2 = nt.nodes.new("ShaderNodeMapRange")
    mr2.inputs["From Min"].default_value = 0.3
    mr2.inputs["From Max"].default_value = 0.7
    mr2.inputs["To Min"].default_value = 0.55
    mr2.inputs["To Max"].default_value = 1.0
    nt.links.new(nz2.outputs["Fac"], mr2.inputs["Value"])
    nt.links.new(mr.outputs["Result"], mul.inputs[0])
    nt.links.new(mr2.outputs["Result"], mul.inputs[1])
    nt.links.new(mul.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(dk.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    mt.blend_method = "HASHED"
    return mt


def blob(name, loc, rx, ry, rz, mat, seed, disp=0.25):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0, location=loc)
    ob = bpy.context.object
    ob.name = name
    ob.scale = (rx, ry, rz)
    tex = bpy.data.textures.new(name + "_t", "CLOUDS")
    tex.noise_scale = 0.45
    tex.noise_depth = 2
    md = ob.modifiers.new("disp", "DISPLACE")
    md.texture = tex
    md.strength = disp
    md.mid_level = 0.5
    md.texture_coords = "LOCAL"
    tex.noise_basis = "BLENDER_ORIGINAL"
    C.set_mat(ob, mat)
    C.smooth(ob, 80)
    return ob


def sph(name, loc, r, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=r, location=loc)
    ob = bpy.context.object
    ob.name = name
    C.set_mat(ob, mat)
    C.smooth(ob, 80)
    return ob


def halo_material(name, center, radius, strength, seed=0.0):
    """Thin redeposited film: transparent outside, a bronze-violet thin-film tint inside an irregular
    but fairly sharp boundary."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    film = nt.nodes.new("ShaderNodeBsdfPrincipled")
    film.inputs["Base Color"].default_value = (0.30, 0.22, 0.30, 1)
    film.inputs["Metallic"].default_value = 1.0
    film.inputs["Roughness"].default_value = 0.12
    mix = nt.nodes.new("ShaderNodeMixShader")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (center[0], center[1], 0)
    nt.links.new(tc.outputs["Object"], sub.inputs[0])
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(sub.outputs[0], ln.inputs[0])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["W"].default_value = seed
    nz.inputs["Scale"].default_value = 0.12
    nz.inputs["Detail"].default_value = 4
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    warp = nt.nodes.new("ShaderNodeMath")
    warp.operation = "MULTIPLY_ADD"
    warp.inputs[1].default_value = 0.6 * radius
    nt.links.new(nz.outputs["Fac"], warp.inputs[0])
    nt.links.new(ln.outputs["Value"], warp.inputs[2])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = radius * 1.3 - 0.8
    mr.inputs["From Max"].default_value = radius * 1.3 + 0.8
    mr.inputs["To Min"].default_value = strength
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(warp.outputs[0], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(film.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mt


def scene():
    C.reset()
    sc = bpy.context.scene
    sc.unit_settings.system = "NONE"
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    # microscope camera: plain sRGB response (no filmic highlight desaturation), exposure set so the
    # gold is the brightest, most saturated feature, as in a real brightfield image
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1)
    m = mats()
    C.box("substrate", (600, 400, 2), (0, 0, -1), m["sub"])
    # ---------------- comb electrodes (fingers along y), bus bars at y = +-60
    pitch = FW + GAP
    L_f, y_bus = 84.0, 47.0
    damaged = {}  # finger index -> list of (y0, y1) removed spans
    main = Vector((6.0, 8.0))
    for k in (-1, 0, 1):
        damaged[k] = [(main.y - 6.5 + 1.5 * abs(k) + rnd.uniform(-1, 1), main.y + 5.0 - 1.2 * abs(k) + rnd.uniform(-1, 1))]
    minor_k = -7
    polys, ends = [], []
    for k in range(-16, 17):
        x = k * pitch + main.x - 0 * pitch
        top = (k % 2 == 0)
        y0, y1 = (y_bus - L_f, y_bus) if top else (-y_bus, -y_bus + L_f)
        if k == minor_k:  # notched tip
            if top:
                y0 += 3.2
            else:
                y1 -= 3.2
            ends.append((x, y0 if top else y1, 0.9))
        spans = [(y0, y1)]
        for (a, b) in damaged.get(k, []):
            new = []
            for (s0, s1) in spans:
                if b <= s0 or a >= s1:
                    new.append((s0, s1))
                    continue
                if a > s0:
                    new.append((s0, a))
                    ends.append((x, a, 1.0))
                if b < s1:
                    new.append((b, s1))
                    ends.append((x, b, 1.0))
            spans = new
        for (s0, s1) in spans:
            polys.append([(x - FW / 2, s0), (x + FW / 2, s0), (x + FW / 2, s1), (x - FW / 2, s1)])
    polys.append([(-300, y_bus), (300, y_bus), (300, y_bus + 40), (-300, y_bus + 40)])
    polys.append([(-300, -y_bus - 40), (300, -y_bus - 40), (300, -y_bus), (-300, -y_bus)])
    C.extrude_polys("comb", polys, 0.0, H_AU, m["au"], bevel=0.06, bevel_segments=2)
    # ---------------- resolidified gold: smooth spheres at the broken finger ends (surface tension),
    # sized from the melted volume (3 fingers x ~11 um x 4 um x 0.2 um ~ 26 um^3 -> six ~1.1 um
    # beads, one fused bridge, sub-um spatter)
    au_s = m["au_smooth"]
    for i, (x, y, s_) in enumerate(ends):
        r = s_ * rnd.uniform(0.95, 1.3)
        sph(f"bead{i}", (x + rnd.uniform(-0.3, 0.3), y, 0.35 * r), r, au_s)
    # fused bridge across one gap between opposite-polarity fingers, just above the cut zone
    br = sph("bridge", (main.x + pitch * 0.5, main.y + 8.2, 0.25), 1.0, au_s)
    br.scale = (4.8, 1.2, 0.4)
    # sub-um spatter, solid spheres, density falling off with distance
    bm = bmesh.new()
    for _ in range(160):
        r_d = 4.0 + 18.0 * rnd.random() ** 1.5
        a = rnd.uniform(0, 2 * math.pi)
        rad = 0.08 + 0.45 * rnd.random() ** 3
        c = Vector((main.x + r_d * math.cos(a), main.y + r_d * math.sin(a), rad * 0.8))
        g = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=rad)
        bmesh.ops.translate(bm, verts=g["verts"], vec=c)
    me = bpy.data.meshes.new("splatter")
    bm.to_mesh(me)
    bm.free()
    sp = C.link(bpy.data.objects.new("splatter", me))
    C.set_mat(sp, au_s)
    C.smooth(sp, 80)
    # small crater in the LN film where the arc rooted, with radial cracks (thermal shock)
    crater = C.principled("crater", color=(0.06, 0.06, 0.06), rough=0.7, bump=1.0, bump_scale=2.0, bump_dist=0.2)
    pts = []
    for i in range(48):
        a = 2 * math.pi * i / 48
        rr = 2.6 * (1 + 0.18 * math.sin(3 * a + 0.7) + 0.10 * rnd.uniform(-1, 1))
        pts.append((main.x + 0.5 + rr * math.cos(a), main.y - 0.5 + rr * math.sin(a)))
    C.extrude_polys("crater", [pts], -0.02, 0.03, crater)
    crack = C.principled("crack", color=(0.02, 0.02, 0.02), rough=0.8)
    for i in range(6):
        a = rnd.uniform(0, 2 * math.pi)
        L = rnd.uniform(6, 15)
        p0 = Vector((main.x + 0.5 + 2.4 * math.cos(a), main.y - 0.5 + 2.4 * math.sin(a), 0.01))
        pts3 = [p0]
        for k in range(1, 5):
            a += rnd.uniform(-0.35, 0.35)
            pts3.append(pts3[-1] + Vector((math.cos(a), math.sin(a), 0)) * (L / 4))
        C.curve_tube(f"crack{i}", pts3, 0.07, crack, bevel_res=1)
    # thin redeposited film around the site: a subtle thin-film colour shift with a fairly defined
    # edge, on the substrate only (under the gold)
    C.box("halo_main", (300, 220, 0.005), (0, 0, 0.004), halo_material("halo_main", main, 13.0, 0.35, seed=1.3))
    minor_c = Vector((minor_k * pitch + main.x, (y_bus - L_f + 3.2) if minor_k % 2 == 0 else (-y_bus + L_f - 3.2)))
    C.box("halo_minor", (300, 220, 0.005), (0, 0, 0.005), halo_material("halo_minor", minor_c, 4.0, 0.3, seed=4.1))
    pit = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        pit.append((minor_c.x + 0.9 * math.cos(a), minor_c.y + 1.6 + 0.9 * math.sin(a)))
    C.extrude_polys("pit_minor", [pit], -0.02, 0.03, crater)
    # particulates on the surface
    dust = C.principled("dust", color=(0.5, 0.48, 0.45), rough=0.9)
    C.scatter_dust("dust", ((-100, 100), (-60, 60)), H_AU, 25, 0.2, 0.9, dust, seed=31)
    # ---------------- camera, coaxial illumination
    field = 200.0
    cam = C.camera((0, 0, 200), (0, 0, 0), ortho_scale=field, clip=(1, 1000))
    cam.location = (8.0, 2.0, 200)
    lamp = C.emission_mat("coax", (1.0, 0.97, 0.92), 0.92)
    # coaxial illuminator limited to the objective NA (0.4, +-24 deg): emitter = field + margin
    mg = 2 * 120 * math.tan(math.radians(24))
    pl = C.box("coax_plane", (field + mg, field * 9 / 16 + mg, 1), (8.0, 2.0, 120), lamp)
    pl.visible_camera = False
    pl.visible_shadow = False
    # scale bar, burned into the lower right corner (20 um)
    bx, by = 8.0 + field / 2 - 22.0, 2.0 - field * 9 / 32 + 9.0
    C.box("bar", (20.0, 1.2, 0.5), (bx, by, 30), m["bar"])
    C.text_mesh("bar_t", "20 µm", 4.2, (bx, by + 4.2, 30), h=0.2, mat=m["bar"])
    C.compositor(vignette=0.30, dispersion=0.012)


if __name__ == "__main__":
    scene()
    C.render(A["out"])
