"""Shot 1: x-cut TFLN MZM die in a black-anodized die fixture on a probe-station chuck.

variant=0 / 90 / 180: die rotation (counterclockwise, degrees) relative to the fixture. The fixture
has fiber slots on the west and east sides and a laser-engraved "+Z" reference arrow; the die has a
gold "+Z" arrow. Correct loading (variant 0): die arrow parallel to the fixture arrow, facets at the
fiber slots, RF pads to the north.
Camera, lights and everything else are identical across variants (choice-card set).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402
import tfln_die  # noqa: E402

A = C.args(out="/tmp/die.jpg", samples=128, variant=0, res=1920, noise=0.015, key_angle=35.0, look="AgX - Punchy", field=30.0)


def fixture_materials():
    m = {}
    m["anod"] = C.principled("anodized_black", color=(0.012, 0.012, 0.014), rough=0.42, rough_var=0.06,
                             rough_scale=3.0, bump=0.3, bump_scale=300.0, bump_dist=0.0004,
                             color_var=0.05, color_scale=0.6)
    m["engr"] = C.principled("laser_mark", color=(0.42, 0.42, 0.43), rough=0.75, bump=0.3, bump_scale=200.0,
                             bump_dist=0.0005)
    m["steel"] = C.principled("screw_steel", color=(0.42, 0.42, 0.43), metallic=1.0, rough=0.38,
                              rough_var=0.08, rough_scale=8.0)
    m["dust"] = C.principled("dust", color=(0.55, 0.53, 0.50), rough=0.9)
    m["brass"] = C.principled("brass", color=(0.85, 0.62, 0.30), metallic=1.0, rough=0.25)
    m["pu_tube"] = C.principled("pu_tube", color=(0.85, 0.88, 0.90), rough=0.15, ior=1.5, transmission=0.85)
    m["lint"] = C.principled("lint", color=(0.75, 0.75, 0.78), rough=0.8, transmission=0.3)
    return m


def chuck_material():
    """Nickel-plated chuck with concentric vacuum grooves (shader) and fine radial polish."""
    m = C.principled("chuck_nickel", color=(0.22, 0.215, 0.205), metallic=1.0, rough=0.42, rough_var=0.07,
                     rough_scale=0.2)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    md = nt.nodes.new("ShaderNodeMath")
    md.operation = "PINGPONG"
    md.inputs[1].default_value = 5.0
    nt.links.new(ln.outputs["Value"], md.inputs[0])
    # groove: |pingpong| < 0.5 mm
    lt = nt.nodes.new("ShaderNodeMapRange")
    lt.inputs["From Min"].default_value = 0.25
    lt.inputs["From Max"].default_value = 0.40
    lt.inputs["To Min"].default_value = 1.0
    lt.inputs["To Max"].default_value = 0.0
    nt.links.new(md.outputs[0], lt.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.22, 0.215, 0.205, 1)
    mix.inputs["B"].default_value = (0.05, 0.05, 0.05, 1)
    nt.links.new(lt.outputs["Result"], mix.inputs["Factor"])
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = 1.0
    b.inputs["Distance"].default_value = 0.3
    b.invert = True
    nt.links.new(lt.outputs["Result"], b.inputs["Height"])
    nt.links.new(b.outputs["Normal"], p.inputs["Normal"])
    return m


SLOT_Y = tfln_die.YC  # fiber slots are centred on the waveguide axis of a correctly loaded die
POCKET_D = 0.20       # pocket depth; the 0.5 mm die stands 0.3 mm proud
ZT = tfln_die.T - POCKET_D  # die top surface height


def build_fixture(m):
    """Machined die fixture, 40 x 28 x 6 mm, black anodized: 6.12 mm square pocket (0.2 mm deep) with
    a vacuum port under the die, two 1.2 mm wide fiber slots on the waveguide line ending at the
    pocket (west = IN, east = OUT), four M3 socket head cap screws (5.5 mm heads) in counterbores
    holding it to the chuck, laser-marked 2 mm text, and a vacuum hose barb on the south face."""
    W, H, D = 40.0, 28.0, 6.0
    blk = C.box("fixture", (W, H, D), (0, 0, -D / 2), m["anod"])
    cutters = [C.box("c_pocket", (6.12, 6.12, 1.0), (0, 0, -POCKET_D + 0.5))]
    cutters.append(C.cylinder("c_vac", 0.5, 4.0, (0.8, 0.9, -2.0), verts=24))
    for sx in (-1, 1):
        cutters.append(C.box("c_slot", (W / 2 - 3.0, 1.2, 1.6), (sx * (3.0 + (W / 2 - 3.0) / 2 + 0.06), SLOT_Y, 0)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            cutters.append(C.cylinder("c_cb", 3.25, 6.4, (sx * 16.0, sy * 10.0, 0), verts=48))
            cutters.append(C.cylinder("c_th", 1.7, 20.0, (sx * 16.0, sy * 10.0, 0), verts=32))
    for c in cutters:
        md = blk.modifiers.new("b", "BOOLEAN")
        md.object = c
        md.solver = "EXACT"
        c.hide_render = True
        c.hide_viewport = True
    # M3 socket head cap screws (head 5.5 x 3 mm, 2.5 mm hex), seated 0.2 mm below the top surface
    for sx in (-1, 1):
        for sy in (-1, 1):
            hx, hy = sx * 16.0, sy * 10.0
            hd = C.cylinder("screw", 2.75, 3.0, (hx, hy, -0.2 - 1.5), m["steel"], verts=48, bevel=0.2)
            hexs = C.cylinder("hex", 1.44, 4.0, (hx, hy, -0.2), None, verts=6)
            md = hd.modifiers.new("b", "BOOLEAN")
            md.object = hexs
            md.solver = "EXACT"
            hexs.hide_render = True
            hexs.hide_viewport = True
            hd.modifiers.move(len(hd.modifiers) - 1, 0)
    # vacuum hose barb (brass) on the south face, clear PU tubing leaving the frame
    C.cylinder("barb_hex", 3.2, 2.5, (0.8, -H / 2 - 1.25, -3.0), m["brass"], verts=6, rot=(math.radians(90), 0, 0))
    C.cylinder("barb", 1.3, 6.0, (0.8, -H / 2 - 5.5, -3.0), m["brass"], verts=32, rot=(math.radians(90), 0, 0))
    C.curve_tube("vac_tube", [Vector((0.8, -H / 2 - 6.0, -3.0)), Vector((0.8, -H / 2 - 25, -3.5)),
                              Vector((-10, -H / 2 - 60, -6.0)), Vector((-40, -H / 2 - 120, -8.5))], 2.0, m["pu_tube"])
    # laser marking (light, matte), 2 um proud
    z = 0.002
    ax = -6.2
    arrow = [(ax - 0.25, 1.0), (ax + 0.25, 1.0), (ax + 0.25, 5.4), (ax + 0.9, 5.4), (ax, 7.2),
             (ax - 0.9, 5.4), (ax - 0.25, 5.4)]
    C.extrude_polys("engr_arrow", [arrow], 0.0, z, m["engr"])
    C.text_mesh("engr_z", "+Z", 2.0, (-9.4, 5.6, 0), h=z, mat=m["engr"])
    C.text_mesh("engr_in", "IN", 2.0, (-9.0, SLOT_Y - 2.2, 0), h=z, mat=m["engr"])
    C.text_mesh("engr_out", "OUT", 2.0, (9.0, SLOT_Y - 2.2, 0), h=z, mat=m["engr"])
    C.text_mesh("engr_id", "LN-MZM 6 MM  X-CUT", 1.5, (0.0, -8.5, 0), h=z, mat=m["engr"])
    return blk


def scene(variant):
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    C.world_hdri("hospital_room", strength=0.35, rot_z_deg=40, bg_color=(0.03, 0.03, 0.032))
    m = fixture_materials()
    build_fixture(m)
    # chuck below the fixture
    ch = C.cylinder("chuck", 150.0, 10.0, (6.0, 10.0, -6.0 - 5.0), chuck_material(), verts=256)
    tfln_die.build("die", rot_z_deg=variant, loc=(0, 0, ZT))
    C.scatter_dust("dust_die", ((-2.9, 2.9), (-2.9, 2.9)), ZT, 22, 0.002, 0.014, m["dust"], seed=3,
                   fibers=1, fiber_len=(0.3, 0.6))
    C.scatter_dust("dust_fix", ((-19, 19), (-13, 13)), 0.0, 120, 0.004, 0.04, m["dust"], seed=5, fibers=2,
                   fiber_len=(0.8, 1.8))
    C.scatter_dust("dust_chuck", ((-60, 60), (-30, 60)), -6.0, 150, 0.01, 0.06, m["dust"], seed=7, fibers=3,
                   fiber_len=(1.0, 3.0))
    # camera: 25 deg tilt from vertical, from the south; field ~ 21 mm
    tgt = Vector((-1.5, 1.2, 0.1))
    el = math.radians(62)
    D = 160.0
    loc = tgt + D * Vector((0.12, -math.cos(el), math.sin(el))).normalized()
    cam = C.camera(loc, tgt, lens=36 * D / A["field"])
    C.set_aperture(cam, tgt, pupil_mm=4.0)
    # key softbox in the specular direction (slightly off) so metal and film read; small fill
    mp = C.mirror_point(loc, (0, 0, ZT), 300.0)
    C.edge_softbox("key", mp, (0, 0, 0), 500.0, 3.5, edge=10.0, angle_deg=A["key_angle"], low=0.06)
    C.area_light("top", (-80, 60, 300), (0, 0, 0), 120, 2.0e4, color=(1.0, 0.97, 0.93))
    C.area_light("fill", (-180, -120, 120), (0, 0, 0), 120, 2.5e4, color=(0.9, 0.95, 1.0))
    C.area_light("rim", (200, 60, 60), (0, 0, 0), 60, 1.5e4)
    C.ceiling(420.0, color=(0.05, 0.05, 0.05))
    C.surround(600.0, 800.0, color=(0.06, 0.06, 0.065))
    C.area_light("fill2", (150, -200, 200), (0, 0, 0), 200, 3.0e4, color=(1.0, 0.98, 0.95))
    C.compositor(vignette=0.18, dispersion=0.004)


if __name__ == "__main__":
    scene(A["variant"])
    C.render(A["out"])
