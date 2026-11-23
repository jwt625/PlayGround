"""PANDA polarization-maintaining fiber end face as seen in a fiber inspection scope (top-down,
coaxial LED illumination, circular field), connector key marker at twelve o'clock.

variant=0 : stress rods at 12 and 6 o'clock (slow axis aligned to the key)
variant=90: same connector, fiber rotated 90 degrees in the ferrule (rods at 3 and 9 o'clock)

Units: 1 Blender unit = 1 um. Geometry (typical PM1550 PANDA): cladding 125 um, core 8.5 um,
stress rods 33 um diameter with centres 28 um from the core, ferrule bore 126 um, thin epoxy ring.
Overlay: inspection-zone rings (core 25 um, cladding 120 um, adhesive 130 um, contact 250 um) as
drawn by inspection software, and a key marker.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402

A = C.args(out="/tmp/panda.jpg", samples=96, res=1920, noise=0.015, variant=0, look="AgX - Medium High Contrast")
rnd = random.Random(11 + A["variant"])  # two different connectors: different debris


def flat_disk(name, r, z, mat, verts=192, h=0.02):
    ob = C.cylinder(name, r, h, (0, 0, z - h / 2), mat, verts=verts)
    return ob


def ring(name, r, w, z, mat, n=128):
    """Thin emissive circle (overlay graphic), tube of diameter w."""
    pts = [Vector((r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z)) for i in range(n)]
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = w / 2
    cu.bevel_resolution = 2
    sp = cu.splines.new("POLY")
    sp.points.add(n - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q.x, q.y, q.z, 1)
    sp.use_cyclic_u = True
    o = C.link(bpy.data.objects.new(name, cu))
    cu.materials.append(mat)
    return o


def scene():
    C.reset()
    bpy.context.scene.unit_settings.system = "NONE"
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    w = bpy.data.worlds.new("W")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    # materials: values chosen for an LED inspection-scope look (cladding bright, ferrule mid grey)
    zr = C.principled("zirconia", color=(0.17, 0.17, 0.18), rough=0.45, rough_var=0.15, rough_scale=0.08,
                      color_var=0.25, color_scale=0.12, bump=0.3, bump_scale=0.5, bump_dist=0.4)
    epoxy = C.principled("epoxy", color=(0.035, 0.03, 0.02), rough=0.4)
    clad = C.principled("cladding", color=(0.60, 0.61, 0.62), rough=0.12, color_var=0.03, color_scale=0.05)
    rod = C.principled("stress_rod", color=(0.50, 0.51, 0.53), rough=0.15, color_var=0.03, color_scale=0.1)
    rod_edge = C.principled("rod_edge", color=(0.15, 0.15, 0.16), rough=0.3)
    core = C.principled("core", color=(0.74, 0.75, 0.76), rough=0.1)
    zone = C.emission_mat("zone", (0.15, 0.9, 0.25), 1.2)
    key = C.emission_mat("key", (0.95, 0.85, 0.15), 1.5)
    flat_disk("ferrule", 600, 0.0, zr)
    flat_disk("epoxy", 63.2, 0.02, epoxy)
    flat_disk("clad", 62.5, 0.04, clad)
    ang = math.radians(90.0 if A["variant"] == 0 else 0.0)  # rods axis direction in the image
    for s in (1, -1):
        c = Vector((28 * math.cos(ang) * s, 28 * math.sin(ang) * s, 0))
        r = flat_disk(f"rod{s}", 16.0, 0.08, rod)
        r.location.x, r.location.y = c.x, c.y
    flat_disk("core", 4.25, 0.10, core, verts=64)
    # a few polish scratches and specks on the ferrule (clean fiber, as a passing end face)
    sc_m = C.principled("scratch", color=(0.12, 0.12, 0.12), rough=0.6)
    for i in range(7):
        r0 = rnd.uniform(80, 220)
        a0 = rnd.uniform(0, 6.28)
        L = rnd.uniform(25, 90)
        p = Vector((r0 * math.cos(a0), r0 * math.sin(a0), 0.03))
        d = Vector((math.cos(a0 + rnd.uniform(1.2, 1.9)), math.sin(a0 + rnd.uniform(1.2, 1.9)), 0))
        C.curve_tube(f"scr{i}", [p, p + d * L / 2 + Vector((rnd.uniform(-2, 2), rnd.uniform(-2, 2), 0)), p + d * L],
                     0.35, sc_m, bevel_res=1)
    dust = C.principled("dust", color=(0.05, 0.05, 0.05), rough=0.8)
    C.scatter_dust("specks", ((-200, 200), (-200, 200)), 0.03, 30, 0.4, 2.5, dust, seed=9 + A["variant"])
    # overlay: zone rings (diameters 25, 120, 130, 250 um) and key marker at 12 o'clock
    for i, dmt in enumerate((25.0, 120.0, 130.0, 250.0)):
        ring(f"zone{i}", dmt / 2, 0.7, 5.0, zone)
    tri = [(-7, 0), (7, 0), (0, -10)]
    # key indicator drawn as an annotation outside the scope's field of view (course overlay)
    C.extrude_polys("key_mark", [[(x, y + 216) for x, y in tri]], 25.0, 0.2, key)
    C.text_mesh("key_t", "KEY", 9.0, (24, 210, 25.0), h=0.2, mat=key)
    # circular field stop (black card with a hole) between sample and camera
    stop = C.cylinder("stop", 2000, 1.0, (0, 0, 20), C.principled("black", color=(0, 0, 0), rough=1.0), verts=256)
    hole = C.cylinder("hole", 196.0, 5.0, (0, 0, 20), None, verts=256)
    md = stop.modifiers.new("b", "BOOLEAN")
    md.object = hole
    md.solver = "EXACT"
    hole.hide_render = True
    hole.hide_viewport = True
    stop.visible_shadow = False
    stop.visible_diffuse = False
    stop.visible_glossy = False
    # camera and coaxial illumination
    C.camera((0, 0, 300), (0, 0, 0), ortho_scale=440 * 16 / 9, clip=(1, 2000))
    lamp = C.emission_mat("coax", (0.95, 0.97, 1.0), 1.3)
    mg = 2 * 150 * math.tan(math.radians(20))
    pl = C.box("coax", (440 * 16 / 9 + mg, 440 + mg, 1), (0, 0, 150), lamp)
    pl.visible_camera = False
    pl.visible_shadow = False
    C.compositor(vignette=0.25, dispersion=0.006)


if __name__ == "__main__":
    scene()
    C.render(A["out"])
