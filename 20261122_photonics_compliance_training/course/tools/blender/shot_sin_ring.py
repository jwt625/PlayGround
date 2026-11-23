"""Shot 6: silicon nitride microring chip under a reflected-light microscope (brightfield, coaxial
illumination, objective NA 0.28), lensed fiber at the input facet. Top-down, field 520 um.

Units: 1 Blender unit = 1 um. Stack: 4 um buried oxide / 800 nm SiN core / 2 um top cladding on Si.
Ring: radius 100 um, waveguide 2.5 um wide (multimode low-loss ring), bus 2.5 um narrowing to a
0.4 um inverse taper over the last 150 um before the facet, ring-bus gap 0.8 um.
Surface colours from color_stack.py (normal-incidence reflectance, D65): over the core
(0.260, 0.272, 0.325), field (0.226, 0.245, 0.304), about 11 % brighter over the core. The
conformal cladding bump over the core (sloped edges outside the objective NA) adds dark outlines.
Depth of field: aperture matched to NA 0.28, focus on the chip surface, so the lensed tip (at the
waveguide height) is sharp and the 125 um fiber body above the surface is soft.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import common as C  # noqa: E402

A = C.args(out="/tmp/ring.jpg", samples=128, res=1920, noise=0.015, look="AgX - Very High Contrast")
rnd = random.Random(3)
X_EDGE = -220.0
CX, CY = -40.0, 0.0  # camera centre (um)
FW = 520.0           # field width (um)
WG_W, BUMP_H, SLOPE = 2.5, 0.8, 0.4  # core width, cladding bump height, slope run per side (63 deg)


def strip_polys(path, w):
    """Polygon for a strip of width w along a polyline path [(x, y), ...]."""
    left, right = [], []
    for i, p in enumerate(path):
        p = Vector(p)
        q = Vector(path[min(i + 1, len(path) - 1)]) - Vector(path[max(i - 1, 0)])
        q.normalize()
        n = Vector((-q.y, q.x))
        left.append(p + n * w / 2)
        right.append(p - n * w / 2)
    ring = right + list(reversed(left))
    return [(v.x, v.y) for v in ring]


def ring_polys(cx, cy, r, w, n=720):
    outer = [(cx + (r + w / 2) * math.cos(2 * math.pi * i / n), cy + (r + w / 2) * math.sin(2 * math.pi * i / n))
             for i in range(n)]
    inner = [(cx + (r - w / 2) * math.cos(2 * math.pi * i / n), cy + (r - w / 2) * math.sin(2 * math.pi * i / n))
             for i in range(n)]
    return outer, inner


def bump_mesh(name, polys_outer, mat, holes=None):
    """Cladding bump: top face at BUMP_H, sides sloped outward by SLOPE (built as extrude + inset)."""
    bm = bmesh.new()
    faces = []
    for poly in polys_outer:
        vs = [bm.verts.new((x, y, BUMP_H)) for x, y in poly]
        f = bm.faces.new(vs)
        if f.normal.z < 0:
            f.normal_flip()
        faces.append(f)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    md = ob.modifiers.new("solid", "SOLIDIFY")
    md.thickness = BUMP_H
    md.offset = -1
    C.set_mat(ob, mat)
    bv = ob.modifiers.new("bevel", "BEVEL")
    bv.width = SLOPE * 0.5
    bv.segments = 3
    bv.limit_method = "ANGLE"
    return ob


def ring_obj(name, cx, cy, r, mat):
    bm = bmesh.new()
    n = 720
    w = WG_W
    vo, vi = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        vo.append(bm.verts.new((cx + (r + w / 2) * math.cos(a), cy + (r + w / 2) * math.sin(a), BUMP_H)))
        vi.append(bm.verts.new((cx + (r - w / 2) * math.cos(a), cy + (r - w / 2) * math.sin(a), BUMP_H)))
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((vi[i], vo[i], vo[j], vi[j]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    md = ob.modifiers.new("solid", "SOLIDIFY")
    md.thickness = BUMP_H
    md.offset = -1
    bv = ob.modifiers.new("bevel", "BEVEL")
    bv.width = SLOPE * 0.5
    bv.segments = 3
    bv.limit_method = "ANGLE"
    C.set_mat(ob, mat)
    return ob


def lensed_fiber(glass, coat, tip, d):
    d = Vector(d).normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bm = bmesh.new()
    rt, rf, Lc = 4.0, 62.5, 160.0
    prof = [(0.0, 0.0)]
    for i in range(1, 9):
        a = math.pi / 2 * i / 8
        prof.append((rt * math.sin(a), -rt + rt * math.cos(a)))
    prof += [(rf, -Lc), (rf, -3000.0)]
    seg = 96
    rings = [[bm.verts.new((r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z))
              for k in range(seg)] for r, z in prof]
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(seg):
            bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    me = bpy.data.meshes.new("fiber")
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new("fiber", me))
    for p in ob.data.polygons:
        p.use_smooth = True
    C.set_mat(ob, glass)
    ob.matrix_world = Matrix.Translation(Vector(tip)) @ rot
    ob.visible_shadow = False
    return ob


def sweep(name, pts, widths, mat, closed=False):
    """Conformal cladding bump along a path: trapezoid cross-section, flat top = core width,
    sloped sides of horizontal run SLOPE down to the field (z = 0)."""
    verts, faces = [], []
    n = len(pts)
    for i, (p, w) in enumerate(zip(pts, widths)):
        p = Vector(p)
        q = Vector(pts[(i + 1) % n] if closed else pts[min(i + 1, n - 1)]) - Vector(
            pts[(i - 1) % n] if closed else pts[max(i - 1, 0)])
        q.normalize()
        nn = Vector((-q.y, q.x))
        for off, z in ((-(w / 2 + SLOPE), 0.0), (-w / 2, BUMP_H), (w / 2, BUMP_H), (w / 2 + SLOPE, 0.0)):
            v = p + nn * off
            verts.append((v.x, v.y, z))
    m = n if closed else n - 1
    for i in range(m):
        a_, b_ = 4 * i, 4 * ((i + 1) % n)
        for j in range(3):
            faces.append((a_ + j, a_ + j + 1, b_ + j + 1, b_ + j))
    ob = C.mesh_obj(name, verts, faces, mat)
    for poly in ob.data.polygons:
        poly.use_smooth = False
    return ob


def scene():
    C.reset()
    bpy.context.scene.unit_settings.system = "NONE"
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    w = bpy.data.worlds.new("W")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1)
    field = C.principled("sin_field", color=(0.226, 0.245, 0.304), metallic=1.0, rough=0.05, rough_var=0.02,
                         rough_scale=0.02)
    wgm = C.principled("sin_wg", color=(0.260, 0.272, 0.325), metallic=1.0, rough=0.05)
    facet = C.principled("si_edge", color=(0.05, 0.05, 0.055), rough=0.35, ior=3.9)
    glass = C.principled("glass", color=(1, 1, 1), rough=0.0, ior=1.46, transmission=1.0)
    # chip body: top at z = 0, west edge at X_EDGE (with slight dicing chips)
    edge = [(X_EDGE - (0.6 * rnd.random() if rnd.random() < 0.08 else 0.0) if 0 < i < 120 else X_EDGE,
             -400 + 800 * i / 120) for i in range(121)]
    poly = [(2000, -400), (2000, 400)] + list(reversed(edge))
    chip = C.extrude_polys("chip", [poly], -500.0, 500.0, None)
    chip.data.materials.append(field)
    chip.data.materials.append(facet)
    for f in chip.data.polygons:
        f.material_index = 0 if f.normal.z > 0.9 else 1
    # bus with inverse taper at the facet
    yb, r, gap, cx = -100.0, 100.0, 0.8, 60.0
    xs = [X_EDGE + 150.0 * i / 60 for i in range(61)] + [X_EDGE + 150.0 + 20.0 * i for i in range(1, 60)]
    ws = [0.4 + (WG_W - 0.4) * min(1.0, (x - X_EDGE) / 150.0) for x in xs]
    sweep("bus", [(x, yb) for x in xs], ws, wgm)
    cy = yb + WG_W + gap + r
    nr = 1440
    sweep("ring", [(cx + r * math.cos(2 * math.pi * i / nr), cy + r * math.sin(2 * math.pi * i / nr)) for i in range(nr)],
          [WG_W] * nr, wgm, closed=True)
    lbl = C.text_mesh("lbl", "R100  W2.5  G0.8", 9.0, (cx, yb - 22.0, 0.0), h=BUMP_H, mat=wgm)
    lbl.data.bevel_depth = 0.35  # cladding over the SiN text: sloped edges like the waveguides
    # drop bus (add-drop ring): same 0.8 um gap at the top of the ring
    yd = cy + r + gap + WG_W
    xs2 = [X_EDGE + 150.0 * i / 60 for i in range(61)] + [X_EDGE + 150.0 + 20.0 * i for i in range(1, 60)]
    ws2 = [0.4 + (WG_W - 0.4) * min(1.0, (x - X_EDGE) / 150.0) for x in xs2]
    sweep("bus_drop", [(x, yd) for x in xs2], ws2, wgm)  # drop port, also edge-coupled
    dust = C.principled("dust", color=(0.12, 0.11, 0.10), rough=0.8)
    C.scatter_dust("dust", ((X_EDGE + 3, 230), (-150, 150)), 0.0, 25, 0.25, 1.5, dust, seed=17)
    # lensed fiber on the bus axis, 3 um from the facet, axis at the core height (2.4 um below the top)
    lensed_fiber(glass, None, (X_EDGE - 3.0, yb, -2.4), (1, 0, 0))
    # camera: orthographic, DOF matched to NA 0.28 (aperture radius = 300 * tan(16 deg) = 86 um)
    cam = C.camera((0, 0, 300), (0, 0, 0), ortho_scale=FW, clip=(1, 5000))
    cam.location = (CX, CY, 300)
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 300.0
    cam.data.dof.aperture_fstop = 1.0 / (2 * 300.0 * math.tan(math.radians(16)))
    # coaxial illuminator limited to the objective NA (+-16 deg): emitter = field + margin
    lamp = C.emission_mat("coax", (1.0, 0.96, 0.9), 1.6)
    m = 2 * 200 * math.tan(math.radians(16))
    pl = C.box("coax", (FW + m, FW * 9 / 16 + m, 1), (CX, CY, 200), lamp)
    pl.visible_camera = False
    pl.visible_shadow = False
    # weak low-angle ring illumination (about 68 deg from the axis): the 34 deg cladding slopes
    # send it into the objective, so waveguide edges glow as in a slightly dark-field image
    ringl = C.emission_mat("ringlight", (1.0, 0.97, 0.92), 0.9)
    band = C.cylinder("ringlight", 4000.0, 400.0, (CX, CY, 1616.0), ringl, verts=128)
    bm = bmesh.new()
    bm.from_mesh(band.data)
    for f in [f for f in bm.faces if abs(f.normal.z) > 0.9]:
        bm.faces.remove(f)
    bm.to_mesh(band.data)
    bm.free()
    band.visible_camera = False
    band.visible_shadow = False
    # scale bar 50 um
    bar = C.principled("bar", color=(0.95, 0.95, 0.95), rough=0.8, emission=(1, 1, 1), emission_strength=0.6)
    bx, by = CX + FW / 2 - 45, CY - FW * 9 / 32 + 14
    C.box("bar", (50, 1.6, 1), (bx, by, 2), bar)
    C.text_mesh("bar_t", "50 \u00b5m", 9, (bx, by + 8, 2), h=0.5, mat=bar)
    C.compositor(vignette=0.30, dispersion=0.01)


if __name__ == "__main__":
    scene()
    C.render(A["out"])
