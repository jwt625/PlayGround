"""Shot 2: the same x-cut TFLN MZM die (correct orientation) on a probe station.

Two GSG microwave probes (150 um pitch) land on the RF-in and termination pads from the north,
lensed fibers sit at the west (input) and east (output) facets, about 8 um from the facet, on the
waveguide axis. A long-working-distance objective (20 mm WD) hangs above, out of focus.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import common as C  # noqa: E402
import shot_die_fixture as F  # noqa: E402
import tfln_die  # noqa: E402

A = C.args(out="/tmp/probe.jpg", samples=128, res=1920, noise=0.015, look="AgX - Punchy", view="close",
           el=34.0, field=30.0, low=0.35, az=-100.0)
ZT = F.ZT  # die top surface height (0.3 mm above the fixture)


def mats():
    m = {}
    m["glass"] = C.principled("fiber_glass", color=(1, 1, 1), rough=0.0, ior=1.46, transmission=1.0)
    m["coat"] = C.principled("acrylate", color=(0.85, 0.88, 0.80), rough=0.25, ior=1.5, transmission=0.6)
    m["alu"] = C.principled("alu_bead", color=(0.80, 0.80, 0.81), metallic=1.0, rough=0.35, rough_var=0.06,
                            rough_scale=2.0, bump=0.3, bump_scale=80.0, bump_dist=0.0008)
    m["nickel"] = C.principled("probe_body", color=(0.70, 0.68, 0.64), metallic=1.0, rough=0.22, rough_var=0.05,
                               rough_scale=3.0)
    m["brass_au"] = C.principled("gold_plate", color=(1.0, 0.74, 0.36), metallic=1.0, rough=0.18)
    m["coax"] = C.principled("semirigid", color=(0.85, 0.80, 0.72), metallic=1.0, rough=0.25)
    m["tung"] = C.principled("tungsten_tip", color=(0.55, 0.52, 0.48), metallic=1.0, rough=0.2)
    m["ceramic"] = C.principled("tip_ceramic", color=(0.85, 0.83, 0.78), rough=0.45)
    m["black"] = C.principled("black_anod", color=(0.015, 0.015, 0.017), rough=0.4, rough_var=0.05, rough_scale=1.0)
    m["engr"] = C.principled("obj_engr", color=(0.8, 0.8, 0.8), rough=0.6)
    m["lens"] = C.principled("obj_lens", color=(1, 1, 1), rough=0.0, ior=1.6, transmission=1.0,
                             film_nm=140.0, film_ior=1.38)
    m["chrome"] = C.principled("obj_chrome", color=(0.9, 0.9, 0.9), metallic=1.0, rough=0.08)
    m["rubber"] = C.principled("strain_boot", color=(0.02, 0.02, 0.02), rough=0.6)
    m["cable_blue"] = C.principled("cable_blue", color=(0.05, 0.12, 0.35), rough=0.45)
    m["yellow"] = C.principled("jacket_yellow", color=(0.75, 0.45, 0.02), rough=0.45)
    return m


BARE = 2.0  # stripped (bare glass) length behind the lensed tip, mm


def lensed_fiber(m, tip, direction, name):
    """Bare 125 um fiber with a conical lensed tip ending at `tip`, pointing along `direction`."""
    d = Vector(direction).normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    parts = []
    # cone + hemispherical end, built along +Z in local frame, tip at origin
    bm = bmesh.new()
    rt, rf, Lc = 0.004, 0.0625, 0.16
    prof = []
    for i in range(9):  # hemisphere
        a = math.pi / 2 * i / 8
        prof.append((rt * math.sin(a), -rt + rt * math.cos(a) - 0.0))
    prof.append((rf, -Lc))
    prof.append((rf, -BARE))
    prof = [(0.0, 0.0)] + [(r, z) for r, z in prof[1:]]
    seg = 48
    rings = []
    for r, z in prof:
        ring = [bm.verts.new((r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z))
                for k in range(seg)]
        rings.append(ring)
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(seg):
            bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    bm.faces.new(list(reversed(rings[-1])))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me))
    C.set_mat(ob, m["glass"])
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.matrix_world = Matrix.Translation(Vector(tip)) @ rot
    ob.visible_shadow = False  # thin glass: no hard shadow (caustics are off)
    parts.append(ob)
    # acrylate coating (250 um) from the end of the stripped section
    base = Vector(tip) - d * BARE
    c = C.cylinder(name + "_coat", 0.125, 14.0, (0, 0, 0), m["coat"], verts=48)
    c.matrix_world = Matrix.Translation(base - d * 7.0) @ rot
    parts.append(c)
    # fiber holder on its own stage, above the fixture: the coated fiber is clamped in a V-groove on
    # the underside of the holder arm (top-down holder), so nothing rests on the fixture
    hb = C.box(name + "_holder", (9.0, 5.0, 3.5), (0, 0, 0), m["alu"], bevel=0.3)
    hb.matrix_world = Matrix.Translation(base - d * 9.5 + Vector((0, 0, 0.125 + 1.75))) @ Matrix.Rotation(
        math.atan2(d.y, d.x), 4, "Z")
    arm = C.box(name + "_harm", (30.0, 5.0, 3.0), (0, 0, 0), m["alu"], bevel=0.3)
    arm.matrix_world = Matrix.Translation(base - d * 27.0 + Vector((0, 0, 3.5))) @ Matrix.Rotation(
        math.atan2(d.y, d.x), 4, "Z")
    parts += [hb, arm]
    return parts


def gsg_probe(m, pad, name, side=-1):
    """Microwave GSG probe (150 um pitch) approaching along x from the west (side=-1) or east (+1)
    at about 30 degrees: compact coplanar tip with three contacts on the in-line pad set (contacts
    along y), semi-rigid coax up to the probe body, 2.92 mm-class connector with a flexible cable,
    and the mounting arm toward an off-frame micropositioner."""
    px, py, pz = pad
    sx = side
    parts = []
    ang = math.radians(30.0)
    # contacts and compact tip: three tapered blades converging on the pads, ~0.5 mm long
    root = Vector((px + sx * 0.55, py, pz + 0.30))
    for k in (-1, 0, 1):
        tip = Vector((px, py + k * 0.150, pz + 0.004))
        r0 = root + Vector((0, k * 0.22, 0))
        n = 12
        verts, faces = [], []
        for i in range(n + 1):
            t = i / n
            p = r0 + (tip - r0) * t
            w = (0.16 if k else 0.09) * (1 - t) + 0.03 * t
            verts += [(p.x, p.y - w / 2, p.z), (p.x, p.y + w / 2, p.z), (p.x, p.y + w / 2, p.z + 0.02),
                      (p.x, p.y - w / 2, p.z + 0.02)]
        for i in range(n):
            o = 4 * i
            for j in range(4):
                faces.append((o + j, o + (j + 1) % 4, o + 4 + (j + 1) % 4, o + 4 + j))
        faces.append((0, 3, 2, 1))
        faces.append((4 * n, 4 * n + 1, 4 * n + 2, 4 * n + 3))
        parts.append(C.mesh_obj(name + f"_c{k}", verts, faces, m["tung"]))
    # ceramic tip carrier with the coplanar lines, then the semi-rigid coax rising at 30 degrees
    tb = C.box(name + "_tipblock", (0.9, 0.62, 0.12), (0, 0, 0), m["ceramic"], bevel=0.02)
    tb.location = root + Vector((sx * 0.40, 0, 0.18))
    tb.rotation_euler = (0, sx * ang, 0)
    parts.append(tb)
    c0 = root + Vector((sx * 0.8, 0, 0.42))
    c1 = c0 + Vector((sx * 9.0 * math.cos(ang), 0, 9.0 * math.sin(ang)))
    parts.append(C.curve_tube(name + "_coax", [c0, (c0 + c1) / 2, c1], 0.43, m["coax"], bevel_res=6))
    # probe body, connector and cable
    body = C.box(name + "_body", (9.0, 7.0, 6.0), (0, 0, 0), m["nickel"], bevel=0.4)
    body.location = c1 + Vector((sx * 4.0, 0, 1.5))
    body.rotation_euler = (0, sx * ang, 0)
    parts.append(body)
    ctop = body.location + Vector((sx * 1.0, 0, 3.6))
    parts.append(C.cylinder(name + "_conn_hex", 2.2, 2.0, tuple(ctop), m["brass_au"], verts=6, bevel=0.1))
    parts.append(C.cylinder(name + "_conn", 1.7, 3.5, tuple(ctop + Vector((0, 0, 2.6))), m["brass_au"], verts=48))
    cab0 = ctop + Vector((0, 0, 4.4))
    parts.append(C.curve_tube(name + "_cable", [cab0, cab0 + Vector((sx * 2, 0, 8)), cab0 + Vector((sx * 20, 6, 22)),
                                                cab0 + Vector((sx * 60, 20, 30))], 1.6, m["cable_blue"], bevel_res=6))
    # mounting arm toward the micropositioner
    arm = C.box(name + "_arm", (45.0, 6.0, 4.0), (0, 0, 0), m["black"], bevel=0.3)
    arm.location = body.location + Vector((sx * 25.0, 0, 2.0))
    parts.append(arm)
    return parts


def objective(m, z_front, name="obj"):
    parts = []
    parts.append(C.cylinder(name + "_nose", 9.5, 6.0, (0, 0, z_front + 3.0), m["black"], verts=96, bevel=0.3,
                            r2=11.0))
    parts.append(C.cylinder(name + "_ring", 11.2, 1.2, (0, 0, z_front + 6.6), m["chrome"], verts=96, bevel=0.1))
    parts.append(C.cylinder(name + "_barrel", 16.0, 40.0, (0, 0, z_front + 27.2), m["black"], verts=96,
                            bevel=0.4))
    parts.append(C.cylinder(name + "_lens", 4.2, 0.6, (0, 0, z_front + 0.1), m["lens"], verts=64))
    parts.append(C.cylinder(name + "_bezel", 5.2, 0.8, (0, 0, z_front - 0.1), m["chrome"], verts=64))
    # engraving around the barrel facing the camera (south)
    t = C.text_mesh(name + "_t", "LWD PLAN 20x / 0.42   f = 200   WD 20", 1.3, (0, -16.02, z_front + 16.0),
                    h=0.02, mat=m["engr"], rot=(math.radians(90), 0, 0))
    parts.append(t)
    return parts


def scene():
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    C.world_hdri("hospital_room", strength=0.30, rot_z_deg=130, bg_color=(0.025, 0.025, 0.027))
    fm = F.fixture_materials()
    F.build_fixture(fm)
    C.cylinder("chuck", 150.0, 10.0, (6.0, 10.0, -11.0), F.chuck_material(), verts=256)
    tfln_die.build("die", rot_z_deg=0, loc=(0, 0, ZT))
    C.scatter_dust("dust_die", ((-2.9, 2.9), (-2.9, 2.9)), ZT, 22, 0.002, 0.014, fm["dust"], seed=3, fibers=1,
                   fiber_len=(0.3, 0.6))
    C.scatter_dust("dust_fix", ((-9.5, 9.5), (-6.5, 6.5)), 0.0, 70, 0.004, 0.03, fm["dust"], seed=5, fibers=2,
                   fiber_len=(0.8, 1.8))
    m = mats()
    wg_z = ZT - 0.0003
    lensed_fiber(m, (-3.0 - 0.008, tfln_die.YC, wg_z), (1, 0, 0), "fibW")
    lensed_fiber(m, (3.0 + 0.008, tfln_die.YC, wg_z), (-1, 0, 0), "fibE")
    for i, (x, y) in enumerate(tfln_die.PADS):
        gsg_probe(m, (x, y, ZT + 0.0015), f"probe{i}", side=-1 if x < 0 else 1)
    objective(m, 20.0)
    # camera: from the south-south-west, low
    if A["view"] == "close":
        tgt, D, pupil, shift = Vector((0.0, tfln_die.YC + 2.0, ZT + 3.0)), 220.0, 3.0, (0.0, 0.0)
    else:  # wide station view
        tgt, D, pupil, shift = Vector((0.0, 4.0, 4.0)), 260.0, 4.0, (0.0, 0.0)
    el = math.radians(A["el"])
    az = math.radians(A["az"])
    loc = tgt + D * Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    cam = C.camera(loc, tgt, lens=36 * D / A["field"], shift=shift)
    C.set_aperture(cam, Vector((0, 0.0, ZT)), pupil_mm=pupil)
    # lights: coaxial microscope illumination from the objective, softbox reflected by the die
    C.area_light("coax_ill", (0, 0, 19.5), (0, 0, 0), 8.0, 6.0, shape="DISK", color=(1.0, 0.96, 0.9))
    mp = C.mirror_point(loc, (0, 0, ZT), 300.0)
    C.edge_softbox("key", mp, (0, 0, 0), 500.0, 3.0, edge=12.0, angle_deg=25.0, low=A["low"])
    C.area_light("side", (-220, 60, 160), (0, 0, 0), 150, 4.0e4, color=(1.0, 0.97, 0.93))
    C.area_light("rim", (200, 120, 90), (0, 0, 0), 100, 2.5e4, color=(0.92, 0.96, 1.0))
    # grazing lights along the fiber axis from behind, so the bare glass picks up highlights
    # (placed in the cylinder-mirror direction of the camera: same axial component, (y, z) mirrored)
    cdir = (loc - Vector((0, tfln_die.YC, ZT))).normalized()
    hdir = Vector((cdir.x, -cdir.y, cdir.z))
    for nm, x0 in (("fib_w", -4.0), ("fib_e", 4.0)):
        p0 = Vector((x0, tfln_die.YC, ZT))
        C.area_light(nm, p0 + 90 * hdir, p0, 12, 5.0e3, color=(1.0, 0.96, 0.92))
    C.ceiling(420.0, color=(0.05, 0.05, 0.05))
    C.surround(600.0, 800.0, color=(0.06, 0.06, 0.065))
    C.compositor(vignette=0.2, dispersion=0.004)


if __name__ == "__main__":
    scene()
    C.render(A["out"])
