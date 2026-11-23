"""Shot 2: the x-cut TFLN MZM die (correct orientation) on a probe station, in its die fixture.

Optical: lensed fibers on the waveguide axis at the west (input) and east (output) facets, about
8 um from the facet, held from above by V-groove holders on arms that run out along the optical axis
to fiber stages on the platen. RF: the CPW bends to pad sets inside the north and south edges
(tfln_die.py); the input GSG probe (150 um pitch) lands from the north, the termination probe
(port 2) from the south, so neither crosses a fiber. Each probe is bolted under an L-shaped arm on a
manual XYZ positioner on the platen (about 30 mm above the die), with a mated 1.85 mm cable plug and
a phase-stable test cable. A long-working-distance objective (33.5 mm WD) hangs above the die.
Parts and dimensions: rf_probe.py and research/ref_probes/README.md.

view=close: die, tips, fibers (course hotspot image). view=wide: whole station around the chuck.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import common as C  # noqa: E402
import rf_probe as R  # noqa: E402
import shot_die_fixture as F  # noqa: E402
import tfln_die  # noqa: E402

# camera presets: the only free sight lines between the four positioners are the diagonals
PRESETS = {
    "close": dict(el=60.0, field=40.0, az=-142.0, pupil=3.0, tx=-2.0, ty=3.0, tz=0.0, exposure=0.8, dist=240.0,
                  key=1.6),
    "wide": dict(el=55.0, field=230.0, az=-135.0, pupil=3.0, tx=0.0, ty=0.0, tz=0.0, exposure=0.6, dist=900.0,
                 key=3.0),
}
VIEW = C.args(view="close")["view"]
A = C.args(out="/tmp/probe.jpg", samples=256, res=1920, noise=0.01, look="AgX - Punchy", view="close", low=0.30,
           back=2.0e5, rimn=6.0e4, frontn=1.0e5, **PRESETS[VIEW])
ZT = F.ZT                 # die top surface height (0.3 mm above the fixture)
Z_PL = 30.0               # platen top: positioner bases about 30 mm above the probe tips
R_POS = 66.0              # probe tip to positioner front face (published RF arm: 137.6 - 70 = 67.6 mm)
R_FIB = 65.5              # fiber tip to fiber-stage front face (front flush with the platen opening)


def mats():
    m = R.materials()
    m["glass"] = C.principled("fiber_glass", color=(1, 1, 1), rough=0.0, ior=1.46, transmission=1.0)
    m["coat"] = C.principled("acrylate", color=(0.85, 0.88, 0.80), rough=0.25, ior=1.5, transmission=0.6)
    m["jacket"] = C.principled("jacket_yellow", color=(0.75, 0.45, 0.02), rough=0.45)
    m["black"] = C.principled("black_anod", color=(0.015, 0.015, 0.017), rough=0.4, rough_var=0.05, rough_scale=1.0)
    m["engr"] = C.principled("obj_engr", color=(0.8, 0.8, 0.8), rough=0.6)
    m["lens"] = C.principled("obj_lens", color=(1, 1, 1), rough=0.0, ior=1.6, transmission=1.0,
                             film_nm=140.0, film_ior=1.38)
    m["chrome"] = C.principled("obj_chrome", color=(0.9, 0.9, 0.9), metallic=1.0, rough=0.08)
    m["scope"] = C.principled("scope_paint", color=(0.55, 0.56, 0.57), rough=0.35, coat=0.3)
    return m


BARE = 2.0  # stripped (bare glass) length behind the lensed tip, mm


def lensed_fiber(m, tip, direction, name):
    """Bare 125 um fiber with a conical lensed tip ending at `tip`, pointing along `direction`, held by a
    V-groove clamp on an arm that runs back along the fiber axis to a fiber stage on the platen."""
    d = Vector(direction).normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bm = bmesh.new()
    rt, rf, Lc = 0.004, 0.0625, 0.16
    prof = []
    for i in range(9):  # hemisphere
        a = math.pi / 2 * i / 8
        prof.append((rt * math.sin(a), -rt + rt * math.cos(a)))
    prof.append((rf, -Lc))
    prof.append((rf, -BARE))
    prof = [(0.0, 0.0)] + prof[1:]
    seg = 48
    rings = []
    for r, z in prof:
        rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / seg), r * math.sin(2 * math.pi * k / seg), z))
                      for k in range(seg)])
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
    # fiber-local frame: +x back along the fiber toward the stage, z up
    sx = -1.0 if d.x > 0 else 1.0   # world x of "back"
    root = C.empty(name + "_root", loc=tip, rot=(0, 0, 0 if sx > 0 else math.pi))
    P = []
    # 250 um acrylate coating from the end of the stripped section to the clamp and beyond
    P.append(R.lathe(name + "_coat", [(0.0, BARE), (0.125, BARE), (0.125, 8.4), (0.0, 8.4)], m["coat"], seg=24))
    P[-1].matrix_world = Matrix.Rotation(math.pi / 2, 4, "Y")
    # V-groove clamp over the coated fiber (5 x 3.5 mm block, lid with one M2 screw) on a black anodized
    # holder bar rising at 12 degrees back to the drop plate on the fiber stage
    xc = BARE + 3.5
    P.append(C.box(name + "_clamp", (5.0, 3.5, 2.4), (xc, 0, 0.125 + 1.2), m["anod_pos"], bevel=0.2))
    P.append(C.box(name + "_lid", (3.0, 3.7, 0.7), (xc, 0, 0.125 + 2.4 + 0.35), m["anod_pos"], bevel=0.12))
    P.append(R.cap_screw(name + "_lidscr", (xc, 0, 0.125 + 3.1), 2.0, m))
    x_bar0, x_drop = xc + 2.0, R_FIB - 8.0
    ang = math.radians(12.0)
    Lb = (x_drop - x_bar0) / math.cos(ang) + 2.0
    zb0 = 0.125 + 2.4
    zc = zb0 + 2.0 + (x_drop - x_bar0) / 2 * math.tan(ang)
    bar = C.box(name + "_bar", (Lb, 5.0, 4.0), ((x_bar0 + x_drop) / 2, 0, zc), m["anod_pos"], bevel=0.3,
                rot=(0, -ang, 0))
    P.append(bar)
    z_top = Z_PL + 76.0 - tip[2]
    z_bar_end = zb0 + (x_drop - x_bar0) * math.tan(ang)
    P.append(C.box(name + "_drop", (8.0, 18.0, z_top - z_bar_end), (x_drop + 4.0, 0, (z_top + z_bar_end) / 2),
                   m["anod_pos"], bevel=0.5))
    # Fiber route (fiber frame, checked with check_clearance.py), as on a real bench: the coated fiber leaves
    # the V-groove straight, steps sideways under the rising bar, is taped along the bar's side, then leaves
    # the bar in a gentle climb past the drop plate onto the platen beside the stage, where the 900 um
    # jacket starts; the jacket lies on the platen along the stage and drapes off the platen edge.
    # Bend radii (fillets): coated fiber 20 mm, jacket 30 mm.
    tb = math.tan(ang)

    def zmid(x):
        return zb0 + 2.0 + (x - x_bar0) * tb

    yf = 2.5 + 0.125 + 0.4
    zpl = Z_PL - tip[2]                      # platen top in the fiber frame
    fib = R.fillet_path([Vector((8.3, 0, 0)), Vector((17.0, 0, 0)), Vector((26.0, yf, 3.0)),
                         Vector((40.0, yf, zmid(40.0))), Vector((68.0, 37.0, 33.0)),
                         Vector((112.0, 40.0, zpl + 0.175))], 20.0, step=0.4)
    P.append(C.curve_tube(name + "_fib", fib, 0.125, m["coat"], resolution=2, bevel_res=3))
    zj = zpl + 0.5
    x_edge = 210.0 - abs(tip[0])             # platen edge in the fiber frame
    jk = R.fillet_path([Vector((109.0, 39.8, zj)), Vector((150.0, 42.0, zj)), Vector((x_edge - 20.0, 46.0, zj)),
                        Vector((x_edge + 32.0, 48.0, zj)), Vector((x_edge + 33.0, 48.5, zj - 90.0))],
                       30.0, step=0.6)
    P.append(C.curve_tube(name + "_jkt", jk, 0.45, m["jacket"], resolution=2, bevel_res=4))
    C.parent(P, root)
    stage = R.positioner(m, name + "_stage", (tip[0] + sx * R_FIB, tip[1]), (-sx, 0.0), Z_PL, micrometers=3)
    return root, stage


def rf_probe_assembly(m, pad, direction, name, cable_route):
    """Probe on the pads with its arm, positioner and cable. `direction`: unit (x, y) from the pads toward
    the probe body. `cable_route`: points (probe frame) after the plug."""
    root, mate, cdir = R.gsg_probe(m, name)
    arm = R.probe_arm(m, name + "_arm", R_POS - 8.0, Z_PL + 80.0 - pad[2])
    C.parent(arm, root)
    R.cable_plug(m, name + "_plug", mate, cdir, cable_route, parent=root)
    root.location = pad
    root.rotation_euler = (0, 0, math.atan2(direction[1], direction[0]))
    pos = R.positioner(m, name + "_pos", (pad[0] + direction[0] * R_POS, pad[1] + direction[1] * R_POS),
                       (-direction[0], -direction[1]), Z_PL)
    return root, pos


def objective(m, z_front, name="obj"):
    parts = []
    parts.append(C.cylinder(name + "_nose", 9.5, 6.0, (0, 0, z_front + 3.0), m["black"], verts=96, bevel=0.3,
                            r2=11.0))
    parts.append(C.cylinder(name + "_ring", 11.2, 1.2, (0, 0, z_front + 6.6), m["chrome"], verts=96, bevel=0.1))
    parts.append(C.cylinder(name + "_barrel", 16.0, 40.0, (0, 0, z_front + 27.2), m["black"], verts=96,
                            bevel=0.4))
    parts.append(C.cylinder(name + "_lens", 4.2, 0.6, (0, 0, z_front + 0.1), m["lens"], verts=64))
    parts.append(C.cylinder(name + "_bezel", 5.2, 0.8, (0, 0, z_front - 0.1), m["chrome"], verts=64))
    t = C.text_mesh(name + "_t", "PLAN APO 10x / 0.28   f = 200   WD 33.5", 1.3, (0, -16.02, z_front + 16.0),
                    h=0.02, mat=m["engr"], rot=(math.radians(90), 0, 0))
    parts.append(t)
    # zoom tube, coaxial illuminator and focus block of the microscope above
    parts.append(C.cylinder(name + "_tube", 19.0, 120.0, (0, 0, z_front + 47.2 + 60.0), m["scope"], verts=96,
                            bevel=0.6))
    parts.append(C.box(name + "_ill", (46.0, 40.0, 34.0), (0, 0, z_front + 47.2 + 70.0), m["scope"], bevel=2.0))
    parts.append(C.cylinder(name + "_lg", 6.0, 30.0, (-35.0, 0, z_front + 47.2 + 70.0), m["black"], verts=32,
                            rot=(0, math.radians(90), 0)))
    return parts


def scene():
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"],
                   exposure=A["exposure"])
    C.world_hdri("hospital_room", strength=0.30, rot_z_deg=130, bg_color=(0.025, 0.025, 0.027))
    fm = F.fixture_materials()
    F.build_fixture(fm)
    C.cylinder("chuck", 150.0, 10.0, (6.0, 10.0, -11.0), F.chuck_material(), verts=256)
    tfln_die.build("die", rot_z_deg=0, loc=(0, 0, ZT), skate=True)
    C.scatter_dust("dust_die", ((-2.9, 2.9), (-2.9, 2.9)), ZT, 22, 0.002, 0.014, fm["dust"], seed=3, fibers=1,
                   fiber_len=(0.3, 0.6))
    C.scatter_dust("dust_fix", ((-19, 19), (-13, 13)), 0.0, 90, 0.004, 0.03, fm["dust"], seed=5, fibers=2,
                   fiber_len=(0.8, 1.8))
    m = mats()
    wg_z = ZT - 0.0003
    lensed_fiber(m, (-3.0 - 0.008, tfln_die.YC, wg_z), (1, 0, 0), "fibW")
    lensed_fiber(m, (3.0 + 0.008, tfln_die.YC, wg_z), (-1, 0, 0), "fibE")
    # cables (probe frame, +x back along the probe): up and back from the plug, then down to the
    # instrument ports off the set with a 60-80 mm bend radius
    route = [Vector((0.0, 0.0, 120.0)), Vector((95.0, 55.0, 150.0)),
             Vector((170.0, 90.0, 140.0)), Vector((280.0, 160.0, 40.0))]
    for i, ((x, y), dr) in enumerate(zip(tfln_die.PADS, tfln_die.PAD_DIRS)):
        rf_probe_assembly(m, (x, y, ZT + tfln_die.H_AU), dr, f"probe{i}", route)
    R.platen(m, Z_PL, opening=(137.0, 137.0))
    objective(m, ZT + 33.5)
    # camera
    if A["view"] == "close":
        tgt = Vector((A["tx"], A["ty"] + tfln_die.YC + 0.2, ZT + A["tz"]))
        D, pupil = A["dist"], A["pupil"]
        focus = Vector((0.0, tfln_die.YC + 0.3, ZT))
    else:
        tgt = Vector((A["tx"], A["ty"], 30.0 + A["tz"]))
        D, pupil = A["dist"], A["pupil"]
        focus = Vector((0.0, 0.0, 10.0))
    el = math.radians(A["el"])
    az = math.radians(A["az"])
    loc = tgt + D * Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
    cam = C.camera(loc, tgt, lens=36 * D / A["field"], clip=(1.0, 1e5))
    C.set_aperture(cam, focus, pupil_mm=pupil)
    # lights: coaxial microscope illumination, softbox reflected by the die, room fill
    C.area_light("coax_ill", (0, 0, ZT + 33.0), (0, 0, 0), 8.0, 12.0, shape="DISK", color=(1.0, 0.96, 0.9))
    # large soft source high behind the set: rim and sheen on the black anodized parts
    C.area_light("back_soft", (160, 260, 320), (0, 0, 0), 300, A["back"], color=(0.95, 0.97, 1.0))
    mp = C.mirror_point(loc, (0, 0, ZT), 300.0)
    C.edge_softbox("key", mp, (0, 0, 0), 500.0, A["key"], edge=12.0, angle_deg=25.0, low=A["low"])
    C.area_light("side", (-220, 60, 260), (0, 0, 0), 200, 6.0e4, color=(1.0, 0.97, 0.93))
    C.area_light("rim", (200, 160, 200), (0, 0, 0), 160, 4.0e4, color=(0.92, 0.96, 1.0))
    C.area_light("fill", (60, -260, 220), (0, 0, 0), 220, 3.0e4, color=(1.0, 0.98, 0.95))
    # rim on the north (input) probe body from behind-left
    C.area_light("rim_n", (-160, 190, 140), (-2.0, 22.0, 6.0), 80, A["rimn"], color=(0.95, 0.97, 1.0))
    C.area_light("front_n", (60, -80, 60), (-2.0, 14.0, 5.0), 60, A["frontn"], color=(1.0, 0.98, 0.95))
    # grazing lights along the fiber axis from behind, so the bare glass picks up highlights
    cdir = (loc - Vector((0, tfln_die.YC, ZT))).normalized()
    hdir = Vector((cdir.x, -cdir.y, cdir.z))
    for nm, x0 in (("fib_w", -4.0), ("fib_e", 4.0)):
        p0 = Vector((x0, tfln_die.YC, ZT))
        C.area_light(nm, p0 + 90 * hdir, p0, 12, 5.0e3, color=(1.0, 0.96, 0.92))
    C.ceiling(600.0, color=(0.05, 0.05, 0.05))
    C.surround(900.0, 1200.0, color=(0.06, 0.06, 0.065))
    C.compositor(vignette=0.2, dispersion=0.004)


if __name__ == "__main__":
    scene()
    C.render(A["out"])
