"""Gel-film die tray (2 inch box) with a 4 x 3 grid of identical x-cut TFLN MZM dies; one die is
rotated 90 degrees (spot-the-hazard step). Prints the normalized image bbox of the rotated die.

Generic gel-film carrier: black conductive base, tacky translucent elastomer film, no branding.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402
import tfln_die  # noqa: E402

A = C.args(out="/tmp/gelpak.jpg", samples=128, res=1920, noise=0.015, look="AgX - Punchy", odd_row=1, odd_col=2)
PITCH_X, PITCH_Y = 9.5, 9.5


def scene():
    C.reset()
    C.setup_render(samples=A["samples"], res=(A["res"], A["res"] * 9 // 16), noise=A["noise"], look=A["look"])
    C.world_hdri("hospital_room", strength=0.35, rot_z_deg=40, bg_color=(0.03, 0.03, 0.032))
    base = C.principled("gp_base", color=(0.018, 0.018, 0.02), rough=0.45, rough_var=0.05, rough_scale=0.5)
    # gel film: satin, dark grey (ESD grade), inset from the box wall
    gel = C.principled("gp_gel", color=(0.07, 0.07, 0.075), rough=0.45, ior=1.41, rough_var=0.06, rough_scale=0.3)
    bench = C.principled("bench", color=(0.80, 0.80, 0.79), rough=0.5, rough_var=0.05, rough_scale=0.02)
    dust = C.principled("dust", color=(0.6, 0.58, 0.55), rough=0.9)
    C.box("bench", (600, 600, 10), (0, 0, -5 - 7.0), bench)
    W = 50.8
    C.box("gp_box", (W, W, 7.0), (0, 0, -3.5), base, bevel=1.2, segments=4)
    rim = C.box("gp_well_cut", (W - 3.0, W - 3.0, 4.0), (0, 0, 0.6), None)
    md = bpy.data.objects["gp_box"].modifiers.new("b", "BOOLEAN")
    md.object = rim
    md.solver = "EXACT"
    bpy.data.objects["gp_box"].modifiers.move(1, 0)
    rim.hide_render = True
    rim.hide_viewport = True
    # gel film in the well: top surface at z = -1.2
    C.box("gp_gel", (W - 5.0, W - 5.0, 0.2), (0, 0, -1.4 + 0.1), gel, bevel=0.3)
    ztop = -1.2
    mats = tfln_die.materials()
    odd = None
    import random
    rnd = random.Random(23)
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * PITCH_X + rnd.uniform(-0.25, 0.25)
            y = (1 - r) * PITCH_Y + rnd.uniform(-0.25, 0.25)
            yaw = rnd.uniform(-2.5, 2.5)  # hand placement
            odd_here = (r == A["odd_row"] and c == A["odd_col"])
            rot = (90.0 if odd_here else 0.0) + yaw
            idx = r * 4 + c + 1
            tfln_die.build(f"die_{r}{c}", mats=mats, rot_z_deg=rot, loc=(x, y, ztop + tfln_die.T),
                           label=f"MZM-{idx:02d}  X-CUT LN")
            if odd_here:
                odd = (x, y)
    C.scatter_dust("dust", ((-24, 24), (-24, 24)), ztop, 60, 0.004, 0.03, dust, seed=41, fibers=2,
                   fiber_len=(0.8, 2.0))
    tgt = Vector((0, 1.5, ztop))
    el = math.radians(66)
    D = 260.0
    loc = tgt + D * Vector((0.10, -math.cos(el), math.sin(el))).normalized()
    cam = C.camera(loc, tgt, lens=36 * D / 60.0)
    C.set_aperture(cam, tgt, pupil_mm=4.0)
    mp = C.mirror_point(loc, (0, 0, ztop + tfln_die.T), 500.0)
    C.edge_softbox("key", mp, (0, 0, 0), 900.0, 3.0, edge=40.0, angle_deg=30.0, low=0.12)
    C.area_light("top", (-120, 80, 400), (0, 0, 0), 200, 6.0e4, color=(1.0, 0.97, 0.93))
    C.area_light("fill", (250, -250, 200), (0, 0, 0), 250, 3.0e4, color=(0.95, 0.97, 1.0))
    C.ceiling(700.0, color=(0.05, 0.05, 0.05))
    C.surround(900.0, 1200.0, color=(0.06, 0.06, 0.065))
    C.compositor(vignette=0.18, dispersion=0.003)
    # normalized image bbox of the rotated die (x right, y down, origin top-left)
    bpy.context.view_layer.update()
    sc = bpy.context.scene
    h = tfln_die.DIE / 2
    pts = [Vector((odd[0] + sx * h, odd[1] + sy * h, ztop + tfln_die.T)) for sx in (-1, 1) for sy in (-1, 1)]
    uv = [world_to_camera_view(sc, cam, p) for p in pts]
    xs = [u.x for u in uv]
    ys = [1 - u.y for u in uv]
    print(f"ODD_BBOX x={min(xs):.3f} y={min(ys):.3f} w={max(xs) - min(xs):.3f} h={max(ys) - min(ys):.3f} "
          f"center=({(min(xs) + max(xs)) / 2:.3f},{(min(ys) + max(ys)) / 2:.3f})")


if __name__ == "__main__":
    scene()
    C.render(A["out"])
