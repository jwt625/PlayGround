"""Print normalized image coordinates (x right, y down, 0..1) of named world points for the
perspective shots, by building each scene (no render) and projecting through its camera.
Used to place hotspot pins in research/media_plan_render.md.

  bslot.sh -b --factory-startup --python project_points.py
"""
import importlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402


def proj(points):
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    for name, p in points.items():
        u = world_to_camera_view(sc, sc.camera, Vector(p))
        print(f"  {name:28s} x={u.x:.3f} y={1 - u.y:.3f}")


def run(label, modname, argv, call, points):
    sys.argv = ["blender", "--"] + argv
    mod = importlib.import_module(modname)
    mod = importlib.reload(mod)
    import common as C
    C.reset()
    if modname == "shot_die_fixture":
        mod.scene(mod.A["variant"])
    else:
        getattr(mod, call)()
    print(label)
    proj(points)


a = math.radians
YC = -0.4
ZT = 0.3
for v, arrow, mz in ((0, (0, 1.15, ZT), (0, YC, ZT)), (90, (-1.15, 0, ZT), (-YC, 0, ZT)),
                     (180, (0, -1.15, ZT), (0, -YC, ZT))):
    run(f"tfln_die_fixture_{v}deg", "shot_die_fixture", [f"variant={v}"], "scene", {
        "die +Z arrow": arrow, "MZM electrode centre": mz, "fixture +Z engraving": (-6.2, 4.5, 0),
        "fiber slot IN (mouth at pocket)": (-3.6, YC, 0), "fiber slot OUT (mouth at pocket)": (3.6, YC, 0),
        "IN label": (-9.0, YC - 2.2, 0), "OUT label": (9.0, YC - 2.2, 0)})
run("tfln_die_fixture_0deg extra", "shot_die_fixture", ["variant=0"], "scene", {
    "RF input pads": (-2.05, 1.975, ZT), "termination pads": (2.05, -1.975, ZT),
    "input MMI splitter": (-2.66, YC, ZT), "output MMI combiner": (2.66, YC, ZT),
    "fixture RF engraving": (-3.7, 5.0, 0)})
run("tfln_probe_station_closeup", "shot_probe_station", ["view=close"], "scene", {
    "GSG probe on RF input pads": (-2.05, 1.975, ZT), "GSG probe on termination pads": (2.05, -1.975, ZT),
    "input probe coax (1 mm up)": (-2.05, 3.0, ZT + 0.9), "termination probe coax (1 mm up)": (2.05, -3.0, ZT + 0.9),
    "lensed fiber at input facet": (-3.01, YC, ZT), "input fiber 1 mm back": (-4.0, YC, ZT),
    "lensed fiber at output facet": (3.01, YC, ZT), "output fiber 1 mm back": (4.0, YC, ZT),
    "die +Z arrow": (0, 1.15, ZT), "fixture +Z engraving": (-6.2, 4.5, 0), "fixture RF engraving": (-3.7, 5.0, 0)})
run("tfln_probe_station_wide", "shot_probe_station", ["view=wide"], "scene", {
    "die centre": (0, YC, ZT), "input probe plug": (-2.05, 22.0, 18.0), "termination probe plug": (2.05, -22.0, 18.0),
    "objective": (0, 0, 40)})
pf = math.sqrt(50 ** 2 - 16.25 ** 2)
ps = math.sqrt(50 ** 2 - 9.0 ** 2)
run("ln_wafer_carrier", "shot_wafer", ["scene=carrier"], "scene", {
    "wafer centre": (0, 0, 3.5), "primary flat": (pf * math.cos(a(-120)), pf * math.sin(a(-120)), 3.5),
    "secondary flat": (ps * math.cos(a(150)), ps * math.sin(a(150)), 3.5), "floor mark seen through": (8, -20, 1.7)})
run("fc_apc_vs_pc_ferrules", "shot_butterfly", ["view=ferrules"], "scene_ferrules", {
    "APC tip": (0, 15, 5.0), "PC tip": (0, 0, 5.0), "APC key tab": (-8.1, 15, 8.4), "PC key tab": (-8.1, 0, 8.4),
    "green boot": (-38, 15, 3.0), "blue boot": (-38, 0, 3.0), "APC knurled nut": (-18.8, 15, 5.5),
    "jacket leaving boot": (-56, 15, 1.2)})
run("ln_wafer_hotplate", "shot_wafer", ["scene=hotplate"], "scene", {
    "wafer centre": (0, 0, 95.5), "primary flat": (pf * math.cos(a(-78)), pf * math.sin(a(-78)), 95.5),
    "temperature display": (-34, -192, 48), "caution label": (-50, -190.5, 8)})
for view, reverse in (("main", "0"), ("iso_close", "1")):
    run(f"butterfly view={view}", "shot_butterfly", [f"view={view}", f"iso_reverse={reverse}"], "scene_main", {
        "butterfly package": (25, 25, 30), "M6 screw (front left)": (-25, 0, 8), "laser D-sub": (64, 25, 15),
        "inline isolator centre": (-60, -45, 2.75), "isolator end (laser side)": (-35.3, -51.2, 2.75),
        "isolator end (output side)": (-84.7, -38.8, 2.75), "splice sleeve (laser side)": (-5.3, -64.5, 1.5),
        "splice sleeve (output side)": (-114.7, -31.3, 1.5), "FC/APC (green, capped)": (62, -97, 5.1),
        "FC/PC (blue, capped)": (68, -122.5, 5.1)})
