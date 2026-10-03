"""Optical connector building blocks shared by several interconnect assets (module receptacles, cables, patch panels).

Local frame of every ferrule/connector: front (mating) face at y = 0 facing -Y, body extends toward +Y. X = row direction
(horizontal), Z = vertical (key up). All mm.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import ic_common as I
from ic_common import MB, _v, mm

# --- dimensions (mm) with provenance, also dumped to the asset metadata
MT = dict(width=6.4, height=2.5, length=8.0, pin_pitch=4.6, pin_dia=0.7, fiber_pitch=0.25, row_pitch=0.5, hole_dia=0.126,
          clad_dia=0.125, core_dia=0.009, apc_deg=8.0)
LC = dict(ferrule_dia=1.25, ferrule_len=6.6, pitch=6.25, body_w=5.6, body_h=8.1, body_len=20.0)
SC = dict(ferrule_dia=2.5, ferrule_len=10.5, body_w=8.4, body_h=9.0, body_len=24.0)


def boolean_cut(target, cutter, solver="EXACT"):
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    md = target.modifiers.new("cut", "BOOLEAN")
    md.operation = "DIFFERENCE"
    md.object = cutter
    md.solver = solver
    bpy.ops.object.modifier_apply(modifier=md.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def mt_ferrule(M, nfib=12, rows=1, apc=False, pins=False, name="mt", fibers=True, pin_len=6.5, pin_protrude=3.5):
    """Return list of objects (unlinked from asset coll): ferrule, [fibers, cores], [pins]. Front face at y=0 (APC: sheared)."""
    W, H, L = MT["width"], MT["height"], MT["length"]
    tan = math.tan(math.radians(MT["apc_deg"])) if apc else 0.0
    mb = MB()
    mb.box((-W / 2, W / 2), (0, L), (-H / 2, H / 2), bev=0.12, seg=1)
    fer = mb.build(name + "_ferrule", M["mt_ferrule"], smooth_deg=40)
    if apc:
        for v in fer.data.vertices:
            if v.co.y < 0.0005 * 1.0:  # front face (y ~ 0)
                v.co.y += v.co.x * tan
        # also shear verts of the bevel ring near the face
        for v in fer.data.vertices:
            if 0.0005 <= v.co.y < 0.00015 + 0.0:  # pragma: no cover
                pass
    # cutters: fiber holes and guide holes
    cm = MB()
    cols = nfib // rows
    for r in range(rows):
        zc = (r - (rows - 1) / 2) * MT["row_pitch"]
        for i in range(cols):
            xc = (i - (cols - 1) / 2) * MT["fiber_pitch"]
            y0 = xc * tan
            cm.cyl((xc, y0 + 0.9, zc), MT["hole_dia"] / 2, 2.4, axis="Y", segs=10)
    for sx in (-1, 1):
        xc = sx * MT["pin_pitch"] / 2
        y0 = xc * tan
        cm.cyl((xc, y0 + 2.4, 0.0), MT["pin_dia"] / 2 + 0.005, 5.6, axis="Y", segs=20)
    cut = cm.build(name + "_cutter", None, smooth_deg=0)
    boolean_cut(fer, cut)
    out = [fer]
    # epoxy window on top
    ew = MB()
    ew.box((-2.0, 2.0), (3.4, 6.4), (H / 2 - 0.02, H / 2 + 0.03), bev=0.05, seg=1)
    out.append(ew.build(name + "_epoxy", M["plastic_black"], smooth_deg=0))
    if fibers:
        fb = MB()
        cb = MB()
        for r in range(rows):
            zc = (r - (rows - 1) / 2) * MT["row_pitch"]
            for i in range(cols):
                xc = (i - (cols - 1) / 2) * MT["fiber_pitch"]
                y0 = xc * tan
                fb.cyl((xc, y0 + 0.5 + 0.002, zc), MT["clad_dia"] / 2, 1.0, axis="Y", segs=8)
                cb.cyl((xc, y0 + 0.0015, zc), MT["core_dia"] / 2, 0.004, axis="Y", segs=6)
        out.append(fb.build(name + "_fibers", M["fiber_glass"], smooth_deg=0))
        out.append(cb.build(name + "_cores", M["fiber_core"], smooth_deg=0))
    if pins:
        pb = MB()
        for sx in (-1, 1):
            xc = sx * MT["pin_pitch"] / 2
            y0 = xc * tan
            ytip = y0 - pin_protrude
            pb.cyl((xc, ytip + pin_len / 2, 0.0), MT["pin_dia"] / 2 - 0.005, pin_len, axis="Y", segs=14)
        out.append(pb.build(name + "_pins", M["steel"], smooth_deg=40))
    return out


def lathe_y(mb, prof, segs=32, cx=0.0, cz=0.0, y_off=0.0):
    """Solid of revolution about the Y axis. prof: list of (radius, y) from front to rear (open polyline); closed by caller profile."""
    bm = mb.bm
    rings = []
    for (r, y) in prof:
        ring = []
        for k in range(segs):
            a = 2 * math.pi * k / segs
            ring.append(bm.verts.new(_v(cx + r * math.cos(a), y + y_off, cz + r * math.sin(a))))
        rings.append(ring)
    for i in range(len(prof) - 1):
        for k in range(segs):
            k2 = (k + 1) % segs
            a, b = rings[i][k], rings[i][k2]
            c, d = rings[i + 1][k2], rings[i + 1][k]
            if prof[i][0] == 0 and prof[i + 1][0] == 0:
                continue
            try:
                if prof[i][0] == 0:
                    bm.faces.new((a, c, d))
                elif prof[i + 1][0] == 0:
                    bm.faces.new((a, b, d))
                else:
                    bm.faces.new((a, b, c, d))
            except ValueError:
                pass
    return rings


def round_ferrule(M, dia, length, bore_dia=0.1255, chamfer=0.12, apc=False, name="ferrule", fiber=True, core_dia=0.009, clad_dia=0.125, segs=40):
    """Ceramic (zirconia) cylindrical ferrule (LC 1.25 mm / SC 2.5 mm) with 125 um bore, fibre cladding and glowing 9 um core. Front face y = 0.
    Returns [ferrule, fiber, core] objects (front face centred on the origin; body toward +Y)."""
    R = dia / 2
    rb = bore_dia / 2
    prof = [(rb, 1.4), (rb, 0.0), (R - chamfer, 0.0), (R, chamfer), (R, length), (rb * 3, length), (rb * 3, length - 0.2)]
    mb = MB()
    lathe_y(mb, prof, segs=segs)
    if apc:
        tan = math.tan(math.radians(8.0))
        for v in mb.bm.verts:
            if v.co.y < (chamfer + 0.01) * 0.001:
                v.co.y += v.co.x * tan
    out = [mb.build(name, M["ferrule_ceramic"], smooth_deg=40)]
    if fiber:
        fb = MB()
        fb.cyl((0, 0.7, 0), clad_dia / 2, 1.4, axis="Y", segs=16)
        out.append(fb.build(name + "_fiber", M["fiber_glass"], smooth_deg=0))
        cb = MB()
        cb.cyl((0, 0.0015, 0), core_dia / 2, 0.004, axis="Y", segs=10)
        out.append(cb.build(name + "_core", M["fiber_core"], smooth_deg=0))
        if apc:
            for o in out[1:]:
                for v in o.data.vertices:
                    if v.co.y < 0.0004:
                        v.co.y += v.co.x * math.tan(math.radians(8.0))
    return out
