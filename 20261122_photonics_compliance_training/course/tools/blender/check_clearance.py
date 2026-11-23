"""Clearance check for the probe-station routes (fibers, jackets, RF cables): builds the scene of
shot_probe_station.py without rendering, samples every route centreline every 0.25 mm and prints each
solid it enters (signed distance from the centreline minus the route radius below -tol), plus the
smallest bend radius of each route.

  bslot.sh -b --factory-startup --python check_clearance.py -- view=wide
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
from mathutils.geometry import interpolate_bezier  # noqa: E402

import shot_probe_station as P  # noqa: E402

# route suffix -> objects (name prefixes after the assembly name) it may touch at its ends
ROUTES = {"_fib": ("_coat", "_clamp", "_jkt"), "_jkt": ("_fib",), "_plug_cable": ("_plug_boot", "_plug_grip")}


def centerline(ob, n=24):
    pts = []
    for sp in ob.data.splines:
        bp = sp.bezier_points
        for a, b in zip(bp[:-1], bp[1:]):
            seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, n)
            pts += seg if not pts else seg[1:]
    return [ob.matrix_world @ p for p in pts]


def resample(pts, step):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        L = (b - a).length
        k = max(1, int(L / step))
        out += [a + (b - a) * (i / k) for i in range(1, k + 1)]
    return out


def bend_radius(pts, k=12):
    """Smallest radius of the circle through samples i-k, i, i+k (3 mm chords at 0.25 mm sampling, long
    enough to average out the polyline vertices of the filleted routes). Returns (radius, point)."""
    best = (1e9, None)
    for i in range(k, len(pts) - k):
        a, b, c = pts[i - k], pts[i], pts[i + k]
        ab, bc, ca = (b - a).length, (c - b).length, (a - c).length
        area2 = (b - a).cross(c - a).length
        if area2 > 1e-9:
            r = ab * bc * ca / (2 * area2)
            if r < best[0]:
                best = (r, b)
    return best


def main(tol=0.02):
    P.scene()
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    solids = {}
    for ob in bpy.data.objects:
        if ob.type not in ("MESH", "CURVE") or ob.hide_render or ob.name.startswith(("dust", "ceiling", "surround")):
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        if me is None or not me.polygons:
            ev.to_mesh_clear()
            continue
        vs = [ob.matrix_world @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        ev.to_mesh_clear()
        lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
        hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
        solids[ob.name] = (BVHTree.FromPolygons(vs, polys), lo, hi)
    bad = 0
    for ob in [o for o in bpy.data.objects if o.type == "CURVE"]:
        suf = next((s for s in ROUTES if ob.name.endswith(s)), None)
        if suf is None:
            continue
        base = ob.name[: -len(suf)]
        touch = tuple(base + t for t in ROUTES[suf])
        rad = ob.data.bevel_depth
        raw = centerline(ob)
        pts = resample(raw, 0.25)
        worst = {}
        for p in pts:
            for sn, (bvh, lo, hi) in solids.items():
                if sn == ob.name or sn.startswith(touch):
                    continue
                m = rad + 0.1
                if not (lo.x - m <= p.x <= hi.x + m and lo.y - m <= p.y <= hi.y + m and lo.z - m <= p.z <= hi.z + m):
                    continue
                inside = lo.x <= p.x <= hi.x and lo.y <= p.y <= hi.y and lo.z <= p.z <= hi.z
                hit = bvh.find_nearest(p) if inside else bvh.find_nearest(p, m)
                if hit[0] is None:
                    continue
                d = (p - hit[0]).length
                sd = d if (p - hit[0]).dot(hit[1]) >= 0 else -d
                c = sd - rad
                if c < worst.get(sn, (1e9, None))[0]:
                    worst[sn] = (c, p)
        for sn, (c, p) in sorted(worst.items(), key=lambda kv: kv[1][0]):
            if c < -tol:
                print(f"CHECK {ob.name}: inside {sn} by {-c:.2f} mm at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
                bad += 1
        br = bend_radius(pts)
        print(f"ROUTE {ob.name:22s} r={rad:.3f} length={sum((b - a).length for a, b in zip(raw[:-1], raw[1:])):.0f} mm "
              "min bend radius={:.1f} mm at ({:.1f}, {:.1f}, {:.1f})".format(br[0], *br[1]))
    print(f"CHECK_DONE violations={bad}")


if __name__ == "__main__":
    main()
