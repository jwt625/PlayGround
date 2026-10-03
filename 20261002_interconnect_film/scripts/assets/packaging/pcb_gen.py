"""Parametric PCB builder (Blender 4.2, headless). All dimensions in mm, board-local frame: X right, Y up (-Y front), origin at the
board center, z = 0 at the board BOTTOM surface, top surface at z = thickness.

Usage:
    import pcb_gen
    b = pcb_gen.PCB(coll, root, "demo", 100.0, 70.0, thickness=1.6, mask="mask_green")
    b.hole(-45, -30, 3.2)                     # plated mounting hole
    b.fiducials()                             # 3 corner fiducials
    b.trace([(x, y), ...], width=0.1)         # single trace (mitered), 45-degree corners are the caller's job (see route45)
    b.diff_pair(path, width=0.1, gap=0.13)    # edge-coupled differential pair
    b.vias([(x, y), ...])                     # via pads (instanced)
    b.pads_grid(cx, cy, nx, ny, pitch, dia)   # round pads (BGA footprint), instanced
    b.pad_rect(x, y, w, d)                    # rectangular pad
    b.text("U1", x, y, size)                  # silkscreen text
    b.finish()                                # builds the objects, returns dict of created objects
Real widths used: signal trace 0.1 mm (4 mil) with 0.13 mm (5 mil) pair gap (85-100 ohm differential), via 0.2 mm drill / 0.45 mm pad,
copper 35 um (1 oz) outer, solder mask 20 um over copper (the 'masked' trace style renders the mask bulge over copper).
"""
import math

import bpy
from mathutils import Vector

import pk
from pk import MB, MM, PI

CU_T = 0.035
MASK_T = 0.02


def route45(p0, p1, first="x"):
    """Manhattan route with one 45-degree bend between two points (first = axis traveled first)."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    c = min(abs(dx), abs(dy))
    if c < 1e-9:
        return [p0, p1]
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    pts = [p0]
    if first == "y":
        if abs(dy) > c + 1e-9:
            pts.append((x0, y0 + sy * (abs(dy) - c)))
        pts.append((x0 + sx * c, y1))
    else:
        if abs(dx) > c + 1e-9:
            pts.append((x0 + sx * (abs(dx) - c), y0))
        pts.append((x1, y0 + sy * c))
    if math.dist(pts[-1], p1) > 1e-9:
        pts.append(p1)
    return pts


def offset_path(pts, dist):
    """Parallel polyline offset by dist (mm, left positive) with miter joins."""
    P = [Vector(p) for p in pts]
    out = []
    n = len(P)
    for i in range(n):
        d1 = (P[i] - P[i - 1]).normalized() if i > 0 else None
        d2 = (P[i + 1] - P[i]).normalized() if i < n - 1 else None
        if d1 is None:
            d1 = d2
        if d2 is None:
            d2 = d1
        n1 = Vector((-d1.y, d1.x))
        n2 = Vector((-d2.y, d2.x))
        m = (n1 + n2)
        m = n1 if m.length < 1e-9 else m.normalized()
        out.append(tuple(P[i] + m * (dist / max(m.dot(n1), 0.3))))
    return out


class PCB:
    def __init__(self, coll, root, name, w, d, thickness=1.6, mask="mask_green", trace_style="masked", prefix=None):
        self.coll, self.root, self.name = coll, root, name
        self.w, self.d, self.t = w, d, thickness
        self.mask = mask
        self.style = trace_style
        self.P = prefix or name
        self.holes = []
        self.cu = MB()
        self.bulge = MB()
        self.gold = MB()
        self.silk = MB()
        self.via_pts = []
        self.pad_pts = []
        self.pad_dia = {}
        self.texts = []
        self.counts = {"traces": 0, "diff_pairs": 0, "vias": 0, "pads": 0, "holes": 0}
        self.trace_len = 0.0

    # ---- features
    def hole(self, x, y, dia, plated_ring=True):
        self.holes.append((x, y, dia))
        self.counts["holes"] += 1
        if plated_ring:
            r = dia / 2
            for z in (self.t, 0.0):
                s = 1 if z else -1
                self.gold.lathe([(r + 0.05, z), (r + 1.4, z), (r + 1.4, z + s * 0.035), (r + 0.05, z + s * 0.035)], c=(x, y, 0), seg=32, mi=0, smooth=False, close=True)

    def fiducials(self, inset=3.0):
        for (sx, sy) in ((-1, -1), (1, -1), (1, 1)):
            x, y = sx * (self.w / 2 - inset), sy * (self.d / 2 - inset)
            self.gold.lathe([(0, self.t + 0.035), (0.5, self.t + 0.035), (0.5, self.t + 0.07), (0, self.t + 0.07)], c=(x, y, 0), seg=20, mi=0, smooth=False)

    def trace(self, pts, width=0.1, layer="top"):
        z0 = self.t
        self.cu.ribbon(pts, width, z0, z0 + CU_T, mi=0)
        if self.style == "masked":
            self.bulge.ribbon(pts, width + 0.12, z0 + CU_T, z0 + CU_T + MASK_T, mi=0)
        self.counts["traces"] += 1
        self.trace_len += sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def diff_pair(self, pts, width=0.1, gap=0.13):
        o = (width + gap) / 2
        self.trace(offset_path(pts, o), width)
        self.trace(offset_path(pts, -o), width)
        self.counts["diff_pairs"] += 1
        self.counts["traces"] -= 2

    def vias(self, pts):
        self.via_pts += [(p[0], p[1]) for p in pts]
        self.counts["vias"] += len(pts)

    def pads_grid(self, cx, cy, nx, ny, pitch, dia, skip=None):
        for j in range(ny):
            for i in range(nx):
                x, y = (i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch
                if skip and skip(i, j, x, y):
                    continue
                self.pad_pts.append((cx + x, cy + y, dia))
                self.counts["pads"] += 1

    def pad_rect(self, x, y, w, d):
        self.gold.box((x, y, self.t + CU_T / 2), (w, d, CU_T), mi=0)
        self.counts["pads"] += 1

    def text(self, body, x, y, size, rot=0.0, align="CENTER"):
        self.texts.append((body, x, y, size, rot, align))

    def outline(self, x0, y0, x1, y1, width=0.15):
        self.silk.ribbon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], width, self.t, self.t + 0.02, mi=0, closed=True)

    # ---- build
    def finish(self):
        P = self.P
        c, r = self.coll, self.root
        out = {}
        mb = MB()
        mb.box((0, 0, self.t / 2), (self.w, self.d, self.t), mi=(0, 1, 0))
        board = mb.to_obj(P + "_board", [pk.mat(self.mask), pk.mat("fr4"), pk.mat("copper")], c, r)
        out["board"] = board
        for k, (x, y, dia) in enumerate(self.holes):
            cm = MB()
            cm.cyl((x, y, self.t / 2), dia / 2, self.t + 0.5, seg=32, mi=2, smooth=False)
            cut = cm.to_obj(P + "_cut%d" % k, [pk.mat(self.mask), pk.mat("fr4"), pk.mat("copper")], c, r)
            md = board.modifiers.new("hole%d" % k, "BOOLEAN")
            md.operation = "DIFFERENCE"
            md.object = cut
            md.solver = "EXACT"
            bpy.context.view_layer.objects.active = board
            bpy.ops.object.modifier_apply(modifier=md.name)
            bpy.data.objects.remove(cut)
        for nm, mbx, m in (("copper_traces", self.cu, "copper"), ("mask_over_traces", self.bulge, "mask_green_light"),
                           ("gold_pads", self.gold, "gold_enig"), ("silkscreen_lines", self.silk, "silkscreen")):
            if len(mbx.bm.verts):
                out[nm] = mbx.to_obj(P + "_" + nm, [pk.mat(m)], c, r)
            else:
                mbx.bm.free()
        # vias: ring + dark hole, instanced
        if self.via_pts:
            vm = MB()
            vm.lathe([(0.10, 0.0), (0.225, 0.0), (0.225, 0.035), (0.10, 0.035)], seg=12, mi=0, smooth=False, close=True)
            vm.lathe([(0.0, 0.0), (0.10, 0.0), (0.10, 0.03), (0.0, 0.03)], seg=12, mi=1, smooth=False)
            vs = pk.source(vm.to_obj(P + "_src_via", [pk.mat("gold_enig"), pk.mat("steel_dark")], c, r))
            out["vias"] = pk.scatter(P + "_vias", [(x, y, self.t) for (x, y) in self.via_pts], vs, c, r)
        # round pads by diameter
        by = {}
        for (x, y, dia) in self.pad_pts:
            by.setdefault(dia, []).append((x, y, self.t))
        for dia, pts in by.items():
            pm = MB()
            pm.lathe([(0, 0), (dia / 2, 0), (dia / 2, CU_T), (0, CU_T)], seg=14, mi=0, smooth=False)
            ps = pk.source(pm.to_obj(P + "_src_pad_%d" % int(dia * 1000), [pk.mat("gold_enig")], c, r))
            out["pads_%d" % int(dia * 1000)] = pk.scatter(P + "_pads_%d" % int(dia * 1000), pts, ps, c, r)
        for k, (body, x, y, size, rot, al) in enumerate(self.texts):
            out["text_%d" % k] = pk.text(P + "_silk_%d" % k, body, size, (x, y, self.t + 0.0), pk.mat("silkscreen"), c, r, rot=(0, 0, rot), extrude_mm=0.02, align=al)
        return out
