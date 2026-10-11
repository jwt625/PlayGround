"""Front cabling for the GB300 NVL72 rack (v4 wave, DevLog-007 section 12): pluggable modules, dust caps, MPO-12/APC
connectors, single-mode fiber jumpers dressed into vertical bundles with velcro straps, slim Cat6 to the in-rack
management switches. All geometry is merged into a few meshes per rack (cheap; the hall instances the rack collection).

Sources (research/racks_refs.md section 6): ConnectX-8 OSFP ports = 4 x 800G per compute tray (Lenovo GB300 UG Fig. 2,
Gigabyte XN15-CB0 spec); NIC-side module = NVIDIA MMS4A20 single-port 800G DR4 OSFP, flat top (RHS), one MPO-12/APC
(green housing), yellow pull tab in NVIDIA's figure; BlueField-3 B3240 2 x QSFP112; NVLink switch trays have no front
optics (NVLink is on the rear cable cartridges).
All lengths in mm in tray-local frames (port face plane x = 0, +x toward the aisle) unless noted.
"""
import math

from mathutils import Vector

import rk_lib as K
from rk_lib import MB

m = lambda x: x * 0.001
FIBER_R = 1.5          # MPO-12 round cable 3.0 mm OD (C: typical 3.0 mm round jumper)
CAT6_R = 2.1           # slim Cat6 patch cord 4.2 mm OD (C: typical 28 AWG slim patch 3.8-4.5 mm)


def osfp_module(mb, P, mi):
    """Populated OSFP (flat-top RHS) at port centre P=(x,y,z) in metres, rack frame (+x out). mi: dict of material idx."""
    x, y, z = P
    def b(x0, x1, y0, y1, z0, z1, k):
        mb.box2((x + m(x0), y + m(y0), z + m(z0)), (x + m(x1), y + m(y1), z + m(z1)), mi[k])
    b(1.5, 15.0, -11.29, 11.29, -6.5, 6.5, "module")                    # module nose (22.58 x 13.00 mm, OSFP MSA)
    b(15.0, 16.0, -9.0, 9.0, -4.8, 4.8, "dark")                          # receptacle face
    b(16.0, 40.0, -6.2, 6.2, -3.7, 3.7, "mpo")                           # MPO-12/APC housing (green)
    mb.cyl((x + m(49), y, z), m(4.6), m(18), "x", 8, mi["boot"], r2=m(2.2))   # strain-relief boot
    loop = [(13, -10, 6.9), (30, -10.5, 6.6), (44, -9, 4.0), (51, 0, -1.0), (44, 9, 4.0), (30, 10.5, 6.6), (13, 10, 6.9)]
    mb.tube(K.smooth_path([(x + m(a), y + m(b_), z + m(c)) for a, b_, c in loop], 4), m(1.1), 6, mi["tab"])   # yellow pull-tab loop
    b(10.0, 14.0, -11.0, 11.0, 6.5, 7.3, "tab")                           # tab root on the module top
    return (x + m(58), y, z)                                             # fiber exit


def qsfp_module(mb, P, mi):
    x, y, z = P
    def b(x0, x1, y0, y1, z0, z1, k):
        mb.box2((x + m(x0), y + m(y0), z + m(z0)), (x + m(x1), y + m(y1), z + m(z1)), mi[k])
    b(1.0, 12.0, -9.17, 9.17, -4.25, 4.25, "module")                     # QSFP112 nose (18.35 x 8.5 mm)
    b(12.0, 36.0, -6.2, 6.2, -3.7, 3.7, "mpo")
    mb.cyl((x + m(45), y, z), m(4.6), m(18), "x", 8, mi["boot"], r2=m(2.2))
    b(10.0, 46.0, -4.0, 4.0, 4.25, 5.0, "tab")
    return (x + m(54), y, z)


def osfp_cap(mb, P, mi):
    x, y, z = P
    mb.box2((x + m(1.5), y - m(10.6), z - m(6.0)), (x + m(9.0), y + m(10.6), z + m(6.0)), mi["cap"])
    mb.box2((x + m(9.0), y - m(4.0), z - m(1.0)), (x + m(17.0), y + m(4.0), z + m(1.0)), mi["cap"])


def qsfp_cap(mb, P, mi):
    x, y, z = P
    mb.box2((x + m(1.0), y - m(8.4), z - m(3.6)), (x + m(8.0), y + m(8.4), z + m(3.6)), mi["cap"])
    mb.box2((x + m(8.0), y - m(3.0), z - m(0.8)), (x + m(14.0), y + m(3.0), z + m(0.8)), mi["cap"])


def rj45_plug(mb, P, mi):
    x, y, z = P
    mb.box2((x + m(1.2), y - m(5.9), z - m(4.6)), (x + m(14.0), y + m(5.9), z + m(4.6)), mi["plug"])
    mb.cyl((x + m(22), y, z), m(5.0), m(16), "x", 8, mi["cat6_boot"], r2=m(2.6))
    return (x + m(30), y, z)


def arc_route(s, c, r_h=0.035, r_v=0.03, n=8):
    """Exit route made of true arcs: straight out of the connector (+x), quarter arc in the horizontal plane toward the
    bundle side, straight run along y, quarter arc in the y-z plane turning up, short straight up. Bend radii r_h / r_v
    (clamped to the available offsets, never below 15 mm). Returns points (m)."""
    sx, sy, sz = s
    cx, cy = c
    dy = cy - sy
    sg = 1.0 if dy >= 0 else -1.0
    d = cx - sx
    rh = max(0.015, min(r_h, d - 0.005, abs(dy) - 0.017))
    rv = max(0.015, min(r_v, abs(dy) - rh - 0.002))
    x0 = cx - rh                                                   # end of the straight run out of the connector
    pts = [(sx, sy, sz)]
    if x0 > sx + 0.002:
        pts.append((x0, sy, sz))
    for k in range(1, n + 1):                                      # horizontal arc: +x -> sg*y, centre (x0, sy + sg*rh)
        a = (math.pi / 2) * k / n
        pts.append((x0 + rh * math.sin(a), sy + sg * rh * (1 - math.cos(a)), sz))
    y1 = cy - sg * rv
    if abs(y1 - (sy + sg * rh)) > 0.002:
        pts.append((cx, y1, sz))
    for k in range(1, n + 1):                                      # vertical arc: sg*y -> +z, centre (cx, y1, sz + rv)
        a = (math.pi / 2) * k / n
        pts.append((cx, y1 + sg * rv * math.sin(a), sz + rv * (1 - math.cos(a))))
    pts.append((cx, cy, sz + rv + 0.04))
    return pts


def min_radius(pts):
    """Smallest circumradius of consecutive point triples (bend radius of the sampled path), metres."""
    best = 9.0
    P = [Vector(p) for p in pts]
    for a, b, c in zip(P, P[1:], P[2:]):
        ab, bc, ca = (b - a).length, (c - b).length, (a - c).length
        if min(ab, bc) < 1e-4:
            continue
        area = (b - a).cross(c - a).length / 2
        if area < 1e-12:
            continue
        best = min(best, ab * bc * ca / (4 * area))
    return best


def hex_offsets(n, pitch):
    """n positions in a compact hex bundle (innermost first), metres."""
    pts = []
    rng = int(math.sqrt(n)) + 3
    for i in range(-rng, rng + 1):
        for j in range(-rng, rng + 1):
            px = (i + 0.5 * j) * pitch
            py = j * pitch * math.sqrt(3) / 2
            pts.append((math.hypot(px, py), math.atan2(py, px), px, py))
    pts.sort()
    return [(p[2], p[3]) for p in pts[:n]]


class Bundle:
    """Vertical bundle at (bx, by) (metres, rack frame). Members join from below in order; each gets a hex offset."""

    def __init__(self, name, bx, by, pitch, r_cable):
        self.name, self.bx, self.by, self.pitch, self.r = name, bx, by, pitch, r_cable
        self.members = []      # (z_join, start point, end info)

    def add(self, start, z_join, end=None):
        self.members.append((z_join, start, end))

    def build(self, mb_cable, mb_strap, z_top, mi_cable=0, mi_strap=0, strap_pitch=0.25, fan_out=None, fwd=0.022):
        """fan_out(member_index_in_top_order, (x, y, z_bundle_top)) -> list of points after the bundle top (optional)."""
        self.members.sort(key=lambda t: t[0])
        offs = hex_offsets(len(self.members), self.pitch)
        tops = []
        for k, ((zj, s, end), (dx, dy)) in enumerate(zip(self.members, offs)):
            cx, cy = self.bx + dx, self.by + dy
            sx, sy, sz = s
            pts = arc_route((sx, sy, sz), (cx, cy))
            self.min_radius = min(getattr(self, "min_radius", 9.0), min_radius(pts))
            top = (cx, cy, z_top)
            if fan_out is not None:
                pts.append((cx, cy, z_top))
                pts += fan_out(k, top, end)
            else:
                pts.append(top)
            mb_cable.tube(pts, self.r, 5, mi_cable)
            tops.append(top)
        # velcro straps: ring around the members already joined at each strap height
        if self.members:
            z0 = self.members[0][0] + 0.18
            z = z0
            while z < z_top - 0.05:
                n = sum(1 for t in self.members if t[0] + 0.13 < z)
                if n:
                    rad = max(math.hypot(dx, dy) for dx, dy in offs[:n]) + self.r + 0.0015
                    mb_strap.ring((self.bx, self.by, z), rad + 0.0012, rad, 0.012, "z", 16, mi_strap)
                z += strap_pitch
        return tops
