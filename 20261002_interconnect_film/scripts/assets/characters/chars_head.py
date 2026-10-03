"""Head, face parts and the expression system (shape keys on every face part, driven from the root's p_ properties).

All coordinates are baseline (crown height 1.75 m, character left = +X, front = -Y). The caller scales by K when it
creates Blender objects (see `make_face_objects`).
"""
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import chars_geo as G

# ---------------------------------------------------------------------------------------------- expressions
DEFAULT = dict(brow_in=0.0, brow_out=0.0, brow_x=0.0, lid_up=0.0, lid_lo=0.0, lid_tilt=0.0, gap=0.002, width=1.0,
               corner=0.0, smirk=0.0, snarl=0.0, pupil=1.0, cheek=0.0, flare=0.0, sweat=0.0, tears=0.0)

EXPR = {
    "neutral": {},
    "worried": dict(brow_in=0.011, brow_out=-0.004, lid_up=0.12, lid_tilt=-0.20, corner=-0.004, gap=0.005, width=0.95, pupil=1.1),
    "sweating": dict(brow_in=0.007, brow_out=-0.002, lid_up=0.10, lid_tilt=-0.12, corner=-0.003, gap=0.006, width=1.12, pupil=0.95, sweat=1.0),
    "dread": dict(brow_in=0.014, brow_out=0.007, lid_up=-0.28, lid_lo=-0.05, corner=-0.008, gap=0.032, width=0.9, pupil=0.5, sweat=0.6),
    "flat": dict(brow_in=-0.002, lid_up=0.30, gap=0.0015, width=0.97, pupil=0.9),
    "sobbing": dict(brow_in=0.013, brow_out=-0.006, lid_up=0.38, lid_lo=0.18, lid_tilt=-0.15, corner=-0.012, gap=0.020, width=1.15, cheek=0.004, tears=1.0),
    "shouting": dict(brow_in=-0.009, brow_out=0.004, lid_up=-0.05, lid_tilt=0.12, gap=0.052, width=1.1, corner=-0.002, snarl=0.4, flare=0.3),
    "smug": dict(brow_in=-0.002, brow_out=0.005, lid_up=0.32, lid_tilt=0.05, corner=0.012, smirk=0.010, width=1.12, gap=0.002, cheek=0.004),
    "shock": dict(brow_in=0.016, brow_out=0.016, lid_up=-0.34, lid_lo=-0.08, gap=0.038, width=0.85, pupil=0.45),
    "dead_eyed": dict(brow_out=-0.003, lid_up=0.46, lid_lo=0.10, corner=-0.002, gap=0.0015, width=0.96, pupil=0.6),
    "angry": dict(brow_in=-0.015, brow_out=0.002, lid_up=0.14, lid_tilt=0.32, corner=-0.010, gap=0.010, snarl=0.7, flare=1.0),
    "scared": dict(brow_in=0.012, brow_out=0.004, lid_up=-0.18, corner=-0.006, gap=0.022, width=0.92, pupil=0.7, sweat=0.4),
    "happy": dict(brow_in=0.004, brow_out=0.004, lid_up=0.08, lid_lo=0.16, corner=0.014, gap=0.018, width=1.2, cheek=0.006),
    "yell": dict(gap=0.066, width=1.1, snarl=0.5, brow_in=-0.004, flare=0.5),
}
KEYS = list(EXPR.keys())


def E(name):
    d = dict(DEFAULT)
    d.update(EXPR[name])
    return d


# ---------------------------------------------------------------------------------------------- head surface
class Head:
    """Head skin grid with a modelled mouth hole and a mouth bag; ears are separate islands (same mesh)."""

    NR = 40       # rings (poles excluded: rings 1..NR-1)
    NS = 56       # columns
    IM = 12       # mouth centre ring
    HOLE_ROWS = {10: 2, 11: 4, 12: 4, 13: 2}  # lower ring of the quad row -> half-width in quads

    def __init__(self, hs):
        self.hs = hs
        self.zc = hs["z_chin"]
        self.hh = hs["hh"]
        self.cy = hs.get("cy", 0.005)
        NR, NS = self.NR, self.NS
        self.j0 = 3 * NS // 4
        verts = [(0.0, self.cy, self.zc)]  # bottom pole (index 0)
        gi = [0]
        gj = [0]
        for i in range(1, NR):
            t = math.pi * i / NR
            s = (1 - math.cos(t)) / 2
            z = self.zc + self.hh * s
            sn = math.sin(t)
            fw = self.fw(s)
            for j in range(NS):
                ph = 2 * math.pi * j / NS
                c, sp = math.cos(ph), math.sin(ph)
                p = hs.get("sup", 2.3)
                xx = hs["w0"] * sn * fw * np.sign(c) * abs(c) ** (2.0 / p)
                b = hs["b_front"] * self.fb(s) if sp < 0 else hs["b_back"]
                yy = self.cy + b * sn * np.sign(sp) * abs(sp) ** (2.0 / p)
                # chin: push the front of the lower head forward a little, flatten the cranial back
                verts.append((xx, yy, z))
                gi.append(i)
                gj.append(j)
        verts.append((0.0, self.cy, self.zc + self.hh))  # top pole
        gi.append(NR)
        gj.append(0)
        self.V0 = np.array(verts)
        self.gi = np.array(gi)
        self.gj = np.array(gj)
        faces = []

        def vid(i, j):
            return 1 + (i - 1) * NS + (j % NS)
        for j in range(NS):
            faces.append((0, vid(1, j + 1), vid(1, j)))
        for i in range(1, NR - 1):
            for j in range(NS):
                faces.append((vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)))
        top = len(verts) - 1
        for j in range(NS):
            faces.append((vid(NR - 1, j), vid(NR - 1, j + 1), top))
        self.faces_full = faces
        self.vid = vid
        # BVH of the intact surface (for placing features)
        self.bvh = BVHTree.FromPolygons([tuple(v) for v in self.V0], [tuple(f) for f in faces])
        # remove the mouth quads
        j0 = self.j0
        hole = set()
        for ring, hw in self.HOLE_ROWS.items():
            for j in range(j0 - hw, j0 + hw):
                hole.add((ring, j % NS))
        keep = []
        nface_grid_start = NS
        for fi, f in enumerate(faces):
            if nface_grid_start <= fi < nface_grid_start + (NR - 2) * NS:
                k = fi - nface_grid_start
                ring = 1 + k // NS
                j = k % NS
                if (ring, j) in hole:
                    continue
            keep.append(f)
        self.faces = keep
        # boundary loop of the hole
        edges = {}
        for f in keep:
            for a, b in zip(f, f[1:] + f[:1]):
                key = (min(a, b), max(a, b))
                edges.setdefault(key, []).append((a, b))
        bnd = [(k, v[0]) for k, v in edges.items() if len(v) == 1]
        nxt = {}
        for k, (a, b) in bnd:
            nxt[a] = b
        start = bnd[0][1][0]
        loop = [start]
        while True:
            n = nxt[loop[-1]]
            if n == start:
                break
            loop.append(n)
        # orientation: the boundary edges of the kept faces run counter-clockwise around the hole interior as seen
        # from inside the faces; the bag faces must run the other way (handled by normal recalculation later)
        self.rim = loop
        self.n_grid = len(self.V0)
        # bag loops
        self.bag_scale = [0.92, 0.72, 0.46, 0.2]
        self.bag_depth = [0.012, 0.028, 0.042, 0.050]
        nl = len(loop)
        self.nl = nl
        self.bag_first = self.n_grid
        bag_faces = []
        for k in range(len(self.bag_scale)):
            for m in range(nl):
                m1 = (m + 1) % nl
                if k == 0:
                    a, b = loop[m], loop[m1]
                else:
                    a, b = self.n_grid + (k - 1) * nl + m, self.n_grid + (k - 1) * nl + m1
                c = self.n_grid + k * nl + m1
                d = self.n_grid + k * nl + m
                bag_faces.append((a, d, c, b))
        capv = self.n_grid + len(self.bag_scale) * nl
        for m in range(nl):
            m1 = (m + 1) % nl
            a = self.n_grid + (len(self.bag_scale) - 1) * nl + m
            b = self.n_grid + (len(self.bag_scale) - 1) * nl + m1
            bag_faces.append((a, capv, b))
        self.bag_faces = bag_faces
        self.n_bag = len(self.bag_scale) * nl + 1
        self.rim_idx = np.array(loop)
        self.rim_di = self.gi[self.rim_idx] - self.IM
        self.rim_dj = ((self.gj[self.rim_idx] - self.j0 + self.NS // 2) % self.NS) - self.NS // 2
        # gap at the base mesh (centre column, upper rim ring minus lower rim ring)
        self.gbase = (self.V0[self.vid(self.IM + 2, self.j0), 2] - self.V0[self.vid(self.IM - 2, self.j0), 2])
        # weights for the field (grid vertices)
        di = (self.gi - self.IM).astype(float)
        dj = ((self.gj - self.j0 + NS // 2) % NS - NS // 2).astype(float)
        self.di, self.dj = di, dj
        adj = np.abs(dj)
        self.wj = np.where(adj <= 4, 1.0, np.exp(-(((adj - 4) / 2.2) ** 2)))
        self.wj_w = np.where(adj <= 4, 1.0, np.exp(-(((adj - 4) / 3.5) ** 2)))
        front = self.V0[:, 1] < (self.cy - 0.01)
        self.front = front
        self.wj = self.wj * front
        self.wj_w = self.wj_w * front
        # vertical fraction f(di): -0.65 at di=-2, +0.35 at di=+2
        f = np.where(di >= -2, np.where(di <= 2, -0.15 + 0.25 * di, 0.35 * np.exp(-(((di - 2) / 1.8) ** 2))),
                     -0.65 * np.exp(-(((-di - 2) / 3.2) ** 2)))
        self.fz = f
        self.wi_both = np.where(np.abs(di) <= 2, 1.0, np.exp(-(((np.abs(di) - 2) / 2.5) ** 2)))
        self.vi_corner = np.exp(-((di / 2.3) ** 2))
        self.h_corner = np.where(adj <= 4, G.smoothstep(0, 4, adj), np.exp(-(((adj - 4) / 2.5) ** 2)))
        self.wcheek = np.exp(-(((adj - 6.5) / 2.5) ** 2)) * np.exp(-(((di - 3.0) / 2.5) ** 2))
        self.up_weight = np.where(di >= 1, np.where(di <= 3, 1.0, np.exp(-(((di - 3) / 1.8) ** 2))), 0.0)
        self.jaw_w_grid = None
        self._ears = None

    def fw(self, s):
        hs = self.hs
        return 1.0 - hs.get("jaw", 0.26) * (1.0 - G.smoothstep(0.0, 0.62, s)) ** 1.2

    def fb(self, s):
        return 1.0 + self.hs.get("chin", 0.06) * (1.0 - G.smoothstep(0.0, 0.35, s))

    # ---------------------------------------------------------------------- queries
    def surface_y(self, x, z, which="front"):
        """y of the intact head surface at (x, z), ray cast from the front (or back)."""
        if which == "front":
            hit = self.bvh.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)))
        else:
            hit = self.bvh.ray_cast(Vector((x, 1.0, z)), Vector((0, -1, 0)))
        if hit[0] is None:
            return None
        return hit[0].y

    def surface_normal(self, x, z):
        hit = self.bvh.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)))
        return np.array(hit[1]) if hit[0] is not None else np.array([0, -1.0, 0])

    # ---------------------------------------------------------------------- deformation
    def deformed(self, e):
        """Return (grid+bag vertices) for expression dict e (absolute positions)."""
        V = self.V0.copy()
        D = e["gap"] - self.gbase
        # vertical open/close field
        V[:, 2] += D * self.fz * self.wj
        # width
        V[:, 0] += (e["width"] - 1.0) * V[:, 0] * self.wj_w * self.wi_both
        # corner lift (+ cheek raise); smirk lifts only the +X corner
        V[:, 2] += e["corner"] * self.h_corner * self.vi_corner * self.front
        sm = G.smoothstep(0.0, 2.0, self.dj) * self.h_corner * self.vi_corner * self.front
        V[:, 2] += e["smirk"] * sm
        V[:, 2] += e["cheek"] * self.wcheek * self.front
        # snarl: upper lip lifts
        V[:, 2] += e["snarl"] * 0.006 * self.up_weight * self.wj
        rim = V[self.rim_idx]
        c = rim.mean(axis=0)
        bag = []
        for sc, dp in zip(self.bag_scale, self.bag_depth):
            q = c + (rim - c) * sc
            q[:, 1] += dp
            bag.append(q)
        capv = c + np.array([0, 0.058, 0.0])
        bagv = np.concatenate(bag + [capv[None, :]], axis=0)
        return np.concatenate([V, bagv], axis=0)

    def jaw_weights(self):
        """Per-vertex jaw bone weight for grid + bag vertices (baseline neutral positions)."""
        V = self.V0
        di = self.di
        wz = G.smoothstep(0.0, -2.5, di)
        wy = G.smoothstep(0.035 + self.cy, -0.01 + self.cy, V[:, 1])
        w = wz * wy
        w[0] = 1.0
        bag = []
        rw = w[self.rim_idx]
        for k in range(len(self.bag_scale)):
            bag.append(rw)
        bag.append(np.array([rw.mean()]))
        return np.concatenate([w] + bag)

    def part(self, hs=None):
        """The neutral head Part (skin material 0, mouth interior material 1), ears excluded."""
        Vn = self.deformed(E("neutral"))
        faces = list(self.faces) + list(self.bag_faces)
        fm = [0] * len(self.faces) + [1] * len(self.bag_faces)
        p = G.Part(Vn, faces, 0, closed=True)
        p.fm = fm
        return p


# ---------------------------------------------------------------------------------------------- face parts
class FacePart:
    """A Part plus a function e -> vertex array (same count)."""

    def __init__(self, name, part, fn, mats):
        self.name, self.part, self.fn, self.mats = name, part, fn, mats


def _brow_part(head, hs, side):
    """Brow ridge on the forehead for one side (+1 = character left)."""
    ze = hs["z_chin"] + hs["eye_z"]
    zb = ze + hs["brow_dz"]
    xs = np.array([0.012, 0.024, 0.036, 0.048, 0.060]) * hs.get("brow_w", 1.0)
    pts = []
    for x in xs:
        y = head.surface_y(x, zb + 0.004 * math.sin(math.pi * (x - 0.012) / 0.05))
        y = y if y is not None else -0.08
        pts.append((side * x, y - 0.002, zb + 0.004 * math.sin(math.pi * (x - 0.012) / 0.05)))
    pts = np.array(pts)
    taper = np.array([0.8, 1.0, 1.0, 0.9, 0.6])
    p = G.tube(pts, hs.get("brow_t", 0.0045) * taper, hs.get("brow_h", 0.0062) * taper, u=(0, -1, 0), nseg=10, ncap=2)
    # normalised position along the brow (0 inner .. 1 outer)
    t = np.clip((np.abs(p.v[:, 0]) - 0.012) / 0.048, 0, 1)
    return p, t


def make_brows(head, hs):
    pl, tl = _brow_part(head, hs, +1)
    pr, tr = _brow_part(head, hs, -1)
    p = G.merge([pl, pr])
    t = np.concatenate([tl, tr])
    base = p.v.copy()

    def fn(e):
        V = base.copy()
        V[:, 2] += e["brow_in"] * (1 - t) + e["brow_out"] * t
        sgn = np.sign(base[:, 0])
        V[:, 0] -= sgn * e["brow_in"] * 0.35 * (1 - t)  # inner end pulled toward the centre as it rises
        V[:, 1] -= 0.0
        return V
    return FacePart("brows", p, fn, ["brow"])


def eye_centres(head, hs):
    ze = hs["z_chin"] + hs["eye_z"]
    out = []
    for side in (+1, -1):
        x = hs["eye_sep"] * side
        y = head.surface_y(x, ze)
        out.append(np.array([x, y + hs["eye_sink"], ze]))
    return out  # [left(+x), right(-x)]


def make_eyes(head, hs):
    r = hs["eye_r"]
    parts_e, parts_p, parts_l = [], [], []
    ctr = eye_centres(head, hs)
    names = ["L", "R"]
    pup_info = []
    lid_info = []
    for side_i, c in enumerate(ctr):
        nm = names[side_i]
        eb = G.ellipsoid(c, (r, r, r), axes=G.axes_from((0, -1, 0), (1, 0, 0)), nseg=24, nrings=14, mat=0)
        eb.set_w("eye_" + nm, 1.0)
        parts_e.append(eb)
        # pupil: spherical cap patch on the front of the eyeball
        pr = hs["pupil_r"]
        ang = math.asin(min(0.99, pr / r))
        rr = r * 1.004
        nang, nph = 5, 20
        verts = [c + np.array([0, -rr, 0])]
        faces = []
        for a in range(1, nang + 1):
            th = ang * a / nang
            for k in range(nph):
                ph = 2 * math.pi * k / nph
                verts.append(c + np.array([rr * math.sin(th) * math.cos(ph), -rr * math.cos(th), rr * math.sin(th) * math.sin(ph)]))
        for k in range(nph):
            faces.append((0, 1 + (k + 1) % nph, 1 + k))
        for a in range(nang - 1):
            for k in range(nph):
                k1 = (k + 1) % nph
                faces.append((1 + a * nph + k, 1 + a * nph + k1, 1 + (a + 1) * nph + k1, 1 + (a + 1) * nph + k))
        pp = G.Part(np.array(verts), faces, 0, closed=False)
        # winding: normals should point outward (-Y at the pole); flip if needed
        pp.f = [tuple(reversed(f)) for f in pp.f]
        pp.set_w("eye_" + nm, 1.0)
        parts_p.append((pp, c.copy(), rr))
        # lids: upper and lower caps of a slightly larger sphere, each with an edge bead
        for which in ("up", "lo"):
            lid, edge_lat = lid_cap(c, r * 1.085, which, hs)
            lid.set_w("lid_%s_%s" % (which, nm), 1.0)
            parts_l.append((lid, c.copy(), which, side_i))
    eyes = G.merge(parts_e)
    pup = G.merge([p for p, _, _ in parts_p])
    pup_base = pup.v.copy()
    # per-vertex pupil centre (for scaling)
    cen = np.concatenate([np.tile(c, (len(p.v), 1)) for p, c, _ in parts_p])
    rrs = parts_p[0][2]

    def pup_fn(e):
        s = e["pupil"]
        V = pup_base.copy()
        d = V - cen
        d[:, 0] *= s
        d[:, 2] *= s
        # re-project onto the eyeball sphere
        xz2 = d[:, 0] ** 2 + d[:, 2] ** 2
        d[:, 1] = -np.sqrt(np.maximum(rrs ** 2 - xz2, 1e-9))
        return cen + d
    lids = G.merge([p for p, _, _, _ in parts_l])
    lid_base = lids.v.copy()
    lcen = np.concatenate([np.tile(c, (len(p.v), 1)) for p, c, _, _ in parts_l])
    kind = np.concatenate([np.full(len(p.v), 1.0 if w == "up" else -1.0) for p, _, w, _ in parts_l])
    sidev = np.concatenate([np.full(len(p.v), 1.0 if si == 0 else -1.0) for p, _, _, si in parts_l])

    def lid_fn(e):
        V = lid_base.copy()
        d = V - lcen
        ang = np.where(kind > 0, e["lid_up"], -e["lid_lo"]).astype(float)
        # tilt: roll about Y (view axis); inner side down for angry (positive), mirrored per side
        tilt = np.where(kind > 0, e["lid_tilt"], 0.0) * sidev
        out = d.copy()
        ca, sa = np.cos(ang), np.sin(ang)
        # rotate about X by ang (positive closes the upper lid: front point moves down)
        y = d[:, 1] * ca - d[:, 2] * sa
        z = d[:, 1] * sa + d[:, 2] * ca
        out[:, 1], out[:, 2] = y, z
        ct, st = np.cos(tilt), np.sin(tilt)
        x2 = out[:, 0] * ct - out[:, 2] * st
        z2 = out[:, 0] * st + out[:, 2] * ct
        out[:, 0], out[:, 2] = x2, z2
        return lcen + out
    return (FacePart("eyes", eyes, None, ["eye_white"]),
            FacePart("pupils", pup, pup_fn, ["pupil"]),
            FacePart("lids", lids, lid_fn, ["skin"]))


def lid_cap(c, r, which, hs):
    """Spherical cap (open, with an edge bead) covering the upper (or lower) part of the eyeball."""
    lat = hs["lid_up_lat"] if which == "up" else -hs["lid_lo_lat"]  # edge latitude from the equator (rad)
    nph, nth = 24, 8
    # polar angle from +Z axis; cap from pole to edge
    th_edge = math.pi / 2 - lat if which == "up" else math.pi / 2 - lat  # lat negative for lower
    verts = []
    faces = []
    if which == "up":
        th_list = [th_edge * k / nth for k in range(nth + 1)]
    else:
        th_list = [math.pi - (math.pi - th_edge) * k / nth for k in range(nth + 1)]
    pole = np.array([0, 0, 1.0 if which == "up" else -1.0])
    verts.append(c + pole * r)
    for th in th_list[1:]:
        for k in range(nph):
            ph = 2 * math.pi * k / nph
            verts.append(c + r * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)]))
    for k in range(nph):
        faces.append((0, 1 + k, 1 + (k + 1) % nph))
    for a in range(nth - 1):
        for k in range(nph):
            k1 = (k + 1) % nph
            faces.append((1 + a * nph + k, 1 + (a + 1) * nph + k, 1 + (a + 1) * nph + k1, 1 + a * nph + k1))
    cap = G.Part(np.array(verts), faces, 0, closed=False)
    # bead along the edge circle
    th = th_edge
    path = np.array([c + r * np.array([math.sin(th) * math.cos(2 * math.pi * k / nph), math.sin(th) * math.sin(2 * math.pi * k / nph), math.cos(th)]) for k in range(nph)])
    bead = G.tube(path, 0.0016, 0.0016, u=(0, 0, 1), nseg=6, closed_path=True)
    cap = G.merge([cap, bead])
    cap.closed = False
    return cap, lat


def make_nose(head, hs):
    zc = hs["z_chin"]
    ytip_surface = head.surface_y(0.0, zc + hs["nose_z"])
    tip = np.array([0.0, ytip_surface - hs["nose_len"], zc + hs["nose_z"]])
    ns = hs.get("nose_s", 1.0)
    bridge_top = np.array([0.0, head.surface_y(0, zc + hs["eye_z"]) + 0.002, zc + hs["eye_z"] + 0.0])
    parts = []
    # bridge + tip as a tapered capsule
    path = np.array([bridge_top, (bridge_top + tip) / 2 + np.array([0, -0.002, 0.004]), tip])
    parts.append(G.tube(path, [0.0075 * ns, 0.0095 * ns, 0.0125 * ns], [0.0070 * ns, 0.0090 * ns, 0.0115 * ns], u=(1, 0, 0), nseg=14, ncap=3))
    wings = []
    for side in (+1, -1):
        c = tip + np.array([side * 0.0125 * ns, 0.0075, -0.0045])
        w = G.ellipsoid(c, (0.0078 * ns, 0.0078 * ns, 0.0075 * ns), nseg=12, nrings=8)
        wings.append((side, c))
        parts.append(w)
    nostril = []
    for side in (+1, -1):
        c = tip + np.array([side * 0.0068 * ns, 0.0035, -0.0112 * ns])
        n = G.ellipsoid(c, (0.0035 * ns, 0.0045 * ns, 0.0028 * ns), nseg=10, nrings=6, mat=1)
        nostril.append((side, c))
        parts.append(n)
    p = G.merge(parts)
    base = p.v.copy()
    # per-vertex flare class: wings and nostrils move outward in x, wings also scale
    nv_tube = len(parts[0].v)
    cls = np.zeros(len(base))
    off = nv_tube
    cen = np.tile(tip, (len(base), 1))
    for k, w in enumerate(wings):
        n = len(parts[1 + k].v)
        cls[off:off + n] = 1.0
        cen[off:off + n] = w[1]
        off += n
    for k, w in enumerate(nostril):
        n = len(parts[3 + k].v)
        cls[off:off + n] = 2.0
        cen[off:off + n] = w[1]
        off += n

    def fn(e):
        V = base.copy()
        f = e["flare"]
        if f:
            sgn = np.sign(V[:, 0] - tip[0] + 1e-9)
            m1 = cls == 1.0
            m2 = cls == 2.0
            V[m1, 0] += sgn[m1] * 0.0022 * f
            V[m1] = cen[m1] + (V[m1] - cen[m1]) * (1.0 + 0.14 * f)
            V[m2, 0] += sgn[m2] * 0.0030 * f
            V[m2] = cen[m2] + (V[m2] - cen[m2]) * np.array([1.0 + 0.30 * f, 1.0, 1.0 + 0.1 * f])
        return V
    return FacePart("nose", p, fn, ["skin", "nostril"]), tip


def make_ears(head, hs):
    es = hs.get("ear_s", 1.0)
    parts = []
    for side in (+1, -1):
        zc = hs["z_chin"] + hs["ear_z"]
        # x at the surface of the head at that height (ray along x from the side)
        hit = head.bvh.ray_cast(Vector((side * 1.0, head.cy + 0.012, zc)), Vector((-side, 0, 0)))
        x0 = hit[0].x if hit[0] is not None else side * 0.075
        c = np.array([x0 + side * 0.004 * es, head.cy + 0.012, zc])
        A = G.axes_from((side * 0.35, -0.2, 0.0), (0, 0, 1))  # z axis: thin direction (outward & slightly forward)
        # disc: thin direction = outward
        ax = np.stack([np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), nrm_(np.array([side * 1.0, -0.25, 0.0]))], axis=1)
        disc = G.ellipsoid(c + np.array([side * 0.006 * es, 0, 0]), (0.0155 * es, 0.0330 * es, 0.0085 * es), axes=ax, nseg=16, nrings=10)
        # helix rim: closed ring tube around the ear outline
        th = np.linspace(0, 2 * math.pi, 22, endpoint=False)
        path = np.array([c + np.array([side * 0.011 * es, 0, 0]) + ax[:, 0] * 0.0160 * es * math.cos(t) * 1.0 + ax[:, 1] * 0.0335 * es * math.sin(t) for t in th])
        rim = G.tube(path, 0.0050 * es, 0.0050 * es, u=ax[:, 2], nseg=8, closed_path=True)
        lobe = G.ellipsoid(c + np.array([side * 0.008 * es, 0.001, -0.032 * es]), (0.0085 * es, 0.0095 * es, 0.0095 * es), nseg=10, nrings=6)
        for q in (disc, rim, lobe):
            q.closed = True
            parts.append(q)
    return G.merge(parts)


def nrm_(v):
    return v / np.linalg.norm(v)


def make_teeth_tongue(head, hs):
    """Upper teeth, lower teeth, tongue as one part each; they follow the mouth rim through deform functions."""
    tw = 0.016
    xs = np.linspace(-tw, tw, 9)
    ymouth = head.surface_y(0.0, hs["z_chin"] + 0.05)
    zc = hs["z_chin"] + 0.05

    def teeth_path(z):
        return np.array([(x, ymouth + 0.004 + 0.9 * (x / 0.03) ** 2 * 0.006 * -1 * -1, z) for x in xs])
    up = G.tube(teeth_path(zc) + np.array([0, 0.004, 0]), 0.0022, 0.0040, u=(0, -1, 0), nseg=8, cap0="round", cap1="round", ncap=2, mat=0)
    lo = G.tube(teeth_path(zc) + np.array([0, 0.004, 0]), 0.0022, 0.0036, u=(0, -1, 0), nseg=8, cap0="round", cap1="round", ncap=2, mat=0)
    tongue = G.ellipsoid(np.array([0, ymouth + 0.03, zc - 0.006]), (0.014, 0.022, 0.0065), nseg=12, nrings=8, mat=0)
    up.set_w("head", 1.0)
    lo.set_w("jaw", 1.0)
    tongue.set_w("jaw", 1.0)
    up0, lo0, tg0 = up.v.copy(), lo.v.copy(), tongue.v.copy()

    def rim_z(e):
        Vd = head.deformed(e)
        zt = Vd[head.vid(head.IM + 2, head.j0), 2]
        zb = Vd[head.vid(head.IM - 2, head.j0), 2]
        return zt, zb, Vd

    def up_fn(e):
        zt, zb, _ = rim_z(e)
        V = up0.copy()
        V[:, 2] += (zt - 0.0050) - zc
        V[:, 0] *= e["width"]
        return V

    def lo_fn(e):
        zt, zb, _ = rim_z(e)
        V = lo0.copy()
        V[:, 2] += (zb + 0.0042) - zc
        V[:, 0] *= e["width"]
        return V

    def tg_fn(e):
        zt, zb, _ = rim_z(e)
        V = tg0.copy()
        V[:, 2] += (zb + 0.004) - (zc - 0.006)
        return V
    return (FacePart("teeth_up", up, up_fn, ["teeth"]), FacePart("teeth_lo", lo, lo_fn, ["teeth"]),
            FacePart("tongue", tongue, tg_fn, ["tongue"]))


def drop_shape(c, length, radius, nseg=10):
    """Teardrop (point up) as a lathe."""
    prof = []
    n = 9
    for k in range(n + 1):
        t = k / n
        # t=0 bottom (round), t=1 top (point)
        h = length * t
        r = radius * math.sqrt(max(0.0, 1 - (2 * min(t, 0.5) - 1) ** 2)) if t <= 0.5 else radius * (1 - (t - 0.5) / 0.5) ** 1.3
        prof.append((max(r, 0.0), h))
    prof[0] = (0.0, 0.0)
    prof[-1] = (0.0, length)
    return G.lathe(prof, center=c, axis=(0, 0, 1), nseg=nseg)


def make_drops(head, hs):
    """Sweat drops on the forehead and tears on the cheeks (anchored; scale 0 in the neutral key)."""
    zc = hs["z_chin"]
    items = []  # (part, anchor, kind)
    sweat_pos = [(0.052, zc + 0.196, 0.0), (-0.044, zc + 0.205, 0.0), (0.020, zc + 0.215, 0.0),
                 (0.076, zc + 0.165, 0.0), (-0.074, zc + 0.172, 0.0)]
    for x, z, _ in sweat_pos:
        y = head.surface_y(x, z)
        y = (y if y is not None else -0.07) - 0.0035
        p = drop_shape(np.array([x, y, z - 0.008]), 0.020, 0.0062)
        items.append((p, np.array([x, y, z + 0.004]), "sweat"))
    ze = zc + hs["eye_z"]
    for side in (+1, -1):
        for k, (dz, dxm) in enumerate([(-0.030, 0.006), (-0.062, 0.010)]):
            x = side * (hs["eye_sep"] + dxm)
            z = ze + dz
            y = head.surface_y(x, z)
            y = (y if y is not None else -0.07) - 0.003
            p = drop_shape(np.array([x, y, z - 0.010]), 0.022 + 0.004 * k, 0.0058)
            items.append((p, np.array([x, y, z + 0.006]), "tears"))
    P = G.merge([i[0] for i in items])
    base = P.v.copy()
    anchor = np.concatenate([np.tile(i[1], (len(i[0].v), 1)) for i in items])
    kind = np.concatenate([np.full(len(i[0].v), 0 if i[2] == "sweat" else 1) for i in items])

    def fn(e):
        s = np.where(kind == 0, e["sweat"], e["tears"])
        s = np.maximum(s, 0.001)[:, None]
        return anchor + (base - anchor) * s
    P.set_w("head", 1.0)
    P.v = fn(E("neutral"))
    return FacePart("drops", P, fn, ["drop"])


# ---------------------------------------------------------------------------------------------- object creation
def key_driver(key_block_owner, key_name, root, expr_name):
    """Drive shape key `key_name` on a Key datablock from root custom properties."""
    path = 'key_blocks["%s"].value' % key_name
    fc = key_block_owner.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    props = ["p_expr_" + key_name]
    expr = "a"
    if key_name == "angry":
        props.append("p_anger")
        expr = "max(a,b)"
    elif key_name == "yell":
        props.append("p_anger")
        expr = "max(a,min(1.0,max(0.0,(b-0.35)/0.65)))"
    for k, pn in enumerate(props):
        var = d.variables.new()
        var.name = "ab"[k]
        var.targets[0].id_type = "OBJECT"
        var.targets[0].id = root
        var.targets[0].data_path = '["%s"]' % pn
    d.expression = expr


def add_expression_keys(ob, fn, root, K, base_verts):
    """Add shape keys (neutral basis + all expressions) to a mesh object, scaled by K, with drivers."""
    ob.shape_key_add(name="neutral", from_mix=False)
    keys = ob.data.shape_keys
    for name in KEYS[1:]:
        V = fn(E(name)) * K
        kb = ob.shape_key_add(name=name, from_mix=False)
        kb.data.foreach_set("co", V.astype(np.float32).ravel())
        kb.slider_min = 0.0
        kb.slider_max = 1.0
        key_driver(keys, name, root, name)
    return keys
