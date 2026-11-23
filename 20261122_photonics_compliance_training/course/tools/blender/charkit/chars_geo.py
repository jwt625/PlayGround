"""Numpy geometry kit for the clay characters (Blender 4.2, headless).

Everything is built as `Part` objects (vertices, faces, per-face material slot index, per-vertex bone weights)
in baseline character coordinates (H = 1.75 m reference, Z up, front = -Y, character left = +X), then
converted to Blender mesh objects with `to_object`.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

TAU = 2.0 * math.pi


def nrm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def interp(xs, ys, x):
    """Smooth (Catmull-Rom) interpolation of a table; xs ascending; ys scalar or vector columns."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    x = np.asarray(x, float)
    xc = np.clip(x, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xc) - 1, 0, len(xs) - 2)
    x0, x1 = xs[i], xs[i + 1]
    t = (xc - x0) / (x1 - x0)
    ip = np.clip(i - 1, 0, len(xs) - 1)
    inn = np.clip(i + 2, 0, len(xs) - 1)
    y0, y1 = ys[i], ys[i + 1]
    m0 = (ys[i + 1] - ys[ip]) / (xs[i + 1] - xs[ip]) * (x1 - x0)
    m1 = (ys[inn] - ys[i]) / (xs[inn] - xs[i]) * (x1 - x0)
    if ys.ndim > 1:
        t = t[..., None]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1


class Part:
    """Mesh under construction."""

    def __init__(self, v=None, f=None, mat=0, closed=True):
        self.v = np.zeros((0, 3)) if v is None else np.asarray(v, float).reshape(-1, 3)
        self.f = [] if f is None else [tuple(int(i) for i in q) for q in f]
        self.fm = [mat] * len(self.f)
        self.w = {}
        self.closed = closed

    # ------------------------------------------------------------ weights
    def set_w(self, bone, value=1.0):
        a = self.w.get(bone, np.zeros(len(self.v)))
        a = a + 0.0
        a[:] = value
        self.w[bone] = a
        return self

    def add_w(self, bone, arr):
        a = self.w.get(bone, np.zeros(len(self.v)))
        self.w[bone] = a + np.broadcast_to(np.asarray(arr, float), (len(self.v),))
        return self

    def only_w(self, wdict):
        self.w = {k: np.broadcast_to(np.asarray(a, float), (len(self.v),)).copy() for k, a in wdict.items()}
        return self

    # ------------------------------------------------------------ transforms
    def copy(self):
        p = Part(self.v.copy(), list(self.f), 0, self.closed)
        p.fm = list(self.fm)
        p.w = {k: a.copy() for k, a in self.w.items()}
        return p

    def tf(self, M):
        """Apply 4x4 (or 3x3) matrix."""
        M = np.asarray(M, float)
        if M.shape == (4, 4):
            self.v = self.v @ M[:3, :3].T + M[:3, 3]
        else:
            self.v = self.v @ M.T
        return self

    def move(self, d):
        self.v = self.v + np.asarray(d, float)
        return self

    def scale(self, s, about=(0, 0, 0)):
        about = np.asarray(about, float)
        self.v = (self.v - about) * np.asarray(s, float) + about
        return self

    def set_mat(self, m):
        self.fm = [m] * len(self.f)
        return self

    def mirror_x(self, rename=True):
        """Mirror across x=0 (faces flipped); bone weight names _L <-> _R."""
        p = self.copy()
        p.v[:, 0] *= -1.0
        p.f = [tuple(reversed(q)) for q in p.f]
        if rename:
            nw = {}
            for k, a in p.w.items():
                if k.endswith("_L"):
                    k = k[:-2] + "_R"
                elif k.endswith("_R"):
                    k = k[:-2] + "_L"
                nw[k] = a
            p.w = nw
        return p

    def weld(self, eps=1e-5):
        """Merge coincident vertices (bone weights taken from the first occurrence); degenerate faces are dropped."""
        key = np.round(self.v / eps).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        self.v = self.v[first]
        nf, nm = [], []
        for q, m in zip(self.f, self.fm):
            r = []
            for i in q:
                j = int(inv[i])
                if not r or r[-1] != j:
                    r.append(j)
            if len(r) > 1 and r[0] == r[-1]:
                r.pop()
            if len(r) >= 3 and len(set(r)) == len(r):
                nf.append(tuple(r))
                nm.append(m)
        self.f, self.fm = nf, nm
        self.w = {k: a[first] for k, a in self.w.items()}
        return self

    def drop_faces(self, pred):
        """Remove faces whose centroid satisfies pred(c) (c: (n,3) array -> bool array); unused vertices pruned."""
        if not self.f:
            return self
        cents = np.array([self.v[list(q)].mean(axis=0) for q in self.f])
        keep = ~pred(cents)
        self.f = [q for q, k in zip(self.f, keep) if k]
        self.fm = [m for m, k in zip(self.fm, keep) if k]
        used = sorted({i for q in self.f for i in q})
        remap = {o: n for n, o in enumerate(used)}
        self.v = self.v[used]
        self.f = [tuple(remap[i] for i in q) for q in self.f]
        self.w = {k: a[used] for k, a in self.w.items()}
        return self


def merge(parts):
    parts = [p for p in parts if p is not None and len(p.v)]
    out = Part()
    out.closed = all(p.closed for p in parts)
    off = 0
    vs, fs, fms = [], [], []
    names = set()
    for p in parts:
        names.update(p.w.keys())
    ws = {k: [] for k in names}
    for p in parts:
        vs.append(p.v)
        fs.extend([tuple(i + off for i in q) for q in p.f])
        fms.extend(p.fm)
        for k in names:
            ws[k].append(p.w.get(k, np.zeros(len(p.v))))
        off += len(p.v)
    out.v = np.concatenate(vs, axis=0)
    out.f = fs
    out.fm = fms
    out.w = {k: np.concatenate(a) for k, a in ws.items()}
    return out


# ------------------------------------------------------------------ primitives
def tube(path, rx, ry=None, u=(1, 0, 0), nseg=24, cap0="round", cap1="round", ncap=3, mat=0, phase=0.0,
         closed_path=False, capf=1.0):
    """Elliptical-section loft along a polyline. Radii rx (along u) and ry (along d x u), scalar or per-point.

    cap: 'round' (ellipsoidal end), 'flat' (fan) or None. Returns a Part (rings are the weights' natural unit).
    """
    path = np.asarray(path, float)
    n = len(path)
    rx = np.broadcast_to(np.asarray(rx, float), (n,)).copy()
    ry = rx.copy() if ry is None else np.broadcast_to(np.asarray(ry, float), (n,)).copy()
    d = np.zeros_like(path)
    if closed_path:
        d = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    else:
        d[1:-1] = path[2:] - path[:-2]
        d[0] = path[1] - path[0]
        d[-1] = path[-1] - path[-2]
    d = nrm(d)
    u = np.asarray(u, float)
    phi = phase + TAU * np.arange(nseg) / nseg
    cp, sp = np.cos(phi), np.sin(phi)

    def ring(P, di, a, b):
        ui = u - di * np.dot(u, di)
        if np.linalg.norm(ui) < 1e-6:
            ui = np.array([0.0, 1.0, 0.0]) - di * di[1]
        ui = ui / np.linalg.norm(ui)
        wi = np.cross(di, ui)
        return P + np.outer(cp, ui) * a + np.outer(sp, wi) * b

    rings = [ring(path[i], d[i], rx[i], ry[i]) for i in range(n)]
    pre, post = [], []
    pole0 = pole1 = None
    if not closed_path:
        if cap0 == "round":
            cl = capf * 0.5 * (rx[0] + ry[0])
            for k in range(ncap, 0, -1):
                a = (math.pi / 2) * k / (ncap + 1)
                pre.append(ring(path[0] - d[0] * cl * math.sin(a), d[0], rx[0] * math.cos(a), ry[0] * math.cos(a)))
            pole0 = path[0] - d[0] * cl
        elif cap0 == "flat":
            pole0 = path[0]
        if cap1 == "round":
            cl = capf * 0.5 * (rx[-1] + ry[-1])
            for k in range(1, ncap + 1):
                a = (math.pi / 2) * k / (ncap + 1)
                post.append(ring(path[-1] + d[-1] * cl * math.sin(a), d[-1], rx[-1] * math.cos(a), ry[-1] * math.cos(a)))
            pole1 = path[-1] + d[-1] * cl
        elif cap1 == "flat":
            pole1 = path[-1]
    allr = pre + rings + post
    verts = [r for r in allr]
    V = np.concatenate(verts, axis=0)
    faces = []
    m = len(allr)
    for i in range(m - 1):
        for j in range(nseg):
            j1 = (j + 1) % nseg
            faces.append((i * nseg + j, i * nseg + j1, (i + 1) * nseg + j1, (i + 1) * nseg + j))
    if closed_path:
        for j in range(nseg):
            j1 = (j + 1) % nseg
            i = m - 1
            faces.append((i * nseg + j, i * nseg + j1, j1, j))
    vl = list(V)
    if pole0 is not None:
        pi_ = len(vl)
        vl.append(pole0)
        for j in range(nseg):
            j1 = (j + 1) % nseg
            faces.append((pi_, j1, j))
    if pole1 is not None:
        pi_ = len(vl)
        vl.append(pole1)
        base = (m - 1) * nseg
        for j in range(nseg):
            j1 = (j + 1) % nseg
            faces.append((base + j, base + j1, pi_))
    return Part(np.array(vl), faces, mat, closed=not closed_path or True)


def axes_from(z, x=None):
    z = nrm(np.asarray(z, float))
    if x is None:
        x = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.asarray(x, float)
    x = nrm(x - z * np.dot(x, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # columns


def ellipsoid(center, radii, axes=None, nseg=16, nrings=10, mat=0, e=None):
    """Ellipsoid (or superellipsoid when e is given: e=1 sphere, smaller = boxier). axes: 3x3 matrix with column axes."""
    A = np.eye(3) if axes is None else np.asarray(axes, float)
    verts = []
    faces = []
    th = np.pi * np.arange(1, nrings) / nrings
    ph = TAU * np.arange(nseg) / nseg

    def sp(c, p):
        return np.sign(c) * np.abs(c) ** p

    ee = 1.0 if e is None else e
    verts.append((0.0, 0.0, -1.0))
    for t in th:
        for p in ph:
            ct, st = math.cos(t), math.sin(t)
            # superellipsoid: z = -sp(cos t), radial = sp(sin t) * (sp(cos p), sp(sin p))
            z = -sp(ct, ee)
            rr = abs(st) ** ee
            xx = rr * sp(math.cos(p), ee)
            yy = rr * sp(math.sin(p), ee)
            verts.append((xx, yy, z))
    verts.append((0.0, 0.0, 1.0))
    V = np.array(verts) * np.asarray(radii, float)
    V = V @ A.T + np.asarray(center, float)
    nr = len(th)
    for j in range(nseg):
        j1 = (j + 1) % nseg
        faces.append((0, 1 + j1, 1 + j))
    for i in range(nr - 1):
        for j in range(nseg):
            j1 = (j + 1) % nseg
            a, b = 1 + i * nseg + j, 1 + i * nseg + j1
            c, d = 1 + (i + 1) * nseg + j1, 1 + (i + 1) * nseg + j
            faces.append((a, b, c, d))
    last = 1 + (nr - 1) * nseg
    top = len(V) - 1
    for j in range(nseg):
        j1 = (j + 1) % nseg
        faces.append((last + j, last + j1, top))
    return Part(V, faces, mat)


def capsule(a, b, ra, rb=None, nseg=14, ncap=3, mat=0, u=(1, 0, 0)):
    rb = ra if rb is None else rb
    return tube(np.array([a, b], float), [ra, rb], [ra, rb], u=u, nseg=nseg, ncap=ncap, mat=mat)


def lathe(profile, center=(0, 0, 0), axis=(0, 0, 1), nseg=24, mat=0, xdir=None):
    """Surface of revolution. profile: list of (radius, height) from bottom to top; radius 0 at an end gives a pole."""
    A = axes_from(axis, xdir)
    ph = TAU * np.arange(nseg) / nseg
    verts = []
    ringidx = []
    poles = {}
    for k, (r, h) in enumerate(profile):
        if r <= 1e-9:
            poles[k] = len(verts)
            verts.append((0, 0, h))
            ringidx.append(None)
        else:
            ringidx.append(len(verts))
            for p in ph:
                verts.append((r * math.cos(p), r * math.sin(p), h))
    V = np.array(verts) @ A.T + np.asarray(center, float)
    faces = []
    for k in range(len(profile) - 1):
        a, b = ringidx[k], ringidx[k + 1]
        for j in range(nseg):
            j1 = (j + 1) % nseg
            if a is None and b is None:
                continue
            if a is None:
                faces.append((poles[k], b + j1, b + j))
            elif b is None:
                faces.append((a + j, a + j1, poles[k + 1]))
            else:
                faces.append((a + j, a + j1, b + j1, b + j))
    return Part(V, faces, mat)


def disc_cap(center, radius, normal, nseg=20, mat=0):
    """Flat or dished circular patch used for small details."""
    A = axes_from(normal)
    ph = TAU * np.arange(nseg) / nseg
    V = [np.zeros(3)] + [np.array([radius * math.cos(p), radius * math.sin(p), 0.0]) for p in ph]
    V = np.array(V) @ A.T + np.asarray(center, float)
    F = [(0, 1 + j, 1 + (j + 1) % nseg) for j in range(nseg)]
    return Part(V, F, mat, closed=False)


def to_matrix(loc=(0, 0, 0), rot=(0, 0, 0), s=1.0):
    M = np.eye(4)
    R = np.array(Matrix.Rotation(rot[2], 3, "Z") @ Matrix.Rotation(rot[1], 3, "Y") @ Matrix.Rotation(rot[0], 3, "X"))
    M[:3, :3] = R * s
    M[:3, 3] = loc
    return M


def rot_about(axis, angle):
    return np.array(Matrix.Rotation(angle, 3, Vector(axis)))


# ------------------------------------------------------------------ mesh objects
def part_to_mesh(part, name, mats, recalc=True):
    """Create a bpy mesh datablock from a Part (materials given as list of Material, indexed by Part.fm)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(x) for x in part.v], [], part.f)
    me.update()
    for m in mats:
        me.materials.append(m)
    if len(part.fm) == len(me.polygons):
        me.polygons.foreach_set("material_index", np.array(part.fm, dtype=np.int32))
    for p in me.polygons:
        p.use_smooth = True
    if recalc and part.closed:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()
    attr = me.attributes.new("rest_pos", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", part.v.astype(np.float32).ravel())
    return me


def to_object(part, name, mats, collection, parent=None, recalc=True):
    me = part_to_mesh(part, name, mats, recalc)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    for bone, w in part.w.items():
        vg = ob.vertex_groups.new(name=bone)
        idx = np.nonzero(w > 1e-5)[0]
        wr = np.round(np.minimum(w[idx], 1.0), 3)
        for val in np.unique(wr):
            vg.add(idx[wr == val].tolist(), float(val), "REPLACE")
    return ob
