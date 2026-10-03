"""S1 helpers (v1.1): baked camera director (eased moves, handheld noise, impact shake) and deterministic effect bakes
(electron pockets along a cable centreline, ballistic clay crumbs). Pure python + bpy; every random number is seeded.
"""
import math
import random

import asm
import bpy
from mathutils import Vector

L = asm.L
F = asm.F
TAU = 2.0 * math.pi
FPS = 30.0


# ------------------------------------------------------------------ easing
def smooth3(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def smooth5(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def ease_out(u):
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 2.6


def ease_in(u):
    u = min(1.0, max(0.0, u))
    return u ** 2.2


EASE = {"io": smooth3, "io5": smooth5, "out": ease_out, "in": ease_in, "lin": lambda u: min(1.0, max(0.0, u))}


def lerp3(a, b, e):
    return (a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e)


def dist3(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ------------------------------------------------------------------ deterministic noise
class Noise:
    FREQ = (0.45, 1.1, 2.3, 4.1)
    AMP = (1.0, 0.6, 0.35, 0.2)

    def __init__(self, seed, channels=6):
        rng = random.Random(seed)
        self.ph = [[rng.uniform(0, TAU) for _ in self.FREQ] for _ in range(channels)]
        self.norm = sum(self.AMP)

    def __call__(self, t, ch):
        return sum(a * math.sin(TAU * f * t + p) for a, f, p in zip(self.AMP, self.FREQ, self.ph[ch])) / self.norm


# ------------------------------------------------------------------ camera director
class Director:
    """Per-frame baked camera. A shot is fn(u, t) -> (pos, tgt, lens) with u in 0..1 over the shot."""

    def __init__(self, t_hit, seed=7):
        self.shots = []
        self.noise = Noise(seed)
        self.shake_ph = [random.Random(seed + 1).uniform(0, TAU) for _ in range(12)]
        self.t_hit = t_hit

    def add(self, t0, t1, fn, hh=1.0, ease="io"):
        self.shots.append((t0, t1, fn, hh, ease))

    def simple(self, t0, t1, p0, a0, p1=None, a1=None, lens=28.0, lens1=None, ease="io", hh=1.0):
        p1 = p0 if p1 is None else p1
        a1 = a0 if a1 is None else a1
        lens1 = lens if lens1 is None else lens1

        def fn(u, t, e=EASE[ease]):
            k = e(u)
            return lerp3(p0, p1, k), lerp3(a0, a1, k), lens + (lens1 - lens) * k
        self.add(t0, t1, fn, hh, ease)

    def _shot(self, t):
        for s in self.shots:
            if s[0] - 1e-6 <= t < s[1] - 1e-6:
                return s
        return self.shots[-1]

    def eval(self, t, noise=True):
        t0, t1, fn, hh, _ = self._shot(t)
        span = max(t1 - t0 - 1.0 / FPS, 1.0 / FPS)
        u = min(1.0, max(0.0, (t - t0) / span))
        pos, tgt, lens = fn(u, t)
        if not noise:
            return pos, tgt, lens
        ls = max(dist3(pos, tgt), 0.02)
        # handheld: small positional wander + smaller aim wander (amplitude grows with distance to the subject)
        ap = hh * (0.0007 + 0.0011 * ls)
        aa = hh * 0.0022 * ls
        n = self.noise
        pos = (pos[0] + ap * n(t, 0), pos[1] + ap * n(t, 1), pos[2] + ap * n(t, 2))
        tgt = (tgt[0] + aa * n(t, 3), tgt[1] + aa * n(t, 4), tgt[2] + aa * n(t, 5))
        # impact shake (decaying, 11-17 Hz) + lens punch
        dt = t - self.t_hit
        if dt >= 0.0:
            d = math.exp(-dt / 0.17) * (1.0 if dt < 0.9 else 0.0)
            ph = self.shake_ph
            sp = 0.006 * ls
            sa = 0.012 * ls
            pos = (pos[0] + sp * d * math.sin(TAU * 13.0 * dt + ph[0]), pos[1] + sp * d * math.sin(TAU * 11.0 * dt + ph[1]),
                   pos[2] + sp * d * math.sin(TAU * 17.0 * dt + ph[2]))
            tgt = (tgt[0] + sa * d * math.sin(TAU * 14.0 * dt + ph[3]), tgt[1] + sa * d * math.sin(TAU * 12.0 * dt + ph[4]),
                   tgt[2] + sa * d * math.sin(TAU * 16.0 * dt + ph[5]))
            lens = lens * (1.0 + 0.035 * math.exp(-dt / 0.09))
        return pos, tgt, lens

    def bake(self, frames=300):
        cam, tgt_o, hold = L.CAM, L.TGT, L.HOLD
        cam.data.clip_start = 0.002
        cam.data.clip_end = 400.0
        for f in range(1, frames + 1):
            t = (f - 1) / FPS
            p, a, lens = self.eval(t)
            cam.location = L.LP(0, p)
            cam.keyframe_insert("location", frame=f)
            tgt_o.location = L.LP(0, a)
            tgt_o.keyframe_insert("location", frame=f)
            cam.data.lens = lens
            cam.data.keyframe_insert("lens", frame=f)
            hold.scale = (L.HOLD_S0 * L.LENS0 / lens,) * 3
            hold.keyframe_insert("scale", frame=f)
        for ob in (cam, tgt_o, cam.data, hold):
            ad = ob.animation_data
            if ad and ad.action:
                for fc in ad.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"


# ------------------------------------------------------------------ polyline helper (cable centreline)
class Polyline:
    def __init__(self, pts):
        self.p = [Vector(q) for q in pts]
        self.c = [0.0]
        for a, b in zip(self.p[:-1], self.p[1:]):
            self.c.append(self.c[-1] + (b - a).length)
        self.length = self.c[-1]

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self.c) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if self.c[m] <= s:
                lo = m
            else:
                hi = m
        seg = max(self.c[hi] - self.c[lo], 1e-9)
        k = (s - self.c[lo]) / seg
        pos = self.p[lo].lerp(self.p[hi], k)
        tan = (self.p[hi] - self.p[lo]).normalized()
        return pos, tan


# ------------------------------------------------------------------ electron pockets
def add_electrons(director, line, mat_fwd, mat_bwd, t_end, n=18, seed=3, center=0.0):
    """Glowing blob pockets travelling along the polyline in both directions (baked per frame, closed form).

    Blob scale grows with the camera distance (stylised so the pockets stay visible in the wide shot)."""
    rng = random.Random(seed)
    mesh = bpy.data.meshes.new("electron_blob")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    f_end = F(t_end)
    out = []
    Ltot = line.length
    for i in range(n):
        fwd = (i % 2 == 0)
        mat = mat_fwd if fwd else mat_bwd
        speed = rng.uniform(0.85, 1.25) * (1.0 if fwd else -1.0)     # relative speed factor; absolute speed ramps up with time (see below)
        s0 = rng.uniform(0.0, Ltot) if i % 3 == 2 else (center + rng.uniform(-0.9, 0.9) - speed * 0.2) % Ltot   # most pockets start near the opening camera
        phi = rng.uniform(0.0, TAU)
        rho = 0.0055 * rng.uniform(0.45, 1.0)
        root = bpy.data.objects.new("electron_pocket_%02d" % i, None)
        bpy.context.scene.collection.objects.link(root)
        blobs = []
        for b in range(4):
            ob = bpy.data.objects.new("electron_%02d_%d" % (i, b), mesh)
            bpy.context.scene.collection.objects.link(ob)
            ob.data.materials.append(mat) if not ob.data.materials else None
            ob.parent = root
            r = rng.uniform(0.0028, 0.0046)
            ob.scale = (r, r, r)
            ob.location = (rng.uniform(-0.014, 0.014), rng.uniform(-0.0022, 0.0022), rng.uniform(-0.0022, 0.0022))
            blobs.append(ob)
        out.append((root, blobs))
        for f in range(1, f_end + 1):
            t = (f - 1) / FPS
            s = (s0 + speed * (0.45 * t + 2.0 * smooth3(t / 2.0) * t * 0.55)) % Ltot
            pos, tan = line.at(s)
            n1 = tan.cross(Vector((0, 0, 1)))
            n1 = n1.normalized() if n1.length > 1e-6 else Vector((0, 1, 0))
            n2 = tan.cross(n1)
            lane = pos + n1 * (rho * math.cos(phi + 0.6 * t)) + n2 * (rho * math.sin(phi + 0.6 * t))
            cam_p = director.eval(t, noise=False)[0]
            d = dist3(cam_p, tuple(lane))
            sc = min(max(d / 0.30, 1.5), 20.0)
            env = smooth3(s / 0.3) * smooth3((Ltot - s) / 0.3)
            fade = 1.0 - smooth3((t - (t_end - 0.45)) / 0.45)
            near = smooth3((d - 0.012) / 0.05)      # no blob may swallow the lens when a pocket flies past the camera
            k = sc * env * fade * near
            if k < 1e-3:
                k = 0.0
            root.location = lane
            root.scale = (k, k, k)
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("scale", frame=f)
        for ob in [root] + blobs:
            L.V(ob, 0, 0.0, t_end + 0.05)
        if root.animation_data:
            for fc in root.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    return out


# ------------------------------------------------------------------ clay crumbs (ballistic, floor bounce)
def add_crumbs(origin, away, t0, mat, n=16, seed=5, floor_z=0.0, dur=1.1, speed=(1.8, 4.2)):
    rng = random.Random(seed)
    mesh = bpy.data.meshes.new("clay_crumb")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = True
    away = Vector(away).normalized()
    side = away.cross(Vector((0, 0, 1))).normalized()
    f0 = F(t0)
    f1 = F(t0 + dur)
    objs = []
    for i in range(n):
        ob = bpy.data.objects.new("clay_crumb_%02d" % i, mesh)
        bpy.context.scene.collection.objects.link(ob)
        ob.data.materials.append(mat) if not ob.data.materials else None
        r = rng.uniform(0.02, 0.045)
        base_scale = (r, r * rng.uniform(0.6, 1.0), r * rng.uniform(0.5, 0.9))
        v = (away * rng.uniform(*speed) + side * rng.uniform(-1.6, 1.6) + Vector((0, 0, rng.uniform(1.2, 3.4)))
             + Vector((0, -1, 0)) * rng.uniform(0.0, 1.5))
        p = Vector(origin) + Vector((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(-0.1, 0.1)))
        spin = Vector((rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(-14, 14)))
        h = 1.0 / 240.0
        t = 0.0
        sim = {}
        steps_per_frame = 8
        for fr in range(f0, f1 + 1):
            sim[fr] = (p.copy(), t)
            for _ in range(steps_per_frame):
                v.z -= 9.81 * h
                p += v * h
                if p.z < floor_z + r * 0.6:
                    p.z = floor_z + r * 0.6
                    if v.z < 0:
                        v.z = -v.z * 0.32
                    v.x *= 0.78
                    v.y *= 0.78
                    if abs(v.z) < 0.35:
                        v.z = 0.0
                t += h
        for fr, (pp, tt) in sim.items():
            ob.location = pp
            ob.rotation_euler = (spin.x * tt, spin.y * tt, spin.z * tt)
            # shrink to nothing in the last 0.25 s so crumbs do not pile up forever
            fade = 1.0 - smooth3((fr - (f1 - 8)) / 8.0)
            ob.scale = tuple(c * fade for c in base_scale)
            ob.keyframe_insert("location", frame=fr)
            ob.keyframe_insert("rotation_euler", frame=fr)
            ob.keyframe_insert("scale", frame=fr)
        for fc in ob.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        L.VA(ob, f0, f1 + 1)
        objs.append(ob)
    return objs


# ------------------------------------------------------------------ damped spring helper
def spring(t, amp, freq, tau):
    """amp * exp(-t/tau) * cos(2 pi f t) for t >= 0 (0 before)."""
    if t < 0.0:
        return 0.0
    return amp * math.exp(-t / tau) * math.cos(TAU * freq * t)
