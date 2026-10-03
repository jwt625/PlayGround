"""S3 effects (v1.1): impact dust puffs, clay crumbs, flying papers and hats. All deterministic (fixed-seed numpy), baked to keys.

Objects share one mesh each; visibility windows are keyed through blender_lib.VA (finalized by asm.finalize).
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

import blender_lib as L
from asm import F

G = 9.81
SHIRT = {"terahop": (0.42, 0.25, 0.69), "molexx": (0.82, 0.13, 0.16), "nubiss": (0.18, 0.62, 0.31), "ayarr": (0.95, 0.76, 0.19)}
SKIN = {"terahop": (0.91, 0.72, 0.58), "molexx": (0.85, 0.63, 0.48), "nubiss": (0.79, 0.56, 0.40), "ayarr": (0.54, 0.35, 0.24)}
TABLE_TOP = 0.78


def ground(x, y):
    """Height of the surface under (x, y): conference table top or floor."""
    return TABLE_TOP if (abs(x) < 2.75 and abs(y) < 1.05) else 0.01


def _coll(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def _mat(name, color, rough=0.8, emit=0.0, two_sided=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    m.use_backface_culling = False
    return m


def _ico(name, sub=1, scale=(1, 1, 1)):
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def _key(o, f, loc=None, rot=None, scale=None):
    if loc is not None:
        o.location = loc
        o.keyframe_insert("location", frame=f)
    if rot is not None:
        o.rotation_euler = rot
        o.keyframe_insert("rotation_euler", frame=f)
    if scale is not None:
        o.scale = scale
        o.keyframe_insert("scale", frame=f)


def _linear(o):
    ad = o.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


class Fx:
    def __init__(self):
        self.coll = _coll("FX_s03")
        self.puff_mesh = _ico("fx_puff", 2)
        self.crumb_mesh = _ico("fx_crumb", 1, (1.0, 0.8, 0.6))
        self.dust_mat = _mat("MAT_s03_dust", (0.80, 0.77, 0.70), 1.0)
        self.crumb_mats = {k: _mat("MAT_s03_crumb_" + k, SHIRT[k], 0.7) for k in SHIRT}
        self.skin_mats = {k: _mat("MAT_s03_crumbskin_" + k, SKIN[k], 0.7) for k in SKIN}
        self.rng = np.random.default_rng(3003)
        self.n = 0

    def _obj(self, name, mesh, mat):
        o = bpy.data.objects.new("%s_%03d" % (name, self.n), mesh)
        self.n += 1
        self.coll.objects.link(o)
        if len(mesh.materials) == 0:
            mesh.materials.append(None)
        o.material_slots[0].link = "OBJECT"
        o.material_slots[0].material = mat
        return o

    # ---------------------------------------------------------------- puffs and crumbs
    def puff(self, t, pt, d, s, n=2):
        f0 = F(t)
        for i in range(n):
            o = self._obj("puff", self.puff_mesh, self.dust_mat)
            r = 0.045 + 0.03 * s + 0.015 * self.rng.random()
            vel = np.array([d * (0.35 + 0.5 * self.rng.random()), -0.25 * self.rng.random() - 0.1, 0.35 + 0.35 * self.rng.random()])
            life = 0.45 + 0.15 * self.rng.random()
            nf = int(life * 30)
            for k in range(nf + 1):
                u = k / nf
                tt = k / 30.0
                p = np.array(pt) + vel * tt * (1 - 0.4 * u) + np.array([0.04 * (i - 0.5), 0, 0])
                sc = r * (0.35 + 1.4 * math.sqrt(u)) * (1.0 - u ** 3)
                _key(o, f0 + k, loc=tuple(p), scale=(sc, sc, sc * 0.85))
            _linear(o)
            L.VA(o, f0 - 1, f0 + nf + 1)

    def crumbs(self, t, pt, d, s, victim, n=5):
        f0 = F(t)
        for i in range(n):
            mat = self.skin_mats[victim] if i % 3 == 0 else self.crumb_mats[victim]
            o = self._obj("crumb", self.crumb_mesh, mat)
            sz = 0.018 + 0.022 * self.rng.random()
            v0 = np.array([d * (0.8 + 2.2 * self.rng.random() * (0.5 + s)), (self.rng.random() - 0.6) * 1.2, 1.2 + 2.0 * self.rng.random()])
            p = np.array(pt, float)
            vel = v0.copy()
            rot = self.rng.random(3) * 6.28
            spin = (self.rng.random(3) - 0.5) * 20
            k = 0
            life = 0.9
            while k <= int(life * 30):
                _key(o, f0 + k, loc=tuple(p), rot=tuple(rot + spin * k / 30.0), scale=(sz, sz, sz) if k < int(life * 30) - 4 else (sz * 0.3, sz * 0.3, sz * 0.3))
                vel[2] -= G / 30.0
                p = p + vel / 30.0
                gz = ground(p[0], p[1])
                if p[2] < gz + sz:
                    p[2] = gz + sz
                    vel = vel * np.array([0.5, 0.5, -0.35])
                k += 1
            _linear(o)
            L.VA(o, f0 - 1, f0 + k)

    # ---------------------------------------------------------------- papers
    def papers(self, t, pt, n, d=1.0, spread=1.0, t_end=10.0):
        """n paper sheets thrown from pt at time t; flutter down (terminal speed about 1.3 m/s), land flat, stay to t_end."""
        me = bpy.data.meshes.get("fx_paper")
        if me is None:
            me = bpy.data.meshes.new("fx_paper")
            hx, hy = 0.105, 0.1485
            me.from_pydata([(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)], [], [(0, 1, 2, 3)])
            me.update()
            self.paper_mat = _mat("MAT_s03_paper", (0.93, 0.93, 0.90), 0.9)
        f0 = F(t)
        total = int(round((t_end - t) * 30))
        for i in range(n):
            o = self._obj("paper", me, self.paper_mat)
            p = np.array(pt, float) + (self.rng.random(3) - 0.5) * 0.2
            vel = np.array([d * (0.5 + 1.8 * self.rng.random()) * spread, (self.rng.random() - 0.5) * 1.5, 1.8 + 2.0 * self.rng.random()])
            ph = self.rng.random(3) * 6.28
            fr = 1.2 + self.rng.random()
            rz = self.rng.random() * 6.28
            spin = (self.rng.random() - 0.5) * 8
            landed = False
            land_k = None
            c = 5.0  # quadratic drag so terminal speed = sqrt(g / c) = 1.4 m/s
            for k in range(total + 1):
                tt = k / 30.0
                if not landed:
                    sp = np.linalg.norm(vel)
                    acc = np.array([0, 0, -G]) - c * vel * sp
                    acc[0] += 2.2 * math.sin(2 * math.pi * fr * tt + ph[0])
                    acc[1] += 2.2 * math.cos(2 * math.pi * fr * 0.8 * tt + ph[1])
                    vel = vel + acc / 30.0
                    p = p + vel / 30.0
                    gz = ground(p[0], p[1])
                    if p[2] <= gz + 0.004 and vel[2] < 0:
                        p[2] = gz + 0.004
                        landed = True
                        land_k = k
                    rx = 0.9 * math.sin(2 * math.pi * fr * tt + ph[0]) + 0.6
                    ry = 0.9 * math.sin(2 * math.pi * fr * 0.7 * tt + ph[1])
                    rz += spin / 30.0
                else:
                    u = min(1.0, (k - land_k) / 6.0)
                    rx = (0.9 * math.sin(2 * math.pi * fr * (land_k / 30.0) + ph[0]) + 0.6) * (1 - u)
                    ry = (0.9 * math.sin(2 * math.pi * fr * 0.7 * (land_k / 30.0) + ph[1])) * (1 - u)
                if landed and k - land_k > 10:
                    _key(o, f0 + k, loc=tuple(p), rot=(0, 0, rz))
                    break
                _key(o, f0 + k, loc=tuple(p), rot=(rx, ry, rz))
            _linear(o)
            L.VA(o, f0 - 1, F(t_end) + 1)

    # ---------------------------------------------------------------- hats
    def cap_mesh(self):
        import bmesh
        me = bpy.data.meshes.get("fx_cap")
        if me is not None:
            return me
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=0.105)
        for vtx in list(bm.verts):
            vtx.co.z *= 0.8
        bmesh.ops.delete(bm, geom=[vtx for vtx in bm.verts if vtx.co.z < -0.005], context="VERTS")
        # brim: flat wedge toward -Y
        bm2 = bmesh.new()
        pts = []
        for k in range(9):
            a = math.pi * (0.15 + 0.7 * k / 8.0)
            pts.append((0.105 * math.cos(a + math.pi), -0.105 * math.sin(a), 0.0))
        vs = [bm.verts.new(p) for p in pts]
        vo = [bm.verts.new((p[0] * 1.18, p[1] * 1.9 - 0.02, 0.0)) for p in pts]
        for k in range(8):
            try:
                bm.faces.new((vs[k], vs[k + 1], vo[k + 1], vo[k]))
            except ValueError:
                pass
        bm2.free()
        me = bpy.data.meshes.new("fx_cap")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        return me

    def hat(self, fighter, color, t_off, vel, spin, head_hook, t_end=10.0, local=(0, 0, -0.085)):
        """Cap worn on the head hook until t_off, then a ballistic flight (spin about its axis), lands, bounces once."""
        me = self.cap_mesh()
        mat = _mat("MAT_s03_cap_" + fighter, color, 0.6)
        worn = self._obj("cap_worn_" + fighter, me, mat)
        worn.parent = head_hook
        worn.matrix_parent_inverse.identity()
        worn.location = local
        worn.rotation_euler = (0, 0, 0)
        L.VA(worn, 1, F(t_off))
        fly = self._obj("cap_fly_" + fighter, me, mat)
        bpy.context.scene.frame_set(F(t_off))
        bpy.context.view_layer.update()
        mw = worn.matrix_world.copy()
        p = np.array(mw.translation)
        e = mw.to_euler()
        rot = np.array([e.x, e.y, e.z])
        vel = np.array(vel, float)
        f0 = F(t_off)
        total = int(round((t_end - t_off) * 30))
        bounces = 0
        for k in range(total + 1):
            _key(fly, f0 + k, loc=tuple(p), rot=tuple(rot))
            if bounces >= 2 and np.linalg.norm(vel) < 0.05:
                break
            vel[2] -= G / 30.0
            p = p + vel / 30.0
            rot = rot + np.array([spin[0], spin[1], spin[2]]) / 30.0 * (1.0 if bounces == 0 else 0.3)
            gz = ground(p[0], p[1])
            if p[2] < gz + 0.03 and vel[2] < 0:
                p[2] = gz + 0.03
                vel = vel * np.array([0.5, 0.5, -0.35])
                spin = [s_ * 0.4 for s_ in spin]
                bounces += 1
                if bounces >= 3:
                    vel[:] = 0
                    spin = [0, 0, 0]
                    rot[0] *= 0.2
                    rot[1] *= 0.2
        _linear(fly)
        L.VA(fly, f0, F(t_end) + 1)
        bpy.context.scene.frame_set(1)
        return worn, fly
