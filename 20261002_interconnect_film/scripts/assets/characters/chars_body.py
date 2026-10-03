"""Character body, armature, drivers, hole system, hooks. Baseline coordinates (crown 1.75 m), scaled by K = crown/1.75."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import common as C  # noqa: E402
import chars_geo as G  # noqa: E402
import chars_head as HD  # noqa: E402
import chars_mat as M  # noqa: E402

FINGERS = ["index", "middle", "ring", "pinky"]
CURL_ANGLES = {"index": (1.25, 1.45, 1.0), "middle": (1.25, 1.45, 1.0), "ring": (1.25, 1.45, 1.0), "pinky": (1.3, 1.45, 1.0),
               "thumb": (0.55, 0.8, 0.8)}
FINGER_DEF = {  # name: (c offset, lengths, radius, splay)
    "index": (-0.030, (0.040, 0.023, 0.018), 0.0092, -0.05),
    "middle": (-0.010, (0.044, 0.026, 0.019), 0.0095, 0.0),
    "ring": (0.010, (0.040, 0.024, 0.018), 0.0090, 0.05),
    "pinky": (0.028, (0.032, 0.019, 0.016), 0.0080, 0.12),
}


def nrm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ortho(v, f):
    v = np.asarray(v, float)
    return nrm(v - f * np.dot(v, f))


def default_spec():
    return dict(
        H=1.75, build=1.0, belly=0.0, shoulders=1.0, arm_r=1.0, leg_r=1.0, arm_len=1.0, leg_len=1.0,
        head=dict(z_chin=1.517, hh=0.233, w0=0.078, b_front=0.090, b_back=0.100, eye_z=0.117, eye_sep=0.034, eye_r=0.0165,
                  eye_sink=0.006, pupil_r=0.0072, lid_up_lat=0.55, lid_lo_lat=0.45, brow_dz=0.034, nose_z=0.080,
                  nose_len=0.024, ear_z=0.105, jaw=0.12, chin=0.06),
    )


def torso_table(spec):
    z = np.array([0.84, 0.90, 0.98, 1.06, 1.15, 1.25, 1.34, 1.40, 1.44, 1.47])
    rx = np.array([.12, .158, .160, .148, .160, .176, .180, .172, .120, .060]) * spec["build"]
    ry = np.array([.085, .105, .107, .098, .102, .112, .106, .095, .075, .058]) * (0.5 + 0.5 * spec["build"])
    cy = np.array([0, 0, 0, -0.003, -0.004, -0.006, 0, 0.004, 0.006, 0.006])
    b = spec["belly"]
    bump = np.exp(-((z - 1.04) / 0.09) ** 2)
    ry = ry * (1 + b * bump)
    cy = cy - b * 0.04 * bump
    rx = rx * (1 + 0.3 * b * bump)
    rx[7:] = rx[7:] * spec["shoulders"] ** 0.5
    return z, rx, ry, cy


def torso_at(spec, z, off=0.0):
    zs, rx, ry, cy = torso_table(spec)
    return G.interp(zs, rx, z) + off, G.interp(zs, ry, z) + off, G.interp(zs, cy, z)


def joints(spec):
    s = spec
    J = {}
    sw = 0.19 * s["shoulders"]
    J["sh"] = np.array([sw, 0.0, 1.405])
    du = nrm((0.20, -0.03, -0.98))
    df = nrm((0.07, -0.20, -0.98))
    dh = nrm((0.04, -0.22, -0.97))
    J["el"] = J["sh"] + 0.3255 * s["arm_len"] * du
    J["wr"] = J["el"] + 0.2555 * s["arm_len"] * df
    J["dh"] = dh
    J["palm_end"] = J["wr"] + 0.095 * dh
    hipx = 0.09 * (0.9 + 0.1 * s["build"])
    J["hip"] = np.array([hipx, 0.0, 0.925])
    J["knee"] = J["hip"] + 0.43 * s["leg_len"] * nrm((0.03, -0.04, -1.0))
    J["ankle"] = J["knee"] + 0.425 * s["leg_len"] * nrm((-0.01, 0.03, -1.0))
    J["ball"] = J["ankle"] + np.array([0.0, -0.145, -0.03])
    return J


def lerp_w(z, centers, names):
    """Partition-of-unity weights along z between bone centres."""
    out = {n: np.zeros_like(z) for n in names}
    cs = np.array(centers)
    for k in range(len(z)):
        zz = z[k]
        if zz <= cs[0]:
            out[names[0]][k] = 1
        elif zz >= cs[-1]:
            out[names[-1]][k] = 1
        else:
            i = np.searchsorted(cs, zz) - 1
            t = (zz - cs[i]) / (cs[i + 1] - cs[i])
            t = t * t * (3 - 2 * t)
            out[names[i]][k] = 1 - t
            out[names[i + 1]][k] = t
    return out


class Character:
    def __init__(self, cid, spec=None, reset=True):
        self.id = cid
        self.spec = spec or default_spec()
        self.K = self.spec["H"] / 1.75
        self.J = joints(self.spec)
        if reset:
            C.reset()
        self.coll, self.root = C.new_asset(cid, accuracy="C")
        self.scn = bpy.context.scene
        self.m = {}
        self.objs = {}
        self.surfaces = []  # (verts, faces) in final space
        self.holes = []
        self.hooks = []
        self.hole_group = None
        self.bone_names = []
        self._build_armature()

    # ------------------------------------------------------------------ helpers
    def k(self, v):
        return np.asarray(v, float) * self.K

    def mat(self, key, m):
        self.m[key] = m
        return m

    def add(self, part, name, mats, skin=True, register=True, subsurf=0, parent=None, solid=0.0):
        """Scale a Part by K, make an object `<id>_<name>` in the asset collection, skin it to the armature."""
        p = part.copy()
        p.v = p.v * self.K
        ml = [self.m[x] if isinstance(x, str) else x for x in mats]
        ob = G.to_object(p, "%s_%s" % (self.id, name), ml, self.coll, parent=parent or self.root)
        if skin and p.w:
            md = ob.modifiers.new("Armature", "ARMATURE")
            md.object = self.arm
        if solid:
            so = ob.modifiers.new("Solidify", "SOLIDIFY")
            so.thickness = solid * self.K
            so.offset = 1.0
            so.use_even_offset = True
        if subsurf:
            sd = ob.modifiers.new("Subsurf", "SUBSURF")
            sd.levels = subsurf
            sd.render_levels = subsurf + 1
        if register:
            self.surfaces.append((p.v.copy(), list(p.f)))
        self.objs[name] = ob
        return ob

    def bone_parent(self, ob, bone, world=None):
        ob.parent = self.arm
        ob.parent_type = "BONE"
        ob.parent_bone = bone
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        P = ob.matrix_world.copy()
        ob.matrix_parent_inverse = P.inverted()
        ob.matrix_basis = world if world is not None else Matrix.Identity(4)

    def hook(self, name, bone, pos, rot=(0, 0, 0), size=0.05, rotm=None):
        h = bpy.data.objects.new("HOOK_" + name, None)
        h.empty_display_type = "ARROWS"
        h.empty_display_size = size
        self.coll.objects.link(h)
        loc = Matrix.Translation(Vector(self.k(pos)))
        R = Matrix.Rotation(rot[2], 4, "Z") @ Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
        if rotm is not None:
            R = Matrix(np.asarray(rotm, float).tolist()).to_4x4()
        self.bone_parent(h, bone, loc @ R)
        self.hooks.append(dict(name="HOOK_" + name, bone=bone, pos=[float(x) for x in self.k(pos)], rot=list(rot), rotm=None if rotm is None else np.asarray(rotm).tolist()))
        return h

    # ------------------------------------------------------------------ armature
    def _build_armature(self):
        J, K = self.J, self.K
        spec = self.spec
        hs = spec["head"]
        ad = bpy.data.armatures.new(self.id + "_rig")
        arm = bpy.data.objects.new(self.id + "_rig", ad)
        self.coll.objects.link(arm)
        arm.parent = self.root
        arm.show_in_front = True
        self.arm = arm
        self.ad = ad
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        eb = ad.edit_bones
        self.finger_info = {}
        self.hand_frames = {}

        def mk(name, head, tail, parent=None, xaxis=None, deform=True, conn=False):
            b = eb.new(name)
            b.head = Vector(np.asarray(head, float) * K)
            b.tail = Vector(np.asarray(tail, float) * K)
            if xaxis is not None:
                y = nrm(np.asarray(tail, float) - np.asarray(head, float))
                z = np.cross(nrm(xaxis), y)
                b.align_roll(Vector(z))
            if parent:
                b.parent = eb[parent]
            b.use_deform = deform
            self.bone_names.append(name)
            return b

        X = (1, 0, 0)
        mk("root", (0, 0, 0), (0, 0, 0.15), None, X, False)
        mk("hips", (0, 0, 0.925), (0, 0, 1.0), "root", X)
        mk("spine_1", (0, 0, 1.0), (0, 0, 1.12), "hips", X)
        mk("spine_2", (0, 0, 1.12), (0, 0, 1.25), "spine_1", X)
        mk("chest", (0, 0, 1.25), (0, 0, 1.41), "spine_2", X)
        mk("neck", (0, 0.005, 1.41), (0, 0.005, 1.55), "chest", X)
        mk("head", (0, 0.005, 1.55), (0, 0.005, 1.75), "neck", X)
        mk("jaw", (0, 0.015, 1.60), (0, -0.075, 1.56), "head", X)
        # eyes and lids: bones at the eyeball centres
        head = HD.Head(hs)
        self.head = head
        ctr = HD.eye_centres(head, hs)
        self.eye_centres = ctr
        for nm, c in zip(("L", "R"), ctr):
            for b in ("eye", "lid_up", "lid_lo"):
                mk("%s_%s" % (b, nm), c, c + np.array([0, -0.03, 0]), "head", X)
        mk("ctl_look", (0, -0.9, 1.63), (0, -0.9, 1.68), "root", X, False)
        for side, sn in ((1, "L"), (-1, "R")):
            sx = np.array([side, 1, 1])
            sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
            mk("clavicle_" + sn, (side * 0.03, -0.01, 1.42), sh, "chest", X)
            mk("upper_arm_" + sn, sh, el, "clavicle_" + sn, X)
            mk("forearm_" + sn, el, wr, "upper_arm_" + sn, X)
            dh = J["dh"] * sx
            f = nrm(dh)
            n = ortho((-side, 0, 0), f)
            c = ortho((0, 1, 0), f)
            self.hand_frames[sn] = (f, n, c)
            mk("hand_" + sn, wr, wr + 0.095 * f, "forearm_" + sn, np.cross(f, n))
            # fingers
            for fn_, (coff, lens, rad, splay) in FINGER_DEF.items():
                base = wr + 0.100 * f + c * coff
                d = nrm(f + c * splay)
                pos = base
                for k_ in range(3):
                    nxt = pos + d * lens[k_]
                    mk("%s_%d_%s" % (fn_, k_ + 1, sn), pos, nxt, ("hand_" + sn) if k_ == 0 else "%s_%d_%s" % (fn_, k_, sn), np.cross(d, n))
                    self.finger_info["%s_%d_%s" % (fn_, k_ + 1, sn)] = (pos.copy(), nxt.copy(), rad * (1 - 0.06 * k_))
                    pos = nxt
            base = wr + 0.028 * f - c * 0.036 - n * 0.0
            d = nrm(0.75 * f - 0.55 * c - 0.10 * n)
            m_ = nrm(0.5 * n + 0.8 * c)
            pos = base
            for k_, ln in enumerate((0.042, 0.032, 0.026)):
                nxt = pos + d * ln
                mk("thumb_%d_%s" % (k_ + 1, sn), pos, nxt, ("hand_" + sn) if k_ == 0 else "thumb_%d_%s" % (k_, sn), np.cross(d, m_))
                self.finger_info["thumb_%d_%s" % (k_ + 1, sn)] = (pos.copy(), nxt.copy(), 0.0105 * (1 - 0.1 * k_))
                pos = nxt
            hip, kn, an, ba = J["hip"] * sx, J["knee"] * sx, J["ankle"] * sx, J["ball"] * sx
            mk("thigh_" + sn, hip, kn, "hips", X)
            mk("shin_" + sn, kn, an, "thigh_" + sn, X)
            mk("foot_" + sn, an, ba, "shin_" + sn, X)
            # IK / control bones (orientation copies of the driven bones)
            hb = eb["hand_" + sn]
            b = mk("ik_hand_" + sn, wr, wr + 0.095 * f, "chest", np.cross(f, n), False)
            b.roll = hb.roll
            fb = eb["foot_" + sn]
            b = mk("ik_foot_" + sn, an, ba, "root", X, False)
            b.roll = fb.roll
            mk("ik_elbow_" + sn, el + np.array([side * 0.10, 0.40, 0.0]), el + np.array([side * 0.10, 0.45, 0.0]), "chest", X, False)
            mk("ik_knee_" + sn, kn + np.array([side * 0.03, -0.55, 0.0]), kn + np.array([side * 0.03, -0.60, 0.0]), "root", X, False)
            mk("ctl_hand_" + sn, wr + np.array([side * 0.16, 0.0, 0.0]), wr + np.array([side * 0.16, 0.0, 0.05]), "chest", X, False)
        mk("ctl_settings", (0.0, 0.35, 0.05), (0.0, 0.35, 0.12), "root", X, False)
        bpy.ops.object.mode_set(mode="OBJECT")
        self._rig_pose_setup()

    def _rig_pose_setup(self):
        arm = self.arm
        pb = arm.pose.bones
        # custom properties on control bones
        cs = pb["ctl_settings"]
        for k in ("ik_arm_L", "ik_arm_R", "ik_leg_L", "ik_leg_R", "look_at"):
            cs[k] = 1.0
        for sn in ("L", "R"):
            ch = pb["ctl_hand_" + sn]
            for f in FINGERS + ["thumb"]:
                ch["curl_" + f] = 0.12
        # constraints
        def drv(idb, path, bone, prop, expr="v", index=-1):
            fc = idb.driver_add(path, index) if index >= 0 else idb.driver_add(path)
            d = fc.driver
            d.type = "SCRIPTED"
            d.expression = expr
            v = d.variables.new()
            v.name = "v"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = arm
            v.targets[0].data_path = 'pose.bones["%s"]["%s"]' % (bone, prop)
        for sn in ("L", "R"):
            c = pb["forearm_" + sn].constraints.new("IK")
            c.target = arm
            c.subtarget = "ik_hand_" + sn
            c.pole_target = arm
            c.pole_subtarget = "ik_elbow_" + sn
            c.chain_count = 2
            c.use_tail = True
            c.use_stretch = False
            drv(c, "influence", "ctl_settings", "ik_arm_" + sn)
            c2 = pb["hand_" + sn].constraints.new("COPY_ROTATION")
            c2.target = arm
            c2.subtarget = "ik_hand_" + sn
            c2.target_space = "WORLD"
            c2.owner_space = "WORLD"
            drv(c2, "influence", "ctl_settings", "ik_arm_" + sn)
            c = pb["shin_" + sn].constraints.new("IK")
            c.target = arm
            c.subtarget = "ik_foot_" + sn
            c.pole_target = arm
            c.pole_subtarget = "ik_knee_" + sn
            c.chain_count = 2
            c.use_tail = True
            c.use_stretch = False
            drv(c, "influence", "ctl_settings", "ik_leg_" + sn)
            c2 = pb["foot_" + sn].constraints.new("COPY_ROTATION")
            c2.target = arm
            c2.subtarget = "ik_foot_" + sn
            c2.target_space = "WORLD"
            c2.owner_space = "WORLD"
            drv(c2, "influence", "ctl_settings", "ik_leg_" + sn)
            # eyes look at ctl_look
            c3 = pb["eye_" + sn].constraints.new("DAMPED_TRACK")
            c3.target = arm
            c3.subtarget = "ctl_look"
            c3.track_axis = "TRACK_Y"
            drv(c3, "influence", "ctl_settings", "look_at")
            # finger curl drivers
            for f in FINGERS + ["thumb"]:
                for k_ in range(3):
                    b = pb["%s_%d_%s" % (f, k_ + 1, sn)]
                    ang = CURL_ANGLES[f][k_]
                    drv(b, "rotation_euler", "ctl_hand_" + sn, "curl_" + f, "v*%.4f" % ang, 0)
        # bone collections
        ctl = self.ad.collections.new("Controls")
        for n in self.bone_names:
            if n.startswith(("ik_", "ctl_")):
                ctl.assign(self.ad.bones[n])
        self.ad.display_type = "OCTAHEDRAL"
        self._calibrate_poles()

    def eval_tail(self, bone, which="tail"):
        dg = bpy.context.evaluated_depsgraph_get()
        e = self.arm.evaluated_get(dg)
        b = e.pose.bones[bone]
        return np.array(b.tail if which == "tail" else b.head)

    def _calibrate_poles(self):
        """Choose IK pole angles so the elbows point back (+Y pole) and knees forward."""
        pb = self.arm.pose.bones
        K = self.K
        for sn, side in (("L", 1), ("R", -1)):
            for limb, tgt, pole, bone, mid in (("arm", "ik_hand_", "ik_elbow_", "forearm_", "forearm_"), ("leg", "ik_foot_", "ik_knee_", "shin_", "shin_")):
                t = pb[tgt + sn]
                t0 = t.location.copy()
                # move the target to bend the chain sharply
                t.location = Vector((0.0, 0.0, 0.0))
                pbn = pb[bone + sn]
                con = [c for c in pbn.constraints if c.type == "IK"][0]
                best, bestd = 0.0, 1e9
                tl = t.location.copy()
                # bend: raise target (arm: forward-up by pose-space delta; leg: lift foot)
                if limb == "arm":
                    t.location = Vector((0.0, 0.18 * K, 0.0))  # bone local y is up
                else:
                    t.location = Vector((0.0, 0.25 * K, 0.0))
                for ang in (0.0, math.pi / 2, math.pi, -math.pi / 2):
                    con.pole_angle = ang
                    bpy.context.view_layer.update()
                    joint = self.eval_tail(pbn.parent.name, "tail")
                    pole_pos = np.array(self.arm.pose.bones[pole + sn].head) * 1.0
                    d = np.linalg.norm(joint - pole_pos)
                    if d < bestd:
                        best, bestd = ang, d
                con.pole_angle = best
                t.location = t0
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------ body geometry (bare skin)
    def torso_part(self, off=0.0, z0=0.84, z1=1.47, front_only=None, mat=0, nseg=32):
        spec = self.spec
        zs = np.arange(z0, z1 + 1e-6, 0.03)
        if zs[-1] < z1 - 1e-6:
            zs = np.append(zs, z1)
        rx, ry, cy = torso_at(spec, zs, off)
        path = np.stack([np.zeros_like(zs), cy, zs], axis=1)
        p = G.tube(path, rx, ry, u=(1, 0, 0), nseg=nseg, cap0="round" if z0 <= 0.85 else None, cap1="flat", ncap=3, mat=mat, phase=0.0)
        self.torso_weights(p)
        return p

    def torso_weights(self, p):
        z = p.v[:, 2]
        w = lerp_w(z, [0.94, 1.06, 1.185, 1.33], ["hips", "spine_1", "spine_2", "chest"])
        p.only_w(w)
        return p

    def skin_parts(self):
        """Nude body: torso, neck, arms, legs, hands. Returns a merged Part (skin material index 0)."""
        spec, J = self.spec, self.J
        ar, lr = spec["arm_r"], spec["leg_r"]
        parts = [self.torso_part()]
        neck_path = np.array([[0, 0.003, 1.42], [0, 0.005, 1.50], [0, 0.005, 1.60]])
        nk = G.tube(neck_path, [0.050, 0.046, 0.044], [0.052, 0.050, 0.048], nseg=20, cap0="flat", cap1="flat", ncap=2)
        nk.set_w("neck", 1.0)
        parts.append(nk)
        for side, sn in ((1, "L"), (-1, "R")):
            sx = np.array([side, 1, 1])
            sh, el, wr = J["sh"] * sx, J["el"] * sx, J["wr"] * sx
            ua = G.tube(np.array([sh, (sh + el) / 2, el]), np.array([0.047, 0.043, 0.039]) * ar, u=(0, 1, 0), nseg=18)
            ua.set_w("upper_arm_" + sn, 1.0)
            fa = G.tube(np.array([el, (el + wr) / 2, wr]), np.array([0.038, 0.034, 0.027]) * ar, u=(0, 1, 0), nseg=18)
            fa.set_w("forearm_" + sn, 1.0)
            parts += [ua, fa, self.hand_part(sn, side)]
            hip, kn, an = J["hip"] * sx, J["knee"] * sx, J["ankle"] * sx
            th = G.tube(np.array([hip, hip + (kn - hip) * 0.5, kn]), np.array([0.084, 0.072, 0.056]) * lr, u=(0, 1, 0), nseg=20)
            th.set_w("thigh_" + sn, 1.0)
            sh_ = G.tube(np.array([kn, kn + (an - kn) * 0.3, an]), np.array([0.056, 0.059, 0.036]) * lr, u=(0, 1, 0), nseg=20)
            sh_.set_w("shin_" + sn, 1.0)
            parts += [th, sh_]
        return G.merge(parts)

    def hand_part(self, sn, side):
        J = self.J
        f, n, c = self.hand_frames[sn]
        sx = np.array([side, 1, 1])
        wr = J["wr"] * sx
        A = np.stack([c, n, f], axis=1)  # columns: x=c, y=n, z=f
        palm = G.ellipsoid(wr + 0.054 * f, (0.043, 0.0165, 0.056), axes=A, nseg=16, nrings=10)
        palm.set_w("hand_" + sn, 1.0)
        parts = [palm]
        for name, (a, b, r) in self.finger_info.items():
            if not name.endswith("_" + sn):
                continue
            seg = G.capsule(a, b, r, r * 0.94, nseg=10, ncap=2)
            seg.set_w(name, 1.0)
            parts.append(seg)
        return G.merge(parts)

    # ------------------------------------------------------------------ face
    def build_face(self, hair_brow=True):
        """Head, ears, eyes, pupils, lids, brows, nose, teeth, tongue, drops with expression shape keys."""
        hs = self.spec["head"]
        head = self.head
        K = self.K
        hp = head.part()
        ears = HD.make_ears(head, hs)
        hjaw = head.jaw_weights()
        nh = len(hp.v)
        full = G.merge([hp, ears])
        wj = np.concatenate([hjaw, np.zeros(len(ears.v))])
        full.only_w({"head": 1.0 - wj, "jaw": wj})
        ears_v = ears.v.copy()

        def head_fn(e):
            return np.concatenate([head.deformed(e), ears_v], axis=0)
        eyes, pupils, lids = HD.make_eyes(head, hs)
        brows = HD.make_brows(head, hs)
        brows.part.set_w("head", 1.0)
        nose, tip = HD.make_nose(head, hs)
        nose.part.set_w("head", 1.0)
        tu, tl, tg = HD.make_teeth_tongue(head, hs)
        drops = HD.make_drops(head, hs)
        self.nose_tip = tip
        items = [("head", full, head_fn, ["skin", "mouth"], 1), ("eyes", eyes.part, None, ["eye_white"], 0),
                 ("pupils", pupils.part, pupils.fn, ["pupil"], 0), ("lids", lids.part, lids.fn, ["skin"], 0),
                 ("brows", brows.part, brows.fn, ["brow"], 0), ("nose", nose.part, nose.fn, ["skin", "nostril"], 0),
                 ("teeth_up", tu.part, tu.fn, ["teeth"], 0), ("teeth_lo", tl.part, tl.fn, ["teeth"], 0),
                 ("tongue", tg.part, tg.fn, ["tongue"], 0), ("drops", drops.part, drops.fn, ["drop"], 0)]
        for name, part, fn, mats, sub in items:
            reg = name in ("head", "lids", "nose")
            ob = self.add(part, name, mats, register=reg, subsurf=sub)
            if fn is not None:
                HD.add_expression_keys(ob, fn, self.root, K, part.v)
        return head

    # ------------------------------------------------------------------ holes
    def setup_holes(self, specs):
        """Create root props, hole empties (bone-parented) and the shared cutout node group.

        specs: list of dict(name, prop, bone, pos (baseline), axis 'Y'|'X', radius (baseline m), half_len (baseline m)).
        """
        gl = []
        for s in specs:
            e = bpy.data.objects.new("HOLE_" + s["name"], None)
            e.empty_display_type = "CIRCLE"
            e.empty_display_size = s["radius"] * self.K
            self.coll.objects.link(e)
            rot = (-math.pi / 2, 0, 0) if s["axis"] == "Y" else (0, math.pi / 2, 0)
            Rm = Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
            self.bone_parent(e, s["bone"], Matrix.Translation(Vector(self.k(s["pos"]))) @ Rm)
            self.root[s["prop"]] = float(s.get("default", 0.0))
            self.root.id_properties_ui(s["prop"]).update(min=0.0, max=1.0, description="hole radius scale (0 = no hole, 1 = base radius)")
            d = dict(s)
            d["empty"] = e
            d["base_radius"] = s["radius"] * self.K
            d["half_len"] = s["half_len"] * self.K
            gl.append(d)
        self.holes = gl
        self.hole_group = M.hole_group(gl, self.root, name='NG_bullet_holes_' + self.id)
        return self.hole_group

    def build_rims(self, mat):
        """Thin inner tube per hole, following the real front/back surfaces (ray casts), scale driven by the radius prop."""
        vs, fs, off = [], [], 0
        for v, f in self.surfaces:
            vs.append(v)
            fs.extend([tuple(i + off for i in q) for q in f])
            off += len(v)
        V = np.concatenate(vs)
        tris = []
        for q in fs:
            for k in range(1, len(q) - 1):
                tris.append((q[0], q[k], q[k + 1]))
        bvh = BVHTree.FromPolygons([tuple(x) for x in V], tris)
        for h in self.holes:
            e = h["empty"]
            M4 = e.matrix_world
            o = np.array(M4.translation)
            ux, uy, uz = [np.array(M4.to_3x3().col[i]).copy() for i in range(3)]
            ux, uy, uz = nrm(ux), nrm(uy), nrm(uz)
            r = h["base_radius"]
            n = 28
            front, back = [], []
            for k in range(n):
                a = 2 * math.pi * k / n
                off_ = ux * r * math.cos(a) + uy * r * math.sin(a)
                p0 = Vector(o + off_ - uz * 0.6)
                hit = bvh.ray_cast(p0, Vector(uz))
                zf = np.dot(np.array(hit[0]) - o, uz) if hit[0] is not None else -0.1
                p1 = Vector(o + off_ + uz * 0.6)
                hit = bvh.ray_cast(p1, Vector(-uz))
                zb = np.dot(np.array(hit[0]) - o, uz) if hit[0] is not None else 0.1
                front.append(zf)
                back.append(zb)
            verts = []
            for k in range(n):
                a = 2 * math.pi * k / n
                verts.append((r * math.cos(a), r * math.sin(a), front[k] - 0.0005))
            for k in range(n):
                a = 2 * math.pi * k / n
                verts.append((r * math.cos(a), r * math.sin(a), back[k] + 0.0005))
            faces = []
            for k in range(n):
                k1 = (k + 1) % n
                faces.append((k, k1, n + k1, n + k))
            me = bpy.data.meshes.new("%s_%s_rim" % (self.id, h["name"]))
            me.from_pydata(verts, [], faces)
            me.materials.append(mat)
            for p in me.polygons:
                p.use_smooth = True
            me.update()
            ob = bpy.data.objects.new("%s_%s_rim" % (self.id, h["name"]), me)
            self.coll.objects.link(ob)
            ob.parent = e
            for i in (0, 1):
                fc = ob.driver_add("scale", i)
                dr = fc.driver
                dr.type = "SCRIPTED"
                dr.expression = "max(v,0.0001)"
                var = dr.variables.new()
                var.name = "v"
                var.targets[0].id_type = "OBJECT"
                var.targets[0].id = self.root
                var.targets[0].data_path = '["%s"]' % h["prop"]
            h["rim"] = ob.name
            h["axis_world"] = [float(x) for x in uz]
            h["origin_world"] = [float(x) for x in o]

    # ------------------------------------------------------------------ root properties
    def root_props(self, extra=None):
        r = self.root
        for k in HD.KEYS[1:]:
            r["p_expr_" + k] = 0.0
            r.id_properties_ui("p_expr_" + k).update(min=0.0, max=1.0)
        r["p_anger"] = 0.0
        r["p_flush"] = 0.0
        for k in ("p_anger", "p_flush"):
            r.id_properties_ui(k).update(min=0.0, max=1.0)
        for k, v in (extra or {}).items():
            r[k] = v


def meta_common(ch):
    holes = []
    for h in ch.holes:
        holes.append(dict(name=h["name"], root_property=h["prop"], empty="HOLE_" + h["name"], parent_bone=h["bone"],
                          position_object_space_m=[round(float(x), 4) for x in ch.k(h["pos"])],
                          axis_object_space_at_rest=h.get("axis_world"), axis_label=h["axis"],
                          base_radius_m=round(float(h["base_radius"]), 4), cutout_half_length_m=round(float(h["half_len"]), 4),
                          rim_object=h.get("rim"), default_value=float(ch.root[h["prop"]])))
    return dict(
        bones=ch.bone_names,
        shape_keys_on_face_objects=HD.KEYS,
        shape_key_objects=[o for o in ch.objs if ch.objs[o].data.shape_keys],
        expression_properties=["p_expr_" + k for k in HD.KEYS[1:]],
        holes=holes,
        hooks=ch.hooks,
        root_custom_properties={k: (float(ch.root[k]) if isinstance(ch.root[k], (int, float)) else str(ch.root[k])) for k in ch.root.keys() if k.startswith("p_")},
        bone_custom_properties={"ctl_settings": ["ik_arm_L", "ik_arm_R", "ik_leg_L", "ik_leg_R", "look_at"],
                                "ctl_hand_L / ctl_hand_R": ["curl_index", "curl_middle", "curl_ring", "curl_pinky", "curl_thumb"]},
        scale_K=ch.K,
    )
