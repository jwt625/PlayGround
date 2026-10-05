"""S3 brawl choreography (v1.2): two pairs, TERAHOP vs MOLEXX (left) and NUBISS vs AYARR (right), scene-local 1.7-7.5 s.

v1.2 replaces the v1.1 per-frame pose engine (s03_fight.py, no longer used) with the baked motion_v2 fight set
(scripts/assets/characters/motion_v2, libraries per rig in assets/components/characters/motion_v2/npc_<x>_v2_motion_v2.blend).
Every attack strip is started so its manifest contact frame lands on the v1.1 impact time (the SFX cue times in
scripts/audio/sfx_cues.json); the receiver's reaction starts on the same frame (hit-stop is inside both actions).

Lead-foot rule: only the left-acting attack variants (`*_L`) and the reactions that share their guard (fL forward) are used,
so consecutive strips do not swap the lead foot (measured in the v1.2 probe: the right-hand variants mirror the stance).

Geometry: each pair stands on an axis through its centre c, tilted by PHI so one fighter shows a 3/4 face to the camera.
pos = c + u * s, u = (cos PHI, -sin PHI); the left fighter has s < 0 and faces +u, the right fighter s > 0 and faces -u.
Distances between roots follow the manifest root_to_root_m x K (K of the pair, about 1.07) so clinches do not interpenetrate.
"""
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
M2DIR = os.path.join(os.path.dirname(HERE), "assets", "characters", "motion_v2")
if M2DIR not in sys.path:
    sys.path.insert(0, M2DIR)
import motion_v2 as M2  # noqa: E402
from asm import F  # noqa: E402

FPS = 30.0
Y_ROW = 1.35
EVENTS = []   # dicts: t, kind, att, vic, s (strength 0..1), pt (world xyz), dir (+1/-1 along x, away from the attacker)
STRIPS = []   # (vid, name, t0) log


def _set_interp(obj, path, f, interp):
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if fc.data_path == path:
            for kp in fc.keyframe_points:
                if abs(kp.co[0] - f) < 0.5:
                    kp.interpolation = interp


class Fighter:
    def __init__(self, vid, asset):
        self.id = vid
        self.a = asset
        self.K = 1.0
        self.pair = None
        self.side = 0      # -1 left, +1 right
        self.s_keys = []   # (t, s, interp)

    # ---- root path along the pair axis
    def s_at(self, t):
        ks = sorted(self.s_keys)
        if not ks:
            return 0.0
        if t <= ks[0][0]:
            return ks[0][1]
        for (t0, s0, _), (t1, s1, e) in zip(ks[:-1], ks[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-6)
                if e != "LINEAR":
                    u = u * u * (3 - 2 * u)
                return s0 + (s1 - s0) * u
        return ks[-1][1]

    def go(self, t0, t1, s):
        """Hold the current offset until t0, then move to s by t1 (eased)."""
        self.s_keys.append((t0, self.s_at(t0), "BEZIER"))
        self.s_keys.append((t1, s, "BEZIER"))

    def pos(self, t):
        p = self.pair
        s = self.s_at(t)
        return (p.c[0] + p.u[0] * s, p.c[1] + p.u[1] * s, 0.0)

    def yaw(self):
        p = self.pair
        d = (p.u[0] * -self.side, p.u[1] * -self.side)     # facing direction: toward the partner
        return math.atan2(d[0], -d[1])

    def play(self, name, t0, speed=1.0, blend_in=4, **kw):
        st = M2.apply(self.a, name, t0, speed=speed, hold=True, blend_in=blend_in, key_root_yaw=False, **kw)
        STRIPS.append((self.id, name, round(t0, 3)))
        return st

    def bake_root(self, t_from, t_to, row_xy, yaw_row, t_turn=(1.70, 1.95)):
        """Key root location/yaw: row slot (yaw_row) until t_turn[0], fight axis from t_turn[1] to t_to."""
        r = self.a.root
        f0, f1 = F(t_turn[0]), F(t_to)
        for f in range(1, F(10.0) + 1):
            t = (f - 1) / FPS
            if t <= t_turn[0]:
                p, y = (row_xy[0], row_xy[1], 0.0), yaw_row
            elif t < t_turn[1]:
                u = (t - t_turn[0]) / (t_turn[1] - t_turn[0])
                u = u * u * (3 - 2 * u)
                q = self.pos(t_turn[1])
                p = (row_xy[0] + (q[0] - row_xy[0]) * u, row_xy[1] + (q[1] - row_xy[1]) * u, 0.0)
                y = yaw_row + (self.yaw() - yaw_row) * u
            elif t <= t_to:
                p, y = self.pos(t), self.yaw()
            else:
                break
            r.location = p
            r.keyframe_insert("location", frame=f)
            r.rotation_euler = (0, 0, y)
            r.keyframe_insert("rotation_euler", frame=f)
        return F(t_to)


class Pair:
    def __init__(self, left, right, c, phi):
        self.c = c
        self.u = (math.cos(phi), -math.sin(phi))
        self.L, self.R = left, right
        left.pair = right.pair = self
        left.side, right.side = -1, +1
        self.K = 0.5 * (left.K + right.K)

    def sep(self, t0, t1, d, bias=0.0):
        """Move both fighters so their roots are d apart (bias shifts the pair centre along u)."""
        self.L.go(t0, t1, -d / 2 + bias)
        self.R.go(t0, t1, d / 2 + bias)


def event(t, kind, att, vic, s, hz=1.3):
    pv, pa = vic.pos(t), att.pos(t)
    dx, dy = pv[0] - pa[0], pv[1] - pa[1]
    n = math.hypot(dx, dy) or 1.0
    d = 1.0 if dx > 0 else -1.0
    pt = (pv[0] - dx / n * 0.16, pv[1] - dy / n * 0.16 - 0.05, hz * vic.K)
    EVENTS.append(dict(t=t, kind=kind, att=att.id, vic=vic.id, s=s, pt=pt, dir=d))


def attack(att, vic, name, t_contact, contact_frame, speed=1.0, react=None, kind="hook", s=0.8, hz=1.5, blend_in=4, react_blend=2):
    att.play(name, t_contact - contact_frame / FPS / speed, speed=speed, blend_in=blend_in)
    if react:
        vic.play(react, t_contact, blend_in=react_blend)
    event(t_contact, kind, att, vic, s, hz)


# --------------------------------------------------------------------------------------------- composition
def compose(T, M, N, Y, K):
    """T/M/N/Y: Fighter objects (TERAHOP, MOLEXX, NUBISS, AYARR). K: pair rig scale (npc_v2 about 1.0-1.07)."""
    EVENTS.clear()
    STRIPS.clear()
    for f in (T, M, N, Y):
        f.K = K
    A = Pair(T, M, (-1.27, Y_ROW), 0.30)
    B = Pair(N, Y, (1.22, Y_ROW), -0.30)
    D_IDLE, D_BUMP, D_HIT, D_HB, D_WM = 0.98, 0.64 * K, 0.88 * K, 0.72 * K, 0.80 * K
    for P in (A, B):
        P.L.s_keys.append((1.95, -D_IDLE / 2, "BEZIER"))
        P.R.s_keys.append((1.95, D_IDLE / 2, "BEZIER"))
    for f in (T, M, N, Y):
        f.play("fight_idle", 1.62, blend_in=6, repeat=1.0)

    # ---------------- pair A: TERAHOP (left) vs MOLEXX (right)
    T.play("shout_rant", 1.72, blend_in=4, repeat=2.0)
    M.play("shout_rant", 1.86, blend_in=4, repeat=2.0, start_frame=18)
    A.sep(1.98, 2.19, D_BUMP)
    T.play("chest_bump", 2.52 - 10 / FPS, blend_in=3)
    M.play("chest_bump", 2.52 - 10 / FPS, blend_in=3)
    event(2.52, "bump", T, M, 0.6, 1.25)
    event(2.52, "bump", M, T, 0.6, 1.25)
    A.sep(2.62, 2.86, D_HIT)
    attack(T, M, "shove", 3.18, 9, react="react_chest_bump", kind="shove", s=1.0, hz=1.3)
    M.go(3.22, 3.55, M.s_at(3.2) + 0.36)                                   # stagger back
    M.go(3.58, 3.86, T.s_at(3.86) + D_HIT)                                 # lunges back in for the collar
    attack(M, T, "collar_grab_shake_L", 3.98, 9, react="collar_shaken", kind="grab", s=0.4, hz=1.4)
    for tt in (4.08, 4.29):
        EVENTS.append(dict(t=tt, kind="tug", att=M.id, vic=T.id, s=0.35, pt=(T.pos(tt)[0], Y_ROW - 0.1, 0.05), dir=-1.0))
    attack(M, T, "hook_L", 4.84, 11, react="react_head_hit_L", kind="hook", s=0.9)
    attack(T, M, "hook_L", 5.36, 11, react="react_head_hit_L", kind="hook", s=0.9)
    A.sep(5.45, 5.62, 0.82 * K)
    attack(M, T, "uppercut_L", 5.98, 12, react="react_head_hit_L", kind="uppercut", s=1.0, hz=1.45)
    T.go(6.02, 6.40, T.s_at(6.0) - 0.40)                                   # knocked back by the uppercut
    T.go(6.40, 6.66, M.s_at(6.6) - 0.92 * K)                               # charges back in
    T.play("whiff_overbalance_L", 6.82 - 15 / FPS, blend_in=3)
    M.play("duck", 6.72, blend_in=2)
    EVENTS.append(dict(t=6.82, kind="whoosh", att=T.id, vic=M.id, s=0.5, pt=(M.pos(6.82)[0], Y_ROW - 0.3, 1.6), dir=1.0))

    # ---------------- pair B: NUBISS (left) vs AYARR (right)
    N.play("shout_rant", 1.84, blend_in=4, repeat=2.0, start_frame=8)
    Y.play("shout_rant", 1.90, blend_in=4, repeat=2.0)
    B.sep(2.05, 2.30, D_HB)
    attack(Y, N, "head_butt", 2.72, 12, react="react_headbutt", kind="headbutt", s=0.7, hz=1.55)
    N.go(2.76, 3.00, N.s_at(2.75) - 0.22)
    N.go(3.00, 3.12, Y.s_at(3.1) - D_HIT)
    attack(N, Y, "shove", 3.46, 9, react="react_chest_bump", kind="shove", s=1.0, hz=1.3)
    Y.go(3.50, 3.82, Y.s_at(3.48) + 0.36)
    Y.go(3.84, 4.02, N.s_at(4.0) + D_WM)
    Y.play("flail_windmill", 4.02, speed=2.35, blend_in=3, repeat=0.6 * 2.35)
    N.play("react_gut_hit", 4.18, blend_in=2)
    for tt in (4.18, 4.35, 4.52):
        event(tt, "punch", Y, N, 0.4, 1.05)
    B.sep(4.50, 4.62, D_HIT)
    attack(N, Y, "slap_L", 4.90, 10, react="react_slapped_L", kind="hook", s=1.0, hz=1.5)
    Y.go(4.94, 5.20, Y.s_at(4.92) + 0.30)
    Y.go(5.12, 5.28, N.s_at(5.28) + D_WM)
    Y.play("flail_windmill", 5.26, speed=2.5, blend_in=3, repeat=0.72 * 2.5)
    N.play("react_gut_hit", 5.40, blend_in=2)
    for tt in (5.40, 5.56, 5.72, 5.88):
        event(tt, "punch", Y, N, 0.4, 1.05 if tt < 5.7 else 1.4)
    B.sep(5.92, 6.0, D_HIT)
    attack(N, Y, "hook_L", 6.22, 11, react="react_head_hit_L", kind="hook", s=0.8)
    attack(N, Y, "collar_grab_shake_L", 6.70, 9, react="collar_shaken", kind="grab", s=0.3, hz=1.4)
    for tt in (6.80, 6.92):
        EVENTS.append(dict(t=tt, kind="tug", att=N.id, vic=Y.id, s=0.35, pt=(Y.pos(tt)[0], Y_ROW - 0.1, 0.05), dir=1.0))
    return A, B
