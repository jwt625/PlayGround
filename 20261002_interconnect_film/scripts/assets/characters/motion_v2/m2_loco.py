"""Locomotion and idles: walk, brisk walk, run, stomp-walk, turn-in-place, idle variations."""
import math

import numpy as np

import m2_engine as E
from m2_engine import V, nrm, ankle_from_pivot, FOOT_HEEL, FOOT_BALL
from m2_lib import action, neutral, Timeline, hand, ss, blink_curve, smooth_loop, stand_feet

TAU = 2 * math.pi


def sm(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------------------------------- gait core
class Gait:
    """Analytic gait: heel strike / flat / heel lift (ball pivot, world-fixed) / swing, per foot, in the body frame."""

    def __init__(self, rig, N, D, st, p1, p2, th0, th1, h, xw, yaw, swing_ease=1.0, ph_off=0.0):
        self.rig, self.N, self.D, self.st, self.p1, self.p2 = rig, N, D, st, p1, p2
        self.th0, self.th1, self.h, self.xw, self.yaw = th0, th1, h, xw, yaw
        self.az = rig.az
        self.sw = swing_ease
        # symmetric stance: shift y0 so that the mean stance ankle y is zero
        self.y0 = 0.0
        ys = [self.stance(p)[1] for p in np.linspace(0, st, 40)]
        self.y0 = -float(np.mean(ys))

    def stance(self, p):
        D, p1, p2, st = self.D, self.p1, self.p2, self.st
        hy = self.y0 + D * (p - 0.0)
        if p < p1:
            th = self.th0 * (1 - sm(p / p1))
            return ankle_from_pivot(hy, "heel", th, self.xw, az=self.az)
        if p < p2:
            return ankle_from_pivot(hy, "heel", 0.0, self.xw, az=self.az)
        hy2 = self.y0 + D * p2
        by = hy2 - (FOOT_HEEL + FOOT_BALL) + D * (p - p2)
        u = (p - p2) / (st - p2)
        th = -self.th1 * u ** 1.6
        return ankle_from_pivot(by, "ball", th, self.xw, az=self.az)

    def foot(self, p, side):
        p = p % 1.0
        if p < self.st:
            s = self.stance(p)
        else:
            A3 = self.stance(self.st)
            A1 = self.stance(0.0)
            u = (p - self.st) / (1.0 - self.st)
            e = sm(u) if self.sw == 1.0 else u ** self.sw * (3 - 2 * u)
            s = A3 + (A1 - A3) * e
            s = s.copy()
            ez = sm(u)
            s[2] = A3[2] + (A1[2] - A3[2]) * ez + self.h * math.sin(math.pi * u) ** 0.8
            s[3] = A3[3] + (A1[3] - A3[3]) * sm(u * 1.1) - 0.05 * math.sin(math.pi * u)
        s = s.copy()
        s[4] = side * self.yaw
        return s


def gait_spec(rig, kind="walk", N=30, D=0.95, st=0.62, p1=0.10, p2=0.38, th0=0.35, th1=0.60, h=0.10, xw=0.092,
              yaw=0.07, arm=0.5, arm_leff=0.52, twist=0.12, lean=0.05, comp=0.0, bob_gain=1.0, sway=0.028, shrug=0.0,
              fist=0.25, stomp=0.0, head_flex=0.0, jaw=0.02, margin=0.975, knee_out=0.03, arm_out=0.07, hand_y=0.0,
              chest_sq=0.012, spring=True):
    G = Gait(rig, N, D, st, p1, p2, th0, th1, h, xw, yaw)
    dims = rig.dims
    hx, hz, hy = float(dims["hip"][0]), float(dims["hip"][2]), float(dims["hip"][1])
    Lm = (rig.Lthigh + rig.Lshin) * margin
    S = dims["sh"].copy()
    ph = np.arange(N) / N
    # --- hips height from leg reach (both legs), with knee flexion `comp` on stance legs
    dz = np.zeros(N)
    sxs = sway * np.sin(TAU * ph)
    for i in range(N):
        best = 9.0
        for side, off in ((1, 0.0), (-1, 0.5)):
            p = (ph[i] + off) % 1.0
            a = G.foot(p, side)
            dx = side * a[0] - (side * hx + sxs[i])
            dy = a[1] - hy
            r2 = Lm * Lm - dx * dx - dy * dy
            z = a[2] + math.sqrt(max(r2, 0.0)) - hz
            if p < st:
                z -= comp * math.sin(math.pi * p / st) ** 1.0
            best = min(best, z)
        dz[i] = best
    dz = smooth_loop(dz.reshape(-1, 1), 1).ravel() * bob_gain + (1 - bob_gain) * dz.mean()
    dz_mean = float(dz.mean())
    # stomp: extra weight dump at contact (decaying)
    if stomp:
        for i in range(N):
            for off in (0.0, 0.5):
                pp = (ph[i] + off) % 1.0
                if pp < 0.22:
                    dz[i] -= stomp * 0.05 * math.exp(-pp / 0.07)

    fn_cache = {}

    def fn(frame):
        i = int(round(frame)) % N
        if i in fn_cache:
            return fn_cache[i]
        pL = ph[i]
        pR = (pL + 0.5) % 1.0
        P = neutral()
        P["foot_L"] = G.foot(pL, 1)
        P["foot_R"] = G.foot(pR, -1)
        c, s = math.cos(TAU * pL), math.sin(TAU * pL)
        bob = (dz[i] - dz_mean)
        P["hips_loc"] = V(sxs[i], 0.0, dz[i])
        yaw_h = -twist * c
        lean_h = -0.04 * s * (1 + 1.0 * (kind == "stomp"))
        P["hips_rot"] = V(0.02 + 0.5 * lean * 0.3, yaw_h, lean_h)
        spine_yaw = 1.6 * twist * c
        spine_lean = 0.04 * s - 0.2 * lean_h
        P["spine"] = V(lean + 0.6 * bob * 0, spine_yaw, spine_lean)
        # head stabilisation: cancel upstream yaw/roll, counter pitch; small nod with bob
        tot_yaw = yaw_h + spine_yaw
        tot_roll = lean_h + spine_lean
        P["neck"] = V(-0.4 * lean - 0.3 * head_flex * 0 + 0.0, -0.45 * tot_yaw, -0.45 * tot_roll)
        P["head"] = V(-0.5 * lean + head_flex + 2.0 * bob, -0.55 * tot_yaw, -0.55 * tot_roll)
        P["jaw"] = V(jaw)
        P["clav_L"] = V(shrug + 0.05 * c, 0.0)
        P["clav_R"] = V(shrug - 0.05 * c, 0.0)
        # arms: counter-swing; left arm forward when the right foot is forward
        for sn, side, pp in (("L", 1, pL), ("R", -1, pR)):
            th = -arm * math.cos(TAU * pp)
            L = arm_leff - 0.05 * max(th, 0.0) / max(arm, 1e-6) * 0.5
            if kind == "run":
                L = 0.40 - 0.03 * math.cos(TAU * pp)
            pos = V(S[0] + arm_out, S[1] - L * math.sin(th), S[2] - L * math.cos(th) + hand_y)
            bent = kind in ("run", "stomp")
            f = (0.0, -0.9 * math.sin(th) - (0.55 if bent else 0.04), -math.cos(th) * (0.7 if bent else 1.0))
            P["hand_" + sn] = hand(pos, f=f, n=(-1, 0, 0.15 if kind == "run" else 0.0), side=side)
            P["hfol_" + sn] = V(1.0)
            P["curl_" + sn] = V(fist) if kind not in ("run", "stomp") else V(1.0, 1.0, 1.0, 1.0, 0.7)
            P["elbow_" + sn] = V(0.20, 0.35, -0.12) if kind == "run" else V(0.17, 0.38, -0.30)
            P["knee_" + sn] = V(knee_out, -0.55, 0.0)
        P["sq_c"] = ss(chest_sq * (bob / 0.04 if abs(dz.max() - dz.min()) > 1e-6 else 0))
        fn_cache[i] = P
        return P

    # events: contacts, down (lowest hips), up (highest hips)
    ev = [("contact_L", 0), ("contact_R", N // 2)]
    i_dn = int(np.argmin(dz[: N // 2]))
    i_up = int(np.argmax(dz[: N // 2]))
    ev += [("down_L", i_dn), ("up_L", i_up), ("down_R", i_dn + N // 2), ("up_R", i_up + N // 2),
           ("toe_off_L", int(round(st * N))), ("toe_off_R", (int(round(st * N)) + N // 2) % N)]
    meta = dict(stride_m=D * rig.K, cycle_frames=N, root_speed_mps=D * rig.K / (N / 30.0), step_m=D * rig.K / 2,
                hips_bob_m=float((dz.max() - dz.min()) * rig.K), stance_fraction=st)
    ov = dict(head=(7.0, 0.5), neck=(8.0, 0.55)) if spring else {}
    return dict(fn=fn, n=N, loop=True, events=ev, meta=meta, overlap=ov)


@action("walk", use="default calm walk: Gary and customers (S1, S2, S6); root speed in manifest", note="heel-toe walk, hips bob from leg reach, pelvis rotation, arm counter-swing, head stabilised")
def b_walk(rig):
    return gait_spec(rig, "walk", N=30, D=0.95, st=0.62, arm=0.50, twist=0.12, lean=0.05, comp=0.012)


@action("walk_brisk", use="hurried walk / power walk: managers, NPC entrances", note="shorter cycle, forward lean")
def b_walk_brisk(rig):
    return gait_spec(rig, "walk", N=24, D=1.05, st=0.58, p1=0.09, p2=0.34, arm=0.62, twist=0.14, lean=0.09, comp=0.015, th1=0.55, h=0.11, fist=0.5)


@action("run", use="chase / panic run", note="two-foot flight phase, 3 m/s baseline, forward lean, pumping arms")
def b_run(rig):
    return gait_spec(rig, "run", N=20, D=2.0, st=0.38, p1=0.06, p2=0.20, th0=0.15, th1=0.9, h=0.22, xw=0.085, arm=0.95, twist=0.20, lean=0.20,
                     comp=0.075, bob_gain=1.0, fist=1.0, margin=0.99, chest_sq=0.02, shrug=0.0, knee_out=0.02)


@action("stomp_walk", use="angry Manager marching (S1 arrival, S2 approach)", note="heavy stomps with weight dump, hunched shoulders, clenched fists, head forward")
def b_stomp(rig):
    sp = gait_spec(rig, "stomp", N=36, D=0.85, st=0.64, p1=0.06, p2=0.40, th0=0.12, th1=0.35, h=0.16, xw=0.105, arm=0.32, arm_leff=0.42, twist=0.09,
                   lean=0.10, comp=0.03, fist=1.0, stomp=1.0, shrug=0.18, head_flex=0.22, jaw=0.25, arm_out=0.10, chest_sq=0.03, yaw=0.05)
    sp["meta"]["note_events"] = "contact_L / contact_R are the stomp impact frames"
    sp["overlap"] = dict(head=(5.5, 0.35), neck=(6.5, 0.4))
    sp["events"] = [("stomp_L", 0), ("stomp_R", 18)] + [e for e in sp["events"] if not e[0].startswith("contact")]
    return sp
