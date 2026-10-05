"""Idle variations and turn-in-place."""
import math

import numpy as np

import m2_engine as E
from m2_engine import V, nrm, ankle_from_pivot
from m2_lib import action, neutral, Timeline, hand, ss, blink_curve, stand_feet

TAU = 2 * math.pi


def idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.31, 0.78), look=0.05, sway=0.03):
    bl = [int(N * b) for b in blinks]
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        u = i / N
        s = math.sin(TAU * u * shift)
        b = math.sin(TAU * u * nb)
        P = neutral()
        P["hips_loc"] = V(sway * s, 0.0, -0.012 - 0.006 * abs(s))
        P["hips_rot"] = V(0.0, 0.03 * math.sin(TAU * u * shift + 1.0), -0.035 * s)
        P["spine"] = V(0.02 - 0.012 * b, 0.02 * math.sin(TAU * u * shift + 2.0), 0.03 * s)
        P["neck"] = V(0.01 * b, 0, 0)
        P["head"] = V(0.02 + 0.012 * b, look * math.sin(TAU * u + 1.0), -0.02 * s)
        P["sq_c"] = ss(0.011 * b)
        P["clav_L"] = V(0.035 * b, 0)
        P["clav_R"] = V(0.035 * b, 0)
        P["foot_L"] = V(0.10, 0.0 - 0.025 * max(-s, 0), az, 0, 0.07)
        P["foot_R"] = V(0.10, 0.0 - 0.025 * max(s, 0), az, 0, -0.07)
        P["blink"] = V(blink_curve(i, N, bl))
        P["hfol_L"] = V(1.0)
        P["hfol_R"] = V(1.0)
        P["curl_L"] = V(0.2)
        P["curl_R"] = V(0.2)
        return P
    return fn


@action("idle_breathe", use="default standing idle: Gary / customers / NPC beats between actions", note="breathing with weight shift, two blinks, tiny head sway; loop 96 frames")
def b_idle_breathe(rig):
    return dict(fn=idle_fn(rig, 96), n=96, loop=True, overlap=dict(head=(6.0, 0.5)))


@action("idle_breathe_tense", use="idle for nervous or angry standing (Manager before the outburst)", note="faster breathing, less shift, clenched hands; loop 72 frames")
def b_idle_tense(rig):
    f0 = idle_fn(rig, 72, nb=3, shift=1.0, blinks=(0.5,), look=0.025, sway=0.012)

    def fn(frame):
        P = f0(frame)
        P["curl_L"] = V(0.9)
        P["curl_R"] = V(0.9)
        P["clav_L"] = P["clav_L"] + V(0.12, 0)
        P["clav_R"] = P["clav_R"] + V(0.12, 0)
        P["hips_loc"] = P["hips_loc"] + V(0, 0, -0.01)
        P["spine"] = P["spine"] + V(0.06, 0, 0)
        P["head"] = P["head"] + V(0.08, 0, 0)
        return P
    return dict(fn=fn, n=72, loop=True, overlap=dict(head=(6.0, 0.5)))


@action("idle_fidget", use="nervous waiting idle: Gary in S1/S2/S6 before things go wrong", note="hands rub at the belt, weight shift, glance up, shoulder roll; loop 120 frames")
def b_idle_fidget(rig):
    N = 120
    base = idle_fn(rig, N, nb=3, shift=1.0, blinks=(0.18, 0.55, 0.9))

    def fn(frame):
        i = int(round(frame)) % N
        u = i / N
        P = base(frame)
        a = TAU * u * 4
        for sn, side in (("L", 1), ("R", -1)):
            ph = a + (0 if side > 0 else math.pi * 0.5)
            c, s = math.cos(ph), math.sin(ph)
            pos = (0.11 + 0.025 * c, -0.31 + 0.025 * s, 1.03 + 0.015 * s)
            P["hand_" + sn] = hand(pos, f=(-0.3, -0.6, -0.3), n=(-0.2, 0.2, 0.9), side=side)
            P["curl_" + sn] = V(0.55)
            P["elbow_" + sn] = V(0.20, 0.30, -0.15)
        glance = math.exp(-((i - 84) / 6.0) ** 2)
        P["head"] = P["head"] + V(0.20 - 0.30 * glance, 0.18 * glance, 0)
        P["gaze"] = V(0.25 * glance, 0.0)
        roll = math.exp(-((i - 100) / 5.0) ** 2)
        P["clav_L"] = P["clav_L"] + V(0.25 * roll, 0)
        P["clav_R"] = P["clav_R"] + V(0.25 * roll, 0)
        P["spine"] = P["spine"] + V(0.10, 0, 0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.4), hand_L=(8.0, 0.5), hand_R=(8.0, 0.5)), events=[("glance_up", 84)])


@action("idle_look_around", use="scanning the room: S3 vendors, S1 Gary before the bang", note="eyes lead, head turns left, pause, sweeps right, returns; loop 132 frames")
def b_idle_look(rig):
    N = 132
    base = idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.14, 0.5, 0.86), look=0.0)
    ks = [(0, 0.0, 0.0, 0.0, "smooth"), (10, 0.0, 0.45, 0.0, "out"), (20, 0.75, 0.0, 0.10, "back"), (46, 0.75, 0.0, 0.10, "smooth"),
          (54, 0.0, -0.40, 0.0, "smooth"), (64, -0.80, 0.0, -0.12, "back"), (92, -0.80, 0.0, -0.12, "smooth"), (108, -0.05, 0.0, 0.0, "back"), (132, 0.0, 0.0, 0.0, "smooth")]

    def interp(i):
        i = i % N
        for (t0, h0, g0, s0, _), (t1, h1, g1, s1, es) in zip(ks[:-1], ks[1:]):
            if t0 <= i <= t1:
                u = (i - t0) / max(t1 - t0, 1)
                e = E.ease(es, u)
                return h0 + (h1 - h0) * e, g0 + (g1 - g0) * E.ease("smooth", u), s0 + (s1 - s0) * e
        return 0.0, 0.0, 0.0

    def fn(frame):
        P = base(frame)
        h, g, s = interp(int(round(frame)))
        P["head"] = P["head"] + V(-0.03 * abs(h), 0.8 * h, 0.0)
        P["neck"] = P["neck"] + V(0, 0.2 * h, 0)
        P["spine"] = P["spine"] + V(0, s, 0)
        P["hips_rot"] = P["hips_rot"] + V(0, 0.4 * s, 0)
        P["gaze"] = V(g, 0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(7.0, 0.45), neck=(8.0, 0.5)), events=[("look_left", 20), ("look_right", 64)])


@action("idle_arms_cross", use="impatient / smug / skeptical standing (Manager, vendors, S6 customers)", note="arms crossed over the chest, foot tap, head tilt, breathing; loop 72 frames")
def b_idle_cross(rig):
    N = 72
    base = idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.4,), look=0.04, sway=0.015)

    def fn(frame):
        i = int(round(frame)) % N
        u = i / N
        P = base(frame)
        P["hand_L"] = hand((-0.20, -0.20, 1.21), f=(-1.0, 0.25, 0.0), n=(0.1, 0.9, -0.3), side=1)
        P["hand_R"] = hand((0.20, -0.14, 1.19), f=(-1.0, 0.25, 0.0), n=(0.1, 0.9, -0.3), side=-1)
        P["elbow_L"] = V(0.12, -0.30, -0.25)
        P["elbow_R"] = V(0.12, -0.30, -0.25)
        P["curl_L"] = V(0.7)
        P["curl_R"] = V(0.7)
        tap = max(0.0, math.sin(TAU * u * 4)) ** 0.7
        P["foot_R"] = ankle_from_pivot(0.10 - 0.04, "heel", 0.30 * tap, 0.12, yaw=-0.10, az=rig.az)
        P["head"] = P["head"] + V(0.04, 0.0, 0.10)
        P["spine"] = P["spine"] + V(0.04, 0, 0)
        P["lid_up"] = V(0.12)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.45)), events=[("tap_%d" % k, 9 * k + 4) for k in range(4)])


# ---------------------------------------------------------------------------------------------------- turn in place
def make_turn(rig, theta, N=26):
    """Turn left by theta (rad, + = left): head and eyes lead, body rotates about the root, feet pivot/step. Ends in a neutral stance rotated by theta.
    The root BONE carries the yaw; motion_v2.apply keys the root object yaw by `root_yaw_delta_rad` at the strip end."""
    az = rig.az
    st = 0.10
    sg = 1.0 if theta > 0 else -1.0

    def foot_final(side):
        p = E.rz(theta) @ V(side * st, 0, 0)
        return V(side * p[0], p[1], az, 0, theta + side * 0.06)

    T = Timeline()
    th = theta
    T.key(0, foot_L=V(st, 0, az, 0, 0.06), foot_R=V(st, 0, az, 0, -0.06), hfol_L=1.0, hfol_R=1.0)
    T.key(5, "out", head=(0.0, 0.6 * th, 0), neck=(0, 0.2 * th, 0), gaze=(0.3 * th, 0), spine=(0.03, -0.10 * th, 0), hips_loc=(0.02 * sg, 0, -0.02))
    outside = "R" if sg > 0 else "L"
    inside = "L" if sg > 0 else "R"
    T.key(11, "snap", root_rot=(0, 0.75 * th, 0), head=(0, 0.35 * th, 0), neck=(0, 0.1 * th, 0), spine=(0.03, 0.2 * th, 0), hips_loc=(0, 0, -0.02),
          **{"foot_" + outside: foot_final(-sg)})
    T.key(19, "out", root_rot=(0, 1.04 * th, 0), head=(0, 0.0, 0), neck=(0, 0, 0), gaze=(0, 0), spine=(0.0, 0, 0), hips_loc=(0, 0, -0.01),
          **{"foot_" + inside: foot_final(sg)})
    T.key(N, "smooth", root_rot=(0, th, 0), hips_loc=(0, 0, 0))
    return dict(timeline=T, n=N, loop=False, overlap=dict(head=(6.0, 0.4), neck=(7.0, 0.45)),
                meta=dict(root_yaw_delta_rad=float(theta)), events=[("body_turn", 11), ("settle", 19)])


@action("turn_left_90", also_mirror="turn_right_90", use="turn in place 90 degrees; object yaw handoff is automatic in apply()", note="head leads, shoulders and hips follow, foot step; ends in a neutral stance rotated 90 degrees")
def b_turn90(rig):
    return make_turn(rig, math.pi / 2, 26)


@action("turn_left_180", also_mirror="turn_right_180", use="turn around in place 180 degrees", note="head leads, two-foot pivot")
def b_turn180(rig):
    return make_turn(rig, math.pi, 34)
