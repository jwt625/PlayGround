"""Gestures, gun handling, whiteboard / typing, anger beats, Gary reactions, hands-busy loops, hug and whip reactions."""
import math

import numpy as np

import m2_engine as E
from m2_engine import V, nrm
from m2_lib import action, neutral, Timeline, hand, ss, blink_curve
from m2_fight import sq, finish, HS, FIST, FLAT, GRIP
from m2_idle import idle_fn

TAU = 2 * math.pi
POINT = dict(index=0.0, middle=0.95, ring=0.95, pinky=0.95, thumb=0.6)


def curl(**kw):
    d = dict(index=0.18, middle=0.18, ring=0.18, pinky=0.18, thumb=0.18)
    d.update(kw)
    return V(d["index"], d["middle"], d["ring"], d["pinky"], d["thumb"])


def stand_keys(rig, **kw):
    az = rig.az
    d = dict(foot_L=V(0.10, 0, az, 0, 0.06), foot_R=V(0.10, 0, az, 0, -0.06))
    d.update(kw)
    return d


def T0(rig, **kw):
    T = Timeline()
    T.key(0, **stand_keys(rig, **kw))
    return T


# ------------------------------------------------------------------------------------------------------------ pointing
@action("point", acting="R", use="pointing at a person / the scope / the screen; jab at frame 14", note="arm draws back, snaps out to a pointing finger with overshoot, two small jabs, holds, retracts")
def b_point(rig):
    T = T0(rig)
    tgt = (0.25, -0.52, 1.42)
    T.key(7, "out", spine=(0.04, -0.1, 0), hips_rot=(0, -0.05, 0), head=(0.0, 0.1, 0), hand_L=hand((0.28, -0.20, 1.18), f=(0.1, -0.8, 0.3), n=(-0.6, 0, -0.8)), curl_L=curl(index=0, middle=0.95, ring=0.95, pinky=0.95, thumb=0.6),
          elbow_L=V(0.25, 0.35, -0.2), **sq(c=-0.03))
    T.key(11, "snap", spine=(0.12, 0.25, 0), hips_rot=(0, 0.1, 0), head=(0.04, 0.35, 0), jaw=0.2, hand_L=hand(tgt, f=(0.12, -1, 0.08), n=(-0.6, 0, -0.8)), look_mix=1.0, look=(0.6, -3, 1.5), **sq(c=0.05))
    T.key(14, "out", hand_L=hand((0.25, -0.57, 1.44), f=(0.12, -1, 0.14), n=(-0.6, 0, -0.8)))
    T.key(18, "smooth", hand_L=hand(tgt, f=(0.12, -1, 0.08), n=(-0.6, 0, -0.8)))
    T.key(21, "out", hand_L=hand((0.25, -0.57, 1.44), f=(0.12, -1, 0.14), n=(-0.6, 0, -0.8)))
    T.key(26, "smooth", hand_L=hand(tgt, f=(0.12, -1, 0.08), n=(-0.6, 0, -0.8)))
    T.key(36, "smooth", jaw=0.0)
    T.key(48, "smooth", **stand_keys(rig), spine=(0, 0, 0), hips_rot=(0, 0, 0), head=(0, 0, 0), look_mix=0.0, hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)), curl_L=curl(), **sq())
    return finish(T, rig, 48, [("jab", 14), ("jab2", 21), ("retract", 36)], overlap=dict(head=(6.0, 0.4), hand_L=(8.0, 0.5)), meta=dict(aim_dir="forward and 20 degrees to the actor's side"))


@action("point_accuse", acting="R", use="angry accusing point with the whole body (Manager at Gary; vendors at each other)", note="big lunge, long arm, stabbing finger jabs with shout; loop-friendly end hold")
def b_point_accuse(rig):
    az = rig.az
    T = T0(rig)
    T.key(6, "out", hips_loc=(0, 0.06, -0.06), spine=(-0.08, -0.15, 0), head=(-0.1, 0.0, 0), hand_L=hand((0.30, 0.0, 1.30), f=(0.1, -0.5, 0.4), n=(-0.6, 0, -0.8)), curl_L=curl(index=0, middle=0.95, ring=0.95, pinky=0.95, thumb=0.6),
          elbow_L=V(0.3, 0.3, -0.1), **sq(c=0.04))
    T.key(10, "in", hips_loc=(0, -0.10, -0.10), hips_rot=(0.0, 0.2, 0), spine=(0.30, 0.35, 0), head=(0.12, 0.45, 0), neck=(0.1, 0.1, 0), jaw=0.6, foot_L=V(0.14, -0.20, az, 0, 0.1),
          hand_L=hand((0.20, -0.66, 1.46), f=(0.1, -1, 0.1), n=(-0.6, 0, -0.8)), look_mix=1.0, look=(0.5, -3, 1.5), **sq(c=0.06, s=0.04))
    for k, (f, d) in enumerate(((13, 0.0), (17, 0.07), (21, 0.0), (25, 0.07), (29, 0.0))):
        T.key(f, "out" if d > 0 else "in", hand_L=hand((0.20, -0.66 - d * 0.5, 1.46 - d * 0.2), f=(0.1, -1, 0.1), n=(-0.6, 0, -0.8)), jaw=0.6 if d == 0 else 0.3,
              spine=(0.30 + 0.05 * (d > 0), 0.35, 0))
    T.key(40, "smooth", jaw=0.2)
    return finish(T, rig, 40, [("jab", 13), ("jab2", 17), ("jab3", 21), ("jab4", 25)], overlap=dict(head=(6.0, 0.4), hand_L=(9.0, 0.5)),
                  face=[(0, {"p_anger": 0.0}), (10, {"p_anger": 1.0, "p_expr_yell": 0.7}), (40, {"p_anger": 1.0, "p_expr_yell": 0.2})], props=["p_anger", "p_expr_yell"], meta=dict(aim_dir="forward-left of the actor"))


# ------------------------------------------------------------------------------------------------------------ shout / rant
@action("shout_rant", use="ranting: Manager, vendors, NPC; whole-body beats every 12 frames, alternating chops; loop 36 frames", note="three emphasis beats per loop with anticipation, hips drop, spine and head slam, jaw open, arm chops and a pointing beat")
def b_rant(rig):
    N = 36
    az = rig.az
    beats = [
        dict(L=((0.30, -0.15, 1.55), (0.22, -0.55, 1.20)), R=((0.20, -0.20, 1.20), (0.20, -0.40, 1.15)), twist=-0.15),
        dict(L=((0.20, -0.30, 1.20), (0.20, -0.30, 1.20)), R=((0.40, -0.10, 1.65), (0.20, -0.62, 1.25)), twist=0.15),
        dict(L=((0.50, -0.05, 1.35), (0.55, -0.25, 1.40)), R=((0.50, -0.05, 1.35), (0.55, -0.25, 1.40)), twist=0.0),
    ]

    def fn(frame):
        i = int(round(frame)) % N
        b = i // 12
        q = (i % 12) / 12.0
        bn = beats[(b + 1) % 3]
        bc = beats[b]
        # envelope: wind-up on q in [0.55, 1] toward next beat's "up" pose, slam at q=0
        slam = math.exp(-q * 5.0)
        wind = max(0.0, (q - 0.55) / 0.45) ** 1.5
        P = neutral()
        P["hips_loc"] = V(0, -0.02, -0.10 - 0.05 * slam + 0.03 * wind)
        P["hips_rot"] = V(0.0, bc["twist"] * 0.5 * (1 - wind) + bn["twist"] * -0.2 * wind, 0)
        P["spine"] = V(0.10 + 0.22 * slam - 0.10 * wind, bc["twist"] * (1 - wind), 0.05 * math.sin(TAU * b / 3))
        P["neck"] = V(0.05 + 0.1 * slam, 0, 0)
        P["head"] = V(0.05 + 0.18 * slam - 0.08 * wind, 0.15 * bc["twist"], 0.05 * (1 if b % 2 else -1))
        P["jaw"] = V(0.15 + 0.55 * slam + 0.1 * math.sin(TAU * i / 4))
        P["foot_L"] = V(0.16, -0.10, az, 0, 0.2)
        P["foot_R"] = V(0.16, 0.10, az, 0, -0.2)
        P["knee_L"] = V(0.10, -0.55, 0)
        P["knee_R"] = V(0.10, -0.55, 0)
        for sn, side in (("L", 1), ("R", -1)):
            up, hit = beats[b][sn]
            upn = beats[(b + 1) % 3][sn][0]
            pos = np.array(hit) * (slam * 0.9) + np.array(up) * (1 - slam) * (1 - wind) + np.array(upn) * (1 - slam) * wind
            P["hand_" + sn] = hand(tuple(pos), f=(0.1, -0.8, 0.5 - 0.7 * slam), n=(-1, 0, 0.3), side=side)
            P["curl_" + sn] = V(1.0 if (b + (sn == "L")) % 3 != 2 else 0.1)
            P["elbow_" + sn] = V(0.30, 0.2, -0.1)
        P["sq_c"] = ss(0.05 * slam - 0.02 * wind)
        P["sq_h"] = ss(0.03 * slam)
        P["clav_L"] = V(0.15 * slam, 0)
        P["clav_R"] = V(0.15 * slam, 0)
        return P
    face = [(0, {"p_anger": 0.8, "p_expr_yell": 0.6}), (N, {"p_anger": 0.8, "p_expr_yell": 0.6})]
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(7.0, 0.38), neck=(8.0, 0.4), hand_L=(10.0, 0.5), hand_R=(10.0, 0.5)), events=[("beat1", 0), ("beat2", 12), ("beat3", 24)],
                face=face, props=["p_anger", "p_expr_yell"], meta=dict(note2="beats are the slam frames: sync shout audio syllables or SFX"))


@action("anger_outburst", use="Manager's explosion after the slow burn: inhale, scream with head thrust and shaking fists, pant", note="leans back to inhale, snaps forward with fists down and body shaking, then pants; 72 frames")
def b_outburst(rig):
    az = rig.az
    T = T0(rig)
    fistdown = dict(hand_L=hand((0.26, -0.18, 0.80), f=(0.1, -0.5, -0.8), n=(-1, 0, 0)), hand_R=hand((0.26, -0.18, 0.80), f=(0.1, -0.5, -0.8), n=(-1, 0, 0), side=-1), curl_L=FIST, curl_R=FIST)
    T.key(10, "out", hips_loc=(0, 0.06, -0.04), spine=(-0.22, 0, 0), neck=(-0.25, 0, 0), head=(-0.2, 0, 0), clav_L=(0.3, 0), clav_R=(0.3, 0), **fistdown, **sq(c=0.06), jaw=0.0)
    T.key(14, "in", hips_loc=(0, -0.08, -0.12), spine=(0.40, 0, 0), neck=(0.15, 0, 0), head=(0.20, 0, 0), jaw=0.75, clav_L=(0.15, 0), clav_R=(0.15, 0), foot_L=V(0.15, -0.10, az, 0, 0.1), foot_R=V(0.15, 0.10, az, 0, -0.1),
          **fistdown, **sq(c=-0.03, h=0.05))
    for k in range(7):
        f = 18 + 4 * k
        s = 1 if k % 2 else -1
        T.key(f, "smooth", hips_loc=(0.015 * s, -0.08, -0.12 - 0.015 * (k % 2)), spine=(0.40 + 0.03 * s, 0.05 * s, 0), head=(0.20, 0.06 * s, 0.05 * s), jaw=0.75 - 0.1 * (k % 2),
              hand_L=hand((0.28, -0.18 - 0.06 * s, 0.82 + 0.05 * s), f=(0.1, -0.5, -0.8), n=(-1, 0, 0)), hand_R=hand((0.28, -0.18 + 0.06 * s, 0.82 - 0.05 * s), f=(0.1, -0.5, -0.8), n=(-1, 0, 0), side=-1))
    T.key(52, "out", hips_loc=(0, 0.0, -0.08), spine=(0.25, 0, 0), jaw=0.2, **sq(c=-0.03))
    T.key(60, "smooth", hips_loc=(0, 0, -0.06 + 0.0), spine=(0.30, 0, 0), jaw=0.15, **sq(c=0.02))
    T.key(66, "smooth", hips_loc=(0, 0, -0.08), spine=(0.34, 0, 0), jaw=0.15, **sq(c=-0.02))
    T.key(72, "smooth", hips_loc=(0, 0, -0.06), spine=(0.30, 0, 0), jaw=0.1, **sq(c=0.02))
    return finish(T, rig, 72, [("inhale_peak", 10), ("scream_start", 14), ("shake_end", 46), ("pant", 60)],
                  overlap=dict(head=(6.0, 0.35), neck=(7.0, 0.4), hand_L=(9.0, 0.5), hand_R=(9.0, 0.5)),
                  face=[(0, {"p_anger": 0.8, "p_flush": 1.0}), (14, {"p_anger": 1.0, "p_expr_yell": 1.0, "p_flush": 1.0}), (52, {"p_expr_yell": 0.3, "p_anger": 1.0}), (72, {"p_expr_yell": 0.0, "p_anger": 0.8})],
                  props=["p_anger", "p_flush", "p_expr_yell"], meta=dict(note2="after the slow burn (p_anger/p_flush already high)"))


@action("manager_slow_burn", use="the Manager's slow-burn anger beat (S1 before the bang, S2 intro): controlled to boiling; 150 frames", note="tension builds: shoulders rise, fists clench, tremor grows, head lowers, breath deepens, finally the head-pop stretch and squash at frame 128")
def b_slow_burn(rig):
    N = 150
    az = rig.az
    base = idle_fn(rig, N, nb=5, shift=1.0, blinks=(0.15, 0.4, 0.62), look=0.02, sway=0.012)

    def fn(frame):
        i = int(round(frame)) % N if False else int(round(frame))
        i = min(max(i, 0), N)
        u = i / N
        p = max(0.0, (i - 20) / 108.0)          # 0 at frame 20, 1 at frame 128
        p = min(p, 1.0) ** 1.4
        P = base(frame)
        trem = math.sin(i * 1.9) * 0.6 + math.sin(i * 3.1) * 0.4
        P["hips_loc"] = P["hips_loc"] + V(0.004 * trem * p, 0, -0.02 * p)
        P["spine"] = P["spine"] + V(0.20 * p, 0, 0)
        P["neck"] = P["neck"] + V(0.14 * p, 0, 0)
        P["head"] = P["head"] + V(0.10 * p + 0.008 * trem * p, 0.01 * trem * p, 0)
        P["clav_L"] = P["clav_L"] + V(0.55 * p, 0.0)
        P["clav_R"] = P["clav_R"] + V(0.55 * p, 0.0)
        P["curl_L"] = V(0.2 + 0.8 * p)
        P["curl_R"] = V(0.2 + 0.8 * p)
        for sn, side in (("L", 1), ("R", -1)):
            P["hand_" + sn] = hand((0.28 - 0.03 * p + 0.004 * trem * p, -0.10 - 0.14 * p, 0.84 + 0.04 * p), f=(0.0, -0.3, -0.9), n=(-1, 0, 0), side=side)
            P["elbow_" + sn] = V(0.22 + 0.1 * p, 0.35, -0.3 + 0.1 * p)
        P["jaw"] = V(0.0)
        P["lid_up"] = V(0.30 * p)
        P["sq_c"] = P["sq_c"] * V(1, 1, 1) * 1.0
        # head-pop: squash at 120, stretch up at 128, settle
        if i >= 118:
            a = i - 118
            if a < 6:
                P["sq_c"] = ss(-0.07 * a / 6)
                P["sq_h"] = ss(-0.04 * a / 6)
                P["hips_loc"] = P["hips_loc"] + V(0, 0, -0.06 * a / 6)
            elif a < 14:
                q = (a - 6) / 8
                P["sq_c"] = ss(-0.07 + 0.16 * math.sin(math.pi * 0.5 * q) - 0.05 * q)
                P["sq_h"] = ss(0.12 * math.sin(math.pi * q))
                P["hips_loc"] = P["hips_loc"] + V(0, 0, -0.06 + 0.05 * q)
                P["head"] = P["head"] + V(-0.35 * q, 0, 0)
                P["neck"] = P["neck"] + V(-0.2 * q, 0, 0)
                P["jaw"] = V(0.7 * q)
                P["clav_L"] = P["clav_L"] + V(0.3 * q, 0)
                P["clav_R"] = P["clav_R"] + V(0.3 * q, 0)
            else:
                q = min((a - 14) / 14, 1.0)
                P["sq_c"] = ss(0.08 * (1 - q) * math.cos(q * 9) * 1.0)
                P["head"] = P["head"] + V(-0.35 * (1 - q) + 0.1 * q, 0, 0)
                P["jaw"] = V(0.7 * (1 - 0.5 * q))
        return P
    face = [(0, {"p_anger": 0.0, "p_flush": 0.0}), (20, {"p_anger": 0.05, "p_flush": 0.0}), (60, {"p_anger": 0.35, "p_flush": 0.35}), (100, {"p_anger": 0.7, "p_flush": 0.8}), (118, {"p_anger": 0.9, "p_flush": 1.0}),
            (128, {"p_anger": 1.0, "p_flush": 1.0, "p_expr_yell": 0.8}), (N, {"p_anger": 1.0, "p_flush": 1.0, "p_expr_yell": 0.4})]
    return dict(fn=fn, n=N, loop=False, overlap=dict(head=(7.0, 0.4), neck=(8.0, 0.45)),
                events=[("tremor_start", 40), ("steam_start", 90), ("squash", 124), ("boil_over", 128)], face=face, props=["p_anger", "p_flush", "p_expr_yell"],
                meta=dict(note2="trigger HOOK_steam_L/R (ear steam VFX) at events steam_start and boil_over"), hooks=["HOOK_steam_L", "HOOK_steam_R"])


# ------------------------------------------------------------------------------------------------------------ whiteboard / typing
@action("whiteboard_write", acting="R", use="writing/marking on a whiteboard, board at 0.45 m ahead (marker in HOOK_hand_R); loop 48 frames; `_L` left-handed", note="scribbling zigzag lines left to right with a marker grip, glancing back, other hand on hip, weight shifts")
def b_wb_write(rig):
    N = 48
    base = idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.4,), look=0.0, sway=0.015)

    def fn(frame):
        i = int(round(frame)) % N
        u = i / N
        P = base(frame)
        # zigzag: x sweeps over 24 frames (a line), small fast vertical scribble; second line below
        line = (i // 24)
        t = (i % 24) / 24.0
        xs = 0.40 - 0.50 * t
        zs = 1.50 - 0.10 * line + 0.035 * math.sin(TAU * t * 6)
        ys = -0.40 + 0.01 * math.sin(TAU * t * 6)
        P["hand_L"] = hand((xs, ys, zs), f=(0.1, -0.7, 0.55), n=(-0.4, 0, -0.9))
        P["curl_L"] = curl(index=0.55, middle=0.7, ring=0.8, pinky=0.8, thumb=0.45)
        P["elbow_L"] = V(0.30, 0.25, -0.05)
        P["hand_R"] = hand((0.26, -0.02, 1.02), f=(0.5, -0.4, -0.6), n=(-1, 0.3, 0.2), side=-1)
        P["curl_R"] = curl(index=0.5, middle=0.5, ring=0.5, pinky=0.5, thumb=0.3)
        P["elbow_R"] = V(0.35, 0.2, 0.0)
        P["spine"] = P["spine"] + V(0.10, 0.25 * (0.5 - t) * 0 + 0.2, 0)
        P["head"] = P["head"] + V(0.04, 0.2 + 0.1 * (0.5 - t) * 2 * 0, 0)
        P["hips_rot"] = P["hips_rot"] + V(0, 0.1, 0)
        P["hfol_L"] = V(0.0)
        P["hfol_R"] = V(0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.4), hand_L=(10.0, 0.5)), events=[("line_end", 24), ("line_end2", 0)], hooks=["HOOK_hand_R"],
                meta=dict(board_distance_m=0.45, note2="marker tip about 0.40 m ahead of the root at 1.40-1.50 m height; character faces the board"))


@action("whiteboard_underline", acting="R", use="emphatic underline then tap on the whiteboard (Manager / vendor presenting); `_L` left-handed", note="wind-up, sweeping underline, double tap, step back; 54 frames")
def b_wb_under(rig):
    T = T0(rig)
    marker = dict(f=(0.1, -0.7, 0.55), n=(-0.4, 0, -0.9))
    T.key(8, "out", spine=(0.08, 0.2, 0), head=(0.04, 0.25, 0), hand_L=hand((0.50, -0.34, 1.30), **marker), curl_L=curl(index=0.55, middle=0.7, ring=0.8, pinky=0.8, thumb=0.45), **sq(c=-0.02))
    T.key(20, "smooth", spine=(0.12, 0.05, 0), hand_L=hand((-0.30, -0.40, 1.25), **marker))
    T.key(24, "smooth", hand_L=hand((-0.34, -0.40, 1.27), **marker))
    T.key(30, "in", hand_L=hand((-0.20, -0.42, 1.40), **marker))
    T.key(33, "in", hand_L=hand((-0.20, -0.42, 1.34), **marker))
    T.key(36, "smooth", hand_L=hand((-0.20, -0.42, 1.40), **marker))
    T.key(39, "in", hand_L=hand((-0.20, -0.42, 1.34), **marker))
    T.key(54, "smooth", spine=(0.04, 0, 0), head=(0, 0, 0), hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)), curl_L=curl(), **sq())
    return finish(T, rig, 54, [("underline_start", 12), ("underline_end", 22), ("tap", 33), ("tap2", 39)], overlap=dict(head=(6.0, 0.4), hand_L=(10.0, 0.5)), hooks=["HOOK_hand_R"],
                  meta=dict(board_distance_m=0.45))


@action("typing", use="typing at a keyboard / laptop / terminal about 1.0 m high and 0.40 m ahead; loop 24 frames", note="both hands on a keyboard, finger waves, wrists bob, head down, eyes on screen")
def b_typing(rig):
    N = 24
    base = idle_fn(rig, 96 // 4, nb=1, shift=1.0, blinks=(0.5,), look=0.0, sway=0.01)

    def fn(frame):
        i = int(round(frame)) % N
        a = TAU * i / N
        P = base(frame)
        for sn, side, ph in (("L", 1, 0.0), ("R", -1, math.pi)):
            P["hand_" + sn] = hand((0.11 + 0.01 * math.sin(a * 2 + ph), -0.40 + 0.01 * math.sin(a + ph), 1.01 + 0.012 * math.sin(a * 2 + ph + 1)), f=(0.0, -1, -0.25), n=(-0.1, 0, -1), side=side)
            c = [0.35 + 0.35 * max(0, math.sin(a * 2 + ph + k * 1.3)) for k in range(4)]
            P["curl_" + sn] = V(c[0], c[1], c[2], c[3], 0.3)
            P["elbow_" + sn] = V(0.25, 0.25, -0.15)
        P["spine"] = P["spine"] + V(0.14, 0.05 * math.sin(a), 0)
        P["head"] = P["head"] + V(0.12 + 0.02 * math.sin(a * 2), 0.04 * math.sin(a), 0)
        P["hfol_L"] = V(0.0)
        P["hfol_R"] = V(0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(), meta=dict(note2="hands at 1.0 m height, 0.40 m ahead"))


# ------------------------------------------------------------------------------------------------------------ shotgun
def gun_hands(rp, d, grip_n=(1, 0, 0)):
    """Right grip hand at rp with barrel direction d; left support hand under the forend 0.26 m ahead (left-side coordinates)."""
    d = nrm(V(*d))
    rp = V(*rp)
    lp = rp + 0.26 * d + V(0.10, 0.0, -0.035)
    return dict(
        hand_R=np.concatenate([rp, d, nrm(V(*grip_n))]),
        hand_L=np.concatenate([lp, d, nrm(V(0.0, 0.0, 1.0))]),
        curl_R=curl(index=0.35, middle=0.8, ring=0.8, pinky=0.8, thumb=0.5), curl_L=curl(index=0.5, middle=0.6, ring=0.6, pinky=0.6, thumb=0.4))


AIM_R = (-0.15, -0.30, 1.32)


def aim_pose(d=(0, -1, 0.0), recoil=0.0):
    """Shouldered pose: hands such that the barrel axis goes through both hands; recoil kicks the hands back/up (muzzle flips up)."""
    rp = V(*AIM_R) + V(0, 0.07 * recoil, 0.02 * recoil)
    dd = nrm(V(*d) + V(0, 0, 0.30 * recoil))
    g = gun_hands(rp, dd)
    g["hand_L"][2] += 0.04 * recoil
    return g


@action("gun_ready", use="shotgun carried at low-ready across the body (Manager walking in / waiting); loop 48 frames", note="barrel up at 45 degrees, right grip at the hip, left supports the forend, slight sway")
def b_gun_ready(rig):
    N = 48
    base = idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.4,), look=0.02, sway=0.012)

    def fn(frame):
        i = int(round(frame)) % N
        P = base(frame)
        s = math.sin(TAU * i / N * 2)
        g = gun_hands((-0.16, -0.26 + 0.004 * s, 1.08 + 0.005 * s), (0.05, -0.7, 0.7))
        P.update({k: v for k, v in g.items()})
        P["elbow_R"] = V(0.15, 0.3, -0.2)
        P["elbow_L"] = V(0.25, 0.2, -0.2)
        P["hfol_L"] = V(1.0)
        P["hfol_R"] = V(1.0)
        P["spine"] = P["spine"] + V(0.04, -0.1, 0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.5)), hooks=["HOOK_gun_grip_R", "HOOK_gun_support_L"], meta=dict(muzzle_dir="up-forward 45 degrees (hand R finger direction)"))


@action("gun_aim_hold", use="shouldered shotgun, aiming along the character's forward axis; stable stance, breathing sway; loop 48 frames", note="two-handed, stock in the right shoulder, cheek weld, left foot forward, muzzle level and stable")
def b_gun_aim_hold(rig):
    N = 48
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        s = math.sin(TAU * i / N)
        P = neutral()
        P["hips_loc"] = V(0, 0.01, -0.07)
        P["hips_rot"] = V(0.0, -0.14, 0)
        P["spine"] = V(0.04 + 0.006 * s, -0.20, 0.03)
        P["head"] = V(0.0, 0.12, -0.10)
        P["neck"] = V(0.0, 0.08, 0)
        P["foot_L"] = V(0.16, -0.14, az, 0, -0.1)
        P["foot_R"] = V(0.17, 0.12, az, 0, -0.65)
        P["knee_L"] = V(0.06, -0.55, 0)
        P["knee_R"] = V(0.08, -0.55, 0)
        g = aim_pose()
        g["hand_R"][2] += 0.003 * s
        g["hand_L"][2] += 0.003 * s
        P.update(g)
        P["elbow_R"] = V(0.28, 0.35, 0.10)
        P["elbow_L"] = V(0.15, 0.30, -0.25)
        P["lid_up"] = V(0.25)
        P["blink"] = V(0.0)
        P["gaze"] = V(0.0, 0.0)
        P["clav_R"] = V(0.05, -0.1)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(), hooks=["HOOK_gun_grip_R", "HOOK_gun_support_L"], meta=dict(muzzle_dir="forward (-y), level; shoulder height 1.31 m (baseline)"))


def _aim_key(T, f, ease_, recoil=0.0, **kw):
    az = 0.0707
    g = aim_pose(recoil=recoil)
    T.key(f, ease_, **g, **kw)


@action("gun_fire", use="fire one barrel from gun_aim_hold: anticipation, recoil with weight transfer, recovery to aim (shot at frame 6)", note="3 frames of inhale/anticipation, bang at 6, hands kick back and up, muzzle flips about 17 degrees, torso rocks back and returns; 26 frames")
def b_gun_fire(rig):
    az = rig.az
    ap = b_gun_aim_hold(rig)
    fn0 = ap["fn"]
    base = fn0(0)
    T = Timeline(base)
    T.key(0)
    T.key(4, "out", spine=(0.03, -0.20, 0.03), hips_loc=(0, 0.01, -0.075))
    kick = dict(hips_loc=(0, 0.05, -0.07), spine=(-0.10, -0.20, 0.03), head=(-0.10, 0.12, -0.10), neck=(-0.05, 0.08, 0), clav_R=(0.1, -0.25), jaw=0.0,
                foot_L=V(0.16, -0.12, az, 0, -0.1), foot_R=V(0.17, 0.13, az, 0, -0.65), **sq(c=-0.04), blink=1.0)
    _aim_key(T, 6, "in2", recoil=1.0, **kick)
    _aim_key(T, 9, "out", recoil=0.7, **dict(kick, hips_loc=(0, 0.045, -0.07), blink=0.0))
    _aim_key(T, 14, "back", recoil=0.2, hips_loc=(0, 0.02, -0.07), spine=(0.0, -0.20, 0.03), head=(-0.02, 0.12, -0.10), clav_R=(0.05, -0.1), **sq())
    _aim_key(T, 26, "smooth", recoil=0.0, hips_loc=(0, 0.01, -0.07), spine=(0.04, -0.20, 0.03), head=(0.0, 0.12, -0.10))
    return finish(T, rig, 26, [("shot", 6), ("recoil_peak", 6), ("recovered", 20)], overlap=dict(head=(6.0, 0.4), neck=(7.0, 0.4)), hooks=["HOOK_gun_grip_R", "HOOK_gun_support_L"],
                  meta=dict(muzzle_dir="forward, flips up by about 17 degrees at the shot", muzzle_flash_frame=6),
                  face=[(0, {"p_expr_shouting": 0.0}), (6, {"p_expr_shouting": 0.0})], props=[])


@action("gun_raise_aim_fire", use="Manager raises the shotgun from low-ready, aims, fires one barrel (S1 shot at frame 44, S2 shot); 78 frames", note="dip anticipation, raise with overshoot to the shoulder, aim hold, 3-frame anticipation, bang at 44, recoil with weight transfer, settle in aim")
def b_gun_raise_fire(rig):
    az = rig.az
    T = Timeline()
    ready = gun_hands((-0.16, -0.26, 1.08), (0.05, -0.7, 0.7))
    T.key(0, **stand_keys(rig), **ready, hfol_L=1.0, hfol_R=1.0, elbow_R=V(0.15, 0.3, -0.2), elbow_L=V(0.25, 0.2, -0.2), spine=(0.04, -0.1, 0))
    # anticipation: gun dips, body gathers
    low = gun_hands((-0.18, -0.24, 0.98), (0.05, -0.45, 0.3))
    T.key(6, "smooth", **low, hips_loc=(0, 0.02, -0.05), spine=(0.10, -0.1, 0), head=(0.06, 0, 0), **sq(c=-0.02))
    aimk = aim_pose()
    stance = dict(hips_loc=(0, 0.01, -0.07), hips_rot=(0, -0.14, 0), spine=(0.04, -0.20, 0.03), head=(0.0, 0.12, -0.10), neck=(0, 0.08, 0), lid_up=0.25,
                  foot_L=V(0.16, -0.14, az, 0, -0.1), foot_R=V(0.17, 0.12, az, 0, -0.65), knee_L=V(0.06, -0.55, 0), knee_R=V(0.08, -0.55, 0),
                  elbow_R=V(0.28, 0.35, 0.10), elbow_L=V(0.15, 0.30, -0.25), hfol_L=0.0, hfol_R=0.0, clav_R=(0.05, -0.1))
    over = aim_pose(d=(0, -1, 0.14))
    T.key(15, "smooth", **over, **stance, **sq(c=0.03))                                    # raise (muzzle high)
    T.key(20, "back", **aimk, **stance, **sq())                                           # settle onto the target
    T.key(38, "smooth", **aimk, **stance, **sq())
    T.key(41, "out", **aimk, **dict(stance, spine=(0.035, -0.20, 0.03)), **sq(c=0.015))     # tiny inhale, hold breath
    kick = dict(stance, hips_loc=(0, 0.05, -0.07), spine=(-0.10, -0.20, 0.03), head=(-0.10, 0.12, -0.10), neck=(-0.05, 0.08, 0), clav_R=(0.1, -0.25))
    T.key(44, "in2", **aim_pose(recoil=1.0), **dict(kick, blink=1.0), **sq(c=-0.04))
    T.key(47, "out", **aim_pose(recoil=0.7), **dict(kick, hips_loc=(0, 0.045, -0.07), blink=0.0))
    T.key(52, "back", **aim_pose(recoil=0.2), **dict(stance, hips_loc=(0, 0.02, -0.07), spine=(0.0, -0.20, 0.03)), **sq())
    T.key(64, "smooth", **aimk, **stance)
    T.key(78, "smooth", **aimk, **stance)
    return finish(T, rig, 78, [("raise_start", 6), ("aim_start", 20), ("shot", 44), ("recoil_peak", 44), ("recovered", 58)], overlap=dict(head=(6.0, 0.4), neck=(7.0, 0.4)),
                  hooks=["HOOK_gun_grip_R", "HOOK_gun_support_L"], meta=dict(muzzle_dir="forward (-y), level at 1.31 m baseline; hands flip the muzzle about 17 degrees at the shot", muzzle_flash_frame=44,
                                                                              note2="the gun is mounted on HOOK_gun_grip_R with rot z=pi as in S1/S2; the barrel axis equals the right hand finger direction"),
                  face=[(0, {"p_anger": 0.9}), (78, {"p_anger": 0.9})], props=["p_anger"])


# ------------------------------------------------------------------------------------------------------------ Gary reactions
@action("flinch", use="quick flinch at a bang / near miss (any character)", note="head ducks, shoulders up, arms to the face, hold, relax; 24 frames")
def b_flinch(rig):
    T = T0(rig)
    cover = dict(hand_L=hand((0.12, -0.24, 1.45), f=(0, -0.5, 0.9), n=(-1, 0, 0.2)), hand_R=hand((0.12, -0.24, 1.45), f=(0, -0.5, 0.9), n=(-1, 0, 0.2), side=-1), curl_L=curl(index=0.6, middle=0.6, ring=0.6, pinky=0.6),
                 curl_R=curl(index=0.6, middle=0.6, ring=0.6, pinky=0.6), elbow_L=V(0.25, 0.1, -0.1), elbow_R=V(0.25, 0.1, -0.1))
    T.key(3, "snap", hips_loc=(0, 0.04, -0.12), spine=(0.30, 0.0, 0), neck=(0.2, 0, 0), head=(0.25, 0.2, 0.1), clav_L=(0.45, 0), clav_R=(0.45, 0), blink=1.0, jaw=0.15, **cover, **sq(c=-0.05, h=-0.03))
    T.key(10, "smooth", **cover)
    T.key(24, "smooth", **stand_keys(rig), hips_loc=(0, 0, 0), spine=(0.0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0), clav_L=(0, 0), clav_R=(0, 0), blink=0.0, jaw=0.0, hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl(), **sq())
    return finish(T, rig, 24, [("flinch", 3), ("release", 10)], overlap=dict(head=(6.0, 0.4), hand_L=(8.0, 0.5), hand_R=(8.0, 0.5)),
                  face=[(0, {"p_expr_scared": 0.0}), (3, {"p_expr_scared": 1.0}), (14, {"p_expr_scared": 0.7}), (24, {"p_expr_scared": 0.0})], props=["p_expr_scared"])


@action("startle", use="startle hop at a shot, a shout or an alarm (feet leave the ground)", note="jump with arms flung out, eyes wide, lands with squash and wobble; 36 frames; contact (landing) at frame 14")
def b_startle(rig):
    az = rig.az
    T = T0(rig)
    T.key(3, "out", hips_loc=(0, 0.02, -0.12), spine=(0.08, 0, 0), **sq(c=-0.05, b=-0.03))
    up = dict(hips_loc=(0, 0, 0.0), spine=(-0.18, 0, 0), neck=(-0.15, 0, 0), head=(-0.2, 0, 0), jaw=0.6, foot_L=V(0.12, 0.04, az + 0.12, 0.35, 0.1), foot_R=V(0.12, 0.02, az + 0.14, 0.35, -0.1),
              hand_L=hand((0.60, -0.10, 1.38), f=(0.8, -0.4, 0.3), n=(-0.3, 0, 1)), hand_R=hand((0.60, -0.10, 1.38), f=(0.8, -0.4, 0.3), n=(-0.3, 0, 1), side=-1), curl_L=curl(index=0, middle=0, ring=0, pinky=0, thumb=0),
              curl_R=curl(index=0, middle=0, ring=0, pinky=0, thumb=0), elbow_L=V(0.4, 0.2, 0), elbow_R=V(0.4, 0.2, 0), **sq(c=0.06, s=0.04))
    T.key(8, "out", **up)
    T.key(14, "in", hips_loc=(0, 0, -0.12), spine=(0.10, 0, 0), neck=(0.05, 0, 0), head=(0.1, 0, 0), foot_L=V(0.12, 0.0, az, 0, 0.06), foot_R=V(0.12, 0.0, az, 0, -0.06), **sq(c=-0.07, b=-0.04), jaw=0.3)
    T.key(20, "back", hips_loc=(0, 0, -0.03), spine=(0.0, 0, 0), jaw=0.2, **sq(c=0.02))
    T.key(36, "smooth", hips_loc=(0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0), jaw=0.0, hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl(), **sq())
    return finish(T, rig, 36, [("jump", 4), ("peak", 8), ("land", 14)], overlap=dict(head=(6.0, 0.4), hand_L=(7.0, 0.4), hand_R=(7.0, 0.4)),
                  face=[(0, {"p_expr_shock": 0.0}), (4, {"p_expr_shock": 1.0}), (22, {"p_expr_shock": 0.6}), (36, {"p_expr_shock": 0.0})], props=["p_expr_shock"])


@action("shrug", use="shrug: 'what can I do' (Gary, vendors); beat at frame 12", note="palms up, shoulders to the ears, head tilt, lips pursed, hold, drop; 40 frames")
def b_shrug(rig):
    T = T0(rig)
    up = dict(hand_L=hand((0.45, -0.28, 1.05), f=(0.3, -1, 0.1), n=(-0.1, 0, 1)), hand_R=hand((0.45, -0.28, 1.05), f=(0.3, -1, 0.1), n=(-0.1, 0, 1), side=-1), curl_L=curl(index=0.1, middle=0.15, ring=0.2, pinky=0.25, thumb=0.1),
              curl_R=curl(index=0.1, middle=0.15, ring=0.2, pinky=0.25, thumb=0.1), elbow_L=V(0.30, 0.1, -0.1), elbow_R=V(0.30, 0.1, -0.1))
    T.key(5, "out", hips_loc=(0, 0.01, -0.03), clav_L=(-0.1, 0), clav_R=(-0.1, 0), head=(0.0, 0, 0), spine=(0.03, 0, 0), **sq(c=-0.02))
    T.key(12, "snap", clav_L=(0.55, 0), clav_R=(0.55, 0), head=(-0.04, 0, 0.22), neck=(0, 0, 0.1), jaw=0.0, spine=(-0.03, 0, 0), **up, **sq(c=0.02), lid_up=0.15)
    T.key(28, "smooth", clav_L=(0.52, 0), clav_R=(0.52, 0), head=(-0.04, 0.1, 0.22))
    T.key(40, "smooth", clav_L=(0, 0), clav_R=(0, 0), head=(0, 0, 0), neck=(0, 0, 0), hips_loc=(0, 0, 0), spine=(0, 0, 0), lid_up=0.0, hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl(), **sq())
    return finish(T, rig, 40, [("shrug_peak", 12), ("release", 28)], overlap=dict(head=(6.0, 0.4), hand_L=(8.0, 0.5), hand_R=(8.0, 0.5)),
                  face=[(0, {"p_expr_flat": 0.0}), (10, {"p_expr_flat": 0.8}), (32, {"p_expr_flat": 0.8}), (40, {"p_expr_flat": 0.0})], props=["p_expr_flat"])


@action("facepalm", acting="R", use="facepalm (Gary realises it all went wrong; Manager exasperated); `_L` left hand", note="slaps the hand to the face, head drops into it, hold, slow drag down with a sigh; 70 frames")
def b_facepalm(rig):
    T = T0(rig)
    face = dict(hand_L=hand((0.04, -0.18, 1.55), f=(0.0, -0.3, 1.0), n=(-1, 0.2, 0.0)), curl_L=curl(index=0.2, middle=0.2, ring=0.3, pinky=0.3, thumb=0.2), elbow_L=V(0.25, 0.1, -0.1))
    T.key(7, "out", hips_loc=(0, 0.02, -0.03), spine=(0.05, 0, 0), hand_L=hand((0.20, -0.25, 1.30), f=(0.0, -0.5, 0.9), n=(-1, 0, 0.2)), **sq(c=0.02))
    T.key(11, "in", spine=(0.12, 0, 0), head=(0.20, 0, 0.05), **face, **sq(c=-0.02))   # slap
    T.hold(14)
    T.key(24, "smooth", spine=(0.16, 0, 0), head=(0.32, 0.0, 0.08), neck=(0.1, 0, 0), jaw=0.0, blink=1.0)
    T.key(44, "smooth", spine=(0.16, 0, 0), head=(0.34, 0.0, 0.08), hand_L=hand((0.04, -0.18, 1.52), f=(0.0, -0.3, 1.0), n=(-1, 0.2, 0.0)), **sq(c=0.03))   # sigh
    T.key(56, "smooth", head=(0.26, 0.0, 0.05), hand_L=hand((0.08, -0.20, 1.38), f=(0.0, -0.4, 0.9), n=(-1, 0.2, 0.0)), blink=0.0, **sq(c=-0.01))
    T.key(70, "smooth", hips_loc=(0, 0, 0), spine=(0.04, 0, 0), head=(0.1, 0, 0), neck=(0, 0, 0), hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)), curl_L=curl(), **sq())
    return finish(T, rig, 70, [("slap", 11), ("hold_end", 44), ("sigh", 44)], overlap=dict(head=(6.0, 0.4), hand_L=(8.0, 0.5)),
                  face=[(0, {"p_expr_dead_eyed": 0.0}), (14, {"p_expr_dead_eyed": 0.8}), (56, {"p_expr_dead_eyed": 0.8}), (70, {"p_expr_dead_eyed": 0.0})], props=["p_expr_dead_eyed"])


@action("panic_wiggle", use="Gary's panic: arms wiggle overhead, feet shuffle, wailing; loop 20 frames", note="both hands flutter above the head, knees bounce, head shakes; loop 20 frames")
def b_panic(rig):
    N = 20
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        a = TAU * i / N
        P = neutral()
        P["hips_loc"] = V(0.02 * math.sin(a), 0, -0.07 + 0.04 * abs(math.sin(a)))
        P["hips_rot"] = V(0, 0.15 * math.sin(a), 0)
        P["spine"] = V(0.05, -0.2 * math.sin(a), 0.08 * math.cos(a))
        P["head"] = V(-0.05, 0.25 * math.sin(a * 2), 0.1 * math.sin(a))
        P["jaw"] = V(0.5 + 0.2 * math.sin(a * 2))
        P["foot_L"] = V(0.15, 0.0, az + 0.04 * max(0, math.sin(a)), 0.1, 0.2)
        P["foot_R"] = V(0.15, 0.0, az + 0.04 * max(0, -math.sin(a)), 0.1, -0.2)
        for sn, side, ph in (("L", 1, 0.0), ("R", -1, math.pi)):
            P["hand_" + sn] = hand((0.30 + 0.07 * math.sin(a * 2 + ph), -0.15, 1.82 + 0.06 * math.cos(a * 2 + ph)), f=(0.2 * math.sin(a * 2 + ph), -0.2, 1), n=(-1, 0, 0.2), side=side)
            P["curl_" + sn] = curl(index=0, middle=0, ring=0.1, pinky=0.1, thumb=0)
            P["elbow_" + sn] = V(0.45, 0.1, 0.1)
        P["sq_c"] = ss(0.02 * math.sin(a * 2))
        P["clav_L"] = V(0.3, 0)
        P["clav_R"] = V(0.3, 0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(7.0, 0.4)), face=[(0, {"p_expr_scared": 1.0, "p_expr_sweating": 0.8}), (N, {"p_expr_scared": 1.0, "p_expr_sweating": 0.8})], props=["p_expr_scared", "p_expr_sweating"])


@action("kneel_down", use="Gary kneels to work on the floor (before tie_fibers_kneel)", note="step, lower to one knee, lean over; 26 frames; ends in the tie_fibers_kneel pose")
def b_kneel_down(rig):
    T = T0(rig)
    kp = kneel_pose(rig)
    T.key(8, "out", hips_loc=(0, -0.05, -0.20), spine=(0.30, 0, 0), foot_L=V(0.12, -0.30, rig.az, 0, 0.1), foot_R=V(0.10, 0.10, rig.az + 0.10, -0.6, 0), knee_L=V(0.12, -0.9, 0.0), knee_R=V(0.10, -0.55, 0))
    T.key(26, "out", **kp)
    return finish(T, rig, 26, [("knee_down", 20)], overlap=dict(head=(6.0, 0.4)))


def kneel_pose(rig):
    az = rig.az
    return dict(hips_loc=(0, -0.05, -0.44), hips_rot=(0.0, 0.0, 0.0), spine=(0.75, 0.0, 0), neck=(0.15, 0, 0), head=(0.2, 0.0, 0), look_mix=0.0,
                foot_L=V(0.12, -0.45, az, 0, 0.1), foot_R=V(0.10, 0.38, 0.065, -1.25, 0.0), knee_L=V(0.12, -0.9, 0.0), knee_R=V(0.10, -0.6, -0.3),
                hand_L=hand((0.10, -0.42, 0.36), f=(0, -0.6, -0.8), n=(-1, 0, 0.2)), hand_R=hand((0.10, -0.42, 0.36), f=(0, -0.6, -0.8), n=(-1, 0, 0.2), side=-1),
                curl_L=curl(index=0.6, middle=0.6, ring=0.6, pinky=0.6, thumb=0.4), curl_R=curl(index=0.6, middle=0.6, ring=0.6, pinky=0.6, thumb=0.4))


@action("tie_fibers_kneel", use="Gary kneeling, tying a bundle of fibers/cables: hands-busy loop (S6); 36 frames", note="kneeling, alternating hand loops and tugs, head bobbing, tongue-out concentration; loop 36 frames")
def b_tie_kneel(rig):
    N = 36
    kp = kneel_pose(rig)

    def fn(frame):
        i = int(round(frame)) % N
        a = TAU * i / N
        P = neutral()
        for k, v in kp.items():
            P[k] = np.array(v, float).reshape(P[k].shape) if np.size(v) == P[k].size else np.full(P[k].shape, float(v))
        ca, cb = math.sin(a), math.sin(a + math.pi)
        for sn, side, c in (("L", 1, ca), ("R", -1, cb)):
            P["hand_" + sn] = hand((0.09 - 0.05 * c, -0.42 + 0.06 * c, 0.36 + 0.03 * max(c, 0)), f=(0, -0.6, -0.8), n=(-1, 0, 0.2), side=side)
            P["curl_" + sn] = curl(index=0.6 + 0.2 * c, middle=0.6 + 0.2 * c, ring=0.6 + 0.2 * c, pinky=0.6 + 0.2 * c, thumb=0.4)
        P["head"] = P["head"] + V(0.03 * ca, 0.08 * ca, 0)
        P["spine"] = P["spine"] + V(0.02 * math.sin(2 * a), 0, 0)
        P["blink"] = V(1.0 if (i == 20 or i == 21) else 0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.4)), meta=dict(note2="hand targets at floor level 0.52 m ahead of the root"))


@action("kneel_up", use="Gary gets up from kneeling", note="push up, step, stand; 26 frames")
def b_kneel_up(rig):
    T = Timeline()
    T.key(0, **kneel_pose(rig))
    T.key(10, "out", hips_loc=(0, -0.05, -0.20), spine=(0.35, 0, 0), foot_L=V(0.12, -0.30, rig.az, 0, 0.1), foot_R=V(0.10, 0.10, rig.az + 0.10, -0.6, 0), knee_L=V(0.12, -0.9, 0.0), knee_R=V(0.10, -0.55, 0),
          hand_L=hand((0.20, -0.30, 0.55), f=(0, -0.6, -0.8), n=(-1, 0, 0.2)), hand_R=hand((0.20, -0.30, 0.55), f=(0, -0.6, -0.8), n=(-1, 0, 0.2), side=-1))
    T.key(26, "out", **stand_keys(rig), hips_loc=(0, 0, 0), spine=(0, 0, 0), neck=(0, 0, 0), head=(0, 0, 0), knee_L=V(0.03, -0.55, 0), knee_R=V(0.03, -0.55, 0),
          hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)), hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl())
    return finish(T, rig, 26, [("standing", 26)], overlap=dict(head=(6.0, 0.4)))


@action("tie_fibers_stand", group="upper", use="upper-body layer: hands busy tying / twisting a bundle at belt height while the legs do anything (idle, walk slowly)", note="both hands work a bundle, alternating finger curls, head down; loop 30 frames; keys upper body only")
def b_tie_stand(rig):
    N = 30
    base = idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.5,), look=0.0, sway=0.0)

    def fn(frame):
        i = int(round(frame)) % N
        a = TAU * i / N
        P = base(frame)
        ca, cb = math.sin(a), math.sin(a + math.pi)
        for sn, side, c in (("L", 1, ca), ("R", -1, cb)):
            P["hand_" + sn] = hand((0.08 - 0.04 * c, -0.34 + 0.05 * c, 1.02 + 0.03 * max(c, 0)), f=(-0.2, -0.7, -0.5), n=(-0.3, 0.2, 0.9), side=side)
            P["curl_" + sn] = curl(index=0.6 + 0.2 * c, middle=0.6 + 0.2 * c, ring=0.6 + 0.2 * c, pinky=0.6 + 0.2 * c, thumb=0.4)
            P["elbow_" + sn] = V(0.28, 0.25, -0.2)
            P["hfol_" + sn] = V(1.0)
        P["spine"] = P["spine"] + V(0.16, 0, 0)
        P["head"] = P["head"] + V(0.22 + 0.03 * ca, 0.06 * ca, 0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(), meta=dict(note2="upper layer"))


@action("plug_connector", use="plugging a connector / fibre / cable end into a port at chest height 0.46 m ahead; insert at frame 22, click at 24", note="lines up, wiggles, pushes in with a body lean, click squash, checks with a tug; 60 frames")
def b_plug(rig):
    T = T0(rig)
    hold = lambda y, z, x=0.07: dict(hand_L=hand((x, y, z), f=(0, -1, 0.1), n=(-1, 0, 0.3)), hand_R=hand((x, y, z), f=(0, -1, 0.1), n=(-1, 0, 0.3), side=-1),
                                      curl_L=curl(index=0.5, middle=0.6, ring=0.6, pinky=0.6, thumb=0.4), curl_R=curl(index=0.5, middle=0.6, ring=0.6, pinky=0.6, thumb=0.4),
                                      elbow_L=V(0.28, 0.3, -0.15), elbow_R=V(0.28, 0.3, -0.15))
    T.key(8, "out", spine=(0.12, 0, 0), head=(0.12, 0.0, 0.0), look_mix=0.0, **hold(-0.34, 1.12), **sq(c=-0.02))
    for k, f in enumerate((12, 15, 18)):
        s = 1 if k % 2 else -1
        T.key(f, "smooth", head=(0.12, 0.05 * s, 0.05 * s), **hold(-0.37, 1.14 + 0.012 * s, 0.07 + 0.01 * s))   # line up wiggle
    T.key(22, "in", spine=(0.22, 0, 0), hips_loc=(0, -0.05, -0.04), **hold(-0.44, 1.15), **sq(c=0.03))      # push
    T.key(24, "in2", spine=(0.25, 0, 0), hips_loc=(0, -0.06, -0.05), **hold(-0.46, 1.15), **sq(c=-0.04), jaw=0.2)   # click
    T.hold(27)
    T.key(34, "out", spine=(0.16, 0, 0), hips_loc=(0, 0, -0.02), jaw=0.0, **hold(-0.38, 1.14), **sq())
    T.key(40, "smooth", **hold(-0.44, 1.15))     # tug test
    T.key(43, "out", **hold(-0.40, 1.14))
    T.key(60, "smooth", spine=(0, 0, 0), head=(0, 0, 0), hips_loc=(0, 0, 0), hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl())
    return finish(T, rig, 60, [("insert", 22), ("click", 24), ("tug", 40)], hitstop=[(24, 27)], overlap=dict(head=(6.0, 0.4), hand_L=(9.0, 0.5), hand_R=(9.0, 0.5)),
                  face=[(0, {"p_expr_worried": 0.0}), (8, {"p_expr_worried": 0.5}), (27, {"p_expr_worried": 0.0, "p_expr_happy": 0.6}), (60, {"p_expr_happy": 0.0})], props=["p_expr_worried", "p_expr_happy"],
                  meta=dict(port_xyz_baseline=[0.0, -0.46, 1.15], note2="hands meet at x=+-0.07, y=-0.37..-0.46, z=1.14"))


@action("hug_squashed", use="Gary being hugged by a customer (S6): lifted, squeezed, released; squeeze loop frames 16-48", note="arms pinned and flapping, body squash and stretch with each squeeze, eyes bulging, feet leave the floor, released with a gasp; 72 frames")
def b_hug_squash(rig):
    az = rig.az
    T = T0(rig)
    pin = dict(hand_L=hand((0.36, -0.10, 1.05), f=(0.3, -0.3, 0.4), n=(-0.5, 0, 1)), hand_R=hand((0.36, -0.10, 1.05), f=(0.3, -0.3, 0.4), n=(-0.5, 0, 1), side=-1), curl_L=curl(index=0, middle=0, ring=0, pinky=0, thumb=0),
               curl_R=curl(index=0, middle=0, ring=0, pinky=0, thumb=0), elbow_L=V(0.4, 0.2, -0.1), elbow_R=V(0.4, 0.2, -0.1))
    T.key(5, "out", hips_loc=(0, 0.04, -0.03), spine=(-0.08, 0, 0), head=(-0.05, 0, 0), **sq(c=0.03), jaw=0.2)
    T.key(14, "in", hips_loc=(0, 0.0, 0.0), spine=(-0.05, 0, 0), foot_L=V(0.10, 0.0, az + 0.10, 0.4, 0.06), foot_R=V(0.10, 0.0, az + 0.10, 0.4, -0.06), **pin, **sq(c=-0.12, s=-0.08, h=0.06), jaw=0.5)
    for k in range(4):
        f = 20 + 8 * k
        T.key(f, "out", **sq(c=0.05, s=0.04, h=-0.04), spine=(-0.12, 0, 0), head=(0.08 * (1 - 2 * (k % 2)), 0.15 * (1 - 2 * (k % 2)), 0.1), jaw=0.8,
              hand_L=hand((0.40, -0.06 + 0.1 * (k % 2), 1.10 + 0.14 * (k % 2)), f=(0.3, -0.3, 0.4), n=(-0.5, 0, 1)),
              hand_R=hand((0.40, -0.06 + 0.1 * ((k + 1) % 2), 1.10 + 0.14 * ((k + 1) % 2)), f=(0.3, -0.3, 0.4), n=(-0.5, 0, 1), side=-1), blink=0.0)
        T.key(f + 4, "in", **sq(c=-0.14, s=-0.09, h=0.07), spine=(-0.02, 0, 0), jaw=0.3, head=(0.0, 0.0, 0.0))
    T.key(56, "out", **sq(c=-0.10, s=-0.06, h=0.05))
    T.key(60, "snap", foot_L=V(0.10, 0.0, az, 0, 0.06), foot_R=V(0.10, 0.0, az, 0, -0.06), hips_loc=(0, 0, -0.10), spine=(0.2, 0, 0), head=(0.2, 0, 0), **sq(c=0.06), jaw=0.6)
    T.key(72, "smooth", hips_loc=(0, 0, -0.01), spine=(0.05, 0, 0), head=(0.05, 0, 0), jaw=0.2, **sq(), hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl())
    return finish(T, rig, 72, [("grabbed", 14), ("squeeze1", 24), ("squeeze2", 32), ("squeeze3", 40), ("squeeze4", 48), ("loop_start", 16), ("loop_end", 48), ("released", 60)],
                  overlap=dict(head=(5.5, 0.35), hand_L=(8.0, 0.4), hand_R=(8.0, 0.4)),
                  face=[(0, {"p_expr_shock": 0.0}), (14, {"p_expr_shock": 1.0, "p_expr_scared": 0.5}), (56, {"p_expr_shock": 0.9}), (66, {"p_expr_shock": 0.0, "p_expr_dead_eyed": 0.6}), (72, {"p_expr_dead_eyed": 0.0})], props=["p_expr_shock", "p_expr_scared", "p_expr_dead_eyed"],
                  meta=dict(note2="feet lift 0.10 m by frame 14: hold the hugger's root 0.30 m ahead of Gary; frames 16-48 can be strip-repeated"))


@action("hug_give", use="the hugger (customer) embracing: arms wide, lunge, 4 squeezes, release; pairs with hug_squashed; 72 frames", note="wide arms anticipation, lunge with bear hug at frame 14, squeeze beats at 24/32/40/48, release")
def b_hug_give(rig):
    az = rig.az
    T = T0(rig)
    wide = dict(hand_L=hand((0.62, -0.30, 1.30), f=(0.5, -0.8, 0.2), n=(-1, 0, 0.3)), hand_R=hand((0.62, -0.30, 1.30), f=(0.5, -0.8, 0.2), n=(-1, 0, 0.3), side=-1), curl_L=curl(), curl_R=curl(), elbow_L=V(0.4, 0.1, 0.0), elbow_R=V(0.4, 0.1, 0.0))
    around = dict(hand_L=hand((0.20, -0.52, 1.28), f=(-0.8, -0.4, 0.0), n=(0.0, 0.3, 1)), hand_R=hand((0.20, -0.52, 1.28), f=(-0.8, -0.4, 0.0), n=(0.0, 0.3, 1), side=-1), curl_L=GRIP, curl_R=GRIP, elbow_L=V(0.5, 0.0, 0.0), elbow_R=V(0.5, 0.0, 0.0))
    T.key(7, "out", hips_loc=(0, 0.04, -0.06), spine=(-0.10, 0, 0), head=(-0.1, 0, 0), jaw=0.3, **wide, **sq(c=0.05))
    T.key(14, "in", hips_loc=(0, -0.20, -0.10), spine=(0.30, 0, 0), head=(0.2, 0, 0), foot_L=V(0.12, -0.28, az, 0, 0.1), **around, **sq(c=-0.05, s=0.03))
    for k in range(4):
        f = 20 + 8 * k
        T.key(f, "out", spine=(0.28, 0, 0), **sq(c=0.03), head=(0.15, 0.1 * (1 - 2 * (k % 2)), 0.1 * (1 - 2 * (k % 2))), hips_loc=(0, -0.20, -0.10))
        T.key(f + 4, "in", spine=(0.36, 0, 0), **sq(c=-0.06), hips_loc=(0, -0.22, -0.11))
    T.key(60, "smooth", spine=(0.25, 0, 0), **wide, hips_loc=(0, -0.10, -0.08), **sq())
    T.key(72, "smooth", **stand_keys(rig), spine=(0, 0, 0), head=(0, 0, 0), hips_loc=(0, 0, 0), jaw=0.0, hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), curl_L=curl(), curl_R=curl(), elbow_L=V(0.16, 0.38, -0.31), elbow_R=V(0.16, 0.38, -0.31))
    return finish(T, rig, 72, [("embrace", 14), ("squeeze1", 24), ("squeeze2", 32), ("squeeze3", 40), ("squeeze4", 48), ("release", 60)], overlap=dict(head=(6.0, 0.4)),
                  face=[(0, {"p_expr_happy": 0.0}), (7, {"p_expr_happy": 1.0}), (60, {"p_expr_happy": 1.0}), (72, {"p_expr_happy": 0.0})], props=["p_expr_happy"],
                  meta=dict(root_to_root_m=0.62, note2="roots 0.62 m apart facing each other (hug_target hook on the hugger)"))


@action("whipped", use="Gary hit by the arm whip (S6): whiplash arch, spin, collapse, dazed on the ground; hit at frame 0; 118 frames", note="hit-stop, violent arch and arm fling, 1.5 turns spin about the vertical axis with stagger, buckles to the floor on his side, dazed head wobble; ends sitting dazed")
def b_whipped(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **stand_keys(rig), hfol_L=1.0, hfol_R=1.0, ffol_L=1.0, ffol_R=1.0)
    T.key(1, "cut", **sq(c=-0.1))
    T.hold(HS)
    h = HS
    fling = dict(hand_L=hand((0.65, 0.20, 1.40), f=(0.5, 0.3, 0.7), n=(-0.3, 0, 1)), hand_R=hand((0.65, 0.20, 1.40), f=(0.5, 0.3, 0.7), n=(-0.3, 0, 1), side=-1), curl_L=curl(index=0, middle=0, ring=0, pinky=0, thumb=0),
                 curl_R=curl(index=0, middle=0, ring=0, pinky=0, thumb=0), elbow_L=V(0.4, 0.2, 0.1), elbow_R=V(0.4, 0.2, 0.1))
    T.key(h + 3, "snap", hips_loc=(0, 0.10, -0.04), hips_rot=(0.0, 0.4, 0), spine=(-0.65, 0.5, 0.2), neck=(-0.3, 0.3, 0), head=(-0.4, 0.4, 0.2), jaw=0.7, root_rot=(0, 0.6, 0), foot_L=V(0.12, 0.0, az + 0.05, 0.3, 0.2),
          foot_R=V(0.12, 0.0, az, 0, -0.1), blink=1.0, **fling, **sq(c=0.07, s=0.05))
    # spin about the vertical axis: 540 degrees total, stagger steps (feet lifting alternately)
    for k, (f, yaw) in enumerate(((h + 10, 2.4), (h + 17, 4.6), (h + 24, 6.6), (h + 31, 8.4))):
        s = 1 if k % 2 else -1
        T.key(f, "smooth", root_rot=(0, yaw, 0), hips_loc=(0.04 * s, 0.05, -0.12), spine=(-0.25, 0.3 * s, 0.15 * s), head=(-0.1, -0.3 * s, 0.1 * s), neck=(-0.1, 0, 0), jaw=0.5,
              foot_L=V(0.14, 0.0, az + (0.10 if s > 0 else 0.0), 0.2, 0.2), foot_R=V(0.14, 0.0, az + (0.0 if s > 0 else 0.10), 0.2, -0.2), blink=0.0,
              hand_L=hand((0.62, -0.10 * s, 1.20 + 0.2 * s), f=(0.5, -0.3, 0.7), n=(-0.3, 0, 1)), hand_R=hand((0.62, 0.10 * s, 1.20 - 0.2 * s), f=(0.5, -0.3, 0.7), n=(-0.3, 0, 1), side=-1), **sq())
    # dizzy settle: slow wobble, then buckle: knees give, slump onto the floor to the side
    T.key(h + 38, "out", root_rot=(0, 9.2, 0), hips_loc=(0, 0.05, -0.25), spine=(0.2, 0.2, 0.1), head=(0.15, 0.2, 0.15), jaw=0.3, hand_L=hand((0.45, -0.1, 1.0), f=(0.3, -0.5, -0.5), n=(-0.3, 0, 1)),
          hand_R=hand((0.45, -0.1, 1.0), f=(0.3, -0.5, -0.5), n=(-0.3, 0, 1), side=-1), foot_L=V(0.16, 0.0, az, 0, 0.2), foot_R=V(0.16, 0.0, az, 0, -0.2), knee_L=V(0.14, -0.55, 0), knee_R=V(0.14, -0.55, 0))
    T.key(h + 48, "in", root_rot=(0, 9.4, 0.0), hips_loc=(0, 0.06, -0.5), spine=(0.35, 0.1, 0.25), head=(0.2, 0.0, 0.3), ground=0.5,
          foot_L=V(0.20, 0.10, az, 0, 0.3), foot_R=V(0.20, -0.1, az, 0, -0.4), knee_L=V(0.20, -0.5, 0), knee_R=V(0.20, -0.5, 0))
    # sits dazed: legs splayed, hands in lap, head wobbling
    sit = dict(root_rot=(0, 9.4, 0.0), hips_loc=(0, 0.06, -0.80), spine=(0.30, 0.0, 0.0), head=(0.20, 0.0, 0.0), neck=(0.1, 0, 0), ground=1,
               foot_L=V(0.30, -0.55, az + 0.0, 0.3, 0.5), foot_R=V(0.30, -0.55, az + 0.0, 0.3, -0.5), knee_L=V(0.25, -0.9, 0.0), knee_R=V(0.25, -0.9, 0.0),
               hand_L=hand((0.24, -0.35, 0.30), f=(0.2, -0.8, -0.5), n=(-0.4, 0, 1)), hand_R=hand((0.24, -0.35, 0.30), f=(0.2, -0.8, -0.5), n=(-0.4, 0, 1), side=-1), elbow_L=V(0.3, 0.3, -0.1), elbow_R=V(0.3, 0.3, -0.1))
    T.key(h + 60, "out", **sit, **sq(c=-0.06, b=-0.03))
    for k, f in enumerate((h + 72, h + 84, h + 96, h + 108)):
        s = 1 if k % 2 else -1
        T.key(f, "smooth", head=(0.18 + 0.05 * s, 0.35 * s, 0.25 * s), neck=(0.1, 0.2 * s, 0.1 * s), spine=(0.30, 0.1 * s, 0.1 * s), **sq(c=0.01 * s))
    N = h + 118
    T.key(N, "smooth", head=(0.2, 0, 0), neck=(0.1, 0, 0), spine=(0.30, 0, 0))
    face = [(0, {"p_expr_shock": 0.0}), (HS + 2, {"p_expr_shock": 1.0}), (HS + 30, {"p_expr_shock": 0.5, "p_expr_dead_eyed": 0.4}), (HS + 60, {"p_expr_shock": 0.0, "p_expr_dead_eyed": 1.0}), (N, {"p_expr_dead_eyed": 1.0})]
    return finish(T, rig, N, [("whip_hit", 0), ("arch", HS + 3), ("spin_start", HS + 10), ("spin_end", HS + 38), ("on_floor", HS + 48), ("dazed", HS + 60)],
                  hitstop=[(0, HS)], overlap=dict(head=(4.5, 0.28), neck=(5.5, 0.3), hand_L=(6.0, 0.3), hand_R=(6.0, 0.3)), face=face, props=["p_expr_shock", "p_expr_dead_eyed"], attach=dict(L="root", R="root"),
                  meta=dict(note2="root bone yaw ends at 9.4 rad (about 1.5 turns); hold the strip; ends sitting dazed on the floor (no get-up from this pose in this version)", ends_sitting_dazed=1))


@action("pull_cable", use="S1: Gary heaves on a taut cable held 0.5 m ahead at belt height (STRETCH); loop 36 frames", note="braced split stance, two-handed heave: reach, lean back, strain face, bounce back; loop 36 frames")
def b_pull_cable(rig):
    N = 36
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        a = 0.5 - 0.5 * math.cos(TAU * i / N)       # 0 reach, 1 heave
        P = neutral()
        P["foot_L"] = V(0.17, -0.30, az, 0, 0.1)
        P["foot_R"] = V(0.17, 0.28, az, 0, -0.5)
        P["hips_loc"] = V(0, 0.08 + 0.10 * a, -0.10 - 0.02 * a)
        P["spine"] = V(0.22 - 0.50 * a, 0.08 * (a - 0.5), 0)
        P["neck"] = V(0.05, 0, 0)
        P["head"] = V(0.10 - 0.15 * a, 0, 0.06 * (a - 0.5))
        P["jaw"] = V(0.15 + 0.35 * a)
        P["knee_L"] = V(0.12, -0.55, 0)
        P["knee_R"] = V(0.12, -0.55, 0)
        y = -0.52 + 0.36 * a
        for sn, side in (("L", 1), ("R", -1)):
            P["hand_" + sn] = hand((0.10, y, 1.00 + 0.05 * a), f=(0, -1, -0.2), n=(-1, 0, 0), side=side)
            P["curl_" + sn] = V(0.9)
            P["elbow_" + sn] = V(0.30, 0.25, -0.1)
        P["sq_c"] = ss(0.03 * (1 - a) - 0.02 * a)
        P["clav_L"] = V(0.2 * a, 0.0)
        P["clav_R"] = V(0.2 * a, 0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(7.0, 0.4)), events=[("reach", 0), ("heave_peak", 18)],
                face=[(0, {"p_expr_worried": 0.5}), (N, {"p_expr_worried": 0.5})], props=["p_expr_worried"], meta=dict(cable_height_m=1.0, cable_ahead_m=0.52))


@action("thinking", acting="R", use="S2/S5: chin-stroking thought beat, Gary or Manager; `_L` left hand; loop 60 frames", note="hand to chin, head tilt, eyes up and away, slow stroke, weight shift; loop 60 frames")
def b_thinking(rig):
    N = 60
    base = idle_fn(rig, N, nb=2, shift=1.0, blinks=(0.35, 0.8), look=0.0, sway=0.015)

    def fn(frame):
        i = int(round(frame)) % N
        a = TAU * i / N
        P = base(frame)
        P["hand_L"] = hand((0.07 + 0.005 * math.sin(a * 2), -0.20, 1.44 + 0.01 * math.sin(a * 2)), f=(-0.3, -0.2, 1), n=(-0.8, 0.5, 0))
        P["curl_L"] = curl(index=0.2, middle=0.8, ring=0.8, pinky=0.8, thumb=0.5)
        P["elbow_L"] = V(0.26, -0.10, -0.15)
        P["hand_R"] = hand((0.02, -0.26, 1.16), f=(1, -0.2, 0), n=(0, 0, 1), side=-1)
        P["hfol_L"] = V(1.0)
        P["hfol_R"] = V(1.0)
        P["head"] = P["head"] + V(0.0, -0.25, -0.12)
        P["gaze"] = V(-0.5, 0.35)
        P["spine"] = P["spine"] + V(0.03, 0, 0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.45), hand_L=(8.0, 0.5)), face=[(0, {"p_expr_worried": 0.3}), (N, {"p_expr_worried": 0.3})], props=["p_expr_worried"])
