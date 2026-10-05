"""Hit reactions, falls, ground settle, get-up, whip reaction. Receivers start AT the attacker's impact frame (frame 0 = impact)."""
import math

import numpy as np

import m2_engine as E
from m2_engine import V
from m2_lib import action, neutral, Timeline, hand, ss
from m2_fight import guard_base, gb, sq, finish, HS, FIST, FLAT, GRIP

ROOTFOL = dict(L="root", R="root")


def stand(rig, **kw):
    az = rig.az
    d = dict(foot_L=V(0.10, 0, az, 0, 0.06), foot_R=V(0.10, 0, az, 0, -0.06), hfol_L=0.0, hfol_R=0.0)
    d.update(kw)
    return d


def face_hit(t_hit=0, shock_to=1.0):
    return [(0, {"p_expr_shock": 0.0}), (t_hit + 1, {"p_expr_shock": shock_to}), (t_hit + 12, {"p_expr_shock": 0.4, "p_expr_dead_eyed": 0.5}), (t_hit + 40, {"p_expr_shock": 0.0, "p_expr_dead_eyed": 0.0})]


# ------------------------------------------------------------------------------------------------------------ reactions
@action("react_head_hit", acting="R", use="receiver of haymaker / hook / uppercut (start at the attacker's contact frame); `_L` = punched from the other side", note="3-frame hit-stop with head squash, head and torso snap, knees buckle, one back-step, dazed wobble, recover to guard")
def b_react_head(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **guard_base(rig), **sq(h=-0.0))
    T.key(0 + 1, "cut", **sq(h=-0.07))
    T.hold(HS)
    # snap (authored for a punch that throws the head toward the character's LEFT = +twist)
    T.key(HS + 3, "snap", neck=(-0.1, 0.55, 0), head=(-0.15, 0.85, 0.35), spine=(0.12, 0.45, 0.10), hips_rot=(0.04, 0.1, 0.0), hips_loc=(-0.04, 0.10, -0.14),
          hand_L=hand((0.40, -0.10, 1.30), f=(0.3, -0.8, 0.3), n=(-1, 0, 0.2)), hand_R=hand((0.30, -0.05, 1.10), f=(0.3, -0.5, -0.5), n=(-1, 0, 0), side=-1), curl_L=FLAT, curl_R=FLAT,
          foot_R=V(0.17, 0.30, az, 0, -0.55), jaw=0.35, **sq(h=0.03, c=-0.03), lid_up=-0.1)
    T.key(HS + 9, "out", neck=(0.0, 0.25, 0), head=(0.05, 0.4, 0.15), spine=(0.2, 0.2, 0.05), hips_loc=(-0.02, 0.12, -0.16), foot_L=V(0.16, -0.05, az, 0, -0.05), **sq(), jaw=0.15)
    # dazed wobble
    T.key(HS + 17, "smooth", neck=(0.05, -0.12, 0), head=(0.12, -0.2, -0.15), spine=(0.22, -0.1, 0), hips_loc=(0.02, 0.10, -0.14))
    T.key(HS + 25, "smooth", neck=(0.05, 0.08, 0), head=(0.12, 0.12, 0.1), hips_loc=(0, 0.06, -0.12), jaw=0.0)
    ov = dict(head=(5.0, 0.3), neck=(6.0, 0.35), hand_L=(7.0, 0.4), hand_R=(7.0, 0.4))
    T.key(HS + 32, "smooth", **guard_base(rig), **sq())
    return finish(T, rig, HS + 32, [("impact", 0), ("snap", HS + 3), ("dazed_end", HS + 25)], hitstop=[(0, HS)], overlap=ov, face=face_hit(), props=["p_expr_shock", "p_expr_dead_eyed"],
                  meta=dict(start_at_attacker="contact frame"))


@action("react_slapped", acting="R", use="receiver of slap; `_L` = slapped from the other side", note="head whips sideways with squash, hand to cheek, eyes shut, recover")
def b_react_slap(rig):
    T = Timeline()
    T.key(0, **guard_base(rig))
    T.key(1, "cut", **sq(h=-0.08))
    T.hold(HS)
    T.key(HS + 3, "snap", neck=(0.0, 0.65, 0), head=(0.0, 1.0, 0.45), spine=(0.2, 0.35, 0.1), hips_loc=(-0.02, 0.06, -0.12), jaw=0.4, blink=1.0, **sq(h=0.04))
    T.key(HS + 10, "out", neck=(0.05, 0.4, 0), head=(0.1, 0.55, 0.2), spine=(0.25, 0.2, 0),
          hand_L=hand((0.12, -0.20, 1.48), f=(0, -0.3, 1), n=(-1, 0, 0)),
          curl_L=FLAT, jaw=0.2, blink=0.5, **sq())
    T.key(HS + 22, "smooth", neck=(0.05, 0.15, 0), head=(0.12, 0.25, 0.1), jaw=0.0, blink=0.0)
    T.key(HS + 30, "smooth", **guard_base(rig), **sq())
    return finish(T, rig, HS + 30, [("impact", 0), ("snap", HS + 3), ("hand_to_cheek", HS + 10)], hitstop=[(0, HS)], overlap=dict(head=(5.0, 0.3), neck=(6.0, 0.35)),
                  face=face_hit(0, 0.8), props=["p_expr_shock"])


@action("react_gut_hit", use="receiver of an uppercut to the belly or a low blow", note="doubles over, hands to belly, knees buckle, step back, wheeze")
def b_react_gut(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **guard_base(rig))
    T.key(1, "cut", **sq(c=-0.10, b=-0.03))
    T.hold(HS)
    T.key(HS + 4, "snap", hips_loc=(0, 0.14, -0.30), hips_rot=(0.15, 0, 0), spine=(0.95, 0, 0), neck=(0.15, 0, 0), head=(0.2, 0, 0), jaw=0.5, blink=1.0,
          hand_L=hand((0.12, -0.22, 0.98), f=(-0.4, -0.3, -0.8), n=(0, 0.5, 0.8)), hand_R=hand((0.12, -0.22, 0.98), f=(-0.4, -0.3, -0.8), n=(0, 0.5, 0.8), side=-1),
          foot_L=V(0.15, -0.05, az, 0, 0.1), foot_R=V(0.15, 0.20, az, 0, -0.1), knee_L=V(0.14, -0.55, 0), knee_R=V(0.14, -0.55, 0), curl_L=GRIP, curl_R=GRIP, **sq(c=0.04))
    T.key(HS + 14, "smooth", hips_loc=(0, 0.12, -0.32), spine=(1.0, 0, 0), head=(0.25, 0, 0), jaw=0.35, **sq(c=-0.03))
    T.key(HS + 24, "smooth", hips_loc=(0, 0.10, -0.30), spine=(0.9, 0, 0), jaw=0.45, **sq(c=0.03))
    T.key(HS + 38, "smooth", hips_loc=(0, 0.06, -0.16), spine=(0.4, 0, 0), head=(0.1, 0, 0), jaw=0.1, blink=0.0, **sq())
    return finish(T, rig, HS + 38, [("impact", 0), ("doubled_over", HS + 4), ("wheeze", HS + 24)], hitstop=[(0, HS)], overlap=dict(head=(5.0, 0.3), neck=(6.0, 0.35), hand_L=(7.0, 0.4)),
                  face=[(0, {"p_expr_shock": 0.0}), (HS + 3, {"p_expr_shock": 0.6, "p_expr_sobbing": 0.4}), (HS + 38, {"p_expr_shock": 0.0, "p_expr_sobbing": 0.0})],
                  props=["p_expr_shock", "p_expr_sobbing"])


@action("react_headbutt", use="receiver of head_butt (start at its contact frame)", note="head snaps back, hands to forehead, knees wobble, step back")
def b_react_hb(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **gb(rig, hand_L=hand((0.22, -0.60, 1.40), f=(0, -0.7, -0.6), n=(-1, 0, 0.2)), hand_R=hand((0.22, -0.60, 1.40), f=(0, -0.7, -0.6), n=(-1, 0, 0.2), side=-1)))
    T.key(1, "cut", **sq(h=-0.08))
    T.hold(HS)
    T.key(HS + 4, "snap", hips_loc=(0, 0.12, -0.14), spine=(-0.15, 0, 0), neck=(-0.5, 0, 0), head=(-0.5, 0, 0), jaw=0.3, blink=1.0,
          hand_L=hand((0.08, -0.20, 1.55), f=(0, -0.6, 0.8), n=(-1, 0, 0.4)), hand_R=hand((0.08, -0.20, 1.55), f=(0, -0.6, 0.8), n=(-1, 0, 0.4), side=-1), curl_L=FLAT, curl_R=FLAT, **sq(h=0.04))
    T.key(HS + 12, "out", hips_loc=(0, 0.14, -0.18), spine=(0.1, 0, 0), neck=(0.1, 0, 0), head=(0.3, 0, 0.1), foot_R=V(0.15, 0.28, az, 0, -0.3), **sq())
    T.key(HS + 22, "smooth", hips_loc=(0.04, 0.12, -0.16), spine=(0.15, 0.1, 0.06), head=(0.25, 0.1, -0.2), blink=0.0)
    T.key(HS + 32, "smooth", hips_loc=(-0.04, 0.10, -0.16), spine=(0.15, -0.1, -0.06), head=(0.25, -0.1, 0.2))
    T.key(HS + 40, "smooth", hips_loc=(0, 0.06, -0.12), spine=(0.2, 0, 0), head=(0.1, 0, 0))
    return finish(T, rig, HS + 40, [("impact", 0), ("snap", HS + 4), ("wobble", HS + 22)], hitstop=[(0, HS)], overlap=dict(head=(5.0, 0.3), neck=(6.0, 0.35)),
                  face=[(0, {"p_expr_shock": 0.0}), (HS + 3, {"p_expr_shock": 1.0}), (HS + 14, {"p_expr_dead_eyed": 0.9, "p_expr_shock": 0.2}), (HS + 40, {"p_expr_dead_eyed": 0.0, "p_expr_shock": 0.0})],
                  props=["p_expr_shock", "p_expr_dead_eyed"])


@action("react_shoved", use="receiver of shove (start at the attacker's contact frame): staggers back two steps, arms flail, then catches balance", note="push-back stagger, 2 steps, flailing arms")
def b_react_shoved(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **gb(rig, hand_L=hand((0.17, -0.40, 1.30), f=(0, -1, 0.2), n=(-1, 0, 0))))
    T.key(1, "cut", **sq(c=-0.10))
    T.hold(HS)
    T.key(HS + 4, "snap", hips_loc=(0, 0.12, -0.10), hips_rot=(0.0, 0, 0), spine=(-0.35, 0, 0), neck=(-0.2, 0, 0), head=(-0.25, 0, 0), jaw=0.4,
          hand_L=hand((0.45, -0.05, 1.40), f=(0.3, -0.6, 0.4), n=(-0.3, 0, 1)), hand_R=hand((0.45, -0.05, 1.40), f=(0.3, -0.6, 0.4), n=(-0.3, 0, 1), side=-1), curl_L=FLAT, curl_R=FLAT,
          foot_R=V(0.14, 0.35, az, 0, -0.2), **sq(c=0.04))
    T.key(HS + 9, "out", hips_loc=(0, 0.28, -0.10), spine=(-0.3, 0.2, 0), head=(-0.2, 0.2, 0), hand_L=hand((0.55, 0.10, 1.25), f=(0.3, -0.4, 0.4), n=(-0.3, 0, 1)),
          hand_R=hand((0.40, -0.25, 1.50), f=(0.2, -0.4, 0.8), n=(-0.3, 0, 1), side=-1), foot_L=V(0.14, 0.40, az, 0, 0.1), **sq())
    T.key(HS + 15, "out", hips_loc=(0, 0.34, -0.12), spine=(-0.15, -0.2, 0), head=(-0.1, -0.2, 0), hand_L=hand((0.50, -0.20, 1.55), f=(0.2, -0.3, 0.8), n=(-0.3, 0, 1)),
          hand_R=hand((0.50, 0.05, 1.20), f=(0.3, -0.4, 0.2), n=(-0.3, 0, 1), side=-1), foot_R=V(0.14, 0.46, az, 0, -0.1))
    T.key(HS + 22, "smooth", hips_loc=(0, 0.28, -0.14), spine=(0.1, 0, 0), head=(0.05, 0, 0), foot_L=V(0.14, 0.40, az, 0, 0.1), jaw=0.1)
    T.key(HS + 32, "smooth", **gb(rig, hips_loc=(0, 0.28, -0.12), spine=(0.2, 0, 0), jaw=0.0))
    return finish(T, rig, HS + 32, [("impact", 0), ("step1", HS + 9), ("step2", HS + 15), ("balanced", HS + 22)], hitstop=[(0, HS)],
                  overlap=dict(head=(5.5, 0.35), neck=(6.5, 0.4), hand_L=(7.0, 0.4), hand_R=(7.0, 0.4)),
                  meta=dict(end_offset_back_m=0.28 * rig.K, note2="body shifts about 0.46 m back by feet stepping; keep the strip held, shift the root object after if the scene needs it"),
                  face=face_hit(0, 0.7), props=["p_expr_shock"])


@action("react_chest_bump", use="receiver / second party of chest_bump when the bump is one-sided", note="knocked back by the chest, arms wing out, rebound")
def b_react_bump(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **guard_base(rig))
    T.key(1, "cut", **sq(c=-0.10))
    T.hold(HS)
    T.key(HS + 5, "snap", hips_loc=(0, 0.18, -0.08), spine=(-0.25, 0, 0), head=(-0.2, 0, 0), jaw=0.3,
          hand_L=hand((0.45, 0.05, 1.20), f=(0.5, -0.3, 0.3), n=(-1, 0, 0)), hand_R=hand((0.45, 0.05, 1.20), f=(0.5, -0.3, 0.3), n=(-1, 0, 0), side=-1), curl_L=FLAT, curl_R=FLAT,
          foot_R=V(0.15, 0.30, az, 0, -0.3), **sq(c=0.05))
    T.key(HS + 14, "smooth", hips_loc=(0, 0.20, -0.12), spine=(0.1, 0, 0), head=(0.05, 0, 0), jaw=0.1, **sq())
    T.key(HS + 24, "smooth", **guard_base(rig))
    return finish(T, rig, HS + 24, [("impact", 0), ("recoil", HS + 5)], hitstop=[(0, HS)], overlap=dict(head=(5.5, 0.35), hand_L=(7.0, 0.4), hand_R=(7.0, 0.4)))


@action("collar_shaken", use="receiver of collar_grab_shake: ragdoll shake loop (24 frames)", note="head and arms flop while being shaken by the collar; loop 24 frames")
def b_collar_shaken(rig):
    N = 24
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        a = 2 * math.pi * i / N * 3
        s, c = math.sin(a), math.cos(a)
        P = neutral()
        P["hips_loc"] = V(0.02 * s, 0.06, -0.08)
        P["spine"] = V(-0.1 + 0.08 * c, 0.10 * s, 0.06 * s)
        P["neck"] = V(0.1 * c, 0.2 * s, 0.1 * s)
        P["head"] = V(-0.1 + 0.25 * c, 0.3 * s, 0.2 * s)
        P["jaw"] = V(0.35 + 0.2 * c)
        P["foot_L"] = V(0.12, 0.04, az, 0, 0.1)
        P["foot_R"] = V(0.12, 0.04, az, 0, -0.1)
        P["hand_L"] = hand((0.40 + 0.05 * s, -0.10 + 0.08 * c, 1.0 + 0.08 * c), f=(0.2, -0.3, -0.9), n=(-1, 0, 0))
        P["hand_R"] = hand((0.40 - 0.05 * s, -0.10 - 0.08 * c, 1.0 + 0.08 * c), f=(0.2, -0.3, -0.9), n=(-1, 0, 0), side=-1)
        P["curl_L"] = V(0.0)
        P["curl_R"] = V(0.0)
        P["sq_c"] = ss(0.02 * c)
        P["blink"] = V(1.0 if (i % 12) < 3 else 0.0)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(5.0, 0.3), neck=(6.0, 0.3), hand_L=(6.0, 0.3), hand_R=(6.0, 0.3)),
                face=[(0, {"p_expr_scared": 0.9}), (N, {"p_expr_scared": 0.9})], props=["p_expr_scared"], meta=dict(note2="partner of collar_grab_shake; neck height about 1.40 m should sit at the attacker's grab point"))


# ------------------------------------------------------------------------------------------------------------ falls
def build_fall(rig, flavor="back", from_guard=False):
    """Stagger, tip, hit the ground, bounce, flop, settle, tiny twitch. flavor: back | shot."""
    az = rig.az
    shot = flavor == "shot"
    T = Timeline()
    base = guard_base(rig) if from_guard else stand(rig)
    base.update(hfol_L=1.0, hfol_R=1.0)
    T.key(0, **base, **sq(), ground=0, ffol_L=0, ffol_R=0)
    T.key(1, "cut", **sq(c=-0.14 if shot else -0.08, b=-0.03))
    T.hold(HS + (1 if shot else 0))
    h = HS + (1 if shot else 0)
    py = 0.55 if shot else 0.36
    # fling: chest caves, head and arms whip back
    T.key(h + 3, "snap", hips_loc=(0, 0.14, -0.10), spine=(-0.55 if shot else -0.4, 0, 0), neck=(-0.35, 0, 0), head=(-0.4, 0, 0), jaw=0.5,
          hand_L=hand((0.50, 0.20, 1.55), f=(0.4, 0.3, 0.8), n=(-0.3, 0, 1)), hand_R=hand((0.50, 0.20, 1.55), f=(0.4, 0.3, 0.8), n=(-0.3, 0, 1), side=-1), curl_L=FLAT, curl_R=FLAT,
          foot_R=V(0.14, 0.30, az, 0, -0.1), **sq(c=0.05))
    if not shot:
        T.key(h + 9, "out", hips_loc=(0, 0.20, -0.14), spine=(-0.3, 0.1, 0), head=(-0.2, 0.2, 0), hand_L=hand((0.62, 0.0, 1.30), f=(0.4, -0.3, 0.6), n=(-0.3, 0, 1)),
              hand_R=hand((0.30, -0.30, 1.60), f=(0.2, -0.4, 0.9), n=(-0.3, 0, 1), side=-1), foot_L=V(0.14, 0.36, az, 0, 0.1))
        T.key(h + 15, "out", hips_loc=(0, 0.30, -0.22), spine=(-0.35, -0.1, 0), head=(-0.25, -0.2, 0), hand_L=hand((0.35, -0.10, 1.65), f=(0.2, -0.3, 0.9), n=(-0.3, 0, 1)),
              hand_R=hand((0.62, 0.10, 1.28), f=(0.4, -0.2, 0.5), n=(-0.3, 0, 1), side=-1), foot_R=V(0.14, 0.46, az, 0, -0.1), jaw=0.6)
        t_tip = h + 21
        T.key(t_tip, "in", hips_loc=(0, 0.30, -0.22), root_loc=(0, py, 0), root_rot=(-0.40, 0, 0), spine=(-0.25, 0, 0), head=(-0.2, 0.1, 0), ground=0.5,
              foot_L=V(0.14, py - 0.06, az, 0, 0.1), foot_R=V(0.14, py + 0.08, az, 0, -0.1))
        t_hit = h + 31
    else:
        # launched: the body leaves the floor
        t_tip = h + 8
        T.key(h + 8, "in", hips_loc=(0, 0.25, -0.15), root_loc=(0, 0.25, 0.10), root_rot=(-0.55, 0, 0), spine=(-0.3, 0, 0), head=(-0.3, 0, 0), ground=0.5,
              hand_L=hand((0.62, 0.30, 1.50), f=(0.4, 0.2, 0.8), n=(-0.3, 0, 1)), hand_R=hand((0.62, 0.30, 1.50), f=(0.4, 0.2, 0.8), n=(-0.3, 0, 1), side=-1),
              foot_L=V(0.14, 0.30, az + 0.12, 0.3, 0.1), foot_R=V(0.14, 0.20, az + 0.18, 0.4, -0.1), ffol_L=0.5, ffol_R=0.5)
        t_hit = h + 18
    # land: back/butt hits the ground (clamped), squash, arms fly up
    T.key(t_hit, "in2", root_loc=(0, py + (0.25 if shot else 0.0), 0), root_rot=(-1.40, 0, 0), hips_loc=(0, 0, 0), spine=(-0.05, 0, 0), neck=(-0.3, 0, 0), head=(-0.3, 0, 0), ground=1,
          hand_L=hand((0.60, 0.20, 1.55), f=(0.4, 0.2, 0.9), n=(-0.3, 0, 1)), hand_R=hand((0.60, 0.20, 1.55), f=(0.4, 0.2, 0.9), n=(-0.3, 0, 1), side=-1),
          foot_L=V(0.14, 0.0, az + 0.30, 0.2, 0.1), foot_R=V(0.14, 0.0, az + 0.40, 0.3, -0.1), ffol_L=1, ffol_R=1, jaw=0.5, **sq(c=-0.07, b=-0.04))
    # bounce
    T.key(t_hit + 4, "out", root_rot=(-1.25, 0, 0), ground=0.5, root_loc=(0, py + (0.30 if shot else 0.04), 0.07), spine=(0.1, 0, 0), neck=(0.1, 0, 0), head=(0.2, 0, 0), jaw=0.2,
          hand_L=hand((0.66, 0.0, 1.35), f=(0.4, 0.0, 0.7), n=(-0.3, 0, 1)), hand_R=hand((0.66, 0.0, 1.35), f=(0.4, 0.0, 0.7), n=(-0.3, 0, 1), side=-1), **sq(c=0.03))
    T.key(t_hit + 9, "in", root_rot=(-1.47, 0, 0), ground=1, root_loc=(0, py + (0.34 if shot else 0.06), 0), spine=(-0.08, 0, 0), neck=(-0.2, 0, 0), head=(-0.25, 0.2, 0.1), jaw=0.15,
          foot_L=V(0.14, 0.0, az + 0.45, 0.4, 0.1), foot_R=V(0.14, 0.0, az + 0.30, 0.3, -0.1), **sq(c=-0.03))
    # flop: arms out, knees fall sideways, head turns
    T.key(t_hit + 18, "out", hand_L=hand((0.68, 0.10, 1.25), f=(1, -0.2, 0), n=(-0.3, 0, 1)), hand_R=hand((0.68, 0.10, 1.25), f=(1, -0.2, 0), n=(-0.3, 0, 1), side=-1),
          foot_L=V(0.30, 0.0, az + 0.22, 0.4, 0.3), foot_R=V(0.22, 0.0, az + 0.36, 0.3, -0.3), head=(-0.15, 0.55, 0.2), neck=(-0.1, 0.2, 0), spine=(-0.02, 0, 0), jaw=0.2, **sq())
    T.key(t_hit + 30, "smooth", foot_L=V(0.40, 0.0, az + 0.10, 0.4, 0.4), foot_R=V(0.18, 0.0, az + 0.25, 0.3, -0.2), jaw=0.1, curl_L=0.12, curl_R=0.12)
    t_twitch = t_hit + 44
    T.key(t_twitch - 4, "smooth")
    T.key(t_twitch, "snap", foot_L=V(0.40, 0.0, az + 0.15, 0.5, 0.4), hand_R=hand((0.66, 0.10, 1.30), f=(1, -0.2, 0.2), n=(-0.3, 0, 1), side=-1), head=(-0.15, 0.50, 0.2), curl_R=0.5)
    T.key(t_twitch + 6, "smooth", foot_L=V(0.40, 0.0, az + 0.10, 0.4, 0.4), hand_R=hand((0.68, 0.10, 1.25), f=(1, -0.2, 0), n=(-0.3, 0, 1), side=-1), curl_R=0.12)
    N = t_twitch + 26
    T.key(N, "smooth")
    face = [(0, {"p_expr_shock": 0.0}), (2, {"p_expr_shock": 1.0}), (t_hit, {"p_expr_shock": 0.5, "p_expr_dead_eyed": 0.3}), (t_hit + 20, {"p_expr_shock": 0.0, "p_expr_dead_eyed": 0.9}), (N, {"p_expr_dead_eyed": 0.9})]
    ev = [("impact", 0), ("hit_fling", h + 3), ("tip", t_tip), ("ground_hit", t_hit), ("bounce", t_hit + 4), ("settle", t_hit + 30), ("twitch", t_twitch)]
    return finish(T, rig, N, ev, hitstop=[(0, h)], overlap=dict(head=(4.5, 0.28), neck=(5.5, 0.3), hand_L=(6.0, 0.3), hand_R=(6.0, 0.3), spine=(8.0, 0.4)),
                  attach=ROOTFOL, face=face, props=["p_expr_shock", "p_expr_dead_eyed"],
                  meta=dict(ends_lying_on_back=1, end_root_loc_y_m=(py + (0.34 if shot else 0.06)) * rig.K, ground_frame=t_hit, note2="end pose: lying on the back, head toward +y (behind), feet toward the start position; hold forward"),
                  group=None)


@action("fall_back", use="generic knockdown (fight loser, any character): stagger, tip, ground hit, bounce, flop, settle, twitch", note="start at the impact frame; ends lying on the back, hold forward")
def b_fall_back(rig):
    return build_fall(rig, "back", from_guard=False)


@action("fall_back_brawl", use="knockdown starting from the brawl guard stance (S3)", note="same as fall_back, starts from fight guard")
def b_fall_back_brawl(rig):
    return build_fall(rig, "back", from_guard=True)


@action("shot_hit_fall", use="shotgun hit (S1 Gary, S2 Gary): hit-stop at the blast, flung back, ground hit, bounce, ragdoll settle, twitch; sync frame 0 to the blast frame", note="launched backward off the floor, lands on the back and slides, flop and twitch (events twitch)")
def b_shot_fall(rig):
    return build_fall(rig, "shot", from_guard=False)


@action("ground_twitch", group="full", use="small twitch loop for a character lying on his back after a fall (after fall_back / shot_hit_fall)", note="finger and foot twitch, breathing; use only after a lying end pose; loop 60 frames")
def b_ground_twitch(rig):
    # the lying pose = last frame of a fall: reproduce it from the fall timeline
    sp = build_fall(rig, "back")
    T = sp["timeline"]
    last = T.keys[-1][1]
    N = 60
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        P = E.copy_pose(last)
        b = math.sin(2 * math.pi * i / N * 2)
        tw = math.exp(-((i - 20) / 2.0) ** 2) + 0.8 * math.exp(-((i - 44) / 1.5) ** 2)
        P["sq_c"] = ss(0.01 * b)
        P["curl_R"] = V(0.12 + 0.5 * tw)
        P["foot_L"] = P["foot_L"] + V(0, 0, 0.03 * tw, 0.15 * tw, 0)
        P["head"] = P["head"] + V(0, 0.03 * tw, 0)
        P["jaw"] = V(0.05 + 0.1 * tw)
        return P
    return dict(fn=fn, n=N, loop=True, attach=ROOTFOL, overlap=dict(), face=[(0, {"p_expr_dead_eyed": 0.9}), (N, {"p_expr_dead_eyed": 0.9})], props=["p_expr_dead_eyed"],
                meta=dict(requires_end_pose_of="fall_back / shot_hit_fall", end_root_loc_y_m=sp["meta"]["end_root_loc_y_m"]), events=[("twitch", 20), ("twitch2", 44)])


@action("get_up", use="from lying on the back (after fall_back/shot_hit_fall/whipped) to standing: sit up, push, squat, stand, brush off", note="starts in the fall's end pose; ends in neutral stance; root offset of the fall must be applied to the object (see manifest end_root_loc_y_m)")
def b_get_up(rig):
    az = rig.az
    sp = build_fall(rig, "back")
    last = sp["timeline"].keys[-1][1]
    py = float(last["root_loc"][1])
    T = Timeline(last)
    T.key(0)
    # sit up (torso rolls forward relative to the pelvis), hands push behind
    T.key(10, "out", spine=(0.85, 0, 0), neck=(0.2, 0, 0), head=(0.25, 0, 0), jaw=0.3, root_rot=(-1.45, 0, 0), hfol_L=0.0, hfol_R=0.0,
          hand_L=hand((0.30, py + 0.25, 0.04), f=(0, -1, 0.1), n=(-1, 0, 0.9)), hand_R=hand((0.30, py + 0.25, 0.04), f=(0, -1, 0.1), n=(-1, 0, 0.9), side=-1), elbow_L=V(0.3, 0.4, 0.1), elbow_R=V(0.3, 0.4, 0.1),
          foot_L=V(0.14, 0.0, az + 0.45, 0.4, 0.2), foot_R=V(0.14, 0.0, az + 0.40, 0.3, -0.2), **sq(c=-0.03))
    T.key(18, "smooth", spine=(1.30, 0, 0), head=(0.1, 0, 0), neck=(0.05, 0, 0), root_rot=(-1.30, 0, 0), jaw=0.1, foot_L=V(0.20, 0.0, az + 0.30, 0.4, 0.2), foot_R=V(0.20, 0.0, az + 0.28, 0.3, -0.2))
    # feet plant, hips lift to a squat
    T.key(28, "out", ground=0.5, root_loc=(0, py, 0), root_rot=(-0.55, 0, 0), ffol_L=0.0, ffol_R=0.0, spine=(0.9, 0, 0), hips_loc=(0, 0.0, -0.30),
          foot_L=V(0.18, py - 0.20, az, 0, 0.2), foot_R=V(0.18, py - 0.18, az, 0, -0.2), knee_L=V(0.12, -0.55, 0), knee_R=V(0.12, -0.55, 0),
          hand_L=hand((0.30, py - 0.30, 0.55), f=(0, -1, -0.4), n=(-1, 0, 0.5)), hand_R=hand((0.30, py - 0.30, 0.55), f=(0, -1, -0.4), n=(-1, 0, 0.5), side=-1))
    T.key(38, "out", ground=0, root_loc=(0, py * 0.2, 0), root_rot=(-0.1, 0, 0), hips_loc=(0, 0.1, -0.30), spine=(0.55, 0, 0), head=(0.15, 0, 0),
          foot_L=V(0.14, py * 0.2 - 0.1, az, 0, 0.1), foot_R=V(0.14, py * 0.2 - 0.05, az, 0, -0.1),
          hand_L=hand((0.28, -0.25, 0.70), f=(0, -1, -0.4), n=(-1, 0, 0.5)), hand_R=hand((0.28, -0.25, 0.70), f=(0, -1, -0.4), n=(-1, 0, 0.5), side=-1))
    T.key(48, "back", root_loc=(0, 0, 0), root_rot=(0, 0, 0), hips_loc=(0, 0.02, -0.05), spine=(0.1, 0, 0), head=(0.0, 0, 0), neck=(0, 0, 0), jaw=0.0,
          foot_L=V(0.10, 0.0, az, 0, 0.06), foot_R=V(0.10, 0.0, az, 0, -0.06), hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
          hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1), **sq())
    # brush off: dust off the chest with one hand, shake head, glance
    T.key(60, "smooth", hips_loc=(0, 0, -0.01), head=(0.1, 0.25, 0), hand_L=hand((0.10, -0.25, 1.20), f=(0, -0.4, 1), n=(-1, 0, 0.3)), curl_L=FLAT)
    T.key(66, "smooth", head=(0.05, -0.15, 0), hand_L=hand((-0.05, -0.28, 1.15), f=(-0.3, -0.4, 1), n=(-1, 0, 0.3)))
    T.key(72, "smooth", head=(0.0, 0, 0), hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)), curl_L=0.18)
    N = 78
    T.key(N, "smooth")
    face = [(0, {"p_expr_dead_eyed": 0.9}), (14, {"p_expr_dead_eyed": 0.0, "p_expr_worried": 0.7}), (60, {"p_expr_worried": 0.7}), (N, {"p_expr_worried": 0.0})]
    # use root-follow for hands only while lying (first keys); the timeline switches off follow at key 10
    return finish(T, rig, N, [("sit_up", 10), ("stand", 48), ("brush_off", 60)], overlap=dict(head=(5.5, 0.35), neck=(6.5, 0.4)), face=face, props=["p_expr_dead_eyed", "p_expr_worried"],
                  attach=ROOTFOL, meta=dict(starts_from="lying end pose of fall_back", start_root_loc_y_m=py * rig.K, note2="the strip begins with the body on the ground, root bone offset equals the fall's end offset; ends at root offset 0 (stand where the fall started)"))
