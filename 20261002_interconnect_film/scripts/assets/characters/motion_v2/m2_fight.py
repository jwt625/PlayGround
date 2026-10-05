"""Fight set (family-agnostic, by bone name): attacks, reactions, falls.

Conventions: character faces -y. Attacks are authored with the LEFT side acting (left coordinates) where asymmetric; the exported
action `<name>` is the mirror (right hand acts) and `<name>_L` the unmirrored left variant. Every attack has `contact` (impact frame, the
frame the receiver's reaction must start) followed by a 3-frame hit-stop hold. Reactions start AT the impact frame of the attacker
(their frame 0 = pose at impact) with their own 3-frame hit-stop. Distances (baseline m, multiply by K): root_to_root_m.
"""
import math

import numpy as np

import m2_engine as E
from m2_engine import V, nrm
from m2_lib import action, neutral, Timeline, hand, ss

HS = 3          # hit-stop frames
FIST = (1.0, 1.0, 1.0, 1.0, 0.8)
FLAT = (0.0, 0.0, 0.0, 0.0, 0.0)
GRIP = (0.9, 0.9, 0.9, 0.9, 0.7)


def az_of(rig):
    return rig.az


def guard_base(rig, lead="L"):
    az = rig.az
    g = dict(
        hips_loc=(0, 0.04, -0.11), hips_rot=(0.04, -0.26, 0), spine=(0.20, -0.22, 0), neck=(0.02, 0.10, 0), head=(0.10, 0.10, 0), jaw=0.05,
        foot_L=V(0.16, -0.20, az, 0, -0.05), foot_R=V(0.17, 0.22, az, 0, -0.55),
        knee_L=V(0.08, -0.55, 0), knee_R=V(0.10, -0.55, 0),
        hand_L=hand((0.17, -0.40, 1.36), f=(0, -1, 0.45), n=(-1, 0, 0)),
        hand_R=hand((0.03, -0.30, 1.30), f=(0, -1, 0.55), n=(-1, 0, 0), side=-1),
        elbow_L=V(0.22, 0.20, -0.22), elbow_R=V(0.18, 0.20, -0.22),
        curl_L=FIST, curl_R=FIST, hfol_L=0.0, hfol_R=0.0, look_mix=0.0,
    )
    return g


def gb(rig, **over):
    d = guard_base(rig)
    d.update(over)
    return d


def start(rig, T=None, **over):
    g = guard_base(rig)
    g.update(over)
    T = T or Timeline()
    T.key(0, **g)
    return T


def sq(c=0.0, s=0.0, h=0.0, b=0.0):
    return dict(sq_c=E.V(*ss(c)), sq_s=E.V(*ss(s)), sq_h=E.V(*ss(h)), sq_b=E.V(*ss(b)))


def finish(T, rig, N, ev, hitstop=None, overlap=None, meta=None, loop=False, group=None, **extra):
    spec = dict(timeline=T, n=N, loop=loop, events=ev, hitstop=hitstop or [], overlap=overlap or dict(head=(6.5, 0.4), neck=(8.0, 0.45)), meta=meta or {})
    spec.update(extra)
    return spec


def recover_to_guard(T, rig, f, ease_="smooth"):
    g = guard_base(rig)
    T.key(f, ease_, **g, **sq())


# ------------------------------------------------------------------------------------------------------------ attacks
@action("fight_idle", use="S3 brawl: standing guard bounce between attacks", note="loose brawl stance, fists up, bounce; loop 24 frames")
def b_fight_idle(rig):
    N = 24

    def fn(frame):
        i = int(round(frame)) % N
        u = i / N
        P = neutral()
        for k, v in guard_base(rig).items():
            P[k] = np.array(v, float).reshape(P[k].shape) if np.size(v) == P[k].size else np.full(P[k].shape, float(v))
        b = math.sin(2 * math.pi * u * 2)
        P["hips_loc"] = P["hips_loc"] + V(0.02 * math.sin(2 * math.pi * u), 0, 0.018 * b)
        P["spine"] = P["spine"] + V(0.02 * b, 0.05 * math.sin(2 * math.pi * u), 0)
        P["head"] = P["head"] + V(-0.03 * b, 0, 0)
        P["sq_c"] = ss(0.012 * b)
        P["hand_L"] = P["hand_L"] + np.array([0.01 * math.sin(2 * math.pi * u * 2 + 1), 0.015 * b, 0.012 * b] + [0] * 6)
        P["hand_R"] = P["hand_R"] + np.array([0.0, -0.015 * b, 0.01 * b] + [0] * 6)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(7.0, 0.5)), meta=dict(note2="gloves-up bounce"))


@action("haymaker", acting="R", use="S3: wild looping punch (vendor brawl); receiver: react_head_hit", note="big wind-up, wide arc, impact at frame 15 with 3-frame hit-stop, off-balance follow-through")
def b_haymaker(rig):
    az = rig.az
    T = start(rig)
    # wind-up: acting (left) shoulder back, weight to the back foot, fist low behind
    T.key(7, "out", hips_loc=(0.03, 0.12, -0.17), hips_rot=(0.05, 0.28, 0.0), spine=(0.12, 0.85, 0.05), head=(0.12, -0.45, 0), neck=(0, -0.15, 0),
          hand_L=hand((0.62, 0.30, 1.22), f=(0.5, 0.6, 0.3), n=(-0.3, 0, 1.0)), elbow_L=V(0.5, 0.30, 0.05), curl_L=FIST, **sq(c=-0.04))
    T.key(10, "smooth", spine=(0.12, 0.95, 0.05), hand_L=hand((0.66, 0.38, 1.28), f=(0.5, 0.6, 0.3), n=(-0.3, 0, 1.0)), **sq(c=-0.06))   # anticipation hold-ish
    # strike: arc through the front, torso uncoils
    T.key(13, "in", hips_loc=(-0.02, -0.05, -0.12), hips_rot=(0.05, -0.20, 0), spine=(0.20, -0.10, 0), head=(0.1, -0.1, 0), hand_L=hand((0.55, -0.35, 1.48), f=(0, -1, 0.1), n=(-1, 0, 0.5)),
          elbow_L=V(0.5, 0.0, 0.1), foot_L=V(0.20, -0.30, az, 0, -0.05), **sq(c=0.03, s=0.03))
    T.key(15, "in2", hips_loc=(-0.04, -0.14, -0.12), hips_rot=(0.06, -0.55, 0), spine=(0.28, -0.85, -0.05), head=(0.08, 0.35, 0), neck=(0, 0.1, 0),
          hand_L=hand((0.12, -0.84, 1.46), f=(0, -1, 0.0), n=(-1, 0, 0.0)), elbow_L=V(0.35, -0.1, 0.1), **sq(c=0.07, s=0.05))
    T.hold(15 + HS)
    # follow-through and over-rotation
    T.key(25, "out", hips_loc=(-0.06, -0.20, -0.14), hips_rot=(0.08, -0.80, 0), spine=(0.35, -1.10, -0.08), head=(0.12, 0.55, 0), hand_L=hand((-0.30, -0.60, 1.30), f=(-0.4, -1, 0), n=(-1, 0, 0.2)),
          foot_R=V(0.15, 0.10, az, 0, -0.9), **sq(c=-0.02, s=-0.02))
    recover_to_guard(T, rig, 44)
    return finish(T, rig, 44, [("wind_peak", 10), ("contact", 15), ("impact", 15), ("recover", 25)], hitstop=[(15, 15 + HS)],
                  overlap=dict(head=(6.0, 0.4), neck=(7.0, 0.45), hand_R=(8.0, 0.5)),
                  meta=dict(contact_frame=15, root_to_root_m=0.92, target="head/cheek of the receiver at about 1.45 m (baseline), slightly to the actor's side"),
                  hooks=[])


@action("hook", acting="R", use="S3: short hook to the head; receiver react_head_hit", note="tight horizontal hook, elbow high, impact at frame 11")
def b_hook(rig):
    az = rig.az
    T = start(rig)
    T.key(5, "out", hips_loc=(0.0, 0.07, -0.14), hips_rot=(0.05, 0.05, 0), spine=(0.22, 0.45, 0), head=(0.1, -0.25, 0),
          hand_L=hand((0.33, -0.18, 1.34), f=(0.4, -0.7, 0.2), n=(-1, 0, 0.4)), elbow_L=V(0.45, 0.1, 0.05), **sq(c=-0.04))
    T.key(9, "in", hips_loc=(-0.02, -0.05, -0.12), hips_rot=(0.06, -0.35, 0), spine=(0.25, -0.35, 0), head=(0.1, 0.1, 0),
          hand_L=hand((0.30, -0.60, 1.43), f=(-0.3, -0.9, 0.1), n=(-1, 0, 0.2)), foot_R=V(0.17, 0.22, az, 0, -0.85), **sq(c=0.04, s=0.03))
    T.key(11, "in2", hips_loc=(-0.03, -0.10, -0.12), hips_rot=(0.06, -0.55, 0), spine=(0.28, -0.65, 0), head=(0.08, 0.3, 0),
          hand_L=hand((0.04, -0.78, 1.44), f=(-0.8, -0.6, 0.0), n=(-1, 0, 0.1)), elbow_L=V(0.45, -0.1, 0.1), **sq(c=0.06, s=0.04))
    T.hold(11 + HS)
    T.key(18, "out", hips_loc=(-0.04, -0.12, -0.13), hips_rot=(0.07, -0.65, 0), spine=(0.30, -0.78, 0), head=(0.1, 0.38, 0),
          hand_L=hand((-0.10, -0.62, 1.40), f=(-0.8, -0.6, 0.0), n=(-1, 0, 0.1)), **sq(c=-0.01))
    recover_to_guard(T, rig, 32)
    return finish(T, rig, 32, [("contact", 11), ("impact", 11), ("recover", 18)], hitstop=[(11, 11 + HS)],
                  meta=dict(contact_frame=11, root_to_root_m=0.88, target="head of the receiver at about 1.45 m"))


@action("uppercut", acting="R", use="S3: rising punch to the chin / belly; receiver react_head_hit or react_gut_hit", note="deep crouch, explosive rise, impact at frame 12, heels lift")
def b_uppercut(rig):
    az = rig.az
    T = start(rig)
    T.key(7, "out", hips_loc=(0.0, 0.06, -0.30), hips_rot=(0.05, 0.1, 0), spine=(0.38, 0.2, 0), head=(0.15, -0.1, 0), neck=(0.05, 0, 0),
          hand_L=hand((0.20, -0.28, 0.80), f=(0, -0.6, -0.7), n=(-1, 0, 0)), elbow_L=V(0.2, 0.3, -0.2), hand_R=hand((0.06, -0.28, 1.28), f=(0, -1, 0.5), n=(-1, 0, 0), side=-1),
          knee_L=V(0.12, -0.55, 0), knee_R=V(0.14, -0.55, 0), **sq(c=-0.07, b=-0.03))
    T.key(10, "in", hips_loc=(-0.01, -0.06, -0.12), hips_rot=(0.05, -0.2, 0), spine=(0.12, -0.3, 0), head=(0.0, 0.1, 0), hand_L=hand((0.16, -0.55, 1.15), f=(0, -0.9, 0.5), n=(-1, 0, 0.2)),
          **sq(c=0.06, s=0.04))
    T.key(12, "in2", hips_loc=(-0.02, -0.10, -0.03), hips_rot=(0.0, -0.45, 0), spine=(-0.08, -0.5, 0), head=(-0.2, 0.2, 0), hand_L=hand((0.12, -0.70, 1.56), f=(0, -0.4, 0.9), n=(-1, 0, 0.1)),
          foot_R=V(0.17, 0.22, az + 0.05, -0.6, -0.55), elbow_L=V(0.25, -0.1, -0.1), **sq(c=0.09, s=0.07))
    T.hold(12 + HS)
    T.key(20, "out", hips_loc=(-0.02, -0.12, -0.10), hips_rot=(0.05, -0.55, 0), spine=(0.0, -0.6, 0), head=(-0.15, 0.3, 0), hand_L=hand((0.14, -0.62, 1.66), f=(0, -0.2, 1), n=(-1, 0, 0)),
          foot_R=V(0.17, 0.22, az, 0, -0.55), **sq(c=-0.02))
    recover_to_guard(T, rig, 36)
    return finish(T, rig, 36, [("crouch_peak", 7), ("contact", 12), ("impact", 12), ("recover", 20)], hitstop=[(12, 12 + HS)],
                  meta=dict(contact_frame=12, root_to_root_m=0.82, target="chin/lower face at about 1.40 m; belly if the receiver is crouched"))


@action("slap", acting="R", use="S3: open-hand slap across the face; receiver react_slapped", note="hand cocked at the actor's side, sweeping flat slap, impact at frame 10, follow-through across the body")
def b_slap(rig):
    T = start(rig)
    T.key(6, "out", hips_loc=(0.0, 0.06, -0.10), hips_rot=(0.04, 0.2, 0), spine=(0.12, 0.55, 0), head=(0.08, -0.2, 0), hand_L=hand((0.50, 0.05, 1.46), f=(0.5, -0.6, 0.4), n=(0, 0.2, 1.0)),
          elbow_L=V(0.4, 0.2, 0.0), curl_L=FLAT, **sq(c=-0.03))
    T.key(8, "in", hips_loc=(0.0, -0.02, -0.10), hips_rot=(0.04, -0.2, 0), spine=(0.16, -0.2, 0), hand_L=hand((0.34, -0.50, 1.50), f=(-0.2, -0.9, 0.2), n=(-0.4, 0.2, 0.9)), **sq(c=0.03))
    T.key(10, "in2", hips_loc=(-0.02, -0.07, -0.10), hips_rot=(0.05, -0.38, 0), spine=(0.2, -0.45, 0), head=(0.08, 0.15, 0), hand_L=hand((0.10, -0.72, 1.50), f=(-0.9, -0.5, 0.2), n=(0.1, 0.1, 1.0)),
          elbow_L=V(0.3, 0.0, 0.0), **sq(c=0.04))
    T.hold(10 + HS)
    T.key(19, "out", hips_loc=(-0.03, -0.09, -0.11), hips_rot=(0.06, -0.55, 0), spine=(0.24, -0.70, 0), head=(0.1, 0.30, 0), hand_L=hand((-0.30, -0.55, 1.42), f=(-1, -0.3, 0.1), n=(0, 0.2, 1.0)))
    recover_to_guard(T, rig, 32)
    return finish(T, rig, 32, [("contact", 10), ("impact", 10), ("recover", 19)], hitstop=[(10, 10 + HS)],
                  meta=dict(contact_frame=10, root_to_root_m=0.88, target="receiver's cheek at about 1.50 m, slightly to the actor's side"))


@action("head_butt", use="S3: head-butt in a clinch; receiver react_headbutt", note="hands grab shoulders, head cocked back, drives forward, impact at frame 12, symmetric")
def b_headbutt(rig):
    az = rig.az
    T = start(rig)
    grab = dict(hand_L=hand((0.22, -0.50, 1.38), f=(0, -0.7, -0.6), n=(-1, 0, 0.2)), hand_R=hand((0.22, -0.50, 1.38), f=(0, -0.7, -0.6), n=(-1, 0, 0.2), side=-1),
                curl_L=GRIP, curl_R=GRIP, elbow_L=V(0.30, 0.1, -0.1), elbow_R=V(0.30, 0.1, -0.1))
    T.key(5, "out", hips_loc=(0, 0.02, -0.10), hips_rot=(0.0, 0.0, 0), spine=(0.12, 0, 0), **grab)
    T.key(9, "out", hips_loc=(0, 0.05, -0.12), spine=(-0.08, 0, 0), neck=(-0.55, 0, 0), head=(-0.40, 0, 0), jaw=0.2, **sq(c=-0.04))
    T.key(11, "in", hips_loc=(0, -0.10, -0.12), spine=(0.35, 0, 0), neck=(0.3, 0, 0), head=(0.25, 0, 0), **sq(c=0.04, s=0.04))
    T.key(12, "in2", hips_loc=(0, -0.20, -0.13), spine=(0.55, 0, 0), neck=(0.35, 0, 0), head=(0.40, 0, 0), jaw=0.0, foot_L=V(0.14, -0.30, az, 0, 0.0), foot_R=V(0.14, 0.12, az, 0, 0.0), **sq(c=0.05, h=-0.06))
    T.hold(12 + HS)
    T.key(20, "out", hips_loc=(0, -0.12, -0.12), spine=(0.45, 0, 0), neck=(0.25, 0, 0), head=(0.15, 0, 0), jaw=0.1, **sq(h=0.03))
    recover_to_guard(T, rig, 34)
    return finish(T, rig, 34, [("contact", 12), ("impact", 12), ("recover", 20)], hitstop=[(12, 12 + HS)],
                  meta=dict(contact_frame=12, root_to_root_m=0.72, target="forehead to forehead / nose, symmetric; the head moves about 0.30 m forward"))


@action("shove", use="S3 clinch / S2: two-handed shove; receiver react_shoved", note="hands to chest, lunge, impact at frame 9, hold, rock back")
def b_shove(rig):
    az = rig.az
    T = start(rig)
    T.key(5, "out", hips_loc=(0, 0.08, -0.15), hips_rot=(0.05, 0, 0), spine=(0.12, 0, 0), head=(0.05, 0, 0),
          hand_L=hand((0.17, -0.22, 1.28), f=(0, -0.3, 1), n=(0, -1, 0)), hand_R=hand((0.17, -0.22, 1.28), f=(0, -0.3, 1), n=(0, -1, 0), side=-1), curl_L=FLAT, curl_R=FLAT,
          elbow_L=V(0.30, 0.3, -0.1), elbow_R=V(0.30, 0.3, -0.1), foot_L=V(0.15, -0.20, az, 0, 0), foot_R=V(0.15, 0.20, az, 0, 0), **sq(c=-0.05))
    T.key(8, "in", hips_loc=(0, -0.10, -0.12), spine=(0.30, 0, 0), hand_L=hand((0.17, -0.52, 1.32), f=(0, -0.3, 1), n=(0, -1, 0)), hand_R=hand((0.17, -0.52, 1.32), f=(0, -0.3, 1), n=(0, -1, 0), side=-1),
          foot_L=V(0.15, -0.34, az, 0, 0), **sq(c=0.05))
    T.key(9, "in2", hips_loc=(0, -0.17, -0.13), spine=(0.40, 0, 0), head=(0.1, 0, 0), hand_L=hand((0.17, -0.70, 1.34), f=(0, -0.2, 1), n=(0, -1, 0)),
          hand_R=hand((0.17, -0.70, 1.34), f=(0, -0.2, 1), n=(0, -1, 0), side=-1), **sq(c=0.06, s=0.04))
    T.hold(9 + HS)
    T.key(18, "out", hips_loc=(0, -0.20, -0.14), spine=(0.45, 0, 0), hand_L=hand((0.18, -0.76, 1.36), f=(0, -0.2, 1), n=(0, -1, 0)), hand_R=hand((0.18, -0.76, 1.36), f=(0, -0.2, 1), n=(0, -1, 0), side=-1), **sq(c=0.0))
    recover_to_guard(T, rig, 32)
    return finish(T, rig, 32, [("contact", 9), ("impact", 9), ("recover", 18)], hitstop=[(9, 9 + HS)],
                  meta=dict(contact_frame=9, root_to_root_m=0.88, target="receiver's chest at about 1.30 m, both palms"))


@action("chest_bump", use="S3: macho chest bump, both vendors play it; or one bumps a passive target", note="puff up, lunge, chest-to-chest impact at frame 10, recoil rebound")
def b_chest_bump(rig):
    az = rig.az
    T = start(rig)
    wing = dict(hand_L=hand((0.40, 0.00, 1.12), f=(0.4, -0.2, -0.9), n=(-1, 0, 0)), hand_R=hand((0.40, 0.00, 1.12), f=(0.4, -0.2, -0.9), n=(-1, 0, 0), side=-1),
                elbow_L=V(0.45, 0.2, -0.1), elbow_R=V(0.45, 0.2, -0.1), curl_L=FIST, curl_R=FIST)
    T.key(5, "out", hips_loc=(0, 0.04, -0.06), hips_rot=(0, 0, 0), spine=(-0.10, 0, 0), head=(-0.15, 0, 0), jaw=0.15, foot_L=V(0.13, -0.05, az, 0, 0), foot_R=V(0.13, 0.05, az, 0, 0), **wing, **sq(c=0.06))
    T.key(9, "in", hips_loc=(0, -0.12, -0.07), spine=(-0.15, 0, 0), head=(-0.05, 0, 0), foot_L=V(0.14, -0.25, az, 0, 0), **sq(c=0.08, s=0.03))
    T.key(10, "in2", hips_loc=(0, -0.20, -0.07), spine=(-0.12, 0, 0), head=(0.0, 0, 0), **sq(c=-0.06, s=-0.03))   # squash on contact
    T.hold(10 + HS)
    T.key(17, "snap", hips_loc=(0, 0.14, -0.06), spine=(-0.18, 0, 0), head=(-0.25, 0, 0), foot_L=V(0.14, -0.05, az, 0, 0), foot_R=V(0.14, 0.18, az, 0, 0), **sq(c=0.08))   # rebound
    recover_to_guard(T, rig, 32)
    return finish(T, rig, 32, [("contact", 10), ("impact", 10), ("rebound", 17)], hitstop=[(10, 10 + HS)],
                  meta=dict(contact_frame=10, root_to_root_m=0.64, target="chest to chest; play on both characters, roots 0.64 m apart at the start, they close to about 0.40 m"))


@action("collar_grab_shake", acting="R", use="S3: grabs the receiver by the collar and shakes (loop frames 14-38); receiver collar_shaken", note="reach, grab at frame 9, pull in, shake loop 14-38, release")
def b_collar(rig):
    az = rig.az
    T = start(rig)
    N = 54
    T.key(4, "out", hips_loc=(0, 0.04, -0.12), spine=(0.15, 0.2, 0), hand_L=hand((0.25, -0.30, 1.30), f=(0, -1, 0.2), n=(-1, 0, 0)), curl_L=FLAT, **sq(c=-0.02))
    T.key(8, "in", hips_loc=(0, -0.10, -0.12), spine=(0.30, -0.15, 0), hand_L=hand((0.08, -0.78, 1.40), f=(0, -1, 0.3), n=(-1, 0, 0)), foot_L=V(0.17, -0.34, az, 0, 0), curl_L=FLAT, **sq(c=0.04))
    T.key(9, "in2", curl_L=GRIP)
    T.hold(9 + 2)
    # pull in and up; head thrust
    T.key(14, "out", hips_loc=(0, -0.12, -0.12), spine=(0.35, -0.2, 0), head=(0.2, 0, 0), jaw=0.3, hand_L=hand((0.10, -0.62, 1.42), f=(0, -1, 0.3), n=(-1, 0, 0)), **sq(c=0.02))
    # shake loop: 24 frames, 3 cycles of 8
    for k in range(1, 7):
        f = 14 + 4 * k
        s = 1 if k % 2 else -1
        T.key(f, "smooth", hips_loc=(0.02 * s, -0.12, -0.12), spine=(0.35, -0.2 + 0.10 * s, 0.05 * s), head=(0.2 + 0.1 * s, 0.1 * s, 0), jaw=0.3 + 0.2 * (s > 0),
              hand_L=hand((0.10 + 0.07 * s, -0.60 - 0.06 * s, 1.42 + 0.03 * s), f=(0, -1, 0.3), n=(-1, 0, 0)))
    T.key(42, "out", hips_loc=(0, -0.14, -0.13), spine=(0.40, -0.25, 0), hand_L=hand((0.12, -0.64, 1.44), f=(0, -1, 0.3), n=(-1, 0, 0)))
    T.key(46, "in", curl_L=FLAT, hand_L=hand((0.20, -0.74, 1.46), f=(0, -1, 0.3), n=(-1, 0, 0)), spine=(0.36, -0.25, 0), **sq(c=0.04))   # shove away
    recover_to_guard(T, rig, N)
    return finish(T, rig, N, [("contact", 9), ("grab", 9), ("loop_start", 14), ("loop_end", 38), ("release", 46)], hitstop=[(9, 11)],
                  overlap=dict(head=(6.0, 0.4), neck=(7.0, 0.45)), meta=dict(contact_frame=9, loop_start=14, loop_end=38, root_to_root_m=0.88,
                                                                          target="collar at about 1.40 m; frames 14-38 can be repeated with strip start_frame/end_frame"))


@action("duck", use="S3: dodging a punch; sync to the attacker's contact frame minus 3", note="quick crouch with head snap down, covers with hands, rises")
def b_duck(rig):
    az = rig.az
    T = start(rig)
    T.key(4, "snap", hips_loc=(0, 0.06, -0.38), hips_rot=(0.05, 0.1, 0), spine=(0.45, 0.2, 0), neck=(0.15, 0, 0), head=(0.25, 0.2, 0),
          hand_L=hand((0.12, -0.20, 1.12), f=(0, -0.3, 1), n=(-1, 0, 0)), hand_R=hand((0.12, -0.20, 1.12), f=(0, -0.3, 1), n=(-1, 0, 0), side=-1),
          knee_L=V(0.14, -0.55, 0), knee_R=V(0.14, -0.55, 0), foot_L=V(0.20, -0.20, az, 0, 0.1), foot_R=V(0.20, 0.20, az, 0, -0.1), **sq(c=-0.08, b=-0.04))
    T.key(14, "smooth", hips_loc=(0.0, 0.06, -0.36), **sq(c=-0.06, b=-0.03))
    T.key(24, "back", **guard_base(rig), **sq())
    return finish(T, rig, 24, [("duck_peak", 4), ("rise", 14)], meta=dict(note2="needs about 10 frames of hold; strip trim to lengthen"))


@action("whiff_overbalance", acting="R", use="S3: a missed haymaker that spins the attacker around and he stumbles", note="wind-up, swing through empty air at frame 15 (no hit-stop), over-rotation, stumble and recover")
def b_whiff(rig):
    az = rig.az
    T = start(rig, hfol_L=1.0, hfol_R=1.0)
    T.key(7, "out", hips_loc=(0.03, 0.12, -0.17), hips_rot=(0.05, 0.28, 0.0), spine=(0.12, 0.85, 0.05), head=(0.12, -0.45, 0), hand_L=hand((0.62, 0.30, 1.22), f=(0.5, 0.6, 0.3), n=(-0.3, 0, 1.0)),
          elbow_L=V(0.5, 0.30, 0.05), **sq(c=-0.04))
    T.key(15, "in", hips_loc=(-0.04, -0.14, -0.12), hips_rot=(0.06, -0.55, 0), spine=(0.28, -0.85, -0.05), head=(0.08, 0.35, 0), hand_L=hand((0.12, -0.84, 1.46), f=(0, -1, 0.0), n=(-1, 0, 0.0)),
          elbow_L=V(0.35, -0.1, 0.1), **sq(c=0.07, s=0.05))
    # nothing there: arm keeps going, body spins and tips forward
    T.key(24, "out2", root_rot=(0.0, -0.9, 0), hips_loc=(-0.08, -0.20, -0.10), hips_rot=(0.18, -1.0, 0.1), spine=(0.40, -1.15, 0.1), head=(0.20, 0.7, 0),
          hand_L=hand((-0.45, -0.30, 1.30), f=(-1, -0.2, 0), n=(-1, 0, 0.2)), hand_R=hand((-0.35, -0.50, 1.00), f=(-0.5, -1, -0.3), n=(1, 0, 0), side=-1),
          foot_L=V(0.14, -0.45, az, 0, -0.2), foot_R=V(0.15, 0.05, az, 0, -1.0), curl_L=FLAT, **sq(c=-0.03))
    T.key(33, "back", root_rot=(0.0, -1.3, 0), hips_loc=(-0.10, -0.26, -0.18), hips_rot=(0.24, -1.1, 0.14), spine=(0.50, -1.0, 0.1), head=(0.28, 0.5, 0),
          hand_R=hand((-0.30, -0.65, 0.80), f=(-0.2, -1, -0.5), n=(1, 0, 0), side=-1), foot_R=V(0.16, -0.40, az, 0, -1.2), foot_L=V(0.20, 0.10, az, 0, 0.2), **sq(c=-0.04))
    T.key(46, "smooth", root_rot=(0.0, -1.3, 0), hips_loc=(-0.02, -0.05, -0.12), hips_rot=(0.06, -0.50, 0.0), spine=(0.25, -0.2, 0), head=(0.1, 0.2, 0), **sq())
    T.key(60, "smooth", root_rot=(0, -1.3, 0))
    return finish(T, rig, 60, [("swing_through", 15), ("overbalance", 24), ("stumble_step", 33), ("recovered", 46)],
                  meta=dict(root_yaw_delta_rad=-1.3, note2="ends rotated about 75 degrees to the actor's right; object yaw handoff via apply()"),
                  overlap=dict(head=(5.5, 0.4), neck=(7.0, 0.45), hand_L=(8.0, 0.5)), attach=dict(L="root", R="root"))


@action("flail_windmill", use="S3: panicked windmilling arms before / during a brawl; loop 24 frames", note="both arms circle at full extension in alternation, knees bent, head bob, mouth open")
def b_windmill(rig):
    N = 24
    az = rig.az
    S = rig.dims["sh"]

    def fn(frame):
        i = int(round(frame)) % N
        u = i / N
        a = 2 * math.pi * u
        P = neutral()
        P["hips_loc"] = V(0.03 * math.sin(a), 0, -0.12 + 0.02 * math.cos(2 * a))
        P["hips_rot"] = V(0.03, 0.2 * math.sin(a), 0)
        P["spine"] = V(0.15, -0.25 * math.sin(a), 0.06 * math.cos(a))
        P["head"] = V(0.1 * math.cos(2 * a), 0.2 * math.sin(a), 0)
        P["jaw"] = V(0.45)
        P["foot_L"] = V(0.20, -0.08, az, 0, 0.3)
        P["foot_R"] = V(0.20, 0.08, az, 0, -0.3)
        P["knee_L"] = V(0.15, -0.55, 0)
        P["knee_R"] = V(0.15, -0.55, 0)
        R = 0.45
        for sn, side, ph in (("L", 1, 0.0), ("R", -1, math.pi)):
            th = a + ph
            pos = (S[0] + 0.14, S[1] - R * math.sin(th), S[2] - R * math.cos(th) * 0.95 + 0.0)
            P["hand_" + sn] = hand(pos, f=(0.2, -math.sin(th), -math.cos(th)), n=(-1, 0, 0), side=side)
            P["curl_" + sn] = V(0.1)
            P["elbow_" + sn] = V(0.35, 0.0, 0.0)
        P["sq_c"] = ss(0.02 * math.cos(2 * a))
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(7.0, 0.4)), meta=dict(note2="arm radius 0.50 baseline"))


@action("grapple", use="S3: two characters locked in a clinch, wrestling sway; play on both (offset one by half a cycle); loop 36 frames", note="arms around the partner's shoulders/back, wide low stance, push-pull sway, head pressed")
def b_grapple(rig):
    N = 36
    az = rig.az

    def fn(frame):
        i = int(round(frame)) % N
        a = 2 * math.pi * i / N
        s, c = math.sin(a), math.cos(a)
        P = neutral()
        P["hips_loc"] = V(0.05 * s, -0.05 - 0.04 * c, -0.20 - 0.02 * abs(s))
        P["hips_rot"] = V(0.10, 0.18 * s, 0)
        P["spine"] = V(0.45 + 0.08 * c, -0.3 * s, 0.10 * s)
        P["neck"] = V(0.2, 0.1 * s, 0)
        P["head"] = V(0.25, 0.3 * s, 0.05 * s)
        P["jaw"] = V(0.25 + 0.15 * s)
        P["foot_L"] = V(0.24, -0.22, az, 0, 0.25)
        P["foot_R"] = V(0.24, 0.22, az, 0, -0.25)
        P["knee_L"] = V(0.18, -0.55, 0)
        P["knee_R"] = V(0.18, -0.55, 0)
        P["hand_L"] = hand((0.30 + 0.04 * s, -0.55 - 0.06 * c, 1.28), f=(-0.5, -0.8, -0.3), n=(-0.3, 0, 0.9))
        P["hand_R"] = hand((0.04 - 0.02 * s, -0.66 - 0.06 * c, 1.12), f=(-0.8, -0.3, -0.5), n=(0.2, 0.2, 0.9), side=-1)
        P["curl_L"] = V(0.9)
        P["curl_R"] = V(0.9)
        P["elbow_L"] = V(0.35, 0.1, -0.2)
        P["elbow_R"] = V(0.30, 0.2, -0.2)
        P["sq_c"] = ss(-0.02 * c)
        return P
    return dict(fn=fn, n=N, loop=True, overlap=dict(head=(6.0, 0.4)), meta=dict(root_to_root_m=0.62, note2="arms reach past the partner's flanks; roots 0.62 m apart facing each other"))
