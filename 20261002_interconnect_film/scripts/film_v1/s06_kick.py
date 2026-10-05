"""S6 v1.2: two scene-local motion_v2 actions for the finale gag "boss's ass gets kicked by the customer".

Authored with the motion_v2 pose language (scripts/assets/characters/motion_v2, baseline character space: x left, y back, z up,
forward = -y, metres at stature 1.75, times the rig K) and baked IN THE SCENE on the appended rig (nothing is written to the
asset libraries):
  kick_butt          attacker (NVYDIA customer): anticipation (leg cocked back, lean back), swing, contact at frame 11 with the
                     right foot at about 0.75 m (baseline) height 0.62 m ahead of the root, 3-frame hit-stop, retract, smug stand.
  react_kicked_butt  receiver (Manager), frame 0 = contact: hips squash and arch, 3-frame hit-stop, launched forward with a hop and
                     stretch (feet ride the root), landing squash at frame 14, stumble step, settle; root bone ends about 1.05 m
                     (baseline) ahead.
Usage (in s06_fiber.py, right after asm.append and before any NLA strip is added to that rig):
    import s06_kick; s06_kick.bake(asset, "kick_butt")  -> bpy Action ACT_<asset_id>_v2_kick_butt (+ _face), then M2.apply(asset, "kick_butt", t)
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
M2DIR = os.path.join(os.path.dirname(HERE), "assets", "characters", "motion_v2")
if M2DIR not in sys.path:
    sys.path.insert(0, M2DIR)
import motion_v2 as M2  # noqa: E402
import m2_engine as E  # noqa: E402
from m2_engine import V  # noqa: E402
from m2_lib import action, Timeline, hand  # noqa: E402
from m2_fight import sq, finish, HS, FLAT  # noqa: E402

KICK_CONTACT = 11          # frame of foot contact in kick_butt
REST_L = dict(hand_L=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0)),
              hand_R=hand((0.285, -0.08, 0.86), f=(0.04, -0.22, -0.97), n=(-1, 0, 0), side=-1))


def _stand(rig, **kw):
    az = rig.az
    d = dict(foot_L=V(0.10, 0, az, 0, 0.06), foot_R=V(0.10, 0, az, 0, -0.06), hfol_L=0.0, hfol_R=0.0)
    d.update(kw)
    return d


@action("kick_butt", use="S6 v1.2: customer kicks the Manager's backside (receiver react_kicked_butt)",
        note="anticipation with the right leg cocked back and the torso leaning back, swing, contact at frame 11, 3-frame hit-stop, retract, smug stand")
def b_kick_butt(rig):
    az = rig.az
    T = Timeline()
    T.key(0, **_stand(rig), **REST_L)
    # anticipation: weight back onto the left leg, right leg cocked back and up, arms counter
    T.key(6, "out", hips_loc=(0.0, 0.05, -0.07), hips_rot=(0.0, 0.10, 0.0), spine=(0.10, 0.12, 0.0), head=(0.10, -0.08, 0.0), jaw=0.05,
          foot_R=V(0.12, 0.34, az + 0.24, -0.70, -0.05), lift=0.0,
          hand_L=hand((0.34, -0.30, 1.10), f=(0.2, -0.9, 0.2), n=(-1, 0, 0)), hand_R=hand((0.36, 0.22, 1.00), f=(0.3, 0.6, -0.4), n=(-1, 0, 0), side=-1),
          elbow_L=V(0.25, 0.25, -0.2), elbow_R=V(0.30, 0.0, -0.2), **sq(c=-0.03))
    # swing through
    T.key(9, "in", hips_loc=(0.0, 0.03, -0.04), hips_rot=(0.0, -0.05, 0.0), spine=(-0.12, -0.05, 0.0), head=(0.12, 0.0, 0.0),
          foot_R=V(0.11, -0.28, az + 0.40, 0.10, 0.0), lift=0.0,
          hand_L=hand((0.40, -0.15, 1.25), f=(0.5, -0.7, 0.3), n=(-1, 0, 0.2)), hand_R=hand((0.40, 0.12, 1.20), f=(0.5, 0.3, 0.3), n=(-1, 0, 0.2), side=-1))
    # contact: leg high and forward, torso leans back, arms out for balance
    T.key(KICK_CONTACT, "in2", hips_loc=(0.0, 0.07, -0.03), hips_rot=(0.0, -0.10, 0.0), spine=(-0.30, -0.08, 0.0), neck=(0.10, 0, 0), head=(0.22, 0.0, 0.0), jaw=0.25,
          foot_R=V(0.10, -0.62, 0.82, 0.55, 0.0), lift=0.0, knee_R=V(0.03, -0.60, 0.25),
          hand_L=hand((0.45, -0.25, 1.35), f=(0.6, -0.6, 0.4), n=(-1, 0, 0.3)), hand_R=hand((0.48, 0.20, 1.25), f=(0.7, 0.3, 0.3), n=(-1, 0, 0.3), side=-1),
          curl_L=FLAT, curl_R=FLAT, **sq(c=0.04))
    T.hold(KICK_CONTACT + HS)
    # retract
    T.key(21, "out", hips_loc=(0.0, 0.04, -0.06), hips_rot=(0.0, 0.0, 0.0), spine=(-0.08, 0.0, 0.0), head=(0.10, 0.0, 0.0), jaw=0.1,
          foot_R=V(0.11, -0.28, az + 0.16, 0.10, 0.0), lift=0.0, knee_R=V(0.03, -0.55, 0.0),
          hand_L=hand((0.36, -0.15, 1.05), f=(0.3, -0.8, 0.0), n=(-1, 0, 0)), hand_R=hand((0.36, 0.05, 1.00), f=(0.3, -0.3, -0.5), n=(-1, 0, 0), side=-1), **sq())
    T.key(27, "smooth", foot_R=V(0.10, -0.10, az, 0.0, -0.06), lift=0.03, hips_loc=(0.0, 0.0, -0.03), spine=(0.0, 0.0, 0.0), head=(0.05, 0.0, 0.0))
    # smug stand
    T.key(36, "smooth", **_stand(rig), hips_loc=(0, 0, 0), hips_rot=(0, 0, 0), spine=(-0.04, 0, 0), head=(-0.05, 0.1, 0.0), jaw=0.0, curl_L=E.V(0.18, 0.18, 0.18, 0.18, 0.18),
          curl_R=E.V(0.18, 0.18, 0.18, 0.18, 0.18), elbow_L=V(0.16, 0.38, -0.31), elbow_R=V(0.16, 0.38, -0.31), **REST_L)
    return finish(T, rig, 36, [("contact", KICK_CONTACT), ("impact", KICK_CONTACT), ("recover", 21)], hitstop=[(KICK_CONTACT, KICK_CONTACT + HS)],
                  overlap=dict(head=(6.5, 0.4), neck=(8.0, 0.45), hand_L=(8.0, 0.45), hand_R=(8.0, 0.45)),
                  face=[(0, {"p_expr_smug": 0.6}), (6, {"p_expr_smug": 0.0, "p_expr_angry": 0.6}), (KICK_CONTACT, {"p_expr_angry": 1.0}),
                        (24, {"p_expr_angry": 0.0, "p_expr_smug": 1.0}), (36, {"p_expr_smug": 1.0})],
                  props=["p_expr_smug", "p_expr_angry"],
                  meta=dict(contact_frame=KICK_CONTACT, target="receiver's backside, right foot 0.62 m ahead of the root at 0.74 m (baseline)"))


@action("react_kicked_butt", use="S6 v1.2: receiver of kick_butt (start at the attacker's contact frame): launched forward",
        note="hips squash and back arch at contact, 3-frame hit-stop, hop forward with stretch and flailing arms, landing squash at frame 14, stumble, settle")
def b_react_kicked(rig):
    az = rig.az
    T = Timeline()
    st = _stand(rig, ffol_L=1.0, ffol_R=1.0)
    T.key(0, **st, **REST_L)
    T.key(1, "cut", hips_loc=(0.0, -0.07, -0.02), spine=(-0.22, 0.0, 0.0), neck=(-0.1, 0, 0), head=(-0.30, 0.0, 0.0), jaw=0.5,
          hand_L=hand((0.34, 0.10, 1.20), f=(0.4, 0.3, 0.6), n=(-1, 0, 0)), hand_R=hand((0.34, 0.10, 1.20), f=(0.4, 0.3, 0.6), n=(-1, 0, 0), side=-1),
          curl_L=FLAT, curl_R=FLAT, **sq(c=-0.05, b=-0.14, s=-0.06))
    T.hold(HS)
    # launch: stretch, feet off the floor (they ride the root bone)
    T.key(HS + 4, "snap", root_loc=(0.0, -0.32, 0.16), hips_loc=(0.0, -0.10, 0.0), spine=(-0.40, 0.0, 0.0), neck=(-0.2, 0, 0), head=(-0.35, 0.0, 0.0), jaw=0.85,
          foot_L=V(0.10, 0.10, az + 0.06, -0.35, 0.06), foot_R=V(0.10, 0.28, az + 0.16, -0.55, -0.06), lift=0.0,
          hand_L=hand((0.45, 0.22, 1.60), f=(0.4, 0.3, 0.8), n=(-1, 0, 0)), hand_R=hand((0.45, 0.22, 1.60), f=(0.4, 0.3, 0.8), n=(-1, 0, 0), side=-1),
          **sq(c=0.08, s=0.07, b=0.06))
    # flight peak: bicycle legs, windmill arms
    T.key(HS + 8, "out", root_loc=(0.0, -0.58, 0.20), hips_loc=(0.0, -0.04, 0.0), spine=(-0.15, 0.10, 0.0), neck=(0.0, 0, 0), head=(-0.1, 0.1, 0.0), jaw=0.7,
          foot_L=V(0.10, -0.18, az + 0.14, 0.2, 0.06), foot_R=V(0.10, 0.24, az + 0.22, -0.40, -0.06), lift=0.0,
          hand_L=hand((0.55, -0.20, 1.45), f=(0.6, -0.5, 0.4), n=(-1, 0, 0.2)), hand_R=hand((0.50, 0.25, 1.30), f=(0.6, 0.4, 0.2), n=(-1, 0, 0.2), side=-1), **sq())
    # landing squash, lurch forward
    T.key(14, "in", root_loc=(0.0, -0.78, 0.0), hips_loc=(0.0, 0.0, -0.17), spine=(0.45, 0.0, 0.0), neck=(0.1, 0, 0), head=(0.15, 0.0, 0.0), jaw=0.5,
          foot_L=V(0.12, -0.22, az, 0.0, 0.06), foot_R=V(0.10, 0.22, az, 0.0, -0.06), lift=0.0,
          hand_L=hand((0.36, -0.42, 1.00), f=(0.2, -0.9, -0.2), n=(-1, 0, 0)), hand_R=hand((0.36, -0.40, 1.05), f=(0.2, -0.9, -0.2), n=(-1, 0, 0), side=-1),
          **sq(c=-0.08, s=-0.07, b=-0.06))
    # stumble step
    T.key(20, "out", root_loc=(0.0, -0.96, 0.0), hips_loc=(0.0, 0.0, -0.12), spine=(0.36, -0.10, 0.0), head=(0.1, -0.15, 0.0), jaw=0.45,
          foot_R=V(0.10, -0.32, az, 0.0, -0.06),
          hand_L=hand((0.50, -0.30, 1.20), f=(0.5, -0.7, 0.2), n=(-1, 0, 0)), hand_R=hand((0.40, -0.10, 0.95), f=(0.4, -0.5, -0.5), n=(-1, 0, 0), side=-1), **sq())
    T.key(28, "smooth", root_loc=(0.0, -1.04, 0.0), hips_loc=(0.0, 0.0, -0.06), spine=(0.14, 0.0, 0.0), head=(0.05, 0.0, 0.0), jaw=0.35,
          foot_L=V(0.11, -0.04, az, 0.0, 0.06), foot_R=V(0.11, -0.10, az, 0.0, -0.06),
          hand_L=hand((0.34, -0.10, 0.95), f=(0.1, -0.5, -0.8), n=(-1, 0, 0)), hand_R=hand((0.34, -0.10, 0.95), f=(0.1, -0.5, -0.8), n=(-1, 0, 0), side=-1))
    T.key(40, "smooth", root_loc=(0.0, -1.04, 0.0), hips_loc=(0, 0, 0), spine=(0.06, 0, 0), neck=(0, 0, 0), head=(0.0, 0.0, 0.0), jaw=0.3,
          foot_L=V(0.10, -0.04, az, 0, 0.06), foot_R=V(0.10, -0.04, az, 0, -0.06), curl_L=E.V(0.18, 0.18, 0.18, 0.18, 0.18), curl_R=E.V(0.18, 0.18, 0.18, 0.18, 0.18), **REST_L)
    return finish(T, rig, 40, [("impact", 0), ("launch", HS + 4), ("land", 14), ("step", 20), ("settle", 28)], hitstop=[(0, HS)],
                  overlap=dict(head=(5.5, 0.35), neck=(6.5, 0.4), hand_L=(7.0, 0.4), hand_R=(7.0, 0.4)),
                  face=[(0, {"p_expr_shock": 0.0}), (1, {"p_expr_shock": 1.0}), (18, {"p_expr_shock": 0.8, "p_expr_scared": 0.5}),
                        (40, {"p_expr_shock": 0.5, "p_expr_scared": 0.4})],
                  props=["p_expr_shock", "p_expr_scared"],
                  meta=dict(end_root_loc_y_m=-1.04 * rig.K, note2="frame 0 = contact; root bone ends 1.04 m (baseline) ahead"))


def bake(asset, name):
    """Bake action `name` on the asset's rig (current scene). Call before any NLA strip is placed on that rig."""
    rig = E.Rig2(asset.root)
    arm = rig.arm
    ad = arm.animation_data
    saved = ad.action if ad else None
    if ad:
        ad.action = None
    hid = [o for o in rig.meshes if not o.hide_viewport]
    for o in hid:
        o.hide_viewport = True
    act, info = M2.bake_one(rig, name)
    for o in hid:
        o.hide_viewport = False
    rig.clear()
    if ad:
        ad.action = saved
    return act, info
