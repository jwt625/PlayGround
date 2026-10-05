"""NPC family v2 (agent C2, 2026-10-03): vendors and customers on the v2 chunky-clay kit.

Run (project root): Blender -b --python scripts/assets/characters/v2/npc/build_npc_v2.py -- <out_dir> <variant> [no_actions]
variant: npc | npc_molexx | npc_nubiss | npc_terahop | npc_ayarr | npc_nvydia | npc_openay
Output: <out_dir>/<variant>_v2.blend + .json (asset id <variant>_v2).

Compatibility with the v1 asset of the same variant: same bone names and hierarchy (plus belly; npc_terahop_v2 also
hat), same rest pose for every non-head bone (same H, build, shoulders, arm_len, leg_len as v1), same HOOK_* names
(hand_L/R, gun_grip_R, gun_support_L, head_top, hug_target), same root props (+ the v2 props and p_accessory), the
14 expression shape keys, hole_1 (same bone, same position; v2 base radius 0.045 m, cut half-length 0.30 m) and the
21 actions ACT_<asset>_<name>.
Heights (stature = skull crown, m) are the v1 design values: 1.75, 1.80, 1.70, 1.88, 1.62, 1.82, 1.40 (0.8 x 1.75).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, V2)
import numpy as np  # noqa: E402
import bpy  # noqa: E402
import chars_geo as G  # noqa: E402
import chars_mat as M  # noqa: E402
import chars_body as B  # noqa: E402
import chars_outfit as O  # noqa: E402
import chars_actions as A  # noqa: E402
import common as C  # noqa: E402
import meta_v2 as MV  # noqa: E402
import npc_kit_v2 as NK  # noqa: E402

CROWN = 1.358 + 0.233 * 1.68   # skull crown height of the v2 head (baseline m), kept for every head shape

# Design values. H, build: v1 (unchanged, they define the skeleton). wide/arm_r/leg_r/belly/head: v2 silhouette design.
VARIANTS = {
    "npc": dict(H=1.75, build=1.0, belly=0.10, wide=1.0, arm_r=1.0, leg_r=1.0,
                skin="#d8a27c", hair="short", hair_col="#3b2a1c", shirt="#8d939c", logo=None, trousers="#3a3f4a", shoe="#d8d8d8",
                headS=(1.90, 1.42, 1.68), head={}, face=[], acc=None, lip="#c9705f",
                look="average build, short hair, plain grey tee (generic extra)"),
    "npc_molexx": dict(H=1.80, build=1.05, belly=0.18, wide=1.12, arm_r=1.12, leg_r=1.06,
                       skin="#d9a07a", hair="flattop", hair_col="#2a1c12", shirt="#d1222a", logo=["MOLEXX"], logo_col="#ffffff", logo_w=0.28,
                       trousers="#2c3140", shoe="#222226", headS=(1.98, 1.40, 1.62), lip="#c06656",
                       head=dict(jaw=0.02, sup=2.9, chin=0.08, brow_t=0.0098, brow_h=0.0125, eye_r=0.0218),
                       face=["moustache"], acc="lanyard", acc_col=("#1d2a66", "#f4f4f0", "#d1222a"),
                       look="broad and boxy: square head, flat-top crew cut, thick moustache, bushy brows, lanyard badge"),
    "npc_nubiss": dict(H=1.70, build=1.22, belly=0.62, wide=1.10, arm_r=1.12, leg_r=1.10,
                       skin="#c98f66", hair="bald", hair_col="#4a3322", shirt="#2f9e4f", logo=["NUBISS"], logo_col="#ffffff", logo_w=0.28,
                       trousers="#4b4f5a", shoe="#3a2a20", headS=(2.04, 1.50, 1.60), lip="#b9614f",
                       head=dict(sup=2.0, jaw=0.14, w0=0.084, nose_s=1.45),
                       face=["beard", "glasses_rect"], beard_col="#5a3b24", glasses_col="#16161a", acc="headset", acc_col=("#26282e", "#b8bcc4"),
                       look="short and round: big belly, round bald head with side fringe, full beard, small rectangular glasses, headset"),
    "npc_terahop": dict(H=1.88, build=0.92, belly=0.0, wide=0.88, arm_r=0.86, leg_r=0.86,
                        skin="#e8b894", hair="ponytail", hair_col="#6a4020", shirt="#6a3fb0", logo=["TERAHOP"], logo_col="#ffffff", logo_w=0.27,
                        trousers="#25303f", shoe="#1a1a1c", headS=(1.70, 1.36, 1.72), lip="#cf7466",
                        head=dict(jaw=0.20, chin=0.07, w0=0.078, ear_s=0.80, nose_s=1.25),
                        face=["goatee"], beard_col="#6a4020", acc="cap", acc_col=("#f1efe8", "#6a3fb0"),
                        extra_bones=[dict(name="hat", head=(0.0, 0.0, 1.66), tail=(0.0, 0.0, 1.80), parent="head")],
                        look="tall and lean: long narrow head, ponytail, goatee, white baseball cap"),
    "npc_ayarr": dict(H=1.62, build=0.95, belly=0.05, wide=0.98, arm_r=0.96, leg_r=0.96,
                      skin="#8a5a3c", hair="afro", hair_col="#241912", shirt="#f2c230", logo=["AYARR"], logo_col="#111111", logo_w=0.21,
                      trousers="#3a3a42", shoe="#e8e8e8", headS=(1.98, 1.46, 1.70), lip="#7a4434",
                      head=dict(sup=2.2, eye_r=0.0255, pupil_r=0.0120),
                      face=["glasses_round"], glasses_col="#f2f2ee", acc="backpack", acc_col=("#2b6f8f", "#1d3b4f"),
                      look="short: big round head, large curly hair, big round white-framed glasses, backpack"),
    "npc_nvydia": dict(H=1.82, build=1.10, belly=0.15, wide=1.04, arm_r=1.04, leg_r=1.02,
                       skin="#f0c4a0", hair="buzz", hair_col="#b8b8bc", shirt="#76c21a", logo=["NVYDIA"], logo_col="#101010", logo_w=0.135,
                       trousers="#3a3d47", shoe="#16161a", jacket="#222227", headS=(1.88, 1.42, 1.66), lip="#cf7466",
                       head=dict(jaw=0.06, sup=2.5), face=[], acc=None,
                       look="solid build, silver buzz cut, black clay leather moto jacket (open V shows the green NVYDIA tee)"),
    "npc_openay": dict(H=1.75 * 0.8, build=0.95, belly=0.0, wide=1.0, arm_r=1.0, leg_r=1.0,
                       skin="#e0b08a", hair="bob", hair_col="#6b4a35", shirt="#e9e1cf", logo=["OPENAY", "ANTHROPY"], logo_col="#22252c", logo_w=0.25,
                       trousers="#4a5068", shoe="#d8d8d8", headS=(1.96, 1.46, 1.72), lip="#cf7466", head={}, face=[], acc=None,
                       look="0.8-scale hugger: bigger head, brown bob, cream tee with OPENAY over ANTHROPY"),
}


def build(var):
    v = VARIANTS[var]
    cid = var + "_v2"
    spec = B.default_spec()
    spec["H"], spec["build"], spec["belly"] = v["H"], v["build"], v["belly"]
    spec["wide"], spec["arm_r"], spec["leg_r"] = v["wide"], v["arm_r"], v["leg_r"]
    spec["head"].update(v["head"])
    spec["headS"] = tuple(v["headS"])
    spec["chin_post"] = CROWN - 0.233 * v["headS"][2]
    if v.get("extra_bones"):
        spec["extra_bones"] = v["extra_bones"]
    ch = B.Character(cid, spec)
    K = ch.K
    ch.jiggles.append(("belly", (0.0, -0.13, 1.04), (0.20, 0.14, 0.13), 0.9))
    ch.root_props()
    holes = [dict(name="hole_1", prop="p_hole_1_radius", bone="chest", pos=(0.04, 0.0, 1.20), axis="Y", radius=0.045, half_len=0.30, default=0.0)]
    hg = ch.setup_holes(holes)
    pre = "MAT_characters_%s_" % cid
    O.face_mats(ch, v["skin"], brow_hex=v["hair_col"] if v["hair"] != "bald" else v.get("beard_col", v["hair_col"]), hole_group=hg, lip_hex=v["lip"])
    ch.mat("shirt", M.clay(pre + "shirt", v["shirt"], rough=0.72, bump=0.28, holes=hg, sheen=0.35))
    ch.mat("logo", M.clay(pre + "logo", v.get("logo_col", "#ffffff"), rough=0.5, bump=0.08, holes=hg))
    ch.mat("trousers", M.clay(pre + "trousers", v["trousers"], rough=0.74, bump=0.28, holes=hg, sheen=0.35, folds=[(0.50, 0.09, 16.0, 0.4), (0.22, 0.07, 22.0, 0.4)]))
    ch.mat("shoe", M.clay(pre + "shoes", v["shoe"], rough=0.55, bump=0.22, holes=hg, sheen=0.2))
    ch.mat("sole", M.clay(pre + "sole", "#ece9e0", rough=0.6, bump=0.1, holes=hg))
    ch.mat("hair", M.clay(pre + "hair", v["hair_col"], rough=0.7, bump=0.3, holes=hg, sheen=0.3))
    J = ch.J
    ch.add(ch.skin_parts(), "body", ["skin"], jig=True)
    ch.build_face()
    is_nv = var == "npc_nvydia"

    # ---------------------------------------------------------------- shirt + wordmark
    if is_nv:
        shirt = NK.tee(ch, z0=0.95, off_low=0.0, flare=0.0, hem=False, sleeve_pad=0.006, sleeve_hem=False, sleeve_frac=0.45)
    else:
        shirt = NK.tee(ch)
    ch.add(shirt, "shirt", ["shirt"], jig=True)
    if v["logo"]:
        if is_nv:
            lg = NK.wordmark(ch, v["logo"], 1.285, v["logo_w"], 0.012, depth=1.2)
        elif len(v["logo"]) == 1:
            lg = NK.wordmark(ch, v["logo"], 1.235, v["logo_w"], 0.012)
        else:
            lg = NK.wordmark(ch, v["logo"], 1.225, v["logo_w"], 0.012, line_gap=1.30)
        ch.add(lg, "logo", ["logo"], jig=True)

    # ---------------------------------------------------------------- trousers, shoes
    tr, cr = NK.trousers(ch)
    ch.add(tr, "trousers", ["trousers"], jig=True)
    ch.add(cr, "creases", ["trousers"])
    for side, sn in ((1, "L"), (-1, "R")):
        sh_ = O.boot(ch, sn, side, shaft_h=0.05, scale=1.06, toe_len=0.235)
        ch.add(G.merge([sh_[0], sh_[2], sh_[3]]), "shoe_" + sn, ["shoe"])
        ch.add(sh_[1], "sole_" + sn, ["sole"], bevel=0.002)

    # ---------------------------------------------------------------- leather jacket (NVYDIA)
    if is_nv:
        ch.mat("leather", M.clay(pre + "leather", v["jacket"], rough=0.34, spec=0.65, bump=0.10, holes=hg, sheen=0.40))
        lm = ch.m["leather"]
        bs = lm.node_tree.nodes["Principled BSDF"]
        bs.inputs["Coat Weight"].default_value = 0.35
        bs.inputs["Coat Roughness"].default_value = 0.25
        bs.inputs["Sheen Tint"].default_value = (0.75, 0.80, 0.90, 1.0)
        ch.mat("zip", M.metal(pre + "zip", "#c4c7cd", 0.28, holes=hg))
        jk = NK.leather_jacket(ch)
        ch.add(jk["leather"], "jacket", ["leather"], solid=0.007, jig=True)
        ch.add(jk["lapels"], "jacket_lapels", ["leather"], jig=True, bevel=0.0015)
        ch.add(jk["collar"], "jacket_collar", ["leather"])
        ch.add(jk["metal"], "jacket_zip", ["zip"])

    # ---------------------------------------------------------------- hair
    hs = v["hair"]
    if hs == "short":
        NK.head_part(ch, NK.hair_shell(ch, 1.655, 1.700, lambda z: 0.010 + 0.006 * G.smoothstep(1.70, 1.75, z)), "hair", ["hair"])
    elif hs == "flattop":
        NK.head_part(ch, NK.hair_shell(ch, 1.650, 1.706, lambda z: 0.007 + 0.034 * G.smoothstep(1.690, 1.750, z), zmax=1.768), "hair", ["hair"])
    elif hs == "buzz":
        NK.head_part(ch, NK.hair_shell(ch, 1.640, 1.690, lambda z: np.full_like(z, 0.0045)), "hair", ["hair"])
    elif hs == "bald":
        NK.head_part(ch, O.hair_band(ch, 1.560, 1.648, 0.008, back_only=True, front_cut=-0.035), "hair", ["hair"])
    elif hs == "ponytail":
        band = O.hair_band(ch, 1.560, 1.700, 0.010, back_only=True, front_cut=-0.020)
        band.v = ch.T(band.v)
        pt, ring = NK.ponytail(ch)
        ch.add(G.merge([band, pt]), "hair", ["hair"])
        ch.mat("hair_tie", M.clay(pre + "hair_tie", v["acc_col"][1], rough=0.5, bump=0.1))
        ch.add(ring, "hair_tie", ["hair_tie"])
    elif hs == "afro":
        base = NK.hair_shell(ch, 1.640, 1.695, lambda z: np.full_like(z, 0.012))
        base.v = ch.T(base.v)
        ch.add(G.merge([base, NK.afro_blobs(ch)]), "hair", ["hair"])
    elif hs == "bob":
        NK.head_part(ch, O.hair_bob(ch), "hair", ["hair"])

    # ---------------------------------------------------------------- face hair, glasses
    for f in v["face"]:
        if f in ("beard", "goatee"):
            ch.mat("beard", M.clay(pre + "beard", v["beard_col"], rough=0.8, bump=0.35))
            NK.face_shell(ch, NK.beard_select(ch, "full" if f == "beard" else "goatee"), "beard", ["beard"])
        elif f == "moustache":
            NK.moustache(ch, "moustache", ["hair"])
        elif f.startswith("glasses"):
            ch.mat("glasses", M.gloss(pre + "glasses", v["glasses_col"], rough=0.25, spec=0.6))
            if f == "glasses_round":
                gp = NK.glasses(ch, "round", rx=0.060, rz=0.054, fwd=0.052, tube_r=0.0085)
            else:
                gp = NK.glasses(ch, "rect", rx=0.054, rz=0.036, fwd=0.050, tube_r=0.0068)
            ch.add(gp, "glasses", ["glasses"])

    # ---------------------------------------------------------------- vendor accessory (p_accessory toggles it)
    acc = v.get("acc")
    acc_objs = []
    if acc == "lanyard":
        ch.mat("acc_a", M.clay(pre + "lanyard", v["acc_col"][0], rough=0.6, bump=0.1, holes=hg))
        ch.mat("acc_b", M.clay(pre + "badge", v["acc_col"][1], rough=0.45, bump=0.05, holes=hg))
        ch.mat("acc_c", M.clay(pre + "badge_stripe", v["acc_col"][2], rough=0.45, bump=0.05, holes=hg))
        cord, badge, stripe, clip = NK.lanyard(ch)
        acc_objs += [ch.add(cord, "lanyard", ["acc_a"], jig=True), ch.add(badge, "badge", ["acc_b"], jig=True),
                     ch.add(stripe, "badge_stripe", ["acc_c"], jig=True), ch.add(clip, "badge_clip", ["zip" if "zip" in ch.m else "acc_a"], jig=True)]
    elif acc == "headset":
        ch.mat("acc_a", M.clay(pre + "headset", v["acc_col"][0], rough=0.45, bump=0.05, sheen=0.3))
        ch.mat("acc_b", M.metal(pre + "headset_metal", v["acc_col"][1], 0.3))
        dark, boom, tip = NK.headset(ch)
        acc_objs += [ch.add(dark, "headset", ["acc_a"]), ch.add(boom, "headset_boom", ["acc_b"]), ch.add(tip, "headset_mic", ["acc_a"])]
    elif acc == "cap":
        ch.mat("acc_a", M.clay(pre + "cap", v["acc_col"][0], rough=0.6, bump=0.2, sheen=0.3))
        ch.mat("acc_b", M.clay(pre + "cap_brim", v["acc_col"][1], rough=0.6, bump=0.2))
        crown, brim, btn = NK.cap(ch)
        acc_objs += [ch.add(crown, "cap", ["acc_a"], solid=0.006), ch.add(brim, "cap_brim", ["acc_b"], solid=0.010, bevel=0.002),
                     ch.add(btn, "cap_button", ["acc_b"])]
    elif acc == "backpack":
        ch.mat("acc_a", M.clay(pre + "backpack", v["acc_col"][0], rough=0.6, bump=0.25, holes=hg, sheen=0.3))
        ch.mat("acc_b", M.clay(pre + "backpack_strap", v["acc_col"][1], rough=0.6, bump=0.2, holes=hg))
        pack, straps = NK.backpack(ch)
        acc_objs += [ch.add(pack, "backpack", ["acc_a"], jig=True), ch.add(straps, "backpack_straps", ["acc_b"], jig=True)]
    if acc_objs:
        NK.accessory_toggle(ch, acc_objs)

    # ---------------------------------------------------------------- hooks (v1 names and meaning)
    O.hook_hand(ch, "L", 1)
    O.hook_hand(ch, "R", -1)
    for nm, sn, side in (("gun_grip_R", "R", -1), ("gun_support_L", "L", 1)):
        f, n, c = ch.hand_frames[sn]
        wr = ch.J["wr"] * np.array([side, 1, 1])
        ch.hook(nm, "hand_" + sn, wr + 0.060 * f + n * 0.020, rotm=np.stack([np.cross(f, n), f, n], axis=1))
    top = 0.0
    for nm in ("head", "hair", "cap", "cap_button", "headset"):
        ob = ch.objs.get(nm)
        if ob is not None:
            top = max(top, max(vv.co.z for vv in ob.data.vertices) / K)
    ch.hook("head_top", "head", (0.0, 0.005, top + 0.005))
    ch.hook("hug_target", "root", (0.0, -0.30, 0.65))
    ch.build_rims(ch.m["rim"])
    ch.root["p_stature_m"] = v["H"]
    ch.root["p_scale"] = 1.0
    ch.root["pv_face_z"] = float(K * (spec["chin_post"] + 0.45 * 0.233 * spec["headS"][2]))
    ch.root["pv_face_dist"] = float(3.0 * K * 0.233 * spec["headS"][2])
    return ch, acc_objs


HOLE_USAGE = dict(
    how_it_works="hole_1 is an Empty (HOLE_hole_1, parented to bone chest) whose local Z axis is the hole axis (front to back). The shared node group NG_bullet_holes_<asset> in every body and clothing material sets alpha 0 inside radius R = base_radius_m x root['p_hole_1_radius'] and |local z| < cutout_half_length_m; the rim tube <asset>_hole_1_rim is scaled by the same property.",
    to_animate="Keyframe ROOT_<asset>['p_hole_1_radius'] (0 intact .. 1 full base radius).",
    defaults="0.0 (intact).",
    render_notes="Materials use DITHERED render method with alpha cutout. v2 base radius 0.045 m (v1 0.042) and cut half-length 0.30 m (v1 0.14), scaled by K (stature / 1.75), so belly, wordmark, lanyard, jacket and backpack layers are cut too.")


def meta(ch, var, acc_objs):
    v = VARIANTS[var]
    cid = var + "_v2"
    m = B.meta_common(ch)
    m.update(description="NPC v2 (%s, chunky clay): %s. Same rig, hooks, root props, shape keys, hole_1 and action names as %s (v1)." % (var, v["look"], var),
             variant={k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items() if k not in ("extra_bones",)},
             sources=[dict(what="Joint table, H and build identical to v1 %s (skeleton compatibility)" % var, used_for="skeleton", access="2026-10-03", level="C"),
                      dict(what="Design values (head scale headS, wide, arm_r, leg_r, belly, hair, accessories)", used_for="v2 silhouette", access="2026-10-03", level="C")])
    notes = [
        "NPC v2 family: npc_v2 (blank grey tee), vendors npc_molexx_v2 (red, lanyard badge), npc_nubiss_v2 (green, headset), npc_terahop_v2 (purple, cap), npc_ayarr_v2 (yellow, backpack); customers npc_nvydia_v2 (black leather moto jacket over the green NVYDIA tee), npc_openay_v2 (0.8-scale hugger, OPENAY over ANTHROPY). Shirt and skin colours are the v1 values (s03_fx.SHIRT / SKIN crumbs still match) except the customers (lighter, see variant).",
        "Switch a scene: asm.append('characters/%s', actions=True) instead of 'characters/%s'; asm.play / asm.walk resolve ACT_%s_<name> from root['asset_id']. Object names carry the _v2 id (e.g. %s_rig, ROOT_%s); code that hard-codes v1 names must change." % (cid, var, cid, cid, cid),
        "Vendor accessory: root['p_accessory'] (1 shown, 0 hidden) drives hide_render/hide_viewport of the accessory objects %s." % [o.name for o in acc_objs] if acc_objs else "No accessory on this asset (p_accessory not present).",
        "Face hair (beard/goatee/moustache) carries the same 14 expression shape keys, generated from the head deformation and driven by the same root props, so it follows every expression; glasses, headset and cap are rigid on the head (cap on bone 'hat', p_jiggle_hat works for npc_terahop_v2).",
        "Wordmarks are raised parody text meshes (Arial Black outlines converted at build time; no font needed at render). No real logos.",
        "Untucked tee: the shirt hem (rolled ring at z 0.90 baseline) flares over the trouser waist and the thigh tops; the trousers pelvis tapers into the legs (v1 diaper seam removed).",
    ]
    if var == "npc_nvydia":
        notes.append("Jacket objects npc_nvydia_v2_jacket / _jacket_lapels / _jacket_collar / _jacket_zip are skinned to the armature (torso weights hips..chest, sleeves upper_arm/forearm); leather material = dark clay with sheen 0.40 (tint light blue-grey) and coat 0.35 for rim highlights on dark sets; the open V (from z 1.10 baseline) shows the green tee with the NVYDIA wordmark. Replaces both npc_nvydia and npc_nvydia_leather.")
    m = MV.extend(m, ch, "npc", notes)
    m["hole_system_usage"] = HOLE_USAGE
    m["dimensions"] = [dict(item="stature, skull crown (m)", value=v["H"], provenance="design value, unchanged from v1 %s" % var, accuracy="C"),
                       dict(item="head height (m)", value=m["measured"]["head_height_m"], provenance="measured on the built asset", accuracy="B")]
    m["v1_counterpart"] = [var] + (["npc_nvydia_leather"] if var == "npc_nvydia" else [])
    return m


if __name__ == "__main__":
    args = C.argv_after_dashes()
    OUT, VAR = args[0], args[1]
    NO_ACTIONS = "no_actions" in args
    ch, acc_objs = build(VAR)
    m = meta(ch, VAR, acc_objs)
    if not NO_ACTIONS:
        names = A.make_all(ch.arm, ch.K, ch.J, ch.hand_frames, VAR + "_v2")
        m["actions"] = A.describe(names)
    bpy.context.view_layer.update()
    C.finish(VAR + "_v2", os.path.join(OUT, VAR + "_v2.blend"), ch.coll, m, preview_dir=None)
