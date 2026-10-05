"""Post-build touch-up of the v2 JSON metadata (plain python3, run from the project root after run_all_v2.sh).

python3 scripts/assets/characters/v2/finalize_meta_v2.py
Adds the v1 keys that the builders do not write (hole_system_usage, dimensions, hole defaults) and corrects notes found in the audit.
"""
import json
import os

D = "assets/components/characters"
V1 = {"gary_v2": "gary", "gary_v2_holes30": "gary_holes30", "manager_v2": "manager"}

HOLE_USAGE = dict(
    how_it_works="Each hole is an Empty (HOLE_<name>, parented to a bone) whose local Z axis is the hole axis. A shared shader node group (NG_bullet_holes_<asset>) is evaluated in every body and clothing material: alpha = 0 where the point is inside radius R and |local z| < cutout_half_length_m. R = base_radius_m x root property. A thin rim tube object (<asset>_<hole>_rim, parented to the empty) shows the clay thickness and is scaled by the same property.",
    to_animate="Keyframe root['p_hole_k_radius'] (k = 1..5) or root['p_head_hole_radius'] on ROOT_<asset> (0 = intact, 1 = full base radius, floor 0.3 after the hit per the storyboard; schedule r = max(0.3, 0.9 ** ((T_now - T_shot) / 1.5))).",
    defaults="all hole properties default to 0.0 (intact) except in gary_v2_holes30 (0.3).",
    render_notes="Materials use DITHERED render method with alpha cutout and transparent shadows. Torso holes cut 0.30 m each side of the hole plane (belly, belt and crease layers reach 0.23 m in front); the head hole cuts 0.26 m each side (hair, head, hat).",
)


def main():
    for aid, v1 in V1.items():
        p = os.path.join(D, aid + ".json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        if m.get("holes"):
            m["hole_system_usage"] = HOLE_USAGE
        m["dimensions"] = [dict(item="stature without hat (m)", value=m["root_custom_properties"].get("p_stature_m"), provenance="design value, unchanged from v1 (Gary crown 1.69 m / 1.75 m with hat, Manager 1.85 m)", accuracy="C"),
                           dict(item="head height (m)", value=m["measured"]["head_height_m"], provenance="measured on the built asset", accuracy="B")]
        notes = []
        for n in m["assembler_notes"]:
            if n.startswith("HOOK_muzzle_self is at the same place as v1"):
                n = ("HOOK_muzzle_self keeps the v1 x and y (0.21 m to the character's right of the head axis) and is 0.016 m lower than v1 (z 1.593 m instead of 1.610 m) so it sits below the hat brim at the temple; the bigger head leaves about 0.06 m between the head side and the hook.")
            notes.append(n)
        notes.append("Manager measured block: top_with_hat_m equals top_without_hat_m because the Manager has no hat (it is the hair top, 1.862 m); head_crown_z_m is the skull crown (1.849 m); the design stature 1.85 m refers to the skull crown.")
        m["assembler_notes"] = notes
        json.dump(m, open(p, "w"), indent=2, sort_keys=True)
        print("updated", p)


main()
