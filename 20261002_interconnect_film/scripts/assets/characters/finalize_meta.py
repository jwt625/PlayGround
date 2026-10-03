"""Extend the JSON metadata of every character asset with sources, dimensions, simplifications and assembler notes.
Run with plain python3 from the project root after the builds: python3 scripts/assets/characters/finalize_meta.py"""
import glob
import json
import os
import sys

OUT = "assets/components/characters"
TMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

COMMON_SOURCES = [
    dict(what="Segment lengths and joint heights as fractions of stature (Drillis and Contini 1966 table, as commonly reproduced)",
         used_for="skeleton: shoulder 0.818 H, elbow 0.63 H, wrist 0.485 H, hip 0.53 H, knee 0.285 H, ankle 0.039 H; upper arm 0.186 H, forearm 0.146 H, hand 0.108 H, thigh 0.245 H, shank 0.246 H",
         access="not re-fetched in this build; values rounded from memory of the standard table; verify before quoting", accuracy="C"),
    dict(what="Typical adult head, torso and limb girths (general anthropometry, rounded)", used_for="head 0.233 m tall x 0.156 m wide x 0.19 m deep at 1.75 m stature, torso and limb radii",
         access="estimates, +-15 percent", accuracy="C"),
    dict(what="Design brief: DevLog-001 Section 3 (cast) and Section 4 (storyboard rev 5), scripts/blender_lib.py (crude v0.2 behaviour)", used_for="cast, holes, hooks, actions", access="2026-10-02", accuracy="A"),
]
SIMPL = [
    "Clay look: stylised, not photo-real; eyes are enlarged (radius 16.5 mm baseline vs about 12 mm real) and the mouth is a modelled slit with a dark bag, teeth bar and tongue.",
    "Limbs are rigid capsules with round caps (ball joints), no weight-painted skinning; torso uses smooth spine weights.",
    "Hands: palm ellipsoid plus 3-phalanx fingers and thumb; finger curl is driven by custom properties on ctl_hand_L / ctl_hand_R, not by per-bone keys.",
    "Hole system: through-and-through cutout is a cylinder in the space of a hole empty (local Z axis); the rim tube follows the surface at the base radius only, so at small radii the tube ends sit slightly inside the surface.",
    "Clothing is thin shells slightly offset from the skin; no cloth simulation; boots/shoes are lofted blobs without a toe bone.",
    "Actions are authored by IK targets and FK angles in baseline coordinates scaled by the rig's K; walk and run are in place (move the ROOT at the action's root_speed_mps).",
]


def hole_notes(meta):
    if not meta.get("holes"):
        return None
    return dict(
        how_it_works="Each hole is an Empty (HOLE_<name>, parented to a bone) whose local Z axis is the hole axis. A shared shader node group (NG_bullet_holes_<asset>) tests every body/clothing material: alpha = 0 where the point is inside radius R and |local z| < cutout_half_length_m. R = base_radius_m * root[prop]. A thin rim tube object (<asset>_<hole>_rim, parented to the empty) shows the clay thickness and is scaled by the same property.",
        to_animate="Set root['p_hole_k_radius'] (k = 1..5) or root['p_head_hole_radius'] (0 = intact, 1 = full base radius, never reach 0 after the hit to follow the storyboard; 0.3 = the 30 percent floor). Keyframe it on ROOT_<asset> (custom property). Shrinking in steps: x0.9 every 1.5 s down to 0.3.",
        defaults="all hole properties default to 0.0 (intact) except in the gary_holes30 asset (0.3).",
        render_notes="Materials use DITHERED render method with alpha cutout; for hole shadows enable Transparent Shadows (already set). A straight-on view shows the background through the hole; oblique views show the rim tube wall because the torso is about 0.22 m deep.",
    )


def main():
    for jp in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        aid = os.path.basename(jp)[:-5]
        m = json.load(open(jp))
        dm = os.path.join(TMP, "mat_%s.json" % aid)
        if os.path.exists(dm):
            x = json.load(open(dm))
            m["material_slots"] = x["materials"]
            m["evaluated_triangles_with_modifiers"] = x["evaluated_triangles"]
        m["sources"] = COMMON_SOURCES
        m["simplifications"] = SIMPL
        m["accuracy_level"] = "C (stylised clay; proportions from standard segment fractions)"
        sz = m.get("size_mm")
        m["dimension_table"] = [
            dict(item="bounding box of undeformed meshes at rest (mm, x y z)", value=sz, provenance="measured from the built asset", accuracy="B"),
            dict(item="stature (crown, no hat) m", value=m.get("root_custom_properties", {}).get("p_stature_m"), provenance="design choice per character (Gary 1.75 with hard hat, Manager 1.85, vendors/customers 1.6-1.9, hugger 0.8 x 1.75)", accuracy="C"),
            dict(item="scale factor K versus the 1.75 m baseline", value=m.get("scale_K"), provenance="stature / 1.75", accuracy="A"),
        ]
        m["origin"] = "ROOT_%s at the floor between the feet; Z up, front = -Y, character left = +X; identity transforms on all meshes; armature object %s_rig is a child of the root" % (aid, aid)
        m["assembler_notes"] = [
            "Expressions: set root['p_expr_<name>'] (0..1) for worried, sweating, dread, flat, sobbing, shouting, smug, shock, dead_eyed, angry, scared, happy, yell. Shape keys exist on the objects: head, pupils, lids, brows, nose, teeth_up, teeth_lo, tongue, drops (sweat drops and tears appear with sweating, dread, scared, sobbing). Keys are driven from the root; do not keyframe the keys directly.",
            "root['p_anger'] (0..1) drives the angry key (brows, lids, snarl), the yell key (mouth open from p_anger 0.35 up), nostril flare and the forehead vein bump (skin shader); root['p_flush'] (0..1) mixes the red flush into the skin above neck height.",
            "Rig: armature <asset>_rig. FK: root, hips, spine_1, spine_2, chest, neck, head, jaw (open = +X rotation), clavicle_L/R, upper_arm, forearm, hand, thigh, shin, foot. IK: ik_hand_L/R, ik_elbow_L/R (pole), ik_foot_L/R, ik_knee_L/R (pole); switch with ctl_settings['ik_arm_L'] etc (1 = IK, 0 = FK). Eyes: eye_L/R track ctl_look (ctl_settings['look_at'] = influence). Lids: lid_up_L/R and lid_lo_L/R rotate about X (upper lid closes with +X). Fingers: ctl_hand_L / ctl_hand_R custom props curl_index, curl_middle, curl_ring, curl_pinky, curl_thumb (0..1).",
            "Hooks are Empties named HOOK_*, bone-parented. Hand hooks (HOOK_hand_L/R, gun/whip/paper grips): origin on the palm, local Y along the fingers, local Z along the palm normal (the direction the palm faces).",
            "Actions are stored with fake users: ACT_%s_<name>. Loops carry a Cycles modifier; assign with armature.animation_data.action. Frame rate 30." % aid,
        ]
        h = hole_notes(m)
        if h:
            m["hole_system_usage"] = h
        json.dump(m, open(jp, "w"), indent=2, sort_keys=True)
        print("updated", jp)


main()
