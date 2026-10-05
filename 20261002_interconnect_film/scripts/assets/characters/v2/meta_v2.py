"""Metadata for the v2 characters: measured values from the built scene plus assembler notes (written into the JSON)."""
import bpy


def _zs(ch, pred):
    zs = []
    dg = bpy.context.evaluated_depsgraph_get()
    for o in ch.coll.all_objects:
        if o.type == "MESH" and pred(o):
            e = o.evaluated_get(dg)
            me = e.to_mesh()
            zs += [(o.matrix_world @ v.co).z for v in me.vertices]
            e.to_mesh_clear()
    return zs


def eval_tris(ch):
    """Triangles of the evaluated meshes with subdivision at the render level."""
    n = 0
    for o in ch.coll.all_objects:
        if o.type != "MESH":
            continue
        for m in o.modifiers:
            if m.type == "SUBSURF":
                m.levels = m.render_levels
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    for o in ch.coll.all_objects:
        if o.type != "MESH":
            continue
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        n += sum(len(p.vertices) - 2 for p in me.polygons)
        e.to_mesh_clear()
    return n


def measured(ch):
    head = _zs(ch, lambda o: o.name.endswith("_head"))
    crown = _zs(ch, lambda o: "_rim" not in o.name and not o.name.endswith("_hardhat"))
    allz = _zs(ch, lambda o: "_rim" not in o.name)
    hh = max(head) - min(head)
    return dict(head_height_m=round(hh, 4), head_chin_z_m=round(min(head), 4), head_crown_z_m=round(max(head), 4),
                top_without_hat_m=round(max(crown), 4), top_with_hat_m=round(max(allz), 4),
                head_height_over_stature=round(hh / max(crown), 4), stature_over_head_height=round(max(crown) / hh, 2))


def material_slots(ch):
    return {o.name: [s.material.name for s in o.material_slots if s.material] for o in ch.coll.all_objects if o.type == "MESH" and o.material_slots}


COMMON_NOTES = [
    "v2 is a drop-in for v1: same bone names and hierarchy (bones added: belly, plus hat for Gary and tie_1..3 for the Manager), same HOOK_* empties, same root props, same shape-key names on the same objects, same hole system, same 21 action names with the asset id in the name (ACT_<asset_id>_<action>, e.g. ACT_gary_v2_walk).",
    "To switch a scene: append characters/gary_v2 (or manager_v2, or gary_v2_holes30) instead of characters/gary; asm.play/walk build the action name from root['asset_id'], so 'walk' resolves to ACT_gary_v2_walk. Any code that hard-codes 'ACT_gary_...' or the object names 'gary_rig' / 'ROOT_gary' must use the _v2 names.",
    "Limb bones, hand frames, finger bones, IK rest and all hand hooks are identical to v1 (rest positions equal within 1e-4 m), so v1 actions (and any action authored on the v1 skeleton) retarget by bone name. Changed rest positions (bigger head): neck tail, head head, jaw, eye_* and lid_* only. The head-bone pivot is 0.077 m lower than v1 (centre of the bigger head), so a v1 head rotation of theta rad displaces the head-bone tail by about 0.077 x theta m (measured worst case over all v1 actions: 0.03 m).",
    "Expressions: root['p_expr_<name>'] for worried, sweating, dread, flat, sobbing, shouting, smug, shock, dead_eyed, angry, scared, happy, yell (0..1); root['p_anger'] (angry + yell + nostril flare + forehead veins), root['p_flush']. Shape keys are on objects head, pupils, glints (new), lids, brows, nose, lips (new), teeth_up, teeth_lo, tongue, drops; keys are driven from the root, do not keyframe them directly. Expression deltas are exaggerated (gain 1.15..1.7) relative to v1 so they read at 200 px head height. Neutral mouth is slightly open (gap 7 mm in v1 head units) so lips and mouth read as a smile line.",
    "Squash and stretch: root['p_squash'] (-0.6..1) scales the whole body about the floor along Z by (1+p) and X, Y by 1/sqrt(1+p) (volume kept; positive = stretch); root['p_squash_head'] does the same for the head about the head pivot. Drivers act on pose-bone scale of 'root' and 'head' (bone Y is the long axis), so keep scale out of your own keys on those bones. Bone-parented hooks and holes scale with the body; keep |p| below about 0.3 when a prop is attached.",
    "Jiggle: root['p_jiggle_belly'] (m, vertical offset of bone 'belly', weights on trunk, clothing and belt), root['p_jiggle_hat'] (rad about X, Gary hat bone), root['p_tie_swing'] (rad, Manager tie chain tie_1..3). Rest values are 0; key them with damped oscillation for secondary motion.",
    "Fingers: v1 left the finger bones in quaternion mode so the euler curl drivers did nothing; v2 sets XYZ mode, so ctl_hand_L/R custom props curl_index, curl_middle, curl_ring, curl_pinky, curl_thumb (0..1) now close the hand (mitten: thumb, index, a merged middle+ring block and pinky; middle and ring bones both drive the block). Actions keep their v1 curl keys, so hands that were visually open in v1 now curl by the authored amounts.",
    "Face preview helpers on the root (not part of the contract): pv_face_z, pv_face_dist.",
    "Holes: same empties, same cut-out semantics (Texture Coordinate on the hole empty; radius = base radius x root property; rim tube scaled by the same property). Base radius is 0.045 m (v1 0.042) and the cut half-length 0.30 m (v1 0.14) because the belly and the outer clothing layers (belt, creases) reach 0.23 m in front of the hole axis.",
]

SIMPL = [
    "Clay look is procedural: matte Principled materials plus the NG_clay_bump thumbprint group; cloth folds are modelled crease tubes plus a z-banded noise bump (NG_folds_*), not simulated.",
    "Limbs are rigid per-bone capsules with ball joints (no weight-blended elbows/knees); clothing sleeves and legs follow the same bones, so extreme bends show the capsule overlap.",
    "Head model is the v1 head mapped by an affine transform (x 1.9, y 1.42, z 1.68 about the chin, shifted 0.02 m forward); eye, lid, lip, glint and tooth geometry were re-proportioned in v1 head units before the transform.",
    "Gary's t-shirt hem and the Manager's shirt bottom are hidden under the belt/jacket (shirt shells start at z 1.0 / 0.96 baseline); shells are open at the bottom.",
    "Hair is a shell over the head surface with a few clay blobs; no strand detail.",
    "Hands are mittens with four finger bones; the middle and ring fingers share one geometry block.",
    "Real-scale dimensions are design values (Gary crown 1.69 m, hat top 1.753 m; Manager 1.85 m skull crown), not measured from a source.",
]


def extend(meta, ch, kind, extra_notes=None):
    m = measured(ch)
    meta["measured"] = m
    meta["evaluated_triangles_with_modifiers"] = eval_tris(ch)
    meta["material_slots"] = material_slots(ch)
    meta["accuracy_level"] = "C (stylised clay; design-value proportions)"
    meta["dimension_table"] = [
        dict(item="head height (chin to skull crown, m)", value=m["head_height_m"], provenance="measured from the built asset (evaluated head mesh)", accuracy="B"),
        dict(item="stature / head height", value=m["stature_over_head_height"], provenance="measured; v1 was about 7.5", accuracy="B"),
        dict(item="top of skull/hair without hat (m)", value=m["top_without_hat_m"], provenance="measured", accuracy="B"),
        dict(item="top with hat (m)", value=m["top_with_hat_m"], provenance="measured", accuracy="B"),
        dict(item="stature design value (m)", value=meta["root_custom_properties"].get("p_stature_m"), provenance="design: unchanged from v1", accuracy="C"),
    ]
    meta["simplifications"] = SIMPL
    meta["assembler_notes"] = COMMON_NOTES + (extra_notes or [])
    meta["origin"] = "ROOT_%s at the floor between the feet; Z up, front = -Y, character left = +X; identity transforms on all meshes; armature object %s_rig is a child of the root" % (ch.id, ch.id)
    return meta
