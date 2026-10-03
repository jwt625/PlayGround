"""AOC whip: bundle of six active optical cables (QSFP-DD style plugs both ends), 3.0 m, 24-bone chain rig.

Run: Blender -b --python scripts/assets/props/build_aoc_whip.py -- assets/components/props

Axes: whip lies along -Y, handle end at y = 0 (origin = grip end, mounting origin), tip end at y = -3.0 m. Z up.
Plug local frame (template): origin at the cable entry (rear of the boot), +Y toward the insertion end, X = width, Z = height.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import props_lib as L  # noqa: E402
from props_lib import C, mm  # noqa: E402

OUT, PREV = L.outdirs()
N_BONES = 24
LEN = 3.0
BL = LEN / N_BONES  # 0.125 m
PLUG_L = 89.4  # mm, module length (QSFP-DD MSA drawing dimension string, not figure-verified)
PLUG_W, PLUG_H = 18.35, 8.5
R_PLUG = 21.0  # mm, plug centre radius in the fan-out
R_BUNDLE = 3.0  # mm, cable centre radius in the bundle (touching ring of six 3 mm cables)
S_FAN0 = PLUG_L / 1000.0  # end of plug zone, m
S_FAN1 = 0.20  # fan-out completes (handle side)
S_TAPE0, S_TAPE1 = 0.20, 0.45
S_TIES = [0.45 + 0.30 * k for k in range(8)]  # 0.45 ... 2.55 m
COLORS = [("red", (0.75, 0.05, 0.04)), ("orange", (0.95, 0.38, 0.03)), ("yellow", (0.95, 0.8, 0.05)),
          ("green", (0.1, 0.6, 0.12)), ("blue", (0.05, 0.25, 0.8)), ("violet", (0.45, 0.12, 0.65))]


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def radius_at(s):
    """Radial offset (mm) of a cable centre from the whip axis at distance s (m) from the handle end."""
    if s <= S_FAN0:
        return R_PLUG
    if s < S_FAN1:
        return R_PLUG + (R_BUNDLE - R_PLUG) * smoothstep(S_FAN0, S_FAN1, s)
    s2 = LEN - s
    if s2 <= S_FAN0:
        return R_PLUG
    if s2 < S_FAN1 + 0.0 and s > LEN - S_FAN1:
        return R_PLUG + (R_BUNDLE - R_PLUG) * smoothstep(S_FAN0, S_FAN1, s2)
    return R_BUNDLE


def cable_path(k, n_step=0.01):
    phi = math.radians(30 + 60 * k)
    pts = []
    s = S_FAN0
    s_end = LEN - S_FAN0
    while s < s_end - 1e-9:
        pts.append(s)
        s += n_step
    pts.append(s_end)
    out = []
    for s in pts:
        r = radius_at(s)
        out.append(Vector((mm(r * math.cos(phi)), -s, mm(r * math.sin(phi)))))
    return out


def plug_template(mats):
    """Return dict name -> mesh data of the plug parts in the plug local frame (mm -> m)."""
    meshes = {}
    ux, vz = Vector((1, 0, 0)), Vector((0, 0, 1))

    def loft_super(name, secs, p, mat, n=28):
        bm = bmesh.new()
        rings = []
        for y, w, h in secs:
            pts = [(mm(a), mm(b)) for a, b in L.superellipse2d(w, h, n, p)]
            # path +Y: u x v = +Y needs u = Z, v = X
            rings.append(L.ring_frame(Vector((0, mm(y), 0)), Vector((0, 0, 1)), Vector((1, 0, 0)), [(b, a) for a, b in pts]))
        L.loft_bm(bm, rings, True, True)
        o = L.obj_from_bm(name, bm, mat, True, True, 45)
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        return me
    meshes["boot"] = loft_super("plug_boot", [(0, 5.0, 5.0), (3, 6.5, 6.0), (12, 10.5, 7.0), (22, 15.0, 7.8), (27, 17.6, 8.1)], 2.4, mats["boot"])
    meshes["housing"] = loft_super("plug_housing", [(26.5, 17.9, 8.2), (28, PLUG_W, PLUG_H), (74, PLUG_W, PLUG_H), (80, PLUG_W - 0.3, PLUG_H - 0.3), (80.5, PLUG_W - 1.0, PLUG_H - 1.0)], 4.0, mats["housing"])
    # PCB paddle
    bm = bmesh.new()
    L.box_bm(bm, (mm(16.8), mm(PLUG_L - 79.0), mm(0.9)), (0, mm((PLUG_L + 79.0) / 2), 0), mm(0.1), 1)
    o = L.obj_from_bm("plug_pcb", bm, mats["pcb"], False, True, 50)
    meshes["pcb"] = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    # gold pads: 2 faces x 2 rows x 19 pads (estimate of the layout), pitch 0.8 mm
    bm = bmesh.new()
    for zs in (1, -1):
        for row, (y0, ln) in enumerate(((PLUG_L - 3.6, 2.2), (PLUG_L - 7.4, 2.8))):
            for i in range(19):
                x = (i - 9) * 0.82
                L.box_bm(bm, (mm(0.5), mm(ln), mm(0.1)), (mm(x), mm(y0), mm(zs * 0.5)), 0, 1)
    o = L.obj_from_bm("plug_pads", bm, mats["gold"], False, True, None)
    meshes["pads"] = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    # pull tab: strap above the housing rear + finger loop
    bm = bmesh.new()
    L.box_bm(bm, (mm(8.0), mm(62.0), mm(0.9)), (0, mm(3.0), mm(PLUG_H / 2 + 0.5)), mm(0.25), 1)
    L.lathe_bm(bm, [(mm(3.2), -mm(0.45)), (mm(6.0), -mm(0.45)), (mm(6.0), mm(0.45)), (mm(3.2), mm(0.45))], 28,
               xf=Matrix.Translation((0, mm(-28.0), mm(PLUG_H / 2 + 0.5))) @ Matrix.Diagonal((1.0, 1.45, 1.0, 1.0)), closed_profile=True)
    o = L.obj_from_bm("plug_tab", bm, mats["tab"], True, True, 50)
    meshes["tab"] = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    return meshes


def weights_for(s):
    """Bone index weights for a vertex at distance s (m) from the handle end: linear blend between adjacent bone centres."""
    idx = s / BL - 0.5
    i0 = int(math.floor(idx))
    t = idx - i0
    if i0 < 0:
        return [(0, 1.0)]
    if i0 >= N_BONES - 1:
        return [(N_BONES - 1, 1.0)]
    return [(i0, 1.0 - t), (i0 + 1, t)]


def skin(obj, arm):
    for i in range(N_BONES):
        obj.vertex_groups.new(name="whip_%02d" % (i + 1))
    groups = {}
    for v in obj.data.vertices:
        s = -v.co.y
        for i, w in weights_for(s):
            if w > 1e-4:
                groups.setdefault((i), []).append((v.index, w))
    for i, lst in groups.items():
        g = obj.vertex_groups["whip_%02d" % (i + 1)]
        for vi, w in lst:
            g.add([vi], w, "REPLACE")
    m = obj.modifiers.new("Armature", "ARMATURE")
    m.object = arm
    obj.parent = arm


def build():
    C.reset()
    coll, root = C.new_asset("aoc_whip", accuracy="B")
    mats = dict(
        boot=C.principled("MAT_props_aoc_boot", (0.04, 0.04, 0.045), rough=0.6),
        housing=C.principled("MAT_props_aoc_housing", (0.62, 0.63, 0.65), metallic=1.0, rough=0.38),
        pcb=C.principled("MAT_props_aoc_pcb", (0.02, 0.15, 0.06), rough=0.45),
        gold=C.principled("MAT_props_aoc_gold", (0.85, 0.65, 0.2), metallic=1.0, rough=0.25),
        tab=None,
        tie=C.principled("MAT_props_aoc_tie", (0.03, 0.03, 0.03), rough=0.5),
        tape=C.principled("MAT_props_aoc_tape", (0.015, 0.015, 0.016), rough=0.38, coat=0.2),
    )
    jackets = {n: C.principled("MAT_props_aoc_jacket_" + n, c, rough=0.42) for n, c in COLORS}
    tabs = {n: C.principled("MAT_props_aoc_tab_" + n, c, rough=0.5) for n, c in COLORS}
    mats["tab"] = tabs["red"]
    tpl = plug_template(mats)
    # ------------------------------------------------------------ armature (chain along -Y)
    bones = []
    for i in range(N_BONES):
        bones.append(dict(name="whip_%02d" % (i + 1), head=(0, -BL * i, 0), tail=(0, -BL * (i + 1), 0),
                          parent=("whip_%02d" % i) if i else None, roll_axis=(0, 0, -1)))
    arm = L.make_armature("aoc_whip_rig", bones, coll, root, display="STICK")
    stiff = []
    for i in range(N_BONES):
        st = 1.0 if i < 4 else max(0.05, math.exp(-(i - 3) / 7.0))
        stiff.append(st)
        arm.pose.bones["whip_%02d" % (i + 1)]["stiffness"] = st
        arm.data.bones["whip_%02d" % (i + 1)]["stiffness"] = st
    # ------------------------------------------------------------ cables
    for k, (cn, _) in enumerate(COLORS):
        pts = cable_path(k)
        bm = bmesh.new()
        prof = [(mm(a), mm(b)) for a, b in L.circle2d(1.5, 14)]
        L.tube_path_bm(bm, pts, prof, binormal=Vector((1, 0, 0)), cap=False)
        o = L.obj_from_bm("aoc_whip_cable_%d_%s" % (k + 1, cn), bm, jackets[cn], True, True, None)
        C.add(o, coll, root)
        skin(o, arm)
    # ------------------------------------------------------------ plugs (linked duplicates share mesh data)
    for end in (0, 1):
        for k, (cn, _) in enumerate(COLORS):
            phi = math.radians(30 + 60 * k)
            rad = Vector((math.cos(phi), 0, math.sin(phi)))
            tan = Vector((-math.sin(phi), 0, math.cos(phi)))
            if end == 0:
                outward = Vector((0, 1, 0))
                xax = -tan
                origin = Vector((mm(R_PLUG) * rad.x, -mm(PLUG_L), mm(R_PLUG) * rad.z))
            else:
                outward = Vector((0, -1, 0))
                xax = tan
                origin = Vector((mm(R_PLUG) * rad.x, -LEN + mm(PLUG_L), mm(R_PLUG) * rad.z))
            M = Matrix((xax, outward, rad)).transposed().to_4x4()
            assert M.determinant() > 0
            M.translation = origin
            bone = "whip_01" if end == 0 else "whip_%02d" % N_BONES
            for part in ("boot", "housing", "pcb", "pads", "tab"):
                o = bpy.data.objects.new("aoc_whip_plug%d_%d_%s" % (end + 1, k + 1, part), tpl[part])
                bpy.context.scene.collection.objects.link(o)
                o.matrix_world = M
                if part == "tab":
                    o.material_slots[0].link = "OBJECT"
                    o.material_slots[0].material = tabs[cn]
                C.add(o, coll, root)
                L.bone_parent(o, arm, bone)
    # ------------------------------------------------------------ cable ties
    ux = Vector((0, 0, 1))
    xf_axis = Matrix.Rotation(math.pi / 2, 4, "X")  # lathe Z -> -Y? rotate +90 about X sends +Z to -Y
    for i, s in enumerate(S_TIES):
        bm = bmesh.new()
        L.lathe_bm(bm, [(mm(5.65), -mm(2.4)), (mm(6.85), -mm(2.4)), (mm(6.85), mm(2.4)), (mm(5.65), mm(2.4))], 28,
                   xf=Matrix.Translation((0, -s, 0)) @ xf_axis, closed_profile=True)
        L.box_bm(bm, (mm(7.5), mm(5.0), mm(5.2)), (0, -s, mm(6.9 + 1.6)), mm(0.8), 1)
        L.box_bm(bm, (mm(1.2), mm(4.8), mm(24.0)), (mm(1.6), -s, mm(12 + 6.0)), mm(0.3), 1, rot=Matrix.Rotation(0.12, 3, "Y"))
        o = L.obj_from_bm("aoc_whip_tie_%d" % (i + 1), bm, mats["tie"], True, True, 50)
        C.add(o, coll, root)
        skin(o, arm)
    # ------------------------------------------------------------ tape wrap (spiral ridges, 19 mm tape, about 50 % overlap)
    prof = []
    n = 250
    for j in range(n + 1):
        s = S_TAPE0 + (S_TAPE1 - S_TAPE0) * j / n
        e = min(j, n - j) / 6.0
        r = 6.2 * min(1.0, 0.55 + 0.45 * smoothstep(0, 1, e)) + 0.35 * math.sin(2 * math.pi * s / 0.0095) * min(1.0, e)
        prof.append((mm(r), s))
    bm = bmesh.new()
    L.lathe_bm(bm, [(0, S_TAPE0)] + prof + [(0, S_TAPE1)], 32, xf=xf_axis)
    o = L.obj_from_bm("aoc_whip_tape", bm, mats["tape"], True, True, None)
    C.add(o, coll, root)
    skin(o, arm)
    # ------------------------------------------------------------ hooks
    sg = 0.5 * (S_TAPE0 + S_TAPE1)
    h = L.hook_at("grip", coll, root, (0, -sg, 0), size=0.04)
    L.bone_parent(h, arm, "whip_%02d" % (int(sg / BL) + 1))
    h = L.hook_at("tip", coll, root, (0, -LEN, 0), size=0.04)
    L.bone_parent(h, arm, "whip_%02d" % N_BONES)
    h = L.hook_at("crack", coll, root, (0, -(LEN - S_FAN1 - 0.02), 0), size=0.04)
    L.bone_parent(h, arm, "whip_%02d" % N_BONES)
    L.add_prop(root, "p_stiffness_note", 1.0, 0.0, 1.0, "see pose bone/bone property 'stiffness' (1 stiff at the handle falling to about 0.05 at the tip)")
    return coll, root, arm, stiff


def make_actions(arm, stiff):
    """POSE_straight, POSE_coiled (helix, R about 0.12 m, pitch about 14 mm) and ACT_whip_crack_demo."""
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
    arm.animation_data_create()

    def new_action(name):
        a = bpy.data.actions.new(name)
        a.use_fake_user = True
        arm.animation_data.action = a
        for pb in arm.pose.bones:
            pb.rotation_euler = (0, 0, 0)
        return a
    a = new_action("POSE_straight")
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_euler", frame=1)
    a = new_action("POSE_coiled")
    th0 = BL / 0.12
    for i, pb in enumerate(arm.pose.bones):
        w = 1.0 - stiff[i]
        pb.rotation_euler = (th0 * w, 0.019 * w, 0)
        pb.keyframe_insert("rotation_euler", frame=1)
    a = new_action("ACT_whip_crack_demo")
    for f in range(1, 49, 1):
        for i, pb in enumerate(arm.pose.bones):
            w = 1.0 - stiff[i]
            if f <= 12:  # wind-up: curl back over the shoulder
                t = f / 12.0
                ang = -0.16 * w * t
            elif f <= 36:  # loop travels to the tip
                c = 3 + (f - 12) / 24.0 * 22.0
                bump = math.exp(-((i - c) / 1.6) ** 2)
                ang = (-0.16 * w * (1 - (f - 12) / 24.0)) + 0.85 * bump * (0.5 + 0.5 * w)
            else:  # settle
                ang = 0.04 * w * math.sin((f - 36) * 1.2) * math.exp(-(f - 36) / 5.0)
            pb.rotation_euler = (ang, 0, 0)
            pb.keyframe_insert("rotation_euler", index=0, frame=f)
    arm.animation_data.action = bpy.data.actions["POSE_straight"]
    for pb in arm.pose.bones:
        pb.rotation_euler = (0, 0, 0)


def main():
    coll, root, arm, stiff = build()
    make_actions(arm, stiff)
    meta = dict(
        category="props", asset_family="aoc_whip", date="2026-10-02", accuracy="B (QSFP-DD envelope from the MSA; plug internals simplified)",
        description="Six-cable AOC bundle used as a whip: 3.0 m tip to tip including plugs, 3 mm jackets in six colours, QSFP-DD-style plug with pull tab on both ends (12 plugs, six fan out around each end), eight cable ties every 300 mm, tape-wrapped handle, 24-bone chain rig.",
        origin="Handle end (y = 0) on the whip axis; the whip extends along -Y to y = -3.0 m. HOOK_grip is 0.325 m from the origin (centre of the tape wrap). Mounting origin, not on z = 0.",
        sources=[
            {"what": "QSFP-DD MSA Hardware Specification: module width 18.35 mm, height 8.5 mm; module length 89.4 mm (dimension string found in the Rev 4.0 PDF, figure assignment not verified); search summary also lists 92.4 mm", "url": "http://www.qsfp-dd.com/wp-content/uploads/2019/07/QSFP-DD-Hardware-rev5p0.pdf (also https://fluxlight.com/content/Tech-Docs/QSFP%20DD%20Hardware%20Specification.pdf, Rev 4.0)", "accessed": "2026-10-02"},
            {"what": "Cable tie 200 x 4.8 mm class", "url": "https://www.hwlok.com/en/product/GT-200ST.html", "accessed": "2026-10-02"},
        ],
        dimensions_mm=[
            {"item": "whip length tip to tip incl. plugs", "value": 3000, "provenance": "task brief", "level": "A"},
            {"item": "cable jacket diameter", "value": 3.0, "provenance": "task brief", "level": "A"},
            {"item": "plug housing width x height", "value": "18.35 x 8.5", "provenance": "QSFP-DD MSA", "level": "A"},
            {"item": "plug length (boot to paddle)", "value": 89.4, "provenance": "QSFP-DD MSA module length dimension string (not figure-verified)", "level": "B"},
            {"item": "boot length", "value": 27, "provenance": "estimate +-5", "level": "C"},
            {"item": "paddle card width / thickness", "value": "16.8 / 0.9", "provenance": "estimate +-0.5", "level": "C"},
            {"item": "gold pads", "value": "2 faces x 2 rows x 19, pitch 0.82 mm", "provenance": "estimate of the 76-contact layout", "level": "C"},
            {"item": "pull tab", "value": "8 mm wide strap, finger loop 12 x 17 mm outer, ends 28 mm behind the boot", "provenance": "estimate", "level": "C"},
            {"item": "plug fan-out radius", "value": 21, "provenance": "design (plugs must not overlap)", "level": "C"},
            {"item": "tape wrap", "value": "s = 0.20..0.45 m, 19 mm tape ridge pitch 9.5 mm, radius 6.2 mm", "provenance": "design", "level": "C"},
            {"item": "cable ties", "value": "s = 0.45, 0.75, ..., 2.55 m (every 300 mm after the tape)", "provenance": "task brief interpretation", "level": "B"},
        ],
        hooks={"HOOK_grip": "centre of the tape wrap (bone whip_03)", "HOOK_tip": "far end of the tip plugs (bone whip_24)", "HOOK_crack": "0.22 m before the tip, where the fan-out starts: snap point for the shockwave and sound (bone whip_24)"},
        rig="Armature 'aoc_whip_rig': whip_01 (handle) ... whip_24 (tip), each 0.125 m, bone local X = world X (bend in the Y-Z plane by rotating X), local Z for side bends, local Y for twist. Cables, ties and tape are skinned (Armature modifier, vertex groups whip_NN, linear blend between adjacent bones); plugs are bone-parented to whip_01 / whip_24. Pose bone and bone custom property 'stiffness': 1.0 on bones 1-4 (handle), then exp(-(i-3)/7) falling to 0.057 at the tip.",
        actions={"POSE_straight": "rest pose (all rotations zero)", "POSE_coiled": "helix about 0.12 m radius, pitch about 14 mm; applied as bone rotation_euler (x = bend, y = twist) scaled by (1 - stiffness); assign with arm.animation_data.action = bpy.data.actions['POSE_coiled'] (frame 1)",
                 "ACT_whip_crack_demo": "48 frames: wind-up, bend wave travelling to the tip (crack at about frame 36), settle; bone rotation X keyframes scaled by (1 - stiffness). Demo only; animate your own."},
        custom_properties={},
        materials=["MAT_props_aoc_jacket_{red,orange,yellow,green,blue,violet}", "MAT_props_aoc_tab_{...}", "MAT_props_aoc_boot", "MAT_props_aoc_housing", "MAT_props_aoc_pcb", "MAT_props_aoc_gold", "MAT_props_aoc_tie", "MAT_props_aoc_tape"],
        simplifications=["plug has no internal parts, latch release or label; housing is a smooth die-cast style shell", "bundle is a touching ring of 6 cables with no inner filler and no twist", "pull tabs are rigid (they do not flex), plugs are rigid",
                         "tape wrap is a rippled tube (ridge geometry), not a real spiral strip", "the fan-out splays the cables over 0.11 m next to each plug with plugs kept parallel to the axis"],
        intended_usage="S6: Manager winds up and cracks the AOC bundle whip at Gary (3.0 m). Assembler parents the root to the Manager's hand (HOOK_grip to the hand).",
    )
    blend = os.path.join(OUT, "aoc_whip.blend")
    items = [dict(asset_id="aoc_whip", coll=coll, root=root, meta={})]
    L.write_family(blend, items, meta, PREV, previews=False)
    L.reopen(blend)
    arm = bpy.data.objects["aoc_whip_rig"]
    coll = bpy.data.collections["ASSET_aoc_whip"]
    pn = []
    views = [
        {"name": "front", "dir": "side", "fit": 0.95},
        {"name": "three_quarter", "dir": (0.7, -0.5, 0.45), "fit": 0.9},
        {"name": "top", "dir": "top", "fit": 0.78},
        {"name": "closeup_plug_handle", "target": Vector((0, -mm(60), 0)), "cam_dir": Vector((0.5, 0.9, 0.45)), "dist": 0.16},
        {"name": "closeup_plug_tip", "target": Vector((0, -LEN + mm(50), 0)), "cam_dir": Vector((0.5, -0.9, 0.45)), "dist": 0.2},
        {"name": "closeup_ties_tape", "target": Vector((0, -0.45, 0)), "cam_dir": Vector((0.6, -0.5, 0.5)), "dist": 0.32},
    ]
    pn += L.render_views([coll], PREV, "aoc_whip", views)
    arm.animation_data.action = bpy.data.actions["POSE_coiled"]
    bpy.context.view_layer.update()
    pn += L.render_views([coll], PREV, "aoc_whip", [{"name": "coiled", "dir": (0.6, -0.8, 0.6), "fit": 1.4}])
    arm.animation_data.action = bpy.data.actions["ACT_whip_crack_demo"]
    bpy.context.scene.frame_set(30)
    pn += L.render_views([coll], PREV, "aoc_whip", [{"name": "crack_frame30", "dir": "side", "fit": 0.95}])
    L.contact_sheet(pn, os.path.join(L.SCRATCH, "sheet_whip.png"), 3, (400, 300))


main()
