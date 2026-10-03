"""Shared furniture builders for lab_office (office chair with armature rig). Units: mm in, same conventions as lo_common."""
import math

import bpy
from mathutils import Matrix, Vector

import lo_common as L
from lo_common import C, M, S, MM


def make_chair(name, loc, yaw_deg=0.0, fabric="fabric_blue", arms=True, seat_h=470.0, swivel_deg=0.0, tilt_deg=0.0):
    """Office chair (five-star base, gas lift, seat, back, arms) with an armature rig.
    Rig: armature `<name>_rig`, bones: base (root), swivel (rotation_euler Y = world Z: seat, arms, back), back_tilt (rotate about X, child of swivel).
    The chair root empty is at the floor centre; its +Y faces the chair back, -Y is the front (the seat faces -Y at yaw 0).
    yaw_deg rotates the root empty about Z.  Returns the root empty."""
    g, e = L.push_group(name, loc, (0, 0, yaw_deg), sub=False)
    fm = M(fabric)
    blk = M("abs_black")
    # --- static base parts (bone: base)
    base_objs = []
    base_objs.append(L.cyl("hub", 45.0, 40.0, (0, 0, 70.0), blk, seg=24, bev=4.0))
    for k in range(5):
        a = math.radians(90 + 72 * k)
        cx, cy = 150.0 * math.cos(a), 150.0 * math.sin(a)
        base_objs.append(L.box("spoke_%d" % k, (320.0, 50.0, 28.0), (cx, cy, 78.0), blk, r=8.0, seg=2, rot=(0, 0, math.degrees(a))))
        tx, ty = 300.0 * math.cos(a), 300.0 * math.sin(a)
        base_objs.append(L.box("caster_fork_%d" % k, (26.0, 40.0, 38.0), (tx, ty, 60.0), blk, r=5.0, rot=(0, 0, math.degrees(a))))
        base_objs.append(L.cyl("caster_%d" % k, 25.0, 20.0, (tx, ty, 28.0), M("rubber"), axis="x", seg=16, rot=(0, 0, math.degrees(a) + 90)))
    base_objs.append(L.cyl("gas_cover", 34.0, 170.0, (0, 0, 150.0 + 20.0), blk, seg=20, bev=3.0))
    # --- swivel parts
    sw = []
    sw.append(L.cyl("piston", 21.0, 200.0, (0, 0, seat_h - 120.0), M("chrome"), seg=16))
    sw.append(L.box("seat_mech", (260.0, 260.0, 30.0), (0, 0, seat_h - 40.0), blk, r=6.0))
    sw.append(L.box("seat", (480.0, 470.0, 70.0), (0, -8.0, seat_h - 5.0), fm, r=26.0, seg=4))
    if arms:
        for sx in (-1, 1):
            sw.append(L.box("arm_post_%s" % ("l" if sx < 0 else "r"), (28.0, 36.0, 190.0), (sx * 245.0, 30.0, seat_h + 90.0), blk, r=6.0))
            sw.append(L.box("arm_pad_%s" % ("l" if sx < 0 else "r"), (60.0, 260.0, 32.0), (sx * 245.0, 20.0, seat_h + 205.0), blk, r=12.0, seg=3))
    # --- back (bone: back_tilt)
    bk = []
    bk.append(L.box("back_stem", (70.0, 22.0, 190.0), (0, 220.0, seat_h + 100.0), blk, r=6.0))
    bk.append(L.box("back", (440.0, 55.0, 500.0), (0, 232.0, seat_h + 370.0), fm, r=28.0, seg=4, rot=(-6.0, 0, 0)))
    bk.append(L.box("back_shell", (400.0, 18.0, 420.0), (0, 262.0, seat_h + 360.0), blk, r=20.0, seg=3, rot=(-6.0, 0, 0)))
    # --- armature
    arm_d = bpy.data.armatures.new(S.aid + "_" + name + "_rig")
    arm = bpy.data.objects.new(S.aid + "_" + name + "_rig", arm_d)
    S.coll.objects.link(arm)
    arm.parent = e
    arm.show_in_front = True
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    b0 = arm_d.edit_bones.new("base")
    b0.head, b0.tail = (0, 0, 0.05), (0, 0, 0.25)
    b1 = arm_d.edit_bones.new("swivel")
    b1.head, b1.tail = (0, 0, 0.25), (0, 0, seat_h * MM + 0.06)
    b1.parent = b0
    b2 = arm_d.edit_bones.new("back_tilt")
    b2.head, b2.tail = (0, 0.22, seat_h * MM + 0.05), (0, 0.22, seat_h * MM + 0.45)
    b2.parent = b1
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    for objs, bone in ((base_objs, "base"), (sw, "swivel"), (bk, "back_tilt")):
        for o in objs:
            wm = o.matrix_world.copy() if False else None
            o.parent = arm
            o.parent_type = "BONE"
            o.parent_bone = bone
            bone_mat = arm.data.bones[bone].matrix_local @ Matrix.Translation((0, arm.data.bones[bone].length, 0))
            # keep the object where it was relative to the chair root: parent inverse = inverse of the bone's parent matrix,
            # object matrix_basis stays as built relative to the root (root is the armature's parent)
            o.matrix_parent_inverse = (arm.matrix_world @ Matrix.Identity(4)).inverted() @ Matrix.Identity(4) if False else bone_mat.inverted()
    # object matrices: location values were set relative to the chair root frame; with the armature parented to the root
    # (identity local transform) bone-parent evaluation reproduces that when the inverse above is applied to root-frame coordinates.
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
    arm.pose.bones["swivel"].rotation_euler = (0, math.radians(swivel_deg), 0)
    arm.pose.bones["back_tilt"].rotation_euler = (math.radians(tilt_deg), 0, 0)
    arm["rig_note"] = "bones: base, swivel (rotate about local Y = world Z), back_tilt (rotate about local X, child of swivel)"
    L.hook("seat_%s" % name.replace("chair_", ""), (0, -10.0, seat_h + 35.0), parent=e, note="seat surface centre (sit point), chair %s" % name)
    L.pop_group()
    return e


def apply_pose(chair_root, swivel_deg=None, tilt_deg=None):
    for ch in chair_root.children:
        if ch.type == "ARMATURE":
            if swivel_deg is not None:
                ch.pose.bones["swivel"].rotation_euler = (0, math.radians(swivel_deg), 0)
            if tilt_deg is not None:
                ch.pose.bones["back_tilt"].rotation_euler = (math.radians(tilt_deg), 0, 0)
