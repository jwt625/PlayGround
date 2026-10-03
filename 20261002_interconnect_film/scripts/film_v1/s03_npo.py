"""S3 'NPO: closer, and everyone has a different idea' (film 20-30 s, scene-local 0-10 s), v1.0 assembly.

Build: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s03_npo.py -- scenes/v1/s03_npo.blend
Timing, captions, labels, cards, narration and FX notes follow scripts/build_crude_film.py s3; geometry uses the asset library.

Layout (room frame = world frame; conference_room root at the origin, 1 unit = 1 m):
  table 5.6 x 2.2 m, top z = 0.76, x in [-2.8, 2.8]. Gary at the +X table end (3.1, -0.2) facing -X.
  Vendors stand at the far (+Y) side, y = 1.4: TERAHOP -2.4, MOLEXX -1.4, NUBISS 1.4, AYARR 2.4.
  The Manager enters through the room door in the -X wall (door hinge at (-4.24, -1.45)).

Hardware scale factors (documented in DevLog-003-scene-s03.md): S_HW = 7 for the XPU package, NPO module and ELS laser
(true relative sizes), S_PKG = 12 for the BGA/LGA packages, S_DIE = 50 (xy) with a 3x z exaggeration for the loose dies.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm  # noqa: E402
import bpy  # noqa: E402
from asm import F, PI  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import blender_lib as L  # noqa: E402
import s03_brawl as BR  # noqa: E402
import s03_cam as CAM  # noqa: E402
import s03_fight as FT  # noqa: E402
import s03_fx as FX  # noqa: E402
import s03_sky as SKY  # noqa: E402

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "scenes/v1/s03_npo.blend"

TABLE_Z = 0.76
S_HW, S_PKG, S_DIE, Z_DIE = 7.0, 12.0, 50.0, 3.0
T_FILM0 = 20.0  # film time of scene-local 0

scn = asm.new_scene(preset="standard")




# ------------------------------------------------------------------ small helpers
def yaw_key(obj, t, yaw, interp="CONSTANT"):
    obj.rotation_euler = (obj.rotation_euler[0], obj.rotation_euler[1], yaw)
    obj.keyframe_insert("rotation_euler", frame=F(t))
    L._interp(obj, "rotation_euler", F(t), interp)


def pos_key(obj, t, pos, interp="LINEAR"):
    obj.location = pos
    obj.keyframe_insert("location", frame=F(t))
    L._interp(obj, "location", F(t), interp)


def rot_key(obj, t, rot, interp="LINEAR"):
    obj.rotation_euler = rot
    obj.keyframe_insert("rotation_euler", frame=F(t))
    L._interp(obj, "rotation_euler", F(t), interp)


def scale_key(obj, t, s, interp="LINEAR"):
    obj.scale = s if not isinstance(s, (int, float)) else (s, s, s)
    obj.keyframe_insert("scale", frame=F(t))
    L._interp(obj, "scale", F(t), interp)


def descendants(o):
    out = [o]
    for ch in o.children:
        out += descendants(ch)
    return out


def loop(asset, action, t0, t1, blend_in=3, speed=1.0):
    """Looping action strip between t0 and t1 (scene seconds)."""
    nm = "ACT_%s_%s" % (asset.root["asset_id"], action)
    act = bpy.data.actions[nm]
    n = max(act.frame_range[1] - act.frame_range[0], 1)
    return asm.play(asset, action, t0, speed=speed, hold=False, repeat=max((t1 - t0) * asm.FPS * speed / n, 0.05), blend_in=blend_in)


def heal_radius(film_t_shot, film_t):
    return max(0.3, 0.9 ** ((film_t - film_t_shot) / 1.5))


def key_heal(root, prop, film_t_shot, t_from=0.0, t_to=10.0, step=0.5):
    """Key a bullet-hole radius following r = max(0.3, 0.9 ** ((T_now - T_shot) / 1.5)) (storyboard rule)."""
    t = t_from
    while t <= t_to + 1e-6:
        asm.key_prop(root, prop, t, heal_radius(film_t_shot, T_FILM0 + t), "LINEAR")
        t += step


def block(name, size, loc, color, rough=0.5, metallic=0.0):
    me = bpy.data.meshes.new(name)
    sx, sy, sz = (s / 2 for s in size)
    v = [(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0), (-sx, -sy, 2 * sz), (sx, -sy, 2 * sz), (sx, sy, 2 * sz), (-sx, sy, 2 * sz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    me.from_pydata(v, [], f)
    me.update()
    o = bpy.data.objects.new(name, me)
    c = bpy.data.collections.new("BLOCK_" + name)
    bpy.context.scene.collection.children.link(c)
    c.objects.link(o)
    m = bpy.data.materials.new("MAT_s03_" + name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    o.data.materials.append(m)
    o.location = loc
    return asm.Asset(c, o, [o])


def smoke_ring(t, parent):
    """Muzzle smoke ring toned down: smaller, translucent grey (own material copy, links to base/emission cut)."""
    a = asm.fx("smoke_ring", t, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), parent=parent, dur=0.9, scale=0.6)
    for o in a.objs:
        for m in o.modifiers:
            if m.type == "NODES" and "Socket_5" in m.keys():
                m["Socket_5"] = 0.13   # Radius End (default 0.32)
                m["Socket_8"] = 0.6    # Travel (default 1.4)
                m["Socket_10"] = 0.5   # Intensity
                mat = m["Socket_11"].copy()
                m["Socket_11"] = mat
                nt = mat.node_tree
                pb = nt.nodes["Principled BSDF"]
                for l in list(nt.links):
                    if l.to_node is pb and l.to_socket.name in ("Base Color", "Emission Color", "Emission Strength"):
                        nt.links.remove(l)
                pb.inputs["Base Color"].default_value = (0.42, 0.43, 0.46, 1)
                pb.inputs["Emission Color"].default_value = (0.42, 0.43, 0.46, 1)
                pb.inputs["Emission Strength"].default_value = 0.1
                al = next((l for l in nt.links if l.to_node is pb and l.to_socket.name == "Alpha"), None)
                if al is not None:
                    mul = nt.nodes.new("ShaderNodeMath")
                    mul.operation = "MULTIPLY"
                    mul.inputs[1].default_value = 0.5
                    src = al.from_socket
                    nt.links.remove(al)
                    nt.links.new(src, mul.inputs[0])
                    nt.links.new(mul.outputs[0], pb.inputs["Alpha"])
                o.update_tag()
    return a


# ------------------------------------------------------------------ room, lights
room = asm.append("lab_office/conference_room")
asm.place(room.root, (0, 0, 0))
for o in room.objs:
    if o.name.startswith(("conference_room_chair_far_", "conference_room_chair_near_")) and o.type == "EMPTY":
        for d in descendants(o):
            d.hide_render = True
            d.hide_viewport = True
door = next(o for o in room.objs if o.name == "conference_room_door")
rot_key(door, 0.0, (0, 0, 0), "CONSTANT")
rot_key(door, 6.2, (0, 0, 0), "BEZIER")
rot_key(door, 6.7, (0, 0, -1.55), "BEZIER")


def area(name, loc, size, watt, color=(1, 1, 1), rot=(0, 0, 0)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "RECTANGLE" if isinstance(size, tuple) else "SQUARE"
    if isinstance(size, tuple):
        ld.size, ld.size_y = size
    else:
        ld.size = size
    ld.energy = watt
    ld.color = color
    o = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    return o


for k, (lx, ly) in enumerate([(-2.4, -1.2), (0, -1.2), (2.4, -1.2), (-2.4, 1.2), (0, 1.2), (2.4, 1.2)]):
    area("ceil_%d" % k, (lx, ly, 2.7), 1.0, 30.0, (1.0, 0.97, 0.92))
# camera-side fill (front wall is hidden), faces +Y
area("front_fill", (0, -3.0, 2.3), (6.0, 2.0), 120.0, (1.0, 0.98, 0.95), rot=(PI / 2 - 0.25, 0, 0))

# ------------------------------------------------------------------ characters
VY = 1.4  # vendor row y
VEND_ASSET = {"terahop": "npc_terahop", "molexx": "npc_molexx", "nubiss": "npc_nubiss", "ayarr": "npc_ayarr"}  # swap an asset id here (e.g. a leather-jacket variant)
VEND = [  # id, x, dir (+1 faces +x)
    ("terahop", -2.4, +1),
    ("molexx", -1.4, -1),
    ("nubiss", 1.4, +1),
    ("ayarr", 2.4, -1),
]
V = {}
for vid, vx, d in VEND:
    a = asm.append("characters/" + VEND_ASSET[vid], actions=True)
    asm.place(a.root, (vx, VY, 0.0), yaw=0.0)
    V[vid] = dict(a=a, x=vx, d=d)

gary = asm.append("characters/gary", actions=True)
mgr = asm.append("characters/manager", actions=True)
GARY_POS = (3.1, -0.9, 0.0)
asm.place(gary.root, GARY_POS, yaw=0.0)
MGR_END = (-2.3, -1.8, 0.0)
asm.place(mgr.root, (-5.5, -1.9, 0.0), yaw=PI / 2)

# ------------------------------------------------------------------ shotguns (clay variant is default-visible)
gun_master = asm.append("props/shotgun")
guns = {"molexx": gun_master, "nubiss": asm.clone(gun_master, "ASSET_shotgun_nubiss"),
        "manager": asm.clone(gun_master, "ASSET_shotgun_manager"), "gary": asm.clone(gun_master, "ASSET_shotgun_gary")}
# gun frame -> hand hook frame (hook: Y along the fingers, Z = palm normal; aim pose: fingers along the barrel, palm toward +X)
M_GUN_HAND = Matrix(((0, 0, 1), (0, -1, 0), (1, 0, 0))).to_4x4()


def hook_of(a, name):
    return a.hook(name)


for who, host in (("molexx", V["molexx"]["a"]), ("nubiss", V["nubiss"]["a"]), ("manager", mgr)):
    g = guns[who]
    asm.attach(g.root, host.hook("gun_grip_R"), (0, 0, 0), M_GUN_HAND.to_euler())

# ------------------------------------------------------------------ table hardware
xpu = asm.append("packaging/xpu_package_rubin_style")
XPU_X = 0.8
asm.place(xpu.root, (XPU_X, 0.0, TABLE_Z + 0.0029 * S_HW), scale=S_HW)

npo = asm.append("photonics/oe_module_npo")
npo.variant("open", False)
NPO_Z = TABLE_Z + 0.0005 * S_HW
asm.place(npo.root, (2.55, 0.0, NPO_Z), scale=S_HW)
pos_key(npo.root, 0.0, (2.55, 0.0, NPO_Z), "LINEAR")
pos_key(npo.root, 0.2, (2.55, 0.0, NPO_Z), "BEZIER")
pos_key(npo.root, 1.0, (1.85, 0.0, NPO_Z), "BEZIER")

els = asm.append("photonics/els_laser_source")
for vn in ("elsfp_open", "elsfp_front_pigtail", "butterfly_laser"):
    els.variant(vn, False)
asm.place(els.root, (-0.55, 0.65, TABLE_Z + 0.004), yaw=PI / 2, scale=S_HW)
asm.show(els, 1.9, 3.0)

# in-package laser: glowing block on the XPU (copper emission, as the crude laser_in)
lin = block("laser_in_pkg", (0.28, 0.28, 0.06), (XPU_X - 0.25, 0.0, TABLE_Z + 0.03), (0.8, 0.42, 0.14))
mm = lin.objs[0].data.materials[0]
mm.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (0.9, 0.45, 0.1, 1)
mm.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 3.0
asm.show(lin, 1.8, 3.0)

# real packages standing with the underside to the camera (bga_lga_family)
PKG = [("fcbga_1p0_35", -2.4, 35.0, "BGA", 3.0), ("lga_1p0_45", -1.65, 45.0, "LGA", 3.1), ("fbga_0p8_23", -0.9, 23.0, "BGA (FINE)", 3.2)]
pk_assets = []
for pid, px, mm_, label, t0 in PKG:
    a = asm.append("packaging/bga_lga_family", only=["ASSET_" + pid])
    sz = mm_ / 1000.0 * S_PKG
    lying = (px, 0.0, TABLE_Z + 0.004)
    stand = (px, 0.0, TABLE_Z + sz / 2 + 0.01)
    asm.place(a.root, lying, rot=(0, 0, 0), scale=S_PKG)
    pos_key(a.root, t0 - 0.05, lying, "CONSTANT")
    pos_key(a.root, t0 + 0.05, lying, "BEZIER")
    pos_key(a.root, t0 + 0.4, stand, "BEZIER")
    rot_key(a.root, t0 - 0.05, (0, 0, 0), "CONSTANT")
    rot_key(a.root, t0 + 0.05, (0, 0, 0), "BEZIER")
    rot_key(a.root, t0 + 0.4, (-PI / 2, 0, 0), "BEZIER")
    # gentle sway while standing
    rot_key(a.root, t0 + 0.8, (-PI / 2 - 0.03, 0.03, 0), "BEZIER")
    rot_key(a.root, t0 + 1.2, (-PI / 2 + 0.02, -0.02, 0.0), "BEZIER")
    asm.show(a, t0 - 0.1, 4.2)
    pk_assets.append((a, px, sz, label, t0))

# loose dies: EIC, PIC, HBM and a generic logic die block, mismatched sizes
DIE_Y = -0.1
die_row = [("eic", -2.5), ("pic", -1.75), ("hbm", -1.0), ("logic", -0.25)]
DIES = {}
eic = asm.append("photonics/eic_die")
eic.variant("hybrid_bond", False)
pic = asm.append("photonics/pic_die")
hbm = asm.append("packaging/hbm_stack")
logic = block("logic_die", (0.6 / S_DIE, 0.6 / S_DIE, 0.9 / 1000.0), (0, 0, 0), (0.55, 0.57, 0.62), rough=0.35, metallic=0.6)
DIES = {"eic": eic, "pic": pic, "hbm": hbm, "logic": logic}
TOWER = (-1.2, DIE_Y)
# stacked heights (m): footprints at 50x, z exaggeration 3x -> logic 0.0405? computed from native thickness below
THK = {"logic": 0.9, "hbm": 0.8, "pic": 0.8, "eic": 0.3}  # mm (native), logic block is a generic 0.9 mm slab
order = ["logic", "hbm", "pic", "eic"]
zacc = TABLE_Z + 0.003
tower_z = {}
for k in order:
    tower_z[k] = zacc
    zacc += THK[k] / 1000.0 * S_DIE * Z_DIE  # native thickness x z scale
for k, x in die_row:
    a = DIES[k]
    sxy = S_DIE
    if k == "logic":
        a.root.scale = (S_DIE, S_DIE, S_DIE * Z_DIE)
    else:
        a.root.scale = (S_DIE, S_DIE, S_DIE * Z_DIE)
    asm.show(a, 4.15, 7.0)
    row_p = (x, DIE_Y, TABLE_Z + 0.003)
    a.root.location = row_p
    pos_key(a.root, 4.15, row_p, "CONSTANT")
    pos_key(a.root, 5.2, row_p, "BEZIER")
    pos_key(a.root, 6.2, (TOWER[0], TOWER[1], tower_z[k]), "BEZIER")
    pos_key(a.root, 7.0, (TOWER[0], TOWER[1], tower_z[k]), "LINEAR")
    # wobbling tower, bigger swing higher up
    amp = {"logic": 0.0, "hbm": 0.02, "pic": 0.05, "eic": 0.09}[k]
    for j, tt in enumerate((6.2, 6.4, 6.6, 6.8, 7.0)):
        s = 1 if j % 2 == 0 else -1
        rot_key(a.root, tt, (amp * s, amp * 0.7 * -s, amp * 0.5 * s * (j % 3)), "BEZIER")

# calendar (+3 MONTHS)
cal = asm.append("props/office_gags", only=["ASSET_wall_calendar"])
CAL_S = 2.2
asm.place(cal.root, (-2.4, -0.55, 0.95), scale=CAL_S)
pos_key(cal.root, 5.4, (-2.4, -0.55, 0.95), "LINEAR")
pos_key(cal.root, 7.0, (-2.4, -0.55, 1.55), "BEZIER")
asm.key_prop(cal.root, "p_flip", 5.4, 0.0)
asm.key_prop(cal.root, "p_flip", 6.9, 3.0)
asm.show(cal, 5.35, 7.0)

# price balloon inflating
bal = asm.append("props/office_gags", only=["ASSET_price_balloon"])
BAL_POS = (0.0, 1.95, 1.95)
asm.place(bal.root, BAL_POS)
asm.key_prop(bal.root, "p_inflate_scale", 5.4, 0.2)
asm.key_prop(bal.root, "p_inflate_scale", 7.0, 1.6)
asm.key_prop(bal.root, "p_inflate_scale", 8.3, 1.6)
asm.show(bal, 5.35, 8.3)

# envelope: A in Gary's hand (6.8-7.5), B flies to AYARR (7.5-8.0), C in AYARR's hand (8.0-8.4)
env_a = asm.append("props/office_gags", only=["ASSET_envelope_bonus"])
env_b = asm.clone(env_a, "ASSET_envelope_bonus_flight")
env_c = asm.clone(env_a, "ASSET_envelope_bonus_held")
ENV_REL = 6.8 + 30.0 / 30.0 / 1.4  # hand_over_envelope releases at frame 30; strip speed 1.4
asm.attach(env_a.root, gary.hook("hand_R"), (0, 0, 0), (0, 0, 0))
asm.show(env_a, 6.7, ENV_REL)
ay = V["ayarr"]["a"]
asm.attach(env_c.root, ay.hook("hand_L"), (0, 0, 0), (0, 0, 0))
asm.show(env_c, 8.0, 8.4)
asm.place(env_b.root, (2.8, -0.6, 1.0))
asm.show(env_b, ENV_REL, 8.02)

# ------------------------------------------------------------------ actions: Gary
asm.play(gary, "idle", 0.0, hold=True, repeat=6.0)
asm.play(gary, "hand_over_envelope", 6.8, speed=1.4, hold=True, blend_in=4)
asm.play(gary, "idle", 8.2, hold=True, repeat=3.0, blend_in=6)
asm.play(gary, "topple_back", 9.35, speed=3.0, hold=True, blend_in=0)  # holds the pose for the head close-up, then drops
asm.key_prop(gary.root, "p_expr_worried", 0.0, 0.0)
asm.key_prop(gary.root, "p_expr_worried", 1.0, 0.7)
asm.key_prop(gary.root, "p_expr_worried", 6.0, 0.7)
asm.key_prop(gary.root, "p_expr_worried", 7.0, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 2.5, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 5.0, 0.8)
asm.key_prop(gary.root, "p_expr_sweating", 8.0, 0.8)
asm.key_prop(gary.root, "p_expr_dread", 6.5, 0.0)
asm.key_prop(gary.root, "p_expr_dread", 8.0, 1.0)
asm.key_prop(gary.root, "p_expr_dread", 8.9, 1.0)
asm.key_prop(gary.root, "p_expr_dead_eyed", 8.9, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_expr_dead_eyed", 8.95, 1.0, "CONSTANT")
# earlier holes heal per the storyboard schedule (film times: hole 1 at 9.2 s, hole 2 at 19.2 s)
key_heal(gary.root, "p_hole_1_radius", 9.2)
key_heal(gary.root, "p_hole_2_radius", 19.2)
asm.key_prop(gary.root, "p_head_hole_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_head_hole_radius", 8.95, 1.0, "CONSTANT")
# Gary faces the camera; turns to AYARR for the handover (6.8-8.1), back to the camera for the self-shot; the fall (backward) goes toward +Y
GARY_FACE_AYARR = asm.yaw_to(GARY_POS, (V["ayarr"]["x"], VY))
yaw_key(gary.root, 0.0, 0.0, "CONSTANT")
yaw_key(gary.root, 6.6, 0.0, "LINEAR")
yaw_key(gary.root, 6.8, GARY_FACE_AYARR, "CONSTANT")
yaw_key(gary.root, 8.0, GARY_FACE_AYARR, "LINEAR")
yaw_key(gary.root, 8.25, 0.0, "CONSTANT")

# Gary's own shotgun at the head (HOOK_muzzle_self); right hand IK-pulled to the fore-end
gg = guns["gary"]
bpy.context.view_layer.update()
bpy.context.scene.frame_set(1)
hk = gary.hook("muzzle_self")
d_local = Vector((0.75, 0.3, 0.4)).normalized()  # fired from Gary's right side (table side) toward his head
rw = gary.root.matrix_world.to_3x3()
d_world = rw @ d_local
q = d_world.to_track_quat("-Y", "Z")
Rw = q.to_matrix()
gg.root.parent = None
gg.root.matrix_world = Matrix.Identity(4)
bpy.context.view_layer.update()
mz = guns["gary"].hook("muzzle_R")
mz_local = gg.root.matrix_world.inverted() @ mz.matrix_world.translation
hk_local = gary.root.matrix_world.inverted() @ hk.matrix_world.translation
p_target = hk.matrix_world.translation + 0.05 * d_world
gun_world_loc = p_target - Rw @ mz_local
gun_world = Matrix.Translation(gun_world_loc) @ Rw.to_4x4()
gg.root.parent = hk
gg.root.matrix_parent_inverse.identity()
gg.root.matrix_basis = hk.matrix_world.inverted() @ gun_world
asm.show(gg, 8.3, 10.0)
# IK: Copy Location on ik_hand_R (keyed influence)
arm = gary.armature
pb = arm.pose.bones["ik_hand_R"]
cl = pb.constraints.new("COPY_LOCATION")
cl.target = gg.hook("foregrip_L")
cl.influence = 0.0
cl.keyframe_insert("influence", frame=F(8.1))
cl.influence = 1.0
cl.keyframe_insert("influence", frame=F(8.45))
cl.keyframe_insert("influence", frame=F(10.0))

# ------------------------------------------------------------------ actions: vendors (v1.1: procedural brawl, baked)
FIGHT_YAW = 0.9
T_B0, T_B1 = 1.70, 7.50   # brawl action span (scene seconds)
FG = {}
for vid, vx, d in VEND:
    a = V[vid]["a"]
    asm.play(a, "idle", 0.0, hold=True, repeat=6.0)
    root = a.root
    yaw_key(root, 0.0, 0.35 * d, "CONSTANT")
    pos_key(root, 0.0, (vx, VY, 0.0), "CONSTANT")
    asm.key_prop(root, "p_expr_shouting", 0.0, 0.0)
    asm.key_prop(root, "p_expr_shouting", 1.8, 0.0)
    asm.key_prop(root, "p_expr_shouting", 2.0, 1.0)
    asm.key_prop(root, "p_expr_angry", 2.9, 0.0)
    asm.key_prop(root, "p_expr_angry", 3.2, 0.8)
    fg = FT.Fighter(vid, a, d, (vx, VY), FIGHT_YAW)
    fg.yaw_keys = [(0.0, 0.35 * d), (1.7, 0.35 * d), (2.0, FIGHT_YAW * d), (T_B1 + 1.0, FIGHT_YAW * d)]
    FG[vid] = fg
for x_, y_ in (("terahop", "molexx"), ("nubiss", "ayarr")):
    FG[x_].partner, FG[y_].partner = FG[y_], FG[x_]
BR.compose(FG["terahop"], FG["molexx"], FG["nubiss"], FG["ayarr"])
for vid, fg in FG.items():
    act, roots = FT.bake_fighter(fg, T_B0, T_B1, log=print)
    FT.key_root(fg, roots)
    st = asm.play(fg.asset, act, T_B0, hold=False, blend_in=3)
    st.blend_out = 6
    asm.key_prop(fg.asset.root, "p_expr_shock", 0.0, 0.0, "CONSTANT")
    pulses = sorted(fg.expr)
    for (te, prop, val) in pulses:
        if val > 0:
            asm.key_prop(fg.asset.root, prop, max(te - 0.04, 0.01), 0.0, "LINEAR")
            asm.key_prop(fg.asset.root, prop, te, val, "LINEAR")
        else:
            asm.key_prop(fg.asset.root, prop, te, 0.0, "LINEAR")

# shooters and victims
BANG_M, BANG_N = 8.3, 8.5
AIM_SPEED = 1.5
for who, bang, tgt in (("molexx", BANG_M, "terahop"), ("nubiss", BANG_N, "ayarr")):
    a = V[who]["a"]
    d = V[who]["d"]
    t_start = bang - 36.0 / 30.0 / AIM_SPEED
    yaw_key(a.root, T_B1, FIGHT_YAW * d, "LINEAR")
    yaw_key(a.root, T_B1 + 0.3, (PI / 2) * d, "CONSTANT")
    asm.play(a, "aim_gun", t_start, speed=AIM_SPEED, hold=True, blend_in=6)
    asm.show(guns[who], t_start - 0.05, 10.0)
    g = guns[who]
    asm.fx("muzzle_flash", bang, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), parent=g.hook("muzzle_R"), dur=0.12)
    smoke_ring(bang + 0.03, g.hook("muzzle_R"))
    V[tgt]["bang"] = bang
for tgt in ("terahop", "ayarr"):
    a = V[tgt]["a"]
    d = V[tgt]["d"]
    t_hit = V[tgt]["bang"]
    # victims keep the 3/4 fight stance so the chest hole faces the camera and the fall (backward) goes out past the table end
    asm.play(a, "topple_back", t_hit, speed=1.6, hold=True, blend_in=0)
    asm.key_prop(a.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_hole_1_radius", t_hit, 1.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_shouting", t_hit, 1.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_shouting", t_hit + 0.05, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_shock", t_hit, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_shock", t_hit + 0.05, 1.0, "CONSTANT")

# NUBISS is shot by the Manager at 8.95 (turns toward the camera, falls toward +Y)
nub = V["nubiss"]["a"]
yaw_key(nub.root, 8.5, (PI / 2), "LINEAR")
yaw_key(nub.root, 8.9, -0.6, "CONSTANT")
asm.play(nub, "topple_back", 8.95, speed=2.0, hold=True, blend_in=0)
asm.key_prop(nub.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(nub.root, "p_hole_1_radius", 8.95, 1.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_shock", 8.9, 0.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_shock", 8.95, 1.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_shouting", 8.9, 1.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_shouting", 8.95, 0.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_angry", 8.4, 0.8, "CONSTANT")
asm.key_prop(nub.root, "p_expr_angry", 8.95, 0.0, "CONSTANT")

# MOLEXX survivor: stays shouting then smug after the last shot
mol = V["molexx"]["a"]
asm.key_prop(mol.root, "p_expr_smug", 8.9, 0.0, "CONSTANT")
asm.key_prop(mol.root, "p_expr_smug", 9.0, 1.0, "CONSTANT")
asm.key_prop(mol.root, "p_expr_shouting", 8.9, 1.0, "CONSTANT")
asm.key_prop(mol.root, "p_expr_shouting", 9.0, 0.0, "CONSTANT")
asm.key_prop(mol.root, "p_expr_angry", 8.9, 0.8, "CONSTANT")
asm.key_prop(mol.root, "p_expr_angry", 9.0, 0.0, "CONSTANT")

# move the bullet hole of the shot NPCs 0.14 m down (below the shirt wordmark); the rim tube is a child of the hole empty
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()
for vid in ("terahop", "ayarr", "nubiss"):
    he = next(o for o in V[vid]["a"].objs if o.name.startswith("HOLE_hole_1"))
    mw = he.matrix_world.copy()
    mw.translation = mw.translation + Vector((0, 0, -0.14))
    he.matrix_world = mw
bpy.context.view_layer.update()

# ------------------------------------------------------------------ Manager
asm.play(mgr, "idle", 0.0, hold=True, repeat=6.0)  # base (parked outside)
t_arr = asm.walk(mgr, 6.7, [(-4.35, -1.9, 0.0), MGR_END], speed_mps=1.9, gait="walk")
aim_yaw = asm.yaw_to(MGR_END, (V["nubiss"]["x"], VY))
yaw_key(mgr.root, t_arr, aim_yaw, "CONSTANT")
pos_key(mgr.root, t_arr, MGR_END, "CONSTANT")
st = asm.play(mgr, "idle", t_arr, hold=True, repeat=6.0, blend_in=4)
asm.play(mgr, "aim_gun", 7.75, hold=True, blend_in=6)
asm.show(guns["manager"], 7.7, 10.0)
asm.fx("muzzle_flash", 8.95, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), parent=guns["manager"].hook("muzzle_R"), dur=0.12)
smoke_ring(8.98, guns["manager"].hook("muzzle_R"))
asm.show(mgr, 7.0, 10.0)
asm.key_prop(mgr.root, "p_anger", 0.0, 0.0)
asm.key_prop(mgr.root, "p_anger", 7.8, 0.5)
asm.key_prop(mgr.root, "p_anger", 8.9, 1.0)

# Gary's muzzle effects
asm.fx("muzzle_flash", 8.95, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), parent=gg.hook("muzzle_R"), dur=0.12)
smoke_ring(8.98, gg.hook("muzzle_R"))

# ------------------------------------------------------------------ fight effects (v1.1): dust puffs, clay crumbs, papers, hats
fx = FX.Fx()
for e in BR.EVENTS:
    if e["kind"] in ("shove", "hook", "uppercut", "punch", "headbutt", "bump"):
        fx.puff(e["t"], e["pt"], e["dir"], e["s"], n=2)
        fx.crumbs(e["t"], e["pt"], e["dir"], e["s"], e["vic"], n=3 + int(4 * e["s"]))
    elif e["kind"] == "tug":
        fx.puff(e["t"], (e["pt"][0], e["pt"][1], 0.06), e["dir"], 0.2, n=1)
PAPER_BURSTS = [(3.18, (-1.6, VY - 0.2, 1.3), 4, 1.0), (4.90, (1.7, VY - 0.2, 1.4), 4, -1.0), (5.98, (-2.1, VY - 0.2, 1.5), 4, -1.0),
                (6.62, (2.0, VY - 0.2, 1.2), 3, 1.0)]
for tp, pt, n_, dr in PAPER_BURSTS:
    fx.papers(tp, pt, n_, d=dr)
# caps: NUBISS (green) flies off at the third flurry hit, TERAHOP (purple) at the uppercut
fx.hat("nubiss", (0.15, 0.55, 0.28), 5.72, (-2.0, 0.3, 2.8), (5.0, 3.0, 14.0), V["nubiss"]["a"].hook("head_top"))
fx.hat("terahop", (0.40, 0.22, 0.65), 5.98, (-1.9, -0.4, 3.4), (4.0, -2.0, -12.0), V["terahop"]["a"].hook("head_top"))

# ------------------------------------------------------------------ labels, captions, narration
asm.timecode(3)
asm.narr([(0.2, 3.0, "Gary moves the optics closer to the logic."), (3.0, 5.4, "Every vendor wants something different."),
          (5.6, 8.2, "Now he waits three more months and pays his bonus.")])
L.OVL["BIGHI"] = dict(L.OVL["BIG"], loc=(0, 1.15))
L.WRAP["BIGHI"] = 10
L.ovt("BIGHI", "BANG BANG", 0, 8.3, 8.9)
L.ovt("BIGHI", "BANG", 0, 8.95, 9.55)
asm.lab(0.0, 1.8, "NPO")
asm.lab(1.8, 3.0, "LASER: IN PACKAGE OR EXTERNAL?")
asm.lab(3.0, 4.2, "BGA vs LGA vs PITCH vs SIZE")
asm.lab(4.2, 5.2, "DIE SIZE")
asm.lab(5.2, 7.0, "2D vs 3D   LEAD TIME   COST")
asm.lab(7.0, 8.3, "THE VENDORS LOSE IT")
asm.card(0.0, 1.8, "NPO: module on package substrate / HDI beside the ASIC (Cheng 2025)")
asm.fxn(1.8, 3.0, "[FX: vendors shout, shove, swing; comic dust clouds]")
asm.fxn(3.0, 4.2, "[FX: packages flip to show underside ball/land arrays]")
asm.fxn(5.2, 7.0, "[FX: die tower wobble, calendar pages fly, price tag inflates]")
asm.fxn(8.3, 9.5, "[FX: muzzle flashes, smoke rings, hole decals]")

asm.wl("NPO MODULE", (1.5, 0.0, 1.45), 0.2, 1.8, size=0.2)
asm.wl("LASER IN PACKAGE", (XPU_X - 0.25, 0.0, 1.7), 1.8, 3.0, size=0.17)
asm.wl("LASER EXTERNAL", (-0.55, 0.65, 1.15), 2.0, 3.0, size=0.17)
for a_, px, sz, label, t0 in pk_assets:
    asm.wl(label, (px, 0.0, TABLE_Z + sz + 0.3), t0, 4.2, size=0.2 if "FINE" not in label else 0.16)
asm.wl("DIE SIZE?", (-1.3, DIE_Y, 1.35), 4.2, 5.2, size=0.18)
asm.wl("2D", (-1.4, DIE_Y, 1.3), 5.2, 6.0, size=0.2)
asm.wl("3D", (TOWER[0], TOWER[1], 1.45), 6.2, 7.0, size=0.2)
ct = asm.wl("+3 MONTHS", (-2.4, -0.7, 1.3), 5.4, 7.0, size=0.22, color=(1, 0.4, 0.3))
pos_key(ct, 5.4, (-2.4, -0.9, 1.3))
pos_key(ct, 7.0, (-2.4, -0.9, 2.0))
et = asm.wl("YEAR-END BONUS", (3.0, -0.8, 1.5), 7.0, 8.3, size=0.14, color=(0.4, 1.0, 0.4))
pos_key(et, 7.0, (3.0, -0.8, 1.5))
pos_key(et, ENV_REL, (3.0, -0.8, 1.5))
pos_key(et, 8.0, (V["ayarr"]["x"], VY - 0.3, 1.5))
pos_key(et, 8.3, (V["ayarr"]["x"], VY - 0.3, 1.5))
for (vid, vx, d), t0 in zip(VEND, (2.2, 3.6, 5.0, 6.4)):
    asm.wl("@#$%!", (vx, VY, 2.2), t0, t0 + 0.9, size=0.3, color=(1.0, 0.35, 0.3))
    asm.wl("!!", (vx + 0.2, VY - 0.2, 2.3), t0 + 2.0, t0 + 2.9, size=0.4, color=(1.0, 0.8, 0.2))
asm.wl("GARY'S GUN", (3.5, -0.9, 2.2), 8.3, 8.95, size=0.2)

# envelope flight (arc from Gary's hand to AYARR's hand)
pos_key(env_b.root, ENV_REL, (2.8, -0.55, 1.1), "LINEAR")
pos_key(env_b.root, (ENV_REL + 8.0) / 2, (2.4, 0.4, 1.7), "BEZIER")
pos_key(env_b.root, 8.0, (V["ayarr"]["x"] + 0.1, VY - 0.4, 1.15), "BEZIER")
rot_key(env_b.root, ENV_REL, (0, 0, 0), "LINEAR")
rot_key(env_b.root, 8.0, (0.4, 0.3, 2.5), "LINEAR")

# ------------------------------------------------------------------ camera (v1.1: eased shots, handheld noise, impact shake, stable finale)
SHOTS = [
    dict(t0=0.0, t1=1.8, p0=(1.6, -3.6, 1.6), a0=(1.9, 0, 0.9), p1=(1.5, -2.8, 1.45), a1=(1.9, 0, 0.9), lens0=24.0, lens1=24.0),
    dict(t0=1.8, t1=3.0, p0=(-2.6, -2.6, 1.6), a0=(-0.3, 0.4, 0.95), p1=(-0.8, -2.6, 1.5), a1=(0.5, 0.3, 0.95), lens0=28.0, lens1=28.0),
    dict(t0=3.0, t1=4.2, p0=(-1.65, -2.9, 1.45), a0=(-1.65, 0.7, 1.3), p1=(-1.65, -2.4, 1.4), a1=(-1.65, 0.7, 1.3), lens0=28.0, lens1=28.0),
    dict(t0=4.2, t1=5.2, p0=(-0.6, -3.5, 1.6), a0=(-0.2, 0.6, 1.15), p1=(-0.5, -3.1, 1.5), a1=(-0.1, 0.6, 1.15), lens0=24.0, lens1=24.0),
    dict(t0=5.2, t1=7.0, p0=(0.0, -3.6, 1.5), a0=(0.0, 0.6, 1.5), p1=(0.3, -5.4, 2.0), a1=(0.0, 0.6, 1.6), lens0=24.0, lens1=24.0),
    # one stable position for the finale (everyone in frame at 4:5), 7.0-10.0: only a slow eased push-in
    dict(t0=7.0, t1=10.0, p0=(0.0, -5.7, 1.7), a0=(0.0, 0.5, 1.0), p1=(0.05, -5.3, 1.65), a1=(0.0, 0.5, 1.0), lens0=22.0, lens1=22.0),
]
SHAKE = [(e["t"], 0.9 * e["s"]) for e in BR.EVENTS if e["kind"] in ("shove", "hook", "uppercut", "headbutt", "bump", "punch")]
SHAKE += [(8.3, 0.8), (8.5, 0.8), (8.95, 1.0), (9.35, 0.4), (9.6, 0.3)]
CAM.bake(SHOTS, SHAKE)
scn.render.motion_blur_shutter = 0.5   # motion blur is enabled by the hero preset; all motion is keyed per frame (no cuts inside a key pair)

# ------------------------------------------------------------------ exterior time-lapse during the +3 MONTHS card (5.4-7.0)
T_SKY0, T_SKY1 = 5.4, 7.0
for o in room.objs:
    if o.name.startswith("conference_room_blind_") or o.name == "conference_room_wall_back_sill":
        o.hide_render = True
        o.hide_viewport = True
world, SKYV = SKY.build_world(scn)
sun_l = SKY.lamp("sun_s03", "SUN", 4.0)
moon_l = SKY.lamp("moon_s03", "SUN", 0.4, (0.65, 0.75, 1.0))
SKY.key_cycle(SKYV, sun_l, moon_l, T_SKY0, T_SKY1, cycles=3.0)
ext = SKY.exterior()
for o in list(ext.objects) + [sun_l, moon_l]:   # exterior and lamps exist only during the card shot (5.2-7.0): saves about 0.8 s per frame elsewhere
    L.VA(o, F(5.2), F(7.0))
# roof lifts off for the time-lapse (dollhouse cut-away) so the sky, sun, moon and stars read from inside the room; it returns by 7.15
for o in room.objs:
    if o.name.startswith(("conference_room_ceiling", "conference_room_light_panel", "conference_room_hvac")):
        z0 = o.location.z
        for tt, dz in ((5.24, 0.0), (5.50, 9.0), (6.72, 9.0), (6.98, 0.0)):
            o.location.z = z0 + dz
            o.keyframe_insert("location", index=2, frame=F(tt))
        o.location.z = z0

asm.finalize(OUT)
