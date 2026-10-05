"""S3 'NPO: closer, and everyone has a different idea' (film 20-30 s, scene-local 0-10 s), v1.2 assembly.

Build: FILM_HUD=0 /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s03_npo.py -- scenes/v1/s03_npo.blend
Timing, captions, labels and cards follow scripts/build_crude_film.py s3; subtitles come from scripts/audio/narration.json (asm.narr_vo).

v1.2 (2026-10-03): v2 characters (gary_v2, manager_v2, npc_<x>_v2), motion_v2 actions (brawl in s03_brawl.py, guns, falls),
eased snap push-in at 0.5-1.05 s, softened 3-cycle day/night (roof stays on, sky carries the change), one stable 3/4 finale
framing over the table (Manager and Gary in the near tier, the four vendors across the table), tamed ceiling lamps.

Layout (room frame = world frame; conference_room root at the origin, 1 unit = 1 m):
  table 5.6 x 2.2 m, top z = 0.76, x in [-2.8, 2.8], y in [-1.1, 1.1].
  Vendors on the far (+Y) side, y = 1.35. Brawl pairs centred at x = -1.27 (TERAHOP/MOLEXX) and +1.22 (NUBISS/AYARR);
  finale slots TERAHOP -1.6, MOLEXX -0.35, NUBISS 0.35, AYARR 1.6.
  Gary at (3.1, -0.9) until the cut at 7.0 s, then at (1.15, -1.95) facing +X (falls toward -X into the frame).
  The Manager enters through the door in the -X wall and stops at (-1.55, -1.25).

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
import s03_fx as FX  # noqa: E402
import s03_sky as SKY  # noqa: E402
M2 = BR.M2

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "scenes/v1/s03_npo.blend"

TABLE_Z = 0.76
S_HW, S_PKG, S_DIE, Z_DIE = 7.0, 12.0, 50.0, 3.0
T_FILM0 = 20.0  # film time of scene-local 0
P_CAM_FIN_Y = -5.1   # finale camera y (see SHOTS)

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
rot_key(door, 4.9, (0, 0, 0), "BEZIER")
rot_key(door, 5.4, (0, 0, -1.55), "BEZIER")


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


def aim_rot(src, dst):
    """Euler that points a light's -Z axis from src to dst."""
    d = Vector(dst) - Vector(src)
    return d.to_track_quat("-Z", "Y").to_euler()


# v1.2: larger, softer ceiling areas (2 m, 22 W) instead of 1 m / 30 W; warm key and cool rim on the people (critique S3 lighting)
for k, (lx, ly) in enumerate([(-2.4, -1.2), (0, -1.2), (2.4, -1.2), (-2.4, 1.2), (0, 1.2), (2.4, 1.2)]):
    area("ceil_%d" % k, (lx, ly, 2.7), 2.0, 22.0, (1.0, 0.95, 0.88))
area("front_fill", (0, -3.0, 2.3), (6.0, 2.0), 90.0, (1.0, 0.96, 0.92), rot=(PI / 2 - 0.25, 0, 0))
area("key_warm", (-2.6, -3.4, 2.6), (2.5, 2.5), 70.0, (1.0, 0.82, 0.62), rot=aim_rot((-2.6, -3.4, 2.6), (0.0, 0.8, 1.2)))
area("rim_cool", (1.5, 3.4, 2.5), (4.0, 1.0), 120.0, (0.70, 0.82, 1.0), rot=aim_rot((1.5, 3.4, 2.5), (0.0, 1.0, 1.3)))
# v1.2: lamps x0.85 so fewer pixels exceed the compositor bloom threshold (1.0 scene-linear; the hard hat and white shirts glowed).
# Note: render_presets.apply_render_preset resets view exposure to 0, so the scene cannot compensate with exposure (framework request).
LIGHT_SCALE = 0.85
for o in bpy.data.objects:
    if o.type == "LIGHT" and o.name.startswith(("ceil_", "front_fill", "key_warm", "rim_cool")):
        o.data.energy *= LIGHT_SCALE
# ceiling light panels: lower emission (blown discs with speckle in v1.1, critique X4)
for m in bpy.data.materials:
    if "light_panel" in m.name and m.use_nodes:
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                n.inputs["Emission Strength"].default_value = min(n.inputs["Emission Strength"].default_value, 0.55)
            elif n.type == "EMISSION":
                n.inputs["Strength"].default_value = min(n.inputs["Strength"].default_value, 0.55)

# ------------------------------------------------------------------ characters (v2)
VY = BR.Y_ROW  # vendor row y
VEND_ASSET = {"terahop": "npc_terahop_v2", "molexx": "npc_molexx_v2", "nubiss": "npc_nubiss_v2", "ayarr": "npc_ayarr_v2"}
VEND = [  # id, row x (0-1.7 s), finale x, dir (+1 faces +x in the finale)
    ("terahop", -2.0, -1.6, +1),
    ("molexx", -0.7, -0.35, -1),
    ("nubiss", 0.7, 0.35, +1),
    ("ayarr", 2.0, 1.6, -1),
]
V = {}
for vid, vx, fx_, d in VEND:
    a = asm.append("characters/" + VEND_ASSET[vid], actions=True)
    asm.place(a.root, (vx, VY, 0.0), yaw=0.0)
    if "p_accessory" in a.root.keys():
        a.root["p_accessory"] = 0.0     # no caps/headsets: the flying caps (s03_fx) are the gag
    V[vid] = dict(a=a, x=vx, fx=fx_, d=d)

GARY_ID, MGR_ID = "gary_v2", "manager_v2"
gary = asm.append("characters/" + GARY_ID, actions=True)
mgr = asm.append("characters/" + MGR_ID, actions=True)
GARY_POS0 = (2.95, -1.3, 0.0)
# hard hat: scene copy with base R 0.78 (asset 1.0) and rougher coat; R above 1.0 scene-linear bloomed into a fireball in the finale
_hh = bpy.data.materials.get("MAT_characters_%s_hardhat" % GARY_ID)
if _hh is not None:
    _bs = _hh.node_tree.nodes["Principled BSDF"]
    _bs.inputs["Base Color"].default_value = (0.78, 0.16, 0.01, 1.0)
    _bs.inputs["Roughness"].default_value = 0.55
GARY_POS = (1.15, -1.95, 0.0)     # finale mark (from the cut at 7.0)
asm.place(gary.root, GARY_POS0, yaw=0.0)
MGR_DOOR, MGR_END = (-4.35, -1.9, 0.0), (-1.6, -1.2, 0.0)
asm.place(mgr.root, (-5.5, -1.9, 0.0), yaw=PI / 2)

# ------------------------------------------------------------------ shotguns (clay variant is default-visible), mounted as in S1/S2 (rot z = pi)
gun_master = asm.append("props/shotgun")
guns = {"molexx": gun_master, "nubiss": asm.clone(gun_master, "ASSET_shotgun_nubiss"),
        "manager": asm.clone(gun_master, "ASSET_shotgun_manager"), "gary": asm.clone(gun_master, "ASSET_shotgun_gary"),
        "gary_drop": asm.clone(gun_master, "ASSET_shotgun_gary_drop")}
for who, host in (("molexx", V["molexx"]["a"]), ("nubiss", V["nubiss"]["a"]), ("manager", mgr)):
    asm.attach(guns[who].root, host.hook("gun_grip_R"), (0, 0, 0), (0, 0, PI))

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
mm.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.2   # v1.2: was 3.0 (bloom smear)
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
CAL_S = 1.3
CAL_P = (-0.05, -0.45)   # v1.2: calendar centre frame between the two pairs (critique: +3 MONTHS read only from the sky)
asm.place(cal.root, (CAL_P[0], CAL_P[1], 1.25), scale=CAL_S)   # pages face the camera (-Y)
for o in cal.objs:   # paper and backboard toned down (white pages clipped to a blank glowing card at draft exposure)
    for sl in getattr(o, "material_slots", []):
        m = sl.material
        if m is not None and m.name in ("MAT_props_paper", "MAT_props_clay_white"):
            if m.name + "_s03" not in bpy.data.materials:
                mc = m.copy()
                mc.name = m.name + "_s03"
                mc.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.62, 0.61, 0.58, 1)
            sl.link = "OBJECT"
            sl.material = bpy.data.materials[m.name + "_s03"]
pos_key(cal.root, 5.4, (CAL_P[0], CAL_P[1], 1.25), "LINEAR")
pos_key(cal.root, 7.0, (CAL_P[0], CAL_P[1], 1.5), "BEZIER")
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
ENV_REL = 7.0 + 30.0 / 30.0 / 1.95  # hand_over_envelope releases at frame 30; strip speed 1.95 from the cut at 7.0 (= 7.513, SFX whoosh 7.514)
asm.attach(env_a.root, gary.hook("hand_R"), (0, 0, 0), (0, 0, 0))
asm.show(env_a, 7.0, ENV_REL)
ay = V["ayarr"]["a"]
asm.attach(env_c.root, ay.hook("hand_L"), (0, 0, 0), (0, 0, 0))
asm.show(env_c, 8.0, 8.5)
asm.place(env_b.root, (1.4, -1.6, 1.1))
asm.show(env_b, ENV_REL, 8.02)

# ------------------------------------------------------------------ actions: vendors (v1.2: motion_v2 brawl, s03_brawl.py)
T_BRAWL_END = 7.06
FG = {vid: BR.Fighter(vid, V[vid]["a"]) for vid in V}
for vid, vx, fx_, d in VEND:
    a = V[vid]["a"]
    M2.apply(a, "idle_breathe", 0.0, hold=True, repeat=2.0, key_root_yaw=False)
    asm.key_prop(a.root, "p_expr_shouting", 0.0, 0.0)
    asm.key_prop(a.root, "p_expr_shouting", 1.7, 0.0)
    asm.key_prop(a.root, "p_expr_shouting", 1.9, 1.0)
    asm.key_prop(a.root, "p_expr_angry", 2.9, 0.0)
    asm.key_prop(a.root, "p_expr_angry", 3.2, 0.8)
PA, PB = BR.compose(FG["terahop"], FG["molexx"], FG["nubiss"], FG["ayarr"], K=1.07)
for vid, vx, fx_, d in VEND:
    FG[vid].bake_root(0.0, T_BRAWL_END, (vx, VY), 0.35 * (1 if vid in ("terahop", "nubiss") else -1))
print("BRAWL strips", len(BR.STRIPS), "events", len(BR.EVENTS))

# break apart (7.06-7.45): short eased slides to the finale slots with a walk strip, then face the partner/target
YAW_PX, YAW_NX = PI / 2, -PI / 2      # facing +X / -X
for vid, vx, fx_, d in VEND:
    a, fg = V[vid]["a"], FG[vid]
    p0 = fg.pos(T_BRAWL_END)
    y0 = fg.yaw()
    y1 = YAW_PX if d > 0 else YAW_NX
    for tt in (T_BRAWL_END, 7.12):
        pos_key(a.root, tt, p0, "BEZIER")
        yaw_key(a.root, tt, y0, "BEZIER")
    pos_key(a.root, 7.48, (fx_, VY, 0.0), "BEZIER")
    yaw_key(a.root, 7.40, y1, "BEZIER")
    M2.apply(a, "walk", 7.08, speed=1.2, hold=True, repeat=0.5, blend_in=4, key_root_yaw=False)

# finale vendors: TERAHOP and AYARR raise their guards (victims), MOLEXX and NUBISS draw shotguns
BANG_M, BANG_N, BANG_MGR = 8.3, 8.5, 8.95
AIM_SPEED = 1.5
for vid in ("terahop", "ayarr"):
    M2.apply(V[vid]["a"], "fight_idle", 7.45, hold=True, repeat=3.0, blend_in=6, key_root_yaw=False)
for who, bang, tgt in (("molexx", BANG_M, "terahop"), ("nubiss", BANG_N, "ayarr")):
    a = V[who]["a"]
    t_start = bang - 44.0 / 30.0 / AIM_SPEED
    M2.apply(a, "gun_raise_aim_fire", t_start, speed=AIM_SPEED, hold=True, blend_in=5, key_root_yaw=False)
    asm.show(guns[who], t_start, 10.0 if who == "molexx" else BANG_MGR + 0.02)   # NUBISS drops his gun at the hit
    g = guns[who]
    asm.fx("muzzle_flash", bang - 1.0 / 30.0, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), scale=2.2, parent=g.hook("muzzle_R"), dur=0.14)
    smoke_ring(bang + 0.03, g.hook("muzzle_R"))
    V[tgt]["bang"] = bang
# victims: fall_back_brawl at 1.2x lands on the floor (ground_hit frame 34) at 9.24 / 9.44 s = the body-fall thud cues
for tgt in ("terahop", "ayarr"):
    a = V[tgt]["a"]
    t_hit = V[tgt]["bang"]
    M2.apply(a, "fall_back_brawl", t_hit, speed=1.2, hold=True, blend_in=0, face=False, key_root_yaw=False)
    asm.key_prop(a.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_hole_1_radius", t_hit, 1.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_shouting", t_hit, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_angry", t_hit, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_scared", 0.0, 0.0, "CONSTANT")
    asm.key_prop(a.root, "p_expr_scared", t_hit, 1.0, "CONSTANT")
# NUBISS turns toward the Manager after his own shot and is hit at 8.95 (shot_hit_fall ground_hit frame 22 at 1.05x -> 9.65)
nub = V["nubiss"]["a"]
YAW_NUB_MGR = asm.yaw_to((0.35, VY), MGR_END)
yaw_key(nub.root, 8.62, YAW_PX, "BEZIER")
yaw_key(nub.root, 8.85, YAW_NUB_MGR, "BEZIER")
M2.apply(nub, "shot_hit_fall", BANG_MGR, speed=22.0 / (0.70 * 30.0), hold=True, blend_in=0, face=False, key_root_yaw=False)
asm.key_prop(nub.root, "p_hole_1_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(nub.root, "p_hole_1_radius", BANG_MGR, 1.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_scared", 0.0, 0.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_scared", 8.75, 1.0, "BEZIER")
asm.key_prop(nub.root, "p_expr_shouting", 8.7, 0.0, "CONSTANT")
asm.key_prop(nub.root, "p_expr_angry", 8.7, 0.0, "CONSTANT")
# MOLEXX survivor: smug after the last shot, lowers the gun
mol = V["molexx"]["a"]
asm.key_prop(mol.root, "p_expr_smug", 0.0, 0.0, "CONSTANT")
asm.key_prop(mol.root, "p_expr_smug", 9.0, 1.0, "BEZIER")
asm.key_prop(mol.root, "p_expr_shouting", 8.9, 0.0, "BEZIER")
asm.key_prop(mol.root, "p_expr_angry", 8.9, 0.0, "BEZIER")

# ------------------------------------------------------------------ Manager
M2.apply(mgr, "idle_breathe", 0.0, hold=True, repeat=4.0, key_root_yaw=False)
pos_key(mgr.root, 0.0, (-5.5, -1.9, 0.0), "CONSTANT")
yaw_key(mgr.root, 0.0, PI / 2 * -1, "CONSTANT")
T_MGR_WALK = 5.38
t_arr = M2.walk(mgr, T_MGR_WALK, [MGR_DOOR, MGR_END], gait="walk_brisk")
aim_yaw = asm.yaw_to(MGR_END, (0.35, VY))
yaw_key(mgr.root, t_arr, asm.yaw_to(MGR_DOOR, MGR_END), "BEZIER")
yaw_key(mgr.root, t_arr + 0.25, aim_yaw, "BEZIER")
M2.apply(mgr, "idle_breathe_tense", t_arr, hold=True, repeat=2.0, blend_in=5, key_root_yaw=False)
T_MGR_RAISE = BANG_MGR - 44.0 / 30.0
M2.apply(mgr, "gun_raise_aim_fire", T_MGR_RAISE, hold=True, blend_in=5, face=False, key_root_yaw=False)
asm.show(guns["manager"], T_MGR_RAISE, 10.0)
asm.fx("muzzle_flash", BANG_MGR - 1.0 / 30.0, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), scale=3.0, parent=guns["manager"].hook("muzzle_R"), dur=0.14)
smoke_ring(BANG_MGR + 0.03, guns["manager"].hook("muzzle_R"))
asm.show(mgr, T_MGR_WALK - 0.1, 10.0)
asm.key_prop(mgr.root, "p_anger", 0.0, 0.0)
asm.key_prop(mgr.root, "p_anger", 7.4, 0.6)
asm.key_prop(mgr.root, "p_anger", 8.9, 1.0)
asm.key_prop(mgr.root, "p_flush", 7.4, 0.0)
asm.key_prop(mgr.root, "p_flush", 8.9, 0.8)

# ------------------------------------------------------------------ actions: Gary
M2.apply(gary, "idle_breathe", 0.0, hold=True, repeat=4.0, key_root_yaw=False)
pos_key(gary.root, 0.0, GARY_POS0, "CONSTANT")
yaw_key(gary.root, 0.0, 0.0, "CONSTANT")
GARY_FACE_AYARR = asm.yaw_to(GARY_POS, (1.6, VY))
pos_key(gary.root, 7.0, GARY_POS, "CONSTANT")          # cut at 7.0: Gary on his finale mark
yaw_key(gary.root, 7.0, GARY_FACE_AYARR, "CONSTANT")
YAW_GARY_FIN = -0.35                                    # 3/4 toward the camera; the gun (from his left, +X) reads in silhouette
yaw_key(gary.root, 7.62, GARY_FACE_AYARR, "BEZIER")
yaw_key(gary.root, 7.95, YAW_GARY_FIN, "BEZIER")
yaw_key(gary.root, 8.96, YAW_GARY_FIN, "BEZIER")
_v = Vector((-GARY_POS[0], P_CAM_FIN_Y - GARY_POS[1], 0.0)).normalized()   # Gary -> finale camera (horizontal)
YAW_GARY_HOLE = math.atan2(-_v.y, -_v.x)   # facing f with right-hand side (f.y, -f.x) = _v
yaw_key(gary.root, 9.06, YAW_GARY_HOLE, "BEZIER")      # the blast spins him so his right side (head-hole axis) faces the camera; he falls toward -X
_st = asm.play(gary, "hand_over_envelope", 7.0, speed=1.95, hold=True, blend_in=0)
# the cut at 7.0 teleports Gary: sub-frame keys 0.25 frame early so the centred motion-blur shutter never sees his old mark
if _st is not None and hasattr(_st, "frame_start_ui"):
    _st.frame_start_ui = _st.frame_start - 0.25
for _fc in gary.root.animation_data.action.fcurves:
    if _fc.data_path in ("location", "rotation_euler"):
        _fc.keyframe_points.insert(F(7.0) - 0.25, _fc.evaluate(F(7.0)), options={"FAST"}).interpolation = "CONSTANT"
        _fc.update()
M2.apply(gary, "idle_breathe", 7.75, hold=True, repeat=2.0, blend_in=8, key_root_yaw=False)
T_GARY_FALL = 9.06   # hit-stop hold 9.06-9.19 with the head-hole axis toward the camera, ground hit 9.79
M2.apply(gary, "shot_hit_fall", T_GARY_FALL, hold=True, blend_in=0, face=False, key_root_yaw=False)
asm.key_prop(gary.root, "p_expr_worried", 0.0, 0.0)
asm.key_prop(gary.root, "p_expr_worried", 1.0, 0.7)
asm.key_prop(gary.root, "p_expr_worried", 6.0, 0.7)
asm.key_prop(gary.root, "p_expr_worried", 7.0, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 2.5, 0.0)
asm.key_prop(gary.root, "p_expr_sweating", 5.0, 0.8)
asm.key_prop(gary.root, "p_expr_sweating", 8.0, 0.8)
asm.key_prop(gary.root, "p_expr_dread", 7.6, 0.0)
asm.key_prop(gary.root, "p_expr_dread", 8.2, 1.0)
asm.key_prop(gary.root, "p_expr_dread", 8.93, 1.0)
asm.key_prop(gary.root, "p_expr_dread", 8.95, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_expr_dead_eyed", 8.92, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_expr_dead_eyed", 8.95, 1.0, "CONSTANT")
# earlier holes heal per the storyboard schedule (film times: hole 1 at 9.2 s, hole 2 at 19.2 s)
key_heal(gary.root, "p_hole_1_radius", 9.2)
key_heal(gary.root, "p_hole_2_radius", 19.2)
asm.key_prop(gary.root, "p_head_hole_radius", 0.0, 0.0, "CONSTANT")
asm.key_prop(gary.root, "p_head_hole_radius", BANG_MGR, 1.0, "CONSTANT")
# readability at 4:5: the property is clamped to 1.0, so the head-hole empty is scaled 1.6x across its axis (cut radius and rim
# follow the empty; base radius 0.035 m -> 0.056 m). Scene-only change, documented in DevLog-003-scene-s03.md (v1.2).
_he = next(o for o in gary.objs if o.name.startswith("HOLE_headhole"))
_he.scale = (_he.scale[0] * 1.6, _he.scale[1] * 1.6, _he.scale[2])

# Gary's own shotgun at the head (HOOK_muzzle_self), fired from his left (+X, frame right) with the stock up and out so the gun
# reads in silhouette; left hand IK-pulled to the fore-end. The gun drops at 9.03.
gg = guns["gary"]
T_GUN_UP, T_GUN_DROP = 8.25, 9.03
bpy.context.scene.frame_set(F(8.6))
bpy.context.view_layer.update()
hk = gary.hook("muzzle_self")
b_dir = Vector((0.88, -0.30, 0.36)).normalized()         # world: from the head toward the stock (Gary faces the camera 3/4, his left is +X)
Rw = b_dir.to_track_quat("Y", "Z").to_matrix()           # gun local +Y -> b (muzzle fires along gun -Y, i.e. into the head)
gg.root.parent = None
gg.root.matrix_world = Matrix.Identity(4)
bpy.context.view_layer.update()
mz = gg.hook("muzzle_R")
mz_local = gg.root.matrix_world.inverted() @ mz.matrix_world.translation
p_target = hk.matrix_world.translation + 0.10 * b_dir
gun_world = Matrix.Translation(p_target - Rw @ mz_local) @ Rw.to_4x4()
gg.root.parent = hk
gg.root.matrix_parent_inverse.identity()
gg.root.matrix_basis = hk.matrix_world.inverted() @ gun_world
asm.show(gg, T_GUN_UP, T_GUN_DROP)
arm = gary.armature
pb = arm.pose.bones["ik_hand_L"]
cl = pb.constraints.new("COPY_LOCATION")
cl.target = gg.hook("foregrip_L")
cl.influence = 0.0
cl.keyframe_insert("influence", frame=F(8.15))
cl.influence = 1.0
cl.keyframe_insert("influence", frame=F(8.45))
cl.keyframe_insert("influence", frame=F(T_GUN_DROP))
cl.influence = 0.0
cl.keyframe_insert("influence", frame=F(T_GUN_DROP + 0.1))
asm.fx("muzzle_flash", BANG_MGR - 1.0 / 30.0, loc=(0, 0, 0), rot=(-PI / 2, 0, 0), scale=1.6, parent=gg.hook("muzzle_R"), dur=0.12)
# dropped gun: free clone keyed from the held transform at the drop frame to the floor (tumble, one bounce)
bpy.context.scene.frame_set(F(T_GUN_DROP))
bpy.context.view_layer.update()
mw0 = gg.root.matrix_world.copy()
gd = guns["gary_drop"]
gd.root.parent = None
p0 = mw0.translation.copy()
e0 = mw0.to_euler()
gd.root.rotation_mode = "XYZ"
DROP = [(0.0, (0.0, 0.0, 0.0), (e0.x, e0.y, e0.z)),
        (0.22, (-0.12, 0.05, -0.55), (e0.x * 0.5, e0.y + 0.6, e0.z)),
        (0.38, (-0.25, 0.10, None), (0.0, PI / 2, e0.z + 0.3)),
        (0.47, (-0.30, 0.11, 0.10), (0.0, PI / 2 - 0.15, e0.z + 0.35)),
        (0.56, (-0.33, 0.12, None), (0.0, PI / 2, e0.z + 0.38))]
for dt, (dx, dy, dz), rr in DROP:
    z = p0.z + dz if (dz is not None and dt < 0.4) else (0.05 if dz is None else dz)
    pos_key(gd.root, T_GUN_DROP + dt, (p0.x + dx, p0.y + dy, z), "BEZIER")
    rot_key(gd.root, T_GUN_DROP + dt, rr, "BEZIER")
asm.show(gd, T_GUN_DROP, 10.0)
bpy.context.scene.frame_set(1)

# ------------------------------------------------------------------ fight effects: dust puffs and clay crumbs on the impact frames, papers, caps
fx = FX.Fx()
for e in BR.EVENTS:
    if e["kind"] in ("shove", "hook", "uppercut", "punch", "headbutt", "bump"):
        fx.puff(e["t"], e["pt"], e["dir"], e["s"], n=2)
        fx.crumbs(e["t"], e["pt"], e["dir"], e["s"], e["vic"], n=3 + int(4 * e["s"]))
    elif e["kind"] == "tug":
        fx.puff(e["t"], (e["pt"][0], e["pt"][1], 0.06), e["dir"], 0.2, n=1)
PAPER_BURSTS = [(3.18, (-1.2, VY - 0.25, 1.3), 4, 1.0), (4.90, (1.4, VY - 0.25, 1.4), 4, -1.0), (5.98, (-1.6, VY - 0.25, 1.5), 4, -1.0),
                (6.62, (1.5, VY - 0.25, 1.2), 3, 1.0)]
for tp, pt, n_, dr in PAPER_BURSTS:
    fx.papers(tp, pt, n_, d=dr)
# caps: NUBISS (green) flies off at the third windmill hit, TERAHOP (purple) at the uppercut (cap_whoosh cues 5.72 / 5.98)
fx.hat("nubiss", (0.15, 0.55, 0.28), 5.72, (-2.0, 0.3, 2.8), (5.0, 3.0, 14.0), V["nubiss"]["a"].hook("head_top"))
fx.hat("terahop", (0.40, 0.22, 0.65), 5.98, (-1.9, -0.4, 3.4), (4.0, -2.0, -12.0), V["terahop"]["a"].hook("head_top"))

# ------------------------------------------------------------------ labels, captions, subtitles (narration.json)
asm.timecode(3)
asm.narr_vo(3)
L.OVL["BIGHI"] = dict(L.OVL["BIG"], size=0.40, loc=(0, 1.62))   # small, above the heads (critique: BANG BANG cropped over the action)
L.WRAP["BIGHI"] = 12
L.ovt("BIGHI", "BANG BANG", 0, BANG_M, BANG_N + 0.4)
L.ovt("BIGHI", "BANG", 0, BANG_MGR, BANG_MGR + 0.55)
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

asm.wl("NPO MODULE", (1.95, -0.42, 0.98), 0.2, 1.8, size=0.11)
asm.wl("LASER IN PACKAGE", (XPU_X - 0.25, -0.45, 0.98), 1.8, 3.0, size=0.10)
asm.wl("LASER EXTERNAL", (-0.55, 0.15, 0.98), 2.0, 3.0, size=0.10)
for a_, px, sz, label, t0 in pk_assets:
    asm.wl(label, (px, -0.25, TABLE_Z + 0.03), t0, 4.2, size=0.11 if "FINE" not in label else 0.09)
asm.wl("DIE SIZE?", (-1.3, DIE_Y - 0.35, 0.98), 4.2, 5.2, size=0.12)
asm.wl("2D", (-1.4, DIE_Y - 0.35, 0.98), 5.2, 6.0, size=0.14)
asm.wl("3D", (TOWER[0] + 0.35, TOWER[1] - 0.2, 0.98), 6.2, 7.0, size=0.14)
ct = asm.wl("+3 MONTHS", (CAL_P[0], CAL_P[1] - 0.15, 2.0), 5.4, 7.0, size=0.20, color=(1, 0.4, 0.3))
pos_key(ct, 5.4, (CAL_P[0], CAL_P[1] - 0.15, 2.45))
pos_key(ct, 7.0, (CAL_P[0], CAL_P[1] - 0.15, 2.6))
et = asm.wl("YEAR-END BONUS", (1.0, -1.95, 2.0), 7.0, 8.3, size=0.10, color=(0.4, 1.0, 0.4))
pos_key(et, 7.0, (1.0, -1.95, 2.0))
pos_key(et, ENV_REL, (1.0, -1.95, 2.0))
pos_key(et, 8.0, (1.6, VY - 0.3, 2.1))
pos_key(et, 8.3, (1.6, VY - 0.3, 2.1))
for (vid, vx, fx_, d), t0 in zip(VEND, (2.2, 3.6, 5.0, 6.4)):
    pa = FG[vid].pos(t0)
    asm.wl("@#$%!", (pa[0], VY + 0.1, 2.25), t0, t0 + 0.9, size=0.22, color=(1.0, 0.35, 0.3))
asm.wl("GARY'S GUN", (GARY_POS[0] + 0.85, GARY_POS[1] - 0.2, 1.75), 8.3, 8.95, size=0.11)

# envelope flight (arc from Gary's hand to AYARR's hand); the handover now plays from the cut at 7.0 (release 7.513)
pos_key(env_b.root, ENV_REL, (GARY_POS[0] + 0.3, GARY_POS[1] + 0.3, 1.15), "LINEAR")
pos_key(env_b.root, (ENV_REL + 8.0) / 2, (1.45, -0.2, 1.9), "BEZIER")
pos_key(env_b.root, 8.0, (V["ayarr"]["fx"] - 0.15, VY - 0.4, 1.15), "BEZIER")
rot_key(env_b.root, ENV_REL, (0, 0, 0), "LINEAR")
rot_key(env_b.root, 8.0, (0.4, 0.3, 2.5), "LINEAR")

# ------------------------------------------------------------------ camera (v1.2): calm first and last 0.5 s, eased snap push-in, one stable finale
P_WIDE, A_WIDE = (1.9, -5.6, 2.05), (1.6, 0.1, 1.0)
P_TBL, A_TBL = (2.2, -4.2, 1.72), (2.0, -0.3, 0.85)
P_FIN0, P_FIN1, A_FIN = (0.0, -5.15, 2.65), (0.0, -5.0, 2.62), (0.0, 0.2, 0.95)   # high 3/4: near-tier heads sit below the vendors' chests
SHOTS = [
    dict(t0=0.0, t1=0.5, p0=P_WIDE, a0=A_WIDE, p1=(1.9, -5.55, 2.04), a1=A_WIDE, lens0=24.0, lens1=24.0, ease=False),   # calm (transition T2)
    dict(t0=0.5, t1=1.05, p0=(1.9, -5.55, 2.04), a0=A_WIDE, p1=P_TBL, a1=A_TBL, lens0=24.0, lens1=30.0),              # snap push-in, eased
    dict(t0=1.05, t1=1.8, p0=P_TBL, a0=A_TBL, p1=(2.15, -4.05, 1.70), a1=(2.0, -0.3, 0.85), lens0=30.0, lens1=30.0),
    dict(t0=1.8, t1=3.0, p0=(0.0, -2.75, 1.55), a0=(0.0, 0.9, 1.15), p1=(0.0, -2.45, 1.5), a1=(0.0, 0.9, 1.15), lens0=28.0, lens1=28.0),
    dict(t0=3.0, t1=4.2, p0=(-1.25, -2.75, 1.42), a0=(-1.05, 0.9, 1.12), p1=(-1.15, -2.45, 1.38), a1=(-0.95, 0.9, 1.12), lens0=29.0, lens1=29.0),
    dict(t0=4.2, t1=5.2, p0=(-0.15, -3.0, 1.5), a0=(-0.05, 0.8, 1.1), p1=(-0.05, -2.7, 1.45), a1=(0.0, 0.8, 1.1), lens0=28.0, lens1=28.0),
    dict(t0=5.2, t1=7.0, p0=(0.0, -2.75, 1.18), a0=(0.0, 1.5, 1.52), p1=(0.1, -3.15, 1.22), a1=(0.0, 1.5, 1.55), lens0=24.0, lens1=24.0),   # low, windows (sky) behind the fights
    # one stable position for the finale (Wentao: everyone in frame at 28-29 s): 3/4 over the table, slow eased push-in to 9.4, then still
    dict(t0=7.0, t1=9.4, p0=P_FIN0, a0=A_FIN, p1=P_FIN1, a1=A_FIN, lens0=33.0, lens1=34.0),
    dict(t0=9.4, t1=10.01, p0=P_FIN1, a0=A_FIN, p1=P_FIN1, a1=A_FIN, lens0=34.0, lens1=34.0),
]
SHAKE = [(e["t"], 0.9 * e["s"]) for e in BR.EVENTS if e["kind"] in ("shove", "hook", "uppercut", "headbutt", "bump", "punch")]
SHAKE += [(BANG_M, 0.6), (BANG_N, 0.6), (BANG_MGR, 0.8)]
CAM.bake(SHOTS, [(t, s) for t, s in SHAKE if t < 9.3], shake_ang=0.006)   # v1.2: half the v1.1 shake (motion blur smeared whole frames)
scn.render.use_motion_blur = True
scn.render.motion_blur_shutter = 0.35

# ------------------------------------------------------------------ exterior time-lapse during the +3 MONTHS card (5.4-7.0)
# v1.2: the roof stays on; blinds and sill are hidden so the sky, sun, moon and stars run in the windows only (interior lighting
# changes by the window lamps only, about 25 percent at most); lifted night sky, warm sunset glow, cool moon (s03_sky)
T_SKY0, T_SKY1 = 5.4, 7.0
for o in room.objs:
    if o.name.startswith(("conference_room_blind_", "conference_room_window_glass_")) or o.name == "conference_room_wall_back_sill":   # v1.2: glass hidden (it greyed the sky)
        o.hide_render = True
        o.hide_viewport = True
world, SKYV = SKY.build_world(scn)
sun_l = SKY.lamp("sun_s03", "SUN", 1.2)
moon_l = SKY.lamp("moon_s03", "SUN", 0.3, (0.62, 0.72, 1.0))
ext = SKY.exterior()
SKY.key_cycle(SKYV, sun_l, moon_l, T_SKY0, T_SKY1, cycles=3.0)
for o in list(ext.objects) + [sun_l, moon_l]:   # exterior and lamps exist only during the card shot (5.2-7.0)
    L.VA(o, F(5.2), F(7.0))

asm.finalize(OUT)
