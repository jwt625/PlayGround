"""hero_cell_mating: the hero's front slot (rack L07, HOOK_bird_1_0_1 = HOOK_cell_1_1_1) as a detailed mating cell for the
S2 cutaway (DevLog-010 section 9 item 3; DevLog-007 section 11).

Frame = the slot hook frame: origin = the bird feet origin on the cell floor, +X = beak / out of the cell (the tier-0 face
plane is at x = +0.030), +Y = the bird's left, Z up. Place it at HOOK_bird_1_0_1 of rack L07 (world transform in
hall_layout.json "hero_cell"). Same envelope as the v2 slot (interior 172 x 173 x 300 mm) plus:
  - cage lip (beak latch) on a hinged pivot: bar under the beak, below eye level (rotate the pivot about local Y to swing it up)
  - host contact plate under the feet: the hero's 60-contact pin map (assets/hero/pigeon_osfp_dr8_pinmap.json) projected at
    hall scale 2.6: TX pads under the right foot, RX under the left foot, GND talon pads, VCC + low-speed pads on a belly post;
    one material per class (MAT_dc_hero_pad_<cls>, emission 0 by default) so the scene can pulse them
  - side wall toward the camera (-Y) in collection CUTAWAY_hero_cell (hide it for the cutaway)
Run: scripts/bslot.sh -b --python scripts/assets/datacenter/build_hero_cell_mating.py -- assets/components/datacenter
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import rk_lib as K  # noqa: E402
import rk_loft as LF  # noqa: E402
from mathutils import Vector  # noqa: E402
from rk_lib import MB, Part  # noqa: E402

OUT, KV = K.parse_args()
AID = "hero_cell_mating"
PROJ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PREV = os.path.join(PROJ, "outputs", "assets", "datacenter")
HERO_GLB = os.path.join(PROJ, "assets", "hero", "pigeon_osfp_dr8.glb")
PINMAP = os.path.join(PROJ, "assets", "hero", "pigeon_osfp_dr8_pinmap.json")
SB = 2.6
m = lambda x: x * 0.001
FACE = LF.FEET_IN                     # face plane x (mm) in the feet frame
SW, SH, SD = LF.SLOT_W, LF.SLOT_H, LF.SLOT_D
XB = FACE - SD                        # back wall


def hero_to_cell(p_mm):
    """Hero frame (mm, -Y beak, -X right, Z up, 1x) -> cell frame (mm, +X beak, +Y left, Z up, x 2.6)."""
    x, y, z = p_mm
    return (-y * SB, x * SB, z * SB)


# ------------------------------------------------------------------ measure the hero beak (for the lip height)
beak = dict(tip_x=166.0, bottom_z=200.0, eye_z=218.0, src="crude-film primitive x 2.6 (fallback)")
if os.path.exists(HERO_GLB):
    K.reset()
    bpy.ops.import_scene.gltf(filepath=HERO_GLB)
    pts, eyes = [], []
    for o in bpy.context.scene.objects:
        if o.type == "MESH" and "beak" in o.name.lower() or (o.type == "MESH" and "mandible" in o.name.lower()):
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
        if o.type == "MESH" and ("lenslet" in o.name.lower() or "eye" in o.name.lower()):
            eyes += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if pts:
        tip = min(p.y for p in pts) * 1000
        bot = min(p.z for p in pts) * 1000
        beak = dict(tip_x=round(-tip * SB, 1), bottom_z=round(bot * SB, 1),  # mandible bbox; finer profile: lowest beak point 196.6 mm at x 150-170
                    eye_z=round((sum(p.z for p in eyes) / len(eyes)) * 1000 * SB, 1) if eyes else 218.0,
                    src="measured on assets/hero/pigeon_osfp_dr8.glb (beak/mandible and eye/lenslet bounding boxes) x 2.6")
print("BEAK", beak)
LIP_TOP = min(beak["bottom_z"] - 4.0, beak["eye_z"] - 15.0)     # bar top 4 mm under the beak, below eye level
LIP_X = beak["tip_x"] - 12.0                                      # under the front of the beak

# ------------------------------------------------------------------ build
K.reset()
L = K.palette()
coll, root = K.new_asset(AID, accuracy="C")
cc = {k: K.sub(coll, k) for k in ("CELL_BASE", "CUTAWAY_hero_cell", "HERO_LIP", "HOST_PADS")}


def fb(mb, x0, x1, y0, y1, z0, z1, mi=0):
    mb.box2((m(x0), m(y0), m(z0)), (m(x1), m(y1), m(z1)), mi)


# cell shell (CELL_BASE): floor board, back wall + host connector, +Y wall, tread above, face slice, strip, lips, wires
base = MB()     # 0 zinc wall, 1 pcb, 2 dark steel, 3 black, 4 gold, 5 led, 6 grille, 7 champagne, 8 nickel_dk, 9 steel
fb(base, XB, FACE, -SW / 2, SW / 2, -2, 0, 1)
fb(base, XB, FACE, -SW / 2 - 18, SW / 2 + 18, -LF.FLOOR_T, -2, 2)                  # floor slab
fb(base, FACE - 2.5, FACE + 0.4, -SW / 2 - 18, SW / 2 + 18, -LF.FLOOR_T, 0, 7)      # label strip
fb(base, XB - 2, XB, -SW / 2, SW / 2, 0, SH, 2)                                    # back wall
fb(base, XB, XB + 30, -0.3 * SW, 0.3 * SW, 0, 45, 3)                               # host connector
for k in range(8):
    yy = -0.26 * SW + k * 0.52 * SW / 7
    fb(base, XB + 30, XB + 30.6, yy - 2, yy + 2, 10, 36, 4)
fb(base, XB, XB + 1.5, -6, 6, SH - 20, SH - 12, 5)
fb(base, XB, FACE - 2.5, SW / 2, SW / 2 + 2, 0, SH, 0)                              # +Y wall
fb(base, XB, FACE - 2.5, SW / 2 + 2, SW / 2 + 18, 0, SH, 2)
fb(base, XB - 10, FACE - 2.5, -SW / 2 - 18, SW / 2 + 18, SH, SH + LF.FLOOR_T - 0.5, 2)   # tread (next tier floor)
fb(base, FACE - 2.5, FACE + 0.4, -SW / 2 - 18, SW / 2 + 18, SH, SH + LF.FLOOR_T, 7)      # tread label strip
fb(base, FACE - 2.5, FACE, SW / 2, SW / 2 + 18, 0, SH, 6)                           # grille slice (+Y side)
fb(base, FACE - 4, FACE + 2, SW / 2, SW / 2 + 5, -5, SH, 8)                          # lips: +Y side and bottom
fb(base, FACE - 4, FACE + 2, -SW / 2 - 5, SW / 2 + 5, -5, 0, 8)
for yy in (72.0, 80.0):
    base.cyl((m(FACE + 1), m(yy), m(SH / 2)), m(1.6), m(SH), "z", 6, 9)
for zz in (55.0, 115.0):
    base.cyl((m(FACE + 1), m((72.0 + SW / 2) / 2), m(zz)), m(1.4), m(SW / 2 - 72.0), "y", 6, 9)
K.instantiate([Part("shell", base, [L["cage_wall_lt"], L["pcb"], L["dark_steel"], L["black"], L["gold"], L["led_g"],
                                    L["perf_black"], L["champagne"], L["nickel_dk"], L["steel"]], uv_tile=m(9.0))],
              cc["CELL_BASE"], root, AID)

# CUTAWAY_hero_cell: the -Y wall (toward the camera), its grille slice, lip and wires
cw = MB()
fb(cw, XB, FACE - 2.5, -SW / 2 - 2, -SW / 2, 0, SH, 0)
fb(cw, XB, FACE - 2.5, -SW / 2 - 18, -SW / 2 - 2, 0, SH, 2)
fb(cw, FACE - 2.5, FACE, -SW / 2 - 18, -SW / 2, 0, SH, 6)
fb(cw, FACE - 4, FACE + 2, -SW / 2 - 5, -SW / 2, -5, SH, 8)
for yy in (72.0, 80.0):
    cw.cyl((m(FACE + 1), m(-yy), m(SH / 2)), m(1.6), m(SH), "z", 6, 9)
for zz in (55.0, 115.0):
    cw.cyl((m(FACE + 1), m(-(72.0 + SW / 2) / 2), m(zz)), m(1.4), m(SW / 2 - 72.0), "y", 6, 9)
K.instantiate([Part("side_wall_cutaway", cw, [L["cage_wall_lt"], L["pcb"], L["dark_steel"], L["black"], L["gold"], L["led_g"],
                                               L["perf_black"], L["champagne"], L["nickel_dk"], L["steel"]], uv_tile=m(9.0))],
              cc["CUTAWAY_hero_cell"], root, AID)

# HERO_LIP: beak latch on a pivot at the tread front edge (rotate the pivot about local Y; negative swings the bar up)
piv = K.empty(AID + "_lip_pivot", cc["HERO_LIP"], root, (m(FACE), 0, m(SH)))
piv["note"] = "hinge at the tread front edge; rotation_euler.y = -1.2 swings the lip up and out of the hop-out path"
lp = MB()      # 0 nickel, 1 champagne tab, 2 rubber
for s in (-1, 1):                                                                    # brackets from the hinge to the bar
    lp.tube([(0, m(s * 60), m(4)), (m((LIP_X - FACE) * 0.5), m(s * 60), m(LIP_TOP - SH + 6)),
             (m(LIP_X - FACE), m(s * 60), m(LIP_TOP - SH - 4))], m(3), 8, 0)       # brackets at |y| 60 clear the hero (<= 53)
    lp.cyl((0, m(s * 60), 0), m(5), m(10), "y", 12, 0)                                # hinge knuckles
fb(lp, LIP_X - FACE - 5, LIP_X - FACE + 5, -63, 63, LIP_TOP - SH - 8, LIP_TOP - SH, 0)  # latch bar
fb(lp, LIP_X - FACE + 3, LIP_X - FACE + 7, -55, 55, LIP_TOP - SH - 16, LIP_TOP - SH - 8, 1)   # down-turned catch
fb(lp, LIP_X - FACE - 5, LIP_X - FACE + 5, -30, 30, LIP_TOP - SH, LIP_TOP - SH + 1.2, 2)      # rubber bite pad
K.instantiate([Part("lip", lp, [L["nickel"], L["champagne"], L["rubber"]], smooth=True)], cc["HERO_LIP"], piv, AID + "_lip")

# HOST_PADS: plate + pads from the hero pin map (feet pads on the floor plate, belly pads on a post)
pins = json.load(open(PINMAP))["pins"] if os.path.exists(PINMAP) else []
plate = MB()
feet_pts = [hero_to_cell(p["centre_mm"]) for p in pins if p["centre_mm"][2] < 2.0]
x0 = min(p[0] for p in feet_pts) - 12 if feet_pts else -10
x1 = max(p[0] for p in feet_pts) + 12 if feet_pts else 90
y1 = max(abs(p[1]) for p in feet_pts) + 12 if feet_pts else 45
fb(plate, x0, x1, -y1, y1, -0.9, 0.02, 0)                                            # contact plate (dark FR4), flush with the floor
fb(plate, x0 + 2, x1 - 2, -y1 + 2, y1 - 2, 0.02, 0.04, 1)                            # silkscreen field
belly = [hero_to_cell(p["centre_mm"]) for p in pins if p["centre_mm"][2] >= 2.0]
if belly:
    bx = sum(p[0] for p in belly) / len(belly)
    bz = belly[0][2]
    plate.cyl((m(bx), 0, m((bz - 1.0) / 2)), m(8), m(bz - 1.0), "z", 16, 0)          # belly pogo post
    fb(plate, bx - 12, bx + 12, -10, 10, bz - 3.0, bz - 1.0, 0)
K.instantiate([Part("host_plate", plate, [L["black"], L["pcb"]])], cc["HOST_PADS"], root, AID)
CLS_COL = dict(TX=(1.0, 0.08, 0.05), RX=(0.1, 0.85, 1.0), GND=(0.8, 0.8, 0.8), VCC=(1.0, 0.55, 0.05), LS=(0.2, 1.0, 0.4))
pad_objs = {}
for cls, col in CLS_COL.items():
    mat = K.M("hero_pad_" + cls, (0.95, 0.72, 0.28), 1.0, 0.25, col, 0.0)          # gold; emission colour = class, strength 0
    mb = MB()
    for p in [q for q in pins if q["cls"] == cls]:
        c = hero_to_cell(p["centre_mm"])
        d = Vector((-p["label_dir"][1], p["label_dir"][0], 0)).normalized()          # pad long axis (hero -> cell rotation)
        L_ = p["length_mm"] * SB * 1.15
        Wd = max(p["width_mm"] * SB, 1.6)
        n = Vector((-d.y, d.x, 0))
        z0 = (c[2] - 1.2) if c[2] > 5 else 0.05                                      # belly pads sit on the post top
        ctr = Vector((c[0], c[1], 0))
        corners = [ctr + d * (sa * L_ / 2) + n * (sb * Wd / 2) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        base_i = len(mb.v)
        for zz in (z0, z0 + 0.45):
            mb.v.extend((m(q.x), m(q.y), m(zz)) for q in corners)
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            mb.f.append(tuple(base_i + i for i in f))
            mb.mi.append(0)
    if mb.f:
        pad_objs[cls] = K.obj_from_mb("%s_pads_%s" % (AID, cls), mb, [mat], cc["HOST_PADS"], root)

# hooks
cx = (x0 + x1) / 2
K.hook("hero_host_pads", cc["HOST_PADS"], root, (m(cx), 0, m(1.5)), note="contact plate centre (top); pads per class in objects hero_cell_mating_pads_<cls>")
K.hook("hero_lip", cc["HERO_LIP"], piv, (m(LIP_X - FACE), 0, m(LIP_TOP - SH)), note="beak bite point (bar top centre), child of the lip pivot")
cam_loc = Vector((m(95), m(-560), m(205)))
cam_tgt = Vector((m(35), 0, m(95)))
rot = (cam_tgt - cam_loc).to_track_quat("-Z", "Y").to_euler()
K.hook("hero_cutaway_cam", coll, root, tuple(cam_loc), tuple(rot), lens_mm=35, target=list(cam_tgt),
       note="suggested cutaway camera (Blender camera axes: -Z looks at target); hide CUTAWAY_hero_cell and the L07 rack slot/chassis in front of it")
bpy.context.view_layer.update()

bb = K.bbox(coll)
os.makedirs(OUT, exist_ok=True)
meta = dict(asset_id=AID, date="2026-10-10", units="m",
            frame="slot hook frame: origin = bird feet origin on the cell floor; +X = beak / out of the cell; +Y = bird's left; Z up; tier-0 face plane at x = +0.030",
            placement="rack L07 HOOK_bird_1_0_1 (= HOOK_cell_1_1_1); world transform in hall_layout.json hero_cell",
            size_m=[round(bb[1][i] - bb[0][i], 4) for i in range(3)], triangles=K.count_tris(coll),
            beak=beak, lip=dict(bar_top_z_m=round(m(LIP_TOP), 4), bar_x_m=round(m(LIP_X), 4), eye_z_m=round(m(beak["eye_z"]), 4),
                                note="bar top 4 mm under the beak bottom and >= 15 mm below eye level; pivot swings it up for the hop-out"),
            pads=dict(source="assets/hero/pigeon_osfp_dr8_pinmap.json (OSFP MSA Rev 5.22 60-contact edge mapped to toes/belly) projected x 2.6; pad length x 1.15 (host pads longer than module pads), width >= 1.6 mm for visibility",
                      counts={k: sum(1 for p in pins if p["cls"] == k) for k in CLS_COL},
                      materials={k: "MAT_dc_hero_pad_%s (emission colour %s, strength 0: animate to pulse)" % (k, CLS_COL[k]) for k in CLS_COL},
                      feet="TX pads under the right foot (-Y), RX under the left foot (+Y), GND = talon-tip and separator pads; VCC + LS on the belly post"),
            collections=dict(CELL_BASE="floor, back wall + host connector, +Y wall, tread, strips, lips, cage wires", CUTAWAY_hero_cell="-Y wall (camera side); hide for the cutaway",
                             HERO_LIP="beak latch on a hinged pivot", HOST_PADS="contact plate, belly post, pads per class"),
            hooks=K.hooks_of(coll))
K.save_blend(os.path.join(OUT, AID + ".blend"))
K.export_glb(coll, os.path.join(OUT, AID + ".glb"))
K.write_json(os.path.join(OUT, AID + ".json"), meta)
print("ASSET", AID, "tris", meta["triangles"], "size_m", meta["size_m"], "pads", meta["pads"]["counts"], "lip", meta["lip"])

# ------------------------------------------------------------------ preview: cutaway with the real hero model (preview only)
if int(KV.get("previews", 1)):
    if os.path.exists(HERO_GLB):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=HERO_GLB)
        new = [o for o in bpy.data.objects if o not in before]
        hr = K.empty("PREVIEW_hero_root", bpy.context.scene.collection, None, (0, 0, 0), (0, 0, math.pi / 2))
        hr.scale = (SB, SB, SB)
        for o in new:
            if o.parent is None:
                o.parent = hr
    for cls, o in pad_objs.items():                                                  # show the pulse colours in the preview
        nt = o.data.materials[0].node_tree
        nt.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.5 if cls in ("TX", "RX") else 0.4
    K.preview_setup(res=(960, 640), samples=16, world=0.3, exposure=-0.2)
    K.floor(6, z=-0.016)
    K.light("key", "AREA", (0.3, -1.2, 0.9), 40, 0.8, (0.03, 0, 0.1))
    K.light("fill", "AREA", (1.0, 0.3, 0.6), 20, 0.8, (0.03, 0, 0.1))
    V = lambda n: os.path.join(PREV, "%s_%s.png" % (AID, n))
    K.set_visible(cc["CUTAWAY_hero_cell"], False)
    K.render_view(V("cutaway"), tuple(cam_loc), tuple(cam_tgt), lens=35)
    K.render_view(V("cutaway_pads_lip"), (m(170), m(-330), m(330)), (m(60), 0, m(110)), lens=35)
    K.render_view(V("lip_closeup"), (m(260), m(-260), m(250)), (m(150), 0, m(195)), lens=50)
    hero_objs = [o for o in bpy.data.objects if o.parent is not None and o.parent.name == "PREVIEW_hero_root"]
    for o in [o for o in bpy.data.objects if o.name.startswith("PREVIEW_hero_root")] + hero_objs:
        for d in [o] + list(o.children_recursive):
            d.hide_render = True
    K.render_view(V("pads_top"), (m(45), m(-1), m(420)), (m(45), 0, 0), lens=50)
    for o in hero_objs:
        for d in [o] + list(o.children_recursive):
            d.hide_render = False
    K.set_visible(cc["CUTAWAY_hero_cell"], True)
    K.render_view(V("front_closed"), (m(700), m(-250), m(260)), (m(40), 0, m(120)), lens=35)
