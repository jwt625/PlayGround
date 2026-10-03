"""Big wall whiteboard (about 3.0 m wide) with aluminium frame, marker tray, markers, eraser and an image-ready writing surface.
Run: Blender -b --python build_whiteboard_big.py -- <out_dir>   (out_dir = assets/components/lab_office)
Origin: bottom centre of the frame on the wall plane (y = 0 is the wall, -Y is the room side).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import lo_common as L
from lo_common import C, M, S

OUT = C.argv_after_dashes()[0]
AID = "whiteboard_big"
L.begin(AID, "B")
root = S.root

# ---------------------------------------------------------------- parameters (mm)
W_OUT = 3000.0            # outer width (parametric)
FR = 30.0                 # visible frame width
WA_W = W_OUT - 2 * FR     # writing area width
WA_H = WA_W * 9.0 / 16.0  # writing area height (16:9 so the 640x360 graph texture is not stretched)
H_OUT = WA_H + 2 * FR
FR_D = 22.0               # frame depth (y)
root["p_width_mm"] = W_OUT
root["p_writing_area_mm"] = (WA_W, WA_H)
root["p_graph_on"] = 0.0
root["p_mount_height_default_mm"] = 900.0

# ---------------------------------------------------------------- materials
m_frame = L.M("alu")
m_frame_an = C.principled("MAT_lab_office_alu_anodized", (0.78, 0.79, 0.81), metallic=1.0, rough=0.38)
S.slots.add("MAT_lab_office_alu_anodized")
m_back = L.M("zinc")


def whiteboard_material():
    m = bpy.data.materials.new("MAT_lab_office_whiteboard")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.location = (500, 0)
    nt.nodes["Material Output"].location = (800, 0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-1300, 0)
    # smudges: stretched noise, horizontal wipe streaks
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.location = (-1100, -300)
    mp.inputs["Scale"].default_value = (1.0, 1.0, 5.0)
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.location = (-900, -300)
    nz.inputs["Scale"].default_value = 2.0
    nz.inputs["Detail"].default_value = 3.0
    nz.inputs["Roughness"].default_value = 0.65
    nz.inputs["Distortion"].default_value = 0.4
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.location = (-650, -300)
    cr.color_ramp.elements[0].position = 0.45
    cr.color_ramp.elements[0].color = (0, 0, 0, 1)
    cr.color_ramp.elements[1].position = 0.75
    cr.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(nz.outputs["Fac"], cr.inputs["Fac"])
    # board colour: white lerp toward grey where smudged
    mix_s = nt.nodes.new("ShaderNodeMix")
    mix_s.data_type = "RGBA"
    mix_s.location = (-350, 100)
    mix_s.inputs["A"].default_value = (0.93, 0.935, 0.93, 1)
    mix_s.inputs["B"].default_value = (0.74, 0.745, 0.75, 1)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.9
    mul.location = (-500, -150)
    nt.links.new(cr.outputs["Color"], mul.inputs[0])
    nt.links.new(mul.outputs[0], mix_s.inputs["Factor"])
    # image overlay (EMPTY node, assembler plugs sequence), multiplied over the board
    im = nt.nodes.new("ShaderNodeTexImage")
    im.name = "SCREEN_IMAGE"
    im.label = "SCREEN_IMAGE (empty: assembler plugs the graph sequence)"
    im.extension = "EXTEND"
    im.interpolation = "Linear"
    im.location = (-650, 400)
    nt.links.new(tc.outputs["UV"], im.inputs["Vector"])
    fac = nt.nodes.new("ShaderNodeValue")
    fac.name = "BOARD_FAC"
    fac.label = "BOARD_FAC (0 blank board, 1 image)"
    fac.outputs[0].default_value = 0.0
    fac.location = (-650, 600)
    ov = nt.nodes.new("ShaderNodeMix")
    ov.data_type = "RGBA"
    ov.blend_type = "MULTIPLY"
    ov.name = "BOARD_OVERLAY"
    ov.location = (-100, 150)
    ov.inputs["B"].default_value = (1, 1, 1, 1)
    nt.links.new(mix_s.outputs["Result"], ov.inputs["A"])
    nt.links.new(im.outputs["Color"], ov.inputs["B"])
    nt.links.new(fac.outputs[0], ov.inputs["Factor"])
    nt.links.new(ov.outputs["Result"], b.inputs["Base Color"])
    # gloss: coat reflection, rougher where smudged
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.location = (-100, -300)
    rr.inputs["To Min"].default_value = 0.22
    rr.inputs["To Max"].default_value = 0.38
    nt.links.new(cr.outputs["Color"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    b.inputs["Coat Weight"].default_value = 0.2
    b.inputs["Coat Roughness"].default_value = 0.08
    b.inputs["Specular IOR Level"].default_value = 0.6
    m["plug_image_node"] = "SCREEN_IMAGE"
    m["plug_toggle_node"] = "BOARD_FAC"
    S.slots.add(m.name)
    return m


m_board = whiteboard_material()

# ---------------------------------------------------------------- board body and writing plane
body_y = -(FR_D - 4.0) + 0.0   # board front face recessed 4 mm behind the frame front (y = -22 + 4 = -18)
body = L.box("board_body", (WA_W + 20, 12.0, WA_H + 20), (0, body_y + 8.0, H_OUT / 2), M("zinc"), r=0.5)  # front face 2 mm behind the writing plane (avoids z-fighting)
back = L.box("back_plate", (W_OUT - 6, 2.0, H_OUT - 6), (0, 1.0, H_OUT / 2), m_back)
screen = L.uvquad("screen", WA_W, WA_H, (0, body_y, H_OUT / 2), m_board, facing="-y")
screen.name = AID + "_screen"
screen["is_image_screen"] = True
screen.visible_shadow = False

# ---------------------------------------------------------------- aluminium frame (4 rails, 30 x 22 profile, mitred look via boxes)
fr = []
fr.append(L.box("frame_top", (W_OUT, FR_D, FR), (0, -FR_D / 2, H_OUT - FR / 2), m_frame_an, r=2.5))
fr.append(L.box("frame_bottom", (W_OUT, FR_D, FR), (0, -FR_D / 2, FR / 2), m_frame_an, r=2.5))
for sx in (-1, 1):
    fr.append(L.box("frame_side_%s" % ("l" if sx < 0 else "r"), (FR, FR_D, H_OUT - 2 * FR + 0.4),
                    (sx * (W_OUT / 2 - FR / 2), -FR_D / 2, H_OUT / 2), m_frame_an, r=2.5))
# front lip (thin inner retaining lip)
for nm, size, loc in (("lip_top", (WA_W, 3, 6), (0, -FR_D + 1.5, H_OUT - FR - 3)),
                      ("lip_bottom", (WA_W, 3, 6), (0, -FR_D + 1.5, FR + 3))):
    L.box(nm, size, loc, m_frame_an, r=0.8)
# corner caps (plastic, grey)
for sx in (-1, 1):
    for z in (FR / 2, H_OUT - FR / 2):
        L.box("corner_cap", (FR + 1, FR_D + 1, FR + 1), (sx * (W_OUT / 2 - FR / 2), -FR_D / 2, z), M("alu_dark"), r=3.5,
              seg=3)

# ---------------------------------------------------------------- marker tray (extruded aluminium: base, front lip, end caps)
TR_L = W_OUT - 400.0
TR_D = 75.0
tray = []
tray.append(L.box("tray_base", (TR_L, TR_D, 3.0), (0, -FR_D - TR_D / 2 + 4.0, 1.5), m_frame, r=0.6))
tray.append(L.box("tray_lip_front", (TR_L, 3.0, 14.0), (0, -FR_D - TR_D + 5.5, 7.0 + 1.5), m_frame, r=1.0))
tray.append(L.box("tray_wall_back", (TR_L, 3.0, 18.0), (0, -FR_D + 3.0, 10.5), m_frame, r=1.0))
for sx in (-1, 1):
    L.box("tray_end_cap", (4.0, TR_D, 20.0), (sx * (TR_L / 2 + 2.0), -FR_D - TR_D / 2 + 4.0, 10.0), M("abs_dark"), r=1.0)
TRAY_TOP_Z = 3.0
TRAY_Y = -FR_D - TR_D / 2 + 4.0

# ---------------------------------------------------------------- markers (chisel-tip dry-erase: barrel OD ~ 18 mm, 140 mm long; estimate)
marker_cols = [("black", (0.02, 0.02, 0.025)), ("blue", (0.05, 0.15, 0.75)), ("red", (0.8, 0.05, 0.04)),
               ("green", (0.05, 0.5, 0.15)), ("orange", (0.95, 0.45, 0.05))]
m_barrel = C.principled("MAT_lab_office_marker_barrel", (0.88, 0.88, 0.86), rough=0.35)
S.slots.add("MAT_lab_office_marker_barrel")
xs = [-400, -240, -80, 80, 240]
yaws = [3, -2, 5, -4, 2]
for k, ((cn, col), x0, yaw) in enumerate(zip(marker_cols, xs, yaws)):
    mc = C.principled("MAT_lab_office_marker_" + cn, col, rough=0.4)
    S.slots.add(mc.name)
    g, e = L.push_group("marker_" + cn, (x0 - 450 + 450, TRAY_Y + (k % 2) * 4, TRAY_TOP_Z + 8.5), (0, 0, yaw), sub=False)
    # barrel along X (pen axis), nib toward +X
    L.lathe("barrel", [(0, -72), (7.4, -72), (8.4, -68), (8.4, 20), (7.0, 44), (5.6, 58), (0, 58)], (0, 0, 0), m_barrel,
            axis="x", seg=24)
    L.lathe("cap", [(0, -73), (9.0, -73), (9.0, -8), (8.6, -8), (0, -8)], (0, 0, 0), mc, axis="x", seg=24)
    L.lathe("label", [(8.7, -2), (8.7, 44)], (0, 0, 0), mc, axis="x", seg=24)
    L.lathe("nib", [(5.4, 56), (3.4, 62), (0, 62)], (0, 0, 0), mc, axis="x", seg=16)
    L.box("clip", (38, 3.0, 2.5), (-42, 0, 9.6), M("alu_dark"), r=0.6)
    L.pop_group()
# eraser
g, e = L.push_group("eraser", (470, TRAY_Y + 3, TRAY_TOP_Z), (0, 0, -6), sub=False)
L.box("body", (127, 50, 28), (0, 0, 14 + 4), C.principled("MAT_lab_office_eraser_body", (0.12, 0.2, 0.5), rough=0.5), r=3.0)
L.box("felt", (125, 48, 4), (0, 0, 2), C.principled("MAT_lab_office_eraser_felt", (0.04, 0.04, 0.045), rough=1.0), r=0.5)
S.slots.update({"MAT_lab_office_eraser_body", "MAT_lab_office_eraser_felt"})
L.pop_group()

# ---------------------------------------------------------------- hooks
L.hook("board_center", (0, body_y - 0.3, H_OUT / 2), note="centre of writing area on the surface; -Y is the normal")
L.hook("board_top_left", (-WA_W / 2, body_y - 0.3, H_OUT - FR), note="top-left corner of the writing area (UV 0,1)")
L.hook("tray_center", (0, TRAY_Y, TRAY_TOP_Z), note="centre of the marker tray")
L.hook("marker_grip", (xs[0], TRAY_Y, TRAY_TOP_Z + 8.5), note="black marker pick-up point")

# ---------------------------------------------------------------- dimensions
L.dim("outer width", W_OUT, "mm", "storyboard 'about 3.0 m class'; standard 4 ft x 10 ft board is 3048 mm wide (common catalogue size)", "B")
L.dim("outer height", round(H_OUT, 1), "mm", "derived: 16:9 writing area so the 640x360 graph sequence is not stretched; spec said about 1.5 m class", "C")
L.dim("writing area", "%.0f x %.0f" % (WA_W, WA_H), "mm", "derived (outer minus 2 x 30 mm frame)", "C")
L.dim("frame profile", "30 x 22", "mm", "estimate: extruded aluminium whiteboard frame, typical 25-35 mm wide", "C")
L.dim("tray depth", TR_D, "mm", "estimate: typical marker tray 70-80 mm", "C")
L.dim("marker barrel", "17 dia x 145", "mm", "estimate: chisel-tip dry-erase marker", "C")
L.dim("eraser", "127 x 50 x 32", "mm", "estimate: standard 5 in x 2 in felt eraser", "C")
L.dim("default mount height (board bottom)", 900, "mm", "estimate: common whiteboard bottom edge 0.9 m from floor", "C")

bp = L.finish(AID, OUT, dict(
    sources=[dict(what="whiteboard standard sizes (4 ft x 10 ft = 3048 mm class)", url="common catalogue knowledge; not a single primary source",
                  access_date="2026-10-02")],
    simplifications=["frame corners are butt-joined boxes with corner caps (no mitre profile)", "no wall mounting hardware or hanging rail",
                     "whiteboard surface reflections are procedural (coat + roughness map), no reflection probe"],
    custom_properties={"p_width_mm": "build-time outer width (rebuild to change)", "p_graph_on": "informational; the live toggle is the BOARD_FAC Value node in MAT_lab_office_whiteboard",
                       "p_mount_height_default_mm": "suggested placement height of the board bottom edge"},
    screens=[dict(object=AID + "_screen", material="MAT_lab_office_whiteboard", image_node="SCREEN_IMAGE", toggle_node="BOARD_FAC (set 1.0 after plugging)",
                  aspect="16:9 (UV 0..1 over the writing area)", note="image is MULTIPLIED over the smudged white board; graph_s2 (640x360) maps 1:1")],
    usage="S2: big whiteboard behind Gary and the Manager with the climbing curves (graph_s2 sequence).",
    coordinate_note="origin = bottom centre of the frame at the wall plane; -Y faces the room; place with z offset p_mount_height_default_mm."))

# ---------------------------------------------------------------- previews (plug the real graph sequence first; blend is already saved)
imgmat = m_board
L.plug_image(imgmat, L.tex_paths("graph_s2", 100)[99])
H = H_OUT / 1000.0
views = [("front", dict(loc=(0, -4.6, H / 2 + 0.05), target=(0, 0, H / 2), lens=50)),
         ("three_quarter", dict(loc=(2.6, -3.2, 1.8), target=(0, 0, 0.75), lens=40)),
         ("top", dict(loc=(0, -0.35, 4.2), target=(0, -0.25, 0), lens=50)),
         ("closeup_tray", dict(loc=(-0.15, -0.5, 0.2), target=(-0.15, -0.11, 0.02), lens=50)),
         ("closeup_surface", dict(loc=(0.9, -0.9, 1.1), target=(0.3, 0, 1.0), lens=50))]
L.render_views(S.top_coll, os.path.join(OUT, "previews"), AID, views, floor=False, world=0.6, sun=1.8, lights="sun", sun_shadow=False)
