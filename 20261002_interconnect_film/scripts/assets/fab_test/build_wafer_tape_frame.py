"""wafer_tape_frame: 300 mm wafer mounted on UV dicing tape in a 12 inch stainless frame.
VARIANT_whole: intact SiPh wafer with die film. VARIANT_diced: every die a separate silicon tile (kerf visible), per-die object colour.
Origin: bottom centre of the tape (z = 0). Frame flats at +-X.
Run: Blender -b --python scripts/assets/fab_test/build_wafer_tape_frame.py -- assets/components/fab_test
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft, wafer_lib as W  # noqa: E402
import bpy  # noqa: E402

OUT = ft.C.argv_after_dashes()[0]
A = ft.Asset("wafer_tape_frame", accuracy="B")
MM = ft.MM
A.src("DISCO tape frames DTF2-12: outside diameter 400 mm, inside diameter 350 mm, stainless steel 1.5 mm (also 1.2 mm)",
      "https://www.disco.co.jp/eg/products/related/tapeframe.html", "frame OD/ID/thickness")
A.src("12 inch frame supplier listing (DTF2-12 style): min outer 380 mm across flats", "https://www.jenyen.com/en/product/tjf-12a-12-inch-wafer-frames/",
      "flats 380 mm across; SEMI G74/G87 style frames")
A.dim("frame OD (round) / across flats", "400 / 380", "mm", "DISCO DTF2-12; supplier min OD 380", "B")
A.dim("frame ID", 350, "mm", "DISCO DTF2-12 (opening 350 mm)", "A")
A.dim("frame thickness", 1.5, "mm", "DISCO DTF2-12", "A")
A.dim("tape thickness", 0.08, "mm", "typical dicing tape 80-150 um, estimate", "C")
A.dim("dicing kerf (diced variant)", 0.05, "mm", "blade 20-50 um class (+ chipping); task brief", "B")
A.dim("diced die piece", "7.05 x 9.05", "mm", "7.1 x 9.1 pitch minus 0.05 kerf; die 7 x 9", "B")
FT, TT, WT = 1.5, 0.08, W.T_WAFER
frame_mat, tape_mat, si = A.mat("steel"), A.mat("tape_uv"), A.mat("silicon")


def r_out(th, R=200.0, flat=190.0):
    c = abs(math.cos(th))
    return min(R, flat / c) if c > 1e-9 else R


angs = sorted(set([-math.pi + 2 * math.pi * k / 360 for k in range(360)] + [s * math.acos(190 / 200) + o for s in (1, -1) for o in (0, math.pi)] + [math.acos(190 / 200), -math.acos(190 / 200), math.pi - math.acos(190 / 200), -math.pi + math.acos(190 / 200)]))
# frame ring
verts, faces = [], []
N = len(angs)
ro = [(r_out(a) * math.cos(a), r_out(a) * math.sin(a)) for a in angs]
ri = [(175 * math.cos(a), 175 * math.sin(a)) for a in angs]
z0, z1 = TT, TT + FT
for (pts, z) in ((ro, z0), (ro, z1), (ri, z0), (ri, z1)):
    for (x, y) in pts:
        verts.append((x * MM, y * MM, z * MM))
o0, o1, i0, i1 = 0, N, 2 * N, 3 * N
for j in range(N):
    k = (j + 1) % N
    faces += [(o1 + j, o1 + k, i1 + k, i1 + j),      # top
              (o0 + k, o0 + j, i0 + j, i0 + k),      # bottom
              (o0 + j, o0 + k, o1 + k, o1 + j),      # outer wall
              (i0 + k, i0 + j, i1 + j, i1 + k)]      # inner wall
A.obj_from_mesh("wafer_tape_frame_ring", verts, faces, frame_mat, smooth=True, sharp_deg=40)
# frame marks: two locating notches as small dark slots on the flats (decor) and a label
A.text("wafer_tape_frame_label", "NC-FRAME-001", 6, (0, -187, z1 + 0.02), "black_plastic")
# tape (follows the frame outline minus 1 mm), alpha-blended blue
tv, tf = [], []
for z in (0.0, TT):
    for a in angs:
        r = r_out(a) - 1.0
        tv.append((r * math.cos(a) * MM, r * math.sin(a) * MM, z * MM))
cb = len(tv); tv.append((0, 0, 0)); ct = len(tv); tv.append((0, 0, TT * MM))
for j in range(N):
    k = (j + 1) % N
    tf += [(j, N + j, N + k, k), (cb, k, j), (ct, N + j, N + k)]
tape = A.obj_from_mesh("wafer_tape_frame_tape", tv, tf, tape_mat, smooth=False)

dies = W.die_layout()
film = W.die_material(A)
gold = A.mat("gold")
# ---- VARIANT_whole
vw = A.variant("whole")
A.cur = vw
W.disc(A, "wafer_tape_frame_wafer", si, z0=TT)
ztop = TT + WT
proto = W.build_die_proto(A, "wafer_tape_frame_die_proto", "light", [film, gold], z_top=ztop)
proto.hide_render = proto.hide_viewport = True
objs_w = W.place_dies(A, proto, dies, "wafer_tape_frame_whole", z_top=ztop)
W.pcm_and_marks(A, dies, ["copper", "gold"], z_top=ztop)
# ---- VARIANT_diced
vd = A.variant("diced")
A.cur = vd
protod = W.build_die_proto(A, "wafer_tape_frame_tile_proto", "light", [film, gold, si], si_thick=WT, size=(W.PITCH_X - 0.05, W.PITCH_Y - 0.05), z_top=ztop)
protod.hide_render = protod.hide_viewport = True
# tile body is the pitch-sized silicon piece; film (7 x 9) sits in the middle
objs_d = W.place_dies(A, protod, dies, "wafer_tape_frame_diced", z_top=ztop)
# waste strips: right bands and top bands of fields fully inside the usable radius
strips = []
for (fi, fj) in W.field_list(dies):
    X0, Y0 = fi * W.FIELD_X - W.FIELD_X / 2, fj * W.FIELD_Y - W.FIELD_Y / 2
    rx0, rx1 = X0 + 3 * W.PITCH_X + 0.025, X0 + W.FIELD_X - 0.025
    if all(math.hypot(x, y) <= W.R_USE for x in (rx0, rx1) for y in (Y0, Y0 + W.FIELD_Y)):
        strips.append(((rx0 + rx1) / 2, Y0 + W.FIELD_Y / 2, ztop - WT / 2, rx1 - rx0, W.FIELD_Y - 0.05, WT, 0))
    ty0, ty1 = Y0 + 3 * W.PITCH_Y + 0.025, Y0 + W.FIELD_Y - 0.025
    if all(math.hypot(x, y) <= W.R_USE for x in (X0, rx0) for y in (ty0, ty1)):
        strips.append(((X0 + rx0 - 0.05) / 2 + 0.025, (ty0 + ty1) / 2, ztop - WT / 2, rx0 - X0 - 0.05, ty1 - ty0, WT, 0))
st = A.boxes_obj("wafer_tape_frame_diced_strips", strips, [si], bottom=True)
st.visible_shadow = False
A.cur = A.coll
A.hook("wafer_center_top", (0, 0, ztop), A.root, size=0.05)
A.hook("frame_flat_left", (-190, 0, TT + FT / 2), A.root, size=0.02)
A.hook("frame_flat_right", (190, 0, TT + FT / 2), A.root, size=0.02)
A.preview_fit = 1.1
A.preview_shadows = False
A.preview_hide = ["VARIANT_diced"]


def show_diced():
    bpy.data.collections["VARIANT_diced"].hide_render = False
    bpy.data.collections["VARIANT_whole"].hide_render = True
meta = {"description": "300 mm wafer on UV tape in a 12 inch frame (round 400 mm with flats at 380 mm across, 350 mm opening).",
        "origin": "bottom centre of the tape at z = 0; frame flats at +-X; wafer notch at -Y",
        "variants": {"VARIANT_whole": "intact SiPh wafer: %d dies as linked-duplicate objects (colour via object colour as in wafer_300mm_siph) + test structures and marks" % len(objs_w),
                     "VARIANT_diced": "%d separate silicon die tiles (kerf 50 um) + waste strips; per-die objects recolourable the same way (MAT_fab_test_die_state, obj.color rgb + glow alpha)" % len(objs_d)},
        "scene_usage": "S5 'DICING + TAPE' station: show VARIANT_whole entering the saw, VARIANT_diced after the cut (toggle collection visibility).",
        "simplifications": ["Edge partial dies omitted in both variants", "Frame locating notches not modelled", "Tape has no adhesive layer; blue alpha blend", "Only full waste strips inside 147 mm kept"]}
A.finish(OUT, meta, views=("three_quarter", "top"),
         closeups=[("diced_overview", (0, -620, 420), (0, 0, 0), 50, show_diced), ("diced_close", (-70, -150, 55), (-40, -100, 1), 70)])
