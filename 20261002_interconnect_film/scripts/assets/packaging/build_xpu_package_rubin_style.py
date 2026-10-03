"""Build xpu_package_rubin_style: a Rubin-Ultra-style XPU package (4 reticle-size dies, 16 HBM stacks, CoWoS-L-class interposer).

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_xpu_package_rubin_style.py -- assets/components/packaging
Package frame: X right (long side), Y depth (-Y front), z = 0 at the substrate underside (land side). Balls hang below z = 0.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import pk  # noqa: E402
import hbm  # noqa: E402
import parts  # noqa: E402
from pk import MB, MM, PI  # noqa: E402

C = pk.C
OUT = C.argv_after_dashes()[0]
AID = "xpu_package_rubin_style"
P = AID + "_"

C.reset()
pk.reset_mats()
coll, root = C.new_asset(AID, accuracy="B")
root["p_heat"] = 0.0
root["p_z_exaggeration"] = 1.0
cb = C.sub_collection(coll, "BASE")
csrc = C.sub_collection(coll, "SOURCES")
cring = C.sub_collection(coll, "VARIANT_lid_frame")
cplate = C.sub_collection(coll, "VARIANT_lid_full_plate")
cballs = C.sub_collection(coll, "VARIANT_bga_balls")
ccut = C.sub_collection(coll, "VARIANT_cutaway")
for c_ in (cplate, cballs, ccut):
    c_.hide_render = True
    c_.hide_viewport = True
dims = pk.Dims()

# ------------------------------------------------------------------------------------------------ parameters (mm)
SUB_X, SUB_Y = 180.0, 100.0           # estimate (C): CoWoS-L 5.5x = 100x100, 9.5x = 120x150; wider here for side OE sites
SUB_T = 2.0                           # estimate (C)
CORE_T, BU_T = 1.2, 0.4
INT_X, INT_Y, INT_T = 120.0, 62.0, 0.10
C4_H = 0.08
Z_SUB = SUB_T
Z_INT0 = Z_SUB + C4_H
Z_INT1 = Z_INT0 + INT_T                # interposer top
DIE_X, DIE_Y, DIE_T = 26.0, 33.0, 0.775
UBUMP_H = 0.03
Z_DIE0 = Z_INT1 + UBUMP_H
Z_DIE1 = Z_DIE0 + DIE_T
DIE_CX = (-45.0, -17.0, 17.0, 45.0)
HBM_DY = 16.5 + 1.5 + 5.5
sp16 = hbm.Spec(16)
Z_HBM0 = Z_INT1                         # hbm source origin = bump tips
LID_OUT = (142.0, 92.0)
LID_IN = (130.0, 80.0)
LID_Z0, LID_Z1 = 2.1, 3.2
PLATE_T = 1.6
OE_X0 = 81.0                            # OE footprint center |x| (estimate C)
OE_PITCH = 11.5
OE_FP = (14.0, 10.0)                    # footprint along X (outward) x along Y (estimate C)

mat = pk.mat
rnd = random.Random(2026)


def add_obj(mb, name, mats, c=cb, parent=root, loc=(0, 0, 0)):
    return mb.to_obj(P + name, [mat(m) if isinstance(m, str) else m for m in mats], c, parent, loc)


# ------------------------------------------------------------------------------------------------ substrate
sub_pts = [(-85, 50), (89, 50), (90, 49), (90, -49), (89, -50), (-89, -50), (-90, -49), (-90, 45)]
b = MB()
b.prism(sub_pts, 0.0, BU_T, mi=(0, 1, 2))
add_obj(b, "substrate_buildup_bottom", ["substrate_land", "buildup_film", "substrate_land"])
b = MB()
b.prism(sub_pts, BU_T, BU_T + CORE_T, mi=(1, 0, 1))
add_obj(b, "substrate_core", ["core_dielectric", "core_dielectric"])
b = MB()
b.prism(sub_pts, BU_T + CORE_T, SUB_T, mi=(0, 1, 2))
sub_top = add_obj(b, "substrate_buildup_top", ["substrate", "buildup_film", "buildup_film"])

# fiducials (copper dot in a mask opening), corner marks, pin-1 triangle
b = MB()
bf = MB()
for (x, y) in ((-86, -46), (86, -46), (86, 46)):
    b.lathe([(0.0, Z_SUB), (0.95, Z_SUB), (0.95, Z_SUB + 0.01), (0.0, Z_SUB + 0.01)], c=(x, y, 0), seg=24, mi=0, smooth=False)
    bf.lathe([(0.0, Z_SUB + 0.01), (0.5, Z_SUB + 0.01), (0.5, Z_SUB + 0.03), (0.0, Z_SUB + 0.03)], c=(x, y, 0), seg=24, mi=0, smooth=False)
add_obj(b, "fiducial_mask_openings", ["substrate_land"])
add_obj(bf, "fiducials", ["gold_enig"])
b = MB()
for sx, sy in ((1, 1), (1, -1), (-1, -1)):
    cx, cy = sx * 86.5, sy * 46.5
    b.ribbon([(cx - sx * 2.5, cy), (cx, cy), (cx, cy - sy * 2.5)], 0.25, Z_SUB, Z_SUB + 0.02, mi=0)
b.prism([(-87.5, 47.8), (-83.5, 47.8), (-87.5, 43.8)], Z_SUB, Z_SUB + 0.02, mi=0)   # pin-1 triangle near the chamfered corner
add_obj(b, "corner_marks_pin1", ["gold_enig"])

# marking text (laser mark on the substrate margin)
pk.text(P + "mark_top", "XPU R1  SAMPLE  0001", 1.6, (-30.0, 48.0, Z_SUB + 0.02), mat("silkscreen"), cb, root, extrude_mm=0.02)

# ------------------------------------------------------------------------------------------------ underfill, interposer
b = MB()
b.loft_rects((INT_X + 2.4, INT_Y + 2.4, Z_SUB), (INT_X, INT_Y, Z_INT0 + INT_T * 0.9), mi=0)
add_obj(b, "underfill_fillet", ["underfill"])
b = MB()
b.box((0, 0, Z_SUB + C4_H / 2), (INT_X - 0.4, INT_Y - 0.4, C4_H), mi=0)
add_obj(b, "c4_underfill_layer", ["underfill"])
b = MB()
b.box((0, 0, Z_INT0 + INT_T / 2), (INT_X, INT_Y, INT_T), mi=0, bevel=0.015, seg=1)
interposer = add_obj(b, "interposer", ["interposer"])
# seal ring + LSI bridge strips + edge pads on the interposer top (flush details)
b = MB()
sr = [(-INT_X / 2 + 0.8, -INT_Y / 2 + 0.8), (INT_X / 2 - 0.8, -INT_Y / 2 + 0.8), (INT_X / 2 - 0.8, INT_Y / 2 - 0.8),
      (-INT_X / 2 + 0.8, INT_Y / 2 - 0.8)]
b.ribbon(sr, 0.18, Z_INT1, Z_INT1 + 0.004, mi=0, closed=True)
add_obj(b, "interposer_seal_ring", ["gold_enig"])
b = MB()
gap_x = (-31.0, 0.0, 31.0)
for gx, gw in ((-31.0, 1.0), (0.0, 3.0), (31.0, 1.0)):
    b.box((gx, 0, Z_INT1 + 0.003), (gw, 16.0, 0.006), mi=0)
for cx in DIE_CX:
    for sy in (-1, 1):
        for sx in (-7.0, 7.0):
            pass
for cx in DIE_CX:
    for sy in (-1, 1):
        b.box((cx, sy * 17.25, Z_INT1 + 0.003), (23.0, 0.9, 0.006), mi=0)   # LSI bridge strip in the die-to-HBM gap
add_obj(b, "interposer_lsi_bridges", ["silicon_light"])

# ------------------------------------------------------------------------------------------------ dies (own objects, own heat-driven materials)
die_objs = []
for k, cx in enumerate(DIE_CX):
    m = mat("die_gold").copy()
    m.name = "MAT_packaging_xpu_die_%d" % k
    pk.emission_driver(m, root, "p_heat", 4.0)
    bd = MB()
    bd.box((0, 0, DIE_T / 2), (DIE_X, DIE_Y, DIE_T), mi=0, bevel=0.12, seg=1)
    o = bd.to_obj(P + "gpu_die_%d" % k, [m], cb, root, loc=(cx, 0, Z_DIE0))
    die_objs.append(o)
# stylised bump-like pad clusters on the die backside (after the reference slide: 2 clusters of 6 x 6 pads per die)
pad_src = MB()
pad_src.lathe([(0.0, 0.0), (0.46, 0.0), (0.42, 0.025), (0.0, 0.03)], seg=10, mi=0, smooth=False)
pad_src_o = pk.source(pad_src.to_obj(P + "src_die_pad", [mat("gold_enig")], csrc, root))
pts = []
for sy in (-1, 1):
    for i in range(6):
        for j in range(6):
            pts.append(((i - 2.5) * 1.5, sy * 8.0 + (j - 2.5) * 1.5, 0.0))
for k, cx in enumerate(DIE_CX):
    pk.scatter(P + "die_pads_%d" % k, pts, pad_src_o, cb, root, loc=(cx, 0, Z_DIE1))

# ------------------------------------------------------------------------------------------------ HBM stacks (16 instances)
hb = hbm.stack_mesh(sp16, "exposed")
hsrc = pk.source(hb.to_obj(P + "src_hbm_stack", hbm.hbm_materials(), csrc, root))
hpts = []
for cx in DIE_CX:
    for sy in (-1, 1):
        for dx in (-7.0, 7.0):
            hpts.append((cx + dx, sy * HBM_DY, Z_HBM0))
hbm_pts_obj = pk.scatter(P + "hbm_stacks", hpts, hsrc, cb, root)

# ------------------------------------------------------------------------------------------------ die-side capacitors (0402 MLCC, instanced)
mb_c, cm = parts.mlcc("0402")
cap_src = pk.source(mb_c.to_obj(P + "src_cap_0402", [mat(n) for n in cm], csrc, root))
dsc = []
dsc_rz = []
x = -63.0
while x <= 63.0:
    for y in (-33.2, -34.8, -36.4, 33.2, 34.8, 36.4):
        dsc.append((x, y, Z_SUB))
        dsc_rz.append(0.0)
    x += 1.7
y = -29.5
while y <= 29.5:
    for xx in (-62.4, -63.8, 62.4, 63.8):
        dsc.append((xx, y, Z_SUB))
        dsc_rz.append(PI / 2)
    y += 1.7
pk.scatter(P + "die_side_caps", dsc, cap_src, cb, root, rz=dsc_rz)

# ------------------------------------------------------------------------------------------------ optical engine sites (hooks + land pads + footprint outlines)
site_pad = MB()
site_pad.box((0, 0, 0.015), (0.8, 0.8, 0.03), mi=0)
site_pad_o = pk.source(site_pad.to_obj(P + "src_oe_pad", [mat("gold_enig")], csrc, root))
ppts = []
bo = MB()
oe_sites = []
for side, sname in ((-1, "L"), (1, "R")):
    for i in range(8):
        sx = side * OE_X0
        sy = (i - 3.5) * OE_PITCH
        oe_sites.append((sname, i, sx, sy))
        for a in range(8):
            for bb_ in range(6):
                ppts.append((sx + (a - 3.5) * 1.4, sy + (bb_ - 2.5) * 1.4, Z_SUB))
        fx, fy = OE_FP
        bo.ribbon([(sx - fx / 2, sy - fy / 2), (sx + fx / 2, sy - fy / 2), (sx + fx / 2, sy + fy / 2), (sx - fx / 2, sy + fy / 2)],
                  0.2, Z_SUB, Z_SUB + 0.02, mi=0, closed=True)
        h = C.hook("oe_%s_%d" % (sname, i), cb, root, loc=(sx * MM, sy * MM, Z_SUB * MM),
                   rot=(0, 0, PI if side < 0 else 0.0))
pk.scatter(P + "oe_site_pads", ppts, site_pad_o, cb, root)
add_obj(bo, "oe_site_outlines", ["gold_enig"])

# ------------------------------------------------------------------------------------------------ lid frame (default variant) and plate
b = MB()
b.ring_rect(LID_OUT[0] - 0.8, LID_OUT[1] - 0.8, LID_IN[0] + 0.8, LID_IN[1] + 0.8, Z_SUB, LID_Z0, rad_o=1.5, mi=(0, 0, 0))
add_obj(b, "lid_adhesive", ["plastic_gray"], c=cring)
b = MB()
b.ring_rect(LID_OUT[0], LID_OUT[1], LID_IN[0], LID_IN[1], LID_Z0, LID_Z1, rad_o=1.5, chamfer=0.45, seg=5, mi=(0, 0, 0))
lid_ring = add_obj(b, "lid_frame", ["gold_lid"], c=cring)
b = MB()
b.box((0, 0, LID_Z1 + PLATE_T / 2), (LID_OUT[0], LID_OUT[1], PLATE_T), mi=0, bevel=0.45, seg=3)
add_obj(b, "lid_plate", ["gold_lid"], c=cplate)
b = MB()
for cx in DIE_CX:
    b.box((cx, 0, (Z_DIE1 + LID_Z1) / 2), (DIE_X - 0.3, DIE_Y - 0.3, LID_Z1 - Z_DIE1), mi=0)
    for sy in (-1, 1):
        for dx in (-7.0, 7.0):
            zt_ = Z_HBM0 + sp16.bump_um * 1e-3 + sp16.height
            b.box((cx + dx, sy * HBM_DY, (zt_ + LID_Z1) / 2), (10.4, 10.4, LID_Z1 - zt_), mi=0)
add_obj(b, "tim_pads", ["thermal_pad"], c=cplate)

# ------------------------------------------------------------------------------------------------ land side: capacitors and BGA balls
mb_l, cml = parts.mlcc("0402", t=0.30)
mb_l.mirror_z()
lsc_src = pk.source(mb_l.to_obj(P + "src_lsc_0402", [mat(n) for n in cml], csrc, root))
lsc = []
lsc_regions = []
for cx in DIE_CX:
    lsc_regions.append((cx - 12.5, cx + 12.5, -14.5, 14.5))
    for i in range(15):
        for j in range(28):
            lsc.append((cx + (i - 7) * 1.6, (j - 13.5) * 1.0, 0.0))
pk.scatter(P + "land_side_caps", lsc, lsc_src, cb, root)
BALL_D, BALL_H, BALL_PITCH = 0.6, 0.48, 1.0
bs = MB()
rr = BALL_D / 2
bs.lathe([(0.24, 0.0), (rr, -BALL_H * 0.5), (0.22, -BALL_H * 0.96), (0.0, -BALL_H)], seg=6, mi=0)
ball_src = pk.source(bs.to_obj(P + "src_ball", [mat("solder")], csrc, root))
bpts = []
nx_, ny_ = 175, 95
for j in range(ny_):
    for i in range(nx_):
        x = (i - (nx_ - 1) / 2) * BALL_PITCH
        y = (j - (ny_ - 1) / 2) * BALL_PITCH
        if any(r[0] <= x <= r[1] and r[2] <= y <= r[3] for r in lsc_regions):
            continue
        if x < -85.0 and y > 43.0:     # pin-1 corner chamfer void
            continue
        bpts.append((x, y, 0.0))
balls_obj = pk.scatter(P + "bga_balls", bpts, ball_src, cballs, root)
b = MB()
b.prism([(-87.5, 47.8), (-83.5, 47.8), (-87.5, 43.8)], -0.001, -0.02, mi=0)
add_obj(b, "pin1_marker_land_side", ["gold_enig"], c=cb)

# ------------------------------------------------------------------------------------------------ hooks
for k, cx in enumerate(DIE_CX):
    C.hook("die_%d" % k, cb, root, loc=(cx * MM, 0, Z_DIE1 * MM))
C.hook("package_center_top", cb, root, loc=(0, 0, Z_DIE1 * MM))
C.hook("package_underside_center", cb, root, loc=(0, 0, 0))
C.hook("pin1", cb, root, loc=(-85.5 * MM, 46 * MM, Z_SUB * MM))
C.hook("lid_top_center", cring, root, loc=(0, 0, (LID_Z1 + PLATE_T) * MM))
C.hook("heat_center", cb, root, loc=(0, 0, Z_DIE1 * MM))

# ------------------------------------------------------------------------------------------------ cutaway variant
zs = pk.z_exaggeration(P + "cutaway_zscale", ccut, root, root)
CUT_D = 8.0


def cut_piece(name, mb, mats, loc=(0, 0, 0)):
    mb.cut_y(0.0)
    return mb.to_obj(P + "cut_" + name, [mat(m) for m in mats], ccut, zs, loc)


def csrc_obj(name, mb, mats):
    mb.cut_y(0.0)
    return pk.source(mb.to_obj(P + "cutsrc_" + name, [mat(m) for m in mats], csrc, root))


# board slice (u runs along the package Y axis; the section plane is XZ at y = 0, cut face normal -Y)
BOARD_T = 2.4
Z_BOARD_TOP = -0.45
U0, U1 = -58.0, 58.0
b = MB()
b.box((0, CUT_D / 2, Z_BOARD_TOP - BOARD_T / 2), (U1 - U0, CUT_D, BOARD_T - 0.06), mi=0)
b.box((0, CUT_D / 2, Z_BOARD_TOP - 0.015), (U1 - U0, CUT_D, 0.03), mi=1)
b.box((0, CUT_D / 2, Z_BOARD_TOP - BOARD_T + 0.015), (U1 - U0, CUT_D, 0.03), mi=1)
for k in range(6):
    b.box((0, CUT_D / 2, Z_BOARD_TOP - 0.25 - k * 0.38), (U1 - U0, CUT_D, 0.035), mi=2)
b.box((0, CUT_D / 2, Z_BOARD_TOP + 0.0), (U1 - U0, CUT_D, 0.001), mi=1)
mb_ = b
cut_piece("board", mb_, ["fr4", "mask_green", "copper"])

# substrate layers (alternating build-up dielectric and copper, plated core with through vias)
def layer_stack(name, z0, z1, ncu, cu_t, dmat="buildup_film"):
    mbl = MB()
    n = ncu
    dt = ((z1 - z0) - n * cu_t) / n
    z = z0
    for k in range(n):
        mbl.box((0, CUT_D / 2, z + dt / 2), (100.0, CUT_D, dt), mi=0)
        z += dt
        mbl.box((0, CUT_D / 2, z + cu_t / 2), (100.0, CUT_D, cu_t), mi=1)
        z += cu_t
    cut_piece(name, mbl, [dmat, "copper"])


layer_stack("buildup_bottom", 0.0, BU_T, 8, 0.012)
layer_stack("buildup_top", BU_T + CORE_T, SUB_T, 8, 0.012)
mbl = MB()
mbl.box((0, CUT_D / 2, BU_T + CORE_T / 2), (100.0, CUT_D, CORE_T - 0.04), mi=0)
mbl.box((0, CUT_D / 2, BU_T + 0.02), (100.0, CUT_D, 0.04), mi=1)
mbl.box((0, CUT_D / 2, BU_T + CORE_T - 0.02), (100.0, CUT_D, 0.04), mi=1)
cut_piece("core", mbl, ["core_dielectric", "copper"])
# plated through vias in the core (0.2 mm drill, 0.5 mm pitch): cylinder wall + resin plug cut at the section
mbv = MB()
mbv.cyl((0, 0, BU_T + CORE_T / 2), 0.10, CORE_T + 0.02, seg=12, mi=0)
mbv.cyl((0, 0, BU_T + CORE_T / 2), 0.075, CORE_T + 0.03, seg=12, mi=1)
via_src = csrc_obj("core_via", mbv, ["copper", "steel_dark"])
pk.scatter(P + "cut_core_vias", [(u, -0.004, 0) for u in [(-48 + 0.5 * i) for i in range(0, 193)]], via_src, ccut, zs)
# stacked microvias in the build-up (cone approximated by a cylinder, 0.06 mm)
mbm = MB()
mbm.cyl((0, 0, 0), 0.03, 0.028, seg=8, mi=0)
muv = csrc_obj("microvia", mbm, ["copper"])
mv_pts = []
z = BU_T + CORE_T
dtb = (BU_T - 8 * 0.012) / 8
for k in range(8):
    zc = z + dtb / 2 + k * (dtb + 0.012)
    for i in range(0, 380):
        mv_pts.append((-48 + 0.26 * i + (0.13 if k % 2 else 0), -0.004, zc))
    zc2 = 0.0 + dtb / 2 + k * (dtb + 0.012)
    for i in range(0, 380):
        mv_pts.append((-48 + 0.26 * i + (0.13 if k % 2 else 0), -0.004, zc2))
pk.scatter(P + "cut_microvias", mv_pts, muv, ccut, zs)

# BGA balls (section row at y = 0 plus rows behind) and land-side caps in the die void
bsc = MB()
bsc.lathe([(0.24, 0.0), (0.30, -BALL_H * 0.5), (0.22, -BALL_H * 0.96), (0.0, -BALL_H)], seg=16, mi=0)
bsc.cut_y(0.0)
ball_cut = pk.source(bsc.to_obj(P + "cutsrc_ball_half", [mat("solder")], csrc, root))
bfull = MB()
bfull.lathe([(0.24, 0.0), (0.30, -BALL_H * 0.5), (0.22, -BALL_H * 0.96), (0.0, -BALL_H)], seg=8, mi=0)
ball_full = pk.source(bfull.to_obj(P + "cutsrc_ball_full", [mat("solder")], csrc, root))
us = [-48 + i for i in range(0, 97)]
void_u = [(-14.5 + 0.0, 14.5)]
bu_row = [(u, 0, 0) for u in us if not (-14.5 <= u <= 14.5)]
pk.scatter(P + "cut_balls_row", bu_row, ball_cut, ccut, zs)
bu_back = [(u, y, 0) for u in us if not (-14.5 <= u <= 14.5) for y in range(1, 8)]
pk.scatter(P + "cut_balls_back", bu_back, ball_full, ccut, zs)
mbl_ = MB()
mbl_.box((0, 0, -0.15), (0.5, 1.0, 0.30), mi=0)   # land-side cap rotated 90 deg: 1.0 mm along the section line? (0402: 1.0 x 0.5 x 0.3)
mbl_.cut_y(0.0)
lsc_cut = pk.source(mbl_.to_obj(P + "cutsrc_lsc", [mat("mlcc_body")], csrc, root))
pk.scatter(P + "cut_land_caps", [(u, 0, 0) for u in [(-12.5 + 1.6 * i) for i in range(0, 16)]], lsc_cut, ccut, zs)
# board pads under the balls (gold) as thin discs
mbp = MB()
mbp.box((0, 0, Z_BOARD_TOP + 0.0), (0.5, 0.5, 0.036), mi=0)
mbp.cut_y(0.0)
pad_cut = pk.source(mbp.to_obj(P + "cutsrc_boardpad", [mat("copper")], csrc, root))
pk.scatter(P + "cut_board_pads", [(u, 0, 0) for u in bu_row_u] if False else [(p[0], 0, 0) for p in bu_row], pad_cut, ccut, zs)

# C4 layer: underfill with C4 bumps, interposer (Si + RDL) with TSVs, microbumps, die, HBM
mbl = MB()
mbl.box((0, CUT_D / 2, Z_SUB + C4_H / 2), (INT_Y - 0.4, CUT_D, C4_H), mi=0)
cut_piece("c4_underfill", mbl, ["underfill"])
mbc = MB()
mbc.lathe([(0.04, 0.0), (0.05, 0.04), (0.04, 0.08), (0.0, 0.08)], seg=10, mi=0)
mbc.cut_y(0.0)
c4_src = pk.source(mbc.to_obj(P + "cutsrc_c4", [mat("solder")], csrc, root))
c4_pts = [(-30.0 + 0.13 * i, -0.004, Z_SUB) for i in range(0, 462)]
pk.scatter(P + "cut_c4", c4_pts, c4_src, ccut, zs)
mbl = MB()
mbl.box((0, CUT_D / 2, Z_INT0 + 0.0425), (INT_Y, CUT_D, 0.085), mi=0)
mbl.box((0, CUT_D / 2, Z_INT0 + 0.085 + 0.0075), (INT_Y, CUT_D, 0.015), mi=1)
cut_piece("interposer", mbl, ["silicon", "interposer"])
mbt = MB()
mbt.box((0, 0.0, 0), (0.010, 0.010, 0.085), mi=0)
mbt.cut_y(0.0)
tsv_i = pk.source(mbt.to_obj(P + "cutsrc_int_tsv", [mat("copper")], csrc, root))
pk.scatter(P + "cut_int_tsv", [(-30.0 + 0.13 * i, -0.004, Z_INT0 + 0.0425) for i in range(0, 462)], tsv_i, ccut, zs)
# GPU die section
mbl = MB()
mbl.box((0, CUT_D / 2, Z_DIE0 + (DIE_T - 0.012) / 2 + 0.012), (DIE_Y, CUT_D, DIE_T - 0.012), mi=0)
mbl.box((0, CUT_D / 2, Z_DIE0 + 0.006), (DIE_Y, CUT_D, 0.012), mi=1)
mbl.box((0, CUT_D / 2, Z_INT1 + UBUMP_H / 2), (DIE_Y, CUT_D, UBUMP_H * 0.95), mi=2)
cut_piece("gpu_die", mbl, ["die_gold", "base_die", "underfill"])
mbj = MB()
mbj.lathe([(0.012, 0.0), (0.0175, UBUMP_H / 2), (0.012, UBUMP_H), (0.0, UBUMP_H)], seg=8, mi=0)
mbj.cut_y(0.0)
ub_src = pk.source(mbj.to_obj(P + "cutsrc_ubump", [mat("solder")], csrc, root))
pk.scatter(P + "cut_die_ubumps", [(-16.0 + 0.055 * i, -0.004, Z_INT1) for i in range(0, 582)], ub_src, ccut, zs)
# HBM sections on both sides
for s_ in (-1, 1):
    hbm.section_parts(sp16, ccut, zs, csrc, root, P + "cut_hbm_%s" % ("a" if s_ < 0 else "b"),
                      loc=(s_ * HBM_DY, 0, Z_HBM0), tsv_pitch=0.11,
                      ubump_pts=[(x, y, z) for (x, y, z) in hbm.bump_points(sp16) if abs(y) < 0.01])
# die-side caps (0402) on the substrate, and the lid (frame legs + plate + TIM), with the TIM gap
mb_cap, cmc = parts.mlcc("0402")
mb_cap.cut_y(0.0)
capc = pk.source(mb_cap.to_obj(P + "cutsrc_cap0402", [mat(n) for n in cmc], csrc, root))
pk.scatter(P + "cut_die_side_caps", [(s_ * (33.2 + 1.6 * k), 0, Z_SUB) for s_ in (-1, 1) for k in range(3)], capc, ccut, zs)
mbl = MB()
for s_ in (-1, 1):
    mbl.box((s_ * (LID_IN[1] / 2 + 3.0), CUT_D / 2, (LID_Z0 + LID_Z1) / 2), (6.0, CUT_D, LID_Z1 - LID_Z0), mi=0, bevel=0.2, seg=1)
    mbl.box((s_ * (LID_IN[1] / 2 + 3.0), CUT_D / 2, (Z_SUB + LID_Z0) / 2), (6.0 - 0.8, CUT_D, LID_Z0 - Z_SUB), mi=1)
mbl.box((0, CUT_D / 2, LID_Z1 + PLATE_T / 2), (LID_OUT[1], CUT_D, PLATE_T), mi=0, bevel=0.3, seg=1)
cut_piece("lid", mbl, ["gold_lid", "plastic_gray"])
mbl = MB()
mbl.box((0, CUT_D / 2, (Z_DIE1 + LID_Z1) / 2), (DIE_Y - 0.3, CUT_D, LID_Z1 - Z_DIE1), mi=0)
zt_ = Z_HBM0 + sp16.bump_um * 1e-3 + sp16.height
for s_ in (-1, 1):
    mbl.box((s_ * HBM_DY, CUT_D / 2, (zt_ + LID_Z1) / 2), (10.4, CUT_D, LID_Z1 - zt_), mi=0)
cut_piece("tim", mbl, ["thermal_pad"])
# remaining substrate (outside the lid) is covered by layer_stack width 100 (u -50..50)

# ------------------------------------------------------------------------------------------------ dims / meta
dims.add("package (substrate) size X x Y", "180 x 100", "mm", "estimate: TSMC public CoWoS-L platform substrates 100x100 mm (5.5x reticle, 2025-26) and 120x150 mm (9.5x, 2027); widened to 180 mm here to fit 8+8 optical engine sites outside the lid (S4 storyboard). Range 150-190 x 95-120", "C")
dims.add("substrate thickness", SUB_T, "mm", "estimate; core 1.2 + 2 x 0.4 build-up (8 Cu layers each side); range 1.2-2.6", "C")
dims.add("interposer size", "120 x 62", "mm", "chosen so that interposer area = 7440 mm2 = 8.7 x 858 mm2 reticle (public: Rubin Ultra 'about 7.5-8x' reticle; TSMC CoWoS-L 9.5x in 2027, ~8150 mm2); layout from reference image", "C")
dims.add("interposer thickness", INT_T * 1000, "um", "CoWoS interposers are thinned to ~100 um (typical public figure); range 50-110", "C")
dims.add("GPU die size", "26 x 33", "mm", "reticle limit 26 x 33 mm = 858 mm2 (ASML scanner field); 'reticle-sized' per reference slide", "A")
dims.add("GPU die thickness", DIE_T * 1000, "um", "standard 775 um wafer thickness (assumed unthinned; range 300-775)", "C")
dims.add("number of GPU dies", 4, "count", "reference slide 'Rubin Ultra: 4 reticle-sized GPUs, 1TB HBM4e' (rubin-ultra.png); note press reports of 2026 that shipping part may be dual-die; film follows the slide", "A")
dims.add("number of HBM stacks", 16, "count", "reference slide (8 above + 8 below the die row); 16 x 64 GB = 1 TB", "A")
dims.add("HBM stack footprint", "11 x 11 (see hbm_stack)", "mm", "see hbm_stack asset", "B")
dims.add("die-to-HBM gap", 1.5, "mm", "estimate (range 1-2.5)", "C")
dims.add("die-to-die gap inside pair / central gap", "2.0 / 8.0", "mm", "estimates from reference layout (two dies each side of a central gap)", "C")
dims.add("lid frame outer / inner", "142 x 92 / 130 x 80", "mm", "estimates; ratio from the reference image (frame width about 6 mm at 142 mm outer)", "C")
dims.add("lid frame height above substrate", LID_Z1 - Z_SUB, "mm", "ring top 3.2 mm above the underside, 0.2 mm above die tops (TIM gap)", "C")
dims.add("lid plate thickness (variant)", PLATE_T, "mm", "estimate; range 1-3", "C")
dims.add("BGA pitch / ball diameter / height", "1.0 / 0.6 / 0.48", "mm", "assumption: 1.0 mm FCBGA pitch with 0.6 mm balls (see bga_lga_family); real large-AI-package pitch not public", "C")
dims.add("BGA ball count (variant)", len(bpts), "count", "array 175 x 95 minus land-side-cap voids and chamfer void", "C")
dims.add("die-side caps", len(dsc), "count", "0402 MLCC rows around the interposer inside the lid; real DSC are 01005/0201 and far denser", "C")
dims.add("land-side caps", len(lsc), "count", "0402 low-profile (0.3 mm) caps under the die area", "C")
dims.add("C4 bump pitch (cutaway)", 130, "um", "typical CoWoS C4 pitch order 130-150 um (public)", "C")
dims.add("microbump pitch die-interposer", 55, "um", "55 um HBM-class pitch; GPU die bump pitch may be finer", "B")
dims.add("OE site pitch", OE_PITCH, "mm", "estimate; 8 sites per side along the 92 mm lid span", "C")
dims.add("OE site footprint (outline)", "14 x 10", "mm", "estimate, to be matched to the photonics OE asset (range 10-25 x 8-15)", "C")
dims.add("OE site center |x|", OE_X0, "mm", "outside the lid frame (outer x = 71 mm), 3 mm gap", "C")

meta = {
    "title": "Rubin-Ultra-style XPU package (CoWoS-L class)",
    "accuracy_level": "B for layout (from reference image), C for most absolute sizes (estimates)",
    "origin": "z = 0 is the substrate underside (land side); X/Y centered on the substrate; BGA balls (variant) hang below z = 0; OE hooks sit on the substrate top surface (z = 2.0 mm)",
    "reference_comparison": {
        "followed": ["gold lid frame ring", "wide central interposer band with 4 dies in a row, two on each side of a central gap", "8 HBM above and 8 below in two groups of 4", "rows of small die-side caps inside the frame near the edges", "fine pad clusters on the dies"],
        "differs": ["real aspect ratio from public substrate numbers (1.8:1 with the OE margins) vs about 2:1 frame in the slide", "interposer extends under the HBMs (physical) whereas the slide shows a teal band only behind the dies", "die/HBM backsides are a metallized tan (as in the slide) but real Si would look darker", "pad clusters are stylised, not a real bump map"],
    },
    "sources": [
        {"what": "reference slide (4 reticle-sized GPUs, 1TB HBM4e)", "file": "references/rubin-ultra.png"},
        {"what": "TSMC CoWoS-L roadmap: 5.5x reticle (100x100 mm substrate) 2025-26, 9.5x reticle (120x150 mm) 2027", "url": "https://www.techpowerup.com/336064/tsmc-outlines-roadmap-for-wafer-scale-packaging-and-bigger-ai-packages", "accessed": "2026-10-02"},
        {"what": "CoWoS-L 9.5x reticle details", "url": "https://www.trendforce.com/news/2025/04/24/news-tsmc-tech-symposium-highlights-a14-set-for-2028-launch-9-5-reticle-cowos-arriving-in-2027/", "accessed": "2026-10-02"},
        {"what": "Rubin Ultra four reticle-sized compute dies, 16 HBM4E, interposer about 7.5-8x reticle (press); later reports of a dual-die plan", "url": "https://www.tomshardware.com/tech-industry/semiconductors/nvidia-enterprise-roadmap-rubin-rubin-ultra-feynman-and-silicon-photonics", "accessed": "2026-10-02"},
        {"what": "package electrical path cross-section (build-up layers / core / core vias / BGA ball / PCB) used for the cutaway layer order", "file": "jwt625.github.io/assets/images/2025/20251219_CPO/ieee-400g-package-electrical-path.jpg"},
        {"what": "HBM stack dimensions", "file": "assets/components/packaging/hbm_stack.json"},
    ],
    "variants": {
        "BASE": "always on: substrate, interposer, 4 GPU dies, 16 HBM (instanced), die-side and land-side caps, fiducials, OE site pads and hooks",
        "VARIANT_lid_frame": "default: gold lid frame ring (as in the reference slide) with adhesive; hide this collection for the lid-removed variant",
        "VARIANT_lid_full_plate": "hidden by default: lid plate + TIM pads (enable together with VARIANT_lid_frame for a closed lid)",
        "VARIANT_bga_balls": "hidden by default: land-side BGA balls (about 13.7k low-poly instances)",
        "VARIANT_cutaway": "hidden by default: a 100 mm long, 8 mm deep cross-section through a GPU die and its two HBM stacks (section plane XZ, cut face toward -Y; local x = package Y). Z exaggeration through custom property p_z_exaggeration on the root",
    },
    "instancing": "HBM stacks, balls, caps, pads, TSV/via/bump rows are Geometry Nodes instances (NG_pk_scatter); sources live in SOURCES (hidden)",
    "simplifications": ["no internal circuit detail, no silkscreen beyond one marking line", "die-side caps are 0402 (real: 0201/01005, denser)", "balls low-poly (6 sided)", "section plane details (microvias, TSV) are stand-ins at true pitch but not true counts"],
    "scene_usage": "S4 (top-down XPU, dolly-zoom on package, OE sites at the L/R edges); S3 reference for die-size gags",
}

blend = os.path.join(OUT, AID + ".blend")
C.save(blend)
pv = os.path.join(OUT, "previews")
bb, tris = pk.eval_stats(coll)
all_pngs = []


def setvis(lid=True, plate=False, balls=False, cut=False):
    cring.hide_render = cring.hide_viewport = not lid
    cplate.hide_render = cplate.hide_viewport = not plate
    cballs.hide_render = cballs.hide_viewport = not balls
    ccut.hide_render = ccut.hide_viewport = not cut
    cb.hide_render = cb.hide_viewport = cut
    bpy.context.view_layer.update()


def R(views, prefix=""):
    global all_pngs
    all_pngs += pk.render_views(coll, pv, AID, [dict(v, name=prefix + v["name"]) for v in views])


setvis(lid=True)
R([dict(name="front", loc=(0, -330, 90), tgt=(0, 0, 2), lens=50, floor=True),
   dict(name="three_quarter", loc=(190, -200, 170), tgt=(0, 0, 2), lens=50, floor=True),
   dict(name="top", loc=(0, -1, 265), tgt=(0, 0, 2), lens=50),
   dict(name="closeup_lid_corner", loc=(-60, -70, 28), tgt=(-62, -40, 2.5), lens=60, floor=True),
   dict(name="closeup_oe_sites", loc=(110, -42, 45), tgt=(80, -22, 2.0), lens=60, floor=True)])
setvis(lid=False)
R([dict(name="top", loc=(0, -1, 262), tgt=(0, 0, 2), lens=50),
   dict(name="three_quarter", loc=(120, -130, 100), tgt=(0, 0, 2.3), lens=50, floor=True),
   dict(name="closeup_dies_hbm", loc=(-40, -50, 38), tgt=(-30, -10, 2.4), lens=60, floor=True),
   dict(name="closeup_caps_corner", loc=(-52, -48, 12), tgt=(-58, -33, 2.2), lens=70, floor=True),
   dict(name="closeup_hbm_edge", loc=(-6, -38, 7), tgt=(-9, -23, 2.5), lens=80, floor=True)], prefix="lid_removed_")
setvis(lid=True, plate=True)
R([dict(name="three_quarter", loc=(190, -200, 170), tgt=(0, 0, 3), lens=50, floor=True)], prefix="lid_full_")
setvis(lid=False, balls=True)
R([dict(name="underside", loc=(0, 2, -300), tgt=(0, 0, 0), lens=50),
   dict(name="underside_closeup", loc=(-14, -38, -26), tgt=(-17, -12, 0), lens=70)], prefix="bga_")
setvis(lid=False, cut=True)
root["p_z_exaggeration"] = 1.0
bpy.context.view_layer.update()
R([dict(name="front", loc=(0, -190, 12), tgt=(0, 0, 2), lens=50, floor=True)], prefix="cutaway_z1_")
root["p_z_exaggeration"] = 10.0
bpy.context.view_layer.update()
R([dict(name="front", loc=(0, -150, 14), tgt=(0, 0, 11), lens=50, floor=True),
   dict(name="closeup_die_hbm", loc=(-12, -33, 33), tgt=(-24, 0, 17), lens=70, floor=True),
   dict(name="closeup_substrate", loc=(-40, -13, 20), tgt=(-40, 0, 8.5), lens=70, floor=True),
   dict(name="three_quarter", loc=(70, -120, 90), tgt=(0, 3, 12), lens=50, floor=True)], prefix="cutaway_z10_")
root["p_z_exaggeration"] = 1.0
setvis(lid=True)
C.save(blend)
hooks, mnames, props = pk.collect_auto_meta(coll, root)
meta.update({"asset_id": AID, "bbox_mm": bb, "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)],
             "triangles_including_instances_default_variant": tris, "dimension_table": dims.rows, "hooks": hooks,
             "material_slots": mnames, "custom_properties": props,
             "custom_properties_doc": {"p_heat": "0..1 drives emission (orange glow) of the 4 GPU die materials via drivers", "p_z_exaggeration": "Z scale of the cutaway variant (1 = true scale)"},
             "hook_convention": "HOOK_oe_L_i / HOOK_oe_R_i (i = 0..7, front to back): origin = OE footprint center on the substrate top; local +X points outward (away from the package center), +Z up. Footprint 14 (X) x 10 (Y) mm.",
             "previews": [os.path.relpath(p, OUT) for p in all_pngs]})
pk.write_json(blend, meta)
print("DONE", AID, tris, "balls", len(bpts))
