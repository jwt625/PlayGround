"""Pluggable transceiver builders (OSFP, QSFP-DD) shared by the module, cable and faceplate assets.

Frame (all mm): origin = bottom centre of the module at the forward-stop plane (datum B for OSFP, datum D for QSFP-DD), +Y toward the
host connector (insertion direction), -Y toward the optical port / pull tab, +Z up.
"""
import math
import random

import bpy
import numpy as np
from mathutils import Vector

import ic_common as I
import ic_optics as O
from ic_common import MB, _v, mm

C = I.C

# ------------------------------------------------------------------ dimension tables (value, unit, source, accuracy)
OSFP_DIMS = [
    ("module width", 22.58, "mm", "OSFP MSA Rev 5.22 Fig 3-2 (+-0.10)", "A"),
    ("module height (in cage, top of heat sink to bottom)", 13.00, "mm", "OSFP MSA Rev 5.22 Fig 3-2 (+0.10/-0.20)", "A"),
    ("module length Type 1 (max)", 100.40, "mm", "OSFP MSA Rev 5.22 Fig 3-2/3-3", "A"),
    ("in-cage body length (front of body to forward stop)", 69.8, "mm", "OSFP MSA Rev 5.22 Fig 3-2 (min)", "A"),
    ("rear overhang beyond forward stop (card-edge guard)", 10.6, "mm", "derived from Fig 3-2: 100.4 - 69.8 - 20", "B"),
    ("front extension outside cage", 20.0, "mm", "derived from Fig 3-2 scale", "B"),
    ("front envelope max width / pull tab", 22.93, "mm", "OSFP MSA Rev 5.22 Fig 3-2 note 1", "A"),
    ("pull tab / latch may extend below bottom", 1.6, "mm", "OSFP MSA Rev 5.22 Fig 3-2 note 1 (max)", "A"),
    ("flat top / heat sink thermal area length from forward stop", 56.0, "mm", "OSFP MSA Rev 5.22 Fig 3-15 (+-0.5)", "A"),
    ("heat sink cavity height", 9.2, "mm", "OSFP MSA Rev 5.22 Fig 3-16", "A"),
    ("fin/vent geometry (8 vents x 1.57, 7 fins x 1.00, rails 2 x 1.50)", 1.57, "mm", "OSFP MSA Rev 5.22 Fig 3-16 (example 2)", "A"),
    ("PCB thickness", 1.00, "mm", "OSFP MSA Rev 5.22 Fig 3-21 (+-0.10)", "A"),
    ("card-edge pad pitch / width", 0.60, "mm", "OSFP MSA Rev 5.22 Fig 3-21 (pad width 0.38 +-0.03)", "A"),
    ("card-edge contacts per side", 30, "count", "OSFP MSA Rev 5.22 (60 contacts total)", "A"),
    ("latch pocket distance from forward stop", 44.59, "mm", "OSFP MSA Rev 5.22 Fig 3-26 (+-0.08)", "A"),
    ("PCB vertical position (bottom of PCB above module bottom)", 4.4, "mm", "read from section drawing Fig 3-16 (dimension 4.40)", "B"),
    ("DSP package 15 x 15 x 1.6 mm, gap pad 16 x 16 x 1.0 mm", 15.0, "mm", "generic 800G DSP size estimate range 12-20 mm", "C"),
    ("optical sub-assembly sizes (PIC 6 x 5 mm, lens block 6.6 x 4 x 2.4 mm, driver 4 x 2 mm)", 6.0, "mm", "estimated from teardown photos (Innolight 800G PSM8 interior), +-30%", "C"),
    ("MPO-16 receptacle window", 12.6, "mm", "estimate (MPO housing 12.4 mm wide + clearance); not read from a primary drawing", "C"),
]
QSFP_DIMS = [
    ("module width", 18.35, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (+-0.1)", "A"),
    ("module height in cage", 8.5, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (+-0.1)", "A"),
    ("front (outside-cage) section max width", 19.0, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (max)", "A"),
    ("front section height reference (13.5 = 8.5 + 3.4 above + 1.6 below)", 13.5, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (ref, 3.4 max above, 1.6 max below)", "A"),
    ("back portion length from front section to rear end", 58.26, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (ref)", "A"),
    ("distance front-section end to datum D", 48.2, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (min)", "A"),
    ("rear overhang beyond datum D", 10.06, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (10.06 +-0.15)", "A"),
    ("Type 1 front section length (max)", 20.0, "mm", "QSFP-DD HW Rev 5.1 Fig 33", "A"),
    ("Type 2A heat sink length (min) / height above module", 22.0, "mm", "QSFP-DD HW Rev 5.1 Fig 34/56 (22 min length; 3.4 max above module)", "A"),
    ("Type 2A fin geometry (10 vent holes, 0.4 mm fins, 3.8 ref)", 0.4, "mm", "QSFP-DD HW Rev 5.1 Fig 56 (ref)", "A"),
    ("module length Type 1 without handle", 78.3, "mm", "derived: 58.26 + 20", "B"),
    ("paddle card width", 16.42, "mm", "QSFP-DD HW Rev 5.1 Fig 36 (+-0.08)", "A"),
    ("pads per side (2 rows x 19, pitch 0.8)", 38, "count", "QSFP-DD HW Rev 5.1 Fig 36/Table 1 (76 pads)", "A"),
    ("PCB thickness", 1.0, "mm", "QSFP-DD HW Rev 5.1 Fig 36 (ref)", "A"),
    ("PCB bottom height above module bottom", 2.25, "mm", "QSFP-DD HW Rev 5.1 Fig 35 (2.25 +-0.1, interpreted)", "B"),
    ("latch position from datum D", 29.6, "mm", "QSFP-DD HW Rev 5.1 Fig 33 (+-0.1)", "A"),
    ("dual LC pitch", 6.25, "mm", "IEC 61754-20 / TIA-604-10 (as referenced in QSFP-DD HW Fig 18)", "B"),
    ("DSP package 12.5 x 12.5 x 1.4 mm", 12.5, "mm", "generic 400G DSP size estimate range 10-17 mm", "C"),
    ("TOSA/ROSA 14 x 6 x 4 mm", 14.0, "mm", "estimated from Innolight 200G FR4 TOSA photo and Intel CWDM4 teardown, +-30%", "C"),
]


def drivers_open(root, top_grp, pcb_grp, top_lift=40.0, pcb_lift=16.0):
    I.add_prop(root, "p_open", 0.0, 0.0, 1.0, "0 closed, 1 exploded (top shell lifts, PCB assembly lifts)")
    I.drive_prop(top_grp, "location", 2, root, "p_open", expr_scale=mm(top_lift))
    I.drive_prop(pcb_grp, "location", 2, root, "p_open", expr_scale=mm(pcb_lift))


def _ov(a, b, pad=0.2):
    return not (a[1] + pad < b[0] or b[1] + pad < a[0] or a[3] + pad < b[2] or b[3] + pad < a[2])


def scatter(rng, rects_zone, exclude, specs, placed, y_lim=None):
    """specs: list of (w, d, count). Returns list of (cx, cy, w, d). rects: (x0,x1,y0,y1)."""
    out = []
    for (w, d, cnt) in specs:
        tries = 0
        k = 0
        while k < cnt and tries < cnt * 60:
            tries += 1
            z = rects_zone[rng.randrange(len(rects_zone))]
            cx = rng.uniform(z[0] + w / 2, z[1] - w / 2)
            cy = rng.uniform(z[2] + d / 2, z[3] - d / 2)
            r = (cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2)
            if any(_ov(r, e) for e in exclude) or any(_ov(r, p, 0.15) for p in placed):
                continue
            placed.append(r)
            out.append((cx, cy, w, d))
            k += 1
    return out


def card_edge_pads(M, y_edge, pitch, width, n, plen_a, plen_b, z_top, z_bot, thick=0.04, rows=1, row_gap=0.0, name="gold_fingers"):
    """Gold fingers: n pads per side, alternating long/short (OSFP) or `rows` rows (QSFP-DD)."""
    mb = MB()
    xs = [(i - (n - 1) / 2) * pitch for i in range(n)]
    for side, z0 in ((1, z_top), (-1, z_bot)):
        for i, x in enumerate(xs):
            plen = plen_a if i % 2 == 0 else plen_b
            if side == 1:
                zr = (z0, z0 + thick)
            else:
                zr = (z0 - thick, z0)
            mb.box((x - width / 2, x + width / 2), (y_edge - plen, y_edge - 0.05), zr)
    return mb.build(name, M["gold"], smooth_deg=0)


def card_edge_pads_2row(M, y_edge, pitch, width, n, len1, len2, gap, z_top, z_bot, thick=0.04, name="gold_fingers"):
    mb = MB()
    xs = [(i - (n - 1) / 2) * pitch for i in range(n)]
    for side, z0 in ((1, z_top), (-1, z_bot)):
        for row, ln, yo in ((0, len1, 0.0), (1, len2, len1 + gap)):
            for i, x in enumerate(xs):
                if side == 1:
                    zr = (z0, z0 + thick)
                else:
                    zr = (z0 - thick, z0)
                x_off = 0.0
                mb.box((x - width / 2 + x_off, x + width / 2 + x_off), (y_edge - yo - ln, y_edge - yo - 0.05), zr)
    return mb.build(name, M["gold"], smooth_deg=0)


def label_and_text(M, coll, parent, body_lines, x_face, y_c, z_c, size_mm, plate_mm, name="label"):
    """Label plate + printed text on the +X side wall (x = x_face). plate_mm = (length along Y, height along Z)."""
    L, Hh = plate_mm
    mb = MB()
    mb.box((x_face, x_face + 0.08), (y_c - L / 2, y_c + L / 2), (z_c - Hh / 2, z_c + Hh / 2))
    pl = mb.build(name + "_plate", M["label_white"], smooth_deg=0)
    C.add(pl, coll, parent)
    objs = [pl]
    for k, line in enumerate(body_lines):
        zz = z_c + (len(body_lines) - 1) / 2 * size_mm * 1.45 - k * size_mm * 1.45
        t = I.text_obj(name + "_text%d" % k, line, size_mm, (x_face + 0.08, y_c, zz), rot=(math.pi / 2, 0, math.pi / 2), mat=M["ink_black"], extrude_mm=0.02)
        C.add(t, coll, parent)
        objs.append(t)
    return objs


# ================================================================== OSFP
def build_osfp(coll, root, M, rng_seed=7, tab_mat="tab_orange", with_variants=True):
    rng = random.Random(rng_seed)
    W, H = 22.58, 13.0
    hw = W / 2
    Y_F, Y_R, Y_BF = -89.8, 10.6, -69.8
    z_split, wall = 7.0, 0.6
    parts = {}

    top_grp = I.empty("osfp_module_top_group", coll, root)
    pcb_grp = I.empty("osfp_module_pcb_group", coll, root)
    tab_grp = I.empty("osfp_module_tab_group", coll, root)
    drivers_open(root, top_grp, pcb_grp)
    I.add_prop(root, "p_pull_mm", 0.0, 0.0, 30.0, "pull-tab / latch release travel toward the front (mm); drives tab group -Y")
    I.drive_prop(tab_grp, "location", 1, root, "p_pull_mm", expr_scale=-0.001)

    # ---- bottom shell (nickel-plated zinc / aluminium look)
    mb = MB()
    mb.box((-hw, hw), (Y_F, 0), (0, 0.6), bev=0.15)                                  # floor
    for s in (-1, 1):
        x0, x1 = (hw - wall, hw) if s > 0 else (-hw, -hw + wall)
        mb.box((x0, x1), (Y_F, 0), (0.6, z_split), bev=0.12)                          # side walls
    # front plate with MPO-16 window 12.6 x 8.4 centred at z = 6.5
    wx, wz0, wz1 = 6.3, 1.9, 10.3
    ZC = 6.1
    fy0, fy1 = Y_F, Y_F + 1.5
    mb.box((-hw, -wx), (fy0, fy1), (0.6, 11.0), bev=0.12)
    mb.box((wx, hw), (fy0, fy1), (0.6, 11.0), bev=0.12)
    mb.box((-wx, wx), (fy0, fy1), (0.6, wz0), bev=0.1)
    mb.box((-wx, wx), (fy0, fy1), (wz1, 11.0), bev=0.1)
    # rear: guard side posts and lower guard plate (protect the card edge)
    for s in (-1, 1):
        mb.box((s * 10.0 - 0.45, s * 10.0 + 0.45), (0, Y_R), (0.6, 3.4), bev=0.1)
    mb.box((-9.4, 9.4), (0.5, Y_R - 1.0), (2.2, 2.8), bev=0.1)
    # forward-stop shoulders at the rear (left/right vertical side walls are the forward stop)
    shell_b = mb.build("osfp_module_shell_bottom", M["nickel"], smooth_deg=35)
    I.place(shell_b, coll, root)
    parts["shell_bottom"] = shell_b

    # ---- receptacle sleeve (MPO-16 housing seen through the window) and ferrule holder
    mb = MB()
    sx0, sx1, sz0, sz1 = 6.3, 6.8, wz0 - 0.5, wz1 + 0.5
    ry0, ry1 = Y_F + 0.2, Y_F + 13.0
    mb.box((-sx1, -wx), (ry0, ry1), (sz0, sz1), bev=0.1, seg=1)
    mb.box((wx, sx1), (ry0, ry1), (sz0, sz1), bev=0.1, seg=1)
    mb.box((-wx, wx), (ry0, ry1), (sz0, wz0), bev=0.1, seg=1)
    mb.box((-wx, wx), (ry0, ry1), (wz1, sz1), bev=0.1, seg=1)
    # key notch bump on top of the receptacle (MPO-16 offset key)
    mb.box((1.0, 3.6), (ry0, ry1 - 1), (wz1 - 0.9, wz1), bev=0.1, seg=1)
    sleeve = mb.build("osfp_module_receptacle", M["plastic_black"], smooth_deg=35)
    I.place(sleeve, coll, root)

    # ---- top shell: walls, base plate, fins (open top) and nose cap with ramp
    mb = MB()
    for s_ in (-1, 1):
        x0, x1 = (hw - wall, hw) if s_ > 0 else (-hw, -hw + wall)
        mb.box((x0, x1), (-60.0, 0), (z_split + 0.02, 8.6), bev=0.12)
        mb.box((x0, x1), (fy1, -60.0), (z_split + 0.02, 11.0), bev=0.12)
    mb.box((-hw, hw), (-60.0, 0), (8.6, 9.2), bev=0.12)                               # base plate
    mb.box((-hw, hw), (fy1, -66.5), (11.0, 11.8), bev=0.15)                            # nose cap
    mb.prism_yz([(-66.5, 11.0), (-60.0, 12.2), (-60.0, 13.0), (-66.5, 11.8)], -hw, hw)  # ramp up to the fin height
    fin_y = (-60.0, -2.2)
    mb.box((-hw, -hw + 1.5), fin_y, (9.2, 13.0), bev=0.15, seg=1)
    mb.box((hw - 1.5, hw), fin_y, (9.2, 13.0), bev=0.15, seg=1)
    xcur = -hw + 1.5 + 1.57
    for k in range(7):
        mb.box((xcur, xcur + 1.0), fin_y, (9.2, 13.0), bev=0.1, seg=1)
        xcur += 1.0 + 1.57
    mb.prism_yz([(-2.2, 9.2), (0.0, 9.2), (0.0, 12.2), (-2.2, 13.0)], -hw, hw)         # rear end block
    top_shell = mb.build("osfp_module_shell_top", M["nickel"], smooth_deg=35)
    I.place(top_shell, coll, top_grp)
    parts["shell_top"] = top_shell
    # flat-top cover (closed-top heat sink variant): a plate over the fins with the 10 tunnels left open at the ends
    mb = MB()
    mb.box((-hw, hw), (-60.0, -2.2), (12.4, 13.0), bev=0.15, seg=1)
    flat_cover = mb.build("osfp_module_flat_top_cover", M["nickel"], smooth_deg=35)
    I.place(flat_cover, coll, top_grp)
    parts["flat_cover"] = flat_cover

    mb = MB()
    mb.box((-8.8, 8.8), (-40.0, -23.0), (8.0, 8.6), bev=0.15, seg=1)                 # copper boss between plate and DSP pad (x ~ +-8)
    boss = mb.build("osfp_module_thermal_boss", M["copper"], smooth_deg=35)
    I.place(boss, coll, top_grp)

    # rear tongue plate over the card edge with four vent slots (OSFP Fig 3-13/3-17)
    mb = MB()
    tz = (7.4, 8.0)
    mb.box((-hw, -9.3), (0, Y_R - 0.8), tz, bev=0.1, seg=1)
    mb.box((9.3, hw), (0, Y_R - 0.8), tz, bev=0.1, seg=1)
    ns, total, slot_w = 4, 18.6, 3.2
    bar = (total - ns * slot_w) / (ns + 1)
    cur = -9.3
    for k in range(ns + 1):
        mb.box((cur, cur + bar), (0, Y_R - 0.8), tz, bev=0.08, seg=1)
        cur += bar + slot_w
    mb.box((-9.3, 9.3), (Y_R - 3.0, Y_R - 0.8), tz, bev=0.08, seg=1)
    mb.box((-9.3, 9.3), (0, 1.0), tz, bev=0.08, seg=1)
    tongue = mb.build("osfp_module_tongue", M["nickel"], smooth_deg=35)
    I.place(tongue, coll, top_grp)

    # ---- PCB assembly
    zpb, zpt = 4.4, 5.4
    pcb_y0 = -66.0
    mb = MB()
    mb.box((-9.5, 9.5), (pcb_y0, Y_R), (zpb, zpt), bev=0.06, seg=1)
    pcb = mb.build("osfp_module_pcb", M["fr4_green"], smooth_deg=30)
    I.place(pcb, coll, pcb_grp)
    parts["pcb"] = pcb
    gf = card_edge_pads(M, Y_R, 0.6, 0.38, 30, 3.2, 2.6, zpt, zpb, name="osfp_module_gold_fingers")
    I.place(gf, coll, pcb_grp)
    parts["gold_fingers"] = gf

    # DSP + substrate + thermal pad
    dx, dy, ds = 0.0, -31.5, 15.0
    mb = MB()
    mb.box((dx - ds / 2, dx + ds / 2), (dy - ds / 2, dy + ds / 2), (zpt, zpt + 0.35), bev=0.1, seg=1)
    sub = mb.build("osfp_module_dsp_substrate", M["fr4_core"], smooth_deg=30)
    I.place(sub, coll, pcb_grp)
    mb = MB()
    mb.box((dx - ds / 2 + 0.3, dx + ds / 2 - 0.3), (dy - ds / 2 + 0.3, dy + ds / 2 - 0.3), (zpt + 0.35, zpt + 1.6), bev=0.15, seg=2)
    dsp = mb.build("osfp_module_dsp", M["mold_black"], smooth_deg=35)
    I.place(dsp, coll, pcb_grp)
    parts["dsp"] = dsp
    mb = MB()
    mb.box((dx - 8.0, dx + 8.0), (dy - 8.0, dy + 8.0), (zpt + 1.6, zpt + 2.6), bev=0.2, seg=2)
    pad = mb.build("osfp_module_thermal_pad", M["thermal_pad"], smooth_deg=35)
    I.place(pad, coll, pcb_grp)
    parts["thermal_pad"] = pad
    # marking on the DSP mold: generic text (chip brand would be a parody; kept generic here)
    t = I.text_obj("osfp_module_dsp_text", "DSP-800G", 1.6, (dx, dy, zpt + 1.6 + 0.02), mat=M["gold"], extrude_mm=0.02)
    C.add(t, coll, pcb_grp)

    # drivers/TIAs, optical engines
    tx_x, rx_x = -5.6, 5.6
    mb = MB()
    for xc in (tx_x, rx_x):
        mb.box((xc - 2.2, xc + 2.2), (-46.5, -44.2), (zpt, zpt + 0.4), bev=0.05, seg=1)
    drv = mb.build("osfp_module_driver_tia", M["mold_black"], smooth_deg=35)
    I.place(drv, coll, pcb_grp)
    parts["driver_tia"] = drv
    mb = MB()
    for xc in (tx_x, rx_x):
        for i in range(8):
            mb.box((xc + (i - 3.5) * 0.45 - 0.15, xc + (i - 3.5) * 0.45 + 0.15), (-48.0, -47.2), (zpt, zpt + 0.05))
    pads = mb.build("osfp_module_bond_pads", M["gold"], smooth_deg=0)
    I.place(pads, coll, pcb_grp)
    mb = MB()
    for xc in (tx_x, rx_x):
        mb.box((xc - 3.0, xc + 3.0), (-54.8, -49.0), (zpt, zpt + 0.35), bev=0.05, seg=1)     # PIC / carrier die
    pic = mb.build("osfp_module_pic_dies", M["silicon"], smooth_deg=30)
    I.place(pic, coll, pcb_grp)
    parts["pic"] = pic
    mb = MB()
    for xc in (tx_x, rx_x):
        mb.box((xc - 3.3, xc + 3.3), (-59.6, -54.8), (zpt, zpt + 2.4), bev=0.12, seg=1)
    lens = mb.build("osfp_module_lens_blocks", M["glass"], smooth_deg=35)
    I.place(lens, coll, pcb_grp)
    parts["lens_blocks"] = lens
    mb = MB()
    for xc in (tx_x, rx_x):
        for i in range(8):
            mb.sphere((xc + (i - 3.5) * 0.45, -59.6, zpt + 1.2), 0.17, segs=8, rings=5, sy=0.5)
    lenslets = mb.build("osfp_module_lenslets", M["glass"], smooth_deg=0)
    I.place(lenslets, coll, pcb_grp)
    mb = MB()
    for xc in (tx_x, rx_x):
        mb.box((xc - 3.6, xc + 3.6), (-55.3, -48.3), (zpt - 0.0, zpt + 1.6), bev=0.35, seg=2)
    pot = mb.build("osfp_module_potting", M["potting"], smooth_deg=40)
    I.place(pot, coll, pcb_grp)
    parts["potting"] = pot

    # MT-16 ferrule (APC) with guide pins in the receptacle, ferrule holder, 16 fibers
    fy = Y_F + 5.0
    mts = O.mt_ferrule(M, 16, 1, apc=True, pins=True, name="osfp_module_mt16", fibers=True)
    for o in mts:
        I.place(o, coll, pcb_grp, loc_mm=(0, fy, ZC))
    mb = MB()
    mb.box((-4.8, 4.8), (fy + 8.0, fy + 11.0), (ZC - 1.9, ZC + 1.9), bev=0.2, seg=1)
    holder = mb.build("osfp_module_ferrule_holder", M["plastic_beige"], smooth_deg=35)
    I.place(holder, coll, pcb_grp)
    # fibers
    paths = []
    for i in range(16):
        xa = (tx_x if i < 8 else rx_x) + ((i % 8) - 3.5) * 0.45
        xe = (i - 7.5) * 0.25
        ys, ye = -59.6, fy + 11.0
        pts = np.array([[xa, ys, zpt + 1.2], [xa, ys - 4.5, zpt + 1.3], [(xa + xe) / 2, (ys + ye) / 2, 6.3], [xe, ye + 3.0, ZC], [xe, ye, ZC]]) * 0.001
        paths.append(I.catmull(pts, per_seg=7))
    fib = I.tube_mesh("osfp_module_fibers", np.array(paths), 0.00012, sides=6, mat=M["fiber_coat"], caps=False)
    I.place(fib, coll, pcb_grp)
    parts["fibers"] = fib

    # passives
    placed = []
    excl = [(dx - ds / 2 - 0.5, dx + ds / 2 + 0.5, dy - ds / 2 - 0.5, dy + ds / 2 + 0.5),
            (tx_x - 4, tx_x + 4, -60, -43.5), (rx_x - 4, rx_x + 4, -60, -43.5), (-9.5, 9.5, 2.2, Y_R)]
    zones = [(-9.0, 9.0, -43.0, -39.5), (-9.0, -8.0 + 0.0, -39.5, -22), (8.0, 9.0, -39.5, -22), (-9.0, 9.0, -22.5, 1.5), (-9.0, 9.0, -64.5, -60.5)]
    zones_big = [(-9.0, 9.0, -43.0, 1.5)]
    caps = scatter(rng, zones_big, excl, [(1.0, 0.5, 70), (0.6, 0.3, 30)], placed)
    ress = scatter(rng, zones_big, excl, [(1.0, 0.5, 36)], placed)
    inds = scatter(rng, zones_big, excl, [(2.6, 2.0, 5)], placed)
    ics = scatter(rng, zones_big, excl, [(3.2, 3.2, 3), (4.0, 3.0, 2)], placed)
    mb = MB()
    for cx, cy, w, d in caps:
        mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + d * 0.9), bev=0.05, seg=1)
    I.place(mb.build("osfp_module_mlcc", M["ceramic_tan"], smooth_deg=40), coll, pcb_grp)
    mb = MB()
    for cx, cy, w, d in ress:
        mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + 0.35), bev=0.04, seg=1)
    I.place(mb.build("osfp_module_resistors", M["mold_black"], smooth_deg=40), coll, pcb_grp)
    mb = MB()
    for cx, cy, w, d in inds:
        mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + 1.3), bev=0.15, seg=1)
    I.place(mb.build("osfp_module_inductors", M["inductor"], smooth_deg=40), coll, pcb_grp)
    mb = MB()
    for cx, cy, w, d in ics:
        mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + 0.9), bev=0.1, seg=1)
    I.place(mb.build("osfp_module_pmic_mcu", M["mold_black"], smooth_deg=40), coll, pcb_grp)
    # mounting screw bosses at PCB corners (as in the teardown photos)
    mb = MB()
    for sx in (-1, 1):
        mb.cyl((sx * 8.2, -63.2, zpt + 0.3), 1.1, 0.6, axis="Z", segs=14)
    I.place(mb.build("osfp_module_screw_pads", M["gold"], smooth_deg=0), coll, pcb_grp)

    # ---- pull tab with loop, latch release rails, label
    tab_z = (-1.6, -0.8)
    mb = MB()
    out = I.rounded_rect(15.0, 60.0, 5.0, n=6, cy=0)
    inn = I.rounded_rect(10.6, 52.5, 3.5, n=6, cy=0)
    # shift: loop strap spans y from -84.0 to -144.0 (60 mm total); rounded outline in XY plane
    out = [(x, y - 114.0) for x, y in out]
    inn = [(x, y - 114.0) for x, y in inn]
    mb.ring_xy(out, inn, tab_z[0], tab_z[1])
    tab = mb.build("osfp_module_pull_tab", M[tab_mat], smooth_deg=35)
    I.place(tab, coll, tab_grp)
    mb = MB()
    for s in (-1, 1):
        mb.box((s * 9.8 - 0.8, s * 9.8 + 0.8), (-86.0, -40.0), (-1.2, 0.0), bev=0.1, seg=1)   # latch release rails under the module
    mb.box((-9.8, 9.8), (-86.0, -84.0), (-1.2, 0.0), bev=0.1, seg=1)
    rails = mb.build("osfp_module_latch_rails", M["plastic_black"], smooth_deg=35)
    I.place(rails, coll, tab_grp)
    # latch pockets on both side walls at 44.59 mm in front of the forward stop
    mb = MB()
    for s in (-1, 1):
        mb.box((hw - 0.02, hw + 0.02) if s > 0 else (-hw - 0.02, -hw + 0.02), (-44.59 - 2.5, -44.59 + 2.5), (1.5, 5.0))
    pockets = mb.build("osfp_module_latch_pockets", M["black_alu"], smooth_deg=0)
    I.place(pockets, coll, root)
    # label on the side flank (finned variant: sides are free), small
    lab = label_and_text(M, coll, root, ["GENERIC 800G DR8", "OSFP  SN 0000000"], hw, -45.0, 4.0, 1.6, (26.0, 5.0), name="osfp_module_label")

    # ---- hooks
    h = C.hook("plug_axis", coll, root, loc=_v(0, 0, 6.5))   # local +Y = insertion direction (toward host connector)
    h.empty_display_size = 0.012
    ho = C.hook("open", coll, top_grp, loc=_v(0, -35.0, 13.0))
    ho.empty_display_size = 0.01
    C.hook("receptacle", coll, root, loc=_v(0, Y_F, ZC))
    C.hook("pull_tab_end", coll, tab_grp, loc=_v(0, -144.0, -1.2))
    C.hook("card_edge", coll, root, loc=_v(0, Y_R, 4.9))
    return dict(top_grp=top_grp, pcb_grp=pcb_grp, tab_grp=tab_grp, parts=parts, size=(W, H, Y_F, Y_R))


# ================================================================== QSFP-DD
def build_qsfpdd(coll, root, M, rng_seed=11, tab_mat="tab_blue"):
    rng = random.Random(rng_seed)
    W, H = 18.35, 8.5
    hw = W / 2
    FW = 19.0               # front section width
    fhw = FW / 2
    Y_R = 10.06
    Y_FS0, Y_FS1 = -48.2, -48.2 - 22.0     # front section (Type 2A heat sink length 22 min); Type 1 uses 20
    Y_F = Y_FS1
    z_split = 4.2
    top_grp = I.empty("qsfp_dd_module_top_group", coll, root)
    pcb_grp = I.empty("qsfp_dd_module_pcb_group", coll, root)
    tab_grp = I.empty("qsfp_dd_module_tab_group", coll, root)
    drivers_open(root, top_grp, pcb_grp, top_lift=30.0, pcb_lift=12.0)
    I.add_prop(root, "p_pull_mm", 0.0, 0.0, 30.0, "pull-tab travel toward the front (mm)")
    I.drive_prop(tab_grp, "location", 1, root, "p_pull_mm", expr_scale=-0.001)
    parts = {}

    # bottom shell: floor, walls to z_split, front section lower block from z = -1.6 with the LC window
    lw, lz0, lz1 = 6.45, 1.6, 7.8
    lc_z = 4.7
    mb = MB()
    mb.box((-hw, hw), (Y_FS0, Y_R - 1.2), (0, 0.5), bev=0.12)
    for s_ in (-1, 1):
        x0, x1 = (hw - 0.5, hw) if s_ > 0 else (-hw, -hw + 0.5)
        mb.box((x0, x1), (Y_FS0, 0.0), (0.5, z_split), bev=0.1)
    mb.box((-fhw, -lw), (Y_FS1, Y_FS0), (-1.6, z_split), bev=0.12)
    mb.box((lw, fhw), (Y_FS1, Y_FS0), (-1.6, z_split), bev=0.12)
    mb.box((-lw, lw), (Y_FS1, Y_FS0), (-1.6, lz0), bev=0.12)
    shell_b = mb.build("qsfp_dd_module_shell_bottom", M["nickel"], smooth_deg=35)
    I.place(shell_b, coll, root)
    parts["shell_bottom"] = shell_b

    # top shell: plate over the in-cage portion, side walls, front-section upper blocks and window top bar
    mb = MB()
    for s_ in (-1, 1):
        x0, x1 = (hw - 0.5, hw) if s_ > 0 else (-hw, -hw + 0.5)
        mb.box((x0, x1), (Y_FS0, 0.0), (z_split + 0.02, 7.9), bev=0.1)
        x0, x1 = (lw, fhw) if s_ > 0 else (-fhw, -lw)
        mb.box((x0, x1), (Y_FS1, Y_FS0), (z_split + 0.02, 8.5), bev=0.12)
    mb.box((-hw, hw), (Y_FS0, 0.0), (7.9, H), bev=0.12)
    mb.box((-lw, lw), (Y_FS1, Y_FS0), (lz1, H), bev=0.12)
    top_shell = mb.build("qsfp_dd_module_shell_top", M["nickel"], smooth_deg=35)
    I.place(top_shell, coll, top_grp)
    parts["shell_top"] = top_shell
    # front-section top variants: Type 1 flat cap (to 11.9) or Type 2A extruded heat sink (11 fins of 0.4 mm, 3.4 mm tall)
    mb = MB()
    mb.box((-fhw, fhw), (Y_FS1, Y_FS0), (8.5, 11.9), bev=0.2)
    flat_cap = mb.build("qsfp_dd_module_front_cap_flat", M["nickel"], smooth_deg=35)
    mb = MB()
    nfin = 11
    span = FW - 0.4
    pitch = span / (nfin - 1)
    for k in range(nfin):
        xc = -span / 2 + k * pitch
        mb.box((xc - 0.2, xc + 0.2), (Y_FS1, Y_FS0), (8.5, 11.9), bev=0.05, seg=1)
    finned = mb.build("qsfp_dd_module_front_fins", M["nickel"], smooth_deg=35)
    I.place(flat_cap, coll, top_grp)
    I.place(finned, coll, top_grp)
    parts["flat_cap"] = flat_cap
    parts["fins"] = finned

    # dual LC receptacle: two sleeves (T/R) pitch 6.25 at z = 4.9
    mb = MB()
    for s_ in (-1, 1):
        xc = s_ * 6.25 / 2
        mb.cyl((xc, Y_FS1 + 6.0, lc_z), 1.6, 12.0, axis="Y", segs=18)
    sleeves = mb.build("qsfp_dd_module_lc_sleeves", M["steel"], smooth_deg=40)
    I.place(sleeves, coll, root)
    # receptacle housing (plastic) behind the window
    rh = MB()
    rh.box((-6.45, 6.45), (Y_FS1 + 1.5, Y_FS1 + 14.0), (lz0 + 0.6, lz1 - 0.6), bev=0.2)
    housing = rh.build("qsfp_dd_module_lc_housing", M["plastic_black"], smooth_deg=35)
    I.place(housing, coll, root)
    # LC ferrules visible in the window (ferrule faces at y = Y_FS1 + 4)
    lcs = []
    for s in (-1, 1):
        for o in O.round_ferrule(M, 1.25, 6.6, name="qsfp_dd_module_lc_ferrule_%s" % ("T" if s < 0 else "R")):
            I.place(o, coll, pcb_grp, loc_mm=(s * 6.25 / 2, Y_FS1 + 4.0, lc_z))

    # PCB assembly
    zpb, zpt = 2.25, 3.25
    pcb_y0 = -43.0
    mb = MB()
    mb.box((-8.3, 8.3), (pcb_y0, Y_R), (zpb, zpt), bev=0.06, seg=1)
    pcb = mb.build("qsfp_dd_module_pcb", M["fr4_green"], smooth_deg=30)
    I.place(pcb, coll, pcb_grp)
    parts["pcb"] = pcb
    gf = card_edge_pads_2row(M, Y_R, 0.8, 0.5, 19, 1.9, 1.5, 0.8, zpt, zpb, name="qsfp_dd_module_gold_fingers")
    I.place(gf, coll, pcb_grp)
    parts["gold_fingers"] = gf
    # DSP
    dx, dy, ds = 0.0, -14.0, 12.5
    mb = MB()
    mb.box((dx - ds / 2, dx + ds / 2), (dy - ds / 2, dy + ds / 2), (zpt, zpt + 0.3), bev=0.08, seg=1)
    I.place(mb.build("qsfp_dd_module_dsp_substrate", M["fr4_core"], smooth_deg=30), coll, pcb_grp)
    mb = MB()
    mb.box((dx - ds / 2 + 0.25, dx + ds / 2 - 0.25), (dy - ds / 2 + 0.25, dy + ds / 2 - 0.25), (zpt + 0.3, zpt + 1.4), bev=0.12)
    dsp = mb.build("qsfp_dd_module_dsp", M["mold_black"], smooth_deg=35)
    I.place(dsp, coll, pcb_grp)
    parts["dsp"] = dsp
    mb = MB()
    mb.box((dx - 6.5, dx + 6.5), (dy - 6.5, dy + 6.5), (zpt + 1.4, zpt + 2.9), bev=0.2)
    pad = mb.build("qsfp_dd_module_thermal_pad", M["thermal_pad"], smooth_deg=35)
    I.place(pad, coll, pcb_grp)
    parts["thermal_pad"] = pad
    mb = MB()
    mb.box((dx - 7.4, dx + 7.4), (dy - 7.4, dy + 7.4), (zpt + 2.9, 7.9), bev=0.2, seg=1)
    boss = mb.build("qsfp_dd_module_thermal_boss", M["copper"], smooth_deg=35)
    I.place(boss, coll, top_grp)
    t = I.text_obj("qsfp_dd_module_dsp_text", "DSP-400G", 1.3, (dx, dy, zpt + 1.4 + 0.02), mat=M["gold"], extrude_mm=0.02)
    C.add(t, coll, pcb_grp)

    # TOSA and ROSA (FR4/LR4 style metal can blocks) with fibre jumpers to the LC ferrules
    mb = MB()
    mb.box((-9.0, -1.0), (-42.0, -29.5), (zpt, zpt + 3.8), bev=0.35)
    tosa = mb.build("qsfp_dd_module_tosa", M["nickel_brushed"], smooth_deg=35)
    mb = MB()
    mb.box((1.0, 9.0), (-42.0, -29.5), (zpt, zpt + 3.8), bev=0.35)
    rosa = mb.build("qsfp_dd_module_rosa", M["nickel_brushed"], smooth_deg=35)
    for o in (tosa, rosa):
        I.place(o, coll, pcb_grp)
    parts["tosa"], parts["rosa"] = tosa, rosa
    # flex tails to PCB and window glass
    mb = MB()
    for s in (-1, 1):
        mb.cyl((s * 5.0, -42.3, zpt + 2.0), 1.4, 0.8, axis="Y", segs=16)
    I.place(mb.build("qsfp_dd_module_tosa_rosa_nozzles", M["gold"], smooth_deg=30), coll, pcb_grp)
    mb = MB()
    mb.box((-8.8, -1.2), (-29.6, -25.5), (zpt, zpt + 0.5), bev=0.05, seg=1)
    mb.box((1.2, 8.8), (-29.6, -25.5), (zpt, zpt + 0.5), bev=0.05, seg=1)
    flex = mb.build("qsfp_dd_module_flex_tails", M["plastic_orange"], smooth_deg=0)
    I.place(flex, coll, pcb_grp)
    # fibre jumpers (0.9 mm yellow) from nozzle front to LC ferrule rears
    paths = []
    for s, xs in ((-1, -5.0), (1, 5.0)):
        xe = s * 6.25 / 2
        ys, ye = -43.0, Y_FS1 + 4.0 + 6.6
        pts = np.array([[xs, ys, zpt + 2.0], [xs, ys - 1.5, zpt + 2.4], [(xs + xe) / 2, (ys + ye) / 2 - 1.0, 5.4], [xe, ye + 2.0, lc_z + 0.2], [xe, ye, lc_z]]) * 0.001
        paths.append(I.catmull(pts, per_seg=8))
    fib = I.tube_mesh("qsfp_dd_module_fiber_jumpers", np.array(paths), 0.00045, sides=8, mat=M["jacket_yellow"], caps=False)
    I.place(fib, coll, pcb_grp)
    parts["fibers"] = fib

    placed = []
    excl = [(dx - ds / 2 - 0.5, dx + ds / 2 + 0.5, dy - ds / 2 - 0.5, dy + ds / 2 + 0.5), (-9.5, 9.5, -43.5, -25.0), (-9.5, 9.5, 3.0, Y_R)]
    zone = [(-7.8, 7.8, -25.0, 3.0)]
    caps = scatter(rng, zone, excl, [(1.0, 0.5, 45), (0.6, 0.3, 20)], placed)
    ress = scatter(rng, zone, excl, [(1.0, 0.5, 20)], placed)
    inds = scatter(rng, zone, excl, [(2.5, 2.0, 3)], placed)
    ics = scatter(rng, zone, excl, [(3.0, 3.0, 3)], placed)
    for nm, lst, mt, h in (("mlcc", caps, "ceramic_tan", 0.45), ("resistors", ress, "mold_black", 0.3), ("inductors", inds, "inductor", 1.2), ("pmic", ics, "mold_black", 0.8)):
        mb = MB()
        for cx, cy, w, d in lst:
            mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + h), bev=0.05, seg=1)
        I.place(mb.build("qsfp_dd_module_" + nm, M[mt], smooth_deg=40), coll, pcb_grp)

    # pull tab: elastomeric loop handle (informative length 118 mm total in the spec appendix)
    tab_z = (-1.6, -0.9)
    out = I.rounded_rect(15.0, 56.0, 5.0, n=6)
    inn = I.rounded_rect(10.6, 48.5, 3.5, n=6)
    out = [(x, y - 98.0) for x, y in out]
    inn = [(x, y - 98.0) for x, y in inn]
    mb = MB()
    mb.ring_xy(out, inn, tab_z[0], tab_z[1])
    I.place(mb.build("qsfp_dd_module_pull_tab", M[tab_mat], smooth_deg=35), coll, tab_grp)
    mb = MB()
    for s in (-1, 1):
        mb.box((s * 6.6 - 0.8, s * 6.6 + 0.8), (Y_FS1 - 1.0, -30.0), (-1.3, -0.0), bev=0.1, seg=1)
    mb.box((-6.6, 6.6), (Y_FS1 - 1.0, Y_FS1 + 1.0), (-1.3, 0.0), bev=0.1, seg=1)
    I.place(mb.build("qsfp_dd_module_latch_rails", M["plastic_black"], smooth_deg=35), coll, tab_grp)
    # label
    label_and_text(M, coll, root, ["GENERIC 400G FR4", "QSFP-DD  SN 0000000"], hw, -25.0, 4.0, 1.3, (22.0, 4.2), name="qsfp_dd_module_label")

    h = C.hook("plug_axis", coll, root, loc=_v(0, 0, 4.25))   # local +Y = insertion direction
    h.empty_display_size = 0.012
    ho = C.hook("open", coll, top_grp, loc=_v(0, -25.0, 8.5))
    ho.empty_display_size = 0.01
    C.hook("receptacle", coll, root, loc=_v(0, Y_FS1, lc_z))
    C.hook("pull_tab_end", coll, tab_grp, loc=_v(0, -130.0, -1.2))
    C.hook("card_edge", coll, root, loc=_v(0, Y_R, 2.75))
    return dict(top_grp=top_grp, pcb_grp=pcb_grp, tab_grp=tab_grp, parts=parts)
