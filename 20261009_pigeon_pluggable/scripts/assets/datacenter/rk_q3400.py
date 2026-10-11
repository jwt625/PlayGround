"""Quantum-X800 Q3400-RA style switch chassis generator (true scale or bird scale), workstream R (DevLog-007).

Front grammar from the NVIDIA Q3400-RA front figure (research/racks_refs.md section 3): black square-hole grille; cage pairs
(upper + lower) with a champagne port-label strip between them, grouped in column groups; left panel = port LED matrix,
FNM (UFM) OSFP port, badge; right panel = LED matrix, status LEDs, USB + 2 x RJ-45. Rear: PSUs stacked at both sides,
fans in the middle with champagne pull handles.

Chassis-local frame (mm): faceplate front at x = 0 (cage lips protrude to +x), body toward -x, y centred (+y = right seen
from the front), z = 0 at the chassis bottom. Returns rk_lib Parts plus cell records.
"""
import math

import rk_lib as K
from rk_lib import MB, Part

m = lambda x: x * 0.001

# true-scale Q3400-RA (B: measured on the NVIDIA front figure at 0.338 mm/px; envelope A: 438 x 177.8 x 850 mm).
# Cage opening 22.8 x 13.3 mm with 0.3 mm lips (pitch 23.4 mm) so an OSFP module (22.58 x 13.00 mm, OSFP MSA Fig 3-2,
# research/hero_800g_osfp_dr8.md) fits; panels 62 / 64 mm as measured -> group gap 15.6 mm (measured about 16 +-1 mm).
REAL = dict(
    name="real", W=438.0, H=177.8, D=850.0, groups=3, cols=4, blocks=3,
    cell_w=22.8, cell_h=13.3, lip=0.3, strip_h=3.0, panel_l=62.0, panel_r=64.0,
    cell_depth=60.0, perf=3.0, label_size=1.7, fingers=0, host_pads=False, ui=1.0,
    psu_rows=4, fan_rows=2, fans=5, ears=True, wall="cage_wall", back="port_hole",
)
# bird-scale chassis: one block row of 2 x 3 portrait cells for a 2.6x cyber-pigeon (0.32 m beak to tail, about
# 0.25 m tall, about 0.13 m wide). Stylized: anisotropic scale vs REAL, documented in DevLog-007 section 3.
BIRD = dict(
    name="bird", W=700.0, H=640.0, D=850.0, groups=3, cols=1, blocks=1,
    cell_w=180.0, cell_h=270.0, lip=6.0, strip_h=34.0, panel_l=46.0, panel_r=46.0,
    cell_depth=360.0, perf=9.0, label_size=16.0, fingers=11, host_pads=True, ui=1.0,
    psu_rows=2, fan_rows=1, fans=5, ears=False, wall="cage_wall_lt", back="dark_steel",
)


def layout(S):
    """Cell rectangles (opening) in chassis-local mm: list of dict(block,row,col_global,group,y0,y1,z0,z1)."""
    ow, oh = S["cell_w"] + 2 * S["lip"], S["cell_h"] + 2 * S["lip"]
    gw = S["cols"] * ow
    avail = S["W"] - S["panel_l"] - S["panel_r"]
    gap = (avail - S["groups"] * gw) / (S["groups"] - 1) if S["groups"] > 1 else 0.0
    bh = 2 * oh + S["strip_h"]
    vgap = 0.0
    if S["blocks"] > 1:
        # real: block pitch 57.5 mm measured; margins from the remainder
        vgap = 57.5 - bh
    tot = S["blocks"] * bh + (S["blocks"] - 1) * vgap
    marg = (S["H"] - tot) / 2
    y_start = -S["W"] / 2 + S["panel_l"]
    cells, groups_y, blocks_z = [], [], []
    for g in range(S["groups"]):
        gy0 = y_start + g * (gw + gap)
        groups_y.append((gy0, gy0 + gw))
    for b in range(S["blocks"]):
        bz1 = S["H"] - marg - b * (bh + vgap)       # block top (block 0 at the top)
        bz0 = bz1 - bh
        blocks_z.append((bz0, bz1))
        for r in range(2):                          # row 0 = upper cage, row 1 = lower cage
            oz1 = bz1 - r * (oh + S["strip_h"])
            oz0 = oz1 - oh
            for g in range(S["groups"]):
                for c in range(S["cols"]):
                    oy0 = groups_y[g][0] + c * ow
                    cells.append(dict(block=b, row=r, group=g, col=g * S["cols"] + c,
                                      y0=oy0 + S["lip"], y1=oy0 + ow - S["lip"], z0=oz0 + S["lip"], z1=oz1 - S["lip"],
                                      oy0=oy0, oy1=oy0 + ow, oz0=oz0, oz1=oz1))
    return dict(cells=cells, groups_y=groups_y, blocks_z=blocks_z, ow=ow, oh=oh, gap=gap, marg=marg, vgap=vgap, bh=bh)


def fbox(mb, y0, y1, z0, z1, x0, x1, mi=0):
    mb.box2((m(x0), m(y0), m(z0)), (m(x1), m(y1), m(z1)), mi)


def osfp_cage(mb, y, z, mi_cage, mi_hole, w=23.9, h=14.3, out=1.5):    # opening 22.7 x 13.1 mm > OSFP 22.58 x 13.0
    K.port(mb, y, z, w, h, 0.6, 9.0, out, mi_cage, mi_hole)


def rj45(mb, y, z, mi_shell, mi_hole):
    K.port(mb, y, z, 16.0, 13.6, 1.6, 6.0, 1.2, mi_shell, mi_hole)


def usb(mb, y, z, mi_shell, mi_hole):
    K.port(mb, y, z, 13.0, 6.0, 0.8, 5.0, 1.0, mi_shell, mi_hole)


def chassis(S, L, port_offset=0):
    """Build parts for one chassis. port_offset = first cage index (cells are numbered 1.. as on the real label strip,
    one cage = 2 logical ports on the real switch; on the bird chassis one cell = one bird)."""
    lay = layout(S)
    W, H, D = S["W"], S["H"], S["D"]
    cells = lay["cells"]
    parts = []

    # ---- faceplate grille (complement of cage outlines), perforated black
    gr = MB()
    gy = lay["groups_y"]
    fbox(gr, -W / 2, gy[0][0], 0, H, -2.5, 0, 0)                                     # left panel
    fbox(gr, gy[-1][1], W / 2, 0, H, -2.5, 0, 0)                                     # right panel
    for g in range(len(gy) - 1):
        fbox(gr, gy[g][1], gy[g + 1][0], 0, H, -2.5, 0, 0)                           # group gaps
    bz = lay["blocks_z"]
    for (y0, y1) in gy:
        fbox(gr, y0, y1, bz[0][1], H, -2.5, 0, 0)                                    # top margin
        fbox(gr, y0, y1, 0, bz[-1][0], -2.5, 0, 0)                                   # bottom margin
        for b in range(len(bz) - 1):
            fbox(gr, y0, y1, bz[b + 1][1], bz[b][0], -2.5, 0, 0)                     # between blocks
    parts.append(Part("grille", gr, [L["perf_black"]], uv_tile=m(S["perf"])))

    # ---- champagne label strips + numbers (cage numbering like the real front: upper odd, lower even)
    st = MB()
    tri = MB()
    labels = []
    oh = lay["oh"]
    for b, (bz0, bz1) in enumerate(bz):
        zs0 = bz1 - oh - S["strip_h"]
        for g, (y0, y1) in enumerate(gy):
            fbox(st, y0, y1, zs0, zs0 + S["strip_h"], -2.5, 0.4, 0)
    pair = 0
    for c in sorted([c for c in cells if c["row"] == 0], key=lambda c: (c["col"], c["block"])):
        pass
    for c in cells:
        if c["row"] != 0:
            continue
        bz0, bz1 = bz[c["block"]]
        zs = bz1 - oh - S["strip_h"] / 2
        # real numbering: column-major per block row (1/2 top-left, then 3/4 in the next block row ...)
        idx = port_offset + c["col"] * S["blocks"] + c["block"]
        n_up, n_lo = 2 * idx + 1, 2 * idx + 2
        yc = (c["y0"] + c["y1"]) / 2
        s = S["label_size"]
        labels.append((n_up, n_lo, yc, zs))
        a = s * 0.32
        tri.tri((m(0.6), m(yc - a), m(zs + 0.1 * s)), (m(0.6), m(yc + a), m(zs + 0.1 * s)), (m(0.6), m(yc), m(zs + 0.45 * s)), 0)
        tri.tri((m(0.6), m(yc - a), m(zs - 0.1 * s)), (m(0.6), m(yc), m(zs - 0.45 * s)), (m(0.6), m(yc + a), m(zs - 0.1 * s)), 0)
    parts.append(Part("label_strips", st, [L["champagne"]]))
    parts.append(Part("label_marks", tri, [L["champ_txt"]]))

    # ---- cages: lips, interior shell, EMI fingers, host pads (bird), back wall
    lip = MB()
    inner = MB()
    fing = MB()
    pads = MB()
    back = MB()
    cd = S["cell_depth"]
    t = max(0.6, S["lip"] * 0.35)
    for c in cells:
        y0, y1, z0, z1 = c["y0"], c["y1"], c["z0"], c["z1"]
        L_ = S["lip"]
        fbox(lip, y0 - L_, y1 + L_, z1, z1 + L_, -4, 2.0, 0)
        fbox(lip, y0 - L_, y1 + L_, z0 - L_, z0, -4, 2.0, 0)
        fbox(lip, y0 - L_, y0, z0, z1, -4, 2.0, 0)
        fbox(lip, y1, y1 + L_, z0, z1, -4, 2.0, 0)
        # interior: side walls + ceiling (zinc), floor (pcb on the bird, zinc on the real)
        fbox(inner, y0 - t, y0, z0, z1, -cd, -2.5, 0)
        fbox(inner, y1, y1 + t, z0, z1, -cd, -2.5, 0)
        fbox(inner, y0, y1, z1, z1 + t, -cd, -2.5, 0)
        fbox(inner, y0, y1, z0 - t, z0, -cd, -2.5, 1 if S["host_pads"] else 0)
        # back wall + host connector
        fbox(back, y0, y1, z0, z1, -cd - t, -cd, 0)
        hc_h = min(0.28 * (z1 - z0), 60)
        fbox(back, y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), z0, z0 + hc_h, -cd, -cd + 0.12 * cd, 1)
        nst = 8 if S["host_pads"] else 3
        for k in range(nst):
            yy = y0 + 0.24 * (y1 - y0) + k * (0.52 * (y1 - y0)) / max(1, nst - 1)
            fbox(back, yy - 0.012 * (y1 - y0), yy + 0.012 * (y1 - y0), z0 + 0.25 * hc_h, z0 + 0.8 * hc_h,
                 -cd + 0.12 * cd, -cd + 0.12 * cd + 0.6, 2)
        if S["host_pads"]:
            fbox(back, (y0 + y1) / 2 - 6, (y0 + y1) / 2 + 6, z1 - 22, z1 - 14, -cd, -cd + 1.5, 3)   # in-cell status LED
        if S["fingers"]:
            n = S["fingers"]
            pitch = (y1 - y0) / n
            for k in range(n):
                yy = y0 + (k + 0.5) * pitch
                for zz, sgn in ((z1, -1), (z0, 1)):
                    fbox(fing, yy - pitch * 0.33, yy + pitch * 0.33, zz + (sgn * 0.0 if sgn > 0 else -2.2), zz + (2.2 if sgn > 0 else 0),
                         -26, -6, 0)
        if S["host_pads"]:
            # gold host-contact field under the bird's feet (feet origin 60 mm behind the face; feet 18 mm ahead of it)
            yc = (y0 + y1) / 2
            for k in range(14):
                yy = yc - 42 + k * 6.46
                fbox(pads, yy - 2.0, yy + 2.0, z0, z0 + 0.6, -110, -20, 0)
            fbox(pads, yc - 48, yc + 48, z0, z0 + 0.3, -116, -14, 1)
    parts.append(Part("cage_lips", lip, [L["nickel_dk"]]))
    parts.append(Part("cage_inner", inner, [L[S["wall"]], L["pcb"]]))
    parts.append(Part("cage_back", back, [L[S["back"]], L["black"], L["gold"], L["led_g"]]))
    if fing.f:
        parts.append(Part("emi_fingers", fing, [L["nickel"]]))
    if pads.f:
        parts.append(Part("host_pads", pads, [L["gold"], L["pcb"]]))

    # ---- left / right panels
    pl = MB()     # mats: 0 black, 1 nickel, 2 port_hole, 3 led_g, 4 champagne, 5 rj45, 6 led_a, 7 badge dark
    yl0, yl1 = -W / 2, gy[0][0]
    yr0, yr1 = gy[-1][1], W / 2
    ylc, yrc = (yl0 + yl1) / 2, (yr0 + yr1) / 2
    nled = len(cells)
    # LED matrices. Real: per the NVIDIA figure, 6 rows in the upper part of each side panel, each row = a group of 4 bars
    # and a group of 2 bars (one bar per cage). Bird: one LED bar per cell, beside its cell row (left panel cols 0-1,
    # right panel col 2 + a spare), so the LEDs read as belonging to the block row.
    if S["name"] == "real":
        for (a, b) in ((yl0, yl1), (yr0, yr1)):
            for i in range(6):
                zz = H - 9.0 - i * 9.6
                for j, yy in enumerate([a + 13 + k * 7.0 for k in range(4)] + [a + 46 + k * 7.0 for k in range(2)]):
                    fbox(pl, yy - 2.6, yy + 2.6, zz - 0.5, zz + 0.5, -0.5, 0.8, 4)
                    fbox(pl, yy - 0.6, yy + 0.6, zz + 1.4, zz + 2.6, -0.5, 0.7, 4)        # tiny number tick
    else:
        for c in cells:
            zc = (c["z0"] + c["z1"]) / 2 + 40
            side = (yl0, yl1) if c["col"] < 2 else (yr0, yr1)
            k = c["col"] if c["col"] < 2 else 0
            yy = side[0] + 12 + k * 14
            fbox(pl, yy - 5, yy + 5, zc - 1.5, zc + 1.5, -0.5, 0.9, 4)
            fbox(pl, yy - 2, yy + 2, zc + 4, zc + 8, -0.5, 1.0, 3)                       # port LED
    # FNM (UFM) OSFP port, true scale, left panel at about 60 percent height (real) / mid (bird)
    fz = H * (0.43 if S["name"] == "real" else 0.55)
    osfp_cage(pl, ylc, fz, 1, 2)
    fbox(pl, ylc - 10, ylc + 10, fz + 10, fz + 12, -0.5, 0.6, 4)                    # "FNM" label bar
    # badge plate (no logo; model text added by the caller as a mesh)
    bw, bh = (min(40.0, (yl1 - yl0) - 6), 24.0) if S["name"] != "real" else (40.0, 26.0)
    bz_ = H * 0.16
    fbox(pl, ylc - bw / 2, ylc + bw / 2, bz_ - bh / 2, bz_ + bh / 2, -0.5, 1.0, 7)
    # right panel: status LEDs, USB, 2 x RJ-45 stacked
    for k in range(5):
        yy = yrc - 10 + k * 5
        fbox(pl, yy - 1.0, yy + 1.0, H * 0.62, H * 0.62 + 2.0, -0.5, 0.9, 3 if k != 2 else 6)
    uz_ = H * (0.44 if S["name"] == "real" else 0.46)
    usb(pl, yrc - 9.5, uz_ + 4.0, 5, 2)                                                 # USB 2.0
    usb(pl, yrc - 9.5, uz_ - 4.0, 5, 2)                                                 # console (USB-style recess)
    rj45(pl, yrc + 9, uz_ + 7.5, 5, 2)
    rj45(pl, yrc + 9, uz_ - 7.5, 5, 2)
    if S["ears"]:
        for s in (-1, 1):
            y0, y1 = sorted((s * W / 2, s * 241.3))
            fbox(pl, y0, y1, 4, H - 4, -3, 0, 0)
    parts.append(Part("panels", pl, [L["black"], L["nickel"], L["port_hole"], L["led_g"], L["champagne"], L["rj45"],
                                     L["led_a"], L["badge"]]))

    # ---- body
    bd = MB()
    xb = -(S["cell_depth"] + 4.0)                                                     # body starts behind the cages
    fbox(bd, -W / 2 + 1, W / 2 - 1, 1, H - 1, -D, xb, 0)
    # thin sheet behind the faceplate between the cages (top/bottom/side skins of the front section)
    fbox(bd, -W / 2 + 1, W / 2 - 1, H - 2, H - 1, xb, -2.5, 0)
    fbox(bd, -W / 2 + 1, W / 2 - 1, 1, 2, xb, -2.5, 0)
    for s_ in (-1, 1):
        fbox(bd, *sorted((s_ * (W / 2 - 1), s_ * (W / 2 - 2))), 1, H - 1, xb, -2.5, 0)
    parts.append(Part("body", bd, [L["dark_steel"]]))

    # ---- rear: PSUs stacked at both sides, fans in the middle with champagne handles
    rr = MB()     # 0 black, 1 brick grille, 2 champagne, 3 rubber, 4 led_r, 5 dark_steel
    xr = -D
    psu_w = W * 0.12
    ph = (H - 8) / S["psu_rows"]
    for s in (-1, 1):
        y0, y1 = sorted((s * (W / 2 - 2), s * (W / 2 - 2 - psu_w)))
        for i in range(S["psu_rows"]):
            z0 = 4 + i * ph
            fbox(rr, y0 + 0.5, y1 - 0.5, z0 + 0.6, z0 + ph - 0.6, xr - 2, xr, 0)
            yi = (y0 + y1) / 2 - s * psu_w * 0.12
            iw, ih = psu_w * 0.42, ph * 0.5                                                                       # IEC inlet
            fbox(rr, yi - iw / 2, yi + iw / 2, z0 + ph / 2 - ih / 2, z0 + ph / 2 + ih / 2, xr - 4, xr - 2, 3)     # rubber surround
            fbox(rr, yi - iw * 0.36, yi + iw * 0.36, z0 + ph / 2 - ih * 0.34, z0 + ph / 2 + ih * 0.34, xr - 2.5, xr - 1.2, 6)  # recess
            for dy, dz in ((-0.18, 0.08), (0.18, 0.08), (0.0, -0.14)):                                          # 3 pins
                fbox(rr, yi + dy * iw - iw * 0.04, yi + dy * iw + iw * 0.04, z0 + ph / 2 + dz * ih - ih * 0.07,
                     z0 + ph / 2 + dz * ih + ih * 0.07, xr - 3.8, xr - 2.4, 5)
            fbox(rr, y1 - s * 6 - 2, y1 - s * 6 + 2, z0 + ph * 0.25, z0 + ph * 0.35, xr - 2.6, xr - 1.8, 4)       # LED
            fbox(rr, y0 + s * 0, y0 + s * 6, z0 + ph * 0.2, z0 + ph * 0.8, xr - 10, xr - 2, 0)                     # handle
            fbox(rr, y0 + s * 1, y0 + s * 5, z0 + ph * 0.62, z0 + ph * 0.74, xr - 11, xr - 9.5, 4)                 # red latch tab
    fy0, fy1 = -W / 2 + 2 + psu_w + 3, W / 2 - 2 - psu_w - 3
    fw = (fy1 - fy0) / S["fans"]
    fh = (H - 8) / S["fan_rows"]
    for r in range(S["fan_rows"]):
        z0 = 4 + r * fh
        for k in range(S["fans"]):
            y0 = fy0 + k * fw
            fbox(rr, y0 + 1, y0 + fw - 1, z0 + 1, z0 + fh - 1, xr - 1.5, xr, 1)
            fbox(rr, y0 + fw * 0.1, y0 + fw * 0.9, z0 + fh * 0.62, z0 + fh * 0.62 + fh * 0.13, xr - fh * 0.1, xr - 1.5, 2)  # handle
    fbox(rr, -W / 2, W / 2, H - 4, H, xr - 2, xr, 5)
    fbox(rr, -W / 2, W / 2, 0, 4, xr - 2, xr, 5)
    parts.append(Part("rear", rr, [L["black"], L["brick"], L["champagne_matte"], L["rubber"], L["led_r"], L["dark_steel"],
                                   L["port_hole"]],
                      uv_tile=m(S["perf"] * 3.5)))
    return parts, lay, labels


def label_texts(S, L, labels, coll, parent, prefix, badge_text):
    """Port numbers on the label strips and the badge text (meshes)."""
    s = S["label_size"]
    out = []
    lay_y_left = -S["W"] / 2 + S["panel_l"] / 2
    for n_up, n_lo, yc, zs in labels:
        dy = s * 0.85 + (0.25 * s if n_lo >= 10 else 0)
        out.append(K.text("%s_lbl_%d" % (prefix, n_up), str(n_up), m(s * 0.78), (m(0.8), m(yc - dy), m(zs)),
                          (math.pi / 2, 0, math.pi / 2), L["champ_txt"], coll, parent))
        out.append(K.text("%s_lbl_%d" % (prefix, n_lo), str(n_lo), m(s * 0.78), (m(0.8), m(yc + dy), m(zs)),
                          (math.pi / 2, 0, math.pi / 2), L["champ_txt"], coll, parent))
    H = S["H"]
    bsize = 3.6 if S["name"] == "real" else 5.2
    out.append(K.text(prefix + "_badge", badge_text, m(bsize), (m(1.2), m(lay_y_left), m(H * 0.16)),
                      (math.pi / 2, 0, math.pi / 2), L["champagne"], coll, parent))
    return out
