"""Cable-end plugs (OSFP / QSFP-DD) for DAC, AEC and AOC assemblies, strain-relief boots, U-bend cable runs and cross-sections.

Plug frame as the modules: origin at the forward-stop plane (bottom centre), +Y insertion direction, -Y toward the cable boot.
"""
import math
import random

import numpy as np

import ic_common as I
import ic_modules as MOD
from ic_common import MB, _v, mm

C = I.C

DIMS = {
    "osfp": dict(W=22.58, H=13.0, Yf=-69.8, Yr=10.6, split=7.0, zpb=4.4, zpt=5.4, pcb_w=19.0, pad_n=30, pad_pitch=0.6, pad_w=0.38, boot_w=17.0, boot_h=10.5),
    "qsfpdd": dict(W=18.35, H=8.5, Yf=-68.2, Yr=10.06, split=4.2, zpb=2.25, zpt=3.25, pcb_w=16.6, pad_n=19, pad_pitch=0.8, pad_w=0.5, boot_w=14.0, boot_h=9.5),
}
BOOT_LEN = 38.0


def build_plug(kind, mode, coll, parent, M, name, openable, cable_od, boot_mat="rubber_boot", shell_mat=None, tab_mat=None, seed=3):
    D = DIMS[kind]
    W, H, Yf, Yr, split = D["W"], D["H"], D["Yf"], D["Yr"], D["split"]
    zpb, zpt = D["zpb"], D["zpt"]
    hw = W / 2
    rng = random.Random(seed)
    sm = shell_mat or ("black_alu" if kind == "osfp" else "nickel")
    top_grp = I.empty(name + "_top_group", coll, parent)
    pcb_grp = I.empty(name + "_pcb_group", coll, parent)
    parts = {}
    # shell
    mb = MB()
    mb.box((-hw, hw), (Yf, 0), (0, 0.6), bev=0.15)
    for s in (-1, 1):
        x0, x1 = (hw - 0.6, hw) if s > 0 else (-hw, -hw + 0.6)
        mb.box((x0, x1), (Yf, 0), (0.6, split), bev=0.12)
    mb.box((-hw, hw), (Yf, Yf + 1.2), (0.6, H - 0.6 if kind == "osfp" else H), bev=0.12)           # front wall (boot attaches)
    for s in (-1, 1):
        mb.box((s * (hw - 1.4) - 0.45, s * (hw - 1.4) + 0.45), (0, Yr), (0.6, zpb - 0.3 if kind == "osfp" else 1.4), bev=0.1)   # guard posts
    parts["shell_bottom"] = mb.build(name + "_shell_bottom", M[sm], smooth_deg=35)
    I.place(parts["shell_bottom"], coll, parent)
    mb = MB()
    for s in (-1, 1):
        x0, x1 = (hw - 0.6, hw) if s > 0 else (-hw, -hw + 0.6)
        mb.box((x0, x1), (Yf + 1.2, 0), (split + 0.02, H - 0.8), bev=0.12)
    mb.box((-hw, hw), (Yf + 1.2, 0), (H - 0.8, H), bev=0.15)
    if kind == "osfp":
        mb.prism_yz([(-3.0, H), (0.0, H), (0.0, H - 1.6)], -hw, hw) if False else None
    # tongue over the card edge
    tz = (split + 0.4, split + 1.0) if kind == "osfp" else (zpt + 3.0, zpt + 3.6)
    mb.box((-hw + 1.0, hw - 1.0), (0, Yr - 1.0), tz, bev=0.1, seg=1)
    parts["shell_top"] = mb.build(name + "_shell_top", M[sm], smooth_deg=35)
    I.place(parts["shell_top"], coll, top_grp)
    if kind == "qsfpdd":
        # front block (19 wide, z -1.6 .. 11.9) around the boot
        mb = MB()
        mb.box((-9.5, 9.5), (Yf, Yf + 20.0), (-1.6, 11.9), bev=0.4)
        parts["front_block"] = mb.build(name + "_front_block", M[sm], smooth_deg=35)
        I.place(parts["front_block"], coll, parent)
    # boot (strain relief): loft from the plug front to the cable
    yb0 = Yf - (0 if kind == "osfp" else 0)
    zc = 6.5 if kind == "osfp" else 4.9
    mb = MB()
    mb.loft_y([(Yf + 1.0, D["boot_w"], D["boot_h"], 4.0), (Yf - 6.0, D["boot_w"] * 0.8, D["boot_h"] * 0.85, 3.0), (Yf - 18.0, cable_od * 2.2, cable_od * 2.0, 2.4),
               (Yf - BOOT_LEN, cable_od * 1.3, cable_od * 1.3, 2.0)], npts=28, cz=zc)
    parts["boot"] = mb.build(name + "_boot", M[boot_mat], smooth_deg=35)
    I.place(parts["boot"], coll, parent)
    # PCB with gold fingers (always present: card edge is the plug interface)
    mb = MB()
    mb.box((-D["pcb_w"] / 2, D["pcb_w"] / 2), (Yf + 16.0, Yr), (zpb, zpt), bev=0.06, seg=1)
    pcb = mb.build(name + "_pcb", M["fr4_green"], smooth_deg=30)
    I.place(pcb, coll, pcb_grp)
    parts["pcb"] = pcb
    if kind == "osfp":
        gf = MOD.card_edge_pads(M, Yr, 0.6, 0.38, 30, 3.2, 2.6, zpt, zpb, name=name + "_gold_fingers")
    else:
        gf = MOD.card_edge_pads_2row(M, Yr, 0.8, 0.5, 19, 1.9, 1.5, 0.8, zpt, zpb, name=name + "_gold_fingers")
    I.place(gf, coll, pcb_grp)
    parts["gold_fingers"] = gf
    # internals
    ys = Yf + 16.0
    if mode in ("dac", "aec"):
        # 8 twinax pairs fan out from the boot to solder pads: 4 pairs on top, 4 on bottom
        pairs = []
        pad_y = -22.0
        mb = MB()
        for side, zs in ((1, zpt), (-1, zpb)):
            for i in range(4):
                xc = (i - 1.5) * (3.4 if kind == "osfp" else 3.0)
                for dx in (-0.45, 0.45):
                    mb.box((xc + dx - 0.3, xc + dx + 0.3), (pad_y - 1.5, pad_y + 1.5), (zs, zs + 0.05) if side > 0 else (zs - 0.05, zs), bev=0.0)
        I.place(mb.build(name + "_solder_pads", M["gold"], smooth_deg=0), coll, pcb_grp)
        sheath, cond = [], []
        for side, zs in ((1, zpt), (-1, zpb)):
            for i in range(4):
                xc = (i - 1.5) * (3.4 if kind == "osfp" else 3.0)
                zq = zs + side * 0.45
                p = np.array([[0, Yf - 2, zc + side * 0.8 * (i % 2)], [0, Yf + 6, zc + side * 0.5], [xc * 0.5, Yf + 15, zq + side * 0.2], [xc, pad_y - 7, zq], [xc, pad_y - 3.0, zq]]) * 0.001
                sheath.append(I.catmull(p, 9))
                for dx in (-0.45, 0.45):
                    cond.append(I.catmull(np.array([[xc + dx, pad_y - 4.0, zq], [xc + dx, pad_y - 1.0, zs + side * 0.25]]) * 0.001, 3))
        sh = I.tube_mesh(name + "_twinax_pairs", np.array(sheath), 0.0006, sides=8, mat=M["dielectric"])
        co = I.tube_mesh(name + "_conductors", np.array(cond), 0.00016, sides=6, mat=M["copper"])
        I.place(sh, coll, pcb_grp)
        I.place(co, coll, pcb_grp)
        parts["twinax"] = sh
        if mode == "aec":
            mb = MB()
            mb.box((-5.5, 5.5), (-12.0, -1.0), (zpt, zpt + 1.0), bev=0.1, seg=1)
            mb.box((-5.5, 5.5), (-12.0, -1.0), (zpb - 1.0, zpb), bev=0.1, seg=1)
            ret = mb.build(name + "_retimer_chips", M["mold_black"], smooth_deg=35)
            I.place(ret, coll, pcb_grp)
            parts["retimer"] = ret
            t = I.text_obj(name + "_retimer_text", "RETIMER", 1.6, (0, -6.5, zpt + 1.02), mat=M["gold"], extrude_mm=0.02)
            C.add(t, coll, pcb_grp)
            placed = []
            zone = [(-7.0, 7.0, -21.0, -13.0), (-7.0, 7.0, -0.5, 6.0)]
            for nm, lst, mt, h in (("mlcc", MOD.scatter(rng, zone, [(-5.7, 5.7, -12.2, -0.8)], [(1.0, 0.5, 30)], placed), "ceramic_tan", 0.45),
                                   ("inductors", MOD.scatter(rng, zone, [(-5.7, 5.7, -12.2, -0.8)], [(2.5, 2.0, 2)], placed), "inductor", 1.2)):
                mb = MB()
                for cx, cy, w, d in lst:
                    mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + h), bev=0.05, seg=1)
                I.place(mb.build(name + "_" + nm, M[mt], smooth_deg=40), coll, pcb_grp)
    else:   # aoc: driver/CDR, VCSEL/PD array, lens block, 8 fibres
        mb = MB()
        mb.box((-4.0, 4.0), (-14.0, -6.0), (zpt, zpt + 0.9), bev=0.1, seg=1)
        I.place(mb.build(name + "_cdr_driver", M["mold_black"], smooth_deg=35), coll, pcb_grp)
        mb = MB()
        mb.box((-3.2, 3.2), (-24.0, -19.0), (zpt, zpt + 0.3), bev=0.05, seg=1)
        arr = mb.build(name + "_vcsel_pd_array", M["silicon"], smooth_deg=30)
        I.place(arr, coll, pcb_grp)
        mb = MB()
        for i in range(8):
            mb.box((-2.8 + i * 0.8, -2.6 + i * 0.8), (-22.0, -21.8), (zpt + 0.3, zpt + 0.34))
        I.place(mb.build(name + "_vcsel_apertures", M["vcsel"], smooth_deg=0), coll, pcb_grp)
        mb = MB()
        mb.box((-3.6, 3.6), (-30.0, -18.0), (zpt + 0.0, zpt + 2.4), bev=0.12, seg=1)
        I.place(mb.build(name + "_lens_block", M["glass"], smooth_deg=35), coll, pcb_grp)
        mb = MB()
        mb.box((-3.7, 3.7), (-33.0, -29.8), (zpt, zpt + 1.4), bev=0.3, seg=1)
        I.place(mb.build(name + "_fibre_array_unit", M["potting"], smooth_deg=35), coll, pcb_grp)
        paths = []
        for i in range(8):
            xe = (i - 3.5) * 0.9
            p = np.array([[xe * 0.2, Yf - 2, zc], [xe * 0.4, Yf + 8, zc], [xe, -33.0 - 8.0, zpt + 1.0], [xe, -33.0, zpt + 1.0]]) * 0.001
            paths.append(I.catmull(p, 8))
        fb = I.tube_mesh(name + "_fibre_ribbon", np.array(paths), 0.00012, sides=6, mat=M["fiber_coat"])
        I.place(fb, coll, pcb_grp)
        placed = []
        zone = [(-7.0, 7.0, -17.0, -14.5), (-7.0, 7.0, -0.5, 6.0)]
        mb = MB()
        for cx, cy, w, d in MOD.scatter(rng, [(-7.0, 7.0, -5.5, 6.0)], [(-4.2, 4.2, -14.2, -5.8)], [(1.0, 0.5, 25)], placed):
            mb.box((cx - w / 2, cx + w / 2), (cy - d / 2, cy + d / 2), (zpt, zpt + 0.45), bev=0.05, seg=1)
        I.place(mb.build(name + "_mlcc", M["ceramic_tan"], smooth_deg=40), coll, pcb_grp)
    # pull tab loop (omitted: cable plugs use the boot as handle) and latch rails
    mb = MB()
    for s in (-1, 1):
        mb.box((s * (hw - 2.2) - 0.8, s * (hw - 2.2) + 0.8), (Yf + 1.0, Yf + 38.0), (-1.2, 0.0), bev=0.1, seg=1)
    I.place(mb.build(name + "_latch_rails", M["plastic_black"], smooth_deg=35), coll, parent)
    if tab_mat:
        out = I.rounded_rect(15.0, 34.0, 4.0, n=6)
        inn = I.rounded_rect(10.6, 27.0, 2.6, n=6)
        out = [(x, y + Yf - 20.0) for x, y in out]
        inn = [(x, y + Yf - 20.0) for x, y in inn]
        mb = MB()
        mb.ring_xy(out, inn, -1.6, -0.9)
        I.place(mb.build(name + "_pull_tab", M[tab_mat], smooth_deg=35), coll, parent)
    h = C.hook(name + "_plug_axis", coll, parent, loc=_v(0, 0, zc))   # local +Y = insertion
    h.empty_display_size = 0.012
    ho = C.hook(name + "_open", coll, top_grp, loc=_v(0, -25.0, H))
    ho.empty_display_size = 0.01
    return top_grp, pcb_grp, parts


def u_cable(M, coll, parent, name, y0, od, mat, R=120.0, L1=160.0, n=70, sides=14):
    """U-bend cable path in plug-A frame from its boot end (0, y0) to plug B boot end (2R, y0). Returns (object, length_mm)."""
    pts = [(0.0, y0 + 2.0), (0.0, y0 - L1 * 0.5)]
    ph = np.linspace(0, math.pi, 13)
    pts += [(R - R * math.cos(p), y0 - L1 - R * math.sin(p)) for p in ph]
    pts += [(2 * R, y0 - L1 * 0.5), (2 * R, y0 + 2.0)]
    P = np.array([(x, y, 6.0 if False else 0.0) for x, y in pts])
    path = I.catmull(np.column_stack([P[:, 0], P[:, 1], np.full(len(P), 0.0)]) * 0.001, 6)
    return path


def cross_section_twinax(M, coll, parent, loc_mm, name, scale=1.0):
    """True-size (mm) cross-section of an 8-pair 28 AWG twinax cable, 1.0 mm slice along Y, under a group empty scaled by `scale`."""
    g = I.empty(name + "_group", coll, parent, loc_mm=loc_mm)
    g.scale = (scale, scale, scale)
    R_ring, od_cond, od_diel = 2.25, 0.321, 0.62
    jr_out, jr_in = 3.5, 2.95
    mb = MB()
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    mb.ring_xz([(jr_out * math.cos(a), jr_out * math.sin(a)) for a in ang], [(jr_in * math.cos(a), jr_in * math.sin(a)) for a in ang], 0.0, 1.0)
    I.place(mb.build(name + "_jacket", M["jacket_black"], smooth_deg=40), coll, g)
    mb = MB()
    mb.ring_xz([(2.95 * math.cos(a), 2.95 * math.sin(a)) for a in ang], [(2.78 * math.cos(a), 2.78 * math.sin(a)) for a in ang], 0.0, 1.0)
    I.place(mb.build(name + "_overall_foil_braid", M["braid"], smooth_deg=40), coll, g)
    cond, diel, foil, drain = MB(), MB(), MB(), MB()
    for k in range(8):
        a = 2 * math.pi * k / 8
        cx, cz = R_ring * math.cos(a), R_ring * math.sin(a)
        tx, tz = -math.sin(a), math.cos(a)    # tangential direction (pair axis)
        for sgn in (-1, 1):
            px, pz = cx + sgn * 0.31 * tx, cz + sgn * 0.31 * tz
            diel.cyl((px, 0.5, pz), od_diel / 2, 1.0, axis="Y", segs=20)
            cond.cyl((px, 0.5, pz), od_cond / 2, 1.04, axis="Y", segs=14)
        # drain wire outside the pair (radially outward)
        drain.cyl((cx + 0.5 * math.cos(a), 0.5, cz + 0.5 * math.sin(a)), 0.1, 1.04, axis="Y", segs=10)
        # foil: stadium ring around the pair
        n = 28
        stad_o, stad_i = [], []
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            for (rad, lst) in ((0.0, None),):
                pass
            ux, uz = math.cos(t), math.sin(t)
            hl = 0.31 + (0.31 + 0.0) * 0.0
            # superellipse-ish stadium: semi-axes (0.62+0.03, 0.31+0.03) along (tangent, radial)
            ex = 0.93 * np.sign(ux) * abs(ux) ** 0.7
            ez = 0.31 * 1.0 * np.sign(uz) * abs(uz) ** 0.7
            for (e_s, lst, th) in ((1.0, stad_o, 0.0),):
                pass
            sx_o, sz_o = (0.62 + 0.31 + 0.03) * np.sign(ux) * abs(ux) ** 0.6, (0.31 + 0.03) * np.sign(uz) * abs(uz) ** 0.6
            sx_i, sz_i = (0.62 + 0.31) * np.sign(ux) * abs(ux) ** 0.6, 0.31 * np.sign(uz) * abs(uz) ** 0.6
            stad_o.append((cx + sx_o * tx + sz_o * math.cos(a), cz + sx_o * tz + sz_o * math.sin(a)))
            stad_i.append((cx + sx_i * tx + sz_i * math.cos(a), cz + sx_i * tz + sz_i * math.sin(a)))
        foil.ring_xz(stad_o, stad_i, 0.0, 1.0)
    # four low-speed signal wires in the centre
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        cond.cyl((0.45 * math.cos(a), 0.5, 0.45 * math.sin(a)), 0.16, 1.04, axis="Y", segs=12)
        diel.cyl((0.45 * math.cos(a), 0.5, 0.45 * math.sin(a)), 0.25, 1.0, axis="Y", segs=14)
    I.place(diel.build(name + "_dielectric", M["dielectric"], smooth_deg=40), coll, g)
    I.place(foil.build(name + "_pair_foil", M["foil"], smooth_deg=40), coll, g)
    I.place(cond.build(name + "_conductors", M["copper"], smooth_deg=40), coll, g)
    I.place(drain.build(name + "_drain_wires", M["tinned_copper"], smooth_deg=40), coll, g)
    return g


def cross_section_aoc(M, coll, parent, loc_mm, name, scale=1.0):
    g = I.empty(name + "_group", coll, parent, loc_mm=loc_mm)
    g.scale = (scale, scale, scale)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    mb = MB()
    mb.ring_xz([(1.5 * math.cos(a), 1.5 * math.sin(a)) for a in ang], [(1.15 * math.cos(a), 1.15 * math.sin(a)) for a in ang], 0.0, 1.0)
    I.place(mb.build(name + "_jacket", M["jacket_aqua"], smooth_deg=40), coll, g)
    mb = MB()
    for k in range(18):
        a = 2 * math.pi * k / 18
        mb.cyl((0.98 * math.cos(a), 0.5, 0.98 * math.sin(a)), 0.1, 1.0, axis="Y", segs=8)
    I.place(mb.build(name + "_aramid_yarn", M["plastic_yellow"], smooth_deg=30), coll, g)
    mb = MB()
    mb.box((-1.0, 1.0), (0, 1.0), (-0.16, 0.16), bev=0.04, seg=1)
    I.place(mb.build(name + "_ribbon_matrix", M["fiber_coat"], smooth_deg=30), coll, g)
    gl, co = MB(), MB()
    for i in range(8):
        x = (i - 3.5) * 0.25
        gl.cyl((x, 0.5, 0), 0.0625, 1.02, axis="Y", segs=10)
        co.cyl((x, 0.5, 0), 0.0045, 1.04, axis="Y", segs=6)
    I.place(gl.build(name + "_fibres", M["fiber_glass"], smooth_deg=30), coll, g)
    I.place(co.build(name + "_cores", M["fiber_core"], smooth_deg=0), coll, g)
    mb = MB()
    for sx in (-1, 1):
        mb.cyl((sx * 0.75, 0.5, 0), 0.25, 1.0, axis="Y", segs=12)
    I.place(mb.build(name + "_strength_rods", M["steel"], smooth_deg=30), coll, g)
    return g
