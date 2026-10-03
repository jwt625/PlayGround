"""Shared PIC / EIC die builders (metres). Die frames: centred on xy origin, bottom at z = 0, +X = fiber/edge-coupler edge.

PIC 7.0 (x) x 9.0 (y) x 0.775 mm; EIC 2.4 x 4.2 x 0.3 mm; EIC site on the PIC: centre (-0.2, 0) mm; bump pitch 40 um (60 x 105).
Waveguide lines and rings are drawn thicker than true (simplifications listed in each asset JSON).
"""
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402

C = P.C
MM, UM, PI = P.MM, P.UM, math.pi

PIC_X, PIC_Y, PIC_T = 7.0 * MM, 9.0 * MM, 0.775 * MM
EIC_X, EIC_Y, EIC_T = 2.4 * MM, 4.2 * MM, 0.30 * MM
BEOL_T = 10 * UM
BUMP_PITCH = 40 * UM
NBX, NBY = 60, 105
EIC_SITE = (-0.2 * MM, 0.0)  # centre of the EIC site on the PIC
LINE_T = 2.0 * UM  # raised height of drawn features above the die top
WG_W = 2.0 * UM  # drawn waveguide width (true 0.5 um)
RING_R = 7.5 * UM
RING_RIB = 1.0 * UM  # drawn rib width (true 0.5 um)


def bump_points(cx, cy, z):
    pts = []
    for i in range(NBX):
        for j in range(NBY):
            pts.append((cx + (i - (NBX - 1) / 2) * BUMP_PITCH, cy + (j - (NBY - 1) / 2) * BUMP_PITCH, z))
    return pts


def hybrid_pad_material(name="hybrid_bond_face", pitch=9 * UM, base=(0.02, 0.03, 0.12)):
    """Procedural hybrid-bond face: Cu pads (~4.5 um dia) in SiO2 at `pitch`, evaluated in object coordinates."""
    full = "MAT_photonics_" + name
    if full in bpy.data.materials:
        return bpy.data.materials[full]
    m = bpy.data.materials.new(full)
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])

    def trig(axis):
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 2 * PI / pitch
        nt.links.new(sep.outputs[axis], mul.inputs[0])
        s = nt.nodes.new("ShaderNodeMath")
        s.operation = "COSINE"
        nt.links.new(mul.outputs[0], s.inputs[0])
        return s

    cx, cy = trig("X"), trig("Y")
    ssum = nt.nodes.new("ShaderNodeMath")
    ssum.operation = "ADD"
    nt.links.new(cx.outputs[0], ssum.inputs[0])
    nt.links.new(cy.outputs[0], ssum.inputs[1])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (*base, 1)
    ramp.color_ramp.elements[1].position = 0.8  # cos x + cos y > 1.2 -> Cu pad (~40% of pitch across)
    ramp.color_ramp.elements[1].color = (0.95, 0.55, 0.38, 1)
    mapr = nt.nodes.new("ShaderNodeMapRange")
    mapr.inputs["From Min"].default_value = -2.0
    mapr.inputs["From Max"].default_value = 2.0
    mapr.inputs["To Min"].default_value = 0.0
    mapr.inputs["To Max"].default_value = 1.0
    nt.links.new(ssum.outputs[0], mapr.inputs["Value"])
    nt.links.new(mapr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.25
    bs.inputs["Metallic"].default_value = 0.5
    return m


def _ring_geo(sy, heater=True):
    g = P.Geo()
    g.annulus(0, 0, RING_R - RING_RIB / 2, RING_R + RING_RIB / 2, 0, LINE_T, n=24)
    h = P.Geo()
    if heater:
        h.annulus(0, 0, RING_R + 1.5 * UM, RING_R + 3.0 * UM, 0, LINE_T * 0.6, n=24, a0=0.4, a1=2 * PI - 0.4)
    return g, h


def gc_geo(x0, y0, direction=1, n_teeth=24, per=0.63 * UM * 4, w=12 * UM, taper=40 * UM):
    """Grating coupler for die-scale decals (period drawn 4x true, flagged)."""
    g = P.Geo()
    ns = 12
    for k in range(ns):
        xa, xb = taper * k / ns, taper * (k + 1) / ns
        wm = WG_W + (w - WG_W) * (k + 0.5) / ns
        g.box(x0 + direction * xa, y0 - wm / 2, 0, x0 + direction * xb, y0 + wm / 2, LINE_T)
    for t in range(n_teeth):
        xa = taper + t * per
        g.box(x0 + direction * xa, y0 - w / 2, 0, x0 + direction * (xa + per * 0.5), y0 + w / 2, LINE_T)
    return g


def build_pic(coll, parent, name, loc=(0, 0, 0), detail="high", with_bumps=True, bond_pads="microbump", thickness=None):
    """PIC die. Returns dict(objs=[...], top=z of die top, site=(cx,cy,sx,sy))."""
    T = thickness or PIC_T
    objs = []
    root = bpy.data.objects.new(name + "_root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.002
    coll.objects.link(root)
    root.parent = parent
    root.location = loc

    def mk(nm, geo, matname, smooth=False, bevel=None, loc_=(0, 0, 0)):
        o = geo.build("%s_%s" % (name, nm), P.mat(matname) if isinstance(matname, str) else matname, coll, root, loc=loc_, smooth=smooth, bevel=bevel)
        objs.append(o)
        return o

    hx, hy = PIC_X / 2, PIC_Y / 2
    mk("substrate_si", P.Geo().box(-hx, -hy, 0, hx, hy, T - BEOL_T), "silicon_die_edge")
    mk("beol_passivation", P.Geo().box(-hx, -hy, T - BEOL_T, hx, hy, T), "die_passivation")
    ztop = T
    # edge coupler window (dark trench strip) and 16 data + 4 laser edge couplers
    ec_y = [(0.125 * MM + 0.25 * MM * k) for k in range(8)] + [(-0.125 * MM - 0.25 * MM * k) for k in range(8)]
    laser_y = [-2.5 * MM - 0.25 * MM * k for k in range(4)]
    g = P.Geo().box(hx - 90 * UM, -hy + 0.4 * MM, ztop, hx, hy - 0.4 * MM, ztop + 0.3 * UM)
    mk("edge_trench", g, "mold_black")
    gw = P.Geo()
    for y in ec_y:
        gw.strip([(hx, y), (EIC_SITE[0] + EIC_X / 2 + 0.9 * MM, y)], WG_W, ztop, ztop + LINE_T)
    for y in laser_y:
        gw.strip([(hx, y), (hx - 0.6 * MM, y), (hx - 0.9 * MM, -2.6 * MM + (laser_y.index(y)) * 0.05 * MM)], WG_W, ztop, ztop + LINE_T)
    # laser manifold: vertical bus feeding the TX channels
    gw.strip([(hx - 0.9 * MM, -2.6 * MM), (hx - 0.9 * MM, 0.125 * MM), (hx - 0.9 * MM, 1.875 * MM)], WG_W, ztop, ztop + LINE_T)
    mk("waveguides", gw, "pic_waveguide")
    # ring banks (TX at x ~ +2.0 mm, RX at +2.35 mm): 8 rings per bus, alternating sides
    gring = P.Geo()
    gheat = P.Geo()
    gpd = P.Geo()
    gfan = P.Geo()
    bank_tx = (EIC_SITE[0] + EIC_X / 2 + 0.9 * MM)  # ring bank start x (right of EIC)
    nr = 8 if detail == "high" else 3
    for ci, y in enumerate(ec_y):
        for k in range(nr):
            sy = 1 if k % 2 == 0 else -1
            rx = bank_tx + 0.2 * MM + k * 40 * UM
            ry = y + sy * (RING_R + RING_RIB / 2 + 0.18 * UM + WG_W / 2)
            g1, h1 = _ring_geo(sy)
            gring.merge(g1, rx, ry, ztop)
            gheat.merge(h1, rx, ry, ztop)
        # Ge photodiode on RX channels (y < 0)
        if y < 0:
            gpd.box(bank_tx + 0.2 * MM + nr * 40 * UM + 20 * UM, y - 6 * UM, ztop, bank_tx + 0.2 * MM + nr * 40 * UM + 80 * UM, y + 6 * UM, ztop + LINE_T * 1.2)
        # fan-out of control lines from the bank to the EIC edge
        if detail == "high":
            for k in range(nr):
                sy = 1 if k % 2 == 0 else -1
                ry = y + sy * 15 * UM
                x_end = EIC_SITE[0] + EIC_X / 2 - 0.02 * MM
                y_end = EIC_SITE[1] + (ci - 7.5) * 0.12 * MM + (k - 3.5) * 14 * UM
                gfan.strip([(bank_tx + 0.2 * MM + k * 40 * UM, ry), (bank_tx + 0.1 * MM, ry), (bank_tx + 0.05 * MM, y_end), (x_end, y_end)], 5 * UM, ztop, ztop + LINE_T)
    mk("rings", gring, "waveguide_si")
    mk("ring_heaters", gheat, "tin_heater")
    mk("ge_photodiodes", gpd, "ge_pd")
    if detail == "high":
        mk("metal_fanout", gfan, "gold")
    # seal ring and fiducials
    gs = P.Geo()
    ins = 60 * UM
    gs.strip([(-hx + ins, -hy + ins), (hx - ins, -hy + ins), (hx - ins, hy - ins), (-hx + ins, hy - ins)], 25 * UM, ztop, ztop + LINE_T * 1.5, closed=True)
    gs.strip([(-hx + ins + 60 * UM, -hy + ins + 60 * UM), (hx - ins - 60 * UM, -hy + ins + 60 * UM), (hx - ins - 60 * UM, hy - ins - 60 * UM),
              (-hx + ins + 60 * UM, hy - ins - 60 * UM)], 6 * UM, ztop, ztop + LINE_T, closed=True)
    mk("die_seal_ring", gs, "copper")
    gf = P.Geo()
    for (fx, fy) in [(-hx + 0.3 * MM, -hy + 0.3 * MM), (-hx + 0.3 * MM, hy - 0.3 * MM), (hx - 0.4 * MM, -hy + 0.3 * MM), (hx - 0.4 * MM, hy - 0.3 * MM)]:
        gf.box(fx - 100 * UM, fy - 10 * UM, ztop, fx + 100 * UM, fy + 10 * UM, ztop + LINE_T * 1.5)
        gf.box(fx - 10 * UM, fy - 100 * UM, ztop, fx + 10 * UM, fy + 100 * UM, ztop + LINE_T * 1.5)
    mk("fiducials", gf, "gold_pad")
    # periphery bond / C4 pads (100 um squares at 150 um pitch)
    gp = P.Geo()
    for i in range(57):
        y = -4.2 * MM + i * 0.15 * MM
        gp.box(-hx + 0.16 * MM, y - 50 * UM, ztop, -hx + 0.26 * MM, y + 50 * UM, ztop + LINE_T * 1.5)
    for sgn in (-1, 1):
        for i in range(38):
            x = -3.0 * MM + i * 0.15 * MM
            if x < hx - 0.9 * MM:
                gp.box(x - 50 * UM, sgn * (hy - 0.26 * MM), ztop, x + 50 * UM, sgn * (hy - 0.16 * MM), ztop + LINE_T * 1.5)
    mk("periphery_pads", gp, "gold_pad")
    if detail == "high":
        # probe pad row (wafer-level test)
        gpr = P.Geo()
        for i in range(10):
            x = 0.9 * MM + i * 0.125 * MM
            gpr.box(x - 40 * UM, -hy + 0.34 * MM, ztop, x + 40 * UM, -hy + 0.42 * MM, ztop + LINE_T * 1.5)
        mk("probe_pads", gpr, "gold_pad")
        # grating-coupler loopback test structures
        gt = P.Geo()
        gl = P.Geo()
        for i in range(8):
            y = -3.55 * MM + i * 0.127 * MM
            gt.merge(gc_geo(-2.5 * MM, y, +1), 0, 0, ztop)
            gt.merge(gc_geo(-1.9 * MM, y, -1), 0, 0, ztop)
            gl.strip([(-2.5 * MM + 80 * UM, y), (-1.9 * MM - 80 * UM, y)], WG_W, ztop, ztop + LINE_T)
        mk("gc_test_structures", gt, "pic_waveguide")
        mk("gc_test_loops", gl, "pic_waveguide")
        # label
        t = C.text_mesh(name + "_label", "NARROWCOM PIC-A1", 0.22 * MM, loc=(-1.3 * MM, hy - 0.5 * MM, ztop + LINE_T), extrude=1.5 * UM, mat=P.mat("etch"))
        C.add(t, coll, root)
        objs.append(t)
    # microbump / hybrid-bond pad field on the EIC site
    cx, cy = EIC_SITE
    site = (cx, cy, EIC_X, EIC_Y)
    if with_bumps:
        if bond_pads == "microbump":
            proto_g = P.Geo().cyl(0, 0, 0, 3 * UM, 10 * UM, n=8)
            proto = proto_g.build(name + "_pad_proto", P.mat("copper"), coll, root, loc=(cx, cy, ztop))
            carrier = P.instancer(name + "_ubump_pads", proto, bump_points(cx, cy, ztop), coll, root)
            objs += [proto, carrier]
        else:
            gpad = P.Geo().box(cx - EIC_X / 2, cy - EIC_Y / 2, ztop, cx + EIC_X / 2, cy + EIC_Y / 2, ztop + 0.5 * UM)
            mk("hybrid_bond_pad_field", gpad, hybrid_pad_material())
    return dict(objs=objs, top=ztop, site=site, root=root, ec_y=ec_y, laser_y=laser_y)


def build_eic(coll, parent, name, loc=(0, 0, 0), bond="microbump", detail="high"):
    """EIC die, active face up (+z). bond: 'microbump' (Cu pillar + solder cap, 40 um pitch) or 'hybrid' (9 um Cu pad texture)."""
    T = EIC_T
    objs = []
    root = bpy.data.objects.new(name + "_root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.002
    coll.objects.link(root)
    root.parent = parent
    root.location = loc
    hx, hy = EIC_X / 2, EIC_Y / 2

    def mk(nm, geo, matname, **kw):
        o = geo.build("%s_%s" % (name, nm), P.mat(matname) if isinstance(matname, str) else matname, coll, root, **kw)
        objs.append(o)
        return o

    mk("substrate_si", P.Geo().box(-hx, -hy, 0, hx, hy, T - BEOL_T), "silicon_die_edge")
    mk("beol_passivation", P.Geo().box(-hx, -hy, T - BEOL_T, hx, hy, T), "eic_passivation")
    z = T
    if detail == "high" and bond == "microbump":
        # floorplan macros (16 lanes x [driver, TIA, SerDes slice]) + PLL + control column
        gd, gt, gs, gc, gp = P.Geo(), P.Geo(), P.Geo(), P.Geo(), P.Geo()
        lane_h = (EIC_Y - 0.5 * MM) / 16
        for i in range(16):
            y0 = -EIC_Y / 2 + 0.25 * MM + i * lane_h
            gd.box(-hx + 0.2 * MM, y0 + 4 * UM, z, -hx + 0.65 * MM, y0 + lane_h - 4 * UM, z + 1.0 * UM)
            gt.box(-hx + 0.7 * MM, y0 + 4 * UM, z, -hx + 1.1 * MM, y0 + lane_h - 4 * UM, z + 1.0 * UM)
            gs.box(-hx + 1.15 * MM, y0 + 4 * UM, z, hx - 0.45 * MM, y0 + lane_h - 4 * UM, z + 1.0 * UM)
        gc.box(hx - 0.4 * MM, -hy + 0.25 * MM, z, hx - 0.2 * MM, -hy + 1.8 * MM, z + 1.0 * UM)
        gp.box(hx - 0.4 * MM, -hy + 1.9 * MM, z, hx - 0.2 * MM, hy - 0.25 * MM, z + 1.0 * UM)
        mk("macro_driver", gd, P.mat("macro_drv", (0.75, 0.35, 0.08), 0.6, 0.3))
        mk("macro_tia", gt, P.mat("macro_tia", (0.1, 0.55, 0.3), 0.6, 0.3))
        mk("macro_serdes", gs, P.mat("macro_serdes", (0.12, 0.2, 0.8), 0.5, 0.3))
        mk("macro_control", gc, P.mat("macro_ctrl", (0.55, 0.15, 0.55), 0.5, 0.3))
        mk("macro_pll", gp, P.mat("macro_pll", (0.45, 0.45, 0.5), 0.5, 0.3))
        gseal = P.Geo()
        ins = 40 * UM
        gseal.strip([(-hx + ins, -hy + ins), (hx - ins, -hy + ins), (hx - ins, hy - ins), (-hx + ins, hy - ins)], 20 * UM, z, z + 2 * UM, closed=True)
        mk("die_seal_ring", gseal, "copper")
    if bond == "microbump":
        pts = bump_points(0, 0, z)
        pil = P.Geo().cyl(0, 0, 0, 12 * UM, 10 * UM, n=8)
        cap = P.Geo().hemisphere(0, 0, 12 * UM, 10 * UM, nseg=8, nring=2, squash=0.5)
        p1 = pil.build(name + "_pillar_proto", P.mat("copper"), coll, root, loc=(0, 0, z))
        p2 = cap.build(name + "_solder_proto", P.mat("solder"), coll, root, loc=(0, 0, z))
        c1 = P.instancer(name + "_cu_pillars", p1, pts, coll, root)
        c2 = P.instancer(name + "_solder_caps", p2, pts, coll, root)
        objs += [p1, p2, c1, c2]
    else:
        gpad = P.Geo().box(-hx + 20 * UM, -hy + 20 * UM, z, hx - 20 * UM, hy - 20 * UM, z + 0.3 * UM)
        mk("hybrid_bond_face", gpad, hybrid_pad_material())
    return dict(objs=objs, top=z, root=root)
