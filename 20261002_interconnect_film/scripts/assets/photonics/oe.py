"""Shared optical-engine pieces: PIC + EIC (flip-chip) + FAU on a carrier, fibre ribbon pigtail, MT-16 ferrule with boot."""
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl as P  # noqa: E402
import dies as D  # noqa: E402

C = P.C
MM, UM, PI = P.MM, P.UM, math.pi


def build_engine(coll, root, name, carrier_top, pic_center=(-2.0 * MM, 0.0), detail="low", fau_n=16, fau_pitch=250 * UM,
                 stack_gap=0.05 * MM, with_fau=True, eic_bond="microbump"):
    """PIC on the carrier, EIC flip-chipped on the PIC (bumps down), FAU butt-coupled to the +X PIC edge.
    Returns dict(pic_top, eic_back, fau_exit=(x,y,z) where coated fibres leave the FAU, fiber_axis_z, pic_info)."""
    bump = P.Geo().box(pic_center[0] - D.PIC_X / 2 + 0.4 * MM, pic_center[1] - D.PIC_Y / 2 + 0.4 * MM, carrier_top,
                       pic_center[0] + D.PIC_X / 2 - 0.4 * MM, pic_center[1] + D.PIC_Y / 2 - 0.4 * MM, carrier_top + stack_gap)
    bump.build(name + "_pic_attach_underfill", P.mat("mold_black"), coll, root)
    z_pic0 = carrier_top + stack_gap
    pic = D.build_pic(coll, root, name + "_pic", loc=(pic_center[0], pic_center[1], z_pic0), detail=detail, with_bumps=False)
    pic_top = z_pic0 + D.PIC_T
    # EIC face down on the PIC site
    sx = pic_center[0] + D.EIC_SITE[0]
    sy = pic_center[1] + D.EIC_SITE[1]
    gap = 17 * UM
    eic = D.build_eic(coll, root, name + "_eic", loc=(sx, sy, pic_top + gap + D.EIC_T), bond=eic_bond, detail="low")
    eic["root"].rotation_euler = (PI, 0, 0)
    eic_back = pic_top + gap + D.EIC_T
    out = dict(pic_top=pic_top, eic_back=eic_back, pic=pic, eic=eic)
    if with_fau:
        base_t = 0.80 * MM
        z0 = carrier_top
        xface = pic_center[0] + D.PIC_X / 2 + 0.010 * MM
        fr = bpy.data.objects.new(name + "_fau_mount", None)
        fr.empty_display_type = "ARROWS"
        fr.empty_display_size = 0.002
        coll.objects.link(fr)
        fr.parent = root
        fr.location = (xface, pic_center[1], z0)
        fr.rotation_euler = (0, 0, PI)
        fau = P.build_fau(coll, fr, name + "_fau", fau_n, fau_pitch, base_t=base_t, lid_t=0.5 * MM, length=5.0 * MM, fiber_len_out=2.5 * MM)
        out["fau"] = fau
        out["fiber_axis_z"] = z0 + fau["z_axis"]
        out["fau_exit"] = (xface + 5.0 * MM + 1.0 * MM + 2.5 * MM, pic_center[1], z0 + fau["z_axis"])
        out["fau_top"] = z0 + base_t + 0.5 * MM
        out["fau_face_x"] = xface
        out["fau_width"] = fau["base_w"]
    return out


def build_pigtail(coll, root, name, start, n_fib, length_pts, pitch=250 * UM, mt=True, mt_n=None):
    """Fibre ribbon pigtail from `start` (x,y,z) through list of waypoints to an MT ferrule (mating face at the last point + 8 mm).
    length_pts: waypoints after start (tuples in metres, absolute). Returns dict(mt_face=(x,y,z), end=(x,y,z))."""
    pts = [start] + list(length_pts)
    path = P.catmull(pts, 10)
    P.ribbon_fibers(path, n_fib, pitch, 125 * UM, coll, root, name + "_pigtail", n_sides=8)
    end = pts[-1]
    res = dict(rear=end)
    if mt:
        mt_n = mt_n or n_fib
        face = (end[0] + 8.0 * MM, end[1], end[2])
        P.build_mt_ferrule(coll, root, name + "_mt", mt_n, pitch, loc=face, with_pins=True)
        # strain-relief boot behind the ferrule
        L = 10 * MM
        g = P.Geo().box(end[0] - L, -3.4 * MM, -1.5 * MM, end[0], 3.4 * MM, 1.5 * MM)
        g.build(name + "_mt_boot", P.mat("boot_black"), coll, root, loc=(0, end[1], end[2]), bevel=(0.4 * MM, 3))
        res["mt_face"] = face
    return res


def carrier_substrate(coll, root, name, sx, sy, t, z0, cx=0.0, cy=0.0, lga_pitch=1.0 * MM, lga_d=0.6 * MM, lga_nx=None, lga_ny=None):
    """Organic carrier with solder-mask top and an LGA pad array on the underside (z0 = underside of pads)."""
    pad_t = 0.03 * MM
    zb = z0 + pad_t
    g = P.Geo().box(cx - sx / 2, cy - sy / 2, zb, cx + sx / 2, cy + sy / 2, zb + t * 0.97)
    g.build(name + "_carrier_core", P.mat("substrate_edge"), coll, root, bevel=(0.08 * MM, 2))
    P.Geo().box(cx - sx / 2 + 0.02 * MM, cy - sy / 2 + 0.02 * MM, zb + t * 0.97, cx + sx / 2 - 0.02 * MM, cy + sy / 2 - 0.02 * MM, zb + t).build(
        name + "_solder_mask", P.mat("organic_substrate"), coll, root)
    nx = lga_nx or int((sx - 1.5 * MM) / lga_pitch)
    ny = lga_ny or int((sy - 1.5 * MM) / lga_pitch)
    pts = [(cx + (i - (nx - 1) / 2) * lga_pitch, cy + (j - (ny - 1) / 2) * lga_pitch, z0) for i in range(nx) for j in range(ny)]
    proto = P.Geo().cyl(0, 0, 0, pad_t, lga_d / 2, n=12).build(name + "_lga_proto", P.mat("lga_pad"), coll, root, loc=(cx, cy, z0))
    carrier = P.instancer(name + "_lga_pads", proto, pts, coll, root)
    return dict(top=zb + t, nx=nx, ny=ny, pads=nx * ny)


def mlcc_row(coll, root, name, pts, z, size=(1.0 * MM, 0.5 * MM, 0.5 * MM)):
    gb = P.Geo()
    ge = P.Geo()
    for (x, y) in pts:
        gb.box_c(x, y, z + size[2] / 2, size[0] - 0.3 * MM, size[1], size[2])
        ge.box_c(x - (size[0] - 0.15 * MM) / 2, y, z + size[2] / 2, 0.15 * MM, size[1] * 1.02, size[2] * 1.02)
        ge.box_c(x + (size[0] - 0.15 * MM) / 2, y, z + size[2] / 2, 0.15 * MM, size[1] * 1.02, size[2] * 1.02)
    gb.build(name + "_mlcc_body", P.mat("cap_mlcc"), coll, root)
    ge.build(name + "_mlcc_terminals", P.mat("cap_end"), coll, root)
