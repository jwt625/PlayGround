"""HBM stack geometry builders (shared by build_hbm_stack.py and build_xpu_package_rubin_style.py). All dimensions in mm.

Stack layer model (bottom to top, z = 0 at microbump tips under the base die):
  microbump dome (20 um) | base die | [bond line + DRAM die] x (n-1) | bond line + thick top die
Total stack thickness is fitted to the JEDEC package height limit (HBM3: 720 um, HBM4: 775 um) by solving the bond-line thickness.
"""
import math

import pk
from pk import MB, MM

# material slot order used by every mesh built here (call hbm_materials() to get the list in this order)
SLOTS = ["base_die", "dram_si", "bond_layer", "solder", "mold_black", "hbm_tan", "copper", "underfill"]


def hbm_materials():
    return [pk.mat(n) for n in SLOTS]


class Spec:
    def __init__(self, n=16, footprint=11.0, dram=10.4, total_um=None, core_um=None, base_um=150.0, top_um=100.0,
                 bump_h_um=20.0):
        self.n = n
        self.W = footprint
        self.D = footprint
        self.dram = dram
        self.total_um = total_um if total_um else (775.0 if n >= 16 else 720.0)
        self.core_um = core_um if core_um else (25.0 if n >= 16 else 30.0)
        self.base_um = base_um
        self.top_um = top_um
        self.bump_um = bump_h_um
        used = base_um + (n - 1) * self.core_um + top_um
        self.gap_um = (self.total_um - used) / n

    def layers(self):
        """List of (kind, z0_mm, z1_mm, index). Bond lines carry kind 'bond'; dies 'base'/'dram'/'top'."""
        z = self.bump_um * 1e-3
        out = [("base", z, z + self.base_um * 1e-3, 0)]
        z += self.base_um * 1e-3
        for i in range(self.n):
            g = self.gap_um * 1e-3
            out.append(("bond", z, z + g, i))
            z += g
            t = (self.top_um if i == self.n - 1 else self.core_um) * 1e-3
            out.append(("top" if i == self.n - 1 else "dram", z, z + t, i))
            z += t
        return out

    @property
    def height(self):
        return self.layers()[-1][2]


def bump_points(spec, pitch_phy=0.055, pitch_pg=0.22, phy=(6.6, 2.4), z=0.0):
    """Underside microbump sites (mm): fine 55 um array in the central PHY band, coarser power/ground array elsewhere.
    Real HBM bump maps are proprietary; this is a plausible stand-in (accuracy C)."""
    pts = []
    nx = int(phy[0] / pitch_phy)
    ny = int(phy[1] / pitch_phy)
    for j in range(ny):
        for i in range(nx):
            pts.append(((i - (nx - 1) / 2) * pitch_phy, (j - (ny - 1) / 2) * pitch_phy, z))
    m = 0.35
    nx2 = int((spec.W - 2 * m) / pitch_pg)
    ny2 = int((spec.D - 2 * m) / pitch_pg)
    for j in range(ny2):
        for i in range(nx2):
            x = (i - (nx2 - 1) / 2) * pitch_pg
            y = (j - (ny2 - 1) / 2) * pitch_pg
            if abs(x) < phy[0] / 2 + 0.1 and abs(y) < phy[1] / 2 + 0.1:
                continue
            pts.append((x, y, z))
    return pts


def bump_source(coll, parent, dia_um=35.0, h_um=20.0, name="hbm_stack_src_ubump"):
    """Low-poly flattened solder dome, centered at its own origin (instance origin at the base-die underside plane)."""
    b = MB()
    r = dia_um * 0.5e-3
    h = h_um * 1e-3
    b.lathe([(r * 0.78, -h), (r * 1.0, -h * 0.5), (r * 0.78, -0.0), (0, 0)], seg=6, mi=0)
    o = b.to_obj(name, [pk.mat("solder")], coll, parent)
    return pk.source(o)


def stack_mesh(spec, kind="exposed", with_bumps=False):
    """MB with the whole stack (no TSV, no microbump instances). kind: 'exposed' | 'molded'."""
    b = MB()
    W, D, dr = spec.W, spec.D, spec.dram
    for (k, z0, z1, i) in spec.layers():
        h = z1 - z0
        c = (0, 0, (z0 + z1) / 2)
        if k == "base":
            b.box(c, (W, D, h), mi=0, bevel=0.012, seg=1)
        elif k == "dram":
            b.box(c, (dr, dr, h), mi=1)
        elif k == "top":
            b.box(c, (dr, dr, h), mi=5 if kind == "exposed" else 5)
        elif k == "bond" and kind == "exposed":
            # underfill / NCF squeeze-out bead protrudes a bit beyond the DRAM edge
            ext = dr + 0.10
            b.box(c, (ext, ext, h * 0.92), mi=2)
    if kind == "molded":
        z0 = spec.layers()[0][2]
        zt = spec.height
        # mold compound around the DRAM stack, flush with the top die; footprint equals the base die
        t = 0.0
        hole = dr
        # four wall slabs (no coplanar overlap) around the stack
        mh = zt - z0
        mc = (z0 + zt) / 2
        wx = (W - hole) / 2
        b.box((-(hole / 2 + wx / 2), 0, mc), (wx, D, mh), mi=4, bevel=0.015)
        b.box((+(hole / 2 + wx / 2), 0, mc), (wx, D, mh), mi=4, bevel=0.015)
        wy = (D - hole) / 2
        b.box((0, -(hole / 2 + wy / 2), mc), (hole, wy, mh), mi=4, bevel=0.0)
        b.box((0, +(hole / 2 + wy / 2), mc), (hole, wy, mh), mi=4, bevel=0.0)
    return b


def section_parts(spec, coll, parent, src_coll, root, prefix, loc=(0.0, 0.0, 0.0), tsv_pitch=0.055, ubump_pts=None,
                  depth=None):
    """HBM section at y = 0 (back half kept, depth = D/2 unless given). Cut face normal is -Y. Returns created objects.

    loc shifts the whole piece (mm). TSV columns, microbump joints (instances) lie on the cut plane."""
    sp = spec
    W, D, dr = sp.W, sp.D, sp.dram
    hd = depth if depth else D / 2
    mats = hbm_materials()
    b = MB()
    z_b = sp.layers()[0][2]
    zt = sp.height
    for (k, z0, z1, i) in sp.layers():
        h = z1 - z0
        cz = (z0 + z1) / 2
        if k == "base":
            b.box((0, hd / 2, cz), (W, hd, h), mi=0)
        elif k in ("dram", "top"):
            b.box((0, hd / 2, cz), (dr, hd, h), mi=1)
        else:
            b.box((0, hd / 2, cz), (dr + 0.10, hd, h * 0.92), mi=2)
    wx = (W - dr) / 2
    mh = zt - z_b
    b.box((-(dr / 2 + wx / 2), hd / 2, (z_b + zt) / 2), (wx, hd, mh), mi=4)
    b.box((+(dr / 2 + wx / 2), hd / 2, (z_b + zt) / 2), (wx, hd, mh), mi=4)
    objs = [b.to_obj(prefix + "_body", mats, coll, parent, loc)]
    nx = int(dr / tsv_pitch) - 2
    xs = [(i - (nx - 1) / 2) * tsv_pitch for i in range(nx)]
    by_t = {}
    for (k, z0, z1, i) in sp.layers():
        by_t.setdefault((k, round(z1 - z0, 6)), []).append((z0 + z1) / 2)
    for (k, h), zs in by_t.items():
        mb = MB()
        if k in ("dram", "top", "base"):
            mb.box((0, 0.0027, 0), (0.006, 0.0060, h), mi=0)
            src = mb.to_obj(prefix + "_src_tsv_" + k, [pk.mat("copper")], src_coll, root)
        else:
            mb.box((0, 0.0087, 0), (0.035, 0.0175, h * 0.92), mi=0)
            src = mb.to_obj(prefix + "_src_jbump", [pk.mat("solder")], src_coll, root)
        pk.source(src)
        pts = [(x, 0.0, z) for z in zs for x in xs]
        objs.append(pk.scatter(prefix + "_cut_" + k, pts, src, coll, parent, loc=loc))
    if ubump_pts is not None:
        src = bump_source(src_coll, root, name=prefix + "_src_ubump")
        objs.append(pk.scatter(prefix + "_ubumps", [p for p in ubump_pts if p[1] >= -1e-6], src, coll, parent, loc=loc))
    return objs
