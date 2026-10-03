"""Component builders (SMD passives, QFN, inductors, connectors, hardware). All return an MB (mm) centered on the
component's footprint center with the board-contact plane at z = 0 (mounted components sit on the pad plane).
Material slot lists are returned alongside (names of pk.mat entries).
"""
import math

import pk
from pk import MB, PI

# EIA imperial size code -> (L, W, T) in mm. Sources: EIA/JEDEC chip sizes, Murata/Samsung/Yageo datasheets (nominal values).
CHIP_SIZES = {
    "0201": (0.60, 0.30, 0.30),
    "0402": (1.00, 0.50, 0.50),
    "0603": (1.60, 0.80, 0.80),
    "0805": (2.00, 1.25, 1.25),
    "1206": (3.20, 1.60, 1.60),
}
# resistors are thinner than MLCC of the same footprint (typ. Yageo RC series)
RES_T = {"0201": 0.23, "0402": 0.35, "0603": 0.45, "0805": 0.55, "1206": 0.55}
# termination band length (end cap length along the chip), typ.
BAND = {"0201": 0.15, "0402": 0.25, "0603": 0.35, "0805": 0.50, "1206": 0.50}


def mlcc(size="0402", t=None, body_mat="mlcc_body"):
    """MLCC: ceramic body (slot 0) and two tin terminations (slot 1). z from 0 to T."""
    L, W, T = CHIP_SIZES[size]
    if t is not None:
        T = t
    b = BAND[size]
    e = 0.03 * T
    mb = MB()
    mb.box((0, 0, T / 2), (L - 2 * e, W - 2 * e, T - 2 * e), mi=0)
    # terminations: the boxes wrap the end caps; they overlap the ceramic volume but no faces are coplanar
    for s in (-1, 1):
        mb.box((s * (L - b) / 2, 0, T / 2), (b, W, T), mi=1, bevel=min(0.06 * T, 0.03), seg=1)
    return mb, [body_mat, "tin"]


def chip_resistor(size="0402"):
    """Thick-film chip resistor: black overcoat top (slot 0), alumina body (slot 1), tin terminations (slot 2)."""
    L, W, T = CHIP_SIZES[size]
    T = RES_T[size]
    b = BAND[size]
    mb = MB()
    mb.box((0, 0, T * 0.35), (L - 2 * b + 0.001, W * 0.98, T * 0.7), mi=1)
    mb.box((0, 0, T * 0.7 + (T * 0.3) / 2), (L - 2 * b - 0.04, W * 0.78, T * 0.3), mi=0)
    for s in (-1, 1):
        mb.box((s * (L - b) / 2, 0, T / 2), (b, W, T), mi=2, bevel=0.02, seg=1)
    return mb, ["res_black", "alumina", "tin"]


def ferrite_bead(size="0603"):
    L, W, T = CHIP_SIZES[size]
    b = BAND[size]
    mb = MB()
    mb.box((0, 0, T / 2), (L - 2 * b + 0.001, W, T), mi=0)
    for s in (-1, 1):
        mb.box((s * (L - b) / 2, 0, T / 2), (b, W * 1.0, T), mi=1, bevel=0.03, seg=1)
    return mb, ["ferrite", "tin"]


def chip_inductor_0603():
    """0603 wire/multilayer chip inductor: gray-blue body with tin terminations (L 1.6, W 0.8, T 0.8)."""
    L, W, T = CHIP_SIZES["0603"]
    mb = MB()
    b = 0.30
    mb.box((0, 0, T / 2), (L - 2 * b + 0.001, W, T), mi=0)
    for s in (-1, 1):
        mb.box((s * (L - b) / 2, 0, T / 2), (b, W, T), mi=1, bevel=0.03, seg=1)
    return mb, ["inductor_body", "tin"]


# ---------------------------------------------------------------------------------------------------------------- more parts
def _box_pockets(mb, c, w, d, h, centers, cw, cd, depth, mi=0, mi_floor=0):
    """Housing box (z from c.z) with rectangular pockets (cw x cd, given depth) cut from the top, built from slabs (no booleans)."""
    cx0, cy0, z0 = c
    mb.box((cx0, cy0, z0 + (h - depth) / 2), (w, d, h - depth), mi=mi_floor)
    xs = sorted({-w / 2, w / 2} | {p[0] - cw / 2 for p in centers} | {p[0] + cw / 2 for p in centers})
    ys = sorted({-d / 2, d / 2} | {p[1] - cd / 2 for p in centers} | {p[1] + cd / 2 for p in centers})
    for a, b in zip(xs[:-1], xs[1:]):
        for e, f in zip(ys[:-1], ys[1:]):
            mx, my = (a + b) / 2, (e + f) / 2
            if any(abs(mx - p[0]) < cw / 2 and abs(my - p[1]) < cd / 2 for p in centers):
                continue
            mb.box((cx0 + mx, cy0 + my, z0 + h - depth / 2), (b - a, f - e, depth), mi=mi)


def tantalum(case="B", polymer=False):
    """Molded SMD tantalum/polymer capacitor, EIA 535BAAC cases A 3216-18, B 3528-21, C 6032-28, D 7343-31."""
    L, W, H = {"A": (3.2, 1.6, 1.8), "B": (3.5, 2.8, 2.1), "C": (6.0, 3.2, 2.8), "D": (7.3, 4.3, 3.1)}[case]
    mb = MB()
    t = 0.12
    bl = L - 2 * 0.0
    mb.box((0, 0, 0.1 + (H - 0.1) / 2), (L * 0.86, W * 0.9, H - 0.1), mi=0, bevel=min(0.15, W * 0.08), seg=1)
    # anode polarity band on top at the -x end
    mb.box((-L * 0.86 / 2 + L * 0.07, 0, H + 0.003), (L * 0.14, W * 0.9 - 0.2, 0.006), mi=1)
    for s in (-1, 1):
        # L-shaped termination: foot under the body end and a vertical strap
        mb.box((s * (L / 2 - 0.55), 0, 0.05), (1.1 if L > 3.3 else 0.9, W * 0.75, 0.1), mi=2)
        mb.box((s * (L * 0.43 + 0.04), 0, 0.1 + (H * 0.55) / 2), (0.12, W * 0.75, H * 0.55), mi=2)
    return mb, ["polymer_cap" if polymer else "tantalum", "silkscreen", "tin"]


def alu_electrolytic(d=6.3, h=7.7):
    """SMD aluminum electrolytic (V-chip) can on a plastic base."""
    mb = MB()
    r = d / 2
    mb.box((0, 0, 0.5), (d + 0.4, d + 0.4, 1.0), mi=0, bevel=0.4, seg=1)
    mb.lathe([(0, 1.0), (r * 0.98, 1.0), (r, 1.3), (r, h - 0.5), (r * 0.94, h), (0, h)], seg=32, mi=1)
    mb.box((0, 0, h + 0.004), (d * 0.5, 0.25, 0.008), mi=2)
    mb.box((0, 0, h + 0.004), (0.25, d * 0.5, 0.008), mi=2)
    for s in (-1, 1):
        mb.box((s * (r + 0.35), 0, 0.15), (1.3, 0.9 if d < 8 else 1.2, 0.3), mi=3)
    return mb, ["plastic_black", "electrolytic_can", "steel", "tin"]


def power_inductor(w, d, h, name=None):
    """Shielded molded power inductor (metal-composite body, wrap-around terminals on the two short sides)."""
    mb = MB()
    mb.box((0, 0, 0.1 + (h - 0.1) / 2), (w, d, h - 0.1), mi=0, bevel=0.35, seg=2)
    tw = w * 0.27
    for s in (-1, 1):
        mb.box((s * (w / 2 - tw / 2), 0, 0.2), (tw, d * 0.7, 0.4), mi=1)
        mb.box((s * (w / 2 + 0.03), 0, 0.2 + (h * 0.45) / 2), (0.1, d * 0.7, h * 0.45), mi=1)
    return mb, ["inductor_body", "tin"]


def toroid(od=12.0, idm=6.0, hh=5.0, turns=28, lead_len=3.0):
    """Ferrite toroid with a copper winding (two leads), standing 1.5 mm above the board plane."""
    mb = MB()
    z0 = 1.5 + hh / 2
    ro, ri = od / 2, idm / 2
    mb.lathe([(ri + 0.4, z0 - hh / 2), (ro - 0.4, z0 - hh / 2), (ro, z0 - hh / 2 + 0.4), (ro, z0 + hh / 2 - 0.4), (ro - 0.4, z0 + hh / 2),
              (ri + 0.4, z0 + hh / 2), (ri, z0 + hh / 2 - 0.4), (ri, z0 - hh / 2 + 0.4)], seg=48, mi=0, close=True)
    rm = (ro + ri) / 2
    a = (ro - ri) / 2 + 0.35
    b = hh / 2 + 0.35
    n_pts = 20
    for k in range(turns):
        th = math.radians(20 + k * (300.0 / (turns - 1)))
        path = []
        for q in range(n_pts):
            ph = 2 * PI * q / n_pts
            rr = rm + a * math.cos(ph)
            path.append((rr * math.cos(th), rr * math.sin(th), z0 + b * math.sin(ph)))
        mb.tube(path, 0.28, seg=6, mi=1, closed=True)
    for s_, th in ((0, 340.0), (1, 40.0)):
        t = math.radians(th)
        mb.cyl((rm * math.cos(t), rm * math.sin(t), (1.5 - lead_len) / 2 + 0.0), 0.35, 1.5 + lead_len, seg=8, mi=2)
    return mb, ["ferrite_core", "enamel_wire", "tin"]


def flyback_transformer(w=13.0, d=11.0):
    """Small SMD flyback transformer: bobbin base with 2 x 5 gull-wing pins, copper winding under kapton, ferrite yoke and legs."""
    mb = MB()
    mb.box((0, 0, 0.3 + 0.5), (w - 1.0, d, 1.0), mi=0, bevel=0.15, seg=1)             # bobbin flange
    mb.box((0, 0, 1.3 + 2.3), (w - 4.2, d - 2.0, 4.6), mi=1, bevel=0.3, seg=2)         # winding with tape
    for s in (-1, 1):
        mb.box((s * (w / 2 - 1.05), 0, 1.3 + 2.8), (2.1, d - 1.6, 5.6), mi=2, bevel=0.15, seg=1)   # ferrite legs
    mb.box((0, 0, 1.3 + 5.6 + 1.0), (w, d - 1.6, 2.0), mi=2, bevel=0.2, seg=1)           # ferrite yoke
    for s in (-1, 1):
        for k in range(5):
            y = (k - 2) * 2.0
            mb.box((s * (w / 2 - 0.3), y, 0.25), (1.4 if False else 1.1, 0.7, 0.15), mi=3)
            mb.box((s * (w / 2 - 0.9), y, 0.5), (0.2, 0.7, 0.55), mi=3)
    return mb, ["plastic_black", "kapton", "ferrite_core", "tin"]


def qfn(w, d, h, pitch, nx, ny, lead_w=0.25, lead_l=0.40, ep=None, pads=None, pin1=True):
    """QFN/PQFN: molded body, perimeter leads (tin) and exposed pad(s). Leads exposed on the underside and the body sides."""
    mb = MB()
    mb.box((0, 0, 0.0 + h / 2 + 0.02), (w, d, h), mi=0, bevel=0.04, seg=1)
    for s in (-1, 1):
        for k in range(nx):                 # leads along x on the -y/+y sides
            x = (k - (nx - 1) / 2) * pitch
            mb.box((x, s * (d / 2 - lead_l / 2 + 0.02), 0.1), (lead_w, lead_l, 0.2), mi=1)
        for k in range(ny):                 # leads along y on the -x/+x sides
            y = (k - (ny - 1) / 2) * pitch
            mb.box((s * (w / 2 - lead_l / 2 + 0.02), y, 0.1), (lead_l, lead_w, 0.2), mi=1)
    if ep:
        mb.box((0, 0, 0.1), (ep[0], ep[1], 0.2), mi=2)
    if pads:
        for (px, py, pw, pd_) in pads:
            mb.box((px, py, 0.1), (pw, pd_, 0.2), mi=2)
    if pin1:
        mb.lathe([(0, 0), (0.28, 0), (0.28, 0.006), (0, 0.006)], c=(-w / 2 + 0.55, d / 2 - 0.55, h + 0.02), seg=14, mi=3, smooth=False)
    return mb, ["mold_black", "tin", "copper_dull", "plastic_black"]


def drmos_5x6():
    return qfn(5.0, 6.0, 0.8, 0.5, 8, 10, lead_w=0.25, lead_l=0.45, ep=None,
               pads=[(-1.25, -1.5, 1.4, 2.4), (1.25, -1.5, 1.4, 2.4), (0.0, 1.55, 3.4, 1.6)])


def dcdc_controller_4x4():
    return qfn(4.0, 4.0, 0.75, 0.5, 6, 6, lead_w=0.25, lead_l=0.40, ep=(2.7, 2.7))


def b2b_receptacle(npos=50, pitch=0.5, W=4.0, H=1.5):
    """Board-to-board receptacle, 2 rows at 0.5 mm pitch (npos total contacts)."""
    per = npos // 2
    L = (per - 1) * pitch + 2.4
    mb = MB()
    mb.box((0, 0, 0.15), (L, W, 0.3), mi=0)
    for s in (-1, 1):
        mb.box((0, s * (W / 2 - 0.3), H / 2), (L, 0.6, H), mi=0)
        mb.box((s * (L / 2 - 0.4), 0, H / 2), (0.8, W - 1.2, H), mi=0)
    mb.box((0, 0, 0.3 + (H - 0.3) / 2), (L - 2.0, 1.1, H - 0.3), mi=1)   # center tongue
    for k in range(per):
        x = (k - (per - 1) / 2) * pitch
        for s in (-1, 1):
            mb.box((x, s * 0.7, 0.3 + (H - 0.5) / 2 + 0.1), (0.2, 0.08, H - 0.5), mi=2)             # contacts on the tongue
            mb.box((x, s * (W / 2 + 0.25), 0.05), (0.22, 0.5, 0.1), mi=3)                          # tails
    return mb, ["plastic_black", "plastic_white", "gold_enig", "tin"]


def b2b_plug(npos=50, pitch=0.5, W=3.6, H=1.2):
    per = npos // 2
    L = (per - 1) * pitch + 2.0
    mb = MB()
    mb.box((0, 0, H / 2), (L, W, H), mi=0)
    mb.box((0, 0, H + 0.0 + 0.3), (L - 1.4, 1.0, 0.6), mi=0)
    for k in range(per):
        x = (k - (per - 1) / 2) * pitch
        for s in (-1, 1):
            mb.box((x, s * 0.55, H + 0.3), (0.2, 0.08, 0.5), mi=1)
            mb.box((x, s * (W / 2 + 0.25), 0.05), (0.22, 0.5, 0.1), mi=2)
    return mb, ["plastic_white", "gold_enig", "tin"]


def power_connector_2x3(pitch=4.2):
    """Mini-Fit-style 2 x 3 vertical header: housing with 6 pockets, pins, and a latch ramp."""
    w, d, h = 3 * pitch + 0.6, 2 * pitch + 0.6, 11.0
    cts = [((i - 1) * pitch, (j - 0.5) * pitch) for i in range(3) for j in range(2)]
    mb = MB()
    _box_pockets(mb, (0, 0, 0.4), w, d, h, cts, 3.0, 3.0, 9.0, mi=0, mi_floor=0)
    for (x, y) in cts:
        mb.box((x, y, 0.4 + 2.2 + 3.0), (1.0, 1.0, 6.0), mi=1)          # pin inside the pocket
        mb.cyl((x, y, -1.0), 0.45, 2.6, seg=8, mi=2)                      # through-hole tail
    mb.box((0, d / 2 + 0.25, 0.4 + h * 0.6), (3.0, 0.5, h * 0.5), mi=0, bevel=0.1, seg=1)   # latch rib
    return mb, ["plastic_white", "gold_enig", "tin"]


def test_point_pad(dia=1.0):
    mb = MB()
    mb.lathe([(0, 0), (dia / 2 + 0.3, 0), (dia / 2 + 0.3, 0.005), (0, 0.005)], seg=24, mi=1, smooth=False)
    mb.lathe([(0, 0.005), (dia / 2, 0.005), (dia / 2, 0.04), (0, 0.04)], seg=24, mi=0, smooth=False)
    return mb, ["gold_enig", "substrate_land"]


def test_point_loop(d=5.0, wire=0.4):
    """Through-hole loop test point (Keystone 5000 style): wire loop with two legs."""
    mb = MB()
    r = d / 2
    path = [(r * math.cos(a), 0.0, 1.5 + r + r * math.sin(a)) for a in [math.radians(-90 + 360 * k / 24) for k in range(0, 24)]]
    mb.tube(path, wire / 2, seg=6, mi=0, closed=True)
    for s in (-1, 1):
        mb.cyl((s * 0.8, 0, 0.2), wire / 2 + 0.05, 3.4, seg=8, mi=0)
    return mb, ["tin"]


def screw_m3(length=6.0, head_d=5.6, head_h=2.4, r=1.5):
    """M3 pan-head cross-recess screw (ISO 7045 class: head dia 5.6, height 2.4), tip down, head seating plane at z = 0 (shaft is below)."""
    mb = MB()
    prof = [(0, -length), (r - 0.25, -length), (r, -length + 0.25)]
    z = -length + 0.25
    while z < -0.3:
        prof += [(r - 0.12, z + 0.12), (r, z + 0.25)]
        z += 0.5
    prof += [(r, 0.0), (head_d / 2 - 0.3, 0.0), (head_d / 2, 0.3), (head_d / 2, head_h * 0.55), (head_d / 2 * 0.8, head_h * 0.9), (head_d / 2 * 0.5, head_h), (0, head_h)]
    mb.lathe(prof, seg=20, mi=0)
    mb.box((0, 0, head_h + 0.004), (head_d * 0.5, 0.5, 0.01), mi=1)
    mb.box((0, 0, head_h + 0.004), (0.5, head_d * 0.5, 0.01), mi=1)
    return mb, ["steel", "steel_dark"]


def standoff_m3_hex(length=8.0, af=5.5):
    """M3 female-female hex standoff, 5.5 mm across flats."""
    mb = MB()
    rc = af / math.sqrt(3)
    pts = [(rc * math.cos(math.radians(60 * k)), rc * math.sin(math.radians(60 * k))) for k in range(6)]
    mb.prism(pts, 0.0, length, mi=0)
    mb.lathe([(0, 0.0), (1.5, 0.0), (1.5, 0.004), (0, 0.004)], seg=16, mi=1, smooth=False)
    mb.lathe([(0, length), (1.5, length), (1.5, length + 0.004), (0, length + 0.004)], seg=16, mi=1, smooth=False)
    return mb, ["steel", "steel_dark"]


def washer_m3():
    mb = MB()
    mb.lathe([(1.6, 0.0), (3.5, 0.0), (3.5, 0.5), (1.6, 0.5)], seg=24, mi=0, smooth=False, close=True)
    return mb, ["steel"]


def led_0603(color="green"):
    mb = MB()
    mb.box((0, 0, 0.15), (1.6, 0.8, 0.3), mi=0, bevel=0.03, seg=1)
    mb.lathe([(0.0, 0.3), (0.34, 0.3), (0.34, 0.38), (0.22, 0.52), (0.0, 0.58)], c=(0, 0, 0), seg=14, mi=1)
    for s in (-1, 1):
        mb.box((s * 0.7, 0, 0.05), (0.3, 0.8, 0.1), mi=2)
    return mb, ["plastic_white", "led_" + color, "tin"]
