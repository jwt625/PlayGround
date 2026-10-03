"""Shared datacenter part generators: compute tray, NVLink switch tray, rack frame helpers.

Tray local frame: x = centre, y = 0 at tray centre (front at y = -D/2, -Y is the front), z = 0 at tray bottom.
All dimension constants below are documented (source / accuracy) in the asset metadata and DevLog.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import dc_common as D  # noqa: E402
from dc_common import MB, mm  # noqa: E402

# tray envelope (mm). H: 1U with 1.5 mm clearance. W: ORV3 21-inch opening (537 mm) minus slide clearance. D: estimate.
TRAY_H = 43.0
TRAY_W = 526.0
TRAY_BODY_W = 510.0
TRAY_D = 820.0
SW_D = 820.0  # switch tray depth (estimate; same envelope as compute tray, flush fronts in photos)

_cache = {}


class Part:
    def __init__(self, name, mb, mats, bevel=None, smooth=False, tag=None):
        self.name, self.mb, self.mats, self.bevel, self.smooth, self.tag = name, mb, mats, bevel, smooth, tag


def instantiate(parts, coll, parent, prefix, loc=(0, 0, 0), key=None, subcolls=None):
    """Create objects from parts, sharing mesh data across calls with the same cache key. Returns {tag/name: obj}."""
    out = {}
    for p in parts:
        ck = (key, p.name) if key else None
        if ck and ck in _cache:
            me = _cache[ck]
        else:
            me = p.mb.mesh(prefix + p.name if not key else "mesh_%s_%s" % (key, p.name))
            for m in p.mats:
                me.materials.append(m)
            if p.smooth:
                for pl in me.polygons:
                    pl.use_smooth = True
            if ck:
                _cache[ck] = me
        o = bpy.data.objects.new("%s_%s" % (prefix, p.name), me)
        if p.bevel:
            C.bevel(o, p.bevel[0], p.bevel[1] if len(p.bevel) > 1 else 2)
        target = (subcolls or {}).get(p.tag, coll)
        target.objects.link(o)
        o.parent = parent
        o.location = loc
        out[p.name] = o
    return out


# ------------------------------------------------------------------ small hardware
def u_handle(mb, x, y_front, z0, z1, depth=22.0, r=2.8, m=0):
    """Vertical loop handle (mm coords) projecting toward -y from the front plane y_front."""
    pts = [(x, y_front, z0), (x, y_front - depth, z0 + 3), (x, y_front - depth, z1 - 3), (x, y_front, z1)]
    mb.tube([tuple(a * 0.001 for a in p) for p in pts], r * 0.001, 8, m)


def qd_male(mb, x, y, z, color_m, m_body=0, length=24.0, axis="y", sign=1):
    """Quick-disconnect (UQD-style) blind-mate male: steel body + coloured collar. Axis along y, pointing sign*y."""
    mb.cyl((x * 0.001, (y + sign * length / 2) * 0.001, z * 0.001), 0.0075, length * 0.001, "y", 16, m_body)
    mb.cyl((x * 0.001, (y + sign * 6) * 0.001, z * 0.001), 0.0098, 9 * 0.001, "y", 16, color_m)


def blindmate_block(mb, cx, cy, cz, w, h, d, m_house=0, m_pin=1, rows=3, cols=None, pitch=2.0, sign=1):
    """Floating blind-mate connector housing with a gold contact field facing sign*y (mm)."""
    mb.boxmm((cx, cy + sign * d / 2, cz), (w, d, h), m_house)
    face = cy + sign * d
    cols = cols or int((w - 8) / pitch)
    rows = max(1, rows)
    for r in range(rows):
        z = cz - (rows - 1) * pitch * 1.4 / 2 + r * pitch * 1.4
        mb.boxmm((cx, face + sign * 0.25, z), (cols * pitch, 0.5, 1.1), m_pin)


# ------------------------------------------------------------------ compute tray
def compute_tray_parts(detail="high", lid=True, name="tray_compute"):
    """Returns list[Part]. detail 'high' = boards, chips, cold plates, fans; 'low' = shell only (lid fixed on).

    mats are looked up from D.lib() so call after reset.
    """
    L = D.lib()
    W2 = TRAY_BODY_W / 2
    Dm = TRAY_D
    y0, y1 = -Dm / 2, Dm / 2
    H = TRAY_H
    parts = []

    # chassis pan
    ch = MB()
    ch.boxmm((0, 0, 1.0), (TRAY_BODY_W, Dm - 6, 2.0))
    ch.boxmm((-W2 + 0.75, 0, H / 2 - 1), (1.5, Dm - 6, H - 3))
    ch.boxmm((W2 - 0.75, 0, H / 2 - 1), (1.5, Dm - 6, H - 3))
    ch.boxmm((0, y1 - 3.75, H / 2 - 1), (TRAY_BODY_W, 1.5, H - 3))
    parts.append(Part("chassis", ch, [L["steel"]], bevel=(mm(0.5), 1)))
    # slide members (alu inner on tray, dark steel outer rack-side)
    for s, side in ((-1, "L"), (1, "R")):
        sl = MB()
        sl.boxmm((s * (W2 + 2.0), -10, H / 2), (4.0, Dm - 80, H - 10))
        parts.append(Part("slide_inner_" + side, sl, [L["alu"]], bevel=(mm(0.4), 1), tag="rails"))
        ol = MB()
        ol.boxmm((s * (W2 + 6.0), 10, H / 2), (4.0, Dm - 50, H - 4))
        parts.append(Part("slide_outer_" + side, ol, [L["dark_steel"]], bevel=(mm(0.4), 1), tag="rails"))
    # front bezel
    fb = MB()
    fb.boxmm((0, y0 - 1.5, H / 2), (TRAY_BODY_W + 12, 3.0, H - 1.5))
    parts.append(Part("front_panel", fb, [L["tray_front"]], bevel=(mm(0.6), 2)))
    # perforated vent plates (two) on the front, hole pattern from shader
    pm = D.perforated_material("perforated_front", (0.09, 0.095, 0.1), 0.8, 0.4, pitch=0.0042, radius=0.0014,
                               scale_axes=(1, 0, 1))
    for nm, xc in (("vent_L", -150.0), ("vent_R", 150.0)):
        v = MB()
        v.boxmm((xc, y0 - 3.35, H / 2), (110, 0.7, H - 10))
        parts.append(Part(nm, v, [pm]))
    # ports: two 100G-class cages, RJ45 mgmt, USB; all at x ~ -35..60
    pk = MB()  # nickel cages (mat 0), black openings (mat 1)
    for xc in (-52.0, -20.0):
        pk.boxmm((xc, y0 - 4.3, H / 2), (28, 2.0, 16), 0)
        pk.boxmm((xc, y0 - 5.4, H / 2), (23, 0.4, 7.5), 1)
    pk.boxmm((18, y0 - 4.1, H / 2), (16, 2.0, 14), 0)
    pk.boxmm((18, y0 - 5.2, H / 2), (12.5, 0.4, 10), 1)
    pk.boxmm((46, y0 - 4.1, H / 2 + 5), (14, 2.0, 7), 0)
    pk.boxmm((46, y0 - 5.2, H / 2 + 5), (11, 0.4, 3.5), 1)
    pk.boxmm((46, y0 - 4.1, H / 2 - 5), (14, 2.0, 7), 0)
    pk.boxmm((46, y0 - 5.2, H / 2 - 5), (11, 0.4, 3.5), 1)
    parts.append(Part("ports", pk, [L["nickel"], L["black"]], bevel=(mm(0.3), 1)))
    # handles (hot-swap latch) and green accent grips
    hd = MB()
    for s in (-1, 1):
        x = s * (W2 - 14)
        pts = [(x, y0 - 3.0, 8), (x, y0 - 24, 11), (x, y0 - 24, H - 11), (x, y0 - 3.0, H - 8)]
        hd.tube([tuple(a * 0.001 for a in p) for p in pts], 0.0028, 10, 0)
    parts.append(Part("handles", hd, [L["alu"]], smooth=True))
    gr = MB()
    for s in (-1, 1):
        gr.cyl((s * (W2 - 14) * 0.001, (y0 - 24) * 0.001, H / 2 * 0.001), 0.0042, (H - 24) * 0.001, "z", 12, 0)
    parts.append(Part("handle_grips", gr, [L["accent"]], smooth=True))
    # status LEDs (own objects, own material slots)
    for i, (nm, key) in enumerate((("led_power", "led_g"), ("led_status", "led_a"), ("led_id", "led_b"))):
        lm = MB()
        lm.boxmm((88 + i * 8, y0 - 3.4, H / 2 + 8), (4, 0.8, 3), 0)
        parts.append(Part(nm, lm, [L[key]]))
    # rear: NVLink blind-mate blocks, power contact, rear QDs
    rr = MB()  # mat0 black housing, mat1 gold contacts
    for s in (-1, 1):
        blindmate_block(rr, s * 110, y1 - 2, H / 2, 130, 30, 22, 0, 1, rows=3, pitch=2.5)
    pw = MB()
    pw.boxmm((0, y1 + 5, H / 2), (62, 14, 32), 0)
    pw.boxmm((-14, y1 + 13, H / 2), (20, 6, 26), 1)
    pw.boxmm((14, y1 + 13, H / 2), (20, 6, 26), 1)
    parts.append(Part("rear_nvlink", rr, [L["black"], L["gold"]]))
    parts.append(Part("rear_power", pw, [L["black"], L["copper"]], bevel=(mm(0.5), 1)))
    for s, nm, key in ((-1, "qd_supply", "qd_blue"), (1, "qd_return", "qd_red")):
        q = MB()
        qd_male(q, s * 222, y1 + 1, H / 2, 1, 0, 26.0, sign=1)
        parts.append(Part(nm, q, [L["nickel"], L[key]], smooth=True))
    rw = MB()
    rw.boxmm((0, y1 - 0.3, H / 2 - 1), (TRAY_BODY_W - 4, 0.8, H - 6), 0)
    parts.append(Part("rear_label", rw, [L["label"]]))

    if detail == "high":
        parts += _compute_interior(L)
    if lid:
        ld = MB()
        ld.boxmm((0, 0, H - 0.75), (TRAY_BODY_W, Dm - 6, 1.5))
        # vent slots hint: two raised ribs
        ld.boxmm((-80, 60, H + 0.3), (3, 500, 1.2))
        ld.boxmm((80, 60, H + 0.3), (3, 500, 1.2))
        parts.append(Part("lid", ld, [L["steel"]], bevel=(mm(0.4), 1), tag="lid"))
    return parts


def _compute_interior(L):
    P = []
    pcb = MB()
    cp = MB()   # cold plates: mat0 nickel, mat1 copper (inlet barbs), mat2 black screws
    pk = MB()   # packages: mat0 substrate (pcb), mat1 mold/black
    vr = MB()   # VRMs, inductors, caps: mat0 black, mat1 nickel, mat2 gold
    chip_y = (-150.0, -10.0, 130.0)
    xs = (-125.0, 125.0)
    for xb in xs:
        pcb.boxmm((xb, 20, 5.8), (240, 700, 1.6), 0)
        # standoffs
        for dx in (-105, 105):
            for dy in (-300, -100, 100, 300):
                pcb.boxmm((xb + dx, 20 + dy, 3.5), (5, 5, 3), 1)
        for k, yc in enumerate(chip_y):
            pk.boxmm((xb, yc, 8.4), (118, 118, 3.6), 0)       # substrate rim
            pk.boxmm((xb, yc, 12.2), (92, 92, 4.0), 1)         # lid/heatspreader (dark)
            cp.boxmm((xb, yc, 22.0), (104, 104, 14.0), 0)      # cold plate
            cp.boxmm((xb, yc, 29.4), (80, 80, 0.8), 0)
            for sx in (-1, 1):
                cp.cyl(((xb + sx * 55.5) * 0.001, yc * 0.001, 0.022), 0.0052, 0.012,
                       "x", 10, 1)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    cp.cyl(((xb + sx * 47) * 0.001, (yc + sy * 47) * 0.001, 0.0296), 0.0032, 0.0016, "z", 8, 2)
        # VRM / power stages beside chips along the board
        for yc in (-80.0, 60.0):
            for dx in (-88, 88):
                vr.boxmm((xb + dx, yc, 8.4), (22, 46, 4.5), 0)
                vr.boxmm((xb + dx, yc - 14, 11.4), (10, 10, 2.5), 1)
                vr.boxmm((xb + dx, yc + 14, 11.4), (10, 10, 2.5), 1)
        for i in range(10):
            vr.boxmm((xb - 60 + i * 13.3, 210, 8.0), (6, 4, 3.0), 2)
            vr.boxmm((xb - 60 + i * 13.3, 222, 8.0), (6, 4, 3.0), 2)
        # connector header toward the rear
        vr.boxmm((xb, 360, 8.6), (80, 12, 5.5), 0)
    P.append(Part("boards", pcb, [L["pcb"], L["dark_steel"]]))
    P.append(Part("packages", pk, [L["pcb"], L["mold"]], bevel=(mm(0.3), 1)))
    P.append(Part("cold_plates", cp, [L["nickel"], L["brass"], L["dark_steel"]], bevel=(mm(0.4), 1), smooth=False))
    P.append(Part("board_components", vr, [L["black"], L["nickel"], L["gold"]]))
    # fans (row of 6, 40 mm) behind the front panel (air-cooled parts)
    fn = MB()   # frame mat0, hub mat1
    for i in range(6):
        x = -215 + i * 86
        fn.boxmm((x, -370, 20), (40, 28, 40), 0)
        fn.cyl((x * 0.001, -355.5 * 0.001, 0.02), 0.0125, 0.0008, "y", 16, 1)
    P.append(Part("fans", fn, [L["black"], L["dark_steel"]], bevel=(mm(0.3), 1)))
    return P


def compute_tray_hoses(coll, parent, prefix, loc=(0, 0, 0)):
    """Coolant hoses (curves): rear supply QD -> board A cold plates -> cross -> board B -> rear return QD."""
    L = D.lib()
    y1 = TRAY_D / 2
    z = 22.0
    A = lambda pts: [(x * 0.001 + loc[0], y * 0.001 + loc[1], zz * 0.001 + loc[2]) for x, y, zz in pts]
    chip_y = (-150.0, -10.0, 130.0)
    out = []
    xa, xb = -125.0, 125.0
    # supply: rear left QD to A1 west barb, series through A chain
    p1 = [(-222, y1 + 12, z), (-222, y1 - 40, z), (-222, chip_y[0], z), (-196, chip_y[0], z)]
    out.append(D.curve(prefix + "_hose_supply", A(p1), 0.0045, L["hose"], coll, parent, res=8))
    for i in range(2):  # jumpers on the east side then west side alternate (simplified: east-side loops)
        yA, yB = chip_y[i], chip_y[i + 1]
        side = 1 if i == 0 else -1
        xs = xa + side * 55 + side * 28
        pts = [(xa + side * 62, yA, z), (xs, yA, z), (xs, yB, z), (xa + side * 62, yB, z)]
        out.append(D.curve("%s_hose_jumperA%d" % (prefix, i), A(pts), 0.0045, L["hose"], coll, parent, res=8))
    # cross from A3 east barb to B3 west barb
    pts = [(xa + 62, chip_y[2], z), (-30, chip_y[2], z), (30, chip_y[2], z), (xb - 62, chip_y[2], z)]
    out.append(D.curve(prefix + "_hose_cross", A(pts), 0.0045, L["hose"], coll, parent, res=8))
    for i in range(2):
        yA, yB = chip_y[2 - i], chip_y[1 - i]
        side = -1 if i == 0 else 1
        xs = xb + side * 55 + side * 28
        pts = [(xb + side * 62, yA, z), (xs, yA, z), (xs, yB, z), (xb + side * 62, yB, z)]
        out.append(D.curve("%s_hose_jumperB%d" % (prefix, i), A(pts), 0.0045, L["hose"], coll, parent, res=8))
    p2 = [(xb + 62, chip_y[0], z), (196, chip_y[0], z), (222, chip_y[0], z), (222, y1 - 40, z), (222, y1 + 12, z)]
    out.append(D.curve(prefix + "_hose_return", A(p2), 0.0045, L["hose"], coll, parent, res=8))
    return out


# ------------------------------------------------------------------ NVLink switch tray
def switch_tray_parts(detail="high", lid=True):
    L = D.lib()
    W2 = TRAY_BODY_W / 2
    Dm = SW_D
    y0, y1 = -Dm / 2, Dm / 2
    H = TRAY_H
    parts = []
    ch = MB()
    ch.boxmm((0, 0, 1.0), (TRAY_BODY_W, Dm - 6, 2.0))
    ch.boxmm((-W2 + 0.75, 0, H / 2 - 1), (1.5, Dm - 6, H - 3))
    ch.boxmm((W2 - 0.75, 0, H / 2 - 1), (1.5, Dm - 6, H - 3))
    ch.boxmm((0, y1 - 3.75, H / 2 - 1), (TRAY_BODY_W, 1.5, H - 3))
    parts.append(Part("chassis", ch, [L["steel"]], bevel=(mm(0.5), 1)))
    for s, side in ((-1, "L"), (1, "R")):
        sl = MB()
        sl.boxmm((s * (W2 + 2.0), -10, H / 2), (4.0, Dm - 80, H - 10))
        parts.append(Part("slide_inner_" + side, sl, [L["alu"]], bevel=(mm(0.4), 1), tag="rails"))
        ol = MB()
        ol.boxmm((s * (W2 + 6.0), 10, H / 2), (4.0, Dm - 50, H - 4))
        parts.append(Part("slide_outer_" + side, ol, [L["dark_steel"]], bevel=(mm(0.4), 1), tag="rails"))
    # front maintenance panel (no OSFP cages: NVLink goes to the rear copper spine)
    fb = MB()
    fb.boxmm((0, y0 - 1.5, H / 2), (TRAY_BODY_W + 12, 3.0, H - 1.5))
    parts.append(Part("front_panel", fb, [L["tray_front"]], bevel=(mm(0.6), 2)))
    pm = D.perforated_material("perforated_front", (0.09, 0.095, 0.1), 0.8, 0.4, pitch=0.0042, radius=0.0014,
                               scale_axes=(1, 0, 1))
    v = MB()
    v.boxmm((-95, y0 - 3.35, H / 2), (260, 0.7, H - 10))
    parts.append(Part("vent", v, [pm]))
    mp = MB()  # maintenance panel: label plate + mgmt RJ45 + USB-C + reset pinhole
    mp.boxmm((115, y0 - 3.2, H / 2), (90, 0.5, 30), 0)
    mp.boxmm((98, y0 - 4.4, H / 2 + 2), (16, 2.2, 14), 1)
    mp.boxmm((98, y0 - 5.4, H / 2 + 2), (12.5, 0.4, 10), 2)
    mp.boxmm((124, y0 - 4.2, H / 2 + 2), (9, 2.0, 4), 1)
    mp.boxmm((124, y0 - 5.2, H / 2 + 2), (7, 0.4, 2), 2)
    mp.boxmm((139, y0 - 3.9, H / 2 + 2), (2, 1.0, 2), 2)
    parts.append(Part("maintenance_panel", mp, [L["label"], L["nickel"], L["black"]]))
    hd = MB()
    for s in (-1, 1):
        x = s * (W2 - 14)
        pts = [(x, y0 - 3.0, 8), (x, y0 - 24, 11), (x, y0 - 24, H - 11), (x, y0 - 3.0, H - 8)]
        hd.tube([tuple(a * 0.001 for a in p) for p in pts], 0.0028, 10, 0)
    parts.append(Part("handles", hd, [L["alu"]], smooth=True))
    gr = MB()
    for s in (-1, 1):
        gr.cyl((s * (W2 - 14) * 0.001, (y0 - 24) * 0.001, H / 2 * 0.001), 0.0042, (H - 24) * 0.001, "z", 12, 0)
    parts.append(Part("handle_grips", gr, [L["accent"]], smooth=True))
    for i, (nm, key) in enumerate((("led_power", "led_g"), ("led_status", "led_a"), ("led_id", "led_b"))):
        lm = MB()
        lm.boxmm((60 + i * 8, y0 - 3.4, H / 2 + 8), (4, 0.8, 3), 0)
        parts.append(Part(nm, lm, [L[key]]))
    # rear: four copper-cartridge blind-mate blocks (two per spine bay), power, QDs
    rr = MB()
    for s in (-1, 1):
        for dx in (70, 140):
            blindmate_block(rr, s * dx, y1 - 2, H / 2, 62, 34, 22, 0, 1, rows=3, pitch=2.2)
    parts.append(Part("rear_nvlink", rr, [L["black"], L["gold"]]))
    pw = MB()
    pw.boxmm((0, y1 + 5, H / 2), (62, 14, 32), 0)
    pw.boxmm((-14, y1 + 13, H / 2), (20, 6, 26), 1)
    pw.boxmm((14, y1 + 13, H / 2), (20, 6, 26), 1)
    parts.append(Part("rear_power", pw, [L["black"], L["copper"]], bevel=(mm(0.5), 1)))
    for s, nm, key in ((-1, "qd_supply", "qd_blue"), (1, "qd_return", "qd_red")):
        q = MB()
        qd_male(q, s * 222, y1 + 1, H / 2, 1, 0, 26.0, sign=1)
        parts.append(Part(nm, q, [L["nickel"], L[key]], smooth=True))
    if detail == "high":
        pcb = MB()
        pk = MB()
        cp = MB()
        vr = MB()
        pcb.boxmm((0, 20, 5.8), (440, 600, 1.6), 0)
        for dx in (-200, 200):
            for dy in (-240, 0, 240):
                pcb.boxmm((dx, 20 + dy, 3.5), (5, 5, 3), 1)
        for xc in (-95, 95):
            pk.boxmm((xc, 40, 8.4), (86, 86, 3.6), 0)
            pk.boxmm((xc, 40, 12.0), (72, 72, 3.6), 1)
            cp.boxmm((xc, 40, 22.0), (94, 94, 14.0), 0)
            cp.boxmm((xc, 40, 29.4), (70, 70, 0.8), 0)
            for sx in (-1, 1):
                cp.cyl(((xc + sx * 50) * 0.001, 0.04, 0.022), 0.0052, 0.012, "x", 10, 1)
        for i in range(16):
            vr.boxmm((-190 + i * 25.3, 270, 8.4), (14, 14, 4.5), 0)
            vr.boxmm((-190 + i * 25.3, 245, 8.0), (6, 4, 3.0), 2)
        for i in range(8):  # retimer-like small packages near the rear blocks (generic, unlabeled)
            vr.boxmm((-170 + i * 48.0, 300, 8.0), (22, 22, 2.4), 0)
        parts.append(Part("boards", pcb, [L["pcb"], L["dark_steel"]]))
        parts.append(Part("packages", pk, [L["pcb"], L["mold"]], bevel=(mm(0.3), 1)))
        parts.append(Part("cold_plates", cp, [L["nickel"], L["brass"], L["dark_steel"]], bevel=(mm(0.4), 1)))
        parts.append(Part("board_components", vr, [L["black"], L["nickel"], L["gold"]]))
        fn = MB()
        for i in range(6):
            x = -215 + i * 86
            fn.boxmm((x, -300, 20), (40, 28, 40), 0)
            fn.cyl((x * 0.001, -285.5 * 0.001, 0.02), 0.0125, 0.0008, "y", 16, 1)
        parts.append(Part("fans", fn, [L["black"], L["dark_steel"]], bevel=(mm(0.3), 1)))
    if lid:
        ld = MB()
        ld.boxmm((0, 0, H - 0.75), (TRAY_BODY_W, Dm - 6, 1.5))
        ld.boxmm((-80, 40, H + 0.3), (3, 400, 1.2))
        ld.boxmm((80, 40, H + 0.3), (3, 400, 1.2))
        parts.append(Part("lid", ld, [L["steel"]], bevel=(mm(0.4), 1), tag="lid"))
    return parts


def switch_tray_hoses(coll, parent, prefix, loc=(0, 0, 0)):
    L = D.lib()
    y1 = SW_D / 2
    z = 22.0
    A = lambda pts: [(x * 0.001 + loc[0], y * 0.001 + loc[1], zz * 0.001 + loc[2]) for x, y, zz in pts]
    out = []
    out.append(D.curve(prefix + "_hose_supply", A([(-222, y1 + 12, z), (-222, y1 - 60, z), (-222, 40, z), (-150, 40, z)]),
                       0.0045, L["hose"], coll, parent, res=8))
    out.append(D.curve(prefix + "_hose_cross", A([(-143, 40, z), (-50, 40, z), (50, 40, z), (143, 40, z)]), 0.0045,
                       L["hose"], coll, parent, res=8))
    out.append(D.curve(prefix + "_hose_return", A([(150, 40, z), (222, 40, z), (222, y1 - 60, z), (222, y1 + 12, z)]),
                       0.0045, L["hose"], coll, parent, res=8))
    return out
