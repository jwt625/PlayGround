"""Procedural layouts + textures for the RX octal TIA die and the 1x8 PIN PD array die (workstream X).

Run from the project folder:
  uv run --with pillow --with numpy python scripts/assets/optics_rx/gen_rx_textures.py

Outputs (assets/components/optics_rx/textures/):
  tia_basecolor.png, tia_height.png, tia_rough.png, pd_basecolor.png, pd_height.png
  rx_layout.json  (die-local layout in mm; the Blender builder reads it so 3D pads/coils match the texture)

Die-local coordinates (mm):
  TIA: y in [-W/2, W/2] along the lane row (lane k centre y = (k-3.5)*0.5); s in [0, D] from the INPUT edge (PD side)
       toward the OUTPUT edge.  Image: column = y, row = s (row 0 = input edge).
  PD : y in [-2, 2] along the row; s in [0, 0.30] from the OPTICAL edge (aperture side) toward the pad edge.
Dimension sources: research/rx_optics_refs.md (PD die: datasheet grade A; TIA die: estimate grade D).
"""
import json
import math
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = sys.argv[1] if len(sys.argv) > 1 else "assets/components/optics_rx/textures"
os.makedirs(OUT, exist_ok=True)
FONT = "/System/Library/Fonts/Menlo.ttc"

# ------------------------------------------------------------------ TIA die parameters (mm)
TIA_W, TIA_D = 4.60, 1.30          # D estimate: 8 lanes x 0.50 pitch + 2 x 0.30 end zones; depth from lane layout
LANE_PITCH = 0.50                  # A: octal TIA lane pitch class (500 um, Semtech GN18xx family)
PXU = 1.6                          # pixels per micrometre
PXM = PXU * 1000.0                 # pixels per mm

# colours (sRGB) tuned to the owner's TIA micrographs (copper/orange top metal on a dark field, silver pads)
C_FIELD = (20, 24, 40)
C_FILL = (78, 52, 40)
C_METAL = (200, 112, 62)
C_COIL = (226, 134, 72)
C_UNDER = (120, 150, 165)
C_PAD = (205, 205, 198)
C_PADRING = (95, 92, 88)
C_MIM = (128, 72, 96)
C_SEAL = (190, 120, 70)
C_GUARD = (120, 96, 120)
C_POLY = (70, 100, 84)
C_TEXT = (214, 150, 90)

rng = random.Random(20261010)


def px(y, s, W, ):
    """die-local (y, s) mm -> image pixel (x, row)."""
    return ((y + W / 2.0) * PXM, s * PXM)


class Canvas:
    def __init__(self, W, D, field, h_field=70, r_field=150):
        self.W, self.D = W, D
        self.size = (int(round(W * PXM)), int(round(D * PXM)))
        self.col = Image.new("RGB", self.size, field)
        self.hgt = Image.new("L", self.size, h_field)
        self.rgh = Image.new("L", self.size, r_field)
        self.dc, self.dh, self.dr = ImageDraw.Draw(self.col), ImageDraw.Draw(self.hgt), ImageDraw.Draw(self.rgh)

    def P(self, y, s):
        return ((y + self.W / 2.0) * PXM, s * PXM)

    def rect(self, y0, s0, y1, s1, color, h=None, r=None):
        a, b = self.P(min(y0, y1), min(s0, s1)), self.P(max(y0, y1), max(s0, s1))
        box = [a[0], a[1], b[0], b[1]]
        self.dc.rectangle(box, fill=color)
        if h is not None:
            self.dh.rectangle(box, fill=h)
        if r is not None:
            self.dr.rectangle(box, fill=r)

    def poly(self, pts, color, h=None, r=None):
        q = [self.P(*p) for p in pts]
        self.dc.polygon(q, fill=color)
        if h is not None:
            self.dh.polygon(q, fill=h)
        if r is not None:
            self.dr.polygon(q, fill=r)

    def line(self, pts, w_mm, color, h=None, r=None):
        q = [self.P(*p) for p in pts]
        w = max(1, int(round(w_mm * PXM)))
        self.dc.line(q, fill=color, width=w, joint="curve")
        if h is not None:
            self.dh.line(q, fill=h, width=w, joint="curve")
        if r is not None:
            self.dr.line(q, fill=r, width=w, joint="curve")

    def circle(self, y, s, rad, color, h=None, r=None, outline=None, ow=0.0):
        a, b = self.P(y - rad, s - rad), self.P(y + rad, s + rad)
        box = [a[0], a[1], b[0], b[1]]
        kw = dict(fill=color)
        if outline is not None:
            kw.update(outline=outline, width=max(1, int(ow * PXM)))
        self.dc.ellipse(box, **kw)
        if h is not None:
            self.dh.ellipse(box, fill=h)
        if r is not None:
            self.dr.ellipse(box, fill=r)

    def arc(self, y, s, rad, w_mm, a0, a1, color, h=None):
        a, b = self.P(y - rad, s - rad), self.P(y + rad, s + rad)
        w = max(1, int(round(w_mm * PXM)))
        self.dc.arc([a[0], a[1], b[0], b[1]], a0, a1, fill=color, width=w)
        if h is not None:
            self.dh.arc([a[0], a[1], b[0], b[1]], a0, a1, fill=h, width=w)

    def text(self, y, s, body, size_mm, color, h=None, rot=0):
        f = ImageFont.truetype(FONT, max(6, int(size_mm * PXM)))
        tw, th = f.getbbox(body)[2:]
        tile = Image.new("L", (tw + 4, th + 4), 0)
        ImageDraw.Draw(tile).text((2, 0), body, font=f, fill=255)
        if rot:
            tile = tile.rotate(rot, expand=True)
        # the die-top UV map (column = +Y, row = -X) is a mirror of the world top view, so text is drawn mirrored
        # here and reads correctly on the model
        tile = ImageOps.mirror(tile)
        x0, r0 = self.P(y, s)
        x0, r0 = int(x0 - tile.size[0] / 2), int(r0 - tile.size[1] / 2)
        self.col.paste(Image.new("RGB", tile.size, color), (x0, r0), tile)
        if h is not None:
            self.hgt.paste(Image.new("L", tile.size, h), (x0, r0), tile)


def octagon_spiral(yc, sc, r_out, turns, w, sp, start_deg=22.5):
    """Octagonal spiral centreline (mm), outer end first; radius shrinks by (w+sp) per turn."""
    pts = []
    n = int(round(turns * 8))
    r0 = r_out - w / 2.0
    for i in range(n + 1):
        r = r0 - (w + sp) * i / 8.0
        a = math.radians(start_deg + 45.0 * i)
        rv = r / math.cos(math.radians(22.5))          # vertex radius so the flat-to-flat size is 2r
        pts.append((yc + rv * math.cos(a), sc + rv * math.sin(a)))
    return pts


# ================================================================== TIA layout
def build_tia():
    cv = Canvas(TIA_W, TIA_D, C_FIELD)
    W, D = TIA_W, TIA_D
    lay = dict(die=dict(W=W, D=D, lane_pitch=LANE_PITCH), pads=[], inductors=[], keepouts=[], blocks=[])

    # keep-out list first (inductors), so dummy fill can skip them
    lanes = [(k - 3.5) * LANE_PITCH for k in range(8)]
    ind_specs = []   # (name, dy, s, r_out, turns, w, sp)
    for k, yc in enumerate(lanes):
        ind_specs += [
            ("L%d_inp" % (k + 1), -0.100, 0.215, 0.036, 3.0, 0.0055, 0.0028),
            ("L%d_inn" % (k + 1), +0.100, 0.215, 0.036, 3.0, 0.0055, 0.0028),
            ("L%d_pk1a" % (k + 1), -0.120, 0.500, 0.048, 3.5, 0.0065, 0.0030),
            ("L%d_pk1b" % (k + 1), +0.120, 0.500, 0.048, 3.5, 0.0065, 0.0030),
            ("L%d_pk2a" % (k + 1), -0.120, 0.850, 0.043, 3.0, 0.0065, 0.0030),
            ("L%d_pk2b" % (k + 1), +0.120, 0.850, 0.043, 3.0, 0.0065, 0.0030),
            ("L%d_outa" % (k + 1), -0.075, 1.120, 0.033, 2.5, 0.0060, 0.0030),
            ("L%d_outb" % (k + 1), +0.075, 1.120, 0.033, 2.5, 0.0060, 0.0030),
        ]
    ind_abs = []
    for k, yc in enumerate(lanes):
        for spec in ind_specs[k * 8:(k + 1) * 8]:
            name, dy, s, r, t, w, sp = spec
            ind_abs.append((name, yc + dy, s, r, t, w, sp))
            lay["keepouts"].append((yc + dy, s, r + 0.018))

    def in_keepout(y, s, m=0.0):
        for (ky, ks, kr) in lay["keepouts"]:
            if (y - ky) ** 2 + (s - ks) ** 2 < (kr + m) ** 2:
                return True
        return False

    # ---- dummy fill (metal squares on an 8 um grid, skipped in inductor keep-outs and pad zones)
    step, sq = 0.008, 0.0032
    y = -W / 2 + 0.03
    while y < W / 2 - 0.03:
        s = 0.03
        while s < D - 0.03:
            if not in_keepout(y, s):
                cv.rect(y, s, y + sq, s + sq, C_FILL, h=95, r=120)
            s += step
        y += step

    # ---- functional dense blocks (transistors / lower metal), drawn as random rectangles over a dark base
    def dense(y0, s0, y1, s1, seed, density=1.0, palette=None):
        r = random.Random(seed)
        cv.rect(y0, s0, y1, s1, (34, 30, 38), h=80, r=140)
        pal = palette or [(110, 68, 44), (150, 88, 52), (70, 100, 84), (95, 60, 70), (60, 66, 90)]
        n = int(density * (y1 - y0) * (s1 - s0) * 42000)
        for _ in range(int(density * (s1 - s0) / 0.006)):      # metal-2 routing tracks
            b = r.uniform(s0, s1)
            a0 = r.uniform(y0, y1)
            cv.rect(a0, b, min(y1, a0 + r.uniform(0.02, 0.12)), b + 0.0016, (150, 90, 55), h=125)
        for _ in range(n):
            a = r.uniform(y0, y1)
            b = r.uniform(s0, s1)
            if r.random() < 0.5:
                ww, hh = r.uniform(0.002, 0.012), r.uniform(0.0015, 0.004)
            else:
                ww, hh = r.uniform(0.0015, 0.004), r.uniform(0.002, 0.012)
            cv.rect(a, b, min(a + ww, y1), min(b + hh, s1), r.choice(pal), h=r.randint(90, 130), r=110)
        lay["blocks"].append((y0, s0, y1, s1))

    def fingers(y0, s0, y1, s1, n, color=C_POLY):
        """transistor finger array (vertical poly fingers between metal straps)."""
        cv.rect(y0, s0, y1, s1, (40, 34, 40), h=85)
        for i in range(n):
            yy = y0 + (y1 - y0) * (i + 0.5) / n
            cv.rect(yy - 0.0012, s0 + 0.002, yy + 0.0012, s1 - 0.002, color, h=110)
        cv.rect(y0, s0, y1, s0 + 0.004, C_METAL, h=170, r=60)
        cv.rect(y0, s1 - 0.004, y1, s1, C_METAL, h=170, r=60)

    def mim(y0, s0, y1, s1, stripes=6):
        cv.rect(y0, s0, y1, s1, C_MIM, h=140, r=90)
        for i in range(stripes):
            ss = s0 + (s1 - s0) * (i + 0.5) / stripes
            cv.rect(y0 + 0.003, ss - 0.0015, y1 - 0.003, ss + 0.0015, (160, 100, 118), h=150)

    def meander(y0, s0, y1, n, w=0.0025, pitch=0.007, color=(90, 140, 120)):
        pts = []
        for i in range(n):
            yy = y0 + i * pitch
            pts += [(yy, s0), (yy, y1)] if i % 2 == 0 else [(yy, y1), (yy, s0)]
        cv.line(pts, w, color, h=120)

    def pad(y, s, size, name, kind, lane=None):
        hs = size / 2.0
        cv.rect(y - hs - 0.006, s - hs - 0.006, y + hs + 0.006, s + hs + 0.006, C_METAL, h=170, r=60)
        cv.rect(y - hs, s - hs, y + hs, s + hs, C_PADRING, h=120, r=90)       # passivation step
        cv.rect(y - hs + 0.004, s - hs + 0.004, y + hs - 0.004, s + hs - 0.004, C_PAD, h=150, r=55)
        lay["pads"].append(dict(name=name, kind=kind, y=round(y, 5), s=round(s, 5), size=size, lane=lane))

    # ---- supply rails across the lane field (top metal), with breaks at inductors
    for (s0, s1, nm) in [(0.420, 0.440, "VCC"), (0.565, 0.585, "GND"), (0.785, 0.800, "VCC"), (0.905, 0.920, "GND")]:
        cv.rect(-2.0, s0, 2.0, s1, C_METAL, h=175, r=55)

    for k, yc in enumerate(lanes):
        L = k + 1
        # lane guard (GND) lines between lanes
        for gy in (yc - 0.25, yc + 0.25):
            if -2.0 <= gy <= 2.0:
                cv.rect(gy - 0.003, 0.13, gy + 0.003, 1.17, C_GUARD, h=140, r=70)   # lower metal (passes under rails)
        # differential input pads G, IN+, IN-, G (IN+/IN- at 0.10 mm = PD P/N pad pitch; G bonds go to the PD submount
        # ground strip).  PD bias is supplied from the TIA side through IN- (on-chip RC filter below).
        pad(yc - 0.170, 0.065, 0.060, "L%d_GIa" % L, "tia_in_gnd", L)
        pad(yc - 0.050, 0.065, 0.060, "L%d_INP" % L, "tia_in_sig", L)
        pad(yc + 0.050, 0.065, 0.060, "L%d_INN" % L, "tia_in_sig", L)
        pad(yc + 0.170, 0.065, 0.060, "L%d_GIb" % L, "tia_in_gnd", L)
        for sgn in (-1, 1):
            cv.line([(yc + sgn * 0.170, 0.095), (yc + sgn * 0.170, 0.118), (yc + sgn * 0.250, 0.118),
                     (yc + sgn * 0.250, 0.135)], 0.010, C_METAL, h=170)                      # input grounds -> guards
            cv.line([(yc + sgn * 0.050, 0.095), (yc + sgn * 0.050, 0.150), (yc + sgn * 0.100, 0.172)], 0.008,
                    C_METAL, h=170)                                                         # IN+/IN- -> input coils
            cv.line([(yc + sgn * 0.100, 0.253), (yc + sgn * 0.060, 0.272)], 0.008, C_METAL, h=170)   # coils -> core
        # ESD diodes between the pair
        fingers(yc - 0.030, 0.105, yc - 0.008, 0.140, 5)
        fingers(yc + 0.008, 0.105, yc + 0.030, 0.140, 5)
        # PD-bias network: meander resistors + MIM filter caps; the +Y branch biases the PD cathode through IN-
        meander(yc - 0.212, 0.110, 0.165, 8)
        meander(yc + 0.157, 0.110, 0.165, 8)
        mim(yc - 0.235, 0.180, yc - 0.165, 0.250)
        mim(yc + 0.165, 0.180, yc + 0.235, 0.250)
        cv.line([(yc + 0.157, 0.165), (yc + 0.080, 0.163)], 0.006, C_METAL, h=170)
        # TIA core (feedback amplifier) + feedback resistor
        dense(yc - 0.110, 0.272, yc + 0.110, 0.405, seed=100 + L)
        fingers(yc - 0.060, 0.300, yc - 0.010, 0.350, 8)
        fingers(yc + 0.010, 0.300, yc + 0.060, 0.350, 8)
        meander(yc + 0.068, 0.290, 0.390, 6, color=(90, 150, 130))
        cv.line([(yc, 0.258), (yc, 0.290)], 0.008, C_METAL, h=170)
        # single-to-differential + VGA / CTLE block
        dense(yc - 0.215, 0.600, yc + 0.215, 0.770, seed=200 + L, density=0.9)
        mim(yc - 0.060, 0.640, yc + 0.060, 0.700, stripes=5)
        # output buffer (CML) big devices
        dense(yc - 0.200, 0.940, yc + 0.200, 1.065, seed=300 + L, density=0.8)
        fingers(yc - 0.150, 0.965, yc - 0.030, 1.040, 14)
        fingers(yc + 0.030, 0.965, yc + 0.150, 1.040, 14)
        # differential routes: buffer -> output inductors -> output pads (paired lines)
        for sgn in (-1, 1):
            cv.line([(yc + sgn * 0.070, 1.065), (yc + sgn * 0.075, 1.080)], 0.010, C_METAL, h=175)
            cv.line([(yc + sgn * 0.075, 1.160), (yc + sgn * 0.050, 1.200)], 0.012, C_METAL, h=175)
            # peaking-pair taps
            cv.line([(yc + sgn * 0.120, 0.550), (yc + sgn * 0.120, 0.600)], 0.008, C_METAL, h=170)
            cv.line([(yc + sgn * 0.120, 0.770), (yc + sgn * 0.120, 0.805)], 0.008, C_METAL, h=170)
            cv.line([(yc + sgn * 0.120, 0.895), (yc + sgn * 0.120, 0.940)], 0.008, C_METAL, h=170)
        cv.line([(yc, 0.405), (yc, 0.452), (yc - 0.12, 0.452)], 0.008, C_METAL, h=170)
        # output pads: G, OUTP, OUTN, G
        pad(yc - 0.190, 1.235, 0.060, "L%d_GNDa" % L, "tia_out_gnd", L)
        pad(yc - 0.050, 1.235, 0.060, "L%d_OUTP" % L, "tia_out_sig", L)     # tight differential pair
        pad(yc + 0.050, 1.235, 0.060, "L%d_OUTN" % L, "tia_out_sig", L)
        pad(yc + 0.190, 1.235, 0.060, "L%d_GNDb" % L, "tia_out_gnd", L)
        cv.line([(yc - 0.190, 1.205), (yc - 0.190, 1.170), (yc - 0.25, 1.170)], 0.010, C_METAL, h=170)
        cv.line([(yc + 0.190, 1.205), (yc + 0.190, 1.170), (yc + 0.25, 1.170)], 0.010, C_METAL, h=170)
        # lane number (top metal text)
        cv.text(yc, 0.190, "%d" % L, 0.026, C_TEXT, h=170)

    # ---- top-metal rails redrawn over the lower-metal lane guards (different layers: no VCC/GND short)
    for (s0, s1) in [(0.420, 0.440), (0.565, 0.585), (0.785, 0.800), (0.905, 0.920)]:
        cv.rect(-2.0, s0, 2.0, s1, C_METAL, h=175, r=55)

    # ---- spiral inductors (drawn last so they sit on top; underpass bridge in a lower metal)
    for (name, y, s, r, t, w, sp) in ind_abs:
        pts = octagon_spiral(y, s, r, t, w, sp)
        cv.circle(y, s, r + 0.012, C_FIELD, h=60, r=160)            # fill keep-out ring (dark)
        cv.line(pts, w, C_COIL, h=230, r=40)
        inner = pts[-1]
        outer = pts[0]
        cv.line([inner, (inner[0], s + r + 0.006)], w * 0.8, C_UNDER, h=120)   # underpass
        cv.line([outer, (outer[0] + 0.010, outer[1] - 0.004)], w, C_COIL, h=230, r=40)   # outer lead
        lay["inductors"].append(dict(name=name, y=round(y, 5), s=round(s, 5), r_out=r, turns=t, w=w, sp=sp,
                                     pts=[(round(a, 6), round(b, 6)) for a, b in pts]))

    # ---- end zones: DC / control pads, bias + I2C blocks, decap arrays, logo
    dc_lo = ["VCC", "GND", "VCC", "GND", "VPD"]               # -Y end
    dc_hi = ["GND", "SCL", "SDA", "RSTn", "GND"]              # +Y end
    s_dc = [0.42, 0.60, 0.78, 0.96, 1.14]
    for nm, s in zip(dc_lo, s_dc):
        pad(-W / 2 + 0.085, s, 0.070, "DC_lo_" + nm + "_%d" % int(s * 100), "tia_dc_" + nm, None)
    for nm, s in zip(dc_hi, s_dc):
        pad(W / 2 - 0.085, s, 0.070, "DC_hi_" + nm + "_%d" % int(s * 100), "tia_dc_" + nm, None)
    # bandgap / LDO / I2C / RSSI blocks
    dense(-2.21, 0.10, -2.02, 0.33, seed=901, density=1.1)
    dense(2.02, 0.10, 2.21, 0.33, seed=902, density=1.2)
    mim(-2.17, 0.36, -2.02, 1.20, stripes=40)                   # supply decap bank (-Y)
    mim(2.02, 0.36, 2.17, 0.62, stripes=12)                     # decap (+Y)
    dense(2.02, 0.66, 2.17, 1.20, seed=903, density=1.0)        # I2C registers / RSSI / LOS
    cv.text(2.095, 0.93, "PGN-T8 OCTAVIAN", 0.034, C_TEXT, h=180, rot=90)
    cv.text(-2.255, 0.75, "COO 2026", 0.022, C_TEXT, h=180, rot=90)
    # supply rails into the end zones
    for (s0, s1) in [(0.420, 0.440), (0.785, 0.800)]:
        cv.rect(-2.02, s0, -2.0, s1, C_METAL, h=175)
    # ---- seal ring + corner alignment marks
    for inset, wd in [(0.006, 0.006), (0.018, 0.004)]:
        cv.rect(-W / 2 + inset, inset, W / 2 - inset, inset + wd, C_SEAL, h=190, r=60)
        cv.rect(-W / 2 + inset, D - inset - wd, W / 2 - inset, D - inset, C_SEAL, h=190, r=60)
        cv.rect(-W / 2 + inset, inset, -W / 2 + inset + wd, D - inset, C_SEAL, h=190, r=60)
        cv.rect(W / 2 - inset - wd, inset, W / 2 - inset, D - inset, C_SEAL, h=190, r=60)
    for (yy, ss) in [(-W / 2 + 0.05, 0.05), (W / 2 - 0.05, 0.05), (-W / 2 + 0.05, D - 0.05), (W / 2 - 0.05, D - 0.05)]:
        cv.rect(yy - 0.018, ss - 0.003, yy + 0.018, ss + 0.003, C_SEAL, h=190)
        cv.rect(yy - 0.003, ss - 0.018, yy + 0.003, ss + 0.018, C_SEAL, h=190)
    return cv, lay


# ================================================================== PD die layout (datasheet drawing, grade A/C)
PD_W, PD_D = 4.00, 0.30
PD_PXM = 2.0 * 1000.0      # 2 px/um (die is small; keeps the 20 um aperture crisp)


def build_pd():
    global PXM
    save = PXM
    PXM = PD_PXM
    cv = Canvas(PD_W, PD_D, (74, 56, 48), h_field=80, r_field=110)     # InP, lapped/passivated look
    lay = dict(die=dict(W=PD_W, D=PD_D, pitch=0.5, aperture_d=0.020, pad_d=0.065), apertures=[], pads=[])
    au = (196, 156, 84)
    # passivation / dicing-street border
    cv.rect(-PD_W / 2, 0, PD_W / 2, 0.012, (102, 108, 128), h=60)
    cv.rect(-PD_W / 2, PD_D - 0.012, PD_W / 2, PD_D, (102, 108, 128), h=60)
    cv.rect(-PD_W / 2, 0, -PD_W / 2 + 0.03, PD_D, (102, 108, 128), h=60)
    cv.rect(PD_W / 2 - 0.03, 0, PD_W / 2, PD_D, (102, 108, 128), h=60)
    s_ap, s_pd = 0.120, 0.235      # aperture 120 um from the optical edge (A, S1); pad row (D, differential variant)
    for k in range(8):
        yc = (k - 3.5) * 0.5
        # mesa + AR-coated aperture + p-ring
        cv.circle(yc, s_ap, 0.030, (92, 72, 64), h=120)
        cv.circle(yc, s_ap, 0.017, au, h=170, r=50)                      # p-contact ring
        cv.circle(yc, s_ap, 0.010, (60, 70, 150), h=110, r=20)           # 20 um AR-coated aperture
        cv.circle(yc - 0.003, s_ap - 0.003, 0.003, (150, 160, 230), h=110)   # specular hint
        # differential-pad variant: P (anode) pad at -0.05, N (cathode) pad at +0.05 (0.10 mm pitch, 65 um pads);
        # P stem from the p-ring, N arm from an n-contact arc on the mesa edge
        cv.line([(yc - 0.006, s_ap + 0.015), (yc - 0.050, s_pd)], 0.010, au, h=170, r=50)
        cv.arc(yc, s_ap, 0.027, 0.006, 135, 450, au, h=170)          # gap where the P stem leaves (~113 deg)
        cv.line([(yc + 0.020, s_ap + 0.018), (yc + 0.050, s_pd)], 0.010, au, h=170, r=50)
        cv.circle(yc - 0.050, s_pd, 0.0325, au, h=175, r=45)
        cv.circle(yc + 0.050, s_pd, 0.0325, au, h=175, r=45)
        cv.text(yc - 0.06, 0.05, "%d" % (k + 1), 0.030, au, h=170)
        lay["apertures"].append(dict(lane=k + 1, y=yc, s=s_ap))
        lay["pads"] += [dict(name="PD%d_P" % (k + 1), kind="pd_p", y=yc - 0.050, s=s_pd, lane=k + 1),
                        dict(name="PD%d_N" % (k + 1), kind="pd_n", y=yc + 0.050, s=s_pd, lane=k + 1)]
    cv.text(-1.93, 0.15, "PGN-PD8", 0.026, au, h=170, rot=90)
    PXM = save
    return cv, lay


if __name__ == "__main__":
    tcv, tlay = build_tia()
    tcv.col.save(os.path.join(OUT, "tia_basecolor.png"))
    tcv.hgt.save(os.path.join(OUT, "tia_height.png"))
    tcv.rgh.save(os.path.join(OUT, "tia_rough.png"))
    pcv, play = build_pd()
    pcv.col.save(os.path.join(OUT, "pd_basecolor.png"))
    pcv.hgt.save(os.path.join(OUT, "pd_height.png"))
    with open(os.path.join(OUT, "rx_layout.json"), "w") as f:
        json.dump(dict(tia=tlay, pd=play, px_per_um=dict(tia=PXU, pd=PD_PXM / 1000.0)), f, indent=1)
    print("TIA", tcv.size, "pads", len(tlay["pads"]), "inductors", len(tlay["inductors"]))
    print("PD", pcv.size, "pads", len(play["pads"]))
