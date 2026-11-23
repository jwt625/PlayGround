# Redrawn schematics and model plots for the feedback course literature tabs (figures that are not CC BY).
# Every curve here comes from a stated formula (coupled-mode doublet, Lorentzian detuning, sawtooth mode selection)
# or from numbers stated in the cited text; sketches are labeled as illustrations on the figure itself.
# Run from the project root:
#   uv run --no-project --with numpy python course/tools/lit/fb_draw.py
import math
import os

import numpy as np

OUT = 'course/media/lit'
FONT = 'Helvetica, Arial, sans-serif'
INK, INK2, INK3, LINE = '#1d2733', '#4d5b6b', '#7b8796', '#d3dbe4'
BRAND, BRAND_BG, BRAND_BG_INK, TEAL = '#0b5cab', '#e4eef9', '#0b4a8a', '#0a7c8c'
BAD, WARN, OK, LASER = '#b3261e', '#9a6400', '#1f7a3a', '#c8743a'


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def T(x, y, s, size=18, fill=INK, anchor='start', weight=400, rot=None, italic=False):
    tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ''
    st = ' font-style="italic"' if italic else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}{tr}>{esc(s)}</text>')


def L(x0, y0, x1, y1, stroke=INK, w=2, dash=None, marker=None, cap='butt'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    return f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{stroke}" stroke-width="{w}" stroke-linecap="{cap}"{d}{m}/>'


def R(x, y, w, h, fill='none', stroke='none', sw=2, extra=''):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'


def P(xs, ys, stroke=BRAND, w=2.5, dash=None, fill='none', marker=None):
    pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" stroke-linejoin="round"{d}{m}/>'


def defs(*colors):
    out = []
    for c in colors:
        k = c.lstrip('#')
        out.append(f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    return '<defs>' + ''.join(out) + '</defs>'


def arrow(c):
    return 'a' + c.lstrip('#')


def svg(name, w, h, body, bg='#ffffff', colors=(INK, BAD, BRAND, TEAL, WARN, OK)):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">'
         + defs(*colors) + (R(0, 0, w, h, fill=bg) if bg else '') + ''.join(body) + '</svg>\n')
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(s)
    print(name, len(s))


class Axes:
    """Linear data-to-pixel mapping inside a plot box."""
    def __init__(self, x0, y0, x1, y1, xr, yr):
        self.b = (x0, y0, x1, y1); self.xr = xr; self.yr = yr

    def X(self, v):
        x0, _, x1, _ = self.b; a, b = self.xr
        return x0 + (np.asarray(v, float) - a) / (b - a) * (x1 - x0)

    def Y(self, v):
        _, y0, _, y1 = self.b; a, b = self.yr
        return y1 - (np.asarray(v, float) - a) / (b - a) * (y1 - y0)

    def frame(self, xlabel, ylabel, xticks=(), yticks=(), size=17, fmt='{:g}'):
        x0, y0, x1, y1 = self.b; o = [L(x0, y1, x1, y1, INK, 1.5), L(x0, y0, x0, y1, INK, 1.5)]
        for t in xticks:
            x = float(self.X(t)); o += [L(x, y1, x, y1 + 6, INK, 1.5), T(x, y1 + 6 + size, fmt.format(t), size - 2, INK2, 'middle')]
        for t in yticks:
            y = float(self.Y(t)); o += [L(x0 - 6, y, x0, y, INK, 1.5), T(x0 - 10, y + 6, fmt.format(t), size - 2, INK2, 'end')]
        o.append(T((x0 + x1) / 2, y1 + (2.9 * size if xticks else 1.5 * size), xlabel, size, INK, 'middle'))
        o.append(T(x0 - (2.7 * size if yticks else 0.8 * size), (y0 + y1) / 2, ylabel, size, INK, 'middle', rot=-90))
        return o


def lor(x, x0, w):
    return 1.0 / (1.0 + (2 * (x - x0) / w) ** 2)


# Canvas: 520 px wide so that 18-22 px text stays legible when the figure is shown about 300 px wide on a phone.
FS, FM, FL, FT = 16, 18, 20, 22   # footnote, label, emphasis, title sizes


# ---------------------------------------------------------------------------------------------
# 1-5. Tkach and Chraplyvy (1986): five feedback regimes. Order from the paper; spectra are sketches.
ROMAN = ['I', 'II', 'III', 'IV', 'V']
DESC = ['Line narrows or broadens with phase', 'Hops between external-cavity modes', 'One external-cavity mode, narrow line',
        'Coherence collapse: broad and noisy', 'Stable external-cavity laser']


def regime(k):
    W, H = 520, 420
    o = [T(20, 30, 'Feedback level', FL, INK, weight=700), L(180, 24, 500, 24, INK2, 2, marker=arrow(INK))]
    bw = 96
    for i in range(5):
        x = 20 + i * bw; act = i == k
        o.append(R(x + 1, 44, bw - 2, 44, fill=BRAND if act else BRAND_BG, stroke=BRAND if act else LINE, sw=1.5))
        o.append(T(x + bw / 2, 74, ROMAN[i], FT, '#ffffff' if act else BRAND_BG_INK, 'middle', 700))
    o.append(T(20, 148, DESC[k], FL, INK, weight=700))
    ax = Axes(46, 170, 500, 370, (-1, 1), (0, 1.14))
    o += ax.frame('Optical frequency', 'Power', size=FM)
    f = np.linspace(-1, 1, 801)
    if k in (1, 2):
        for e in np.arange(-0.84, 0.85, 0.12):
            o.append(L(float(ax.X(e)), float(ax.Y(0)), float(ax.X(e)), float(ax.Y(0.98)), LINE, 1.4, dash='3 4'))
        o.append(T(54, 188, 'dotted: cavity modes', FS, INK3))
    if k == 0:
        o.append(P(ax.X(f), ax.Y(lor(f, 0, 0.30)), TEAL, 3, dash='7 5'))
        o.append(P(ax.X(f), ax.Y(lor(f, 0, 0.08)), BRAND, 3))
        o += [T(float(ax.X(0.08)), float(ax.Y(0.9)), 'narrower', FM, BRAND), T(float(ax.X(0.24)), float(ax.Y(0.42)), 'broader', FM, TEAL)]
    elif k == 1:
        o.append(P(ax.X(f), ax.Y(lor(f, -0.12, 0.05)), BRAND, 3))
        o.append(P(ax.X(f), ax.Y(0.8 * lor(f, 0.12, 0.05)), BAD, 3, dash='6 4'))
        o.append(f'<path d="M{float(ax.X(-0.1)):.1f},{float(ax.Y(1.0)):.1f} Q{float(ax.X(0.0)):.1f},{float(ax.Y(1.18)):.1f} '
                 f'{float(ax.X(0.1)):.1f},{float(ax.Y(0.86)):.1f}" fill="none" stroke="{BAD}" stroke-width="2.5" marker-end="url(#{arrow(BAD)})"/>')
        o.append(T(float(ax.X(0.17)), float(ax.Y(0.9)), 'hop', FL, BAD, weight=700))
    elif k == 2:
        o.append(P(ax.X(f), ax.Y(lor(f, 0.0, 0.025)), BRAND, 3))
    elif k == 3:
        rng = np.random.default_rng(1550)
        base = np.exp(-(f / 0.42) ** 2) * 0.55
        noise = np.convolve(rng.normal(0, 1, f.size), np.ones(7) / 7, 'same')
        o.append(P(ax.X(f), ax.Y(np.clip(base * (1 + 0.45 * noise) + 0.02, 0, None)), BAD, 2.2))
    elif k == 4:
        o.append(P(ax.X(f), ax.Y(lor(f, -0.55, 0.025)), BRAND, 3))
        x0, y0 = 230.0, 236.0
        o += [R(x0, y0, 110, 40, fill='#f3d9c6', stroke=LASER, sw=2), T(x0 + 55, y0 + 27, 'gain chip', FS, INK, 'middle'),
              L(x0 + 110, y0 - 6, x0 + 110, y0 + 46, TEAL, 5), T(x0 + 110, y0 + 70, 'AR coating', FS, TEAL, 'middle'),
              L(x0 + 122, y0 + 20, x0 + 196, y0 + 20, BRAND, 2.5, marker=arrow(BRAND)),
              R(x0 + 202, y0 - 10, 10, 60, fill=INK2), T(x0 + 207, y0 + 92, 'reflector', FS, INK2, 'middle')]
    svg(f'fb_tkach1986_regime{k + 1}.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Marinins et al. (2022): far-end grating coupler feedback. Dark full-bleed card for a split slide (object-fit: cover).
# All content inside the central square (x 300-1300) so a 1:1 desktop crop and the 16:10 phone frame both keep it.
def marinins():
    W, H = 1600, 1000
    o = ['<linearGradient id="g" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#0a1628"/><stop offset="1" stop-color="#14365a"/></linearGradient>',
         R(0, 0, W, H, fill='url(#g)')]
    for gx in range(0, W, 80):
        o.append(L(gx, 0, gx, H, 'rgba(255,255,255,0.05)', 1))
    for gy in range(0, H, 80):
        o.append(L(0, gy, W, gy, 'rgba(255,255,255,0.05)', 1))
    wh, mu, cy, red, amb = '#e8eef5', '#9fb3c8', '#5fd0e0', '#ff7a6e', '#ffc15a'
    y = 470
    o += [T(800, 170, 'The far end was the mirror', 66, wh, 'middle', 700),
          T(800, 230, 'After Marinins et al., IEEE JSTQE 2022', 40, mu, 'middle')]
    o += [R(320, y - 70, 210, 140, fill='#7a4a2a', stroke=amb, sw=4), T(425, y - 6, 'InP DFB', 48, wh, 'middle', 700),
          T(425, y + 46, 'laser', 42, wh, 'middle')]
    o += [L(548, y - 86, 548, y + 76, mu, 3, dash='8 8'), T(548, y - 104, 'gap < 1 µm', 44, mu, 'middle')]
    o += [L(566, y, 1140, y, cy, 14, cap='round'), T(860, y - 36, 'SiN waveguide', 44, cy, 'middle')]
    for i in range(9):
        x = 1150 + i * 15
        o.append(L(x, y - 52 - i * 4, x, y + 52 + i * 4, cy, 7))
    o.append(T(1210, y - 104, 'grating', 44, cy, 'middle'))
    o.append(L(700, y, 1000, y, '#ffffff', 5, marker=arrow('#ffffff')))
    o += [L(1160, y + 110, 552, y + 110, red, 8, dash='22 12', marker=arrow(red)),
          T(860, y + 178, '4 percent (-14 dB) returns', 52, red, 'middle', 700)]
    by = y + 260
    o += [L(540, by, 1210, by, amb, 4), L(540, by - 20, 540, by + 20, amb, 4), L(1210, by - 20, 1210, by + 20, amb, 4),
          T(875, by + 64, 'This length fits the 25 to 30 mA', 46, amb, 'middle'),
          T(875, by + 120, 'ripple period. The gap does not.', 46, amb, 'middle')]
    svg('fb_marinins2022_far_end.svg', W, H, o, bg=None, colors=('#ffffff', red))


# ---------------------------------------------------------------------------------------------
# Celo et al. (2017) and Tian et al. (2025): crossings and splitter trees reflect.
def crossing_tree():
    W, H = 520, 400
    o = [T(125, 30, 'Crossing', FL, INK, 'middle', 700), T(390, 30, 'Splitter tree', FL, INK, 'middle', 700)]
    cx, cy = 125, 200
    o += [L(14, cy, 236, cy, TEAL, 10), L(cx, 60, cx, 330, TEAL, 10), R(cx - 22, cy - 22, 44, 44, fill=TEAL)]
    o += [L(20, cy - 26, 86, cy - 26, BRAND, 3, marker=arrow(BRAND)), T(20, cy - 38, 'in', FM, BRAND)]
    o += [L(92, cy + 28, 22, cy + 28, BAD, 3, marker=arrow(BAD)), T(14, cy + 58, 'reflection', FM, BAD)]
    o += [L(cx + 26, cy - 30, cx + 26, cy - 100, WARN, 2.5, dash='5 4', marker=arrow(WARN)), T(cx + 34, cy - 80, 'crosstalk', FM, WARN)]
    o.append(T(125, 372, 'cavities, multipath', FS, INK2, 'middle'))
    xs = [280, 345, 410, 475]
    ys = {0: [200], 1: [130, 270], 2: [95, 165, 235, 305], 3: [78, 112, 148, 182, 218, 252, 288, 322]}
    for lv in range(3):
        for i, yy in enumerate(ys[lv]):
            for c in (ys[lv + 1][2 * i], ys[lv + 1][2 * i + 1]):
                o.append(f'<path d="M{xs[lv]},{yy} C{xs[lv] + 32},{yy} {xs[lv + 1] - 32},{c} {xs[lv + 1]},{c}" fill="none" stroke="{TEAL}" stroke-width="4"/>')
            o.append(L(xs[lv] - 2, yy - 10, xs[lv] - 28, yy - 10, BAD, 2.5, marker=arrow(BAD)))
    for yy in ys[3]:
        o.append(L(475, yy, 510, yy, TEAL, 4))
    o.append(L(262, 200, 280, 200, TEAL, 4))
    o += [T(390, 362, 'one reflection per junction:', FS, INK2, 'middle'), T(390, 384, 'more levels, more returns', FS, INK2, 'middle')]
    svg('fb_crossings_splitters.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Cherchi et al. (2014): double-MMI with end-wall back-reflections; ripple only with enough modes in simulation.
def mmi():
    W, H = 520, 400
    o = [T(260, 30, 'Double MMI splitter', FT, INK, 'middle', 700)]
    y = 104
    for x in (90, 300):
        o += [R(x, y - 40, 130, 80, fill='#d6eef1', stroke=TEAL, sw=2), L(x, y - 40, x, y + 40, BAD, 6), L(x + 130, y - 40, x + 130, y + 40, BAD, 6)]
    for x0, x1 in ((14, 90), (220, 300), (430, 506)):
        o += [L(x0, y - 20, x1, y - 20, TEAL, 6), L(x0, y + 20, x1, y + 20, TEAL, 6)]
    o += [L(130, y + 58, 76, y + 58, BAD, 3, marker=arrow(BAD)), L(340, y + 58, 286, y + 58, BAD, 3, marker=arrow(BAD)),
          T(260, y + 92, 'End walls reflect light back', FM, BAD, 'middle', 700)]
    o.append(T(260, 236, 'Simulated transmission', FS, INK3, 'middle', italic=True))
    f = np.linspace(0, 1, 600)
    smooth = 0.5 + 0.18 * np.cos(2 * np.pi * (f - 0.5) * 0.9)
    ripple = smooth + 0.035 * np.sin(2 * np.pi * 14 * f)
    for i, (lab, yv, col) in enumerate([('few modes: smooth', smooth, INK2), ('over 30 modes: ripple', ripple, BRAND)]):
        ax = Axes(20 + i * 255, 252, 245 + i * 255, 340, (0, 1), (0.25, 0.75))
        o += [L(ax.b[0], ax.b[3], ax.b[2], ax.b[3], INK, 1.2), L(ax.b[0], ax.b[1], ax.b[0], ax.b[3], INK, 1.2)]
        o.append(P(ax.X(f), ax.Y(yv), col, 2.5))
        o.append(T((ax.b[0] + ax.b[2]) / 2, 370, lab, FM, col, 'middle'))
    o.append(T(260, 394, 'versus wavelength', FS, INK3, 'middle'))
    svg('fb_cherchi2014_mmi_walls.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Van Vaerenbergh et al. (2021): perfectly vertical vs tilted grating coupler.
def vgc():
    W, H = 520, 360
    o = []
    for i, (title, tilt, rw, lab) in enumerate([('Vertical fiber', 0, 6, 'large reflection'), ('Tilted fiber', 10, 2.5, 'small reflection')]):
        ox = 10 + i * 260; y = 236
        o.append(T(ox + 120, 30, title, FL, INK, 'middle', 700))
        o.append(R(ox, y, 240, 28, fill='#d6eef1'))
        for k in range(8):
            o.append(R(ox + 108 + k * 15, y - 10, 8, 10, fill=TEAL))
        cxf = ox + 168
        o.append(f'<g transform="rotate({tilt} {cxf} {y - 18})">' + R(cxf - 26, y - 168, 52, 150, fill='#eef2f6', stroke=INK3, sw=1.5)
                 + R(cxf - 6, y - 168, 12, 150, fill='#cfe2f5') + '</g>')
        o.append(L(ox + 10, y + 14, ox + 96, y + 14, BRAND, 3, marker=arrow(BRAND)))
        o.append(T(ox + 6, y - 14, 'from chip', FS, BRAND))
        o.append(L(ox + 96, y + 48, ox + 10, y + 48, BAD, rw, marker=arrow(BAD)))
        o.append(T(ox + 120, y + 96, lab, FL, BAD, 'middle', 700))
    svg('fb_vaerenbergh2021_vertical_gc.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Contreras-Teran et al. (2023): core-offset Mach-Zehnder from three spliced single-mode fiber segments.
def offset_splice():
    W, H = 520, 320
    o = [T(260, 30, 'Core-offset fiber interferometer', FT, INK, 'middle', 700)]
    y = 150; d = 18
    for x0, x1, off in [(10, 160, 0), (160, 360, d), (360, 510, 0)]:
        o.append(R(x0, y - 52 + off, x1 - x0, 104, fill='#eef2f6', stroke=INK3, sw=1.5))
        o.append(R(x0, y - 7 + off, x1 - x0, 14, fill='#cfe2f5'))
    o += [L(160, y - 74, 160, y + 84, INK2, 1.5, dash='4 4'), L(360, y - 74, 360, y + 84, INK2, 1.5, dash='4 4'),
          T(160, y + 106, 'offset splice', FS, INK2, 'middle'), T(360, y + 106, 'offset splice', FS, INK2, 'middle')]
    o += [P([10, 160], [y, y], BRAND, 3.5), P([160, 360], [y + d, y + d], BRAND, 3.5), P([360, 510], [y, y], BRAND, 3.5, marker=arrow(BRAND))]
    o.append(f'<path d="M160,{y} C210,{y - 34} 310,{y - 34} 360,{y}" fill="none" stroke="{WARN}" stroke-width="3" stroke-dasharray="6 4"/>')
    o.append(f'<path d="M160,{y + 4} C210,{y + 74} 310,{y + 74} 360,{y + 4}" fill="none" stroke="{WARN}" stroke-width="3" stroke-dasharray="6 4"/>')
    o += [T(260, y - 38, 'cladding', FM, WARN, 'middle'), T(260, y + 8, 'core', FM, BRAND, 'middle'),
          T(260, 300, 'Two paths recombine: an interferometer', FM, INK, 'middle')]
    svg('fb_contreras2023_offset_splice.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# van Rees et al. (2020): 0.22 nm (28 GHz) mode-hop-free tuning, six times the cavity FSR.
# Model: with a fixed filter the lasing mode is the comb line nearest the filter peak (sawtooth, a hop every FSR);
# with the filter tuned in step with the cavity phase the same line stays selected.
def vanrees():
    W, H = 520, 400
    FSR = 28.0 / 6
    o = [T(270, 30, 'Mode-hop-free span', FT, INK, 'middle', 700)]
    ax = Axes(80, 54, 500, 304, (0, 6), (-4, 30))
    o += ax.frame('Cavity tuning (free spectral ranges)', 'Frequency change (GHz)', xticks=range(7), yticks=(0, 10, 20, 28), size=FM)
    x = np.linspace(0, 6, 1200)
    saw = FSR * (np.mod(x + 0.5, 1.0) - 0.5)
    seg, segs = [], []
    for xi, si in zip(x, saw):
        if seg and si < seg[-1][1] - FSR / 2:
            segs.append(seg); seg = []
        seg.append((xi, si))
    segs.append(seg)
    for sg in segs:
        o.append(P(ax.X([q[0] for q in sg]), ax.Y([q[1] for q in sg]), INK2, 2.5))
    o.append(P(ax.X(x), ax.Y(FSR * x), BRAND, 3.5))
    o += [T(float(ax.X(0.15)), float(ax.Y(26)), 'filter tracks the cavity:', FM, BRAND),
          T(float(ax.X(0.15)), float(ax.Y(22.5)), '28 GHz (0.22 nm), no hop', FM, BRAND, weight=700),
          T(float(ax.X(5.9)), float(ax.Y(5.6)), 'fixed filter: hops', FM, INK2, 'end')]
    svg('fb_vanrees2020_mhf_tuning.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Kashyap and Blow (1988): fiber fuse travels back toward the source.
def fuse():
    W, H = 520, 260
    o = [T(260, 30, 'Fiber fuse', FT, INK, 'middle', 700)]
    y = 130; xf = 300
    o += [R(10, y - 8, 40, 16, fill=LASER), T(30, y - 46, 'source', FM, INK2, 'middle'),
          R(50, y - 34, 460, 68, fill='#eef2f6', stroke=INK3, sw=1.5), R(50, y - 7, 460, 14, fill='#cfe2f5')]
    o += [L(80, y - 52, 240, y - 52, BRAND, 3, marker=arrow(BRAND)), T(250, y - 46, 'laser light', FM, BRAND)]
    rng = np.random.default_rng(1988)
    x = xf + 22
    while x < 500:
        w = 6 + 4 * rng.random()
        o.append(f'<ellipse cx="{x:.1f}" cy="{y}" rx="{w:.1f}" ry="5" fill="#5a3b2a"/>')
        x += 14 + 6 * rng.random()
    o.append('<radialGradient id="fz"><stop offset="0" stop-color="#ffffff"/><stop offset="0.35" stop-color="#ffd27a"/>'
             '<stop offset="1" stop-color="#ff8a3d" stop-opacity="0"/></radialGradient>')
    o.append(f'<ellipse cx="{xf}" cy="{y}" rx="34" ry="26" fill="url(#fz)"/>')
    o += [L(xf - 30, y + 56, xf - 170, y + 56, BAD, 3.5, marker=arrow(BAD)),
          T(xf - 100, y + 92, 'moves toward the source', FL, BAD, 'middle', 700),
          T(xf + 110, y + 62, 'damaged core', FM, INK2, 'middle')]
    svg('fb_kashyap1988_fiber_fuse.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Coupled-mode doublet (Kippenberg et al. 2002): T = |1 - (kex/2)/(i(d-g)+k/2) - (kex/2)/(i(d+g)+k/2)|^2, k = k0 + kex.
def doublet_T(d, k0, kex, g):
    k = k0 + kex
    return np.abs(1 - (kex / 2) / (1j * (d - g) + k / 2) - (kex / 2) / (1j * (d + g) + k / 2)) ** 2


def fit_single(d, t):
    """Least-squares single Lorentzian dip centred at zero: width by grid search, depth in closed form."""
    best = None
    for w in np.linspace(0.3, 8, 771):
        base = lor(d, 0.0, w)
        A = float(np.dot(1 - t, base) / np.dot(base, base))
        err = float(np.sum((1 - A * base - t) ** 2))
        if best is None or err < best[0]:
            best = (err, w, A)
    return best[1], best[2]


def doublet_fit():
    W, H = 520, 420
    k0, kex, g = 1.0, 0.6, 1.2
    d = np.linspace(-6, 6, 1201)
    t = doublet_T(d, k0, kex, g)
    w, A = fit_single(d, t)
    ratio = w / (k0 + kex)
    o = [T(270, 30, 'One Lorentzian fitted to a doublet', FT, INK, 'middle', 700)]
    ax = Axes(76, 54, 500, 270, (-6, 6), (0, 1.05))
    o += ax.frame('Detuning (intrinsic linewidths)', 'Transmission', xticks=(-6, -3, 0, 3, 6), yticks=(0, 0.5, 1), size=FM)
    o.append(P(ax.X(d), ax.Y(t), BRAND, 3.5))
    o.append(P(ax.X(d), ax.Y(1 - A * lor(d, 0, w)), BAD, 3, dash='8 5'))
    o += [L(40, 358, 72, 358, BRAND, 3.5), T(80, 364, 'doublet model', FM, BRAND),
          L(270, 358, 302, 358, BAD, 3, dash='8 5'), T(310, 364, 'single Lorentzian', FM, BAD),
          T(260, 402, f'Fit {ratio:.1f}\u00d7 too wide: loaded Q {ratio:.1f}\u00d7 low at this splitting', FM, INK, 'middle', 700)]
    svg('fb_doublet_single_fit.svg', W, H, o)
    return ratio


# ---------------------------------------------------------------------------------------------
# Hausmann et al. (2014): segmented e-beam writing -> periodic scatterers -> split resonances; continuous writing -> none.
def stitch():
    W, H = 520, 400
    o = [T(260, 30, 'E-beam writing of a ring', FT, INK, 'middle', 700)]
    d = np.linspace(-6, 6, 801)
    for row, (lab, seg, g, l2, col) in enumerate([('Segmented', True, 1.4, 'joints scatter: split', BAD),
                                                  ('Continuous', False, 0.0, 'no joints: one dip', OK)]):
        cy = 120 + row * 160; cx = 66
        if seg:
            for sgi in range(8):
                a0 = sgi * 45 + 3; a1 = sgi * 45 + 42
                r = 46 + (3 if sgi % 2 else -3)
                p0 = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
                p1 = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
                o.append(f'<path d="M{p0[0]:.1f},{p0[1]:.1f} A{r},{r} 0 0 1 {p1[0]:.1f},{p1[1]:.1f}" fill="none" stroke="{TEAL}" stroke-width="9"/>')
                pj = (cx + 46 * math.cos(math.radians(a1 + 1.5)), cy + 46 * math.sin(math.radians(a1 + 1.5)))
                o.append(f'<circle cx="{pj[0]:.1f}" cy="{pj[1]:.1f}" r="5" fill="{BAD}"/>')
        else:
            o.append(f'<circle cx="{cx}" cy="{cy}" r="46" fill="none" stroke="{TEAL}" stroke-width="9"/>')
        o.append(T(130, cy - 8, lab, FL, INK, weight=700))
        o.append(T(130, cy + 20, l2, FM, col))
        ax = Axes(340, cy - 50, 506, cy + 40, (-6, 6), (0, 1.05))
        o += [L(ax.b[0], ax.b[3], ax.b[2], ax.b[3], INK, 1.2), L(ax.b[0], ax.b[1], ax.b[0], ax.b[3], INK, 1.2)]
        o.append(P(ax.X(d), ax.Y(doublet_T(d, 1.0, 0.8, g)), BRAND, 3))
    o.append(T(423, 384, 'transmission', FS, INK3, 'middle'))
    svg('fb_hausmann2014_stitching.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# White et al. (2024): back-reflection reaching the laser vs input power. Back-scattered power ~ P, attenuated by the
# Lorentzian of the Kerr-detuned counter-propagating mode, I = 1 / (1 + (2 Q dw / w0)^2) with dw proportional to P
# (their Eqs. 1-2, arXiv:2404.03093). Normalized x = 2 Q dw / w0. Right edge: 15 dB attenuation, the maximum reported.
def white():
    W, H = 520, 400
    xmax = math.sqrt(10 ** 1.5 - 1)
    x = np.linspace(0, xmax, 600)
    pr = x / (1 + x ** 2)
    o = [T(270, 30, 'Back-reflection reaching the laser', FT, INK, 'middle', 700)]
    ax = Axes(56, 54, 500, 318, (0, xmax), (0, 1.6))
    o += ax.frame('Input power (normalized)', 'Back-reflection (normalized)', size=FM)
    xl = x[x <= 1.55]
    o.append(P(ax.X(xl), ax.Y(xl), INK3, 2.5, dash='7 5'))
    o.append(P(ax.X(x), ax.Y(pr), BRAND, 3.5))
    o += [T(float(ax.X(1.62)), float(ax.Y(1.48)), 'linear: constant ratio', FM, INK2),
          T(float(ax.X(1.1)), float(ax.Y(0.62)), 'Kerr detuning: less returns', FM, BRAND),
          T(float(ax.X(xmax)), float(ax.Y(0.42)), '15 dB below linear', FM, BRAND, 'end', 700)]
    svg('fb_white2024_backreflection.svg', W, H, o)


# ---------------------------------------------------------------------------------------------
# Chowdhury et al. (2024): free spectral range of the same rings, values stated in the text.
def chowdhury():
    W, H = 520, 300
    vals = [('Simulated', 110, INK3), ('Benchtop laser', 120, TEAL), ('On-chip laser', 160, BAD)]
    o = [T(260, 30, 'FSR of the same rings', FT, INK, 'middle', 700)]
    ax = Axes(150, 60, 430, 230, (0, 160), (0, 3))
    for i, (lab, v, c) in enumerate(vals):
        y0 = 62 + i * 56
        o.append(R(150, y0, float(ax.X(v)) - 150, 38, fill=c))
        o.append(T(140, y0 + 26, lab, FM, INK, 'end'))
        o.append(T(float(ax.X(v)) + 8, y0 + 26, f'{v} pm', FL, INK, weight=700))
    o += [L(150, 54, 150, 236, INK, 1.5), T(260, 276, 'The on-chip laser heated the rings', FM, INK2, 'middle')]
    svg('fb_chowdhury2024_fsr.svg', W, H, o)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for k in range(5):
        regime(k)
    marinins(); crossing_tree(); mmi(); vgc(); offset_splice(); vanrees(); fuse()
    print('doublet width ratio', round(doublet_fit(), 3))
    stitch(); white(); chowdhury()
