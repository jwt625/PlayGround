# Figures reproduced from scratch for the ferroelectric course literature tabs (sources not CC BY).
# Every number drawn here is either stated in the cited paper's text, quoted in the course text with its
# source, or computed from a stated textbook model; illustrative parameters are labelled as such in the figure.
# Run from the project root:
#   uv run --no-project --with matplotlib==3.10.0 --with numpy python course/tools/lit/fe_figs.py
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = 'course/media/lit'
INK, INK2, INK3, LINE = '#1d2733', '#4d5b6b', '#7b8796', '#d3dbe4'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

plt.rcParams.update({
    'svg.fonttype': 'none', 'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 13, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, format='svg', bbox_inches='tight', pad_inches=0.12, metadata={'Date': None})
    plt.close(fig)
    print(name, os.path.getsize(p) // 1024, 'KB')


def write(name, svg):
    p = os.path.join(OUT, name)
    with open(p, 'w') as f:
        f.write(svg)
    print(name, os.path.getsize(p) // 1024, 'KB')


# ---------- Valdez et al. 2023: quadrature set by laser wavelength on an unbalanced MZI ----------
# T = (1 + cos(2 pi (lambda - lambda0) / FSR)) / 2, FSR illustrative (the paper's FSR is not used).
def valdez():
    fsr = 2.0
    lam = np.linspace(1549.0, 1553.0, 800)
    lam0 = 1550.0
    T = 0.5 * (1 + np.cos(2 * np.pi * (lam - lam0) / fsr))
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.plot(lam, T, color=BLUE, lw=2)
    q = lam0 + fsr / 4
    ax.plot([q, q], [0, 1.5], color=ORANGE, lw=1.5, ls='--')
    ax.plot([q], [0.5], 'o', ms=9, color=ORANGE, mec='white', mew=2, zorder=5)
    ax.text(q + 0.07, 1.37, 'laser tuned to quadrature\napplied DC bias: 0 V', ha='left', va='center',
            fontsize=12.5, color=INK)
    ax.annotate('', xy=(lam0 + fsr, 1.04), xytext=(lam0, 1.04),
                arrowprops=dict(arrowstyle='<->', color=INK2, lw=1))
    ax.text(lam0 + fsr / 2, 1.07, 'FSR 2 nm', ha='center', va='bottom', fontsize=11.5, color=INK2)
    ax.set_xlim(1549, 1553)
    ax.set_xticks([1549, 1550, 1551, 1552, 1553])
    ax.set_ylim(0, 1.55)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('Laser wavelength (nm)')
    ax.set_ylabel('Transmission')
    ax.grid(axis='y', color=LINE, lw=0.8)
    save(fig, 'fe_valdez2023_quadrature.svg')


# ---------- Which V pi: electrical dB of the EO response vs drive needed ----------
# EO S21 (electrical dB) = 20 log10(m/m0); same phase needs V x 10^(-S/20) and power x 10^(-S/10).
def vpi_db():
    s = np.linspace(0, -8, 200)
    v = 10 ** (-s / 20)
    p = 10 ** (-s / 10)
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(s, p, color=ORANGE, lw=2)
    ax.plot(s, v, color=BLUE, lw=2)
    for x in (-3, -6):
        ax.axvline(x, color=LINE, lw=1, zorder=0)
    pts = [(-3, 10 ** 0.15, 'x1.41', (10, -14)), (-3, 2.0, 'x2', (-8, 10)),
           (-6, 2.0, 'x2: V$_\\pi$ doubled', (10, -16)), (-6, 10 ** 0.6, 'x4', (-8, 10))]
    for x, y, t, off in pts:
        c = BLUE if t.startswith('x1') or 'doubled' in t else ORANGE
        ax.plot([x], [y], 'o', ms=8, color=c, mec='white', mew=2, zorder=5)
        ax.annotate(t, (x, y), textcoords='offset points', xytext=off, fontsize=12.5, color=INK,
                    ha='left' if off[0] > 0 else 'right', va='center')
    ax.annotate('drive power', (-8, 10 ** 0.8), textcoords='offset points', xytext=(-4, 12), ha='right',
                fontsize=12.5, color=INK)
    ax.annotate('drive voltage', (-8, 10 ** 0.4), textcoords='offset points', xytext=(-4, 13), ha='right',
                fontsize=12.5, color=INK)
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(['1', '2', '4', '8'])
    ax.minorticks_off()
    ax.set_xlim(0.2, -8.4)
    ax.set_ylim(0.9, 9)
    ax.set_xticks([0, -2, -3, -4, -6, -8])
    ax.set_xlabel('EO response S$_{21}$ (electrical dB)')
    ax.set_ylabel('Multiple for the same phase')
    save(fig, 'fe_vpi_electrical_db.svg')


# ---------- Weis and Gaylord 1985: clamped vs unclamped r33 at 633 nm (ranges quoted in the tab) ----------
def weis():
    rows = [('Unclamped (constant stress)\nDC and low frequency', 32.2, 34.0, ORANGE),
            ('Clamped (constant strain)\nRF', 28.8, 30.8, BLUE)]
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    for i, (lab, lo, hi, c) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=c, lw=9, solid_capstyle='butt')
        ax.text(lo - 0.25, i, f'{lo:.1f}', ha='right', va='center', fontsize=12, color=INK)
        ax.text(hi + 0.25, i, f'{hi:.1f}', ha='left', va='center', fontsize=12, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(26.5, 36)
    ax.set_xlabel('r$_{33}$ at 633 nm (pm/V)')
    ax.grid(axis='x', color=LINE, lw=0.8)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    save(fig, 'fe_weis1985_r33.svg')


# ---------- Liang et al. 2017: sequence described in the text, plus a generic nanobeam sketch ----------
def liang():
    W, H = 600, 410
    fs = 15
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}">', f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    # nanobeam sketch (generic, not to scale): holes shrink toward the centre
    y0 = 40
    s.append(f'<rect x="20" y="{y0 - 17}" width="560" height="34" fill="#e4eef9" stroke="{INK2}" stroke-width="1"/>')
    for k in range(-13, 14):
        if k == 0:
            continue
        x = 300 + k * 20.5
        r = 10.5 - 4.0 * np.exp(-(k / 4.5) ** 2)
        s.append(f'<ellipse cx="{x:.1f}" cy="{y0}" rx="{r * 0.55:.1f}" ry="{r:.1f}" fill="#ffffff" '
                 f'stroke="{INK2}" stroke-width="0.8"/>')
    env = ' '.join(f'{300 + t:.1f},{y0 + 15 - 30 * np.exp(-(t / 55) ** 2):.1f}' for t in np.linspace(-150, 150, 151))
    s.append(f'<polyline points="{env}" fill="none" stroke="{ORANGE}" stroke-width="2"/>')
    s.append(f'<text x="300" y="{y0 + 40}" text-anchor="middle" font-size="{fs - 1}" fill="{INK2}">'
             'nanobeam cavity; orange: mode envelope</text>')
    # sequence panel
    L, R, T, B = 104, 590, 150, 322
    y_of = {0: T + 22, -55: B - 34}
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="{INK2}" stroke-width="1"/>')
    for v, lab in ((0, '0'), (-55, 'about -55')):
        s.append(f'<line x1="{L}" y1="{y_of[v]}" x2="{R}" y2="{y_of[v]}" stroke="{LINE}" stroke-width="1"/>')
        s.append(f'<text x="{L - 8}" y="{y_of[v] + 5}" text-anchor="end" font-size="{fs}" fill="{INK}">{lab}</text>')
    s.append(f'<text transform="translate(18 {(T + B) / 2}) rotate(-90)" text-anchor="middle" font-size="{fs}" '
             f'fill="{INK}">shift (pm)</text>')
    stages = [(0, ['330 nW'], 'start'), (-55, ['8 µW'], 'blue shift'),
              (-55, ['41 µW'], 'saturates'), (0, ['41 µW', '10 min'], 'returns'),
              (0, ['any', 'power'], 'quenched'), (None, ['hours', 'dark'], '')]
    n = len(stages)
    xs = [L + 40 + i * (R - L - 70) / (n - 1) for i in range(n)]
    prev = None
    for (v, lab, note), x in zip(stages, xs):
        for j, t in enumerate(lab):
            s.append(f'<text x="{x:.1f}" y="{B + 22 + 18 * j}" text-anchor="middle" font-size="{fs}" fill="{INK}">{t}</text>')
        if v is None:
            for j, t in enumerate(['recovers,', 'shift', 'returns']):
                s.append(f'<text x="{x:.1f}" y="{(y_of[0] + y_of[-55]) / 2 - 12 + 18 * j}" text-anchor="middle" '
                         f'font-size="{fs}" fill="{INK2}">{t}</text>')
            continue
        y = y_of[v]
        if prev is not None:
            s.append(f'<line x1="{prev[0] + 9:.1f}" y1="{prev[1]}" x2="{x - 9:.1f}" y2="{y}" stroke="{INK3}" '
                     'stroke-width="1.3" stroke-dasharray="4 4"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{BLUE}" stroke="#ffffff" stroke-width="2"/>')
        s.append(f'<text x="{x:.1f}" y="{y - 14 if v == 0 else y + 25}" text-anchor="middle" font-size="{fs}" '
                 f'fill="{INK2}">{note}</text>')
        prev = (x, y)
    s.append(f'<text x="{(L + R) / 2}" y="{H - 8}" text-anchor="middle" font-size="{fs - 1}" fill="{INK2}">'
             'input power; held about 10 min at 41 µW</text>')
    s.append(f'<text x="{L + 6}" y="{T - 14}" font-size="{fs}" fill="{INK}">'
             'initial shift only: about 87 cavity photons, about 84 MHz per photon</text>')
    s.append('</svg>')
    write('fe_liang2017_sequence.svg', '\n'.join(s))


# ---------- Xie et al. 2026: conventional vs folded AWG layout (redrawn concept, not to scale) ----------
def xie():
    W, H = 560, 360
    fs = 15
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}">', f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    wg = f'fill="none" stroke="{BLUE}" stroke-width="1.6"'
    # panel a: conventional AWG, nested arcs (every arm passes through many directions)
    cx, cy = 135, 236
    for sx in (-1, 1):
        x = cx + sx * 92
        s.append(f'<path d="M{x - 26},{cy - 26} h52 l-16,26 h-20 z" fill="#cfe1f6" stroke="{BLUE}" stroke-width="1.2"/>')
        s.append(f'<line x1="{x}" y1="{cy}" x2="{x}" y2="{cy + 26}" stroke="{BLUE}" stroke-width="1.6"/>')
    for k in range(7):
        h = 120 + 13 * k
        x1, x2 = cx - 92 + 18 - 6 * k, cx + 92 - 18 + 6 * k
        s.append(f'<path d="M{x1},{cy - 26} C{x1},{cy - 26 - h} {x2},{cy - 26 - h} {x2},{cy - 26}" {wg}/>')
    s.append(f'<text x="{cx}" y="22" text-anchor="middle" font-size="{fs + 1}" font-weight="600" fill="{INK}">conventional</text>')
    for j, t in enumerate(['arms turn through', 'many angles; each angle,', 'a different TE index']):
        s.append(f'<text x="{cx}" y="{cy + 50 + 18 * j}" text-anchor="middle" font-size="{fs}" fill="{INK2}">{t}</text>')
    # panel b: folded AWG: one slab; every arm is a straight section along the same direction (z), the length
    # step between arms is only in that straight length, and each arm ends in a loop reflector
    ox, oy = 300, 64
    s.append(f'<path d="M{ox},{oy + 16} l34,-38 v100 l-34,-38 z" fill="#cfe1f6" stroke="{BLUE}" stroke-width="1.2"/>')
    for dy in (22, 34):
        s.append(f'<line x1="{ox - 26}" y1="{oy + dy}" x2="{ox}" y2="{oy + dy}" stroke="{BLUE}" stroke-width="1.6"/>')
    for k in range(7):
        y = oy - 14 + 14 * k
        xr = 400 + 18 * k
        s.append(f'<line x1="{ox + 34}" y1="{y}" x2="{xr}" y2="{y}" {wg}/>')
        s.append(f'<circle cx="{xr + 6}" cy="{y}" r="6" fill="none" stroke="{BLUE}" stroke-width="1.6"/>')
    s.append(f'<text x="430" y="22" text-anchor="middle" font-size="{fs + 1}" font-weight="600" fill="{INK}">folded</text>')
    for j, t in enumerate(['length steps in', 'straight sections', 'along one direction;', 'loop reflectors', 'send light back']):
        s.append(f'<text x="430" y="{176 + 18 * j}" text-anchor="middle" font-size="{fs}" fill="{INK2}">{t}</text>')
    # in-plane axes of the x-cut film
    ax0, ay0 = 300, 340
    s.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0 + 26}" y2="{ay0}" stroke="{INK}" stroke-width="1.4"/>')
    s.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay0 - 26}" stroke="{INK}" stroke-width="1.4"/>')
    s.append(f'<text x="{ax0 + 31}" y="{ay0 + 5}" font-size="{fs}" fill="{INK}">z</text>')
    s.append(f'<text x="{ax0 - 4}" y="{ay0 - 31}" font-size="{fs}" fill="{INK}">y</text>')
    s.append(f'<text x="{ax0 + 50}" y="{ay0 + 5}" font-size="{fs - 1}" fill="{INK2}">x-cut film, in-plane axes</text>')
    s.append('</svg>')
    write('fe_xie2026_awg_folded.svg', '\n'.join(s))


if __name__ == '__main__':
    valdez()
    vpi_db()
    weis()
    liang()
    xie()
