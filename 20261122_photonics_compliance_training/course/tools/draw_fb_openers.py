# Title-card backgrounds for FB-1550 module openers. 1920x1080 JPEG. No axes, no text.
import sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
OUT = sys.argv[1]
W, H, SS = 1920, 1080, 2
def base(c0=(10, 22, 40), c1=(18, 48, 82)):
    y = np.linspace(0, 1, H * SS)[:, None]; x = np.linspace(0, 1, W * SS)[None, :]
    t = np.clip(0.55 * x + 0.45 * (1 - y), 0, 1)
    img = np.zeros((H * SS, W * SS, 3))
    for k in range(3): img[..., k] = c0[k] + (c1[k] - c0[k]) * t
    im = Image.fromarray(img.astype(np.uint8)); d = ImageDraw.Draw(im, 'RGBA')
    for gx in range(0, W * SS, 80 * SS): d.line([(gx, 0), (gx, H * SS)], fill=(255, 255, 255, 10), width=SS)
    for gy in range(0, H * SS, 80 * SS): d.line([(0, gy), (W * SS, gy)], fill=(255, 255, 255, 10), width=SS)
    return im
def curve(im, xs, ys, col, w, glow=True):
    pts = [(float(x) * SS, float(y) * SS) for x, y in zip(xs, ys)]
    if glow:
        g = Image.new('RGBA', im.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
        gd.line(pts, fill=col[:3] + (90,), width=w * 5 * SS, joint='curve'); g = g.filter(ImageFilter.GaussianBlur(8 * SS))
        im.paste(g, (0, 0), g)
    ImageDraw.Draw(im, 'RGBA').line(pts, fill=col, width=w * SS, joint='curve')
def finish(im, name):
    im = im.resize((W, H), Image.LANCZOS).convert('RGB')
    # darker band at the bottom for the title
    a = np.asarray(im).astype(float); y = np.linspace(0, 1, H)[:, None, None]
    a *= 1 - 0.45 * np.clip((y - 0.55) / 0.45, 0, 1)
    Image.fromarray(a.astype(np.uint8)).save(OUT + '/' + name, quality=88, optimize=True, progressive=True)

# 1. Airy transmission of a lossless etalon, T = 1 / (1 + F sin^2(phi/2)), F = 4R / (1 - R)^2; three mirror reflectances
im = base()
f = np.linspace(-2.6, 2.6, 3000)               # frequency in units of FSR
xs = 80 + (f + 2.6) / 5.2 * (W - 160)
for R, col, y0, amp in [(0.3, (120, 170, 220, 200), 760, 520), (0.7, (90, 200, 255, 230), 760, 520), (0.93, (255, 200, 90, 255), 760, 520)]:
    F = 4 * R / (1 - R) ** 2
    T = 1 / (1 + F * np.sin(np.pi * f) ** 2)
    curve(im, xs, y0 - amp * T, col, 3 if R > 0.9 else 2)
finish(im, 'opener_airy.jpg')

# 2. Mode-hop staircase with hysteresis: comb rate a_c, filter rate a_f, hop when |m + a_c T - a_f T| > 0.5 + h
im = base((14, 18, 36), (40, 30, 70))
ac, af, h, TM = 0.25, 0.8, 0.2, 10
def sweep(T0, T1):
    Ts = np.linspace(T0, T1, 4000); m = round(af * T0 - ac * T0); pts = []
    for T in Ts:
        if abs(m + ac * T - af * T) > 0.5 + h: m = round(af * T - ac * T)
        pts.append((T, m + ac * T))
    return pts
for pts, col, dy in [(sweep(0, TM), (90, 200, 255, 240), 0), (sweep(TM, 0), (255, 120, 140, 230), 0)]:
    segs, cur = [], [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if abs(b[1] - a[1]) > 0.5: segs.append(cur); cur = [b]
        else: cur.append(b)
    segs.append(cur)
    for sgm in segs:
        x = [140 + p[0] / TM * (W - 280) for p in sgm]; y = [940 - (p[1] + 1) / (af * TM + 2) * 820 for p in sgm]
        curve(im, x, y, col, 4)
x = [140, W - 140]; y = [940 - 1 / (af * TM + 2) * 820, 940 - (af * TM + 1) / (af * TM + 2) * 820]
ImageDraw.Draw(im, 'RGBA').line([(x[0] * SS, y[0] * SS), (x[1] * SS, y[1] * SS)], fill=(255, 200, 90, 120), width=2 * SS)
finish(im, 'opener_modehop.jpg')

# 3. Backscatter doublet, coupled-mode model: D = 1/2 - i x, t = 1 - eta D/(D^2 + b^2), r = i b eta/(D^2 + b^2); eta = 0.5
im = base((8, 24, 30), (16, 60, 70))
x = np.linspace(-6, 6, 4000)
for b, col, w in [(0.0, (120, 170, 200, 170), 2), (1.2, (90, 230, 210, 240), 3), (2.5, (255, 200, 90, 250), 4)]:
    D = 0.5 - 1j * x; den = D ** 2 + b ** 2
    T = np.abs(1 - 0.5 * D / den) ** 2
    curve(im, 80 + (x + 6) / 12 * (W - 160), 150 + 560 * (1 - T), col, w)
    if b > 2:
        Rr = np.abs(1j * b * 0.5 / den) ** 2
        curve(im, 80 + (x + 6) / 12 * (W - 160), 930 - 180 * Rr / 0.25, (255, 120, 140, 200), 2)
d = ImageDraw.Draw(im, 'RGBA')
cx, cy, r = 1560 * SS, 300 * SS, 170 * SS
for k in range(6): d.ellipse([cx - r - k * 2 * SS, cy - r - k * 2 * SS, cx + r + k * 2 * SS, cy + r + k * 2 * SS], outline=(90, 230, 210, 40), width=SS)
d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90, 230, 210, 230), width=5 * SS)
finish(im, 'opener_doublet.jpg')

# 4. Faraday isolator: input polarizer at 0 deg, 45 deg non-reciprocal rotation, output polarizer at 45 deg.
#    Forward: 0 -> 45 (passes). Backward: 45 -> 90 relative to the input polarizer (blocked).
im = base((12, 20, 44), (30, 40, 90)); d = ImageDraw.Draw(im, 'RGBA')
yF, yB = 420, 700
d.line([(100 * SS, yF * SS), (1820 * SS, yF * SS)], fill=(90, 200, 255, 90), width=3 * SS)
d.line([(100 * SS, yB * SS), (1820 * SS, yB * SS)], fill=(255, 120, 140, 90), width=3 * SS)
def pol(xc, yc, ang, col, L=70):
    a = math.radians(ang); dx, dy = L * math.sin(a), -L * math.cos(a)
    d.line([((xc - dx) * SS, (yc - dy) * SS), ((xc + dx) * SS, (yc + dy) * SS)], fill=col, width=6 * SS)
def plate(xc, ang, col):
    for yc in (yF, yB):
        d.rectangle([(xc - 14) * SS, (yc - 120) * SS, (xc + 14) * SS, (yc + 120) * SS], fill=(255, 255, 255, 18), outline=(255, 255, 255, 60), width=SS)
        a = math.radians(ang)
        for k in range(-4, 5):
            yy = yc + k * 24; dx, dy = 10 * math.sin(a), -10 * math.cos(a)
            d.line([((xc - dx) * SS, (yy - dy) * SS), ((xc + dx) * SS, (yy + dy) * SS)], fill=col, width=2 * SS)
plate(520, 0, (255, 255, 255, 150))
d.rectangle([780 * SS, (yF - 140) * SS, 1140 * SS, (yB + 140) * SS], fill=(255, 200, 90, 22), outline=(255, 200, 90, 120), width=2 * SS)
plate(1400, 45, (255, 255, 255, 150))
for xc, ang in [(250, 0), (650, 0), (900, 15), (1020, 30), (1270, 45), (1600, 45), (1750, 45)]: pol(xc, yF, ang, (90, 200, 255, 235))
for xc, ang in [(1650, 45), (1270, 45), (1020, 60), (900, 75), (650, 90)]: pol(xc, yB, ang, (255, 120, 140, 235))
d.line([(470 * SS, (yB - 70) * SS), (570 * SS, (yB + 70) * SS)], fill=(255, 120, 140, 200), width=5 * SS)
d.line([(470 * SS, (yB + 70) * SS), (570 * SS, (yB - 70) * SS)], fill=(255, 120, 140, 200), width=5 * SS)
finish(im, 'opener_faraday.jpg')
print('done')
