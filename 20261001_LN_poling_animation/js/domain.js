// Domain geometry in normalized units: period = 1, electrode gap LG; y measured from the +V electrode edge (y = 0) toward ground (y = LG).
// g(x, y) = shape parameter s in [0,1] at which the point is first inside a growing stripe domain.
// Domain = {dome(dx) <= y <= P(dx, s)} with a sharp, randomly spiky front P, capped at a final half-width wider than the electrode.
const DOM = (() => {
  const LG = 2.2, NS = 5, HW = 0.275, DOME = 0.19, B = 1.5; // electrode half-width, dome height, wedge slope
  let seed = 11;
  const rnd = () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const xs = [], Wf = [], dly = [], needles = [], wob = [];
  for (let i = 0; i < NS; i++) {
    xs.push(i - 2);
    Wf.push(0.33 * (1 + 0.06 * (rnd() - 0.5) * 2)); // final half-width: domain ~20% wider than the electrode
    dly.push(0.05 * rnd());
    const nn = 4 + Math.floor(rnd() * 3), arr = [];
    for (let k = 0; k < nn; k++) arr.push({ c: (rnd() * 2 - 1) * 0.9 * Wf[i], a: 0.2 + 0.5 * rnd(), b: 0.035 + 0.05 * rnd() });
    needles.push(arr);
    wob.push([rnd() * 6.28, rnd() * 6.28, 28 + 10 * rnd(), 71 + 20 * rnd()]);
  }
  const A1 = LG + B * 0.38 + 0.1; // front reaches the far electrode everywhere inside the final width
  const dome = (ad) => (ad < HW ? DOME * Math.sqrt(1 - (ad / HW) ** 2) : 0);
  const capW = (i, y) => Wf[i] * (1 + 0.04 * (0.6 * Math.sin(wob[i][2] * y + wob[i][0]) + 0.4 * Math.sin(wob[i][3] * y + wob[i][1])));
  const inside = (i, dx, y, s) => {
    const ad = Math.abs(dx);
    if (y < 0 || y > LG || ad > capW(i, y)) return false;
    const yd = dome(ad);
    if (y < yd || y > LG - yd) return false;
    let adv = 0;
    for (const n of needles[i]) { const t = 1 - Math.abs(dx - n.c) / n.b; if (t > 0 && n.a * t > adv) adv = n.a * t; }
    return y <= A1 * s - B * ad + adv;
  };
  const g = (x, y) => {
    const i = Math.min(NS - 1, Math.max(0, Math.round(x) + 2)), dx = x - xs[i];
    if (y < 0 || y > LG || Math.abs(dx) > Wf[i] * 1.06 || !inside(i, dx, y, 1)) return Infinity;
    let lo = 0, hi = 1;
    for (let it = 0; it < 22; it++) { const m = 0.5 * (lo + hi); if (inside(i, dx, y, m)) hi = m; else lo = m; }
    return dly[i] + (1 - dly[i]) * hi;
  };

  // Pixel grid shared by the top view and the area CDF (140 px per period).
  const grid = { x0: -2.6, x1: 2.6, y0: -0.45, y1: LG + 0.45, w: 728, h: 434 };
  grid.g = new Float32Array(grid.w * grid.h);
  const finite = [];
  for (let r = 0; r < grid.h; r++) {
    const y = grid.y0 + (r + 0.5) / grid.h * (grid.y1 - grid.y0);
    for (let c = 0; c < grid.w; c++) {
      const x = grid.x0 + (c + 0.5) / grid.w * (grid.x1 - grid.x0);
      const v = g(x, y);
      grid.g[r * grid.w + c] = v;
      if (isFinite(v)) finite.push(v);
    }
  }
  const sorted = Float32Array.from(finite).sort();
  const n = sorted.length;
  // Shape parameter reached when switched-area fraction is f (area = integral of switching current / Q).
  const hOf = (f) => (f * n < 1 ? -1 : sorted[Math.min(n - 1, Math.floor(f * n))]);
  // Switched-area fraction at which shape parameter gv is reached.
  const fOfG = (gv) => {
    if (!isFinite(gv)) return Infinity;
    let lo = 0, hi = n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= gv) lo = m + 1; else hi = m; }
    return Math.max(lo, 1) / n;
  };
  return { LG, NS, xs, Wf, HW, DOME, g, grid, hOf, fOfG, n };
})();
