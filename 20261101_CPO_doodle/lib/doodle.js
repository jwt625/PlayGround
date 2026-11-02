// Isometric doodle renderer on Canvas2D. No libraries, no images, deterministic (same pattern as the hybrid bonding
// short runtime): a scene registers with DD.scene({build, meta}); the renderer calls DD.frame(t) per frame.
// URL params: style=A..E, play=1 (browser preview loop).
// World axes: +X screen right-down, +Y screen right-up, +Z up. Visible faces of any solid: +X, -Y, +Z.
(() => {
  const params = Object.fromEntries(new URLSearchParams(location.search));
  const W = Number(params.w) || 1920, H = Number(params.h) || 1080, BOIL_CYCLE = 4;
  let BOIL_FPS = 8;
  const C30 = Math.cos(Math.PI / 6);

  const PALETTE = {
    carrier: '#7FB08A', pic: '#5B5F97', eic: '#D9DCE3', glass: '#BFE6EE', fiber: '#F2C94C', ferrule: '#F4EFE3',
    smd: '#D8B98A', pad: '#E2B13C', ring: '#FF8A5C', wg: '#FFD9A0', metal: '#CDBB94', gold_die: '#A99A55',
    qr: '#F7F7F2', substrate: '#6FA7B5', bcm_body: '#C9D6EE', bcm_strip: '#59627A', bcm_block: '#EEF3F7',
    floor: '#EFE6D2', frame: '#8FB7D9', frame2: '#F2A65A', belt: '#5A5A66', rail: '#C8CCD2', tray: '#B8C96B',
    screen: '#2F3A4A', trace: '#7CF0B0', ok: '#6CCB5F', ng: '#E8574A', bin: '#9A8C7A', ink: '#1E1A18',
    wm: '#8F8676', carton: '#C9A36B', cartonIn: '#A98650', tape: '#EAD9B0', lampoff: '#B9B3A5', asic: '#9AA0A8', board: '#4F8F6B', chassis: '#D7DCE2', lid: '#C8CCD2', module: '#F2C94C',
  };

  const STYLES = {
    // Felt-tip rakugaki: even thick marker line, pastel flat fill shifted off the line.
    A: { name: 'marker', paper: '#F6F1E3', ink: '#1E1A18', thick: 7, inner: 4, fine: 2.4, amp: 2.2, period: 60, vj: 2.0,
         over: 3, wnoise: 0.14, fill: 'flat', shift: [3, 2] },
    // Heta-uma crayon: fat unsteady line, scribbled fill that misses the edges.
    B: { name: 'crayon', paper: '#FBF7EC', ink: '#23201D', thick: 11, inner: 5.5, fine: 2.8, amp: 3.0, period: 55, vj: 3.5,
         over: 6, wnoise: 0.28, fill: 'crayon', shift: [5, 3] },
    // Manga ink: pen-weighted black line, white fills, screentone for shadow, one spot color.
    C: { name: 'manga', paper: '#FAFAF6', ink: '#0E0E10', thick: 7.5, inner: 3, fine: 1.6, amp: 0.8, period: 90, vj: 0.7,
         over: 2, wnoise: 0.08, fill: 'tone', shift: [0, 0], pen: true, shadow: true, spot: '#E8452C',
         spotKeys: ['fiber', 'ring', 'ng', 'ok', 'frame2'], darkKeys: ['pic', 'belt', 'screen', 'ink', 'bcm_strip'] },
    // Picture-book cel: dark navy line, flat color plus hard face tones and a cast shadow.
    D: { name: 'cel', paper: '#E9EFE4', ink: '#232B4A', thick: 6, inner: 3, fine: 1.6, amp: 0.9, period: 100, vj: 0.9,
         over: 0.8, wnoise: 0.06, fill: 'cel', shift: [0, 0], shadow: true },
    // Blackboard chalk: broken pale line on dark, thin dusty scribble.
    E: { name: 'chalk', paper: '#20282B', ink: '#EFEBDD', thick: 6.5, inner: 3.5, fine: 1.9, amp: 1.7, period: 60, vj: 1.8,
         over: 4, wnoise: 0.2, fill: 'chalk', shift: [2, 1], chalk: true },
  };

  STYLES.L = { ...STYLES.A, name: 'marker, dense line scene', thick: 4.2, inner: 2.3, fine: 1.5, amp: 1.2, period: 45, vj: 1.1, over: 1.6, shift: [1.6, 1], grain: false };

  // ---------------------------------------------------------------- noise
  const hash = (a, b = 0) => {
    let h = Math.imul(a | 0, 0x9E3779B1) ^ Math.imul(b | 0, 0x85EBCA77);
    h ^= h >>> 15; h = Math.imul(h, 0xC2B2AE3D); h ^= h >>> 13;
    return h >>> 0;
  };
  const r01 = (a, b = 0) => hash(a, b) / 4294967296;
  const r11 = (a, b = 0) => 2 * r01(a, b) - 1;
  const vn = (seed, x) => { // 1D value noise, -1..1
    const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f);
    return r11(seed, i) * (1 - u) + r11(seed, i + 1) * u;
  };
  const rng = (seed) => {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  const rgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const mix = (h, m) => { const c = rgb(h); return `rgb(${c[0] * m[0] | 0},${c[1] * m[1] | 0},${c[2] * m[2] | 0})`; };

  // ---------------------------------------------------------------- world model
  const solids = [], overlays = [];
  let nextId = 1, ordCounter = 0, dynKey = null, dynCount = 0, flying = false;
  // Dance, after the Helltaker mini-character idle: feet planted, hips swing one way, shoulders tilt the other way, the head only
  // translates, and the body stretches in transit and settles at each hold. Offsets run along (1, 1, 0), pure left-right on screen.
  // A 'bend' solid is a body with curved sides; a 'ride' solid is attached to the head of a body of height h.
  const HIP = 1.25, HEAD = 0.75, STRETCH = 0.07, HIP_F = 0.38, SHOULDER_F = 0.82;
  let danceV = 0, danceS = 0, swayTag = null;
  const smooth = (u) => { const w = Math.max(0, Math.min(1, u)); return w * w * (3 - 2 * w); };
  function bodyCurve(f, amp) {
    const hip = HIP * danceV * amp, head = -HEAD * danceV * amp;
    if (f < HIP_F) return hip * smooth(f / HIP_F);
    if (f < SHOULDER_F) return hip + (head - hip) * smooth((f - HIP_F) / (SHOULDER_F - HIP_F));
    return head;
  }
  // Returns [offset along (1, 1, 0), rise in z] for a point of solid s at height z.
  function danceAt(s, z) {
    const t = s.sway;
    if (!t) return [0, 0];
    if (t.ride) return [-HEAD * danceV * t.amp, STRETCH * danceS * t.amp * t.h];
    return [bodyCurve(Math.max(0, Math.min(1, z / t.h)), t.amp), STRETCH * danceS * t.amp * z];
  }
  const PD = (s, x, y, z) => { const [d, r] = danceAt(s, z); return P(x + d, y + d, z + r); };
  const add = (s) => {
    s.sway = swayTag;
    s.id = dynKey === null ? nextId++ : hash(dynKey, dynCount++);
    s.ord = ordCounter++; s.fly = flying; s.decals = []; solids.push(s); return s;
  };

  function prism(poly, z0, z1, key, o = {}) {
    const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
    return add({ kind: 'prism', poly, z0, z1, key, tier: o.tier || 'main', smooth: !!o.smooth, ground: !!o.ground,
      convex: o.convex !== false, x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) });
  }
  const box = (x0, y0, z0, x1, y1, z1, key, o = {}) =>
    prism([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], z0, z1, key, o);
  function cyl(cx, cy, z0, z1, r, key, o = {}) {
    const n = 28, poly = [];
    for (let i = 0; i < n; i++) poly.push([cx + r * Math.cos(2 * Math.PI * i / n), cy + r * Math.sin(2 * Math.PI * i / n)]);
    return prism(poly, z0, z1, key, { ...o, smooth: true });
  }
  function rotBox(cx, cy, hx, hy, z0, z1, a, key, o = {}) {
    const c = Math.cos(a), s = Math.sin(a);
    return prism([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([u, v]) => [cx + u * c - v * s, cy + u * s + v * c]), z0, z1, key, o);
  }
  // One flat polygon in any plane, for hinged parts (carton walls and flaps). Painter order uses its AABB.
  function panel(pts, key, o = {}) {
    const v = (i) => pts.map((p) => p[i]);
    return add({ kind: 'panel', pts, key, tier: o.tier || 'fine', closedLine: o.line !== false, x0: Math.min(...v(0)), x1: Math.max(...v(0)),
      y0: Math.min(...v(1)), y1: Math.max(...v(1)), z0: Math.min(...v(2)), z1: Math.max(...v(2)) });
  }
  // Group of tubes along 3D polylines, sorted as one solid. width in world units.
  function tubes(paths, width, key) {
    const all = paths.flat();
    const v = (i) => all.map((p) => p[i]);
    return add({ kind: 'tubes', paths, width, key, tier: 'fine', x0: Math.min(...v(0)), x1: Math.max(...v(0)),
      y0: Math.min(...v(1)), y1: Math.max(...v(1)), z0: Math.min(...v(2)), z1: Math.max(...v(2)) });
  }
  // Decals live on a solid's top or wall and draw with it, so they occlude correctly.
  const rectTop = (s, x0, y0, x1, y1, key) => { s.decals.push({ t: 'rect', x0, y0, x1, y1, key }); return s; };
  const discTop = (s, cx, cy, r, key) => { s.decals.push({ t: 'disc', cx, cy, r, key }); return s; };
  const text = (s, str, p, face, size, color) => { s.decals.push({ t: 'text', str, p, face, size, color }); return s; };
  const xMark = (s, cx, cy, r) => { s.decals.push({ t: 'x', cx, cy, r }); return s; };
  const xWall = (s, face, c, cz, r) => { s.decals.push({ t: 'xw', face, c, cz, r }); return s; };
  const guide = (a, b) => overlays.push({ t: 'guide', a, b });
  const label = (p, str, dx, dy) => overlays.push({ t: 'label', p, str, dx, dy });

  // ---------------------------------------------------------------- camera
  const raw = (x, y, z) => [(x + y) * C30, (x - y) * 0.5 - z];
  let cam = { s: 1, ox: 0, oy: 0 };
  const P = (x, y, z) => { const r = raw(x, y, z); return [cam.ox + r[0] * cam.s, cam.oy + r[1] * cam.s]; };

  function fit(margin = 70) {
    let a = 1e9, b = -1e9, c = 1e9, d = -1e9;
    const ext = (x, y, z) => { const r = raw(x, y, z); a = Math.min(a, r[0]); b = Math.max(b, r[0]); c = Math.min(c, r[1]); d = Math.max(d, r[1]); };
    for (const s of solids) for (const x of [s.x0, s.x1]) for (const y of [s.y0, s.y1]) for (const z of [s.z0, s.z1]) ext(x, y, z);
    const sc = Math.min((W - 2 * margin) / (b - a), (H - 2 * margin) / (d - c));
    cam = { s: sc, ox: W / 2 - sc * (a + b) / 2, oy: H / 2 - sc * (c + d) / 2 };
  }

  // Painter order for AABBs: a separating plane tells which of two overlapping solids is behind.
  function behind(A, B) {
    const e = 1e-6;
    if (A.x1 <= B.x0 + e) return true;
    if (B.x1 <= A.x0 + e) return false;
    if (A.y0 >= B.y1 - e) return true;
    if (B.y0 >= A.y1 - e) return false;
    if (A.z1 <= B.z0 + e) return true;
    if (B.z1 <= A.z0 + e) return false;
    if (A.fly !== B.fly) return B.fly;
    return A.ord < B.ord;
  }
  // Screen overlap of two AABBs in true isometric: their projected hexagons overlap iff the ranges overlap on all three
  // axes that are invariant along the view direction (1, -1, 1).
  function hexOverlap(A, B) {
    const e = 1e-6;
    return A.x0 + A.y0 < B.x1 + B.y1 - e && B.x0 + B.y0 < A.x1 + A.y1 - e
      && A.y0 + A.z0 < B.y1 + B.z1 - e && B.y0 + B.z0 < A.y1 + A.z1 - e
      && A.x0 - A.z1 < B.x1 - B.z0 - e && B.x0 - B.z1 < A.x1 - A.z0 - e;
  }
  function separated(A, B) {
    const e = 1e-6;
    return A.x1 <= B.x0 + e || B.x1 <= A.x0 + e || A.y0 >= B.y1 - e || B.y0 >= A.y1 - e || A.z1 <= B.z0 + e || B.z1 <= A.z0 + e;
  }
  function sorted() {
    const live = solids.filter((s) => {
      let a = 1e9, b = -1e9, c = 1e9, d = -1e9;
      for (const x of [s.x0, s.x1]) for (const y of [s.y0, s.y1]) for (const z of [s.z0, s.z1]) {
        const r = raw(x, y, z); a = Math.min(a, r[0]); b = Math.max(b, r[0]); c = Math.min(c, r[1]); d = Math.max(d, r[1]);
      }
      return cam.ox + b * cam.s > -20 && cam.ox + a * cam.s < W + 20 && cam.oy + d * cam.s > -20 && cam.oy + c * cam.s < H + 20;
    });
    return sortList(live);
  }
  function sortList(solids) {
    const n = solids.length;
    const after = solids.map(() => []), deg = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (!hexOverlap(solids[i], solids[j]) || !separated(solids[i], solids[j])) continue;
      if (behind(solids[i], solids[j])) { after[i].push(j); deg[j]++; } else { after[j].push(i); deg[i]++; }
    }
    const out = [], done = new Array(n).fill(false);
    for (let k = 0; k < n; k++) {
      let pick = -1;
      for (let i = 0; i < n; i++) if (!done[i] && deg[i] === 0) { pick = i; break; }
      if (pick < 0) for (let i = 0; i < n; i++) if (!done[i]) { pick = i; break; } // cycle: take the earliest
      done[pick] = true; out.push(solids[pick]);
      for (const j of after[pick]) deg[j]--;
    }
    return out;
  }

  // ---------------------------------------------------------------- hand line
  let g, ST, boil = 0;

  // Resample a polyline and push it off its own path with slow noise. Seeded, so a fill and a stroke over the same
  // edge agree. Offset tapers to zero at open ends so corners stay joined.
  function wobble(pts, seed, closed = false, scale = 1) {
    const out = [], step = 5;
    let total = 0;
    const seg = [];
    const m = closed ? pts.length : pts.length - 1;
    for (let i = 0; i < m; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      seg.push({ a, b, L, u0: total }); total += L;
    }
    if (total < 0.5) return pts.map((p) => p.slice());
    const amp = ST.amp * scale * Math.min(1, total / 50), per = ST.period, s = hash(seed, boil);
    const count = Math.max(2, Math.ceil(total / step));
    let si = 0;
    for (let k = 0; k <= count; k++) {
      const u = total * k / count;
      while (si < seg.length - 1 && u > seg[si].u0 + seg[si].L) si++;
      const q = seg[si], f = q.L > 0 ? (u - q.u0) / q.L : 0;
      const tx = (q.b[0] - q.a[0]) / (q.L || 1), ty = (q.b[1] - q.a[1]) / (q.L || 1);
      const env = closed ? 1 : Math.min(1, Math.min(u, total - u) / 12);
      const off = (vn(s, u / per) + 0.35 * vn(s + 31, u / (per * 0.27))) * amp * env;
      out.push([q.a[0] + (q.b[0] - q.a[0]) * f - ty * off, q.a[1] + (q.b[1] - q.a[1]) * f + tx * off]);
    }
    return out;
  }

  function drawStroke(pts, w, seed, closed = false) {
    if (pts.length < 2) return;
    const s = hash(seed, boil + 77);
    g.lineCap = ST.chalk ? 'butt' : 'round';
    g.lineJoin = 'round';
    g.strokeStyle = ST.ink;
    let u = 0;
    const n = pts.length - 1;
    for (let i = 0; i < n; i++) {
      const a = pts[i], b = pts[i + 1], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      let lw = w * (1 + ST.wnoise * vn(s, u / 36));
      if (ST.pen) { // G-pen: wider across the stroke direction, taper at both ends
        const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
        const tap = closed ? 1 : Math.min(1, 0.35 + Math.min(i, n - i) / 7);
        lw = lw * (0.5 + 0.7 * Math.abs(Math.sin(ang - 1.0))) * tap;
      }
      if (ST.chalk) {
        const dust = 0.55 + 0.45 * vn(s + 5, u / 7);
        if (r01(s + 9, i) < 0.08) { u += L; continue; }
        g.globalAlpha = Math.max(0.2, dust);
      }
      g.lineWidth = lw;
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
      u += L;
    }
    g.globalAlpha = 1;
    if (ST.fill === 'crayon') { // second, looser pass: the line a child goes over twice
      g.lineWidth = w * 0.45; g.globalAlpha = 0.55;
      g.beginPath();
      pts.forEach((p, i) => {
        const dx = 1.6 * vn(s + 13, i / 7), dy = 1.6 * vn(s + 17, i / 7);
        if (i) g.lineTo(p[0] + dx, p[1] + dy); else g.moveTo(p[0] + dx, p[1] + dy);
      });
      g.stroke(); g.globalAlpha = 1;
    }
  }

  function hull(points) { // Andrew monotone chain over indices
    const idx = points.map((_, i) => i).sort((a, b) => points[a][0] - points[b][0] || points[a][1] - points[b][1]);
    const cross = (o, a, b) => (points[a][0] - points[o][0]) * (points[b][1] - points[o][1]) - (points[a][1] - points[o][1]) * (points[b][0] - points[o][0]);
    const lo = [], up = [];
    for (const i of idx) { while (lo.length > 1 && cross(lo[lo.length - 2], lo[lo.length - 1], i) <= 0) lo.pop(); lo.push(i); }
    for (const i of idx.slice().reverse()) { while (up.length > 1 && cross(up[up.length - 2], up[up.length - 1], i) <= 0) up.pop(); up.push(i); }
    lo.pop(); up.pop();
    return lo.concat(up);
  }

  const pathOf = (pts) => { g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath(); };

  // ---------------------------------------------------------------- fills
  let tones = {};
  function tonePattern(density) { // round-dot sheet at 45 deg
    if (tones[density]) return tones[density];
    const pitch = 9, c = document.createElement('canvas');
    c.width = c.height = pitch * 2;
    const q = c.getContext('2d');
    q.fillStyle = ST.ink;
    const r = density * pitch * 0.62;
    for (const [x, y] of [[0, 0], [pitch, pitch], [pitch * 2, 0], [0, pitch * 2], [pitch * 2, pitch * 2]]) {
      q.beginPath(); q.arc(x, y, r, 0, 7); q.fill();
    }
    return (tones[density] = g.createPattern(c, 'repeat'));
  }

  function scribble(pts, color, seed, spacing, width, alpha, angle) {
    g.save();
    pathOf(pts); g.clip();
    let a = 1e9, b = -1e9, c = 1e9, d = -1e9;
    for (const p of pts) { a = Math.min(a, p[0]); b = Math.max(b, p[0]); c = Math.min(c, p[1]); d = Math.max(d, p[1]); }
    const cx = (a + b) / 2, cy = (c + d) / 2, R = Math.hypot(b - a, d - c) / 2 + 8;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    g.strokeStyle = color; g.lineWidth = width; g.lineCap = 'round'; g.lineJoin = 'round'; g.globalAlpha = alpha;
    g.beginPath();
    let row = 0;
    for (let v = -R; v < R; v += spacing, row++) {
      const dir = row % 2 ? 1 : -1;
      for (let k = 0; k <= 10; k++) {
        const u = dir * (-R + 2 * R * k / 10), j = 2.2 * vn(hash(seed, row), k * 0.7);
        const x = cx + u * ca - (v + j) * sa, y = cy + u * sa + (v + j) * ca;
        if (row === 0 && k === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
    }
    g.stroke();
    g.restore(); g.globalAlpha = 1;
  }

  function fillFace(pts, key, face, seed) {
    const col = PALETTE[key] || '#ff00ff';
    const f = ST.fill;
    if (f === 'flat') { pathOf(pts); g.fillStyle = col; g.fill(); return; }
    if (f === 'cel') {
      const m = face === 'top' ? [1, 1, 1] : face === 'my' ? [0.86, 0.87, 0.93] : [0.68, 0.70, 0.84];
      pathOf(pts); g.fillStyle = mix(col, m); g.fill(); return;
    }
    if (f === 'tone') {
      pathOf(pts);
      if (ST.darkKeys.includes(key)) { g.fillStyle = face === 'top' ? '#26262B' : '#141417'; g.fill(); return; }
      if (ST.spotKeys.includes(key)) {
        g.fillStyle = ST.spot; g.fill();
        if (face !== 'top') { g.fillStyle = tonePattern(face === 'my' ? 0.34 : 0.56); g.fill(); }
        return;
      }
      g.fillStyle = ST.paper; g.fill();
      if (face !== 'top') { g.fillStyle = tonePattern(face === 'my' ? 0.34 : 0.56); g.fill(); }
      return;
    }
    if (f === 'crayon') {
      const ang = face === 'top' ? -0.5 : face === 'my' ? 0.5 : 1.25;
      const sp = face === 'top' ? 7.5 : face === 'my' ? 6.5 : 5.5;
      scribble(pts, col, seed, sp, 7, 0.9, ang); return;
    }
    if (f === 'chalk') {
      const ang = face === 'top' ? -0.5 : face === 'my' ? 0.5 : 1.25;
      const sp = face === 'top' ? 13 : face === 'my' ? 10 : 7.5;
      scribble(pts, col, seed, sp, 3.5, 0.4, ang); return;
    }
  }

  function decal(s, d) {
    const z = s.z1;
    const q = (x, y) => PD(s, x, y, z);
    if (d.t === 'rect') {
      const pts = [q(d.x0, d.y0), q(d.x1, d.y0), q(d.x1, d.y1), q(d.x0, d.y1)];
      g.fillStyle = decalColor(d.key); pathOf(pts); g.fill();
    } else if (d.t === 'disc') {
      const pts = [];
      for (let i = 0; i < 14; i++) pts.push(q(d.cx + d.r * Math.cos(i / 14 * 6.2832), d.cy + d.r * Math.sin(i / 14 * 6.2832)));
      g.fillStyle = decalColor(d.key); pathOf(pts); g.fill();
    } else if (d.t === 'x') {
      const a = q(d.cx - d.r, d.cy - d.r), b = q(d.cx + d.r, d.cy + d.r), c = q(d.cx - d.r, d.cy + d.r), e = q(d.cx + d.r, d.cy - d.r);
      g.strokeStyle = PALETTE.ng; g.lineWidth = Math.max(2, ST.inner); g.lineCap = 'round';
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.moveTo(c[0], c[1]); g.lineTo(e[0], e[1]); g.stroke();
    } else if (d.t === 'xw') {
      const w = (u, v) => (d.face === 'my' ? P(d.c + u, s.y0, d.cz + v) : P(s.x1, d.c + u, d.cz + v));
      const a = w(-d.r, -d.r), b = w(d.r, d.r), c = w(-d.r, d.r), e = w(d.r, -d.r);
      g.strokeStyle = PALETTE.ng; g.lineWidth = Math.max(2, ST.inner); g.lineCap = 'round';
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.moveTo(c[0], c[1]); g.lineTo(e[0], e[1]); g.stroke();
    } else if (d.t === 'text') {
      const o = PD(s, d.p[0], d.p[1], d.p[2]), sc = cam.s;
      const basis = d.face === 'my' ? [C30, 0.5, 0, 1] : d.face === 'px' ? [C30, -0.5, 0, 1] : [C30, 0.5, -C30, 0.5];
      g.save();
      g.transform(basis[0] * sc, basis[1] * sc, basis[2] * sc, basis[3] * sc, o[0], o[1]);
      g.font = `${d.size}px 'DD Hand'`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      const lines = d.str.split('\n');
      if (s.key !== 'screen' && ['tone', 'crayon', 'chalk'].includes(ST.fill)) { // sign plate so lettering reads over tone or scribble
        const tw = Math.max(...lines.map((l) => g.measureText(l).width)) + d.size * 0.7, th = lines.length * d.size * 1.08 + d.size * 0.3;
        g.fillStyle = ST.paper; g.fillRect(-tw / 2, -th / 2, tw, th);
        g.strokeStyle = ST.ink; g.lineWidth = 2 / sc; g.strokeRect(-tw / 2, -th / 2, tw, th);
      }
      g.fillStyle = d.color ? PALETTE[d.color] : s.key === 'screen' ? PALETTE.trace : ST.ink;
      lines.forEach((line, i, all) => g.fillText(line, 0, (i - (all.length - 1) / 2) * d.size * 1.08));
      g.restore();
    }
  }
  const decalColor = (key) => {
    if (ST.fill === 'tone') return ST.spotKeys.includes(key) ? ST.spot : ST.darkKeys.includes(key) ? ST.ink : ST.paper;
    return PALETTE[key] || '#ff00ff';
  };

  // ---------------------------------------------------------------- solids
  // A box whose four vertical edges follow the body curve, so the walls have real hips and shoulders.
  function drawBent(s) {
    const seed = hash(s.id, 4242), [x0, x1] = [s.x0, s.x1], [y0, y1] = [s.y0, s.y1], LV = 12;
    const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    const edge = corners.map(([x, y], c) => {
      const pts = [];
      for (let i = 0; i <= LV; i++) pts.push(PD(s, x, y, s.z0 + (s.z1 - s.z0) * i / LV));
      return wobble(pts, hash(seed, c), false, 0.8);
    });
    const top = (c) => edge[c][edge[c].length - 1], bot = (c) => edge[c][0], line = (a, b, sd) => wobble([a, b], hash(seed, sd));
    const rev = (q) => q.slice().reverse();
    const wall = (a, b, sd) => [...edge[a], ...line(top(a), top(b), sd + 10), ...rev(edge[b]), ...line(bot(b), bot(a), sd + 20)];
    const sil = [...edge[0], ...line(top(0), top(3), 31), ...line(top(3), top(2), 32), ...rev(edge[2]), ...line(bot(2), bot(1), 33), ...line(bot(1), bot(0), 34)];
    pathOf(sil); g.fillStyle = ST.paper; g.fill();
    g.save();
    g.translate(ST.shift[0], ST.shift[1]);
    fillFace(wall(0, 1, 1), s.key, 'my', hash(seed, 1));
    fillFace(wall(1, 2, 2), s.key, 'px', hash(seed, 2));
    fillFace([...line(top(0), top(1), 11), ...line(top(1), top(2), 12), ...line(top(2), top(3), 32), ...line(top(3), top(0), 31)], s.key, 'top', hash(seed, 3));
    g.restore();
    for (const d of s.decals) decal(s, d);
    drawStroke(edge[1], ST.inner, hash(seed, 41));
    drawStroke(line(top(0), top(1), 11), ST.inner, hash(seed, 42));
    drawStroke(line(top(1), top(2), 12), ST.inner, hash(seed, 43));
    drawStroke(edge[0], ST.thick, hash(seed, 44));
    drawStroke(edge[2], ST.thick, hash(seed, 45));
    drawStroke(line(top(0), top(3), 31), ST.thick, hash(seed, 46));
    drawStroke(line(top(3), top(2), 32), ST.thick, hash(seed, 47));
    drawStroke(line(bot(2), bot(1), 33), ST.thick, hash(seed, 48));
    drawStroke(line(bot(1), bot(0), 34), ST.thick, hash(seed, 49));
  }

  function drawPrism(s) {
    if (s.sway && !s.sway.ride) return drawBent(s);
    const n = s.poly.length, seed = hash(s.id, 4242);
    const V = [];
    const vj = s.smooth ? 0 : ST.vj * Math.min(1, cam.s * Math.min(s.x1 - s.x0, s.y1 - s.y0) / 30);
    const jit = (id) => [vj * vn(hash(seed, boil), id * 7.13), vj * vn(hash(seed, boil + 9), id * 3.77)];
    for (let i = 0; i < n; i++) { const p = PD(s, s.poly[i][0], s.poly[i][1], s.z0), j = jit(i); V.push([p[0] + j[0], p[1] + j[1]]); }
    for (let i = 0; i < n; i++) { const p = PD(s, s.poly[i][0], s.poly[i][1], s.z1), j = jit(n + i); V.push([p[0] + j[0], p[1] + j[1]]); }
    const edge = (a, b) => {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const e = wobble([V[lo], V[hi]], hash(seed, lo * 1000 + hi));
      return a === lo ? e : e.slice().reverse();
    };
    const visible = [];
    for (let i = 0; i < n; i++) {
      const a = s.poly[i], b = s.poly[(i + 1) % n], nx = b[1] - a[1], ny = a[0] - b[0];
      if (nx - ny > 1e-9) visible.push(i);
    }
    const loop = (ids) => { let out = []; ids.forEach((id, i) => { out = out.concat(edge(id, ids[(i + 1) % ids.length]).slice(0, -1)); }); return out; };
    const topIds = Array.from({ length: n }, (_, i) => n + i);
    const hullIds = s.convex ? hull(V) : null;

    // knockout under the fill so hidden lines are really hidden, including where the fill misses the line
    const sil = s.smooth ? wobble(hullIds.map((i) => V[i]), seed, true, 0.7) : s.convex ? loop(hullIds) : null;
    if (sil) { pathOf(sil); g.fillStyle = ST.paper; g.fill(); }
    else { for (const i of visible) { pathOf(loop([i, (i + 1) % n, n + (i + 1) % n, n + i])); g.fillStyle = ST.paper; g.fill(); } pathOf(loop(topIds)); g.fill(); }

    g.save();
    g.translate(ST.shift[0], ST.shift[1]);
    if (s.smooth) {
      // single tone wall from the hull, then the top ellipse
      const wall = sil;
      fillFace(wall, s.key, 'my', seed);
      const top = wobble(topIds.map((i) => V[i]), seed + 3, true, 0.5);
      fillFace(top, s.key, 'top', seed + 1);
    } else {
      for (const i of visible) {
        const a = s.poly[i], b = s.poly[(i + 1) % n];
        const face = (b[1] - a[1]) > (a[0] - b[0]) ? 'px' : 'my';
        fillFace(loop([i, (i + 1) % n, n + (i + 1) % n, n + i]), s.key, face, hash(seed, i));
      }
      fillFace(loop(topIds), s.key, 'top', seed + 1);
    }
    g.restore();
    for (const d of s.decals) decal(s, d);

    const inner = s.tier === 'fine' ? ST.fine : ST.inner;
    const outer = s.tier === 'fine' ? ST.fine * 1.25 : ST.thick;
    const sOver = (pts, w, sd) => {
      if (ST.over <= 0 || pts.length < 2) return drawStroke(pts, w, sd);
      const a = pts[0], b = pts[1], y = pts[pts.length - 1], z = pts[pts.length - 2];
      const L = Math.hypot(y[0] - a[0], y[1] - a[1]), ov = Math.min(ST.over, L * 0.12) * (0.4 + 0.8 * r01(sd, boil));
      const na = Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, nz = Math.hypot(y[0] - z[0], y[1] - z[1]) || 1;
      drawStroke([[a[0] + (a[0] - b[0]) / na * ov, a[1] + (a[1] - b[1]) / na * ov], ...pts,
        [y[0] + (y[0] - z[0]) / nz * ov, y[1] + (y[1] - z[1]) / nz * ov]], w, sd);
    };
    if (s.smooth) {
      drawStroke(wobble(topIds.map((i) => V[i]), seed + 3, true, 0.5).concat([]), inner, seed + 5, true);
      drawStroke(sil.concat([sil[0]]), outer, seed + 6, true);
      return;
    }
    const inHull = new Set();
    if (hullIds) hullIds.forEach((a, i) => { const b = hullIds[(i + 1) % hullIds.length]; inHull.add(Math.min(a, b) * 1000 + Math.max(a, b)); });
    const inners = [];
    for (let i = 0; i < n; i++) inners.push([n + i, n + (i + 1) % n]);
    for (const i of visible) {
      inners.push([i, (i + 1) % n]);
      inners.push([i, n + i]);
      inners.push([(i + 1) % n, n + (i + 1) % n]);
    }
    const seen = new Set();
    for (const [a, b] of inners) {
      const k = Math.min(a, b) * 1000 + Math.max(a, b);
      if (seen.has(k) || inHull.has(k)) continue;
      seen.add(k);
      sOver(edge(a, b), hullIds ? inner : Math.max(inner, outer * 0.7), hash(seed, k));
    }
    if (hullIds) hullIds.forEach((a, i) => { const b = hullIds[(i + 1) % hullIds.length]; sOver(edge(a, b), outer, hash(seed, a * 1000 + b + 1)); });
  }

  function drawPanel(s) {
    const seed = hash(s.id, 77), n = s.pts.length;
    const scr = s.pts.map((p) => PD(s, p[0], p[1], p[2]));
    let path = [];
    for (let i = 0; i < n; i++) path = path.concat(wobble([scr[i], scr[(i + 1) % n]], hash(seed, i), false, 0.7).slice(0, -1));
    pathOf(path); g.fillStyle = PALETTE[s.key] || '#ff00ff'; g.fill();
    if (s.closedLine) drawStroke(path.concat([path[0]]), s.tier === 'fine' ? ST.fine * 1.25 : ST.inner, seed + 1, true);
  }

  function drawTubes(s) {
    const seed = hash(s.id, 99);
    const w = Math.max(2.4, s.width * cam.s);
    const paths = s.paths.map((path, i) => wobble(path.map((p) => P(p[0], p[1], p[2])), hash(seed, i), false, 0.5));
    g.lineCap = 'round'; g.lineJoin = 'round';
    const col = ST.fill === 'tone' ? ST.spot : PALETTE[s.key];
    paths.forEach((pts) => {
      const trace = () => { g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); };
      trace(); g.strokeStyle = ST.ink; g.lineWidth = w + 2 * ST.fine * 0.7; g.stroke();
      trace(); g.strokeStyle = col; g.lineWidth = w; g.stroke();
    });
  }

  // Cast shadow on z = 0: hull of the footprint and the top shifted away from the sun.
  function drawShadow(s) {
    if (s.kind !== 'prism' || s.ground || s.tier === 'fine' || s.z0 > 0.001 && !s.floating) return;
    const k = 0.42, pts = [];
    for (const p of s.poly) { pts.push(raw(p[0], p[1], 0)); pts.push(raw(p[0] + (s.z1) * k, p[1] + (s.z1) * k * 1.2, 0)); }
    const scr = pts.map((r) => [cam.ox + r[0] * cam.s, cam.oy + r[1] * cam.s]);
    pathOf(hull(scr).map((i) => scr[i]));
    if (ST.fill === 'tone') { g.fillStyle = tonePattern(0.4); g.fill(); } else { g.fillStyle = 'rgba(35,43,74,0.16)'; g.fill(); }
  }

  function drawOverlays() {
    for (const o of overlays) {
      if (o.t === 'guide') {
        const a = P(...o.a), b = P(...o.b), L = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const dash = 15, gap = 11;
        let t = 0, i = 0;
        while (t < L) {
          const e = Math.min(t + dash, L);
          const pa = [a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L];
          const pb = [a[0] + (b[0] - a[0]) * e / L, a[1] + (b[1] - a[1]) * e / L];
          drawStroke(wobble([pa, pb], hash(i, 5), false, 0.6), Math.max(ST.fine, 2.4), hash(i++, 6));
          t += dash + gap;
        }
      } else {
        const p = P(...o.p), tx = p[0] + o.dx, ty = p[1] + o.dy;
        drawStroke(wobble([p, [tx, ty]], hash(o.dx, o.dy), false, 0.8), ST.fine, 3);
        g.font = "42px 'DD Hand'";
        g.textBaseline = 'middle'; g.fillStyle = ST.ink;
        g.textAlign = o.dx < 0 ? 'right' : 'left';
        g.fillText(o.str, tx + (o.dx < 0 ? -8 : 8), ty);
      }
    }
  }

  // ---------------------------------------------------------------- runtime
  let grain = null;
  function buildGrain() {
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const q = c.getContext('2d'), im = q.createImageData(256, 256), r = rng(1234);
    for (let i = 0; i < im.data.length; i += 4) {
      const v = 245 + 10 * r();
      im.data[i] = im.data[i + 1] = im.data[i + 2] = v; im.data[i + 3] = 255;
    }
    q.putImageData(im, 0, 0);
    return c;
  }

  const DD = { W, H, box, cyl, prism, rotBox, panel, tubes, rectTop, discTop, text, xMark, xWall, guide, label, rng, hash, r01, PALETTE };
  // Solids created between dyn(key) calls keep a stable seed across frames, so a moving part keeps its own wobble.
  DD.dyn = (key) => { dynKey = key; dynCount = 0; };
  DD.flying = (v) => { flying = v; };
  DD.dance = (v, st) => { danceV = v; danceS = st; };
  DD.swayMode = (mode, h, amp = 1) => { swayTag = mode ? { ride: mode === 'ride', h, amp } : null; };
  // Frame an exact rectangle of the raw projection (units before scaling), left and top edges on the canvas edges.
  DD.cameraRaw = (a, b, c) => { const sc = W / (b - a); cam = { s: sc, ox: -a * sc, oy: -c * sc }; };
  // Frame the isometric projection on a world box: fit the box's width to the canvas, centered on the box.
  DD.camera = (x0, x1, y0, y1, z0, z1, margin = 0) => {
    const c = []; for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [z0, z1]) c.push(raw(x, y, z));
    const a = Math.min(...c.map((r) => r[0])), b = Math.max(...c.map((r) => r[0])), e = Math.min(...c.map((r) => r[1])), f = Math.max(...c.map((r) => r[1]));
    const sc = (W - 2 * margin) / (b - a);
    cam = { s: sc, ox: W / 2 - sc * (a + b) / 2, oy: H / 2 - sc * (e + f) / 2 };
  };
  let spec = null, order = null, canvas = null;

  DD.scene = (s) => { spec = s; };
  DD.liveCount = () => (order ? order.length : 0);
  // Exact draw-order check: for every pair of solids whose hexagons overlap on screen and that a plane separates, the
  // one behind must be drawn first. Returns the count and samples of pairs drawn in the wrong order.
  DD.violations = (t) => {
    DD.frame(t);
    let n = 0, pairs = 0; const out = [];
    const f = (s) => `${s.key}${s.fly ? '*' : ''}[${s.x0.toFixed(1)},${s.x1.toFixed(1)} ${s.y0.toFixed(1)},${s.y1.toFixed(1)} ${s.z0.toFixed(1)},${s.z1.toFixed(1)}]`;
    for (let i = 0; i < order.length; i++) for (let j = i + 1; j < order.length; j++) {
      const A = order[i], B = order[j]; // A is drawn first
      if (A.ground || B.ground || !hexOverlap(A, B)) continue;
      pairs++;
      if (separated(A, B) && !behind(A, B)) { n++; if (out.length < 12) out.push(`drawn first ${f(A)} should be after ${f(B)}`); }
    }
    return { n, pairs, solids: order.length, out };
  };
  // Layer check: painter order is only ambiguous for solids whose AABBs intersect. Lists those pairs for this frame,
  // skipping parts in flight. Zero pairs means every overlap on screen is resolved by a separating plane.
  DD.overlaps = (t) => {
    DD.frame(t);
    const e = 1e-3, out = [], list = order.filter((s) => !s.fly && !s.ground);
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const A = list[i], B = list[j];
      if (A.x1 - e > B.x0 && B.x1 - e > A.x0 && A.y1 - e > B.y0 && B.y1 - e > A.y0 && A.z1 - e > B.z0 && B.z1 - e > A.z0) {
        const f = (s) => `${s.key}[${s.x0.toFixed(1)},${s.x1.toFixed(1)} ${s.y0.toFixed(1)},${s.y1.toFixed(1)} ${s.z0.toFixed(1)},${s.z1.toFixed(1)}]`;
        out.push(`${f(A)} x ${f(B)}`);
      }
    }
    return out;
  };
  let staticCount = 0;
  DD.frame = (t) => {
    boil = Math.floor(t * BOIL_FPS + 1e-6) % BOIL_CYCLE;
    if (spec.update) {
      solids.length = staticCount;
      ordCounter = staticCount;
      spec.update(t);
      dynKey = null; flying = false; swayTag = null;
      order = sorted();
    }
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = ST.paper; g.fillRect(0, 0, W, H);
    for (const s of order) if (s.ground) drawPrism(s);
    if (ST.shadow) for (const s of order) drawShadow(s);
    for (const s of order) { if (s.ground) continue; if (s.kind === 'tubes') drawTubes(s); else if (s.kind === 'panel') drawPanel(s); else drawPrism(s); }
    drawOverlays();
    if (ST.grain !== false) { g.save(); g.globalCompositeOperation = 'multiply'; g.fillStyle = g.createPattern(grain, 'repeat'); g.fillRect(0, 0, W, H); g.restore(); }
  };

  window.DD = DD;
  window.addEventListener('DOMContentLoaded', async () => {
    document.body.style.cssText = 'margin:0;background:#000';
    canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H;
    canvas.style.cssText = 'display:block;width:100vw;max-width:1920px';
    document.body.appendChild(canvas);
    g = canvas.getContext('2d');
    ST = STYLES[params.style || spec.style || 'A'];
    const face = new FontFace('DD Hand', "url('../lib/fonts/xkcd-script.ttf')");
    await face.load(); document.fonts.add(face);
    grain = buildGrain();
    if (spec.boilFps) BOIL_FPS = spec.boilFps;
    spec.build(params);
    if (!spec.manualCamera) fit(spec.margin === undefined ? 70 : spec.margin);
    staticCount = solids.length;
    order = sorted();
    DD.frame(0);
    window.DD_READY = true;
    if (params.play) { const t0 = performance.now(); const loop = () => { DD.frame((performance.now() - t0) / 1000 % (spec.period || 1e9)); requestAnimationFrame(loop); }; loop(); }
  });
})();
