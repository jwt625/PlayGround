// Looping CPO assembly line, station order from the flow slide. Stateless: everything derives from t mod T, and each
// part's fate is a hash of (line, gate, part index mod N), so random rejects repeat exactly and the loop closes.
// Section A: EIC and PIC dies, Known Good Die gate. B: OE build, Known Good OE gate. C: package and system.
// Buffers between sections decouple the rates: 2 dies per OE, 8 OEs per package. Geometry is illustrative.
(() => {
  const { box, cyl, rotBox, panel, tubes, rectTop, text, xMark, xWall, r01, hash } = DD;
  // Clock: the track is 150 BPM and the video runs scene time at 2x, so one beat is 0.8 scene s = 0.4 video s = 12 frames at 30 fps.
  // URL params: bars = loop length; dance=0 keeps the stations still; pace=B doubles the die and OE belts. The package line always runs one package per bar,
  // so two OEs land around the ASIC per beat: the belt moves for the first 8th note, then a landing on every 8th.
  const QS = new URLSearchParams(location.search), DOUBLE = QS.get('pace') === 'B';
  const BEAT = 0.8, BARS = Number(QS.get('bars')) || 8, T = BARS * 4 * BEAT;
  const TA = (DOUBLE ? 4 : 8) * BEAT, TB = DOUBLE ? BEAT / 2 : BEAT, TC = 4 * BEAT, MOVE = 0.125;
  // Each station dances a little differently, by a fixed factor from its position.
  const AMP = (x) => 0.9 + 0.3 * r01(Math.round(x), 3);
  const S16 = BEAT / 4; // 16th note = 3 video frames
  const NA = Math.round(T / TA), NB = Math.round(T / TB), NC = Math.round(T / TC);
  const REJECT = { die: 0.12, oe: 0.16, cpo: 0.2, sys: 0.2 };
  const LINES = [{ y: -140, phase: 5 * BEAT, seed: 101 }, { y: -70, phase: 9 * BEAT, seed: 31 }, { y: 0, phase: 0, seed: 11 }, { y: 70, phase: 17 * BEAT, seed: 57 }, { y: 140, phase: 26 * BEAT, seed: 83 }];
  // Section A runs wafers at a steady speed of one die pitch per 8th, feed to pop in exactly 9 beats: Known Good Die probe, then the dicing saw right next to it.
  const VA = 2.75, XA0 = -98, XG1 = -87, XCUT = -76, XA1 = -69, WR = 4.2, DP = 1.1, XS0 = XA0 + WR + 0.3, XP1 = XA1 - WR - 0.5;
  const DIES = [];
  for (let i = -4; i <= 4; i++) for (let j = -4; j <= 4; j++) {
    const dx = i * DP, dy = j * DP;
    if (Math.hypot(Math.abs(dx) + DP / 2, Math.abs(dy) + DP / 2) < WR - 0.1) DIES.push([dx, dy]);
  }
  const XB0 = -58, PB = 9, XC0 = 14, PC = 15, ZB = 5;
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const ease = (x) => { const u = clamp(x); return u * u * (3 - 2 * u); };
  const mod = (a, n) => ((a % n) + n) % n;

  // Fate tables per line: which part indices each gate rejects, at least one per loop per gate.
  function fates(seed, gate, n, p, lanes = 1) {
    const out = new Array(n * lanes).fill(false);
    let any = false;
    for (let i = 0; i < n * lanes; i++) if (r01(hash(seed, gate), i) < p) { out[i] = true; any = true; }
    if (!any) out[hash(seed, gate + 9) % (n * lanes)] = true;
    return out;
  }

  // Pile slots: a pyramid. Layers are separated in z and cells in a layer in xy, so stacked parts never intersect and
  // the painter order is exact. Slot i is fixed, so reject j of a period always lands in the same place.
  const PILES = {
    die: { cx: 3.5, cy: 3.5, h: 0.95, nx: 4, ny: 3, base: 16, jit: 0.35, xy: 0 },
    oe: { cx: 9, cy: 7.6, h: 2.3, nx: 3, ny: 2, base: 4, jit: 0, xy: 0.5 },
    cpo: { cx: 14.5, cy: 14.5, h: 2.5, nx: 1, ny: 2, base: 3, jit: 0, xy: 0.7 },
    sys: { cx: 15, cy: 16.2, h: 5.7, nx: 1, ny: 2, base: 2, jit: 0, xy: 0.5 },
  };
  function slotOf(name, i, seed) {
    const P = { minCells: 2, ...PILES[name] };
    let nx = P.nx, ny = P.ny, layer = 0, rest = i;
    while (rest >= nx * ny) { rest -= nx * ny; layer++; if (nx * ny > P.minCells) { if (nx >= ny && nx > 1) nx--; else if (ny > 1) ny--; } }
    const ix = rest % nx, iy = Math.floor(rest / nx);
    return { dx: (ix - (nx - 1) / 2) * P.cx + P.xy * (2 * r01(seed + 5, i) - 1), dy: (iy - (ny - 1) / 2) * P.cy + P.xy * (2 * r01(seed + 6, i) - 1), z: layer * P.h, a: P.jit * (2 * r01(seed, i) - 1) };
  }

  // ---------------------------------------------------------------- parts, cx cy z = center of the footprint on its seat
  const die = (kind, x, y, z, a = 0, bad = false, sc = 1) => {
    const s = rotBox(x, y, 1.3 * sc, 1.3 * sc, z, z + 0.9, a, kind, { tier: 'fine' });
    if (bad) xMark(s, x, y, 0.9 * sc);
  };
  const rot = (x, y, a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
  // OE after the ingested references: EIC face-down on the PIC, FAU on the PIC beside it (PIC 6 x 10.7 mm, EIC about
  // 4.6 x 5.8 mm on the ISSCC 2026 stack drawing; proportions kept, heights exaggerated). Before the PIC arrives the
  // EIC sits on the carrier, per the slide's EEngine-then-OEngine order; lift raises it for the PIC to slide under.
  function oe(x, y, z, stage, a = 0, bad = false, lift = 0) {
    rotBox(x, y, 3.9, 3.1, z, z + 0.6, a, 'carrier', { tier: 'fine' });
    const part = (dx, dy, hx, hy, z0, h, key) => { const r = rot(dx, dy, a); return rotBox(x + r[0], y + r[1], hx, hy, z + z0, z + z0 + h, a, key, { tier: 'fine' }); };
    if (stage >= 2) {
      part(0, 0, 3.2, 1.8, 0.6, 0.7, 'pic');
      const e = part(-1.6, 0, 1.4, 1.7, 1.3, 0.9, 'eic');
      part(1.6, 0, 1.3, 1.3, 1.3, 0.7, 'glass');
      part(3.55, 0, 0.35, 0.9, 0.6, 0.7, 'ferrule');
      if (bad) { const r = rot(-1.6, 0, a); xMark(e, x + r[0], y + r[1], 1); }
    } else if (stage >= 1) part(-1.6, 0, 1.4, 1.7, 0.6 + lift, 0.9, 'eic');
  }
  const FULLPKG = { asic: true, oes: 8, lid: true };
  const SITES = [[-2.1, 4.9], [2.1, 4.9], [-2.1, -4.9], [2.1, -4.9], [-4.9, -2.1], [-4.9, 2.1], [4.9, -2.1], [4.9, 2.1]];
  function pkg(x, y, z, st, a = 0, bad = false) { // st: {asic, oes, lid}
    const c = rotBox(x, y, 6.5, 6.5, z, z + 0.8, a, 'carrier');
    const zt = z + 0.8;
    const part = (dx, dy, hx, hy, z0, z1, key) => { const r = rot(dx, dy, a); return rotBox(x + r[0], y + r[1], hx, hy, z0, z1, a, key, { tier: 'fine' }); };
    for (let i = 0; i < (st.oes || 0); i++) {
      const [dx, dy] = SITES[i], side = Math.abs(dx) > 3;
      part(dx, dy, side ? 1 : 1.7, side ? 1.7 : 1, zt, zt + 0.6, 'pic');
      const off = 0.55, ex = side ? dx - Math.sign(dx) * off : dx, ey = side ? dy : dy - Math.sign(dy) * off;
      part(ex, ey, side ? 0.45 : 1.25, side ? 1.25 : 0.45, zt + 0.6, zt + 1.1, 'eic');
    }
    if (st.lid) {
      part(0, -6.25, 6.5, 0.25, zt, zt + 1.6, 'lid');
      part(0, 6.25, 6.5, 0.25, zt, zt + 1.6, 'lid');
      part(-6.25, 0, 0.25, 6, zt, zt + 1.6, 'lid');
      part(6.25, 0, 0.25, 6, zt, zt + 1.6, 'lid');
    }
    const top = st.asic ? part(0, 0, 2.4, 2.4, zt, zt + 1.4, 'asic') : c;
    if (bad) xMark(top, x, y, st.asic ? 1.7 : 3.2);
  }
  function fibers(x, y, z, n) { // OE tops to the front (+X) edge of the board
    if (n <= 0) return;
    const zt = z + 2.6, paths = [];
    for (let i = 0; i < n; i++) {
      const [dx, dy] = SITES[i], ex = x + 6.8, ey = y - 5.5 + 11 * i / 7, path = [];
      for (let j = 0; j <= 8; j++) {
        const u = j / 8, e = u * u;
        path.push([x + dx + (ex - x - dx) * u, y + dy + (ey - y - dy) * e, zt + 2.2 * Math.sin(u * 3.14)]);
      }
      paths.push(path);
    }
    tubes(paths, 0.22, 'fiber');
  }
  function sysBox(x, y, z, a = 0, bad = false) { // a rejected closed system, for the pile
    const s = rotBox(x, y, 6.8, 7.5, z, z + 5.5, a, 'chassis');
    if (bad) { // top and both visible walls, so a box stacked under another still shows its cross
      xMark(s, x, y, 3.2);
      xWall(s, 'my', x, z + 2.75, 1.8);
      xWall(s, 'px', y, z + 2.75, 1.8);
    }
  }
  // Open carton as a real tube of four walls with hinged flaps. phi: flat-packed parallelogram (small) to square (90 deg).
  // flaps: hinge angle per pair, 0 = pointing straight out, 180 = folded flat inward.
  const BX0 = -6.85, BX1 = 8, BY = 7.7, BH = 6.4;
  function cartonOpen(x, y, z, phi, flapX, flapY) {
    const W = BX1 - BX0, D = 2 * BY, cx = x + (BX0 + BX1) / 2;
    const dx = D * Math.cos(phi), dy = D * Math.sin(phi);
    const c = [[-W / 2 - dx / 2, -dy / 2], [W / 2 - dx / 2, -dy / 2], [W / 2 + dx / 2, dy / 2], [-W / 2 + dx / 2, dy / 2]].map((q) => [cx + q[0], y + q[1]]);
    const top = z + BH;
    for (let i = 0; i < 4; i++) {
      const a = c[i], b = c[(i + 1) % 4], ex = b[0] - a[0], ey = b[1] - a[1], L = Math.hypot(ex, ey), nx = ey / L, ny = -ex / L;
      panel([[a[0], a[1], z], [b[0], b[1], z], [b[0], b[1], top], [a[0], a[1], top]], nx - ny > 0 ? 'carton' : 'cartonIn');
      const yEdge = i % 2 === 0, beta = yEdge ? flapY : flapX, len = yEdge ? BY - 0.02 : 5, zf = top + (yEdge ? 0.03 : 0);
      const fx = nx * Math.cos(beta) * len, fy = ny * Math.cos(beta) * len, fz = Math.sin(beta) * len;
      panel([[a[0], a[1], zf], [b[0], b[1], zf], [b[0] + fx, b[1] + fy, zf + Math.max(0, fz)], [a[0] + fx, a[1] + fy, zf + Math.max(0, fz)]], 'carton');
    }
  }
  function carton(x, y, z) { // the closed shipping box; its footprint fits the 15-unit pitch
    const c = box(x - 6.85, y - 7.7, z, x + 8, y + 7.7, z + 6.4, 'carton');
    rectTop(c, x - 6.85, y - 0.9, x + 8, y + 0.9, 'tape');
    text(c, 'outside\nfive sigma', [x + 0.575, y - 7.7, z + 3.2], 'my', 1.9, 'wm');
  }
  function system(x, y, z, st) { // st: {oes, fibers, cover, modules}; board with the package on it, then the closed box
    const b = box(x - 6.8, y - 7.5, z, x + 6.8, y + 7.5, z + 0.9, 'board');
    if (!st.cover) {
      pkg(x, y, z + 0.9, FULLPKG);
      fibers(x, y, z + 0.9, st.fibers);
      return b;
    }
    const c = box(x - 6.8, y - 7.5, z + 0.9, x + 6.8, y + 7.5, z + 5.5, 'chassis');
    for (let i = 0; i < st.modules; i++) {
      const py = y - 6 + i * 3;
      box(x + 6.8, py - 1, z + 2, x + 7.9, py + 1, z + 3.8, 'module', { tier: 'fine' });
    }
    return c;
  }

  // ---------------------------------------------------------------- static machines of one line
  const stations = [];
  function column(ly, x, label, key, halfW, y0, h, reach) {
    DD.swayMode('bend', h, AMP(x));
    const c = box(x - halfW, ly + y0, 0, x + halfW, ly + y0 + 6, h, key);
    DD.swayMode('ride', h, AMP(x));
    text(c, label, [x + 1.7, ly + y0, h * 0.58], 'my', 1.9);
    box(x - 1.2, ly - reach, h - 2.6, x + 1.2, ly + y0, h, key, { tier: 'fine' });
    DD.swayMode(null);
  }
  function belt(ly, x0, x1, hw, key = 'belt') {
    box(x0, ly - hw, ZB - 1, x1, ly + hw, ZB, key);
    box(x0, ly - hw - 0.8, ZB - 2.2, x1, ly - hw, ZB + 0.4, 'rail', { tier: 'fine' });
    box(x0, ly + hw, ZB - 2.2, x1, ly + hw + 0.8, ZB + 0.4, 'rail', { tier: 'fine' });
    for (let x = x0 + 3; x < x1; x += 22) {
      box(x - 0.8, ly - hw - 0.8, 0, x + 0.8, ly - hw + 0.8, ZB - 2.2, 'rail', { tier: 'fine' });
      box(x - 0.8, ly + hw - 0.8, 0, x + 0.8, ly + hw + 0.8, ZB - 2.2, 'rail', { tier: 'fine' });
    }
  }
  const PILE = { die: [XCUT, -11.5], oe: [XB0 + 5 * PB, -14.5], cpo: [XC0 + 4 * PC, -25], sys: [XC0 + 8 * PC, -27] };

  function buildLine(L) {
    const ly = L.y;
    // Section A: wafers feed, two lanes
    const feed = box(XA0 - 8, ly - 6, 0, XA0, ly + 6, 9, 'frame');
    text(feed, 'EIC  PIC', [XA0 - 4, ly - 6, 5], 'my', 2.15);
    belt(ly, XA0, XA1, 4.4);
    column(ly, XG1, 'KNOWN\nGOOD DIE', 'frame2', 5.4, 7, 15, 4);
    column(ly, XCUT, 'DICING', 'frame', 5.4, 7, 15, 4);
    box(XA1, ly - 5, 0, XA1 + 7, ly + 5, 6.4, 'tray');
    // Section B
    box(XB0 - 2.5, ly + 5.2, 0, XB0 + 2.5, ly + 12, 9, 'rail');
    belt(ly, XB0 - 4, XB0 + 6 * PB + 4, 4.4);
    column(ly, XB0 + PB, 'EENGINE\nASSEMBLY', 'frame', 5.8, 7, 15, 3);
    column(ly, XB0 + 3 * PB, 'OENGINE\nASSEMBLY', 'frame', 5.8, 7, 15, 3);
    column(ly, XB0 + 5 * PB, 'KNOWN\nGOOD OE', 'frame2', 5.8, 7, 15, 4);
    box(XB0 + 6 * PB + 4, ly - 5, 0, XB0 + 6 * PB + 10, ly + 5, 6.4, 'tray');
    // Section C
    box(XC0 - 2, ly + 10, 0, XC0 + 6, ly + 15.5, 9, 'rail');
    belt(ly, XC0 - 7, XC0 + 14 * PC, 8.6);
    const labels = [[1, 'ASIC TO\nSUBSTRATE', 'frame'], [2, 'OENGINE TO\nSUBSTRATE', 'frame'], [3, 'FINAL CPO\nASSY + TEST', 'frame'], [4, 'KNOWN\nGOOD CPO', 'frame2'],
      [5, 'CPO TO\nSYSTEM', 'frame'], [6, 'FIBER\nROUTING', 'frame'], [7, 'FINAL ASSY\n+ TEST', 'frame'], [8, 'KNOWN\nGOOD SYSTEM', 'frame2'], [9, 'BOX +\nSHIP', 'frame']];
    for (const [i, lab, key] of labels) if (i < 9) column(ly, XC0 + i * PC, lab, key, 6.8, 10.5, 19, 2);
    { // Boxing station: taller, so the flaps clear the arm; a cross beam carries the tape head along the seam
      const x = XC0 + 9 * PC;
      DD.swayMode('bend', 26, AMP(x));
      const c = box(x - 6.8, ly + 10.5, 0, x + 6.8, ly + 16.5, 26, 'frame');
      DD.swayMode('ride', 26, AMP(x));
      text(c, 'BOX +\nSHIP', [x + 1.7, ly + 10.5, 15], 'my', 1.9);
      box(x - 1.2, ly + 1.2, 23.4, x + 1.2, ly + 10.5, 26, 'frame', { tier: 'fine' });
      box(x - 8.5, ly - 1.2, 23.4, x + 9.5, ly + 1.2, 26, 'frame', { tier: 'fine' });
      DD.swayMode(null);
    }
    // Feeders in front of the belt
    const asicTray = box(XC0 + PC - 4, ly - 17, 0, XC0 + PC + 4, ly - 11.5, 3, 'tray');
    for (const dx of [-2, 2]) box(XC0 + PC + dx - 1.4, ly - 15.6, 3, XC0 + PC + dx + 1.4, ly - 12.8, 3.7, 'asic', { tier: 'fine' });
    text(asicTray, 'ASIC', [XC0 + PC, ly - 17, 1.5], 'my', 2);
    box(XC0 + 3 * PC - 5, ly - 17, 0, XC0 + 3 * PC + 5, ly - 11.5, 2.5, 'lid');
    box(XC0 + 5 * PC - 6, ly - 18, 0, XC0 + 5 * PC + 6, ly - 11.5, 2.8, 'board');
    const modTray = box(XC0 + 7 * PC - 5, ly - 17, 0, XC0 + 7 * PC + 5, ly - 11.5, 3, 'tray');
    text(modTray, 'LASER', [XC0 + 7 * PC, ly - 17, 1.5], 'my', 2);
    box(XC0 + 7 * PC - 3.6, ly - 15.4, 3, XC0 + 7 * PC - 0.6, ly - 13, 4.5, 'module', { tier: 'fine' });
    box(XC0 + 7 * PC + 0.6, ly - 15.4, 3, XC0 + 7 * PC + 3.6, ly - 13, 4.5, 'module', { tier: 'fine' });
    DD.swayMode('ride', 19, AMP(XC0 + 6 * PC));
    cyl(XC0 + 6 * PC + 3, ly + 13.5, 19, 21.4, 3, 'fiber', { tier: 'fine' });
    DD.swayMode(null);
    box(XC0 + 9 * PC - 6, ly - 18, 0, XC0 + 9 * PC + 6, ly - 11.5, 2.8, 'carton');
    // Base piles: always there, under the live slots
    for (const name of ['die', 'oe', 'cpo', 'sys']) {
      const [px, py] = PILE[name];
      for (let i = 0; i < PILES[name].base; i++) {
        const q = slotOf(name, i, L.seed), x = px + q.dx, y = ly + py + q.dy;
        if (name === 'die') die(r01(L.seed + 3, i) > 0.5 ? 'eic' : 'pic', x, y, q.z, q.a, true);
        if (name === 'oe') oe(x, y, q.z, 2, q.a, true);
        if (name === 'cpo') pkg(x, y, q.z, FULLPKG, q.a, true);
        if (name === 'sys') sysBox(x, y, q.z, q.a, true);
      }
    }
    // Part caches
    L.fate = {
      die: fates(L.seed, 1, NA, REJECT.die, DIES.length), oe: fates(L.seed, 2, NB, REJECT.oe),
      cpo: fates(L.seed, 3, NC, REJECT.cpo), sys: fates(L.seed, 4, NC, REJECT.sys),
    };
    L.slot = {};
    for (const k of ['die', 'oe', 'cpo', 'sys']) {
      const idx = []; L.fate[k].forEach((b, i) => { if (b) idx.push(i); });
      L.slot[k] = new Map(idx.map((v, j) => [v, j]));
      L[`${k}Heap`] = idx.map((_, j) => slotOf(k, PILES[k].base + j, L.seed));
    }
    // FAIL blinks per wafer, in 16ths of the 4-beat probe window: each bad die holds 2 16ths from when it crosses the probe, then at
    // least 1 16th of PASS, pulled earlier if the window would overflow. The window fits 5; further bad dies on the wafer share a blink.
    L.blink = Array.from({ length: NA }, (_, km) => {
      const out = [];
      for (let i = DIES.length - 1, free = 0; i >= 0 && out.length < 5; i--) if (L.fate.die[km * DIES.length + i]) {
        const s = Math.max(free, Math.ceil((XG1 - DIES[i][0] - XS0 - VA * BEAT) / VA / S16));
        out.push(s); free = s + 3;
      }
      return out.map((s, j) => Math.min(s, 14 - 3 * (out.length - 1 - j)));
    });
  }

  const hop = (a, b, u, h) => {
    const e = clamp(u);
    return [a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e, a[2] + (b[2] - a[2]) * e + h * 4 * e * (1 - e)];
  };
  function fly(key, fn) { DD.dyn(key); DD.flying(true); fn(); DD.flying(false); }
  const lamp = (x, y, z, key) => cyl(x, y, z, z + 2, 1.3, key, { tier: 'fine' });
  function gateTop(x, ly, y0, h, bad, active) { // lamp and result screen, on top of a gate column
    DD.swayMode('ride', h, AMP(x));
    lamp(x + 3.5, ly + y0 + 3, h, active ? (bad ? 'ng' : 'ok') : 'lampoff');
    const s = box(x - 4, ly + y0 + 1, h, x + 1.5, ly + y0 + 4, h + 3.6, 'screen', { tier: 'fine' });
    if (active) text(s, bad ? 'FAIL' : 'PASS', [x - 1.25, ly + y0 + 1, h + 1.8], 'my', 2.3, bad ? 'ng' : 'trace');
    DD.swayMode(null);
  }

  // ---------------------------------------------------------------- per-frame state of one line
  function updateLine(L, li, t0) {
    const t = mod(t0 + L.phase + 1e-9, T), ly = L.y; // the nudge keeps a frame on a beat from rounding to just before it, where cycle and phase disagree
    const K = (name, k) => hash(hash(li, name), k);

    // Section A. Wafers ride the belt; EIC and PIC alternate. dt counts from the hop, on beat 6.5 of the wafer's 8-beat period:
    // lands on 7, probe bar on 8-12, saw bar on 12-16 (16 cuts on 16ths), pop on 16, the period's downbeat.
    const HOP = BEAT / 2, W0 = 6.5 * BEAT, tProbe = HOP + BEAT, tCut0 = tProbe + 4 * BEAT, tPop = tCut0 + 4 * BEAT, DN = DIES.length;
    const FLY = BEAT, SPREAD = 2 * BEAT, CUTS = 16;
    let gateBad = false, probing = false, cutting = null, probeFiber = false;
    for (let k = Math.floor((t - W0 - tPop - SPREAD - FLY - 1) / TA); k <= Math.floor(t / TA); k++) {
      const dt = t - W0 - k * TA, km = mod(k, NA), kind = km % 2 ? 'pic' : 'eic', key = K('w', km);
      if (dt < 0) continue;
      if (dt < tPop) {
        const x = dt < HOP ? XS0 : XS0 + VA * (dt - HOP);
        const p = dt < HOP ? hop([XA0 - 4, ly, 9], [XS0, ly, ZB], dt / HOP, 1.5) : [x, ly, ZB];
        const draw = () => {
          const w = cyl(p[0], p[1], p[2], p[2] + 0.6, WR, 'lid', { tier: 'fine' });
          DIES.forEach(([dx, dy], i) => {
            rectTop(w, p[0] + dx - 0.47, p[1] + dy - 0.47, p[0] + dx + 0.47, p[1] + dy + 0.47, kind);
            if (p[0] + dx > XG1) DD.discTop(w, p[0] + dx, p[1] + dy, 0.3, L.fate.die[km * DN + i] ? 'ng' : 'ok');
          });
          if (dt > tCut0) { // saw lines: 8 along the belt, then 8 across
            const u = clamp((dt - tCut0) / (tPop - tCut0)) * CUTS, n = Math.floor(u), f = u - n;
            for (let c = 0; c <= Math.min(n, CUTS - 1); c++) {
              const q = c < 8 ? (c - 3.5) * DP : (c - 11.5) * DP, e = c === n ? f : 1, half = Math.sqrt(Math.max(1, WR * WR - q * q));
              if (c < 8) rectTop(w, p[0] - half, p[1] + q - 0.05, p[0] - half + 2 * half * e, p[1] + q + 0.05, 'ink');
              else rectTop(w, p[0] + q - 0.05, p[1] - half, p[0] + q + 0.05, p[1] - half + 2 * half * e, 'ink');
            }
          }
        };
        if (dt < HOP) fly(key, draw); else { DD.dyn(key); draw(); }
        if (dt >= tProbe && dt < tCut0) {
          probing = true; probeFiber = kind === 'pic';
          const n = Math.floor((dt - tProbe) / S16), b = L.blink[km];
          gateBad = b.includes(n) || b.includes(n - 1);
        }
        if (dt > tCut0) cutting = { x, u: clamp((dt - tCut0) / (tPop - tCut0)) * CUTS };
      } else {
        const x = XP1;
        DIES.forEach(([dx, dy], i) => {
          const id = km * DN + i, bad = L.fate.die[id], key2 = K('wd', id), rel = i % 8 ? Math.floor(8 * r01(hash(L.seed, 7), id)) : 0, u = (dt - tPop - rel * S16) / FLY; // released on 16ths over 2 beats, some on the downbeat
          if (u < 0) { DD.dyn(key2); die(kind, x + dx, ly + dy, ZB, 0, bad, 0.36); return; }
          if (u > 1) return;
          // Pop big in flight, land at true size. Bad dies are thrown high onto the pile, good ones into the tray.
          const big = 0.36 + 0.64 * u + 2.6 * Math.sin(Math.PI * u);
          if (bad) {
            const q = L.dieHeap[L.slot.die.get(id)], [px, py] = PILE.die, r = hop([x + dx, ly + dy, ZB], [px + q.dx, ly + py + q.dy, q.z], u, 11);
            fly(key2, () => die(kind, r[0], r[1], r[2], q.a * u + 1.2 * (1 - u), true, big));
          } else {
            const r = hop([x + dx, ly + dy, ZB], [XA1 + 3.5, ly + 2 * Math.sin(i), 6.4], u, 9);
            fly(key2, () => die(kind, r[0], r[1], r[2], 0, false, big));
          }
        });
      }
    }
    // Live pile slots, always drawn: a reject lands on its own identical copy
    L.slot.die.forEach((j, id) => {
      const q = L.dieHeap[j], [px, py] = PILE.die; DD.dyn(K('dp', id));
      die(mod(Math.floor(id / DN), 2) ? 'pic' : 'eic', px + q.dx, ly + py + q.dy, q.z, q.a, true);
    });
    // Probe head: comically fast raster over the passing wafer
    DD.dyn(K('ga', 0));
    {
      DD.swayMode('ride', 15, AMP(XG1));
      const sweep = probing ? 3.6 * Math.sin(2 * Math.PI * t / S16) : 0, jx = probing ? 0.8 * Math.sin(2 * Math.PI * t / (2 * S16)) : 0;
      const px = XG1 + jx, py = ly + sweep, z0 = ZB + 1.6 + (probing ? 0 : 1.5);
      box(px - 0.45, py - 0.45, z0 + 0.5, px + 0.45, py + 0.45, 12.4, 'rail', { tier: 'fine' });
      box(px - 0.8, py - 0.8, z0, px + 0.8, py + 0.8, z0 + 0.5, 'pad', { tier: 'fine' });
      if (probing && probeFiber) tubes([[[px + 1.5, py, z0 + 1.2], [px + 1.1, py, z0 + 0.2], [px + 1.0, py, ZB + 0.7]]], 0.14, 'fiber');
      DD.swayMode(null);
    }
    L.gateBad = gateBad && probing;
    gateTop(XG1, ly, 7, 15, gateBad, probing);
    // Dicing head: blade zips along each saw line
    {
      let bx = XCUT, by = ly, across = false, spinning = false;
      if (cutting) {
        const n = Math.min(Math.floor(cutting.u), CUTS - 1), f = cutting.u - n, q = n < 8 ? (n - 3.5) * DP : (n - 11.5) * DP, half = Math.sqrt(Math.max(1, WR * WR - q * q));
        across = n >= 8; spinning = true;
        bx = n < 8 ? cutting.x - half + 2 * half * f : cutting.x + q;
        by = n < 8 ? ly + q : ly - half + 2 * half * f;
        bx = Math.max(bx, XG1 + 2.6); // stays clear of the probe head
      }
      const zc = ZB + (spinning ? 2.2 : 4), R = 1.5, teeth = [], a0 = 2 * Math.PI * 1.25 * t / S16;
      for (let i = 0; i < 16; i++) {
        const r = R * (i % 2 ? 1 : 0.84), a = a0 + i / 16 * 2 * Math.PI, c = r * Math.cos(a), v = zc + r * Math.sin(a);
        teeth.push(across ? [bx, by + c, v] : [bx + c, by, v]);
      }
      DD.swayMode('ride', 15, AMP(XCUT));
      box(bx - 0.4, by - 0.4, zc + R, bx + 0.4, by + 0.4, 12.4, 'rail', { tier: 'fine' });
      DD.swayMode(null);
      panel(teeth, spinning ? 'rail' : 'lid');
    }

    // Section B. Index motion: the belt moves on the off-beat 8th and stops on the beat, as the dies land and the heads hit. tB and
    // fB count from that off-beat, so fB = 0.5 is the beat.
    const tB = t + TB / 2, fB = mod(tB, TB) / TB, dwellB = (fB - 0.5) / 0.5;
    for (let k = Math.floor(tB / TB) - 8; k <= Math.floor(tB / TB); k++) {
      const dt = tB - k * TB, c = Math.floor(dt / TB), km = mod(k, NB), bad = L.fate.oe[km];
      const idx = c + ease(fB / 0.5), x = XB0 + PB * idx, key = K('o', km);
      const stage = c > 2 || (c === 2 && dwellB >= 0) ? 2 : c > 0 || (c === 0 && dwellB >= 0) ? 1 : 0;
      const lift = c === 2 && stage === 1 ? 1.2 * ease(fB / 0.5) : 0;
      if (c === 0) { // EIC hops in from the die tray
        const p = hop([XA1 + 3.5, ly, 6.4], [XB0 + PB - 2, ly, ZB + 1], fB / 0.5, 9);
        if (dwellB < 0) fly(K('oe', km), () => die('eic', p[0], p[1], p[2], 0));
      }
      if (c === 2) { // PIC hops in from the die tray
        const p = hop([XA1 + 3.5, ly, 6.4], [XB0 + 3 * PB + 1, ly, ZB + 1], fB / 0.5, 13);
        if (dwellB < 0) fly(K('op', km), () => die('pic', p[0], p[1], p[2], 0));
      }
      if (bad && c >= 4 && (c > 4 || dwellB >= 0.5)) { // rejected at the gate, 16th after the beat, lands on the next beat
        const q = L.oeHeap[L.slot.oe.get(km)], [px, py] = PILE.oe;
        const u = (dt - 4.75 * TB) / (3 * S16);
        if (u <= 1) {
          const p = hop([XB0 + 5 * PB, ly, ZB], [px + q.dx, ly + py + q.dy, q.z], u, 7);
          fly(key, () => oe(p[0], p[1], p[2], 2, q.a * u, true));
        }
      } else if (c <= 5) {
        DD.dyn(key);
        oe(x, ly, ZB, stage, 0, false, lift);
      } else if (c === 6) { // lands in the OE tray
        const p = hop([XB0 + 6 * PB + 0, ly, ZB], [XB0 + 6 * PB + 7, ly, 6.4], fB / 0.5, 3);
        if (fB < 0.5) fly(key, () => oe(p[0], p[1], p[2], 2));
      }
      if (c === -1 && fB > 0.7) { // new carrier from the magazine
        const p = hop([XB0, ly + 8, 9], [XB0, ly, ZB], (fB - 0.7) / 0.3, 2);
        fly(K('oc', km), () => oe(p[0], p[1], p[2], 0));
      }
    }
    L.slot.oe.forEach((j, km) => {
      const q = L.oeHeap[j], [px, py] = PILE.oe; DD.dyn(K('op2', km));
      oe(px + q.dx, ly + py + q.dy, q.z, 2, q.a, true);
    });
    DD.dyn(K('gb', 0));
    const hB = 2.4 * Math.min(1, Math.abs(fB - 0.5) * 4); // heads hit on the beat, up by a 16th either side
    for (const [slot, gate] of [[1, false], [3, false], [5, true]]) {
      const x = XB0 + slot * PB, z0 = ZB + 3 + hB;
      DD.swayMode('ride', 15, AMP(x));
      box(x - 0.9, ly - 0.9, z0 + 0.8, x + 0.9, ly + 0.9, 12.4, 'rail', { tier: 'fine' });
      if (gate) box(x - 3, ly - 2.6, z0, x + 3, ly + 2.6, z0 + 0.8, 'pad', { tier: 'fine' });
      else box(x - 1.4, ly - 1.4, z0, x + 1.4, ly + 1.4, z0 + 0.8, 'ink', { tier: 'fine' });
    }
    DD.swayMode(null);
    const atGate = Math.floor(tB / TB) - 4, gbad = L.fate.oe[mod(atGate, NB)];
    gateTop(XB0 + 5 * PB, ly, 7, 15, gbad, dwellB >= 0);
    if (gbad && dwellB >= 0.5) box(XB0 + 5 * PB - 0.6, ly + 3.2, ZB + 0.5, XB0 + 5 * PB + 0.6, ly + 7, ZB + 1.2, 'ink', { tier: 'fine' });

    // Section C, one package per bar. b counts beats, 0 to 4. The belt moves for the first 8th. Every part moves fast, in an
    // 8th or so, and lands with its arm's hit, then idles. Arms hit one per beat, left to right: arm s hits on beat s % 4 + 1.
    const fC = mod(t, TC) / TC, b = fC * 4, q8 = fC * 8;
    const Xs = (s) => XC0 + PC * s;
    for (let k = Math.floor(t / TC) - 15; k <= Math.floor(t / TC); k++) {
      const dt = t - k * TC, c = Math.floor(dt / TC + 1e-9), km = mod(k, NC);
      const idx = c + ease(fC / MOVE), x = XC0 + PC * idx, key = K('c', km), y = ly;
      const badC = L.fate.cpo[km], badS = L.fate.sys[km];
      if (c === 0 && b >= 1.5 && b < 2) { // ASIC lands on arm 1's hit, beat 2
        const p = hop([Xs(1), ly - 14, 3.7], [Xs(1) - 2.4, y - 2.4, ZB + 0.8], (b - 1.5) / 0.5, 11);
        fly(K('ca', km), () => box(p[0], p[1], p[2], p[0] + 4.8, p[1] + 4.8, p[2] + 1.4, 'asic', { tier: 'fine' }));
      }
      const asic = c > 0 || (c === 0 && b >= 2);
      let oes = c > 1 ? 8 : 0;
      if (c === 1) { // 8 OEs fly in from the OE tray, one landing on every 8th note: two per beat
        const i = Math.min(Math.floor(q8), 8);
        oes = i;
        if (i < 8) {
          const [dx, dy] = SITES[i], p = hop([XB0 + 6 * PB + 7, ly, 6.4], [Xs(2) + dx, y + dy, ZB + 0.8], q8 - i, 16);
          fly(K('cf', km * 8 + i), () => oe(p[0], p[1], p[2], 2));
        }
      }
      const lid = c > 2;
      if (c === 2 && b >= 3.5) { // stiffener ring lands on arm 3's hit, beat 4
        const p = hop([Xs(3), ly - 14, 2.5], [Xs(3), y, ZB + 0.8], (b - 3.5) / 0.5, 12);
        fly(K('cl', km), () => {
          box(p[0] - 6.5, p[1] - 6.5, p[2], p[0] + 6.5, p[1] - 6, p[2] + 1.6, 'lid', { tier: 'fine' });
          box(p[0] - 6.5, p[1] + 6, p[2], p[0] + 6.5, p[1] + 6.5, p[2] + 1.6, 'lid', { tier: 'fine' });
          box(p[0] - 6.5, p[1] - 6, p[2], p[0] - 6, p[1] + 6, p[2] + 1.6, 'lid', { tier: 'fine' });
          box(p[0] + 6, p[1] - 6, p[2], p[0] + 6.5, p[1] + 6, p[2] + 1.6, 'lid', { tier: 'fine' });
        });
      }
      const isSys = c > 4 || (c === 4 && b >= 2);
      if (c === 4 && !badC && b >= 1.5 && b < 2) { // system board lands on arm 5's hit, beat 2
        const p = hop([Xs(5), ly - 14.5, 2.8], [Xs(5), y, ZB], (b - 1.5) / 0.5, 12);
        fly(K('cb', km), () => box(p[0] - 6.8, p[1] - 7.5, p[2], p[0] + 6.8, p[1] + 7.5, p[2] + 0.9, 'board'));
      }
      const fib = c > 5 ? 8 : c === 5 ? Math.floor(clamp((b - 2.5) / 0.5) * 8) : 0; // fibers burst in, done on arm 6's hit, beat 3
      const cover = c > 6 || (c === 6 && b >= 3.5);
      const mods = c > 6 ? 5 : c === 6 ? Math.floor(clamp((b - 3.5) / 0.5) * 5) : 0;
      if (c === 6 && !badC && b >= 3 && b < 3.5) { // chassis cover drops
        const zc = ZB + 0.9 + 9 * (1 - ease((b - 3) / 0.5));
        fly(K('cv', km), () => box(Xs(7) - 6.8, y - 7.5, zc, Xs(7) + 6.8, y + 7.5, zc + 4.6, 'chassis'));
      }
      if (c === 6 && b >= 3.5) { // laser modules plug in, done on arm 7's hit, beat 4
        const n = (b - 3.5) / 0.5 * 5, i = Math.floor(n);
        if (i < 5) {
          const p = hop([Xs(7), ly - 14, 4.5], [Xs(7) + 7.35, y - 6 + i * 3, ZB + 2], n - i, 8);
          fly(K('cm', km * 5 + i), () => box(p[0] - 0.55, p[1] - 1, p[2], p[0] + 0.55, p[1] + 1, p[2] + 1.8, 'module', { tier: 'fine' }));
        }
      }
      const rejC = badC && (c > 3 || (c === 3 && b >= 1.5)); // arm 4 tests on beat 1, thrown on beat 1.5
      const rejS = !rejC && badS && (c > 7 || (c === 7 && b >= 1.5));
      if (rejC) {
        const q = L.cpoHeap[L.slot.cpo.get(km)], [px, py] = PILE.cpo;
        const u = (dt - 3 * TC - 1.5 * BEAT) / (0.75 * BEAT);
        if (u <= 1) {
          const p = hop([Xs(4), ly, ZB], [px + q.dx, ly + py + q.dy, q.z], u, 10);
          fly(key, () => pkg(p[0], p[1], p[2], FULLPKG, q.a * u, true));
        }
      } else if (rejS) {
        const q = L.sysHeap[L.slot.sys.get(km)], [px, py] = PILE.sys;
        const u = (dt - 7 * TC - 1.5 * BEAT) / (0.75 * BEAT);
        if (u <= 1) {
          const p = hop([Xs(8), ly, ZB], [px + q.dx, ly + py + q.dy, q.z], u, 10);
          fly(key, () => sysBox(p[0], p[1], p[2], q.a * u, true));
        }
      } else if (c >= 0 && c <= 14) { // past slot 9 the closed box rides the belt out of the frame
        DD.dyn(key);
        const closed = c > 8 || (c === 8 && b >= 2.5);
        if (closed) carton(x, y, ZB);
        else if (isSys) system(x, y, ZB, { fibers: fib, cover, modules: mods });
        else pkg(x, y, ZB, { asic, oes, lid });
        if (c === 8 && !closed && b >= 0.5) { // carton opens in the air, drops over the system, flaps fold, then the tape rolls from beat 2
          const OPEN = 1, DROP = 1.25, SHORT = 1.5, LONG = 2, TAPE0 = 2, TAPE1 = 2.5;
          const X = Xs(9), splayX = 0.75, splayY = 1.22, deg90 = Math.PI / 2;
          const fX = splayX + (Math.PI - splayX) * ease((b - DROP) / (SHORT - DROP));
          const fY = splayY + (Math.PI - splayY) * ease((b - SHORT) / (LONG - SHORT));
          if (b < DROP) {
            const u = b < OPEN ? (b - 0.5) / (OPEN - 0.5) : 1, phi = 0.2 + (deg90 - 0.2) * ease(u);
            const p = b < OPEN ? hop([X, ly - 14.75, 2.8], [X, y, ZB + 9], u, 6) : [X, y, ZB + 9 * (1 - ease((b - OPEN) / (DROP - OPEN)))];
            fly(K('cx', km), () => cartonOpen(p[0], p[1], p[2], phi, splayX, splayY));
          } else {
            DD.dyn(K('cx', km));
            cartonOpen(X, y, ZB, deg90, fX, fY);
            if (b > TAPE0) {
              const xe = X + BX0 + (BX1 - BX0) * clamp((b - TAPE0) / (TAPE1 - TAPE0)), zt = ZB + BH + 0.08;
              panel([[X + BX0, y - 0.9, zt], [xe, y - 0.9, zt], [xe, y + 0.9, zt], [X + BX0, y + 0.9, zt]], 'tape', { line: false });
            }
          }
        }
      }
      if (c === -1 && b >= 3.5) { // new substrate from the magazine
        const p = hop([XC0 + 2, ly + 12.5, 9], [XC0, ly, ZB], (b - 3.5) / 0.5, 2);
        fly(K('cs', km), () => pkg(p[0], p[1], p[2], {}));
      }
    }
    L.slot.cpo.forEach((j, km) => {
      const q = L.cpoHeap[j], [px, py] = PILE.cpo; DD.dyn(K('cp', km));
      pkg(px + q.dx, ly + py + q.dy, q.z, FULLPKG, q.a, true);
    });
    L.slot.sys.forEach((j, km) => {
      const q = L.sysHeap[j], [px, py] = PILE.sys; DD.dyn(K('sp', km));
      sysBox(px + q.dx, ly + py + q.dy, q.z, q.a, true);
    });
    // Section C tool heads and gate tops, by slot occupancy of this period
    DD.dyn(K('gc', 0));
    const curr = Math.floor(t / TC);
    const present = (s, back = 0) => { const km = mod(curr - back - (s - 1), NC); return !(L.fate.cpo[km] && s > 4) && !(L.fate.sys[km] && s > 8); };
    // Same stroke as the OE stations: down in a 16th to the hit, up in a 16th, idle the rest of the bar.
    // Arm 2 instead taps on every OE landing, two per beat.
    const head = (s, key, w, reach = 3) => {
      const hitAt = s % 4 + 1, d = Math.min(Math.abs(b - hitAt), Math.abs(b - hitAt - 4), Math.abs(b - hitAt + 4));
      const on = present(s, hitAt === 4 && b < 1 ? 1 : 0), stroke = s === 2 ? Math.abs(Math.cos(Math.PI * q8)) : Math.max(0, 1 - d / 0.25);
      const x = Xs(s), down = (on ? stroke : 0) * reach, zTop = 16.4, z0 = 11.5 - down;
      DD.swayMode('ride', 19, AMP(x));
      box(x - 0.9, ly - 0.9, z0 + 0.8, x + 0.9, ly + 0.9, zTop, 'rail', { tier: 'fine' });
      box(x - w, ly - w, z0, x + w, ly + w, z0 + 0.8, key, { tier: 'fine' });
    };
    head(1, 'ink', 1.5);
    head(2, 'ink', 1.5);
    head(3, 'ink', 1.5);
    head(4, 'pad', 4.4);
    head(5, 'ink', 1.5);
    head(6, 'fiber', 1.3, 0.7);
    head(7, 'ink', 1.5, 1);
    head(8, 'pad', 4.4, 1);
    DD.swayMode('ride', 26, AMP(Xs(9)));
    { // Tape head on the cross beam: parked high while the flaps fold, then rolls along the seam
      const X = Xs(9), run = b < 2 ? 0 : b < 2.5 ? (b - 2) / 0.5 : b < 3 ? 1 - (b - 2.5) / 0.5 : 0, down = present(9) && b >= 2 && b < 2.6;
      const xh = X + BX0 + (BX1 - BX0) * run, z0 = down ? ZB + BH + 0.12 : ZB + BH + 8;
      box(xh - 0.5, ly - 0.5, z0 + 3.4, xh + 0.5, ly + 0.5, 23.4, 'rail', { tier: 'fine' });
      box(xh - 1.5, ly - 1.1, z0 + 2.6, xh + 1.5, ly + 1.1, z0 + 3.4, 'ink', { tier: 'fine' });
      const roll = (r, key, yy, a0 = 0) => {
        const q = [];
        for (let i = 0; i < 16; i++) { const a = a0 + i / 16 * 6.2832; q.push([xh + r * Math.cos(a), yy, z0 + 1.45 + r * Math.sin(a)]); }
        panel(q, key);
      };
      roll(1.4, 'tape', ly - 1.15);
      roll(0.6, 'carton', ly - 1.2);
    }
    DD.swayMode(null);
    for (const s of [4, 8]) {
      const km = mod(curr - (s - 1), NC), hasPart = present(s);
      const bad = s === 4 ? L.fate.cpo[km] : L.fate.sys[km] && !L.fate.cpo[km];
      const here = s === 4 || !L.fate.cpo[km];
      const active = b >= 1 && here && hasPart !== undefined; // lights on the arm's hit, beat 1
      gateTop(Xs(s), ly, 10.5, 19, active && bad, active);
      if (active && bad && (b >= 1.5 && b < 2)) box(Xs(s) - 0.8, ly + 7.6, ZB + 0.8, Xs(s) + 0.8, ly + 10.5, ZB + 1.8, 'ink', { tier: 'fine' });
    }
  }

  DD.scene({
    manualCamera: true,
    style: 'L',
    period: T,
    boilFps: 5,
    build() {
      box(-500, -500, -2, 500, 500, 0, 'floor', { ground: true });
      for (const L of LINES) buildLine(L);
      DD.cameraRaw(-98, 150, -70.5);
    },
    update(t) {
      // Stations dance in unison on a 2-beat cycle: a hold on each beat, a quick swing between, stretching in transit and settling at the hold.
      const c = Math.cos(Math.PI * t / BEAT);
      if (QS.get('dance') === '0') DD.dance(0, 0); else DD.dance(Math.sign(c) * Math.pow(Math.abs(c), 0.45), 1 - c * c);
      LINES.forEach((L, i) => updateLine(L, i, t));
    },
  });

  window.LOOP = { T, LINES, BEAT };
})();
