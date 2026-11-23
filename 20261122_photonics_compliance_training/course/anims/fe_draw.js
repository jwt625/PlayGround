/* Code-drawn video keyframes for the ferroelectric course: window.LMS_DRAW.fe
   Contract: scene(ctx, W, H, u, k) with W, H in device pixels, u in 0..1 within keyframe k.
   Chips are drawn in their own frame in micrometres (x east along the waveguides, y south, as on the canvas)
   and placed with a view { x, y, k, rot } (die centre in canvas px, px per um, rotation in rad).
   Bright-field colours follow the course renders (tools/blender/tfln_die.py): 600 nm x-cut LN film on 4.7 um oxide
   on Si (mauve grey), trench-etched ribs (green-grey slab lines), evaporated gold. No DOM access at load time. */
(function () {
  'use strict';
  var root = typeof window !== 'undefined' ? window : this;
  var D = (root.LMS_DRAW = root.LMS_DRAW || {});
  var fe = (D.fe = D.fe || {});
  var PI = Math.PI, TAU = 2 * PI;

  /* ---------- palette ---------- */
  var P = fe.P = {
    film: '#a499a3', filmHi: '#b2a8b1', filmLo: '#91858f',     // 600 nm LN film (render: linear 0.374 0.32 0.369)
    slab: '#8bab9a',                                              // 300 nm slab in the trenches
    buffer: '#c2aab7',                                            // thick oxide window at waveguide crossings
    au: '#efcb72', auHi: '#fde7a6', auLo: '#c99a40', auEdge: '#8c6a26',
    resist: '#5f5048',                                            // thin-film resistor (dark brown grey)
    si: '#26282d', siEdge: '#121316',
    chuck: '#56595f', chuckLo: '#3d4045',
    light: '#63e6ff', lightTM: '#ff6b6b', rfP: '#ff3d3d', rfN: '#2f6dff',
    ok: '#3fae69', bad: '#e0524c', warn: '#e3b04b',
    ink: '#e9eef3', inkDim: '#a9b4bf', panel: 'rgba(10,14,19,0.84)', line: 'rgba(233,238,243,0.75)',
    screen: '#0a0e13', grid: 'rgba(120,140,160,0.20)', axis: 'rgba(200,210,220,0.65)'
  };
  var SANS = 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
  var MONO = 'ui-monospace, Menlo, Consolas, monospace';

  /* ---------- math and timing ---------- */
  var clamp = fe.clamp = function (v, a, b) { return v < a ? a : v > b ? b : v; };
  var lerp = fe.lerp = function (a, b, t) { return a + (b - a) * t; };
  var smooth = fe.smooth = function (t) { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  var ease = fe.ease = function (t) { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  var seg = fe.seg = function (u, a, b) { return clamp((u - a) / (b - a), 0, 1); };
  var eseg = fe.eseg = function (u, a, b) { return ease(seg(u, a, b)); };

  /* ---------- canvas units ---------- */
  // U.s: px per design px (900 wide frame); U.f(n): font px, never below about 8.5 css px on phones
  function units(ctx, W, H) {
    var dpr = (root.devicePixelRatio || 1);
    var s = W / 900;
    return { W: W, H: H, s: s, dpr: dpr, f: function (n) { return Math.round(Math.max(n * s, 8.5 * dpr)); } };
  }
  fe.units = units;
  function font(ctx, px, w, mono) { ctx.font = (w ? w + ' ' : '') + px + 'px ' + (mono ? MONO : SANS); }
  fe.font = font;

  /* ---------- annotation ---------- */
  // Plain box callout. o: { to: [x, y] leader target, align: 'l'|'c'|'r', col, bg, size, alpha, bold }
  function callout(ctx, U, x, y, text, o) {
    o = o || {};
    var a = o.alpha == null ? 1 : o.alpha;
    if (a <= 0.01) return null;
    ctx.save(); ctx.globalAlpha *= a;
    var fs = U.f(o.size || 12), pad = Math.round(fs * 0.45);
    font(ctx, fs, o.bold ? '600' : '', o.mono);
    var lines = String(text).split('\n'), tw = 0;
    lines.forEach(function (l) { tw = Math.max(tw, ctx.measureText(l).width); });
    var bw = tw + 2 * pad, bh = lines.length * fs * 1.22 + 2 * pad - fs * 0.22;
    var bx = o.align === 'r' ? x - bw : o.align === 'c' ? x - bw / 2 : x;
    var by = o.valign === 'b' ? y - bh : o.valign === 'm' ? y - bh / 2 : y;
    bx = clamp(bx, 2, U.W - bw - 2); by = clamp(by, 2, U.H - bh - 2);
    if (o.to) {
      var tx = o.to[0], ty = o.to[1];
      var ax = clamp(tx, bx, bx + bw), ay = clamp(ty, by, by + bh);
      if (ax === tx && ay === ty) { ax = bx + bw / 2; ay = by + bh; }
      ctx.strokeStyle = o.lineCol || P.line; ctx.lineWidth = Math.max(1, U.s * 1.1);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.fillStyle = o.lineCol || P.line; ctx.beginPath(); ctx.arc(tx, ty, Math.max(1.5, 2.2 * U.s), 0, TAU); ctx.fill();
    }
    ctx.fillStyle = o.bg || P.panel; ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = o.border || o.col || 'rgba(233,238,243,0.55)'; ctx.lineWidth = Math.max(1, U.s);
    ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
    ctx.fillStyle = o.col || P.ink; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    lines.forEach(function (l, i) { ctx.fillText(l, bx + pad, by + pad + i * fs * 1.22); });
    ctx.restore();
    return { x: bx, y: by, w: bw, h: bh };
  }
  fe.callout = callout;

  // Rubber stamp: slams in from 1.5x over 0.25 s-equivalent of u, then holds. ok: green OKAY style
  function stamp(ctx, U, text, ok, v, x, y) {
    if (v <= 0) return;
    var col = ok ? P.ok : P.bad;
    var sc = 1 + 0.5 * Math.pow(1 - clamp(v / 0.18, 0, 1), 2);
    ctx.save();
    ctx.translate(x == null ? U.W * 0.72 : x, y == null ? U.H * 0.32 : y); ctx.rotate(-0.16); ctx.scale(sc, sc);
    ctx.globalAlpha = clamp(v / 0.08, 0, 1);
    var fs = U.f(30); font(ctx, fs, '800');
    var tw = ctx.measureText(text).width + fs * 0.9, th = fs * 1.5;
    ctx.fillStyle = 'rgba(10,12,14,0.55)'; ctx.fillRect(-tw / 2, -th / 2, tw, th);
    ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, 3.2 * U.s);
    ctx.strokeRect(-tw / 2, -th / 2, tw, th);
    ctx.strokeRect(-tw / 2 + 5 * U.s, -th / 2 + 5 * U.s, tw - 10 * U.s, th - 10 * U.s);
    ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, 0, fs * 0.04);
    ctx.restore();
  }
  fe.stamp = stamp;

  // Scale bar: picks 1-2-5 x 10^n um near the target length in px; label in um or mm
  function scaleBar(ctx, U, x, y, kpx, target, col) {
    target = target || 110 * U.s;
    var um = target / kpx, p10 = Math.pow(10, Math.floor(Math.log(um) / Math.LN10)), m = um / p10;
    var n = (m >= 5 ? 5 : m >= 2 ? 2 : 1) * p10, len = n * kpx;
    var lab = n >= 1000 ? (n / 1000) + ' mm' : n + ' µm';
    ctx.save();
    var fs = U.f(11); font(ctx, fs, '600');
    ctx.fillStyle = 'rgba(10,14,19,0.55)'; ctx.fillRect(x - 6 * U.s, y - fs - 8 * U.s, len + 12 * U.s, fs + 16 * U.s);
    ctx.fillStyle = col || '#ffffff'; ctx.fillRect(x, y, len, Math.max(2, 3 * U.s));
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(lab, x + len / 2, y - 3 * U.s);
    ctx.restore();
    return len;
  }
  fe.scaleBar = scaleBar;

  function arrow(ctx, x0, y0, x1, y1, col, lw, head) {
    var a = Math.atan2(y1 - y0, x1 - x0), hs = head || lw * 4;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw; ctx.lineCap = 'butt';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - hs * 0.8 * Math.cos(a), y1 - hs * 0.8 * Math.sin(a)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hs * Math.cos(a - 0.38), y1 - hs * Math.sin(a - 0.38));
    ctx.lineTo(x1 - hs * Math.cos(a + 0.38), y1 - hs * Math.sin(a + 0.38));
    ctx.closePath(); ctx.fill();
  }
  fe.arrow = arrow;

  // Crystal axis glyph for x-cut: y along the waveguides, z across the electrode gaps, x out of the plane.
  // rot: die rotation (the glyph turns with the crystal). o: { yLab, zLab, q (question marks), alpha, col }
  function axes(ctx, U, x, y, rot, o) {
    o = o || {};
    var L = (o.len || 46) * U.s, lw = Math.max(1.5, 2.2 * U.s), fs = U.f(o.size || 12);
    ctx.save(); ctx.globalAlpha *= (o.alpha == null ? 1 : o.alpha);
    var bw = L * 2.15, bh = L * 2.15;
    if (o.box !== false) { ctx.fillStyle = 'rgba(10,14,19,0.62)'; ctx.fillRect(x - bw * 0.28, y - bh * 0.7, bw, bh); }
    var cy = Math.cos(rot), sy = Math.sin(rot);
    var yx = x + L * cy, yy = y + L * sy, zx = x + L * sy, zy = y - L * cy;   // z is 90 deg ccw of y on screen (north when rot 0)
    arrow(ctx, x, y, yx, yy, o.yCol || '#9fe3ff', lw, 8 * U.s);
    arrow(ctx, x, y, zx, zy, o.zCol || '#ffc46b', lw, 8 * U.s);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = Math.max(1, 1.3 * U.s);
    ctx.beginPath(); ctx.arc(x, y, 5 * U.s, 0, TAU); ctx.stroke();
    ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(x, y, 1.6 * U.s, 0, TAU); ctx.fill();
    font(ctx, fs, '700'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = o.yCol || '#9fe3ff'; ctx.fillText(o.yLab || 'y', yx + 11 * U.s * cy + 2 * U.s, yy + 11 * U.s * sy);
    ctx.fillStyle = o.zCol || '#ffc46b'; ctx.fillText(o.zLab || 'z', zx + 11 * U.s * sy, zy - 11 * U.s * cy);
    ctx.fillStyle = '#ffffff'; font(ctx, U.f(10)); ctx.fillText(o.xLab || 'x', x - 11 * U.s, y + 10 * U.s);
    if (o.title !== false) { font(ctx, U.f(10), '600'); ctx.fillStyle = P.inkDim; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(o.title || 'x-cut', x + bw * 0.69, y + bh * 0.28); }
    ctx.restore();
  }
  fe.axes = axes;

  /* ---------- backgrounds ---------- */
  // Probe station chuck under the stereo camera: nickel-plated, vacuum holes, soft vignette
  function chuckBg(ctx, U, o) {
    o = o || {};
    var W = U.W, H = U.H;
    var g = ctx.createRadialGradient(W * 0.5, H * 0.45, H * 0.1, W * 0.5, H * 0.5, W * 0.75);
    g.addColorStop(0, o.c0 || '#62656b'); g.addColorStop(1, o.c1 || '#2d2f33');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // machining marks: faint concentric arcs around an off-frame centre
    ctx.save(); ctx.globalAlpha = 0.06; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
    var cx = W * 0.5 + (o.ox || 0), cy = H * 2.4 + (o.oy || 0);
    for (var r = H * 1.6; r < H * 3.2; r += 7 * U.s) { ctx.beginPath(); ctx.arc(cx, cy, r, PI * 1.25, PI * 1.75); ctx.stroke(); }
    ctx.restore();
    if (o.holes !== false && o.k) {   // vacuum holes on a 5 mm grid, 0.5 mm dia, in chuck coordinates
      ctx.save(); ctx.fillStyle = 'rgba(12,13,15,0.22)';
      var p = 5000 * o.k, r0 = Math.max(1.5, 150 * o.k);
      var x0 = ((o.hx || 0) % p + p) % p, y0 = ((o.hy || 0) % p + p) % p;
      for (var xx = x0 - p; xx < W + p; xx += p) for (var yy = y0 - p; yy < H + p; yy += p) { ctx.beginPath(); ctx.arc(xx, yy, r0, 0, TAU); ctx.fill(); }
      ctx.restore();
    }
  }
  fe.chuckBg = chuckBg;

  function vignette(ctx, U, strength) {
    var W = U.W, H = U.H;
    var g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.hypot(W, H) * 0.55);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,' + (strength == null ? 0.45 : strength) + ')');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  }
  fe.vignette = vignette;

  // Circular eyepiece mask (microscope ocular view)
  function eyepiece(ctx, U, cx, cy, r) {
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, U.W, U.H); ctx.arc(cx, cy, r, 0, TAU, true);
    ctx.fillStyle = '#050607'; ctx.fill('evenodd');
    var g = ctx.createRadialGradient(cx, cy, r * 0.82, cx, cy, r);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,0.75)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.fill();
    ctx.restore();
  }
  fe.eyepiece = eyepiece;

  /* ---------- traveling-wave MZM die layout (um, die frame) ---------- */
  // Defaults reproduce the course render die (tfln_die.py, MZM-07, 6 x 6 mm): GSG 300/40/300 um with 8 um gaps,
  // straight section |x| <= 1600, CPW bends north at the RF input (west) end and south at the far end,
  // GSG pads 80 um signal / 40 um gaps / 150 um pitch / 150 um long. Arms run straight through the bends and pass
  // under the bent CPW inside a thick-oxide crossing window; the splitters sit outside the bends.
  function mzm(o) {
    o = o || {};
    var L = {
      w: o.w || 6000, h: o.h || 6000, yc: o.yc != null ? o.yc : 400,
      xe: o.xe || 1600, rb: o.rb || 450, ws: 40, g: 8, wg: 300, ps: 80, pg: 40, pitch: 150, lpad: 150, ltaper: 450,
      padN: o.padN != null ? o.padN : 2050, padS: o.padS != null ? o.padS : 2050,   // |y| of the outer pad ends
      term: o.term || 'pads', mmiL: 60, mmiW: 8,
      label: o.label || 'MZM-07  X-CUT LN', marks: o.marks !== false, refY: o.refY != null ? o.refY : 2450,
      rfLabel: o.rfLabel || 'RF IN', termLabel: o.termLabel || (o.term === 'onchip' ? '50 Ω' : 'TERM')
    };
    L.xv = L.xe + L.rb;
    L.mmiX = o.mmiX || (L.w / 2 - 340);
    return L;
  }
  fe.mzm = mzm;

  // Geometry for a given minimum drawn gap (um), so the 8 um gaps stay visible at low zoom
  function geom(L, gmin) {
    var g = Math.max(L.g, gmin || 0), key = Math.round(g * 10);
    L._g = L._g || {};
    if (L._g[key]) return L._g[key];
    var ws = L.ws, wg = L.wg, rb = L.rb, xe = L.xe, xv = L.xv, yc = L.yc, step = 6;
    var pts = [];
    function line(x0, y0, x1, y1) {
      var d = Math.hypot(x1 - x0, y1 - y0), n = Math.max(2, Math.ceil(d / step)), tx = (x1 - x0) / d, ty = (y1 - y0) / d;
      for (var i = 0; i < n; i++) { var t = i / n; pts.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, tx, ty]); }
    }
    function arc(cx, cy, a0, a1) {
      var n = Math.max(8, Math.ceil(Math.abs(a1 - a0) * rb / step)), sg = a1 > a0 ? 1 : -1;
      for (var i = 0; i < n; i++) { var a = a0 + (a1 - a0) * i / n; pts.push([cx + rb * Math.cos(a), cy + rb * Math.sin(a), -sg * Math.sin(a), sg * Math.cos(a)]); }
    }
    var yEnd = L.term === 'onchip' ? yc + rb + 420 : L.padS;
    line(-xv, -L.padN, -xv, yc - rb);
    arc(-xe, yc - rb, PI, PI / 2);
    line(-xe, yc, xe, yc);
    arc(xe, yc + rb, -PI / 2, 0);
    line(xv, yc + rb, xv, yEnd);
    pts.push([xv, yEnd, 0, 1]);
    var s = 0;
    for (var i = 0; i < pts.length; i++) { if (i) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); pts[i].push(s); }
    var Ltot = s, onchip = L.term === 'onchip';
    function uu(sv) {   // 0 on the line, 1 on the pads
      var d = onchip ? sv : Math.min(sv, Ltot - sv);
      if (onchip && sv > Ltot / 2) d = Ltot - sv + L.lpad - 60;  // short pad-like flare before the resistor
      return 1 - smooth((d - L.lpad) / L.ltaper);
    }
    function side(fn, a, b) {
      var out = [];
      for (var i = 0; i < pts.length; i++) { var p = pts[i]; if (p[4] < a || p[4] > b) continue; var d = fn(p[4]); out.push([p[0] - p[3] * d, p[1] + p[2] * d]); }
      return out;
    }
    var hs = function (sv) { return ws / 2 + (L.ps / 2 - ws / 2) * uu(sv); };
    var gi = function (sv) { return ws / 2 + g + ((L.ps / 2 + L.pg) - (ws / 2 + g)) * uu(sv); };
    var go = function () { return ws / 2 + g + wg; };
    var sEndSig = onchip ? Ltot - 150 : Ltot;
    var sig = side(hs, 0, sEndSig).concat(side(function (sv) { return -hs(sv); }, 0, sEndSig).reverse());
    var gl = side(go, 0, Ltot).concat(side(gi, 0, Ltot).reverse());
    var gr = side(function (sv) { return -gi(sv); }, 0, Ltot).concat(side(function (sv) { return -go(); }, 0, Ltot).reverse());
    var G = { g: g, pts: pts, Ltot: Ltot, cond: [gl, sig, gr], off: ws / 2 + g / 2, sig: sig };
    if (onchip) {   // ground bar across the end joins both grounds; thin-film resistor from the signal end to the bar
      var pe = pts[pts.length - 1], hw = go();
      G.bar = [[pe[0] - hw, pe[1] - 60], [pe[0] + hw, pe[1] - 60], [pe[0] + hw, pe[1] + 120], [pe[0] - hw, pe[1] + 120]];
      G.res = [[pe[0] - 34, pe[1] - 162], [pe[0] + 34, pe[1] - 162], [pe[0] + 34, pe[1] - 52], [pe[0] - 34, pe[1] - 52]];
    }
    // crossing windows: arms pass under the bent CPW (outer ground; and the signal for the inner-gap arm)
    var Rout = rb + ws / 2 + g + wg, dx = Math.sqrt(Rout * Rout - Math.pow(rb - G.off, 2));
    G.xa = xe + dx + 50;
    G.win = [[-G.xa + 15, xe + 25], [xe + 25, G.xa - 15]].map(function (r, i) {
      var x0 = i ? r[0] : -r[1], x1 = i ? r[1] : -r[0];
      return [x0, yc - G.off - 40, x1 - x0, 2 * G.off + 80];
    });
    // pad contact centres (signal pad) and outward direction
    G.padIn = { x: -xv, y: -L.padN + L.lpad / 2, dx: 0, dy: -1 };
    G.padOut = onchip ? null : { x: xv, y: L.padS - L.lpad / 2, dx: 0, dy: 1 };
    G.resC = onchip ? { x: xv, y: pts[pts.length - 1][1] - 107 } : null;
    L._g[key] = G;
    return G;
  }
  fe.geom = geom;

  // Optical arm centre y at die x for arm sg (+1 south, -1 north); null where the arm does not exist
  function armY(L, G, x, sg) {
    var ax = Math.abs(x), x0 = G.xa, x1 = L.mmiX - L.mmiL / 2, mo = L.mmiW / 4;
    if (ax <= x0) return L.yc + sg * G.off;
    if (ax < x1) { var t = (x1 - ax) / (x1 - x0); var f = t - Math.sin(TAU * t) / TAU; return L.yc + sg * (mo + (G.off - mo) * f); }
    if (ax <= L.mmiX + L.mmiL / 2) return L.yc + sg * mo;
    return null;
  }
  fe.armY = armY;

  function polyPath(ctx, pts) { ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.closePath(); }
  function sbendPath(ctx, xa, xb, ya, yb, n) {
    for (var i = 0; i <= n; i++) { var t = i / n, x = xa + (xb - xa) * t, f = t - Math.sin(TAU * t) / TAU, y = ya + (yb - ya) * f; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
  }

  function goldFill(ctx, x0, y0, x1, y1) {
    var g = ctx.createLinearGradient(x0, y0, x1, y1);
    g.addColorStop(0, P.auHi); g.addColorStop(0.35, P.au); g.addColorStop(0.7, '#e6bc5c'); g.addColorStop(1, P.auHi);
    return g;
  }

  function vapply(ctx, v) { ctx.translate(v.x, v.y); ctx.rotate(v.rot || 0); ctx.scale(v.k, v.k); }
  fe.vapply = vapply;
  // die (um) -> canvas
  function d2c(v, x, y) { var c = Math.cos(v.rot || 0), s = Math.sin(v.rot || 0); return [v.x + v.k * (x * c - y * s), v.y + v.k * (x * s + y * c)]; }
  fe.d2c = d2c;

  // Draw the MZM die. o: { shadow, lift (0..1, raises shadow), detail, skate, noText, fade }
  function drawDie(ctx, L, v, o) {
    o = o || {};
    var k = v.k, px = 1 / k, G = geom(L, 2 * px);
    var w = L.w, h = L.h;
    ctx.save();
    // shadow on the chuck
    var lift = o.lift || 0;
    ctx.save();
    ctx.translate(v.x + (4 + 26 * lift) * (o.us || 1), v.y + (6 + 34 * lift) * (o.us || 1)); ctx.rotate(v.rot || 0); ctx.scale(k, k);
    ctx.fillStyle = 'rgba(0,0,0,' + (0.42 - 0.18 * lift) + ')';
    if (ctx.filter !== undefined) ctx.filter = 'blur(' + Math.round((3 + 10 * lift) * (o.us || 1)) + 'px)';
    ctx.fillRect(-w / 2, -h / 2, w, h);
    ctx.restore();
    vapply(ctx, v);
    // body: diced sidewalls show as a dark rim; facets (east, west) polished
    ctx.fillStyle = P.siEdge; ctx.fillRect(-w / 2 - 1.5 * px, -h / 2 - 1.5 * px, w + 3 * px, h + 3 * px);
    var fg = ctx.createLinearGradient(-w / 2, -h / 2, w / 2, h / 2);
    fg.addColorStop(0, P.filmHi); fg.addColorStop(0.55, P.film); fg.addColorStop(1, P.filmLo);
    ctx.fillStyle = fg; ctx.fillRect(-w / 2, -h / 2, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.18)'; ctx.fillRect(-w / 2, -h / 2, Math.max(8, 1.2 * px), h); ctx.fillRect(w / 2 - Math.max(8, 1.2 * px), -h / 2, Math.max(8, 1.2 * px), h);
    // crossing windows (thick oxide buffer under the CPW where the arms pass)
    ctx.fillStyle = P.buffer; G.win.forEach(function (r) { ctx.fillRect(r[0], r[1], r[2], r[3]); });
    // waveguides: trench pair (7.5 um band of slab colour) with the 1.5 um rib in the middle
    var tw = Math.max(7.5, 1.1 * px), rw = 1.5;
    function wgPaths() {
      var x1 = L.mmiX - L.mmiL / 2, mo = L.mmiW / 4;
      [-1, 1].forEach(function (sx) {
        ctx.moveTo(sx * w / 2, L.yc); ctx.lineTo(sx * (L.mmiX + L.mmiL / 2), L.yc);
        [-1, 1].forEach(function (sg) { sbendPath(ctx, sx * x1, sx * G.xa, L.yc + sg * mo, L.yc + sg * G.off, 30); });
      });
      [-1, 1].forEach(function (sg) { ctx.moveTo(-G.xa, L.yc + sg * G.off); ctx.lineTo(G.xa, L.yc + sg * G.off); });
      if (L.refY) { ctx.moveTo(-w / 2, L.refY); ctx.lineTo(w / 2, L.refY); }
    }
    ctx.lineCap = 'butt';
    ctx.strokeStyle = P.slab; ctx.lineWidth = tw; ctx.beginPath(); wgPaths(); ctx.stroke();
    [-1, 1].forEach(function (sx) { ctx.fillStyle = P.slab; ctx.fillRect(sx * L.mmiX - L.mmiL / 2 - 3, L.yc - L.mmiW / 2 - 3, L.mmiL + 6, L.mmiW + 6); });
    if (rw * k > 0.7) {
      ctx.strokeStyle = P.film; ctx.lineWidth = rw; ctx.beginPath(); wgPaths(); ctx.stroke();
      [-1, 1].forEach(function (sx) { ctx.fillStyle = P.film; ctx.fillRect(sx * L.mmiX - L.mmiL / 2, L.yc - L.mmiW / 2, L.mmiL, L.mmiW); });
    }
    // edge couplers: slab taper at each facet (6 um to the tip over 300 um), drawn as a faint wedge
    if (k > 0.05) {
      ctx.fillStyle = 'rgba(139,171,154,0.55)';
      [-1, 1].forEach(function (sx) { ctx.beginPath(); ctx.moveTo(sx * w / 2, L.yc - 3 - 1.5 * px); ctx.lineTo(sx * (w / 2 - 300), L.yc - 12); ctx.lineTo(sx * (w / 2 - 300), L.yc + 12); ctx.lineTo(sx * w / 2, L.yc + 3 + 1.5 * px); ctx.closePath(); ctx.fill(); });
    }
    // gold: three conductors, edges darker (sidewall)
    var gf = goldFill(ctx, -w / 2, -h / 2, w / 2, h / 2);
    var conds = G.cond.slice(); if (G.bar) conds.push(G.bar);
    conds.forEach(function (pp) { polyPath(ctx, pp); ctx.fillStyle = gf; ctx.fill(); ctx.strokeStyle = P.auEdge; ctx.lineWidth = Math.min(2, 0.9 * px); ctx.stroke(); });
    ctx.strokeStyle = 'rgba(120,84,30,0.6)'; ctx.lineWidth = Math.min(1.5, 0.9 * px); G.win.forEach(function (r) { ctx.strokeRect(r[0], r[1], r[2], r[3]); });
    if (G.res) { polyPath(ctx, G.res); ctx.fillStyle = P.resist; ctx.fill(); ctx.strokeStyle = 'rgba(30,24,20,0.8)'; ctx.lineWidth = Math.min(1.5, 0.7 * px); ctx.stroke(); }
    if (o.skate) skates(ctx, L, G, o.skate);
    if (L.marks && !o.noText) marksAndText(ctx, L, G, gf, k);
    else if (!o.noText) textOnly(ctx, L, G, gf, k);
    ctx.restore();
    return G;
  }
  fe.drawDie = drawDie;

  function skates(ctx, L, G, which) {
    ctx.fillStyle = 'rgba(120,90,40,0.55)';
    var pads = which === 'out' ? [G.padOut] : which === 'both' ? [G.padIn, G.padOut] : [G.padIn];
    pads.forEach(function (p) { if (!p) return; for (var i = -1; i <= 1; i++) ctx.fillRect(p.x + i * L.pitch - 6, p.y + (p.dy < 0 ? -26 : 0), 12, 26); });
  }

  function dieText(ctx, t, x, y, size, align) {
    ctx.font = '600 ' + size + 'px ' + SANS; ctx.textAlign = align || 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, x, y);
  }
  function textOnly(ctx, L, G, gf, k) {
    if (k < 0.012) return;
    ctx.fillStyle = gf;
    dieText(ctx, L.label, 0, -L.h / 2 + 420, 190);
    dieText(ctx, L.rfLabel, G.padIn.x + 470, G.padIn.y, 120, 'left');
    if (G.padOut) dieText(ctx, L.termLabel, G.padOut.x - 470, G.padOut.y, 120, 'right');
    if (G.resC) dieText(ctx, L.termLabel, G.resC.x - 470, G.resC.y, 120, 'right');
    dieText(ctx, 'IN', -L.w / 2 + 230, L.yc + 210, 120); dieText(ctx, 'OUT', L.w / 2 - 280, L.yc + 210, 120);
  }
  function marksAndText(ctx, L, G, gf, k) {
    ctx.fillStyle = gf;
    var c = L.w / 2 - 280, ch = L.h / 2 - 280, Lm = 420, wm = 50;
    ctx.beginPath(); ctx.moveTo(-c, -ch); ctx.lineTo(-c + Lm, -ch); ctx.lineTo(-c + Lm, -ch + wm); ctx.lineTo(-c + wm, -ch + wm); ctx.lineTo(-c + wm, -ch + Lm); ctx.lineTo(-c, -ch + Lm); ctx.closePath(); ctx.fill();
    [[c, ch], [-c, ch], [c, -ch + 520]].forEach(function (p) { ctx.fillRect(p[0] - 100, p[1] - 15, 200, 30); ctx.fillRect(p[0] - 15, p[1] - 100, 30, 200); });
    // +Z arrow (die north = crystal +z)
    var ay0 = L.yc - 950, ay1 = Math.max(-L.h / 2 + 800, L.yc - 2150);
    if (ay0 - ay1 > 500) {
      ctx.fillRect(-30, ay1 + 300, 60, ay0 - ay1 - 300);
      ctx.beginPath(); ctx.moveTo(0, ay1); ctx.lineTo(110, ay1 + 300); ctx.lineTo(-110, ay1 + 300); ctx.closePath(); ctx.fill();
      dieText(ctx, '+Z', 300, ay1 + 260, 300, 'left');
    } else dieText(ctx, '+Z ↑', 0, L.yc - 700, 200);
    textOnly(ctx, L, G, gf, k);
    if (L.refY) dieText(ctx, 'REF', 0, L.refY - 150, 110);
  }

  /* ---------- light and RF overlays ---------- */
  // Optical pulse at die x (um); split into the two arms between the MMIs. col: colour; a: alpha
  function lightPulse(ctx, L, v, x, o) {
    o = o || {};
    var G = geom(L, 2 / v.k), ys = [];
    if (Math.abs(x) > L.w / 2 + 50) return;
    var a1 = armY(L, G, x, 1), a2 = armY(L, G, x, -1);
    if (a1 == null) ys = [L.yc]; else ys = [a1, a2];
    var r = (o.r || 9) * (o.us || 1);
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    ys.forEach(function (y) {
      var p = d2c(v, x, y);
      var g = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], r);
      g.addColorStop(0, 'rgba(255,255,255,' + (0.9 * (o.a || 1)) + ')'); g.addColorStop(0.3, hexA(o.col || P.light, 0.85 * (o.a || 1))); g.addColorStop(1, hexA(o.col || P.light, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, TAU); ctx.fill();
    });
    ctx.restore();
  }
  fe.lightPulse = lightPulse;

  // Faint glow along the whole optical path (light present in the waveguides)
  function lightPath(ctx, L, v, a, col, x0, x1) {
    var G = geom(L, 2 / v.k);
    x0 = x0 == null ? -L.w / 2 : x0; x1 = x1 == null ? L.w / 2 : x1;
    ctx.save(); ctx.strokeStyle = hexA(col || P.light, a); ctx.lineWidth = Math.max(1.5, 2 * (v.us || 1)); ctx.lineCap = 'round';
    [1, -1].forEach(function (sg) {
      ctx.beginPath(); var first = true;
      for (var x = x0; x <= x1; x += 20) { var y = armY(L, G, x, sg); if (y == null) y = L.yc; var p = d2c(v, x, y); if (first) { ctx.moveTo(p[0], p[1]); first = false; } else ctx.lineTo(p[0], p[1]); }
      ctx.stroke();
    });
    ctx.restore();
  }
  fe.lightPath = lightPath;

  // RF wave packet on the CPW: stripes across signal and gaps, sign coloured, travelling along arc length.
  // sc: packet centre (um of arc length from the RF input pad end); dir: +1 travels from the input pads; cyc: carrier period (um)
  function rfPacket(ctx, L, v, sc, o) {
    o = o || {};
    var G = geom(L, 2 / v.k), pts = G.pts, half = o.half || 900, per = o.per || 700, ph = o.ph || 0;
    var hw = Math.max(L.ws / 2 + G.g + 30, 7 / v.k);
    ctx.save(); ctx.lineCap = 'butt';
    ctx.lineWidth = Math.max(1.5, per / 7 * v.k);
    for (var i = 0; i < pts.length; i += 2) {
      var p = pts[i], ds = p[4] - sc; if (Math.abs(ds) > half) continue;
      if (p[4] > G.Ltot - (L.term === 'onchip' ? 150 : 0)) continue;
      var env = Math.pow(Math.cos(ds / half * PI / 2), 2), c = Math.cos(TAU * (p[4] - sc) / per + ph);
      if (Math.abs(c) < 0.25) continue;
      var a = env * (o.a == null ? 0.85 : o.a) * Math.abs(c);
      var nx = -p[3], ny = p[2];
      var q0 = d2c(v, p[0] + nx * hw, p[1] + ny * hw), q1 = d2c(v, p[0] - nx * hw, p[1] - ny * hw);
      ctx.strokeStyle = hexA(c > 0 ? P.rfP : P.rfN, a);
      ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q1[0], q1[1]); ctx.stroke();
    }
    ctx.restore();
  }
  fe.rfPacket = rfPacket;
  // arc length of the CPW point nearest to die x on the straight section
  fe.cpwS = function (L, v, x) { var G = geom(L, 2 / v.k); var s0 = G.pts.filter(function (p) { return Math.abs(p[1] - L.yc) < 1 && Math.abs(p[2]) > 0.99; }); return s0.length ? s0[0][4] + (x - s0[0][0]) : 0; };

  function hexA(hex, a) {
    if (hex.charAt(0) !== '#') return hex;
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + clamp(a, 0, 1).toFixed(3) + ')';
  }
  fe.hexA = hexA;

  /* ---------- tools seen from above ---------- */
  // GSG probe, top view. Tip contact row centred at (x, y) canvas px, body pointing along ang (rad, canvas).
  // k: px per um; pitch: um; lift: 0 landed .. 1 raised (shadow moves away, contacts lose their skate)
  function gsgProbe(ctx, x, y, ang, k, o) {
    o = o || {};
    var pitch = o.pitch || 150, lift = o.lift || 0, us = o.us || 1;
    // shadow first
    ctx.save();
    ctx.translate(x + (6 + 40 * lift) * us, y + (8 + 52 * lift) * us); ctx.rotate(ang + PI / 2); ctx.scale(k, k);
    ctx.fillStyle = 'rgba(0,0,0,' + (0.35 - 0.15 * lift) + ')';
    if (ctx.filter !== undefined) ctx.filter = 'blur(' + Math.round((2 + 8 * lift) * us) + 'px)';
    probeShape(ctx, pitch, true);
    ctx.restore();
    ctx.save();
    ctx.translate(x, y); ctx.rotate(ang + PI / 2); ctx.scale(k * (1 + 0.04 * lift), k * (1 + 0.04 * lift));
    probeShape(ctx, pitch, false);
    ctx.restore();
  }
  // local frame: contacts at y = 0 along x, body toward -y
  function probeShape(ctx, pitch, shadow) {
    var fill = function (c) { ctx.fillStyle = shadow ? ctx.fillStyle : c; };
    // coax line and body (only the first 9 mm, the rest leaves the frame)
    if (!shadow) { var gc = ctx.createLinearGradient(-600, 0, 600, 0); gc.addColorStop(0, '#5c6168'); gc.addColorStop(0.45, '#d7dde3'); gc.addColorStop(0.6, '#a6adb5'); gc.addColorStop(1, '#4a4f55'); ctx.fillStyle = gc; }
    ctx.beginPath(); ctx.moveTo(-600, -1300); ctx.lineTo(-600, -9000); ctx.lineTo(600, -9000); ctx.lineTo(600, -1300); ctx.closePath(); ctx.fill();
    // body block beyond 9 mm (black anodized, tapered)
    fill('#1b1c1f'); ctx.beginPath(); ctx.moveTo(-3000, -9000); ctx.lineTo(-6000, -16000); ctx.lineTo(6000, -16000); ctx.lineTo(3000, -9000); ctx.closePath(); ctx.fill();
    // conical tip, coax end ground flat
    if (!shadow) { var gt = ctx.createLinearGradient(-500, 0, 500, 0); gt.addColorStop(0, '#6c7178'); gt.addColorStop(0.5, '#e1e6ea'); gt.addColorStop(1, '#5a5f66'); ctx.fillStyle = gt; }
    ctx.beginPath(); ctx.moveTo(-600, -1300); ctx.lineTo(-240, -380); ctx.lineTo(240, -380); ctx.lineTo(600, -1300); ctx.closePath(); ctx.fill();
    // ceramic / dielectric tip plate
    fill('#d9d2c3'); ctx.fillRect(-230, -420, 460, 70);
    // three contacts converging to the pad pitch
    fill('#b9bec4');
    [-1, 0, 1].forEach(function (i) {
      ctx.beginPath(); ctx.moveTo(i * 110 - 22, -380); ctx.lineTo(i * 110 + 22, -380); ctx.lineTo(i * pitch + 15, 0); ctx.lineTo(i * pitch - 15, 0); ctx.closePath(); ctx.fill();
    });
    if (!shadow) { ctx.fillStyle = 'rgba(255,255,255,0.55)'; [-1, 0, 1].forEach(function (i) { ctx.fillRect(i * pitch - 4, -60, 8, 50); }); }
  }
  fe.gsgProbe = gsgProbe;

  // Lensed fibre, top view: 125 um cladding, lensed tip at (x, y) canvas px, fibre running along ang (rad) away from the tip
  function lensedFiber(ctx, x, y, ang, k, o) {
    o = o || {};
    var us = o.us || 1, len = o.len || 9000;
    ctx.save();
    ctx.translate(x + 5 * us, y + 7 * us); ctx.rotate(ang); ctx.scale(k, k);
    ctx.fillStyle = 'rgba(0,0,0,0.28)'; if (ctx.filter !== undefined) ctx.filter = 'blur(' + Math.round(3 * us) + 'px)';
    ctx.fillRect(0, -62, len, 124);
    ctx.restore();
    ctx.save();
    ctx.translate(x, y); ctx.rotate(ang); ctx.scale(k, k);
    var r = 62.5, gw = Math.max(r, 1.2 / k);
    var g = ctx.createLinearGradient(0, -gw, 0, gw);
    g.addColorStop(0, 'rgba(40,46,56,0.95)'); g.addColorStop(0.32, 'rgba(150,170,190,0.65)'); g.addColorStop(0.5, 'rgba(235,245,255,0.92)'); g.addColorStop(0.68, 'rgba(150,170,190,0.65)'); g.addColorStop(1, 'rgba(40,46,56,0.95)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(len, -gw); ctx.lineTo(140, -gw); ctx.quadraticCurveTo(10, -gw * 0.6, 0, 0); ctx.quadraticCurveTo(10, gw * 0.6, 140, gw); ctx.lineTo(len, gw); ctx.closePath(); ctx.fill();
    // V-groove holder block (stainless) holding the fibre about 3 mm back from the tip
    if (o.holder !== false) {
      var hg = ctx.createLinearGradient(0, -900, 0, 900); hg.addColorStop(0, '#7b8087'); hg.addColorStop(0.5, '#b8bec5'); hg.addColorStop(1, '#6a6f76');
      ctx.fillStyle = hg; ctx.fillRect(3000, -900, len, 1800);
      ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 1.5 / k; ctx.strokeRect(3000, -900, len, 1800);
    }
    ctx.restore();
  }
  fe.lensedFiber = lensedFiber;

  // Fine tweezers from above, tips at (x, y), pointing along ang; open: tip separation in px
  function tweezers(ctx, x, y, ang, len, open, us) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
    [-1, 1].forEach(function (sg) {
      var g = ctx.createLinearGradient(0, sg * open / 2, -len, sg * 6 * us); g.addColorStop(0, '#aeb4ba'); g.addColorStop(0.5, '#e4e8ec'); g.addColorStop(1, '#8d939a');
      ctx.fillStyle = g; ctx.beginPath();
      ctx.moveTo(0, sg * open / 2); ctx.lineTo(-len, sg * 3 * us); ctx.lineTo(-len, sg * 13 * us); ctx.lineTo(-len * 0.08, sg * (open / 2 + 4 * us)); ctx.lineTo(0, sg * (open / 2 + 2.5 * us));
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = Math.max(1, us); ctx.stroke();
    });
    ctx.restore();
  }
  fe.tweezers = tweezers;

  /* ---------- plots and instrument screens ---------- */
  // Instrument plot. r: { x, y, w, h } canvas px. o: { x0, x1, y0, y1, xt: [ticks], yt, xl, yl, title, log }
  function plot(ctx, U, r, o) {
    var X = function (v) { return r.x + (v - o.x0) / (o.x1 - o.x0) * r.w; };
    var Y = function (v) { return r.y + r.h - (v - o.y0) / (o.y1 - o.y0) * r.h; };
    ctx.save();
    ctx.fillStyle = o.bg || P.screen; ctx.fillRect(r.x - (o.padL || 46) * U.s, r.y - (o.padT || 26) * U.s, r.w + ((o.padL || 46) + 14) * U.s, r.h + ((o.padT || 26) + 38) * U.s);
    ctx.strokeStyle = 'rgba(160,175,190,0.35)'; ctx.lineWidth = Math.max(1, U.s); ctx.strokeRect(r.x - (o.padL || 46) * U.s + 0.5, r.y - (o.padT || 26) * U.s + 0.5, r.w + ((o.padL || 46) + 14) * U.s - 1, r.h + ((o.padT || 26) + 38) * U.s - 1);
    ctx.strokeStyle = P.grid; ctx.lineWidth = 1;
    var fs = U.f(10); font(ctx, fs, '', true); ctx.fillStyle = P.inkDim;
    (o.xt || []).forEach(function (t) { var x = X(t); ctx.beginPath(); ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h); ctx.stroke(); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(o.xf ? o.xf(t) : String(t), x, r.y + r.h + 4 * U.s); });
    (o.yt || []).forEach(function (t) { var y = Y(t); ctx.beginPath(); ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y); ctx.stroke(); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(o.yf ? o.yf(t) : String(t), r.x - 5 * U.s, y); });
    ctx.strokeStyle = P.axis; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    font(ctx, U.f(10.5), '600'); ctx.fillStyle = P.ink;
    if (o.xl) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(o.xl, r.x + r.w / 2, r.y + r.h + 4 * U.s + fs * 1.25); }
    if (o.yl) { ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(o.yl, r.x - (o.padL || 46) * U.s + 6 * U.s, r.y - 6 * U.s); }
    ctx.restore();
    return { X: X, Y: Y, r: r };
  }
  fe.plot = plot;
  function trace(ctx, pl, xs, f, col, lw, clipX) {
    ctx.save(); ctx.beginPath(); ctx.rect(pl.r.x, pl.r.y, pl.r.w, pl.r.h); ctx.clip();
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineJoin = 'round'; ctx.beginPath();
    var first = true;
    xs.forEach(function (x) { if (clipX != null && !clipX(x)) return; var y = f(x); if (y == null || !isFinite(y)) return; if (first) { ctx.moveTo(pl.X(x), pl.Y(y)); first = false; } else ctx.lineTo(pl.X(x), pl.Y(y)); });
    ctx.stroke(); ctx.restore();
  }
  fe.trace = trace;
  function range(a, b, n) { var o = []; for (var i = 0; i <= n; i++) o.push(a + (b - a) * i / n); return o; }
  fe.range = range;

  // Electro-optic response of a traveling-wave modulator (electrical dB), length L (m), index sum or difference dn,
  // microwave loss a0 (dB per cm per sqrt GHz). |(1 - exp(-g L)) / (g L)| with g = alpha + i beta.
  function eoResp(fGHz, Lm, dn, a0) {
    if (fGHz <= 0) return 0;
    var al = (a0 == null ? 0.5 : a0) / 8.686 * 100 * Math.sqrt(fGHz), be = TAU * fGHz * 1e9 * dn / 2.998e8;
    var gr = al * Lm, gi = be * Lm, e = Math.exp(-gr);
    var nr = 1 - e * Math.cos(gi), ni = e * Math.sin(gi);
    var m = Math.hypot(nr, ni) / Math.hypot(gr, gi);
    return 20 * Math.log(Math.max(m, 1e-6)) / Math.LN10;
  }
  fe.eoResp = eoResp;

  /* ---------- scene registry ---------- */
  var S = fe.scenes = fe.scenes || {};
  fe.scene = function (name, k, o) { return function (ctx, W, H, u) { var f = S[name]; if (f) f(ctx, W, H, u, k, o || {}); }; };

  /* ===== Incident: the chip that was turned around ===== */
  // 8 x 4 mm die, 5 mm traveling-wave electrode, on-chip 50 ohm termination at the far (east) end, no orientation marks
  var TA = mzm({ w: 8000, h: 4000, yc: 300, xe: 2500, rb: 400, padN: 1450, term: 'onchip', label: 'MZM-21   X-CUT LN', marks: false, refY: 1650, mmiX: 3640 });
  fe.dieTA = TA;

  function benchCard(ctx, U, x, y, w, rot) {
    // laminated card: sketch of the die in the intended orientation, RF input left, light left to right
    var h = w * 0.62;
    ctx.save(); ctx.translate(x, y); ctx.rotate(rot || 0);
    ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(-w / 2 + 4 * U.s, -h / 2 + 6 * U.s, w, h);
    ctx.fillStyle = '#eef0ea'; ctx.fillRect(-w / 2, -h / 2, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2 * U.s; ctx.strokeRect(-w / 2 + 1, -h / 2 + 1, w - 2, h - 2);
    var fs = Math.max(U.f(9), w * 0.07); font(ctx, fs, '700'); ctx.fillStyle = '#1b2026'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('LOAD THIS WAY', -w / 2 + w * 0.06, -h / 2 + h * 0.07);
    var dw = w * 0.62, dh = dw * 0.5, dx = -dw / 2 + w * 0.06, dy = -h * 0.12;
    ctx.strokeStyle = '#1b2026'; ctx.lineWidth = Math.max(1, 1.4 * U.s); ctx.strokeRect(dx, dy, dw, dh);
    ctx.fillStyle = '#c9a443'; ctx.fillRect(dx + dw * 0.18, dy + dh * 0.5, dw * 0.64, dh * 0.12);
    ctx.fillRect(dx + dw * 0.12, dy + dh * 0.12, dw * 0.06, dh * 0.38);
    font(ctx, Math.max(U.f(8), w * 0.055), '600'); ctx.fillStyle = '#1b2026';
    ctx.fillText('RF IN', dx + dw * 0.02, dy - fs * 1.15);
    ctx.fillText('50 Ω', dx + dw * 0.72, dy + dh + fs * 0.25);
    arrow(ctx, -w / 2 + w * 0.04, dy + dh * 0.56, dx - 2, dy + dh * 0.56, '#c0392b', Math.max(1.5, 2 * U.s), 6 * U.s);
    ctx.fillStyle = '#c0392b'; ctx.fillText('FIBER', -w / 2 + w * 0.04, dy + dh + fs * 0.25);
    ctx.restore();
  }

  function gelpak(ctx, U, r) {
    ctx.save();
    ctx.fillStyle = '#18191c'; ctx.fillRect(r.x - 10 * U.s, r.y - 10 * U.s, r.w + 20 * U.s, r.h + 20 * U.s);
    var g = ctx.createLinearGradient(r.x, r.y, r.x + r.w, r.y + r.h); g.addColorStop(0, '#3b3d42'); g.addColorStop(0.5, '#2c2e32'); g.addColorStop(1, '#35373b');
    ctx.fillStyle = g; ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.restore();
  }

  // Keyframe 0: the shared gel-pak holds dies in both orientations (nobody recorded which); Gary picks one that sits
  // at 180 degrees and sets it on the chuck as it is. Cut at u = 0.5 from the gel-pak to the chuck.
  var GP_ORI = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0], GP_PICK = 6;
  S.turned0 = function (ctx, W, H, u) {
    var U = units(ctx, W, H), lift, tx, ty, open;
    if (u < 0.5) {
      var a = u / 0.5;
      chuckBg(ctx, U, { holes: false, c0: '#4b4d52', c1: '#1c1d20' });
      var gp = { x: W * 0.05, y: H * 0.1, w: W * 0.9, h: H * 0.76 };
      gelpak(ctx, U, gp);
      var cellW = gp.w / 4, cellH = gp.h / 3, k = Math.min(cellW * 0.82 / TA.w, cellH * 0.66 / TA.h);
      for (var i = 0; i < 12; i++) {
        if (i === GP_PICK) continue;
        drawDie(ctx, TA, { x: gp.x + cellW * (i % 4 + 0.5), y: gp.y + cellH * (Math.floor(i / 4) + 0.5), k: k, rot: GP_ORI[i] * PI }, { us: U.s });
      }
      lift = eseg(a, 0.45, 0.8);
      var px = gp.x + cellW * (GP_PICK % 4 + 0.5), py = gp.y + cellH * (Math.floor(GP_PICK / 4) + 0.5) - lift * H * 0.03;
      drawDie(ctx, TA, { x: px, y: py, k: k * (1 + 0.05 * lift), rot: PI }, { lift: lift, us: U.s });
      var ta = eseg(a, 0.05, 0.35);
      tx = lerp(px + W * 0.3, px, ta); ty = lerp(py - H * 0.25, py, ta);
      open = lerp(TA.h * k + 34 * U.s, TA.h * k * (1 + 0.05 * lift) + 3 * U.s, seg(a, 0.33, 0.42));
      tweezers(ctx, tx, ty, PI, W * 0.42, open, U.s);
      callout(ctx, U, gp.x, gp.y - 8 * U.s, 'shared gel-pak', { valign: 'b', size: 12, bold: true });
    } else {
      var b2 = (u - 0.5) / 0.5;
      var v = { x: W * 0.4, y: H * 0.42, k: W * 0.58 / TA.w, rot: PI, us: U.s };
      chuckBg(ctx, U, { k: v.k, hx: v.x, hy: v.y });
      benchCard(ctx, U, W * 0.84, H * 0.72, W * 0.26, 0);
      lift = 1 - eseg(b2, 0.05, 0.4);
      drawDie(ctx, TA, { x: v.x, y: v.y, k: v.k * (1 + 0.05 * lift), rot: PI }, { lift: lift, us: U.s });
      var out = eseg(b2, 0.45, 0.75);
      open = TA.h * v.k * (1 + 0.05 * lift) + lerp(3, 40, seg(b2, 0.4, 0.48)) * U.s;
      if (b2 < 0.78) tweezers(ctx, v.x + out * W * 0.25, v.y - out * H * 0.75, PI, W * 0.5, open, U.s);
      callout(ctx, U, W * 0.98, H * 0.04, 'probe station chuck', { align: 'r', size: 12, bold: true });
      callout(ctx, U, W * 0.7, H * 0.94, 'bench card', { valign: 'b', align: 'r', size: 11, to: [W * 0.72, H * 0.66], alpha: seg(b2, 0.1, 0.2) });
      var tl = d2c(v, TA.w / 2, -TA.h / 2);
      callout(ctx, U, Math.min(tl[0], d2c(v, -TA.w / 2, TA.h / 2)[0]), Math.min(tl[1], d2c(v, -TA.w / 2, TA.h / 2)[1]) - 10 * U.s, 'turned 180°', { valign: 'b', size: 12, alpha: seg(b2, 0.6, 0.7), col: '#ffd0cc', border: P.bad });
    }
    vignette(ctx, U, 0.35);
  };

  // shared layout of the probe-station view used by keyframes 1-3
  function taView(U, rot) { return { x: U.W * 0.5, y: U.H * 0.44, k: U.W * 0.74 / TA.w, rot: rot, us: U.s }; }
  function taStation(ctx, U, v, o) {
    o = o || {};
    chuckBg(ctx, U, { k: v.k, hx: v.x - 2500 * v.k, hy: v.y + 2500 * v.k });
    var G = drawDie(ctx, TA, v, { us: U.s, skate: o.skate });
    // lensed fibres stay put on the lab frame: input on the left facet, output on the right facet
    var L0 = d2c(v, (v.rot ? 1 : -1) * TA.w / 2, TA.yc), R0 = d2c(v, (v.rot ? -1 : 1) * TA.w / 2, TA.yc);
    var gapPx = 6 * U.s;
    lensedFiber(ctx, Math.min(L0[0], R0[0]) - gapPx, L0[1], PI, v.k, { us: U.s, len: 30000 });
    lensedFiber(ctx, Math.max(L0[0], R0[0]) + gapPx, R0[1], 0, v.k, { us: U.s, len: 30000 });
    return G;
  }
  // probe contact position (canvas) for the RF input pads in view v
  function padPos(v, G) { var p = d2c(v, G.padIn.x, G.padIn.y), d = d2c(v, G.padIn.x, G.padIn.y - 1000); return { x: p[0], y: p[1], ang: Math.atan2(d[1] - p[1], d[0] - p[0]) }; }

  // Keyframe 1: light enters on the left (the designer's output); the probe moves to the RF input pads, now on the right
  S.turned1 = function (ctx, W, H, u) {
    var U = units(ctx, W, H), v = taView(U, PI);
    var G = taStation(ctx, U, v, { skate: u > 0.86 ? 'in' : null });
    var pin = padPos(v, G);
    // where the bench card says the RF input is: upper left (lab frame), above the termination today
    var term = d2c(v, G.resC.x, G.resC.y);
    var startX = term[0], startY = term[1] - 0 * U.s, startAng = -PI / 2;
    // motion: lift (0.05-0.2), swing over (0.2-0.7), lower and land (0.7-0.86)
    var tUp = eseg(u, 0.05, 0.2), tMv = eseg(u, 0.2, 0.7), tDn = eseg(u, 0.7, 0.86);
    var lift = tUp * (1 - tDn);
    var arcY = Math.sin(PI * tMv) * -U.H * 0.18;
    var x = lerp(startX, pin.x, tMv), y = lerp(startY, pin.y, tMv) + arcY;
    var ang = lerp(startAng, pin.ang, eseg(u, 0.3, 0.62));
    // light: continuous in the waveguides from the left
    lightPath(ctx, TA, v, 0.35 + 0.1 * Math.sin(u * 30), P.light);
    var lx = lerp(-TA.w / 2 - 300, TA.w / 2 + 300, (u * 2.2) % 1);
    lightPulse(ctx, TA, v, -lx, { us: U.s });   // die is rotated: canvas left is die +x
    gsgProbe(ctx, x, y, ang, v.k, { lift: lift, us: U.s });
    var a1 = seg(u, 0.02, 0.12);
    callout(ctx, U, U.W * 0.02, U.H * 0.62, 'light in', { to: [d2c(v, TA.w / 2, TA.yc)[0] - 30 * U.s, d2c(v, TA.w / 2, TA.yc)[1] + 4 * U.s], size: 11, alpha: a1 });
    callout(ctx, U, U.W * 0.03, U.H * 0.05, 'on-chip 50 Ω termination', { to: [term[0], term[1]], size: 11, alpha: seg(u, 0.08, 0.18) });
    var ip = d2c(v, G.padIn.x, G.padIn.y);
    callout(ctx, U, U.W * 0.98, U.H * 0.94, 'RF IN pads', { align: 'r', valign: 'b', to: [ip[0] + 14 * U.s, ip[1] - 10 * U.s], size: 11, alpha: seg(u, 0.55, 0.65) });
    axes(ctx, U, U.W * 0.93, U.H * 0.17, PI, { len: 30, size: 11 });
    scaleBar(ctx, U, U.W * 0.04, U.H * 0.94, v.k, 120 * U.s);
    vignette(ctx, U, 0.3);
  };

  // co / counter propagation along the electrode; returns nothing. dir: -1 counter (RF right to left in the frame)
  function travel(ctx, U, v, G, u, counter, speed) {
    var ph = (u * (speed || 1.6)) % 1;
    // light in canvas runs left to right; with rot = PI, canvas left is die +x
    var flip = v.rot ? -1 : 1;
    var xl = lerp(-TA.w / 2 - 200, TA.w / 2 + 200, ph) * flip;
    // RF: from the RF input pads, along the line; in die frame it runs west to east, i.e. same as die +x
    var sIn = fe.cpwS(TA, v, -TA.xe);
    var sOptEq = sIn + (xl - (-TA.xe));          // arc length where the optical pulse is, if it were on the line
    var sRF;
    if (counter) sRF = fe.cpwS(TA, v, -TA.xe) + (TA.xe * 2) * ph * 1.25 - TA.xe * 0.25;   // die-frame west to east = canvas right to left
    else sRF = sOptEq;
    lightPath(ctx, TA, v, 0.28, P.light);
    rfPacket(ctx, TA, v, sRF, { half: 1100, per: 900 });
    lightPulse(ctx, TA, v, xl, { us: U.s });
  }

  // Keyframe 2: counter-propagating waves on the die (top) and the EO response being measured (bottom)
  S.turned2 = function (ctx, W, H, u) {
    var U = units(ctx, W, H);
    var v = { x: W * 0.5, y: H * 0.255, k: Math.min(W * 0.62 / TA.w, H * 0.34 / TA.h), rot: PI, us: U.s };
    ctx.fillStyle = '#121519'; ctx.fillRect(0, 0, W, H);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H * 0.44); ctx.clip();
    var G = taStation(ctx, U, v, { skate: 'in' });
    var pin = padPos(v, G);
    travel(ctx, U, v, G, u, true, 2.4);
    gsgProbe(ctx, pin.x, pin.y, pin.ang, v.k, { us: U.s });
    vignette(ctx, U, 0.25);
    ctx.restore();
    ctx.fillStyle = 'rgba(10,14,19,0.78)'; ctx.fillRect(W * 0.24, H * 0.008, W * 0.52, H * 0.055);
    arrow(ctx, W * 0.33, H * 0.035, W * 0.45, H * 0.035, P.light, Math.max(2, 2.6 * U.s), 9 * U.s);
    arrow(ctx, W * 0.67, H * 0.035, W * 0.55, H * 0.035, P.rfP, Math.max(2, 2.6 * U.s), 9 * U.s);
    font(ctx, U.f(11), '600'); ctx.textBaseline = 'middle'; ctx.fillStyle = P.light; ctx.textAlign = 'right'; ctx.fillText('light', W * 0.32, H * 0.035);
    ctx.fillStyle = P.rfP; ctx.textAlign = 'left'; ctx.fillText('RF wave', W * 0.68, H * 0.035);
    // network analyzer: EO S21 with last week's reference
    var r = { x: W * 0.12, y: H * 0.535, w: W * 0.8, h: H * 0.27 };
    var pl = plot(ctx, U, r, { x0: 0, x1: 40, y0: -30, y1: 2, xt: [0, 10, 20, 30, 40], yt: [0, -10, -20, -30], xl: 'frequency (GHz)', yl: 'electro-optic response (dB)', padT: 30 });
    var fs = range(0.05, 40, 400);
    ctx.save(); ctx.setLineDash([5 * U.s, 4 * U.s]); trace(ctx, pl, [0, 40], function () { return -3; }, 'rgba(227,176,75,0.7)', Math.max(1, U.s)); ctx.restore();
    trace(ctx, pl, fs, function (f) { return eoResp(f, 0.005, 0.05, 0.5); }, 'rgba(200,205,212,0.75)', Math.max(1.5, 2 * U.s));
    var fcur = 40 * eseg(u, 0.05, 0.85);
    trace(ctx, pl, fs, function (f) { return eoResp(f, 0.005, 4.45, 0.5); }, P.bad, Math.max(2, 2.6 * U.s), function (f) { return f <= fcur; });
    if (fcur > 0.1) { var yy = eoResp(fcur, 0.005, 4.45, 0.5); ctx.fillStyle = P.bad; ctx.beginPath(); ctx.arc(pl.X(fcur), pl.Y(Math.max(-30, yy)), 3.5 * U.s, 0, TAU); ctx.fill(); }
    callout(ctx, U, pl.X(40) - 4 * U.s, pl.Y(-3) - 20 * U.s, 'last week', { align: 'r', valign: 'b', size: 10, col: '#d6dbe0' });
    if (fcur > 7) callout(ctx, U, pl.X(9), pl.Y(-6), '−3 dB near 6 GHz', { size: 10, to: [pl.X(6.0), pl.Y(-3)], col: '#ffd0cc', border: P.bad });
    if (fcur > 15) callout(ctx, U, pl.X(13.5) + 16 * U.s, pl.Y(-26), 'null', { size: 10, valign: 'b', to: [pl.X(13.5), pl.Y(-28)], col: '#ffd0cc', border: P.bad });
  };

  // Keyframe 3: freeze on the null, NOT OKAY; then the corrected take (die turned back, probe at the left, flat response)
  S.turned3 = function (ctx, W, H, u) {
    var U = units(ctx, W, H);
    var fix = eseg(u, 0.4, 0.62);
    var rot = PI * (1 - fix);
    var bw = Math.abs(TA.w * Math.cos(rot)) + Math.abs(TA.h * Math.sin(rot)), bh = Math.abs(TA.w * Math.sin(rot)) + Math.abs(TA.h * Math.cos(rot));
    var v = { x: W * 0.5, y: H * 0.255, k: Math.min(W * 0.62 / bw, H * 0.34 / bh), rot: rot, us: U.s };
    ctx.fillStyle = '#121519'; ctx.fillRect(0, 0, W, H);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H * 0.44); ctx.clip();
    chuckBg(ctx, U, { k: v.k, hx: v.x, hy: v.y });
    var G = drawDie(ctx, TA, v, { us: U.s, lift: Math.sin(PI * fix) * 0.6 });
    var L0 = d2c(v, -TA.w / 2, TA.yc);
    var fibOut = Math.sin(PI * fix) * 40 * U.s;
    var lx = Math.min(d2c(v, TA.w / 2, TA.yc)[0], d2c(v, -TA.w / 2, TA.yc)[0]), rx = Math.max(d2c(v, TA.w / 2, TA.yc)[0], d2c(v, -TA.w / 2, TA.yc)[0]);
    lensedFiber(ctx, lx - 6 * U.s - fibOut, v.y + TA.yc * v.k * Math.cos(rot), PI, v.k, { us: U.s, len: 30000 });
    lensedFiber(ctx, rx + 6 * U.s + fibOut, v.y - TA.yc * v.k * Math.cos(rot) * -1, 0, v.k, { us: U.s, len: 30000 });
    if (u < 0.36) {           // frozen frame of the counter-propagating state
      travel(ctx, U, v, G, 0.37, true, 1);
    } else if (u > 0.66) {
      var vv = v; travel(ctx, U, vv, G, (u - 0.66) * 1.8, false, 1.6);
    }
    // probe: on the right pads while frozen, lifted away during the turn, landed on the left pads after
    var pin = padPos(v, G);
    if (u < 0.4) gsgProbe(ctx, pin.x, pin.y, pin.ang, v.k, { us: U.s });
    else if (u > 0.62) { var dn = eseg(u, 0.62, 0.7); gsgProbe(ctx, pin.x, pin.y - (1 - dn) * 30 * U.s, pin.ang, v.k, { us: U.s, lift: 1 - dn }); }
    vignette(ctx, U, 0.25);
    ctx.restore();
    var r = { x: W * 0.12, y: H * 0.535, w: W * 0.8, h: H * 0.27 };
    var pl = plot(ctx, U, r, { x0: 0, x1: 40, y0: -30, y1: 2, xt: [0, 10, 20, 30, 40], yt: [0, -10, -20, -30], xl: 'frequency (GHz)', yl: 'electro-optic response (dB)', padT: 30 });
    var fs = range(0.05, 40, 400);
    ctx.save(); ctx.setLineDash([5 * U.s, 4 * U.s]); trace(ctx, pl, [0, 40], function () { return -3; }, 'rgba(227,176,75,0.7)', Math.max(1, U.s)); ctx.restore();
    trace(ctx, pl, fs, function (f) { return eoResp(f, 0.005, 0.05, 0.5); }, 'rgba(200,205,212,0.75)', Math.max(1.5, 2 * U.s));
    var badA = 1 - seg(u, 0.4, 0.5) * 0.7;
    ctx.save(); ctx.globalAlpha = badA; trace(ctx, pl, fs, function (f) { return eoResp(f, 0.005, 4.45, 0.5); }, P.bad, Math.max(2, 2.6 * U.s)); ctx.restore();
    if (u > 0.68) { var fc = 40 * eseg(u, 0.68, 0.88); trace(ctx, pl, fs, function (f) { return eoResp(f, 0.005, 0.05, 0.5); }, P.ok, Math.max(2, 2.8 * U.s), function (f) { return f <= fc; }); }
    if (u < 0.4) stamp(ctx, U, 'NOT OKAY', false, (u - 0.06) / 0.3, W * 0.76, H * 0.64);
    else if (u > 0.88) stamp(ctx, U, 'OKAY', true, (u - 0.88) / 0.1, W * 0.76, H * 0.64);
    if (u > 0.42 && u < 0.9) callout(ctx, U, W * 0.02, H * 0.02, 'corrected: loaded as on the bench card', { size: 10.5, alpha: seg(u, 0.42, 0.5) });
  };
})();

/* ===== cold opens and incidents built on the same library ===== */
(function () {
  'use strict';
  var fe = (typeof window !== 'undefined' ? window : this).LMS_DRAW.fe;
  var S = fe.scenes, P = fe.P, PI = Math.PI, TAU = 2 * PI;
  var units = fe.units, callout = fe.callout, stamp = fe.stamp, arrow = fe.arrow, axes = fe.axes, font = fe.font;
  var clamp = fe.clamp, lerp = fe.lerp, seg = fe.seg, eseg = fe.eseg, ease = fe.ease, hexA = fe.hexA;

  function plainBg(ctx, U, c0, c1) {
    var g = ctx.createRadialGradient(U.W * 0.5, U.H * 0.42, U.H * 0.1, U.W * 0.5, U.H * 0.5, U.W * 0.7);
    g.addColorStop(0, c0 || '#3a3d42'); g.addColorStop(1, c1 || '#17191c');
    ctx.fillStyle = g; ctx.fillRect(0, 0, U.W, U.H);
  }
  fe.plainBg = plainBg;
  function shortest(a0, a1) { var d = ((a1 - a0) % 360 + 540) % 360 - 180; return a0 + d; }

  /* ----- Many hands: the course render die (MZM-07, 6 x 6 mm) passed from hand to hand ----- */
  var M07 = fe.mzm({});
  fe.dieM07 = M07;
  var HANDS = [
    { labels: ['one die'], turns: [0] },
    { labels: ['spin coater', 'mask aligner chuck', 'mode solver', 'wafer supplier', 'shared gel-pak'], turns: [0, 90, 90, 180, 270] },
    { labels: ['partner lab', 'your own bench', 'a deal', 'the lab itself'], turns: [180, 270, 0, 90] },
    { labels: ['the test bench'], turns: [90] }
  ];
  function handAngle(k, u) {
    var hk = HANDS[k], n = hk.labels.length, j = Math.min(n - 1, Math.floor(u * n));
    var prev = j > 0 ? hk.turns[j - 1] : (k > 0 ? HANDS[k - 1].turns[HANDS[k - 1].turns.length - 1] : 0);
    var t = eseg(u * n - j, 0.15, 0.55);
    return { a: lerp(prev, shortest(prev, hk.turns[j]), t), j: j, label: hk.labels[j], moving: t > 0 && t < 1, t: t };
  }
  function inkDot(ctx, v, L) {   // red marker dot next to the L mark (die frame), turns with the die
    var p = fe.d2c(v, -L.w / 2 + 400, -L.h / 2 + 1300);
    ctx.fillStyle = 'rgba(205,30,40,0.9)'; ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(4, 160 * v.k), 0, TAU); ctx.fill();
  }
  S.hands = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    plainBg(ctx, U, '#45484d', '#1a1c1f');
    var cx = W * 0.5, cy = H * 0.45, kk = H * 0.58 / M07.w * (1 + 0.04 * u);
    // turntable
    ctx.save(); ctx.fillStyle = '#d9dbde'; ctx.beginPath(); ctx.ellipse(cx, cy + H * 0.02, H * 0.47, H * 0.44, 0, 0, TAU); ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 2 * U.s; ctx.stroke(); ctx.restore();
    var ha = handAngle(k, u), rot = ha.a * PI / 180;
    // the start orientation as a dashed outline, so a turn is visible
    if (k > 0) {
      ctx.save(); ctx.setLineDash([6 * U.s, 5 * U.s]); ctx.strokeStyle = 'rgba(40,44,50,0.7)'; ctx.lineWidth = 1.5 * U.s;
      ctx.strokeRect(cx - M07.w * kk / 2, cy - M07.h * kk / 2, M07.w * kk, M07.h * kk); ctx.restore();
    }
    var v = { x: cx, y: cy, k: kk, rot: rot, us: U.s };
    fe.drawDie(ctx, M07, v, { us: U.s, lift: ha.moving ? 0.35 * Math.sin(PI * ha.t) : 0 });
    inkDot(ctx, v, M07);
    axes(ctx, U, W * 0.86, H * 0.2, rot, { len: 34, size: 11 });
    // who is holding it, and the net turn from the start
    var hk = HANDS[k], jj = ha.t >= 1 ? ha.j : ha.j - 1, settled = jj >= 0 ? hk.turns[jj] : (k > 0 ? HANDS[k - 1].turns[HANDS[k - 1].turns.length - 1] : 0);
    var deg = ((settled % 360) + 360) % 360;
    callout(ctx, U, W * 0.04, H * 0.06, ha.label, { size: 15, bold: true });
    callout(ctx, U, W * 0.04, H * 0.06 + U.f(15) * 2.1, 'turned ' + deg + '° from the start', { size: 11, col: deg ? '#ffd0cc' : P.ink, border: deg ? P.bad : undefined });
    if (k === 3) stamp(ctx, U, 'NOT OKAY', false, (u - 0.45) / 0.4);
    fe.vignette(ctx, U, 0.3);
  };

  /* ----- wafers ----- */
  // 100 mm wafer with a primary flat. fa: direction of the flat (rad, canvas; 0 = three o'clock). o: { tint, lines (fn) }
  function waferPath(ctx, cx, cy, r, fa) {
    var phi = Math.acos(0.92);   // flat chord at 0.92 r (about 32 mm flat on 100 mm)
    ctx.beginPath(); ctx.arc(cx, cy, r, fa + phi, fa - phi + TAU); ctx.closePath();
  }
  fe.waferPath = waferPath;
  function wafer(ctx, U, cx, cy, r, fa, o) {
    o = o || {};
    ctx.save();
    ctx.save(); ctx.translate(5 * U.s, 7 * U.s); waferPath(ctx, cx, cy, r, fa); ctx.fillStyle = 'rgba(0,0,0,0.35)'; if (ctx.filter !== undefined) ctx.filter = 'blur(' + Math.round(5 * U.s) + 'px)'; ctx.fill(); ctx.restore();
    waferPath(ctx, cx, cy, r, fa);
    var g = ctx.createLinearGradient(cx - r, cy - r, cx + r, cy + r);
    g.addColorStop(0, o.c0 || 'rgba(232,214,160,0.92)'); g.addColorStop(0.5, o.c1 || 'rgba(196,176,120,0.88)'); g.addColorStop(1, o.c0 || 'rgba(226,208,150,0.92)');
    ctx.fillStyle = g; ctx.fill();
    ctx.clip();
    if (o.inner) o.inner(ctx);
    // soft specular band
    var sg = ctx.createLinearGradient(cx - r, cy - r * 0.3, cx + r, cy + r * 0.3);
    sg.addColorStop(0.35, 'rgba(255,255,255,0)'); sg.addColorStop(0.5, 'rgba(255,255,255,0.18)'); sg.addColorStop(0.65, 'rgba(255,255,255,0)');
    ctx.fillStyle = sg; ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);
    ctx.restore();
    ctx.save(); waferPath(ctx, cx, cy, r, fa); ctx.strokeStyle = 'rgba(90,70,30,0.9)'; ctx.lineWidth = Math.max(1.5, 2 * U.s); ctx.stroke(); ctx.restore();
  }
  fe.wafer = wafer;

  // yellow-room lighting overlay (lithography area)
  function yellowRoom(ctx, U, a) { ctx.fillStyle = 'rgba(255,190,40,' + (a == null ? 0.10 : a) + ')'; ctx.fillRect(0, 0, U.W, U.H); }

  // die grid on a wafer in the mask frame (waveguides vertical on the mask): small MZM layouts
  function maskDies(ctx, cx, cy, r, kk, alpha, layers) {
    var pitchX = 8000 * kk, pitchY = 4500 * kk;
    ctx.save(); ctx.globalAlpha *= alpha;
    for (var ix = -7; ix <= 7; ix++) for (var iy = -12; iy <= 12; iy++) {
      var x = cx + ix * pitchY, y = cy + iy * pitchX;
      if (Math.hypot(x - cx, y - cy) > r - 5000 * kk) continue;
      // one MZM per die, vertical: waveguide line, electrode bar and pads
      ctx.fillStyle = 'rgba(60,70,80,0.85)'; ctx.fillRect(x - 0.6, y - 3600 * kk, 1.2, 7200 * kk);
      if (layers > 1) { ctx.fillStyle = 'rgba(214,170,60,0.95)'; ctx.fillRect(x - 330 * kk, y - 2500 * kk, 660 * kk, 5000 * kk); ctx.fillRect(x - 700 * kk, y - 3000 * kk, 1000 * kk, 450 * kk); ctx.fillRect(x - 300 * kk, y + 2550 * kk, 1000 * kk, 450 * kk); }
    }
    ctx.restore();
  }

  /* ----- Incident: the flat at three o'clock ----- */
  S.flat = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    var cx = k === 4 ? W * 0.72 : W * 0.5, cy = H * 0.47, r = k === 4 ? H * 0.3 : H * 0.36;
    if (k === 0) {   // spin coater bowl, chuck decelerating, wafer stops at an arbitrary angle
      plainBg(ctx, U, '#4a4c50', '#1b1c1f');
      ctx.fillStyle = '#26282b'; ctx.beginPath(); ctx.arc(cx, cy, r * 1.22, 0, TAU); ctx.fill();
      ctx.strokeStyle = '#5b5e63'; ctx.lineWidth = 6 * U.s; ctx.stroke();
      var spin = 2.3 * TAU * Math.pow(1 - clamp(u * 1.25, 0, 1), 2);
      var fa = 30 * PI / 180 + spin;
      wafer(ctx, U, cx, cy, r, fa);
      var w = 2.3 * TAU * 2.5 * (1 - clamp(u * 1.25, 0, 1));   // angular speed proxy for blur streaks
      if (w > 0.3) { ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,' + Math.min(0.25, w * 0.02) + ')'; ctx.lineWidth = 2 * U.s; for (var i = 0; i < 4; i++) { ctx.beginPath(); ctx.arc(cx, cy, r * (0.5 + 0.12 * i), fa, fa + Math.min(2.5, w * 0.08)); ctx.stroke(); } ctx.restore(); }
      ctx.fillStyle = '#9da2a8'; ctx.beginPath(); ctx.arc(cx, cy, r * 0.2, 0, TAU); ctx.globalAlpha = 0.25; ctx.fill(); ctx.globalAlpha = 1;
      callout(ctx, U, W * 0.04, H * 0.06, 'spin coater', { size: 13, bold: true });
      callout(ctx, U, cx + r * 1.05 * Math.cos(fa) + 20 * U.s, cy + r * 1.05 * Math.sin(fa), 'flat', { to: [cx + r * 0.93 * Math.cos(fa), cy + r * 0.93 * Math.sin(fa)], size: 11, alpha: seg(u, 0.75, 0.85) });
      yellowRoom(ctx, U); fe.vignette(ctx, U, 0.35);
      return;
    }
    if (k === 1 || k === 3 || k === 4) {
      plainBg(ctx, U, '#5c5f64', '#202226');
      // aligner vacuum chuck with three alignment pins: two for the flat at six o'clock, one at three o'clock
      ctx.fillStyle = '#7c8087'; ctx.beginPath(); ctx.arc(cx, cy, r * 1.12, 0, TAU); ctx.fill();
      ctx.strokeStyle = 'rgba(30,32,36,0.45)'; ctx.lineWidth = 2 * U.s; [0.3, 0.55, 0.8].forEach(function (q) { ctx.beginPath(); ctx.arc(cx, cy, r * q, 0, TAU); ctx.stroke(); });
      var pins = [[cx - r * 0.28, cy + r * 0.92 + 7 * U.s], [cx + r * 0.28, cy + r * 0.92 + 7 * U.s], [cx + r * 0.92 + 7 * U.s, cy]];
      var slide = k === 1 ? eseg(u, 0.05, 0.55) : 1;
      var wx = lerp(cx - W * 0.55, cx, slide);
      var inner = null;
      if (k >= 3) inner = function (c) { maskDies(c, cx, cy, r, r / 50000, 1, 2); };
      wafer(ctx, U, wx, cy, r, 0, { inner: inner });
      pins.forEach(function (p) { ctx.fillStyle = '#d7dbe0'; ctx.beginPath(); ctx.arc(p[0], p[1], 7 * U.s, 0, TAU); ctx.fill(); ctx.strokeStyle = '#2a2c30'; ctx.lineWidth = 1.5 * U.s; ctx.stroke(); });
      if (k === 1) {
        // laminated procedure card taped to the bezel
        var cw = W * 0.24, chh = H * 0.15, cx0 = W * 0.73, cy0 = H * 0.76;
        ctx.save(); ctx.translate(cx0, cy0); ctx.rotate(-0.04);
        ctx.fillStyle = '#f1efe6'; ctx.fillRect(0, 0, cw, chh); ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 3 * U.s; ctx.strokeRect(0, 0, cw, chh);
        ctx.fillStyle = 'rgba(230,220,150,0.6)'; ctx.fillRect(cw * 0.35, -6 * U.s, cw * 0.3, 14 * U.s);
        font(ctx, Math.max(U.f(12), cw * 0.075), '800'); ctx.fillStyle = '#16191d'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText("FLAT AT SIX O'CLOCK", cw / 2, chh * 0.32);
        ctx.strokeStyle = '#16191d'; ctx.lineWidth = 1.5 * U.s; var mx = cw / 2, my = chh * 0.7, mr = chh * 0.22;
        waferPath(ctx, mx, my, mr, PI / 2); ctx.stroke();
        ctx.restore();
        callout(ctx, U, W * 0.04, H * 0.06, 'mask aligner chuck', { size: 13, bold: true });
        callout(ctx, U, cx + r + 28 * U.s, cy - 40 * U.s, "flat at three o'clock", { to: [cx + r * 0.93, cy], size: 11, alpha: seg(u, 0.55, 0.65), col: '#ffd0cc', border: P.bad });
        yellowRoom(ctx, U); fe.vignette(ctx, U, 0.35);
        return;
      }
      if (k === 3 || k === 4) {
        var ga = eseg(u, 0.25, 0.6);
        if (k === 4) ga = 1;
        // crystal axes on this wafer: y perpendicular to the flat (horizontal here), z along the flat (vertical)
        ctx.save(); ctx.globalAlpha = ga;
        ctx.setLineDash([8 * U.s, 6 * U.s]); arrow(ctx, cx - r * 0.75, cy - r * 0.08, cx + r * 0.75, cy - r * 0.08, '#4fd27f', Math.max(2, 3 * U.s), 11 * U.s); ctx.setLineDash([]);
        ctx.restore();
        callout(ctx, U, cx + r * 0.2, cy - r * 0.08 - 34 * U.s, 'intended: along y', { size: 11, col: '#c8f5d6', border: '#4fd27f', alpha: ga });
        callout(ctx, U, cx - r * 0.12, cy + r * 0.5, 'waveguides: along z', { size: 11, col: '#ffd0cc', border: P.bad, alpha: ga, align: 'r', to: [cx - r * 0.14 + 0, cy + r * 0.42] });
        if (k === 3) axes(ctx, U, W * 0.87, H * 0.2, 0, { len: 34, size: 11, alpha: ga });
        callout(ctx, U, W * 0.04, H * 0.06, k === 3 ? 'layers 2 to 5: aligned to layer 1' : 'six weeks later', { size: 13, bold: true });
        if (k === 4) {
          // transfer curve on the test bay monitor: reference and measured (about 8x V pi)
          var rr = { x: W * 0.09, y: H * 0.24, w: W * 0.38, h: H * 0.42 };
          var pl = fe.plot(ctx, U, rr, { x0: -8, x1: 8, y0: 0, y1: 1.05, xt: [-8, -4, 0, 4, 8], yt: [0, 0.5, 1], xl: 'voltage / reference Vπ', yl: 'transmission', padL: 34, padT: 22 });
          var xs = fe.range(-8, 8, 400);
          fe.trace(ctx, pl, xs, function (x) { return Math.pow(Math.cos(PI * x / 2 / 1 + PI / 4), 2); }, 'rgba(200,205,212,0.7)', Math.max(1, 1.4 * U.s));
          var cut = -8 + 16 * eseg(u, 0.05, 0.5);
          fe.trace(ctx, pl, xs, function (x) { return Math.pow(Math.cos(PI * x / 2 / 8 + PI / 4), 2); }, P.bad, Math.max(1.5, 2.4 * U.s), function (x) { return x <= cut; });
          stamp(ctx, U, 'NOT OKAY', false, (u - 0.55) / 0.35, W * 0.74, H * 0.16);
        }
        fe.vignette(ctx, U, 0.3);
      }
      return;
    }
    if (k === 2) {   // eyepiece: chrome waveguide lines on the mask slide over the wafer
      ctx.fillStyle = '#0b0c0e'; ctx.fillRect(0, 0, W, H);
      var er = H * 0.43, ex = W * 0.42;
      ctx.save(); ctx.beginPath(); ctx.arc(ex, cy, er, 0, TAU); ctx.clip();
      ctx.fillStyle = '#c9b682'; ctx.fillRect(0, 0, W, H);
      var sh = (1 - eseg(u, 0.05, 0.7)) * er * 0.9;
      ctx.globalAlpha = clamp(u * 2.5, 0, 1);
      for (var i = -8; i <= 8; i++) { var lx = ex + i * er * 0.13 + sh; ctx.fillStyle = 'rgba(40,40,44,0.9)'; ctx.fillRect(lx - 2.2 * U.s, 0, 4.4 * U.s, H); }
      ctx.globalAlpha = 1;
      ctx.restore();
      fe.eyepiece(ctx, U, ex, cy, er);
      // lot traveler sketch: waveguides perpendicular to the flat (flat at six o'clock)
      var tx = W * 0.74, ty = H * 0.14, tw = W * 0.22, th = H * 0.46;
      ctx.save(); ctx.translate(tx, ty); ctx.rotate(0.03);
      ctx.fillStyle = '#f4f2ea'; ctx.fillRect(0, 0, tw, th);
      ctx.strokeStyle = 'rgba(30,40,120,0.25)'; ctx.lineWidth = 1; for (var y = th * 0.12; y < th; y += th * 0.06) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(tw, y); ctx.stroke(); }
      font(ctx, Math.max(U.f(10), tw * 0.075), '700'); ctx.fillStyle = '#1c2a6b'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('LOT TRAVELER, LAYER 1', tw * 0.06, th * 0.03);
      var mx = tw * 0.5, my = th * 0.56, mr = tw * 0.32;
      ctx.strokeStyle = '#1c2a6b'; ctx.lineWidth = 1.6 * U.s; waferPath(ctx, mx, my, mr, PI / 2); ctx.stroke();
      ctx.save(); waferPath(ctx, mx, my, mr, PI / 2); ctx.clip(); for (var j = -3; j <= 3; j++) { ctx.beginPath(); ctx.moveTo(mx + j * mr * 0.25, my - mr); ctx.lineTo(mx + j * mr * 0.25, my + mr); ctx.stroke(); } ctx.restore();
      ctx.fillStyle = '#1c2a6b'; ctx.textAlign = 'center'; ctx.fillText('flat', mx, my + mr * 1.05);
      ctx.restore();
      callout(ctx, U, W * 0.04, H * 0.06, 'aligner eyepiece', { size: 13, bold: true });
      yellowRoom(ctx, U, 0.06);
    }
  };

  /* ----- Cold open: which edge is z ----- */
  // blank square x-cut chip (no pattern, no flat, no mark), 5 mm
  function blankChip(ctx, U, cx, cy, s, rot) {
    ctx.save(); ctx.translate(cx + 5 * U.s, cy + 7 * U.s); ctx.rotate(rot); ctx.fillStyle = 'rgba(0,0,0,0.4)'; if (ctx.filter !== undefined) ctx.filter = 'blur(' + Math.round(5 * U.s) + 'px)'; ctx.fillRect(-s / 2, -s / 2, s, s); ctx.restore();
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
    ctx.fillStyle = P.siEdge; ctx.fillRect(-s / 2 - 2, -s / 2 - 2, s + 4, s + 4);
    var g = ctx.createLinearGradient(-s / 2, -s / 2, s / 2, s / 2); g.addColorStop(0, P.filmHi); g.addColorStop(0.6, P.film); g.addColorStop(1, P.filmLo);
    ctx.fillStyle = g; ctx.fillRect(-s / 2, -s / 2, s, s);
    ctx.restore();
  }
  fe.blankChip = blankChip;
  S.whichz = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    if (k === 0 || k === 1) {
      plainBg(ctx, U, '#3f4246', '#141517');
      var push = k === 0 ? 1 + 0.25 * ease(u) : 1.25 + 0.5 * eseg(u, 0, 0.15);
      var cx = W * 0.5, cy = H * 0.46, s = H * 0.42 * push;
      // gel-pack below (out of focus)
      if (k === 0) { ctx.fillStyle = '#222326'; ctx.fillRect(W * 0.18, H * 0.66, W * 0.64, H * 0.4); ctx.fillStyle = 'rgba(255,255,255,0.04)'; ctx.fillRect(W * 0.18, H * 0.66, W * 0.64, H * 0.05); }
      var wob = k === 0 ? 0.08 * Math.sin(u * 5) : 0;
      blankChip(ctx, U, cx, cy, s, wob);
      fe.tweezers(ctx, cx + s * 0.06, cy - s * 0.42, PI / 2 + 0.22 + wob, H * 0.9, 3 * U.s, U.s);
      if (k === 0) {
        // dimension lines
        ctx.save(); ctx.strokeStyle = P.line; ctx.lineWidth = Math.max(1, U.s); var y0 = cy + s * 0.5 + 18 * U.s;
        ctx.beginPath(); ctx.moveTo(cx - s / 2, y0); ctx.lineTo(cx + s / 2, y0); ctx.stroke(); ctx.restore();
        callout(ctx, U, cx, y0 - 2 * U.s, '5 mm', { align: 'c', size: 11, alpha: seg(u, 0.2, 0.3) });
        callout(ctx, U, W * 0.04, H * 0.06, 'no flat, no mark', { size: 13, bold: true, alpha: seg(u, 0.35, 0.45) });
        yellowRoom(ctx, U, 0.12);
      } else {
        // axes jump by 90 degrees at each blink
        var n = Math.floor(u * 4), f = u * 4 - n, blink = f < 0.06;
        axes(ctx, U, cx - s * 0.18, cy + s * 0.18, n * PI / 2, { len: s * 0.3 / U.s, size: 14, yLab: 'y?', zLab: 'z?', box: false, title: false });
        if (blink) { ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.fillRect(0, 0, W, H); }
        yellowRoom(ctx, U, 0.12);
      }
      fe.vignette(ctx, U, 0.4);
      return;
    }
    // k 2, 3: device frame vs model frame
    ctx.fillStyle = '#101317'; ctx.fillRect(0, 0, W, H);
    // left: the chip as loaded (axes as they actually are)
    var lx = W * 0.24, ly = H * 0.44, ls = H * 0.36;
    blankChip(ctx, U, lx, ly, ls, 0);
    axes(ctx, U, lx - ls * 0.15, ly + ls * 0.15, 0, { len: ls * 0.3 / U.s, size: 13, box: false, title: false });
    callout(ctx, U, lx, ly - ls * 0.5 - 14 * U.s, 'device', { align: 'c', valign: 'b', size: 12, bold: true });
    // right: model viewport, x-cut ridge cross-section with the crystal frame set wrong (z out of the section)
    var mx = W * 0.47, my = H * 0.1, mw = W * 0.49, mh = H * 0.66;
    ctx.fillStyle = '#1d232a'; ctx.fillRect(mx, my, mw, mh); ctx.strokeStyle = 'rgba(160,175,190,0.4)'; ctx.lineWidth = 1; ctx.strokeRect(mx + 0.5, my + 0.5, mw - 1, mh - 1);
    var bx = mx + mw * 0.5, by = my + mh * 0.62, sc = mw / 14;   // um to px
    ctx.fillStyle = '#5d6b78'; ctx.fillRect(mx + 4, by + 0.3 * sc, mw - 8, mh * 0.3);           // oxide
    ctx.fillStyle = '#9b8fa1'; ctx.fillRect(mx + 4, by, mw - 8, 0.3 * sc);                        // slab
    ctx.beginPath(); ctx.moveTo(bx - 0.9 * sc, by); ctx.lineTo(bx - 0.75 * sc, by - 0.3 * sc); ctx.lineTo(bx + 0.75 * sc, by - 0.3 * sc); ctx.lineTo(bx + 0.9 * sc, by); ctx.closePath(); ctx.fill();
    ctx.fillStyle = P.au; ctx.fillRect(bx - 5.5 * sc, by - 0.6 * sc, 3.2 * sc, 0.6 * sc); ctx.fillRect(bx + 2.3 * sc, by - 0.6 * sc, 3.2 * sc, 0.6 * sc);
    // model coordinate dropdown
    font(ctx, U.f(11), '', false); ctx.fillStyle = '#c7d0d9'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('Material frame', mx + 12 * U.s, my + 18 * U.s);
    ctx.fillStyle = '#0e1216'; ctx.fillRect(mx + 12 * U.s, my + 30 * U.s, mw * 0.55, 22 * U.s); ctx.strokeStyle = '#6d7a87'; ctx.strokeRect(mx + 12 * U.s + 0.5, my + 30 * U.s + 0.5, mw * 0.55, 22 * U.s);
    ctx.fillStyle = '#e6ebf0'; ctx.fillText('Global coordinate system  ▾', mx + 18 * U.s, my + 41 * U.s);
    // model triad: section plane is the x-z plane of the device, but the model puts crystal z along the waveguide (out of the section)
    var tx = mx + mw * 0.83, ty = my + mh * 0.3, tl = 34 * U.s;
    arrow(ctx, tx, ty, tx + tl, ty, '#9fe3ff', 2 * U.s, 8 * U.s); arrow(ctx, tx, ty, tx, ty - tl, '#ffffff', 2 * U.s, 8 * U.s);
    ctx.strokeStyle = '#ffc46b'; ctx.lineWidth = 2 * U.s; ctx.beginPath(); ctx.arc(tx, ty, 6 * U.s, 0, TAU); ctx.moveTo(tx - 4 * U.s, ty - 4 * U.s); ctx.lineTo(tx + 4 * U.s, ty + 4 * U.s); ctx.moveTo(tx + 4 * U.s, ty - 4 * U.s); ctx.lineTo(tx - 4 * U.s, ty + 4 * U.s); ctx.stroke();
    font(ctx, U.f(12), '700'); ctx.textAlign = 'center'; ctx.fillStyle = '#9fe3ff'; ctx.fillText('y', tx + tl + 10 * U.s, ty); ctx.fillStyle = '#ffffff'; ctx.fillText('x', tx, ty - tl - 10 * U.s); ctx.fillStyle = '#ffc46b'; ctx.fillText('z', tx - 14 * U.s, ty + 12 * U.s);
    callout(ctx, U, mx + mw / 2, my - 6 * U.s, 'model', { align: 'c', valign: 'b', size: 12, bold: true });
    // calendar: weeks pass
    var wk = k === 2 ? 1 + Math.floor(12 * ease(u)) : 13;
    callout(ctx, U, mx + mw - 4 * U.s, my + mh + 10 * U.s, 'week ' + wk, { align: 'r', size: 12, mono: true });
    if (k === 3) {
      callout(ctx, U, W * 0.04, H * 0.84, 'device: z across the electrode gap    model: z along the waveguide', { size: 11, col: '#ffd0cc', border: P.bad, valign: 'b', alpha: seg(u, 0.1, 0.2) });
      stamp(ctx, U, 'NOT OKAY', false, (u - 0.3) / 0.4);
    }
  };
})();

/* ===== rings, plots, cross-sections, fixtures ===== */
(function () {
  'use strict';
  var fe = (typeof window !== 'undefined' ? window : this).LMS_DRAW.fe;
  var S = fe.scenes, P = fe.P, PI = Math.PI, TAU = 2 * PI;
  var units = fe.units, callout = fe.callout, stamp = fe.stamp, arrow = fe.arrow, axes = fe.axes, font = fe.font;
  var clamp = fe.clamp, lerp = fe.lerp, seg = fe.seg, eseg = fe.eseg, ease = fe.ease, hexA = fe.hexA, range = fe.range;

  function filmBg(ctx, U) {
    var g = ctx.createRadialGradient(U.W * 0.5, U.H * 0.45, U.H * 0.05, U.W * 0.5, U.H * 0.5, U.W * 0.7);
    g.addColorStop(0, P.filmHi); g.addColorStop(0.7, P.film); g.addColorStop(1, P.filmLo);
    ctx.fillStyle = g; ctx.fillRect(0, 0, U.W, U.H);
  }
  function mix(c1, c2, t) {
    var a = parseInt(c1.slice(1), 16), b = parseInt(c2.slice(1), 16);
    var r = Math.round(lerp((a >> 16) & 255, (b >> 16) & 255, t)), g = Math.round(lerp((a >> 8) & 255, (b >> 8) & 255, t)), bl = Math.round(lerp(a & 255, b & 255, t));
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
  }
  function glow(ctx, x, y, r, col, a) {
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    var g = ctx.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, 'rgba(255,255,255,' + (0.85 * a) + ')'); g.addColorStop(0.3, hexA(col, 0.85 * a)); g.addColorStop(1, hexA(col, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill(); ctx.restore();
  }

  /* ----- Cold open: the gentle bend (x-cut ring, TE and TM effective indices cross at four azimuths) ----- */
  // TE effective index rises from about n_e (propagation along y) to about n_o (along z); TM (field along x) stays near n_o
  // minus the geometric offset, so the two cross where the propagation direction is TC from y (four azimuths)
  var TC = 65 * PI / 180, SC = Math.sin(TC);
  function crossAngles() { return [PI / 2 - TC, PI / 2 + TC, -PI / 2 - TC, -PI / 2 + TC]; }
  function ringScene(ctx, U, R, k, u, o) {
    var W = U.W, H = U.H, cx = W * 0.5, cy = H * 0.44, kk = H * 0.33 / R;
    filmBg(ctx, U);
    var tw = Math.max(7.5 * kk, 2.2 * U.s), rw = 1.5 * kk;
    var gap = R > 300 ? 0.6 : 0.6, ybus = cy + R * kk + (gap + 1.5 + 0.75) * kk + tw * 0.5;
    ctx.strokeStyle = P.slab; ctx.lineWidth = tw;
    ctx.beginPath(); ctx.arc(cx, cy, R * kk, 0, TAU); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, ybus); ctx.lineTo(W, ybus); ctx.stroke();
    if (rw > 0.8) { ctx.strokeStyle = P.film; ctx.lineWidth = rw; ctx.beginPath(); ctx.arc(cx, cy, R * kk, 0, TAU); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, ybus); ctx.lineTo(W, ybus); ctx.stroke(); }
    // crossing azimuths: small ticks outside the ring
    var ca = crossAngles();
    ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = Math.max(1.5, 2 * U.s);
    ca.forEach(function (a) { var r0 = R * kk + 8 * U.s, r1 = R * kk + 20 * U.s; ctx.beginPath(); ctx.moveTo(cx + r0 * Math.cos(a), cy + r0 * Math.sin(a)); ctx.lineTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a)); ctx.stroke(); });
    // light in the bus, pulse in the ring travelling counterclockwise from the coupler at six o'clock
    var laps = o.laps || 1.25, ph = (o.ph != null ? o.ph : u) * laps;
    var alpha = PI / 2 - ph * TAU;     // canvas angle, decreasing = counterclockwise on screen
    var passed = 0, aa = PI / 2 - ph * TAU;
    // count crossings passed since the start
    for (var n = 0; n < 8; n++) ca.forEach(function (c) { var t = c - n * TAU; if (t < PI / 2 && t >= aa) passed++; });
    var tm = o.big ? (passed > 0 ? 1 : 0) : Math.min(0.45, passed * 0.12);
    if (o.big && passed > 0) { var cnear = ca.map(function (c) { var d = ((alpha - c) % TAU + TAU) % TAU; return TAU - d; }); var dmin = Math.min.apply(null, cnear) * R * kk; tm = clamp(dmin / (40 * U.s), 0, 1); if (passed > 1) tm = 1; }
    var col = mix(P.light, P.lightTM, tm);
    ctx.save(); ctx.strokeStyle = hexA(P.light, 0.5); ctx.lineWidth = Math.max(1.5, 2 * U.s); ctx.beginPath(); ctx.moveTo(0, ybus); ctx.lineTo(cx, ybus); ctx.stroke(); ctx.restore();
    // trail
    for (var i = 0; i < 14; i++) { var a2 = alpha + i * 0.035 * (100 / Math.max(100, R)) * 3; glow(ctx, cx + R * kk * Math.cos(a2), cy + R * kk * Math.sin(a2), 9 * U.s * (1 - i / 16), col, 0.5 * (1 - i / 14)); }
    glow(ctx, cx + R * kk * Math.cos(alpha), cy + R * kk * Math.sin(alpha), 13 * U.s, col, 1);
    if (o.exit) {   // the pulse leaving through the bus on the right
      var ex = cx + (W - cx) * o.exit;
      ctx.save(); ctx.strokeStyle = hexA(P.lightTM, 0.6); ctx.lineWidth = Math.max(1.5, 2 * U.s); ctx.beginPath(); ctx.moveTo(cx, ybus); ctx.lineTo(ex, ybus); ctx.stroke(); ctx.restore();
      glow(ctx, ex, ybus, 13 * U.s, P.lightTM, 1);
    }
    axes(ctx, U, W * 0.88, H * 0.2, 0, { len: 32, size: 11 });
    fe.scaleBar(ctx, U, W * 0.04, H * 0.8, kk, 110 * U.s);
    // legend
    font(ctx, U.f(11), '700'); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    var lx = W * 0.04, ly = H * 0.08;
    ctx.fillStyle = 'rgba(10,14,19,0.7)'; ctx.fillRect(lx - 6 * U.s, ly - 12 * U.s, 120 * U.s, 46 * U.s);
    glow(ctx, lx + 8 * U.s, ly, 7 * U.s, P.light, 1); ctx.fillStyle = P.ink; ctx.fillText('TE', lx + 22 * U.s, ly);
    glow(ctx, lx + 8 * U.s, ly + 22 * U.s, 7 * U.s, P.lightTM, 1); ctx.fillStyle = P.ink; ctx.fillText('TM', lx + 22 * U.s, ly + 22 * U.s);
    return { cx: cx, cy: cy, r: R * kk, ca: ca };
  }
  // effective index vs propagation direction (TE follows the crystal, TM does not)
  function neffInset(ctx, U, r) {
    var pl = fe.plot(ctx, U, r, { x0: 0, x1: 180, y0: 0, y1: 1, xt: [0, 90, 180], xf: function (t) { return t === 90 ? 'z' : 'y'; }, yt: [], xl: 'propagation direction', yl: 'effective index', padL: 16, padT: 22 });
    var xs = range(0, 180, 120);
    fe.trace(ctx, pl, xs, function (d) { var c = Math.sin(d * PI / 180); return 0.15 + 0.75 * c * c; }, P.light, Math.max(1.5, 2 * U.s));
    fe.trace(ctx, pl, xs, function () { return 0.15 + 0.75 * SC * SC; }, P.lightTM, Math.max(1.5, 2 * U.s));
    [TC, PI - TC].forEach(function (a) { var d = a * 180 / PI; ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(pl.X(d), pl.Y(0.15 + 0.75 * SC * SC), 3.5 * U.s, 0, TAU); ctx.fill(); });
  }
  S.gentle = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    if (k === 0) { ringScene(ctx, U, 100, k, u, { laps: 1.6 }); fe.vignette(ctx, U, 0.3); return; }
    if (k === 1) {   // layout editor: radius handle dragged from 100 um to 1000 um
      ctx.fillStyle = '#0d1014'; ctx.fillRect(0, 0, W, H);
      var R = 100 * Math.pow(10, eseg(u, 0.1, 0.8)), cx = W * 0.5, cy = H * 0.47;
      var kk = H * 0.36 / 1000 * (1 + 2.2 * (1 - eseg(u, 0.1, 0.8)));
      var gp = 50 * kk; if (gp < 10 * U.s) gp *= Math.ceil(10 * U.s / gp);
      ctx.fillStyle = 'rgba(140,160,180,0.35)';
      for (var gx = cx % gp; gx < W; gx += gp) for (var gy = cy % gp; gy < H; gy += gp) ctx.fillRect(gx, gy, 1.2, 1.2);
      var rp = R * kk;
      ctx.lineWidth = Math.max(2, 1.5 * kk + 2); ctx.strokeStyle = 'rgba(214,94,214,0.95)'; ctx.beginPath(); ctx.arc(cx, cy, rp, 0, TAU); ctx.stroke();
      ctx.strokeStyle = 'rgba(214,94,214,0.95)'; ctx.beginPath(); ctx.moveTo(0, cy + rp + 4); ctx.lineTo(W, cy + rp + 4); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.setLineDash([5 * U.s, 4 * U.s]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + rp * 0.707, cy - rp * 0.707); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(cx + rp * 0.707 - 5 * U.s, cy - rp * 0.707 - 5 * U.s, 10 * U.s, 10 * U.s);
      callout(ctx, U, cx + rp * 0.707 + 14 * U.s, cy - rp * 0.707 - 30 * U.s, 'R = ' + Math.round(R) + ' µm', { size: 13, mono: true });
      fe.scaleBar(ctx, U, W * 0.04, H * 0.92, kk, 110 * U.s);
      callout(ctx, U, W * 0.04, H * 0.05, 'layout editor', { size: 12, bold: true });
      return;
    }
    var o = { big: true, laps: k === 2 ? 0.95 : 0.95, ph: k === 2 ? eseg(u, 0, 1) * 0.92 : 0.92 };
    if (k === 3) o.exit = eseg(u, 0.0, 0.3) * 0.85;
    var rs = ringScene(ctx, U, 1000, k, u, o);
    if (k === 2) {
      var a0 = rs.ca[0];
      callout(ctx, U, rs.cx + rs.r + 30 * U.s, rs.cy + rs.r * 0.05, 'TE and TM indices equal', { size: 11, to: [rs.cx + (rs.r + 14 * U.s) * Math.cos(a0), rs.cy + (rs.r + 14 * U.s) * Math.sin(a0)], alpha: seg(u, 0.1, 0.2) });
      neffInset(ctx, U, { x: W * 0.06, y: H * 0.3, w: W * 0.19, h: H * 0.22 });
    }
    if (k === 3) stamp(ctx, U, 'NOT OKAY', false, (u - 0.35) / 0.4);
    fe.vignette(ctx, U, 0.3);
  };

  /* ----- Cold open: the record Q (photorefractive blue shift; sweep direction decides the shape) ----- */
  // the resonance shifts blue with stored energy E and relaxes toward -BETA E with a response length TAU
  // (in detuning units of the sweep): short to long the resonance runs into the laser (narrow dip),
  // long to short it runs ahead of it (broad dip)
  var BETA = 4, TRESP = 1;
  function sweep(xs, dir) {
    var out = {}, n = 1200, x0 = -8, x1 = 4, r = 0, dx = (x1 - x0) / n;
    for (var i = 0; i <= n; i++) {
      var x = dir > 0 ? x0 + i * dx : x1 - i * dx, E = 1 / (1 + 4 * (x - r) * (x - r));
      out[Math.round(x * 100)] = 1 - 0.9 * E;
      r += (-BETA * E - r) * dx / TRESP;
    }
    return function (x) { return out[Math.round(x * 100)]; };
  }
  var QX = range(-5, 3, 800), QF = null, QB = null, QFW = null;
  function qTraces() {
    if (QF) return;
    QF = sweep(QX, 1); QB = sweep(QX, -1);
    // apparent FWHM of the forward dip
    var tmin = 1, xs = []; QX.forEach(function (x) { tmin = Math.min(tmin, QF(x)); });
    QX.forEach(function (x) { if (QF(x) < (1 + tmin) / 2) xs.push(x); });
    QFW = xs.length ? xs[xs.length - 1] - xs[0] : 1;
  }
  function qPlot(ctx, U, r, title) {
    return fe.plot(ctx, U, r, { x0: -5, x1: 3, y0: 0, y1: 1.08, xt: [-4, -2, 0, 2], yt: [0, 0.5, 1], xl: 'laser detuning (cold linewidths)', yl: title || 'transmission', padL: 34 });
  }
  S.recordq = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H); qTraces();
    ctx.fillStyle = '#101317'; ctx.fillRect(0, 0, W, H);
    var lw = Math.max(2, 2.6 * U.s);
    if (k <= 1) {
      var r = { x: W * 0.12, y: H * 0.12, w: W * 0.76, h: H * 0.56 };
      var pl = qPlot(ctx, U, r);
      var xc = k === 0 ? -5 + 8 * eseg(u, 0.05, 0.9) : 3;
      fe.trace(ctx, pl, QX, QF, P.light, lw, function (x) { return x <= xc; });
      if (k === 0) { ctx.strokeStyle = P.warn; ctx.setLineDash([4 * U.s, 4 * U.s]); ctx.lineWidth = Math.max(1, U.s); ctx.beginPath(); ctx.moveTo(pl.X(xc), r.y); ctx.lineTo(pl.X(xc), r.y + r.h); ctx.stroke(); ctx.setLineDash([]); }
      callout(ctx, U, r.x + 8 * U.s, r.y + r.h * 0.62, 'sweep: short to long wavelength', { size: 11, col: '#bfefff' });
      if (k === 1) {
        // Lorentzian fit to the narrow dip, and the message
        var tmin = 1, xm = 0; QX.forEach(function (x) { if (QF(x) < tmin) { tmin = QF(x); xm = x; } });
        var hw = QFW / 2;
        ctx.save(); ctx.setLineDash([6 * U.s, 4 * U.s]); fe.trace(ctx, pl, QX, function (x) { var d = (x - xm) / hw; return 1 - (1 - tmin) / (1 + d * d); }, '#ffffff', Math.max(1, 1.4 * U.s)); ctx.restore();
        callout(ctx, U, pl.X(xm) + 40 * U.s, pl.Y(0.42), 'fit: Q × ' + (1 / QFW).toFixed(1), { size: 11, to: [pl.X(xm + hw), pl.Y((1 + tmin) / 2)], alpha: seg(u, 0.1, 0.2), mono: true });
        // chat message tile
        var a = seg(u, 0.25, 0.35), cx0 = W * 0.6, cy0 = H * 0.74;
        ctx.save(); ctx.globalAlpha = a;
        ctx.fillStyle = '#1f6feb'; var mw = W * 0.24, mh = H * 0.08; ctx.fillRect(cx0, cy0, mw, mh);
        font(ctx, U.f(16), '800'); ctx.fillStyle = '#ffffff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('RECORD Q', cx0 + mw / 2, cy0 + mh / 2);
        font(ctx, U.f(10)); ctx.fillStyle = P.inkDim; ctx.textAlign = 'right'; ctx.fillText('Gary', cx0 + mw, cy0 - 8 * U.s);
        ctx.restore();
      }
      return;
    }
    if (k === 2) {
      var r2 = { x: W * 0.12, y: H * 0.12, w: W * 0.76, h: H * 0.56 };
      var pl2 = qPlot(ctx, U, r2);
      fe.trace(ctx, pl2, QX, QF, hexA(P.light, 0.35), Math.max(1.5, 2 * U.s));
      var xc2 = 3 - 8 * eseg(u, 0.05, 0.9);
      fe.trace(ctx, pl2, QX, QB, P.lightTM, lw, function (x) { return x >= xc2; });
      ctx.strokeStyle = P.warn; ctx.setLineDash([4 * U.s, 4 * U.s]); ctx.lineWidth = Math.max(1, U.s); ctx.beginPath(); ctx.moveTo(pl2.X(xc2), r2.y); ctx.lineTo(pl2.X(xc2), r2.y + r2.h); ctx.stroke(); ctx.setLineDash([]);
      callout(ctx, U, r2.x + r2.w - 8 * U.s, r2.y + r2.h * 0.62, 'sweep: long to short wavelength', { size: 11, align: 'r', col: '#ffd0cc' });
      return;
    }
    // split screen
    var a1 = { x: W * 0.08, y: H * 0.14, w: W * 0.36, h: H * 0.5 }, a2 = { x: W * 0.58, y: H * 0.14, w: W * 0.36, h: H * 0.5 };
    var p1 = qPlot(ctx, U, a1, 'short to long'), p2 = qPlot(ctx, U, a2, 'long to short');
    fe.trace(ctx, p1, QX, QF, P.light, lw); fe.trace(ctx, p2, QX, QB, P.lightTM, lw);
    stamp(ctx, U, 'NOT OKAY', false, (u - 0.3) / 0.4, W * 0.5, H * 0.5);
  };

  /* ----- Cold open: the hotplate (z-cut wafer, pyroelectric charge; heating leaves the +z face negative) ----- */
  function hotplateScene(ctx, U, o) {
    var W = U.W, H = U.H;
    fe.plainBg(ctx, U, '#4a4c51', '#18191c');
    // hotplate (left) and cool metal block (right)
    var hx = W * 0.08, hy = H * 0.42, hw = W * 0.46, hh = H * 0.4;
    ctx.fillStyle = '#d9dcdf'; ctx.fillRect(hx, hy + H * 0.03, hw, hh);
    var heat = o.plateT != null ? clamp((o.plateT - 25) / 90, 0, 1) : 0;
    ctx.fillStyle = 'rgb(' + Math.round(150 + 80 * heat) + ',' + Math.round(150 - 40 * heat) + ',' + Math.round(150 - 70 * heat) + ')'; ctx.fillRect(hx - 4 * U.s, hy, hw + 8 * U.s, H * 0.03);
    ctx.fillStyle = '#121416'; ctx.fillRect(hx + hw * 0.08, hy + H * 0.12, hw * 0.4, H * 0.12);
    font(ctx, U.f(20), '700', true); ctx.fillStyle = '#ff5b3a'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText((o.plateT != null ? o.plateT : 25).toFixed(1) + ' °C', hx + hw * 0.28, hy + H * 0.18);
    ctx.fillStyle = '#80868d'; ctx.fillRect(W * 0.57, hy + H * 0.06, W * 0.4, H * 0.36);
    ctx.fillStyle = '#9aa0a7'; ctx.fillRect(W * 0.57, hy + H * 0.06, W * 0.4, H * 0.015);
    // wafer cross-section
    var wx = lerp(hx + hw * 0.08, W * 0.578, o.move || 0), wy = lerp(hy, hy + H * 0.06, o.move || 0) - (o.lift || 0) * H * 0.1 - H * 0.03;
    var ww = hw * 0.84, wt = H * 0.035;
    ctx.fillStyle = o.foil ? '#b9bec4' : 'rgba(225,212,170,0.95)'; ctx.fillRect(wx, wy, ww, wt);
    if (o.foil) { ctx.strokeStyle = '#7d838a'; ctx.lineWidth = 1.5 * U.s; ctx.beginPath(); for (var i = 0; i <= 24; i++) { var x = wx - 4 * U.s + i * (ww + 8 * U.s) / 24; ctx.lineTo(x, wy - 4 * U.s + (i % 2) * 3 * U.s); } ctx.stroke(); ctx.strokeRect(wx - 4 * U.s, wy - 4 * U.s, ww + 8 * U.s, wt + 8 * U.s); }
    ctx.strokeStyle = 'rgba(80,60,20,0.9)'; ctx.lineWidth = Math.max(1, U.s); ctx.strokeRect(wx, wy, ww, wt);
    // surface charge: sign on the +z face (top) is q (-1 heating, +1 cooling); n symbols per face
    var n = Math.round(o.n || 0), q = o.q || -1;
    font(ctx, U.f(16), '800'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (var j = 0; j < n; j++) {
      var xx = wx + (j + 0.5) * ww / 16;
      ctx.fillStyle = q < 0 ? '#6fa8ff' : '#ff6b6b'; ctx.fillText(q < 0 ? '−' : '+', xx, wy - 12 * U.s);
      if ((o.lift || 0) > 0.2) { ctx.fillStyle = q < 0 ? '#ff6b6b' : '#6fa8ff'; ctx.fillText(q < 0 ? '+' : '−', xx, wy + wt + 12 * U.s); }
    }
    callout(ctx, U, wx + ww - 10 * U.s, wy - 44 * U.s, '+z face', { size: 10, to: [wx + ww - 6 * U.s, wy + 2 * U.s], alpha: o.labelA == null ? 1 : o.labelA });
    return { wx: wx, wy: wy, ww: ww, wt: wt };
  }
  function spark(ctx, U, x0, y0, x1, y1, seed, a) {
    ctx.save(); ctx.strokeStyle = 'rgba(190,220,255,' + a + ')'; ctx.lineWidth = Math.max(1.5, 2.2 * U.s); ctx.shadowColor = '#9fd0ff'; ctx.shadowBlur = 12 * U.s;
    ctx.beginPath(); ctx.moveTo(x0, y0);
    for (var i = 1; i < 7; i++) { var t = i / 7, r = Math.sin(seed * 12.9898 + i * 78.233) * 43758.5453; r = r - Math.floor(r); ctx.lineTo(lerp(x0, x1, t) + (r - 0.5) * 18 * U.s, lerp(y0, y1, t) + (r - 0.5) * 10 * U.s); }
    ctx.lineTo(x1, y1); ctx.stroke(); ctx.restore();
  }
  S.hotplate = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H), w;
    if (k === 0) { var T = 25 + 85 * eseg(u, 0.05, 0.95); w = hotplateScene(ctx, U, { plateT: T, n: Math.round(5 * seg(u, 0.3, 1)), q: -1, labelA: seg(u, 0.5, 0.6) }); }
    else if (k === 1) { w = hotplateScene(ctx, U, { plateT: 110, n: 5 + Math.round(10 * eseg(u, 0, 0.8)), q: -1 }); }
    else if (k === 2) {
      var mv = eseg(u, 0.05, 0.45);
      var T2 = lerp(110, 45, eseg(u, 0.3, 1));
      w = hotplateScene(ctx, U, { plateT: 110, move: mv, lift: Math.sin(PI * mv), n: Math.round(14 * eseg(u, 0.35, 0.8)), q: 1 });
      fe.tweezers(ctx, w.wx - 2 * U.s, w.wy + w.wt / 2, 0, W * 0.3, w.wt + 4 * U.s, U.s);
      if (u > 0.55) { var fl = Math.floor(u * 40); if (fl % 3 !== 0) spark(ctx, U, w.wx + 20 * U.s, w.wy - 4 * U.s, w.wx - 16 * U.s, w.wy - 22 * U.s, fl, 0.95); if (fl % 4 === 1) spark(ctx, U, w.wx + w.ww * 0.6, w.wy - 2 * U.s, w.wx + w.ww * 0.75, w.wy - 30 * U.s, fl + 3, 0.8); }
      callout(ctx, U, W * 0.6, H * 0.3, 'wafer ' + Math.round(T2) + ' °C', { size: 12, mono: true });
    } else {
      var fixd = u > 0.45;
      if (!fixd) {
        w = hotplateScene(ctx, U, { plateT: 110, move: 1, n: 14, q: 1 });
        fe.tweezers(ctx, w.wx - 2 * U.s, w.wy + w.wt / 2, 0, W * 0.3, w.wt + 4 * U.s, U.s);
        spark(ctx, U, w.wx + 20 * U.s, w.wy - 4 * U.s, w.wx - 16 * U.s, w.wy - 22 * U.s, 7, 0.95);
        stamp(ctx, U, 'NOT OKAY', false, (u - 0.05) / 0.3);
      } else {
        var T3 = lerp(110, 30, eseg(u, 0.5, 0.95));
        w = hotplateScene(ctx, U, { plateT: T3, foil: true, n: 0 });
        callout(ctx, U, W * 0.66, H * 0.3, 'ramp 2 °C/min', { size: 12, mono: true });
        stamp(ctx, U, 'OKAY', true, (u - 0.8) / 0.15, W * 0.8, H * 0.14);
      }
    }
    fe.vignette(ctx, U, 0.3);
  };

  /* ----- Cold open: quadrature on Monday (DC drift: the transfer curve moves under constant voltage) ----- */
  // bias history: returns { V, Vd } in units of V pi at keyframe k, phase u
  function biasState(k, u) {
    var tau = 40, V = 0.5, Vd = 0, t = 0;
    function run(dt) { Vd = 0.9 * V + (Vd - 0.9 * V) * Math.exp(-dt / tau); }
    if (k === 0) return { V: 0.5, Vd: 0, t: 0 };
    if (k >= 1) { var dt = 60 * (k === 1 ? eseg(u, 0.05, 0.95) : 1); run(dt); t += dt; }
    if (k >= 2) {
      var pushes = [0.08, 0.42, 0.68], uu = k === 2 ? u : 1;
      var last = 0;
      for (var i = 0; i < pushes.length; i++) {
        if (uu < pushes[i]) break;
        run((pushes[i] - last) * 80); t += (pushes[i] - last) * 80; last = pushes[i];
        V = Vd + 0.5;          // Gary turns the knob back to quadrature
      }
      run((uu - last) * 80); t += (uu - last) * 80;
    }
    return { V: V, Vd: Vd, t: t };
  }
  S.quadrature = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    ctx.fillStyle = '#101317'; ctx.fillRect(0, 0, W, H);
    var b = biasState(k, k === 3 ? 1 : u);
    var Tf = function (V, Vd) { var s = Math.sin(PI * (V - Vd) / 2); return s * s; };
    // instrument panel
    var px = W * 0.04, py = H * 0.12, pw = W * 0.26;
    function readout(y, lab, val, col) {
      ctx.fillStyle = '#05070a'; ctx.fillRect(px, y, pw, H * 0.13); ctx.strokeStyle = 'rgba(160,175,190,0.35)'; ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, y + 0.5, pw - 1, H * 0.13 - 1);
      font(ctx, U.f(10), '600'); ctx.fillStyle = P.inkDim; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(lab, px + 8 * U.s, y + 6 * U.s);
      font(ctx, U.f(20), '700', true); ctx.fillStyle = col; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(val, px + pw - 8 * U.s, y + H * 0.13 - 6 * U.s);
    }
    var T = Tf(b.V, b.Vd);
    var mins = Math.round(b.t), hh = 12 + Math.floor(mins / 60), mm = mins % 60;
    readout(py, 'DC SOURCE', '+' + b.V.toFixed(2) + ' Vπ', '#7cf0a0');
    readout(py + H * 0.17, 'PHOTODETECTOR', Math.round(T * 100) + ' %', Math.abs(T - 0.5) < 0.06 ? '#7cf0a0' : '#ff8a80');
    readout(py + H * 0.34, 'WALL CLOCK', hh + ':' + (mm < 10 ? '0' : '') + mm, '#e6ebf0');
    // transfer curve
    var r = { x: W * 0.4, y: H * 0.12, w: W * 0.55, h: H * 0.56 };
    var pl = fe.plot(ctx, U, r, { x0: 0, x1: 3, y0: 0, y1: 1.05, xt: [0, 0.5, 1, 1.5, 2, 2.5, 3], yt: [0, 0.5, 1], xl: 'DC voltage (Vπ)', yl: 'transmission', padL: 34 });
    var xs = range(0, 3, 300);
    if (k > 0) fe.trace(ctx, pl, xs, function (V) { return Tf(V, 0); }, 'rgba(200,205,212,0.35)', Math.max(1, 1.4 * U.s));
    fe.trace(ctx, pl, xs, function (V) { return Tf(V, b.Vd); }, '#e6ebf0', Math.max(1.5, 2.2 * U.s));
    ctx.fillStyle = Math.abs(T - 0.5) < 0.06 ? P.ok : P.bad; ctx.beginPath(); ctx.arc(pl.X(b.V), pl.Y(T), 8 * U.s, 0, TAU); ctx.fill();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5 * U.s; ctx.stroke();
    ctx.strokeStyle = 'rgba(227,176,75,0.6)'; ctx.setLineDash([4 * U.s, 4 * U.s]); ctx.beginPath(); ctx.moveTo(r.x, pl.Y(0.5)); ctx.lineTo(r.x + r.w, pl.Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
    callout(ctx, U, r.x + r.w - 4 * U.s, pl.Y(0.5) - 6 * U.s, 'quadrature', { size: 10, align: 'r', valign: 'b' });
    if (k === 3) stamp(ctx, U, 'NOT OKAY', false, (u - 0.3) / 0.4);
  };

  /* ----- Cold open: Q = 7e13 (quarter model with two symmetry planes vs full model) ----- */
  function qMeter(ctx, U, r, logQ) {
    ctx.fillStyle = '#05070a'; ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = 'rgba(160,175,190,0.4)'; ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    var bx = r.x + r.w * 0.05, bw = r.w * 0.9, by = r.y + r.h * 0.62;
    font(ctx, U.f(10), '', true); ctx.fillStyle = P.inkDim; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var e = 2; e <= 14; e += 2) { var x = bx + (e - 2) / 12 * bw; ctx.fillRect(x - 0.5, by - 6 * U.s, 1, 12 * U.s); ctx.fillText('10' + sup(e), x, by + 9 * U.s); }
    ctx.fillStyle = 'rgba(200,210,220,0.3)'; ctx.fillRect(bx, by - 1, bw, 2);
    var mx = bx + clamp((logQ - 2) / 12, 0, 1) * bw;
    ctx.fillStyle = logQ > 8 ? P.bad : P.ok; ctx.beginPath(); ctx.moveTo(mx, by - 4 * U.s); ctx.lineTo(mx - 8 * U.s, by - 18 * U.s); ctx.lineTo(mx + 8 * U.s, by - 18 * U.s); ctx.closePath(); ctx.fill();
    var m = Math.pow(10, logQ - Math.floor(logQ));
    font(ctx, U.f(20), '700', true); ctx.fillStyle = '#e6ebf0'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('Q = ' + m.toFixed(1) + ' × 10' + sup(Math.floor(logQ)), r.x + 10 * U.s, r.y + 8 * U.s);
  }
  function sup(n) { var m = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' }; return String(n).split('').map(function (c) { return m[c] || c; }).join(''); }
  fe.sup = sup;
  function jet(t) { t = clamp(t, 0, 1); var r = clamp(1.5 - Math.abs(4 * t - 3), 0, 1), g = clamp(1.5 - Math.abs(4 * t - 2), 0, 1), b = clamp(1.5 - Math.abs(4 * t - 1), 0, 1); return 'rgb(' + Math.round(255 * r) + ',' + Math.round(255 * g) + ',' + Math.round(255 * b) + ')'; }
  function nanobeam(ctx, U, r, quarter, leak, ph) {
    // beam along x with elliptical holes, clamped at both ends to the slab; displacement colour map
    var cx = r.x + r.w / 2, cy = r.y + r.h / 2, L = r.w * 0.86, bw = r.h * 0.16;
    ctx.fillStyle = '#2a323b'; ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.fillStyle = '#3c4652'; ctx.fillRect(r.x, r.y, r.w * 0.07, r.h); ctx.fillRect(r.x + r.w * 0.93, r.y, r.w * 0.07, r.h);   // anchors / slab
    var cell = L / 31;
    for (var i = 0; i < 31; i++) {
      var x0 = cx - L / 2 + i * cell, xm = x0 + cell / 2, d = (xm - cx) / (L * 0.16);
      var amp = Math.exp(-d * d) * Math.abs(Math.cos(ph)), lk = leak * 0.35 * Math.abs(Math.sin((xm - cx) / cell * 1.3 + ph));
      ctx.fillStyle = jet(clamp(amp + lk, 0, 1) * 0.9 + 0.05); ctx.fillRect(x0, cy - bw / 2, cell + 0.5, bw);
      ctx.fillStyle = '#2a323b'; ctx.beginPath(); ctx.ellipse(xm, cy, cell * 0.28, bw * 0.3 * (1 - 0.3 * Math.exp(-d * d)), 0, 0, TAU); ctx.fill();
    }
    if (leak > 0) {   // energy radiating into the anchors
      ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip(); ctx.globalAlpha = leak;
      [r.x + r.w * 0.07, r.x + r.w * 0.93].forEach(function (ax, s) { for (var j = 1; j < 5; j++) { ctx.strokeStyle = jet(0.75 - j * 0.1); ctx.lineWidth = 2 * U.s; ctx.beginPath(); ctx.arc(ax, cy, j * r.h * 0.11 + (ph * 6 % 1) * r.h * 0.11, s ? -PI / 2 : PI / 2, s ? PI / 2 : PI * 1.5); ctx.stroke(); } });
      ctx.restore();
    }
    // mesh lines
    ctx.save(); ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1;
    for (var m = 0; m <= 62; m++) { var x = cx - L / 2 + m * L / 62; ctx.beginPath(); ctx.moveTo(x, cy - bw / 2); ctx.lineTo(x + L / 124, cy + bw / 2); ctx.stroke(); }
    ctx.restore();
    if (quarter < 1) {   // hidden three quarters
      var q = quarter;   // 0: quarter model, rising to 1: full
      ctx.save(); ctx.fillStyle = 'rgba(70,72,78,' + (0.96 * (1 - q)) + ')';
      ctx.fillRect(r.x, r.y, r.w / 2, r.h);
      ctx.fillRect(r.x + r.w / 2, r.y, r.w / 2, r.h / 2);
      ctx.restore();
      ctx.save(); ctx.strokeStyle = '#ffd36b'; ctx.lineWidth = 2 * U.s; ctx.setLineDash([10 * U.s, 4 * U.s, 2 * U.s, 4 * U.s]);
      ctx.globalAlpha = 1 - q;
      ctx.beginPath(); ctx.moveTo(cx, r.y); ctx.lineTo(cx, r.y + r.h); ctx.moveTo(r.x, cy); ctx.lineTo(r.x + r.w, cy); ctx.stroke(); ctx.restore();
    }
  }
  S.qnano = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    ctx.fillStyle = '#101317'; ctx.fillRect(0, 0, W, H);
    var r = { x: W * 0.05, y: H * 0.08, w: W * 0.9, h: H * 0.42 };
    var quarter = k === 2 ? eseg(u, 0.05, 0.5) : k === 3 ? 0.5 : 0;
    var leak = k === 2 ? eseg(u, 0.3, 0.7) : k === 3 ? 1 : 0;
    nanobeam(ctx, U, r, quarter, leak, u * 12);
    if (k === 0) callout(ctx, U, r.x + r.w * 0.52, r.y + r.h * 0.04, 'symmetry planes', { size: 11, to: [r.x + r.w / 2, r.y + r.h * 0.25] });
    var logQ = k === 0 ? 4 + 9.85 * eseg(u, 0.4, 0.95) * 0 + 3.34 : k === 1 ? lerp(3.34, Math.log(7e13) / Math.LN10, eseg(u, 0.05, 0.35)) : k === 2 ? lerp(Math.log(7e13) / Math.LN10, Math.log(2.2e3) / Math.LN10, eseg(u, 0.35, 0.8)) : Math.log(2.2e3) / Math.LN10;
    if (k === 0) logQ = lerp(2, 3.34, eseg(u, 0.5, 0.95));
    qMeter(ctx, U, { x: W * 0.2, y: H * 0.56, w: W * 0.6, h: H * 0.2 }, logQ);
    callout(ctx, U, W * 0.05, H * 0.015, k >= 2 ? 'full geometry' : 'quarter model', { size: 12, bold: true });
    if (k === 3) stamp(ctx, U, 'NOT OKAY', false, (u - 0.3) / 0.4, W * 0.8, H * 0.6);
  };

  /* ----- Incident: the rods at twelve o'clock (PANDA fibre; slow axis along the line through the rods) ----- */
  function endFace(ctx, U, cx, cy, r, rot, o) {
    o = o || {};
    ctx.fillStyle = '#05070a'; ctx.beginPath(); ctx.arc(cx, cy, r * 1.08, 0, TAU); ctx.fill();
    var g = ctx.createRadialGradient(cx, cy, r * 0.1, cx, cy, r); g.addColorStop(0, '#c9d2dc'); g.addColorStop(1, '#8d99a6');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.fill();
    var dx = Math.cos(rot), dy = Math.sin(rot);
    ctx.fillStyle = '#4e5864';
    [-1, 1].forEach(function (s) { ctx.beginPath(); ctx.arc(cx + s * 0.46 * r * dx, cy + s * 0.46 * r * dy, 0.27 * r, 0, TAU); ctx.fill(); });
    ctx.fillStyle = o.lit ? '#ffffff' : '#eef3f8'; ctx.beginPath(); ctx.arc(cx, cy, 0.07 * r, 0, TAU); ctx.fill();
    if (o.axis) { ctx.save(); ctx.setLineDash([5 * U.s, 4 * U.s]); ctx.strokeStyle = '#ffd36b'; ctx.lineWidth = 2 * U.s; ctx.beginPath(); ctx.moveTo(cx - r * 1.2 * dx, cy - r * 1.2 * dy); ctx.lineTo(cx + r * 1.2 * dx, cy + r * 1.2 * dy); ctx.stroke(); ctx.restore(); }
  }
  function xsec(ctx, U, r, te) {
    // x-cut rib: 600 nm film, 300 nm etch, 1.5 um top width, gold 4 um from the rib; vertical scale stretched x2
    var sc = r.w / 16, cx = r.x + r.w / 2, base = r.y + r.h * 0.62, vs = 2;
    ctx.fillStyle = '#2b2f35'; ctx.fillRect(r.x, base + 4.7 * sc * 0.35, r.w, r.h);    // Si (cut)
    ctx.fillStyle = '#6e7f8f'; ctx.fillRect(r.x, base, r.w, 4.7 * sc * 0.35);           // SiO2
    ctx.fillStyle = '#a28fb0'; ctx.fillRect(r.x, base - 0.3 * sc * vs, r.w, 0.3 * sc * vs);
    ctx.beginPath(); ctx.moveTo(cx - 1.1 * sc, base - 0.3 * sc * vs); ctx.lineTo(cx - 0.75 * sc, base - 0.6 * sc * vs); ctx.lineTo(cx + 0.75 * sc, base - 0.6 * sc * vs); ctx.lineTo(cx + 1.1 * sc, base - 0.3 * sc * vs); ctx.closePath(); ctx.fill();
    ctx.fillStyle = P.au; ctx.fillRect(cx - 7.5 * sc, base - 0.3 * sc * vs - 0.8 * sc * vs, 3.0 * sc, 0.8 * sc * vs); ctx.fillRect(cx + 4.5 * sc, base - 0.3 * sc * vs - 0.8 * sc * vs, 3.0 * sc, 0.8 * sc * vs);
    // RF field across the gap (along z)
    ctx.save(); ctx.globalAlpha = 0.5; for (var i = -1; i <= 1; i++) arrow(ctx, cx - 4.2 * sc, base - 0.45 * sc * vs + i * 6 * U.s, cx + 4.2 * sc, base - 0.45 * sc * vs + i * 6 * U.s, '#ffb35c', Math.max(1, 1.2 * U.s), 6 * U.s); ctx.restore();
    // optical mode and its field direction
    var my = base - 0.42 * sc * vs;
    var gm = ctx.createRadialGradient(cx, my, 0, cx, my, 1.3 * sc); gm.addColorStop(0, 'rgba(255,255,255,0.75)'); gm.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.save(); ctx.scale(1, 0.55); ctx.fillStyle = gm; ctx.beginPath(); ctx.arc(cx, my / 0.55, 1.3 * sc, 0, TAU); ctx.restore();
    ctx.fillStyle = gm; ctx.beginPath(); ctx.ellipse(cx, my, 1.3 * sc, 0.7 * sc, 0, 0, TAU); ctx.fill();
    var L = 1.4 * sc;
    if (te) { arrow(ctx, cx - L, my, cx + L, my, '#4fd27f', 3 * U.s, 10 * U.s); arrow(ctx, cx + L, my, cx - L, my, '#4fd27f', 3 * U.s, 10 * U.s); }
    else { arrow(ctx, cx, my + L * 0.8, cx, my - L * 0.8, P.bad, 3 * U.s, 10 * U.s); arrow(ctx, cx, my - L * 0.8, cx, my + L * 0.8, P.bad, 3 * U.s, 10 * U.s); }
    return { cx: cx, my: my, sc: sc, base: base };
  }
  S.rods = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H);
    ctx.fillStyle = '#101317'; ctx.fillRect(0, 0, W, H);
    if (k === 0 || k === 1) {
      // fibre inspection scope view of the patch cord end face
      var ex = W * 0.36, ey = H * 0.45, er = H * 0.33;
      ctx.fillStyle = '#20262d'; ctx.beginPath(); ctx.arc(ex, ey, er * 1.25, 0, TAU); ctx.fill();
      var rot = k === 0 ? 0 : PI / 2 * eseg(u, 0.25, 0.75);
      endFace(ctx, U, ex, ey, er, rot, { axis: true });
      fe.eyepiece(ctx, U, ex, ey, er * 1.25);
      callout(ctx, U, ex + er * 1.3, ey - er * 0.9, 'slow axis', { size: 11, col: '#ffe6a6', border: '#ffd36b', to: [ex + er * 1.12 * Math.cos(rot), ey + er * 1.12 * Math.sin(rot)] });
      callout(ctx, U, W * 0.04, H * 0.05, k === 0 ? 'old patch cord' : 'new patch cord', { size: 13, bold: true });
      if (k === 1) {
        // sticky note on the monitor bezel
        var nx = W * 0.72, ny = H * 0.52, nw = W * 0.22, nh = H * 0.24;
        ctx.save(); ctx.translate(nx, ny); ctx.rotate(0.05);
        ctx.fillStyle = '#f3e27a'; ctx.fillRect(0, 0, nw, nh); ctx.fillStyle = 'rgba(0,0,0,0.08)'; ctx.fillRect(0, 0, nw, nh * 0.12);
        font(ctx, Math.max(U.f(12), nw * 0.085), '700'); ctx.fillStyle = '#2b2a6b'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('SLOW AXIS', nw / 2, nh * 0.38); ctx.fillText('⊥ RODS', nw / 2, nh * 0.62);
        ctx.restore();
      }
      return;
    }
    if (k === 2) {
      // computed RF field lines and optical mode (separate module), when loaded; this drawing is the fallback
      var LD = (typeof window !== 'undefined' ? window : this).LMS_DRAW;
      if (LD && typeof LD.fe_rodsFields === 'function') { LD.fe_rodsFields(ctx, W, H, u, {}); return; }
      var r = { x: W * 0.12, y: H * 0.06, w: W * 0.76, h: H * 0.7 };
      var xs = xsec(ctx, U, r, false);
      callout(ctx, U, xs.cx + 1.8 * xs.sc, xs.my - 2.6 * xs.sc, 'TM: field along x', { size: 11, col: '#ffd0cc', border: P.bad, to: [xs.cx + 6 * U.s, xs.my - 1.0 * xs.sc] });
      callout(ctx, U, xs.cx - 4.5 * xs.sc, xs.base - 2.4 * xs.sc, 'RF field along z', { size: 11, align: 'r', col: '#ffe0bf', border: '#ffb35c', to: [xs.cx - 3.5 * xs.sc, xs.base - 0.45 * xs.sc * 2] });
      // axes for the section: z to the right, x up, y out of the page
      var ax = W * 0.9, ay = H * 0.2, l = 30 * U.s;
      arrow(ctx, ax, ay, ax + l, ay, '#ffc46b', 2 * U.s, 8 * U.s); arrow(ctx, ax, ay, ax, ay - l, '#ffffff', 2 * U.s, 8 * U.s);
      ctx.strokeStyle = '#9fe3ff'; ctx.lineWidth = 1.5 * U.s; ctx.beginPath(); ctx.arc(ax, ay, 5 * U.s, 0, TAU); ctx.stroke(); ctx.fillStyle = '#9fe3ff'; ctx.beginPath(); ctx.arc(ax, ay, 1.6 * U.s, 0, TAU); ctx.fill();
      font(ctx, U.f(11), '700'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#ffc46b'; ctx.fillText('z', ax + l + 9 * U.s, ay); ctx.fillStyle = '#ffffff'; ctx.fillText('x', ax, ay - l - 9 * U.s); ctx.fillStyle = '#9fe3ff'; ctx.fillText('y', ax - 12 * U.s, ay + 10 * U.s);
      return;
    }
    var rr = { x: W * 0.1, y: H * 0.12, w: W * 0.8, h: H * 0.52 };
    var pl = fe.plot(ctx, U, rr, { x0: -4, x1: 4, y0: 0, y1: 1.05, xt: [-4, -2, 0, 2, 4], yt: [0, 0.5, 1], xl: 'drive voltage (reference Vπ)', yl: 'transmission', padL: 34 });
    var xs2 = range(-4, 4, 400);
    fe.trace(ctx, pl, xs2, function (x) { var c = Math.cos(PI * x / 2 + PI / 4); return c * c; }, 'rgba(200,205,212,0.6)', Math.max(1, 1.4 * U.s));
    var cut = -4 + 8 * eseg(u, 0.03, 0.5);
    fe.trace(ctx, pl, xs2, function (x) { var c = Math.cos(PI * x / 6 + PI / 4); return c * c; }, P.bad, Math.max(2, 2.6 * U.s), function (x) { return x <= cut; });
    stamp(ctx, U, 'NOT OKAY', false, (u - 0.55) / 0.35);
  };

  /* ----- Incident: the hot plug (bulk z-cut chip on a thermoelectric cooler, pads floating, then a hot plug) ----- */
  function fixture(ctx, U, o) {
    var W = U.W, H = U.H;
    fe.plainBg(ctx, U, '#45484d', '#151618');
    var x0 = W * 0.08, w = W * 0.5, yb = H * 0.86;
    // fan + heatsink
    ctx.fillStyle = '#1c1e21'; ctx.fillRect(x0, yb - H * 0.08, w, H * 0.08);
    ctx.fillStyle = '#8e959d'; for (var i = 0; i < 14; i++) ctx.fillRect(x0 + i * w / 14 + 2, yb - H * 0.2, w / 14 - 5, H * 0.12);
    ctx.fillStyle = '#a8aeb5'; ctx.fillRect(x0, yb - H * 0.23, w, H * 0.03);
    // TEC: ceramic plates and pellets
    var tx = x0 + w * 0.12, tw = w * 0.76, ty = yb - H * 0.31;
    ctx.fillStyle = '#ece8df'; ctx.fillRect(tx, ty + H * 0.065, tw, H * 0.015); ctx.fillRect(tx, ty, tw, H * 0.015);
    ctx.fillStyle = '#5a6068'; for (var j = 0; j < 12; j++) ctx.fillRect(tx + 6 + j * (tw - 12) / 12, ty + H * 0.015, (tw - 12) / 24, H * 0.05);
    var hot = o.hot || 0;
    if (hot > 0) { ctx.fillStyle = 'rgba(255,90,40,' + (0.35 * hot) + ')'; ctx.fillRect(tx, ty, tw, H * 0.015); }
    // gold-plated fixture block
    var fx = x0 + w * 0.06, fw = w * 0.88, fy = ty - H * 0.09;
    var g = ctx.createLinearGradient(0, fy, 0, fy + H * 0.09); g.addColorStop(0, '#f1d27a'); g.addColorStop(1, '#b88d36');
    ctx.fillStyle = g; ctx.fillRect(fx, fy, fw, H * 0.09);
    // chip: bulk z-cut LN, 0.5 mm thick (drawn thicker), electrodes on top
    var cx = fx + fw * 0.08, cw = fw * 0.7, ct = H * 0.035, cy = fy - ct;
    ctx.fillStyle = 'rgba(210,205,190,0.95)'; ctx.fillRect(cx, cy, cw, ct);
    ctx.fillStyle = P.au; ctx.fillRect(cx + cw * 0.05, cy - 2 * U.s, cw * 0.9, 2.5 * U.s);
    // RF connector (end launch) on the right of the fixture, wire bonds to the chip pads
    var kx = fx + fw - W * 0.005, ky = fy + H * 0.01;
    ctx.fillStyle = '#c9ced4'; ctx.fillRect(kx, ky, W * 0.05, H * 0.07);
    ctx.fillStyle = '#e8c86a'; ctx.fillRect(kx + W * 0.05, ky + H * 0.022, W * 0.025, H * 0.026);
    ctx.strokeStyle = '#f5dc8c'; ctx.lineWidth = Math.max(1, 1.2 * U.s);
    ctx.beginPath(); ctx.moveTo(cx + cw * 0.95, cy - 1); ctx.quadraticCurveTo(cx + cw + W * 0.03, cy - H * 0.05, kx + 1, ky + H * 0.035); ctx.stroke();
    // charge on the top (+z) face
    var n = Math.round(o.n || 0);
    font(ctx, U.f(14), '800'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#6fa8ff';
    for (var q = 0; q < n; q++) ctx.fillText('−', cx + (q + 0.5) * cw / 12, cy - 13 * U.s);
    return { pin: [kx + W * 0.075, ky + H * 0.035], chip: [cx, cy, cw, ct] };
  }
  function cable(ctx, U, x, y, a) {   // cable plug: nut at (x, y), cable going to the amplifier box at the upper right
    var W = U.W, H = U.H;
    ctx.save(); ctx.globalAlpha = a == null ? 1 : a;
    ctx.strokeStyle = '#2a2d31'; ctx.lineWidth = 9 * U.s; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x + W * 0.04, y); ctx.bezierCurveTo(x + W * 0.15, y, W * 0.78, H * 0.5, W * 0.8, H * 0.36); ctx.stroke();
    ctx.fillStyle = '#c3c9cf'; ctx.fillRect(x, y - H * 0.03, W * 0.04, H * 0.06);
    ctx.restore();
  }
  function amp(ctx, U, dead, u) {
    var W = U.W, H = U.H, ax = W * 0.7, ay = H * 0.14, aw = W * 0.22, ah = H * 0.22;
    ctx.fillStyle = '#3a3e44'; ctx.fillRect(ax, ay, aw, ah); ctx.strokeStyle = '#5f656c'; ctx.lineWidth = 1.5 * U.s; ctx.strokeRect(ax, ay, aw, ah);
    ctx.fillStyle = '#23262a'; for (var i = 0; i < 6; i++) ctx.fillRect(ax + aw * 0.1 + i * aw * 0.13, ay + ah * 0.12, aw * 0.07, ah * 0.3);
    ctx.fillStyle = dead ? '#2a2a2a' : '#39d353'; ctx.beginPath(); ctx.arc(ax + aw * 0.85, ay + ah * 0.75, 5 * U.s, 0, TAU); ctx.fill();
    font(ctx, U.f(10), '600'); ctx.fillStyle = P.inkDim; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('driver amplifier', ax, ay + ah + 6 * U.s);
    if (dead) { ctx.save(); for (var k = 0; k < 3; k++) { var t = (u * 1.5 + k * 0.33) % 1; ctx.strokeStyle = 'rgba(200,200,200,' + (0.5 * (1 - t)) + ')'; ctx.lineWidth = 4 * U.s; ctx.beginPath(); var sx = ax + aw * (0.3 + 0.15 * k), sy = ay - t * H * 0.12; ctx.moveTo(sx, sy); ctx.bezierCurveTo(sx + 10 * U.s, sy - 10 * U.s, sx - 10 * U.s, sy - 20 * U.s, sx + 4 * U.s, sy - 30 * U.s); ctx.stroke(); } ctx.restore(); }
  }
  function burnInset(ctx, U, r) {
    ctx.fillStyle = '#5d6066'; ctx.fillRect(r.x, r.y, r.w, r.h);    // bulk LN in bright field
    var sc = r.h / 120;     // px per um
    var sy = r.y + r.h * 0.5;
    var g = ctx.createLinearGradient(0, r.y, 0, r.y + r.h); g.addColorStop(0, P.auHi); g.addColorStop(1, P.au);
    ctx.fillStyle = g;
    ctx.fillRect(r.x, sy - 4 * sc, r.w, 8 * sc);                      // signal, 8 um over the waveguide
    ctx.fillRect(r.x, r.y, r.w, sy - 4 * sc - 25 * sc - r.y);          // ground, 25 um gap
    ctx.fillRect(r.x, sy + 4 * sc + 25 * sc, r.w, r.y + r.h - (sy + 4 * sc + 25 * sc));
    // burn track across the gap between signal and the upper ground
    ctx.save(); ctx.strokeStyle = 'rgba(25,18,14,0.9)'; ctx.lineWidth = 3 * sc; ctx.lineCap = 'round';
    var bx = r.x + r.w * 0.52;
    ctx.beginPath(); ctx.moveTo(bx, sy - 3 * sc); ctx.lineTo(bx - 3 * sc, sy - 12 * sc); ctx.lineTo(bx + 4 * sc, sy - 20 * sc); ctx.lineTo(bx - 1 * sc, sy - 31 * sc); ctx.stroke();
    ctx.fillStyle = 'rgba(40,28,20,0.85)'; ctx.beginPath(); ctx.ellipse(bx, sy - 3 * sc, 6 * sc, 3 * sc, 0, 0, TAU); ctx.fill(); ctx.beginPath(); ctx.ellipse(bx - 1 * sc, sy - 31 * sc, 7 * sc, 3.5 * sc, 0, 0, TAU); ctx.fill();
    ctx.restore();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    fe.scaleBar(ctx, U, r.x + 10 * U.s, r.y + r.h - 10 * U.s, sc, 60 * U.s);
  }
  function controller(ctx, U, x, y, set, act, ramp) {
    var w = U.W * 0.26, h = U.H * 0.2;
    ctx.fillStyle = '#05070a'; ctx.fillRect(x, y, w, h); ctx.strokeStyle = 'rgba(160,175,190,0.4)'; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    font(ctx, U.f(10), '600'); ctx.fillStyle = P.inkDim; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('SET', x + 8 * U.s, y + 8 * U.s); ctx.fillText('ACTUAL', x + 8 * U.s, y + h * 0.38); ctx.fillText('RAMP', x + 8 * U.s, y + h * 0.7);
    font(ctx, U.f(16), '700', true); ctx.textAlign = 'right'; ctx.fillStyle = '#ffb35c';
    ctx.fillText(set.toFixed(1) + ' °C', x + w - 8 * U.s, y + 6 * U.s); ctx.fillStyle = '#7cf0a0'; ctx.fillText(act.toFixed(1) + ' °C', x + w - 8 * U.s, y + h * 0.36); ctx.fillStyle = ramp === 'MAX' ? '#ff8a80' : '#e6ebf0'; ctx.fillText(ramp, x + w - 8 * U.s, y + h * 0.68);
  }
  S.hotplug = function (ctx, W, H, u, k) {
    var U = units(ctx, W, H), f;
    if (k === 0) {
      f = fixture(ctx, U, { n: 0 });
      cable(ctx, U, f.pin[0] + W * 0.12 * eseg(u, 0.15, 0.7), f.pin[1]);
      amp(ctx, U, false, u);
      callout(ctx, U, f.chip[0], f.chip[1] - 44 * U.s, 'pads floating', { size: 11, alpha: seg(u, 0.7, 0.8), to: [f.chip[0] + f.chip[2] * 0.5, f.chip[1] - 3 * U.s] });
    } else if (k === 1) {
      var T = lerp(25, 70, eseg(u, 0.1, 0.45));
      f = fixture(ctx, U, { n: Math.round(12 * eseg(u, 0.15, 0.7)), hot: seg(u, 0.1, 0.45) });
      cable(ctx, U, f.pin[0] + W * 0.12, f.pin[1]);
      amp(ctx, U, false, u);
      controller(ctx, U, W * 0.04, H * 0.06, 70, T, 'MAX');
      callout(ctx, U, f.chip[0] + f.chip[2] * 0.6, f.chip[1] - 60 * U.s, '+z face', { size: 10, to: [f.chip[0] + f.chip[2] * 0.75, f.chip[1] + 2 * U.s] });
    } else if (k === 2) {
      f = fixture(ctx, U, { n: u > 0.62 ? 0 : 12, hot: 1 });
      var plug = eseg(u, 0.05, 0.6);
      cable(ctx, U, f.pin[0] + W * 0.12 * (1 - plug) + 2 * U.s, f.pin[1]);
      amp(ctx, U, false, u);
      if (u > 0.56 && u < 0.66) {
        spark(ctx, U, f.pin[0] - 4 * U.s, f.pin[1], f.pin[0] + 10 * U.s, f.pin[1] - 2 * U.s, 3, 1);
        ctx.fillStyle = 'rgba(220,235,255,' + (0.5 * (1 - seg(u, 0.58, 0.66))) + ')'; ctx.fillRect(0, 0, W, H);
      }
    } else {
      var fixd = k === 4 && u > 0.45;
      f = fixture(ctx, U, { n: 0, hot: fixd ? lerp(1, 0, eseg(u, 0.5, 0.95)) : 1 });
      if (!fixd) {
        cable(ctx, U, f.pin[0] + 2 * U.s, f.pin[1]);
        amp(ctx, U, true, u);
        burnInset(ctx, U, { x: W * 0.04, y: H * 0.05, w: W * 0.34, h: H * 0.36 });
        if (k === 4) stamp(ctx, U, 'NOT OKAY', false, (u - 0.05) / 0.3);
      } else {
        // corrected take: shorting cap on the connector during the thermal step, slow ramp
        ctx.fillStyle = '#c0392b'; ctx.fillRect(f.pin[0] - 2 * U.s, f.pin[1] - H * 0.025, W * 0.03, H * 0.05);
        callout(ctx, U, f.pin[0] + W * 0.05, f.pin[1] + H * 0.08, 'shorting cap', { size: 11, to: [f.pin[0] + W * 0.015, f.pin[1] + H * 0.025] });
        amp(ctx, U, false, u);
        controller(ctx, U, W * 0.04, H * 0.06, 25, lerp(70, 25, eseg(u, 0.5, 0.95)), '1.0 °C/min');
        stamp(ctx, U, 'OKAY', true, (u - 0.85) / 0.12);
      }
    }
    fe.vignette(ctx, U, 0.3);
  };
})();
