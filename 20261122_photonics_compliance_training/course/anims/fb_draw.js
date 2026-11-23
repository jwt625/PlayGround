/* Feedback course: code-drawn video keyframes, window.LMS_DRAW.fb.
   Every scene is fn(ctx, W, H, u, o): u is 0..1 inside the keyframe, o holds scene options.
   Scenes draw in a virtual 1600 x 900 frame (16:9, letterboxed); the bottom band (y > 760) is left for the caption.
   Device views follow real topology and proportions (references: research/ref_micrographs/README_fb.md):
   flip-chip dies are seen from the backside, laser output facets face the chip input, isolator arrows point along the
   forward path, ring heaters stay off the coupling region, instrument traces use the physics stated in the narration.
   Classic script, no DOM access at load time. */
(function () {
  'use strict';
  var root = typeof window !== 'undefined' ? window : globalThis;
  root.LMS_DRAW = root.LMS_DRAW || {};
  var FB = {};
  var VW = 1600, VH = 900;
  var P = {
    bg: '#0b0f14', ink: '#e6ebf0', ink2: '#a3aeba', ink3: '#66727e', line: '#2a333d',
    fwd: '#56d4ff', back: '#ff5a4f', warn: '#f2b84b', ok: '#3fb950', bad: '#e5534b',
    oxide: '#6e6a94', oxideHi: '#7c78a2', oxideLo: '#5b5780', wg: '#c2c9f2', wgEdge: '#45416a',
    trench: '#2a2735', trenchRim: '#8e8ab4', gold: '#d6af58', goldHi: '#f1d78e', goldLo: '#97762f',
    die: '#a58c55', dieLo: '#7d6a40', si: '#4f5664', box: '#8a93b2', inp: '#7f838c', solder: '#cdb98a', mqw: '#ff9a52',
    scr: '#06090c', grid: '#18222b', axis: '#53616e', tr1: '#f2c94c', tr2: '#56d4ff', tr3: '#ff7a6e',
    bench: '#23272c', hole: '#14171a', jacket: '#e8c547', steel: '#aeb5bd', steelLo: '#6c737b', pcb: '#1f5a3a'
  };

  /* ---------- frame, text, boxes ---------- */
  var S = { k: 1, mf: 14, narrow: false, ctx: null };
  function begin(ctx, W, H, bg) {
    var k = Math.min(W / VW, H / VH);
    ctx.fillStyle = bg || P.bg; ctx.fillRect(0, 0, W, H);
    ctx.translate((W - VW * k) / 2, (H - VH * k) / 2); ctx.scale(k, k);
    var css = (ctx.canvas && ctx.canvas.clientWidth) || W, dpr = W / Math.max(1, css);
    S.k = k; S.mf = 6.5 * dpr / k; S.narrow = css < 560; S.ctx = ctx;
    ctx.lineJoin = 'round'; ctx.lineCap = 'butt'; ctx.textBaseline = 'middle';
    return S;
  }
  var SANS = '"Helvetica Neue", Arial, system-ui, sans-serif', MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
  function font(px, wt, mono) { var s = Math.max(px, S.mf); S.ctx.font = (wt || 400) + ' ' + s.toFixed(1) + 'px ' + (mono ? MONO : SANS); return s; }
  function txt(s, x, y, px, col, al, wt, mono) {
    var c = S.ctx; font(px, wt, mono); c.fillStyle = col || P.ink; c.textAlign = al || 'left'; c.textBaseline = 'middle'; c.fillText(s, x, y);
    return c.measureText(s).width;
  }
  function tw(s, px, wt, mono) { font(px, wt, mono); return S.ctx.measureText(s).width; }
  function clamp(v, a, b) { a = a === undefined ? 0 : a; b = b === undefined ? 1 : b; return v < a ? a : v > b ? b : v; }
  function ease(t) { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function seg(u, a, b) { return ease((u - a) / (b - a)); }
  function lin(u, a, b) { return clamp((u - a) / (b - a)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function rng(seed) { var s = seed % 2147483647; if (s <= 0) s += 2147483646; return function () { s = s * 16807 % 2147483647; return (s - 1) / 2147483646; }; }
  function rect(x, y, w, h, fill, stroke, lw) { var c = S.ctx; if (fill) { c.fillStyle = fill; c.fillRect(x, y, w, h); } if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw || 2; c.strokeRect(x, y, w, h); } }
  function line(pts, col, w, dash) {
    var c = S.ctx; c.strokeStyle = col; c.lineWidth = w || 2; if (dash) c.setLineDash(dash);
    c.beginPath(); pts.forEach(function (p, i) { if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }); c.stroke(); if (dash) c.setLineDash([]);
  }
  function dot(x, y, r, col) { var c = S.ctx; c.fillStyle = col; c.beginPath(); c.arc(x, y, r, 0, 6.2832); c.fill(); }
  function arrowHead(x, y, ang, sz, col) {
    var c = S.ctx; c.fillStyle = col; c.beginPath(); c.moveTo(x, y);
    c.lineTo(x - sz * Math.cos(ang - 0.45), y - sz * Math.sin(ang - 0.45)); c.lineTo(x - sz * Math.cos(ang + 0.45), y - sz * Math.sin(ang + 0.45)); c.closePath(); c.fill();
  }
  function arrow(x0, y0, x1, y1, col, w, sz) { line([[x0, y0], [x1, y1]], col, w || 2); arrowHead(x1, y1, Math.atan2(y1 - y0, x1 - x0), sz || 12, col); }
  /* plain box callout: lines[0] in ink, the rest in ink2; o.anchor [x, y] draws a leader line */
  function box(x, y, lines, o) {
    o = o || {}; lines = [].concat(lines); var c = S.ctx, px = o.px || 21, sz = font(px, o.wt || 500), pad = sz * 0.5, lh = sz * 1.28;
    var w = 0; lines.forEach(function (l, i) { font(px, i ? 400 : (o.wt || 500), o.mono); w = Math.max(w, c.measureText(l).width); });
    var bw = w + 2 * pad, bh = lines.length * lh + pad * 0.8;
    var bx = o.al === 'right' ? x - bw : o.al === 'center' ? x - bw / 2 : x, by = o.va === 'bottom' ? y - bh : o.va === 'middle' ? y - bh / 2 : y;
    bx = clamp(bx, 6, VW - bw - 6); by = clamp(by, 6, VH - bh - 6);
    if (o.anchor) {
      var ax = o.anchor[0], ay = o.anchor[1], ex = clamp(ax, bx, bx + bw), ey = clamp(ay, by, by + bh);
      if (ex > bx && ex < bx + bw && ey > by && ey < by + bh) ey = by + bh;
      line([[ex, ey], [ax, ay]], o.lc || o.border || '#9aa6b2', 1.1, [5, 4]); dot(ax, ay, 3, o.lc || o.border || '#9aa6b2');
    }
    c.save(); if (o.alpha !== undefined) c.globalAlpha = o.alpha;
    rect(bx, by, bw, bh, o.fill || 'rgba(8,11,15,0.88)', o.border || '#8f9ba7', 1.6);
    lines.forEach(function (l, i) { txt(l, bx + pad, by + pad * 0.4 + lh * (i + 0.5), px, i ? (o.col2 || P.ink2) : (o.col || P.ink), 'left', i ? 400 : (o.wt || 500), o.mono); });
    c.restore();
    return { x: bx, y: by, w: bw, h: bh };
  }
  /* verdict stamp: plain box, top right, appears at u = 0.55 */
  function stamp(u, s, ok) {
    if (!s || u < 0.55) return;
    var c = S.ctx, col = ok ? P.ok : P.bad, a = clamp((u - 0.55) / 0.06), w = tw(s, 34, 800) + 44;
    c.save(); c.globalAlpha = a; rect(VW - 36 - w, 30, w, 64, 'rgba(8,11,15,0.86)', col, 5); txt(s, VW - 36 - w / 2, 63, 34, col, 'center', 800); c.restore();
  }
  /* small tag such as a microscope overlay or panel name */
  function tag(s, x, y, o) { o = o || {}; var w = tw(s, o.px || 17, 600, true) + 16; rect(o.al === 'right' ? x - w : x, y, w, 28, o.fill || 'rgba(0,0,0,0.6)', o.border || null, 1); txt(s, (o.al === 'right' ? x - w : x) + 8, y + 14.5, o.px || 17, o.col || P.ink2, 'left', 600, true); return w; }

  /* ---------- paths and light ---------- */
  function mkPath(pts) { var L = [0]; for (var i = 1; i < pts.length; i++) L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])); return { p: pts, L: L, len: L[L.length - 1] }; }
  function ptAt(pa, s) {
    var d = clamp(s) * pa.len, i = 1; while (i < pa.L.length - 1 && pa.L[i] < d) i++;
    var a = pa.p[i - 1], b = pa.p[i], sl = pa.L[i] - pa.L[i - 1] || 1, t = (d - pa.L[i - 1]) / sl;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, Math.atan2(b[1] - a[1], b[0] - a[0])];
  }
  function subPts(pa, s0, s1) {
    s0 = clamp(s0); s1 = clamp(s1); if (s1 <= s0) return [];
    var out = [ptAt(pa, s0)], d0 = s0 * pa.len, d1 = s1 * pa.len;
    for (var i = 1; i < pa.p.length - 1; i++) if (pa.L[i] > d0 && pa.L[i] < d1) out.push(pa.p[i]);
    out.push(ptAt(pa, s1)); return out;
  }
  function sbend(x0, y0, x1, y1, n) { var p = []; n = n || 24; for (var i = 0; i <= n; i++) { var t = i / n; p.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * (1 - Math.cos(Math.PI * t)) / 2]); } return p; }
  function arcPts(cx, cy, r, a0, a1, n) { var p = []; n = n || 48; for (var i = 0; i <= n; i++) { var a = a0 + (a1 - a0) * i / n; p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); } return p; }
  function glow(pts, col, w, blur, alpha) {
    if (pts.length < 2) return; var c = S.ctx; c.save(); c.globalAlpha = alpha === undefined ? 1 : alpha; c.shadowColor = col; c.shadowBlur = blur || 14; c.lineCap = 'round';
    line(pts, col, w || 5); c.restore();
  }
  /* a light pulse of fractional length ln travelling along the path, head at s (reverse paths are given reversed) */
  function pulse(pa, s, col, ln, w, alpha) { if (s <= 0 || s - (ln || 0.08) >= 1) return; glow(subPts(pa, s - (ln || 0.08), s), col, w || 6, 16, alpha); }
  function flash(x, y, r, col, a) { var c = S.ctx; c.save(); c.globalAlpha = clamp(a); var g = c.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, col); g.addColorStop(1, 'rgba(0,0,0,0)'); c.fillStyle = g; c.beginPath(); c.arc(x, y, r, 0, 6.2832); c.fill(); c.restore(); }

  /* ---------- plots (instrument screens) ---------- */
  function niceTicks(a, b, step) { var t = [], s = Math.ceil(a / step - 1e-9) * step; for (var v = s; v <= b + step * 1e-6; v += step) t.push(+v.toFixed(10)); return t; }
  /* o: x, y, w, h, xr, yr, xs, ys (tick steps), xd, yd (decimals), xl, yl (axis labels with units), bg */
  function plot(o) {
    var c = S.ctx, m = { x: o.x, y: o.y, w: o.w, h: o.h, o: o };
    m.X = function (v) { return o.x + (v - o.xr[0]) / (o.xr[1] - o.xr[0]) * o.w; };
    m.Y = function (v) { return o.y + o.h - (v - o.yr[0]) / (o.yr[1] - o.yr[0]) * o.h; };
    rect(o.x, o.y, o.w, o.h, o.bg || P.scr, null);
    var px = o.px || 17, fs = Math.max(px, S.mf), xt = niceTicks(o.xr[0], o.xr[1], o.xs), yt = niceTicks(o.yr[0], o.yr[1], o.ys);
    font(px, 400, true);
    var xw = 0; xt.forEach(function (v) { xw = Math.max(xw, c.measureText(v.toFixed(o.xd || 0)).width); });
    var yw = 0; yt.forEach(function (v) { yw = Math.max(yw, c.measureText(v.toFixed(o.yd || 0)).width); });
    var xsp = xt.length > 1 ? Math.abs(m.X(xt[1]) - m.X(xt[0])) : o.w, ysp = yt.length > 1 ? Math.abs(m.Y(yt[1]) - m.Y(yt[0])) : o.h;
    var xk = Math.max(1, Math.ceil((xw + fs * 0.6) / xsp)), yk = Math.max(1, Math.ceil(fs * 1.15 / ysp));
    xt.forEach(function (v, i) { var X = m.X(v); line([[X, o.y], [X, o.y + o.h]], P.grid, 1.2); if (!o.noXt && i % xk === 0) txt(v.toFixed(o.xd || 0), X, o.y + o.h + fs * 0.95, px, P.ink2, 'center', 400, true); });
    yt.forEach(function (v, i) { var Y = m.Y(v); line([[o.x, Y], [o.x + o.w, Y]], P.grid, 1.2); if (!o.noYt && i % yk === 0) txt(v.toFixed(o.yd || 0), o.x - 8, Y, px, P.ink2, 'right', 400, true); });
    rect(o.x, o.y, o.w, o.h, null, P.axis, 1.6);
    var fl = Math.max(px + 1, S.mf);
    if (o.xl) txt(o.xl, o.x + o.w / 2, o.y + o.h + (o.noXt ? fl * 0.9 : fs * 1.9 + fl * 0.55), px + 1, P.ink2, 'center');
    if (o.yl) { c.save(); c.translate(o.x - Math.max(o.ylo || 62, (o.noYt ? 0 : yw + 8) + fl * 0.75), o.y + o.h / 2); c.rotate(-Math.PI / 2); txt(o.yl, 0, 0, px + 1, P.ink2, 'center'); c.restore(); }
    return m;
  }
  /* trace a function of x over [x0, x1] inside plot m */
  function ftrace(m, f, x0, x1, col, lw, n, dash) {
    var c = S.ctx, pts = [], N = n || 400; if (x1 <= x0) return;
    for (var i = 0; i <= N; i++) { var x = x0 + (x1 - x0) * i / N, y = f(x); if (y === null || y === undefined || isNaN(y)) continue; pts.push([m.X(x), clamp(m.Y(y), m.y, m.y + m.h)]); }
    c.save(); c.beginPath(); c.rect(m.x, m.y, m.w, m.h); c.clip(); line(pts, col, lw || 2.4, dash); c.restore();
  }
  function ptrace(m, pts, col, lw, dash) { var c = S.ctx; c.save(); c.beginPath(); c.rect(m.x, m.y, m.w, m.h); c.clip(); line(pts.map(function (p) { return [m.X(p[0]), m.Y(p[1])]; }), col, lw || 2.4, dash); c.restore(); }
  /* instrument bezel with a name; returns the inner screen rect */
  function inst(x, y, w, h, name, o) {
    o = o || {}; rect(x, y, w, h, o.body || '#161b21', '#3a4550', 2);
    if (name) txt(name, x + 12, y + 17, 15, P.ink3, 'left', 700, true);
    return { x: x + 10, y: y + 32, w: w - 20, h: h - 42 };
  }
  function readout(x, y, w, h, label, val, unit, col) {
    rect(x, y, w, h, '#05080a', '#2f3a45', 1.5);
    txt(label, x + 10, y + 15, 14, P.ink3, 'left', 600, true);
    var vs = Math.min(h * 0.55, 44); txt(val, x + w - 12 - (unit ? tw(unit, vs * 0.55, 500, true) + 8 : 0), y + h * 0.62, vs, col || '#7fe8a2', 'right', 500, true);
    if (unit) txt(unit, x + w - 10, y + h * 0.66, vs * 0.55, P.ink2, 'right', 500, true);
  }

  /* ---------- microscope primitives ---------- */
  function scopeTag(x, y, s) { if (S.narrow) return; tag(s, x, y, { col: '#d8dee6', fill: 'rgba(0,0,0,0.55)' }); }
  /* scale bar in the lower right of a view: len in micrometres, v = virtual units per micrometre */
  function scaleBar(x1, y, um, v, lab) {
    var L = um * v; rect(x1 - L, y - 6, L, 7, '#f4f6f8', null);
    txt(lab || (um >= 1000 ? (um / 1000) + ' mm' : um + ' µm'), x1 - L / 2, y - 22, 17, '#f4f6f8', 'center', 700);
  }
  function vignette(x, y, w, h, a) {
    var c = S.ctx, g = c.createRadialGradient(x + w / 2, y + h / 2, Math.min(w, h) * 0.35, x + w / 2, y + h / 2, Math.hypot(w, h) * 0.62);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(0,0,0,' + (a || 0.5) + ')'); c.fillStyle = g; c.fillRect(x, y, w, h);
  }
  /* oxide-clad chip surface with dicing edge and a few particles */
  function chipSurf(x, y, w, h, seed, col) {
    var c = S.ctx, g = c.createLinearGradient(x, y, x + w, y + h);
    g.addColorStop(0, col ? col[0] : P.oxideHi); g.addColorStop(1, col ? col[1] : P.oxideLo);
    c.fillStyle = g; c.fillRect(x, y, w, h);
    var r = rng(seed || 7); c.fillStyle = 'rgba(30,26,50,0.35)';
    for (var i = 0; i < 22; i++) { var px = x + r() * w, py = y + r() * h; c.fillRect(px, py, 1.5 + r() * 2.5, 1.5 + r() * 2.5); }
    rect(x, y, w, h, null, 'rgba(210,205,240,0.55)', 2.5);
    c.fillStyle = '#0b0f14';
    for (var j = 0; j < 9; j++) { var e = r(), sz = 2 + r() * 6; if (j % 2) c.fillRect(x + e * w, y + (j % 4 < 2 ? 0 : h - sz), sz * 1.6, sz); else c.fillRect(x + (j % 4 < 2 ? 0 : w - sz), y + e * h, sz, sz * 1.6); }
  }
  /* waveguide as seen in bright field: thin bright core with dark edges */
  function wg(pts, w, col) { line(pts, P.wgEdge, (w || 3) + 3); line(pts, col || P.wg, w || 3); }
  function goldRect(x, y, w, h) {
    var c = S.ctx, g = c.createLinearGradient(x, y, x + w, y + h); g.addColorStop(0, P.goldHi); g.addColorStop(0.45, P.gold); g.addColorStop(1, P.goldLo);
    c.fillStyle = g; c.fillRect(x, y, w, h); rect(x, y, w, h, null, 'rgba(90,64,20,0.8)', 1.4);
  }
  function goldPath(pts, w) { line(pts, P.goldLo, w + 2.5); line(pts, P.gold, w); line(pts, 'rgba(255,240,190,0.35)', w * 0.3); }
  function trench(x, y, w, h) { rect(x - 5, y - 5, w + 10, h + 10, P.trenchRim, null); rect(x, y, w, h, P.trench, null); rect(x, y, w, h, null, 'rgba(0,0,0,0.6)', 2); }
  /* flip-chip die from the top: lapped backside with n-metal, cleaved facets left and right */
  function dieBack(x, y, w, h, seed) {
    var c = S.ctx, g = c.createLinearGradient(x, y, x, y + h); g.addColorStop(0, '#c2a868'); g.addColorStop(0.5, P.die); g.addColorStop(1, P.dieLo);
    c.fillStyle = g; c.fillRect(x, y, w, h);
    var r = rng(seed || 3); c.strokeStyle = 'rgba(255,240,200,0.16)'; c.lineWidth = 1;
    for (var i = 0; i < 26; i++) { var a = x + r() * w, b = y + r() * h; c.beginPath(); c.moveTo(a, b); c.lineTo(a + 30 + r() * 60, b - 8 - r() * 12); c.stroke(); }
    line([[x, y], [x, y + h]], '#efe3bd', 2.5); line([[x + w, y], [x + w, y + h]], '#efe3bd', 2.5);
    rect(x, y, w, h, null, 'rgba(40,30,10,0.9)', 1.6);
  }
  /* focusing grating coupler at low magnification: fan taper and teeth, light exits along +x */
  function gcSmall(x, y, ang, sz) {
    var c = S.ctx; c.save(); c.translate(x, y); c.rotate(ang || 0); sz = sz || 1;
    c.fillStyle = 'rgba(194,201,242,0.35)'; c.beginPath(); c.moveTo(0, 0); c.arc(0, 0, 26 * sz, -0.36, 0.36); c.closePath(); c.fill();
    c.strokeStyle = 'rgba(225,230,255,0.85)'; c.lineWidth = 1.3;
    for (var i = 0; i < 6; i++) { c.beginPath(); c.arc(0, 0, (12 + i * 2.6) * sz, -0.4, 0.4); c.stroke(); }
    c.restore();
  }

  /* =====================================================================
     Flip-chip laser on a silicon photonics chip with SiN waveguide and a far-end grating coupler.
     Geometry (top view, 0.42 units per um): chip 3.0 x 1.0 mm, recess 680 x 300 um, die 600 x 250 um,
     waveguide path from the laser front facet to the grating about 1.9 mm.
     o.mode: 'bond' (die lowered by the bonder), 'gap' (Gary reworks the bond), 'whole' (the whole path is the cavity)
     ===================================================================== */
  var FC = { cx0: 170, cx1: 1430, cy0: 92, cy1: 512, rx: 230, ry: 232, rw: 286, rh: 126, dx: 263, dw: 256, dy: 242, dh: 105, yw: 295 };
  FC.path = mkPath([[FC.rx + FC.rw, FC.yw], [700, FC.yw]].concat(sbend(700, FC.yw, 860, 400, 30).slice(1), [[1290, 400]]));
  function liP(I) { var Ith = 18, eta = 0.085; if (I < Ith) return 0.0025 * I * I / Ith; var d = I - Ith; return 0.0025 * Ith + eta * d * (1 + 0.02 * Math.sin(2 * Math.PI * d / 27 - 0.6)); }
  function flipTop(u, o, lift) {
    var c = S.ctx;
    rect(0, 70, VW, 455, '#101318', null);
    chipSurf(FC.cx0, FC.cy0, FC.cx1 - FC.cx0, FC.cy1 - FC.cy0, 11);
    /* alignment marks */
    [[205, 125], [1395, 125]].forEach(function (p) { line([[p[0] - 12, p[1]], [p[0] + 12, p[1]]], '#d9dcf2', 2.5); line([[p[0], p[1] - 12], [p[0], p[1] + 12]], '#d9dcf2', 2.5); });
    /* fiber-array alignment loop: two grating couplers on the 127 um array pitch, in line with the output grating */
    var gp = 127 * 0.42, g1 = 400 - 5 * gp, g2 = 400 - 4 * gp;
    wg([[1290, g1], [1200, g1]].concat(arcPts(1200, (g1 + g2) / 2, gp / 2, -Math.PI / 2, -3 * Math.PI / 2, 24).slice(1), [[1290, g2]]), 2.2);
    gcSmall(1290, g1, 0, 1); gcSmall(1290, g2, 0, 1);
    /* probe pads and host-chip traces that run under the die (flip-chip contacts are on the host chip) */
    [[292, 'p'], [392, 'n']].forEach(function (q, i) {
      goldPath([[q[0] + 21, 155], [q[0] + 21, 262 + i * 40], [FC.dx + 60 + i * 120, 262 + i * 40]], 13);
      goldRect(q[0], 112, 42, 42);
    });
    trench(FC.rx, FC.ry, FC.rw, FC.rh);
    /* traces continue on the recess floor (metal down the sloped wall) */
    goldPath([[313, FC.ry - 4], [313, 262], [FC.dx + 60, 262]], 13); goldPath([[413, FC.ry - 4], [413, 302], [FC.dx + 180, 302]], 13);
    goldRect(FC.dx + 20, 252, 90, 22); goldRect(FC.dx + 140, 292, 90, 22);
    /* waveguide from the recess wall to the far-end grating */
    wg(FC.path.p, 3);
    gcSmall(1290, 400, 0, 1);
    /* the die, seen from its backside; ridge underneath shown as a hidden line */
    var dy = FC.dy - (lift || 0) * 70, sh = (lift || 0) * 26;
    if (sh > 0.5) { c.save(); c.globalAlpha = 0.45; rect(FC.dx + sh, FC.dy + sh, FC.dw, FC.dh, '#000', null); c.restore(); }
    dieBack(FC.dx, dy, FC.dw, FC.dh, 5);
    line([[FC.dx + 4, dy + FC.dh / 2], [FC.dx + FC.dw - 4, dy + FC.dh / 2]], 'rgba(255,255,255,0.75)', 2, [10, 7]);
    if (lift > 0.02) { /* bonder collet outline */ c.save(); c.globalAlpha = Math.min(1, lift * 2); rect(FC.dx + 60, dy - 28, FC.dw - 120, FC.dh + 56, 'rgba(160,170,180,0.18)', '#c3cad2', 2); c.restore(); }
    scaleBar(1350, 490, 500, 0.42);
    scopeTag(180, 470, 'Top view, 5x');
  }
  /* section A-A' along the waveguide: recess, solder, flipped die with active layer aligned to the SiN core */
  function flipSection(x, y, w, h, u, o, lift) {
    var c = S.ctx, xw = x + w * 0.56, zc = y + h * 0.56; /* recess wall, core height */
    rect(x, y, w, h, '#0d1014', '#2f3a45', 1.5);
    c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip();
    rect(x, zc + 34, w, y + h - zc - 34, P.si, null);                 /* Si substrate */
    rect(xw, zc - 40, w, 74, P.box, null);                            /* oxide stack right of the wall */
    rect(xw, zc - 5, w, 10, '#dfe4ff', null);                         /* SiN core */
    rect(x, zc + 34, xw - x, 6, '#3d434e', null);                     /* recess floor */
    for (var p = 0; p < 2; p++) rect(x + 70 + p * 170, zc + 4, 60, 30, '#b7bfcc', null); /* stand-off pedestals */
    rect(x + 160, zc + 28, 50, 6, P.gold, null);                      /* floor metal under the solder */
    var dz = -(lift || 0) * 150, gx = xw - 3;
    c.save(); c.translate(0, dz);
    rect(x + 160, zc + 8 - (lift > 0.05 ? 6 : 0), 50, 20, P.solder, null);          /* AuSn solder */
    rect(x + 30, zc - 110, gx - x - 30, 108, P.inp, null);           /* InP substrate, upside down */
    rect(x + 30, zc - 112, gx - x - 30, 6, P.gold, null);            /* n-metal on the backside (top) */
    rect(x + 30, zc - 6, gx - x - 30, 10, '#9aa0aa', null);          /* epitaxial layers */
    rect(x + 30, zc - 1.5, gx - x - 30, 3, P.mqw, null);             /* active layer */
    rect(x + 150, zc + 4, 70, 6, P.gold, null);                      /* p-metal on the ridge */
    line([[gx, zc - 112], [gx, zc + 6]], '#f0f2f6', 2);              /* front facet */
    c.restore();
    c.restore();
    rect(x, y, w, h, null, '#2f3a45', 1.5);
    if (!S.narrow) txt('Section along the waveguide', x + w - 12, y + 18, 15, P.ink2, 'right', 600, true);
    if (!S.narrow) {
      if (lift < 0.3) txt('InP laser, p-side down', x + 46, zc - 70 + dz, 17, '#eef1f5', 'left', 600);
      txt('Si substrate', x + 14, y + h - 16, 15, '#c8ced8', 'left', 500);
      txt('SiN waveguide', xw + 14, zc - 22, 15, '#20253a', 'left', 700);
    }
    return { xw: xw, zc: zc, gx: gx };
  }
  FB.flipChip = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var mode = o.mode || 'bond', lift = 0, rework = 0;
    if (mode === 'bond') lift = 1 - seg(u, 0.02, 0.55);
    if (mode === 'gap') { var cyc = (u * 3) % 1; rework = Math.min(2, Math.floor(u * 3)); lift = cyc < 0.5 ? Math.sin(Math.PI * cyc / 0.5) : 0; }
    flipTop(u, o, lift);
    var on = mode === 'bond' ? u > 0.6 : lift < 0.05;
    var t = (u * (mode === 'bond' ? 3 : 2.2)) % 1;
    if (on) {
      flash(FC.dx + FC.dw, FC.yw, 26, 'rgba(255,170,90,0.9)', 0.9);
      pulse(FC.path, t * 1.25, P.fwd, 0.1, 5);
      if (t * 1.25 > 1) flash(1300, 400, 46, mode === 'whole' ? 'rgba(255,90,80,0.95)' : 'rgba(86,212,255,0.8)', 1 - (t * 1.25 - 1) * 2);
      var rp = mkPath(FC.path.p.slice().reverse()), tb = t * 1.25 - 1;
      if (tb > 0) pulse(rp, tb * 1.25, P.back, 0.1, mode === 'whole' ? 6 : 4, mode === 'whole' ? 1 : 0.7);
    }
    var sec = flipSection(40, 540, 900, 210, u, o, lift);
    if (on) {
      var k = (u * 4) % 1; glow([[sec.gx + 2, sec.zc], [sec.gx + 2 + k * (940 - sec.gx), sec.zc]], P.fwd, 4, 10, 0.8);
      if (mode === 'gap') { flash(sec.gx + 1, sec.zc, 22, 'rgba(255,90,80,0.9)', 0.5 + 0.5 * Math.sin(u * 40)); }
    }
    /* right panel */
    if (mode === 'bond') {
      gcZoom(970, 540, 590, 210, u);
      box(1060, 300, ['Far-end grating coupler'], { anchor: [1296, 400], px: 19 });
      box(560, 440, ['Laser output facet'], { anchor: [FC.dx + FC.dw + 2, FC.yw + 4], px: 19 });
    } else if (mode === 'gap') {
      var m = plot({ x: 1040, y: 560, w: 500, h: 150, xr: [0, 120], yr: [0, 10], xs: 20, ys: 5, xl: 'Current (mA)', yl: 'Power (mW)', px: 14, ylo: 56 });
      for (var r = 0; r <= rework; r++) ftrace(m, liP, 0, 120, ['#f2c94c', '#56d4ff', '#ff7a6e'][r], 2, 300);
      txt('Rework ' + (rework + 1) + ' of 3: same ripple', 1040, 548, 16, P.ink2, 'left', 600);
      box(560, 445, ['Suspect: the gap, < 1 µm'], { anchor: [sec.gx, sec.zc], px: 20, border: P.bad });
      line([[FC.rx + FC.rw, FC.yw - 34], [FC.rx + FC.rw, FC.yw + 34]], P.bad, 3);
    } else {
      /* ruler along the whole optical path */
      var kk = seg(u, 0.05, 0.5), rp2 = mkPath(FC.path.p.map(function (q) { return [q[0], q[1] + 34]; }));
      line(subPts(rp2, 0, kk), P.ok, 3, [12, 6]);
      if (kk > 0) { var e = ptAt(rp2, kk); dot(e[0], e[1], 6, P.ok); }
      dot(FC.rx + FC.rw, FC.yw + 34, 6, P.ok);
      if (kk > 0.98) box(820, 112, ['Cavity: laser facet to grating, about 1.9 mm'], { al: 'center', px: 20, border: P.ok, col: '#bff0c8' });
      box(990, 560, ['Gap < 1 µm: period too long', 'Whole chip, 1.9 mm: period matches'], { px: 18 });
      box(1000, 300, ['Grating end reflects back to the laser'], { anchor: [1296, 400], px: 19, border: P.bad });
    }
    stamp(u, o.stamp, o.ok);
  };
  /* grating coupler at 50x: taper and focusing teeth, 630 nm period on 220 nm silicon */
  function gcZoom(x, y, w, h, u) {
    var c = S.ctx; rect(x, y, w, h, P.oxideLo, '#2f3a45', 1.5);
    c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip();
    var ox = x + 70, oy = y + h / 2, v = 9.5; /* units per um */
    wg([[x, oy], [ox, oy]], 4);
    c.fillStyle = 'rgba(194,201,242,0.45)'; c.beginPath(); c.moveTo(ox, oy); c.arc(ox, oy, 30 * v, -0.38, 0.38); c.closePath(); c.fill();
    c.strokeStyle = '#e4e8ff'; c.lineWidth = 0.315 * v;
    for (var i = 0; i < 26; i++) { c.beginPath(); c.arc(ox, oy, (14 + i * 0.63) * v, -0.42, 0.42); c.stroke(); }
    c.restore(); vignette(x, y, w, h, 0.4);
    scaleBar(x + w - 16, y + h - 16, 10, v); scopeTag(x + 8, y + 8, '50x');
  }

  /* light versus current with the ripple: P(I) and dP/dI, current swept with a cursor */
  FB.liCurve = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var Imax = 120, Ic = Math.min(Imax, Imax * seg(u, 0, 0.85) * 1.0001);
    var m = plot({ x: 150, y: 70, w: 1000, h: 360, xr: [0, Imax], yr: [0, 10], xs: 20, ys: 2, xl: null, yl: 'Power at the grating (mW)', noXt: true });
    ftrace(m, liP, 0, Ic, P.tr1, 2.6, 500);
    var m2 = plot({ x: 150, y: 470, w: 1000, h: 190, xr: [0, Imax], yr: [0, 0.14], xs: 20, ys: 0.05, yd: 2, xl: 'Laser current (mA)', yl: 'dP/dI (W/A)' });
    ftrace(m2, function (I) { return (liP(I + 0.25) - liP(I - 0.25)) / 0.5; }, 1, Ic, P.tr2, 2.2, 500);
    line([[m.X(Ic), m.y], [m.X(Ic), m2.y + m2.h]], 'rgba(255,255,255,0.35)', 1.5, [6, 6]);
    var rx = 1200;
    readout(rx, 80, 360, 84, 'LASER CURRENT', Ic.toFixed(1), 'mA', '#7fe8a2');
    readout(rx, 176, 360, 84, 'POWER', liP(Ic).toFixed(2), 'mW', '#f2c94c');
    if (u > 0.55) {
      var a = seg(u, 0.55, 0.7), dfn = function (I) { return (liP(I + 0.25) - liP(I - 0.25)) / 0.5; }, pk = [];
      for (var I = 40; I < 119; I += 0.25) if (dfn(I) > dfn(I - 0.25) && dfn(I) >= dfn(I + 0.25)) pk.push(I);
      var p0 = pk[pk.length - 2], p1 = pk[pk.length - 1];
      var c = S.ctx; c.save(); c.globalAlpha = a;
      var yb = m2.y + 14; line([[m2.X(p0), yb], [m2.X(p1), yb]], '#ffffff', 2); line([[m2.X(p0), yb - 10], [m2.X(p0), yb + 10]], '#fff', 2); line([[m2.X(p1), yb - 10], [m2.X(p1), yb + 10]], '#fff', 2);
      txt('period ' + (p1 - p0).toFixed(0) + ' mA', (m2.X(p0) + m2.X(p1)) / 2, m2.y - 14, 17, '#fff', 'center', 600);
      c.restore();
      box(rx, 290, ['Periodic ripple in power', 'Same period every sweep', 'The slope dP/dI shows it best'], { px: 19, alpha: a });
    }
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Modulator wiggles: unbalanced Mach-Zehnder transmission (period 1.6 nm) with a fine Fabry-Perot ripple from the two
     chip facets. Chip 8.0 mm facet to facet, group index 2.3: ripple period lambda^2 / (2 n L) = 0.065 nm.
     ===================================================================== */
  var MZ = { ng: 2.3, L: 8.0 };
  function fpPeriod(Lmm, ng) { return 1550 * 1550 / (2 * (ng || MZ.ng) * Lmm * 1e6); }
  function mzT(l) { return 0.5 * (1 + 0.92 * Math.cos(2 * Math.PI * (l - 1548.3) / 1.6)); }
  function mzData(l) { return mzT(l) * (1 + 0.07 * Math.cos(2 * Math.PI * (l - 1548) / fpPeriod(MZ.L) + 0.4)) + 0.004 * Math.sin(l * 9137.1) * Math.cos(l * 3311.7); }
  FB.mzmTrace = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var lend = 1548 + 4 * seg(u, 0, o.fit ? 0.01 : 0.8);
    var m = plot({ x: 150, y: 70, w: 1080, h: o.fit ? 380 : 560, xr: [1548, 1552], yr: [0, 1.1], xs: 0.5, ys: 0.2, xd: 1, yd: 1, yl: 'Transmission (normalized)', xl: o.fit ? null : 'Wavelength (nm)', noXt: !!o.fit });
    ftrace(m, mzData, 1548, lend, P.tr1, 1.6, 2400);
    var rx = 1270;
    readout(rx, 120, 290, 84, 'LASER WAVELENGTH', lend.toFixed(3), 'nm', '#7fe8a2');
    if (!o.fit) {
      if (u > 0.5) box(rx, 230, ['Fine ripple on every fringe', 'Period about 0.065 nm'], { px: 19, alpha: seg(u, 0.5, 0.6) });
      /* zoom of the ripple */
      if (u > 0.6) {
        var z = plot({ x: rx + 10, y: 330, w: 270, h: 200, xr: [1549.5, 1549.8], yr: [0.6, 1.05], xs: 0.1, ys: 0.2, xd: 1, yd: 1, px: 13, ylo: 40 });
        ftrace(z, mzData, 1549.5, 1549.8, P.tr1, 1.8, 400); txt('zoom', rx + 16, 316, 15, P.ink3, 'left', 600, true);
      }
    } else {
      var k = seg(u, 0.2, 0.55);
      var fit = function (l) { return mzT(l) * (1 + k * 0.07 * Math.cos(2 * Math.PI * (l - 1548) / fpPeriod(MZ.L) + 0.4)); };
      ftrace(m, fit, 1548, 1552, P.tr2, 1.4, 2400, [6, 4]);
      var r = plot({ x: 150, y: 490, w: 1080, h: 180, xr: [1548, 1552], yr: [-0.08, 0.08], xs: 0.5, ys: 0.04, xd: 1, yd: 2, xl: 'Wavelength (nm)', yl: 'Residual' });
      ftrace(r, function (l) { return mzData(l) - fit(l); }, 1548, 1552, P.tr3, 1.4, 2400);
      box(rx, 230, ['Fit model', 'a + b cos(2πλ / 1.60 nm + φ₁)', k > 0.05 ? '+ c cos(2πλ / 0.065 nm + φ₂)' : ' '], { px: 18, mono: false });
      readout(rx, 420, 290, 84, 'FIT RESIDUAL RMS', (0.05 * (1 - k) + 0.003).toFixed(3), '', k > 0.9 ? '#7fe8a2' : '#f2c94c');
      var lgx = rx; lgx += txt('fit (dashed)', lgx, 540, 16, P.tr2, 'left', 600) + 24; txt('data', lgx, 540, 16, P.tr1, 'left', 600);
    }
    stamp(u, o.stamp, o.ok);
  };
  /* thin-film lithium niobate MZM top view: edge couplers, MMIs, GSG traveling-wave electrodes fed from pads on the top
     edge. The upper arm passes under the S lead and both arms pass under the lower-ground lead (cladded crossings). */
  FB.mzmChip = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var c = S.ctx, x0 = 80, x1 = 1520, mm = (x1 - x0) / MZ.L, yc = 470, yu = 435, yd = 505;
    rect(0, 230, VW, 470, '#101318', null);
    chipSurf(x0, 250, x1 - x0, 430, 21, ['#4b4f66', '#3f4357']);
    var upP = [[x0, yc], [165, yc]].concat(sbend(205, yc - 5, 285, yu, 16), [[1235, yu]], sbend(1235, yu, 1315, yc - 5, 16), [[1435, yc], [x1, yc]]);
    var dnP = [[205, yc + 5]].concat(sbend(205, yc + 5, 285, yd, 16).slice(1), [[1235, yd]], sbend(1235, yd, 1315, yc + 5, 16));
    wg(upP.slice(0, 2), 3); wg(upP.slice(2, -2), 3); wg(dnP, 3); wg(upP.slice(-2), 3);
    rect(165, yc - 9, 40, 18, '#c6cbe8', '#45416a', 1.5); rect(1315, yc - 9, 40, 18, '#c6cbe8', '#45416a', 1.5); /* MMIs */
    line([[1355, yc], [1435, yc]], P.wgEdge, 6); line([[1355, yc], [1435, yc]], P.wg, 3);
    /* electrodes: G | gap (upper arm) | S | gap (lower arm) | G, gaps exaggerated */
    var Gt = [420, 1215, 335, yu - 9], Sg = [yu + 9, yd - 9], Gb = yd + 9;
    function el(x, y, w, h) { goldRect(x, y, w, h); }
    el(Gt[0], Gt[2], Gt[1] - Gt[0], Gt[3] - Gt[2]);
    el(372, Sg[0], 843, Sg[1] - Sg[0]); el(320, Gb, 895, 600 - Gb);
    /* left feed: pads on the top edge (G S G = lower ground, signal, upper ground) */
    el(320, 270, 44, 330); el(372, 270, 40, Sg[0] - 270 + 1); el(420, 270, 44, 66);
    /* right termination feed, mirrored */
    el(1171 + 0, 270, 44, 66); el(1223, 270, 40, Sg[0] - 270 + 1); el(1271, 270, 44, 330); el(1215, Gb, 100, 600 - Gb); el(1215, Sg[0], 48, Sg[1] - Sg[0]);
    /* hidden waveguide under the metal leads */
    line(upP.slice(2, -2), 'rgba(225,230,255,0.5)', 1.5, [5, 4]); line(dnP, 'rgba(225,230,255,0.5)', 1.5, [5, 4]);
    [[342, 'G'], [392, 'S'], [442, 'G'], [1193, 'G'], [1243, 'S'], [1293, 'G']].forEach(function (q) { txt(q[1], q[0], 290, 17, '#3a2a08', 'center', 800); });
    scaleBar(1500, 660, 1000, mm / 1000, '1 mm'); scopeTag(90, 640, 'Top view');
    /* reflectors and the pairs they form */
    var R = [{ x: x0, n: 'facet' }, { x: 185, n: 'MMI' }, { x: 342, n: 'crossing' }, { x: 1293, n: 'crossing' }, { x: 1335, n: 'MMI' }, { x: x1, n: 'facet' }];
    var path = mkPath(upP), t = (u * 2.4) % 1;
    pulse(path, t * 1.1, P.fwd, 0.06, 5);
    R.forEach(function (r) { var d = Math.abs(ptAt(path, t * 1.1)[0] - r.x); var a = clamp(1 - d / 60); flash(r.x, r.n === 'crossing' ? yu : yc, 40, 'rgba(255,90,80,0.95)', 0.25 + 0.75 * a); });
    var pairs = S.narrow ? [[0, 5], [1, 4]] : [[0, 5], [1, 4], [2, 3], [0, 1]], k = seg(u, 0.1, 0.7);
    pairs.forEach(function (pr, i) {
      var a = R[pr[0]], b = R[pr[1]], L = (b.x - a.x) / mm, per = fpPeriod(L), hit = i === 0, al = clamp(k * 4 - i);
      if (al <= 0) return;
      var top = 215 - i * Math.max(42, S.mf * 1.5), col = hit && u > 0.62 ? P.ok : '#c8d0da';
      c.save(); c.globalAlpha = al;
      line([[a.x, 250], [a.x, top], [b.x, top], [b.x, 250]], col, hit && u > 0.62 ? 2 : 1.1, [6, 4]);
      var lab = a.n + ' to ' + b.n + ': ' + L.toFixed(L < 1 ? 2 : 1) + ' mm, ripple ' + (per < 0.1 ? per.toFixed(3) : per.toFixed(2)) + ' nm';
      if (i === 3) txt(lab, b.x + 14, top, 17, col, 'left', 600); else txt(lab, (a.x + b.x) / 2, top - 14, 17, col, 'center', 600);
      c.restore();
    });
    box(VW / 2, 694, ['Measured ripple 0.065 nm: facet to facet. L = λ² / (2 n₉ Δλ), n₉ = 2.3'.replace(/₉/g, 'g')], { al: 'center', px: 20, border: u > 0.62 ? P.ok : '#8f9ba7', alpha: seg(u, 0.55, 0.65) });
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Mode-hop model used by every hop scene. External-cavity comb: spacing 0.15 nm, moves 0.02 nm/K.
     Filter (distributed feedback grating): moves 0.10 nm/K. The laser keeps its mode until the filter is more than
     half a spacing plus 0.03 nm away: up and down sweeps hop at different temperatures (hysteresis about 0.75 K).
     ===================================================================== */
  var HOP = { l0: 1550.0, T0: 25, fsr: 0.15, kf: 0.10, kc: 0.02, d: 0.03 };
  function modeL(m, T) { return HOP.l0 + m * HOP.fsr + HOP.kc * (T - HOP.T0); }
  function filtL(T) { return HOP.l0 + HOP.kf * (T - HOP.T0); }
  function nearestM(T) { return Math.round((filtL(T) - modeL(0, T)) / HOP.fsr); }
  function sweep(Ta, Tb, dT) {
    var m = nearestM(Ta), out = [], n = Math.max(1, Math.round(Math.abs(Tb - Ta) / (dT || 0.01)));
    for (var i = 0; i <= n; i++) {
      var T = Ta + (Tb - Ta) * i / n, det = filtL(T) - modeL(m, T), hop = false;
      if (det > HOP.fsr / 2 + HOP.d) { m++; hop = true; } else if (det < -HOP.fsr / 2 - HOP.d) { m--; hop = true; }
      out.push({ T: T, m: m, l: modeL(m, T), hop: hop });
    }
    return out;
  }
  function hopsOf(sw) { return sw.filter(function (s) { return s.hop; }).map(function (s) { return s.T; }); }
  /* spectrum: comb lines under the filter, lasing line highlighted */
  function combPanel(x, y, w, h, T, m, o) {
    o = o || {};
    var pm = plot({ x: x, y: y, w: w, h: h, xr: [1549.75, 1551.35], yr: [0, 1.15], xs: 0.25, ys: 0.5, xd: 2, yd: 1, xl: 'Wavelength (nm)', yl: 'Relative power', px: 15, ylo: 50 });
    var fc = filtL(T), lor = function (l) { return 1 / (1 + Math.pow((l - fc) / 0.15, 2)); };
    ftrace(pm, lor, 1549.75, 1551.35, P.tr1, 2, 300);
    for (var k = -4; k < 14; k++) {
      var l = modeL(k, T); if (l < 1549.75 || l > 1551.35) continue;
      var on = k === m && !o.off, X = pm.X(l);
      line([[X, pm.Y(0)], [X, pm.Y(on ? 1.0 : 0.05 + 0.12 * lor(l))]], on ? P.tr2 : '#7d8995', on ? 4 : 2);
      if (on) S.ctx.save(), flash(X, pm.Y(1.0), 18, 'rgba(86,212,255,0.8)', 0.8), S.ctx.restore();
    }
    if (!S.narrow && !o.noLegend) { txt('filter', pm.X(fc) + 34, pm.Y(0.62), 16, P.tr1, 'left', 600); txt('cavity modes, 0.15 nm apart', x + w - 10, y + 16, 15, '#9aa6b2', 'right', 600); }
    return pm;
  }
  /* temperature sweep with the wavemeter: o.dir 'up' | 'down', o.prior shows the earlier up sweep */
  FB.hopSweep = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var up = o.dir !== 'down', Ta = up ? 25 : 35, Tb = up ? 35 : 25, k = seg(u, 0.02, 0.92), T = Ta + (Tb - Ta) * k;
    var sw = sweep(Ta, T, 0.01), cur = sw[sw.length - 1];
    readout(50, 70, 360, 90, 'TEC SETPOINT', T.toFixed(2), '°C', '#f2c94c');
    readout(50, 172, 360, 90, 'WAVEMETER', cur.l.toFixed(3), 'nm', '#7fe8a2');
    combPanel(520, 70, 1030, 250, T, cur.m);
    /* log: wavelength versus temperature, logged every 0.25 K */
    var lm = plot({ x: 560, y: 400, w: 990, h: 260, xr: [25, 35], yr: [1549.95, 1551.05], xs: 1, ys: 0.2, xd: 0, yd: 1, xl: 'Temperature (°C)', yl: S.narrow ? null : 'Logged wavelength (nm)', px: 15, ylo: 74 });
    function logDots(s, col, r) { s.forEach(function (q, i) { if (i % 25 === 0) dot(lm.X(q.T), lm.Y(q.l), r || 4.5, col); }); }
    var hopC = function (s, col) { hopsOf(s).forEach(function (Th) { var a = s.filter(function (q) { return Math.abs(q.T - Th) < 0.006; })[0]; S.ctx.strokeStyle = col; S.ctx.lineWidth = 2.5; S.ctx.beginPath(); S.ctx.arc(lm.X(Th), lm.Y(a.l + (s[0].T > s[s.length - 1].T ? 1 : -1) * HOP.fsr / 2), 15, 0, 6.3); S.ctx.stroke(); }); };
    if (o.prior) { var pr = sweep(25, 35, 0.01); logDots(pr, 'rgba(86,212,255,0.55)', 3.5); hopC(pr, 'rgba(86,212,255,0.7)'); }
    logDots(sw, up ? P.tr2 : P.tr1);
    if (o.prior) hopC(sw, P.bad);
    /* table of the last readings */
    var rsp = Math.max(36, Math.max(19, S.mf) * 1.15), nr = Math.max(3, Math.floor((350 - 2 * Math.max(30, S.mf * 1.15)) / rsp)), rows = sw.filter(function (q, i) { return i % 25 === 0; }).slice(-nr);
    rect(50, 290, 360, 380, '#0d1116', '#2f3a45', 1.5); var th = Math.max(30, S.mf * 1.15);
    txt(up ? 'Log, sweep up' : 'Log, sweep down', 62, 310, 16, P.ink2, 'left', 700); txt('T (°C)', 62, 310 + th, 15, P.ink3, 'left', 600, true); txt('λ (nm)', 250, 310 + th, 15, P.ink3, 'left', 600, true);
    rows.forEach(function (q, i) { var prev = i ? rows[i - 1] : null, jump = prev && Math.abs(q.l - prev.l) > 0.08; txt(q.T.toFixed(2), 62, 310 + 2 * th + i * rsp, 19, jump ? P.bad : P.ink, 'left', 500, true); txt(q.l.toFixed(3), 250, 310 + 2 * th + i * rsp, 19, jump ? P.bad : P.ink, 'left', 500, true); });
    if (o.prior && u > 0.8) box(560, 410, ['Hops going down happen about 0.75 K lower than going up'], { px: 18, border: P.bad });
    stamp(u, o.stamp, o.ok);
  };
  /* wavelength versus temperature, up and down: two offset staircases */
  FB.hopLoop = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var m = plot({ x: 150, y: 70, w: 1040, h: 560, xr: [25, 35], yr: [1549.95, 1551.05], xs: 1, ys: 0.1, xd: 0, yd: 1, xl: 'Laser temperature (°C)', yl: 'Lasing wavelength (nm)', ylo: 84 });
    ftrace(m, filtL, 25, 35, 'rgba(242,201,76,0.45)', 2, 50, [8, 6]);
    var su = sweep(25, 35, 0.01), sd = sweep(35, 25, 0.01), ku = seg(u, 0, 0.45), kd = seg(u, 0.45, 0.9);
    function stair(s, k, col) { var n = Math.floor((s.length - 1) * k), pts = []; for (var i = 0; i <= n; i++) { if (s[i].hop && i) pts.push([s[i].T, s[i - 1].l]); pts.push([s[i].T, s[i].l]); } if (pts.length > 1) ptrace(m, pts, col, 3); }
    stair(su, ku, P.tr2); stair(sd, kd, P.tr1);
    var hu = hopsOf(su), hd = hopsOf(sd);
    if (u > 0.9) {
      var Th = hu[2], Tl = hd.filter(function (t) { return t < Th; })[0] || hd[2];
      S.ctx.save(); S.ctx.fillStyle = 'rgba(255,255,255,0.08)'; S.ctx.fillRect(m.X(Tl), m.y, m.X(Th) - m.X(Tl), m.h); S.ctx.restore();
    }
    var rx = 1230;
    txt('Sweep up', rx, 130, 19, P.tr2, 'left', 700); txt('Sweep down', rx, 160, 19, P.tr1, 'left', 700); txt('Filter', rx, 190, 19, 'rgba(242,201,76,0.8)', 'left', 700);
    if (ku > 0.99) box(rx, 230, ['Hops going up (°C)', hu.map(function (t) { return t.toFixed(2); }).join('  ')], { px: 17, mono: true });
    if (kd > 0.99) box(rx, 330, ['Hops going down (°C)', hd.map(function (t) { return t.toFixed(2); }).join('  ')], { px: 17, mono: true });
    if (u > 0.9) box(rx, 430, ['Both modes are stable', 'in the shaded band:', 'the laser keeps the one it had'], { px: 18 });
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Screen documents: tickets, forms, logs, invitations, messages. o.title, o.app, o.rows [[label, value, flags]],
     flags: 'bad' | 'ok' | 'strike' | 'type' (typed out over the keyframe) | 'hl' (highlighted) | 'muted';
     o.readouts [{ label, value, unit, col }] shown in a column on the right; o.note: plain box under the window.
     ===================================================================== */
  FB.doc = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var side = o.readouts && o.readouts.length, x = o.x || (side ? 60 : 220), w = o.w || (side ? 1060 : 1160), y = 110, rows = o.rows || [];
    var lh = Math.max(o.lh || 54, Math.max(21, S.mf) * 1.5), h = o.h || Math.min(700, 110 + rows.length * lh + (o.button ? 74 : 10));
    var lw = 0; rows.forEach(function (r) { lw = Math.max(lw, tw(r[0], 18, 600)); });
    rect(x + 10, y + 12, w, h, 'rgba(0,0,0,0.5)', null);
    rect(x, y, w, h, '#eef1f5', '#9aa4b0', 1.5);
    rect(x, y, w, 46, '#27303a', null);
    txt(o.app || 'Lab tracker', x + 18, y + 23, 18, '#c9d3dd', 'left', 600);
    txt(o.title || '', x + 18, y + 80, 28, '#18202a', 'left', 700);
    var typed = seg(u, 0.1, 0.6);
    rows.forEach(function (r, i) {
      var ry = y + 128 + i * lh, f = r[2] || '', v = String(r[1]);
      if (f.indexOf('type') >= 0) v = v.slice(0, Math.round(v.length * typed)) + (typed < 1 && (u * 6) % 1 < 0.5 ? '|' : '');
      line([[x + 18, ry - lh / 2 + 2], [x + w - 18, ry - lh / 2 + 2]], '#d6dbe2', 1);
      txt(r[0], x + 18, ry, 18, '#5b6673', 'left', 600);
      var col = f.indexOf('bad') >= 0 ? '#b3261e' : f.indexOf('ok') >= 0 ? '#1f7a3a' : f.indexOf('muted') >= 0 ? '#7b8796' : '#18202a';
      var vx = x + Math.max(o.lx || 300, lw + 40);
      if (f.indexOf('hl') >= 0) rect(vx - 6, ry - 18, tw(v, 21, 500) + 12, 36, '#ffe58a', null);
      var vw = txt(v, vx, ry, 21, col, 'left', f.indexOf('bad') >= 0 || f.indexOf('ok') >= 0 ? 700 : 500);
      if (f.indexOf('strike') >= 0) line([[vx - 4, ry], [vx + vw + 4, ry]], '#b3261e', 3);
    });
    if (o.button) { var bw = tw(o.button, 20, 700) + 40, bx = x + w - bw - 18, by = y + h - 62, pr = u > 0.7 && u < 0.78; rect(bx, by, bw, 44, pr ? '#08427e' : '#0b5cab', null); txt(o.button, bx + bw / 2, by + 22, 20, '#fff', 'center', 700); }
    if (side) o.readouts.forEach(function (r, i) { readout(1160, 130 + i * 112, 400, 96, r.label, r.value, r.unit, r.col); if (r.sub) txt(r.sub, 1170, 130 + i * 112 + 108, 15, P.ink3, 'left', 500); });
    if (o.note) box(x, y + h + 26, [].concat(o.note), { px: 20, border: o.noteBad ? P.bad : o.noteOk ? P.ok : '#8f9ba7', alpha: seg(u, 0.35, 0.45) });
    stamp(u, o.stamp, o.ok);
  };

  /* wavemeter versus time at the demo: o.phase 'hop' (mode jumps by one comb spacing) or 'restart' (driver off, on) */
  FB.wmTimeline = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var restart = o.phase === 'restart', steady = o.phase === 'steady', T = 29.6, l2 = modeL(2, T), l3 = modeL(3, T);
    var tEnd = restart ? 60 + 60 * seg(u, 0, 0.8) : 60 * seg(u, 0, 0.85), tHop = steady ? 1e9 : 33, tOff = 72, tOn = 76;
    var wl = function (t) { if (t < tHop) return l2; if (!restart || t < tOff) return l3; if (t < tOn) return null; return l2; };
    var m = plot({ x: 150, y: 120, w: 980, h: 420, xr: [0, restart ? 120 : 60], yr: [1550.30, 1550.62], xs: restart ? 20 : 10, ys: 0.05, xd: 0, yd: 2, xl: 'Time (s)', yl: 'Wavemeter reading (nm)', ylo: 100 });
    line([[m.X(0), m.Y(l2)], [m.X(restart ? 120 : 60), m.Y(l2)]], 'rgba(63,185,80,0.5)', 2, [10, 6]);
    txt('customer channel', m.X(1), m.Y(l2) + 18, 16, '#7fd18f', 'left', 600);
    var pts = []; for (var t = 0; t <= tEnd; t += 0.1) { var v = wl(t); if (v === null) { if (pts.length > 1) ptrace(m, pts, P.tr2, 3); pts = []; continue; } if (pts.length && Math.abs(pts[pts.length - 1][1] - v) > 0.05) pts.push([t, pts[pts.length - 1][1]]); pts.push([t, v]); }
    if (pts.length > 1) ptrace(m, pts, P.tr2, 3);
    var now = wl(tEnd), lock = now !== null && Math.abs(now - l2) < 0.01;
    readout(1180, 130, 380, 96, 'WAVEMETER', now === null ? '- - - -' : now.toFixed(3), 'nm', '#7fe8a2');
    rect(1180, 242, 380, 96, '#05080a', '#2f3a45', 1.5); txt('CUSTOMER RECEIVER', 1190, 257, 14, P.ink3, 'left', 600, true);
    rect(1196, 280, 348, 44, lock ? '#14431f' : '#5a1512', lock ? P.ok : P.bad, 2); txt(lock ? 'LOCKED' : 'NO LOCK', 1370, 302, 26, lock ? '#8ff0a4' : '#ff9d96', 'center', 800, true);
    var on = !(restart && tEnd > tOff && tEnd < tOn);
    readout(1180, 354, 380, 96, 'LASER DRIVER', on ? '120.0' : '0.0', 'mA', on ? '#f2c94c' : '#66727e');
    txt(on ? 'OUTPUT ON' : 'OUTPUT OFF', 1190, 466, 16, on ? '#f2c94c' : P.bad, 'left', 700, true);
    if (tEnd > tHop) box(m.X(tHop) - 14, m.Y(l3) - 20, ['Jump of 0.150 nm: one cavity mode'], { px: 18, border: P.bad, al: 'right', va: 'bottom' });
    if (restart && tEnd > tOn + 4) { box(m.X(tOn) + 20, m.Y(l2) + 30, ['"It re-locked."'], { px: 24, wt: 700 }); box(150, 610, ['Lab log: no entry. Hop point, temperature and sweep direction not recorded.'], { px: 19, border: P.bad, alpha: seg(u, 0.6, 0.7) }); }
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Edge coupling: flat-cleaved fiber (125 um cladding, 8.2 um core) facing a cleaved silicon chip facet with an
     inverse-taper waveguide. 2.4 units per um. Air gap Fabry-Perot: period lambda^2 / (2 L) = 24.0 nm at 50 um.
     ===================================================================== */
  function gapT(l, L, gel) {
    var R1 = gel ? 0.0001 : 0.034, R2 = gel ? 0.0004 : 0.03, ph = 4 * Math.PI * L * 1000 / l;
    var fp = (1 - R1) * (1 - R2) / (1 + R1 * R2 - 2 * Math.sqrt(R1 * R2) * Math.cos(ph));
    return 10 * Math.log10(fp * 0.065 * (1 - 0.6 * Math.pow((l - 1560) / 140, 2)));
  }
  function edgeView(x, y, w, h, gapUm, o) {
    var c = S.ctx, v = o.v || 2.4, xf = x + w * 0.62, ya = y + h / 2, xt = xf - gapUm * v;
    c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip();
    rect(x, y, w, h, '#1b1d22', null);
    /* chip: top surface and the cleaved facet */
    var g = c.createLinearGradient(xf, 0, x + w, 0); g.addColorStop(0, '#77738f'); g.addColorStop(1, '#5d5a75'); c.fillStyle = g; c.fillRect(xf, y + 20, x + w - xf, h - 40);
    line([[xf, y + 20], [xf, y + h - 20]], '#e9ecf5', 3);
    /* inverse taper: 180 nm tip at the facet widening into the waveguide */
    c.fillStyle = '#d4d9f5'; c.beginPath(); c.moveTo(xf, ya - 0.5); c.lineTo(xf + 300, ya - 2.2); c.lineTo(x + w, ya - 2.2); c.lineTo(x + w, ya + 2.2); c.lineTo(xf + 300, ya + 2.2); c.lineTo(xf, ya + 0.5); c.fill();
    if (o.gel) { c.save(); c.globalAlpha = 0.55 * o.gel; c.fillStyle = '#8fc7e8'; c.beginPath(); c.ellipse(xf - gapUm * v / 2, ya, gapUm * v / 2 + 60 * o.gel, 120 + 60 * o.gel, 0, 0, 6.2832); c.fill(); c.restore(); }
    /* fiber: glass cylinder with highlights and a faint core */
    var fr = 62.5 * v, fg = c.createLinearGradient(0, ya - fr, 0, ya + fr);
    fg.addColorStop(0, 'rgba(210,225,235,0.55)'); fg.addColorStop(0.18, 'rgba(120,140,155,0.35)'); fg.addColorStop(0.5, 'rgba(160,180,195,0.25)'); fg.addColorStop(0.82, 'rgba(120,140,155,0.35)'); fg.addColorStop(1, 'rgba(230,240,248,0.7)');
    c.fillStyle = fg; c.fillRect(x, ya - fr, xt - x, 2 * fr);
    line([[x, ya - fr], [xt, ya - fr]], '#dfe8ee', 2); line([[x, ya + fr], [xt, ya + fr]], '#dfe8ee', 2); line([[xt, ya - fr], [xt, ya + fr]], '#f2f6f9', 3);
    rect(x, ya - 4.1 * v, xt - x, 8.2 * v, 'rgba(230,240,255,0.18)', null);
    c.restore(); rect(x, y, w, h, null, '#2f3a45', 1.5);
    return { xf: xf, xt: xt, ya: ya, v: v };
  }
  /* o.mode: 'approach' | 'backoff' | 'gel'; o.trace draws the swept transmission below */
  FB.edgeCouple = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var mode = o.mode || 'approach', gap = mode === 'approach' ? lerp(140, 50, seg(u, 0, 0.45)) : mode === 'backoff' ? lerp(50, 55, seg(u, 0.15, 0.45)) : 50;
    var gel = mode === 'gel' ? seg(u, 0.45, 0.75) : 0, vh = o.trace ? 300 : 560;
    var ev = edgeView(40, 70, 1100, vh, gap, { gel: gel, v: o.trace ? 1.3 : 2.4 });
    var t = (u * 2.5) % 1, yA = ev.ya;
    /* forward light in the core, partial reflections bouncing in the gap */
    glow([[40, yA], [lerp(40, ev.xt, Math.min(1, t * 2)), yA]], P.fwd, 5, 12, 0.85);
    if (t > 0.5) glow([[ev.xf, yA], [lerp(ev.xf, 1140, (t - 0.5) * 2), yA]], P.fwd, 3, 10, 0.8);
    if (gel < 0.5) for (var b = 0; b < 3; b++) { var tb = ((u * 7) + b / 3) % 1, xb = b % 2 ? lerp(ev.xf, ev.xt, tb) : lerp(ev.xt, ev.xf, tb); dot(xb, yA + 9 * (b - 1), 4, 'rgba(255,90,80,' + (0.9 - 0.5 * gel) + ')'); }
    flash(ev.xt, yA, 26, 'rgba(255,90,80,0.9)', (gel < 0.5 ? 0.7 : 0.1)); flash(ev.xf, yA, 26, 'rgba(255,90,80,0.9)', (gel < 0.5 ? 0.7 : 0.1));
    scaleBar(1120, 70 + vh - 18, 50, ev.v); scopeTag(52, 82, 'Stereo microscope, top view');
    readout(1180, 120, 380, 96, 'FIBER STAGE, GAP', gap.toFixed(1), 'µm', '#7fe8a2');
    if (!o.trace && mode !== 'gel') {
      box(ev.xt - 20, yA + 170, ['Flat-cleaved fiber, 3.3 % reflection'], { px: 18, anchor: [ev.xt, yA + 60], al: 'right' });
      box(ev.xf + 30, yA + 110, ['Cleaved chip facet'], { px: 18, anchor: [ev.xf, yA + 60] });
    }
    if (mode === 'gel') {
      var rk = seg(u, 0.05, 0.35); line([[ev.xt, yA + 190], [lerp(ev.xt, ev.xf, rk), yA + 190]], P.ok, 3); line([[ev.xt, yA + 175], [ev.xt, yA + 205]], P.ok, 3); line([[ev.xf, yA + 175], [ev.xf, yA + 205]], P.ok, 3);
      txt('L = 50 µm', (ev.xt + ev.xf) / 2, yA + 222, 20, '#bff0c8', 'center', 700);
      box(1180, 250, ['Δλ = λ² / (2 L)', '(1550 nm)² / (2 × 50 µm)', '= 24.0 nm'], { px: 21, border: P.ok });
      if (gel > 0.2) box(1180, 420, ['Index-matching gel in the gap:', 'both reflections drop, the ripple goes'], { px: 18, alpha: gel });
    }
    if (o.trace) {
      var m = plot({ x: 150, y: 430, w: 1000, h: 250, xr: [1500, 1630], yr: [-14, -10.5], xs: 10, ys: 1, xd: 0, yd: 0, xl: 'Wavelength (nm)', yl: 'Transmission (dB)', px: 15, ylo: 56 });
      var lend = 1500 + 130 * seg(u, 0, 0.5);
      if (mode === 'backoff') { ftrace(m, function (l) { return gapT(l, 50); }, 1500, 1630, P.tr2, 2, 800); ftrace(m, function (l) { return gapT(l, gap); }, 1500, 1630, P.tr1, 2.2, 800); var gl = Math.max(30, S.mf * 1.3); txt('gap 50 µm', 1180, 450, 18, P.tr2, 'left', 700); txt('gap ' + gap.toFixed(1) + ' µm', 1180, 450 + gl, 18, P.tr1, 'left', 700);
        var per = 1550 * 1550 / (2 * gap * 1000); box(1180, 470 + 2 * Math.max(30, S.mf * 1.3), ['Period at 1550 nm', '50 µm: 24.0 nm', gap.toFixed(1) + ' µm: ' + per.toFixed(1) + ' nm'], { px: 18, mono: false }); }
      else ftrace(m, function (l) { return gapT(l, gap); }, 1500, lend, P.tr1, 2.2, 800);
    }
    stamp(u, o.stamp, o.ok);
  };
  /* the reported ripple: o.report overlays the report window with the fit-and-subtract result */
  FB.gapTrace = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var m = plot({ x: 150, y: 120, w: 1000, h: 400, xr: [1500, 1630], yr: [-14, -10.5], xs: 10, ys: 0.5, xd: 0, yd: 1, xl: 'Wavelength (nm)', yl: 'Transmission (dB)', ylo: 66 });
    var lend = o.report ? 1630 : 1500 + 130 * seg(u, 0, 0.8);
    ftrace(m, function (l) { return gapT(l, 50); }, 1500, lend, P.tr1, 2.4, 800);
    var a = 1550 - 12, b = 1550 + 12; /* bracket one period around 1550 nm: peaks of cos(4 pi L / lambda) */
    var pk = []; for (var l = 1520; l < 1600; l += 0.01) { var f0 = gapT(l - 0.01, 50), f1 = gapT(l, 50), f2 = gapT(l + 0.01, 50); if (f1 > f0 && f1 >= f2) pk.push(l); }
    var p0 = pk.filter(function (q) { return q < 1550; }).pop(), p1 = pk.filter(function (q) { return q > 1550; })[0];
    if (lend > p1 + 2) {
      var yb = m.Y(-10.75); line([[m.X(p0), yb], [m.X(p1), yb]], '#fff', 2); line([[m.X(p0), yb - 10], [m.X(p0), yb + 10]], '#fff', 2); line([[m.X(p1), yb - 10], [m.X(p1), yb + 10]], '#fff', 2);
      txt((p1 - p0).toFixed(0) + ' nm', (m.X(p0) + m.X(p1)) / 2, yb - 20, 19, '#fff', 'center', 700);
    }
    readout(1190, 130, 370, 96, 'TUNABLE LASER', lend.toFixed(2), 'nm', '#7fe8a2');
    if (o.report) {
      var k = seg(u, 0.15, 0.5);
      ftrace(m, function (l) { return lerp(gapT(l, 50), -12.25 - 0.3 * Math.pow((l - 1560) / 140, 2) * 2.3, k); }, 1500, 1630, P.tr2, 2, 800, [7, 5]);
      rect(1190, 250, 370, 300, '#eef1f5', '#9aa4b0', 1.5); rect(1190, 250, 370, 40, '#27303a', null); txt('Report draft', 1204, 270, 17, '#c9d3dd', 'left', 600);
      rect(1204, 306, 330, 34, '#ffe58a', null); txt('"Intrinsic ripple"', 1210, 323, 19, '#18202a', 'left', 700);
      txt('Fitted, subtracted.', 1210, 370, 18, '#18202a', 'left', 500); txt('Cause: device.', 1210, 370 + Math.max(32, S.mf * 1.3), 18, '#18202a', 'left', 500);
      var pr = u > 0.7 && u < 0.78; rect(1430, 490, 110, 42, pr ? '#08427e' : '#0b5cab', null); txt('Send', 1485, 511, 19, '#fff', 'center', 700);
    }
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Fiber bench parts, top view, s units per mm. 14-pin butterfly (7 pins per long side, 2.54 mm pitch, body
     20.8 x 12.7 mm, flange tabs with 2.7 mm holes, fiber snout on the short end), inline isolator (5.5 x 35 mm,
     printed arrow along the forward direction), FC adapters, patch cords, splitter tube, spool, detector heads.
     ===================================================================== */
  function fiber(pts, w, col) { line(pts, '#5c4a10', w + 2); line(pts, col || P.jacket, w); line(pts, 'rgba(255,255,255,0.3)', Math.max(1, w * 0.25)); }
  function metalRect(x, y, w, h, hi, lo, vert) { var c = S.ctx, g = vert ? c.createLinearGradient(x, y, x + w, y) : c.createLinearGradient(x, y, x, y + h); g.addColorStop(0, hi); g.addColorStop(0.5, lo); g.addColorStop(1, hi); c.fillStyle = g; c.fillRect(x, y, w, h); rect(x, y, w, h, null, 'rgba(0,0,0,0.6)', 1.2); }
  /* butterfly laser on a mount: x, y is the body centre, fiber leaves toward +x; returns the boot end */
  function butterfly(x, y, s, o) {
    o = o || {}; var c = S.ctx;
    if (!o.noMount) { rect(x - 24 * s, y - 21 * s, 48 * s, 42 * s, '#1a1d21', '#33383f', 1.5); [[-21, -18], [21, -18], [-21, 18], [21, 18]].forEach(function (p) { dot(x + p[0] * s, y + p[1] * s, 1.6 * s, '#4a5058'); dot(x + p[0] * s, y + p[1] * s, 0.8 * s, '#111'); }); }
    for (var i = 0; i < 7; i++) { var px = x + (i - 3) * 2.54 * s; [-1, 1].forEach(function (sd) { metalRect(px - 0.3 * s, y + sd * 6.35 * s + (sd < 0 ? -7 * s : 0), 0.6 * s, 7 * s, P.goldHi, P.gold, true); }); }
    if (!o.noMount) [-1, 1].forEach(function (sd) { rect(x - 11 * s, y + sd * 10 * s - 1.6 * s, 22 * s, 3.2 * s, '#2a2f35', '#454c55', 1); });
    metalRect(x - 15 * s, y - 3.5 * s, 30 * s, 7 * s, '#c7cbd0', '#8c9198');
    [-13.2, 13.2].forEach(function (hx) { dot(x + hx * s, y, 1.35 * s, '#1a1d21'); });
    metalRect(x - 10.4 * s, y - 6.35 * s, 20.8 * s, 12.7 * s, '#d9dce0', '#a5aab1');
    rect(x - 9.4 * s, y - 5.35 * s, 18.8 * s, 10.7 * s, null, 'rgba(0,0,0,0.25)', 1);
    dot(x + 8.2 * s, y - 4.2 * s, 0.6 * s, '#333');
    metalRect(x + 10.4 * s, y - 1.75 * s, 5.5 * s, 3.5 * s, '#d0d4d9', '#959aa1');
    c.fillStyle = o.boot || '#2a2d33'; c.beginPath(); c.moveTo(x + 15.9 * s, y - 2 * s); c.lineTo(x + 31 * s, y - 1 * s); c.lineTo(x + 31 * s, y + 1 * s); c.lineTo(x + 15.9 * s, y + 2 * s); c.closePath(); c.fill();
    if (o.glow) flash(x + 10.6 * s, y, 7 * s, o.glow, o.glowA === undefined ? 0.8 : o.glowA);
    return [x + 31 * s, y];
  }
  /* inline isolator from x0 along +x; dir +1 arrow toward +x; returns [in, out] fiber points */
  function inlineIso(x0, y, s, dir, o) {
    o = o || {}; var c = S.ctx, L = 35 * s, r = 2.75 * s, b = 9 * s;
    c.fillStyle = '#26292e'; [[x0 - b, x0], [x0 + L, x0 + L + b]].forEach(function (q, i) { c.beginPath(); c.moveTo(q[i ? 0 : 1], y - r * 0.8); c.lineTo(q[i ? 1 : 0], y - r * 0.3); c.lineTo(q[i ? 1 : 0], y + r * 0.3); c.lineTo(q[i ? 0 : 1], y + r * 0.8); c.closePath(); c.fill(); });
    metalRect(x0, y - r, L, 2 * r, '#e3e6ea', '#9aa0a8');
    var ax0 = x0 + L * 0.22, ax1 = x0 + L * 0.78; if (dir < 0) { var t = ax0; ax0 = ax1; ax1 = t; }
    line([[ax0, y], [ax1 - dir * 1.6 * s, y]], '#14181d', 0.55 * s); arrowHead(ax1, y, dir > 0 ? 0 : Math.PI, 2.2 * s, '#14181d');
    if (o.label && !S.narrow) txt('1550 nm', x0 + L / 2, y + r + 3.2 * s, Math.max(15, 2.6 * s), '#cfd6de', 'center', 600);
    return [[x0 - b, y], [x0 + L + b, y]];
  }
  /* FC adapter (mating sleeve) with two connector boots; o.open: right side has no mate, flat ferrule end */
  function fcPair(x, y, s, o) {
    o = o || {}; metalRect(x - 4 * s, y - 5 * s, 8 * s, 10 * s, '#c9ced4', '#7f858d', true);
    [-1, 1].forEach(function (sd) { if (o.open && sd > 0) return; metalRect(x + sd * 4 * s - (sd < 0 ? 6 * s : 0), y - 2.6 * s, 6 * s, 5.2 * s, '#d5d9de', '#8a9098', true); rect(x + sd * 10 * s - (sd < 0 ? 9 * s : 0), y - 2 * s, 9 * s, 4 * s, o.boot || '#2a62b8', null); });
    return [[x - 19 * s, y], [x + 19 * s, y]];
  }
  /* single FC/PC connector end, open: ferrule end at +x */
  function fcOpen(x, y, s, o) {
    o = o || {}; rect(x - 13 * s, y - 2 * s, 9 * s, 4 * s, o.boot || '#2a62b8', null); metalRect(x - 4 * s, y - 2.6 * s, 6 * s, 5.2 * s, '#d5d9de', '#8a9098', true);
    rect(x + 2 * s, y - 1.25 * s, 3 * s, 2.5 * s, '#efe9dc', null); line([[x + 5 * s, y - 1.25 * s], [x + 5 * s, y + 1.25 * s]], '#ffffff', 2);
    return [x + 5 * s, y];
  }
  function spool(x, y, r) { dot(x, y, r, '#2b2f35'); dot(x, y, r * 0.86, '#d8b947'); for (var i = 0; i < 6; i++) { S.ctx.strokeStyle = 'rgba(90,70,10,0.5)'; S.ctx.lineWidth = 1; S.ctx.beginPath(); S.ctx.arc(x, y, r * (0.45 + i * 0.07), 0, 6.3); S.ctx.stroke(); } dot(x, y, r * 0.42, '#2b2f35'); dot(x, y, r * 0.12, '#111'); }
  function pmHead(x, y, s) { dot(x, y, 12.5 * s, '#23272c'); dot(x, y, 10 * s, '#3a4048'); dot(x, y, 4 * s, '#111417'); line([[x + 12 * s, y], [x + 30 * s, y]], '#111', 2.5 * s); }
  function benchBg(y0, y1) { rect(0, y0, VW, y1 - y0, P.bench, null); for (var x = 25; x < VW; x += 50) for (var y = y0 + 25; y < y1; y += 50) { dot(x, y, 5.5, P.hole); dot(x - 1, y - 1, 3, '#0d0f11'); } }
  function telegraphState(t) { var k = Math.floor(t * 9.7), h = Math.sin(k * 12.9898 + 4.1) * 43758.5453; h -= Math.floor(h); return h > 0.45 ? 1 : 0; }
  /* oscilloscope: fast photodiode voltage switching between two levels */
  function scopePanel(x, y, w, h, u, o) {
    o = o || {}; var sc = inst(x, y, w, h, o.name || 'OSCILLOSCOPE, FAST PHOTODIODE');
    var m = plot({ x: sc.x + 62, y: sc.y + 6, w: sc.w - 74, h: sc.h - Math.max(52, S.mf * 3.3), xr: [0, 10], yr: [0, 400], xs: 1, ys: 100, xd: 0, yd: 0, xl: o.xl || 'Time (µs)', yl: 'mV', px: 13, ylo: 50 });
    var pts = [], off = u * 3; for (var t = 0; t <= 10; t += 0.01) { var st = telegraphState(t / 10 * 3 + off), v = (st ? 290 : 170) + 9 * Math.sin(t * 913.1 + off * 50) * Math.cos(t * 271.3); pts.push([t, v]); }
    ptrace(m, pts, '#f2e94c', 1.6); return m;
  }
  /* electrical spectrum analyzer, delayed self-heterodyne beat note at 80 MHz; width in MHz */
  function esaPanel(x, y, w, h, width, o) {
    o = o || {}; var sc = inst(x, y, w, h, 'ELECTRICAL SPECTRUM ANALYZER');
    var m = plot({ x: sc.x + 62, y: sc.y + 6, w: sc.w - 74, h: sc.h - Math.max(52, S.mf * 3.3), xr: [30, 130], yr: [-90, -20], xs: 20, ys: 20, xd: 0, yd: 0, xl: 'Frequency (MHz)', yl: 'dBm', px: 13, ylo: 50 });
    ftrace(m, function (f) { var l = 1 / (1 + Math.pow((f - 80) / (width / 2), 2)); return -82 + 10 * Math.log10(1e-6 + l * Math.pow(10, 5.6 * Math.min(1, 1.2 / Math.sqrt(width))) ) + 0.4 * Math.sin(f * 37.1) * Math.cos(f * 13.7); }, 30, 130, '#f2e94c', 1.6, 600);
    if (o.ref) ftrace(m, function (f) { var l = 1 / (1 + Math.pow((f - 80) / 0.25, 2)); return -82 + 10 * Math.log10(1e-6 + l * Math.pow(10, 5.6)); }, 30, 130, 'rgba(86,212,255,0.6)', 1.4, 1200, [5, 4]);
    return m;
  }
  /* o.variant: 'budget' | 'remove' | 'customer' | 'reversed' */
  FB.linkBench = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var v = o.variant || 'budget', s = 5, y = 250, c = S.ctx;
    benchBg(110, 420);
    var noIso = v === 'remove' || v === 'customer', lift = v === 'remove' ? seg(u, 0.05, 0.4) : v === 'customer' ? 1 : 0, rev = v === 'reversed';
    var hot = (noIso && (v === 'customer' || u > 0.45)) || rev;
    var bootEnd = butterfly(170, y, s, { glow: hot ? 'rgba(255,90,80,0.95)' : 'rgba(255,170,90,0.7)', glowA: hot ? 0.5 + 0.4 * Math.sin(u * 50) : 0.6 });
    var p1 = [[bootEnd[0], y], [430, y]];
    fiber(p1, 4); var a1 = fcPair(470, y, s);
    var path = [[bootEnd[0] - 80, y], [565, y]];
    if (v === 'customer' || v === 'reversed') {
      if (rev) {
        var io = inlineIso(620, y, s, -1, { label: true }); fiber([a1[1], io[0]], 4); fiber([io[1], [843, y]], 4);
        rect(900, y - 70, 260, 140, '#2b3036', '#4b545e', 2); txt('Optical amplifier', 1030, y - 40, 19, '#dfe5ea', 'center', 700); if (!S.narrow) txt('erbium-doped fiber', 1030, y - 14, 15, P.ink2, 'center', 500);
        fcPair(900, y, 3); fcPair(1160, y, 3); txt('IN', 940, y + 44, 14, P.ink2, 'center', 700); txt('OUT', 1120, y + 44, 14, P.ink2, 'center', 700);
        fiber([[1217, y], [1540, y]], 4);
        var fp = mkPath([[bootEnd[0] - 80, y], [620 + 175 + 45, y]]), t = (u * 2.2) % 1, sb = fp.L[1] ? (620 - 45 - fp.p[0][0]) / fp.len : 0.6;
        var hs = t * 1.1; if (hs < sb + 0.05) pulse(fp, hs, P.fwd, 0.12, 5); else flash(620 - 45, y, 30, 'rgba(86,212,255,0.8)', 1 - (hs - sb) * 3);
        var bp = mkPath([[880, y], [bootEnd[0] - 80, y]]); for (var k = 0; k < 4; k++) pulse(bp, ((u * 3) + k / 4) % 1 * 1.1, P.back, 0.06, 5, 0.85);
        box(430, 330, ['Arrow points at the laser'], { px: 19, border: P.bad, anchor: [707, y - 14] });
        box(980, 125, ['Backward amplified spontaneous emission'], { px: 18, border: P.bad, anchor: [890, y - 8] });
        doRow(450, ['Laser light going forward', 'blocked: isolation', P.bad], ['Amplifier light going backward', 'passes: insertion loss only', P.bad]);
      } else {
        fiber([a1[1], [700, y]], 6);
        rect(700, 130, 600, 260, P.pcb, '#2f7a52', 2);
        for (var q = 0; q < 6; q++) rect(720 + q * 34, 350, 20, 22, '#1b1e22', null);
        metalRect(760, y - 8, 270, 16, '#e1e4e8', '#8f959c');
        txt('1 x 2 splitter, 50:50', 895, y - 26, 16, '#dfeee5', 'center', 600);
        fiber([[700, y], [760, y]], 4); fiber([[1030, y - 4], [1120, y - 50], [1200, y - 50]], 4); fiber([[1030, y + 4], [1120, y + 50], [1160, y + 50]], 4);
        fcPair(1220, y - 50, 3); fiber([[1277, y - 50], [1560, y - 50]], 6); txt('to receiver', 1540, y - 74, 16, P.ink2, 'right', 600);
        var oe = fcOpen(1200, y + 50, 3);
        rect(1240, y + 70, 14, 10, '#1f4a9a', null); txt('dust cap', 1262, y + 92, 14, P.ink2, 'left', 600);
        flash(oe[0], oe[1], 40, 'rgba(255,90,80,0.95)', 0.6 + 0.3 * Math.sin(u * 40));
        box(640, 22, S.narrow ? ['Open flat port: 3.3 % back, 0.8 % home'] : ['Open port, flat-polished: 3.3 % back', 'Passes the splitter twice: 0.8 %, about -21 dB'], { px: 17, border: P.bad, anchor: [oe[0], oe[1] - 8] });
        var fw = mkPath([[bootEnd[0] - 80, y], [1030, y], [1120, y + 50], [oe[0], y + 50]]), t2 = (u * 2.6) % 1;
        pulse(fw, t2, P.fwd, 0.08, 5); pulse(mkPath(fw.p.slice().reverse()), ((u * 2.6) + 0.5) % 1, P.back, 0.08, 5);
        scopePanel(40, 440, 740, 300, u); esaPanel(820, 440, 740, 300, 22, { ref: true });
      }
    } else {
      /* isolator between two adapters, arrow away from the laser */
      c.save(); c.translate(0, -150 * lift); c.globalAlpha = 1 - 0.4 * lift;
      var iso = inlineIso(580, y, s, 1, { label: true }); fiber([a1[1], iso[0]], 4); fiber([iso[1], [800, y]], 4);
      c.restore();
      if (lift > 0.5) { rect(560, 112, 260, 70, 'rgba(200,210,220,0.18)', '#aab4be', 1.5); txt('parts bag', 690, 172, 15, P.ink2, 'center', 600); }
      if (lift > 0.6) fiber([a1[1], [800, y]], 4);
      var a2 = fcPair(840, y, s); fiber([a2[1], [960, y], [960, y - 60]], 6); spool(1010, y - 60, 70); fiber([[1080, y - 60], [1150, y], [1240, y]], 6);
      pmHead(1270, y, s);
      var pwr = v === 'remove' ? lerp(4.2, 4.7, seg(u, 0.35, 0.5)) : 4.2;
      readout(1350, 200, 210, 100, 'POWER METER', pwr.toFixed(1), 'dBm', '#f2c94c');
      var fp2 = mkPath([[bootEnd[0] - 80, y], [1240, y]]), t3 = (u * 2.2) % 1; pulse(fp2, t3 * 1.1, P.fwd, 0.08, 5);
      var bp2 = mkPath([[1240, y], [bootEnd[0] - 80, y]]), tb = ((u * 2.2) + 0.5) % 1;
      if (lift > 0.6) { pulse(bp2, tb * 1.1, P.back, 0.08, 5); box(300, 330, ['Isolator removed: every reflection now reaches the laser'], { px: 20, border: P.bad, alpha: seg(u, 0.45, 0.55) }); }
      else if (!lift) { var bh = tb * 1.1, sx = (1240 - 800) / (1240 - bootEnd[0] + 80); if (bh < sx) pulse(bp2, bh, P.back, 0.06, 4, 0.8); else flash(800, y, 24, 'rgba(255,90,80,0.8)', 1 - (bh - sx) * 4); }
      if (v === 'budget') {
        var rows = [['Laser output, fiber', '+6.0 dBm'], ['Isolator', '-0.5 dB'], ['Connectors, 3 matings', '-0.9 dB'], ['Fiber spool', '-0.4 dB'], ['Splitter at the customer', '-3.5 dB'], ['Received', '+0.7 dBm'], ['Required', '+1.0 dBm'], ['Margin', '-0.3 dB']];
        rect(260, 440, 760, 310, '#0d1116', '#2f3a45', 1.5); txt('LINK BUDGET', 276, 460, 15, P.ink3, 'left', 700, true);
        var rsp = Math.max(32, Math.max(19, S.mf) * 1.15); if (S.narrow) rows = rows.filter(function (r, i) { return i === 1 || i >= 5; });
        rows.forEach(function (r, i) { var yy = 492 + i * rsp, bad = r[0] === 'Margin', hl = r[0] === 'Isolator' && u > 0.5; if (hl) rect(268, yy - rsp / 2, 744, rsp, 'rgba(242,184,75,0.18)', null); txt(r[0], 280, yy, 19, bad ? P.bad : P.ink, 'left', bad ? 700 : 500); txt(r[1], 1000, yy, 19, bad ? P.bad : P.ink, 'right', bad ? 700 : 500, true); if (r[0] === 'Splitter at the customer') line([[276, yy + rsp / 2], [1004, yy + rsp / 2]], '#2f3a45', 1); });
        box(1060, 440, ['Short by 0.3 dB', 'Demo on Monday'], { px: 20, border: P.bad });
      }
    }
    stamp(u, o.stamp, o.ok);
  };
  function doRow(y, a, b) { [a, b].forEach(function (r, i) { var x = 260 + i * 560; rect(x, y, 520, 100, '#0d1116', '#2f3a45', 1.5); txt(r[0], x + 16, y + 30, 20, P.ink, 'left', 600); txt(r[1], x + 16, y + 66, 20, r[2] || (i ? P.bad : P.ok), 'left', 700); }); }

  /* free-space polarization-dependent isolator in a rotation mount: input polarizer at theta from the laser
     polarization; transmission cos^2(theta) (law of Malus). o.rot: animate 0 to 90 degrees; o.fixed: angle in degrees */
  FB.fsIso = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var th = o.fixed !== undefined ? o.fixed : 90 * seg(u, 0.1, 0.55), T = Math.pow(Math.cos(th * Math.PI / 180), 2), y = 250, c = S.ctx;
    benchBg(110, 400);
    /* laser diode in a mount with a collimation tube */
    rect(130, y - 60, 90, 120, '#1c1f23', '#3a4048', 1.5); metalRect(220, y - 22, 110, 44, '#3c4249', '#22262b'); txt('laser', 175, y + 84, 16, P.ink2, 'center', 600);
    /* beam */
    glow([[330, y], [640, y]], P.fwd, 10, 18, 0.75);
    if (T > 0.01) glow([[880, y], [1240, y]], P.fwd, 10, 18, 0.75 * T);
    /* isolator: input polarizer, Faraday rotator in its magnet, output polarizer, in a rotation mount */
    rect(620, y - 90, 280, 180, '#202429', '#454c55', 2);
    metalRect(650, y - 60, 220, 120, '#3b4148', '#1d2126');
    rect(650, y - 60, 18, 120, '#4c5560', null); rect(852, y - 60, 18, 120, '#4c5560', null);
    line([[700, y], [820, y - 0.01]], '#d6dde5', 3); arrowHead(822, y, 0, 14, '#d6dde5');
    txt('input', 659, y - 76, 15, P.ink2, 'center', 600); txt('output', 861, y - 76, 15, P.ink2, 'center', 600);
    /* detector card */
    rect(1240, y - 70, 24, 140, '#e8e4cf', '#8a8670', 1.5); flash(1240, y, 46, 'rgba(255,240,150,0.95)', T);
    txt('detector card', 1252, y + 92, 16, P.ink2, 'center', 600);
    /* end view of the input face: laser E field vertical, polarizer axis rotating */
    var ex = 260, ey = 600, R = 120;
    dot(ex, ey, R + 18, '#1d2126'); dot(ex, ey, R, '#2f353c');
    for (var a = 0; a < 360; a += 10) { var ra = a * Math.PI / 180, l = a % 90 ? 8 : 16; line([[ex + (R + 18) * Math.cos(ra), ey + (R + 18) * Math.sin(ra)], [ex + (R + 18 - l) * Math.cos(ra), ey + (R + 18 - l) * Math.sin(ra)]], '#9aa4ae', 1.5); }
    var pa = -Math.PI / 2 + th * Math.PI / 180;
    for (var k = -4; k <= 4; k++) { var ox = Math.cos(pa + Math.PI / 2) * k * 22, oy = Math.sin(pa + Math.PI / 2) * k * 22, hl = Math.sqrt(Math.max(0, R * R - (k * 22) * (k * 22))) * 0.92; line([[ex + ox - Math.cos(pa) * hl, ey + oy - Math.sin(pa) * hl], [ex + ox + Math.cos(pa) * hl, ey + oy + Math.sin(pa) * hl]], 'rgba(200,210,225,0.35)', 2); }
    line([[ex - Math.cos(pa) * R, ey - Math.sin(pa) * R], [ex + Math.cos(pa) * R, ey + Math.sin(pa) * R]], P.warn, 4);
    arrow(ex, ey + 70, ex, ey - 70, P.fwd, 5, 18); arrow(ex, ey - 70, ex, ey + 70, P.fwd, 5, 18);
    txt('Input face, seen from the laser', ex, ey - R - 40, 17, P.ink2, 'center', 600);
    txt('laser polarization', ex + R + 34, ey - 20, 17, P.fwd, 'left', 600); txt('input polarizer axis', ex + R + 34, ey + 12, 17, P.warn, 'left', 600);
    readout(800, 470, 330, 96, 'ANGLE', th.toFixed(0), 'deg', '#f2c94c');
    readout(800, 580, 330, 96, 'T = cos²(angle)', (100 * T).toFixed(1), '%', T > 0.5 ? '#7fe8a2' : '#ff9d96');
    if (!S.narrow) box(1170, 470, ['Forward path inside the isolator', 'Input polarizer: along the laser', 'Faraday rotator: +45 degrees', 'Output polarizer: at 45 degrees'], { px: 18 });
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Rings. Coupled-mode doublet: clockwise and counter-clockwise modes coupled at rate beta; total loss kappa =
     kappa0 + kappaE (all in MHz, angular factors dropped). t = 1 - kE (iD - k/2) / ((iD - k/2)^2 + b^2),
     r = i b kE / ((iD - k/2)^2 + b^2). Defaults follow the barely resolved doublet in Raja et al. (2019): splitting
     118 MHz with a loaded linewidth near 110 MHz.
     ===================================================================== */
  function dbl(D, p) {
    p = p || {}; var k0 = p.k0 || 70, ke = p.ke || 40, b = p.b === undefined ? 59 : p.b, k = k0 + ke;
    var ar = -k / 2, ai = D, zr = ar * ar - ai * ai + b * b, zi = 2 * ar * ai, den = zr * zr + zi * zi;
    var qr = (ar * zr + ai * zi) / den, qi = (ai * zr - ar * zi) / den; /* (a) / z */
    var tr = 1 + ke * qr, ti = ke * qi;
    var rr = b * ke * zi / den, ri = b * ke * zr / den; /* i b kE / z */
    return { T: tr * tr + ti * ti, R: rr * rr + ri * ri };
  }
  /* ring top view at 1.6 units per um (R = 100 um), bus below the ring, light enters the bus from the left */
  function ringView(x, y, w, h, u, o) {
    var c = S.ctx, v = 1.6, cx = x + w * 0.5, cy = y + h * 0.44, R = 100 * v, wgw = 1.6 * v, yb = cy + R + 1.1 * v + wgw;
    c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip();
    chipSurf(x + 70, y - 10, w - 140, h + 20, 31, ['#5c5f78', '#4c4f66']);
    wg(arcPts(cx, cy, R, 0, 6.2832, 120), wgw); wg([[x + 70, yb], [x + w - 70, yb]], wgw);
    /* lensed fibers at both facets (125 um fiber, short cone, tip at the facet) */
    [[x + 70, -1], [x + w - 70, 1]].forEach(function (f) {
      var tx = f[0] + f[1] * 6, r = 62.5 * v, cl = 70 * v, bx = tx + f[1] * cl, g = c.createLinearGradient(0, yb - r, 0, yb + r);
      g.addColorStop(0, 'rgba(235,242,248,0.85)'); g.addColorStop(0.5, 'rgba(170,190,205,0.55)'); g.addColorStop(1, 'rgba(235,242,248,0.85)');
      c.fillStyle = g; c.beginPath(); c.moveTo(bx, yb - r); c.lineTo(tx + f[1] * 8, yb - 8); c.quadraticCurveTo(tx, yb, tx + f[1] * 8, yb + 8); c.lineTo(bx, yb + r); c.lineTo(bx + f[1] * 400, yb + r); c.lineTo(bx + f[1] * 400, yb - r); c.closePath(); c.fill();
      c.strokeStyle = '#dfe8ee'; c.lineWidth = 1.5; c.stroke();
    });
    c.restore(); rect(x, y, w, h, null, '#2f3a45', 1.5);
    return { cx: cx, cy: cy, R: R, yb: yb, v: v, x0: x + 70, x1: x + w - 70 };
  }
  /* angle on screen (y down): forward light from a left-to-right bus below the ring circulates counter-clockwise */
  function ringPulse(g, a0, a1, col, w) { glow(arcPts(g.cx, g.cy, g.R, a0, a1, 14), col, w || 5, 14); }
  /* o.variant: 'sweep' | 'standing' */
  FB.ringScatter = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var g = ringView(40, 70, 1000, 560, u, o), sa = -0.7, sp = [g.cx + g.R * Math.cos(sa), g.cy + g.R * Math.sin(sa)], t = u * 3;
    /* forward bus light and counter-clockwise circulation (angle decreasing from the bottom) */
    var bus = mkPath([[g.x0, g.yb], [g.x1, g.yb]]); pulse(bus, (t % 1) * 1.1, P.fwd, 0.1, 5);
    for (var k = 0; k < 3; k++) { var a = Math.PI / 2 - ((t * 0.9 + k / 3) % 1) * 2 * Math.PI; ringPulse(g, a + 0.35, a, P.fwd, 5); }
    /* backscatter: clockwise light from the scatterer, and light leaving the bus backward */
    for (var j = 0; j < 2; j++) { var b0 = sa + ((t * 0.9 + j / 2) % 1) * 2 * Math.PI; ringPulse(g, b0 - 0.3, b0, P.back, 4); }
    pulse(mkPath([[g.cx, g.yb], [g.x0, g.yb]]), ((t * 1.3) % 1) * 1.1, P.back, 0.1, 4, 0.9);
    dot(sp[0], sp[1], 7, '#1b1a22'); flash(sp[0], sp[1], 26, 'rgba(255,240,200,0.9)', 0.5 + 0.3 * Math.sin(u * 30));
    box(sp[0] + 40, sp[1] - 90, ['Scatterer on the ring'], { px: 19, anchor: [sp[0] + 6, sp[1] - 6] });
    scaleBar(1010, 610, 100, g.v); scopeTag(52, 82, 'Top view, 20x');
    if (o.variant === 'standing') {
      /* the two standing waves: antinode or node at the scatterer */
      [['Mode at lower frequency', 0], ['Mode at higher frequency', 1]].forEach(function (q, i) {
        var bx = 1080, by = 90 + i * 220, bw = 480, bh = 190; rect(bx, by, bw, bh, '#0d1116', '#2f3a45', 1.5); txt(q[0], bx + 12, by + 20, 17, P.ink2, 'left', 600);
        var x0 = bx + 20, x1 = bx + bw - 20, yy = by + 120; line([[x0, yy], [x1, yy]], '#7d86a6', 10);
        var pts = []; for (var xx = x0; xx <= x1; xx += 2) { var ph = (xx - (x0 + x1) / 2) / 26; var I = Math.pow(q[1] ? Math.sin(ph) : Math.cos(ph), 2); pts.push([xx, yy - 8 - 70 * I * (0.6 + 0.4 * Math.sin(u * 20 + i * 1.5) * Math.sin(u * 20 + i * 1.5))]); }
        line(pts, i ? P.tr2 : P.tr1, 2.6); dot((x0 + x1) / 2, yy, 7, '#1b1a22'); txt(q[1] ? 'node at the scatterer' : 'antinode at the scatterer', (x0 + x1) / 2, yy + 34, 16, P.ink2, 'center', 600);
      });
      var m = plot({ x: 1140, y: 540, w: 410, h: 150, xr: [-300, 300], yr: [0, 1.05], xs: 150, ys: 0.5, xd: 0, yd: 1, xl: 'Detuning (MHz)', px: 13, ylo: 40 });
      ftrace(m, function (D) { return dbl(D).T; }, -300, 300, P.tr1, 2.2);
      dot(m.X(-59), m.Y(dbl(-59).T) - 14, 5, P.tr1); dot(m.X(59), m.Y(dbl(59).T) - 14, 5, P.tr2);
    } else {
      var m2 = plot({ x: 1140, y: 120, w: 410, h: 260, xr: [-300, 300], yr: [0, 1.05], xs: 150, ys: 0.2, xd: 0, yd: 1, xl: 'Detuning (MHz)', yl: 'Transmission', px: 14, ylo: 48 });
      ftrace(m2, function (D) { return dbl(D, o.b ? { b: o.b } : {}).T; }, -300, -300 + 600 * seg(u, 0.05, 0.8), P.tr1, 2.4);
    }
    stamp(u, o.stamp, o.ok);
  };
  /* doublet plots: o.variant 'singlefit' (one Lorentzian through both dips), 'twomodes' (labelled as two families) */
  FB.doubletPlot = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var p = o.variant === 'twomodes' ? { b: 90 } : {}, f0 = 193.4e6; /* MHz at 1550 nm */
    var m = plot({ x: 150, y: 110, w: 1000, h: 520, xr: [-400, 400], yr: [0, 1.05], xs: 100, ys: 0.2, xd: 0, yd: 1, xl: 'Detuning (MHz)', yl: 'Transmission (normalized)', ylo: 70 });
    var tr = function (D) { return dbl(D, p).T + 0.006 * Math.sin(D * 1.37) * Math.cos(D * 0.71); };
    ftrace(m, tr, -400, -400 + 800 * seg(u, 0, 0.35), P.tr1, 2.6, 600);
    var kL = (p.k0 || 70) + (p.ke || 40);
    if (o.variant === 'twomodes') {
      if (u > 0.4) { var a = seg(u, 0.4, 0.5); S.ctx.save(); S.ctx.globalAlpha = a; [[-90, 'mode 1'], [90, 'mode 2']].forEach(function (q) { var X = m.X(q[0]), Y = m.Y(tr(q[0])); arrow(X, Y + 120, X, Y + 14, '#fff', 2.5, 14); txt(q[1], X, Y + 140, 22, '#fff', 'center', 700); }); S.ctx.restore(); }
      rect(1190, 120, 370, 60, '#eef1f5', null); txt('Dual-mode ring', 1206, 150, 24, '#18202a', 'left', 800);
    } else {
      var mn = 1; for (var D = -300; D <= 300; D += 1) mn = Math.min(mn, tr(D)); var half = 1 - (1 - mn) / 2, xl = -300, xr = 300;
      for (var D2 = -300; D2 < 0; D2 += 0.5) if (tr(D2) < half) { xl = D2; break; } for (var D3 = 300; D3 > 0; D3 -= 0.5) if (tr(D3) < half) { xr = D3; break; }
      var fw = xr - xl, fit = function (D) { return 1 - (1 - mn) / (1 + Math.pow(D / (fw / 2), 2)); };
      if (u > 0.4) ftrace(m, fit, -400, -400 + 800 * seg(u, 0.4, 0.6), P.tr2, 2.2, 400, [8, 5]);
      readout(1190, 120, 370, 96, 'Q, ONE LORENTZIAN', (f0 / fw / 1e6).toFixed(2), 'million', '#f2c94c');
      readout(1190, 230, 370, 96, 'Q, COUPLED MODES', (f0 / kL / 1e6).toFixed(2), 'million', '#7fe8a2');
      txt('data', 1190, 350, 17, P.tr1, 'left', 700); txt('one-Lorentzian fit', 1270, 350, 17, P.tr2, 'left', 700);
      if (u > 0.6) box(1190, 390, ['Wafer scrap request', 'Reason: Q too low'], { px: 19, border: P.bad, alpha: seg(u, 0.6, 0.7) });
    }
    stamp(u, o.stamp, o.ok);
  };
  /* fit tool: one shallow dip, coupling branch chosen. Tmin = 0.25 gives Q0 = 2 QL / (1 +/- 0.5): 2.0 or 6.0 million */
  FB.fitTool = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var x = 120, y = 100, w = 1360, h = 600;
    rect(x, y, w, h, '#eef1f5', '#9aa4b0', 1.5); rect(x, y, w, 46, '#27303a', null); txt('Resonance fit', x + 18, y + 23, 18, '#c9d3dd', 'left', 600);
    var m = plot({ x: x + 90, y: y + 80, w: 700, h: 400, xr: [-400, 400], yr: [0, 1.05], xs: 200, ys: 0.25, xd: 0, yd: 2, xl: 'Detuning (MHz)', yl: 'Transmission', px: 15, ylo: 62, bg: '#fbfcfd' });
    var QL = 1.5, fw = 193.4e6 / (QL * 1e6), lor = function (D) { return 1 - 0.75 / (1 + Math.pow(D / (fw / 2), 2)); };
    ftrace(m, function (D) { return lor(D) + 0.008 * Math.sin(D * 1.9) * Math.cos(D * 0.37); }, -400, 400, '#2a6fb0', 2.2, 500); ftrace(m, lor, -400, 400, '#b3261e', 2, 300, [7, 4]);
    var over = u > 0.35, fx = x + 860;
    txt('Coupling branch', fx, y + 100, 19, '#5b6673', 'left', 700);
    [['under-coupled', !over], ['over-coupled', over]].forEach(function (q, i) { var bx = fx + i * 230; rect(bx, y + 120, 220, 44, q[1] ? '#0b5cab' : '#dfe4ea', '#9aa4b0', 1); txt(q[0], bx + 110, y + 142, 18, q[1] ? '#fff' : '#18202a', 'center', 700); });
    var rows = [['Dip depth, T min', '0.25'], ['Loaded Q', QL.toFixed(2) + ' million'], ['Intrinsic Q', (over ? '6.00' : '2.00') + ' million']];
    rows.forEach(function (r, i) { var ry = y + 220 + i * 60; line([[fx, ry - 28], [x + w - 30, ry - 28]], '#d6dbe2', 1); txt(r[0], fx, ry, 19, '#5b6673', 'left', 600); txt(r[1], x + w - 30, ry, 22, i === 2 && over ? '#b3261e' : '#18202a', 'right', 700); });
    txt('Q0 = 2 QL / (1 ± √Tmin)', fx, y + 420, 18, '#5b6673', 'left', 500);
    if (u > 0.5) { rect(fx, y + 470, 460, 50, '#ffe58a', null); txt('Slide title: "Record intrinsic Q"', fx + 12, y + 495, 19, '#18202a', 'left', 700); }
    stamp(u, o.stamp, o.ok);
  };
  /* reflection check: circulator at the input; reflection peaks at every doublet, splitting pinned across resonances */
  FB.ringReflect = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    /* setup strip */
    var y = 150, c = S.ctx; benchBg(90, 220);
    rect(60, y - 40, 190, 80, '#1c2128', '#4b545e', 1.5); txt('Tunable laser', 155, y, 17, '#dfe5ea', 'center', 700);
    dot(420, y, 34, '#2b3036'); c.strokeStyle = '#9aa4ae'; c.lineWidth = 2; c.beginPath(); c.arc(420, y, 18, -2.4, 1.9); c.stroke(); arrowHead(420 + 18 * Math.cos(1.9), y + 18 * Math.sin(1.9), 1.9 + 1.57, 10, '#9aa4ae');
    txt('circulator', 420, y - 50, 15, P.ink2, 'center', 600);
    fiber([[250, y], [386, y]], 4); fiber([[454, y], [700, y]], 4); fiber([[420, y + 34], [420, y + 70], [560, y + 70]], 4);
    rect(560, y + 50, 150, 40, '#1c2128', '#4b545e', 1.5); txt('reflection', 635, y + 70, 15, '#ff9d96', 'center', 700);
    rect(700, y - 30, 300, 60, '#5c5f78', '#8f9ba7', 1.5); c.strokeStyle = P.wg; c.lineWidth = 2; c.beginPath(); c.arc(850, y - 4, 22, 0, 6.3); c.stroke(); line([[700, y + 22], [1000, y + 22]], P.wg, 2);
    fiber([[1000, y + 22], [1200, y + 22]], 4); rect(1200, y + 2, 160, 40, '#1c2128', '#4b545e', 1.5); txt('transmission', 1280, y + 22, 15, '#f2c94c', 'center', 700);
    /* four resonances across the band: doublet and reflection peak stay pinned together */
    var fsr = 1.92, k = seg(u, 0.05, 0.85);
    for (var i = 0; i < 4; i++) {
      var px = 60 + i * 380, al = clamp(k * 4 - i); if (al <= 0) continue;
      c.save(); c.globalAlpha = al;
      var m = plot({ x: px + 50, y: 290, w: 300, h: 300, xr: [-400, 400], yr: [0, 1.05], xs: 200, ys: 0.5, xd: 0, yd: 1, xl: 'MHz', px: 13, ylo: 40 });
      ftrace(m, function (D) { return dbl(D, { b: 90 }).T; }, -400, 400, P.tr1, 2.2); ftrace(m, function (D) { return dbl(D, { b: 90 }).R; }, -400, 400, P.tr3, 2.2);
      var walk = -240 + i * 160; ftrace(m, function (D) { return 1 - 0.55 / (1 + Math.pow((D - walk) / 40, 2)); }, -400, 400, 'rgba(200,210,220,0.45)', 1.6, 300, [5, 4]);
      txt((1550 + i * fsr).toFixed(2) + ' nm', px + 200, 276, 16, P.ink2, 'center', 600, true);
      c.restore();
    }
    var lx2 = 110; [['transmission', P.tr1], ['reflection', P.tr3], ['two independent mode families would drift apart', '#aab4be']].forEach(function (q) { lx2 += txt(q[0], lx2, 705, 17, q[1], 'left', 700) + 40; });
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Instruments that integrate: wavemeter, oscilloscope, on-chip source heating the ring under test
     (free spectral range 160 pm with the on-chip laser, 120 pm with a benchtop laser, 110 pm simulated:
     Chowdhury et al., Opt. Express 32, 3085, 2024).
     ===================================================================== */
  FB.steadyNumber = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var upd = Math.floor(u * 6);
    readout(60, 120, 520, 130, 'WAVEMETER, 1 UPDATE PER SECOND', (1550.392 + 0.0004 * Math.sin(upd * 2.1)).toFixed(3), 'nm', '#7fe8a2');
    rect(60, 262, 520, 30, '#0d1116', '#2f3a45', 1); for (var i = 0; i < 6; i++) rect(66 + i * 85, 268, 78, 18, i <= upd ? '#2f6f45' : '#1b232b', null);
    txt('updates', 70, 310, 15, P.ink3, 'left', 600, true);
    readout(60, 350, 520, 130, 'POWER METER, AVERAGED', '3.21', 'mW', '#f2c94c');
    scopePanel(620, 110, 940, 520, u * 3);
    stamp(u, o.stamp, o.ok);
  };
  FB.onChipHeat = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var c = S.ctx, v = 0.13, x0 = 90, y0 = 100, cw = 7200 * v, ch = 4600 * v;
    rect(0, 70, VW, 620, '#101318', null);
    chipSurf(x0, y0, cw, ch, 41, ['#5c5f78', '#4c4f66']);
    var lx = x0 + 60, ly = y0 + ch - 50, cx = x0 + 520, R = 1740 * v, cy = ly - R - 4;
    /* DFB die, ridge up, wire-bonded p-pad; photonic wire bond to the chip waveguide */
    goldRect(lx - 40, ly - 60, 30, 30); rect(lx, ly - 19, 60, 38, '#8a8e96', '#555', 1); rect(lx + 4, ly - 13, 52, 26, P.gold, null); line([[lx, ly], [lx + 60, ly]], '#3a2a08', 1.5);
    c.strokeStyle = '#e8d38c'; c.lineWidth = 1.5; c.beginPath(); c.moveTo(lx + 30, ly - 8); c.quadraticCurveTo(lx, ly - 70, lx - 25, ly - 45); c.stroke();
    c.strokeStyle = 'rgba(225,235,255,0.85)'; c.lineWidth = 2; c.beginPath(); c.moveTo(lx + 60, ly); c.quadraticCurveTo(lx + 78, ly - 14, lx + 96, ly); c.stroke();
    wg([[lx + 96, ly], [x0 + cw - 20, ly]], 2); wg(arcPts(cx, cy, R, 0, 6.2832, 160), 2); gcSmall(x0 + cw - 20, ly, 0, 0.8);
    /* heat spreading from the laser */
    var k = seg(u, 0.05, 0.7), g = c.createRadialGradient(lx + 30, ly, 4, lx + 30, ly, 80 + 520 * k);
    g.addColorStop(0, 'rgba(255,80,40,0.75)'); g.addColorStop(0.35, 'rgba(255,160,40,0.45)'); g.addColorStop(1, 'rgba(255,200,60,0)');
    c.fillStyle = g; c.fillRect(x0, y0, cw, ch);
    rect(x0, y0, cw, ch, null, 'rgba(210,205,240,0.55)', 2);
    scaleBar(x0 + cw - 20, y0 + ch - 20, 1000, v, '1 mm'); scopeTag(x0 + 10, y0 + ch - 44, 'Top view with thermal overlay');
    box(lx - 20, y0 + 16, ['On-chip laser'], { px: 18, anchor: [lx + 30, ly - 20] });
    /* sweep through three resonances: apparent spacing grows as the chip warms */
    var m = plot({ x: 1100, y: 120, w: 450, h: 260, xr: [0, 400], yr: [0, 1.05], xs: 100, ys: 0.5, xd: 0, yd: 1, xl: 'Detuning (pm)', px: 14, ylo: 44 });
    var dips = [40, 200, 360], des = [40, 150, 260];
    ftrace(m, function (x) { var t = 1; dips.forEach(function (d) { t -= 0.8 / (1 + Math.pow((x - d) / 6, 2)); }); return t; }, 0, 400 * seg(u, 0.1, 0.8), P.tr1, 2.2, 800);
    des.forEach(function (d) { line([[m.X(d), m.y], [m.X(d), m.y + m.h]], 'rgba(86,212,255,0.6)', 1.6, [5, 4]); });
    readout(1100, 505, 450, 86, 'FSR, ON-CHIP LASER', u > 0.75 ? '160' : '- - -', 'pm', '#ff9d96');
    readout(1100, 600, 450, 86, 'FSR, DESIGN', '110', 'pm', '#56d4ff');
    stamp(u, o.stamp, o.ok);
  };
  /* bench with a separate source: tunable laser, isolator (arrow away from the laser), chip, fast photodiode */
  FB.separateBench = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var y = 250; benchBg(110, 400);
    rect(40, y - 70, 250, 140, '#1c2128', '#4b545e', 2); txt('Tunable laser', 165, y - 40, 18, '#dfe5ea', 'center', 700); readout(54, y - 16, 222, 70, '', '1550.000', 'nm', '#7fe8a2');
    fcPair(300, y, 4); var iso = inlineIso(420, y, 5, 1, { label: true }); fiber([[376, y], iso[0]], 4); fiber([iso[1], [760, y]], 4);
    rect(760, y - 50, 240, 100, '#5c5f78', '#8f9ba7', 1.5); S.ctx.strokeStyle = P.wg; S.ctx.lineWidth = 2; S.ctx.beginPath(); S.ctx.arc(880, y - 14, 26, 0, 6.3); S.ctx.stroke(); line([[760, y + 14], [1000, y + 14]], P.wg, 2);
    fiber([[1000, y + 14], [1120, y + 14]], 4); rect(1120, y - 10, 90, 48, '#2b3036', '#4b545e', 1.5); txt('fast PD', 1165, y + 14, 15, '#dfe5ea', 'center', 700);
    line([[1210, y + 14], [1250, y + 14], [1250, 440]], '#111', 5);
    var fp = mkPath([[290, y], [760, y]]); pulse(fp, (u * 2.5 % 1) * 1.1, P.fwd, 0.1, 5);
    var bp = mkPath([[760, y], [iso[1][0], y]]), tb = (u * 2.5 + 0.5) % 1; if (tb < 0.9) pulse(bp, tb * 1.1, P.back, 0.12, 4, 0.8); else flash(iso[1][0], y, 24, 'rgba(255,90,80,0.8)', 1);
    scopePanel(1000, 440, 560, 250, 0, { name: 'OSCILLOSCOPE' });
    /* thermal camera: uniform chip */
    rect(60, 440, 420, 240, '#0d1116', '#2f3a45', 1.5); txt('THERMAL CAMERA', 72, 458, 14, P.ink3, 'left', 700, true);
    var c = S.ctx, g = c.createLinearGradient(90, 0, 450, 0); g.addColorStop(0, '#24407a'); g.addColorStop(1, '#2b4b86'); c.fillStyle = g; c.fillRect(90, 490, 360, 170);
    txt('uniform', 270, 575, 20, '#cfe0ff', 'center', 700);
    readout(540, 440, 420, 110, 'FSR, BENCHTOP LASER', '120', 'pm', '#7fe8a2'); readout(540, 570, 420, 110, 'FSR, DESIGN', '110', 'pm', '#56d4ff');
    stamp(u, o.stamp, o.ok);
  };
  /* self-injection locking slide: DFB laser on a submount butt-coupled to a SiN ring chip; backscatter returns */
  FB.silSlide = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {};
    var c = S.ctx; rect(100, 70, 1400, 640, '#eef1f5', '#9aa4b0', 1.5); txt('Policy update: backscatter is now mandatory', 130, 116, 32, '#18202a', 'left', 800);
    txt('Self-injection locking needs light to come back', 130, 156, 20, '#5b6673', 'left', 500);
    var y = 430, vx = 140, vw = 1320; rect(vx, 200, vw, 460, '#2a2d36', null);
    /* submount (AlN), laser die with ridge and p-pad, wire bonds */
    rect(180, y - 120, 330, 240, '#e6e2d8', '#b9b3a6', 1.5); goldRect(200, y - 100, 120, 70);
    rect(330, y - 60, 170, 120, '#80848c', '#55595f', 1.5); goldRect(338, y - 34, 154, 68); line([[338, y], [500, y]], '#3a2a08', 3);
    c.strokeStyle = '#e8d38c'; c.lineWidth = 2; for (var i = 0; i < 3; i++) { c.beginPath(); c.moveTo(370 + i * 40, y - 20); c.quadraticCurveTo(330 + i * 20, y - 110, 260 + i * 20, y - 70); c.stroke(); }
    txt('DFB laser', 345, y + 150, 18, '#cfd6de', 'center', 700);
    /* ring chip, input facet facing the laser front facet */
    chipSurf(510, 240, 920, 390, 51, ['#5c5f78', '#4c4f66']);
    var cx = 960, R = 100, cy = y - R - 6; wg(arcPts(cx, cy, R, 0, 6.2832, 120), 3); wg([[510, y], [1430, y]], 3);
    flash(500, y, 26, 'rgba(255,170,90,0.9)', 0.8);
    var t = u * 3, bus = mkPath([[500, y], [1430, y]]); pulse(bus, (t % 1) * 1.1, P.fwd, 0.08, 5);
    for (var k = 0; k < 2; k++) { var a = Math.PI / 2 - ((t * 0.9 + k / 2) % 1) * 2 * Math.PI; glow(arcPts(cx, cy, R, a + 0.35, a, 12), P.fwd, 5, 14); var b = -0.6 + ((t * 0.9 + k / 2) % 1) * 2 * Math.PI; glow(arcPts(cx, cy, R, b - 0.3, b, 12), P.back, 4, 14); }
    pulse(mkPath([[cx, y], [500, y]]), ((t * 1.2) % 1) * 1.1, P.back, 0.12, 5);
    if ((t * 1.2) % 1 > 0.85) flash(500, y, 40, 'rgba(255,90,80,0.9)', 0.9);
    box(560, 520, ['Backscatter from the ring returns into the laser and locks it'], { px: 19, border: P.ok });
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Asymmetric mode converter (reciprocal). Layout view, a one-mode measurement, and the reflection that retraces
     its path: fundamental (one lobe) to first-order (two lobes) forward, and back to fundamental on the way home.
     ===================================================================== */
  function convPoly(x, y, L, s) { /* 0.5 um single mode to 1.6 um, one edge straight, one flared: units s per um */
    return [[x, y - 0.25 * s], [x + L, y - 0.25 * s], [x + L, y + 1.35 * s], [x + L * 0.55, y + 0.95 * s], [x, y + 0.25 * s]];
  }
  function fillPoly(pts, fill, stroke) { var c = S.ctx; c.beginPath(); pts.forEach(function (p, i) { if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }); c.closePath(); if (fill) { c.fillStyle = fill; c.fill(); } if (stroke) { c.strokeStyle = stroke; c.lineWidth = 1.5; c.stroke(); } }
  function modeLobe(x, y, h, order, col, a) { /* transverse field magnitude plotted across the guide, vertical */
    var pts = []; for (var i = -1; i <= 1.0001; i += 0.05) { var f = order ? Math.abs(Math.sin(Math.PI * (i + 1) / 2 * 2)) : Math.cos(Math.PI * i / 2); pts.push([x - 50 * f, y + i * h / 2]); }
    S.ctx.save(); S.ctx.globalAlpha = a === undefined ? 1 : a; line(pts, col, 3); line([[x, y - h / 2], [x, y + h / 2]], col, 1); S.ctx.restore();
  }
  FB.modeConv = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {}; var v = o.variant || 'layout', c = S.ctx;
    if (v === 'layout') {
      rect(40, 70, 1520, 640, '#0b0d10', '#2f3a45', 1.5); rect(40, 70, 1520, 40, '#1b2128', null); txt('Layout editor', 56, 90, 17, '#c9d3dd', 'left', 600);
      for (var gx = 80; gx < 1240; gx += 40) for (var gy = 140; gy < 700; gy += 40) dot(gx, gy, 1.2, '#2a333d');
      var s = 120, y = 360; c.save(); var pts = [[60, y - 0.25 * s], [360, y - 0.25 * s]].concat(convPoly(360, y, 600, s).slice(1, 4).reverse().reverse(), [[360, y + 0.25 * s], [60, y + 0.25 * s]]);
      fillPoly([[60, y - 0.25 * s], [360, y - 0.25 * s], [360, y + 0.25 * s], [60, y + 0.25 * s]], 'rgba(70,140,255,0.35)', '#5aa0ff');
      fillPoly(convPoly(360, y, 600, s), 'rgba(70,140,255,0.35)', '#5aa0ff');
      fillPoly([[960, y - 0.25 * s], [1220, y - 0.25 * s], [1220, y + 1.35 * s], [960, y + 1.35 * s]], 'rgba(70,140,255,0.35)', '#5aa0ff'); c.restore();
      line([[360, y - 0.25 * s - 30], [960, y - 0.25 * s - 30]], '#9aa6b2', 1.5); txt('60 µm', 660, y - 0.25 * s - 48, 17, '#c8d0da', 'center', 600);
      txt('0.5 µm', 200, y + 0.25 * s + 26, 16, '#c8d0da', 'center', 600); txt('1.6 µm', 1090, y + 1.35 * s + 26, 16, '#c8d0da', 'center', 600);
      var name = 'ISOLATOR', n = Math.round(name.length * seg(u, 0.15, 0.6));
      rect(560, 560, 300, 54, 'rgba(0,0,0,0.6)', '#e3b04b', 1.5); txt('cell: ' + name.slice(0, n) + ((u * 6) % 1 < 0.5 && n < name.length ? '|' : ''), 576, 587, 24, '#e3b04b', 'left', 700, true);
      rect(1260, 130, 280, 560, '#12171d', '#2f3a45', 1); txt('Layers', 1276, 152, 16, P.ink2, 'left', 700);
      [['waveguide', '#5aa0ff'], ['slab etch', '#7fd18f'], ['metal', '#e3b04b'], ['cell labels', '#c9d3dd']].forEach(function (q, i) { rect(1276, 180 + i * 40, 22, 22, q[1], null); txt(q[0], 1310, 191 + i * 40, 17, P.ink, 'left', 500); });
    } else if (v === 'measure') {
      [['Forward', 0, 'fundamental in', 'first-order out', 'all-mode detector', 'high', P.ok], ['Backward', 1, 'fundamental in', 'other mode out', 'single-mode detector', 'low', P.bad]].forEach(function (r, i) {
        var y = 200 + i * 260, s = 60; rect(40, y - 110, 1520, 230, '#0d1116', '#2f3a45', 1.5); txt(r[0], 60, y - 88, 20, P.ink, 'left', 700);
        fillPoly([[200, y - 0.25 * s], [500, y - 0.25 * s], [500, y + 0.25 * s], [200, y + 0.25 * s]], '#3a4a6a', null); fillPoly(convPoly(500, y, 500, s), '#3a4a6a', null); fillPoly([[1000, y - 0.25 * s], [1200, y - 0.25 * s], [1200, y + 1.35 * s], [1000, y + 1.35 * s]], '#3a4a6a', null);
        var k = seg(u, 0.05, 0.6), xs = i ? lerp(1200, 300, k) : lerp(200, 1200, k), ord = i ? (xs < 600 ? 1 : 0) : (xs > 900 ? 1 : 0), ym = ord ? y + 0.55 * s : (xs < 600 ? y : y + 0.55 * s);
        modeLobe(xs, ym, ord || xs > 900 ? 1.6 * s : 0.5 * s * 2, ord, P.fwd);
        txt(r[2], i ? 1200 : 200, y + 100, 17, P.ink2, i ? 'right' : 'left', 600); txt(r[3], i ? 200 : 1200, y + 100, 17, P.ink2, i ? 'left' : 'right', 600);
        rect(1330, y - 60, 200, 120, '#05080a', '#2f3a45', 1.5); txt(r[4], 1430, y - 76, 14, P.ink2, 'center', 600);
        var bh = i ? 8 : 70, bk = seg(u, 0.6, 0.75); rect(1400, y + 16 - bh * bk, 60, bh * bk, r[6], null); txt(r[5], 1430, y + 36, 18, r[6], 'center', 700);
      });
    } else {
      /* replay: laser, converter, reflection at the far facet, then a magneto-optic isolator slides in */
      var y = 330, s = 70; benchBg(110, 560);
      rect(60, y - 70, 180, 140, '#1c2128', '#4b545e', 1.5); txt('laser', 150, y + 92, 17, P.ink2, 'center', 600);
      var iso = seg(u, 0.62, 0.78), ix = 320, gapw = 200 * iso;
      fillPoly([[240, y - 0.25 * s], [560, y - 0.25 * s], [560, y + 0.25 * s], [240, y + 0.25 * s]], '#3a4a6a', null);
      fillPoly(convPoly(560, y, 520, s), '#3a4a6a', null); fillPoly([[1080, y - 0.25 * s], [1420, y - 0.25 * s], [1420, y + 1.35 * s], [1080, y + 1.35 * s]], '#3a4a6a', null);
      line([[1420, y - 40], [1420, y + 130]], '#f0f2f6', 4); txt('far facet', 1420, y - 58, 17, P.ink2, 'center', 600);
      if (iso > 0) { c.save(); c.globalAlpha = iso; c.translate(0, (1 - iso) * -160);
        rect(ix - 20, y - 80, 220, 160, '#20242a', '#5a636e', 2); rect(ix - 10, y - 66, 26, 132, '#4c5560', null); metalRect(ix + 30, y - 66, 120, 132, '#3b4148', '#1d2126'); rect(ix + 164, y - 66, 26, 132, '#4c5560', null);
        txt('polarizer, Faraday rotator, polarizer at 45 degrees', ix + 90, y - 100, 16, '#cfd6de', 'center', 600); c.restore(); }
      var t = (u * 2) % 1, fw = t < 0.5, sp = fw ? lerp(250, 1410, t * 2) : lerp(1410, 250, (t - 0.5) * 2);
      var ordF = sp > 900 ? 1 : 0, yy = sp > 900 ? y + 0.55 * s : sp > 560 ? lerp(y, y + 0.55 * s, (sp - 560) / 340) : y;
      var blocked = iso > 0.95 && !fw && sp < ix + 200;
      if (!blocked) modeLobe(sp, yy, sp > 600 ? 1.6 * s : 0.5 * s * 2, ordF, fw ? P.fwd : P.back);
      if (!fw && !blocked && sp < 300) flash(240, y, 60, 'rgba(255,90,80,0.95)', 0.9);
      if (blocked) flash(ix + 200, y, 40, 'rgba(255,90,80,0.7)', 0.8);
      doRow(600, ['Reflection returns as', 'the mode it left in', P.ink2], ['Arrives at the laser as', iso > 0.95 ? 'blocked by a real isolator' : 'the fundamental mode', iso > 0.95 ? P.ok : P.bad]);
    }
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Outgoing inspection: swept optical spectrum analyzer (resolution bandwidth 0.06 nm), averaging power meter,
     a laser hopping between two adjacent modes 0.15 nm apart. A mode shows in a sweep only if the laser is in it
     when the sweep passes that wavelength.
     ===================================================================== */
  function osaTrace(seed, k, rbw) {
    var lA = 1550.00, lB = 1550.15, pts = [];
    for (var i = 0; i <= 600 * k; i++) {
      var l = 1549.7 + 0.6 * i / 600, tt = seed * 7.3 + i / 600 * 0.3, st = seed < 0 ? 0 : telegraphState(tt), lc = st ? lB : lA;
      var g = Math.exp(-Math.pow((l - lc) / (rbw / 1.665), 2)), sm = Math.exp(-Math.pow((l - (st ? lA : lB)) / (rbw / 1.665), 2));
      pts.push([l, Math.max(-75 + 1.5 * Math.sin(i * 1.7 + seed), 10 * Math.log10(1e-9 + g * 2.0 + sm * 2e-5))]);
    }
    return pts;
  }
  function osaPanel(x, y, w, h, seed, k, rbw, o) {
    o = o || {}; var sc = inst(x, y, w, h, o.name || 'OPTICAL SPECTRUM ANALYZER  RBW ' + rbw.toFixed(2) + ' nm');
    var m = plot({ x: sc.x + 60, y: sc.y + 6, w: sc.w - 72, h: sc.h - (o.xl === false ? Math.max(30, S.mf * 1.4) : Math.max(50, S.mf * 3.3)), xr: [1549.7, 1550.3], yr: [-80, 10], xs: 0.1, ys: 20, xd: 1, yd: 0, xl: o.xl === false ? null : 'Wavelength (nm)', yl: 'dBm', px: 13, ylo: 46, noXt: o.noXt });
    ptrace(m, osaTrace(seed, k, rbw), o.col || P.tr1, 1.8);
    if (o.seed2 !== undefined) ptrace(m, osaTrace(o.seed2, k, rbw), P.tr2, 1.6);
    if (k < 1) { var X = m.X(1549.7 + 0.6 * k); line([[X, m.y], [X, m.y + m.h]], 'rgba(255,255,255,0.35)', 1.5); }
    return m;
  }
  FB.inspect = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {}; var v = o.variant || 'station', c = S.ctx;
    if (v === 'station') {
      osaPanel(40, 110, 900, 420, -1, seg(u, 0.05, 0.9), 0.06);
      readout(980, 120, 580, 120, 'POWER METER, AVERAGING 1 s', '3.20', 'mW', '#f2c94c');
      readout(980, 260, 580, 120, 'WAVEMETER', '1550.000', 'nm', '#7fe8a2');
      /* tray of butterfly lasers in foam */
      rect(40, 560, 1520, 160, '#1d2024', '#3a4048', 1.5);
      for (var i = 0; i < 7; i++) { rect(70 + i * 210, 585, 180, 110, '#2a2e33', null); butterfly(160 + i * 210, 640, 3, { noMount: true }); }
    } else if (v === 'split') {
      scopePanel(40, 110, 740, 420, u, { name: 'FAST PHOTODIODE (NOT IN THE PROCEDURE)' });
      osaPanel(820, 110, 740, 420, 5, seg(u, 0.05, 0.9), 0.06);
      readout(820, 560, 740, 120, 'POWER METER, AVERAGING 1 s', '3.20', 'mW', '#f2c94c');
      readout(40, 560, 740, 120, 'WAVEMETER', '1550.0' + (telegraphState(u * 30) ? '75' : '75'), 'nm', '#7fe8a2');
    } else if (v === 'runs') {
      for (var r = 0; r < 9; r++) { var px = 40 + (r % 3) * 330, py = 110 + Math.floor(r / 3) * 205, al = clamp(seg(u, 0, 0.5) * 9 - r); if (al <= 0) continue; c.save(); c.globalAlpha = r < 8 && u > 0.6 ? 0.35 : al; osaPanel(px, py, 320, 195, 11 + r * 3.1, 1, 0.06, { name: 'RUN ' + (r + 1), xl: false, noXt: true }); c.restore(); }
      rect(1050, 110, 510, 560, '#eef1f5', '#9aa4b0', 1.5); rect(1050, 110, 510, 46, '#27303a', null); txt('Inspection sheet', 1068, 133, 18, '#c9d3dd', 'left', 600);
      [['Laser', 'serial 0412'], ['Runs recorded', 'run 9 of 9'], ['Runs 1 to 8', 'not attached'], ['Second peaks', 'analyzer artifacts'], ['Result', 'PASS']].forEach(function (q, i) { var yy = 200 + i * 62; txt(q[0], 1068, yy, 19, '#5b6673', 'left', 600); txt(q[1], 1300, yy, 21, i === 4 ? '#1f7a3a' : i === 2 || i === 3 ? '#b3261e' : '#18202a', 'left', 700); });
    } else if (v === 'customer') {
      var m = plot({ x: 150, y: 120, w: 1000, h: 500, xr: [0, 60], yr: [-5, 25], xs: 10, ys: 5, xd: 0, yd: 0, xl: 'Time (min)', yl: 'Frequency offset (GHz)', ylo: 60 });
      var pts = [], tEnd = 60 * seg(u, 0, 0.85); for (var t = 0; t <= tEnd; t += 0.02) { var st = telegraphState(t * 0.08 + 2); var fv = st ? 18.7 : 0; if (pts.length && pts[pts.length - 1][1] !== fv) pts.push([t, pts[pts.length - 1][1]]); pts.push([t, fv + 0.3 * Math.sin(t * 13)]); }
      ptrace(m, pts, P.tr2, 2);
      readout(1190, 120, 370, 96, 'COHERENT RECEIVER', 'HOPS', '', '#ff9d96');
      box(1190, 240, ['Jumps of 18.7 GHz:', '0.15 nm at 1550 nm'], { px: 19 });
    } else {
      scopePanel(40, 110, 740, 330, u, { name: 'FAST PHOTODIODE' });
      osaPanel(820, 110, 740, 330, 5, 1, 0.02, { name: 'OSA  RBW 0.02 nm  SWEEPS UP AND DOWN', seed2: 8 });
      rect(40, 470, 1520, 230, '#0d1116', '#2f3a45', 1.5); txt('RUNS KEPT', 56, 490, 15, P.ink3, 'left', 700, true);
      for (var j = 0; j < 9; j++) { var bx = 60 + j * 166, two = j % 3 !== 1; rect(bx, 512, 150, 120, '#05080a', two ? P.bad : '#2f3a45', 1.5); txt('run ' + (j + 1), bx + 75, 532, 15, P.ink2, 'center', 600); line([[bx + 50, 620], [bx + 50, 560]], P.tr1, 3); if (two) line([[bx + 100, 620], [bx + 100, 575]], P.tr1, 3); }
      txt('Hops in 6 of 9 runs: laser rejected', 60, 670, 21, P.bad, 'left', 700);
    }
    stamp(u, o.stamp, o.ok);
  };

  /* =====================================================================
     Gary's bench, top view: butterfly laser, patch cords from the drawer, an inline isolator, lensed fibers on a chip
     stage, the analyzer input, a shipping box. o.stage: 'pan' | 'tags' | 'trade' | 'arcs'
     ===================================================================== */
  var BT = { y: 330, R: [{ x: 300 }, { x: 470 }, { x: 640 }, { x: 990 }, { x: 1000 }, { x: 1335 }],
    tags: [[1060, 280, 'Intracavity', 1000, 150], [470, 320, 'External', 420, 150], [1335, 410, 'Instrument-Originated', 1420, 150], [650, 320, 'Third-Party', 700, 200], [530, 580, 'Intra-Group', 530, 640], [1030, 580, 'Cross-Institutional', 1030, 640]] };
  BT.R[3].x = 985; BT.R[4].x = 1002;
  function benchScene(u, o, pan) {
    var c = S.ctx, y = BT.y; c.save(); c.translate(pan || 0, 0);
    benchBg(90, 600);
    butterfly(140, y, 5); fiber([[295, y], [300, y]], 4); fcPair(320, y, 4); fiber([[396, y], [450, y]], 6);
    fcPair(470, y, 4); fiber([[546, y], [560, y]], 4); inlineIso(580, y, 4, 1); fiber([[736, y], [800, y]], 4);
    /* chip on a stage with two lensed fibers */
    rect(820, y - 110, 360, 220, '#2c3138', '#4b545e', 1.5); chipSurf(1000 - 0, y - 60, 120, 120, 61, ['#5c5f78', '#4c4f66']); wg([[1000, y], [1120, y]], 2.5);
    fiber([[800, y], [840, y]], 4); line([[840, y], [990, y]], '#cfd8df', 4); fiber([[1180, y], [1220, y]], 4); line([[1130, y], [1180, y]], '#cfd8df', 4);
    /* analyzer */
    rect(1270, y - 120, 300, 240, '#1c2128', '#4b545e', 2); rect(1290, y - 100, 200, 130, '#05080a', null); txt('Spectrum analyzer', 1420, y + 70, 16, '#dfe5ea', 'center', 600);
    fiber([[1220, y], [1240, y], [1240, y + 90], [1320, y + 90]], 4); fcOpen(1335, y + 90, 3);
    /* drawer of patch cords and a shipping box */
    rect(380, 470, 300, 110, '#3a3f46', '#5a616a', 1.5); for (var i = 0; i < 6; i++) { c.strokeStyle = P.jacket; c.lineWidth = 3; c.beginPath(); c.arc(430 + i * 40, 525, 22, 0, 6.3); c.stroke(); } txt('drawer', 530, 595, 14, P.ink2, 'center', 600);
    rect(900, 470, 260, 110, '#b08a58', '#7c5f3a', 1.5); rect(930, 490, 140, 50, '#f3efe6', null); txt('shipping box', 1030, 595, 14, P.ink2, 'center', 600);
    c.restore();
  }
  FB.benchTop = function (ctx, W, H, u, o) {
    begin(ctx, W, H); o = o || {}; var st = o.stage || 'pan', c = S.ctx, y = BT.y;
    var pan = st === 'pan' ? lerp(60, -60, ease(u)) : 0;
    benchScene(u, o, pan);
    var t = (u * 2) % 1, path = mkPath([[300, y], [1240, y]]);
    pulse(path, t * 1.1, P.fwd, 0.06, 5);
    if (st === 'tags') BT.tags.forEach(function (r, i) { var a = clamp(u * 7 - i * 0.9); if (a <= 0) return; var dy = (1 - ease(a)) * -80, sw = Math.sin(u * 12 + i) * 4, tx = r[3] + sw, ty = r[4] + dy; c.save(); c.globalAlpha = a; line([[r[0], r[1]], [tx, ty + 19]], '#c8d0da', 1.5); dot(r[0], r[1], 4, '#c8d0da'); var tw2 = tw(r[2], 18, 700) + 20; tx = clamp(tx, tw2 / 2 + 6, VW - tw2 / 2 - 6); rect(tx - tw2 / 2, ty, tw2, 38, '#f3efe6', '#7c5f3a', 1.5); txt(r[2], tx, ty + 19, 18, '#18202a', 'center', 700); c.restore(); });
    if (st === 'arcs' || st === 'pan') BT.R.forEach(function (r, i) { var a = st === 'arcs' ? 0.5 + 0.5 * Math.sin(u * 20 - i) : 0.25; flash(r.x, y, 34, 'rgba(255,90,80,0.9)', a); });
    if (st === 'arcs') {
      var n = 0, k = seg(u, 0.05, 0.6) * 15;
      for (var i = 0; i < 6; i++) for (var j = i + 1; j < 6; j++) { if (n++ >= k) continue; var a0 = BT.R[i].x, a1 = BT.R[j].x, mid = (a0 + a1) / 2, rr = (a1 - a0) / 2; c.strokeStyle = 'rgba(255,120,100,0.55)'; c.lineWidth = 2; c.beginPath(); c.ellipse(mid, y - 20, rr, Math.min(170, rr * 0.55), 0, Math.PI, 2 * Math.PI); c.stroke(); }
      butterfly(140, y, 5, { glow: 'rgba(255,90,80,0.95)', glowA: 0.6 + 0.4 * Math.sin(u * 40) });
      readout(1190, 620, 370, 90, 'CAVITIES', String(Math.min(15, Math.floor(k))), '', '#ff9d96');
    }
    if (st === 'trade') {
      rect(0, 90, VW, 510, 'rgba(8,11,15,0.65)', null);
      rect(260, 160, 460, 330, '#0d1116', '#2f3a45', 1.5); inlineIso(370, 300, 6, 1, { label: true }); txt('Isolator', 490, 200, 22, P.ink, 'center', 700); txt('Keeps reflections out of the laser', 490, 400, 18, P.ok, 'center', 600);
      rect(880, 160, 460, 330, '#eef1f5', '#9aa4b0', 1.5); txt('Link budget', 900, 200, 22, '#18202a', 'left', 700); txt('Isolator', 900, 260, 20, '#5b6673', 'left', 600); txt('-0.5 dB', 1320, 260, 20, '#18202a', 'right', 700, true);
      line([[900, 260], [1320, 260]], '#b3261e', 3); txt('Recovered', 900, 320, 20, '#5b6673', 'left', 600); txt('+0.5 dB', 1320, 320, 20, '#1f7a3a', 'right', 700, true); txt('Isolation', 900, 380, 20, '#5b6673', 'left', 600); txt('lost', 1320, 380, 20, '#b3261e', 'right', 700);
      var tg = 'Quid Pro Quo', tw3 = tw(tg, 26, 800) + 30, dy = (1 - seg(u, 0.1, 0.35)) * -200; rect(800 - tw3 / 2, 520 + dy, tw3, 50, '#f3efe6', '#7c5f3a', 2); txt(tg, 800, 545 + dy, 26, '#18202a', 'center', 800);
      txt('⇄', 800, 320, 60, '#c8d0da', 'center', 700);
    }
    stamp(u, o.stamp, o.ok);
  };

  root.LMS_DRAW.fb = FB;
})();
