/* Rods incident, cross-section keyframe: computed RF field lines and optical modes of an x-cut thin-film LN rib in
   one gap of a G-S-G line (data: anims/fe_rods_fields_data.js, generator tools/fields/rods_fields.py).
   window.LMS_DRAW.fe_rodsFields(ctx, W, H, u, opts): W, H in device pixels, u in 0..1 within the keyframe.
   Reads window.LMS_DRAW_DATA.rodsFields at draw time (load order does not matter). No DOM access at load time. */
(function () {
  'use strict';
  var root = typeof window !== 'undefined' ? window : this;
  var D = (root.LMS_DRAW = root.LMS_DRAW || {});
  var TAU = 2 * Math.PI;
  var SANS = 'ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
  var C = {
    bg: '#0e1217', air: '#11161c', ox: '#1f2a35', oxHi: '#283643', ln: '#4b4058', lnEdge: '#8d7fa0',
    au: '#efcb72', auHi: '#fde7a6', auLo: '#b98d3a', auEdge: '#7c5d22',
    rf: '#ffc46b', tm: '#ff6b6b', te: '#63e6ff', ink: '#e9eef3', dim: '#9aa6b2', panel: 'rgba(10,14,19,0.86)'
  };

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function seg(u, a, b) { return clamp((u - a) / (b - a), 0, 1); }
  function ease(t) { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function fpx(U, n) { return Math.round(Math.max(n * U.s, 8.5 * U.dpr)); }
  function font(ctx, px, w) { ctx.font = (w ? w + ' ' : '') + px + 'px ' + SANS; }

  function arrow(ctx, x0, y0, x1, y1, col, lw, hs) {
    var a = Math.atan2(y1 - y0, x1 - x0);
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * hs * 0.6, y1 - Math.sin(a) * hs * 0.6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hs * Math.cos(a - 0.42), y1 - hs * Math.sin(a - 0.42));
    ctx.lineTo(x1 - hs * Math.cos(a + 0.42), y1 - hs * Math.sin(a + 0.42));
    ctx.closePath(); ctx.fill();
  }
  function dbl(ctx, cx, cy, dx, dy, col, lw, hs) { arrow(ctx, cx, cy, cx + dx, cy + dy, col, lw, hs); arrow(ctx, cx, cy, cx - dx, cy - dy, col, lw, hs); }

  // boxed label; o: { col, border, to: [x, y], align: 'l'|'c'|'r', size, alpha, lines (array) }
  function label(ctx, U, x, y, text, o) {
    o = o || {};
    var a = o.alpha == null ? 1 : o.alpha;
    if (a <= 0.01) return null;
    ctx.save(); ctx.globalAlpha *= a;
    var fs = fpx(U, o.size || 12), pad = Math.round(fs * 0.45);
    var lines = String(text).split('\n'), tw = 0;
    font(ctx, fs, o.bold ? '600' : '');
    lines.forEach(function (l) { tw = Math.max(tw, ctx.measureText(l).width); });
    var bw = tw + 2 * pad, bh = lines.length * fs * 1.25 + 2 * pad - fs * 0.25;
    var bx = o.align === 'r' ? x - bw : o.align === 'c' ? x - bw / 2 : x;
    var by = o.valign === 'b' ? y - bh : y;
    bx = clamp(bx, 2, U.W - bw - 2); by = clamp(by, 2, U.H - bh - 2);
    if (o.to) {
      var tx = o.to[0], ty = o.to[1], ax = clamp(tx, bx, bx + bw), ay = clamp(ty, by, by + bh);
      ctx.strokeStyle = 'rgba(233,238,243,0.7)'; ctx.lineWidth = Math.max(1, 1.1 * U.s);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.fillStyle = 'rgba(233,238,243,0.85)'; ctx.beginPath(); ctx.arc(tx, ty, Math.max(1.5, 2.2 * U.s), 0, TAU); ctx.fill();
    }
    ctx.fillStyle = o.bg || C.panel; ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = o.border || 'rgba(233,238,243,0.45)'; ctx.lineWidth = Math.max(1, U.s);
    ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
    ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    lines.forEach(function (l, i) {
      font(ctx, fs, (o.bold && i === 0) || o.boldAll ? '600' : '');
      ctx.fillStyle = (o.cols && o.cols[i]) || o.col || C.ink;
      ctx.fillText(l, bx + pad, by + pad + i * fs * 1.25);
    });
    ctx.restore();
    return { x: bx, y: by, w: bw, h: bh };
  }

  /* ---------- data decoding (once) ---------- */
  var cache = null;
  function b64bytes(s) {
    var bin = root.atob(s), n = bin.length, a = new Uint8Array(n);
    for (var i = 0; i < n; i++) a[i] = bin.charCodeAt(i);
    return a;
  }
  function mapCanvas(nx, ny, bytes, rgbFn) {
    var cv = root.document.createElement('canvas'); cv.width = nx; cv.height = ny;
    var g = cv.getContext('2d'), im = g.createImageData(nx, ny), d = im.data;
    for (var i = 0; i < nx * ny; i++) {
      var c = rgbFn(bytes[i] / 255);
      d[4 * i] = c[0]; d[4 * i + 1] = c[1]; d[4 * i + 2] = c[2]; d[4 * i + 3] = c[3];
    }
    g.putImageData(im, 0, 0);
    return cv;
  }
  function prepare(F) {
    if (cache && cache.F === F) return cache;
    var lines = F.lines.map(function (l) {
      var pts = [], cum = [0];
      for (var i = 0; i < l.length; i += 2) pts.push([l[i], l[i + 1]]);
      for (var j = 1; j < pts.length; j++) cum.push(cum[j - 1] + Math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1]));
      return { pts: pts, cum: cum, len: cum[cum.length - 1] };
    });
    var emap = mapCanvas(F.emap.nx, F.emap.ny, b64bytes(F.emap.b64), function (q) {
      // log |E|: dark to amber, alpha rising with field
      var a = Math.pow(q, 2.2);
      return [255, Math.round(150 + 60 * q), Math.round(60 + 40 * q), Math.round(255 * 0.34 * a)];
    });
    function modeMap(m, rgb) {
      return mapCanvas(m.nx, m.ny, b64bytes(m.b64), function (v) {
        var I = v * v; // stored as |E| / max |E|
        var a = Math.pow(I, 0.8);
        return [rgb[0], Math.round(rgb[1] + (255 - rgb[1]) * I * I * 0.7), Math.round(rgb[2] + (255 - rgb[2]) * I * I * 0.7), Math.round(255 * 0.92 * a)];
      });
    }
    cache = { F: F, lines: lines, emap: emap, tm: modeMap(F.modes.TM, [255, 80, 80]), te: modeMap(F.modes.TE, [70, 220, 255]) };
    return cache;
  }

  /* ---------- the frame ---------- */
  D.fe_rodsFields = function (ctx, W, H, u, opts) {
    opts = opts || {};
    var DD = root.LMS_DRAW_DATA, F = DD && DD.rodsFields;
    var U = { W: W, H: H, s: W / 900, dpr: root.devicePixelRatio || 1 };
    var narrow = W / U.dpr < 560; // phone: captions cover the lower half
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    if (!F) return;
    var K = prepare(F), G = F.geom, v = F.view, N = F.numbers;
    var vw = v[1] - v[0], vh = v[3] - v[2];
    // panel: full width, top 82 % of the height (the bottom band stays clear for captions)
    var ax0 = W * 0.02, ay0 = H * 0.03, aw = W * 0.96, ah = H * 0.79;
    var k = Math.min(aw / vw, ah / vh);
    var ox = ax0 + (aw - vw * k) / 2, oy = ay0 + (ah - vh * k) / 2;
    function X(x) { return ox + (x - v[0]) * k; }
    function Y(y) { return oy + (v[3] - y) * k; }
    var px0 = X(v[0]), py0 = Y(v[3]), pw = vw * k, ph = vh * k;

    ctx.save();
    ctx.beginPath(); ctx.rect(px0, py0, pw, ph); ctx.clip();
    // materials
    ctx.fillStyle = C.air; ctx.fillRect(px0, py0, pw, ph);
    var gox = ctx.createLinearGradient(0, Y(G.box_top), 0, Y(v[2]));
    gox.addColorStop(0, C.oxHi); gox.addColorStop(1, C.ox);
    ctx.fillStyle = gox; ctx.fillRect(px0, Y(G.box_top), pw, Y(v[2]) - Y(G.box_top) + 1);
    ctx.fillStyle = C.ln;
    ctx.fillRect(px0, Y(G.slab_top), pw, Y(G.box_top) - Y(G.slab_top));
    ctx.beginPath(); G.rib.forEach(function (p, i) { if (i) ctx.lineTo(X(p[0]), Y(p[1])); else ctx.moveTo(X(p[0]), Y(p[1])); }); ctx.closePath(); ctx.fill();
    // |E_RF| shading
    var eA = ease(seg(u, 0.05, 0.45));
    if (eA > 0) {
      ctx.save(); ctx.globalAlpha = eA; ctx.imageSmoothingEnabled = true;
      ctx.drawImage(K.emap, X(F.emap.extent[0]), Y(F.emap.extent[3]), (F.emap.extent[1] - F.emap.extent[0]) * k, (F.emap.extent[3] - F.emap.extent[2]) * k);
      ctx.restore();
    }
    // LN outline (film surface and rib)
    ctx.strokeStyle = C.lnEdge; ctx.lineWidth = Math.max(1, 1.2 * U.s);
    ctx.beginPath();
    ctx.moveTo(px0, Y(G.slab_top)); ctx.lineTo(X(G.rib[0][0]), Y(G.slab_top));
    ctx.lineTo(X(G.rib[3][0]), Y(G.rib[3][1])); ctx.lineTo(X(G.rib[2][0]), Y(G.rib[2][1])); ctx.lineTo(X(G.rib[1][0]), Y(G.rib[1][1]));
    ctx.lineTo(px0 + pw, Y(G.slab_top)); ctx.stroke();
    ctx.strokeStyle = 'rgba(160,180,200,0.35)'; ctx.beginPath(); ctx.moveTo(px0, Y(G.box_top)); ctx.lineTo(px0 + pw, Y(G.box_top)); ctx.stroke();

    // optical modes: TM in the rib (main), TE as an inset later
    var me = F.modes.extent;
    var tmA = ease(seg(u, 0.28, 0.55));
    if (tmA > 0) {
      ctx.save(); ctx.globalAlpha = tmA; ctx.globalCompositeOperation = 'lighter'; ctx.imageSmoothingEnabled = true;
      ctx.drawImage(K.tm, X(me[0]), Y(me[3]), (me[1] - me[0]) * k, (me[3] - me[2]) * k);
      ctx.restore();
    }

    // RF field lines, drawn in from the signal electrode
    var p = seg(u, 0.0, 0.42), nL = K.lines.length;
    var lw = Math.max(1, 1.3 * U.s);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    K.lines.forEach(function (L, i) {
      var f = clamp(p * 1.7 - 0.7 * (i % 7) / 7, 0, 1);
      if (f <= 0) return;
      var lim = L.len * ease(f);
      ctx.strokeStyle = 'rgba(255,196,107,0.82)'; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(X(L.pts[0][0]), Y(L.pts[0][1]));
      var last = L.pts[0];
      for (var j = 1; j < L.pts.length; j++) {
        if (L.cum[j] <= lim) { ctx.lineTo(X(L.pts[j][0]), Y(L.pts[j][1])); last = L.pts[j]; continue; }
        var t = (lim - L.cum[j - 1]) / Math.max(1e-9, L.cum[j] - L.cum[j - 1]);
        last = [L.pts[j - 1][0] + t * (L.pts[j][0] - L.pts[j - 1][0]), L.pts[j - 1][1] + t * (L.pts[j][1] - L.pts[j - 1][1])];
        ctx.lineTo(X(last[0]), Y(last[1]));
        break;
      }
      ctx.stroke();
      // direction chevron at half length, once drawn past it
      var half = L.len * 0.5;
      if (lim > half && L.len > 1.0) {
        for (var m = 1; m < L.pts.length; m++) if (L.cum[m] >= half) break;
        var a = L.pts[m - 1], b = L.pts[Math.min(m, L.pts.length - 1)];
        var tt = (half - L.cum[m - 1]) / Math.max(1e-9, L.cum[m] - L.cum[m - 1]);
        var cx = X(a[0] + tt * (b[0] - a[0])), cy = Y(a[1] + tt * (b[1] - a[1]));
        var ang = Math.atan2(-(b[1] - a[1]), b[0] - a[0]), hs = Math.max(3, 5 * U.s);
        ctx.fillStyle = 'rgba(255,196,107,0.95)';
        ctx.beginPath(); ctx.moveTo(cx + hs * Math.cos(ang), cy + hs * Math.sin(ang));
        ctx.lineTo(cx + hs * Math.cos(ang + 2.5), cy + hs * Math.sin(ang + 2.5));
        ctx.lineTo(cx + hs * Math.cos(ang - 2.5), cy + hs * Math.sin(ang - 2.5)); ctx.closePath(); ctx.fill();
      }
    });

    // electrodes (on top of the field lines, which end on their surfaces)
    function gold(r) {
      var x0 = X(Math.max(r[0], v[0] - 1)), x1 = X(Math.min(r[1], v[1] + 1)), y0 = Y(r[3]), y1 = Y(r[2]);
      var gg = ctx.createLinearGradient(0, y0, 0, y1); gg.addColorStop(0, C.auHi); gg.addColorStop(0.35, C.au); gg.addColorStop(1, C.auLo);
      ctx.fillStyle = gg; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = C.auEdge; ctx.lineWidth = Math.max(1, U.s); ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
    }
    gold(G.signal); gold(G.ground);
    var fs = fpx(U, 12);
    font(ctx, fs, '700'); ctx.textBaseline = 'middle'; ctx.fillStyle = '#3a2a08';
    var ey = Y((G.signal[2] + G.signal[3]) / 2);
    ctx.textAlign = 'right'; ctx.fillText('signal', X(G.signal[1]) - 8 * U.s, ey);
    ctx.textAlign = 'left'; ctx.fillText('ground', X(G.ground[0]) + 8 * U.s, ey);
    // material names
    font(ctx, fpx(U, 11), '600'); ctx.fillStyle = 'rgba(233,238,243,0.55)'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('LiNbO₃, x-cut', X(v[0]) + 8 * U.s, Y(G.slab_top / 2));
    ctx.fillText('SiO₂', X(v[0]) + 8 * U.s, Y(-0.55));
    if (!narrow) ctx.fillText('air', X(v[0]) + 8 * U.s, Y(G.signal[3]) - 14 * U.s);
    ctx.restore();

    // polarization arrows: TM light along x (vertical), RF along z (horizontal, in the gap)
    var pk = F.modes.TM.peak, mx = X(pk[0]), my = Y(pk[1]);
    var aL = 0.42 * k, alw = Math.max(2, 3 * U.s), ahs = Math.max(6, 9 * U.s);
    if (tmA > 0) {
      ctx.save(); ctx.globalAlpha = ease(seg(u, 0.4, 0.6));
      ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 4 * U.s;
      dbl(ctx, mx, my, 0, -aL, '#ffe1de', alw, ahs);
      ctx.restore();
    }
    var rfA = ease(seg(u, 0.35, 0.55));
    if (rfA > 0) {
      ctx.save(); ctx.globalAlpha = rfA; ctx.shadowColor = 'rgba(0,0,0,0.8)'; ctx.shadowBlur = 4 * U.s;
      var gy = Y(0.95);
      arrow(ctx, X(G.signal[1]) + 0.35 * k, gy, X(G.ground[0]) - 0.35 * k, gy, C.rf, alw, ahs);
      ctx.restore();
    }

    // labels
    var lA = ease(seg(u, 0.5, 0.66));
    label(ctx, U, X(0), Y(G.signal[3]) - 0.18 * k, 'RF field: along z', { col: '#ffe3bd', border: C.rf, align: 'c', valign: 'b', alpha: lA, size: 12, to: [X(0), Y(0.95) - 2 * U.s] });
    // TM label: below the ground electrode; on phones above the signal electrode (captions cover the lower half)
    if (narrow) label(ctx, U, px0 + 4 * U.dpr, Y(G.signal[3]) - 0.18 * k, 'TM light: along x', { col: '#ffd6d2', border: C.tm, valign: 'b', alpha: lA, size: 12, to: [mx - 0.1 * k, my - 0.12 * k] });
    else label(ctx, U, X(G.ground[0]) - 0.6 * k, Y(-0.12), 'TM light: along x', { col: '#ffd6d2', border: C.tm, alpha: lA, size: 12, to: [mx + 0.14 * k, my + 0.04 * k] });
    // axes glyph (top left): z right, x up, y out of the page
    var gA = ease(seg(u, 0.5, 0.66));
    if (gA > 0) {
      ctx.save(); ctx.globalAlpha = gA;
      var gx0 = px0 + Math.max(16 * U.s, 12 * U.dpr), gy0 = py0 + Math.max(46 * U.s, 34 * U.dpr), gl = Math.max(28 * U.s, 18 * U.dpr), gw = Math.max(1.5, 2 * U.s), gh = Math.max(5, 7 * U.s);
      arrow(ctx, gx0, gy0, gx0 + gl, gy0, C.rf, gw, gh); arrow(ctx, gx0, gy0, gx0, gy0 - gl, '#ffffff', gw, gh);
      ctx.strokeStyle = '#9fe3ff'; ctx.lineWidth = Math.max(1, 1.4 * U.s); ctx.beginPath(); ctx.arc(gx0, gy0, 4.5 * U.s, 0, TAU); ctx.stroke();
      ctx.fillStyle = '#9fe3ff'; ctx.beginPath(); ctx.arc(gx0, gy0, 1.5 * U.s, 0, TAU); ctx.fill();
      font(ctx, fpx(U, 11), '700'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = C.rf; ctx.fillText('z', gx0 + gl + 9 * U.s, gy0);
      ctx.fillStyle = '#ffffff'; ctx.fillText('x', gx0, gy0 - gl - 9 * U.s);
      ctx.fillStyle = '#9fe3ff'; ctx.fillText('y', gx0 - 11 * U.s, gy0 + 10 * U.s);
      ctx.restore();
    }

    // TE inset (comparison), lower right over the oxide
    var iA = narrow ? 0 : ease(seg(u, 0.68, 0.85));
    if (iA > 0) {
      ctx.save(); ctx.globalAlpha = iA;
      var iw = Math.max(pw * 0.2, 118 * U.s), ih = iw * (me[3] - me[2]) / (me[1] - me[0]) * 0.78;
      var ix = px0 + pw - iw - 10 * U.s, iy = py0 + ph - ih - 10 * U.s;
      ctx.fillStyle = 'rgba(8,11,15,0.9)'; ctx.fillRect(ix, iy, iw, ih);
      ctx.save(); ctx.beginPath(); ctx.rect(ix, iy, iw, ih); ctx.clip();
      var ik = iw / (me[1] - me[0]), iy0 = iy + ih * 0.5 + 0.3 * ik;
      var IX = function (x) { return ix + (x - me[0]) * ik; }, IY = function (y) { return iy0 - y * ik; };
      ctx.fillStyle = C.ln; ctx.fillRect(ix, IY(G.slab_top), iw, IY(0) - IY(G.slab_top));
      ctx.beginPath(); G.rib.forEach(function (q, i) { if (i) ctx.lineTo(IX(q[0]), IY(q[1])); else ctx.moveTo(IX(q[0]), IY(q[1])); }); ctx.closePath(); ctx.fill();
      ctx.fillStyle = C.ox; ctx.fillRect(ix, IY(0), iw, ih);
      ctx.globalCompositeOperation = 'lighter';
      ctx.drawImage(K.te, IX(me[0]), IY(me[3]), (me[1] - me[0]) * ik, (me[3] - me[2]) * ik);
      ctx.globalCompositeOperation = 'source-over';
      var tp = F.modes.TE.peak;
      dbl(ctx, IX(tp[0]), IY(tp[1]), 0.45 * ik, 0, '#dff8ff', Math.max(1.5, 2.2 * U.s), Math.max(5, 7 * U.s));
      ctx.restore();
      ctx.strokeStyle = C.te; ctx.lineWidth = Math.max(1, U.s); ctx.strokeRect(ix + 0.5, iy + 0.5, iw - 1, ih - 1);
      font(ctx, fpx(U, 11), '600'); ctx.fillStyle = '#d6f7ff'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('TE light: along z', ix + 6 * U.s, iy + 5 * U.s);
      ctx.restore();
    }

    // numbers (computed for this cross-section)
    var nA = ease(seg(u, 0.74, 0.9));
    var g2 = function (x) { return x.toFixed(2); };
    label(ctx, U, px0 + pw - 8 * U.s, py0 + 8 * U.s,
      'TE: r33 = ' + N.r33_rf + ' pm/V, Γ = ' + g2(N.gamma_TE) + '\nTM: r13 = ' + N.r13_rf + ' pm/V, Γ = ' + g2(N.gamma_TM) + '\nVπ(TM) / Vπ(TE) ≈ ' + N.ratio_rf.toFixed(1),
      { align: 'r', alpha: nA, size: 12, cols: ['#d6f7ff', '#ffd6d2', '#ffffff'], border: 'rgba(233,238,243,0.5)' });

    // scale bar (1 um), lower left over the oxide
    var sA = ease(seg(u, 0.5, 0.66));
    if (sA > 0) {
      ctx.save(); ctx.globalAlpha = sA;
      var sx = px0 + Math.max(84 * U.s, 62 * U.dpr), sy = py0 + Math.max(40 * U.s, 30 * U.dpr);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(sx, sy, k, Math.max(2, 3 * U.s));
      font(ctx, fpx(U, 11), '600'); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText('1 µm', sx + k / 2, sy - 3 * U.s);
      ctx.restore();
    }
  };
})();
