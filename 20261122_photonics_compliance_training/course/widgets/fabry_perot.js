/* Parasitic Fabry-Perot etalon (feedback course, Module 2).
   Model: two real interfaces with normal-incidence Fresnel reflectance R = ((n1 - n2) / (n1 + n2))^2,
   a gap of length L and group index n_g, and a single-pass power transmission eta in the gap.
   Lossy Airy transmission T = (1 - R1)(1 - R2) eta / (1 + rho^2 - 2 rho cos d), rho = sqrt(R1 R2) eta,
   d = 2 pi (nu - nu_peak) / FSR, FSR = c / (2 n_g L), ripple period in wavelength lambda^2 / (2 n_g L).
   Reflection seen from side 1: (R1 + R2 eta^2 - 2 sqrt(R1 R2) eta cos d) / (1 + rho^2 - 2 rho cos d).
   Finesse F = pi sqrt(rho) / (1 - rho) (the coefficient of finesse would be 4 rho / (1 - rho)^2).
   Indices at 1550 nm: fused silica 1.444 (Malitson 1965), silicon 3.48 (Li 1980), InP 3.165 (Pettit and
   Turner 1965 Sellmeier), congruent LiNbO3 extraordinary 2.138 (Zelmon, Small and Jundt 1997 Sellmeier),
   Si3N4 1.996 (Sellmeier of Luke et al., Opt. Lett. 40, 4823 (2015), 1.9963 at 1550 nm). Fiber group index 1.4682 at 1550 nm (Corning SMF-28e+ product information, 2014).
   Overrides the in-course widget of the same name. */
(function () {
  'use strict';
  var STYLE = '.fbw{color:var(--ink,#e3e6ea);font-size:13px;line-height:1.35}.fbw *{border-radius:0;box-sizing:border-box}' +
    '.fbw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px 18px;margin:8px 0}' +
    '.fbw-ctl label{display:flex;justify-content:space-between;gap:6px;font-size:12px;color:var(--ink-2,#8b939c)}.fbw-ctl label b{color:var(--ink,#e3e6ea);font-weight:600;white-space:nowrap}' +
    '.fbw-ctl input[type=range]{width:100%}.fbw-row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}' +
    '.fbw-btn{background:var(--panel,#1d2228);color:var(--ink,#e3e6ea);border:1px solid var(--line,#3a414a);padding:4px 10px;cursor:pointer;font:inherit}' +
    '.fbw-btn:hover,.fbw-btn.on{border-color:var(--accent,#4ea1ff);color:var(--accent,#4ea1ff)}.fbw-btn.primary{border-color:var(--accent,#4ea1ff);color:var(--accent,#4ea1ff);font-weight:600}' +
    '.fbw-btn:disabled{opacity:.5;cursor:default}' +
    '.fbw-read{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:3px 14px;font-size:12px;font-variant-numeric:tabular-nums;margin:6px 0}' +
    '.fbw-read span{color:var(--ink-2,#8b939c)}.fbw-read b{color:var(--accent,#4ea1ff);font-weight:600}.fbw-read b.warn{color:var(--warn,#d29922)}' +
    '.fbw-tag{font-size:11px;color:var(--ink-2,#8b939c);letter-spacing:.05em;text-transform:uppercase;margin:4px 0}' +
    '.fbw-note{font-size:11.5px;color:var(--ink-2,#8b939c);margin:6px 0;line-height:1.45}' +
    '.fbw-det{font-size:11.5px;color:var(--ink-2,#8b939c);margin:6px 0;line-height:1.45}.fbw-det summary{cursor:pointer;color:var(--accent,#4ea1ff);width:max-content}.fbw-det p{margin:4px 0}' +
    '.fbw-hint{outline:2px solid var(--accent,#4ea1ff);outline-offset:3px;animation:fbwP 1.1s ease-in-out infinite}' +
    '@keyframes fbwP{0%,100%{outline-color:var(--accent,#4ea1ff)}50%{outline-color:transparent}}' +
    '.fbw-hand{display:inline-block;margin-left:8px;font-size:11px;font-weight:600;color:var(--accent,#4ea1ff);animation:fbwB .9s ease-in-out infinite}' +
    '@keyframes fbwB{0%,100%{transform:translateX(0)}50%{transform:translateX(-6px)}}' +
    '.fbw select,.fbw input[type=number]{background:var(--panel-2,#14171b);color:var(--ink,#e3e6ea);border:1px solid var(--line,#3a414a);padding:3px 5px;font:inherit;max-width:100%}' +
    '.fbw input[type=number]{width:90px}.fbw-ifc{border:1px solid var(--line,#3a414a);padding:6px 8px;background:var(--panel-2,#14171b)}' +
    '.fbw-ifc select{width:100%;margin:3px 0}.fbw-ifc .fbw-sub{font-size:12px;color:var(--ink-2,#8b939c);font-variant-numeric:tabular-nums}.fbw-ifc .fbw-sub b{color:var(--accent,#4ea1ff)}';

  function shell(root) { var w = document.createElement('div'); w.className = 'fbw'; var st = document.createElement('style'); st.textContent = STYLE; w.appendChild(st); root.appendChild(w); return w; }
  function el(tag, cls, html, parent) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.appendChild(e); return e; }
  function slider(parent, lab, min, max, step, val, fmt, on) {
    var box = el('div', 'fbw-ctl', null, parent);
    var l = el('label', null, '<span>' + lab + '</span><b class="nomath"></b>', box);
    var i = el('input', null, null, box); i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = val;
    var b = l.querySelector('b'); var upd = function () { b.textContent = fmt(+i.value); };
    i.addEventListener('input', function () { upd(); on(+i.value); }); upd();
    return { input: i, box: box, set: function (v) { i.value = v; upd(); } };
  }
  function hint(target, label) {
    target.classList.add('fbw-hint');
    var h = el('span', 'fbw-hand', '&larr; ' + (label || 'start here'));
    if (target.tagName === 'BUTTON') target.insertAdjacentElement('afterend', h); else { var lb = target.querySelector('label span, .fbw-tag'); (lb || target).appendChild(h); }
    return function () { target.classList.remove('fbw-hint'); if (h.parentNode) h.parentNode.removeChild(h); };
  }
  function canvas(parent, hpx) {
    var cv = el('canvas', null, null, parent); cv.style.width = '100%'; cv.style.height = hpx + 'px'; cv.style.display = 'block';
    return function () {
      var dpr = window.devicePixelRatio || 1, w = Math.max(260, parent.clientWidth || 600);
      cv.width = Math.round(w * dpr); cv.height = Math.round(hpx * dpr);
      var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx: ctx, w: w, h: hpx };
    };
  }
  function ticks(a, b, n) {
    var span = b - a; if (!(span > 0)) return [a];
    var st = Math.pow(10, Math.floor(Math.log10(span / n))), e = span / n / st;
    st *= e >= 5 ? 5 : e >= 2 ? 2 : 1;
    var out = []; for (var v = Math.ceil(a / st) * st; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v); return out;
  }
  function fmtN(v) { var a = Math.abs(v); if (a === 0) return '0'; if (a >= 1e5 || a < 0.01) return v.toExponential(1); return (+v.toPrecision(3)).toString(); }
  function axes(ctx, w, h, xr, yr, xl, yl, c, pad, xfmt) {
    pad = pad || { l: 58, r: 24, t: 10, b: 36 };
    var L = pad.l, Rr = w - pad.r, T = pad.t, B = h - pad.b;
    var X = function (x) { return L + (x - xr[0]) / (xr[1] - xr[0]) * (Rr - L); };
    var Y = function (y) { return B - (y - yr[0]) / (yr[1] - yr[0]) * (B - T); };
    ctx.fillStyle = c.bg; ctx.fillRect(L, T, Rr - L, B - T);
    ctx.strokeStyle = c.border; ctx.lineWidth = 1; ctx.strokeRect(L, T, Rr - L, B - T);
    ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = c.muted;
    ticks(xr[0], xr[1], Math.max(3, Math.min(6, Math.floor((Rr - L) / 85)))).forEach(function (t) { var px = X(t); ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, B); ctx.stroke(); ctx.globalAlpha = 1; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText((xfmt || fmtN)(t), px, B + 4); });
    ticks(yr[0], yr[1], 4).forEach(function (t) { var py = Y(t); ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(Rr, py); ctx.stroke(); ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmtN(t), L - 4, py); });
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(xl, (L + Rr) / 2, h - 2);
    ctx.save(); ctx.translate(11, (T + B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(yl, 0, 0); ctx.restore();
    return { X: X, Y: Y, L: L, R: Rr, T: T, B: B };
  }
  /* legend above the plot: items [[color, text, dash]]; returns the top padding needed; draws when draw is true */
  function legend(ctx, items, L, R, draw) {
    ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    var x = L, row = 0;
    items.forEach(function (it) {
      var wd = 20 + ctx.measureText(it[1]).width + 14;
      if (x + wd > R && x > L) { x = L; row++; }
      if (draw) {
        var y = 9 + row * 15;
        ctx.strokeStyle = it[0]; ctx.lineWidth = 2; ctx.setLineDash(it[2] || []); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 14, y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = it[0]; ctx.fillText(it[1], x + 19, y);
      }
      x += wd;
    });
    return 12 + (row + 1) * 15;
  }
  function apiCols(api) {
    var g = function (n, fb) { try { var v = api.css(n); return (v && String(v).trim()) || fb; } catch (e) { return fb; } };
    return { bg: g('--panel-2', '#14171b'), panel: g('--panel', '#1d2228'), fg: g('--ink', '#e3e6ea'), muted: g('--ink-2', '#8b939c'), accent: g('--accent', '#4ea1ff'), bad: g('--bad', '#e5534b'), ok: g('--ok', '#3fb950'), warn: g('--warn', '#d29922'), border: g('--line', '#3a414a') };
  }
  function db(x) { return 10 * Math.log10(x); }
  function fmtLen(m) { return m < 1e-3 ? (m * 1e6).toPrecision(3) + ' um' : m < 1 ? (m * 1e3).toPrecision(3) + ' mm' : m.toPrecision(3) + ' m'; }
  function fmtPm(pm) { return pm >= 1000 ? (+(pm / 1000).toPrecision(3)) + ' nm' : (+pm.toPrecision(3)) + ' pm'; }
  function fmtHz(hz) { return hz >= 1e12 ? (+(hz / 1e12).toPrecision(3)) + ' THz' : hz >= 1e9 ? (+(hz / 1e9).toPrecision(3)) + ' GHz' : (+(hz / 1e6).toPrecision(3)) + ' MHz'; }

  var CL = 299792458, LAM0 = 1550e-9, NU0 = CL / LAM0;
  /* Interface catalog: [label, n1, n2, source]. Fresnel at normal incidence, plane wave. */
  var IFC = {
    silica_air: ['Fiber end, flat cleave, in air (silica 1.444)', 1.444, 1, 'Malitson 1965'],
    sin_air: ['Silicon nitride facet in air (1.996)', 1.996, 1, 'Luke et al. 2015'],
    ln_air: ['Lithium niobate facet in air, extraordinary (2.138)', 2.138, 1, 'Zelmon et al. 1997'],
    inp_air: ['Uncoated InP laser facet in air (3.165)', 3.165, 1, 'Pettit and Turner 1965'],
    si_air: ['Silicon facet in air (3.48)', 3.48, 1, 'Li 1980'],
    silica_si: ['Silica on silicon interface (1.444 / 3.48)', 1.444, 3.48, 'Malitson 1965; Li 1980'],
    ar: ['Anti-reflection coated surface (set residual)', null, null, 'set by you'],
    custom: ['Any reflector (set reflectance)', null, null, 'set by you']
  };

  function fabryPerot(root, params, api) {
    var s = { i1: 'silica_air', i2: 'ln_air', ar1: -30, ar2: -30, cu1: -20, cu2: -20, logL: Math.log10(100e-6), ng: 1.0, loss: 0, span: 50, step: 100 };
    var touched = {}, sweptOnce = false, done = false, raf = 0, sweep = null;
    var wrap = shell(root);
    el('div', 'fbw-note', 'Two real interfaces facing each other across a gap form an etalon: Fresnel reflectances at 1550 nm, lossy Airy transmission.', wrap);
    var fit = canvas(wrap, 250);
    var rd = el('div', 'fbw-read nomath', null, wrap);

    var ifRow = el('div', 'fbw-grid', null, wrap);
    function ifcBox(k, title) {
      var box = el('div', 'fbw-ifc', null, ifRow);
      el('div', 'fbw-tag', title, box);
      var sel = el('select', null, null, box);
      Object.keys(IFC).forEach(function (key) { var o = el('option', null, IFC[key][0], sel); o.value = key; });
      sel.value = s['i' + k];
      var sub = el('div', 'fbw-sub nomath', null, box);
      var sl = slider(box, 'Residual reflectance', -60, -10, 1, s['ar' + k], function (v) { return v + ' dB'; }, function (v) { if (s['i' + k] === 'ar') s['ar' + k] = v; else s['cu' + k] = v; mark('r' + k); draw(); });
      sel.addEventListener('change', function () { s['i' + k] = sel.value; sync(); mark('r' + k); draw(); });
      function sync() {
        var key = s['i' + k], show = key === 'ar' || key === 'custom';
        sl.box.style.display = show ? '' : 'none';
        if (key === 'ar') { sl.box.querySelector('label span').textContent = 'Residual reflectance of the coating'; sl.set(s['ar' + k]); }
        if (key === 'custom') { sl.box.querySelector('label span').textContent = 'Reflectance'; sl.set(s['cu' + k]); }
      }
      sync();
      return { box: box, sel: sel, sub: sub };
    }
    var I1 = ifcBox(1, 'Interface 1 (toward the laser)'), I2 = ifcBox(2, 'Interface 2');

    var g = el('div', 'fbw-grid', null, wrap);
    var sL = slider(g, 'Gap length $L$', -5, 0, 0.01, s.logL, function (v) { return fmtLen(Math.pow(10, v)); }, function (v) { s.logL = v; mark('L'); draw(); });
    var sN = slider(g, 'Group index in the gap $n_g$', 1, 4.5, 0.0001, s.ng, function (v) { return v.toFixed(4); }, function (v) { s.ng = v; mark('ng'); draw(); });
    slider(g, 'Single-pass loss in the gap', 0, 20, 0.1, s.loss, function (v) { return v.toFixed(1) + ' dB'; }, function (v) { s.loss = v; mark('loss'); draw(); });
    var nRow = el('div', 'fbw-row', '<span class="fbw-tag">Gap medium</span>', wrap);
    [['Air', 1.0], ['Single-mode fiber, 1.4682', 1.4682]].forEach(function (p) {
      var b = el('button', 'fbw-btn', p[0], nRow); b.type = 'button';
      b.addEventListener('click', function () { s.ng = p[1]; sN.set(p[1]); mark('ng'); draw(); });
    });

    var sw = el('div', 'fbw-row', '<span class="fbw-tag">Tunable laser sweep</span>', wrap);
    el('span', 'fbw-note', 'span', sw);
    var spanSel = el('select', null, null, sw);
    [[0.05, '0.05 nm'], [0.5, '0.5 nm'], [5, '5 nm'], [50, '50 nm']].forEach(function (p) { var o = el('option', null, p[1], spanSel); o.value = p[0]; });
    spanSel.value = s.span;
    el('span', 'fbw-note', 'step', sw);
    var stepSel = el('select', null, null, sw);
    [[0.1, '0.1 pm'], [1, '1 pm'], [10, '10 pm'], [100, '100 pm']].forEach(function (p) { var o = el('option', null, p[1], stepSel); o.value = p[0]; });
    stepSel.value = s.step;
    var bRun = el('button', 'fbw-btn primary', 'Run laser sweep', sw); bRun.type = 'button';
    spanSel.addEventListener('change', function () { s.span = +spanSel.value; sweep = null; mark('span'); draw(); });
    stepSel.addEventListener('change', function () { s.step = +stepSel.value; sweep = null; mark('step'); draw(); });

    var inv = el('div', 'fbw-row', '<span class="fbw-tag">Ripple-to-length calculator</span> <span>observed period</span> <input type="number" min="0" step="any" value="100"> <span>pm at 1550 nm, with the group index above, implies a spacing of</span> <b class="fbw-invout nomath"></b>', wrap);
    var invIn = inv.querySelector('input'), invOut = inv.querySelector('.fbw-invout');
    invOut.style.color = 'var(--accent,#4ea1ff)';
    invIn.addEventListener('input', function () { mark('inv'); draw(); });
    el('details', 'fbw-det', '<summary>Model details</summary><p>Two real interfaces facing each other across a gap form an etalon. Reflectances from the refractive indices at 1550 nm by normal-incidence Fresnel, $R = \\left(\\frac{n_1 - n_2}{n_1 + n_2}\\right)^2$; lossy Airy transmission $T = \\frac{(1-R_1)(1-R_2)\\,\\eta}{1 + \\rho^2 - 2\\rho\\cos\\delta}$ with $\\rho = \\sqrt{R_1 R_2}\\,\\eta$ and $\\eta$ the single-pass power transmission of the gap; ripple period $\\Delta\\lambda = \\lambda^2 / (2 n_g L)$. A guided mode at a waveguide facet reflects somewhat differently from a plane wave; the plane-wave value is the starting estimate.</p><p>Indices at 1550 nm: fused silica 1.444 (Malitson, J. Opt. Soc. Am. 1965); silicon 3.48 (Li, J. Phys. Chem. Ref. Data 1980); InP 3.165 (Sellmeier fit to Pettit and Turner, J. Appl. Phys. 1965); congruent lithium niobate, extraordinary, 2.138 (Sellmeier of Zelmon, Small and Jundt, J. Opt. Soc. Am. B 1997); silicon nitride 1.996 (Sellmeier of Luke et al., Opt. Lett. 2015). Fiber group index 1.4682 at 1550 nm (Corning SMF-28e+ product information, 2014). Coating residuals, custom reflectances and gap loss are values you set. Fringe positions are placed with a transmission peak at 1550 nm; only the period and depth are physical.</p>', wrap);

    var unh = [hint(bRun, 'start here')];
    var unL = null;
    function mark(k) {
      touched[k] = 1;
      if (k !== 'span' && k !== 'step' && unL) { unL(); unL = null; }
      check();
    }
    function check() { if (!done && sweptOnce && Object.keys(touched).filter(function (k) { return k !== 'span' && k !== 'step'; }).length >= 1) { done = true; api.complete(); } }

    function refl(k) {
      var key = s['i' + k], d = IFC[key];
      if (key === 'ar') return Math.pow(10, s['ar' + k] / 10);
      if (key === 'custom') return Math.pow(10, s['cu' + k] / 10);
      var r = (d[1] - d[2]) / (d[1] + d[2]); return r * r;
    }
    function model() {
      var R1 = refl(1), R2 = refl(2), eta = Math.pow(10, -s.loss / 10), L = Math.pow(10, s.logL);
      var rho = Math.sqrt(R1 * R2) * eta, fsr = CL / (2 * s.ng * L), per = LAM0 * LAM0 / (2 * s.ng * L);
      return { R1: R1, R2: R2, eta: eta, L: L, rho: rho, fsr: fsr, per: per,
        T: function (lam) { var d = 2 * Math.PI * (CL / lam - NU0) / fsr; return (1 - R1) * (1 - R2) * eta / (1 + rho * rho - 2 * rho * Math.cos(d)); },
        Rb: function (lam) { var d = 2 * Math.PI * (CL / lam - NU0) / fsr; return (R1 + R2 * eta * eta - 2 * Math.sqrt(R1 * R2) * eta * Math.cos(d)) / (1 + rho * rho - 2 * rho * Math.cos(d)); } };
    }

    function runSweep() {
      if (sweep && sweep.running) return;
      unh.forEach(function (f) { f(); }); unh = [];
      var n = Math.round(s.span * 1000 / s.step) + 1, lam0 = LAM0 - s.span * 1e-9 / 2;
      sweep = { running: true, n: n, k: 0, lam0: lam0, dl: s.step * 1e-12, pts: [] };
      bRun.disabled = true;
      var t0 = null, dur = 2600;
      function frame(ts) {
        if (t0 === null) t0 = ts;
        var u = Math.min(1, (ts - t0) / dur), kk = Math.round(u * (n - 1));
        for (; sweep.k <= kk; sweep.k++) sweep.pts.push(lam0 + sweep.k * sweep.dl);
        draw();
        if (u < 1) raf = requestAnimationFrame(frame);
        else {
          sweep.running = false; bRun.disabled = false; sweptOnce = true; draw();
          if (!Object.keys(touched).filter(function (k) { return k !== 'span' && k !== 'step'; }).length && !unL) unL = hint(sL.box, 'now change the gap and sweep again');
          check();
        }
      }
      raf = requestAnimationFrame(frame);
    }
    bRun.addEventListener('click', runSweep);

    function draw() {
      var o = fit(), ctx = o.ctx, w = o.w, h = o.h, c = apiCols(api), m = model();
      ctx.fillStyle = c.panel; ctx.fillRect(0, 0, w, h);
      var Tmax = (1 - m.R1) * (1 - m.R2) * m.eta / Math.pow(1 - m.rho, 2), Tmin = (1 - m.R1) * (1 - m.R2) * m.eta / Math.pow(1 + m.rho, 2);
      var dbMax = db(Tmax), dbMin = db(Tmin), mid = (dbMax + dbMin) / 2, half = Math.max((dbMax - dbMin) / 2 * 1.3, 0.01);
      var spanM = s.span * 1e-9, lo = LAM0 - spanM / 2, hi = LAM0 + spanM / 2;
      var npx = Math.max(2, Math.round(w - 82)), perPx = m.per / (spanM / npx), sampPer = m.per / (s.step * 1e-12);
      var leg = [[c.muted, perPx < 3 ? 'true response (ripple finer than a pixel: envelope shown)' : 'true response'], [c.accent, 'measured by the laser sweep, ' + fmtPm(s.step) + ' steps']];
      if (sweep && !sweep.running && sampPer < 2) leg.push([c.warn, 'fewer than two samples per ripple period: the sweep aliases the ripple']);
      var A = axes(ctx, w, h, [lo * 1e9, hi * 1e9], [mid - half, mid + half], 'wavelength (nm)', 'transmission (dB)', c, { l: 58, r: 24, t: legend(ctx, leg, 58, w - 24, false), b: 36 }, function (v) { return (+v.toFixed(4)).toString(); });
      legend(ctx, leg, 58, w - 24, true);
      /* true response: exact curve if resolvable, otherwise its envelope */
      if (perPx < 3) {
        ctx.fillStyle = c.muted; ctx.globalAlpha = 0.3; ctx.fillRect(A.L, A.Y(dbMax), A.R - A.L, A.Y(dbMin) - A.Y(dbMax)); ctx.globalAlpha = 1;
      } else {
        ctx.strokeStyle = c.muted; ctx.lineWidth = 1.2; ctx.beginPath();
        var N = Math.min(6000, npx * 4);
        for (var i = 0; i < N; i++) { var lam = lo + spanM * i / (N - 1), y = A.Y(db(m.T(lam))); if (i) ctx.lineTo(A.X(lam * 1e9), y); else ctx.moveTo(A.X(lam * 1e9), y); }
        ctx.stroke();
      }
      /* sampled laser sweep */
      if (sweep && sweep.pts.length) {
        var pts = sweep.pts.map(function (lam) { return [lam, m.T(lam)]; }); ctx.strokeStyle = c.accent; ctx.lineWidth = 1.8;
        if (pts.length <= npx * 2) {
          ctx.beginPath();
          for (var j = 0; j < pts.length; j++) { var px = A.X(pts[j][0] * 1e9), py = A.Y(db(pts[j][1])); if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
          ctx.stroke();
          if (pts.length <= 120) { ctx.fillStyle = c.accent; pts.forEach(function (p) { ctx.beginPath(); ctx.arc(A.X(p[0] * 1e9), A.Y(db(p[1])), 2.2, 0, 7); ctx.fill(); }); }
        } else {
          /* many samples per pixel: min / max per pixel column */
          var col = {}, k;
          for (k = 0; k < pts.length; k++) { var cx = Math.floor(A.X(pts[k][0] * 1e9)), v = db(pts[k][1]); var e = col[cx]; if (!e) col[cx] = [v, v]; else { if (v < e[0]) e[0] = v; if (v > e[1]) e[1] = v; } }
          ctx.beginPath(); Object.keys(col).forEach(function (cx) { var e = col[cx]; ctx.moveTo(+cx + 0.5, A.Y(e[1])); ctx.lineTo(+cx + 0.5, A.Y(e[0]) + 0.5); }); ctx.stroke();
        }
        if (sweep.running) { var last = pts[pts.length - 1]; ctx.strokeStyle = c.warn; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(A.X(last[0] * 1e9), A.T); ctx.lineTo(A.X(last[0] * 1e9), A.B); ctx.stroke(); }
      }

      var Rmax = Math.pow(Math.sqrt(m.R1) + Math.sqrt(m.R2) * m.eta, 2) / Math.pow(1 + m.rho, 2);
      var F = Math.PI * Math.sqrt(m.rho) / (1 - m.rho), pv = 20 * Math.log10((1 + m.rho) / (1 - m.rho));
      [[I1, 1], [I2, 2]].forEach(function (p) { var R = p[1] === 1 ? m.R1 : m.R2, src = IFC[s['i' + p[1]]][3]; p[0].sub.innerHTML = 'R = <b>' + fmtN(R * 100) + ' %</b> (' + db(R).toFixed(1) + ' dB), ' + src; });
      rd.innerHTML =
        '<div><span>Ripple period at 1550 nm </span><b>' + fmtPm(m.per * 1e12) + '</b></div>' +
        '<div><span>Free spectral range </span><b>' + fmtHz(m.fsr) + '</b></div>' +
        '<div><span>Peak-to-valley ripple </span><b>' + fmtN(pv) + ' dB</b></div>' +
        '<div><span>Finesse </span><b>' + fmtN(F) + '</b></div>' +
        '<div><span>Round-trip amplitude factor </span><b>' + fmtN(m.rho) + '</b></div>' +
        '<div><span>Worst-case reflection back to the laser </span><b>' + db(Rmax).toFixed(1) + ' dB</b></div>' +
        '<div><span>Sweep samples per ripple period </span><b' + (sampPer < 2 ? ' class="warn"' : '') + '>' + fmtN(sampPer) + '</b></div>';
      var pvv = parseFloat(invIn.value);
      invOut.textContent = pvv > 0 ? fmtLen(LAM0 * LAM0 / (2 * s.ng * pvv * 1e-12)) : 'N/A';
    }
    var onR = function () { draw(); }; window.addEventListener('resize', onR);
    api.onCleanup(function () { cancelAnimationFrame(raf); window.removeEventListener('resize', onR); });
    draw();
  }

  window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  window.LMS_WIDGETS.fabryPerot = fabryPerot;
})();
