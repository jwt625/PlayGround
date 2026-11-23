  var CL = 299792458;
  var TAU_IN = 8e-12;            /* laser internal round-trip time, Rogister 2001, Table 2-I */
  var N_INP = 3.165;             /* InP index at 1550 nm, Sellmeier fit to Pettit and Turner 1965 */
  var R1 = Math.pow((N_INP - 1) / (N_INP + 1), 2);   /* cleaved facet, normal-incidence Fresnel */
  var NG_FIBER = 1.4682;         /* Corning SMF-28e+ product information, 2014, 1550 nm */
  var NSWEEP = 3;                /* sweep span in external free spectral ranges */

  function modeHop(root, params, api) {
    var s = { fdb: -55, logL: Math.log10(0.3), ng: NG_FIBER, alpha: 4, ph: 0, X: 0, up: [], down: [], hopsUp: [], hopsDown: [], didUp: false, didDown: false, busy: false };
    var done = false, raf = 0;
    var wrap = shell(root);
    el('div', 'fbw-note', 'Simulated: Lang-Kobayashi steady state of a single-mode laser with one distant reflector.', wrap);
    var fit1 = canvas(wrap, 240), fit2 = canvas(wrap, 230);
    var rd = el('div', 'fbw-read nomath', null, wrap);
    var row = el('div', 'fbw-row', null, wrap);
    var bUp = el('button', 'fbw-btn primary', 'Sweep up', row); bUp.type = 'button';
    var bDn = el('button', 'fbw-btn primary', 'Sweep down', row); bDn.type = 'button';
    var bClr = el('button', 'fbw-btn', 'Clear traces', row); bClr.type = 'button';
    var g = el('div', 'fbw-grid', null, wrap);
    var sP = slider(g, 'Free-running frequency shift, by temperature or current (manual)', 0, 1, 0.001, 0, function (v) { return fmtF(v * NSWEEP / tauX()); }, function (v) { if (s.busy) return; moveTo(v * NSWEEP * 2 * Math.PI, null, null); draw(); });
    slider(g, 'Feedback level $f_{ext}$', -90, -5, 0.5, s.fdb, function (v) { return v.toFixed(1) + ' dB'; }, function (v) { s.fdb = v; reset(); draw(); });
    slider(g, 'Distance to the reflector $L$', -3, Math.log10(3), 0.005, s.logL, function (v) { var m = Math.pow(10, v); return m < 1 ? (m * 100).toPrecision(3) + ' cm' : m.toPrecision(3) + ' m'; }, function (v) { s.logL = v; reset(); draw(); });
    slider(g, 'Linewidth enhancement factor $\\alpha$', 0, 8, 0.1, s.alpha, function (v) { return v.toFixed(1); }, function (v) { s.alpha = v; reset(); draw(); });
    var mRow = el('div', 'fbw-row', '<span class="fbw-tag">Path to the reflector</span>', wrap);
    var mBtns = [['Single-mode fiber, group index 1.4682', NG_FIBER], ['Air', 1.0]].map(function (p) {
      var b = el('button', 'fbw-btn' + (p[1] === s.ng ? ' on' : ''), p[0], mRow); b.type = 'button';
      b.addEventListener('click', function () { s.ng = p[1]; mBtns.forEach(function (x) { x.classList.remove('on'); }); b.classList.add('on'); reset(); draw(); });
      return b;
    });
    var fit3 = canvas(wrap, 220);
    el('details', 'fbw-det', '<summary>Model details</summary><p>Lang-Kobayashi steady state of a single-mode laser with one distant reflector. The lasing frequency $\\omega_s$ solves $(\\omega_s - \\omega_{th})\\tau = -C \\sin(\\omega_s \\tau + \\arctan\\alpha)$, with external round-trip time $\\tau$, feedback parameter $C = \\kappa \\frac{\\tau}{\\tau_{in}} \\sqrt{1+\\alpha^2}$ and coupling $\\kappa = \\frac{1-R_1}{\\sqrt{R_1}}\\sqrt{f_{ext}}$, where $f_{ext}$ is the fraction of the emitted power that returns into the laser. For $C < 1$ there is one solution; for $C > 1$ external-cavity modes and anti-modes appear in pairs, and a sweep follows one mode until it vanishes, then hops.</p><p>Laser: internal round-trip time $\\tau_{in}$ = 8 ps and $\\alpha$ = 4 (parameter table of Rogister, PhD thesis, Faculte Polytechnique de Mons, 2001); output facet reflectance $R_1$ = 0.27 from Fresnel at the InP index 3.165 (Pettit and Turner, J. Appl. Phys. 1965). This is the Fabry-Perot facet formula; in a DFB laser the grating and the facet phases change the coupling. Model: Lang and Kobayashi, IEEE J. Quantum Electron. 1980. Regime lines in the lower panel: Tkach and Chraplyvy, J. Lightwave Technol. 1986, measured on a 1.5 um DFB laser at twice threshold (regime I below about -70 dB, coherence collapse from about -45 dB to -10 dB, stable external-cavity operation above about -10 dB, which needs an anti-reflection coated facet or low current), as summarized by Rogister; the levels shift with laser structure, current and delay. The steady state shown here does not include the relaxation-oscillation instabilities behind coherence collapse.</p>', wrap);
    var unUp = hint(bUp, 'start here'), unDn = null;

    function tauX() { return 2 * s.ng * Math.pow(10, s.logL) / CL; }
    function kap() { return (1 - R1) / Math.sqrt(R1) * Math.sqrt(Math.pow(10, s.fdb / 10)); }
    function Cp() { return kap() * tauX() / TAU_IN * Math.sqrt(1 + s.alpha * s.alpha); }
    function fmtF(hz) { var a = Math.abs(hz); if (a < 1) return '0'; return a >= 1e9 ? (+(hz / 1e9).toPrecision(3)) + ' GHz' : a >= 1e6 ? (+(hz / 1e6).toPrecision(3)) + ' MHz' : (+(hz / 1e3).toPrecision(3)) + ' kHz'; }
    /* phase condition in X = (w_s - w_th) tau, with ph = w_th tau: g(X) = X + C sin(X + ph + atan a) = 0 */
    function gf(X, ph, C, th) { return X + C * Math.sin(X + ph + th); }
    function gd(X, ph, C, th) { return 1 + C * Math.cos(X + ph + th); }
    function roots(ph, C, th, lo, hi) {
      var out = [], n = Math.max(200, Math.ceil((hi - lo) / 0.01)), dx = (hi - lo) / n, x0 = lo, g0 = gf(x0, ph, C, th);
      for (var i = 1; i <= n; i++) {
        var x1 = lo + i * dx, g1 = gf(x1, ph, C, th);
        if (g0 === 0 || g0 * g1 < 0) {
          var a = x0, b = x1, ga = g0;
          for (var k = 0; k < 50; k++) { var m = 0.5 * (a + b), gm = gf(m, ph, C, th); if (ga * gm <= 0) b = m; else { a = m; ga = gm; } }
          var r = 0.5 * (a + b); out.push({ X: r, stable: gd(r, ph, C, th) > 0 });
        }
        x0 = x1; g0 = g1;
      }
      return out;
    }
    function startMode(ph) {
      var C = Cp(), th = Math.atan(s.alpha), best = null;
      roots(ph, C, th, -C - 1, C + 1).forEach(function (r) { if (!r.stable) return; var gain = Math.cos(r.X + ph); if (!best || gain > best.gain) best = { X: r.X, gain: gain }; });
      return best ? best.X : 0;
    }
    /* follow the current mode; when it disappears (saddle-node), hop to the nearest remaining stable mode */
    function stepTo(ph, rec, hops, depth) {
      var C = Cp(), th = Math.atan(s.alpha), X = s.X, ok = false;
      for (var k = 0; k < 40; k++) { var d = gd(X, ph, C, th); if (d <= 1e-9) break; var dX = gf(X, ph, C, th) / d; dX = Math.max(-0.2, Math.min(0.2, dX)); X -= dX; if (Math.abs(dX) < 1e-10) { ok = true; break; } }
      if (!ok || gd(X, ph, C, th) <= 0 || Math.abs(X - s.X) > 0.3) {
        depth = depth || 0;
        if (depth < 10 && Math.abs(ph - s.ph) > 1e-6) { var mid = 0.5 * (s.ph + ph); stepTo(mid, null, hops, depth + 1); stepTo(ph, null, hops, depth + 1); if (rec) rec.push([ph, s.X]); return; }
        /* the mode we were in no longer exists: hop to the nearest remaining stable mode */
        var rs = roots(ph, C, th, s.X - 8, s.X + 8).filter(function (r) { return r.stable; }), best = null;
        rs.forEach(function (r) { if (!best || Math.abs(r.X - s.X) < Math.abs(best.X - s.X)) best = r; });
        if (best) { X = best.X; if (C > 1 && hops) hops.push(ph); } else X = s.X;
      }
      s.X = X; s.ph = ph;
      if (rec) rec.push([ph, X]);
    }
    function moveTo(ph, rec, hops) { var n = Math.max(1, Math.ceil(Math.abs(ph - s.ph) / 0.01)), p0 = s.ph; for (var i = 1; i <= n; i++) stepTo(p0 + (ph - p0) * i / n, rec, hops); }
    function reset() { s.up = []; s.down = []; s.hopsUp = []; s.hopsDown = []; s.ph = 0; s.X = startMode(0); sP.set(0); }
    function sweep(dir) {
      if (s.busy) return; s.busy = true; bUp.disabled = bDn.disabled = bClr.disabled = true;
      var PH = NSWEEP * 2 * Math.PI, rec, hops;
      if (dir > 0) { s.up = []; s.hopsUp = []; rec = s.up; hops = s.hopsUp; s.ph = 0; s.X = startMode(0); }
      else { s.down = []; s.hopsDown = []; rec = s.down; hops = s.hopsDown; if (Math.abs(s.ph - PH) > 1e-9) { s.ph = PH; s.X = startMode(PH); } }
      rec.push([s.ph, s.X]);
      var t0 = null, from = s.ph, to = dir > 0 ? PH : 0, dur = 3500;
      function frame(ts) {
        if (t0 === null) t0 = ts; var k = Math.min(1, (ts - t0) / dur);
        moveTo(from + (to - from) * k, rec, hops); sP.set(s.ph / PH); draw();
        if (k < 1) raf = requestAnimationFrame(frame);
        else {
          s.busy = false; bUp.disabled = bDn.disabled = bClr.disabled = false;
          if (dir > 0) { s.didUp = true; if (unUp) { unUp(); unUp = null; } if (!s.didDown && !unDn) unDn = hint(bDn, 'now sweep down'); }
          else { s.didDown = true; if (unDn) { unDn(); unDn = null; } }
          if (s.didUp && s.didDown && !done) { done = true; api.complete(); }
        }
      }
      raf = requestAnimationFrame(frame);
    }
    bUp.addEventListener('click', function () { sweep(1); });
    bDn.addEventListener('click', function () { sweep(-1); });
    bClr.addEventListener('click', function () { if (!s.busy) { reset(); draw(); } });

    function draw() {
      var c = apiCols(api), C = Cp(), th = Math.atan(s.alpha), tau = tauX(), f = 1 / (2 * Math.PI * tau);
      var unit = f * 8 >= 1e9 ? 1e9 : 1e6, uL = unit === 1e9 ? 'GHz' : 'MHz', k = f / unit;  /* X (rad) to frequency in display units */
      /* panel 1: phase condition at the current free-running frequency */
      var o = fit1(), ctx = o.ctx; ctx.fillStyle = c.panel; ctx.fillRect(0, 0, o.w, o.h);
      var W = Math.min(C + 2.5, 40), xr = [-W * k, W * k], yr = [-W, W];
      var leg1 = [[c.muted, 'left side: (ws - wth) tau'], [c.warn, 'right side: -C sin(ws tau + atan alpha)'], [c.fg, 'dot: external-cavity mode, circle: anti-mode'], [c.accent, 'ring: lasing now']];
      var A = axes(ctx, o.w, o.h, xr, yr, 'lasing frequency minus free-running frequency (' + uL + ')', 'phase (rad)', c, { l: 58, r: 24, t: legend(ctx, leg1, 58, o.w - 24, false), b: 36 });
      legend(ctx, leg1, 58, o.w - 24, true);
      ctx.save(); ctx.beginPath(); ctx.rect(A.L, A.T, A.R - A.L, A.B - A.T); ctx.clip();
      ctx.strokeStyle = c.muted; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(A.X(-W * k), A.Y(-W)); ctx.lineTo(A.X(W * k), A.Y(W)); ctx.stroke();
      ctx.strokeStyle = c.warn; ctx.lineWidth = 1.6; ctx.beginPath();
      var N = Math.min(4000, Math.max(800, Math.round(W * 40)));
      for (var i = 0; i < N; i++) { var X = -W + 2 * W * i / (N - 1), y = -C * Math.sin(X + s.ph + th); if (i) ctx.lineTo(A.X(X * k), A.Y(y)); else ctx.moveTo(A.X(X * k), A.Y(y)); }
      ctx.stroke();
      var rs = roots(s.ph, C, th, -W, W), nM = 0, nA = 0;
      rs.forEach(function (r) {
        var px = A.X(r.X * k), py = A.Y(r.X);
        ctx.beginPath(); ctx.arc(px, py, 3.6, 0, 7);
        if (r.stable) { nM++; ctx.fillStyle = c.fg; ctx.fill(); } else { nA++; ctx.strokeStyle = c.fg; ctx.lineWidth = 1.2; ctx.stroke(); }
      });
      ctx.strokeStyle = c.accent; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(A.X(s.X * k), A.Y(s.X), 7, 0, 7); ctx.stroke();
      ctx.restore();
      if (C + 2.5 > 40) { ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = c.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('central part of the solution range shown', A.R - 6, A.T + 4); }

      /* panel 2: frequency pulling during the sweeps */
      var o2 = fit2(), c2 = o2.ctx; c2.fillStyle = c.panel; c2.fillRect(0, 0, o2.w, o2.h);
      var PH = NSWEEP * 2 * Math.PI, ymax = Math.max(1, C + 0.3) * k, xmax = PH * k;
      var leg2 = [[c.accent, 'sweep up'], [c.bad, 'sweep down', [6, 4]], [c.muted, 'hop', [1, 3]]];
      var B2 = axes(c2, o2.w, o2.h, [0, xmax], [-ymax, ymax], 'free-running frequency shift (' + uL + ')', 'pulling (' + uL + ')', c, { l: 58, r: 24, t: legend(c2, leg2, 58, o2.w - 24, false), b: 36 });
      legend(c2, leg2, 58, o2.w - 24, true);
      function trace(arr, col, w, dash) {
        if (!arr.length) return; c2.strokeStyle = col; c2.lineWidth = w; c2.setLineDash(dash || []); c2.beginPath();
        var jumps = [];
        for (var j = 0; j < arr.length; j++) { var p = arr[j], px = B2.X(p[0] * k), py = B2.Y(p[1] * k); if (j && Math.abs(p[1] - arr[j - 1][1]) > 0.3) { jumps.push([px, B2.Y(arr[j - 1][1] * k), py]); c2.stroke(); c2.beginPath(); c2.moveTo(px, py); } else if (j) c2.lineTo(px, py); else c2.moveTo(px, py); }
        c2.stroke(); c2.setLineDash([1, 3]); c2.lineWidth = 1.2;
        jumps.forEach(function (q) { c2.beginPath(); c2.moveTo(q[0], q[1]); c2.lineTo(q[0], q[2]); c2.stroke(); });
        c2.setLineDash([]);
      }
      trace(s.up, c.accent, 2.6); trace(s.down, c.bad, 1.8, [6, 4]);
      c2.fillStyle = c.fg; c2.beginPath(); c2.arc(B2.X(s.ph * k), B2.Y(s.X * k), 4, 0, 7); c2.fill();

      /* panel 3: regime map, feedback level vs distance */
      var o3 = fit3(), c3 = o3.ctx; c3.fillStyle = c.panel; c3.fillRect(0, 0, o3.w, o3.h);
      var leg3 = [[c.accent, 'C = 1 (model); shaded: C > 1'], [c.bad, 'levels measured by Tkach and Chraplyvy (1986)', [4, 3]], [c.fg, 'dot: this laser']];
      var D = axes(c3, o3.w, o3.h, [-90, -5], [-3, Math.log10(3)], 'feedback level, returned power / emitted power (dB)', 'distance, log10(m)', c, { l: 58, r: 24, t: legend(c3, leg3, 58, o3.w - 24, false), b: 36 });
      legend(c3, leg3, 58, o3.w - 24, true);
      /* C = 1 boundary: f = [ sqrt(R1) tau_in / ((1 - R1) tau sqrt(1 + a^2)) ]^2 */
      c3.save(); c3.beginPath(); c3.rect(D.L, D.T, D.R - D.L, D.B - D.T); c3.clip();
      c3.beginPath();
      for (var q = 0; q <= 60; q++) {
        var lg = -3 + (Math.log10(3) + 3) * q / 60, tt = 2 * s.ng * Math.pow(10, lg) / CL;
        var fb = 20 * Math.log10(Math.sqrt(R1) * TAU_IN / ((1 - R1) * tt * Math.sqrt(1 + s.alpha * s.alpha)));
        if (q) c3.lineTo(D.X(fb), D.Y(lg)); else c3.moveTo(D.X(fb), D.Y(lg));
      }
      c3.lineTo(D.X(-5), D.Y(Math.log10(3))); c3.lineTo(D.X(-5), D.Y(-3)); c3.closePath();
      c3.fillStyle = c.accent; c3.globalAlpha = 0.12; c3.fill(); c3.globalAlpha = 1; c3.strokeStyle = c.accent; c3.lineWidth = 1.6; c3.stroke();
      var narrow = D.R - D.L < 560;
      [[-70, narrow ? 'II' : 'regime II onset near -70 dB'], [-45, narrow ? 'IV' : 'coherence collapse from -45 dB'], [-10, narrow ? 'V' : 'regime V above -10 dB']].forEach(function (p, idx) {
        c3.strokeStyle = c.bad; c3.setLineDash([4, 3]); c3.lineWidth = 1; c3.beginPath(); c3.moveTo(D.X(p[0]), D.T); c3.lineTo(D.X(p[0]), D.B); c3.stroke(); c3.setLineDash([]);
        c3.fillStyle = c.bad; c3.font = '10.5px system-ui, sans-serif'; c3.textAlign = idx === 2 ? 'right' : 'left'; c3.textBaseline = 'top';
        c3.fillText(p[1], D.X(p[0]) + (idx === 2 ? -4 : 4), D.B - 16);
      });
      c3.fillStyle = c.fg; c3.beginPath(); c3.arc(D.X(s.fdb), D.Y(s.logL), 5, 0, 7); c3.fill();
      c3.restore();

      var f2 = function (a) { return a.length ? a.slice(0, 4).map(function (p) { return fmtF(p / (2 * Math.PI * tau)); }).join(', ') + (a.length > 4 ? ' ...' : '') : (C < 1 ? 'none (C < 1)' : 'none yet'); };
      var reg = s.fdb >= -10 ? 'above -10 dB: stable external-cavity operation only with an anti-reflection coated facet or low current' :
        s.fdb >= -45 ? 'coherence-collapse range in the 1986 measurement: steady states exist here but lose stability' :
        C < 1 ? 'one solution: the laser is pulled, it cannot hop' : 'several external-cavity modes: hops and hysteresis';
      rd.innerHTML =
        '<div><span>Feedback parameter C </span><b>' + fmtN(C) + '</b></div>' +
        '<div><span>Coupling kappa </span><b>' + kap().toExponential(2) + '</b></div>' +
        '<div><span>External round trip tau </span><b>' + (tau >= 1e-9 ? (+(tau * 1e9).toPrecision(3)) + ' ns' : (+(tau * 1e12).toPrecision(3)) + ' ps') + '</b></div>' +
        '<div><span>External-cavity mode spacing 1/tau </span><b>' + fmtF(1 / tau) + '</b></div>' +
        '<div><span>Modes / anti-modes now </span><b>' + nM + ' / ' + nA + (C + 2.5 > 40 ? ' (in view)' : '') + '</b></div>' +
        '<div><span>Largest pulling </span><b>' + fmtF(C / (2 * Math.PI * tau)) + '</b></div>' +
        '<div><span>Hops going up at shift </span><b>' + f2(s.hopsUp) + '</b></div>' +
        '<div><span>Hops going down at shift </span><b>' + f2(s.hopsDown.slice().reverse()) + '</b></div>' +
        '<div style="grid-column:1/-1"><span>Regime </span><b' + (s.fdb >= -45 ? ' class="warn"' : '') + '>' + reg + '</b></div>';
    }
    var onR = function () { draw(); }; window.addEventListener('resize', onR);
    api.onCleanup(function () { cancelAnimationFrame(raf); window.removeEventListener('resize', onR); });
    reset(); draw();
  }

  window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  window.LMS_WIDGETS.modeHop = modeHop;
})();
