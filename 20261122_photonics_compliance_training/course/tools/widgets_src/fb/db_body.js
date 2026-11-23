  var CL = 299792458, LAM0 = 1550e-9, NU0 = CL / LAM0, W0 = 2 * Math.PI * NU0;
  /* Presets (rates are angular: kappa = 2 pi x linewidth in Hz).
     Ji et al., Optica 4, 619 (2017): Si3N4 ring, radius 115 um, intrinsic Q 37 million, loaded FWHM 5.6 MHz.
     Nominal bus coupling inferred here from those two numbers: (5.6 MHz - nu/Q0) / (nu/Q0) = 0.071,
     which makes the preset under-coupled (an inference, not a reported value).
     Bahadori, Rumley, Cheng and Bergman, IEEE Optical Interconnects Conference (2018): silicon add-drop ring, radius 5 um,
     FSR 17.2 nm, extracted field coupling 0.1222 per bus, round trip 0.9998 (taken as field transmission),
     backscatter coefficient 0.0124 per round trip. Rates: kappa_e = 0.1222^2 FSR, kappa_0 = (1 - 0.9998^2) FSR,
     beta = 0.0124 FSR, with FSR = c d-lambda / lambda^2 = 1 / (round-trip time). */
  var FSR_SI = CL * 17.2e-9 / (LAM0 * LAM0); /* round trips per second, 1 / T_rt */
  var PRESETS = {
    sin: { label: 'Silicon nitride ring, intrinsic Q 37 million (Ji et al. 2017)', q0: 37e6, ratio: (5.6e6 - NU0 / 37e6) / (NU0 / 37e6), drop: false, beta: 0 },
    si: { label: 'Silicon add-drop ring, 5 um radius (Bahadori et al. 2018)', q0: W0 / ((1 - 0.9998 * 0.9998) * FSR_SI), ratio: (0.1222 * 0.1222) / (1 - 0.9998 * 0.9998), drop: true, beta: 0.0124 * FSR_SI }
  };

  function doublet(root, params, api) {
    var s = { q0: PRESETS.sin.q0, ratio: PRESETS.sin.ratio, drop: false, lb: 2, preset: 'sin', log: false }; /* lb: log10(beta / 2 pi in Hz); <= 2 means none */
    var done = false;
    var wrap = shell(root);
    el('div', 'fbw-note', 'Simulated: coupled-mode model of a ring with clockwise and counter-clockwise modes at 1550 nm.', wrap);
    var fit = canvas(wrap, 290);
    var rd = el('div', 'fbw-read nomath', null, wrap);
    var pRow = el('div', 'fbw-row', '<span class="fbw-tag">Device</span>', wrap);
    var pBtns = Object.keys(PRESETS).map(function (k) {
      var b = el('button', 'fbw-btn' + (k === s.preset ? ' on' : ''), PRESETS[k].label, pRow); b.type = 'button';
      b.addEventListener('click', function () { load(k); pBtns.forEach(function (x) { x.classList.remove('on'); }); b.classList.add('on'); draw(); });
      return b;
    });
    var g = el('div', 'fbw-grid', null, wrap);
    var un = null;
    var sB = slider(g, 'Backscatter rate $\\beta / 2\\pi$', 2, 11, 0.01, s.lb, function (v) { return v <= 2.001 ? 'none' : fmtF(Math.pow(10, v)); }, function (v) { s.lb = v; s.preset = null; pBtns.forEach(function (x) { x.classList.remove('on'); }); if (un) { un(); un = null; } draw(); });
    var sQ = slider(g, 'Intrinsic quality factor $Q_0$', 3, 9, 0.0001, Math.log10(s.q0), function (v) { return fmtQ(Math.pow(10, v)); }, function (v) { s.q0 = Math.pow(10, v); s.preset = null; pBtns.forEach(function (x) { x.classList.remove('on'); }); draw(); });
    var sR = slider(g, 'Bus coupling $\\kappa_e / \\kappa_0$', -2, 2, 0.0001, Math.log10(s.ratio), function (v) { var r = Math.pow(10, v); return fmtN(r) + (s.drop ? ' per bus' : Math.abs(v) < 0.02 ? ', critical' : r < 1 ? ', under-coupled' : ', over-coupled'); }, function (v) { s.ratio = Math.pow(10, v); s.preset = null; pBtns.forEach(function (x) { x.classList.remove('on'); }); draw(); });
    var bRow = el('div', 'fbw-row', '<span class="fbw-tag">Buses</span>', wrap);
    var bAll = el('button', 'fbw-btn on', 'One bus (all-pass)', bRow); bAll.type = 'button';
    var bAdd = el('button', 'fbw-btn', 'Two buses (add-drop, equal gaps)', bRow); bAdd.type = 'button';
    function setDrop(v) { s.drop = v; bAll.classList.toggle('on', !v); bAdd.classList.toggle('on', v); sR.set(Math.log10(s.ratio)); }
    bAll.addEventListener('click', function () { setDrop(false); draw(); });
    bAdd.addEventListener('click', function () { setDrop(true); draw(); });
    el('span', 'fbw-tag', 'Scale', bRow);
    var bLin = el('button', 'fbw-btn on', 'Linear', bRow); bLin.type = 'button';
    var bLog = el('button', 'fbw-btn', 'dB', bRow); bLog.type = 'button';
    function setLog(v) { s.log = v; bLin.classList.toggle('on', !v); bLog.classList.toggle('on', v); draw(); }
    bLin.addEventListener('click', function () { setLog(false); });
    bLog.addEventListener('click', function () { setLog(true); });
    el('details', 'fbw-det', '<summary>Model details</summary><p>Coupled-mode model of a ring with clockwise and counter-clockwise modes $a_{cw}$, $a_{ccw}$ at 1550 nm: intrinsic loss rate $\\kappa_0 = \\omega_0 / Q_0$, bus coupling $\\kappa_e$ (a second bus for the add-drop case), backscatter coupling $\\beta$ between the two directions; total $\\kappa = \\kappa_0 + \\kappa_e (+ \\kappa_e)$. With $D = \\kappa/2 - i\\Delta$: through $t = 1 - \\frac{\\kappa_e D}{D^2 + \\beta^2}$, reflection $r = \\frac{i \\beta \\kappa_e}{D^2 + \\beta^2}$. The standing-wave supermodes sit at $\\Delta = \\pm\\beta$, split by $2\\beta$, and the dip visibly splits once $2\\beta$ exceeds about $\\kappa$ (Kippenberg, Spillane and Vahala, Opt. Lett. 2002; first seen by Weiss et al., Opt. Lett. 1995).</p><p>Silicon nitride device: Ji et al., Optica 2017 (radius 115 um, intrinsic Q 37 million, loaded linewidth 5.6 MHz; the nominal bus coupling, and so the under-coupled state, is our inference from those two numbers; the backscatter rate is yours to set). Silicon device: Bahadori, Rumley, Cheng and Bergman, IEEE Optical Interconnects Conference 2018 (radius 5 um, free spectral range 17.2 nm, field coupling 0.1222 per bus, round-trip transmission 0.9998, backscatter 0.0124 per round trip), converted to rates by multiplying each per-round-trip power or field coefficient by the free spectral range; this gives a loaded linewidth near 10 GHz, where the authors quote 12 GHz. The dashed curve is a least-squares single-Lorentzian fit to the through port, with the under-coupled extraction $Q_0 = 2Q_L / (1 + \\sqrt{T_0})$.</p>', wrap);
    un = hint(sB.box, 'drag right');

    function load(k) {
      var p = PRESETS[k]; s.preset = k; s.q0 = p.q0; s.ratio = p.ratio; setDrop(p.drop);
      s.lb = p.beta > 0 ? Math.log10(p.beta / (2 * Math.PI)) : 2;
      sQ.set(Math.log10(s.q0)); sR.set(Math.log10(s.ratio)); sB.set(s.lb);
    }
    function fmtF(hz) { var a = Math.abs(hz); return a >= 1e9 ? (+(hz / 1e9).toPrecision(3)) + ' GHz' : a >= 1e6 ? (+(hz / 1e6).toPrecision(3)) + ' MHz' : (+(hz / 1e3).toPrecision(3)) + ' kHz'; }
    function fmtQ(q) { return q >= 1e6 ? (+(q / 1e6).toPrecision(3)) + ' million' : Math.round(q).toLocaleString('en-US'); }
    function rates() {
      var k0 = W0 / s.q0, ke = s.ratio * k0, ke2 = s.drop ? ke : 0, beta = s.lb <= 2.001 ? 0 : 2 * Math.PI * Math.pow(10, s.lb);
      return { k0: k0, ke: ke, ke2: ke2, k: k0 + ke + ke2, b: beta };
    }
    /* complex helpers on [re, im] */
    function resp(d, R) {
      var Dr = R.k / 2, Di = -d;                       /* D = kappa/2 - i Delta */
      var Er = Dr * Dr - Di * Di + R.b * R.b, Ei = 2 * Dr * Di; /* D^2 + beta^2 */
      var den = Er * Er + Ei * Ei;
      var qr = (Dr * Er + Di * Ei) / den, qi = (Di * Er - Dr * Ei) / den; /* D / (D^2 + beta^2) */
      var tr = 1 - R.ke * qr, ti = -R.ke * qi;
      var D2 = Dr * Dr + Di * Di;
      return { T: tr * tr + ti * ti, Rf: R.ke * R.ke * R.b * R.b / den, Dp: R.ke * R.ke2 * D2 / den, Ad: R.ke * R.ke2 * R.b * R.b / den };
    }
    function lorFit(xs, ts, k) {
      /* T_fit = 1 - A / (1 + (2x/w)^2), center fixed at 0 by symmetry; A by linear least squares for each w */
      function sse(w) {
        var sl = 0, sll = 0, i, L;
        for (i = 0; i < xs.length; i++) { L = 1 / (1 + Math.pow(2 * xs[i] / w, 2)); sl += (1 - ts[i]) * L; sll += L * L; }
        var A = sl / sll, e = 0; for (i = 0; i < xs.length; i++) { L = 1 / (1 + Math.pow(2 * xs[i] / w, 2)); e += Math.pow(1 - A * L - ts[i], 2); }
        return { e: e, A: A, w: w };
      }
      var best = null, lw;
      for (lw = Math.log(0.2 * k); lw <= Math.log(60 * k); lw += 0.02) { var r = sse(Math.exp(lw)); if (!best || r.e < best.e) best = r; }
      var lo = Math.log(best.w) - 0.02, hi = Math.log(best.w) + 0.02;
      for (var it = 0; it < 40; it++) { var m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (sse(Math.exp(m1)).e < sse(Math.exp(m2)).e) hi = m2; else lo = m1; }
      return sse(Math.exp(0.5 * (lo + hi)));
    }
    function draw() {
      var o = fit(), ctx = o.ctx, c = apiCols(api); ctx.fillStyle = c.panel; ctx.fillRect(0, 0, o.w, o.h);
      var R = rates(), half = Math.max(3 * R.k, R.b + 2.5 * R.k), toHz = 1 / (2 * Math.PI);
      var unit = half * toHz >= 2e9 ? 1e9 : half * toHz >= 2e6 ? 1e6 : 1e3, uL = unit === 1e9 ? 'GHz' : unit === 1e6 ? 'MHz' : 'kHz';
      var N = 1201, xs = [], P = { T: [], Rf: [], Dp: [], Ad: [] }, tmin = 2, rmax = 0, dmax = 0;
      for (var i = 0; i < N; i++) {
        var d = -half + 2 * half * i / (N - 1), v = resp(d, R); xs.push(d);
        P.T.push(v.T); P.Rf.push(v.Rf); P.Dp.push(v.Dp); P.Ad.push(v.Ad);
        if (v.T < tmin) tmin = v.T; if (v.Rf > rmax) rmax = v.Rf; if (v.Dp > dmax) dmax = v.Dp;
      }
      var F = lorFit(xs, P.T, R.k);
      var leg = [[c.accent, 'through port |t|^2'], [c.bad, 'reflected toward the laser |r|^2'], [c.warn, 'single-Lorentzian fit to the through port', [5, 4]]];
      if (s.drop) leg.push([c.ok, 'drop port'], [c.muted, 'add port, backscattered (equals the reflection for equal gaps)', [2, 2]]);
      if (R.b > 0) leg.push([c.muted, 'supermodes at +/- beta', [1, 3]]);
      var lo = 0, hi = 1.05, yl = 'power fraction', Yv = function (v) { return v; };
      if (s.log) {
        var vmin = Math.min(tmin, rmax > 0 ? rmax : 1, s.drop && dmax > 0 ? dmax : 1);
        lo = Math.max(-60, Math.floor((10 * Math.log10(Math.max(vmin, 1e-6)) - 20) / 10) * 10); hi = 1; yl = 'power (dB)';
        Yv = function (v) { return Math.max(lo - 5, 10 * Math.log10(Math.max(v, 1e-12))); };
      }
      var A = axes(ctx, o.w, o.h, [-half * toHz / unit, half * toHz / unit], [lo, hi], 'laser detuning from the bare resonance (' + uL + ')', yl, c, { l: 58, r: 24, t: legend(ctx, leg, 58, o.w - 24, false), b: 36 });
      legend(ctx, leg, 58, o.w - 24, true);
      ctx.save(); ctx.beginPath(); ctx.rect(A.L, A.T, A.R - A.L, A.B - A.T); ctx.clip();
      function line(arr, col, wdt, dash) {
        ctx.strokeStyle = col; ctx.lineWidth = wdt; ctx.setLineDash(dash || []); ctx.beginPath();
        for (var j = 0; j < N; j++) { var px = A.X(xs[j] * toHz / unit), py = A.Y(Yv(arr[j])); if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
        ctx.stroke(); ctx.setLineDash([]);
      }
      if (s.drop) { line(P.Dp, c.ok, 1.6); line(P.Ad, c.muted, 1.4, [2, 2]); }
      line(P.Rf, c.bad, 2);
      line(P.T, c.accent, 2.2);
      line(xs.map(function (x) { return 1 - F.A / (1 + Math.pow(2 * x / F.w, 2)); }), c.warn, 1.4, [5, 4]);
      if (R.b > 0) {
        ctx.strokeStyle = c.muted; ctx.lineWidth = 1; ctx.setLineDash([1, 3]);
        [-R.b, R.b].forEach(function (x) { var px = A.X(x * toHz / unit); ctx.beginPath(); ctx.moveTo(px, A.T); ctx.lineTo(px, A.B); ctx.stroke(); });
        ctx.setLineDash([]);
      }
      ctx.restore();

      var QL = W0 / R.k, QLfit = W0 / F.w, T0 = Math.max(0, 1 - F.A), Q0fit = 2 * QLfit / (1 + Math.sqrt(T0));
      var ratio = 2 * R.b / R.k;
      rd.innerHTML =
        '<div><span>Intrinsic Q (true) </span><b>' + fmtQ(s.q0) + '</b></div>' +
        '<div><span>Loaded Q (true) </span><b>' + fmtQ(QL) + '</b></div>' +
        '<div><span>Loaded linewidth </span><b>' + fmtF(R.k * toHz) + '</b></div>' +
        '<div><span>Splitting 2 beta / 2 pi </span><b>' + (R.b > 0 ? fmtF(2 * R.b * toHz) : 'none') + '</b></div>' +
        '<div><span>Splitting / loaded linewidth </span><b>' + fmtN(ratio) + '</b></div>' +
        '<div><span>Peak reflection toward the laser </span><b>' + (rmax > 1e-12 ? (10 * Math.log10(rmax)).toFixed(1) + ' dB' : 'none') + '</b></div>' +
        '<div><span>Single-Lorentzian fit: loaded Q </span><b' + (Math.abs(QLfit / QL - 1) > 0.1 ? ' class="warn"' : '') + '>' + fmtQ(QLfit) + '</b></div>' +
        (s.drop ? '<div><span>Fit, intrinsic Q </span><b>N/A (the extraction formula is for one bus)</b></div>' :
        '<div><span>Fit, under-coupled formula: intrinsic Q </span><b' + (Math.abs(Q0fit / s.q0 - 1) > 0.1 ? ' class="warn"' : '') + '>' + fmtQ(Q0fit) + ' (' + (Q0fit / s.q0 >= 1 ? '+' : '') + Math.round((Q0fit / s.q0 - 1) * 100) + ' %)</b></div>');
      if (!done && ratio >= 1.5) { done = true; api.complete(); }
    }
    var onR = function () { draw(); }; window.addEventListener('resize', onR);
    api.onCleanup(function () { window.removeEventListener('resize', onR); });
    draw();
  }

  window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  window.LMS_WIDGETS.doublet = doublet;
})();
