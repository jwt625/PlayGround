/* Widget prSweep (ferroelectric course): time-domain resonance sweep with thermal (red, fast) and
   photorefractive (blue, slow, multi-component) self-shifts. Overrides the in-course widget.
   Model after Sun et al., Opt. Express 25, 13504 (2017): adiabatic cavity, first-order relaxation of each shift.
   Photorefractive relaxation components: Jiang et al., Opt. Lett. 42, 3267 (2017), Table 1 (1548.78 nm pump). */
(function () {
  'use strict';
  var W = window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  var SAY_DONE = 'You scanned both ways, at two speeds. Whether the two directions agree depends on how fast you sweep compared with how fast the crystal responds. Scan both ways.';
  window.LMS_WIDGET_SPEECH = window.LMS_WIDGET_SPEECH || {};
  (window.LMS_WIDGET_SPEECH.ferroelectric = window.LMS_WIDGET_SPEECH.ferroelectric || []).push(SAY_DONE);

  /* ---------- shared helpers (copied per widget file) ---------- */
  function tok(api, n, fb) {
    var v = '';
    try { v = api && api.css ? api.css(n) : getComputedStyle(document.documentElement).getPropertyValue(n); } catch (e) { v = ''; }
    v = v && String(v).trim();
    return v || fb;
  }
  function pal(api, root) {
    var inh = '#cfd6de';
    try { inh = getComputedStyle(root).color || inh; } catch (e) { /* ignore */ }
    return {
      fg: tok(api, '--ink', inh), muted: tok(api, '--ink-3', 'rgba(140,150,160,0.9)'), grid: tok(api, '--line', 'rgba(140,150,160,0.35)'),
      accent: tok(api, '--accent', '#2f7de1'), danger: tok(api, '--bad', '#d9534f'), ok: tok(api, '--ok', '#3a9d5d'), warn: tok(api, '--warn', '#d39b2a')
    };
  }
  function ensureStyle() {
    if (document.getElementById('fe-w1-style')) return;
    var s = document.createElement('style');
    s.id = 'fe-w1-style';
    s.textContent =
      '@keyframes feW1Pulse{0%{box-shadow:0 0 0 0 var(--accent,#2f7de1)}70%{box-shadow:0 0 0 9px rgba(0,0,0,0)}100%{box-shadow:0 0 0 0 rgba(0,0,0,0)}}' +
      '@keyframes feW1Bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}' +
      '.fw1{font:inherit;color:inherit;display:flex;flex-direction:column;gap:8px}' +
      '.fw1 .row{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center}' +
      '.fw1 .lab{min-width:9em;opacity:0.85}' +
      '.fw1 button{font:inherit;color:inherit;background:transparent;border:1px solid currentColor;border-radius:0;padding:4px 10px;cursor:pointer}' +
      '.fw1 button.on{background:var(--accent,#2f7de1);border-color:var(--accent,#2f7de1);color:#fff}' +
      '.fw1 button:disabled{opacity:0.45;cursor:default}' +
      '.fw1 input[type=range]{accent-color:var(--accent,#2f7de1);min-width:160px;flex:1}' +
      '.fw1 label{display:flex;gap:6px;align-items:center;flex-wrap:wrap}' +
      '.fw1 .hint{animation:feW1Pulse 1.3s infinite;outline:2px solid var(--accent,#2f7de1);outline-offset:2px}' +
      '.fw1 .hand{display:inline-flex;align-items:center;gap:4px;color:var(--accent,#2f7de1);font-weight:600;animation:feW1Bob 0.9s infinite;white-space:nowrap}' +
      '.fw1 .note{opacity:0.82;font-size:0.88em;line-height:1.45}' +
      '.fw1 .note details{margin-top:4px}.fw1 .note summary{cursor:pointer;color:var(--accent,#4ea1ff);opacity:1;width:max-content}' +
      '.fw1 .tag{display:inline-block;border:1px solid currentColor;padding:0 6px;font-size:0.8em;opacity:0.85;margin-right:4px}' +
      '.fw1 .read{font-variant-numeric:tabular-nums;line-height:1.5}' +
      '.fw1 .read b{font-weight:600}' +
      '.fw1 canvas{border:1px solid var(--line,rgba(127,127,127,0.3))}';
    document.head.appendChild(s);
  }
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }
  function hand(text) {
    return el('span', { 'class': 'hand' }, '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1v10M3 7l5 6 5-6" fill="none" stroke="currentColor" stroke-width="2"/></svg>' + text);
  }
  function makeCanvas(root, aspect) {
    var w = Math.max(320, Math.min(980, root.clientWidth || 640));
    var h = Math.round(w * aspect);
    var dpr = window.devicePixelRatio || 1;
    var c = document.createElement('canvas');
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    c.style.width = '100%'; c.style.maxWidth = w + 'px'; c.style.height = 'auto'; c.style.display = 'block';
    var ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c: c, ctx: ctx, w: w, h: h };
  }
  function onTheme(api, fn) {
    try {
      var mo = new MutationObserver(fn);
      mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] });
      if (api && api.onCleanup) api.onCleanup(function () { mo.disconnect(); });
    } catch (e) { /* ignore */ }
  }
  function say(api, t) { try { if (api && api.say) return api.say(t); } catch (e) { /* ignore */ } return null; }
  function num2(v) { return Math.abs(v) >= 100 ? String(Math.round(v)) : String(+v.toPrecision(2)); }
  function fmtT(s) {
    var a = Math.abs(s);
    if (a < 1e-3) return num2(s * 1e6) + ' \u00b5s';
    if (a < 1) return num2(s * 1e3) + ' ms';
    if (a < 120) return num2(s) + ' s';
    if (a < 7200) return num2(s / 60) + ' min';
    return num2(s / 3600) + ' h';
  }

  /* ---------- model constants (sources in the on-screen note) ---------- */
  var TAU_T = 10e-6;                         // thermal relaxation, about 10 us (Sun et al. 2017: Gamma_T = 100 kHz)
  var PRC = [                                // Jiang et al. 2017, Table 1, pump 1548.78 nm: amplitude (pm), time constant (s)
    { a: 23.535, t: 25.67e-3 }, { a: 21.250, t: 187.7e-3 }, { a: 24.314, t: 1.959 }, { a: 7.255, t: 28.44 }
  ];
  var PRSUM = PRC.reduce(function (s, c) { return s + c.a; }, 0);
  PRC.forEach(function (c) { c.w = c.a / PRSUM; });
  var LW_PM = 1511.10 / 5.3e5 * 1e3;         // loaded linewidth in pm (Sun et al. 2017: loaded Q 5.3e5 at 1511.10 nm)
  var DEPTH = 0.9;                           // coupling depth about 90 % (Sun et al. 2017)
  var A_TH = 4 * Math.sqrt(3) / 9;           // steady-state bistability threshold of a Lorentzian self-shift (0.770 linewidths)
  var XR = 10, NOUT = 800, BOTH_PR = 1.5;    // span +-10 linewidths; 'Both': photorefractive magnitude = 1.5 x thermal (illustrative)

  function lor(d) { return 1 / (1 + 4 * d * d); }

  W.prSweep = function (root, params, api) {
    ensureStyle();
    var P = pal(api, root);
    var wrap = el('div', { 'class': 'fw1' }); root.appendChild(wrap);
    var cv = makeCanvas(root, 0.42); wrap.appendChild(cv.c);
    var cv2 = makeCanvas(root, 0.15); wrap.appendChild(cv2.c);

    var rowM = el('div', { 'class': 'row' });
    rowM.appendChild(el('span', { 'class': 'lab' }, 'Mechanism'));
    var mechs = [['th', 'Thermal (red shift, fast)'], ['pr', 'Photorefractive (blue shift, slow)'], ['both', 'Both']];
    var mb = {};
    mechs.forEach(function (m) { var b = el('button', { type: 'button' }, m[1]); mb[m[0]] = b; rowM.appendChild(b); });
    var rowP = el('div', { 'class': 'row' });
    var lp = el('label', null, '<span class="lab">Optical power</span>');
    var sp = el('input', { type: 'range', min: '-1.3', max: '1.5', step: '0.02', value: String(Math.log10(0.5)) });
    var lpv = el('span', { 'class': 'read' }, '');
    rowP.appendChild(lp); rowP.appendChild(sp); rowP.appendChild(lpv);
    var rowT = el('div', { 'class': 'row' });
    var lt = el('label', null, '<span class="lab">Sweep time per pass</span>');
    var st = el('input', { type: 'range', min: '-5', max: '3', step: '0.05', value: String(Math.log10(2 * XR * LW_PM * 1e-3 / 6)) });
    var ltv = el('span', { 'class': 'read' }, '');
    rowT.appendChild(lt); rowT.appendChild(st); rowT.appendChild(ltv);
    var rowS = el('div', { 'class': 'row' });
    var bUp = el('button', { type: 'button' }, 'Sweep short to long'), bDn = el('button', { type: 'button' }, 'Sweep long to short');
    var bRep = el('button', { type: 'button' }, 'Ten passes back and forth'), bRst = el('button', { type: 'button' }, 'Relax in the dark (reset)');
    bUp.className = 'hint'; bDn.className = 'hint';
    var hd = hand('sweep both ways');
    [bUp, bDn, hd, bRep, bRst].forEach(function (b) { rowS.appendChild(b); });
    var scales = el('div', { 'class': 'read' });
    var read = el('div', { 'class': 'read' }, 'Sweep to measure. Passes follow each other without a pause, as in a triangle scan; the crystal keeps its charge between passes.');
    var note = el('div', { 'class': 'note' },
      'Model: thermal and photorefractive resonance shifts, each with its own response time; power in linewidths. ' +
      '<details><summary>Model details</summary>' +
      '<span class="tag">Illustrative model</span> The cavity follows the laser adiabatically (photon lifetime about 0.4 ns), intracavity energy $U = 1/(1+4(\\Delta - s)^2)$ with laser detuning $\\Delta$ and total shift $s$ in loaded linewidths. ' +
      'Each shift relaxes toward its steady value, $\\tau_i\\,\\dot s_i = a_i U - s_i$, as in the coupled model of Sun et al., Optics Express 2017. ' +
      'Thermal: red shift, $\\tau_T \\approx 10\\ \\mu s$ (Sun et al.). Photorefractive: blue shift with four components, 26 ms, 0.19 s, 2.0 s and 28 s, weighted by the amplitudes measured by Jiang et al., Optics Letters 2017 (thin-film microdisk, 1549 nm pump); build-up is assumed to share those time constants. ' +
      'Some devices hold photorefractive changes for hours (recovery over 10 hours, Ren et al., npj Nanophotonics 2025); this model forgets after about a minute. ' +
      'Scan rate assumes a 2.85 pm linewidth (loaded Q 5.3e5 at 1511 nm, Sun et al.; their scans ran at about 6 nm/s). ' +
      'Power is normalized as the steady on-resonance shift in linewidths; in "Both" the photorefractive magnitude is set to 1.5 times the thermal one. Steady-state bistability threshold: $4\\sqrt{3}/9 \\approx 0.77$ linewidths.' + '</details>');
    [rowM, rowP, rowT, rowS, scales, read, note].forEach(function (n) { wrap.appendChild(n); });

    var mech = 'pr';
    var sT = 0, sE = [0, 0, 0, 0], tTot = 0, nPass = 0;
    var hist = [];                     // {t, th, pr}
    var last = { 1: null, '-1': null };
    var anim = null, busy = false, done = false, doneDirs = {}, speeds = [], speedHintShown = false;

    function aVals() {
      var p = pw();
      return { aT: mech === 'pr' ? 0 : p, aE: mech === 'th' ? 0 : (mech === 'both' ? BOTH_PR * p : p) };
    }
    function pw() { return Math.pow(10, +sp.value); }
    function Tpass() { return Math.pow(10, +st.value); }
    function key() { return mech + '|' + sp.value + '|' + st.value; }
    function sESum() { return sE[0] + sE[1] + sE[2] + sE[3]; }

    function solveT(xe, a, prev) {
      // root of g(s) = s - a lor(xe - s) nearest the previous value (adiabatic thermal shift)
      if (Math.abs(a) < 1e-12) return 0;
      var lo = Math.min(0, a), hi = Math.max(0, a), s = Math.min(hi, Math.max(lo, prev)), ok = false;
      for (var k = 0; k < 14; k++) {
        var d = xe - s, q = 1 + 4 * d * d, g = s - a / q, gp = 1 - a * 8 * d / (q * q);
        if (Math.abs(gp) < 1e-6) break;
        var sn = s - g / gp;
        if (sn < lo - 1e-9 || sn > hi + 1e-9) break;
        if (Math.abs(sn - s) < 1e-10) { s = sn; ok = true; break; }
        s = sn;
      }
      if (ok && Math.abs(s - prev) < 0.25 * Math.max(1, Math.abs(a))) return s;
      var M = 500, roots = [], ps = lo, pg = lo - a * lor(xe - lo);
      for (var i = 1; i <= M; i++) {
        var x = lo + (hi - lo) * i / M, gx = x - a * lor(xe - x);
        if (pg === 0) roots.push(ps);
        else if (pg * gx < 0) {
          var A = ps, B = x, gA = pg;
          for (var j = 0; j < 40; j++) { var m = (A + B) / 2, gm = m - a * lor(xe - m); if (gA * gm <= 0) B = m; else { A = m; gA = gm; } }
          roots.push((A + B) / 2);
        }
        ps = x; pg = gx;
      }
      if (!roots.length) return ok ? s : prev;
      var best = roots[0];
      roots.forEach(function (r) { if (Math.abs(r - prev) < Math.abs(best - prev)) best = r; });
      return best;
    }

    function simulate(dir) {
      var T = Tpass(), av = aVals(), aT = av.aT, aE = av.aE;
      var dtOut = T / NOUT;
      var explicitT = T <= 0.1;   // resolve the thermal time constant directly when affordable; otherwise it follows adiabatically
      var nsub = Math.max(1, Math.ceil(dtOut / (PRC[0].t / 20)));
      if (explicitT) nsub = Math.max(nsub, Math.ceil(dtOut / (TAU_T / 10)));
      var dt = dtOut / nsub, kT = 1 - Math.exp(-dt / TAU_T), kE = PRC.map(function (c) { return 1 - Math.exp(-dt / c.t); });
      var pts = [], x0 = dir > 0 ? -XR : XR, xPrev = x0, histEvery = Math.max(1, Math.round(NOUT / 200));
      function step(xs) {
        var e = sESum(), U;
        if (explicitT) { U = lor(xs - sT - e); sT += (aT * U - sT) * kT; }
        else { sT = solveT(xs - e, aT, sT); U = lor(xs - sT - e); }
        for (var c = 0; c < 4; c++) sE[c] += (-PRC[c].w * aE * U - sE[c]) * kE[c];
      }
      for (var i = 0; i <= NOUT; i++) {
        var x = x0 + dir * 2 * XR * i / NOUT;
        if (i > 0) for (var k = 1; k <= nsub; k++) step(xPrev + (x - xPrev) * k / nsub);
        xPrev = x;
        var e = sESum();
        if (!explicitT) sT = solveT(x - e, aT, sT);
        var s = sT + e, U = lor(x - s);
        pts.push({ x: x, T: 1 - DEPTH * U, s: s });
        if (i > 0) tTot += dtOut;
        if (i % histEvery === 0) hist.push({ t: tTot, th: sT, pr: e });
      }
      if (hist.length > 6000) hist = hist.filter(function (h, j) { return j % 2 === 0 || j === hist.length - 1; });
      nPass++;
      return { dir: dir, pts: pts, key: key(), pass: nPass, T: T, a: av };
    }

    function apparent(tr) {
      var thr = 1 - DEPTH / 2, xs = [], minT = 2, xMin = 0;
      tr.pts.forEach(function (p) { if (p.T < thr) xs.push(p.x); if (p.T < minT) { minT = p.T; xMin = p.x; } });
      return { w: xs.length ? Math.max.apply(null, xs) - Math.min.apply(null, xs) : 0, c: xMin };
    }

    function draw(partial) {
      P = pal(api, root);
      var ctx = cv.ctx, w = cv.w, h = cv.h; ctx.clearRect(0, 0, w, h);
      var x0 = 46, x1 = w - 12, y0 = 12, y1 = h - 36;
      function X(x) { return x0 + (x + XR) / (2 * XR) * (x1 - x0); }
      function Y(T) { return y1 - T * (y1 - y0); }
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = P.fg; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
      for (var t = -XR; t <= XR; t += 2) { ctx.fillText(String(t), X(t), y1 + 14); ctx.strokeStyle = P.grid; ctx.beginPath(); ctx.moveTo(X(t), y1); ctx.lineTo(X(t), y1 + 3); ctx.stroke(); }
      ctx.fillText('laser detuning from the cold resonance (loaded linewidths); right = longer wavelength', (x0 + x1) / 2, y1 + 29);
      ctx.textAlign = 'right'; ctx.fillText('1', x0 - 5, Y(1) + 4); ctx.fillText('0', x0 - 5, Y(0) + 4);
      ctx.save(); ctx.translate(13, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('transmission', 0, 0); ctx.restore();
      // cold line
      ctx.strokeStyle = P.muted; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.4; ctx.beginPath();
      for (var i = 0; i <= 400; i++) { var xx = -XR + 2 * XR * i / 400, yy = Y(1 - DEPTH * lor(xx)); if (i === 0) ctx.moveTo(X(xx), yy); else ctx.lineTo(X(xx), yy); }
      ctx.stroke(); ctx.setLineDash([]);
      var k = key();
      [1, -1].forEach(function (d) {
        var tr = last[d]; if (!tr) return;
        var n = tr === drawing && partial != null ? Math.max(1, Math.floor(partial * tr.pts.length)) : tr.pts.length;
        ctx.globalAlpha = tr.key === k ? 1 : 0.3;
        ctx.strokeStyle = d > 0 ? P.accent : P.danger; ctx.lineWidth = d > 0 ? 3.4 : 1.6; ctx.beginPath();
        for (var j = 0; j < n; j++) { var p = tr.pts[j]; if (j === 0) ctx.moveTo(X(p.x), Y(p.T)); else ctx.lineTo(X(p.x), Y(p.T)); }
        ctx.stroke();
        if (tr === drawing && partial != null && n < tr.pts.length) {
          var pl = tr.pts[n - 1];
          ctx.fillStyle = d > 0 ? P.accent : P.danger; ctx.beginPath(); ctx.arc(X(pl.x), Y(pl.T), 4, 0, 2 * Math.PI); ctx.fill();
          // current resonance position
          ctx.strokeStyle = P.warn; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(Math.max(-XR, Math.min(XR, pl.s))), y0); ctx.lineTo(X(Math.max(-XR, Math.min(XR, pl.s))), y0 + 10); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      });
      ctx.textAlign = 'left'; ctx.font = '11px sans-serif';
      ctx.fillStyle = P.muted; ctx.fillText('dashed: cold resonance', x0 + 6, y0 + 14);
      ctx.fillStyle = P.accent; ctx.fillText('short to long' + (last[1] ? ' (pass ' + last[1].pass + ')' : ''), x0 + 6, y0 + 28);
      ctx.fillStyle = P.danger; ctx.fillText('long to short' + (last[-1] ? ' (pass ' + last[-1].pass + ')' : ''), x0 + 6, y0 + 42);
      if ((last[1] && last[1].key !== k) || (last[-1] && last[-1].key !== k)) { ctx.fillStyle = P.muted; ctx.fillText('faded: taken with other settings', x0 + 6, y0 + 56); }
      drawStrip();
    }

    function drawStrip() {
      var ctx = cv2.ctx, w = cv2.w, h = cv2.h; ctx.clearRect(0, 0, w, h);
      var x0 = 46, x1 = w - 12, y0 = 8, y1 = h - 22;
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.font = '11px sans-serif'; ctx.fillStyle = P.fg; ctx.textAlign = 'left';
      if (hist.length < 2) { ctx.fillStyle = P.muted; ctx.fillText('Resonance shift versus time since reset appears here.', x0 + 6, (y0 + y1) / 2 + 4); return; }
      var tMax = hist[hist.length - 1].t || 1, sMax = 0.5;
      hist.forEach(function (q) { sMax = Math.max(sMax, Math.abs(q.th), Math.abs(q.pr), Math.abs(q.th + q.pr)); });
      function X(t) { return x0 + t / tMax * (x1 - x0); }
      function Y(s) { return (y0 + y1) / 2 - s / sMax * (y1 - y0) / 2 * 0.92; }
      ctx.strokeStyle = P.grid; ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x1, Y(0)); ctx.stroke();
      [['th', P.warn], ['pr', P.accent], ['tot', P.fg]].forEach(function (L) {
        ctx.strokeStyle = L[1]; ctx.lineWidth = L[0] === 'tot' ? 1.8 : 1.2; ctx.beginPath();
        hist.forEach(function (q, j) { var v = L[0] === 'th' ? q.th : L[0] === 'pr' ? q.pr : q.th + q.pr; if (j === 0) ctx.moveTo(X(q.t), Y(v)); else ctx.lineTo(X(q.t), Y(v)); });
        ctx.stroke();
      });
      ctx.fillStyle = P.fg; ctx.textAlign = 'right';
      ctx.fillText('+' + sMax.toFixed(1), x0 - 4, y0 + 9); ctx.fillText('-' + sMax.toFixed(1), x0 - 4, y1);
      ctx.textAlign = 'left'; ctx.fillText('0', x0 + 3, y1 + 14);
      ctx.textAlign = 'right'; ctx.fillText(fmtT(tMax), x1, y1 + 14);
      ctx.textAlign = 'center'; ctx.fillText('time since reset; resonance shift in linewidths (up = red)', (x0 + x1) / 2, y1 + 14);
      ctx.textAlign = 'left';
      ctx.fillStyle = P.warn; ctx.fillText('thermal', x0 + 6, y0 + 11);
      ctx.fillStyle = P.accent; ctx.fillText('photorefractive', x0 + 58, y0 + 11);
      ctx.fillStyle = P.fg; ctx.fillText('total', x0 + 152, y0 + 11);
    }

    function regime(tLw, tau) { var r = tLw / tau; return r > 10 ? 'follows' : r < 0.1 ? 'frozen' : 'lags'; }
    function updateLabels() {
      var p = pw(), T = Tpass(), tLw = T / (2 * XR), av = aVals();
      lpv.innerHTML = 'steady shift on resonance <b>' + num2(p) + '</b> linewidths (' + (Math.max(av.aT, av.aE) < A_TH ? 'below' : 'above') + ' the 0.77 bistability threshold)';
      var rate = 2 * XR * LW_PM * 1e-3 / T;
      ltv.innerHTML = '<b>' + fmtT(T) + '</b> (' + (rate >= 0.01 ? rate.toPrecision(2) : rate.toExponential(1)) + ' nm/s)';
      var parts = ['Time to cross one linewidth: <b>' + fmtT(tLw) + '</b>.'];
      if (av.aT > 0) {
        var rt = regime(tLw, TAU_T);
        parts.push('Thermal response 10 µs: <b>' + (rt === 'follows' ? 'follows the sweep' : rt === 'frozen' ? 'too slow to follow' : 'lags the sweep') + '</b>.');
      }
      if (av.aE > 0) {
        var fast = regime(tLw, PRC[0].t), slow = regime(tLw, PRC[3].t);
        var txt = slow === 'follows' ? 'follows the sweep' : (T < 0.1 * PRC[0].t ? 'frozen within a pass; it integrates over many passes' : (fast === 'frozen' ? 'mostly frozen; it accumulates over passes' : 'partly follows and lags'));
        parts.push('Photorefractive response 26 ms to 28 s: <b>' + txt + '</b>.');
      }
      scales.innerHTML = parts.join(' ');
    }

    function verdict(tr) {
      var ap = apparent(tr), s = tr.pts[tr.pts.length - 1].s;
      var msg = 'Pass ' + tr.pass + ' (' + (tr.dir > 0 ? 'short to long' : 'long to short') + '): apparent linewidth <b>' + ap.w.toFixed(2) + '</b> (cold 1.00), dip at <b>' + ap.c.toFixed(2) + '</b> linewidths. Shift left in the crystal at the end of the pass: <b>' + s.toFixed(2) + '</b> linewidths (' + (s * LW_PM).toFixed(1) + ' pm).';
      var o = last[-tr.dir];
      if (o && o.key === tr.key) {
        var up = tr.dir > 0 ? tr : o, dn = tr.dir > 0 ? o : tr, md = 0;
        for (var i = 0; i <= NOUT; i++) md = Math.max(md, Math.abs(up.pts[i].T - dn.pts[NOUT - i].T));
        var dc = Math.abs(apparent(up).c - apparent(dn).c);
        var p = pw(), tLw = tr.T / (2 * XR), av = tr.a;
        var follows = (av.aT === 0 || regime(tLw, TAU_T) === 'follows') && (av.aE === 0 || regime(tLw, PRC[3].t) === 'follows');
        var frozen = (av.aT === 0 || regime(tLw, TAU_T) === 'frozen') && (av.aE === 0 || tr.T < 0.1 * PRC[0].t);
        if (md < 0.03) {
          msg += '<br>Up and down traces <b>coincide</b> (largest transmission difference ' + md.toFixed(3) + ', dips ' + dc.toFixed(2) + ' linewidths apart). ';
          if (p < 0.05) msg += 'At this power there is almost no self-shift.';
          else if (follows && Math.max(av.aT, av.aE) < A_TH) msg += 'The sweep is slow compared with every response time and the self-shift is below the bistability threshold, so the shifted line is single valued.';
          else if (frozen) msg += 'The sweep is too fast for the shift to change during a pass, so both directions see the same frozen shift. That is not the cold line: the dip sits ' + ap.c.toFixed(2) + ' linewidths from it.';
          else msg += 'Agreement here does not mean the line is unshifted: compare the dip with the dashed cold resonance.';
        } else {
          msg += '<br>Up and down traces <b>differ</b> (largest transmission difference ' + md.toFixed(2) + ', dips ' + dc.toFixed(2) + ' linewidths apart): hysteresis. ';
          if (Math.max(av.aT, av.aE) >= A_TH) msg += 'The self-shift is above the steady-state bistability threshold' + (follows ? '.' : ', and the shift also lags the sweep.');
          else msg += 'The shift lags the sweep, so the line depends on sweep direction and history even below the steady-state bistability threshold.';
        }
      } else if (o) msg += '<br>The other direction was taken with different settings. Sweep it again to compare.';
      read.innerHTML = msg;
    }

    var drawing = null;
    function setBusy(b) { busy = b; [bUp, bDn, bRep, bRst].forEach(function (x) { x.disabled = b; }); }
    function animatePasses(dirs, msPer, cb) {
      var i = 0;
      function next() {
        if (i >= dirs.length) { drawing = null; draw(); setBusy(false); if (cb) cb(); return; }
        var tr = simulate(dirs[i]); last[tr.dir] = tr; drawing = tr;
        var t0 = performance.now();
        function frame(now) {
          var u = Math.min(1, (now - t0) / msPer); draw(u);
          if (u < 1) anim = requestAnimationFrame(frame);
          else { anim = null; verdict(tr); afterPass(tr); i++; next(); }
        }
        anim = requestAnimationFrame(frame);
      }
      setBusy(true); next();
    }
    function afterPass(tr) {
      if (pw() > 0.1) {
        doneDirs[tr.dir] = true;
        speeds.push(Math.log10(tr.T));
      }
      if (doneDirs[1] && doneDirs[-1]) {
        bUp.classList.remove('hint'); bDn.classList.remove('hint'); if (hd.parentNode) hd.parentNode.removeChild(hd);
        var spread = Math.max.apply(null, speeds) - Math.min.apply(null, speeds);
        if (spread < 2 && !speedHintShown) { speedHintShown = true; st.classList.add('hint'); rowT.appendChild(hand('now a very different sweep time, both ways')); }
        if (!done && spread >= 2) {
          done = true; st.classList.remove('hint'); var hh = rowT.querySelector('.hand'); if (hh) hh.parentNode.removeChild(hh);
          say(api, SAY_DONE); api.complete();
        }
      }
    }
    function lastDir() { var a = last[1], b = last[-1]; if (!a && !b) return -1; if (!a) return -1; if (!b) return 1; return a.pass > b.pass ? 1 : -1; }

    function setMech(m) { mech = m; mechs.forEach(function (q) { mb[q[0]].className = q[0] === m ? 'on' : ''; }); updateLabels(); draw(); }
    mechs.forEach(function (m) { mb[m[0]].addEventListener('click', function () { if (!busy) setMech(m[0]); }); });
    sp.addEventListener('input', function () { updateLabels(); draw(); });
    st.addEventListener('input', function () { updateLabels(); draw(); });
    bUp.addEventListener('click', function () { if (!busy) animatePasses([1], 1300); });
    bDn.addEventListener('click', function () { if (!busy) animatePasses([-1], 1300); });
    bRep.addEventListener('click', function () {
      if (busy) return;
      var d = -lastDir(), seq = [];
      for (var i = 0; i < 10; i++) { seq.push(d); d = -d; }
      animatePasses(seq, 320);
    });
    bRst.addEventListener('click', function () {
      if (busy) return;
      sT = 0; sE = [0, 0, 0, 0]; tTot = 0; nPass = 0; hist = []; last = { 1: null, '-1': null };
      read.innerHTML = 'Charge relaxed. The next pass starts from the cold resonance.'; draw();
    });
    api.onCleanup && api.onCleanup(function () { if (anim) cancelAnimationFrame(anim); });
    onTheme(api, function () { if (!busy) draw(); });
    setMech('pr');
  };
})();
