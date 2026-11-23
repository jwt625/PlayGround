/* Widget biasDrift (ferroelectric course): DC bias point of a lithium niobate MZM over a log time axis (1 us to 1 year).
   Two-layer Maxwell-Wagner stack: field in the crystal after a voltage step = r + (1 - r) exp(-t/tau).
   Time scales from Holzgrafe et al., Opt. Express 32, 3619 (2024). Overrides the in-course widget. */
(function () {
  'use strict';
  var W = window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  var SAY_DONE = 'Thank you. Holding a D C voltage is not the same as holding a bias point. Watch your charge.';
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
    if (a < 1e-3) return num2(s * 1e6) + ' µs';
    if (a < 1) return num2(s * 1e3) + ' ms';
    if (a < 120) return num2(s) + ' s';
    if (a < 7200) return num2(s / 60) + ' min';
    if (a < 2 * 86400) return num2(s / 3600) + ' h';
    if (a < 365.25 * 86400) return num2(s / 86400) + ' days';
    return num2(s / (365.25 * 86400)) + ' years';
  }
  function fmtF(f) {
    if (f >= 1e3) return num2(f / 1e3) + ' kHz';
    if (f >= 1) return num2(f) + ' Hz';
    if (f >= 1e-3) return num2(f * 1e3) + ' mHz';
    return num2(f * 1e6) + ' µHz';
  }

  /* ---------- model ---------- */
  var LT0 = -6, LT1 = Math.log10(365.25 * 86400), NPT = 900;   // log time axis, 1 us to 1 year
  var V0 = 0.5;                                                  // bias for quadrature, in units of V pi (balanced MZM)
  var DEV = {
    clad: { name: 'Thin film, oxide clad', tau: 1 / (2 * Math.PI * 100), r: 0 },
    anneal: { name: 'Thin film, improved contact and anneal', tau: 86400, r: 0 },
    bulk: { name: 'Bulk crystal', tau: 1e7, r: 0 }
  };

  // returns {V, y} in units of V pi at time t for a strategy
  function series(mode, tau, r, rail) {
    var out = [], tc = Infinity, zc = 0;
    if (mode === 'loop' && Math.abs(r - 1) > 1e-9) {
      // unclamped controller output that keeps the crystal field at V0
      if (r > 0) { var vinf = V0 / r; if (vinf > rail) tc = -tau / r * Math.log((vinf - rail) / (vinf - V0)); }
      else tc = tau * (rail / V0 - 1);
      if (isFinite(tc)) zc = (rail - V0) / (1 - r);
    }
    for (var i = 0; i <= NPT; i++) {
      var t = Math.pow(10, LT0 + (LT1 - LT0) * i / NPT), V, y;
      if (mode === 'heat') { V = 0; y = V0; }
      else if (mode === 'hold') { V = V0; y = V0 * (r + (1 - r) * Math.exp(-t / tau)); }
      else {
        if (t < tc) {
          V = Math.abs(r - 1) < 1e-9 ? V0 : (r > 0 ? V0 * (1 / r - (1 / r - 1) * Math.exp(-r * t / tau)) : V0 * (1 + t / tau));
          y = V0;
        } else {
          V = rail;
          var z = rail + (zc - rail) * Math.exp(-(t - tc) / tau);
          y = V - (1 - r) * z;
        }
      }
      var ph = Math.PI * y;
      out.push({ t: t, V: V, y: y, T: Math.pow(Math.cos(ph / 2), 2) });
    }
    return { pts: out, tc: tc };
  }

  W.biasDrift = function (root, params, api) {
    ensureStyle();
    var P = pal(api, root);
    var wrap = el('div', { 'class': 'fw1' }); root.appendChild(wrap);
    var cv = makeCanvas(root, 0.30); wrap.appendChild(cv.c);
    var cv2 = makeCanvas(root, 0.17); wrap.appendChild(cv2.c);

    var rowD = el('div', { 'class': 'row' });
    rowD.appendChild(el('span', { 'class': 'lab' }, 'Device'));
    var db = {};
    Object.keys(DEV).forEach(function (k) { var b = el('button', { type: 'button' }, DEV[k].name); db[k] = b; rowD.appendChild(b); });
    var rowTau = el('div', { 'class': 'row' });
    var sTau = el('input', { type: 'range', min: '-4', max: '8', step: '0.05', value: '0' });
    var lTau = el('span', { 'class': 'read' });
    rowTau.appendChild(el('span', { 'class': 'lab' }, 'Relaxation time')); rowTau.appendChild(sTau); rowTau.appendChild(lTau);
    var rowR = el('div', { 'class': 'row' });
    var sR = el('input', { type: 'range', min: '0', max: '1.5', step: '0.05', value: '0' });
    var lR = el('span', { 'class': 'read' });
    rowR.appendChild(el('span', { 'class': 'lab' }, 'Long-time field ratio r')); rowR.appendChild(sR); rowR.appendChild(lR);
    var rowRail = el('div', { 'class': 'row' });
    var sRail = el('input', { type: 'range', min: '1', max: '20', step: '1', value: '5' });
    var lRail = el('span', { 'class': 'read' });
    rowRail.appendChild(el('span', { 'class': 'lab' }, 'Controller range')); rowRail.appendChild(sRail); rowRail.appendChild(lRail);
    var rowS = el('div', { 'class': 'row' });
    rowS.appendChild(el('span', { 'class': 'lab' }, 'Strategy'));
    var modes = [['hold', 'Hold a constant DC voltage'], ['loop', 'Bias controller (re-adjusts the voltage)'], ['heat', 'Thermo-optic heater']];
    var sb = {};
    modes.forEach(function (m) { var b = el('button', { type: 'button' }, m[1]); b.className = 'hint'; sb[m[0]] = b; rowS.appendChild(b); });
    var hd = hand('run two strategies'); rowS.appendChild(hd);
    var read = el('div', { 'class': 'read' }, 'Pick a strategy. The modulator starts at quadrature at time zero.');
    var note = el('div', { 'class': 'note' },
      'Model: two-layer charge relaxation in the electrode stack; time scales from Holzgrafe et al., 2024. ' +
      '<details><summary>Model details</summary>' +
      '<span class="tag">Model</span> Two-layer Maxwell-Wagner stack: after a voltage step the field in the lithium niobate follows $E(t)/E(0) = r + (1-r)\\,e^{-t/\\tau}$, where $\\tau$ is the stack\'s dielectric relaxation time (one material: $\\tau = \\epsilon_r\\epsilon_0/\\sigma$) and $r$ is the resistive over the capacitive voltage division. ' +
      'Balanced Mach-Zehnder, $T = \\cos^2(\\pi V_{\\mathrm{LN}}/2V_\\pi)$, quadrature at $V_\\pi/2$; an ideal controller raises the applied voltage to keep the crystal field constant until it reaches its range. ' +
      'Time scales from Holzgrafe et al., Optics Express 2024: oxide-clad thin film relaxes within a few milliseconds (response suppressed below about 100 Hz; fitted conductivities about $6\\times10^{-6}$ S/m for the film and $2\\times10^{-10}$ S/m for the cladding, so $r \\approx 0$); controlling the metal to lithium niobate contact (cladding opened under the electrodes, or omitted) and annealing moved the corner from 100 Hz to below 10 mHz, with stable DC tuning over hours and drift over days; bulk devices relax over hours to years. ' +
      'The "improved contact and anneal" and "bulk crystal" presets place $\\tau$ at one day and about four months, inside those reported ranges. ' +
      'Heater: no DC field in the crystal here; of order 100 mW per $2\\pi$ of phase (Holzgrafe et al.). ' +
      'Not modeled: the 2.5 to 6 dB response enhancement near 1 kHz that Holzgrafe et al. attribute to a conductive surface layer, multi-exponential drift, and bias hysteresis after long biasing (Yeh et al., arXiv 2026).' + '</details>');
    [rowD, rowTau, rowR, rowRail, rowS, read, note].forEach(function (n) { wrap.appendChild(n); });

    var dev = 'clad', runs = [], used = {}, done = false, anim = null, tauV = DEV.clad.tau, rV = DEV.clad.r;
    function tau() { return tauV; }
    function rr() { return rV; }
    function rail() { return +sRail.value; }
    function labels() {
      var t = tau();
      lTau.innerHTML = '<b>' + fmtT(t) + '</b> (corner ' + fmtF(1 / (2 * Math.PI * t)) + ')';
      lR.innerHTML = '<b>' + rr().toFixed(2) + '</b> ' + (rr() < 0.999 ? '(field in the crystal decays)' : rr() > 1.001 ? '(field in the crystal grows)' : '(no relaxation)');
      lRail.innerHTML = 'output limited to <b>±' + rail() + '</b> Vπ';
    }
    function setDev(k) {
      dev = k; Object.keys(DEV).forEach(function (q) { db[q].className = q === k ? 'on' : ''; });
      tauV = DEV[k].tau; rV = DEV[k].r; sTau.value = String(Math.log10(tauV)); sR.value = String(rV);
      labels(); recompute(); draw();
    }
    function recompute() { runs = runs.map(function (x) { var s = series(x.mode, tau(), rr(), rail()); return { mode: x.mode, pts: s.pts, tc: s.tc }; }); }

    var LTk = [[-6, '1 µs'], [-3, '1 ms'], [0, '1 s'], [Math.log10(60), '1 min'], [Math.log10(3600), '1 h'], [Math.log10(86400), '1 day'], [LT1, '1 year']];
    function axes(c, x0, x1, y0, y1, P, lastPanel) {
      var ctx = c.ctx;
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.font = '11px sans-serif'; ctx.fillStyle = P.fg; ctx.textAlign = 'center';
      LTk.forEach(function (k) {
        var x = x0 + (k[0] - LT0) / (LT1 - LT0) * (x1 - x0);
        ctx.strokeStyle = P.grid; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]);
        if (lastPanel) { ctx.textAlign = k[0] >= LT1 - 1e-9 ? 'right' : 'center'; ctx.fillText(k[1], k[0] >= LT1 - 1e-9 ? x1 : x, y1 + 14); }
      });
    }
    var COL = function (m) { return m === 'hold' ? P.danger : m === 'loop' ? P.accent : P.warn; };
    function draw(partial) {
      P = pal(api, root);
      var t = tau();
      // panel 1: transmission
      var c = cv, ctx = c.ctx, w = c.w, h = c.h; ctx.clearRect(0, 0, w, h);
      var x0 = 48, x1 = w - 12, y0 = 10, y1 = h - 10;
      function X(tt) { return x0 + (Math.log10(tt) - LT0) / (LT1 - LT0) * (x1 - x0); }
      function Y(T) { return y1 - T * (y1 - y0); }
      axes(c, x0, x1, y0, y1, P, false);
      ctx.fillStyle = P.fg; ctx.textAlign = 'right'; ctx.font = '11px sans-serif';
      ctx.fillText('1', x0 - 4, Y(1) + 4); ctx.fillText('0.5', x0 - 4, Y(0.5) + 4); ctx.fillText('0', x0 - 4, Y(0) + 2);
      ctx.strokeStyle = P.ok; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x0, Y(0.5)); ctx.lineTo(x1, Y(0.5)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = 'left'; ctx.fillStyle = P.ok; ctx.fillText('quadrature', x0 + 6, Y(0.5) - 5);
      ctx.fillStyle = P.fg; ctx.fillText('MZM transmission', x0 + 6, y0 + 13);
      // tau marker
      if (t > 1e-6 && t < Math.pow(10, LT1)) {
        ctx.strokeStyle = P.muted; ctx.beginPath(); ctx.moveTo(X(t), y0); ctx.lineTo(X(t), y1); ctx.stroke();
        ctx.fillStyle = P.muted; ctx.fillText('τ', X(t) + 4, y1 - 6);
      }
      runs.forEach(function (rn, idx) {
        var n = idx === runs.length - 1 && partial != null ? Math.max(1, Math.floor(partial * rn.pts.length)) : rn.pts.length;
        ctx.strokeStyle = COL(rn.mode); ctx.lineWidth = 2.4; ctx.beginPath();
        for (var k = 0; k < n; k++) { var p = rn.pts[k]; if (k === 0) ctx.moveTo(X(p.t), Y(p.T)); else ctx.lineTo(X(p.t), Y(p.T)); }
        ctx.stroke();
      });
      // panel 2: applied voltage
      c = cv2; ctx = c.ctx; w = c.w; h = c.h; ctx.clearRect(0, 0, w, h);
      y0 = 8; y1 = h - 22;
      var vmax = rail() * 1.4;
      function YV(v) { return y1 - (v + 0.1 * vmax) / (1.1 * vmax) * (y1 - y0); }
      axes(c, x0, x1, y0, y1, P, true);
      ctx.strokeStyle = P.danger; ctx.setLineDash([6, 3]); ctx.beginPath(); ctx.moveTo(x0, YV(rail())); ctx.lineTo(x1, YV(rail())); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = P.grid; ctx.beginPath(); ctx.moveTo(x0, YV(0)); ctx.lineTo(x1, YV(0)); ctx.stroke();
      ctx.font = '11px sans-serif'; ctx.fillStyle = P.fg; ctx.textAlign = 'right';
      ctx.fillText(String(rail()), x0 - 4, YV(rail()) + 4); ctx.fillText('0', x0 - 4, YV(0) + 4);
      ctx.textAlign = 'left'; ctx.fillText('applied voltage / Vπ', x0 + 6, y0 + 12);
      ctx.fillStyle = P.danger; ctx.fillText('controller limit', x0 + 6, YV(rail()) + 13);
      runs.forEach(function (rn, idx) {
        var n = idx === runs.length - 1 && partial != null ? Math.max(1, Math.floor(partial * rn.pts.length)) : rn.pts.length;
        ctx.strokeStyle = COL(rn.mode); ctx.lineWidth = 2; ctx.beginPath();
        for (var k = 0; k < n; k++) { var p = rn.pts[k]; if (k === 0) ctx.moveTo(X(p.t), YV(Math.min(vmax, p.V))); else ctx.lineTo(X(p.t), YV(Math.min(vmax, p.V))); }
        ctx.stroke();
      });
      // legend
      ctx.textAlign = 'right'; var lx = x1 - 6;
      [['heat', 'heater'], ['loop', 'controller'], ['hold', 'constant voltage']].forEach(function (L) {
        if (!runs.some(function (q) { return q.mode === L[0]; })) return;
        ctx.fillStyle = COL(L[0]); ctx.fillText(L[1], lx, y0 + 12); lx -= ctx.measureText(L[1]).width + 14;
      });
    }
    function describe() {
      var parts = [];
      runs.forEach(function (rn) {
        if (rn.mode === 'hold') {
          var off = null; for (var i = 0; i < rn.pts.length; i++) if (Math.abs(rn.pts[i].T - 0.5) > 0.1) { off = rn.pts[i].t; break; }
          parts.push('<b>Constant voltage:</b> ' + (off == null ? 'stays within 0.1 of quadrature for the whole year.' : 'leaves quadrature by more than 0.1 after <b>' + fmtT(off) + '</b>; after one year T = <b>' + rn.pts[rn.pts.length - 1].T.toFixed(2) + '</b>.'));
        } else if (rn.mode === 'loop') {
          var vEnd = rn.pts[rn.pts.length - 1].V, vw = vEnd > V0 + 1e-6 ? 'rises to' : vEnd < V0 - 1e-6 ? 'falls to' : 'stays at';
          parts.push('<b>Controller:</b> ' + (isFinite(rn.tc) && rn.tc < Math.pow(10, LT1) ? 'holds quadrature by raising the voltage until it reaches ±' + rail() + ' Vπ after <b>' + fmtT(rn.tc) + '</b>, then the bias walks off.' : 'holds quadrature for the whole year; the voltage it needs ' + vw + ' <b>' + vEnd.toFixed(2) + '</b> Vπ.'));
        } else parts.push('<b>Heater:</b> holds quadrature with no DC field in the crystal; about a quarter of 100 mW per 2π, so of order 25 mW, continuously.');
      });
      var t = tau();
      if (Math.abs(rr() - 1) > 1e-9) parts.push('Relaxation corner ' + fmtF(1 / (2 * Math.PI * t)) + ': a Vπ measured with a triangle wave slower than this reads ' + (rr() < 1 ? 'high' : 'low') + '.');
      read.innerHTML = parts.join('<br>');
    }
    function run(mode) {
      if (anim) cancelAnimationFrame(anim);
      runs = runs.filter(function (x) { return x.mode !== mode; });
      var s = series(mode, tau(), rr(), rail());
      runs.push({ mode: mode, pts: s.pts, tc: s.tc });
      used[mode] = true; sb[mode].classList.remove('hint');
      modes.forEach(function (m) { if (used[m[0]]) sb[m[0]].classList.add('on'); });
      var t0 = performance.now(), D = 1800;
      function step(now) {
        var u = Math.min(1, (now - t0) / D); draw(u);
        if (u < 1) anim = requestAnimationFrame(step);
        else {
          anim = null; describe();
          if (!done && Object.keys(used).length >= 2) {
            done = true; modes.forEach(function (m) { sb[m[0]].classList.remove('hint'); });
            if (hd.parentNode) hd.parentNode.removeChild(hd);
            rowD.appendChild(hand('now try another device'));
            say(api, SAY_DONE);
            api.complete();
          }
        }
      }
      anim = requestAnimationFrame(step);
    }
    Object.keys(DEV).forEach(function (k) { db[k].addEventListener('click', function () { var hh = rowD.querySelector('.hand'); if (hh) hh.parentNode.removeChild(hh); setDev(k); if (runs.length) describe(); }); });
    [sTau, sR, sRail].forEach(function (s) { s.addEventListener('input', function () { if (s === sTau) tauV = Math.pow(10, +sTau.value); if (s === sR) rV = +sR.value; Object.keys(DEV).forEach(function (q) { db[q].className = (Math.abs(DEV[q].tau - tauV) < 1e-9 * DEV[q].tau && Math.abs(DEV[q].r - rV) < 1e-9) ? 'on' : ''; }); labels(); recompute(); draw(); if (runs.length) describe(); }); });
    modes.forEach(function (m) { sb[m[0]].addEventListener('click', function () { run(m[0]); }); });
    api.onCleanup && api.onCleanup(function () { if (anim) cancelAnimationFrame(anim); });
    onTheme(api, function () { draw(); });
    setDev('clad');
  };
})();
