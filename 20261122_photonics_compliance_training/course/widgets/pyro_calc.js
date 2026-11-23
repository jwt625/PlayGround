/* Widget pyroCalc (ferroelectric course): pyroelectric field during a hot plate ramp, hold and cool-down,
   with screening, for a z-cut wafer, a z-cut thin film on oxide, or an x-cut wafer. Overrides the in-course widget. */
(function () {
  'use strict';
  var W = window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  var SAY_DONE = 'Thank you. A hotplate step of tens of kelvin can give kilovolts per millimeter before screening. Watch your charge.';
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
    if (a < 120) return num2(s) + ' s';
    if (a < 7200) return num2(s / 60) + ' min';
    if (a < 2 * 86400) return num2(s / 3600) + ' h';
    return num2(s / 86400) + ' days';
  }

  /* ---------- constants (sources in the on-screen note) ---------- */
  var EPS0 = 8.854e-12, EPS33 = 27.9;   // Zhu et al., Adv. Opt. Photon. 13, 242 (2021), Table 1
  var EC = 21;                           // kV/mm, congruent LN coercive field (Volk and Wohlecke 2008)
  var PASCHEN = 327;                     // V, Paschen minimum of air at 1 atm (Paschen's law, air constants)
  var HOLD = 600;                        // s, scenario hold time on the plate
  var STRUCT = {
    zw: { name: 'z-cut wafer on a grounded plate', kind: 'wafer', faces: true },
    zf: { name: 'z-cut thin film on oxide on silicon', kind: 'film', faces: true },
    xw: { name: 'x-cut wafer', kind: 'wafer', faces: false }
  };

  W.pyroCalc = function (root, params, api) {
    ensureStyle();
    var P = pal(api, root);
    var wrap = el('div', { 'class': 'fw1' }); root.appendChild(wrap);
    var cv = makeCanvas(root, 0.36); wrap.appendChild(cv.c);
    var rowG = el('div', { 'class': 'row' }); rowG.appendChild(el('span', { 'class': 'lab' }, 'Structure'));
    var gb = {};
    Object.keys(STRUCT).forEach(function (k) { var b = el('button', { type: 'button' }, STRUCT[k].name); gb[k] = b; rowG.appendChild(b); });
    function sliderRow(label, attrs) {
      var r = el('div', { 'class': 'row' }), s = el('input', attrs), v = el('span', { 'class': 'read' });
      r.appendChild(el('span', { 'class': 'lab' }, label)); r.appendChild(s); r.appendChild(v); return { row: r, s: s, v: v };
    }
    var rT = sliderRow('Plate above room', { type: 'range', min: '1', max: '150', step: '1', value: '10' });
    var rR = sliderRow('Ramp rate', { type: 'range', min: String(Math.log10(0.5)), max: '3', step: '0.02', value: '1' });
    var rD = sliderRow('Thickness', { type: 'range', min: '0.1', max: '1', step: '0.05', value: '0.5' });
    var rS = sliderRow('Screening time', { type: 'range', min: '0', max: '6', step: '0.05', value: '4' });
    var rowP = el('div', { 'class': 'row' }); rowP.appendChild(el('span', { 'class': 'lab' }, 'Pyroelectric coefficient'));
    var bLo = el('button', { type: 'button' }, '4.0e-5 C/(m\u00b2K)'), bHi = el('button', { type: 'button' }, '9.5e-5 C/(m\u00b2K)');
    rowP.appendChild(bLo); rowP.appendChild(bHi);
    var read = el('div', { 'class': 'read' });
    var note = el('div', { 'class': 'note' },
      'Model: pyroelectric charge from heating, screened by conduction. ' +
      '<details><summary>Model details</summary>' +
      '<span class="tag">Model</span> Heating releases bound charge $\\sigma = p\\,\\Delta T$ on the faces normal to the polar axis z. In a planar stack with no free charge the whole pyroelectric field sits in the lithium niobate, $E = \\sigma/(\\epsilon_0\\epsilon_{33})$ with $\\epsilon_{33} = 27.9$ (Zhu et al., Advances in Optics and Photonics 2021); a buried oxide under a thin film carries none, and the top face sits at $V = E\\,d$ above the grounded plate. ' +
      'Conduction screens the field: $\\dot E = (p/\\epsilon_0\\epsilon_{33})\\,\\dot T - E/\\tau_s$. The screening time $\\tau_s$ is not a material constant (surfaces, humidity, coatings, a foil wrap all shorten it); the dielectric relaxation of bulk lithium niobate at room temperature is hours to years (Holzgrafe et al., Optics Express 2024, citing bulk studies), and its rise in conductivity with temperature is not modeled. ' +
      'Coefficient magnitude $4.0\\times10^{-5}$ C/(m$^2$K) (Weis and Gaylord, Applied Physics A 1985, after Savage 1966) or about $9.5\\times10^{-5}$ C/(m$^2$K) (Popescu et al., Journal of Applied Physics 2013), held constant. Coercive field about 21 kV/mm for congruent lithium niobate (Volk and Wohlecke, Lithium Niobate, Springer 2008). Paschen minimum of air about 327 V near a 7.5 \u00b5m gap at 1 atm: below it no spark crosses an air gap, for gaps above a few micrometers (in smaller gaps field emission lowers the breakdown voltage). ' +
      'x-cut: the polar axis lies in the plane, so the faces carry no pyroelectric charge; it collects on the edges normal to z, and that fringing field needs a 3D solve (not computed). Scenario: ramp up, hold 10 minutes, ramp back down at the same rate.' + '</details>');
    [rowG, rT.row, rR.row, rD.row, rS.row, rowP, read, note].forEach(function (n) { wrap.appendChild(n); });
    rT.s.className = 'hint'; var hT = hand('drag'); rT.row.appendChild(hT);

    var st = 'zw', p = 4.0e-5, touchedT = false, touchedR = false, done = false, anim = null, ph = 0;
    function dT() { return +rT.s.value; }
    function rate() { return Math.pow(10, +rR.s.value) / 60; }          // K/s
    function tauS() { return Math.pow(10, +rS.s.value); }
    function thick() { return STRUCT[st].kind === 'film' ? +rD.s.value * 1e-6 : +rD.s.value * 1e-3; } // m

    function simulate() {
      var k = p / (EPS0 * EPS33) / 1e6;  // kV/mm per K
      var R = rate(), D = dT(), tr = D / R, tau = tauS(), tEnd = 2 * tr + HOLD, N = 700, pts = [];
      var faces = STRUCT[st].faces;
      function seg(E0, slope, t) { var a = k * slope * tau; return a + (E0 - a) * Math.exp(-t / tau); }
      var E1 = seg(0, R, tr), E2 = seg(E1, 0, HOLD);
      for (var i = 0; i <= N; i++) {
        var t = tEnd * i / N, T, E;
        if (t <= tr) { T = R * t; E = seg(0, R, t); }
        else if (t <= tr + HOLD) { T = D; E = seg(E1, 0, t - tr); }
        else { T = D - R * (t - tr - HOLD); E = seg(E2, -R, t - tr - HOLD); }
        pts.push({ t: t, T: T, E: faces ? E : 0 });
      }
      var pk = 0, tpk = 0; pts.forEach(function (q) { if (Math.abs(q.E) > Math.abs(pk)) { pk = q.E; tpk = q.t; } });
      return { pts: pts, tEnd: tEnd, tr: tr, pk: pk, tpk: tpk, unscr: k * D, k: k };
    }

    function draw() {
      P = pal(api, root);
      var sim = simulate(), ctx = cv.ctx, w = cv.w, h = cv.h; ctx.clearRect(0, 0, w, h);
      // schematic
      var sx0 = 14, sx1 = w * 0.27, sy = h * 0.52, film = STRUCT[st].kind === 'film', faces = STRUCT[st].faces;
      var last = sim.pts[sim.pts.length - 1], cur = sim.pts[Math.floor((ph % 1) * (sim.pts.length - 1))] || last;
      ctx.fillStyle = 'rgba(150,150,150,0.25)'; ctx.fillRect(sx0 - 6, sy + 30, sx1 - sx0 + 12, 10);
      ctx.fillStyle = P.muted; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('grounded hot plate', (sx0 + sx1) / 2, sy + 54);
      if (film) {
        ctx.fillStyle = 'rgba(120,120,140,0.35)'; ctx.fillRect(sx0, sy + 4, sx1 - sx0, 26);
        ctx.fillStyle = 'rgba(200,200,220,0.25)'; ctx.fillRect(sx0, sy - 8, sx1 - sx0, 12);
        ctx.fillStyle = 'rgba(160,170,190,0.55)'; ctx.fillRect(sx0, sy - 13, sx1 - sx0, 5);
        ctx.fillStyle = P.muted; ctx.textAlign = 'left';
        ctx.fillText('silicon', sx0 + 4, sy + 22); ctx.fillText('oxide', sx0 + 4, sy + 1); ctx.fillText('LN film', sx0 + 4, sy - 17);
      } else {
        ctx.fillStyle = 'rgba(160,170,190,0.35)'; ctx.fillRect(sx0, sy - 22, sx1 - sx0, 52);
        ctx.strokeStyle = P.fg; ctx.strokeRect(sx0, sy - 22, sx1 - sx0, 52);
      }
      var q = Math.abs(cur.E) / Math.max(1e-9, Math.abs(sim.unscr) || 1), n = Math.round(Math.min(1, q) * 9), sgn = cur.E >= 0 ? 1 : -1;
      ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center';
      if (faces) {
        var top = film ? sy - 15 : sy - 25, bot = film ? sy - 4 : sy + 41;
        for (var i = 0; i < n; i++) {
          var x = sx0 + (i + 0.5) * (sx1 - sx0) / Math.max(1, n);
          ctx.fillStyle = sgn > 0 ? P.danger : P.accent; ctx.fillText(sgn > 0 ? '+' : '-', x, top);
          if (!film) { ctx.fillStyle = sgn > 0 ? P.accent : P.danger; ctx.fillText(sgn > 0 ? '-' : '+', x, bot); }
        }
        ctx.fillStyle = P.muted; ctx.font = '11px sans-serif'; ctx.fillText('z normal to the surface', (sx0 + sx1) / 2, 14);
      } else {
        for (var j = 0; j < Math.min(4, n); j++) {
          var yy = sy - 14 + j * 12;
          ctx.fillStyle = sgn >= 0 ? P.danger : P.accent; ctx.fillText(sgn >= 0 ? '+' : '-', sx1 + 8, yy + 4);
          ctx.fillStyle = sgn >= 0 ? P.accent : P.danger; ctx.fillText(sgn >= 0 ? '-' : '+', sx0 - 8, yy + 4);
        }
        ctx.fillStyle = P.muted; ctx.font = '11px sans-serif'; ctx.fillText('z in the plane: charge on the edges', (sx0 + sx1) / 2, 14);
      }
      // plot
      var x0 = w * 0.33, x1 = w - 12, y0 = 12, y1 = h - 34;
      var emax = Math.max(Math.abs(sim.pk) * 1.25, 0.5);
      function X(t) { return x0 + t / sim.tEnd * (x1 - x0); }
      function Y(E) { return (y0 + y1) / 2 - E / emax * (y1 - y0) / 2; }
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x1, Y(0)); ctx.stroke();
      // temperature (scaled to the top half)
      ctx.strokeStyle = P.warn; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]); ctx.beginPath();
      sim.pts.forEach(function (q2, i2) { var yv = Y(0) - q2.T / Math.max(1, dT()) * ((y1 - y0) / 2 * 0.9); if (i2 === 0) ctx.moveTo(X(q2.t), yv); else ctx.lineTo(X(q2.t), yv); });
      ctx.stroke(); ctx.setLineDash([]);
      // coercive lines if on scale
      ctx.font = '11px sans-serif';
      [EC, -EC].forEach(function (e) {
        if (Math.abs(e) <= emax) { ctx.strokeStyle = P.danger; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x0, Y(e)); ctx.lineTo(x1, Y(e)); ctx.stroke(); ctx.setLineDash([]); }
      });
      // field
      ctx.strokeStyle = P.accent; ctx.lineWidth = 2.4; ctx.beginPath();
      sim.pts.forEach(function (q2, i2) { if (i2 === 0) ctx.moveTo(X(q2.t), Y(q2.E)); else ctx.lineTo(X(q2.t), Y(q2.E)); });
      ctx.stroke();
      // cursor
      ctx.fillStyle = P.accent; ctx.beginPath(); ctx.arc(X(cur.t), Y(cur.E), 3.5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = P.fg; ctx.textAlign = 'right';
      ctx.fillText('+' + num2(emax), x0 - 4, y0 + 10); ctx.fillText('0', x0 - 4, Y(0) + 4); ctx.fillText('-' + num2(emax), x0 - 4, y1);
      ctx.textAlign = 'left'; ctx.fillText('0', x0 + 2, y1 + 14);
      ctx.textAlign = 'right'; ctx.fillText(fmtT(sim.tEnd), x1, y1 + 14);
      ctx.textAlign = 'center'; ctx.fillText('time', (x0 + x1) / 2, y1 + 14);
      ctx.textAlign = 'left';
      ctx.fillStyle = P.accent; ctx.fillText('field in the crystal along z (kV/mm)', x0 + 6, y0 + 12);
      ctx.fillStyle = P.warn; ctx.fillText('plate temperature (scaled)', x0 + 6, y0 + 26);
      ctx.fillStyle = P.danger; ctx.fillText(Math.abs(EC) <= emax ? 'dotted: coercive field \u00b121 kV/mm' : 'coercive field 21 kV/mm is off this scale', x0 + 6, y1 - 6);
      return sim;
    }

    function readout(sim) {
      var d = thick(), film = STRUCT[st].kind === 'film', faces = STRUCT[st].faces;
      var sig = p * dT() * 1e2;  // uC/cm^2 : C/m^2 * 1e6 uC / 1e4 cm^2
      var vpk = Math.abs(sim.pk) * 1e6 * d;  // V (kV/mm = 1e6 V/m)
      var vun = sim.unscr * 1e6 * d;
      rT.v.innerHTML = '<b>' + dT() + '</b> K';
      rR.v.innerHTML = '<b>' + num2(rate() * 60) + '</b> K/min (ramp takes ' + fmtT(dT() / rate()) + ')';
      rD.v.innerHTML = film ? '<b>' + (+rD.s.value).toFixed(2) + '</b> \u00b5m film' : '<b>' + (+rD.s.value).toFixed(2) + '</b> mm wafer';
      rS.v.innerHTML = '<b>' + fmtT(tauS()) + '</b>';
      var parts = ['Bound charge released: <b>' + sig.toFixed(3) + '</b> \u00b5C/cm\u00b2 ' + (faces ? 'on each face.' : 'on the edges normal to z.')];
      if (faces) {
        parts.push('Unscreened (instant step): <b>' + sim.unscr.toFixed(2) + '</b> kV/mm, top face at <b>' + num2(vun) + '</b> V.');
        parts.push('With this ramp and screening: peak <b>' + Math.abs(sim.pk).toFixed(2) + '</b> kV/mm (' + (sim.pk >= 0 ? 'while heating' : 'while cooling, opposite sign') + '), top face at <b>' + num2(vpk) + '</b> V. ');
        parts.push('Coercive field: <b>' + (Math.abs(sim.pk) / EC * 100).toFixed(0) + '%</b>' + (Math.abs(sim.pk) >= EC ? ' (domains can flip)' : '') + '. Paschen minimum 327 V: ' + (vpk >= PASCHEN ? '<b>exceeded</b>, a grounded tweezer can draw a spark' : '<b>not reached</b>, no spark across an air gap above a few micrometers') + (film ? '; the thin film keeps the voltage small, not the field.' : '.'));
      } else {
        parts.push('The top face stays near zero potential in this planar model. Edge fields are not computed; compare z-cut for the face-normal case.');
      }
      read.innerHTML = parts.join(' ');
    }

    function loop() { ph += 0.0025; readoutCache(); anim = requestAnimationFrame(loop); }
    var lastKey = '';
    function readoutCache() {
      var sim = draw();
      var k = [st, p, rT.s.value, rR.s.value, rD.s.value, rS.s.value].join('|');
      if (k !== lastKey) { lastKey = k; readout(sim); }
    }
    function setStruct(k) {
      st = k; Object.keys(STRUCT).forEach(function (q) { gb[q].className = q === k ? 'on' : ''; });
      if (STRUCT[k].kind === 'film') { rD.s.min = '0.3'; rD.s.max = '0.9'; rD.s.step = '0.05'; rD.s.value = '0.6'; }
      else { rD.s.min = '0.1'; rD.s.max = '1'; rD.s.step = '0.05'; rD.s.value = '0.5'; }
      lastKey = '';
    }
    function check() {
      if (done || !(touchedT && touchedR)) return;
      done = true; rR.s.classList.remove('hint'); var hh = rR.row.querySelector('.hand'); if (hh) hh.parentNode.removeChild(hh);
      say(api, SAY_DONE); api.complete();
    }
    rT.s.addEventListener('input', function () {
      touchedT = true;
      if (rT.s.classList.contains('hint')) { rT.s.classList.remove('hint'); if (hT.parentNode) hT.parentNode.removeChild(hT); if (!touchedR) { rR.s.classList.add('hint'); rR.row.appendChild(hand('now the ramp rate')); } }
      check();
    });
    rR.s.addEventListener('input', function () { touchedR = true; check(); });
    Object.keys(STRUCT).forEach(function (k) { gb[k].addEventListener('click', function () { setStruct(k); }); });
    bLo.addEventListener('click', function () { p = 4.0e-5; bLo.className = 'on'; bHi.className = ''; });
    bHi.addEventListener('click', function () { p = 9.5e-5; bHi.className = 'on'; bLo.className = ''; });
    bLo.className = 'on';
    api.onCleanup && api.onCleanup(function () { if (anim) cancelAnimationFrame(anim); });
    setStruct('zw');
    loop();
  };
})();
