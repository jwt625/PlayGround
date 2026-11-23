/* Widget eulerSign (ferroelectric course): active versus passive rotation of the lithium niobate electro-optic tensor
   on an x-cut die. Full third-rank tensor rotation; coefficients from Zhu et al., Adv. Opt. Photon. 13, 242 (2021), Table 1.
   Overrides the in-course widget. */
(function () {
  'use strict';
  var W = window.LMS_WIDGETS = window.LMS_WIDGETS || {};
  var SAY_DONE = 'Excellent. A sign error is invisible at zero degrees and loud in between. Know your axes.';
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
  function arrow(ctx, x0, y0, x1, y1, col, lw) {
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw || 2;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    var a = Math.atan2(y1 - y0, x1 - x0), s = 8;
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - s * Math.cos(a - 0.4), y1 - s * Math.sin(a - 0.4));
    ctx.lineTo(x1 - s * Math.cos(a + 0.4), y1 - s * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fill();
  }

  /* ---------- lithium niobate electro-optic tensor (point group 3m, mirror plane normal to x) ---------- */
  var R13 = 9.6, R22 = 6.8, R33 = 30.9, R51 = 32.6;   // pm/V, Zhu et al. 2021, Table 1 (r22 6.8 is unclamped; clamped about 3.4)
  var CON = [                                         // contracted r_Ik, I = xx, yy, zz, yz, xz, xy
    [0, -R22, R13], [0, R22, R13], [0, 0, R33], [0, R51, 0], [R51, 0, 0], [-R22, 0, 0]
  ];
  var IDX = [[0, 5, 4], [5, 1, 3], [4, 3, 2]];
  var RT = [0, 1, 2].map(function (i) { return [0, 1, 2].map(function (j) { return [0, 1, 2].map(function (k) { return CON[IDX[i][j]][k]; }); }); });
  function rotX(th) { var c = Math.cos(th), s = Math.sin(th); return [[1, 0, 0], [0, c, -s], [0, s, c]]; }
  function tr(M) { return [[M[0][0], M[1][0], M[2][0]], [M[0][1], M[1][1], M[2][1]], [M[0][2], M[1][2], M[2][2]]]; }
  // lab-frame r'_{zzz} of the tensor transformed by matrix Q: r'_{ijk} = Q_il Q_jm Q_kn r_lmn
  function reffLab(Q) {
    var s = 0;
    for (var l = 0; l < 3; l++) for (var m = 0; m < 3; m++) for (var n = 0; n < 3; n++) s += Q[2][l] * Q[2][m] * Q[2][n] * RT[l][m][n];
    return s;
  }
  function reff(thDeg, active) { var R = rotX(thDeg * Math.PI / 180); return reffLab(active ? R : tr(R)); }

  W.eulerSign = function (root, params, api) {
    ensureStyle();
    var P = pal(api, root);
    var wrap = el('div', { 'class': 'fw1' }); root.appendChild(wrap);
    var cv = makeCanvas(root, 0.42); wrap.appendChild(cv.c);
    var r1 = el('div', { 'class': 'row' });
    var s1 = el('input', { type: 'range', min: '-90', max: '90', step: '1', value: '0' }); s1.className = 'hint';
    var l1 = el('span', { 'class': 'read' });
    var h1 = hand('drag to 0 or 90, and to an angle in between');
    r1.appendChild(el('span', { 'class': 'lab' }, 'Crystal rotation θ')); r1.appendChild(s1); r1.appendChild(l1); r1.appendChild(h1);
    var read = el('div', { 'class': 'read' });
    var note = el('div', { 'class': 'note' },
      'Model: full rotated electro-optic tensor of lithium niobate; coefficients from Zhu et al., 2021. ' +
      '<details><summary>Model details</summary>' +
      '<span class="tag">Model</span> x-cut die, x out of the page. The waveguide runs along the lab $y\'$ axis; TE light and the electrode gap field both point along the lab $z\'$ axis (in plane, across the waveguide). The crystal is rotated by $\\theta$ about x. ' +
      'Left: the intended convention, rotate the crystal, $r\'_{ijk} = R_{il}R_{jm}R_{kn}\\,r_{lmn}$ with $R = R_x(\\theta)$. Right: the same matrix used as a coordinate rotation by mistake, which is $R^{\\mathsf T} = R_x(-\\theta)$. ' +
      'The plotted quantity is the lab component $r\'_{z\'z\'z\'}$ of the full rotated tensor of lithium niobate (point group 3m), with $r_{13} = 9.6$, $r_{22} = 6.8$, $r_{33} = 30.9$, $r_{51} = 32.6$ pm/V (Zhu et al., Advances in Optics and Photonics 2021, Table 1); this $r_{22}$ is the low-frequency (unclamped) value, the clamped RF value is about 3.4 pm/V (Weis and Gaylord 1985). ' +
      'In closed form $r_{\\mathrm{eff}} = r_{33}\\cos^3\\theta + (r_{13} + 2r_{51})\\sin^2\\theta\\cos\\theta \\pm r_{22}\\sin^3\\theta$: the two conventions differ by $2r_{22}\\sin^3\\theta$, zero at 0 degrees and a sign flip at 90 degrees. The sign of $r_{22}$ itself is tied to the choice of +y (IEEE axis convention), so state that too. $V_\\pi$ scales as $1/r_{\\mathrm{eff}}$ for a fixed mode overlap.' + '</details>');
    [r1, read, note].forEach(function (n) { wrap.appendChild(n); });
    var seen0 = false, seen90 = false, seenGen = false, done = false;

    function panel(ctx, cx, cy, R, ang, title, col) {
      // die outline, lab axes, waveguide; crystal y and z axes rotated by ang (radians, counterclockwise)
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1; ctx.strokeRect(cx - R, cy - R, 2 * R, 2 * R);
      ctx.strokeStyle = P.muted; ctx.lineWidth = 7; ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(cx - R * 0.95, cy); ctx.lineTo(cx + R * 0.95, cy); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.font = '11px sans-serif'; ctx.fillStyle = P.muted; ctx.textAlign = 'center';
      ctx.fillText('waveguide along lab y\'', cx, cy + 16);
      arrow(ctx, cx, cy, cx, cy - R * 0.55, P.warn, 1.6); ctx.fillStyle = P.warn; ctx.fillText('vertical: field and TE (lab z\')', cx, cy + R + 14);
      var zx = Math.cos(Math.PI / 2 + ang), zy = Math.sin(Math.PI / 2 + ang), yx = Math.cos(ang), yy = Math.sin(ang);
      arrow(ctx, cx, cy, cx + R * 0.85 * zx, cy - R * 0.85 * zy, col, 3);
      arrow(ctx, cx, cy, cx + R * 0.6 * yx, cy - R * 0.6 * yy, col, 1.6);
      ctx.fillStyle = col; ctx.font = '12px sans-serif';
      ctx.fillText('z', cx + R * 0.97 * zx, cy - R * 0.97 * zy + 4);
      ctx.fillText('y', cx + R * 0.72 * yx, cy - R * 0.72 * yy + 4);
      ctx.fillStyle = P.fg; ctx.fillText(title, cx, cy - R - 8);
    }
    function draw() {
      P = pal(api, root);
      var th = +s1.value, a = th * Math.PI / 180;
      var ctx = cv.ctx, w = cv.w, h = cv.h; ctx.clearRect(0, 0, w, h);
      var R = Math.min(w * 0.11, h * 0.36), cy = h * 0.52;
      panel(ctx, w * 0.13, cy, R, a, 'Rotate the crystal by +θ', P.accent);
      panel(ctx, w * 0.38, cy, R, -a, 'Matrix used as coordinate rotation', P.danger);
      // r_eff plot
      var x0 = w * 0.58, x1 = w - 14, y0 = 14, y1 = h - 32, rmin = -10, rmax = 45;
      function X(t) { return x0 + (t + 90) / 180 * (x1 - x0); }
      function Y(r) { return y1 - (r - rmin) / (rmax - rmin) * (y1 - y0); }
      ctx.strokeStyle = P.grid; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x1, Y(0)); ctx.stroke();
      ctx.font = '11px sans-serif'; ctx.fillStyle = P.fg; ctx.textAlign = 'center';
      for (var t = -90; t <= 90; t += 45) ctx.fillText(String(t), X(t), y1 + 13);
      ctx.fillText('θ (degrees)', (x0 + x1) / 2, y1 + 26);
      ctx.textAlign = 'right';
      [0, 20, 40].forEach(function (r) { ctx.fillText(String(r), x0 - 4, Y(r) + 4); });
      ctx.save(); ctx.translate(x0 - 30, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('r_eff (pm/V)', 0, 0); ctx.restore();
      ctx.textAlign = 'center';
      [[true, P.accent], [false, P.danger]].forEach(function (L) {
        ctx.strokeStyle = L[1]; ctx.lineWidth = L[0] ? 2.6 : 1.8; if (!L[0]) ctx.setLineDash([6, 4]); ctx.beginPath();
        for (var d = -90; d <= 90; d++) { var y = Y(reff(d, L[0])); if (d === -90) ctx.moveTo(X(d), y); else ctx.lineTo(X(d), y); }
        ctx.stroke(); ctx.setLineDash([]);
      });
      ctx.strokeStyle = P.fg; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(th), y0); ctx.lineTo(X(th), y1); ctx.stroke(); ctx.setLineDash([]);
      var ra = reff(th, true), rp = reff(th, false);
      [[ra, P.accent], [rp, P.danger]].forEach(function (L) { ctx.fillStyle = L[1]; ctx.beginPath(); ctx.arc(X(th), Y(L[0]), 4.5, 0, 2 * Math.PI); ctx.fill(); });
      ctx.fillStyle = P.accent; ctx.fillText('solid: rotate the crystal', (x0 + x1) / 2, Y(-2));
      ctx.fillStyle = P.danger; ctx.fillText('dashed: coordinate rotation by mistake', (x0 + x1) / 2, Y(-2) + 14);
      // readout
      l1.innerHTML = '<b>' + th + '</b> deg';
      var ya = Math.sin(a), yp = -Math.sin(a), z = Math.cos(a);
      var agree = Math.abs(ra - rp) < 1e-9;
      var txt = 'Crystal z in lab components (y\', z\'): intended (' + (-ya).toFixed(3) + ', ' + z.toFixed(3) + '), mistaken (' + (-yp).toFixed(3) + ', ' + z.toFixed(3) + '). ';
      txt += 'r_eff: intended <b>' + ra.toFixed(1) + '</b> pm/V, mistaken <b>' + rp.toFixed(1) + '</b> pm/V. ';
      if (agree) txt += 'The conventions <b>agree</b> here, so a test at this angle cannot catch the error.';
      else if (Math.abs(Math.abs(th) - 90) < 1e-9) txt += 'Same magnitude, <b>opposite sign</b>: a test that checks only magnitude, or which axis the vector lands on, still passes.';
      else txt += 'They <b>disagree</b> by ' + Math.abs(ra - rp).toFixed(1) + ' pm/V, so the predicted Vπ is off by <b>' + (Math.abs(ra / rp - 1) * 100).toFixed(0) + '%</b>' + (ra * rp < 0 ? ' and the sign of the response flips.' : '.');
      read.innerHTML = txt;
    }
    s1.addEventListener('input', function () {
      var d = +s1.value;
      if (d === 0) seen0 = true;
      if (Math.abs(d) === 90) seen90 = true;
      if (Math.abs(d) >= 20 && Math.abs(d) <= 70) seenGen = true;
      draw();
      if (!done && seenGen && (seen0 || seen90)) {
        done = true; s1.classList.remove('hint'); if (h1.parentNode) h1.parentNode.removeChild(h1);
        say(api, SAY_DONE);
        api.complete();
      }
    });
    onTheme(api, draw);
    draw();
  };
})();
