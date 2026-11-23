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

