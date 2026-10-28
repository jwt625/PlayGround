// Shared renderer for traced patent figures. Used by the viewer and by pipeline/overlay.py.
// A trace is data/<pn>/traces/fig-<id>.json; all coordinates are pixels of the upright crop
// work/<pn>/figs/fig-<id>.png (origin top-left). Schema: see pipeline/TRACE_SCHEMA.md.
(function (root) {
  const NS = 'http://www.w3.org/2000/svg';
  const mk = (tag, a) => { const n = document.createElementNS(NS, tag); for (const k in a) if (a[k] != null) n.setAttribute(k, a[k]); return n; };
  const pts = d => d.map(p => p[0] + ',' + p[1]).join(' ');

  function defs(svg, uid) {
    const d = mk('defs', {});
    // arrowheads sized in stroke widths so they scale with the line they sit on
    [['end', 'auto'], ['start', 'auto-start-reverse']].forEach(([k, o]) => {
      const m = mk('marker', { id: `${uid}-ar-${k}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: o, markerUnits: 'strokeWidth' });
      m.appendChild(mk('path', { d: 'M0,0 L10,5 L0,10 z', class: 'tr-arrow' }));
      d.appendChild(m);
    });
    const p = mk('pattern', { id: `${uid}-hatch`, width: 14, height: 14, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
    p.appendChild(mk('line', { x1: 0, y1: 0, x2: 0, y2: 14, class: 'tr-hatchline' }));
    d.appendChild(p);
    const st = mk('pattern', { id: `${uid}-stipple`, width: 18, height: 18, patternUnits: 'userSpaceOnUse' });
    [[4, 4], [13, 9], [6, 14]].forEach(([x, y]) => st.appendChild(mk('circle', { cx: x, cy: y, r: 1.6, class: 'tr-arrow' })));
    d.appendChild(st);
    const ds = mk('pattern', { id: `${uid}-dashes`, width: 28, height: 20, patternUnits: 'userSpaceOnUse' });
    ds.appendChild(mk('line', { x1: 3, y1: 5, x2: 13, y2: 5, class: 'tr-hatchline' }));
    ds.appendChild(mk('line', { x1: 17, y1: 15, x2: 27, y2: 15, class: 'tr-hatchline' }));
    d.appendChild(ds);
    svg.appendChild(d);
    return d;
  }

  // hatch with custom pitch/angle (px, degrees), defined once per combination
  function hatchId(ctx, pitch, angle, hw) {
    const id = `${ctx.uid}-hatch-${pitch}-${angle}-${hw || 0}`.replace(/\./g, '_');
    if (!ctx.made.has(id)) {
      ctx.made.add(id);
      const p = mk('pattern', { id, width: pitch, height: pitch, patternUnits: 'userSpaceOnUse', patternTransform: `rotate(${angle})` });
      p.appendChild(mk('line', { x1: 0, y1: 0, x2: 0, y2: pitch, class: 'tr-hatchline', style: hw ? `stroke-width:${hw}` : null }));
      ctx.defs.appendChild(p);
    }
    return id;
  }

  function shape(s, ctx, dflt) {
    const uid = ctx.uid;
    const w = s.w != null ? s.w : dflt;
    const pat = ['hatch', 'stipple', 'dashes'].includes(s.fill);
    const fill = s.fill === 'solid' ? 'tr-fill-solid' : s.fill === 'light' ? 'tr-fill-light' : s.fill === 'paper' ? 'tr-fill-paper' : pat ? 'tr-fill-pat' : 'tr-fill-none';
    const a = { class: `tr-s ${fill}${s.dash ? ' tr-dash' : ''}${s.ink === 'paper' ? ' tr-ink-paper' : ''}`, 'stroke-width': w };
    const patId = s.fill === 'hatch' && (s.pitch || s.angle != null || s.hw) ? hatchId(ctx, s.pitch || 14, s.angle != null ? s.angle : 45, s.hw) : `${uid}-${s.fill}`;
    // inline style: stylesheet rules on .tr-s would override presentation attributes
    a.style = (pat ? `fill:url(#${patId});` : '') + (s.w === 0 ? 'stroke:none;' : '') || null;
    if (s.dash) a['stroke-dasharray'] = Array.isArray(s.dash) ? s.dash.join(' ') : `${w * 4} ${w * 3}`;
    let n;
    switch (s.t) {
      case 'r': n = mk('rect', { ...a, x: s.d[0], y: s.d[1], width: s.d[2] - s.d[0], height: s.d[3] - s.d[1] }); break;
      case 'c': n = mk('circle', { ...a, cx: s.d[0], cy: s.d[1], r: s.d[2] }); break;
      case 'e': n = mk('ellipse', { ...a, cx: s.d[0], cy: s.d[1], rx: s.d[2], ry: s.d[3] }); break;
      case 'p': n = mk('polygon', { ...a, points: pts(s.d) }); break;
      case 'l': n = mk('polyline', { ...a, points: pts(s.d), class: a.class.replace(/tr-fill-\w+/, 'tr-fill-none') }); break;
      case 'path': n = mk('path', { ...a, d: s.d }); break;
      case 't': {
        const sz = s.size || 28;
        n = mk('text', { class: `tr-t${s.italic ? ' tr-it' : ''}${s.bold ? ' tr-b' : ''}${s.ink === 'paper' ? ' tr-ink-paper' : ''}`, x: s.x, y: s.y, 'font-size': sz, 'text-anchor': s.anchor || 'start',
          transform: s.rot ? `rotate(${s.rot} ${s.x} ${s.y})` : null, 'textLength': s.len || null, lengthAdjust: s.len ? 'spacingAndGlyphs' : null });
        // subscripts: "V0_{12}" renders 12 as a subscript
        const parts = String(s.s).split(/(_\{[^}]*\}|\^\{[^}]*\})/);
        parts.forEach(p => {
          if (!p) return;
          const sub = p.startsWith('_{'), sup = p.startsWith('^{');
          // subscript size/drop: per shape, else per trace, else typographic default
          const ss = s.sub || ctx.sub || {}, k = ss.scale != null ? ss.scale : 0.68;
          const shift = sub ? (ss.drop != null ? -ss.drop : 'sub') : (ss.rise != null ? ss.rise : 'super');
          const ts = mk('tspan', sub || sup ? { 'font-size': sz * k, 'baseline-shift': shift } : {});
          ts.textContent = sub || sup ? p.slice(2, -1) : p;
          n.appendChild(ts);
        });
        break;
      }
      default: return null;
    }
    if (s.t === 'l' || s.t === 'path') {
      if (s.arrow === 'end' || s.arrow === 'both') n.setAttribute('marker-end', `url(#${uid}-ar-end)`);
      if (s.arrow === 'start' || s.arrow === 'both') n.setAttribute('marker-start', `url(#${uid}-ar-start)`);
    }
    return n;
  }

  // Render `trace` into `svg` (an <svg> element). Returns { groups: {elementId: <g>}, byRef: {ref: [<g>]} }.
  function renderTrace(svg, trace, opt = {}) {
    const uid = opt.uid || ('t' + Math.random().toString(36).slice(2, 8));
    svg.setAttribute('viewBox', `0 0 ${trace.w} ${trace.h}`);
    const ctx = { uid, defs: defs(svg, uid), made: new Set(), sub: trace.sub };
    const body = mk('g', { class: 'tr-body' });
    svg.appendChild(body);
    const groups = {}, byRef = {};
    const dflt = trace.stroke || 3;
    (trace.elements || []).forEach(el => {
      const g = mk('g', { class: `tr-el tr-k-${el.kind || 'line'}` });
      g.dataset.id = el.id;
      if (el.ref) g.dataset.ref = el.ref;
      if (el.step != null) g.dataset.step = el.step;
      (el.shapes || []).forEach(s => { const n = shape(s, ctx, dflt); if (n) g.appendChild(n); });
      body.appendChild(g);
      groups[el.id] = g;
      if (el.ref) (byRef[el.ref] = byRef[el.ref] || []).push(g);
    });
    return { groups, byRef, uid };
  }

  // End and control points of an absolute-command path (M L H V C S Q T A Z); arc radii/flags are skipped.
  function pathPoints(d) {
    const out = [], toks = d.match(/[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?/g) || [];
    let cmd = 'M', i = 0, x = 0, y = 0;
    const num = () => +toks[i++];
    while (i < toks.length) {
      if (/[A-Za-z]/.test(toks[i])) cmd = toks[i++].toUpperCase();
      if (cmd === 'Z') continue;
      if (i >= toks.length || /[A-Za-z]/.test(toks[i])) continue;
      if (cmd === 'H') x = num();
      else if (cmd === 'V') y = num();
      else if (cmd === 'A') { i += 5; x = num(); y = num(); }
      else { const n = { M: 1, L: 1, T: 1, Q: 2, S: 2, C: 3 }[cmd] || 1; for (let k = 0; k < n; k++) { x = num(); y = num(); if (k < n - 1) out.push([x, y]); } }
      out.push([x, y]);
    }
    return out;
  }

  // Bounding box of an element in trace pixels (for hit areas, fly-to, tags).
  function elementBox(el) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const add = (x, y) => { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); };
    (el.shapes || []).forEach(s => {
      if (s.t === 'r') { add(s.d[0], s.d[1]); add(s.d[2], s.d[3]); }
      else if (s.t === 'c') { add(s.d[0] - s.d[2], s.d[1] - s.d[2]); add(s.d[0] + s.d[2], s.d[1] + s.d[2]); }
      else if (s.t === 'e') { add(s.d[0] - s.d[2], s.d[1] - s.d[3]); add(s.d[0] + s.d[2], s.d[1] + s.d[3]); }
      else if (s.t === 'p' || s.t === 'l') s.d.forEach(p => add(p[0], p[1]));
      else if (s.t === 't') { const sz = s.size || 28, n = String(s.s).replace(/[_^]\{|\}/g, '').length * sz * 0.6;
        const ax = s.anchor === 'middle' ? n / 2 : s.anchor === 'end' ? n : 0;
        if (s.rot === -90 || s.rot === 270) { add(s.x - sz, s.y - n + ax); add(s.x, s.y + ax); }
        else if (s.rot === 90) { add(s.x, s.y - ax); add(s.x + sz, s.y + n - ax); }
        else { add(s.x - ax, s.y - sz); add(s.x - ax + n, s.y + sz * 0.25); } }
      else if (s.t === 'path') pathPoints(s.d).forEach(p => add(p[0], p[1]));
    });
    return isFinite(x0) ? [x0, y0, x1, y1] : null;
  }

  root.TraceRender = { renderTrace, elementBox };
})(typeof window !== 'undefined' ? window : globalThis);
