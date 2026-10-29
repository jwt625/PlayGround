// Patent figure gallery + viewer. Data: window.PATENTS (pipeline/build.py). Traces: window.TraceRender.
// window.SITE: {links: {pn: file}} on the gallery page (light records), {home: file} on a per-patent page.
// Routes: #/  |  #/<pn>/figures/<fig>[/<ref>]  |  #/<pn>/text[/<pid>]  |  #/<pn>/claims  |  #/<pn>/about
(function () {
  const PATS = window.PATENTS, SITE = window.SITE || {}, byPn = Object.fromEntries(PATS.map(p => [p.pn, p]));
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s, r = document) => r.querySelector(s);
  const h = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
  const mk = (tag, a = {}) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); return n; };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const subsup = s => esc(s).replace(/_\{([^}]*)\}/g, '<sub>$1</sub>').replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>');
  const cite = (col, line) => col ? `${col}:${line || '?'}` : '';
  const fmtDate = d => d || '';
  PATS.forEach(p => p.figById = Object.fromEntries(p.figures.map(f => [f.id, f])));

  // ---------- theme ----------
  const themeBtn = $('#themeBtn');
  function curTheme() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  function setTheme(t) { document.documentElement.dataset.theme = t; themeBtn.textContent = t === 'dark' ? 'Light' : 'Dark'; try { localStorage.setItem('cray-theme', t); } catch (e) { } }
  let saved = null; try { saved = localStorage.getItem('cray-theme'); } catch (e) { }
  setTheme((/[?&]theme=(light|dark)/.exec(location.search) || [])[1] || saved || 'dark'); // ?theme= for headless review
  themeBtn.onclick = () => setTheme(curTheme() === 'dark' ? 'light' : 'dark');

  // ---------- tooltip ----------
  const tip = $('#tip');
  function showTip(ev, html) { tip.innerHTML = html; tip.style.display = 'block'; moveTip(ev); }
  function moveTip(ev) { const r = tip.getBoundingClientRect(); let x = ev.clientX + 14, y = ev.clientY + 14; if (x + r.width > innerWidth - 4) x = ev.clientX - r.width - 10; if (y + r.height > innerHeight - 4) y = ev.clientY - r.height - 10; tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
  function hideTip() { tip.style.display = 'none'; }
  const partName = (pat, ref) => (pat.parts[ref] && pat.parts[ref].name) || '';
  const UNNAMED = 'no name in the text';
  const partTip = (pat, ref) => `<span class="num">${esc(ref)}</span>${esc(partName(pat, ref) || UNNAMED)}`;

  // ---------- pan / zoom on an <svg> viewBox ----------
  class PanZoom {
    constructor(box, svg, onChange, onClick) {
      this.box = box; this.svg = svg; this.onChange = onChange || (() => { }); this.full = [0, 0, 100, 100]; this.vb = this.full.slice(); this.anim = 0;
      let drag = null;
      box.addEventListener('mousedown', ev => { if (ev.button === 0 && !ev.target.closest('button')) drag = { x: ev.clientX, y: ev.clientY, vb: this.vb.slice(), moved: false }; });
      window.addEventListener('mousemove', ev => {
        if (!drag) return;
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; box.classList.add('panning'); hideTip(); }
        if (drag.moved) { const k = this.k(); this.set([drag.vb[0] - dx * k, drag.vb[1] - dy * k, drag.vb[2], drag.vb[3]]); }
      });
      window.addEventListener('mouseup', ev => {
        if (!drag) return; const moved = drag.moved; drag = null; box.classList.remove('panning');
        if (!moved && box.contains(ev.target) && onClick) onClick(ev);
      });
      box.addEventListener('wheel', ev => {
        ev.preventDefault();
        const p = this.toUser(ev.clientX, ev.clientY), f = Math.exp(ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015));
        const [x, y, w, hh] = this.vb, nw = Math.min(Math.max(w * f, this.full[2] / 40), this.full[2] * 2.5), nh = nw * hh / w;
        this.set([p[0] - (p[0] - x) * nw / w, p[1] - (p[1] - y) * nh / hh, nw, nh]);
      }, { passive: false });
      box.addEventListener('dblclick', ev => { if (!ev.target.closest('button')) this.fit(); });
      let tp = null;
      box.addEventListener('touchstart', ev => { tp = { t: [...ev.touches].map(t => [t.clientX, t.clientY]), vb: this.vb.slice(), moved: false }; }, { passive: true });
      box.addEventListener('touchmove', ev => {
        if (!tp) return; ev.preventDefault(); tp.moved = true;
        const t = [...ev.touches].map(t => [t.clientX, t.clientY]), r = box.getBoundingClientRect(), k0 = tp.vb[2] / r.width;
        if (t.length === 1 && tp.t.length === 1) this.set([tp.vb[0] - (t[0][0] - tp.t[0][0]) * k0, tp.vb[1] - (t[0][1] - tp.t[0][1]) * k0, tp.vb[2], tp.vb[3]]);
        else if (t.length === 2 && tp.t.length === 2) {
          const d0 = Math.hypot(tp.t[0][0] - tp.t[1][0], tp.t[0][1] - tp.t[1][1]), d1 = Math.hypot(t[0][0] - t[1][0], t[0][1] - t[1][1]);
          const f = d0 / Math.max(d1, 1), cx = (tp.t[0][0] + tp.t[1][0]) / 2 - r.left, cy = (tp.t[0][1] + tp.t[1][1]) / 2 - r.top;
          const ix = tp.vb[0] + cx * k0, iy = tp.vb[1] + cy * k0, nw = tp.vb[2] * f, nh = nw * tp.vb[3] / tp.vb[2];
          this.set([ix - cx * nw / r.width, iy - cy * nh / r.height, nw, nh]);
        }
      }, { passive: false });
      box.addEventListener('touchend', ev => { if (tp && !tp.moved && onClick && ev.changedTouches[0]) onClick({ target: document.elementFromPoint(ev.changedTouches[0].clientX, ev.changedTouches[0].clientY) }); tp = null; });
      new ResizeObserver(() => this.set(this.vb)).observe(box);
    }
    k() { const r = this.box.getBoundingClientRect(); return this.vb[2] / Math.max(r.width, 1); }
    toUser(cx, cy) { const r = this.box.getBoundingClientRect(); return [this.vb[0] + (cx - r.left) * this.vb[2] / r.width, this.vb[1] + (cy - r.top) * this.vb[3] / r.height]; }
    aspect(vb) {
      const r = this.box.getBoundingClientRect(), a = r.height / Math.max(r.width, 1);
      let [x, y, w, hh] = vb; const cx = x + w / 2, cy = y + hh / 2;
      if (!(a > 0)) return vb;
      if (hh / w > a) w = hh / a; else hh = w * a;
      return [cx - w / 2, cy - hh / 2, w, hh];
    }
    set(vb) { this.vb = this.aspect(vb); this.svg.setAttribute('viewBox', this.vb.join(' ')); this.onChange(this); }
    setFull(full) { this.full = full; cancelAnimationFrame(this.anim); this.set(full.map((v, i) => i < 2 ? v - full[2 + i] * 0.02 : v * 1.04)); }
    fit() { const f = this.full; this.animate([f[0] - f[2] * 0.02, f[1] - f[3] * 0.02, f[2] * 1.04, f[3] * 1.04]); }
    animate(to, ms = 450) {
      cancelAnimationFrame(this.anim);
      const from = this.vb.slice(), tgt = this.aspect(to), t0 = performance.now();
      const step = t => {
        const u = Math.min(1, (t - t0) / ms), q = u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        const w = Math.exp(Math.log(from[2]) + (Math.log(tgt[2]) - Math.log(from[2])) * q);
        const cx = from[0] + from[2] / 2 + (tgt[0] + tgt[2] / 2 - from[0] - from[2] / 2) * q;
        const cy = from[1] + from[3] / 2 + (tgt[1] + tgt[3] / 2 - from[1] - from[3] / 2) * q;
        const hh = w * tgt[3] / tgt[2];
        this.set([cx - w / 2, cy - hh / 2, w, hh]);
        if (u < 1) this.anim = requestAnimationFrame(step);
      };
      this.anim = requestAnimationFrame(step);
    }
  }

  // ---------- figure view: scan + trace + hit layer ----------
  const shapeArea = b => b ? (b[2] - b[0]) * (b[3] - b[1]) : 0;
  class FigureView {
    constructor(container, opt = {}) {
      this.el = h('div', 'fv m-trace'); container.appendChild(this.el);
      this.svg = mk('svg', { class: 'stage', preserveAspectRatio: 'xMidYMid meet' }); this.el.appendChild(this.svg);
      this.tag = h('span', 'fvTag'); this.el.appendChild(this.tag);
      this.zoomEl = h('span', 'fvZoom'); this.el.appendChild(this.zoomEl);
      this.opt = opt; this.mode = 'trace'; this.sel = null; this.step = null;
      this.pz = new PanZoom(this.el, this.svg, pz => { if (this.fig) this.zoomEl.textContent = Math.round(100 * this.fig.w / pz.vb[2] / (this.el.clientWidth / this.fig.w || 1) * (this.el.clientWidth / this.fig.w)) / 100 + 'x'; },
        ev => { const g = ev.target && ev.target.closest && ev.target.closest('[data-ref]'); (opt.onClick || (() => { }))(g ? g.dataset.ref : null); });
      this.svg.addEventListener('mouseover', ev => { const g = ev.target.closest('[data-ref]'); if (!g || !this.pat) return; showTip(ev, partTip(this.pat, g.dataset.ref)); (opt.onHover || (() => { }))(g.dataset.ref); });
      this.svg.addEventListener('mousemove', ev => { if (tip.style.display === 'block') moveTip(ev); });
      this.svg.addEventListener('mouseout', ev => { const g = ev.target.closest('[data-ref]'); if (!g) return; const to = ev.relatedTarget && ev.relatedTarget.closest && ev.relatedTarget.closest('[data-ref]'); if (to && to.dataset.ref === g.dataset.ref) return; hideTip(); (opt.onHover || (() => { }))(null); });
    }
    show(pat, fig) {
      if (this.pat === pat && this.fig === fig) return;
      this.pat = pat; this.fig = fig; this.sel = null; this.step = null;
      this.svg.innerHTML = '';
      const vp = mk('g'); this.svg.appendChild(vp);
      const img = mk('image', { class: 'scan', href: fig.img, x: 0, y: 0, width: fig.w, height: fig.h }); vp.appendChild(img);
      this.byRef = {}; this.groups = {}; this.boxes = {};
      if (fig.trace) {
        const tmp = mk('svg');
        const r = TraceRender.renderTrace(tmp, fig.trace, { uid: `u${pat.pn}-${fig.id}-${Math.random().toString(36).slice(2, 6)}` });
        const tg = mk('g', { class: 'trace' }); [...tmp.childNodes].forEach(n => tg.appendChild(n)); vp.appendChild(tg);
        this.byRef = r.byRef; this.groups = r.groups;
        fig.trace.elements.forEach(e => { if (e.ref) { const b = TraceRender.elementBox(e); if (b) (this.boxes[e.ref] = this.boxes[e.ref] || []).push(b); } });
        vp.appendChild(this.buildHit(fig.trace));
        this.maxStep = Math.max(-1, ...fig.trace.elements.map(e => e.step == null ? -1 : e.step));
      } else this.maxStep = -1;
      this.setMode(fig.trace ? this.mode : 'scan', true);
      this.tag.textContent = this.opt.tag ? this.opt.tag(fig) : '';
      this.pz.setFull([0, 0, fig.w, fig.h]);
    }
    buildHit(trace) {
      const g = mk('g', { class: 'hit' }), items = [];
      trace.elements.forEach(e => {
        if (!e.ref) return;
        (e.shapes || []).forEach(s => {
          const b = TraceRender.elementBox({ shapes: [s] }); if (!b) return;
          let n;
          const big = (e.kind === 'outline');
          if (s.t === 't' || e.kind === 'label') { const p = 8; n = mk('rect', { x: b[0] - p, y: b[1] - p, width: b[2] - b[0] + 2 * p, height: b[3] - b[1] + 2 * p, 'pointer-events': 'all' }); }
          else if (s.t === 'l' || (s.t === 'path' && s.fill !== 'solid') || big) {
            n = s.t === 'l' ? mk('polyline', { points: s.d.map(q => q.join(',')).join(' ') }) : s.t === 'path' ? mk('path', { d: s.d }) :
              s.t === 'r' ? mk('rect', { x: s.d[0], y: s.d[1], width: s.d[2] - s.d[0], height: s.d[3] - s.d[1] }) : mk('rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1] });
            n.setAttribute('stroke-width', 22); n.setAttribute('pointer-events', 'stroke'); n.setAttribute('fill', 'none');
          }
          else if (s.t === 'r') n = mk('rect', { x: s.d[0], y: s.d[1], width: s.d[2] - s.d[0], height: s.d[3] - s.d[1], 'pointer-events': 'all' });
          else if (s.t === 'c') n = mk('circle', { cx: s.d[0], cy: s.d[1], r: s.d[2], 'pointer-events': 'all' });
          else if (s.t === 'e') n = mk('ellipse', { cx: s.d[0], cy: s.d[1], rx: s.d[2], ry: s.d[3], 'pointer-events': 'all' });
          else if (s.t === 'p') n = mk('polygon', { points: s.d.map(q => q.join(',')).join(' '), 'pointer-events': 'all' });
          else n = mk('path', { d: s.d, 'pointer-events': 'all' });
          n.dataset.ref = e.ref;
          items.push({ n, a: (s.t === 't' || e.kind === 'label') ? -1 : (n.getAttribute('pointer-events') === 'stroke' ? 0 : shapeArea(b)) });
        });
      });
      // largest areas underneath, strokes above areas, labels on top
      items.sort((p, q) => (q.a === -1) - (p.a === -1) || (p.a === 0) - (q.a === 0) || q.a - p.a);
      items.forEach(i => g.appendChild(i.n));
      return g;
    }
    setMode(m, quiet) {
      if (!this.fig || (!this.fig.trace && m !== 'scan')) m = 'scan';
      this.mode = m; this.el.classList.remove('m-trace', 'm-scan', 'm-overlay'); this.el.classList.add('m-' + m);
      if (!quiet && this.opt.onMode) this.opt.onMode(m);
    }
    highlight(ref) {
      this.sel = ref;
      Object.values(this.groups).forEach(g => g.classList.remove('on'));
      const on = ref && this.byRef[ref];
      this.el.classList.toggle('sel', !!on);
      if (on) on.forEach(g => g.classList.add('on'));
    }
    has(ref) { return !!(this.byRef && this.byRef[ref]); }
    flyTo(ref) {
      const bs = this.boxes[ref]; if (!bs) return;
      const b = bs.reduce((a, c) => [Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.max(a[2], c[2]), Math.max(a[3], c[3])]);
      const f = this.fig, minW = f.w * 0.3;
      const W = Math.max((b[2] - b[0]) * 1.5, minW), H = Math.max((b[3] - b[1]) * 1.5, minW * 0.6);
      const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
      if (W >= f.w * 0.9 && H >= f.h * 0.9) return this.pz.fit();
      this.pz.animate([cx - W / 2, cy - H / 2, W, H]);
    }
    setStep(n) {
      this.step = n;
      if (!this.fig || !this.fig.trace) return;
      this.fig.trace.elements.forEach(e => {
        const g = this.groups[e.id]; if (!g || e.step == null) return;
        g.classList.toggle('fut', n != null && e.step > n);
        g.classList.toggle('now', n != null && e.step === n);
      });
    }
  }

  // ---------- rich text ----------
  function rich(text, spans, q) {
    let out = '', i = 0;
    const plain = s => { let r = subsup(s); if (q) r = r.replace(q, m => `<mark>${m}</mark>`); return r; };
    (spans || []).forEach(([s, e, typ, key]) => {
      if (s < i) return;
      out += plain(text.slice(i, s));
      const t = text.slice(s, e);
      out += typ === 'ref' ? `<a class="rf" data-ref="${esc(key)}">${esc(t)}</a>` : `<a class="fg" data-fig="${esc(key)}">${esc(t)}</a>`;
      i = e;
    });
    return out + plain(text.slice(i));
  }
  function mentionHtml(pat, m, ref) {
    const p = pat.paras[m[0]], s = p.text.slice(m[1], m[2]);
    const spans = p.spans.filter(x => x[0] >= m[1] && x[1] <= m[2]).map(x => [x[0] - m[1], x[1] - m[1], x[2], x[3]]);
    let out = '', i = 0;
    spans.forEach(([a, b, typ, key]) => { out += subsup(s.slice(i, a)); out += typ === 'ref' && key === ref ? `<b>${esc(s.slice(a, b))}</b>` : esc(s.slice(a, b)); i = b; });
    return out + subsup(s.slice(i));
  }

  // ---------- scan page modal ----------
  let modalView = null;
  function openScan(pat, col, label, pg) {
    const page = pg != null && pg !== '' && !isNaN(pg) ? +pg : col ? pat.col1_page + Math.floor((col - 1) / 2) : 0;
    const src = pat.pages[page]; if (!src) return;
    $('#modal').hidden = false;
    $('#modalTitle').innerHTML = `<span class="num">${esc(pat.number)}</span> &nbsp;page ${page + 1} of ${pat.pages.length}${col ? `, column ${col}` : ''}${label ? ` &nbsp;<span class="cite">${esc(label)}</span>` : ''}`;
    const body = $('#modalBody'); body.innerHTML = '';
    const fv = h('div', 'fv'); body.appendChild(fv);
    const svg = mk('svg', { class: 'stage', preserveAspectRatio: 'xMidYMid meet' }); fv.appendChild(svg);
    const [W, H] = (pat.page_wh && pat.page_wh[page]) || [2320, 3408], k = W / 2320;
    svg.appendChild(mk('image', { class: 'scan', href: src, x: 0, y: 0, width: W, height: H }));
    if (col) { const left = col % 2 === 1; svg.appendChild(mk('rect', { class: 'colmark', x: (left ? 120 : 1180) * k, y: 330 * k, width: 1020 * k, height: 2960 * k, opacity: .35 })); }
    modalView = new PanZoom(fv, svg);
    requestAnimationFrame(() => {
      modalView.setFull([0, 0, W, H]);
      if (col) { const left = col % 2 === 1; modalView.set([(left ? 60 : 1120) * k, 200 * k, 1140 * k, 1600 * k]); }
    });
  }
  $('#modalClose').onclick = () => { $('#modal').hidden = true; $('#modalBody').innerHTML = ''; };
  addEventListener('keydown', ev => { if (ev.key === 'Escape' && !$('#modal').hidden) $('#modalClose').onclick(); });

  // ---------- routing ----------
  const main = $('#main');
  let view = null; // current view object: { pat, tab, update(route) }
  function parse(hash = location.hash) {
    const [pn, tab, a, b] = hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent);
    return { pn: byPn[pn] ? pn : null, tab: tab || 'figures', a, b };
  }
  const go = (pat, tab, a, b, replace) => {
    const hsh = '#/' + [pat.pn, tab, a, b].filter(x => x != null && x !== '').map(encodeURIComponent).join('/');
    if (replace) history.replaceState(null, '', hsh); else location.hash = hsh;
  };
  function route(hash) {
    const r = parse(typeof hash === 'string' ? hash : location.hash);
    hideTip();
    if (!r.pn && SITE.home) { go(PATS[0], 'figures', undefined, undefined, true); return route(); }
    if (!r.pn) { const hs = typeof hash === 'string' ? hash : location.hash; if (view && view.gallery) view.update(hs); else { view = null; renderGallery(hs); } return; }
    const pat = byPn[r.pn];
    if (view && view.cleanup) view.cleanup();
    if (!view || view.pat !== pat || view.tab !== r.tab) {
      main.innerHTML = '';
      view = ({ figures: FiguresTab, text: TextTab, claims: ClaimsTab, about: AboutTab }[r.tab] || FiguresTab)(pat);
      view.pat = pat; view.tab = r.tab;
    }
    renderBar(pat, r.tab);
    view.update(r);
  }
  addEventListener('hashchange', route);

  function renderBar(pat, tab) {
    $('#crumb').innerHTML = pat ? `<span class="num">${esc(pat.number)}</span><span class="tt">${esc(pat.title)}</span>` : '';
    $('#tabs').innerHTML = pat ? [['figures', 'Figures'], ['text', 'Description'], ['claims', 'Claims'], ['about', 'About']]
      .map(([k, n]) => `<a href="#/${pat.pn}/${k}" class="${k === tab ? 'on' : ''}">${n}</a>`).join('') : '';
    document.title = pat ? `US ${pat.number}` : 'Patent Figures';
  }

  // ---------- gallery: search, facets, sort, cards / table ----------
  // State lives in the hash: #/?q=..&v=cards|table&s=<sort>&d=asc|desc&group=a|b&decade=..&kind=..&country=..
  const GROUP_ORDER = [...new Set(PATS.map(p => p.group || ''))];
  const firstOrg = p => { const a = (p.assignees || [])[0]; return a ? (a.org || a.name) : ''; };
  const where = p => { const a = (p.assignees || []).find(a => a.place) || (p.inventors || []).find(i => i.place); return a ? a.place : ''; };
  const country = p => { const a = (p.assignees || []).find(a => a.country) || (p.inventors || []).find(i => i.country); return a ? a.country : ''; };
  const surname = n => (n || '').replace(/,?\s+(Jr|Sr|II|III|IV)\.?$/i, '').trim().split(/\s+/).pop() || '';
  const year = d => (d || '').slice(0, 4);
  const pct = (a, b) => b ? Math.round(100 * a / b) : null;
  const FACETS = [
    ['group', 'Group', p => [p.group || '']],
    ['decade', 'Issued', p => year(p.issued) ? [year(p.issued).slice(0, 3) + '0s'] : []],
    ['kind', 'Assignee', p => [...new Set((p.assignees || []).map(a => a.kind).filter(Boolean))]],
    ['country', 'Country', p => country(p) ? [country(p)] : []],
    ['status', 'Status', p => p.status ? [p.status.state] : []],
  ];
  const SORTS = {
    group: ['Group', p => [GROUP_ORDER.indexOf(p.group || ''), p.issued || '']],
    issued: ['Issued', p => p.issued || ''], filed: ['Filed', p => p.filed || ''],
    number: ['Patent number', p => +p.pn], title: ['Title', p => (p.title || '').toLowerCase()],
    inventor: ['First inventor', p => surname(((p.inventors || [])[0] || {}).name).toLowerCase()],
    assignee: ['Assignee', p => firstOrg(p).toLowerCase()], place: ['Location', p => where(p).toLowerCase()],
    figures: ['Figures', p => p.counts.figures], traced: ['Traced', p => pct(p.counts.traced, p.counts.figures)],
    claims: ['Claims', p => p.counts.claims], parts: ['Numbered parts', p => p.counts.parts],
    term: ['Term end', p => (p.status || {}).term_end || ''],
  };
  const statusTip = s => s ? `${esc(s.state)} ${esc(s.term_end)}: ${esc(s.basis)}${s.google_expiration && s.google_expiration !== s.term_end ? `; Google Patents lists ${esc(s.google_expiration)}` : ''}; checked ${esc(s.checked)}` : 'unknown';
  PATS.forEach(p => {
    if (!p.counts) return;
    p.hay = [p.pn, p.number, p.title, p.group, p.issued, p.filed, p.abstract,
      ...(p.inventors || []).flatMap(i => [i.name, i.place]), ...(p.assignees || []).flatMap(a => [a.name, a.org, a.kind, a.place]),
      ...(p.partnames || [])].filter(Boolean).join(' \u0001 ').toLowerCase();
    p.facet = Object.fromEntries(FACETS.map(([k, , fn]) => [k, fn(p)]));
  });
  function galleryState(hash = location.hash) {
    const qs = new URLSearchParams((hash.split('?')[1]) || '');
    const st = { q: qs.get('q') || '', v: qs.get('v') === 'table' ? 'table' : 'cards', s: SORTS[qs.get('s')] ? qs.get('s') : 'group', d: qs.get('d') === 'desc' ? -1 : 1, f: {} };
    FACETS.forEach(([k]) => st.f[k] = new Set((qs.get(k) || '').split('|').filter(Boolean)));
    return st;
  }
  function galleryHash(st) {
    const qs = new URLSearchParams();
    if (st.q) qs.set('q', st.q);
    if (st.v !== 'cards') qs.set('v', st.v);
    if (st.s !== 'group') qs.set('s', st.s);
    if (st.d < 0) qs.set('d', 'desc');
    FACETS.forEach(([k]) => { if (st.f[k].size) qs.set(k, [...st.f[k]].join('|')); });
    const s = qs.toString();
    return '#/' + (s ? '?' + s : '');
  }
  const cmp = (a, b) => Array.isArray(a) ? (cmp(a[0], b[0]) || cmp(a[1], b[1])) : (a == null || a === '') - (b == null || b === '') || (a < b ? -1 : a > b ? 1 : 0);

  function renderGallery(hash) {
    renderBar(null);
    main.innerHTML = '';
    const st = galleryState(hash);
    const g = h('div', 'gallery'); main.appendChild(g);
    const bar = h('div', 'gbar'), facets = h('div', 'gfacets'), out = h('div', 'gout');
    g.append(bar, facets, out);
    const q = h('input'); q.type = 'search'; q.placeholder = 'Search patents, people, places, parts'; q.value = st.q;
    const views = h('span', 'grp');
    const vb = { cards: h('button', '', 'Cards'), table: h('button', '', 'Table') };
    views.append(vb.cards, vb.table);
    const sortSel = h('select'); Object.entries(SORTS).forEach(([k, [n]]) => sortSel.appendChild(new Option(n, k)));
    const dirBtn = h('button'); dirBtn.title = 'Sort direction';
    const count = h('span', 'lbl num');
    bar.append(q, h('span', 'lbl', 'Sort'), sortSel, dirBtn, views, h('span', 'spacer'), count);

    const commit = () => { history.replaceState(null, '', galleryHash(st)); draw(); };
    q.oninput = () => { st.q = q.value.trim(); commit(); };
    sortSel.onchange = () => { st.s = sortSel.value; st.d = 1; commit(); };
    dirBtn.onclick = () => { st.d = -st.d; commit(); };
    Object.entries(vb).forEach(([k, b]) => b.onclick = () => { st.v = k; commit(); });
    const onKey = ev => { if (ev.key === '/' && document.activeElement !== q && view && view.gallery) { ev.preventDefault(); q.focus(); } };
    addEventListener('keydown', onKey);

    function matches(p, skip) {
      const toks = st.q.toLowerCase().split(/\s+/).filter(Boolean);
      if (toks.some(t => !p.hay.includes(t.replace(/,/g, '')) && !p.hay.includes(t))) return false;
      return FACETS.every(([k]) => k === skip || !st.f[k].size || p.facet[k].some(v => st.f[k].has(v)));
    }
    function drawFacets() {
      facets.innerHTML = '';
      FACETS.forEach(([k, name]) => {
        const vals = new Map();
        PATS.forEach(p => p.facet[k].forEach(v => vals.set(v, 0)));
        PATS.filter(p => matches(p, k)).forEach(p => p.facet[k].forEach(v => vals.set(v, vals.get(v) + 1)));
        if (vals.size < 2 && !st.f[k].size) return;
        const order = k === 'group' ? GROUP_ORDER : [...vals.keys()].sort();
        const row = h('div', 'frow'); row.appendChild(h('span', 'lbl', name));
        order.filter(v => vals.has(v)).forEach(v => {
          const b = h('button', 'fchip' + (st.f[k].has(v) ? ' on' : ''), `${esc(v || 'Other')} <span class="num">${vals.get(v)}</span>`);
          b.onclick = () => { st.f[k].has(v) ? st.f[k].delete(v) : st.f[k].add(v); commit(); };
          row.appendChild(b);
        });
        facets.appendChild(row);
      });
      if (FACETS.some(([k]) => st.f[k].size) || st.q) {
        const clr = h('button', 'fchip', 'Clear'); clr.onclick = () => { st.q = q.value = ''; FACETS.forEach(([k]) => st.f[k].clear()); commit(); };
        (facets.lastChild || facets).appendChild(clr);
      }
    }
    function draw() {
      sortSel.value = st.s; dirBtn.textContent = st.d > 0 ? 'Asc' : 'Desc';
      Object.entries(vb).forEach(([k, b]) => b.classList.toggle('on', st.v === k));
      drawFacets();
      const key = SORTS[st.s][1];
      const rows = PATS.filter(p => matches(p)).sort((a, b) => st.d * cmp(key(a), key(b)) || cmp(+a.pn, +b.pn));
      count.textContent = `${rows.length} of ${PATS.length}`;
      out.innerHTML = '';
      if (!rows.length) { out.appendChild(h('div', 'gnone lbl', 'No match')); return; }
      (st.v === 'table' ? drawTable : drawCards)(rows);
    }
    const link = (p, fig) => `${(SITE.links || {})[p.pn] || ''}#/${p.pn}/figures/${encodeURIComponent(fig.id)}`;
    const frontFig = p => p.figures.find(f => f.front) || p.figures[0];

    function drawCards(rows) {
      const io = new IntersectionObserver(es => es.forEach(e => {
        if (!e.isIntersecting) return;
        io.unobserve(e.target); const p = byPn[e.target.dataset.pn], fig = frontFig(p);
        const s = mk('svg', { class: 'fv' }); e.target.appendChild(s);
        e.target.classList.add('fv', 'm-trace');
        TraceRender.renderTrace(s, fig.trace, { uid: 'g' + p.pn });
        s.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      }), { rootMargin: '400px' });
      const grouped = st.s === 'group';
      let cards = null, cur = null;
      rows.forEach(p => {
        if (!cards || (grouped && (p.group || '') !== cur)) {
          cur = p.group || ''; cards = out.appendChild(h('div', 'cards'));
          if (grouped && cur) cards.appendChild(h('div', 'ghead', esc(cur)));
        }
        const fig = frontFig(p);
        const c = h('a', 'card'); c.href = link(p, fig);
        const th = h('div', 'thumb');
        if (fig.trace) { th.dataset.pn = p.pn; io.observe(th); }
        else { const im = h('img'); im.src = fig.img; im.style.filter = 'var(--scan-filter)'; im.style.objectFit = 'contain'; th.appendChild(im); }
        c.appendChild(th);
        const orgs = (p.assignees || []).map(a => `${esc(a.org || a.name)}${a.place ? ` <span class="pl">${esc(a.place)}</span>` : ''}`).join('; ');
        c.appendChild(h('div', 'meta', `
          <div class="row"><span class="num">US ${esc(p.number)}</span><span class="num">${esc(fmtDate(p.issued))}</span></div>
          <div class="t">${esc(p.title)}</div>
          <div class="row"><span>${(p.inventors || []).map(i => esc(i.name)).join(', ')}</span></div>
          <div class="row"><span>${orgs || '<span class="unk">No assignee</span>'}</span></div>
          <div class="counts num">${p.counts.figures} figures &middot; ${p.counts.claims} claims &middot; ${p.counts.parts} numbered parts &middot; filed ${esc(fmtDate(p.filed))}</div>`));
        cards.appendChild(c);
      });
    }

    function drawTable(rows) {
      const wrap = h('div', 'gtable'), t = h('table');
      const cols = [
        ['number', 'Patent', p => `<span class="num">${esc(p.number)}</span>`, 'num'],
        ['issued', 'Issued', p => esc(p.issued || ''), 'num'],
        ['filed', 'Filed', p => esc(p.filed || ''), 'num'],
        ['title', 'Title', p => esc(p.title), 'tt'],
        ['inventor', 'Inventors', p => (p.inventors || []).map(i => esc(i.name)).join(', ')],
        ['assignee', 'Assignee', p => esc(firstOrg(p)) || '<span class="unk">none</span>'],
        ['place', 'Location', p => esc(where(p)) || '<span class="unk">unknown</span>'],
        ['group', 'Group', p => esc(p.group || '')],
        ['term', 'Status', p => p.status ? `<span class="st" title="${statusTip(p.status)}">${esc(p.status.state)} <span class="num">${esc(p.status.term_end)}</span></span>` : '<span class="unk">unknown</span>'],
        ['figures', 'Figs', p => p.counts.figures, 'r'],
        ['traced', 'Traced', p => `${pct(p.counts.traced, p.counts.figures)} %`, 'r'],
        ['claims', 'Claims', p => p.counts.claims, 'r'],
        ['parts', 'Parts', p => p.counts.parts, 'r'],
      ];
      t.innerHTML = `<thead><tr>${cols.map(([k, n, , c]) => `<th data-k="${k}" class="${c === 'r' ? 'r' : ''}${st.s === k ? ' on' : ''}">${n}${st.s === k ? (st.d > 0 ? ' &#9650;' : ' &#9660;') : ''}</th>`).join('')}</tr></thead>`;
      const tb = h('tbody'); t.appendChild(tb);
      rows.forEach(p => {
        const tr = h('tr'); tr.dataset.pn = p.pn;
        tr.innerHTML = cols.map(([k, , fn, c]) => `<td class="${c || ''}">${fn(p)}</td>`).join('');
        tb.appendChild(tr);
      });
      t.querySelector('thead').onclick = ev => {
        const th = ev.target.closest('th'); if (!th) return;
        if (st.s === th.dataset.k) st.d = -st.d; else { st.s = th.dataset.k; st.d = 1; }
        commit();
      };
      tb.onclick = ev => { const tr = ev.target.closest('tr'); if (tr) { const p = byPn[tr.dataset.pn]; location.href = link(p, frontFig(p)); } };
      tb.onmouseover = ev => {
        const tr = ev.target.closest('tr'); if (!tr) return;
        const p = byPn[tr.dataset.pn], fig = frontFig(p);
        const ppl = (p.inventors || []).map(i => `<div>${esc(i.name)}${i.place ? ` <span class="pl">${esc(i.place)}</span>` : ''}</div>`).join('');
        showTip(ev, `<img class="tthumb" src="${fig.img}" alt=""><div class="tmeta"><span class="num">US ${esc(p.number)}</span> ${esc(fig.label || '')}</div>${ppl}`);
      };
      tb.onmousemove = ev => { if (tip.style.display === 'block') moveTip(ev); };
      tb.onmouseleave = hideTip;
      wrap.appendChild(t); out.appendChild(wrap);
    }

    draw();
    view = { gallery: true, update(hash) { const s2 = galleryState(hash); Object.assign(st, s2); if (document.activeElement !== q) q.value = st.q; draw(); }, cleanup() { removeEventListener('keydown', onKey); } };
  }

  // ---------- figures tab ----------
  function FiguresTab(pat) {
    const root = h('div', 'figs'); main.appendChild(root);
    const list = h('div', 'flist'), center = h('div', 'center'), side = h('div', 'side');
    root.append(list, center, side);
    const tools = h('div', 'tools'); center.appendChild(tools);
    const stageBox = h('div', ''); stageBox.style.cssText = 'position:relative;min-height:0'; center.appendChild(stageBox);
    const info = h('div', 'info'), plist = h('div', 'plist'); side.append(info, plist);
    const st = { fig: null, pin: null, hover: null, tour: null, stepTimer: null };
    const fv = new FigureView(stageBox, {
      onHover: ref => { st.hover = ref; paint(); },
      onClick: ref => { stopTour(); pin(ref && ref === st.pin ? null : ref, false); },
      onMode: () => drawTools(),
    });
    fv.el.style.cssText = 'position:absolute;inset:0';
    pat.figures.forEach(f => {
      const it = h('div', 'fitem', `<img src="${f.thumb}" alt=""><div><div class="fl">${esc(f.label)}</div><div class="fs">Sheet ${esc(f.sheet || '')}${f.trace ? '' : ' &middot; scan only'}</div></div>`);
      it.dataset.fig = f.id; it.onclick = () => go(pat, 'figures', f.id); list.appendChild(it);
    });

    function pin(ref, fly = true) {
      st.pin = ref; go(pat, 'figures', st.fig.id, ref || undefined, true);
      paint(); if (ref && fly) fv.flyTo(ref);
    }
    function paint() {
      const ref = st.hover || st.pin;
      fv.highlight(ref);
      plist.querySelectorAll('.prow').forEach(r => r.classList.toggle('on', r.dataset.ref === ref));
      renderInfo(ref);
    }
    function renderInfo(ref) {
      const f = st.fig;
      if (!ref) {
        const note = pat.notes && pat.notes.figs && pat.notes.figs[f.id];
        const paras = f.paras.map(pid => pat.paras[pid]);
        info.innerHTML = `<h2>${esc(f.label)}</h2>
          ${f.brief.text ? `<p class="quote">${subsup(f.brief.text)} <span class="cite" data-col="${f.brief.col || ''}">${cite(f.brief.col, f.brief.line)}</span></p>` : ''}
          ${note ? `<div class="note"><span class="lbl">Note</span>${subsup(note)}</div>` : ''}
          ${paras.length ? `<div class="lbl sub">Referenced in ${paras.length} paragraph${paras.length > 1 ? 's' : ''}</div><div class="mlist">${paras.map(p =>
            `<div class="mitem" data-pid="${p.id}"><span class="cite">${cite(p.col, p.line)}</span> ${subsup(p.text.slice(0, 150))}${p.text.length > 150 ? '...' : ''}</div>`).join('')}</div>` : ''}`;
      } else {
        const p = pat.parts[ref] || { name: '', mentions: [], figs: [] };
        const note = pat.notes && pat.notes.parts && pat.notes.parts[ref];
        const figs = pat.figures.filter(x => (p.figs || []).includes(x.id) || (x.trace && x.trace.elements.some(e => e.ref === ref)));
        info.innerHTML = `<h2><span class="num">${esc(ref)}</span>${esc(p.name || UNNAMED)}</h2>
          ${p.alt && p.alt.length ? `<div class="alt">also: ${p.alt.map(esc).join('; ')}</div>` : ''}
          <div class="chips">${figs.map(x => `<span class="chip${x === f ? ' on' : ''}" data-fig="${x.id}">${esc(x.label)}</span>`).join('')}</div>
          ${note ? `<div class="note"><span class="lbl">Note</span>${subsup(note)}</div>` : ''}
          <div class="lbl sub">${p.mentions.length} mention${p.mentions.length === 1 ? '' : 's'} in the description</div>
          <div class="mlist">${p.mentions.map(m => `<div class="mitem" data-pid="${m[0]}"><span class="cite">${cite(pat.paras[m[0]].col, pat.paras[m[0]].line)}</span> ${mentionHtml(pat, m, ref)}</div>`).join('')}</div>`;
      }
      info.querySelectorAll('.mitem').forEach(n => n.onclick = () => go(pat, 'text', n.dataset.pid));
      info.querySelectorAll('.chip[data-fig]').forEach(n => n.onclick = () => go(pat, 'figures', n.dataset.fig, ref));
      info.querySelectorAll('.quote .cite').forEach(n => n.onclick = () => openScan(pat, +n.dataset.col, n.textContent));
    }
    function buildParts() {
      const f = st.fig;
      plist.innerHTML = `<div class="head"><span class="lbl">Parts</span><span class="lbl num">${f.refs.length}</span></div>`;
      f.refs.forEach(ref => {
        const r = h('div', 'prow', `<span class="num">${esc(ref)}</span><span>${esc(partName(pat, ref) || UNNAMED)}</span>`);
        r.dataset.ref = ref;
        r.onmouseenter = () => { st.hover = ref; paint(); };
        r.onmouseleave = () => { st.hover = null; paint(); };
        r.onclick = () => { stopTour(); pin(ref === st.pin ? null : ref); };
        if (!fv.has(ref)) r.style.color = 'var(--text-muted)';
        plist.appendChild(r);
      });
    }
    // tour through parts in order of first mention; step through timing figures
    function tourRefs() { return st.fig.refs.filter(r => fv.has(r)); }
    function tourStep(d) {
      const rs = tourRefs(); if (!rs.length) return;
      let i = rs.indexOf(st.pin); i = i < 0 ? (d > 0 ? 0 : rs.length - 1) : (i + d + rs.length) % rs.length;
      pin(rs[i]);
      const row = plist.querySelector(`.prow[data-ref="${CSS.escape(rs[i])}"]`); if (row) row.scrollIntoView({ block: 'nearest' });
    }
    function stopTour() { if (st.tour) { clearInterval(st.tour); st.tour = null; drawTools(); } }
    function stopStep() { if (st.stepTimer) { clearInterval(st.stepTimer); st.stepTimer = null; } }
    function drawTools() {
      const f = st.fig, has = !!f.trace;
      tools.innerHTML = '';
      const modes = h('span', 'grp');
      [['trace', 'Redraw'], ['scan', 'Scan'], ['overlay', 'Overlay']].forEach(([m, n]) => {
        const b = h('button', fv.mode === m ? 'on' : '', n); b.disabled = !has && m !== 'scan';
        b.onclick = () => { fv.setMode(m); drawTools(); }; modes.appendChild(b);
      });
      tools.append(h('span', 'lbl', 'View'), modes);
      const tg = h('span', 'grp');
      const bp = h('button', '', '&lt;'), bplay = h('button', st.tour ? 'on' : '', st.tour ? 'Pause' : 'Play'), bn = h('button', '', '&gt;');
      bp.title = 'Previous part (Left)'; bn.title = 'Next part (Right)'; bplay.title = 'Step through parts in order of first mention';
      bp.onclick = () => { stopTour(); tourStep(-1); }; bn.onclick = () => { stopTour(); tourStep(1); };
      bplay.onclick = () => { if (st.tour) return stopTour(); tourStep(1); st.tour = setInterval(() => tourStep(1), 2200); drawTools(); };
      [bp, bplay, bn].forEach(b => { b.disabled = !has; tg.appendChild(b); });
      tools.append(h('span', 'lbl', 'Parts tour'), tg);
      if (fv.maxStep >= 0) {
        const steps = (f.trace.steps && f.trace.steps.names) || [];
        const lbl = (f.trace.steps && f.trace.steps.label) || 'Step';
        const sl = h('input'); sl.type = 'range'; sl.min = -1; sl.max = fv.maxStep; sl.step = 1; sl.value = fv.step == null ? fv.maxStep : fv.step;
        const v = h('span', 'stepv num');
        const upd = () => { const n = +sl.value; fv.setStep(n === fv.maxStep && !st.stepTimer && sl.dataset.touched !== '1' ? null : n); v.innerHTML = n < 0 ? 'start' : subsup(steps[n] || `${n}`); };
        sl.oninput = () => { sl.dataset.touched = '1'; stopStep(); upd(); };
        const sp = h('button', '', 'Play'); sp.onclick = () => {
          if (st.stepTimer) { stopStep(); sp.textContent = 'Play'; return; }
          sl.dataset.touched = '1'; if (+sl.value >= fv.maxStep) sl.value = -1; upd(); sp.textContent = 'Pause';
          st.stepTimer = setInterval(() => { if (+sl.value >= fv.maxStep) { stopStep(); sp.textContent = 'Play'; return; } sl.value = +sl.value + 1; upd(); }, 650);
        };
        tools.append(h('span', 'lbl', esc(lbl)), sp, sl, v);
        const qs = /[?&]step=(\d+)/.exec(location.search); // headless review hook
        if (qs) { sl.value = qs[1]; sl.dataset.touched = '1'; }
        upd();
      }
      const fit = h('button', '', 'Fit'); fit.title = 'Fit (F)'; fit.onclick = () => fv.pz.fit();
      tools.append(h('span', 'spacer'), fit);
    }
    const key = ev => {
      if (ev.target.closest('input,select,textarea') || !$('#modal').hidden) return;
      if (ev.key === 'ArrowRight') { stopTour(); tourStep(1); ev.preventDefault(); }
      else if (ev.key === 'ArrowLeft') { stopTour(); tourStep(-1); ev.preventDefault(); }
      else if (ev.key === 'f' || ev.key === 'F') fv.pz.fit();
      else if (ev.key === 'Escape') { stopTour(); pin(null, false); fv.pz.fit(); }
      else if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
        const i = pat.figures.indexOf(st.fig) + (ev.key === 'ArrowDown' ? 1 : -1);
        if (pat.figures[i]) { go(pat, 'figures', pat.figures[i].id); ev.preventDefault(); }
      }
    };
    addEventListener('keydown', key);
    const obs = new MutationObserver(() => { if (!root.isConnected) { removeEventListener('keydown', key); stopTour(); stopStep(); obs.disconnect(); } });
    obs.observe(main, { childList: true });
    return {
      update(r) {
        const f = pat.figById[r.a] || pat.figures[0];
        if (f !== st.fig) {
          stopTour(); stopStep(); st.fig = f; st.pin = null; st.hover = null;
          fv.show(pat, f); drawTools(); buildParts();
          list.querySelectorAll('.fitem').forEach(n => n.classList.toggle('on', n.dataset.fig === f.id));
          const li = list.querySelector('.fitem.on'); if (li) li.scrollIntoView({ block: 'nearest', inline: 'nearest' });
          if (!r.a) go(pat, 'figures', f.id, undefined, true);
        }
        if ((r.b || null) !== st.pin) { st.pin = r.b || null; paint(); if (st.pin) requestAnimationFrame(() => fv.flyTo(st.pin)); }
        else paint();
      }
    };
  }

  // ---------- description / claims with a linked figure pane ----------
  function linkedPane(pat, host) {
    const pane = h('div', 'pane'); host.appendChild(pane);
    const tools = h('div', 'tools'); pane.appendChild(tools);
    const box = h('div', ''); box.style.cssText = 'position:relative;min-height:0'; pane.appendChild(box);
    const st = { fig: null, ref: null, lock: null };
    const fv = new FigureView(box, { onClick: ref => { if (ref) go(pat, 'figures', st.fig.id, ref); }, tag: f => f.label });
    fv.el.style.cssText = 'position:absolute;inset:0';
    const sel = h('select'); pat.figures.forEach(f => sel.appendChild(new Option(f.label, f.id)));
    sel.onchange = () => { st.lock = sel.value; show(pat.figById[sel.value]); };
    const open = h('button', '', 'Open in Figures'); open.onclick = () => go(pat, 'figures', st.fig.id, st.ref || undefined);
    const modes = h('span', 'grp');
    const drawModes = () => { modes.innerHTML = ''; [['trace', 'Redraw'], ['scan', 'Scan']].forEach(([m, n]) => { const b = h('button', fv.mode === m ? 'on' : '', n); b.disabled = !st.fig.trace && m !== 'scan'; b.onclick = () => { fv.setMode(m); drawModes(); }; modes.appendChild(b); }); };
    tools.append(sel, modes, h('span', 'spacer'), open);
    function show(f) { if (!f) return; if (f !== st.fig) { st.fig = f; fv.show(pat, f); sel.value = f.id; drawModes(); } }
    function figFor(ref) {
      if (st.fig && fv.has(ref)) return st.fig;
      const p = pat.parts[ref];
      const ids = (p && p.figs) || [];
      return pat.figures.find(f => ids.includes(f.id) && f.trace) || pat.figures.find(f => ids.includes(f.id)) || null;
    }
    show(pat.figures[0]);
    return {
      hoverRef(ref) {
        if (ref) { const f = figFor(ref); if (f) show(f); }
        st.ref = ref; fv.highlight(ref); if (ref && fv.has(ref)) fv.flyTo(ref); else if (!ref) fv.pz.fit();
      },
      showFig(id) { st.lock = null; show(pat.figById[id]); },
      follow(id) { if (!st.lock && !st.ref) show(pat.figById[id]); },
    };
  }

  function bindRich(container, pat, pane) {
    container.addEventListener('mouseover', ev => {
      const a = ev.target.closest('a.rf, a.fg'); if (!a) return;
      if (a.classList.contains('rf')) { showTip(ev, partTip(pat, a.dataset.ref)); pane.hoverRef(a.dataset.ref); container.querySelectorAll(`a.rf[data-ref="${CSS.escape(a.dataset.ref)}"]`).forEach(x => x.classList.add('on')); }
      else pane.showFig(a.dataset.fig);
    });
    container.addEventListener('mousemove', ev => { if (tip.style.display === 'block') moveTip(ev); });
    container.addEventListener('mouseout', ev => {
      const a = ev.target.closest('a.rf'); if (!a) return;
      hideTip(); container.querySelectorAll('a.rf.on').forEach(x => x.classList.remove('on')); pane.hoverRef(null);
    });
    container.addEventListener('click', ev => {
      const a = ev.target.closest('a.rf, a.fg, .cite'); if (!a) return;
      if (a.classList.contains('rf')) { const f = pat.figures.find(f => (pat.parts[a.dataset.ref] || {}).figs && pat.parts[a.dataset.ref].figs.includes(f.id)); if (f) go(pat, 'figures', f.id, a.dataset.ref); }
      else if (a.classList.contains('fg')) go(pat, 'figures', a.dataset.fig);
      else if (a.dataset.col || a.dataset.page) openScan(pat, +a.dataset.col, a.textContent, a.dataset.page);
    });
  }

  function TextTab(pat) {
    const root = h('div', 'textv'); main.appendChild(root);
    const col = h('div', 'tcol'); root.appendChild(col);
    const pane = linkedPane(pat, root);
    const bar = h('div', 'tbar'); col.appendChild(bar);
    const q = h('input'); q.type = 'search'; q.placeholder = 'Search description';
    const secSel = h('select'); pat.sections.forEach((s, i) => secSel.appendChild(new Option(s.heading || 'Opening (no heading)', i)));
    const count = h('span', 'lbl num');
    bar.append(q, secSel, count);
    const body = h('div', 'tbody'); col.appendChild(body);
    const abs = h('div', ''); body.appendChild(abs);
    abs.innerHTML = `<div class="tsec">Abstract</div><div class="para"><span class="cite" data-col="">front</span><p>${subsup(pat.abstract || '')}</p></div>`;
    const nodes = [];
    pat.sections.forEach((s, si) => {
      const hd = h('div', 'tsec', esc(s.heading)); hd.dataset.sec = si; if (!s.heading) hd.style.margin = '12px 0 0'; body.appendChild(hd);
      pat.paras.filter(p => p.sec === si).forEach(p => {
        const n = h('div', 'para'); n.dataset.pid = p.id;
        const c = `<span class="cite" data-col="${p.col || ''}" data-page="${p.page != null ? p.page : ''}" title="Show on the scanned page">${p.page != null && !p.col ? `p.${p.page + 1}` : cite(p.col, p.line)}</span>`;
        n.innerHTML = c + (p.table ? `<div><table>${p.table.map(r => `<tr>${r.map(x => `<td>${subsup(x)}</td>`).join('')}</tr>`).join('')}</table></div>` : `<p>${rich(p.text, p.spans)}</p>`);
        body.appendChild(n); nodes.push([p, n]);
      });
    });
    bindRich(body, pat, pane);
    secSel.onchange = () => { const hd = body.querySelector(`.tsec[data-sec="${secSel.value}"]`); if (hd) hd.scrollIntoView({ block: 'start' }); };
    let qt = 0;
    q.oninput = () => {
      clearTimeout(qt); qt = setTimeout(() => {
        const s = q.value.trim(); let hit = 0;
        const re = s ? new RegExp(s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi') : null;
        nodes.forEach(([p, n]) => {
          if (p.table) { n.style.display = re ? 'none' : ''; return; }
          const ok = !re || re.test(p.text); if (re) re.lastIndex = 0;
          n.style.display = ok ? '' : 'none'; if (ok && re) hit++;
          const pp = n.querySelector('p'); if (pp) pp.innerHTML = rich(p.text, p.spans, ok ? re : null);
        });
        body.querySelectorAll('.tsec').forEach(x => x.style.display = re ? 'none' : '');
        abs.style.display = re ? 'none' : '';
        count.textContent = re ? `${hit} paragraphs` : '';
      }, 120);
    };
    // the pane follows the figure referenced by the topmost paragraph in view
    const io = new IntersectionObserver(es => {
      const vis = es.filter(e => e.isIntersecting).map(e => e.target).sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
      for (const n of vis) { const p = pat.paras[+n.dataset.pid]; const fs = p && p.spans.find(s => s[2] === 'fig'); if (fs) { pane.follow(fs[3]); break; } }
    }, { root: col, rootMargin: '0px 0px -60% 0px' });
    nodes.forEach(([, n]) => io.observe(n));
    return {
      update(r) {
        if (r.a != null && r.a !== '') {
          const n = body.querySelector(`.para[data-pid="${CSS.escape(r.a)}"]`);
          if (n) { n.scrollIntoView({ block: 'center' }); n.classList.add('flash'); setTimeout(() => n.classList.remove('flash'), 1600); }
        }
      }
    };
  }

  function ClaimsTab(pat) {
    const root = h('div', 'textv'); main.appendChild(root);
    const col = h('div', 'tcol'); root.appendChild(col);
    const pane = linkedPane(pat, root);
    const bar = h('div', 'tbar'); col.appendChild(bar);
    const q = h('input'); q.type = 'search'; q.placeholder = 'Search claims';
    const ind = h('button', 'on', 'Independent + dependent'); const only = h('button', '', 'Independent only');
    const grp = h('span', 'grp'); grp.append(ind, only);
    const count = h('span', 'lbl num'); bar.append(q, grp, count);
    const body = h('div', 'tbody'); col.appendChild(body);
    const nodes = pat.claims.map(c => {
      const n = h('div', 'claim' + (c.depends ? ' dep' : ''));
      n.innerHTML = `<div class="lbl"><span class="num">${c.n}</span>${c.depends ? ` &middot; depends on ${c.depends}` : ' &middot; independent'} <span class="cite" data-col="${c.col || ''}" data-page="${c.page != null ? c.page : ''}">${cite(c.col, c.line)}</span></div><p>${rich(c.text, c.spans)}</p>`;
      body.appendChild(n); return [c, n];
    });
    bindRich(body, pat, pane);
    let indOnly = false;
    const apply = () => {
      const s = q.value.trim(), re = s ? new RegExp(s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi') : null; let k = 0;
      nodes.forEach(([c, n]) => {
        const ok = (!indOnly || !c.depends) && (!re || re.test(c.text)); if (re) re.lastIndex = 0;
        n.style.display = ok ? '' : 'none'; if (ok) k++;
        n.querySelector('p').innerHTML = rich(c.text, c.spans, ok ? re : null);
      });
      count.textContent = `${k} of ${nodes.length}`;
    };
    q.oninput = apply;
    ind.onclick = () => { indOnly = false; ind.className = 'on'; only.className = ''; apply(); };
    only.onclick = () => { indOnly = true; only.className = 'on'; ind.className = ''; apply(); };
    apply();
    return { update() { } };
  }

  function AboutTab(pat) {
    const ent = pat.entities || {};
    const personHtml = e => `${esc(e.name)}${[e.city, e.region, e.country !== 'US' ? e.country : null].filter(Boolean).length ? ` <span class="pl">${esc([e.city, e.region, e.country !== 'US' ? e.country : null].filter(Boolean).join(', '))}</span>` : ''}${e.kind ? ` <span class="lbl">${esc(e.kind)}</span>` : ''}`;
    const root = h('div', 'about'); main.appendChild(root);
    const n = pat.notes || {};
    const row = (k, v) => v ? `<tr><th>${k}</th><td>${v}</td></tr>` : '';
    root.innerHTML = `<div class="wrap">
      ${n.patent ? `<div class="note"><span class="lbl">Note</span>${subsup(n.patent)}</div>` : ''}
      <h3>Bibliographic data</h3>
      <table>
        ${row('Patent', `<span class="num">US ${esc(pat.number)}</span>`)}
        ${row('Title', esc(pat.title))}
        ${row('Inventors', ent.inventors ? ent.inventors.map(personHtml).join('<br>') : esc((pat.inventors || []).join('; ')))}
        ${row('Term', pat.status ? `${esc(pat.status.state)} <span class="num">${esc(pat.status.term_end)}</span> <span class="pl">${esc(pat.status.basis)}${pat.status.google_expiration && pat.status.google_expiration !== pat.status.term_end ? `; Google Patents lists ${esc(pat.status.google_expiration)}` : ''}; checked ${esc(pat.status.checked)}</span>` : '<span class="unk">unknown</span>')}
        ${row('Assignee', ent.assignees ? (ent.assignees.map(personHtml).join('<br>') || '<span class="unk">none printed</span>') : esc(pat.assignee))}
        ${row('Application no.', `<span class="num">${esc(pat.appl_no)}</span>`)}
        ${row('Filed', `<span class="num">${esc(pat.filed)}</span>`)}
        ${row('Issued', `<span class="num">${esc(pat.issued)}</span>`)}
        ${row('Int. Cl.', esc(pat.int_cl))}
        ${row('U.S. Cl.', esc(pat.us_cl))}
        ${row('Examiners', esc(pat.examiners))}
        ${row('Claims / figures', `<span class="num">${pat.claims.length} / ${pat.figures.length}</span>`)}
      </table>
      ${n.glossary && n.glossary.length ? `<h3>Glossary</h3><dl>${n.glossary.map(g => `<dt>${esc(g.term)}</dt><dd>${subsup(g.def)}</dd>`).join('')}</dl>` : ''}
      <h3>Parts index</h3>
      <table>${Object.values(pat.parts).sort((a, b) => parseInt(a.ref) - parseInt(b.ref) || a.ref.localeCompare(b.ref)).map(p =>
        `<tr><th class="num"><a href="#/${pat.pn}/figures/${esc((p.figs || [])[0] || pat.figures[0].id)}/${esc(p.ref)}">${esc(p.ref)}</a></th><td>${esc(p.name)}${p.alt && p.alt.length ? ` <span class="lbl">also</span> ${p.alt.map(esc).join('; ')}` : ''}</td><td class="num">${(p.figs || []).join(', ')}</td></tr>`).join('')}</table>
      ${pat.refs_patents && pat.refs_patents.length ? `<h3>U.S. patent documents cited</h3><table>${pat.refs_patents.map(r =>
        `<tr><td class="num">${esc(r.no)}</td><td class="num">${esc(r.date)}</td><td>${esc(r.name)}</td><td class="num">${esc(r.cl)}</td></tr>`).join('')}</table>` : ''}
      ${pat.refs_other && pat.refs_other.length ? `<h3>Other publications</h3><table>${pat.refs_other.map(r => `<tr><td>${esc(r)}</td></tr>`).join('')}</table>` : ''}
      <div class="sources">Text: OCR of the USPTO scan, checked against the scan and ${pat.google ? `<a href="${esc(pat.google)}" target="_blank" rel="noopener">Google Patents</a>` : 'Google Patents'}; ${pat.ocr_fixes} OCR corrections; spelling kept as printed. Citations column:line. Redraws traced from the drawing sheets; Scan view shows the original. Notes are added context, not patent text.</div>
    </div>`;
    return { update() { } };
  }

  if (SITE.home) $('#home').href = SITE.home;
  route();

  // ?selftest: drive every route, figure, part and hover; report into the #selftest block (read with --dump-dom)
  if (/selftest/.test(location.search)) {
    const errs = [], log = [];
    addEventListener('error', e => errs.push(`${e.message} @${e.lineno}`));
    addEventListener('unhandledrejection', e => errs.push(`rejection: ${e.reason && e.reason.stack || e.reason}`));
    const wait = ms => new Promise(r => setTimeout(r, ms));
    (async () => {
      if (SITE.links) {  // gallery page: cards only
        route('#/'); await wait(100);
        const cards = document.querySelectorAll('.card'), bad = [...cards].filter(c => !/^p\/\d+\.html#\//.test(c.getAttribute('href')));
        if (bad.length || cards.length !== PATS.length) errs.push(`gallery: ${cards.length} cards for ${PATS.length} patents, ${bad.length} bad links`);
        log.push(`gallery: ${cards.length} cards, ${document.querySelectorAll('.card svg .tr-el, .card svg [data-ref]').length} trace elements`);
        const n = () => document.querySelectorAll('.card, .gtable tbody tr').length;
        for (const v of ['cards', 'table']) for (const k of Object.keys(SORTS)) for (const d of ['', '&d=desc']) {
          route(`#/?v=${v}&s=${k}${d}`); await wait(5);
          if (n() !== PATS.length) errs.push(`${v} sort ${k}${d}: ${n()} rows`);
        }
        const probe = PATS[0], term = ((probe.inventors || [])[0] || {}).name || probe.title;
        route(`#/?v=table&q=${encodeURIComponent(term)}`); await wait(5);
        if (!document.querySelector(`.gtable tr[data-pn="${probe.pn}"]`)) errs.push(`search "${term}" misses ${probe.pn}`);
        route(`#/?q=zzzz-no-match`); await wait(5);
        if (!document.querySelector('.gnone')) errs.push('no-match state missing');
        for (const [k] of FACETS) for (const v of new Set(PATS.flatMap(p => p.facet[k]))) {
          route(`#/?v=table&${k}=${encodeURIComponent(v)}`); await wait(5);
          const want = PATS.filter(p => p.facet[k].includes(v)).length;
          if (n() !== want) errs.push(`facet ${k}=${v}: ${n()} rows, want ${want}`);
        }
        log.push(`gallery: ${Object.keys(SORTS).length} sorts x 2 views x 2 dirs, search, facets checked`);
      } else for (const p of PATS) {
        for (const f of p.figures) {
          route(`#/${p.pn}/figures/${f.id}`); await wait(30);
          const refs = f.refs.slice(0, 40);
          for (const r of refs) { route(`#/${p.pn}/figures/${f.id}/${r}`); await wait(5); }
          const hits = document.querySelectorAll('.fv .hit [data-ref]').length, on = document.querySelectorAll('.fv .tr-el.on').length;
          log.push(`${p.pn} fig ${f.id}: trace=${!!f.trace} hit=${hits} lastSelOn=${on} refs=${f.refs.length} missingInTrace=${f.trace ? f.refs.filter(r => !f.trace.elements.some(e => e.ref === r)).join(',') : '-'}`);
        }
        for (const t of ['text', 'claims', 'about']) {
          route(`#/${p.pn}/${t}`); await wait(120);
          const a = document.querySelector('a.rf');
          if (a) { a.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: 10, clientY: 10 })); await wait(20); a.dispatchEvent(new MouseEvent('mouseout', { bubbles: true })); }
          log.push(`${p.pn} ${t}: view=${view && view.pat.pn}/${view && view.tab} rf=${document.querySelectorAll('a.rf').length} fg=${document.querySelectorAll('a.fg').length}`);
        }
      }
      const pre = document.createElement('pre'); pre.id = 'selftest';
      pre.textContent = `SELFTEST ${errs.length ? 'FAIL' : 'PASS'}\n` + errs.join('\n') + '\n' + log.join('\n');
      document.body.appendChild(pre);
    })();
  }
})();
