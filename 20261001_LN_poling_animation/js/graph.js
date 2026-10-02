// Voltage / current plot with moving dashed cursor and markers.
class Graph {
  constructor(canvas, onScrub) {
    this.cv = canvas; this.ctx = canvas.getContext('2d'); this.onScrub = onScrub;
    this.m = { l: 52, r: 54, t: 12, b: 32 };
    this.hover = null; this.drag = false;
    const pos = (e) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    canvas.addEventListener('pointerdown', (e) => { this.drag = true; canvas.setPointerCapture(e.pointerId); this.scrub(pos(e).x); });
    canvas.addEventListener('pointermove', (e) => { const p = pos(e); this.hover = p; if (this.drag) this.scrub(p.x); });
    canvas.addEventListener('pointerup', () => { this.drag = false; });
    canvas.addEventListener('pointerleave', () => { this.hover = null; });
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }
  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.cv.getBoundingClientRect();
    this.w = r.width; this.h = r.height;
    this.cv.width = Math.max(1, Math.round(r.width * dpr)); this.cv.height = Math.max(1, Math.round(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  tx(t) { return this.m.l + t / WF.TMAX * (this.w - this.m.l - this.m.r); }
  yI(v) { return this.m.t + (400 - v) / 450 * (this.h - this.m.t - this.m.b); }
  yV(v) { return this.m.t + (600 - v) / 675 * (this.h - this.m.t - this.m.b); }
  scrub(x) { this.onScrub(Math.min(WF.TMAX, Math.max(0, (x - this.m.l) / (this.w - this.m.l - this.m.r) * WF.TMAX))); }
  draw(t) {
    const c = this.ctx, { l, r, t: mt, b } = this.m, w = this.w, h = this.h;
    c.clearRect(0, 0, w, h);
    c.font = '11px -apple-system, Segoe UI, Roboto, sans-serif';
    c.lineWidth = 1;
    // grid + left ticks (current)
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const v of [-50, 0, 100, 200, 300, 400]) {
      const y = Math.round(this.yI(v)) + 0.5;
      c.strokeStyle = v === 0 ? '#3a4048' : '#20252b'; c.beginPath(); c.moveTo(l, y); c.lineTo(w - r, y); c.stroke();
      c.fillStyle = '#5794f2'; c.fillText(String(v), l - 6, y);
    }
    c.textAlign = 'left';
    for (const v of [0, 100, 200, 300, 400, 500, 600]) { c.fillStyle = '#ff9830'; c.fillText(String(v), w - r + 6, this.yV(v)); }
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (let v = 0; v <= 10; v += 2) {
      const x = Math.round(this.tx(v)) + 0.5;
      c.strokeStyle = '#20252b'; c.beginPath(); c.moveTo(x, mt); c.lineTo(x, h - b); c.stroke();
      c.fillStyle = '#9aa4ae'; c.fillText(String(v), x, h - b + 5);
    }
    c.strokeStyle = '#3a4048'; c.strokeRect(l + 0.5, mt + 0.5, w - l - r, h - mt - b);
    c.fillStyle = '#9aa4ae'; c.fillText('Time (ms)', (l + w - r) / 2, h - 14);
    c.save(); c.translate(12, (mt + h - b) / 2); c.rotate(-Math.PI / 2); c.fillStyle = '#5794f2'; c.fillText('Current (nA)', 0, 0); c.restore();
    c.save(); c.translate(w - 10, (mt + h - b) / 2); c.rotate(Math.PI / 2); c.fillStyle = '#ff9830'; c.fillText('Voltage (V)', 0, 0); c.restore();
    // curves
    const step = 0.01;
    c.save(); c.beginPath(); c.rect(l, mt, w - l - r, h - mt - b); c.clip();
    c.lineWidth = 1.5; c.lineJoin = 'round';
    c.strokeStyle = '#5794f2'; c.beginPath();
    for (let x = 0; x <= WF.TMAX + 1e-9; x += step) { const px = this.tx(x), py = this.yI(WF.I(x)); x === 0 ? c.moveTo(px, py) : c.lineTo(px, py); }
    c.stroke();
    c.strokeStyle = '#ff9830'; c.beginPath();
    for (let x = 0; x <= WF.TMAX + 1e-9; x += step) { const px = this.tx(x), py = this.yV(WF.V(x)); x === 0 ? c.moveTo(px, py) : c.lineTo(px, py); }
    c.stroke();
    c.restore();
    // cursor + markers
    const X = this.tx(t);
    c.setLineDash([5, 4]); c.strokeStyle = '#e6edf3'; c.beginPath(); c.moveTo(X + 0.5, mt); c.lineTo(X + 0.5, h - b); c.stroke(); c.setLineDash([]);
    for (const [y, col] of [[this.yI(WF.I(t)), '#5794f2'], [this.yV(WF.V(t)), '#ff9830']]) {
      c.fillStyle = col; c.strokeStyle = '#0b0d10'; c.lineWidth = 2; c.beginPath(); c.arc(X, y, 4.5, 0, 6.2832); c.stroke(); c.fill();
    }
    // hover readout
    if (this.hover && this.hover.x > l && this.hover.x < w - r) {
      const th = (this.hover.x - l) / (w - l - r) * WF.TMAX;
      c.strokeStyle = 'rgba(230,237,243,.25)'; c.lineWidth = 1; c.beginPath(); c.moveTo(this.hover.x + 0.5, mt); c.lineTo(this.hover.x + 0.5, h - b); c.stroke();
      const lines = [`t = ${th.toFixed(2)} ms`, `V = ${WF.V(th).toFixed(0)} V`, `I = ${WF.I(th).toFixed(1)} nA`, `Q = ${WF.Q(th).toFixed(1)} pC`];
      c.font = '11px ui-monospace, Menlo, monospace';
      const bw = 96, bh = 14 * lines.length + 8;
      let bx = this.hover.x + 10; if (bx + bw > w - r) bx = this.hover.x - bw - 10;
      c.fillStyle = '#1c2026'; c.strokeStyle = '#4a525b'; c.fillRect(bx, mt + 6, bw, bh); c.strokeRect(bx + 0.5, mt + 6.5, bw, bh);
      c.fillStyle = '#c7d0d9'; c.textAlign = 'left'; c.textBaseline = 'top';
      lines.forEach((s, i) => c.fillText(s, bx + 6, mt + 11 + 14 * i));
    }
  }
}
