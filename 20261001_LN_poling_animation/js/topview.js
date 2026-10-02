// Top view of the progressing domain rendered like an SHG microscope image (SH intensity, white = 1):
// dark domain walls, slightly darker poled domains, black electrode blocks at both ends.
class TopView {
  constructor(canvas, track) {
    this.cv = canvas; this.ctx = canvas.getContext('2d'); this.track = track;
    const G = DOM.grid;
    this.off = document.createElement('canvas'); this.off.width = G.w; this.off.height = G.h;
    this.octx = this.off.getContext('2d'); this.img = this.octx.createImageData(G.w, G.h);
    this.grain = new Float32Array(G.w * G.h);
    let s = 12345; for (let i = 0; i < this.grain.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; this.grain[i] = s / 4294967296; }
    this.mask = new Uint8Array(G.w * G.h);
    this.wall = new Float32Array(G.w * G.h); this.tmp = new Float32Array(G.w * G.h);
    this.r0 = Math.round(-G.y0 / (G.y1 - G.y0) * G.h); this.r1 = this.r0 + Math.round(DOM.LG / (G.y1 - G.y0) * G.h);
    this.lastH = NaN;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }
  resize() {
    const dpr = window.devicePixelRatio || 1, r = this.cv.getBoundingClientRect();
    this.w = r.width; this.h = r.height;
    this.cv.width = Math.max(1, Math.round(r.width * dpr)); this.cv.height = Math.max(1, Math.round(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.lastH = NaN;
  }
  blur(a) { // two passes of [1 2 1]/4 in x and y
    const G = DOM.grid, W = G.w, H = G.h, t = this.tmp;
    for (let pass = 0; pass < 2; pass++) {
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const i = r * W + c; t[i] = 0.25 * a[c > 0 ? i - 1 : i] + 0.5 * a[i] + 0.25 * a[c < W - 1 ? i + 1 : i];
      }
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const i = r * W + c; a[i] = 0.25 * t[r > 0 ? i - W : i] + 0.5 * t[i] + 0.25 * t[r < H - 1 ? i + W : i];
      }
    }
  }
  render(h) {
    const G = DOM.grid, d = this.img.data, m = this.mask, wl = this.wall, W = G.w, H = G.h, r0 = this.r0, r1 = this.r1;
    for (let i = 0; i < m.length; i++) m[i] = G.g[i] <= h ? 1 : 0;
    // continue the stripe state under the electrodes (rows outside the gap copy the nearest gap row)
    for (let r = 0; r < r0; r++) m.copyWithin(r * W, r0 * W, r0 * W + W);
    for (let r = r1; r < H; r++) m.copyWithin(r * W, (r1 - 1) * W, (r1 - 1) * W + W);
    wl.fill(0);
    for (let r = r0; r < r1; r++) for (let c = 0; c < W; c++) {
      const i = r * W + c;
      if (m[i] !== (c > 0 ? m[i - 1] : m[i]) || m[i] !== (c < W - 1 ? m[i + 1] : m[i]) ||
        (r > r0 && m[i] !== m[i - W]) || (r < r1 - 1 && m[i] !== m[i + W])) wl[i] = 1;
    }
    this.blur(wl);
    const LG = DOM.LG;
    for (let r = 0; r < H; r++) {
      const y = G.y0 + (r + 0.5) / H * (G.y1 - G.y0);
      const near = (y >= 0 && y <= LG) ? 1 - 0.3 * Math.exp(-y / 0.16) - 0.3 * Math.exp(-(LG - y) / 0.16) : 1;
      for (let c = 0; c < W; c++) {
        const i = r * W + c, k = i * 4;
        let I = (m[i] ? 0.7 : 0.86) * near + (this.grain[i] - 0.5) * 0.07;
        I *= 1 - Math.min(1, wl[i] * 1.5) * 0.9;
        const v = Math.max(0, Math.min(1, I)) * 255;
        d[k] = v; d[k + 1] = v; d[k + 2] = v; d[k + 3] = 255;
      }
    }
    this.octx.putImageData(this.img, 0, 0);
  }
  draw(t) {
    const G = DOM.grid, c = this.ctx, f = WF.F(t), h = DOM.hOf(f);
    if (h !== this.lastH) { this.render(h); this.lastH = h; }
    c.fillStyle = '#0e1013'; c.fillRect(0, 0, this.w, this.h);
    const ar = G.w / G.h, padL = 10, padR = 58, padT = 30, padB = 10;
    let dw = this.w - padL - padR, dh = dw / ar;
    if (dh > this.h - padT - padB) { dh = this.h - padT - padB; dw = dh * ar; }
    const ox = padL + (this.w - padL - padR - dw) / 2, oy = padT + (this.h - padT - padB - dh) / 2;
    const sx = dw / (G.x1 - G.x0), sy = dh / (G.y1 - G.y0);
    const X = (x) => ox + (x - G.x0) * sx, Y = (y) => oy + (y - G.y0) * sy, LG = DOM.LG;
    c.imageSmoothingEnabled = true; c.drawImage(this.off, ox, oy, dw, dh);
    // electrode blocks: black bars under the stripes, light slits between (SH is blocked by metal)
    const v = WF.V(t) / WF.P.Vmax;
    const top = `rgb(${Math.round(10 + 40 * v)},10,10)`;
    const hw = DOM.HW, dh2 = DOM.DOME;
    for (const x of DOM.xs) {
      c.fillStyle = top; c.beginPath(); c.moveTo(X(x - hw), Y(G.y0)); c.lineTo(X(x - hw), Y(0));
      c.ellipse(X(x), Y(0), hw * sx, dh2 * sy, 0, Math.PI, 0, true); c.lineTo(X(x + hw), Y(G.y0)); c.closePath(); c.fill();
      c.fillStyle = '#0a0a0a'; c.beginPath(); c.moveTo(X(x - hw), Y(G.y1)); c.lineTo(X(x - hw), Y(LG));
      c.ellipse(X(x), Y(LG), hw * sx, dh2 * sy, 0, Math.PI, 0, false); c.lineTo(X(x + hw), Y(G.y1)); c.closePath(); c.fill();
    }
    c.fillStyle = '#e6edf3'; c.font = '600 11px -apple-system, Segoe UI, sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText('+V', X(0), Y(G.y0 + 0.22)); c.fillText('GND', X(0), Y(G.y1 - 0.22));
    // polarization arrow row above the image (down = poled)
    const arrow = (x0, y0, x1, y1, col, lw) => {
      c.strokeStyle = col; c.fillStyle = col; c.lineWidth = lw; c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
      const an = Math.atan2(y1 - y0, x1 - x0); c.beginPath(); c.moveTo(x1, y1);
      c.lineTo(x1 - 6 * Math.cos(an - 0.45), y1 - 6 * Math.sin(an - 0.45)); c.lineTo(x1 - 6 * Math.cos(an + 0.45), y1 - 6 * Math.sin(an + 0.45));
      c.closePath(); c.fill();
    };
    const colAt = (x) => Math.min(G.w - 1, Math.max(0, Math.floor((x - G.x0) / (G.x1 - G.x0) * G.w)));
    const ay = oy - 6, L = 18;
    DOM.xs.forEach((x, i) => {
      const down = this.mask[(this.r0 + Math.round((DOM.DOME + 0.04) / (G.y1 - G.y0) * G.h)) * G.w + colAt(x)] === 1;
      down ? arrow(X(x), ay - L, X(x), ay, '#3b78e0', 1.5) : arrow(X(x), ay, X(x), ay - L, '#e0a24a', 1.5);
      if (i < DOM.xs.length - 1) arrow(X(x + 0.5), ay, X(x + 0.5), ay - L, '#e0a24a', 1.5);
    });
    // colorbar (0 black at top, 1 white at bottom, as in SH intensity maps)
    const cx0 = ox + dw + 14, cy0 = oy, ch = dh * 0.62;
    const grd = c.createLinearGradient(0, cy0, 0, cy0 + ch); grd.addColorStop(0, '#000'); grd.addColorStop(1, '#fff');
    c.fillStyle = grd; c.fillRect(cx0, cy0, 11, ch); c.strokeStyle = '#4a525b'; c.lineWidth = 1; c.strokeRect(cx0 + 0.5, cy0 + 0.5, 11, ch);
    c.fillStyle = '#9aa4ae'; c.font = '10px -apple-system, Segoe UI, sans-serif'; c.textAlign = 'left'; c.textBaseline = 'middle';
    c.fillText('0', cx0 + 15, cy0 + 5); c.fillText('1', cx0 + 15, cy0 + ch - 5);
    c.save(); c.translate(cx0 + 36, cy0 + ch / 2); c.rotate(Math.PI / 2); c.textAlign = 'center'; c.fillText('SH intensity (norm.)', 0, 0); c.restore();
    // crystal axes: z in the image plane along the field, x out of the page (x-cut)
    const bx = cx0 + 4, by = oy + dh - 6;
    arrow(bx, by, bx, by - 30, '#e6edf3', 1.5); arrow(bx, by, bx + 30, by, '#e6edf3', 1.5);
    c.fillStyle = '#e6edf3'; c.font = '600 11px -apple-system, Segoe UI, sans-serif'; c.textAlign = 'left';
    c.fillText('+z', bx + 5, by - 28); c.fillText('y', bx + 33, by);
    c.strokeStyle = '#e6edf3'; c.lineWidth = 1.5; c.beginPath(); c.arc(bx, by, 3.5, 0, 6.2832); c.stroke();
    c.beginPath(); c.arc(bx, by, 1.2, 0, 6.2832); c.fill();
    // overview window and tracked cell
    c.setLineDash([4, 3]); c.strokeStyle = 'rgba(255,152,48,.9)'; c.lineWidth = 1;
    c.strokeRect(X(-0.5) + 0.5, Y(0) + 0.5, sx, LG * sy);
    c.setLineDash([]);
    c.strokeStyle = '#ff9830'; c.fillStyle = 'rgba(11,13,16,.6)'; c.lineWidth = 1.5;
    c.beginPath(); c.arc(X(this.track.x), Y(this.track.y), 5, 0, 6.2832); c.fill(); c.stroke();
  }
}
