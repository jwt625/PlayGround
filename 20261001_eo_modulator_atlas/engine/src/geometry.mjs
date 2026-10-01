// 2D geometry helpers (lengths in micrometers inside the engine plane).
// Part of the eo-atlas engine. License: GPL-3.0-or-later (see engine/LICENSE note in package.json).

/** @typedef {[number, number]} Pt */

/** Axis-aligned rectangle {x:[x0,x1], y:[y0,y1]} (um) to a ccw polygon. */
export function rectToPolygon(x, y) {
  return [
    [x[0], y[0]],
    [x[1], y[0]],
    [x[1], y[1]],
    [x[0], y[1]],
  ];
}

export function polygonArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    a += x0 * y1 - x1 * y0;
  }
  return 0.5 * a;
}

/** Return polygon in counter-clockwise order. */
export function ensureCcw(poly) {
  return polygonArea(poly) < 0 ? poly.slice().reverse() : poly.slice();
}

export function polygonBBox(poly) {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const [x, y] of poly) {
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1 };
}

/** Even-odd point in polygon. Points exactly on the boundary may go either way. */
export function pointInPolygon(px, py, poly) {
  let inside = false;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i][0],
      yi = poly[i][1],
      xj = poly[j][0],
      yj = poly[j][1];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Distance from point to segment. */
export function distPointSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax,
    dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let t = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const qx = ax + t * dx - px,
    qy = ay + t * dy - py;
  return Math.hypot(qx, qy);
}

/** Distance from point to polygon (0 if inside when filled=true). */
export function distPointPolygon(px, py, poly, filled = true) {
  if (filled && pointInPolygon(px, py, poly)) return 0;
  let d = Infinity;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i],
      b = poly[(i + 1) % n];
    const di = distPointSegment(px, py, a[0], a[1], b[0], b[1]);
    if (di < d) d = di;
  }
  return d;
}

/** Liang-Barsky clip of a segment to a rectangle. Returns null or [x0,y0,x1,y1]. */
export function clipSegmentToRect(ax, ay, bx, by, r) {
  let t0 = 0,
    t1 = 1;
  const dx = bx - ax,
    dy = by - ay;
  const p = [-dx, dx, -dy, dy];
  const q = [ax - r.x0, r.x1 - ax, ay - r.y0, r.y1 - ay];
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return null;
    } else {
      const t = q[i] / p[i];
      if (p[i] < 0) {
        if (t > t1) return null;
        if (t > t0) t0 = t;
      } else {
        if (t < t0) return null;
        if (t < t1) t1 = t;
      }
    }
  }
  if (t1 - t0 <= 0) return null;
  return [ax + t0 * dx, ay + t0 * dy, ax + t1 * dx, ay + t1 * dy];
}

/**
 * Build a planar straight-line graph from tagged polygon boundaries.
 * Segments are clipped to the rectangle `clip`, split at mutual intersections and T-junctions,
 * vertices merged within `tol`, duplicate pieces merged (tags are OR-combined).
 *
 * @param {{poly: Pt[], kind: number, id: number}[]} loops  kind bit: 1 region, 2 electrode, 4 domain
 * @param {{x0:number,y0:number,x1:number,y1:number}} clip  hull rectangle (always added as domain loop)
 * @param {number} tol
 * @returns {{verts: Pt[], segs: {a:number,b:number,mask:number,elec:number}[]}}
 */
export function buildPslg(loops, clip, tol) {
  /** @type {{ax:number,ay:number,bx:number,by:number,mask:number,elec:number}[]} */
  const raw = [];
  const addLoop = (poly, mask, elec) => {
    const n = poly.length;
    for (let i = 0; i < n; i++) {
      const a = poly[i],
        b = poly[(i + 1) % n];
      const c = clipSegmentToRect(a[0], a[1], b[0], b[1], clip);
      if (!c) continue;
      if (Math.hypot(c[2] - c[0], c[3] - c[1]) <= tol) continue;
      raw.push({ ax: c[0], ay: c[1], bx: c[2], by: c[3], mask, elec });
    }
  };
  for (const l of loops) addLoop(l.poly, l.kind, l.kind & 2 ? l.id : -1);
  addLoop(
    [
      [clip.x0, clip.y0],
      [clip.x1, clip.y0],
      [clip.x1, clip.y1],
      [clip.x0, clip.y1],
    ],
    4,
    -1,
  );

  // collect split points per raw segment
  const splits = raw.map(() => /** @type {number[]} */ ([]));
  const n = raw.length;
  for (let i = 0; i < n; i++) {
    const s = raw[i];
    const dx = s.bx - s.ax,
      dy = s.by - s.ay;
    const len = Math.hypot(dx, dy);
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const u = raw[j];
      // endpoints of u lying on interior of s
      for (const [px, py] of [
        [u.ax, u.ay],
        [u.bx, u.by],
      ]) {
        const t = ((px - s.ax) * dx + (py - s.ay) * dy) / (len * len);
        if (t * len > tol && (1 - t) * len > tol) {
          const qx = s.ax + t * dx - px,
            qy = s.ay + t * dy - py;
          if (Math.hypot(qx, qy) <= tol) splits[i].push(t);
        }
      }
      // proper crossing
      const ex = u.bx - u.ax,
        ey = u.by - u.ay;
      const den = dx * ey - dy * ex;
      if (Math.abs(den) > 1e-14 * len * Math.hypot(ex, ey)) {
        const t = ((u.ax - s.ax) * ey - (u.ay - s.ay) * ex) / den;
        const w = ((u.ax - s.ax) * dy - (u.ay - s.ay) * dx) / den;
        const lu = Math.hypot(ex, ey);
        if (t * len > tol && (1 - t) * len > tol && w * lu > tol && (1 - w) * lu > tol) splits[i].push(t);
      }
    }
  }

  // vertex dedupe via spatial hash
  /** @type {Pt[]} */
  const verts = [];
  const cell = Math.max(tol * 4, 1e-12);
  const hash = new Map();
  const keyOf = (ix, iy) => `${ix},${iy}`;
  const snap = (v, lo, hi) => (Math.abs(v - lo) <= tol ? lo : Math.abs(v - hi) <= tol ? hi : v);
  const getVertex = (x0, y0) => {
    const x = snap(x0, clip.x0, clip.x1),
      y = snap(y0, clip.y0, clip.y1);
    const ix = Math.floor(x / cell),
      iy = Math.floor(y / cell);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++) {
        const lst = hash.get(keyOf(ix + dx, iy + dy));
        if (lst) for (const v of lst) if (Math.hypot(verts[v][0] - x, verts[v][1] - y) <= tol) return v;
      }
    const id = verts.length;
    verts.push([x, y]);
    const k = keyOf(ix, iy);
    if (!hash.has(k)) hash.set(k, []);
    hash.get(k).push(id);
    return id;
  };

  const segMap = new Map();
  for (let i = 0; i < n; i++) {
    const s = raw[i];
    const ts = [0, ...splits[i].sort((p, q) => p - q), 1];
    let prev = getVertex(s.ax, s.ay);
    for (let k = 1; k < ts.length; k++) {
      const t = ts[k];
      const v = t === 1 ? getVertex(s.bx, s.by) : getVertex(s.ax + t * (s.bx - s.ax), s.ay + t * (s.by - s.ay));
      if (v !== prev) {
        const key = prev < v ? `${prev}_${v}` : `${v}_${prev}`;
        const ex = segMap.get(key);
        if (ex) {
          ex.mask |= s.mask;
          if (s.elec >= 0) {
            if (ex.elec >= 0 && ex.elec !== s.elec) throw new Error(`electrodes ${ex.elec} and ${s.elec} share an edge`);
            ex.elec = s.elec;
          }
        } else segMap.set(key, { a: Math.min(prev, v), b: Math.max(prev, v), mask: s.mask, elec: s.elec });
        prev = v;
      }
    }
  }
  return { verts, segs: [...segMap.values()] };
}
