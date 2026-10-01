// Conforming 2D quality mesher: Bowyer-Watson Delaunay + Ruppert/Chew-style refinement.
//
// Why own code: the available permissive npm packages either do unconstrained Delaunay only
// (delaunator, ISC) or constrained Delaunay without quality refinement / size fields
// (cdt2d MIT, poly2tri BSD); Shewchuk Triangle is not freely usable and is excluded by project rule.
// This module therefore implements incremental Delaunay (robust predicates from the public-domain
// `robust-predicates` package), conforming segment recovery by encroachment splitting
// (input segments become unions of Delaunay edges, so no constrained-edge flipping is required),
// and circumcenter refinement driven by a radius-edge bound and a graded size field.
//
// Units: coordinates are micrometers (engine plane convention).
// License: GPL-3.0-or-later.

import { orient2d, incircle } from 'robust-predicates';
import { buildPslg, pointInPolygon, polygonBBox, distPointPolygon, distPointSegment } from './geometry.mjs';

const orient = (ax, ay, bx, by, cx, cy) => -orient2d(ax, ay, bx, by, cx, cy); // >0 if ccw

/** Binary max-heap of [priority, tri, serial]. */
class Heap {
  constructor() {
    this.p = [];
    this.t = [];
    this.s = [];
  }
  get size() {
    return this.p.length;
  }
  push(pri, tri, ser) {
    let i = this.p.length;
    this.p.push(pri);
    this.t.push(tri);
    this.s.push(ser);
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (this.p[par] >= this.p[i]) break;
      this._swap(i, par);
      i = par;
    }
  }
  pop() {
    const out = [this.p[0], this.t[0], this.s[0]];
    const lp = this.p.pop(),
      lt = this.t.pop(),
      ls = this.s.pop();
    if (this.p.length) {
      this.p[0] = lp;
      this.t[0] = lt;
      this.s[0] = ls;
      let i = 0;
      const n = this.p.length;
      for (;;) {
        let l = 2 * i + 1,
          r = l + 1,
          m = i;
        if (l < n && this.p[l] > this.p[m]) m = l;
        if (r < n && this.p[r] > this.p[m]) m = r;
        if (m === i) break;
        this._swap(i, m);
        i = m;
      }
    }
    return out;
  }
  _swap(i, j) {
    [this.p[i], this.p[j]] = [this.p[j], this.p[i]];
    [this.t[i], this.t[j]] = [this.t[j], this.t[i]];
    [this.s[i], this.s[j]] = [this.s[j], this.s[i]];
  }
}

const EKEY = 1 << 21;
const ekey = (a, b) => (a < b ? a * EKEY + b : b * EKEY + a);

class Triangulation {
  constructor(cap = 1024) {
    this.nv = 0;
    this.px = new Float64Array(cap);
    this.py = new Float64Array(cap);
    this.vtri = new Int32Array(cap);
    this.nt = 0; // slots used
    this.tv = new Int32Array(3 * cap);
    this.tn = new Int32Array(3 * cap);
    this.alive = new Uint8Array(cap);
    this.serial = new Uint32Array(cap);
    this.nextSerial = 1;
    this.free = [];
    this.stampIn = new Uint32Array(cap);
    this.stampOut = new Uint32Array(cap);
    this.stamp = 1;
    this.last = 0;
  }
  _growV() {
    const cap = this.px.length * 2;
    const g = (a, T) => {
      const b = new T(cap);
      b.set(a);
      return b;
    };
    this.px = g(this.px, Float64Array);
    this.py = g(this.py, Float64Array);
    this.vtri = g(this.vtri, Int32Array);
  }
  _growT() {
    const cap = this.alive.length * 2;
    const g = (a, T, k = 1) => {
      const b = new T(cap * k);
      b.set(a);
      return b;
    };
    this.tv = g(this.tv, Int32Array, 3);
    this.tn = g(this.tn, Int32Array, 3);
    this.alive = g(this.alive, Uint8Array);
    this.serial = g(this.serial, Uint32Array);
    this.stampIn = g(this.stampIn, Uint32Array);
    this.stampOut = g(this.stampOut, Uint32Array);
  }
  addVertex(x, y) {
    if (this.nv >= this.px.length) this._growV();
    if (this.nv >= EKEY) throw new Error('mesh: too many vertices');
    this.px[this.nv] = x;
    this.py[this.nv] = y;
    this.vtri[this.nv] = -1;
    return this.nv++;
  }
  _newTri(a, b, c) {
    let t;
    if (this.free.length) t = this.free.pop();
    else {
      if (this.nt >= this.alive.length) this._growT();
      t = this.nt++;
    }
    this.tv[3 * t] = a;
    this.tv[3 * t + 1] = b;
    this.tv[3 * t + 2] = c;
    this.tn[3 * t] = this.tn[3 * t + 1] = this.tn[3 * t + 2] = -1;
    this.alive[t] = 1;
    this.serial[t] = this.nextSerial++;
    return t;
  }
  _kill(t) {
    this.alive[t] = 0;
    this.free.push(t);
  }
  /** Initialise with two triangles covering the rectangle corners (vertex ids 0..3 ccw). */
  initRect(x0, y0, x1, y1) {
    const v0 = this.addVertex(x0, y0),
      v1 = this.addVertex(x1, y0),
      v2 = this.addVertex(x1, y1),
      v3 = this.addVertex(x0, y1);
    // diagonal v0-v2
    const t0 = this._newTri(v0, v1, v2),
      t1 = this._newTri(v0, v2, v3);
    // t0: edge opposite v1 is (v2,v0) -> shared with t1 ; t1: edge opposite v3 is (v0,v2)... opposite v3 is (v0,v2)
    this.tn[3 * t0 + 1] = t1;
    this.tn[3 * t1 + 2] = t0;
    for (const t of [t0, t1]) for (let i = 0; i < 3; i++) this.vtri[this.tv[3 * t + i]] = t;
    this.last = t0;
  }
  /** Walk to the triangle containing p. Returns {tri} or {outside:true, tri, edge}. */
  locate(x, y, start) {
    let t = start >= 0 && this.alive[start] ? start : this.last;
    if (!this.alive[t]) t = this._anyAlive();
    const maxSteps = 4 * this.nt + 100;
    let rot = 0;
    for (let step = 0; step < maxSteps; step++) {
      let moved = false;
      for (let k = 0; k < 3; k++) {
        const i = (k + rot) % 3;
        const a = this.tv[3 * t + ((i + 1) % 3)],
          b = this.tv[3 * t + ((i + 2) % 3)];
        if (orient(this.px[a], this.py[a], this.px[b], this.py[b], x, y) < 0) {
          const n = this.tn[3 * t + i];
          if (n < 0) return { outside: true, tri: t, edge: i };
          t = n;
          moved = true;
          rot = (i + 1) % 3;
          break;
        }
      }
      if (!moved) {
        this.last = t;
        return { tri: t };
      }
    }
    // fallback brute force
    for (let q = 0; q < this.nt; q++) {
      if (!this.alive[q]) continue;
      let ok = true;
      for (let i = 0; i < 3 && ok; i++) {
        const a = this.tv[3 * q + ((i + 1) % 3)],
          b = this.tv[3 * q + ((i + 2) % 3)];
        if (orient(this.px[a], this.py[a], this.px[b], this.py[b], x, y) < 0) ok = false;
      }
      if (ok) return { tri: q };
    }
    return { outside: true, tri: t, edge: 0 };
  }
  _anyAlive() {
    for (let t = 0; t < this.nt; t++) if (this.alive[t]) return t;
    return -1;
  }
  inCircle(t, x, y) {
    const a = this.tv[3 * t],
      b = this.tv[3 * t + 1],
      c = this.tv[3 * t + 2];
    return incircle(this.px[a], this.py[a], this.px[b], this.py[b], this.px[c], this.py[c], x, y);
  }
  /** Compute the Bowyer-Watson cavity of point (x,y) starting from triangle t0 (containing it). */
  cavity(x, y, t0) {
    const st = ++this.stamp;
    const cav = [t0];
    this.stampIn[t0] = st;
    const bnd = []; // {a,b,outer,ct}
    for (let h = 0; h < cav.length; h++) {
      const t = cav[h];
      for (let i = 0; i < 3; i++) {
        const n = this.tn[3 * t + i];
        const a = this.tv[3 * t + ((i + 1) % 3)],
          b = this.tv[3 * t + ((i + 2) % 3)];
        if (n >= 0) {
          if (this.stampIn[n] === st) continue;
          if (this.stampOut[n] !== st) {
            if (this.inCircle(n, x, y) > 0) {
              this.stampIn[n] = st;
              cav.push(n);
              continue;
            }
            this.stampOut[n] = st;
          }
        }
        bnd.push({ a, b, outer: n, ct: t });
      }
    }
    return { cav, bnd };
  }
  /** Commit cavity with new vertex v (already added). Returns array of new triangle ids. */
  commit(v, cav, bnd) {
    const x = this.px[v],
      y = this.py[v];
    const byStart = new Map(),
      byEnd = new Map();
    const created = [];
    const info = [];
    for (const e of bnd) {
      if (e.outer < 0 && Math.abs(orient(this.px[e.a], this.py[e.a], this.px[e.b], this.py[e.b], x, y)) === 0) continue;
      const t = this._newTri(v, e.a, e.b);
      this.tn[3 * t] = e.outer;
      created.push(t);
      byStart.set(e.a, t);
      byEnd.set(e.b, t);
      info.push({ t, e });
    }
    for (const { t, e } of info) {
      const n2 = byEnd.get(e.a); // across edge (v,a): opposite b -> index 2
      if (n2 !== undefined) this.tn[3 * t + 2] = n2;
      const n1 = byStart.get(e.b); // across edge (b,v): opposite a -> index 1
      if (n1 !== undefined) this.tn[3 * t + 1] = n1;
      if (e.outer >= 0) {
        for (let i = 0; i < 3; i++) if (this.tn[3 * e.outer + i] === e.ct) this.tn[3 * e.outer + i] = t;
      }
      this.vtri[e.a] = t;
      this.vtri[e.b] = t;
    }
    this.vtri[v] = created[0];
    for (const t of cav) this._kill(t);
    this.last = created[0];
    return created;
  }
  /** Find the triangle (and local edge index i) having directed edge a->b, or -1. */
  findEdge(a, b) {
    const t0 = this.vtri[a];
    if (t0 < 0) return null;
    // rotate around a in both directions
    for (const dir of [0, 1]) {
      let t = t0;
      for (let guard = 0; guard < 1000; guard++) {
        // local index of a
        let ia = 0;
        if (this.tv[3 * t + 1] === a) ia = 1;
        else if (this.tv[3 * t + 2] === a) ia = 2;
        const u = this.tv[3 * t + ((ia + 1) % 3)],
          w = this.tv[3 * t + ((ia + 2) % 3)];
        if (u === b) return { t, i: (ia + 2) % 3, forward: true };
        if (w === b) return { t, i: (ia + 1) % 3, forward: false };
        // move to neighbour across edge (a,u) [opposite w, index ia+2] for dir 0, across (w,a) [opposite u, ia+1] for dir 1
        const n = this.tn[3 * t + (dir === 0 ? (ia + 2) % 3 : (ia + 1) % 3)];
        if (n < 0 || n === t0) break;
        t = n;
      }
    }
    return null;
  }
  circum(t) {
    const a = this.tv[3 * t],
      b = this.tv[3 * t + 1],
      c = this.tv[3 * t + 2];
    const ax = this.px[a],
      ay = this.py[a];
    const bx = this.px[b] - ax,
      by = this.py[b] - ay,
      cx = this.px[c] - ax,
      cy = this.py[c] - ay;
    const d = 2 * (bx * cy - by * cx);
    const b2 = bx * bx + by * by,
      c2 = cx * cx + cy * cy;
    const ux = (cy * b2 - by * c2) / d,
      uy = (bx * c2 - cx * b2) / d;
    return { x: ax + ux, y: ay + uy, r: Math.hypot(ux, uy) };
  }
}

/**
 * @typedef {Object} SizeSource
 * @property {number[][]} poly polygon (um)
 * @property {boolean} filled distance zero inside
 * @property {number} h0 edge length at distance 0
 */

/**
 * Mesh a rectangular domain containing tagged polygon loops.
 *
 * @param {Object} spec
 * @param {{x0:number,y0:number,x1:number,y1:number}} spec.rect hull rectangle (um)
 * @param {{poly:number[][], kind:number, id:number}[]} spec.loops PSLG loops (kind 1 region boundary, 2 electrode boundary)
 * @param {(x:number,y:number)=>{region:number, electrode:number}} spec.classify point classification (painter order)
 * @param {number} spec.maxEdge global max edge length (um)
 * @param {SizeSource[]} [spec.sources] graded size sources
 * @param {number[]} [spec.regionMaxEdge] per-region max edge (um) (index by region id; Infinity = none)
 * @param {number} [spec.grade] size growth per unit distance (default 0.35)
 * @param {number} [spec.minAngleBound] radius-edge ratio bound B (default 1.35 ~ 21.7 deg)
 * @param {number} [spec.minEdge] absolute floor for refinement (um)
 * @param {number} [spec.maxVertices]
 */
export function meshDomain(spec) {
  const t0 = performance.now();
  const { rect, loops, classify } = spec;
  const scale = Math.hypot(rect.x1 - rect.x0, rect.y1 - rect.y0);
  const tol = 1e-9 * scale;
  const grade = spec.grade ?? 0.35;
  const B = spec.minAngleBound ?? 1.35;
  const maxEdge = spec.maxEdge;
  const minEdge = spec.minEdge ?? 1e-4 * maxEdge;
  const maxVertices = spec.maxVertices ?? 600000;
  const sources = spec.sources ?? [];
  const regionMax = spec.regionMaxEdge ?? [];

  const pslg = buildPslg(loops, rect, tol);
  const T = new Triangulation(Math.max(1024, pslg.verts.length * 8));

  // vertices: rect corners first (ids 0..3), then PSLG verts mapped
  T.initRect(rect.x0, rect.y0, rect.x1, rect.y1);
  const vmap = new Int32Array(pslg.verts.length).fill(-1);
  const corner = (x, y) => {
    for (let i = 0; i < 4; i++) if (Math.hypot(T.px[i] - x, T.py[i] - y) <= tol) return i;
    return -1;
  };
  for (let i = 0; i < pslg.verts.length; i++) {
    const [x, y] = pslg.verts[i];
    const c = corner(x, y);
    if (c >= 0) vmap[i] = c;
  }
  for (let i = 0; i < pslg.verts.length; i++) {
    if (vmap[i] >= 0) continue;
    const [x, y] = pslg.verts[i];
    const loc = T.locate(x, y, T.last);
    if (loc.outside) throw new Error('mesh: PSLG vertex outside hull');
    const v = T.addVertex(x, y);
    const { cav, bnd } = T.cavity(x, y, loc.tri);
    T.commit(v, cav, bnd);
    vmap[i] = v;
  }

  // subsegments: key -> {elec, mask}
  const seg = new Map();
  for (const s of pslg.segs) seg.set(ekey(vmap[s.a], vmap[s.b]), { a: vmap[s.a], b: vmap[s.b], mask: s.mask, elec: s.elec });
  // initial DT may lack some segments: handled via encroachment splitting.

  const segQueue = [];
  const segQueued = new Set();
  const enqueueSeg = (a, b, force = false) => {
    const k = ekey(a, b);
    if (!seg.has(k)) return;
    if (!force && segQueued.has(k)) return;
    segQueued.add(k);
    segQueue.push([a, b, force]);
  };

  const encroaches = (a, b, x, y) => {
    const dx1 = x - T.px[a],
      dy1 = y - T.py[a],
      dx2 = x - T.px[b],
      dy2 = y - T.py[b];
    return dx1 * dx2 + dy1 * dy2 < -1e-12 * (dx1 * dx1 + dy1 * dy1 + dx2 * dx2 + dy2 * dy2);
  };

  /** Is subsegment (a,b) encroached by any existing vertex? */
  const segEncroached = (a, b) => {
    const e = T.findEdge(a, b);
    if (e) {
      // apex test (valid when the segment is a Delaunay edge)
      const t = e.t;
      const apex = T.tv[3 * t + e.i];
      if (encroaches(a, b, T.px[apex], T.py[apex])) return true;
      const n = T.tn[3 * t + e.i];
      if (n >= 0) {
        for (let k = 0; k < 3; k++) {
          const vv = T.tv[3 * n + k];
          if (vv !== a && vv !== b && T.tn[3 * n + k] === t) {
            if (encroaches(a, b, T.px[vv], T.py[vv])) return true;
          }
        }
      }
      return false;
    }
    for (let v = 0; v < T.nv; v++) {
      if (v === a || v === b) continue;
      if (encroaches(a, b, T.px[v], T.py[v])) return true;
    }
    return true; // not a DT edge => necessarily non-Gabriel
  };

  // initial encroachment scan
  for (const k of seg.keys()) {
    const s = seg.get(k);
    if (segEncroached(s.a, s.b)) enqueueSeg(s.a, s.b);
  }

  // size evaluation
  const sizeAt = (x, y) => {
    let h = maxEdge;
    const c = classify(x, y);
    const rm = regionMax[c.region];
    if (rm !== undefined && rm < h) h = rm;
    for (const s of sources) {
      const d = distPointPolygon(x, y, s.poly, s.filled);
      const hh = s.h0 + (s.grade ?? grade) * d;
      if (hh < h) h = hh;
    }
    return h;
  };

  const bad = new Heap();
  const retries = new Map();
  const retry = (t, ser) => {
    const n = (retries.get(ser) ?? 0) + 1;
    retries.set(ser, n);
    if (n <= 3) bad.push(0.5, t, ser);
  };
  const frozen = (x, y) => classify(x, y).electrode >= 0;

  const evalTri = (t) => {
    if (!T.alive[t]) return;
    const a = T.tv[3 * t],
      b = T.tv[3 * t + 1],
      c = T.tv[3 * t + 2];
    const cx = (T.px[a] + T.px[b] + T.px[c]) / 3,
      cy = (T.py[a] + T.py[b] + T.py[c]) / 3;
    if (frozen(cx, cy)) return;
    const lab = Math.hypot(T.px[a] - T.px[b], T.py[a] - T.py[b]),
      lbc = Math.hypot(T.px[b] - T.px[c], T.py[b] - T.py[c]),
      lca = Math.hypot(T.px[c] - T.px[a], T.py[c] - T.py[a]);
    const lmin = Math.min(lab, lbc, lca),
      lmax = Math.max(lab, lbc, lca);
    if (lmax < 2 * minEdge) return;
    const cc = T.circum(t);
    const ratio = cc.r / lmin;
    const h = sizeAt(cx, cy);
    const sz = lmax / h;
    let pri = 0;
    if (sz > 1) pri = Math.max(pri, 10 + sz);
    if (ratio > B && lmin > minEdge) pri = Math.max(pri, ratio);
    if (pri > 0) bad.push(pri, t, T.serial[t]);
  };

  const afterInsert = (v, created) => {
    for (const t of created) {
      // new triangle (v,a,b): edge (a,b) opposite v could be an encroached subsegment
      const a = T.tv[3 * t + 1],
        b = T.tv[3 * t + 2];
      const k = ekey(a, b);
      if (seg.has(k) && encroaches(a, b, T.px[v], T.py[v])) enqueueSeg(a, b);
    }
    for (const t of created) evalTri(t);
  };

  const splitSeg = (a, b) => {
    const k = ekey(a, b);
    const s = seg.get(k);
    if (!s) return;
    const len = Math.hypot(T.px[a] - T.px[b], T.py[a] - T.py[b]);
    if (len < 2 * minEdge) return;
    // Concentric-shell style split: if an endpoint is a "shared" vertex (belongs to another segment),
    // move the split point to a power-of-two distance from it to avoid ping-pong at small input angles.
    let t = 0.5;
    const sharedA = vertDegree.get(a) > 2,
      sharedB = vertDegree.get(b) > 2;
    if (sharedA !== sharedB) {
      // split at distance from shared vertex rounded to a power of two fraction of len
      const frac = 0.5;
      const dsh = len * frac;
      const p2 = Math.pow(2, Math.round(Math.log2(dsh / minEdge))) * minEdge;
      const tt = Math.min(0.75, Math.max(0.25, p2 / len));
      t = sharedA ? tt : 1 - tt;
    }
    const x = T.px[a] + t * (T.px[b] - T.px[a]),
      y = T.py[a] + t * (T.py[b] - T.py[a]);
    const loc = T.locate(x, y, T.vtri[a] >= 0 ? T.vtri[a] : T.last);
    if (loc.outside) return;
    const { cav, bnd } = T.cavity(x, y, loc.tri);
    // other subsegments destroyed / encroached by the midpoint must be re-queued
    const seenE = new Set();
    for (const tr of cav)
      for (let i = 0; i < 3; i++) {
        const ea = T.tv[3 * tr + ((i + 1) % 3)],
          eb = T.tv[3 * tr + ((i + 2) % 3)];
        const ek = ekey(ea, eb);
        if (ek === k || seenE.has(ek)) continue;
        seenE.add(ek);
        if (seg.has(ek) && encroaches(ea, eb, x, y)) enqueueSeg(ea, eb);
      }
    const v = T.addVertex(x, y);
    const created = T.commit(v, cav, bnd);
    seg.delete(k);
    segQueued.delete(k);
    seg.set(ekey(a, v), { a, b: v, mask: s.mask, elec: s.elec });
    seg.set(ekey(v, b), { a: v, b, mask: s.mask, elec: s.elec });
    vertDegree.set(v, 2);
    // new subsegments may be non-Gabriel / not DT edges
    for (const [p, q] of [
      [a, v],
      [v, b],
    ]) {
      if (segEncroached(p, q)) enqueueSeg(p, q);
    }
    afterInsert(v, created);
  };

  const vertDegree = new Map();
  for (const s of seg.values()) {
    vertDegree.set(s.a, (vertDegree.get(s.a) ?? 0) + 1);
    vertDegree.set(s.b, (vertDegree.get(s.b) ?? 0) + 1);
  }

  let nSegSplits = 0;
  const processSegs = () => {
    while (segQueue.length) {
      if (T.nv > maxVertices) throw new Error('mesh: vertex budget exceeded (check hints / small input angles)');
      const [a, b, force] = segQueue.pop();
      const k = ekey(a, b);
      segQueued.delete(k);
      if (!seg.has(k)) continue;
      if (force || segEncroached(a, b)) {
        splitSeg(a, b);
        nSegSplits++;
      }
    }
  };
  processSegs();

  // classify all initial triangles
  for (let t = 0; t < T.nt; t++) if (T.alive[t]) evalTri(t);

  let nInsert = 0;
  let guard = 0;
  while (bad.size) {
    if (T.nv > maxVertices) throw new Error('mesh: vertex budget exceeded (check hints / small input angles)');
    if (++guard > 5e7) throw new Error('mesh: iteration guard');
    const [, t, ser] = bad.pop();
    if (!T.alive[t] || T.serial[t] !== ser) continue;
    const cc = T.circum(t);
    if (!(Number.isFinite(cc.x) && Number.isFinite(cc.y))) continue;
    // clamp huge circumcenters (degenerate) -> treat as outside
    const loc = T.locate(cc.x, cc.y, t);
    if (loc.outside) {
      const tt = loc.tri;
      const i = loc.edge;
      const a = T.tv[3 * tt + ((i + 1) % 3)],
        b = T.tv[3 * tt + ((i + 2) % 3)];
      if (seg.has(ekey(a, b))) {
        enqueueSeg(a, b, true);
        processSegs();
        if (T.alive[t] && T.serial[t] === ser) retry(t, ser);
        else if (T.alive[t]) evalTri(t);
        continue;
      }
      continue; // cannot resolve (should not happen)
    }
    const { cav, bnd } = T.cavity(cc.x, cc.y, loc.tri);
    // encroachment of subsegments on cavity edges
    let enc = false;
    const seen = new Set();
    for (const tr of cav) {
      for (let i = 0; i < 3; i++) {
        const a = T.tv[3 * tr + ((i + 1) % 3)],
          b = T.tv[3 * tr + ((i + 2) % 3)];
        const k = ekey(a, b);
        if (seen.has(k)) continue;
        seen.add(k);
        if (seg.has(k) && encroaches(a, b, cc.x, cc.y)) {
          enqueueSeg(a, b, true);
          enc = true;
        }
      }
    }
    if (enc) {
      processSegs();
      if (T.alive[t]) {
        if (T.serial[t] === ser) retry(t, ser);
        else evalTri(t);
      }
      continue;
    }
    // too close to existing vertices guard
    const v = T.addVertex(cc.x, cc.y);
    const created = T.commit(v, cav, bnd);
    nInsert++;
    afterInsert(v, created);
    processSegs();
  }

  // ---- extract mesh ----
  const triList = [];
  const triRegion = [];
  const triElec = [];
  for (let t = 0; t < T.nt; t++) {
    if (!T.alive[t]) continue;
    const a = T.tv[3 * t],
      b = T.tv[3 * t + 1],
      c = T.tv[3 * t + 2];
    const cx = (T.px[a] + T.px[b] + T.px[c]) / 3,
      cy = (T.py[a] + T.py[b] + T.py[c]) / 3;
    const cl = classify(cx, cy);
    triList.push(t);
    triRegion.push(cl.region);
    triElec.push(cl.electrode);
  }

  return { T, seg, triList, triRegion, triElec, stats: { nSegSplits, nInsert, ms: performance.now() - t0 } };
}

/**
 * Compact a raw mesh to the dielectric (non-conductor) triangles.
 * @returns {{x:Float64Array,y:Float64Array,tri:Int32Array,triRegion:Int32Array,
 *  nodeElectrode:Int16Array, edges:{a:number,b:number,tri:number,electrode:number}[], domainEdges:number[]}}
 */
export function extractMesh(raw, { keepConductors = false } = {}) {
  const { T, seg, triList, triRegion, triElec } = raw;
  const used = new Int32Array(T.nv).fill(-1);
  const tris = [];
  const regs = [];
  const els = [];
  for (let k = 0; k < triList.length; k++) {
    if (!keepConductors && triElec[k] >= 0) continue;
    const t = triList[k];
    tris.push(t);
    regs.push(triRegion[k]);
    els.push(triElec[k]);
  }
  let nn = 0;
  for (const t of tris) for (let i = 0; i < 3; i++) if (used[T.tv[3 * t + i]] < 0) used[T.tv[3 * t + i]] = nn++;
  const x = new Float64Array(nn),
    y = new Float64Array(nn);
  for (let v = 0; v < T.nv; v++)
    if (used[v] >= 0) {
      x[used[v]] = T.px[v];
      y[used[v]] = T.py[v];
    }
  const tri = new Int32Array(3 * tris.length);
  tris.forEach((t, k) => {
    for (let i = 0; i < 3; i++) tri[3 * k + i] = used[T.tv[3 * t + i]];
  });
  const nodeElectrode = new Int16Array(nn).fill(-1);
  const edgeInfo = [];
  for (const s of seg.values()) {
    if (s.elec >= 0 && used[s.a] >= 0 && used[s.b] >= 0) {
      for (const v of [s.a, s.b]) {
        const u = used[v];
        if (nodeElectrode[u] >= 0 && nodeElectrode[u] !== s.elec) throw new Error('mesh: node shared by two electrodes');
        nodeElectrode[u] = s.elec;
      }
      edgeInfo.push({ a: used[s.a], b: used[s.b], electrode: s.elec });
    }
  }
  return {
    x,
    y,
    tri,
    triRegion: Int32Array.from(regs),
    triElectrode: Int16Array.from(els),
    nodeElectrode,
    electrodeEdges: edgeInfo,
    nNodes: nn,
    nTris: tris.length,
    stats: raw.stats,
  };
}

/** Triangle quality summary (min angle in degrees, area sum). */
export function meshQuality(m) {
  let minAng = 180,
    maxAng = 0,
    area = 0,
    amin = Infinity,
    amax = 0;
  for (let t = 0; t < m.nTris; t++) {
    const i = m.tri[3 * t],
      j = m.tri[3 * t + 1],
      k = m.tri[3 * t + 2];
    const P = [
      [m.x[i], m.y[i]],
      [m.x[j], m.y[j]],
      [m.x[k], m.y[k]],
    ];
    const a = 0.5 * ((P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[2][0] - P[0][0]) * (P[1][1] - P[0][1]));
    area += a;
    if (a < amin) amin = a;
    if (a > amax) amax = a;
    for (let q = 0; q < 3; q++) {
      const A = P[q],
        Bp = P[(q + 1) % 3],
        C = P[(q + 2) % 3];
      const v1 = [Bp[0] - A[0], Bp[1] - A[1]],
        v2 = [C[0] - A[0], C[1] - A[1]];
      const ang = (Math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(...v1) * Math.hypot(...v2))) * 180) / Math.PI;
      if (ang < minAng) minAng = ang;
      if (ang > maxAng) maxAng = ang;
    }
  }
  return { minAngleDeg: minAng, maxAngleDeg: maxAng, totalArea: area, minTriArea: amin, maxTriArea: amax };
}

export { polygonBBox, pointInPolygon, distPointSegment };
