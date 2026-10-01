// Sparse symmetric linear algebra: CSR pattern from triangle meshes, nested-dissection ordering by
// node coordinates, and an up-looking sparse Cholesky (elimination-tree based; algorithm after the
// textbook CSparse cs_chol, T. Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006,
// re-implemented here from the published description).
// License: GPL-3.0-or-later.

/**
 * Build symmetric CSR pattern (full storage, including diagonal) from triangle connectivity.
 * @param {number} n
 * @param {Int32Array} tri
 */
export function buildPattern(n, tri) {
  const nt = tri.length / 3;
  const cnt = new Int32Array(n + 1);
  // collect entries (i,j) for all i,j in triangle (9 per tri), unique after sort
  const rows = new Array(n);
  for (let i = 0; i < n; i++) rows[i] = [i];
  for (let t = 0; t < nt; t++) {
    const a = tri[3 * t],
      b = tri[3 * t + 1],
      c = tri[3 * t + 2];
    rows[a].push(b, c);
    rows[b].push(a, c);
    rows[c].push(a, b);
  }
  let nnz = 0;
  const cols = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = rows[i].sort((p, q) => p - q);
    const u = [];
    for (let k = 0; k < r.length; k++) if (k === 0 || r[k] !== r[k - 1]) u.push(r[k]);
    cols[i] = u;
    nnz += u.length;
    cnt[i + 1] = nnz;
  }
  const rowptr = cnt;
  const colidx = new Int32Array(nnz);
  for (let i = 0; i < n; i++) colidx.set(cols[i], rowptr[i]);
  return { n, rowptr, colidx, nnz };
}

/** Index of entry (i,j) in CSR pattern (binary search). */
export function entryIndex(pat, i, j) {
  let lo = pat.rowptr[i],
    hi = pat.rowptr[i + 1] - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    const c = pat.colidx[m];
    if (c === j) return m;
    if (c < j) lo = m + 1;
    else hi = m - 1;
  }
  return -1;
}

export function matVec(pat, val, x, out) {
  const n = pat.n;
  out = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) s += val[p] * x[pat.colidx[p]];
    out[i] = s;
  }
  return out;
}

/** Nested dissection ordering by recursive coordinate bisection. Returns perm (new index -> old index). */
export function ndOrder(pat, x, y, leaf = 48) {
  const n = pat.n;
  const side = new Int8Array(n); // 0 unassigned, 1 A, 2 B
  const order = [];
  const stackTop = [];
  const rec = (nodes) => {
    if (nodes.length <= leaf) {
      for (const v of nodes) order.push(v);
      return;
    }
    let x0 = Infinity,
      x1 = -Infinity,
      y0 = Infinity,
      y1 = -Infinity;
    for (const v of nodes) {
      if (x[v] < x0) x0 = x[v];
      if (x[v] > x1) x1 = x[v];
      if (y[v] < y0) y0 = y[v];
      if (y[v] > y1) y1 = y[v];
    }
    const key = x1 - x0 >= y1 - y0 ? x : y;
    const sorted = nodes.slice().sort((p, q) => key[p] - key[q]);
    const half = sorted.length >> 1;
    for (let i = 0; i < sorted.length; i++) side[sorted[i]] = i < half ? 1 : 2;
    const A = [],
      S = [],
      Bn = [];
    for (let i = 0; i < sorted.length; i++) {
      const v = sorted[i];
      if (i < half) {
        let sep = false;
        for (let p = pat.rowptr[v]; p < pat.rowptr[v + 1]; p++) {
          const w = pat.colidx[p];
          if (w !== v && side[w] === 2) {
            sep = true;
            break;
          }
        }
        if (sep) S.push(v);
        else A.push(v);
      } else Bn.push(v);
    }
    for (const v of sorted) side[v] = 0;
    rec(A);
    rec(Bn);
    for (const v of S) order.push(v);
  };
  const all = new Array(n);
  for (let i = 0; i < n; i++) all[i] = i;
  void stackTop;
  rec(all);
  return Int32Array.from(order);
}

/**
 * Sparse Cholesky for a fixed symmetric pattern with fill-reducing permutation.
 * factor(values) may be called repeatedly (symbolic analysis is reused).
 */
export class SparseCholesky {
  /**
   * @param {{n:number,rowptr:Int32Array,colidx:Int32Array}} pat full symmetric CSR pattern
   * @param {Int32Array} perm new->old
   */
  constructor(pat, perm) {
    const n = pat.n;
    this.n = n;
    this.perm = perm;
    const pinv = new Int32Array(n);
    for (let i = 0; i < n; i++) pinv[perm[i]] = i;
    this.pinv = pinv;
    // permuted upper-triangular CSC: entries (r,c) r<=c ; column c lists rows
    const cnt = new Int32Array(n + 1);
    for (let i = 0; i < n; i++)
      for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) {
        const j = pat.colidx[p];
        const pi = pinv[i],
          pj = pinv[j];
        if (pi <= pj) cnt[pj + 1]++;
      }
    for (let i = 0; i < n; i++) cnt[i + 1] += cnt[i];
    const Cp = cnt;
    const nnzC = Cp[n];
    const Ci = new Int32Array(nnzC);
    this.srcIndex = new Int32Array(nnzC);
    const next = Cp.slice(0, n);
    for (let i = 0; i < n; i++)
      for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) {
        const j = pat.colidx[p];
        const pi = pinv[i],
          pj = pinv[j];
        if (pi <= pj) {
          const q = next[pj]++;
          Ci[q] = pi;
          this.srcIndex[q] = p;
        }
      }
    this.Cp = Cp;
    this.Ci = Ci;
    // etree
    const parent = new Int32Array(n).fill(-1);
    const anc = new Int32Array(n).fill(-1);
    for (let k = 0; k < n; k++) {
      for (let p = Cp[k]; p < Cp[k + 1]; p++) {
        let i = Ci[p];
        while (i !== -1 && i < k) {
          const inext = anc[i];
          anc[i] = k;
          if (inext === -1) parent[i] = k;
          i = inext;
        }
      }
    }
    this.parent = parent;
    // column counts via ereach
    const mark = new Int32Array(n);
    const s = new Int32Array(n);
    const colCount = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const top = this._ereach(k, mark, s, k + 1);
      for (let q = top; q < n; q++) colCount[s[q]]++;
      colCount[k]++;
    }
    const Lp = new Int32Array(n + 1);
    for (let k = 0; k < n; k++) Lp[k + 1] = Lp[k] + colCount[k];
    this.Lp = Lp;
    this.Li = new Int32Array(Lp[n]);
    this.Lx = new Float64Array(Lp[n]);
    this._mark = mark;
    this._s = s;
    this._x = new Float64Array(n);
    this._c = new Int32Array(n);
    this.nnzL = Lp[n];
    this.markStamp = n + 2;
  }
  _ereach(k, mark, s, stamp) {
    const { Cp, Ci, parent, n } = this;
    let top = n;
    mark[k] = stamp;
    for (let p = Cp[k]; p < Cp[k + 1]; p++) {
      let i = Ci[p];
      if (i > k) continue;
      let len = 0;
      for (; mark[i] !== stamp; i = parent[i]) {
        s[len++] = i;
        mark[i] = stamp;
      }
      while (len > 0) s[--top] = s[--len];
    }
    return top;
  }
  /**
   * Numeric factorization of A = values (aligned with the CSR pattern).
   * Returns true on success; false if matrix is not positive definite.
   */
  factor(values) {
    const { n, Cp, Ci, Lp, Li, Lx, srcIndex } = this;
    const mark = this._mark,
      s = this._s,
      x = this._x,
      c = this._c;
    mark.fill(0);
    let stamp = 1;
    for (let k = 0; k < n; k++) c[k] = Lp[k];
    for (let k = 0; k < n; k++) {
      stamp++;
      const top = this._ereach(k, mark, s, stamp);
      for (let p = Cp[k]; p < Cp[k + 1]; p++) x[Ci[p]] = values[srcIndex[p]];
      let d = x[k];
      x[k] = 0;
      for (let q = top; q < n; q++) {
        const i = s[q];
        const lki = x[i] / Lx[Lp[i]];
        x[i] = 0;
        const ci = c[i];
        for (let p = Lp[i] + 1; p < ci; p++) x[Li[p]] -= Lx[p] * lki;
        d -= lki * lki;
        Li[ci] = k;
        Lx[ci] = lki;
        c[i] = ci + 1;
      }
      if (!(d > 0)) {
        this.ok = false;
        return false;
      }
      Li[c[k]] = k;
      Lx[c[k]] = Math.sqrt(d);
      c[k]++;
    }
    this.ok = true;
    return true;
  }
  /** Solve A x = b. b and result in original ordering. */
  solve(b, out) {
    const { n, Lp, Li, Lx, perm } = this;
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = b[perm[i]];
    for (let j = 0; j < n; j++) {
      y[j] /= Lx[Lp[j]];
      const yj = y[j];
      for (let p = Lp[j] + 1; p < Lp[j + 1]; p++) y[Li[p]] -= Lx[p] * yj;
    }
    for (let j = n - 1; j >= 0; j--) {
      let v = y[j];
      for (let p = Lp[j] + 1; p < Lp[j + 1]; p++) v -= Lx[p] * y[Li[p]];
      y[j] = v / Lx[Lp[j]];
    }
    out = out ?? new Float64Array(n);
    for (let i = 0; i < n; i++) out[perm[i]] = y[i];
    return out;
  }
}

/** Convenience: ND-ordered Cholesky of a SPD CSR matrix. */
export function cholesky(pat, val, x, y) {
  const perm = ndOrder(pat, x, y);
  const ch = new SparseCholesky(pat, perm);
  if (!ch.factor(val)) throw new Error('cholesky: matrix not positive definite');
  return ch;
}

// ---- small dense helpers ----

/** Jacobi eigenvalue iteration for a symmetric dense matrix (row-major p x p). Returns {values, vectors(columns)}. */
export function jacobiEigen(A, p) {
  const a = Float64Array.from(A);
  const v = new Float64Array(p * p);
  for (let i = 0; i < p; i++) v[i * p + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += a[i * p + j] * a[i * p + j];
    if (off < 1e-30) break;
    for (let i = 0; i < p; i++)
      for (let j = i + 1; j < p; j++) {
        const aij = a[i * p + j];
        if (Math.abs(aij) < 1e-300) continue;
        const theta = (a[j * p + j] - a[i * p + i]) / (2 * aij);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cc = 1 / Math.sqrt(t * t + 1),
          ss = t * cc;
        for (let k = 0; k < p; k++) {
          const aik = a[i * p + k],
            ajk = a[j * p + k];
          a[i * p + k] = cc * aik - ss * ajk;
          a[j * p + k] = ss * aik + cc * ajk;
        }
        for (let k = 0; k < p; k++) {
          const aki = a[k * p + i],
            akj = a[k * p + j];
          a[k * p + i] = cc * aki - ss * akj;
          a[k * p + j] = ss * aki + cc * akj;
        }
        for (let k = 0; k < p; k++) {
          const vki = v[k * p + i],
            vkj = v[k * p + j];
          v[k * p + i] = cc * vki - ss * vkj;
          v[k * p + j] = ss * vki + cc * vkj;
        }
      }
  }
  const idx = Array.from({ length: p }, (_, i) => i).sort((i, j) => a[i * p + i] - a[j * p + j]);
  const values = idx.map((i) => a[i * p + i]);
  const vectors = new Float64Array(p * p);
  idx.forEach((src, dst) => {
    for (let k = 0; k < p; k++) vectors[k * p + dst] = v[k * p + src];
  });
  return { values, vectors };
}
