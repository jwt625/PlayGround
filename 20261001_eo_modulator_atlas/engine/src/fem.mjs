// P1 triangle finite elements: anisotropic Poisson (electrostatics) and scalar Helmholtz-type matrices.
// Coordinates are micrometers; matrices are dimensionless 2D forms (capacitance per length is scale free).
// License: GPL-3.0-or-later.

import { buildPattern, entryIndex, ndOrder, SparseCholesky } from './sparse.mjs';

/** Per-triangle geometry: area and shape function gradients. */
export function triangleGeometry(mesh) {
  const nt = mesh.nTris;
  const area = new Float64Array(nt);
  const gx = new Float64Array(3 * nt),
    gy = new Float64Array(3 * nt);
  for (let t = 0; t < nt; t++) {
    const i = mesh.tri[3 * t],
      j = mesh.tri[3 * t + 1],
      k = mesh.tri[3 * t + 2];
    const x1 = mesh.x[i],
      y1 = mesh.y[i],
      x2 = mesh.x[j],
      y2 = mesh.y[j],
      x3 = mesh.x[k],
      y3 = mesh.y[k];
    const a2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    area[t] = 0.5 * a2;
    gx[3 * t] = (y2 - y3) / a2;
    gx[3 * t + 1] = (y3 - y1) / a2;
    gx[3 * t + 2] = (y1 - y2) / a2;
    gy[3 * t] = (x3 - x2) / a2;
    gy[3 * t + 1] = (x1 - x3) / a2;
    gy[3 * t + 2] = (x2 - x1) / a2;
  }
  return { area, gx, gy };
}

/**
 * Assemble the global matrix  K = sum_T A_T [gradN]^T C_T [gradN]  with 2x2 symmetric tensor
 * C_T = [cxx cxy; cxy cyy] (3 values per triangle) and optionally a mass-type term  m_T * (A/12)(1+delta_ij).
 * @param {Object} mesh
 * @param {{pat:any}} ctx
 * @param {Float64Array|null} cTri 3 per tri (xx, xy, yy) or null
 * @param {Float64Array|null} mTri 1 per tri or null
 */
export function assemble(mesh, geo, pat, cTri, mTri, val = null) {
  val = val ?? new Float64Array(pat.nnz);
  const nt = mesh.nTris;
  for (let t = 0; t < nt; t++) {
    const A = geo.area[t];
    const v = [mesh.tri[3 * t], mesh.tri[3 * t + 1], mesh.tri[3 * t + 2]];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        let s = 0;
        if (cTri) {
          const cxx = cTri[3 * t],
            cxy = cTri[3 * t + 1],
            cyy = cTri[3 * t + 2];
          const gax = geo.gx[3 * t + a],
            gay = geo.gy[3 * t + a],
            gbx = geo.gx[3 * t + b],
            gby = geo.gy[3 * t + b];
          s += A * (gax * (cxx * gbx + cxy * gby) + gay * (cxy * gbx + cyy * gby));
        }
        if (mTri) s += mTri[t] * ((A / 12) * (a === b ? 2 : 1));
        if (s !== 0) val[entryIndex(pat, v[a], v[b])] += s;
      }
    }
  }
  return val;
}

/** Restrict a full CSR matrix to a set of "free" rows/cols. Returns sub pattern/values and index maps. */
export function restrictMatrix(pat, val, isFree) {
  const n = pat.n;
  const map = new Int32Array(n).fill(-1);
  let nf = 0;
  for (let i = 0; i < n; i++) if (isFree[i]) map[i] = nf++;
  const rowptr = new Int32Array(nf + 1);
  let nnz = 0;
  for (let i = 0; i < n; i++) {
    if (map[i] < 0) continue;
    for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) if (map[pat.colidx[p]] >= 0) nnz++;
    rowptr[map[i] + 1] = nnz;
  }
  const colidx = new Int32Array(nnz);
  const src = new Int32Array(nnz);
  const v = new Float64Array(nnz);
  let q = 0;
  for (let i = 0; i < n; i++) {
    if (map[i] < 0) continue;
    for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) {
      const c = map[pat.colidx[p]];
      if (c >= 0) {
        colidx[q] = c;
        src[q] = p;
        v[q] = val[p];
        q++;
      }
    }
  }
  return { pat: { n: nf, rowptr, colidx, nnz }, val: v, src, map, nf };
}

/**
 * Anisotropic Poisson problem  div( eps grad phi ) = 0  with Dirichlet nodes.
 * Natural (zero normal D) boundary elsewhere.
 */
export class PoissonSolver {
  /**
   * @param {Object} mesh extracted mesh
   * @param {Float64Array} epsTri 3 per tri (xx,xy,yy), relative permittivity in-plane block
   * @param {Uint8Array} isDirichlet per node
   */
  constructor(mesh, epsTri, isDirichlet, geo = null, pat = null) {
    this.mesh = mesh;
    this.geo = geo ?? triangleGeometry(mesh);
    this.pat = pat ?? buildPattern(mesh.nNodes, mesh.tri);
    this.epsTri = epsTri;
    this.K = assemble(mesh, this.geo, this.pat, epsTri, null);
    this.isDirichlet = isDirichlet;
    const isFree = new Uint8Array(mesh.nNodes);
    for (let i = 0; i < mesh.nNodes; i++) isFree[i] = isDirichlet[i] ? 0 : 1;
    this.sub = restrictMatrix(this.pat, this.K, isFree);
    const fx = new Float64Array(this.sub.nf),
      fy = new Float64Array(this.sub.nf);
    for (let i = 0; i < mesh.nNodes; i++)
      if (this.sub.map[i] >= 0) {
        fx[this.sub.map[i]] = mesh.x[i];
        fy[this.sub.map[i]] = mesh.y[i];
      }
    const perm = ndOrder(this.sub.pat, fx, fy);
    this.chol = new SparseCholesky(this.sub.pat, perm);
    if (!this.chol.factor(this.sub.val)) throw new Error('PoissonSolver: stiffness not positive definite (floating region?)');
  }
  /**
   * @param {Float64Array} phiD potential at Dirichlet nodes (full-length array; free entries ignored)
   * @returns {{phi:Float64Array, residual:Float64Array}}
   */
  solve(phiD) {
    const n = this.mesh.nNodes,
      { pat, K } = this;
    const rhs = new Float64Array(this.sub.nf);
    for (let i = 0; i < n; i++) {
      const fi = this.sub.map[i];
      if (fi < 0) continue;
      let s = 0;
      for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) {
        const j = pat.colidx[p];
        if (this.isDirichlet[j]) s += K[p] * phiD[j];
      }
      rhs[fi] = -s;
    }
    const xf = this.chol.solve(rhs);
    const phi = new Float64Array(n);
    for (let i = 0; i < n; i++) phi[i] = this.sub.map[i] >= 0 ? xf[this.sub.map[i]] : phiD[i];
    // nodal residual r = K phi (charge-like quantity at Dirichlet nodes)
    const residual = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) s += K[p] * phi[pat.colidx[p]];
      residual[i] = s;
    }
    return { phi, residual };
  }
  /** Energy form  phi^T K phi  (dimensionless; multiply by eps0/2 for J/m). */
  energyForm(phi) {
    const { pat, K } = this;
    let e = 0;
    for (let i = 0; i < pat.n; i++) {
      let s = 0;
      for (let p = pat.rowptr[i]; p < pat.rowptr[i + 1]; p++) s += K[p] * phi[pat.colidx[p]];
      e += phi[i] * s;
    }
    return e;
  }
  /** Per-triangle energy form phi^T Ke phi. */
  triEnergyForm(phi) {
    const { mesh, geo, epsTri } = this;
    const out = new Float64Array(mesh.nTris);
    for (let t = 0; t < mesh.nTris; t++) {
      let ex = 0,
        ey = 0;
      for (let a = 0; a < 3; a++) {
        const p = phi[mesh.tri[3 * t + a]];
        ex += geo.gx[3 * t + a] * p;
        ey += geo.gy[3 * t + a] * p;
      }
      const cxx = epsTri[3 * t],
        cxy = epsTri[3 * t + 1],
        cyy = epsTri[3 * t + 2];
      out[t] = geo.area[t] * (ex * (cxx * ex + cxy * ey) + ey * (cxy * ex + cyy * ey));
    }
    return out;
  }
  /** Gradient of phi per triangle (V/um). */
  gradient(phi) {
    const { mesh, geo } = this;
    const gx = new Float64Array(mesh.nTris),
      gy = new Float64Array(mesh.nTris);
    for (let t = 0; t < mesh.nTris; t++)
      for (let a = 0; a < 3; a++) {
        const p = phi[mesh.tri[3 * t + a]];
        gx[t] += geo.gx[3 * t + a] * p;
        gy[t] += geo.gy[3 * t + a] * p;
      }
    return { gx, gy };
  }
}
