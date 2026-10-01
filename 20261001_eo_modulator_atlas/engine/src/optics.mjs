// Scalar optical mode solver on a conforming triangle mesh (P1 FEM), shift-and-invert subspace iteration.
//
// Every result of this module carries the label `scalar_optical_not_full_vector`.
// Two scalar weak forms are available (both exact in their slab limits and tested against the slab
// dispersion relations):
//   form 'E'  (quasi-TE, psi = E_x):    -lap psi - k0^2 n^2 psi = -beta^2 psi
//   form 'H'  (quasi-TM, psi = H_x):    -div(n^-2 grad psi) + k0^2 psi = beta^2 n^-2 psi
// Quasi-TE uses the lab-frame n_xx component of each material, quasi-TM uses n_yy.
// License: GPL-3.0-or-later.

import { assemble, restrictMatrix, triangleGeometry } from './fem.mjs';
import { buildPattern, ndOrder, SparseCholesky, jacobiEigen, matVec } from './sparse.mjs';
import { CrossSection } from './section.mjs';
import { epsOptLab } from './materials.mjs';

export const OPTICAL_LABELS = ['scalar_optical_not_full_vector'];

/**
 * Build the optical sub-model (window mesh) of a cross-section.
 * Electrodes inside the window are painted over the regions with their material (needs a real index).
 */
export function buildOpticalModel(section, mats, opts) {
  const { window: win, polarization, lambda0Um } = opts;
  const g = section.geom;
  const wnd = win ?? g.optical_window;
  if (!wnd) throw new Error(`cross-section ${section.name}: optical_window is required for the optical mode stage`);
  const regions = [...g.regions.map((r) => ({ name: r.name, material: r.material, poly: r.poly }))];
  for (const e of g.electrodes) regions.push({ name: `electrode:${e.name}`, material: e.material, poly: e.poly });
  const geomOpt = {
    symmetry: 'none',
    mirror_x_um: 0,
    domain: { x: wnd.x, y: wnd.y },
    regions,
    electrodes: [],
    optical_window: null,
    mesh: {},
    boundary: opts.boundary ?? {},
  };
  const os = new CrossSection(geomOpt, `${section.name}:optical`);
  const comp = polarization === 'TM' ? 4 : 0; // eps_opt_yy index 4, eps_opt_xx index 0
  // reference index at lambda0 per region (selected component)
  const regionN0 = regions.map((r) => {
    const m = mats[r.material];
    if (!m) throw new Error(`optical: unknown material ${r.material}`);
    const e = epsOptLab(m, lambda0Um, lambda0Um);
    return e ? Math.sqrt(e[comp]) : null;
  });
  // which regions intersect the window (bounding-box test is enough for sizing)
  const inWin = (r) => {
    const b = os.regions.find((q) => q.name === r.name)?.bbox;
    return b && b.x1 >= wnd.x[0] && b.x0 <= wnd.x[1] && b.y1 >= wnd.y[0] && b.y0 <= wnd.y[1];
  };
  let nMax = 0;
  regions.forEach((r, i) => {
    if (inWin(r) && regionN0[i] !== null) nMax = Math.max(nMax, regionN0[i]);
  });
  const core = [];
  regions.forEach((r, i) => {
    if (inWin(r) && regionN0[i] !== null && regionN0[i] >= nMax - 0.02 * nMax) core.push(r);
  });
  const lam = lambda0Um;
  const hints = opts.hints ?? {};
  const scale = opts.meshScale ?? 1;
  const coreEdge = (hints.core_edge_um ?? lam / (nMax * 22)) * scale;
  const maxEdge = (hints.max_edge_um ?? lam / 8) * scale;
  const grade = hints.grade ?? 0.25;
  const sources = core.map((r) => ({ poly: r.poly, filled: true, h0: coreEdge, grade }));
  const mesh = os.buildMesh({ sourcesOnly: true, hints: { max_edge_um: maxEdge, grade }, extraSources: sources, scale: 1, minAngleBound: 1.35 });
  mesh.rect = os.rect;
  const geo = triangleGeometry(mesh);
  const pat = buildPattern(mesh.nNodes, mesh.tri);
  return { section, os, mats, mesh, geo, pat, regions, polarization, lambda0Um, window: wnd, boundary: opts.boundary ?? {}, nMax, coreNames: core.map((c) => c.name), coreEdge, maxEdge };
}

/** Per-triangle n^2 for a wavelength. */
export function n2Triangles(model, lambdaUm) {
  const comp = model.polarization === 'TM' ? 4 : 0;
  const per = model.regions.map((r) => {
    const m = model.mats[r.material];
    const e = epsOptLab(m, lambdaUm, model.lambda0Um);
    if (!e) throw new Error(`material ${r.material} has no optical index (conductor without n)`);
    return e[comp];
  });
  const out = new Float64Array(model.mesh.nTris);
  for (let t = 0; t < model.mesh.nTris; t++) out[t] = per[model.mesh.triRegion[t]];
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Solve for the guided modes at one wavelength.
 * @returns {{neff:number[], psi:Float64Array[], theta:number[], iterations:number, form:string, warm:any}}
 */
export function solveModesAt(model, lambdaUm, { numModes = 2, form = null, warm = null, tol = 1e-11, maxIter = 120 } = {}) {
  const { mesh, geo, pat } = model;
  const pol = model.polarization;
  const useForm = form ?? (pol === 'TM' ? 'H' : 'E');
  const k0 = (2 * Math.PI) / lambdaUm;
  const n2 = n2Triangles(model, lambdaUm);
  const nT = mesh.nTris;
  let nMax2 = 0;
  for (const v of n2) if (v > nMax2) nMax2 = v;
  let cTri, mA, mB;
  if (useForm === 'E') {
    cTri = new Float64Array(3 * nT);
    mA = new Float64Array(nT);
    mB = new Float64Array(nT).fill(1);
    for (let t = 0; t < nT; t++) {
      cTri[3 * t] = 1;
      cTri[3 * t + 2] = 1;
      mA[t] = -k0 * k0 * n2[t];
    }
  } else {
    cTri = new Float64Array(3 * nT);
    mA = new Float64Array(nT).fill(-k0 * k0);
    mB = new Float64Array(nT);
    for (let t = 0; t < nT; t++) {
      cTri[3 * t] = 1 / n2[t];
      cTri[3 * t + 2] = 1 / n2[t];
      mB[t] = 1 / n2[t];
    }
  }
  const Afull = assemble(mesh, geo, pat, cTri, mA);
  const Bfull = assemble(mesh, geo, pat, null, mB);

  // Dirichlet at window boundary (per side)
  const bc = model.boundary ?? {};
  const rect = mesh.rect;
  const tolx = 1e-9 * Math.hypot(rect.x1 - rect.x0, rect.y1 - rect.y0);
  const isFree = new Uint8Array(mesh.nNodes).fill(1);
  const side = (name) => (bc[name] ?? 'dirichlet') === 'dirichlet';
  for (let i = 0; i < mesh.nNodes; i++) {
    const x = mesh.x[i],
      y = mesh.y[i];
    if ((side('left') && Math.abs(x - rect.x0) < tolx) || (side('right') && Math.abs(x - rect.x1) < tolx) || (side('bottom') && Math.abs(y - rect.y0) < tolx) || (side('top') && Math.abs(y - rect.y1) < tolx)) isFree[i] = 0;
  }
  const sub = restrictMatrix(pat, Afull, isFree);
  const nf = sub.nf;
  const Av = sub.val;
  const Bv = new Float64Array(sub.pat.nnz);
  for (let q = 0; q < Bv.length; q++) Bv[q] = Bfull[sub.src[q]];
  const fx = new Float64Array(nf),
    fy = new Float64Array(nf);
  for (let i = 0; i < mesh.nNodes; i++)
    if (sub.map[i] >= 0) {
      fx[sub.map[i]] = mesh.x[i];
      fy[sub.map[i]] = mesh.y[i];
    }
  const perm = warm?.perm && warm.nf === nf ? warm.perm : ndOrder(sub.pat, fx, fy);
  const chol = new SparseCholesky(sub.pat, perm);
  const Cv = new Float64Array(sub.pat.nnz);
  const factorShift = (sigma) => {
    for (let q = 0; q < Cv.length; q++) Cv[q] = Av[q] - sigma * Bv[q];
    return chol.factor(Cv);
  };

  // lower bound of spectrum:  lambda_min >= -k0^2 n_max^2  (E form);  H form: -k0^2 n_max^2 as well
  let sigma = -k0 * k0 * nMax2 * (1 + 1e-3);
  if (!factorShift(sigma)) throw new Error('optics: shifted matrix not positive definite (check mesh / materials)');
  const p = numModes + 3;
  // initial vectors
  let X;
  if (warm?.X && warm.nf === nf && warm.X.length >= p) X = warm.X.slice(0, p).map((v) => Float64Array.from(v));
  else {
    X = [];
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296 - 0.5;
    };
    for (let k = 0; k < p; k++) {
      const v = new Float64Array(nf);
      // seed with smooth bump centered on the highest-index region plus randomness
      for (let i = 0; i < nf; i++) v[i] = rnd();
      X.push(v);
    }
  }
  const sp = { n: nf, rowptr: sub.pat.rowptr, colidx: sub.pat.colidx };
  const mv = (val, x) => matVec(sp, val, x);
  let theta = new Float64Array(p).fill(Infinity);
  let iters = 0;
  let converged = false;
  let shiftUpdated = false;
  let stable = 0;
  for (let it = 1; it <= maxIter; it++) {
    iters = it;
    const Y = X.map((x) => chol.solve(mv(Bv, x)));
    // Rayleigh-Ritz
    const AY = Y.map((y) => mv(Av, y));
    const BY = Y.map((y) => mv(Bv, y));
    const H = new Float64Array(p * p),
      M = new Float64Array(p * p);
    for (let i = 0; i < p; i++)
      for (let j = i; j < p; j++) {
        const h = dot(Y[i], AY[j]),
          m = dot(Y[i], BY[j]);
        H[i * p + j] = H[j * p + i] = h;
        M[i * p + j] = M[j * p + i] = m;
      }
    // Cholesky of M (dense)
    const L = new Float64Array(p * p);
    let okc = true;
    for (let i = 0; i < p && okc; i++)
      for (let j = 0; j <= i; j++) {
        let s = M[i * p + j];
        for (let k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
        if (i === j) {
          if (!(s > 0)) {
            okc = false;
            break;
          }
          L[i * p + i] = Math.sqrt(s);
        } else L[i * p + j] = s / L[j * p + j];
      }
    if (!okc) throw new Error('optics: subspace became rank deficient');
    // G = L^-1 H L^-T
    const T = new Float64Array(p * p);
    for (let c = 0; c < p; c++)
      for (let i = 0; i < p; i++) {
        let s = H[i * p + c];
        for (let k = 0; k < i; k++) s -= L[i * p + k] * T[k * p + c];
        T[i * p + c] = s / L[i * p + i];
      }
    const G = new Float64Array(p * p);
    for (let r = 0; r < p; r++)
      for (let i = 0; i < p; i++) {
        let s = T[r * p + i];
        for (let k = 0; k < i; k++) s -= L[i * p + k] * G[r * p + k];
        G[r * p + i] = s / L[i * p + i];
      }
    for (let i = 0; i < p; i++) for (let j = 0; j < i; j++) G[i * p + j] = G[j * p + i] = 0.5 * (G[i * p + j] + G[j * p + i]);
    const { values, vectors } = jacobiEigen(G, p);
    // back-transform vectors: V = L^-T W
    const V = new Float64Array(p * p);
    for (let c = 0; c < p; c++)
      for (let i = p - 1; i >= 0; i--) {
        let s = vectors[i * p + c];
        for (let k = i + 1; k < p; k++) s -= L[k * p + i] * V[k * p + c];
        V[i * p + c] = s / L[i * p + i];
      }
    const Xn = [];
    for (let c = 0; c < p; c++) {
      const v = new Float64Array(nf);
      for (let i = 0; i < p; i++) {
        const w = V[i * p + c];
        const yi = Y[i];
        for (let q = 0; q < nf; q++) v[q] += w * yi[q];
      }
      Xn.push(v);
    }
    let dmax = 0;
    for (let k = 0; k < numModes; k++) dmax = Math.max(dmax, Math.abs(values[k] - theta[k]) / Math.max(1, Math.abs(values[k])));
    theta = Float64Array.from(values);
    X = Xn;
    if (dmax < tol) {
      stable++;
      if (stable >= 2) {
        converged = true;
        break;
      }
    } else stable = 0;
    // one-time shift update closer to the lowest eigenvalue
    if (!shiftUpdated && it >= 4 && dmax < 1e-4) {
      const s2 = theta[0] - 0.15 * (theta[0] - sigma);
      if (factorShift(s2)) sigma = s2;
      else factorShift(sigma);
      shiftUpdated = true;
    }
  }
  // assemble outputs
  const neff = [];
  const psi = [];
  for (let k = 0; k < numModes; k++) {
    const th = theta[k];
    neff.push(Math.sqrt(Math.max(-th, 0)) / k0);
    const full = new Float64Array(mesh.nNodes);
    for (let i = 0; i < mesh.nNodes; i++) if (sub.map[i] >= 0) full[i] = X[k][sub.map[i]];
    // normalise: max |psi| = 1 with positive peak
    let mx = 0,
      sg = 1;
    for (let i = 0; i < full.length; i++)
      if (Math.abs(full[i]) > mx) {
        mx = Math.abs(full[i]);
        sg = Math.sign(full[i]);
      }
    for (let i = 0; i < full.length; i++) full[i] *= sg / mx;
    psi.push(full);
  }
  return { neff, psi, theta: Array.from(theta).slice(0, numModes), iterations: iters, converged, form: useForm, n2, k0, lambdaUm, warm: { X, nf, perm }, labels: OPTICAL_LABELS };
}

/** Group index by central finite difference of n_eff(lambda) with material dispersion included. */
export function groupIndex(model, lambdaUm, modeIndex, { delta = 0.01, ...solveOpts } = {}) {
  const base = solveModesAt(model, lambdaUm, { numModes: modeIndex + 2, ...solveOpts });
  const lo = solveModesAt(model, lambdaUm - delta, { numModes: modeIndex + 2, warm: base.warm, ...solveOpts });
  const hi = solveModesAt(model, lambdaUm + delta, { numModes: modeIndex + 2, warm: base.warm, ...solveOpts });
  const dn = (hi.neff[modeIndex] - lo.neff[modeIndex]) / (2 * delta);
  return { ng: base.neff[modeIndex] - lambdaUm * dn, dneffDlambda: dn, base, lo, hi };
}

/** Integral of f^2 over each triangle for a P1 field. */
export function triIntegralSquare(mesh, geo, f) {
  const out = new Float64Array(mesh.nTris);
  for (let t = 0; t < mesh.nTris; t++) {
    const a = f[mesh.tri[3 * t]],
      b = f[mesh.tri[3 * t + 1]],
      c = f[mesh.tri[3 * t + 2]];
    out[t] = (geo.area[t] / 6) * (a * a + b * b + c * c + a * b + b * c + c * a);
  }
  return out;
}
