// 2D quasi-static electrostatics of a cross-section: anisotropic eps tensor, Dirichlet electrodes with
// weights, capacitance by energy and by terminal charge, C0' (all eps = 1), n_rf, Z0.
//
// Voltage normalisation (exact definition used everywhere in the engine):
//   Electrode i has potential  phi_i = w_i / V_line   so that the line voltage V_line_eff = 1 V, with
//     V_line = max_i(w_i) - min_i(w_i)            for symmetry none / x_mirror_even
//     V_line = 2 max_i |w_i|                       for symmetry x_mirror_odd (partner electrodes at -w_i)
//   (differential drive V+/V-: weights +-0.5 or +-1 give the same normalised problem; single ended: w = 1, 0).
//   C'  = 2 W' / (1 V)^2,   W' = energy per length of the FULL structure (x2 of the solved half for symmetry),
//   which equals the series capacitance seen by the line voltage; Z0 = 1/(c sqrt(C' C0')).
//   Terminal-charge form: C' = sum_i Q_i phi_i  (full structure), Q_i = eps0 * nodal residual sum on electrode i.
//
// License: GPL-3.0-or-later.

import { PoissonSolver, triangleGeometry } from './fem.mjs';
import { buildPattern } from './sparse.mjs';
import { epsRfLab } from './materials.mjs';

export const EPS0 = 8.8541878128e-12;
export const MU0 = 1.25663706212e-6;
export const C_LIGHT = 299792458;

/** Grid-accelerated triangle locator over a mesh. */
export class TriLocator {
  constructor(mesh, geo) {
    this.mesh = mesh;
    const { x, y, tri, nTris } = mesh;
    let x0 = Infinity,
      x1 = -Infinity,
      y0 = Infinity,
      y1 = -Infinity;
    for (let i = 0; i < x.length; i++) {
      if (x[i] < x0) x0 = x[i];
      if (x[i] > x1) x1 = x[i];
      if (y[i] < y0) y0 = y[i];
      if (y[i] > y1) y1 = y[i];
    }
    const n = Math.max(8, Math.ceil(Math.sqrt(nTris / 2)));
    this.x0 = x0;
    this.y0 = y0;
    this.nx = n;
    this.ny = n;
    this.dx = (x1 - x0) / n || 1;
    this.dy = (y1 - y0) / n || 1;
    // bucket by triangle bbox (cap bucket spans)
    const buckets = new Array(n * n);
    for (let t = 0; t < nTris; t++) {
      const a = tri[3 * t],
        b = tri[3 * t + 1],
        c = tri[3 * t + 2];
      const tx0 = Math.min(x[a], x[b], x[c]),
        tx1 = Math.max(x[a], x[b], x[c]),
        ty0 = Math.min(y[a], y[b], y[c]),
        ty1 = Math.max(y[a], y[b], y[c]);
      const i0 = Math.max(0, Math.floor((tx0 - x0) / this.dx)),
        i1 = Math.min(n - 1, Math.floor((tx1 - x0) / this.dx));
      const j0 = Math.max(0, Math.floor((ty0 - y0) / this.dy)),
        j1 = Math.min(n - 1, Math.floor((ty1 - y0) / this.dy));
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const k = j * n + i;
          (buckets[k] ??= []).push(t);
        }
    }
    this.buckets = buckets;
  }
  /** Triangle index containing (px,py) or -1. */
  find(px, py) {
    const { mesh, nx, ny } = this;
    const i = Math.floor((px - this.x0) / this.dx),
      j = Math.floor((py - this.y0) / this.dy);
    if (i < 0 || j < 0 || i >= nx || j >= ny) return -1;
    const lst = this.buckets[j * nx + i];
    if (!lst) return -1;
    const eps = 1e-12;
    let best = -1,
      bestMin = -Infinity;
    for (const t of lst) {
      const a = mesh.tri[3 * t],
        b = mesh.tri[3 * t + 1],
        c = mesh.tri[3 * t + 2];
      const x1 = mesh.x[a],
        y1 = mesh.y[a],
        x2 = mesh.x[b],
        y2 = mesh.y[b],
        x3 = mesh.x[c],
        y3 = mesh.y[c];
      const d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
      const l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / d;
      const l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / d;
      const l3 = 1 - l1 - l2;
      const m = Math.min(l1, l2, l3);
      if (m >= -eps) return t;
      if (m > bestMin) {
        bestMin = m;
        best = t;
      }
    }
    return bestMin > -1e-6 ? best : -1;
  }
}

/** Per-triangle in-plane relative permittivity block [xx, xy, yy]. */
export function epsTriangles(mesh, regionMaterials, isotropicOne = false) {
  const out = new Float64Array(3 * mesh.nTris);
  const cache = regionMaterials.map((m) => {
    if (isotropicOne || !m) return [1, 0, 1];
    const e = epsRfLab(m);
    return [e[0], e[1], e[4]];
  });
  for (let t = 0; t < mesh.nTris; t++) {
    const c = cache[mesh.triRegion[t]];
    out[3 * t] = c[0];
    out[3 * t + 1] = c[1];
    out[3 * t + 2] = c[2];
  }
  return out;
}

/**
 * Normalised electrode potentials (V) for unit line voltage.
 * @param {{weight:number}[]} electrodes
 * @param {string} symmetry
 */
export function normalizeWeights(electrodes, symmetry) {
  const w = electrodes.map((e) => e.weight);
  if (!w.length) throw new Error('no electrodes');
  let vline;
  if (symmetry === 'x_mirror_odd') vline = 2 * Math.max(...w.map(Math.abs));
  else vline = Math.max(...w) - Math.min(...w);
  if (!(vline > 0)) throw new Error('electrode weights give zero line voltage (need at least two distinct potentials)');
  return { vline, phi: w.map((v) => v / vline) };
}

/**
 * Solve a cross-section.
 * @param {import('./section.mjs').CrossSection} section
 * @param {{mesh?:any, meshScale?:number, resolvedMaterials:Object<string,any>}} opts
 */
export function solveElectrostatics(section, opts) {
  const t0 = performance.now();
  const g = section.geom;
  const mesh = opts.mesh ?? section.buildMesh({ scale: opts.meshScale ?? 1 });
  const tMesh = performance.now();
  const regionMats = section.regions.map((r) => opts.resolvedMaterials[r.material]);
  for (let i = 0; i < regionMats.length; i++) if (!regionMats[i]) throw new Error(`region ${section.regions[i].name}: unknown material ${section.regions[i].material}`);
  const geo = triangleGeometry(mesh);
  const pat = buildPattern(mesh.nNodes, mesh.tri);
  const { vline, phi: elecPhi } = normalizeWeights(section.electrodes, g.symmetry);
  const symFactor = g.symmetry === 'none' ? 1 : 2;

  // Dirichlet set: electrodes, mirror plane (odd), optional outer sides
  const isD = new Uint8Array(mesh.nNodes);
  const phiD = new Float64Array(mesh.nNodes);
  const tol = 1e-9 * Math.hypot(mesh.rect.x1 - mesh.rect.x0, mesh.rect.y1 - mesh.rect.y0);
  const bc = g.boundary ?? {};
  const rect = mesh.rect;
  for (let i = 0; i < mesh.nNodes; i++) {
    const x = mesh.x[i],
      y = mesh.y[i];
    let d = false;
    if (bc.left === 'dirichlet' && Math.abs(x - rect.x0) < tol) d = true;
    if (bc.right === 'dirichlet' && Math.abs(x - rect.x1) < tol) d = true;
    if (bc.bottom === 'dirichlet' && Math.abs(y - rect.y0) < tol) d = true;
    if (bc.top === 'dirichlet' && Math.abs(y - rect.y1) < tol) d = true;
    if (g.symmetry === 'x_mirror_odd' && Math.abs(x - g.mirror_x_um) < tol) d = true;
    if (d) {
      isD[i] = 1;
      phiD[i] = 0;
    }
  }
  for (let i = 0; i < mesh.nNodes; i++) {
    const e = mesh.nodeElectrode[i];
    if (e >= 0) {
      isD[i] = 1;
      phiD[i] = elecPhi[e];
    }
  }
  const nElec = section.electrodes.length;
  const epsT = epsTriangles(mesh, regionMats);
  const epsAir = epsTriangles(mesh, regionMats, true);
  const solver = new PoissonSolver(mesh, epsT, isD, geo, pat);
  const sol = solver.solve(phiD);
  const solver0 = new PoissonSolver(mesh, epsAir, isD, geo, pat);
  const sol0 = solver0.solve(phiD);

  const charges = (res) => {
    const q = new Float64Array(nElec);
    for (let i = 0; i < mesh.nNodes; i++) if (mesh.nodeElectrode[i] >= 0) q[mesh.nodeElectrode[i]] += res[i] * EPS0;
    return q;
  };
  const Q = charges(sol.residual);
  const Q0 = charges(sol0.residual);
  let sumQphi = 0,
    sumQ0phi = 0;
  for (let e = 0; e < nElec; e++) {
    sumQphi += Q[e] * elecPhi[e];
    sumQ0phi += Q0[e] * elecPhi[e];
  }
  const energyForm = solver.energyForm(sol.phi);
  const energyForm0 = solver0.energyForm(sol0.phi);
  const W = 0.5 * EPS0 * energyForm * symFactor; // J/m for V_line = 1 V (full structure)
  const W0 = 0.5 * EPS0 * energyForm0 * symFactor;
  const C_energy = 2 * W;
  const C0_energy = 2 * W0;
  const C_charge = sumQphi * symFactor;
  const C0_charge = sumQ0phi * symFactor;
  const Cp = C_energy; // report energy-based value as primary
  const C0p = C0_energy;
  const nrf = Math.sqrt(Cp / C0p);
  const Z0 = 1 / (C_LIGHT * Math.sqrt(Cp * C0p));
  const Lp = 1 / (C_LIGHT * C_LIGHT * C0p);

  // energy partition per region (for dielectric loss), fraction of total
  const triE = solver.triEnergyForm(sol.phi);
  const regionEnergy = new Float64Array(section.regions.length);
  let tot = 0;
  for (let t = 0; t < mesh.nTris; t++) {
    regionEnergy[mesh.triRegion[t]] += triE[t];
    tot += triE[t];
  }
  const regionEnergyFraction = Array.from(regionEnergy, (v) => v / tot);

  const grad = solver.gradient(sol.phi);
  const grad0 = solver0.gradient(sol0.phi);
  const t1 = performance.now();
  return {
    mesh,
    geo,
    pat,
    vline,
    symmetryFactor: symFactor,
    electrodePotentials: elecPhi,
    phi: sol.phi,
    phi0: sol0.phi,
    residual: sol.residual,
    residual0: sol0.residual,
    grad, // V/um per triangle (V_line = 1 V)
    grad0,
    epsTri: epsT,
    solver,
    solver0,
    isDirichlet: isD,
    charges: Q,
    charges0: Q0,
    capacitance: { energy: C_energy, charge: C_charge },
    capacitance0: { energy: C0_energy, charge: C0_charge },
    cPul: Cp,
    c0Pul: C0p,
    lPul: Lp,
    nRf: nrf,
    z0: Z0,
    regionEnergyFraction,
    timings_ms: { mesh: tMesh - t0, solve: t1 - tMesh, total: t1 - t0 },
    stats: { nNodes: mesh.nNodes, nTris: mesh.nTris },
  };
}
