import { capacitanceUnits, EPS0 } from "./constants.js";
import { parallelPlateCapacitance, twoCylinderCapacitance } from "./analytic.js";
import { electrodeContains, parseDomain, parseElectrodes } from "./geometry.js";
import { materialPropertyField, parseMaterials } from "./materials.js";
import { validateConfig } from "./validation.js";

export function solveConfig(config) {
  validateConfig(config);
  const sim = config.Simulation ?? {};
  const nx = Number(sim.mesh_nx ?? 81);
  const ny = Number(sim.mesh_ny ?? 61);
  const domain = parseDomain(config);
  const epsR = Number(config.Materials?.background?.eps_r ?? 1.0);
  const materials = parseMaterials(config);
  const electrodes = parseElectrodes(config);
  const mesh = makeStructuredTriMesh(domain, nx, ny);
  const stiffness = assembleStiffness(mesh, epsR);
  const { values: dirichlet, labels } = dirichletNodes(mesh, electrodes);
  const { phi, iterations, residual } = solveDirichlet(stiffness, dirichlet);
  const fields = computeFieldComponents(mesh, phi);
  const energy = fieldEnergy(stiffness, phi);
  const signalName = String(sim.signal_electrode ?? "signal");
  const signalNodes = [];
  for (const [node, label] of labels.entries()) {
    if (label === signalName) signalNodes.push(node);
  }
  const vSignal = electrodePotential(electrodes, signalName);
  const qSignal = electrodeCharge(stiffness, phi, signalNodes);
  const capacitanceEnergy = (2 * energy) / (vSignal * vSignal);
  const capacitanceCharge = Math.abs(qSignal / vSignal);
  const reference = referenceCapacitance(config, epsR);
  return {
    capacitanceEnergy,
    capacitanceCharge,
    energyPerLength: energy,
    phi,
    mesh,
    fields,
    materials,
    electrodeLabels: Array.from({ length: mesh.nodes.length }, (_, i) => labels.get(i) ?? null),
    iterations,
    residual,
    units: capacitanceUnits(capacitanceEnergy),
    reference,
  };
}

export function computeFieldComponents(mesh, phi) {
  const ex = new Float64Array(phi.length);
  const ey = new Float64Array(phi.length);
  const norm = new Float64Array(phi.length);
  const dx = (mesh.domain.xMax - mesh.domain.xMin) / (mesh.nx - 1);
  const dy = (mesh.domain.yMax - mesh.domain.yMin) / (mesh.ny - 1);
  for (let j = 0; j < mesh.ny; j += 1) {
    for (let i = 0; i < mesh.nx; i += 1) {
      const idx = j * mesh.nx + i;
      const left = j * mesh.nx + Math.max(i - 1, 0);
      const right = j * mesh.nx + Math.min(i + 1, mesh.nx - 1);
      const down = Math.max(j - 1, 0) * mesh.nx + i;
      const up = Math.min(j + 1, mesh.ny - 1) * mesh.nx + i;
      const ddx = i === 0 || i === mesh.nx - 1 ? dx : 2 * dx;
      const ddy = j === 0 || j === mesh.ny - 1 ? dy : 2 * dy;
      ex[idx] = -(phi[right] - phi[left]) / ddx;
      ey[idx] = -(phi[up] - phi[down]) / ddy;
      norm[idx] = Math.hypot(ex[idx], ey[idx]);
    }
  }
  return { Ex: ex, Ey: ey, normE: norm };
}

export function getPlotValues(result, quantity) {
  if (quantity === "phi") return result.phi;
  if (quantity === "Ex") return result.fields.Ex;
  if (quantity === "Ey") return result.fields.Ey;
  if (quantity === "normE") return result.fields.normE;
  return materialPropertyField(result.mesh, result.materials, quantity);
}

export function makeStructuredTriMesh(domain, nx, ny) {
  if (nx < 2 || ny < 2) throw new Error("mesh_nx and mesh_ny must be at least 2");
  const nodes = new Array(nx * ny);
  for (let j = 0; j < ny; j += 1) {
    const y = domain.yMin + ((domain.yMax - domain.yMin) * j) / (ny - 1);
    for (let i = 0; i < nx; i += 1) {
      const x = domain.xMin + ((domain.xMax - domain.xMin) * i) / (nx - 1);
      nodes[j * nx + i] = [x, y];
    }
  }
  const triangles = [];
  for (let j = 0; j < ny - 1; j += 1) {
    for (let i = 0; i < nx - 1; i += 1) {
      const n00 = j * nx + i;
      const n10 = n00 + 1;
      const n01 = n00 + nx;
      const n11 = n01 + 1;
      triangles.push([n00, n10, n11], [n00, n11, n01]);
    }
  }
  return { domain, nx, ny, nodes, triangles };
}

export function assembleStiffness(mesh, epsR) {
  const rows = Array.from({ length: mesh.nodes.length }, () => new Map());
  const eps = EPS0 * epsR;
  for (const tri of mesh.triangles) {
    const pts = tri.map((i) => mesh.nodes[i]);
    const area2 =
      (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) -
      (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]);
    const area = Math.abs(area2) / 2;
    if (area === 0) continue;
    const b = [
      pts[1][1] - pts[2][1],
      pts[2][1] - pts[0][1],
      pts[0][1] - pts[1][1],
    ];
    const c = [
      pts[2][0] - pts[1][0],
      pts[0][0] - pts[2][0],
      pts[1][0] - pts[0][0],
    ];
    for (let a = 0; a < 3; a += 1) {
      const ia = tri[a];
      for (let d = 0; d < 3; d += 1) {
        const value = (eps * (b[a] * b[d] + c[a] * c[d])) / (4 * area);
        rows[ia].set(tri[d], (rows[ia].get(tri[d]) ?? 0) + value);
      }
    }
  }
  return mapRowsToCsr(rows);
}

function mapRowsToCsr(rows) {
  const rowPtr = new Int32Array(rows.length + 1);
  let nnz = 0;
  for (let i = 0; i < rows.length; i += 1) {
    nnz += rows[i].size;
    rowPtr[i + 1] = nnz;
  }
  const colIdx = new Int32Array(nnz);
  const values = new Float64Array(nnz);
  let cursor = 0;
  for (let i = 0; i < rows.length; i += 1) {
    const entries = Array.from(rows[i].entries()).sort((a, b) => a[0] - b[0]);
    for (const [col, value] of entries) {
      colIdx[cursor] = col;
      values[cursor] = value;
      cursor += 1;
    }
  }
  return { n: rows.length, rowPtr, colIdx, values };
}

function dirichletNodes(mesh, electrodes) {
  const values = new Map();
  const labels = new Map();
  for (let i = 0; i < mesh.nodes.length; i += 1) {
    const [x, y] = mesh.nodes[i];
    for (const electrode of electrodes) {
      if (electrodeContains(electrode, x, y)) {
        values.set(i, electrode.potential);
        labels.set(i, electrode.name);
        break;
      }
    }
  }
  if (new Set(labels.values()).size < 2) {
    throw new Error("at least two electrodes must intersect mesh nodes");
  }
  return { values, labels };
}

function solveDirichlet(matrix, dirichlet, tol = 1e-10, maxiter = 20000) {
  const n = matrix.n;
  const free = [];
  const freeIndex = new Int32Array(n);
  freeIndex.fill(-1);
  const phi = new Float64Array(n);
  for (const [node, value] of dirichlet.entries()) phi[node] = value;
  for (let i = 0; i < n; i += 1) {
    if (!dirichlet.has(i)) {
      freeIndex[i] = free.length;
      free.push(i);
    }
  }
  const freeNodes = Int32Array.from(free);
  const rhs = new Float64Array(free.length);
  for (let rowIndex = 0; rowIndex < free.length; rowIndex += 1) {
    const node = free[rowIndex];
    let value = 0;
    for (let cursor = matrix.rowPtr[node]; cursor < matrix.rowPtr[node + 1]; cursor += 1) {
      const col = matrix.colIdx[cursor];
      if (dirichlet.has(col)) value -= matrix.values[cursor] * phi[col];
    }
    rhs[rowIndex] = value;
  }
  const cg = conjugateGradient(matrix, freeNodes, freeIndex, rhs, tol, maxiter);
  for (let i = 0; i < freeNodes.length; i += 1) phi[freeNodes[i]] = cg.x[i];
  return { phi, iterations: cg.iterations, residual: cg.residual };
}

function conjugateGradient(matrix, freeNodes, freeIndex, rhs, tol, maxiter) {
  const x = new Float64Array(rhs.length);
  const r = new Float64Array(rhs);
  const p = new Float64Array(r);
  const ap = new Float64Array(rhs.length);
  let rsold = dot(r, r);
  const rhsNorm = Math.sqrt(rsold) || 1.0;
  if (rhsNorm === 0) return { x, iterations: 0, residual: 0 };
  let iteration = 0;
  for (iteration = 1; iteration <= maxiter; iteration += 1) {
    csrFreeMatvec(matrix, freeNodes, freeIndex, p, ap);
    const denom = dot(p, ap);
    if (denom === 0) break;
    const alpha = rsold / denom;
    for (let i = 0; i < x.length; i += 1) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }
    const rsnew = dot(r, r);
    const residual = Math.sqrt(rsnew) / rhsNorm;
    if (residual < tol) return { x, iterations: iteration, residual };
    const beta = rsnew / rsold;
    for (let i = 0; i < p.length; i += 1) p[i] = r[i] + beta * p[i];
    rsold = rsnew;
  }
  return { x, iterations: iteration, residual: Math.sqrt(rsold) / rhsNorm };
}

function csrFreeMatvec(matrix, freeNodes, freeIndex, vec, out) {
  out.fill(0.0);
  for (let rowIndex = 0; rowIndex < freeNodes.length; rowIndex += 1) {
    const node = freeNodes[rowIndex];
    let value = 0;
    for (let cursor = matrix.rowPtr[node]; cursor < matrix.rowPtr[node + 1]; cursor += 1) {
      const colIndex = freeIndex[matrix.colIdx[cursor]];
      if (colIndex >= 0) value += matrix.values[cursor] * vec[colIndex];
    }
    out[rowIndex] = value;
  }
}

function fieldEnergy(matrix, phi) {
  let value = 0;
  for (let i = 0; i < matrix.n; i += 1) {
    let kphi = 0;
    for (let cursor = matrix.rowPtr[i]; cursor < matrix.rowPtr[i + 1]; cursor += 1) {
      kphi += matrix.values[cursor] * phi[matrix.colIdx[cursor]];
    }
    value += phi[i] * kphi;
  }
  return 0.5 * value;
}

function electrodeCharge(matrix, phi, nodes) {
  let charge = 0;
  for (const i of nodes) {
    for (let cursor = matrix.rowPtr[i]; cursor < matrix.rowPtr[i + 1]; cursor += 1) {
      charge += matrix.values[cursor] * phi[matrix.colIdx[cursor]];
    }
  }
  return charge;
}

function referenceCapacitance(config, epsR) {
  const outputs = config.Outputs ?? {};
  if (outputs.reference === "parallel_plate") {
    const capacitance = parallelPlateCapacitance(epsR, Number(outputs.plate_width), Number(outputs.plate_gap));
    return { name: "parallel_plate", capacitance, ...capacitanceUnits(capacitance) };
  }
  if (outputs.reference === "two_cylinders") {
    const capacitance = twoCylinderCapacitance(
      epsR,
      Number(outputs.radius),
      Number(outputs.center_distance),
    );
    return { name: "two_cylinders", capacitance, ...capacitanceUnits(capacitance) };
  }
  return null;
}

function electrodePotential(electrodes, name) {
  const electrode = electrodes.find((item) => item.name === name);
  if (!electrode) throw new Error(`signal electrode ${name} not found`);
  if (electrode.potential === 0) throw new Error("signal electrode potential cannot be zero");
  return electrode.potential;
}

function dot(a, b) {
  let value = 0;
  for (let i = 0; i < a.length; i += 1) value += a[i] * b[i];
  return value;
}
