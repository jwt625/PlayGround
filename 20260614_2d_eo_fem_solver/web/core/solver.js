import { capacitanceUnits, EPS0 } from "./constants.js";
import { parallelPlateCapacitance, twoCylinderCapacitance } from "./analytic.js";
import { electrodeContains, parseDomain, parseElectrodes } from "./geometry.js";
import {
  epsilonTensorAt,
  materialPropertyField,
  parseMaterials,
  usesSpatialPermittivity,
  usesTensorPermittivity,
} from "./materials.js";
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
  const mesh = makeStructuredTriMesh(domain, nx, ny, {
    refinement: sim.refinement,
    materials,
    electrodes,
  });
  const stiffness = assembleStiffness(mesh, materials);
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
    permittivityModel: permittivityModelDescription(materials, epsR),
  };
}

export function computeFieldComponents(mesh, phi) {
  const ex = new Float64Array(phi.length);
  const ey = new Float64Array(phi.length);
  const norm = new Float64Array(phi.length);
  const xCoords = mesh.xCoords;
  const yCoords = mesh.yCoords;
  for (let j = 0; j < mesh.ny; j += 1) {
    for (let i = 0; i < mesh.nx; i += 1) {
      const idx = j * mesh.nx + i;
      const left = j * mesh.nx + Math.max(i - 1, 0);
      const right = j * mesh.nx + Math.min(i + 1, mesh.nx - 1);
      const down = Math.max(j - 1, 0) * mesh.nx + i;
      const up = Math.min(j + 1, mesh.ny - 1) * mesh.nx + i;
      const ddx = xCoords[Math.min(i + 1, mesh.nx - 1)] - xCoords[Math.max(i - 1, 0)];
      const ddy = yCoords[Math.min(j + 1, mesh.ny - 1)] - yCoords[Math.max(j - 1, 0)];
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

export function makeStructuredTriMesh(domain, nx, ny, options = {}) {
  if (nx < 2 || ny < 2) throw new Error("mesh_nx and mesh_ny must be at least 2");
  const refined = options.refinement?.enabled === true;
  const xCoords = refined
    ? makeRefinedCoordinates(domain.xMin, domain.xMax, nx, collectMandatoryCoordinates("x", options), options.refinement)
    : linspace(domain.xMin, domain.xMax, nx);
  const yCoords = refined
    ? makeRefinedCoordinates(domain.yMin, domain.yMax, ny, collectMandatoryCoordinates("y", options), options.refinement)
    : linspace(domain.yMin, domain.yMax, ny);
  const actualNx = xCoords.length;
  const actualNy = yCoords.length;
  const nodes = new Array(actualNx * actualNy);
  for (let j = 0; j < actualNy; j += 1) {
    const y = yCoords[j];
    for (let i = 0; i < actualNx; i += 1) {
      const x = xCoords[i];
      nodes[j * actualNx + i] = [x, y];
    }
  }
  const triangles = [];
  for (let j = 0; j < actualNy - 1; j += 1) {
    for (let i = 0; i < actualNx - 1; i += 1) {
      const n00 = j * actualNx + i;
      const n10 = n00 + 1;
      const n01 = n00 + actualNx;
      const n11 = n01 + 1;
      triangles.push([n00, n10, n11], [n00, n11, n01]);
    }
  }
  return {
    domain,
    nx: actualNx,
    ny: actualNy,
    nodes,
    triangles,
    xCoords,
    yCoords,
    stats: meshStats(xCoords, yCoords, triangles.length, refined ? "structured refined" : "structured uniform"),
  };
}

function linspace(min, max, count) {
  const values = new Float64Array(count);
  for (let i = 0; i < count; i += 1) values[i] = min + ((max - min) * i) / (count - 1);
  return values;
}

function collectMandatoryCoordinates(axis, options) {
  const coordinates = [];
  for (const material of options.materials ?? []) {
    collectShapeCoordinates(axis, material.shape, material.params, coordinates);
  }
  for (const electrode of options.electrodes ?? []) {
    collectShapeCoordinates(axis, electrode.shape, electrode.params, coordinates);
  }
  return coordinates;
}

function collectShapeCoordinates(axis, shape, params, coordinates) {
  if (!params) return;
  if (shape === "rectangle") {
    coordinates.push(axis === "x" ? params.x_min : params.y_min);
    coordinates.push(axis === "x" ? params.x_max : params.y_max);
  } else if (shape === "circle") {
    const center = axis === "x" ? params.x : params.y;
    coordinates.push(center - params.radius, center, center + params.radius);
  }
}

function makeRefinedCoordinates(min, max, fallbackCount, mandatory, refinement) {
  const span = max - min;
  const fallbackStep = span / (fallbackCount - 1);
  const hMax = positiveNumber(refinement?.h_max) ?? fallbackStep;
  const hMin = positiveNumber(refinement?.h_min) ?? hMax / 4;
  const guardLayers = Math.max(0, Math.round(Number(refinement?.guard_layers ?? 2)));
  const seeds = [min, max];
  for (const value of mandatory) {
    if (!Number.isFinite(value) || value < min || value > max) continue;
    seeds.push(value);
    for (let layer = 1; layer <= guardLayers; layer += 1) {
      seeds.push(value - layer * hMin, value + layer * hMin);
    }
  }
  const sorted = uniqueSorted(seeds.filter((value) => value >= min && value <= max), span);
  const coordinates = [];
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const start = sorted[i];
    const end = sorted[i + 1];
    if (i === 0) coordinates.push(start);
    const interval = end - start;
    const segments = Math.max(1, Math.ceil(interval / hMax));
    for (let segment = 1; segment <= segments; segment += 1) {
      coordinates.push(start + (interval * segment) / segments);
    }
  }
  return Float64Array.from(uniqueSorted(coordinates, span));
}

function uniqueSorted(values, span) {
  const tolerance = Math.max(Math.abs(span) * 1e-12, 1e-18);
  const sorted = [...values].sort((a, b) => a - b);
  const unique = [];
  for (const value of sorted) {
    if (unique.length === 0 || Math.abs(value - unique[unique.length - 1]) > tolerance) {
      unique.push(value);
    }
  }
  return unique;
}

function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function meshStats(xCoords, yCoords, triangleCount, type) {
  const dx = coordinateSpacings(xCoords);
  const dy = coordinateSpacings(yCoords);
  return {
    type,
    nodes: xCoords.length * yCoords.length,
    triangles: triangleCount,
    minDx: Math.min(...dx),
    maxDx: Math.max(...dx),
    minDy: Math.min(...dy),
    maxDy: Math.max(...dy),
  };
}

function coordinateSpacings(coords) {
  const spacings = [];
  for (let i = 0; i < coords.length - 1; i += 1) spacings.push(coords[i + 1] - coords[i]);
  return spacings;
}

export function assembleStiffness(mesh, materialsOrEpsR) {
  const rows = Array.from({ length: mesh.nodes.length }, () => new Map());
  const useMaterials = Array.isArray(materialsOrEpsR);
  const homogeneousTensor = useMaterials
    ? null
    : { xx: Number(materialsOrEpsR), yy: Number(materialsOrEpsR), xy: 0.0 };
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
    const eps = useMaterials
      ? epsilonTensorAt(
          materialsOrEpsR,
          (pts[0][0] + pts[1][0] + pts[2][0]) / 3,
          (pts[0][1] + pts[1][1] + pts[2][1]) / 3,
        )
      : homogeneousTensor;
    for (let a = 0; a < 3; a += 1) {
      const ia = tri[a];
      for (let d = 0; d < 3; d += 1) {
        const value =
          (EPS0 *
            (eps.xx * b[a] * b[d] +
              eps.xy * b[a] * c[d] +
              eps.xy * c[a] * b[d] +
              eps.yy * c[a] * c[d])) /
          (4 * area);
        rows[ia].set(tri[d], (rows[ia].get(tri[d]) ?? 0) + value);
      }
    }
  }
  return mapRowsToCsr(rows);
}

function permittivityModelDescription(materials, backgroundEpsR) {
  const spatial = usesSpatialPermittivity(materials);
  const tensor = usesTensorPermittivity(materials);
  if (tensor && spatial) return "spatial anisotropic eps_r tensor (triangle centroid)";
  if (tensor) return "homogeneous anisotropic eps_r tensor";
  if (spatial) return "spatial scalar eps_r (triangle centroid)";
  return `homogeneous scalar eps_r=${backgroundEpsR}`;
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

function solveDirichlet(matrix, dirichlet, tol = 1e-6, maxiter = 20000) {
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
  const z = new Float64Array(rhs.length);
  const p = new Float64Array(rhs.length);
  const ap = new Float64Array(rhs.length);
  const invDiag = makeFreeInverseDiagonal(matrix, freeNodes);
  let rsold = dot(r, r);
  const rhsNorm = Math.sqrt(rsold) || 1.0;
  if (rhsNorm === 0) return { x, iterations: 0, residual: 0 };
  applyJacobi(invDiag, r, z);
  p.set(z);
  let rzold = dot(r, z);
  let iteration = 0;
  for (iteration = 1; iteration <= maxiter; iteration += 1) {
    csrFreeMatvec(matrix, freeNodes, freeIndex, p, ap);
    const denom = dot(p, ap);
    if (denom === 0) break;
    const alpha = rzold / denom;
    for (let i = 0; i < x.length; i += 1) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }
    const rsnew = dot(r, r);
    const residual = Math.sqrt(rsnew) / rhsNorm;
    if (residual < tol) return { x, iterations: iteration, residual };
    applyJacobi(invDiag, r, z);
    const rznew = dot(r, z);
    const beta = rznew / rzold;
    for (let i = 0; i < p.length; i += 1) p[i] = z[i] + beta * p[i];
    rsold = rsnew;
    rzold = rznew;
  }
  return { x, iterations: iteration, residual: Math.sqrt(rsold) / rhsNorm };
}

function makeFreeInverseDiagonal(matrix, freeNodes) {
  const invDiag = new Float64Array(freeNodes.length);
  for (let i = 0; i < freeNodes.length; i += 1) {
    const node = freeNodes[i];
    let diagonal = 0.0;
    for (let cursor = matrix.rowPtr[node]; cursor < matrix.rowPtr[node + 1]; cursor += 1) {
      if (matrix.colIdx[cursor] === node) {
        diagonal = matrix.values[cursor];
        break;
      }
    }
    invDiag[i] = diagonal !== 0 ? 1 / diagonal : 1.0;
  }
  return invDiag;
}

function applyJacobi(invDiag, input, output) {
  for (let i = 0; i < input.length; i += 1) output[i] = invDiag[i] * input[i];
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
