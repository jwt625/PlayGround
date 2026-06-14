import { parseDomain } from "./geometry.js";
import { materialPropertyField, opticalIndexAt, parseMaterials } from "./materials.js";
import { makeStructuredTriMesh } from "./solver.js";
import { validateConfig } from "./validation.js";

const C0 = 299792458;

export function solveOpticalModeConfig(config) {
  validateConfig(config);
  const sim = config.Simulation ?? {};
  const nx = Number(sim.mesh_nx ?? 81);
  const ny = Number(sim.mesh_ny ?? 61);
  const wavelength = Number(sim.wavelength ?? sim.wavelength_m ?? 1.55e-6);
  const numModes = Math.max(1, Math.min(4, Math.round(Number(sim.num_modes ?? 1))));
  const maxIterations = Math.max(50, Math.round(Number(sim.mode_max_iterations ?? 450)));
  const tolerance = Number(sim.mode_tolerance ?? 1e-8);
  if (sim.refinement?.enabled === true) {
    throw new Error("optical_mode currently supports the uniform structured mesh only");
  }
  const domain = parseDomain(config);
  const materials = parseMaterials(config);
  const mesh = makeStructuredTriMesh(domain, nx, ny, { refinement: { enabled: false }, materials });
  const nField = materialPropertyField(mesh, materials, "n");
  const modes = solveScalarModes(mesh, materials, wavelength, numModes, maxIterations, tolerance);
  return {
    physics: "optical_mode",
    wavelength,
    frequency: C0 / wavelength,
    mesh,
    materials,
    nField,
    modes,
    mode: modes[0],
    scalarModel:
      "scalar finite-difference Helmholtz eigenmode, Dirichlet outer boundary, beta^2 largest-real modes",
  };
}

export function getModePlotValues(result, quantity) {
  if (quantity === "mode") return result.mode.field;
  if (quantity === "mode_abs") return result.mode.absField;
  if (quantity === "mode_intensity") return result.mode.intensity;
  if (quantity === "n") return result.nField;
  return materialPropertyField(result.mesh, result.materials, quantity);
}

export function solveScalarModes(mesh, materials, wavelength, numModes = 1, maxIterations = 450, tolerance = 1e-8) {
  const k0 = (2 * Math.PI) / wavelength;
  const dx = mesh.xCoords[1] - mesh.xCoords[0];
  const dy = mesh.yCoords[1] - mesh.yCoords[0];
  const active = [];
  const activeIndex = new Int32Array(mesh.nodes.length);
  activeIndex.fill(-1);
  for (let j = 1; j < mesh.ny - 1; j += 1) {
    for (let i = 1; i < mesh.nx - 1; i += 1) {
      const node = j * mesh.nx + i;
      activeIndex[node] = active.length;
      active.push(node);
    }
  }
  const n2 = new Float64Array(active.length);
  for (let p = 0; p < active.length; p += 1) {
    const [x, y] = mesh.nodes[active[p]];
    const n = opticalIndexAt(materials, x, y);
    n2[p] = n * n;
  }
  const shift = 4 / (dx * dx) + 4 / (dy * dy);
  const previous = [];
  const modes = [];
  for (let modeIndex = 0; modeIndex < numModes; modeIndex += 1) {
    let vector = seededVector(active.length, modeIndex + 1);
    orthonormalize(vector, previous);
    let residual = Infinity;
    let iterations = 0;
    for (iterations = 1; iterations <= maxIterations; iterations += 1) {
      const next = applyShiftedOperator(mesh, active, activeIndex, n2, k0, dx, dy, shift, vector);
      orthonormalize(next, previous);
      normalize(next);
      residual = vectorDistance(next, vector);
      vector = next;
      if (residual < tolerance) break;
    }
    iterations = Math.min(iterations, maxIterations);
    const av = applyHelmholtzOperator(mesh, active, activeIndex, n2, k0, dx, dy, vector);
    const beta2 = dot(vector, av);
    const beta = Math.sqrt(Math.max(beta2, 0));
    const nEff = beta / k0;
    const field = expandModeField(mesh, active, vector);
    const mode = makeModeResult(mesh, materials, field, wavelength, beta, nEff, iterations, residual, modeIndex);
    modes.push(mode);
    previous.push(vector);
  }
  return modes;
}

function applyShiftedOperator(mesh, active, activeIndex, n2, k0, dx, dy, shift, vector) {
  const out = applyHelmholtzOperator(mesh, active, activeIndex, n2, k0, dx, dy, vector);
  for (let i = 0; i < out.length; i += 1) out[i] += shift * vector[i];
  return out;
}

function applyHelmholtzOperator(mesh, active, activeIndex, n2, k0, dx, dy, vector) {
  const out = new Float64Array(vector.length);
  const invDx2 = 1 / (dx * dx);
  const invDy2 = 1 / (dy * dy);
  for (let p = 0; p < active.length; p += 1) {
    const node = active[p];
    const left = activeIndex[node - 1];
    const right = activeIndex[node + 1];
    const down = activeIndex[node - mesh.nx];
    const up = activeIndex[node + mesh.nx];
    let value = (k0 * k0 * n2[p] - 2 * invDx2 - 2 * invDy2) * vector[p];
    if (left >= 0) value += invDx2 * vector[left];
    if (right >= 0) value += invDx2 * vector[right];
    if (down >= 0) value += invDy2 * vector[down];
    if (up >= 0) value += invDy2 * vector[up];
    out[p] = value;
  }
  return out;
}

function makeModeResult(mesh, materials, field, wavelength, beta, nEff, iterations, residual, modeIndex) {
  let maxAbs = 0;
  for (const value of field) maxAbs = Math.max(maxAbs, Math.abs(value));
  if (maxAbs > 0) {
    for (let i = 0; i < field.length; i += 1) field[i] /= maxAbs;
  }
  const absField = new Float64Array(field.length);
  const intensity = new Float64Array(field.length);
  let sumI = 0;
  let sumI2 = 0;
  let coreI = 0;
  const background = materials.find((material) => material.shape === "background");
  const backgroundN = background?.properties.n ?? Math.sqrt(background?.properties.eps_r ?? 1.0);
  const cellArea = coordinateStep(mesh.xCoords) * coordinateStep(mesh.yCoords);
  for (let i = 0; i < field.length; i += 1) {
    const value = field[i];
    const abs = Math.abs(value);
    const ii = value * value;
    absField[i] = abs;
    intensity[i] = ii;
    sumI += ii * cellArea;
    sumI2 += ii * ii * cellArea;
    const [x, y] = mesh.nodes[i];
    if (opticalIndexAt(materials, x, y) > backgroundN + 1e-6) coreI += ii * cellArea;
  }
  return {
    modeIndex,
    wavelength,
    beta,
    nEff,
    field,
    absField,
    intensity,
    confinement: sumI > 0 ? coreI / sumI : 0,
    modeArea: sumI2 > 0 ? (sumI * sumI) / sumI2 : Infinity,
    iterations,
    residual,
  };
}

function expandModeField(mesh, active, vector) {
  const field = new Float64Array(mesh.nodes.length);
  for (let i = 0; i < active.length; i += 1) field[active[i]] = vector[i];
  let maxIndex = 0;
  for (let i = 1; i < field.length; i += 1) {
    if (Math.abs(field[i]) > Math.abs(field[maxIndex])) maxIndex = i;
  }
  if (field[maxIndex] < 0) {
    for (let i = 0; i < field.length; i += 1) field[i] = -field[i];
  }
  return field;
}

function seededVector(length, seed) {
  const values = new Float64Array(length);
  for (let i = 0; i < length; i += 1) {
    const x = Math.sin((i + 1) * (12.9898 + seed) + seed * 78.233) * 43758.5453;
    values[i] = x - Math.floor(x) - 0.5;
  }
  normalize(values);
  return values;
}

function orthonormalize(vector, basis) {
  for (const base of basis) {
    const projection = dot(vector, base);
    for (let i = 0; i < vector.length; i += 1) vector[i] -= projection * base[i];
  }
  normalize(vector);
}

function normalize(vector) {
  const norm = Math.sqrt(dot(vector, vector)) || 1.0;
  for (let i = 0; i < vector.length; i += 1) vector[i] /= norm;
}

function vectorDistance(a, b) {
  let same = 0;
  let opposite = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d0 = a[i] - b[i];
    const d1 = a[i] + b[i];
    same += d0 * d0;
    opposite += d1 * d1;
  }
  return Math.sqrt(Math.min(same, opposite));
}

function dot(a, b) {
  let value = 0;
  for (let i = 0; i < a.length; i += 1) value += a[i] * b[i];
  return value;
}

function coordinateStep(coords) {
  return coords.length > 1 ? coords[1] - coords[0] : 1;
}
