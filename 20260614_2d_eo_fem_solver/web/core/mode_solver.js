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
  const polarization = normalizePolarization(sim.mode_polarization ?? sim.polarization ?? "scalar");
  if (sim.refinement?.enabled === true) {
    throw new Error("optical_mode currently supports the uniform structured mesh only");
  }
  const domain = sim.mode_window ? regionToDomain(parseModeRectangle(sim.mode_window, "mode_window")) : parseDomain(config);
  const materials = parseMaterials(config);
  const mesh = makeStructuredTriMesh(domain, nx, ny, { refinement: { enabled: false }, materials });
  const nField = opticalIndexField(mesh, materials, polarization);
  const targetNeff = resolveTargetNeff(sim.target_neff, nField);
  const targetRegion = sim.mode_region ? parseModeRectangle(sim.mode_region, "mode_region") : null;
  const modes = solveScalarModes(
    mesh,
    materials,
    wavelength,
    numModes,
    maxIterations,
    tolerance,
    polarization,
    targetNeff,
    targetRegion,
  );
  return {
    physics: "optical_mode",
    wavelength,
    frequency: C0 / wavelength,
    mesh,
    materials,
    nField,
    polarization,
    targetNeff,
    targetRegion,
    modes,
    mode: modes[0],
    scalarModel:
      `tensor-aware scalar finite-difference Helmholtz eigenmode (${polarization}), Dirichlet outer boundary, target n_eff=${targetNeff.toFixed(4)}`,
  };
}

export function getModePlotValues(result, quantity) {
  const mode = selectedMode(result);
  if (quantity === "mode") return mode.field;
  if (quantity === "mode_abs") return mode.absField;
  if (quantity === "mode_intensity") return mode.intensity;
  if (quantity === "n") return result.nField;
  if (quantity === "n_xx" || quantity === "n_yy" || quantity === "n_zz") {
    return materialPropertyField(result.mesh, result.materials, quantity);
  }
  return materialPropertyField(result.mesh, result.materials, quantity);
}

function selectedMode(result) {
  const modes = result.modes ?? [];
  const index = Number(result.selectedModeIndex ?? 0);
  return modes[Number.isInteger(index) && index >= 0 && index < modes.length ? index : 0] ?? result.mode;
}

export function solveScalarModes(
  mesh,
  materials,
  wavelength,
  numModes = 1,
  maxIterations = 450,
  tolerance = 1e-8,
  polarization = "scalar",
  targetNeff = null,
  targetRegion = null,
) {
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
    const n = opticalIndexAt(materials, x, y, polarization);
    n2[p] = n * n;
  }
  const shift = 4 / (dx * dx) + 4 / (dy * dy);
  const previous = [];
  const modes = [];
  const candidateCount = targetRegion ? Math.max(numModes, 4) : numModes;
  for (let modeIndex = 0; modeIndex < candidateCount; modeIndex += 1) {
    let vector = initialModeVector(mesh, active, modeIndex, targetRegion);
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
    const mode = makeModeResult(
      mesh,
      materials,
      field,
      wavelength,
      beta,
      nEff,
      iterations,
      residual,
      modeIndex,
      polarization,
      targetNeff,
      targetRegion,
    );
    mode.nodeScore = modeNodeScore(mesh, mode.field);
    modes.push(mode);
    previous.push(vector);
  }
  return targetRegion ? selectGuidedModes(modes, numModes) : modes;
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

function makeModeResult(
  mesh,
  materials,
  field,
  wavelength,
  beta,
  nEff,
  iterations,
  residual,
  modeIndex,
  polarization,
  targetNeff,
  targetRegion,
) {
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
  let targetI = 0;
  const background = materials.find((material) => material.shape === "background");
  const backgroundN = opticalIndexForMaterial(background, polarization);
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
    if (opticalIndexAt(materials, x, y, polarization) > backgroundN + 1e-6) coreI += ii * cellArea;
    if (targetRegion && containsRegion(targetRegion, x, y)) targetI += ii * cellArea;
  }
  return {
    modeIndex,
    polarization,
    targetNeff,
    wavelength,
    beta,
    nEff,
    field,
    absField,
    intensity,
    confinement: sumI > 0 ? coreI / sumI : 0,
    targetOverlap: targetRegion && sumI > 0 ? targetI / sumI : null,
    modeArea: sumI2 > 0 ? (sumI * sumI) / sumI2 : Infinity,
    iterations,
    residual,
  };
}

function selectGuidedModes(modes, count) {
  return [...modes]
    .sort((a, b) => {
      const nodeDiff = a.nodeScore - b.nodeScore;
      if (nodeDiff !== 0) return nodeDiff;
      return b.nEff - a.nEff;
    })
    .slice(0, count)
    .map((mode, index) => ({ ...mode, selectedModeIndex: index }));
}

function modeNodeScore(mesh, field) {
  const nx = mesh.nx;
  const ny = mesh.ny;
  let maxIndex = 0;
  for (let i = 1; i < field.length; i += 1) {
    if (Math.abs(field[i]) > Math.abs(field[maxIndex])) maxIndex = i;
  }
  const i0 = maxIndex % nx;
  const j0 = Math.floor(maxIndex / nx);
  return signCrossingsAlongRow(field, nx, j0) + signCrossingsAlongColumn(field, nx, ny, i0);
}

function signCrossingsAlongRow(field, nx, j) {
  let count = 0;
  let previous = 0;
  for (let i = 0; i < nx; i += 1) {
    const sign = Math.sign(field[j * nx + i]);
    if (sign !== 0 && previous !== 0 && sign !== previous) count += 1;
    if (sign !== 0) previous = sign;
  }
  return count;
}

function signCrossingsAlongColumn(field, nx, ny, i) {
  let count = 0;
  let previous = 0;
  for (let j = 0; j < ny; j += 1) {
    const sign = Math.sign(field[j * nx + i]);
    if (sign !== 0 && previous !== 0 && sign !== previous) count += 1;
    if (sign !== 0) previous = sign;
  }
  return count;
}

function parseModeRectangle(region, name) {
  if (!region) return null;
  if (region.shape !== "rectangle") {
    throw new Error(`Simulation.${name} currently supports shape: rectangle`);
  }
  return {
    shape: "rectangle",
    xMin: Number(region.x_min),
    xMax: Number(region.x_max),
    yMin: Number(region.y_min),
    yMax: Number(region.y_max),
  };
}

function containsRegion(region, x, y) {
  return x >= region.xMin && x <= region.xMax && y >= region.yMin && y <= region.yMax;
}

function regionToDomain(region) {
  return {
    xMin: region.xMin,
    xMax: region.xMax,
    yMin: region.yMin,
    yMax: region.yMax,
  };
}

function resolveTargetNeff(rawTarget, nField) {
  if (rawTarget === undefined || rawTarget === null || String(rawTarget).toLowerCase() === "auto") {
    return maxFinite(nField);
  }
  const value = Number(rawTarget);
  if (!Number.isFinite(value) || value <= 0) return maxFinite(nField);
  return value;
}

function maxFinite(values) {
  let max = 0;
  for (const value of values) {
    if (Number.isFinite(value) && value > max) max = value;
  }
  return max || 1.0;
}

function opticalIndexForMaterial(material, polarization) {
  if (!material) return 1.0;
  if (polarization === "Ex") return material.properties.n_xx ?? material.properties.n ?? fallbackMaterialIndex(material);
  if (polarization === "Ey") return material.properties.n_yy ?? material.properties.n ?? fallbackMaterialIndex(material);
  if (polarization === "Ez") return material.properties.n_zz ?? material.properties.n ?? fallbackMaterialIndex(material);
  return material.properties.n ?? material.properties.n_xx ?? fallbackMaterialIndex(material);
}

function fallbackMaterialIndex(material) {
  return Math.sqrt(material.properties.eps_r ?? material.properties.eps_r_xx ?? 1.0);
}

function opticalIndexField(mesh, materials, polarization) {
  const values = new Float64Array(mesh.nodes.length);
  for (let i = 0; i < mesh.nodes.length; i += 1) {
    const [x, y] = mesh.nodes[i];
    values[i] = opticalIndexAt(materials, x, y, polarization);
  }
  return values;
}

function normalizePolarization(value) {
  const key = String(value).trim().toLowerCase();
  if (key === "ex" || key === "x" || key === "te_x" || key === "horizontal") return "Ex";
  if (key === "ey" || key === "y" || key === "tm_y" || key === "vertical") return "Ey";
  if (key === "ez" || key === "z" || key === "longitudinal") return "Ez";
  return "scalar";
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

function initialModeVector(mesh, active, modeIndex, targetRegion) {
  if (!targetRegion) return seededVector(active.length, modeIndex + 1);
  const cx = 0.5 * (targetRegion.xMin + targetRegion.xMax);
  const cy = 0.5 * (targetRegion.yMin + targetRegion.yMax);
  const sx = Math.max((targetRegion.xMax - targetRegion.xMin) / 2, (mesh.domain.xMax - mesh.domain.xMin) / 12);
  const sy = Math.max((targetRegion.yMax - targetRegion.yMin) / 2, (mesh.domain.yMax - mesh.domain.yMin) / 12);
  const values = new Float64Array(active.length);
  for (let p = 0; p < active.length; p += 1) {
    const [x, y] = mesh.nodes[active[p]];
    const gx = (x - cx) / sx;
    const gy = (y - cy) / sy;
    let value = Math.exp(-0.5 * (gx * gx + gy * gy));
    if (modeIndex === 1) value *= gx;
    else if (modeIndex === 2) value *= gy;
    else if (modeIndex === 3) value *= gx * gy;
    values[p] = value;
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
