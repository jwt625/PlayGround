import { parseDomain } from "./geometry.js";
import { materialPropertyField, opticalIndexAt, parseMaterials } from "./materials.js";
import {
  containsRegion,
  coordinateStep,
  opticalIndexField,
  parseModeRectangle,
  regionToDomain,
  resolveTargetNeff,
  selectedMode,
  solveScalarModes,
} from "./mode_solver.js";
import { makeStructuredTriMesh } from "./solver.js";
import { validateConfig } from "./validation.js";

const C0 = 299792458;
const MU0 = 4e-7 * Math.PI;

export function solveVectorModeConfig(config) {
  validateConfig(config);
  const sim = config.Simulation ?? {};
  const nx = Number(sim.mesh_nx ?? 81);
  const ny = Number(sim.mesh_ny ?? 61);
  const wavelength = Number(sim.wavelength ?? sim.wavelength_m ?? 1.55e-6);
  const numModes = Math.max(1, Math.min(4, Math.round(Number(sim.num_modes ?? 1))));
  const maxIterations = Math.max(50, Math.round(Number(sim.mode_max_iterations ?? 650)));
  const tolerance = Number(sim.mode_tolerance ?? 2e-4);
  if (sim.refinement?.enabled === true) {
    throw new Error("vector_mode currently supports the uniform structured mesh only");
  }
  const domain = sim.mode_window ? regionToDomain(parseModeRectangle(sim.mode_window, "mode_window")) : parseDomain(config);
  const materials = parseMaterials(config);
  const mesh = makeStructuredTriMesh(domain, nx, ny, { refinement: { enabled: false }, materials });
  const nField = opticalIndexField(mesh, materials, "scalar");
  const targetNeff = resolveTargetNeff(sim.target_neff, nField);
  const targetRegion = sim.mode_region ? parseModeRectangle(sim.mode_region, "mode_region") : null;
  const candidateCount = Math.max(numModes, targetRegion ? 4 : numModes);
  const exCandidates = solveScalarModes(
    mesh,
    materials,
    wavelength,
    candidateCount,
    maxIterations,
    tolerance,
    "Ex",
    targetNeff,
    targetRegion,
  );
  const eyCandidates = solveScalarModes(
    mesh,
    materials,
    wavelength,
    candidateCount,
    maxIterations,
    tolerance,
    "Ey",
    targetNeff,
    targetRegion,
  );
  const modes = [...exCandidates, ...eyCandidates]
    .map((candidate, index) => reconstructVectorMode(mesh, materials, candidate, wavelength, targetNeff, targetRegion, index))
    .sort((a, b) => modeSortKey(a, targetNeff) - modeSortKey(b, targetNeff))
    .slice(0, numModes)
    .map((mode, modeIndex) => ({ ...mode, modeIndex }));
  return {
    physics: "vector_mode",
    wavelength,
    frequency: C0 / wavelength,
    mesh,
    materials,
    nField,
    targetNeff,
    targetRegion,
    polarization: "vector",
    modes,
    mode: modes[0],
    boundaryCondition: String(sim.optical_boundary ?? "pec"),
    vectorModel:
      `isotropic vector FDFD reconstruction, uniform grid, ${String(sim.optical_boundary ?? "pec")} outer boundary, target n_eff=${targetNeff.toFixed(4)}`,
  };
}

export function getVectorModePlotValues(result, quantity) {
  const mode = selectedMode(result);
  if (quantity === "mode_Ex") return mode.Ex;
  if (quantity === "mode_Ey") return mode.Ey;
  if (quantity === "mode_Ez") return mode.Ez;
  if (quantity === "mode_Hx") return mode.Hx;
  if (quantity === "mode_Hy") return mode.Hy;
  if (quantity === "mode_Hz") return mode.Hz;
  if (quantity === "mode_normE" || quantity === "mode_abs") return mode.normE;
  if (quantity === "mode_normH") return mode.normH;
  if (quantity === "mode" || quantity === "mode_intensity") return mode.intensity;
  if (quantity === "n") return result.nField;
  if (quantity === "n_xx" || quantity === "n_yy" || quantity === "n_zz") {
    return materialPropertyField(result.mesh, result.materials, quantity);
  }
  return materialPropertyField(result.mesh, result.materials, quantity);
}

function reconstructVectorMode(mesh, materials, candidate, wavelength, targetNeff, targetRegion, candidateIndex) {
  const length = mesh.nodes.length;
  const ex = new Float64Array(length);
  const ey = new Float64Array(length);
  const dominant = candidate.polarization === "Ey" ? ey : ex;
  dominant.set(candidate.field);
  const beta = candidate.beta;
  const omegaMu = (2 * Math.PI * C0 * MU0) / wavelength;
  const n2 = opticalN2Field(mesh, materials);
  const divEpsEt = divergenceEpsEt(mesh, n2, ex, ey);
  const ez = new Float64Array(length);
  if (beta > 0) {
    for (let i = 0; i < length; i += 1) ez[i] = -divEpsEt[i] / (beta * n2[i]);
  }
  const dEzDx = derivativeX(mesh, ez);
  const dEzDy = derivativeY(mesh, ez);
  const dEyDx = derivativeX(mesh, ey);
  const dExDy = derivativeY(mesh, ex);
  const hx = new Float64Array(length);
  const hy = new Float64Array(length);
  const hz = new Float64Array(length);
  const normE = new Float64Array(length);
  const normH = new Float64Array(length);
  const intensity = new Float64Array(length);
  let sumI = 0;
  let sumI2 = 0;
  let targetI = 0;
  let coreI = 0;
  let transverseE = 0;
  let exPower = 0;
  let eyPower = 0;
  const cellArea = coordinateStep(mesh.xCoords) * coordinateStep(mesh.yCoords);
  const backgroundN = backgroundIndex(materials);
  for (let i = 0; i < length; i += 1) {
    hx[i] = (beta * ey[i] - dEzDy[i]) / omegaMu;
    hy[i] = (dEzDx[i] - beta * ex[i]) / omegaMu;
    hz[i] = (dEyDx[i] - dExDy[i]) / omegaMu;
    normE[i] = Math.hypot(ex[i], ey[i], ez[i]);
    normH[i] = Math.hypot(hx[i], hy[i], hz[i]);
    intensity[i] = n2[i] * normE[i] * normE[i];
    const weightedI = intensity[i] * cellArea;
    sumI += weightedI;
    sumI2 += intensity[i] * intensity[i] * cellArea;
    transverseE += (ex[i] * ex[i] + ey[i] * ey[i]) * cellArea;
    exPower += ex[i] * ex[i] * cellArea;
    eyPower += ey[i] * ey[i] * cellArea;
    const [x, y] = mesh.nodes[i];
    if (Math.sqrt(n2[i]) > backgroundN + 1e-6) coreI += weightedI;
    if (targetRegion && containsRegion(targetRegion, x, y)) targetI += weightedI;
  }
  normalizeFields([ex, ey, ez, hx, hy, hz, normE, normH], intensity);
  const teFraction = transverseE > 0 ? exPower / transverseE : 0;
  const tmFraction = transverseE > 0 ? eyPower / transverseE : 0;
  return {
    modeIndex: candidateIndex,
    sourcePolarization: candidate.polarization,
    targetNeff,
    wavelength,
    beta,
    nEff: candidate.nEff,
    Ex: ex,
    Ey: ey,
    Ez: ez,
    Hx: hx,
    Hy: hy,
    Hz: hz,
    normE,
    normH,
    intensity,
    field: normE,
    absField: normE,
    teFraction,
    tmFraction,
    confinement: sumI > 0 ? coreI / sumI : 0,
    targetOverlap: targetRegion && sumI > 0 ? targetI / sumI : null,
    modeArea: sumI2 > 0 ? (sumI * sumI) / sumI2 : Infinity,
    iterations: candidate.iterations,
    residual: candidate.residual,
  };
}

function opticalN2Field(mesh, materials) {
  const values = new Float64Array(mesh.nodes.length);
  for (let i = 0; i < mesh.nodes.length; i += 1) {
    const [x, y] = mesh.nodes[i];
    const n = opticalIndexAt(materials, x, y, "scalar");
    values[i] = n * n;
  }
  return values;
}

function divergenceEpsEt(mesh, n2, ex, ey) {
  const epsEx = new Float64Array(ex.length);
  const epsEy = new Float64Array(ey.length);
  for (let i = 0; i < ex.length; i += 1) {
    epsEx[i] = n2[i] * ex[i];
    epsEy[i] = n2[i] * ey[i];
  }
  const dEpsExDx = derivativeX(mesh, epsEx);
  const dEpsEyDy = derivativeY(mesh, epsEy);
  const out = new Float64Array(ex.length);
  for (let i = 0; i < out.length; i += 1) out[i] = dEpsExDx[i] + dEpsEyDy[i];
  return out;
}

function derivativeX(mesh, values) {
  const out = new Float64Array(values.length);
  for (let j = 0; j < mesh.ny; j += 1) {
    for (let i = 0; i < mesh.nx; i += 1) {
      const left = j * mesh.nx + Math.max(i - 1, 0);
      const right = j * mesh.nx + Math.min(i + 1, mesh.nx - 1);
      const dx = mesh.xCoords[Math.min(i + 1, mesh.nx - 1)] - mesh.xCoords[Math.max(i - 1, 0)];
      out[j * mesh.nx + i] = dx > 0 ? (values[right] - values[left]) / dx : 0;
    }
  }
  return out;
}

function derivativeY(mesh, values) {
  const out = new Float64Array(values.length);
  for (let j = 0; j < mesh.ny; j += 1) {
    for (let i = 0; i < mesh.nx; i += 1) {
      const down = Math.max(j - 1, 0) * mesh.nx + i;
      const up = Math.min(j + 1, mesh.ny - 1) * mesh.nx + i;
      const dy = mesh.yCoords[Math.min(j + 1, mesh.ny - 1)] - mesh.yCoords[Math.max(j - 1, 0)];
      out[j * mesh.nx + i] = dy > 0 ? (values[up] - values[down]) / dy : 0;
    }
  }
  return out;
}

function normalizeFields(fields, intensity) {
  let maxE = 0;
  for (const value of fields[6]) maxE = Math.max(maxE, value);
  const eScale = maxE > 0 ? 1 / maxE : 1;
  for (const field of fields.slice(0, 3)) {
    for (let i = 0; i < field.length; i += 1) field[i] *= eScale;
  }
  for (let i = 0; i < fields[6].length; i += 1) fields[6][i] *= eScale;
  let maxH = 0;
  for (const value of fields[7]) maxH = Math.max(maxH, value);
  const hScale = maxH > 0 ? 1 / maxH : 1;
  for (const field of fields.slice(3, 6)) {
    for (let i = 0; i < field.length; i += 1) field[i] *= hScale;
  }
  for (let i = 0; i < fields[7].length; i += 1) fields[7][i] *= hScale;
  let maxI = 0;
  for (const value of intensity) maxI = Math.max(maxI, value);
  const iScale = maxI > 0 ? 1 / maxI : 1;
  for (let i = 0; i < intensity.length; i += 1) intensity[i] *= iScale;
}

function modeSortKey(mode, targetNeff) {
  const targetPenalty = Math.abs(mode.nEff - targetNeff);
  const overlapBonus = mode.targetOverlap === null ? 0 : mode.targetOverlap;
  return targetPenalty - 0.2 * overlapBonus - 0.05 * mode.confinement;
}

function backgroundIndex(materials) {
  const background = materials.find((material) => material.shape === "background");
  if (!background) return 1.0;
  if (background.properties.n !== undefined) return background.properties.n;
  if (background.properties.n_xx !== undefined) return background.properties.n_xx;
  return Math.sqrt(background.properties.eps_r ?? background.properties.eps_r_xx ?? 1.0);
}
