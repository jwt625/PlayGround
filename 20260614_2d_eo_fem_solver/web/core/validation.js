import { MATERIAL_PROPERTY_KEYS } from "./materials.js";

const RECT_KEYS = ["x_min", "x_max", "y_min", "y_max"];
const CIRCLE_KEYS = ["x", "y", "radius"];

export function validateConfig(config) {
  const errors = [];
  requireBlock(config, "Simulation", errors);
  requireBlock(config, "Domain", errors);
  requireBlock(config, "Materials", errors);
  requireBlock(config, "Electrodes", errors);
  validateDomain(config.Domain, errors);
  validateSimulation(config.Simulation ?? {}, errors);
  validateMaterials(config.Materials ?? {}, errors);
  validateElectrodes(config.Electrodes ?? {}, errors);
  validateOutputs(config.Outputs ?? {}, errors);
  if (errors.length > 0) {
    throw new Error(`Invalid simulation config:\n- ${errors.join("\n- ")}`);
  }
  return config;
}

function requireBlock(config, key, errors) {
  if (!isPlainObject(config?.[key])) errors.push(`${key} must be a mapping`);
}

function validateDomain(domain, errors) {
  if (!isPlainObject(domain)) return;
  for (const key of RECT_KEYS) requireFinite(domain, key, `Domain.${key}`, errors);
  const xMin = Number(domain.x_min);
  const xMax = Number(domain.x_max);
  const yMin = Number(domain.y_min);
  const yMax = Number(domain.y_max);
  if (Number.isFinite(xMin) && Number.isFinite(xMax) && xMax <= xMin) {
    errors.push("Domain width must be positive: x_max > x_min");
  }
  if (Number.isFinite(yMin) && Number.isFinite(yMax) && yMax <= yMin) {
    errors.push("Domain height must be positive: y_max > y_min");
  }
}

function validateSimulation(simulation, errors) {
  const nx = Number(simulation.mesh_nx ?? 81);
  const ny = Number(simulation.mesh_ny ?? 61);
  if (!Number.isInteger(nx) || nx < 3 || nx > 401) {
    errors.push("Simulation.mesh_nx must be an integer from 3 to 401");
  }
  if (!Number.isInteger(ny) || ny < 3 || ny > 401) {
    errors.push("Simulation.mesh_ny must be an integer from 3 to 401");
  }
}

function validateMaterials(materials, errors) {
  if (!isPlainObject(materials)) return;
  for (const [name, material] of Object.entries(materials)) {
    if (!isPlainObject(material)) {
      errors.push(`Materials.${name} must be a mapping`);
      continue;
    }
    const shape = material.shape ?? "background";
    if (!["background", "rectangle", "circle"].includes(shape)) {
      errors.push(`Materials.${name}.shape must be background, rectangle, or circle`);
    }
    if (material.eps_r === undefined && material.eps_r_xx === undefined) {
      errors.push(`Materials.${name} must define eps_r or eps_r_xx`);
    }
    for (const key of MATERIAL_PROPERTY_KEYS) {
      if (material[key] !== undefined) {
        requireFinite(material, key, `Materials.${name}.${key}`, errors);
      }
    }
    for (const key of ["eps_r", "eps_r_xx", "eps_r_yy"]) {
      if (material[key] !== undefined) {
        const value = Number(material[key]);
        if (Number.isFinite(value) && value <= 0) {
          errors.push(`Materials.${name}.${key} must be positive`);
        }
      }
    }
    validateShapeParams(material, `Materials.${name}`, errors);
  }
}

function validateElectrodes(electrodes, errors) {
  if (!isPlainObject(electrodes)) return;
  const names = Object.keys(electrodes);
  if (names.length < 2) errors.push("Electrodes must define at least two conductors");
  let hasNonzero = false;
  let hasZero = false;
  let hasPositive = false;
  let hasNegative = false;
  const potentials = new Set();
  for (const [name, electrode] of Object.entries(electrodes)) {
    if (!isPlainObject(electrode)) {
      errors.push(`Electrodes.${name} must be a mapping`);
      continue;
    }
    if (!["rectangle", "circle"].includes(electrode.shape)) {
      errors.push(`Electrodes.${name}.shape must be rectangle or circle`);
    }
    requireFinite(electrode, "potential", `Electrodes.${name}.potential`, errors);
    const potential = Number(electrode.potential);
    if (Number.isFinite(potential)) {
      potentials.add(potential);
      if (potential === 0) hasZero = true;
      else hasNonzero = true;
      if (potential > 0) hasPositive = true;
      if (potential < 0) hasNegative = true;
    }
    validateShapeParams(electrode, `Electrodes.${name}`, errors);
  }
  if (potentials.size < 2) errors.push("Electrodes must define at least two distinct potentials");
  if (!hasZero && !(hasPositive && hasNegative)) {
    errors.push("Electrodes should include either a 0 V reference conductor or bipolar differential drive");
  }
  if (!hasNonzero) errors.push("Electrodes should include at least one non-zero signal conductor");
}

function validateOutputs(outputs, errors) {
  if (!isPlainObject(outputs) || outputs.reference === undefined) return;
  if (outputs.reference === "parallel_plate") {
    requirePositive(outputs, "plate_width", "Outputs.plate_width", errors);
    requirePositive(outputs, "plate_gap", "Outputs.plate_gap", errors);
  } else if (outputs.reference === "two_cylinders") {
    requirePositive(outputs, "radius", "Outputs.radius", errors);
    requirePositive(outputs, "center_distance", "Outputs.center_distance", errors);
    const radius = Number(outputs.radius);
    const distance = Number(outputs.center_distance);
    if (Number.isFinite(radius) && Number.isFinite(distance) && distance <= 2 * radius) {
      errors.push("Outputs.center_distance must be larger than 2 * Outputs.radius");
    }
  } else {
    errors.push("Outputs.reference must be parallel_plate or two_cylinders");
  }
}

function validateShapeParams(block, path, errors) {
  if (block.shape === "rectangle") {
    for (const key of RECT_KEYS) requireFinite(block, key, `${path}.${key}`, errors);
    const xMin = Number(block.x_min);
    const xMax = Number(block.x_max);
    const yMin = Number(block.y_min);
    const yMax = Number(block.y_max);
    if (Number.isFinite(xMin) && Number.isFinite(xMax) && xMax <= xMin) {
      errors.push(`${path} rectangle width must be positive`);
    }
    if (Number.isFinite(yMin) && Number.isFinite(yMax) && yMax <= yMin) {
      errors.push(`${path} rectangle height must be positive`);
    }
  }
  if (block.shape === "circle") {
    for (const key of CIRCLE_KEYS) requireFinite(block, key, `${path}.${key}`, errors);
    const radius = Number(block.radius);
    if (Number.isFinite(radius) && radius <= 0) errors.push(`${path}.radius must be positive`);
  }
}

function requirePositive(block, key, path, errors) {
  requireFinite(block, key, path, errors);
  const value = Number(block?.[key]);
  if (Number.isFinite(value) && value <= 0) errors.push(`${path} must be positive`);
}

function requireFinite(block, key, path, errors) {
  const value = Number(block?.[key]);
  if (block?.[key] === undefined || !Number.isFinite(value)) {
    errors.push(`${path} must be a finite number`);
  }
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
