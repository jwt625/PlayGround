import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import { parallelPlateCapacitance, twoCylinderCapacitance } from "../web/core/analytic.js";
import { EPS0 } from "../web/core/constants.js";
import { parseSimpleYaml } from "../web/core/config.js";
import { materialPropertyField } from "../web/core/materials.js";
import { quantityInfo } from "../web/core/quantities.js";
import { solveConfig } from "../web/core/solver.js";
import { validateConfig } from "../web/core/validation.js";

const PARALLEL_PLATE_CONFIG = `
Simulation:
  name: parallel_plate_width_gt_gap
  mesh_nx: 121
  mesh_ny: 81
  signal_electrode: signal
Domain:
  x_min: -12e-6
  x_max: 12e-6
  y_min: -8e-6
  y_max: 8e-6
Materials:
  background:
    eps_r: 3.9
Electrodes:
  signal:
    shape: rectangle
    potential: 1.0
    x_min: -5e-6
    x_max: 5e-6
    y_min: 1e-6
    y_max: 1.4e-6
  ground:
    shape: rectangle
    potential: 0.0
    x_min: -5e-6
    x_max: 5e-6
    y_min: -1.4e-6
    y_max: -1e-6
Outputs:
  reference: parallel_plate
  plate_width: 10e-6
  plate_gap: 2e-6
`;

test("parseSimpleYaml parses nested scalar mappings", () => {
  const config = parseSimpleYaml(`
Simulation:
  name: demo
  mesh_nx: 11
Materials:
  background:
    eps_r: 3.9
`);
  assert.equal(config.Simulation.name, "demo");
  assert.equal(config.Simulation.mesh_nx, 11);
  assert.equal(config.Materials.background.eps_r, 3.9);
});

test("analytic references match closed forms", () => {
  assert.equal(parallelPlateCapacitance(2.0, 4.0, 8.0), EPS0);
  assert.ok(
    Math.abs(twoCylinderCapacitance(1.0, 0.5, 3.0) - Math.PI * EPS0 / Math.acosh(3.0)) <
      1e-26,
  );
});

test("parallel plate browser solver is close to analytic reference", () => {
  const config = parseSimpleYaml(PARALLEL_PLATE_CONFIG);
  const result = solveConfig(config);
  const relErr = Math.abs(result.capacitanceEnergy - result.reference.capacitance) / result.reference.capacitance;
  assert.ok(relErr < 0.35, `relative error ${relErr}`);
});

test("two cylinder browser solver is same order as analytic reference", () => {
  const config = parseSimpleYaml(fs.readFileSync("examples/two_cylinders.yaml", "utf8"));
  const result = solveConfig(config);
  const ratio = result.capacitanceEnergy / result.reference.capacitance;
  assert.ok(ratio > 0.45 && ratio < 1.8, `ratio ${ratio}`);
});

test("spatial scalar epsilon increases capacitance for a high-k slab", () => {
  const low = structuredSlabConfig(1.0);
  const high = structuredSlabConfig(30.0);
  const lowResult = solveConfig(low);
  const highResult = solveConfig(high);
  assert.ok(highResult.capacitanceEnergy > 3 * lowResult.capacitanceEnergy);
  assert.match(highResult.permittivityModel, /spatial scalar eps_r/);
});

test("overlapping material regions use later non-background material in assembly", () => {
  const highLast = overlappingSlabConfig(2.0, 20.0);
  const lowLast = overlappingSlabConfig(20.0, 2.0);
  const highLastResult = solveConfig(highLast);
  const lowLastResult = solveConfig(lowLast);
  assert.ok(highLastResult.capacitanceEnergy > 2 * lowLastResult.capacitanceEnergy);
});

test("solver exposes field components on the mesh", () => {
  const config = parseSimpleYaml(PARALLEL_PLATE_CONFIG);
  const result = solveConfig(config);
  assert.equal(result.fields.Ex.length, result.mesh.nodes.length);
  assert.equal(result.fields.Ey.length, result.mesh.nodes.length);
  assert.equal(result.fields.normE.length, result.mesh.nodes.length);
  assert.ok(Math.max(...result.fields.normE) > 0);
});

test("material property maps expose isotropic and anisotropic placeholders", () => {
  const config = parseSimpleYaml(fs.readFileSync("examples/material_stack.yaml", "utf8"));
  const result = solveConfig(config);
  const epsR = materialPropertyField(result.mesh, result.materials, "eps_r");
  const epsRxx = materialPropertyField(result.mesh, result.materials, "eps_r_xx");
  const r13 = materialPropertyField(result.mesh, result.materials, "r13");
  assert.ok(Math.max(...epsR) >= 30.0);
  assert.ok(Math.max(...epsRxx) >= 43.0);
  assert.ok(Math.max(...r13) >= 8.6e-12);
});

test("EO modulator examples parse, validate, solve, and expose EO maps", () => {
  for (const path of [
    "examples/tfln_partial_etched_mzm.yaml",
    "examples/bto_on_sin_plasmonic.yaml",
  ]) {
    const config = parseSimpleYaml(fs.readFileSync(path, "utf8"));
    validateConfig(config);
    const result = solveConfig(config);
    const rEff = materialPropertyField(result.mesh, result.materials, "r_eff");
    assert.ok(result.capacitanceEnergy > 0, `${path} capacitance`);
    assert.ok(Math.max(...rEff) > 0, `${path} r_eff`);
  }
});

test("validation rejects non-finite and non-physical inputs before solve", () => {
  const config = parseSimpleYaml(PARALLEL_PLATE_CONFIG);
  config.Domain.x_max = config.Domain.x_min;
  assert.throws(() => validateConfig(config), /Domain width must be positive/);

  const badMaterial = parseSimpleYaml(PARALLEL_PLATE_CONFIG);
  badMaterial.Materials.background.eps_r = 0;
  assert.throws(() => solveConfig(badMaterial), /eps_r must be positive/);
});

test("quantity metadata exposes labels, descriptions, and expressions", () => {
  assert.equal(quantityInfo("phi").description, "electrostatic potential");
  assert.equal(quantityInfo("Ex").expression, "-d(phi)/dx");
});

function structuredSlabConfig(epsR) {
  const config = parseSimpleYaml(PARALLEL_PLATE_CONFIG);
  config.Simulation.mesh_nx = 61;
  config.Simulation.mesh_ny = 61;
  config.Domain.x_min = -6e-6;
  config.Domain.x_max = 6e-6;
  config.Domain.y_min = -4e-6;
  config.Domain.y_max = 4e-6;
  config.Materials = {
    background: { eps_r: 1.0 },
    slab: {
      shape: "rectangle",
      eps_r: epsR,
      x_min: -6e-6,
      x_max: 6e-6,
      y_min: -1e-6,
      y_max: 1e-6,
    },
  };
  config.Electrodes.signal.y_min = 1e-6;
  config.Electrodes.signal.y_max = 1.2e-6;
  config.Electrodes.ground.y_min = -1.2e-6;
  config.Electrodes.ground.y_max = -1e-6;
  delete config.Outputs;
  return config;
}

function overlappingSlabConfig(firstEpsR, secondEpsR) {
  const config = structuredSlabConfig(1.0);
  config.Materials = {
    background: { eps_r: 1.0 },
    first: {
      shape: "rectangle",
      eps_r: firstEpsR,
      x_min: -6e-6,
      x_max: 6e-6,
      y_min: -1e-6,
      y_max: 1e-6,
    },
    second: {
      shape: "rectangle",
      eps_r: secondEpsR,
      x_min: -6e-6,
      x_max: 6e-6,
      y_min: -1e-6,
      y_max: 1e-6,
    },
  };
  return config;
}
