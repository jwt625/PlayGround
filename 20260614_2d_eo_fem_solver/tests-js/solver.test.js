import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import { parallelPlateCapacitance, twoCylinderCapacitance } from "../web/core/analytic.js";
import { EPS0 } from "../web/core/constants.js";
import { parseSimpleYaml } from "../web/core/config.js";
import { materialPropertyField } from "../web/core/materials.js";
import { getModePlotValues, solveOpticalModeConfig } from "../web/core/mode_solver.js";
import { quantityInfo } from "../web/core/quantities.js";
import { makeStructuredTriMesh, solveConfig } from "../web/core/solver.js";
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

test("isotropic tensor assembly matches scalar assembly", () => {
  const scalar = structuredSlabConfig(12.0);
  const tensor = structuredSlabConfig(1.0);
  tensor.Materials.slab = {
    ...tensor.Materials.slab,
    eps_r_xx: 12.0,
    eps_r_yy: 12.0,
    eps_r_xy: 0.0,
  };
  delete tensor.Materials.slab.eps_r;

  const scalarResult = solveConfig(scalar);
  const tensorResult = solveConfig(tensor);
  const relDiff =
    Math.abs(scalarResult.capacitanceEnergy - tensorResult.capacitanceEnergy) /
    scalarResult.capacitanceEnergy;
  assert.ok(relDiff < 1e-10, `relative difference ${relDiff}`);
  assert.match(tensorResult.permittivityModel, /anisotropic eps_r tensor/);
});

test("vertical-field capacitance responds more to eps_r_yy than eps_r_xx", () => {
  const highX = anisotropicSlabConfig(30.0, 1.0);
  const highY = anisotropicSlabConfig(1.0, 30.0);
  const highXResult = solveConfig(highX);
  const highYResult = solveConfig(highY);
  assert.ok(highYResult.capacitanceEnergy > 2 * highXResult.capacitanceEnergy);
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

test("structured refinement snaps mesh coordinates to rectangle boundaries", () => {
  const config = structuredSlabConfig(10.0);
  config.Simulation.refinement = {
    enabled: true,
    h_min: 0.1e-6,
    h_max: 1.2e-6,
    guard_layers: 1,
  };
  const result = solveConfig(config);
  assert.equal(result.mesh.stats.type, "structured refined");
  assert.ok(hasCoordinate(result.mesh.yCoords, -1e-6));
  assert.ok(hasCoordinate(result.mesh.yCoords, 1e-6));
  assert.ok(hasCoordinate(result.mesh.xCoords, -5e-6));
  assert.ok(hasCoordinate(result.mesh.xCoords, 5e-6));
  assert.ok(result.mesh.stats.minDy < result.mesh.stats.maxDy);
  assert.ok(result.capacitanceEnergy > 0);
});

test("makeStructuredTriMesh preserves uniform coordinates when refinement is disabled", () => {
  const config = parseSimpleYaml(PARALLEL_PLATE_CONFIG);
  const mesh = makeStructuredTriMesh(
    { xMin: -1, xMax: 1, yMin: -2, yMax: 2 },
    5,
    3,
    { refinement: { enabled: false } },
  );
  assert.deepEqual(Array.from(mesh.xCoords), [-1, -0.5, 0, 0.5, 1]);
  assert.deepEqual(Array.from(mesh.yCoords), [-2, 0, 2]);
  assert.equal(mesh.stats.type, "structured uniform");
  assert.equal(config.Simulation.mesh_nx, 121);
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

test("scalar optical mode example solves a guided Si strip mode", () => {
  const config = parseSimpleYaml(fs.readFileSync("examples/si_strip_mode.yaml", "utf8"));
  validateConfig(config);
  const result = solveOpticalModeConfig(config);
  assert.equal(result.physics, "optical_mode");
  assert.ok(result.mode.nEff > 1.444, `n_eff ${result.mode.nEff}`);
  assert.ok(result.mode.nEff < 3.476, `n_eff ${result.mode.nEff}`);
  assert.ok(result.mode.confinement > 0.1, `confinement ${result.mode.confinement}`);
  assert.equal(getModePlotValues(result, "mode_intensity").length, result.mesh.nodes.length);
});

test("scalar optical mode effective index increases with core index", () => {
  const low = parseSimpleYaml(fs.readFileSync("examples/si_strip_mode.yaml", "utf8"));
  const high = parseSimpleYaml(fs.readFileSync("examples/si_strip_mode.yaml", "utf8"));
  low.Simulation.mesh_nx = 81;
  low.Simulation.mesh_ny = 61;
  low.Simulation.mode_max_iterations = 450;
  high.Simulation.mesh_nx = 81;
  high.Simulation.mesh_ny = 61;
  high.Simulation.mode_max_iterations = 450;
  low.Materials.silicon_core.n = 2.0;
  high.Materials.silicon_core.n = 3.4;
  const lowResult = solveOpticalModeConfig(low);
  const highResult = solveOpticalModeConfig(high);
  assert.ok(highResult.mode.nEff > lowResult.mode.nEff + 0.05);
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

function anisotropicSlabConfig(epsRxx, epsRyy) {
  const config = structuredSlabConfig(1.0);
  config.Materials.slab = {
    ...config.Materials.slab,
    eps_r_xx: epsRxx,
    eps_r_yy: epsRyy,
    eps_r_xy: 0.0,
  };
  delete config.Materials.slab.eps_r;
  return config;
}

function hasCoordinate(coords, target) {
  return Array.from(coords).some((value) => Math.abs(value - target) < 1e-15);
}
