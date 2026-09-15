import { describe, expect, it } from "vitest";
import { createArrayConfig, wavenumber } from "../src/optics/geometry";
import { seededRandom, uniformCommands, commandsToActual } from "../src/optics/channels";
import { launchPower, sampleLaunchField, idealLaunchPower } from "../src/optics/launch";
import {
  channelFieldFast,
  farFieldAngular,
  intensityFast,
  launchQ,
  rayleighRange,
  totalFieldFast,
} from "../src/optics/gaussian";
import {
  angularCentroid,
  angularPib,
  angularSpectrumCriticalZ,
  computeFarFieldMap,
  fresnelFieldAtPoint,
  propagateAngularSpectrum,
  totalAngularPower,
} from "../src/optics/reference";
import { makeGrid } from "../src/optics/grid";
import {
  coincidentConfig,
  singleChannelConfig,
  statesFromPistons,
  twoEmitterXConfig,
  uniformActual,
} from "./helpers";

const c0 = createArrayConfig();

function magnitude2(u: { re: number; im: number }): number {
  return u.re * u.re + u.im * u.im;
}

function peakFarFieldMagnitude2(
  config: Parameters<typeof farFieldAngular>[0],
  states: Parameters<typeof farFieldAngular>[1],
  range: number,
  step: number,
): { sx: number; sy: number; value: number } {
  let best = { sx: 0, sy: 0, value: -Infinity };
  for (let sx = -range; sx <= range; sx += step) {
    for (let sy = -range; sy <= range; sy += step) {
      const v = magnitude2(farFieldAngular(config, states, sx, sy));
      if (v > best.value) best = { sx, sy, value: v };
    }
  }
  return best;
}

/** 1/e^2 intensity radius on the +X axis at plane z, by bisection. */
function measuredRadiusX(
  config: Parameters<typeof intensityFast>[0],
  states: Parameters<typeof intensityFast>[1],
  z: number,
): number {
  const peak = intensityFast(config, states, 0, 0, z);
  const target = peak / Math.E ** 2;
  let lo = 0;
  let hi = config.launchRadius_m;
  while (intensityFast(config, states, hi, 0, z) > target) hi *= 2;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (intensityFast(config, states, mid, 0, z) > target) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

describe("T02 independent optical reference", () => {
  it("1. coincident fields add as complex amplitudes (4x intensity, pi cancels)", () => {
    const config = coincidentConfig(2, { launchRadius_m: 0.18e-3, apertureRadius_m: 0.6e-3 });
    const inPhase = statesFromPistons(config, [0, 0]);
    const anti = statesFromPistons(config, [0, Math.PI]);
    const p = { x: 0.2e-3, y: 0.13e-3, z: 0 };
    const single = magnitude2(channelFieldFast(config, 0, inPhase[0], p.x, p.y, p.z));
    const both = magnitude2(totalFieldFast(config, inPhase, p.x, p.y, p.z));
    const cancelled = magnitude2(totalFieldFast(config, anti, p.x, p.y, p.z));
    expect(single).toBeGreaterThan(0);
    expect(both / single).toBeCloseTo(4, 6);
    expect(cancelled).toBeLessThan(single * 1e-12);
  });

  it("2. global piston leaves intensity, centroid, and PIB invariant", () => {
    const mapGrid = makeGrid(256, 24e-6);
    const s0 = uniformActual(c0);
    const s1 = statesFromPistons(c0, Array(c0.channelIds.length).fill(0.37));
    const map0 = computeFarFieldMap(c0, s0, mapGrid);
    const map1 = computeFarFieldMap(c0, s1, mapGrid);

    const p0 = totalAngularPower(map0);
    const p1 = totalAngularPower(map1);
    expect(Math.abs(p1 - p0) / p0).toBeLessThan(1e-9);

    const ctr0 = angularCentroid(map0);
    const ctr1 = angularCentroid(map1);
    expect(ctr0).not.toBeNull();
    expect(ctr1).not.toBeNull();
    expect(ctr1!.x).toBeCloseTo(ctr0!.x, 9);
    expect(ctr1!.y).toBeCloseTo(ctr0!.y, 9);

    const target = { x: 0, y: 0, z: 1 };
    const pib0 = angularPib(map0, target, 0.01);
    const pib1 = angularPib(map1, target, 0.01);
    expect(Math.abs(pib1 - pib0)).toBeLessThan(1e-9);

    const p = { x: 0.4e-3, y: -0.2e-3, z: 0.3 };
    const i0 = magnitude2(totalFieldFast(c0, s0, p.x, p.y, p.z));
    const i1 = magnitude2(totalFieldFast(c0, s1, p.x, p.y, p.z));
    expect(Math.abs(i1 - i0) / i0).toBeLessThan(1e-12);
  });

  it("3. positive/negative phase ramps steer both axes toward the predicted direction", () => {
    const lambda = 1550e-9;
    const config = createArrayConfig({
      pitch_m: 5 * lambda,
      launchRadius_m: 2e-6,
      apertureRadius_m: 2e-5,
    });
    const step = 5e-4;
    const range = 0.06;

    const cases: [number, number][] = [
      [0.02, 0],
      [-0.02, 0],
      [0, 0.02],
      [0, -0.02],
    ];
    for (const [sx, sy] of cases) {
      const k = wavenumber(config.wavelength_m);
      const pistons = config.channelIds.map((_, i) =>
        k * (sx * config.x_m[i] + sy * config.y_m[i]),
      );
      const states = statesFromPistons(config, pistons);
      const peak = peakFarFieldMagnitude2(config, states, range, step);
      expect(peak.sx).toBeCloseTo(sx, 2);
      expect(peak.sy).toBeCloseTo(sy, 2);
    }
  });

  it("3b. piston wraparound: adding 2pi does not change the far field", () => {
    const config = createArrayConfig({ pitch_m: 5 * 1550e-9, launchRadius_m: 2e-6 });
    const base = 1.234;
    const a = statesFromPistons(config, Array(config.channelIds.length).fill(base));
    const b = statesFromPistons(config, Array(config.channelIds.length).fill(base + 2 * Math.PI));
    const va = magnitude2(farFieldAngular(config, a, 0.01, -0.005));
    const vb = magnitude2(farFieldAngular(config, b, 0.01, -0.005));
    expect(vb).toBeCloseTo(va, 12);
  });

  it("4. Gaussian radius matches q theory; curvature moves the waist, piston does not", () => {
    const config = singleChannelConfig();
    const s0 = uniformActual(config);
    const zR = rayleighRange(config);

    for (const z of [0.05, 0.2, 0.4]) {
      const theory = config.launchRadius_m * Math.sqrt(1 + (z / zR) ** 2);
      expect(measuredRadiusX(config, s0, z)).toBeCloseTo(theory, 4);
    }

    const withPiston = statesFromPistons(config, [1.1]);
    expect(measuredRadiusX(config, withPiston, 0.2)).toBeCloseTo(
      measuredRadiusX(config, s0, 0.2),
      9,
    );

    const curvature = 2; // 1/m, waist downstream
    const curved = uniformActual(config, { curvature_per_m: curvature });
    let minZ = 0;
    let minW = Infinity;
    for (let z = -0.02; z <= 0.08; z += 0.0005) {
      const w = measuredRadiusX(config, curved, z);
      if (w < minW) {
        minW = w;
        minZ = z;
      }
    }
    const q = launchQ(config, curvature);
    const predictedWaistZ = -q.re;
    expect(minZ).toBeCloseTo(predictedWaistZ, 3);
  });

  it("5. two separated apertures produce predicted fringes; pi shift swaps maxima/minima", () => {
    const d = 50e-6;
    const config = twoEmitterXConfig(d, { launchRadius_m: 2e-6, apertureRadius_m: 1e-5 });
    const inPhase = statesFromPistons(config, [0, 0]);
    const anti = statesFromPistons(config, [0, Math.PI]);

    const theta0 = config.wavelength_m / (Math.PI * config.launchRadius_m);
    const envelope = (sx: number, sy: number) =>
      Math.exp(-((sx * sx + sy * sy) / (theta0 * theta0)));

    const arrayFactor = (states: ReturnType<typeof statesFromPistons>, sx: number) =>
      magnitude2(farFieldAngular(config, states, sx, 0)) / envelope(sx, 0) ** 2;

    expect(arrayFactor(inPhase, 0)).toBeCloseTo(4, 2);
    expect(arrayFactor(inPhase, config.wavelength_m / (2 * d))).toBeLessThan(1e-6);
    expect(arrayFactor(anti, 0)).toBeLessThan(1e-6);
    expect(arrayFactor(anti, config.wavelength_m / (2 * d))).toBeCloseTo(4, 2);
  });

  it("6. finite-target phase alignment maximizes target intensity via the reference", () => {
    const config = createArrayConfig({ launchRadius_m: 0.18e-3 });
    const grid = makeGrid(256, 20e-6);
    const target = { x: 1.2e-3, y: -0.7e-3, z: 0.2 };

    const idealPistons: number[] = [];
    const amplitudes: number[] = [];
    const fastPistons: number[] = [];
    for (let i = 0; i < config.channelIds.length; i++) {
      const cmds = uniformCommands(config);
      cmds.forEach((cmd, j) => (cmd.enabled = j === i));
      const launch = sampleLaunchField(config, commandsToActual(config, cmds), grid);
      const ref = fresnelFieldAtPoint(launch, grid, target.x, target.y, target.z, config.wavelength_m);
      const refPhase = Math.atan2(ref.im, ref.re);
      amplitudes.push(Math.hypot(ref.re, ref.im));
      // Ideal piston is relative to the channel's own carrier; the reference
      // already carries the propagation phase, so negate it directly.
      idealPistons.push(-refPhase);

      const fast = channelFieldFast(config, i, commandsToActual(config, cmds)[i], target.x, target.y, target.z);
      fastPistons.push(-Math.atan2(fast.im, fast.re));
    }

    const coherentSum = amplitudes.reduce((a, b) => a + b, 0);
    const aligned = statesFromPistons(config, idealPistons);
    const alignedField = fresnelFieldAtPoint(
      sampleLaunchField(config, aligned, grid),
      grid,
      target.x,
      target.y,
      target.z,
      config.wavelength_m,
    );
    const alignedI = magnitude2(alignedField);

    // The reference is linear, so ideal alignment reaches the coherent sum.
    expect(alignedI).toBeGreaterThan((coherentSum * 0.9) ** 2);

    const rand = seededRandom(11);
    const randomStates = statesFromPistons(
      config,
      config.channelIds.map(() => (rand() * 2 - 1) * Math.PI),
    );
    const randomField = fresnelFieldAtPoint(
      sampleLaunchField(config, randomStates, grid),
      grid,
      target.x,
      target.y,
      target.z,
      config.wavelength_m,
    );
    const randomI = magnitude2(randomField);
    expect(alignedI).toBeGreaterThan(randomI * 5);

    // Cross-model: phases derived from the fast evaluator must also beat random.
    const fastAlignedField = fresnelFieldAtPoint(
      sampleLaunchField(config, statesFromPistons(config, fastPistons), grid),
      grid,
      target.x,
      target.y,
      target.z,
      config.wavelength_m,
    );
    expect(magnitude2(fastAlignedField)).toBeGreaterThan(randomI * 3);
  });

  it("7. power is conserved by angular-spectrum propagation within the sampling limit", () => {
    const grid = makeGrid(256, 20e-6);
    const states = uniformActual(c0);
    const launch = sampleLaunchField(c0, states, grid);
    const p0 = launchPower(launch, grid);
    const z = 0.02;
    expect(z).toBeLessThan(angularSpectrumCriticalZ(grid, c0.wavelength_m));
    const propagated = propagateAngularSpectrum(launch, grid, z, c0.wavelength_m);
    const p1 = launchPower(propagated, grid);
    expect(Math.abs(p1 - p0) / p0).toBeLessThan(1e-9);

    // Truncation/grid loss for one isolated channel must be negligible.
    const single = states.map((s, i) => ({ ...s, enabled: i === 0 }));
    expect(launchPower(sampleLaunchField(c0, single, grid), grid)).toBeCloseTo(1, 5);

    // The coherent total exceeds the incoherent sum by the overlap cross terms;
    // this is constructive interference, not a power gain. Report it.
    const ideal = idealLaunchPower(states);
    const overlapExcess = (p0 - ideal) / ideal;
    expect(overlapExcess).toBeGreaterThan(0);
    expect(overlapExcess).toBeLessThan(0.2);
  });

  it("8. far-field metrics converge with sampling and expose an undersampled failure", () => {
    const states = uniformActual(c0);
    const extent = 8.192e-3;
    const gridCoarse = makeGrid(256, extent / 256);
    const gridFine = makeGrid(512, extent / 512);

    const relativePeak = (grid: ReturnType<typeof makeGrid>) => {
      const map = computeFarFieldMap(c0, states, grid);
      const total = totalAngularPower(map);
      let peak = 0;
      for (const s of map.samples) peak = Math.max(peak, s.power);
      return peak / total;
    };

    const coarse = relativePeak(gridCoarse);
    const fine = relativePeak(gridFine);
    expect(Math.abs(fine - coarse) / fine).toBeLessThan(0.03);

    const undersampled = relativePeak(makeGrid(64, 200e-6));
    expect(Math.abs(undersampled - fine) / fine).toBeGreaterThan(0.1);
  });

  it("9. fast evaluator agrees with the independent Fresnel reference", () => {
    const config = singleChannelConfig();
    const states = uniformActual(config);
    const grid = makeGrid(256, 15e-6);
    const launch = sampleLaunchField(config, states, grid);
    const zR = rayleighRange(config);
    const Z = 0.25;
    const w = config.launchRadius_m * Math.sqrt(1 + (Z / zR) ** 2);

    for (const x of [-w, -w / 2, 0, w / 2, w]) {
      const fast = channelFieldFast(config, 0, states[0], x, 0, Z);
      const ref = fresnelFieldAtPoint(launch, grid, x, 0, Z, config.wavelength_m);
      const fi = magnitude2(fast);
      const ri = magnitude2(ref);
      expect(Math.abs(fi - ri) / ri).toBeLessThan(0.05);
    }
  });
});
