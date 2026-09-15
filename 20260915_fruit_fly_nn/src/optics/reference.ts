import type { ArrayConfig } from "./geometry";
import { wavenumber } from "./geometry";
import type { ChannelActual } from "./types";
import { ComplexField, fft2d, fftFrequency } from "./complex";
import { gridCoord, type Grid2D } from "./grid";
import { sampleLaunchField } from "./launch";

/**
 * Independent numerical optical reference (docs/CODING_TASKS.md T02).
 *
 * Two methods, neither sharing code with the fast analytic evaluator:
 *  - sampled angular-spectrum propagation (plane-to-plane),
 *  - direct paraxial Fresnel diffraction integral evaluated at arbitrary points.
 *
 * The launch field itself is sampled from the declared aperture formula, so
 * truncation is included. Approximations/limits are documented here and in the
 * task report: scalar/paraxial, transverse grid with spacing dx, paraxial angle
 * limit lambda/(2 dx), and angular-spectrum critical sampling
 * z <= n dx^2 / lambda.
 */

/** Critical propagation distance for alias-free angular-spectrum transfer. */
export function angularSpectrumCriticalZ(grid: Grid2D, wavelength_m: number): number {
  return (grid.n * grid.dx * grid.dx) / wavelength_m;
}

/** Propagate a sampled field over z using the angular-spectrum transfer function. */
export function propagateAngularSpectrum(
  field: ComplexField,
  grid: Grid2D,
  z: number,
  wavelength_m: number,
): ComplexField {
  const out = field.clone();
  const { n, dx } = grid;
  fft2d(out.re, out.im, false);
  const k = wavenumber(wavelength_m);
  for (let iy = 0; iy < n; iy++) {
    const fy = fftFrequency(iy, n, dx);
    for (let ix = 0; ix < n; ix++) {
      const fx = fftFrequency(ix, n, dx);
      const s2 = (wavelength_m * fx) ** 2 + (wavelength_m * fy) ** 2;
      const idx = iy * n + ix;
      if (s2 >= 1) {
        out.re[idx] = 0;
        out.im[idx] = 0;
        continue;
      }
      const kz = k * Math.sqrt(1 - s2);
      const ph = kz * z;
      const c = Math.cos(ph);
      const s = Math.sin(ph);
      const re = out.re[idx];
      const im = out.im[idx];
      out.re[idx] = re * c - im * s;
      out.im[idx] = re * s + im * c;
    }
  }
  fft2d(out.re, out.im, true);
  return out;
}

/**
 * Direct paraxial Fresnel diffraction integral at a single point (x, y) in the
 * plane z = Z. O(n^2) per point; used for tests and cross-checks, not realtime.
 */
export function fresnelFieldAtPoint(
  launch: ComplexField,
  grid: Grid2D,
  x: number,
  y: number,
  Z: number,
  wavelength_m: number,
): { re: number; im: number } {
  if (Z <= 0) throw new Error("fresnelFieldAtPoint: Z must be positive");
  const { n, dx } = grid;
  const k = wavenumber(wavelength_m);
  const outerPhase = (k * (x * x + y * y)) / (2 * Z);
  const cosO = Math.cos(k * Z + outerPhase);
  const sinO = Math.sin(k * Z + outerPhase);
  let sr = 0;
  let si = 0;
  for (let iy = 0; iy < n; iy++) {
    const yj = gridCoord(iy, n, dx);
    for (let ix = 0; ix < n; ix++) {
      const xj = gridCoord(ix, n, dx);
      const u = launch.re[iy * n + ix];
      const v = launch.im[iy * n + ix];
      if (u === 0 && v === 0) continue;
      const ph = (k * (xj * xj + yj * yj)) / (2 * Z) - (k * (x * xj + y * yj)) / Z;
      const c = Math.cos(ph);
      const s = Math.sin(ph);
      sr += u * c - v * s;
      si += u * s + v * c;
    }
  }
  const scale = dx * dx / (wavelength_m * Z);
  // 1/i = -i, so rotate by -90 degrees.
  const re = (sr * cosO + si * sinO) * scale;
  const im = (si * cosO - sr * sinO) * scale;
  return { re, im };
}

export interface AngleSample {
  sx: number;
  sy: number;
  sz: number;
  /** Solid-angle weight dOmega of the sample (0 if evanescent). */
  weight: number;
  /** Power in the sample (linear, consistent with launch-plane power). */
  power: number;
}

export interface FarFieldMap {
  readonly n: number;
  readonly samples: readonly AngleSample[];
}

/**
 * Far-field angular power map from the sampled launch field via a single FFT.
 *
 *   A(fx, fy) = FFT[U(x, y, 0)]
 *   sx = lambda fx, sy = lambda fy, sz = sqrt(1 - sx^2 - sy^2)
 *   dOmega = lambda^2 / (n^2 dx^2 sz),  dP = |A|^2 dx^2 / n^2
 *
 * Total dP sums to the launch-plane power (Parseval). Only propagating
 * directions (sx^2 + sy^2 < 1) are retained; others are masked.
 */
export function computeFarFieldMap(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  grid: Grid2D,
): FarFieldMap {
  const launch = sampleLaunchField(config, states, grid);
  const spec = launch.clone();
  const { n, dx } = grid;
  fft2d(spec.re, spec.im, false);
  const samples: AngleSample[] = new Array(n * n);
  const df = 1 / (n * dx);
  const norm = (dx * dx) / (n * n);
  for (let iy = 0; iy < n; iy++) {
    const fy = fftFrequency(iy, n, dx);
    for (let ix = 0; ix < n; ix++) {
      const fx = fftFrequency(ix, n, dx);
      const sx = config.wavelength_m * fx;
      const sy = config.wavelength_m * fy;
      const s2 = sx * sx + sy * sy;
      const idx = iy * n + ix;
      if (s2 >= 1) {
        samples[idx] = { sx, sy, sz: 0, weight: 0, power: 0 };
        continue;
      }
      const sz = Math.sqrt(1 - s2);
      const a2 = spec.re[idx] * spec.re[idx] + spec.im[idx] * spec.im[idx];
      samples[idx] = {
        sx,
        sy,
        sz,
        weight: (config.wavelength_m * config.wavelength_m * df * df) / sz,
        power: a2 * norm,
      };
    }
  }
  return { n, samples };
}

export function totalAngularPower(map: FarFieldMap): number {
  let p = 0;
  for (const s of map.samples) p += s.power;
  return p;
}

/** Power-weighted mean direction; returns null when power is negligible. */
export function angularCentroid(
  map: FarFieldMap,
  minPower = 1e-12,
): { x: number; y: number; z: number; concentration: number } | null {
  let px = 0;
  let py = 0;
  let pz = 0;
  let p = 0;
  for (const s of map.samples) {
    if (s.power <= 0) continue;
    px += s.sx * s.power;
    py += s.sy * s.power;
    pz += s.sz * s.power;
    p += s.power;
  }
  if (p < minPower) return null;
  const mx = px / p;
  const my = py / p;
  const mz = pz / p;
  const mag = Math.hypot(mx, my, mz);
  if (mag < minPower) return null;
  return { x: mx / mag, y: my / mag, z: mz / mag, concentration: mag };
}

/** Fraction of power inside an angular cone about a target unit direction. */
export function angularPib(
  map: FarFieldMap,
  target: { x: number; y: number; z: number },
  halfAngle_rad: number,
): number {
  const cosLimit = Math.cos(halfAngle_rad);
  let inside = 0;
  let total = 0;
  for (const s of map.samples) {
    if (s.power <= 0) continue;
    total += s.power;
    const cos = s.sx * target.x + s.sy * target.y + s.sz * target.z;
    if (cos >= cosLimit) inside += s.power;
  }
  return total > 0 ? inside / total : 0;
}
