import type { ArrayConfig } from "./geometry";
import { wavenumber } from "./geometry";
import type { ChannelActual } from "./types";
import { launchAmplitudeScale } from "./launch";
import { ComplexField } from "./complex";
import { gridCoord, type Grid2D } from "./grid";

/**
 * Fast analytic Gaussian-beam evaluator (docs/CODING_TASKS.md T02).
 *
 * Each channel is modelled as a paraxial Gaussian beam with launch waist radius
 * w0, an optional launch curvature C (lens power C = 1/f), a propagation
 * direction (tiltX, tiltY) = (sx, sy), and piston. The launch complex beam
 * parameter is
 *
 *   q0 = i z_R,   z_R = pi w0^2 / lambda
 *   1/q_launch = 1/q0 - C
 *
 * so a positive curvature places the waist downstream. At a field point the
 * distance along the tilted axis is t = r . s and the transverse offset is
 * r_perp = r - t s. The field is
 *
 *   U = A0 * a * (w0/w(t)) * exp(-|r_perp|^2 / w^2)
 *       * exp{i[k t - (psi(t) - psi(0)) + k|r_perp|^2/(2R(t)) + piston]}
 *
 * which reduces to the declared launch aperture at t = 0 (Gouy phase referenced
 * to the launch plane). It is valid in the paraxial, small-angle domain and is
 * cross-checked against the independent Fresnel/angular-spectrum reference.
 */

export interface ComplexQ {
  re: number;
  im: number;
}

export function rayleighRange(config: ArrayConfig): number {
  return (Math.PI * config.launchRadius_m * config.launchRadius_m) / config.wavelength_m;
}

/** Launch complex beam parameter for a given signed curvature, 1/m. */
export function launchQ(config: ArrayConfig, curvature_per_m: number): ComplexQ {
  const zR = rayleighRange(config);
  const a = -curvature_per_m;
  const b = -1 / zR;
  const den = a * a + b * b;
  return { re: a / den, im: -b / den };
}

function normalizedDirection(sx: number, sy: number): { sx: number; sy: number; sz: number } {
  const s2 = sx * sx + sy * sy;
  if (s2 > 1 + 1e-9) throw new Error(`direction outside paraxial domain: |s|^2 = ${s2}`);
  const clamped = Math.min(s2, 1);
  return { sx, sy, sz: Math.sqrt(Math.max(0, 1 - clamped)) };
}

export interface FieldSample {
  re: number;
  im: number;
}

/** Fast complex field of one channel at an arbitrary world point. */
export function channelFieldFast(
  config: ArrayConfig,
  channelIndex: number,
  state: ChannelActual,
  x: number,
  y: number,
  z: number,
): FieldSample {
  if (!state.enabled || state.amplitude === 0) return { re: 0, im: 0 };
  const { sx, sy, sz } = normalizedDirection(state.tiltX, state.tiltY);
  const k = wavenumber(config.wavelength_m);
  const w0 = config.launchRadius_m;

  const xc = config.x_m[channelIndex];
  const yc = config.y_m[channelIndex];
  const zc = config.z_m[channelIndex];
  const rx = x - xc;
  const ry = y - yc;
  const rz = z - zc;

  const t = rx * sx + ry * sy + rz * sz;
  const px = rx - t * sx;
  const py = ry - t * sy;
  const pz = rz - t * sz;
  const r2 = px * px + py * py + pz * pz;

  const q0 = launchQ(config, state.curvature_per_m);
  const qRe = q0.re + t;
  const qIm = q0.im;
  const den = qRe * qRe + qIm * qIm;
  const invqRe = qRe / den;
  const invqIm = -qIm / den;

  const w2 = config.wavelength_m / (Math.PI * -invqIm);
  const w = Math.sqrt(w2);
  const R = invqRe === 0 ? Infinity : 1 / invqRe;

  const psi = Math.atan2(qRe, qIm);
  const psi0 = Math.atan2(q0.re, q0.im);
  const curvaturePhase = Number.isFinite(R) ? (k * r2) / (2 * R) : 0;
  const phase = k * t - (psi - psi0) + curvaturePhase + state.piston_rad;

  const amp =
    launchAmplitudeScale(config) * state.amplitude * (w0 / w) * Math.exp(-r2 / w2);
  return { re: amp * Math.cos(phase), im: amp * Math.sin(phase) };
}

export function totalFieldFast(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  x: number,
  y: number,
  z: number,
): FieldSample {
  let re = 0;
  let im = 0;
  for (let c = 0; c < states.length; c++) {
    const u = channelFieldFast(config, c, states[c], x, y, z);
    re += u.re;
    im += u.im;
  }
  return { re, im };
}

export function intensityFast(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  x: number,
  y: number,
  z: number,
): number {
  const u = totalFieldFast(config, states, x, y, z);
  return u.re * u.re + u.im * u.im;
}

/** Sample the fast summed field on a transverse plane at world z. */
export function sampleFieldFastOnPlane(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  z: number,
  grid: Grid2D,
): ComplexField {
  if (states.length !== config.channelIds.length) {
    throw new Error(
      `sampleFieldFastOnPlane: expected ${config.channelIds.length} states, got ${states.length}`,
    );
  }
  const { n, dx } = grid;
  const out = new ComplexField(n * n);
  for (let iy = 0; iy < n; iy++) {
    const y = gridCoord(iy, n, dx);
    for (let ix = 0; ix < n; ix++) {
      const x = gridCoord(ix, n, dx);
      const u = totalFieldFast(config, states, x, y, z);
      const idx = iy * n + ix;
      out.re[idx] = u.re;
      out.im[idx] = u.im;
    }
  }
  return out;
}

/**
 * Far-field angular array factor of the launch field in the direction (sx, sy),
 * including each emitter's Gaussian far-field element envelope centered on its
 * own tip/tilt direction:
 *
 *   E(s) ∝ sum_n a_n g_n(s) exp{i[piston_n - k(sx x_n + sy y_n)]}
 *
 * `g_n` has 1/e^2 half-angle lambda / (pi w0) about the channel axis. This is
 * used for the real-time dome and objective; the FFT angular spectrum in
 * `reference.ts` is the independent check.
 */
export function farFieldAngular(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  sx: number,
  sy: number,
): FieldSample {
  const { sx: ux, sy: uy } = normalizedDirection(sx, sy);
  const k = wavenumber(config.wavelength_m);
  const theta0 = config.wavelength_m / (Math.PI * config.launchRadius_m);
  const invTheta2 = 1 / (theta0 * theta0);

  let re = 0;
  let im = 0;
  for (let c = 0; c < states.length; c++) {
    const s = states[c];
    if (!s.enabled || s.amplitude === 0) continue;
    const t2 = s.tiltX * s.tiltX + s.tiltY * s.tiltY;
    if (t2 >= 1) continue;
    const tz = Math.sqrt(1 - t2);
    const sz = Math.sqrt(Math.max(0, 1 - ux * ux - uy * uy));
    // Component of s perpendicular to the channel axis d.
    const dot = ux * s.tiltX + uy * s.tiltY + sz * tz;
    const px = ux - dot * s.tiltX;
    const py = uy - dot * s.tiltY;
    const pz = sz - dot * tz;
    const sin2 = px * px + py * py + pz * pz;
    const element = Math.exp(-sin2 * invTheta2);
    const phase = s.piston_rad - k * (ux * config.x_m[c] + uy * config.y_m[c]);
    re += s.amplitude * element * Math.cos(phase);
    im += s.amplitude * element * Math.sin(phase);
  }
  return { re, im };
}
