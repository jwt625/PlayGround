import type { ArrayConfig } from "./geometry";
import { wavenumber } from "./geometry";
import type { ChannelActual } from "./types";
import { farFieldAngular } from "./gaussian";
import { wrapAngleDiff } from "./channels";

export interface Direction {
  sx: number;
  sy: number;
  sz: number;
}

export function directionFromTransverse(sx: number, sy: number): Direction {
  const s2 = sx * sx + sy * sy;
  if (s2 >= 1) throw new Error(`direction outside paraxial domain: |s|^2=${s2}`);
  return { sx, sy, sz: Math.sqrt(1 - s2) };
}

export function directionToPoint(x: number, y: number, z: number): Direction {
  const n = Math.hypot(x, y, z);
  if (n === 0) throw new Error("directionToPoint: zero vector");
  return { sx: x / n, sy: y / n, sz: z / n };
}

function sumAmplitudes(states: readonly ChannelActual[]): number {
  return states.reduce((a, s) => a + (s.enabled ? s.amplitude : 0), 0);
}

/**
 * Normalized far-field intensity in a direction: element envelope included and
 * divided by the ideal in-phase value (sum of amplitudes)^2. Equals 1 for a
 * perfectly phase-locked array with no element-envelope loss.
 */
export function normalizedFarField(config: ArrayConfig, states: readonly ChannelActual[], dir: Direction): number {
  const total = sumAmplitudes(states);
  if (total <= 0) return 0;
  const u = farFieldAngular(config, states, dir.sx, dir.sy);
  const i = u.re * u.re + u.im * u.im;
  return i / (total * total);
}

/** Ideal piston phase that steers the collimated array toward a direction. */
export function idealSteeringPistons(config: ArrayConfig, dir: Direction): number[] {
  const k = wavenumber(config.wavelength_m);
  return config.channelIds.map((_, i) => k * (dir.sx * config.x_m[i] + dir.sy * config.y_m[i]));
}

/** Strehl-like ratio versus the ideal steering ramp at the same direction. */
export function strehlAtDirection(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  dir: Direction,
): number {
  const total = sumAmplitudes(states);
  if (total <= 0) return 0;
  const here = farFieldAngular(config, states, dir.sx, dir.sy);
  const refPistons = idealSteeringPistons(config, dir);
  const idealStates = states.map((s, i) => ({
    ...s,
    piston_rad: refPistons[i],
    amplitude: s.enabled ? s.amplitude : 0,
  }));
  const ref = farFieldAngular(config, idealStates, dir.sx, dir.sy);
  const iHere = here.re * here.re + here.im * here.im;
  const iRef = ref.re * ref.re + ref.im * ref.im;
  return iRef > 0 ? iHere / iRef : 0;
}

/** RMS wrapped piston error against a target phase set, rad. */
export function phaseRmsRad(states: readonly ChannelActual[], idealPistons: readonly number[]): number {
  if (states.length !== idealPistons.length) throw new Error("phaseRmsRad: length mismatch");
  let sum = 0;
  states.forEach((s, i) => {
    const d = wrapAngleDiff(s.piston_rad, idealPistons[i]);
    sum += d * d;
  });
  return Math.sqrt(sum / states.length);
}
