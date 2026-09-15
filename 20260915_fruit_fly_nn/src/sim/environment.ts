import type { ArrayConfig } from "../optics/geometry";
import {
  randomHiddenErrors,
  resolveActual,
  type HiddenChannelErrors,
} from "../optics/channels";
import { farFieldAngular } from "../optics/gaussian";
import { normalizedFarField, directionToPoint, type Direction } from "../optics/metrics";
import type { ChannelActual, ChannelCommand } from "../optics/types";
import type { TargetMotion } from "./target";
import { staticTarget } from "./target";

export const OBSERVATION_SIZE = 8;

export interface RewardWeights {
  target: number;
  pib: number;
  pointing: number;
  energy: number;
}

export const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  target: 1.0,
  pib: 0.25,
  pointing: 0.1,
  energy: 0.001,
};

export interface EnvironmentConfig {
  array: ArrayConfig;
  dt_s: number;
  episodeSteps: number;
  target: TargetMotion;
  hiddenPistonScale_rad: number;
  hiddenGainScale: number;
  rewardWeights: RewardWeights;
  scanSamples: number;
  bucketHalfAngle_rad: number;
  seed: number;
}

export interface Observation {
  targetSx: number;
  targetSy: number;
  beamSx: number;
  beamSy: number;
  deltaSx: number;
  deltaSy: number;
  peak: number;
  pib: number;
}

export interface StepMetrics {
  targetIntensity: number;
  pib: number;
  peak: number;
  pointingAngle_rad: number;
  beamSx: number;
  beamSy: number;
  targetSx: number;
  targetSy: number;
}

export interface StepResult {
  reward: number;
  observation: Observation;
  metrics: StepMetrics;
  actual: ChannelActual[];
  targetPosition: { x: number; y: number; z: number };
}

export function observationToVector(o: Observation): Float64Array {
  return Float64Array.of(
    o.targetSx,
    o.targetSy,
    o.beamSx,
    o.beamSy,
    o.deltaSx,
    o.deltaSy,
    o.peak,
    o.pib,
  );
}

/**
 * Fixed-step optical environment. Commands (from manual controls, SPGD, or the
 * connectome readout) go through the same hidden-error path. The learner never
 * receives hidden phase offsets; it only sees the declared observations.
 */
export class Environment {
  readonly config: EnvironmentConfig;
  readonly array: ArrayConfig;
  private hidden: HiddenChannelErrors[];
  private previousCommands: ChannelCommand[];
  private actual: ChannelActual[] = [];
  time_s = 0;
  stepIndex = 0;

  constructor(config: EnvironmentConfig) {
    this.config = config;
    this.array = config.array;
    this.hidden = randomHiddenErrors(
      config.array.channelIds.length,
      config.seed,
      config.hiddenPistonScale_rad,
      config.hiddenGainScale,
    );
    this.previousCommands = [];
  }

  reset(seed = this.config.seed): void {
    this.time_s = 0;
    this.stepIndex = 0;
    this.hidden = randomHiddenErrors(
      this.array.channelIds.length,
      seed,
      this.config.hiddenPistonScale_rad,
      this.config.hiddenGainScale,
    );
    this.previousCommands = [];
  }

  targetPosition(): { x: number; y: number; z: number } {
    return this.config.target.positionAt(this.time_s);
  }

  targetDirection(): Direction {
    const p = this.targetPosition();
    return directionToPoint(p.x, p.y, p.z);
  }

  /** Ground-truth hidden errors; telemetry only, never learner input. */
  hiddenErrors(): readonly HiddenChannelErrors[] {
    return this.hidden;
  }

  /** Power-in-bucket: fraction of radiated power inside a cone about a direction. */
  pibBucket(dir: Direction, halfAngle_rad: number): number {
    const theta0 = this.array.wavelength_m / (Math.PI * this.array.launchRadius_m);
    const elementIntegral = (Math.PI * theta0 * theta0) / 2;
    let emitted = 0;
    for (const s of this.actual) {
      if (s.enabled) emitted += s.amplitude * s.amplitude * elementIntegral;
    }
    if (emitted <= 0) return 0;

    const ref =
      Math.abs(dir.sz) < 0.9 ? { x: 0, y: 0, z: 1 } : { x: 1, y: 0, z: 0 };
    const u = {
      x: ref.y * dir.sz - ref.z * dir.sy,
      y: ref.z * dir.sx - ref.x * dir.sz,
      z: ref.x * dir.sy - ref.y * dir.sx,
    };
    const uLen = Math.hypot(u.x, u.y, u.z) || 1;
    u.x /= uLen;
    u.y /= uLen;
    u.z /= uLen;
    const v = {
      x: dir.sy * u.z - dir.sz * u.y,
      y: dir.sz * u.x - dir.sx * u.z,
      z: dir.sx * u.y - dir.sy * u.x,
    };

    const nR = 8;
    const dAngle = halfAngle_rad / nR;
    let inside = 0;
    for (let i = -nR; i <= nR; i++) {
      for (let j = -nR; j <= nR; j++) {
        const a = i * dAngle;
        const b = j * dAngle;
        if (Math.hypot(a, b) > halfAngle_rad) continue;
        let sx = dir.sx + a * u.x + b * v.x;
        let sy = dir.sy + a * u.y + b * v.y;
        let sz = dir.sz + a * u.z + b * v.z;
        const norm = Math.hypot(sx, sy, sz) || 1;
        sx /= norm;
        sy /= norm;
        sz /= norm;
        const sample = farFieldAngular(this.array, this.actual, sx, sy);
        const intensity = sample.re * sample.re + sample.im * sample.im;
        inside += (intensity * dAngle * dAngle) / Math.max(1e-9, sz);
      }
    }
    return inside / emitted;
  }

  scanBeam(): {
    beamSx: number;
    beamSy: number;
    peak: number;
    peakSx: number;
    peakSy: number;
    pib: number;
    intensityAtTarget: number;
  } {
    const target = this.targetDirection();
    const half = this.array.steeringLimit * 1.4;
    const n = this.config.scanSamples;
    const step = n > 1 ? (2 * half) / (n - 1) : 0;

    let wsum = 0;
    let wx = 0;
    let wy = 0;

    for (let iy = 0; iy < n; iy++) {
      const sy = -half + iy * step;
      for (let ix = 0; ix < n; ix++) {
        const sx = -half + ix * step;
        const u = farFieldAngular(this.array, this.actual, sx, sy);
        const intensity = u.re * u.re + u.im * u.im;
        wsum += intensity;
        wx += sx * intensity;
        wy += sy * intensity;
      }
    }
    // Intensity exactly at the target direction (analytic, not on the scan grid).
    const targetSample = farFieldAngular(this.array, this.actual, target.sx, target.sy);
    const atTarget = targetSample.re * targetSample.re + targetSample.im * targetSample.im;

    // Second, fine stage. The coarse grid step is ~7 mrad, far larger than the
    // diffraction spot, so a single-pass centroid is biased by up to half a
    // step and visibly offsets the measurement plane from the beam. Refine
    // around the coarse centroid to sub-mrad resolution.
    const coarseSx = wsum > 0 ? wx / wsum : 0;
    const coarseSy = wsum > 0 ? wy / wsum : 0;
    const fineHalf = step > 0 ? step : this.array.steeringLimit * 0.05;
    const nf = 17;
    const fstep = nf > 1 ? (2 * fineHalf) / (nf - 1) : 0;
    let fw = 0;
    let fwx = 0;
    let fwy = 0;
    let fPeak = 0;
    let fPeakSx = coarseSx;
    let fPeakSy = coarseSy;
    for (let iy = 0; iy < nf; iy++) {
      const sy = coarseSy - fineHalf + iy * fstep;
      for (let ix = 0; ix < nf; ix++) {
        const sx = coarseSx - fineHalf + ix * fstep;
        const u = farFieldAngular(this.array, this.actual, sx, sy);
        const intensity = u.re * u.re + u.im * u.im;
        fw += intensity;
        fwx += sx * intensity;
        fwy += sy * intensity;
        if (intensity > fPeak) {
          fPeak = intensity;
          fPeakSx = sx;
          fPeakSy = sy;
        }
      }
    }

    return {
      beamSx: fw > 0 ? fwx / fw : coarseSx,
      beamSy: fw > 0 ? fwy / fw : coarseSy,
      peak: fPeak,
      peakSx: fPeakSx,
      peakSy: fPeakSy,
      pib: this.pibBucket(target, this.config.bucketHalfAngle_rad),
      intensityAtTarget: atTarget,
    };
  }

  setCommands(commands: readonly ChannelCommand[]): ChannelActual[] {
    this.actual = resolveActual(this.array, commands, this.hidden);
    return this.actual;
  }

  /** Current actual actuator state (for telemetry/visualization). */
  currentActual(): readonly ChannelActual[] {
    return this.actual;
  }

  /** Observation for a command set without advancing time (episode bootstrap). */
  observeWithCommands(commands: readonly ChannelCommand[]): Observation {
    this.setCommands(commands);
    const scan = this.scanBeam();
    const target = this.targetDirection();
    return {
      targetSx: target.sx,
      targetSy: target.sy,
      beamSx: scan.beamSx,
      beamSy: scan.beamSy,
      deltaSx: target.sx - scan.beamSx,
      deltaSy: target.sy - scan.beamSy,
      peak: scan.peak,
      pib: scan.pib,
    };
  }

  step(commands: readonly ChannelCommand[]): StepResult {
    const actuallyResolved = this.setCommands(commands);
    const target = this.targetDirection();
    const scan = this.scanBeam();
    const targetSample = farFieldAngular(this.array, actuallyResolved, target.sx, target.sy);
    const total = actuallyResolved.reduce((a, s) => a + (s.enabled ? s.amplitude : 0), 0);
    const normalizedTarget = total > 0 ? (targetSample.re ** 2 + targetSample.im ** 2) / total ** 2 : 0;

    const directionToBeam = directionToPoint(scan.beamSx, scan.beamSy, Math.sqrt(Math.max(0, 1 - scan.beamSx ** 2 - scan.beamSy ** 2)));
    const pointing = Math.acos(
      Math.min(1, Math.max(-1, directionToBeam.sx * target.sx + directionToBeam.sy * target.sy + directionToBeam.sz * target.sz)),
    );

    let energy = 0;
    if (this.previousCommands.length === commands.length) {
      for (let i = 0; i < commands.length; i++) {
        const d = commands[i].piston_rad - this.previousCommands[i].piston_rad;
        energy += d * d;
      }
      energy /= commands.length;
    }
    this.previousCommands = commands.map((c) => ({ ...c }));

    const w = this.config.rewardWeights;
    const reward =
      w.target * normalizedTarget +
      w.pib * scan.pib -
      w.pointing * pointing -
      w.energy * energy;

    const targetPos = this.targetPosition();
    const observation: Observation = {
      targetSx: target.sx,
      targetSy: target.sy,
      beamSx: scan.beamSx,
      beamSy: scan.beamSy,
      deltaSx: target.sx - scan.beamSx,
      deltaSy: target.sy - scan.beamSy,
      peak: scan.peak,
      pib: scan.pib,
    };
    const metrics: StepMetrics = {
      targetIntensity: normalizedTarget,
      pib: scan.pib,
      peak: scan.peak,
      pointingAngle_rad: pointing,
      beamSx: scan.beamSx,
      beamSy: scan.beamSy,
      targetSx: target.sx,
      targetSy: target.sy,
    };

    this.time_s += this.config.dt_s;
    this.stepIndex += 1;
    return { reward, observation, metrics, actual: actuallyResolved, targetPosition: targetPos };
  }
}

export function defaultEnvironmentConfig(array: ArrayConfig, overrides: Partial<EnvironmentConfig> = {}): EnvironmentConfig {
  return {
    array,
    dt_s: 1 / 60,
    episodeSteps: 180,
    target: staticTarget(0, 0, 1),
    hiddenPistonScale_rad: 0,
    hiddenGainScale: 0,
    rewardWeights: DEFAULT_REWARD_WEIGHTS,
    scanSamples: 15,
    bucketHalfAngle_rad: 1e-3,
    seed: 1,
    ...overrides,
  };
}

/** Spot check used by tests: normalized intensity proxy for a direction. */
export function normalizedTargetIntensity(
  array: ArrayConfig,
  actual: readonly ChannelActual[],
  dir: Direction,
): number {
  return normalizedFarField(array, actual, dir);
}
