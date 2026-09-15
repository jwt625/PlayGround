import type { ArrayConfig } from "./geometry";
import { defaultChannelCommand, type ChannelActual, type ChannelCommand } from "./types";

export interface HiddenChannelErrors {
  staticPiston_rad: number;
  gain: number;
}

export function zeroHiddenErrors(count: number): HiddenChannelErrors[] {
  return Array.from({ length: count }, () => ({ staticPiston_rad: 0, gain: 1 }));
}

/**
 * Resolve commands to actual actuator state. Hidden piston offsets and gain
 * errors act between command and actual state. Manual controls and learned
 * controllers use the same path.
 */
export function resolveActual(
  config: ArrayConfig,
  commands: readonly ChannelCommand[],
  hidden: readonly HiddenChannelErrors[],
): ChannelActual[] {
  if (commands.length !== config.channelIds.length) {
    throw new Error(`resolveActual: expected ${config.channelIds.length} commands`);
  }
  if (hidden.length !== commands.length) throw new Error("resolveActual: hidden length mismatch");
  return commands.map((cmd, i) => {
    const h = hidden[i];
    const gain = Math.max(0, cmd.amplitude * h.gain);
    return {
      piston_rad: wrapPiston(cmd.piston_rad + h.staticPiston_rad),
      amplitude: clamp(gain, config.amplitudeMin, config.amplitudeMax),
      tiltX: cmd.tiltX,
      tiltY: cmd.tiltY,
      curvature_per_m: clamp(
        cmd.curvature_per_m,
        -config.curvatureLimit_per_m,
        config.curvatureLimit_per_m,
      ),
      enabled: cmd.enabled,
      hiddenPiston_rad: h.staticPiston_rad,
      hiddenGain: h.gain,
    };
  });
}

export function commandsToActual(
  config: ArrayConfig,
  commands: readonly ChannelCommand[],
): ChannelActual[] {
  return resolveActual(config, commands, zeroHiddenErrors(commands.length));
}

export function uniformCommands(config: ArrayConfig, patch: Partial<ChannelCommand> = {}): ChannelCommand[] {
  return config.channelIds.map(() => ({ ...defaultChannelCommand(), ...patch }));
}

export function uniformStates(config: ArrayConfig, patch: Partial<ChannelCommand> = {}): ChannelActual[] {
  return commandsToActual(config, uniformCommands(config, patch));
}

/** Wrap a piston command into (-pi, pi]. */
export function wrapPiston(value: number): number {
  const twoPi = 2 * Math.PI;
  let v = value % twoPi;
  if (v <= -Math.PI) v += twoPi;
  if (v > Math.PI) v -= twoPi;
  return v;
}

export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}

/** Difference of two angles wrapped into (-pi, pi]. */
export function wrapAngleDiff(a: number, b: number): number {
  return wrapPiston(a - b);
}

export function seededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomHiddenErrors(
  count: number,
  seed: number,
  pistonScale_rad = 1,
  gainScale = 0.15,
): HiddenChannelErrors[] {
  const rand = seededRandom(seed);
  return Array.from({ length: count }, () => ({
    staticPiston_rad: (rand() * 2 - 1) * pistonScale_rad,
    gain: 1 + (rand() * 2 - 1) * gainScale,
  }));
}
