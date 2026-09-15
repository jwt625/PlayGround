import type { ArrayConfig } from "../optics/geometry";
import {
  resolveActual,
  uniformCommands,
  type HiddenChannelErrors,
} from "../optics/channels";
import { channelFieldFast } from "../optics/gaussian";
import { normalizedFarField, type Direction } from "../optics/metrics";
import type { ChannelActual, ChannelCommand } from "../optics/types";

/**
 * Build a scalar objective over piston commands for a fixed direction and a
 * fixed hidden-error realization. Each evaluation goes through the same
 * command -> actual path as the UI, then measures the optical far field.
 */
export function makeDirectionObjective(
  config: ArrayConfig,
  hidden: readonly HiddenChannelErrors[],
  dir: Direction,
): (pistons: readonly number[]) => number {
  const buffer: ChannelCommand[] = uniformCommands(config);
  for (const cmd of buffer) {
    cmd.tiltX = dir.sx;
    cmd.tiltY = dir.sy;
  }
  return (pistons: readonly number[]) => {
    if (pistons.length !== buffer.length) throw new Error("objective: piston length mismatch");
    for (let i = 0; i < pistons.length; i++) buffer[i].piston_rad = pistons[i];
    const actual = resolveActual(config, buffer, hidden);
    return normalizedFarField(config, actual, dir);
  };
}

/**
 * Ideal piston phases that make the current channel fields add in phase at a
 * finite target point. Uses the fast analytic evaluator, so it includes
 * propagation, curvature, and Gouy phase relative to each launch aperture.
 */
export function alignPistonsToTarget(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  target: { x: number; y: number; z: number },
): number[] {
  return states.map((state, i) => {
    const zeroPhase = { ...state, piston_rad: 0 };
    const u = channelFieldFast(config, i, zeroPhase, target.x, target.y, target.z);
    return -Math.atan2(u.im, u.re);
  });
}
