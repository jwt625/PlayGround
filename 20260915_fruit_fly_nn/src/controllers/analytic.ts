import type { ArrayConfig } from "../optics/geometry";
import { uniformCommands } from "../optics/channels";
import { idealSteeringPistons } from "../optics/metrics";
import type { Direction } from "../optics/metrics";
import type { ChannelCommand } from "../optics/types";

/**
 * Analytic phase/focus reference. V1 steering uses common tip/tilt (all
 * channels pointed at the target) plus the piston ramp that aligns their
 * phases at that direction. Tip/tilt and focus are exposed on the command path
 * for later curricula; the fly's connectome initially learns piston
 * corrections on top of this reference.
 */
export function analyticSteeringCommands(config: ArrayConfig, dir: Direction): ChannelCommand[] {
  const pistons = idealSteeringPistons(config, dir);
  return uniformCommands(config).map((cmd, i) => ({
    ...cmd,
    piston_rad: pistons[i],
    tiltX: dir.sx,
    tiltY: dir.sy,
  }));
}

export function analyticSteeringPistons(config: ArrayConfig, dir: Direction): number[] {
  return idealSteeringPistons(config, dir);
}
