import type { ArrayConfig } from "../optics/geometry";
import { seededRandom, uniformCommands } from "../optics/channels";
import type { ChannelCommand } from "../optics/types";

/** Uniformly random piston commands in [-range, range]. Expected to fail. */
export function randomCommands(
  config: ArrayConfig,
  seed: number,
  range = Math.PI,
): ChannelCommand[] {
  const rand = seededRandom(seed);
  return uniformCommands(config).map((cmd) => ({
    ...cmd,
    piston_rad: (rand() * 2 - 1) * range,
  }));
}

export function randomPistons(config: ArrayConfig, seed: number, range = Math.PI): number[] {
  const rand = seededRandom(seed);
  return config.channelIds.map(() => (rand() * 2 - 1) * range);
}
