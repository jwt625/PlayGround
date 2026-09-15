import {
  createArrayConfigFromSites,
  type ArrayConfig,
  type ArrayConfigOverrides,
} from "../src/optics/geometry";
import { commandsToActual, uniformCommands } from "../src/optics/channels";
import type { ChannelActual, ChannelCommand } from "../src/optics/types";

/** Config with coincident emitters at the origin (for algebra tests). */
export function coincidentConfig(count: number, overrides: ArrayConfigOverrides = {}): ArrayConfig {
  return createArrayConfigFromSites(
    Array.from({ length: count }, () => ({ q: 0, r: 0 })),
    overrides,
  );
}

/** Config with two emitters separated by `d` along X (pitch = d). */
export function twoEmitterXConfig(d: number, overrides: ArrayConfigOverrides = {}): ArrayConfig {
  return createArrayConfigFromSites([{ q: 0, r: 0 }, { q: 1, r: 0 }], {
    pitch_m: d,
    ...overrides,
  });
}

export function singleChannelConfig(overrides: ArrayConfigOverrides = {}): ArrayConfig {
  return createArrayConfigFromSites([{ q: 0, r: 0 }], overrides);
}

export function rampCommands(config: ArrayConfig, sx: number, sy: number): ChannelCommand[] {
  const k = (2 * Math.PI) / config.wavelength_m;
  const cmds = uniformCommands(config);
  return cmds.map((cmd, i) => ({
    ...cmd,
    piston_rad: k * (sx * config.x_m[i] + sy * config.y_m[i]),
  }));
}

export function statesFromPistons(config: ArrayConfig, pistons: readonly number[]): ChannelActual[] {
  const cmds = uniformCommands(config);
  pistons.forEach((p, i) => (cmds[i].piston_rad = p));
  return commandsToActual(config, cmds);
}

export function uniformActual(
  config: ArrayConfig,
  patch: Partial<ChannelCommand> = {},
): ChannelActual[] {
  return commandsToActual(config, uniformCommands(config, patch));
}
