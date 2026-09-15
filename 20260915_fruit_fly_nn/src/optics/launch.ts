import type { ArrayConfig } from "./geometry";
import { wavenumber } from "./geometry";
import type { ChannelActual } from "./types";
import { ComplexField } from "./complex";
import { gridCoord, type Grid2D } from "./grid";

/**
 * Launch-aperture field model (docs/CODING_TASKS.md T02; DevLog/001 "Optical contract").
 *
 * Per channel, relative to that emitter's centre (dx, dy):
 *
 *   U_n = A_n * exp(-(dx^2+dy^2)/w0^2)
 *         * exp{i[piston + k(sx*dx + sy*dy) - k*C*(dx^2+dy^2)/2]}
 *
 * with a declared circular aperture truncation radius. Piston is referenced to
 * the launch aperture. The amplitude A_n is normalized so an enabled channel
 * with unit `amplitude` carries unit power in the untruncated Gaussian limit:
 *
 *   integral |A_n|^2 exp(-2r^2/w0^2) dA = A_n^2 * pi w0^2 / 2 = 1
 *
 * so A_n = sqrt(2 / (pi w0^2)).
 */
export function launchAmplitudeScale(config: ArrayConfig): number {
  return Math.sqrt(2 / (Math.PI * config.launchRadius_m * config.launchRadius_m));
}

/** Complex launch field of a single channel at a point in the z = 0 plane. */
export function channelLaunchFieldAt(
  config: ArrayConfig,
  channelIndex: number,
  state: ChannelActual,
  x: number,
  y: number,
): { re: number; im: number } {
  if (!state.enabled) return { re: 0, im: 0 };
  const dx = x - config.x_m[channelIndex];
  const dy = y - config.y_m[channelIndex];
  const r2 = dx * dx + dy * dy;
  if (r2 > config.apertureRadius_m * config.apertureRadius_m) return { re: 0, im: 0 };

  const k = wavenumber(config.wavelength_m);
  const w0 = config.launchRadius_m;
  const envelope = Math.exp(-r2 / (w0 * w0));
  const phase =
    state.piston_rad +
    k * (state.tiltX * dx + state.tiltY * dy) -
    (k * state.curvature_per_m * r2) / 2;
  const amp = launchAmplitudeScale(config) * state.amplitude * envelope;
  return { re: amp * Math.cos(phase), im: amp * Math.sin(phase) };
}

/** Sample the summed launch field on a transverse grid at z = 0. */
export function sampleLaunchField(
  config: ArrayConfig,
  states: readonly ChannelActual[],
  grid: Grid2D,
): ComplexField {
  if (states.length !== config.channelIds.length) {
    throw new Error(
      `sampleLaunchField: expected ${config.channelIds.length} states, got ${states.length}`,
    );
  }
  const { n, dx } = grid;
  const field = new ComplexField(n * n);
  for (let iy = 0; iy < n; iy++) {
    const y = gridCoord(iy, n, dx);
    for (let ix = 0; ix < n; ix++) {
      const x = gridCoord(ix, n, dx);
      let re = 0;
      let im = 0;
      for (let c = 0; c < states.length; c++) {
        const u = channelLaunchFieldAt(config, c, states[c], x, y);
        re += u.re;
        im += u.im;
      }
      const idx = iy * n + ix;
      field.re[idx] = re;
      field.im[idx] = im;
    }
  }
  return field;
}

/** Total launch-plane power integral, sum |U|^2 dx^2. */
export function launchPower(field: ComplexField, grid: Grid2D): number {
  let sum = 0;
  for (let i = 0; i < field.length; i++) {
    sum += field.re[i] * field.re[i] + field.im[i] * field.im[i];
  }
  return sum * grid.dx * grid.dx;
}

/**
 * Untruncated analytic launch power for a set of channel states, used as a
 * declared reference so truncation loss can be reported separately.
 */
export function idealLaunchPower(states: readonly ChannelActual[]): number {
  return states.reduce((acc, s) => acc + (s.enabled ? s.amplitude * s.amplitude : 0), 0);
}
