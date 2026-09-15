/** Perceptual colormaps for intensity and cyclic phase rings. */

type RGB = [number, number, number];

const VIRIDIS_STOPS: RGB[] = [
  [0.267, 0.005, 0.329],
  [0.188, 0.407, 0.554],
  [0.128, 0.567, 0.551],
  [0.369, 0.789, 0.383],
  [0.993, 0.906, 0.144],
];

export function intensityColor(t: number): RGB {
  const x = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  const scaled = x * (VIRIDIS_STOPS.length - 1);
  const i = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = VIRIDIS_STOPS[i];
  const b = VIRIDIS_STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Logarithmic normalization: floor of `dynamicRange` decades below max. */
export function logNormalize(intensity: number, maxIntensity: number, dynamicRange = 4): number {
  if (maxIntensity <= 0) return 0;
  const v = Math.max(intensity, maxIntensity * 10 ** -dynamicRange);
  const lo = Math.log10(maxIntensity * 10 ** -dynamicRange);
  const hi = Math.log10(maxIntensity);
  return Math.min(1, Math.max(0, (Math.log10(v) - lo) / (hi - lo)));
}

/** Cyclic phase color: phi = 0 and 2pi share a hue, pi is opposite. */
export function phaseColor(phase_rad: number): RGB {
  const hue = (((phase_rad % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI);
  return hslToRgb(hue, 0.85, 0.55);
}

export function hslToRgb(h: number, s: number, l: number): RGB {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h * 6;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const m = l - c / 2;
  return [r + m, g + m, b + m];
}
