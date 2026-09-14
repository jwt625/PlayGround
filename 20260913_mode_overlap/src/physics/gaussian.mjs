/** Scalar paraxial, axis-aligned Gaussian fields at a common waist plane.
 * MFD is the intensity 1/e² diameter. All lengths are in µm.
 */
export const GAUSSIAN_VERSION = 'gaussian-waist-1';
export function positive(value, name) {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be finite and positive`);
  return value;
}
export function validateMode(mode) {
  positive(mode.mfdXUm, 'mfdXUm');
  positive(mode.mfdYUm, 'mfdYUm');
  return mode;
}
export function transformDefaults(t = {}) {
  const keys = ['dxUm', 'dyUm', 'thetaXRad', 'thetaYRad', 'polarizationRad', 'gapIndex'];
  for (const key of Object.keys(t)) if (!keys.includes(key)) throw new TypeError(`Unsupported transform: ${key}`);
  const result = {dxUm: 0, dyUm: 0, thetaXRad: 0, thetaYRad: 0, polarizationRad: 0, gapIndex: 1, ...t};
  for (const [key, value] of Object.entries(result)) if (!Number.isFinite(value)) throw new RangeError(`${key} must be finite`);
  positive(result.gapIndex, 'gapIndex');
  if (Math.abs(result.thetaXRad) > .1 || Math.abs(result.thetaYRad) > .1) throw new RangeError('Paraxial tilt limit is 0.1 rad per axis');
  return result;
}
export function gaussianOverlap(a, b, wavelengthUm = 1.55, transform = {}) {
  validateMode(a); validateMode(b); positive(wavelengthUm, 'wavelengthUm');
  const t = transformDefaults(transform);
  const k = 2 * Math.PI * t.gapIndex / wavelengthUm;
  let eta = Math.cos(t.polarizationRad) ** 2;
  // Treat machine-precision orthogonal linear polarization as exactly zero.
  if (eta < 1e-30) eta = 0;
  for (const [key, d, theta] of [['mfdXUm', t.dxUm, t.thetaXRad], ['mfdYUm', t.dyUm, t.thetaYRad]]) {
    const wa = a[key] / 2, wb = b[key] / 2, s = wa * wa + wb * wb;
    const q = k * Math.sin(theta);
    eta *= (2 * wa * wb / s) * Math.exp(-2 * d * d / s - q * q * wa * wa * wb * wb / (2 * s));
  }
  if (!Number.isFinite(eta)) throw new RangeError('Gaussian calculation overflow');
  return eta;
}
export function mismatchLoss(eta) {
  if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new RangeError('eta must be in [0,1]');
  // JSON cannot encode Infinity. Null plus an explicit flag is the storage contract.
  return {lossDB: eta === 0 ? null : -10 * Math.log10(eta), lossIsInfinite: eta === 0};
}
/** Complex normalized scalar field; rendering must never feed the overlap engine. */
export function gaussianField(mode, xUm, yUm, wavelengthUm = 1.55, transform = {}) {
  validateMode(mode); positive(wavelengthUm, 'wavelengthUm');
  if (!Number.isFinite(xUm) || !Number.isFinite(yUm)) throw new RangeError('Coordinates must be finite');
  const t = transformDefaults(transform), wx = mode.mfdXUm / 2, wy = mode.mfdYUm / 2;
  const amplitude = Math.sqrt(2 / (Math.PI * wx * wy)) * Math.exp(-(((xUm - t.dxUm) / wx) ** 2 + ((yUm - t.dyUm) / wy) ** 2));
  const phase = 2 * Math.PI * t.gapIndex / wavelengthUm * (Math.sin(t.thetaXRad) * xUm + Math.sin(t.thetaYRad) * yUm);
  return {re: amplitude * Math.cos(phase), im: amplitude * Math.sin(phase)};
}
