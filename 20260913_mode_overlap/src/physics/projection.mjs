/** Selected-mode projection (C5.1).
 *
 * Projects a mode onto a set of forward guided modes of the SAME guide. Only
 * then is the basis orthonormal and Ση the captured fraction. `1 − Ση` is
 * "not captured by the selected forward guided modes", not radiation loss
 * (K002). The result is qualified by the caller; this module never relabels the
 * remainder as scattering.
 */
import { overlapModes } from './overlap.mjs';

export const PROJECTION_VERSION = 'projection-1';

export function projectModeOntoBasis(target, basis) {
	if (!Array.isArray(basis) || basis.length === 0) throw new RangeError('basis required');
	const components = [];
	let captured = 0;
	for (const mode of basis) {
		const overlap = overlapModes(target, mode);
		const eta = overlap.eta ?? 0;
		captured += eta;
		components.push({
			modeIndex: mode.modeIndex ?? components.length,
			neff: mode.neff,
			eta,
			cAB: overlap.cAB,
			qualified: overlap.positivePower
		});
	}
	return {
		projectionVersion: PROJECTION_VERSION,
		coefficients: components,
		captured,
		remainder: 1 - captured,
		remainderLabel: 'not captured by the selected forward guided modes'
	};
}

/** Combine modes on a shared mesh with complex weights into one bundle. */
export function combineModes(modes, weights) {
	if (modes.length !== weights.length) throw new RangeError('weights must match modes');
	if (modes.length === 0) throw new RangeError('at least one mode required');
	const template = modes[0];
	const names = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'];
	const length = template.mesh.xUm.length;
	const fields = {};
	for (const name of names) fields[name] = { re: new Float64Array(length), im: new Float64Array(length) };
	for (let m = 0; m < modes.length; m++) {
		const weight = weights[m];
		const wRe = typeof weight === 'number' ? weight : weight.re;
		const wIm = typeof weight === 'number' ? 0 : weight.im;
		for (const name of names) {
			const source = modes[m].fields[name];
			for (let i = 0; i < length; i++) {
				fields[name].re[i] += wRe * source.re[i] - wIm * source.im[i];
				fields[name].im[i] += wRe * source.im[i] + wIm * source.re[i];
			}
		}
	}
	return { ...template, fields };
}
