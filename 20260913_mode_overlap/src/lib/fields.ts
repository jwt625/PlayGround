import { gaussianField } from './engine';
import type { FieldComponent, FieldGrid, GaussianMode, ResolvedTransform, ScaleKind } from './types';

/** Shared physical extent derived from both modes. Never used to compute overlap. */
export function sharedExtent(a: GaussianMode, b: GaussianMode, factor = 1.6): { x: number; y: number } {
	const spanX = Math.max(a.mfdXUm, b.mfdXUm) * factor;
	const spanY = Math.max(a.mfdYUm, b.mfdYUm) * factor;
	return { x: Math.max(spanX / 2, 1), y: Math.max(spanY / 2, 1) };
}

export interface FieldSampleOptions {
	mode: GaussianMode;
	transform: ResolvedTransform;
	wavelengthUm: number;
	extentXUm: number;
	extentYUm: number;
	samples: number;
}

/**
 * Reconstruct the complex field on an explicit physics grid. The render canvas
 * samples this grid; canvas size never changes the underlying physics.
 */
export function sampleField(options: FieldSampleOptions): FieldGrid {
	const { mode, transform, wavelengthUm, extentXUm, extentYUm, samples } = options;
	const re = new Float64Array(samples * samples);
	const im = new Float64Array(samples * samples);
	for (let iy = 0; iy < samples; iy++) {
		const y = samples === 1 ? 0 : -extentYUm + (2 * extentYUm * iy) / (samples - 1);
		for (let ix = 0; ix < samples; ix++) {
			const x = samples === 1 ? 0 : -extentXUm + (2 * extentXUm * ix) / (samples - 1);
			const value = gaussianField(mode, x, y, wavelengthUm, transform);
			const index = iy * samples + ix;
			re[index] = value.re;
			im[index] = value.im;
		}
	}
	return { extentXUm, extentYUm, samples, re, im };
}

/** Map a complex grid to a real display component. */
export function componentValues(grid: FieldGrid, component: FieldComponent): Float64Array {
	const { re, im, samples } = grid;
	const out = new Float64Array(samples * samples);
	for (let i = 0; i < out.length; i++) {
		switch (component) {
			case 'intensity':
				out[i] = re[i] * re[i] + im[i] * im[i];
				break;
			case 'amplitude':
				out[i] = Math.hypot(re[i], im[i]);
				break;
			case 'real':
				out[i] = re[i];
				break;
			case 'phase':
				out[i] = Math.atan2(im[i], re[i]);
				break;
		}
	}
	return out;
}

export interface ValueRange {
	min: number;
	max: number;
	symmetric: boolean;
}

export function rangeOf(values: Float64Array, component: FieldComponent): ValueRange {
	let min = Infinity;
	let max = -Infinity;
	for (const value of values) {
		if (value < min) min = value;
		if (value > max) max = value;
	}
	if (component === 'phase') return { min: -Math.PI, max: Math.PI, symmetric: true };
	if (component === 'real') {
		const bound = Math.max(Math.abs(min), Math.abs(max)) || 1;
		return { min: -bound, max: bound, symmetric: true };
	}
	return { min: 0, max: max || 1, symmetric: false };
}

/** Normalize a single value to [0,1] for colormap lookup. */
export function normalizeValue(value: number, range: ValueRange, scale: ScaleKind): number {
	let { min, max } = range;
	let v = value;
	if (scale === 'log' && !range.symmetric) {
		min = Math.log10(Math.max(min, 1e-12));
		max = Math.log10(Math.max(max, 1e-11));
		v = Math.log10(Math.max(v, 1e-12));
	}
	if (range.symmetric) {
		const bound = Math.max(Math.abs(min), Math.abs(max)) || 1;
		return (v / bound + 1) / 2;
	}
	return (v - min) / (max - min || 1);
}

/** Union of ranges used when two panels share one color scale. */
export function mergeRanges(a: ValueRange, b: ValueRange, component: FieldComponent): ValueRange {
	if (component === 'phase') return { min: -Math.PI, max: Math.PI, symmetric: true };
	if (a.symmetric || b.symmetric) {
		const bound = Math.max(Math.abs(a.min), Math.abs(a.max), Math.abs(b.min), Math.abs(b.max)) || 1;
		return { min: -bound, max: bound, symmetric: true };
	}
	return { min: Math.min(a.min, b.min), max: Math.max(a.max, b.max), symmetric: false };
}

export interface IntegrandGrid {
	grid: FieldGrid;
	re: Float64Array;
	im: Float64Array;
}

/**
 * Raw scalar overlap integrand u_A* conj(u_B) at each physics-grid point.
 * This is a display aid; the authoritative eta comes from gaussianOverlap.
 */
export function sampleIntegrand(
	a: FieldSampleOptions,
	b: FieldSampleOptions,
	samples: number
): IntegrandGrid {
	const gridA = sampleField({ ...a, samples });
	const gridB = sampleField({ ...b, samples });
	const re = new Float64Array(samples * samples);
	const im = new Float64Array(samples * samples);
	for (let i = 0; i < re.length; i++) {
		re[i] = gridA.re[i] * gridB.re[i] + gridA.im[i] * gridB.im[i];
		im[i] = gridA.im[i] * gridB.re[i] - gridA.re[i] * gridB.im[i];
	}
	return { grid: gridA, re, im };
}
