import type { Colormap } from './types';

type RGB = [number, number, number];

const ANCHORS: Record<Colormap, RGB[]> = {
	viridis: [
		[68, 1, 84],
		[59, 82, 139],
		[33, 145, 140],
		[94, 201, 98],
		[253, 231, 37]
	],
	magma: [
		[0, 0, 4],
		[81, 18, 124],
		[183, 55, 121],
		[252, 137, 97],
		[252, 253, 191]
	],
	diverging: [
		[59, 130, 246],
		[17, 17, 17],
		[239, 68, 68]
	],
	gray: [
		[0, 0, 0],
		[255, 255, 255]
	],
	cyclic: []
};

function cyclicRGB(t: number): RGB {
	const hue = (((t % 1) + 1) % 1) * 360;
	const saturation = 0.85;
	const value = 0.95;
	const c = value * saturation;
	const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
	const m = value - c;
	const [r, g, b] =
		hue < 60 ? [c, x, 0] :
		hue < 120 ? [x, c, 0] :
		hue < 180 ? [0, c, x] :
		hue < 240 ? [0, x, c] :
		hue < 300 ? [x, 0, c] :
		[c, 0, x];
	return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/** Interpolate a colormap at t in [0,1]. Diverging maps t=0.5 to zero. */
export function colormapRGB(name: Colormap, t: number): RGB {
	if (name === 'cyclic') return cyclicRGB(t);
	const anchors = ANCHORS[name] ?? ANCHORS.viridis;
	const clamped = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
	const span = anchors.length - 1;
	const scaled = clamped * span;
	const index = Math.min(span - 1, Math.floor(scaled));
	const f = scaled - index;
	const a = anchors[index];
	const b = anchors[index + 1];
	return [
		Math.round(a[0] + (b[0] - a[0]) * f),
		Math.round(a[1] + (b[1] - a[1]) * f),
		Math.round(a[2] + (b[2] - a[2]) * f)
	];
}

export function colormapCSS(name: Colormap, steps = 8): string {
	const stops: string[] = [];
	for (let i = 0; i < steps; i++) {
		const t = i / (steps - 1);
		const [r, g, b] = colormapRGB(name, t);
		stops.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(0)}%`);
	}
	return `linear-gradient(90deg, ${stops.join(', ')})`;
}
