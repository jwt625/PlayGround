/** On-the-fly waveguide questions from the solved modes.
 *
 * Picks two DIFFERENT solved (guide, mode) entries and evaluates the scalar
 * overlap integral of their actual fields. No field shifting or manual
 * alignment: the two modes are independent eigen-solutions. Returns questions
 * in the same shape as the analytic bank plus a `view` for the toggling viewer.
 */
import { bucketIndex, rng } from '../scenarios/generator.mjs';
import { mismatchLoss } from '../physics/gaussian.mjs';
import type { CachedQuestion, ModeGalleryEntry, StoredFieldView } from './types';

function trapezoid(edges: ArrayLike<number>): Float64Array {
	const n = edges.length;
	const w = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const left = i > 0 ? edges[i] - edges[i - 1] : edges[1] - edges[0];
		const right = i < n - 1 ? edges[i + 1] - edges[i] : edges[i] - edges[i - 1];
		w[i] = 0.5 * (left + right);
	}
	return w;
}

function unionSorted(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
	const out: number[] = [];
	let i = 0;
	let j = 0;
	while (i < a.length || j < b.length) {
		let value: number;
		if (i >= a.length) value = b[j++];
		else if (j >= b.length) value = a[i++];
		else if (a[i] < b[j]) value = a[i++];
		else if (b[j] < a[i]) value = b[j++];
		else {
			value = a[i];
			i++;
			j++;
		}
		if (out.length === 0 || Math.abs(out[out.length - 1] - value) > 1e-12) out.push(value);
	}
	return Float64Array.from(out);
}

interface GridField {
	x: Float64Array;
	y: Float64Array;
	field: Float64Array;
	nx: number;
	ny: number;
}

function toGrid(entry: ModeGalleryEntry): GridField {
	const n = entry.axes.nx;
	const m = entry.axes.ny;
	const x = new Float64Array(n);
	const y = new Float64Array(m);
	for (let i = 0; i < n; i++) x[i] = entry.axes.x0 + ((entry.axes.x1 - entry.axes.x0) * i) / (n - 1);
	for (let j = 0; j < m; j++) y[j] = entry.axes.y0 + ((entry.axes.y1 - entry.axes.y0) * j) / (m - 1);
	return { x, y, field: Float64Array.from(entry.field), nx: n, ny: m };
}

function sample(grid: GridField, x: number, y: number): number {
	if (x < grid.x[0] || x > grid.x[grid.nx - 1] || y < grid.y[0] || y > grid.y[grid.ny - 1]) return 0;
	let i0 = 0;
	while (i0 < grid.nx - 2 && grid.x[i0 + 1] < x) i0++;
	let j0 = 0;
	while (j0 < grid.ny - 2 && grid.y[j0 + 1] < y) j0++;
	const tx = (x - grid.x[i0]) / (grid.x[i0 + 1] - grid.x[i0] || 1);
	const ty = (y - grid.y[j0]) / (grid.y[j0 + 1] - grid.y[j0] || 1);
	const v00 = grid.field[j0 * grid.nx + i0];
	const v10 = grid.field[j0 * grid.nx + i0 + 1];
	const v01 = grid.field[(j0 + 1) * grid.nx + i0];
	const v11 = grid.field[(j0 + 1) * grid.nx + i0 + 1];
	return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

/** Scalar overlap of two stored fields on their common grid: η = |⟨a|b⟩|²/(⟨a|a⟩⟨b|b⟩). */
export function overlapStored(a: GridField, b: GridField): number {
	const ux = unionSorted(a.x, b.x);
	const uy = unionSorted(a.y, b.y);
	const wx = trapezoid(ux);
	const wy = trapezoid(uy);
	let na = 0;
	let nb = 0;
	let c = 0;
	for (let j = 0; j < uy.length; j++) {
		for (let i = 0; i < ux.length; i++) {
			const w = wx[i] * wy[j];
			const va = sample(a, ux[i], uy[j]);
			const vb = sample(b, ux[i], uy[j]);
			na += w * va * va;
			nb += w * vb * vb;
			c += w * va * vb;
		}
	}
	return na > 0 && nb > 0 ? (c * c) / (na * nb) : 0;
}

function viewOf(entry: ModeGalleryEntry): StoredFieldView {
	return {
		id: entry.id,
		nx: entry.axes.nx,
		ny: entry.axes.ny,
		x0: entry.axes.x0,
		x1: entry.axes.x1,
		y0: entry.axes.y0,
		y1: entry.axes.y1,
		field: entry.field,
		core: entry.core,
		backgroundIndex: entry.backgroundIndex,
		indexRegions: entry.indexRegions
	};
}

const PROVENANCE = {
	cacheVersion: 'cache-1',
	generatorVersion: 'waveguide-2',
	presetsVersion: 'presets-1',
	presetsHash: 'runtime',
	examplesHash: 'runtime'
};

export interface WaveguideOptions {
	seed: number;
	count?: number;
}

/**
 * Pair two distinct solved modes at random and return the overlap question.
 * Repeated pairings are avoided so a session does not ask the same comparison.
 */
export function createWaveguideQuestions(entries: ModeGalleryEntry[], options: WaveguideOptions): CachedQuestion[] {
	const random = rng(options.seed >>> 0);
	const count = options.count ?? 12;
	const usable = entries.filter((entry) => entry.confinement >= 0.6 && entry.residual < 1e-4);
	if (usable.length < 2) return [];
	const byOrder = new Map<number, ModeGalleryEntry[]>();
	for (const entry of usable) {
		if (!byOrder.has(entry.order)) byOrder.set(entry.order, []);
		byOrder.get(entry.order)?.push(entry);
	}
	const pairedOrders = [...byOrder.entries()].filter(([, list]) => list.length >= 2).map(([order]) => order);
	const draw = <T>(items: T[]): T => items[Math.floor(random() * items.length) % items.length];
	const used = new Set<string>();
	const questions: CachedQuestion[] = [];
	for (let index = 0; index < count; index++) {
		let a: ModeGalleryEntry = usable[0];
		let b: ModeGalleryEntry = usable[1];
		let guard = 0;
		const wantSameOrder = pairedOrders.length > 0 && random() < 0.8;
		do {
			if (wantSameOrder) {
				// Same mode order across different guides: informative and non-zero.
				const list = byOrder.get(draw(pairedOrders)) as ModeGalleryEntry[];
				a = draw(list);
				b = draw(list);
			} else {
				a = draw(usable);
				b = draw(usable);
			}
		} while ((a.id === b.id || used.has([a.id, b.id].sort().join('|'))) && guard++ < 40);
		if (a.id === b.id) {
			// Exhausted unique pairs: force a different entry so we still return count.
			const alternatives = (byOrder.get(a.order) ?? usable).filter((entry) => entry.id !== a.id);
			if (alternatives.length === 0) continue;
			b = alternatives[0];
		}
		used.add([a.id, b.id].sort().join('|'));
		const eta = overlapStored(toGrid(a), toGrid(b));
		const explanation = `Two independently solved guides: ${a.id} (${a.modeLabel}) vs ${b.id} (${b.modeLabel}). Their normalized mode overlap is η ${eta.toFixed(3)}.`;
		questions.push(makeQuestion(a, b, viewOf(a), viewOf(b), eta, explanation, 'waveguide-overlap', index));
	}
	return questions;
}

function makeQuestion(
	a: ModeGalleryEntry,
	b: ModeGalleryEntry,
	viewA: StoredFieldView,
	viewB: StoredFieldView,
	eta: number,
	explanation: string,
	category: string,
	index: number
): CachedQuestion {
	const loss = mismatchLoss(eta);
	return {
		id: `wg-${a.id}-${b.id}-${index}`,
		title: 'Waveguide mode overlap',
		seed: 0,
		category,
		status: 'ready-analytic',
		model: 'scalar-waveguide-2d',
		generatorVersion: 'waveguide-2',
		solverVersion: 'scalar-fem-2d-1',
		materialDBVersion: 'materials-1',
		meshSettings: { kind: 'scalar-fem-2d-display' },
		wavelengthUm: a.wavelengthUm,
		a: { mfdXUm: a.dimensions.widthUm, mfdYUm: a.dimensions.heightUm },
		b: { mfdXUm: b.dimensions.widthUm, mfdYUm: b.dimensions.heightUm },
		transform: { dxUm: 0, dyUm: 0, thetaXRad: 0, thetaYRad: 0, polarizationRad: 0, gapIndex: 1 },
		answer: { eta, ...loss },
		bucket: bucketIndex(eta),
		provenance: PROVENANCE,
		view: { a: viewA, b: viewB, explanation, note: 'Scalar quasi-TE FEM fields; experimental, not full-vector.' }
	};
}
