import test from 'node:test';
import assert from 'node:assert/strict';
import { createWaveguideQuestions, overlapStored } from '../src/lib/waveguide.ts';
import { bucketIndex } from '../src/scenarios/generator.mjs';

function entry(id, widthUm = 0.45, heightUm = 0.22) {
	const nx = 41;
	const ny = 41;
	const x0 = -1.2;
	const x1 = 1.2;
	const field = [];
	for (let j = 0; j < ny; j++) {
		const y = -1.2 + (2.4 * j) / (ny - 1);
		for (let i = 0; i < nx; i++) {
			const x = -1.2 + (2.4 * i) / (nx - 1);
			field.push(Math.exp(-((x / 0.3) ** 2 + (y / 0.25) ** 2)));
		}
	}
	return {
		id,
		name: id,
		order: 0,
		category: 'silicon',
		modeLabel: 'quasi-TE0',
		materialLabel: 'Si / SiO2',
		cladding: 'SiO2',
		wavelengthUm: 1.55,
		status: 'experimental-scalar-quasi-te',
		neff: 2.5,
		confinement: 0.95,
		residual: 1e-12,
		dimensions: { widthUm, heightUm, slabUm: null },
		core: { x0: -widthUm / 2, x1: widthUm / 2, y0: -heightUm / 2, y1: heightUm / 2 },
		axes: { x0, x1, y0: x0, y1: x1, nx, ny },
		backgroundIndex: 1.444,
		indexRegions: [{ x0: -widthUm / 2, x1: widthUm / 2, y0: -heightUm / 2, y1: heightUm / 2, index: 3.475, material: 'si' }],
		field
	};
}

test('stored-field self overlap is unity', () => {
	const e = entry('a');
	const grid = {
		x: Float64Array.from({ length: e.axes.nx }, (_, i) => e.axes.x0 + ((e.axes.x1 - e.axes.x0) * i) / (e.axes.nx - 1)),
		y: Float64Array.from({ length: e.axes.ny }, (_, j) => e.axes.y0 + ((e.axes.y1 - e.axes.y0) * j) / (e.axes.ny - 1)),
		field: Float64Array.from(e.field),
		nx: e.axes.nx,
		ny: e.axes.ny
	};
	assert.ok(Math.abs(overlapStored(grid, grid) - 1) < 1e-9);
});

test('waveguide questions pair two independently solved distinct modes', () => {
	const entries = [entry('soi-strip'), entry('soi-strip-b', 0.5, 0.22), entry('sin-thick', 1.5, 0.8)];
	const first = createWaveguideQuestions(entries, { seed: 5, count: 5 });
	const second = createWaveguideQuestions(entries, { seed: 5, count: 5 });
	assert.equal(first.length, 5);
	assert.deepEqual(first.map((q) => q.answer.eta), second.map((q) => q.answer.eta));
	for (const question of first) {
		assert.ok(question.answer.eta >= 0 && question.answer.eta <= 1);
		assert.equal(question.bucket, bucketIndex(question.answer.eta));
		assert.ok(question.view, 'waveguide question needs a view');
		assert.notEqual(question.view.a.id, question.view.b.id, 'pair must be two different solved modes');
		assert.equal(question.category, 'waveguide-overlap');
		assert.equal(question.transform.dxUm, 0);
		assert.equal(question.transform.dyUm, 0);
		assert.equal(question.view.a.field.length, question.view.a.nx * question.view.a.ny);
	}
});

test('no usable entries produces no questions', () => {
	const weak = [{ ...entry('weak'), confinement: 0.2 }];
	assert.deepEqual(createWaveguideQuestions(weak, { seed: 1, count: 4 }), []);
});
