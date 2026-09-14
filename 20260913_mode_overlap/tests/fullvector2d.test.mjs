import test from 'node:test';
import assert from 'node:assert/strict';
import { assembleFullVector2D, solveNear } from '../src/physics/fullvector2d.mjs';
import { buildGrid2D, rasterizeRegions } from '../src/physics/mesh.mjs';

const wavelengthUm = 1.55;

function fixture() {
	const grid = buildGrid2D({ xEdges: [0, 0.3, 0.6, 1.0], yEdges: [0, 0.4, 0.8] });
	const nodeEps = rasterizeRegions(
		grid,
		[{ x0: 0.3, x1: 0.6, y0: 0, y1: 0.8, material: 'si', priority: 1 }],
		wavelengthUm,
		'sio2'
	);
	const gridM = {
		nx: grid.nx,
		ny: grid.ny,
		x: Float64Array.from(grid.x, (v) => v * 1e-6),
		y: Float64Array.from(grid.y, (v) => v * 1e-6)
	};
	const k0 = (2 * Math.PI) / (wavelengthUm * 1e-6);
	return { grid, nodeEps, assembled: assembleFullVector2D(gridM, nodeEps, k0), k0 };
}

test('full-vector matrices are finite and symmetric', () => {
	const { assembled } = fixture();
	for (const name of ['K0', 'K1', 'K2']) {
		const rows = assembled[name];
		for (let i = 0; i < rows.length; i++) {
			for (const [j, value] of rows[i]) {
				assert.ok(Number.isFinite(value), `${name}[${i}][${j}] not finite`);
				if (i !== j) assert.ok(Math.abs(value - (rows[j].get(i) ?? 0)) < 1e-3, `${name} not symmetric at ${i},${j}`);
			}
		}
	}
});

test('full-vector Rayleigh iteration returns a finite eigenpair on a seeded guess', () => {
	const { assembled } = fixture();
	const { dofs } = assembled;
	const fixed = new Uint8Array(dofs);
	const result = solveNear({ K0: assembled.K0, K1: assembled.K1, K2: assembled.K2, dofs, fixed, sigma0: 4e6 });
	assert.ok(Number.isFinite(result.beta));
	assert.ok(Number.isFinite(result.residual));
	assert.equal(result.vector.length, dofs);
});
