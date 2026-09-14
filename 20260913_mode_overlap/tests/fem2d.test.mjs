import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGrid2D, buildSlabMesh, rasterizeRegions } from '../src/physics/mesh.mjs';
import { refractiveIndex } from '../src/physics/materials.mjs';
import { solveScalar2D } from '../src/physics/fem2d.mjs';
import { analyticSlabModes } from '../src/physics/slab.mjs';
import { overlapScalarModes } from '../src/physics/overlap.mjs';

const wavelengthUm = 1.55;

function slabScenario(halfWidthY = 0.4, minPerFeature = 48) {
	const mesh = buildSlabMesh({
		core: { material: 'si', thicknessUm: 0.22 },
		substrate: 'sio2',
		cladding: 'sio2',
		wavelengthUm,
		minPerFeature
	});
	const grid = buildGrid2D({ xEdges: mesh.x, yEdges: [0, halfWidthY] });
	const eps = rasterizeRegions(
		grid,
		[{ x0: mesh.coreStartUm, x1: mesh.coreEndUm, y0: -1, y1: 1, material: 'si', priority: 1 }],
		wavelengthUm,
		'sio2'
	);
	return { grid, eps };
}

function analyticTE() {
	return analyticSlabModes({
		coreIndex: refractiveIndex('si', wavelengthUm),
		substrateIndex: refractiveIndex('sio2', wavelengthUm),
		claddingIndex: refractiveIndex('sio2', wavelengthUm),
		thicknessUm: 0.22,
		wavelengthUm,
		polarization: 'TE',
		modeCount: 1
	})[0].neff;
}

test('2D scalar solve reproduces the slab TE benchmark in the y-invariant limit', () => {
	const { grid, eps } = slabScenario(0.4, 36);
	const result = solveScalar2D({
		grid,
		nodeEps: eps,
		wavelengthUm,
		modeCount: 1,
		dirichlet: { xMin: true, xMax: true, yMin: false, yMax: false },
		lanczosSteps: 250
	});
	const reference = analyticTE();
	assert.ok(Math.abs(result.modes[0].neff - reference) < 1.5e-3, `${result.modes[0].neff} vs ${reference}`);
	// Lanczos without shift-invert needs many steps for 1e-8; see C3.2 notes.
	assert.ok(result.modes[0].residual < 1e-4);
});

test('2D scalar refinement reduces the slab-limit error', () => {
	const coarse = slabScenario(0.4, 12);
	const fine = slabScenario(0.4, 36);
	const reference = analyticTE();
	const coarseResult = solveScalar2D({ grid: coarse.grid, nodeEps: coarse.eps, wavelengthUm, modeCount: 1, dirichlet: { xMin: true, xMax: true, yMin: false, yMax: false }, lanczosSteps: 200 });
	const fineResult = solveScalar2D({ grid: fine.grid, nodeEps: fine.eps, wavelengthUm, modeCount: 1, dirichlet: { xMin: true, xMax: true, yMin: false, yMax: false }, lanczosSteps: 200 });
	assert.ok(Math.abs(fineResult.modes[0].neff - reference) < Math.abs(coarseResult.modes[0].neff - reference));
});

test('scalar overlap is unity for self and near zero for distinct guided modes', () => {
	const { grid, eps } = slabScenario(0.4, 24);
	const result = solveScalar2D({
		grid,
		nodeEps: eps,
		wavelengthUm,
		modeCount: 2,
		dirichlet: { xMin: true, xMax: true, yMin: false, yMax: false },
		lanczosSteps: 250
	});
	assert.ok(result.modes.length >= 2);
	const [first, second] = result.modes;
	assert.ok(Math.abs(overlapScalarModes(first, first).eta - 1) < 1e-9);
	assert.ok(overlapScalarModes(first, second).eta < 1e-4);
});
