import test from 'node:test';
import assert from 'node:assert/strict';
import {
	refractiveIndex,
	relativePermittivity,
	isReleaseEligible
} from '../src/physics/materials.mjs';
import {
	buildAlignedMesh1D,
	buildGrid2D,
	buildSlabMesh,
	rasterizeRegions
} from '../src/physics/mesh.mjs';
import { solveModes } from '../src/physics/solve-modes.mjs';

test('materials carry validity windows and sourced reference values', () => {
	assert.ok(Math.abs(refractiveIndex('sio2', 1.55) - 1.444) < 2e-3);
	assert.ok(Math.abs(refractiveIndex('si', 1.55) - 3.4757) < 6e-3);
	assert.ok(Math.abs(refractiveIndex('air', 1.55) - 1) < 1e-12);
	assert.equal(relativePermittivity('si', 1.55).toFixed(2), '12.08');
});

test('materials refuse silent extrapolation outside the validity window', () => {
	assert.throws(() => refractiveIndex('si', 0.8), /valid/);
	assert.throws(() => refractiveIndex('sio2', 0.1), /valid/);
	assert.throws(() => refractiveIndex('nope', 1.55), /Unknown/);
});

test('placeholders are not release-eligible', () => {
	assert.equal(isReleaseEligible('sio2'), true);
	assert.equal(isReleaseEligible('si'), true);
	assert.equal(isReleaseEligible('sin'), false);
});

test('aligned 1D mesh puts segment boundaries exactly on nodes', () => {
	const mesh = buildAlignedMesh1D(
		[
			{ material: 'sio2', thicknessUm: 0.1 },
			{ material: 'si', thicknessUm: 0.05 }
		],
		{ maxStepUm: 0.02 }
	);
	assert.equal(mesh.x[0], 0);
	assert.ok(Math.abs(mesh.x[mesh.x.length - 1] - 0.15) < 1e-12);
	assert.ok([...mesh.x].some((x) => Math.abs(x - 0.1) < 1e-12), 'interface node missing');
	assert.equal(mesh.elementMaterial.filter((m) => m === 'si').length, Math.ceil(0.05 / 0.02));
});

test('slab mesh aligns the core boundaries and keeps one material per element', () => {
	const mesh = buildSlabMesh({
		core: { material: 'si', thicknessUm: 0.22 },
		substrate: 'sio2',
		cladding: 'sio2',
		wavelengthUm: 1.55,
		minPerFeature: 12
	});
	assert.ok([...mesh.x].some((x) => Math.abs(x - mesh.coreStartUm) < 1e-12));
	assert.ok([...mesh.x].some((x) => Math.abs(x - mesh.coreEndUm) < 1e-12));
	const epsCore = relativePermittivity('si', 1.55);
	assert.ok(mesh.eps.some((eps) => Math.abs(eps - epsCore) < 1e-9));
	assert.ok(mesh.eps.every((eps) => eps > 1));
});

test('2D grid trapezoidal weights integrate area and rasterization respects priority', () => {
	const grid = buildGrid2D({ xEdges: [0, 1, 3], yEdges: [0, 2] });
	const area = [...grid.weights].reduce((sum, w) => sum + w, 0);
	assert.ok(Math.abs(area - 6) < 1e-12);
	const eps = rasterizeRegions(
		grid,
		[
			{ x0: 0, x1: 1, y0: 0, y1: 2, material: 'sio2', priority: 1 },
			{ x0: 0.5, x1: 1, y0: 0, y1: 2, material: 'si', priority: 2 }
		],
		1.55
	);
	const at = (x, y) => {
		const ix = grid.x.indexOf(x);
		const iy = grid.y.indexOf(y);
		return eps[iy * grid.nx + ix];
	};
	assert.ok(Math.abs(at(0, 0) - relativePermittivity('sio2', 1.55)) < 1e-9);
	assert.ok(Math.abs(at(1, 0) - relativePermittivity('si', 1.55)) < 1e-9);
});

test('solveModes dispatches the slab backend and rejects unimplemented 2D', () => {
	const result = solveModes({
		geometry: { kind: 'slab', core: { material: 'si', thicknessUm: 0.22 }, substrate: 'sio2', cladding: 'sio2' },
		wavelengthUm: 1.55,
		polarization: 'TE',
		solver: { modeCount: 1, minPerFeature: 24 }
	});
	assert.equal(result.schemaVersion, 'solve-modes-1');
	assert.equal(result.model, 'slab-fem-1d');
	assert.equal(result.modes.length, 1);
	assert.throws(
		() => solveModes({ geometry: { kind: 'rect' }, wavelengthUm: 1.55 }),
		/2D full-vector/
	);

	const scalar = solveModes({
		geometry: { kind: 'scalar-2d', xEdges: [0, 0.3, 0.6], yEdges: [0, 0.3, 0.6], background: 'si' },
		wavelengthUm: 1.55,
		solver: { modeCount: 1 }
	});
	assert.equal(scalar.releaseEligible, false);
	assert.match(scalar.modelNote, /not release-eligible/);
});
