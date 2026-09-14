import test from 'node:test';
import assert from 'node:assert/strict';
import { solveSlab } from '../src/physics/slab.mjs';
import { combineModes, projectModeOntoBasis } from '../src/physics/projection.mjs';

const wavelengthUm = 1.55;
const substrate = 'sio2';

function guide(thicknessUm, modeCount, polarization = 'TE') {
	return solveSlab({
		core: { material: 'si', thicknessUm },
		substrate,
		cladding: substrate,
		wavelengthUm,
		polarization,
		modeCount,
		minPerFeature: 48
	}).modes;
}

test('a mode projects fully onto its own orthonormal same-guide basis', () => {
	const [te0, te1] = guide(0.5, 2);
	const projection = projectModeOntoBasis(te0, [te0, te1]);
	assert.ok(Math.abs(projection.captured - 1) < 1e-9);
	assert.ok(Math.abs(projection.remainder) < 1e-9);
	assert.ok(Math.abs(projection.coefficients[1].eta) < 1e-12);
});

test('a combination inside the basis span is fully captured with no remainder', () => {
	const [te0, te1] = guide(0.5, 2);
	const combo = combineModes([te0, te1], [1, 0.5]);
	const projection = projectModeOntoBasis(combo, [te0, te1]);
	assert.ok(Math.abs(projection.captured - 1) < 1e-9);
	assert.ok(projection.remainder >= -1e-9 && projection.remainder < 1e-9);
});

test('a mode outside the basis leaves a qualified remainder', () => {
	const [te0] = guide(0.5, 1);
	const [other] = guide(0.9, 1);
	const projection = projectModeOntoBasis(other, [te0]);
	assert.ok(projection.captured > 0 && projection.captured < 1);
	assert.ok(projection.remainder > 0);
	assert.equal(projection.remainderLabel, 'not captured by the selected forward guided modes');
});

test('complex phase weights combine and project consistently', () => {
	const [te0, te1] = guide(0.5, 2);
	const combo = combineModes([te0, te1], [{ re: 0, im: 1 }, { re: 0.5, im: 0 }]);
	const projection = projectModeOntoBasis(combo, [te0, te1]);
	assert.ok(Math.abs(projection.captured - 1) < 1e-9);
});
