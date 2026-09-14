import test from 'node:test';
import assert from 'node:assert/strict';
import { eligibleForWarmStart, validateModeBundle } from '../src/solver/provenance.mjs';

const expected = {
	solverVersion: 'slab-fem-1d-1',
	materialDBVersion: 'materials-1',
	meshVersion: 'mesh-1',
	wavelengthUm: 1.55
};

const record = { ...expected, neff: 2.4, power: 1 };

test('matching provenance is eligible for warm start', () => {
	assert.equal(eligibleForWarmStart(record, expected), true);
	assert.deepEqual(validateModeBundle(record, expected).reasons, []);
});

test('solver, material, mesh and wavelength mismatches are rejected with reasons', () => {
	for (const key of ['solverVersion', 'materialDBVersion', 'meshVersion']) {
		const result = validateModeBundle({ ...record, [key]: 'stale' }, expected);
		assert.equal(result.ok, false, key);
		assert.ok(result.reasons.some((reason) => reason.includes(key)));
	}
	const wavelength = validateModeBundle({ ...record, wavelengthUm: 1.31 }, expected);
	assert.equal(wavelength.ok, false);
	assert.ok(wavelength.reasons.includes('wavelength mismatch'));
});

test('records missing metadata, finite neff or positive power are ineligible', () => {
	assert.equal(eligibleForWarmStart(null, expected), false);
	assert.equal(validateModeBundle({ ...record, neff: NaN }, expected).ok, false);
	assert.equal(validateModeBundle({ ...record, power: 0 }, expected).ok, false);
});
