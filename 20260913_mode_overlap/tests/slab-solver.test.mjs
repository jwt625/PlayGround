import test from 'node:test';
import assert from 'node:assert/strict';
import { solveSlab, analyticSlabModes } from '../src/physics/slab.mjs';
import { refractiveIndex } from '../src/physics/materials.mjs';

const wavelengthUm = 1.55;
const thicknessUm = 0.22;
const core = { material: 'si', thicknessUm };
const substrate = 'sio2';
const cladding = 'sio2';

function analytic(polarization) {
	return analyticSlabModes({
		coreIndex: refractiveIndex('si', wavelengthUm),
		substrateIndex: refractiveIndex('sio2', wavelengthUm),
		claddingIndex: refractiveIndex('sio2', wavelengthUm),
		thicknessUm,
		wavelengthUm,
		polarization,
		modeCount: 1
	})[0].neff;
}

test('slab FDE matches closed-form TE and TM dispersion within 1e-4', () => {
	for (const polarization of ['TE', 'TM']) {
		const result = solveSlab({
			core,
			substrate,
			cladding,
			wavelengthUm,
			polarization,
			modeCount: 1,
			minPerFeature: 96
		});
		const finite = result.modes[0];
		assert.ok(Math.abs(finite.neff - analytic(polarization)) < 1e-4, `${polarization}: ${finite.neff}`);
		assert.ok(finite.residual < 1e-8);
		assert.ok(finite.power > 0);
		assert.ok(Math.abs(finite.power - 1) < 1e-9);
		assert.ok(finite.boundaryEnergyFraction < 1e-6);
	}
});

test('uniform guide makes TE and TM degenerate', () => {
	const uniform = { core: { material: 'sin', thicknessUm: 0.5 }, substrate: 'sin', cladding: 'sin' };
	const te = solveSlab({ ...uniform, wavelengthUm, polarization: 'TE', modeCount: 1, minPerFeature: 24 });
	const tm = solveSlab({ ...uniform, wavelengthUm, polarization: 'TM', modeCount: 1, minPerFeature: 24 });
	assert.ok(Math.abs(te.modes[0].neff - tm.modes[0].neff) < 1e-9);
});

test('mesh refinement converges and expansion does not move the mode', () => {
	const coarse = solveSlab({ core, substrate, cladding, wavelengthUm, polarization: 'TE', modeCount: 1, minPerFeature: 12 });
	const fine = solveSlab({ core, substrate, cladding, wavelengthUm, polarization: 'TE', modeCount: 1, minPerFeature: 96 });
	const reference = analytic('TE');
	assert.ok(Math.abs(fine.modes[0].neff - reference) < Math.abs(coarse.modes[0].neff - reference));
	assert.ok(Math.abs(fine.modes[0].neff - reference) < 1e-4);

	const narrow = solveSlab({ core, substrate, cladding, wavelengthUm, polarization: 'TE', modeCount: 1, minPerFeature: 48, paddingUm: 1.2 });
	const wide = solveSlab({ core, substrate, cladding, wavelengthUm, polarization: 'TE', modeCount: 1, minPerFeature: 48, paddingUm: 4 });
	assert.ok(Math.abs(narrow.modes[0].neff - wide.modes[0].neff) < 1e-4);
	assert.ok(wide.modes[0].boundaryEnergyFraction < 1e-6);
});

test('guided modes are contained and ordered by descending neff', () => {
	const result = solveSlab({
		core: { material: 'si', thicknessUm: 0.5 },
		substrate,
		cladding,
		wavelengthUm,
		polarization: 'TE',
		modeCount: 2,
		minPerFeature: 48
	});
	assert.ok(result.modes.length >= 2);
	assert.ok(result.modes[0].neff > result.modes[1].neff);
	assert.ok(result.modes[0].coreConfinement > 0.8);
	assert.ok(result.modes[1].neff > refractiveIndex('sio2', wavelengthUm));
});
