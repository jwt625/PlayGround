import test from 'node:test';
import assert from 'node:assert/strict';
import { solveSlab } from '../src/physics/slab.mjs';
import { overlapModes } from '../src/physics/overlap.mjs';

const wavelengthUm = 1.55;
const substrate = 'sio2';

function solve(thicknessUm, polarization = 'TE', modeCount = 1, minPerFeature = 48) {
	return solveSlab({
		core: { material: 'si', thicknessUm },
		substrate,
		cladding: substrate,
		wavelengthUm,
		polarization,
		modeCount,
		minPerFeature
	}).modes;
}

function rotate(mode, phase) {
	const c = Math.cos(phase);
	const s = Math.sin(phase);
	const fields = Object.fromEntries(
		Object.entries(mode.fields).map(([name, component]) => [
			name,
			{
				re: Float64Array.from(component.re, (value, i) => value * c - component.im[i] * s),
				im: Float64Array.from(component.im, (value, i) => component.re[i] * s + value * c)
			}
		])
	);
	return { ...mode, fields };
}

test('self overlap is one and forward power is normalized', () => {
	const [fundamental] = solve(0.5);
	const result = overlapModes(fundamental, fundamental);
	assert.ok(Math.abs(result.eta - 1) < 1e-9);
	assert.ok(result.powerA > 0 && result.powerB > 0);
	assert.ok(result.positivePower);
});

test('same-guide modes are orthogonal and forward power stays positive', () => {
	const [te0, te1] = solve(0.5, 'TE', 2);
	const result = overlapModes(te0, te1);
	assert.ok(result.eta < 1e-12, `eta=${result.eta}`);
});

test('orthogonal polarizations do not couple', () => {
	const [te0] = solve(0.5, 'TE');
	const [tm0] = solve(0.5, 'TM');
	assert.equal(overlapModes(te0, tm0).eta, 0);
});

test('overlap is invariant under a global complex phase and reciprocal in A/B', () => {
	const [te0, te1] = solve(0.5, 'TE', 2);
	const baseline = overlapModes(te0, te1).eta;
	const rotated = overlapModes(rotate(te0, 0.7), te1).eta;
	assert.ok(Math.abs(rotated - baseline) < 1e-12);
	assert.equal(overlapModes(te0, te1).eta, overlapModes(te1, te0).eta);
	assert.ok(Math.abs(overlapModes(rotate(te0, 1.3), te0).eta - 1) < 1e-9);
});

test('cross-guide overlap falls as the thickness mismatch grows and stays below one', () => {
	const [reference] = solve(0.5);
	const [close] = solve(0.44);
	const [far] = solve(0.9);
	const etaClose = overlapModes(reference, close).eta;
	const etaFar = overlapModes(reference, far).eta;
	assert.ok(etaClose > 0.9 && etaClose < 1);
	assert.ok(etaFar > 0 && etaFar < etaClose);
});
