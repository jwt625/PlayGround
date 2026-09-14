import test from 'node:test';
import assert from 'node:assert/strict';
import { balancedMixedBank } from '../src/scenarios/generator.mjs';

test('mixed Gaussian bank is deterministic and covers every bucket', () => {
	const bank = balancedMixedBank(7, 2);
	assert.deepEqual(bank, balancedMixedBank(7, 2));
	const counts = Array(7).fill(0);
	for (const question of bank) {
		counts[question.bucket]++;
		assert.equal(question.category, 'gaussian-mixed');
	}
	assert.deepEqual(counts, Array(7).fill(2));
});

test('mixed questions include aspect-ratio variation and non-trivial effects', () => {
	const bank = balancedMixedBank(3, 3);
	const elliptical = bank.filter((q) => Math.abs(q.a.mfdXUm - q.a.mfdYUm) > 1e-6);
	assert.ok(elliptical.length > 0, 'expected some elliptical modes');
	const tilted = bank.filter((q) => q.transform.thetaXRad !== 0 || q.transform.thetaYRad !== 0);
	assert.ok(tilted.length > 0, 'expected some tilt');
	assert.ok(bank.every((q) => Math.abs(q.answer.eta - 1) > 1e-9 || q.bucket === 6));
});

test('misalignment stays bounded (not too far off)', () => {
	for (const question of balancedMixedBank(11, 2)) {
		const sx = Math.sqrt((question.a.mfdXUm ** 2 + question.b.mfdXUm ** 2) / 2) / 2;
		const sy = Math.sqrt((question.a.mfdYUm ** 2 + question.b.mfdYUm ** 2) / 2) / 2;
		const normalized = Math.hypot(question.transform.dxUm / sx, question.transform.dyUm / sy);
		assert.ok(normalized <= 2.5 + 1e-9, `${normalized}`);
	}
});
