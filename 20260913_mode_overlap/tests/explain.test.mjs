import test from 'node:test';
import assert from 'node:assert/strict';
import { explainQuestion } from '../src/lib/explain.ts';
import { analyticQuestion } from '../src/scenarios/generator.mjs';

const base = { mfdXUm: 4, mfdYUm: 4 };

test('dominant effect is identified by the largest counterfactual improvement', () => {
	const offset = analyticQuestion({ id: 'o', title: 'offset', a: base, b: base, transform: { dxUm: 2 } });
	const offsetExplanation = explainQuestion(offset);
	assert.equal(offsetExplanation.dominant, 'lateral offset');
	assert.ok(offsetExplanation.sentence.includes('lateral offset'));

	const size = analyticQuestion({ id: 's', title: 'size', a: { mfdXUm: 4, mfdYUm: 4 }, b: { mfdXUm: 8, mfdYUm: 8 } });
	assert.equal(explainQuestion(size).dominant, 'mode-size mismatch');

	const tilt = analyticQuestion({ id: 't', title: 'tilt', a: base, b: base, transform: { thetaXRad: 0.05 } });
	assert.equal(explainQuestion(tilt).dominant, 'angular tilt');

	const polarization = analyticQuestion({
		id: 'p',
		title: 'polarization',
		a: base,
		b: base,
		transform: { polarizationRad: Math.PI / 4 }
	});
	assert.equal(explainQuestion(polarization).dominant, 'polarization rotation');
});

test('identical modes report no dominant perturbation', () => {
	const identical = analyticQuestion({ id: 'i', title: 'identical', a: base, b: base });
	const explanation = explainQuestion(identical);
	assert.equal(explanation.dominant, 'none');
	assert.equal(explanation.candidates.length, 0);
	assert.ok(explanation.sentence.includes('identical'));
});

test('interacting effects are described as a largest single improvement', () => {
	const combined = analyticQuestion({
		id: 'c',
		title: 'combined',
		a: { mfdXUm: 4, mfdYUm: 4 },
		b: { mfdXUm: 8, mfdYUm: 8 },
		transform: { dxUm: 1 }
	});
	const explanation = explainQuestion(combined);
	assert.ok(explanation.candidates.length >= 2);
	assert.ok(explanation.sentence.startsWith('Largest single improvement'));
});
