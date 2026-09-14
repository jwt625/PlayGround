import test from 'node:test';
import assert from 'node:assert/strict';
import {
	aggregate,
	buildContinuousAnswer,
	buildIntervalAnswer,
	selectQuestions,
	BUCKET_TARGET
} from '../src/lib/session-core.ts';
import { analyticQuestion, balancedAlignmentBank } from '../src/scenarios/generator.mjs';

const identical = () =>
	analyticQuestion({ id: 'anchor', title: 'Identical', a: { mfdXUm: 4, mfdYUm: 4 }, b: { mfdXUm: 4, mfdYUm: 4 } });

test('quota selection is deterministic and fills G008 bucket targets', () => {
	const bank = balancedAlignmentBank(7, 5);
	const first = selectQuestions(bank, { seed: 123, limit: 20 });
	const second = selectQuestions(bank, { seed: 123, limit: 20 });
	assert.deepEqual(first.questions.map((q) => q.id), second.questions.map((q) => q.id));
	assert.equal(first.questions.length, 20);
	assert.deepEqual(first.shortages, []);
	const counts = Array(7).fill(0);
	for (const question of first.questions) counts[question.bucket]++;
	assert.deepEqual(counts, BUCKET_TARGET);
	// No question can appear twice in one draw.
	assert.equal(new Set(first.questions.map((q) => q.id)).size, 20);
});

test('selection reports shortages and backfills without exceeding the limit', () => {
	const pool = Array.from({ length: 20 }, (_, index) =>
		analyticQuestion({ id: `same-${index}`, title: 'Same', a: { mfdXUm: 4, mfdYUm: 4 }, b: { mfdXUm: 4, mfdYUm: 4 } })
	);
	const result = selectQuestions(pool, { seed: 5, limit: 20 });
	assert.equal(result.questions.length, 20);
	assert.equal(result.shortages.length, 6);
	assert.ok(result.shortages.every((entry) => entry.available === 0));
});

test('continuous and interval answers record signed errors and scores', () => {
	const question = identical();
	const perfect = buildContinuousAnswer({ question, mode: 'continuous', guessEta: 1, durationMs: 500 });
	assert.equal(perfect.score, 100);
	assert.equal(perfect.absEtaError, 0);
	assert.equal(perfect.durationMs, 500);
	const wrong = buildContinuousAnswer({ question, mode: 'continuous', guessEta: 0, durationMs: 250 });
	assert.ok(Math.abs(wrong.score - 100 * Math.exp(-20)) < 1e-9);
	assert.equal(wrong.absEtaError, 1);
	assert.equal(wrong.trueLossDB, 0);
	assert.equal(wrong.guessLossDB, 60);

	const correctBin = buildIntervalAnswer({ question, mode: 'interval', intervalIndex: 5, durationMs: 100 });
	assert.equal(correctBin.score, 100);
	assert.equal(correctBin.guessEta, 0.975);
	const wrongBin = buildIntervalAnswer({ question, mode: 'interval', intervalIndex: 0, durationMs: 100 });
	assert.ok(wrongBin.score < 100);
});

test('aggregation reports MAE, RMSE, category and bucket summaries', () => {
	const questionA = identical();
	const questionB = analyticQuestion({
		id: 'offset',
		title: 'Offset',
		a: { mfdXUm: 4, mfdYUm: 4 },
		b: { mfdXUm: 4, mfdYUm: 4 },
		transform: { dxUm: 2 }
	});
	const records = [
		buildContinuousAnswer({ question: questionA, mode: 'continuous', guessEta: 1, durationMs: 1000 }),
		buildContinuousAnswer({ question: questionB, mode: 'continuous', guessEta: 1, durationMs: 3000 })
	];
	const results = aggregate(records);
	assert.equal(results.count, 2);
	assert.equal(results.mae, records[1].absEtaError / 2);
	assert.ok(results.rmse > 0);
	assert.equal(results.meanDurationMs, 2000);
	assert.equal(results.medianDurationMs, 2000);
	assert.equal(results.categoryBreakdown.length, 1);
	assert.equal(results.biasByBucket.reduce((sum, row) => sum + row.count, 0), 2);
});
