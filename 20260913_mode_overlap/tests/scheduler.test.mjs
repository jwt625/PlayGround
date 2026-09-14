import test from 'node:test';
import assert from 'node:assert/strict';
import { scheduleQuestions, BUCKET_TARGET } from '../src/lib/session-core.ts';
import { analyticQuestion, balancedAlignmentBank } from '../src/scenarios/generator.mjs';

function anchor(id, category) {
	return analyticQuestion({ id, title: id, category, a: { mfdXUm: 4, mfdYUm: 4 }, b: { mfdXUm: 4, mfdYUm: 4 } });
}

test('schedule fills G008 bucket quotas deterministically', () => {
	const bank = balancedAlignmentBank(3, 6);
	const first = scheduleQuestions(bank, { seed: 11, length: 20 });
	const second = scheduleQuestions(bank, { seed: 11, length: 20 });
	assert.deepEqual(first.questions.map((q) => q.id), second.questions.map((q) => q.id));
	assert.deepEqual(first.bucketCounts, BUCKET_TARGET);
	assert.deepEqual(first.shortages, []);
	assert.equal(first.questions.length, 20);
	assert.equal(new Set(first.questions.map((q) => q.id)).size, 20);
});

test('category caps are enforced and reported without starving the session', () => {
	const pool = balancedAlignmentBank(5, 6);
	// Force every question into one category so a cap of one would starve.
	const single = pool.map((q, i) => ({ ...q, category: 'gaussian-alignment', id: `${i}` }));
	const report = scheduleQuestions(single, { seed: 2, length: 20, maxPerCategory: { 'gaussian-alignment': 1 } });
	assert.equal(report.questions.length, 20, 'anti-starvation relaxation must still fill the session');
	assert.ok(report.rejected > 0, 'the cap must reject during the quota pass');
});

test('category caps hold when the pool has enough alternatives', () => {
	const pool = [
		...balancedAlignmentBank(9, 6),
		...Array.from({ length: 40 }, (_, i) => anchor(`a${i}`, 'gaussian-anchor'))
	];
	const report = scheduleQuestions(pool, { seed: 4, length: 20, maxPerCategory: { 'gaussian-anchor': 2 } });
	assert.ok(report.categoryCounts['gaussian-anchor'] <= 2);
	assert.equal(report.questions.length, 20);
});

test('shortages are reported per bucket when the pool is thin', () => {
	const pool = Array.from({ length: 5 }, (_, i) => anchor(`only${i}`, 'gaussian-anchor'));
	const report = scheduleQuestions(pool, { seed: 1, length: 20 });
	assert.ok(report.shortages.length >= 1);
	assert.ok(report.shortages.some((entry) => entry.available === 0));
	assert.equal(report.questions.length, 5);
});
