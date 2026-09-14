import test from 'node:test';
import assert from 'node:assert/strict';
import {
	DB_FLOOR,
	INTERVALS,
	intervalContains,
	intervalForEta,
	intervalMidpoint,
	scoreContinuous,
	scoreInterval,
	scoreRanking,
	scoringLossDB
} from '../src/lib/scoring.ts';
import { mismatchLoss } from '../src/physics/gaussian.mjs';

test('interval boundaries respect the inclusive upper edge of the last bin', () => {
	for (const [eta, index] of [
		[0, 0],
		[0.099999, 0],
		[0.1, 1],
		[0.3, 2],
		[0.6, 3],
		[0.8, 4],
		[0.95, 5],
		[0.99, 5],
		[1, 5]
	]) {
		assert.equal(intervalForEta(eta), index, `eta=${eta}`);
	}
	assert.ok(intervalContains(0, 0));
	assert.ok(!intervalContains(0, 0.1));
	assert.ok(intervalContains(1, 0.1));
	assert.ok(intervalContains(5, 0.95));
	assert.ok(intervalContains(5, 1));
	assert.ok(!intervalContains(5, 0.949));
	assert.equal(INTERVALS.length, 6);
});

test('exact zero overlap uses the 60 dB scoring floor, never as a physical answer', () => {
	assert.equal(DB_FLOOR, 60);
	assert.equal(scoringLossDB(0), 60);
	assert.ok(scoringLossDB(1) === 0);
	const physical = mismatchLoss(0);
	assert.equal(physical.lossDB, null);
	assert.ok(physical.lossIsInfinite, 'physical zero overlap must remain infinite');
});

test('continuous scoring is maximal at the true value and decays in dB', () => {
	assert.equal(scoreContinuous(1, 1), 100);
	assert.equal(scoreContinuous(0, 0), 100);
	const near = scoreContinuous(0.9, 0.95);
	const far = scoreContinuous(0.5, 0.95);
	assert.ok(near > far);
	assert.ok(near <= 100 && far >= 0);
	assert.equal(scoreContinuous(2, 1), scoreContinuous(1, 1));
	assert.equal(scoreContinuous(-1, 0), scoreContinuous(0, 0));
});

test('interval scoring is zero distance inside and dB distance outside', () => {
	for (let index = 0; index < INTERVALS.length; index++) {
		assert.equal(scoreInterval(index, intervalMidpoint(index)), 100);
	}
	assert.equal(scoreInterval(5, 1), 100);
	// True eta 0.2 (bucket 1) against guessed bucket 0: distance to 0.1 edge.
	const score = scoreInterval(0, 0.2);
	const expected = 100 * Math.exp(-Math.abs(scoringLossDB(0.1) - scoringLossDB(0.2)) / 3);
	assert.ok(Math.abs(score - expected) < 1e-9);
});

test('ranking excludes sub-threshold ties and scores pairwise order', () => {
	const etas = [0.9, 0.5, 0.2];
	assert.equal(scoreRanking([0, 1, 2], etas).score, 100);
	assert.equal(scoreRanking([2, 1, 0], etas).score, 0);
	const tied = [0.5, 0.5001, 0.2];
	const result = scoreRanking([0, 1, 2], tied);
	assert.equal(result.informativePairs, 2);
	assert.equal(result.excludedPairs, 1);
	assert.equal(result.score, 100);
	const allTied = scoreRanking([0, 1], [0.5, 0.5001]);
	assert.equal(allTied.informativePairs, 0);
	assert.equal(allTied.score, 100);
});
