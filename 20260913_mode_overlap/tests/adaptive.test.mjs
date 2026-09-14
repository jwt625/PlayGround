import test from 'node:test';
import assert from 'node:assert/strict';
import {
	adaptBucketTarget,
	clampDifficulty,
	createAdaptiveProfile,
	recommendedDifficulty,
	updateAdaptiveProfile
} from '../src/lib/adaptive.ts';
import { BUCKET_TARGET } from '../src/lib/session-core.ts';

function answer(category, absEtaError, bucket) {
	return { category, absEtaError, bucket };
}

test('difficulty is clamped so unvalidated vector levels are never selected', () => {
	assert.equal(clampDifficulty(1).difficulty, 1);
	assert.equal(clampDifficulty(2).difficulty, 2);
	const decision = clampDifficulty(4);
	assert.equal(decision.difficulty, 2);
	assert.equal(decision.clamped, true);
	assert.match(decision.reason, /validated vector solver/);
});

test('poor performance recommends level 1 and good performance level 2', () => {
	const poor = updateAdaptiveProfile(createAdaptiveProfile(), [
		answer('gaussian-alignment', 0.5, 0),
		answer('gaussian-alignment', 0.4, 1)
	]);
	assert.equal(poor.difficulty, 1);
	assert.equal(recommendedDifficulty(poor), 1);

	const good = updateAdaptiveProfile(createAdaptiveProfile(), [
		answer('gaussian-alignment', 0.01, 0),
		answer('gaussian-anchor', 0.02, 3)
	]);
	assert.equal(good.difficulty, 2);
	assert.equal(good.runs, 1);
});

test('profile updates are deterministic and bucket counts accumulate', () => {
	const base = createAdaptiveProfile();
	const first = updateAdaptiveProfile(base, [answer('a', 0.1, 2), answer('a', 0.2, 2)]);
	const second = updateAdaptiveProfile(base, [answer('a', 0.1, 2), answer('a', 0.2, 2)]);
	assert.deepEqual(first, second);
	assert.equal(first.bucketCount[2], 2);
	assert.equal(first.categoryCount.a, 2);
});

test('adaptive bucket targets preserve the total and chase under-sampled buckets', () => {
	const profile = createAdaptiveProfile();
	profile.bucketCount = [50, 0, 0, 0, 0, 0, 0];
	const adapted = adaptBucketTarget(profile, BUCKET_TARGET, 2);
	assert.equal(adapted.reduce((sum, value) => sum + value, 0), BUCKET_TARGET.reduce((sum, value) => sum + value, 0));
	assert.ok(adapted[1] > BUCKET_TARGET[1]);
	assert.ok(adapted[0] < BUCKET_TARGET[0]);
});
