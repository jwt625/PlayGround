/** C6.1 adaptive profile, difficulty clamp and campaign scheduling.
 *
 * Adaptation is deterministic and never selects unvalidated content: the
 * vector levels 3–5 stay clamped to 2 until C3 supplies released fields.
 */
import type { AnswerRecord } from './types';

export const ADAPTIVE_VERSION = 'adaptive-1';

export interface AdaptiveProfile {
	schemaVersion: string;
	runs: number;
	categoryMae: Record<string, number>;
	categoryCount: Record<string, number>;
	bucketCount: number[];
	difficulty: 1 | 2;
}

export function createAdaptiveProfile(): AdaptiveProfile {
	return {
		schemaVersion: ADAPTIVE_VERSION,
		runs: 0,
		categoryMae: {},
		categoryCount: {},
		bucketCount: Array(7).fill(0),
		difficulty: 2
	};
}

function runningMean(previous: number | undefined, count: number, value: number): number {
	if (!count || previous === undefined) return value;
	return previous + (value - previous) / count;
}

export function updateAdaptiveProfile(profile: AdaptiveProfile, answers: AnswerRecord[]): AdaptiveProfile {
	const next: AdaptiveProfile = {
		...profile,
		categoryMae: { ...profile.categoryMae },
		categoryCount: { ...profile.categoryCount },
		bucketCount: [...profile.bucketCount]
	};
	for (const answer of answers) {
		const count = (next.categoryCount[answer.category] ?? 0) + 1;
		next.categoryCount[answer.category] = count;
		next.categoryMae[answer.category] = runningMean(next.categoryMae[answer.category], count - 1, answer.absEtaError);
		if (answer.bucket >= 0 && answer.bucket < 7) next.bucketCount[answer.bucket]++;
	}
	next.runs++;
	next.difficulty = recommendedDifficulty(next);
	return next;
}

export function recommendedDifficulty(profile: AdaptiveProfile): 1 | 2 {
	const categories = Object.values(profile.categoryMae);
	if (categories.length === 0) return 2;
	const mean = categories.reduce((sum, value) => sum + value, 0) / categories.length;
	if (mean > 0.25) return 1;
	return 2;
}

export interface DifficultyDecision {
	difficulty: 1 | 2;
	requested: number;
	clamped: boolean;
	reason: string | null;
}

/** Guard against selecting unvalidated levels 3–5. */
export function clampDifficulty(requested: number): DifficultyDecision {
	if (requested <= 1) return { difficulty: 1, requested, clamped: false, reason: null };
	if (requested <= 2) return { difficulty: 2, requested, clamped: false, reason: null };
	return {
		difficulty: 2,
		requested,
		clamped: true,
		reason: `level ${requested} requires the validated vector solver (C3)`
	};
}

/**
 * Move a bounded number of questions from the most-sampled bucket to the
 * least-sampled one, preserving the total and never over-correcting.
 */
export function adaptBucketTarget(profile: AdaptiveProfile, base: number[], maxMoves = 2): number[] {
	const target = [...base];
	const counts = profile.bucketCount;
	for (let move = 0; move < maxMoves; move++) {
		let minIndex = -1;
		for (let i = 0; i < counts.length; i++) {
			if (target[i] >= 1 && (minIndex < 0 || counts[i] < counts[minIndex])) minIndex = i;
		}
		let maxIndex = -1;
		for (let i = 0; i < counts.length; i++) {
			if (i !== minIndex && (maxIndex < 0 || counts[i] > counts[maxIndex])) maxIndex = i;
		}
		if (minIndex < 0 || maxIndex < 0) break;
		if (counts[maxIndex] <= counts[minIndex] || target[maxIndex] <= 1) break;
		target[maxIndex]--;
		target[minIndex]++;
	}
	return target;
}
