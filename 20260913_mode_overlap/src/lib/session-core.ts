import { rng } from '../scenarios/generator.mjs';
import {
	scoreContinuous,
	scoreInterval,
	intervalForEta,
	intervalMidpoint,
	scoringLossDB
} from './scoring.ts';
import type { AnswerMode, AnswerRecord, CachedQuestion, SessionResults } from './types';

/** G008 target bucket counts for the canonical 20-question speedrun. */
export const BUCKET_TARGET = [2, 2, 3, 3, 4, 3, 3];
export const SPEEDRUN_LENGTH = 20;

export interface SelectOptions {
	seed: number;
	/** null = unlimited practice. */
	limit: number | null;
}

export interface SelectionResult {
	questions: CachedQuestion[];
	shortages: { bucket: number; requested: number; available: number }[];
}

function shuffle<T>(items: T[], random: () => number): T[] {
	const out = [...items];
	for (let i = out.length - 1; i > 0; i--) {
		const j = Math.floor(random() * (i + 1));
		[out[i], out[j]] = [out[j], out[i]];
	}
	return out;
}

function allocate(limit: number): number[] {
	if (limit === SPEEDRUN_LENGTH) return [...BUCKET_TARGET];
	const raw = BUCKET_TARGET.map((target) => (limit * target) / SPEEDRUN_LENGTH);
	const base = raw.map(Math.floor);
	let remaining = limit - base.reduce((sum, value) => sum + value, 0);
	const order = raw
		.map((value, index) => ({ index, fraction: value - Math.floor(value) }))
		.sort((a, b) => b.fraction - a.fraction);
	for (let i = 0; remaining > 0; i++, remaining--) base[order[i % order.length].index]++;
	return base;
}

/**
 * Deterministic, quota-aware selection. Draws each bucket without replacement
 * and reports shortages instead of silently over-drawing one bucket.
 */
export function selectQuestions(pool: CachedQuestion[], options: SelectOptions): SelectionResult {
	const random = rng(options.seed >>> 0);
	if (options.limit === null) return { questions: shuffle(pool, random), shortages: [] };
	const byBucket: CachedQuestion[][] = Array.from({ length: BUCKET_TARGET.length }, () => []);
	for (const question of shuffle(pool, random)) byBucket[question.bucket]?.push(question);
	const target = allocate(options.limit);
	const questions: CachedQuestion[] = [];
	const shortages: SelectionResult['shortages'] = [];
	for (let bucket = 0; bucket < target.length; bucket++) {
		const available = byBucket[bucket];
		const take = Math.min(target[bucket], available.length);
		if (take < target[bucket]) shortages.push({ bucket, requested: target[bucket], available: available.length });
		questions.push(...available.slice(0, take));
	}
	const picked = new Set(questions.map((question) => question.id));
	const leftover = pool.filter((question) => !picked.has(question.id));
	while (questions.length < options.limit && leftover.length > 0) {
		questions.push(leftover.shift() as CachedQuestion);
	}
	return { questions: shuffle(questions, random), shortages };
}

export interface BuildAnswerInput {
	question: CachedQuestion;
	mode: Extract<AnswerMode, 'continuous'>;
	guessEta: number;
	durationMs: number;
}

export interface BuildIntervalAnswerInput {
	question: CachedQuestion;
	mode: Extract<AnswerMode, 'interval'>;
	intervalIndex: number;
	durationMs: number;
}

/**
 * One-shot answer construction. The guess is immutable once built; the caller
 * is responsible for refusing a second submission.
 */
export function buildContinuousAnswer(input: BuildAnswerInput): AnswerRecord {
	const { question, guessEta, durationMs } = input;
	const guess = Math.min(1, Math.max(0, guessEta));
	return finishAnswer(question, 'continuous', guess, scoreContinuous(guess, question.answer.eta), durationMs);
}

export function buildIntervalAnswer(input: BuildIntervalAnswerInput): AnswerRecord {
	const { question, intervalIndex, durationMs } = input;
	if (!Number.isInteger(intervalIndex) || intervalIndex < 0 || intervalIndex > 5) {
		throw new RangeError('intervalIndex must be 0..5');
	}
	const representative = intervalMidpoint(intervalIndex);
	return finishAnswer(
		question,
		'interval',
		representative,
		scoreInterval(intervalIndex, question.answer.eta),
		durationMs
	);
}

function finishAnswer(
	question: CachedQuestion,
	mode: AnswerMode,
	guessEta: number,
	score: number,
	durationMs: number
): AnswerRecord {
	const trueEta = question.answer.eta;
	const guessLossDB = scoringLossDB(guessEta);
	const trueLossDB = scoringLossDB(trueEta);
	const signedEtaError = guessEta - trueEta;
	const signedDbError = guessLossDB - trueLossDB;
	return {
		questionId: question.id,
		category: question.category,
		bucket: question.bucket,
		mode,
		guessEta,
		trueEta,
		guessLossDB,
		trueLossDB,
		signedEtaError,
		absEtaError: Math.abs(signedEtaError),
		signedDbError,
		absDbError: Math.abs(signedDbError),
		score,
		durationMs
	};
}

function mean(values: number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function aggregate(records: AnswerRecord[]): SessionResults {
	const categoryMap = new Map<string, AnswerRecord[]>();
	const bucketMap = new Map<number, AnswerRecord[]>();
	for (const record of records) {
		if (!categoryMap.has(record.category)) categoryMap.set(record.category, []);
		categoryMap.get(record.category)?.push(record);
		if (!bucketMap.has(record.bucket)) bucketMap.set(record.bucket, []);
		bucketMap.get(record.bucket)?.push(record);
	}
	return {
		count: records.length,
		score: mean(records.map((record) => record.score)),
		mae: mean(records.map((record) => record.absEtaError)),
		medianAbsEtaError: median(records.map((record) => record.absEtaError)),
		rmse: Math.sqrt(mean(records.map((record) => record.signedEtaError ** 2))),
		meanDbError: mean(records.map((record) => record.absDbError)),
		meanDurationMs: mean(records.map((record) => record.durationMs)),
		medianDurationMs: median(records.map((record) => record.durationMs)),
		categoryBreakdown: [...categoryMap.entries()]
			.map(([category, items]) => ({
				category,
				count: items.length,
				mae: mean(items.map((record) => record.absEtaError)),
				score: mean(items.map((record) => record.score))
			}))
			.sort((a, b) => b.count - a.count),
		biasByBucket: [...bucketMap.entries()]
			.map(([bucket, items]) => ({
				bucket,
				count: items.length,
				signedBias: mean(items.map((record) => record.signedEtaError))
			}))
			.sort((a, b) => a.bucket - b.bucket),
		answers: records
	};
}

export function representativeIntervalForEta(eta: number): number {
	return intervalForEta(eta);
}

export interface ScheduleOptions {
	seed: number;
	length: number;
	bucketTarget?: number[];
	maxPerCategory?: Record<string, number>;
}

export interface ScheduleReport {
	questions: CachedQuestion[];
	shortages: { bucket: number; requested: number; available: number; unfilled: number }[];
	bucketCounts: number[];
	categoryCounts: Record<string, number>;
	rejected: number;
	attempts: number;
	coverageRule: string;
}

/**
 * C4.2 constraint-aware draw: per-bucket quotas, optional category caps, a
 * bounded attempt budget and an explicit shortage/coverage report. Quotas are
 * relaxed only in a final pass so that a saturated category cap can never
 * starve the session.
 */
export function scheduleQuestions(pool: CachedQuestion[], options: ScheduleOptions): ScheduleReport {
	const random = rng(options.seed >>> 0);
	const target = options.bucketTarget ?? BUCKET_TARGET;
	const maxPerCategory = options.maxPerCategory ?? {};
	const byBucket: CachedQuestion[][] = Array.from({ length: target.length }, () => []);
	for (const question of shuffle(pool, random)) byBucket[question.bucket]?.push(question);
	const chosen: CachedQuestion[] = [];
	const picked = new Set<string>();
	const categoryCounts: Record<string, number> = {};
	const canTake = (question: CachedQuestion) => {
		const cap = maxPerCategory[question.category];
		return cap === undefined || (categoryCounts[question.category] ?? 0) < cap;
	};
	const take = (question: CachedQuestion) => {
		chosen.push(question);
		picked.add(question.id);
		categoryCounts[question.category] = (categoryCounts[question.category] ?? 0) + 1;
	};
	let rejected = 0;
	let attempts = 0;
	const shortages: ScheduleReport['shortages'] = [];
	for (let bucket = 0; bucket < target.length; bucket++) {
		let need = target[bucket];
		for (const question of byBucket[bucket]) {
			if (need <= 0) break;
			attempts++;
			if (canTake(question)) {
				take(question);
				need--;
			} else {
				rejected++;
			}
		}
		if (need > 0) shortages.push({ bucket, requested: target[bucket], available: byBucket[bucket].length, unfilled: need });
	}
	const remaining = pool.filter((question) => !picked.has(question.id));
	for (const question of remaining) {
		if (chosen.length >= options.length) break;
		attempts++;
		if (canTake(question)) take(question);
		else rejected++;
	}
	if (chosen.length < options.length) {
		for (const question of remaining) {
			if (chosen.length >= options.length) break;
			if (!picked.has(question.id)) take(question);
		}
	}
	const bucketCounts = Array(target.length).fill(0);
	for (const question of chosen) bucketCounts[question.bucket]++;
	return {
		questions: shuffle(chosen, random),
		shortages,
		bucketCounts,
		categoryCounts,
		rejected,
		attempts,
		coverageRule: 'G008 bucket quotas with category caps and final anti-starvation relaxation'
	};
}
