import {
	aggregate,
	buildContinuousAnswer,
	buildIntervalAnswer,
	scheduleQuestions,
	selectQuestions,
	SPEEDRUN_LENGTH
} from './session-core';
import type {
	AnswerMode,
	AnswerRecord,
	CachedQuestion,
	SessionConfig,
	SessionPhase,
	SessionResults
} from './types';

export interface SessionInit {
	pool: CachedQuestion[];
	answerMode: AnswerMode;
	limit: number | null;
	category: string | null;
	seed: number;
	maxPerCategory?: Record<string, number>;
}

const noopNow = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/**
 * Session state machine outside Svelte components. Runes keep it reactive while
 * the scoring, selection and aggregation rules live in the pure core module.
 * Submission is one-shot: a question accepts exactly one answer before reveal.
 */
export class Session {
	config = $state<SessionConfig>({
		answerMode: 'continuous',
		questionCount: SPEEDRUN_LENGTH,
		category: null,
		seed: 0,
		limit: SPEEDRUN_LENGTH
	});
	phase = $state<SessionPhase>('idle');
	index = $state(0);
	questions = $state<CachedQuestion[]>([]);
	records = $state<AnswerRecord[]>([]);
	questionStartedAt = $state(0);
	shortages = $state<{ bucket: number; requested: number; available: number }[]>([]);

	#pool: CachedQuestion[] = [];
	#submitted = false;
	#now: () => number;

	constructor(now: () => number = noopNow) {
		this.#now = now;
	}

	get current(): CachedQuestion | undefined {
		return this.questions[this.index];
	}

	get isLast(): boolean {
		return this.index >= this.questions.length - 1;
	}

	get progressLabel(): string {
		return `${Math.min(this.index + 1, this.questions.length)}/${this.questions.length}`;
	}

	start(init: SessionInit): void {
		this.#pool = init.pool;
		this.config = {
			answerMode: init.answerMode,
			questionCount: init.limit ?? this.#pool.length,
			category: init.category,
			seed: init.seed,
			limit: init.limit,
			maxPerCategory: init.maxPerCategory
		};
		const length = init.limit ?? this.#pool.length;
		const selection = scheduleQuestions(this.#pool, {
			seed: init.seed,
			length,
			maxPerCategory: init.maxPerCategory
		});
		this.questions = selection.questions;
		this.shortages = selection.shortages.map((entry) => ({
			bucket: entry.bucket,
			requested: entry.requested,
			available: entry.available
		}));
		this.records = [];
		this.index = 0;
		this.#submitted = false;
		this.questionStartedAt = this.#now();
		this.phase = this.questions.length > 0 ? 'playing' : 'results';
	}

	/** Restart with the same configuration and a fresh seed. */
	restart(seed = (this.config.seed + 1) >>> 0): void {
		this.start({
			pool: this.#pool,
			answerMode: this.config.answerMode,
			limit: this.config.limit,
			category: this.config.category,
			seed,
			maxPerCategory: this.config.maxPerCategory
		});
	}

	submitContinuous(guessEta: number): AnswerRecord {
		const question = this.#requirePlaying();
		const record = buildContinuousAnswer({
			question,
			mode: 'continuous',
			guessEta,
			durationMs: this.#duration()
		});
		return this.#commit(record);
	}

	submitInterval(intervalIndex: number): AnswerRecord {
		const question = this.#requirePlaying();
		const record = buildIntervalAnswer({
			question,
			mode: 'interval',
			intervalIndex,
			durationMs: this.#duration()
		});
		return this.#commit(record);
	}

	next(): void {
		if (this.phase !== 'revealed') return;
		if (this.isLast) {
			if (this.config.limit !== null) {
				this.phase = 'results';
				return;
			}
			this.#refill();
			if (this.questions.length === this.index + 1) {
				this.phase = 'results';
				return;
			}
		}
		this.index += 1;
		this.#submitted = false;
		this.questionStartedAt = this.#now();
		this.phase = 'playing';
	}

	get results(): SessionResults {
		return aggregate(this.records);
	}

	/** End an unlimited practice session early to review statistics. */
	finish(): void {
		if (this.records.length > 0 && (this.phase === 'playing' || this.phase === 'revealed')) {
			this.phase = 'results';
		}
	}

	elapsedMs(): number {
		return this.phase === 'playing' ? this.#now() - this.questionStartedAt : 0;
	}

	#duration(): number {
		return Math.max(0, this.#now() - this.questionStartedAt);
	}

	#requirePlaying(): CachedQuestion {
		if (this.phase !== 'playing') throw new Error('No question is accepting an answer');
		if (this.#submitted) throw new Error('This question was already submitted once');
		const question = this.current;
		if (!question) throw new Error('No current question');
		return question;
	}

	#commit(record: AnswerRecord): AnswerRecord {
		this.#submitted = true;
		this.records = [...this.records, record];
		this.phase = 'revealed';
		return record;
	}

	#refill(): void {
		const selection = selectQuestions(this.#pool, { seed: (this.config.seed + this.questions.length) >>> 0, limit: null });
		this.questions = [...this.questions, ...selection.questions];
	}
}
