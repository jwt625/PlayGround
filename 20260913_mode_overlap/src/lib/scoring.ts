/** Transparent G003 scoring.
 *
 * Physical answers keep exact zero overlap as infinite dB. Scoring only uses
 * the explicit 1e-6 (60 dB) floor for finite arithmetic; the floor is never
 * substituted for the displayed physical answer.
 */
export const SCORE_SCALE_DB = 3;
export const SCORING_EPSILON = 1e-6;
export const DB_FLOOR = -10 * Math.log10(SCORING_EPSILON); // 60 dB
export const RANK_TIE_THRESHOLD = 0.005;

export interface Interval {
	index: number;
	low: number;
	high: number;
	lowInclusive: boolean;
	highInclusive: boolean;
	label: string;
}

export const INTERVALS: Interval[] = [
	{ index: 0, low: 0, high: 0.1, lowInclusive: true, highInclusive: false, label: '[0, 0.1)' },
	{ index: 1, low: 0.1, high: 0.3, lowInclusive: true, highInclusive: false, label: '[0.1, 0.3)' },
	{ index: 2, low: 0.3, high: 0.6, lowInclusive: true, highInclusive: false, label: '[0.3, 0.6)' },
	{ index: 3, low: 0.6, high: 0.8, lowInclusive: true, highInclusive: false, label: '[0.6, 0.8)' },
	{ index: 4, low: 0.8, high: 0.95, lowInclusive: true, highInclusive: false, label: '[0.8, 0.95)' },
	{ index: 5, low: 0.95, high: 1, lowInclusive: true, highInclusive: true, label: '[0.95, 1]' }
];

/** Finite dB value used for scoring; exact zero maps to the 60 dB floor. */
export function scoringLossDB(eta: number): number {
	if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new RangeError('eta must be in [0,1]');
	const db = -10 * Math.log10(Math.max(eta, SCORING_EPSILON));
	return db === 0 ? 0 : db;
}

export function scoreFromLossDB(guessLossDB: number, trueLossDB: number): number {
	return 100 * Math.exp(-Math.abs(guessLossDB - trueLossDB) / SCORE_SCALE_DB);
}

export function scoreContinuous(guessEta: number, trueEta: number): number {
	if (!Number.isFinite(guessEta)) throw new RangeError('guess must be finite');
	const guess = Math.min(1, Math.max(0, guessEta));
	return scoreFromLossDB(scoringLossDB(guess), scoringLossDB(trueEta));
}

export function intervalForEta(eta: number): number {
	if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new RangeError('eta must be in [0,1]');
	if (eta === 1) return INTERVALS.length - 1;
	return INTERVALS.findIndex(
		(interval) => eta >= interval.low && (eta < interval.high || (interval.highInclusive && eta <= interval.high))
	);
}

export function intervalContains(index: number, eta: number): boolean {
	const interval = INTERVALS[index];
	if (!interval) throw new RangeError('unknown interval');
	if (eta < interval.low || eta > interval.high) return false;
	if (eta === interval.low) return interval.lowInclusive;
	if (eta === interval.high) return interval.highInclusive;
	return true;
}

/** Zero when the true eta lies in the chosen interval; otherwise dB distance to it. */
export function scoreInterval(guessIndex: number, trueEta: number): number {
	if (!INTERVALS[guessIndex]) throw new RangeError('unknown interval');
	if (intervalContains(guessIndex, trueEta)) return 100;
	const interval = INTERVALS[guessIndex];
	const lowLoss = scoringLossDB(interval.high); // lower loss edge (vice versa)
	const highLoss = scoringLossDB(interval.low);
	const trueLoss = scoringLossDB(trueEta);
	if (trueLoss < lowLoss) return scoreFromLossDB(lowLoss, trueLoss);
	if (trueLoss > highLoss) return scoreFromLossDB(highLoss, trueLoss);
	return 100;
}

export function intervalMidpoint(index: number): number {
	const interval = INTERVALS[index];
	if (!interval) throw new RangeError('unknown interval');
	return (interval.low + interval.high) / 2;
}

export interface RankingResult {
	score: number;
	correctPairs: number;
	informativePairs: number;
	excludedPairs: number;
}

/**
 * Normalized pairwise correctness. Excludes pairs whose true eta differ by
 * less than RANK_TIE_THRESHOLD (treated as ties) and accepts tied guesses.
 */
export function scoreRanking(userOrder: number[], trueEtas: number[]): RankingResult {
	if (userOrder.length !== trueEtas.length) throw new RangeError('order and etas must match');
	const position = new Array(userOrder.length);
	userOrder.forEach((item, pos) => (position[item] = pos));
	let correctPairs = 0;
	let informativePairs = 0;
	let excludedPairs = 0;
	for (let i = 0; i < trueEtas.length; i++) {
		for (let j = i + 1; j < trueEtas.length; j++) {
			if (Math.abs(trueEtas[i] - trueEtas[j]) < RANK_TIE_THRESHOLD) {
				excludedPairs++;
				continue;
			}
			informativePairs++;
			const higherEtaFirst = position[i] < position[j];
			const iIsHigher = trueEtas[i] > trueEtas[j];
			if (higherEtaFirst === iIsHigher) correctPairs++;
		}
	}
	const score = informativePairs === 0 ? 100 : (100 * correctPairs) / informativePairs;
	return { score, correctPairs, informativePairs, excludedPairs };
}
