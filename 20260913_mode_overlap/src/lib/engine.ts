/** Frontend bridge to the shared foundation modules.
 *
 * The physics, generator and key derivation live in `src/physics`,
 * `src/scenarios` and `src/storage`, which are plain browser-compatible ESM.
 * They are the single source of truth for Node scripts, tests and the UI.
 */
export {
	gaussianOverlap,
	gaussianField,
	mismatchLoss,
	transformDefaults,
	validateMode,
	GAUSSIAN_VERSION
} from '$foundation/physics/gaussian.mjs';

export {
	bucketIndex,
	BUCKET_EDGES,
	GENERATOR_VERSION,
	rng,
	triangular,
	samplePreset,
	analyticQuestion,
	balancedAlignmentBank,
	balancedMixedBank
} from '$foundation/scenarios/generator.mjs';

export {
	canonicalJSON,
	contentKey,
	modeKey,
	pairKey,
	CACHE_VERSION
} from '$foundation/storage/cache.mjs';

export type {
	GaussianMode,
	GaussianTransform,
	ResolvedTransform,
	MismatchLoss
} from '$foundation/physics/gaussian.mjs';

/** Physical mismatch loss in dB. eta = 0 is an exact zero overlap, not a floor. */
export function mismatchLossDB(eta: number): number {
	if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new RangeError('eta must be in [0,1]');
	return eta === 0 ? Infinity : -10 * Math.log10(eta);
}

export function formatEta(eta: number, digits = 3): string {
	return eta.toFixed(digits);
}

export function formatLossDB(lossDB: number, digits = 3): string {
	return Number.isFinite(lossDB) ? `${lossDB.toFixed(digits)} dB` : 'inf dB';
}
