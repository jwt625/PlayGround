import {gaussianOverlap, mismatchLoss, transformDefaults, GAUSSIAN_VERSION} from '../physics/gaussian.mjs';
export const GENERATOR_VERSION = 'generator-1';
export const BUCKET_EDGES = [0, .1, .3, .6, .8, .95, .99, 1];
export function bucketIndex(eta) {
  if (!Number.isFinite(eta) || eta < 0 || eta > 1) throw new RangeError('Invalid eta');
  return eta === 1 ? 6 : BUCKET_EDGES.findIndex((edge, i) => i < 7 && eta >= edge && eta < BUCKET_EDGES[i + 1]);
}
export function rng(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new RangeError('Seed must be uint32');
  let state = seed;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function triangular(random, low, mode, high) {
  if (![low, mode, high].every(Number.isFinite) || low > mode || mode > high) throw new RangeError('Invalid triangular bounds');
  if (low === high) return low;
  const u = random(), f = (mode - low) / (high - low);
  return u < f ? low + Math.sqrt(u * (high - low) * (mode - low)) : high - Math.sqrt((1 - u) * (high - low) * (high - mode));
}
export function samplePreset(preset, seed) {
  const random = rng(seed);
  for (let attempt = 0; attempt < 100; attempt++) {
    const parameters = {...preset.nominal};
    for (const [name, [low, high]] of Object.entries(preset.ranges)) parameters[name] = triangular(random, low, preset.nominal[name], high);
    if (parameters.slab_um !== undefined && parameters.slab_um >= parameters.height_um) continue;
    if (parameters.etch_um !== undefined && parameters.etch_um >= parameters.film_um) continue;
    if (parameters.outer_width_um !== undefined && (parameters.tip_width_um >= parameters.outer_width_um || parameters.height_um >= parameters.outer_height_um)) continue;
    return {presetId: preset.id, seed, generatorVersion: GENERATOR_VERSION, wavelengthUm: preset.wavelength_um, parameters, status: preset.model === 'scalar-gaussian' ? 'analytic-parameters' : 'pending-vector-solver'};
  }
  throw new Error(`Could not sample valid geometry: ${preset.id}`);
}
export function analyticQuestion({id, title, a, b, wavelengthUm = 1.55, transform = {}, seed = 0, category = 'gaussian-anchor'}) {
  rng(seed); // Validate seed even for deterministic anchors.
  const t = transformDefaults(transform), eta = gaussianOverlap(a, b, wavelengthUm, t);
  return {id, title, seed, category, status: 'ready-analytic', model: 'scalar-paraxial-gaussian-at-waist', generatorVersion: GENERATOR_VERSION, solverVersion: GAUSSIAN_VERSION, materialDBVersion: 'explicit-gap-index-1', meshSettings: {kind: 'analytic-no-mesh'}, wavelengthUm, a, b, transform: t, answer: {eta, ...mismatchLoss(eta)}, bucket: bucketIndex(eta)};
}
/**
 * On-the-fly mixed Gaussian bank: random aspect ratio, size mismatch, small
 * tilt and polarization, then an inverse-computed misalignment that lands the
 * requested eta bucket. Misalignment is bounded (normalized shift <= 2.5) so
 * questions stay "not too far off".
 */
export function balancedMixedBank(seed = 20260913, perBucket = 3) {
	if (!Number.isInteger(perBucket) || perBucket < 1 || perBucket > 1000) throw new RangeError('perBucket must be 1..1000');
	const random = rng(seed), questions = [];
	const kmax = 2.5;
	for (let bucket = 0; bucket < 7; bucket++) {
		let accepted = 0;
		for (let attempt = 0; accepted < perBucket && attempt < perBucket * 500; attempt++) {
			const low = Math.max(BUCKET_EDGES[bucket], Math.exp(-(kmax ** 2)));
			const high = BUCKET_EDGES[bucket + 1];
			const target = low + (high - low) * (0.002 + 0.996 * random());
			const a = { mfdXUm: 3 + 8 * random(), mfdYUm: 2.5 + 8 * random() };
			const rx = 1 + (random() * 2 - 1) * 0.35 * (1 - Math.sqrt(target));
			const ry = 1 + (random() * 2 - 1) * 0.35 * (1 - Math.sqrt(target));
			const b = { mfdXUm: a.mfdXUm * rx, mfdYUm: a.mfdYUm * ry };
			const thetaXRad = (random() * 2 - 1) * 0.04 * (1 - target);
			const thetaYRad = (random() * 2 - 1) * 0.04 * (1 - target);
			const polarizationRad = random() < 0.8 ? 0 : random() * 0.4;
			const base = gaussianOverlap(a, b, 1.55, { thetaXRad, thetaYRad, polarizationRad });
			if (!(base > target)) continue;
			const k = Math.sqrt(-Math.log(target / base));
			if (!(k <= kmax)) continue;
			const direction = 2 * Math.PI * random();
			const sx = Math.sqrt((a.mfdXUm ** 2 + b.mfdXUm ** 2) / 2) / 2;
			const sy = Math.sqrt((a.mfdYUm ** 2 + b.mfdYUm ** 2) / 2) / 2;
			const dxUm = k * sx * Math.cos(direction);
			const dyUm = k * sy * Math.sin(direction);
			const question = analyticQuestion({
				id: `mixed-${bucket}-${accepted}`,
				title: 'Gaussian mode matching',
				category: 'gaussian-mixed',
				a,
				b,
				transform: { dxUm, dyUm, thetaXRad, thetaYRad, polarizationRad },
				seed
			});
			if (question.bucket !== bucket) continue;
			questions.push(question);
			accepted++;
		}
		if (accepted !== perBucket) throw new Error(`Unfilled mixed bucket ${bucket}: ${accepted}/${perBucket}`);
	}
	return questions;
}

/** Balanced diagnostic bank. Target eta is converted to a bounded normalized shift.
 * Currently alignment-only; anchors add other effects. Not a full game scheduler.
 */
export function balancedAlignmentBank(seed = 20260913, perBucket = 20) {
  if (!Number.isInteger(perBucket) || perBucket < 1 || perBucket > 10000) throw new RangeError('perBucket must be 1..10000');
  const random = rng(seed), questions = [];
  for (let bucket = 0; bucket < 7; bucket++) {
    let accepted = 0;
    for (let attempt = 0; accepted < perBucket && attempt < perBucket * 100; attempt++) {
      const low = Math.max(BUCKET_EDGES[bucket], Math.exp(-(2.5 ** 2)));
      const high = BUCKET_EDGES[bucket + 1];
      const target = low + (high - low) * (.001 + .998 * random());
      const a = {mfdXUm: 2 + 10 * random(), mfdYUm: 2 + 10 * random()};
      const radius = Math.sqrt(-Math.log(target)), direction = 2 * Math.PI * random();
      const q = analyticQuestion({id: `alignment-${bucket}-${accepted}`, title: 'Elliptical mode alignment', seed, category: 'gaussian-alignment', a, b: {...a}, transform: {dxUm: radius * a.mfdXUm / 2 * Math.cos(direction), dyUm: radius * a.mfdYUm / 2 * Math.sin(direction)}});
      if (q.bucket !== bucket) continue;
      questions.push(q); accepted++;
    }
    if (accepted !== perBucket) throw new Error(`Unfilled bucket ${bucket}: ${accepted}/${perBucket}`);
  }
  return questions;
}
