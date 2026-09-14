/** Warm-start provenance gate (C6.6).
 *
 * A cached numerical mode may only be imported when its solver, material,
 * mesh and wavelength provenance match the current request. Either the whole
 * bundle validates or it is rejected with explicit reasons; missing metadata
 * never counts as a match.
 */
export const PROVENANCE_VERSION = 'provenance-1';

const REQUIRED = ['solverVersion', 'materialDBVersion', 'meshVersion'];

export function validateModeBundle(record, expected = {}) {
	const reasons = [];
	if (!record || typeof record !== 'object') {
		return { ok: false, reasons: ['missing mode record'], provenanceVersion: PROVENANCE_VERSION };
	}
	for (const key of REQUIRED) {
		if (expected[key] === undefined) continue;
		if (record[key] !== expected[key]) reasons.push(`${key} mismatch: ${record[key]} != ${expected[key]}`);
	}
	if (expected.wavelengthUm !== undefined) {
		if (typeof record.wavelengthUm !== 'number' || Math.abs(record.wavelengthUm - expected.wavelengthUm) > 1e-12) {
			reasons.push('wavelength mismatch');
		}
	}
	if (!Number.isFinite(record.neff)) reasons.push('missing finite neff');
	if (record.power !== undefined && !(record.power > 0)) reasons.push('non-positive forward power');
	return { ok: reasons.length === 0, reasons, provenanceVersion: PROVENANCE_VERSION };
}

export function eligibleForWarmStart(record, expected) {
	return validateModeBundle(record, expected).ok;
}
