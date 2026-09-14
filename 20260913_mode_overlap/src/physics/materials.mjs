/** Versioned isotropic material functions (C3.1).
 *
 * Every entry carries its supported wavelength window and a source note.
 * Values outside the window throw: a solver must never silently extrapolate.
 * Process-specific indices (SiN, LN) are explicitly flagged as placeholders
 * and are not eligible for released answers until K2.2 ties them to data.
 */
export const MATERIAL_DB_VERSION = 'materials-1';

function sellmeier(terms) {
	return (um) => {
		let n2 = 1;
		for (const [B, C] of terms) n2 += (B * um * um) / (um * um - C);
		return Math.sqrt(n2);
	};
}

export const MATERIALS = {
	air: {
		id: 'air',
		name: 'Air',
		kind: 'isotropic',
		status: 'reference',
		validityUm: [0.3, 2.0],
		source: 'n = 1 approximation over the near-IR game window',
		n: () => 1
	},
	sio2: {
		id: 'sio2',
		name: 'Fused silica (SiO2)',
		kind: 'isotropic',
		status: 'reference',
		validityUm: [0.21, 3.7],
		source: 'Malitson 1965 Sellmeier, fused silica',
		n: sellmeier([
			[0.6961663, 0.0684043 ** 2],
			[0.4079426, 0.1162414 ** 2],
			[0.8974794, 9.896161 ** 2]
		])
	},
	si: {
		id: 'si',
		name: 'Crystalline silicon',
		kind: 'isotropic',
		status: 'reference',
		validityUm: [1.1, 2.0],
		source: 'Salzberg & Villa 1957 room-temperature dispersion (n=3.475 at 1.55 µm)',
		n: (um) => Math.sqrt(11.6858 + 0.939816 / (um * um) + 0.000993358 / (um * um - 1.22567))
	},
	sin: {
		id: 'sin',
		name: 'Stoichiometric silicon nitride (placeholder)',
		kind: 'isotropic',
		status: 'placeholder-process-specific',
		validityUm: [1.5, 1.6],
		source: 'Process-dependent; constant n=2.00 placeholder pending K2.2 primary data',
		n: () => 2.0
	},
	ln: {
		id: 'ln',
		name: 'Lithium niobate (isotropic proxy, disabled)',
		kind: 'anisotropic-disabled',
		status: 'disabled-requires-tensor',
		validityUm: [0.4, 4.0],
		source: 'Isotropic proxy only; K2.5 requires crystal cut and tensor axes',
		n: () => 2.21
	}
};

export function getMaterial(id) {
	const material = MATERIALS[id];
	if (!material) throw new RangeError(`Unknown material: ${id}`);
	return material;
}

/** Refractive index with an explicit validity check; never extrapolates. */
export function refractiveIndex(id, wavelengthUm) {
	const material = getMaterial(id);
	if (!Number.isFinite(wavelengthUm) || wavelengthUm <= 0) {
		throw new RangeError('wavelengthUm must be finite and positive');
	}
	const [low, high] = material.validityUm;
	if (wavelengthUm < low || wavelengthUm > high) {
		throw new RangeError(`${id} is only valid for ${low}–${high} µm, got ${wavelengthUm} µm`);
	}
	const n = material.n(wavelengthUm);
	if (!Number.isFinite(n) || n <= 0) throw new RangeError(`Invalid index for ${id} at ${wavelengthUm} µm`);
	return n;
}

export function relativePermittivity(id, wavelengthUm) {
	const n = refractiveIndex(id, wavelengthUm);
	return n * n;
}

/** True when a material may participate in a released (non-placeholder) solve. */
export function isReleaseEligible(id) {
	const material = getMaterial(id);
	return material.status !== 'placeholder-process-specific' && material.status !== 'disabled-requires-tensor';
}
