/** Common-grid mode overlap integral (C3.3).
 *
 * K002 forward-power and symmetrized flux conventions:
 *   P_A  = 0.5 Re ∫ (E_A x H_A*) · z dA
 *   C_AB = 0.25 ∫ [E_A x H_B* + E_B* x H_A] · z dA
 *   η    = |C_AB|² / (P_A P_B)
 *
 * Fields are sampled from each mode onto one common grid before integration, so
 * the result is independent of either solver mesh. η is reported raw: the
 * cross-impedance branch is not clamped to one.
 */
export const OVERLAP_VERSION = 'flux-overlap-1';
const UM = 1e-6;

function unionSorted(a, b) {
	const out = [];
	let i = 0;
	let j = 0;
	while (i < a.length || j < b.length) {
		let value;
		if (i >= a.length) value = b[j++];
		else if (j >= b.length) value = a[i++];
		else if (Math.abs(a[i] - b[j]) < 1e-12) {
			value = a[i];
			i++;
			j++;
		} else if (a[i] < b[j]) value = a[i++];
		else value = b[j++];
		if (out.length === 0 || Math.abs(out[out.length - 1] - value) > 1e-12) out.push(value);
	}
	return Float64Array.from(out);
}

function trapezoidWeightsMeters(xUm) {
	const n = xUm.length;
	const w = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const left = i > 0 ? xUm[i] - xUm[i - 1] : xUm[1] - xUm[0];
		const right = i < n - 1 ? xUm[i + 1] - xUm[i] : xUm[i] - xUm[i - 1];
		w[i] = (0.5 * (left + right)) * UM;
	}
	return w;
}

function sampleComplex(mode, component, xUm) {
	const grid = mode.mesh.xUm;
	const field = mode.fields[component];
	const n = grid.length;
	if (xUm <= grid[0]) return { re: field.re[0], im: field.im[0] };
	if (xUm >= grid[n - 1]) return { re: field.re[n - 1], im: field.im[n - 1] };
	let lo = 0;
	let hi = n - 1;
	while (hi - lo > 1) {
		const mid = (lo + hi) >> 1;
		if (grid[mid] <= xUm) lo = mid;
		else hi = mid;
	}
	const t = (xUm - grid[lo]) / (grid[hi] - grid[lo] || 1);
	return {
		re: field.re[lo] * (1 - t) + field.re[hi] * t,
		im: field.im[lo] * (1 - t) + field.im[hi] * t
	};
}

function conj(a) {
	return { re: a.re, im: -a.im };
}

function mulConj(a, b) {
	// a * conj(b)
	return { re: a.re * b.re + a.im * b.im, im: a.im * b.re - a.re * b.im };
}

function conjMul(a, b) {
	// conj(a) * b
	return { re: a.re * b.re + a.im * b.im, im: a.re * b.im - a.im * b.re };
}

function sub(a, b) {
	return { re: a.re - b.re, im: a.im - b.im };
}

function add(a, b) {
	return { re: a.re + b.re, im: a.im + b.im };
}

function scale(a, s) {
	return { re: a.re * s, im: a.im * s };
}

/** Complex (E x H*)·z for one mode. */
function sz(fields, x) {
	const ex = sampleComplex({ mesh: fields.mesh, fields: fields.fields }, 'Ex', x);
	const ey = sampleComplex({ mesh: fields.mesh, fields: fields.fields }, 'Ey', x);
	const hx = sampleComplex({ mesh: fields.mesh, fields: fields.fields }, 'Hx', x);
	const hy = sampleComplex({ mesh: fields.mesh, fields: fields.fields }, 'Hy', x);
	return sub(mulConj(ex, hy), mulConj(ey, hx));
}

function unionSortedArray(a, b) {
	return unionSorted(a, b);
}

function trapezoid2D(xUm, yUm) {
	const wx = trapezoidWeightsMeters(xUm);
	const wy = trapezoidWeightsMeters(yUm);
	const out = new Float64Array(xUm.length * yUm.length);
	for (let j = 0; j < yUm.length; j++) for (let i = 0; i < xUm.length; i++) out[j * xUm.length + i] = wx[i] * wy[j];
	return out;
}

function sampleBilinear(field, grid, x, y) {
	const { xUm, yUm, nx } = grid;
	const ny = yUm.length;
	const find = (edges, value) => {
		if (value <= edges[0]) return { lo: 0, hi: 0, t: 0 };
		if (value >= edges[edges.length - 1]) return { lo: edges.length - 1, hi: edges.length - 1, t: 0 };
		let lo = 0;
		let hi = edges.length - 1;
		while (hi - lo > 1) {
			const mid = (lo + hi) >> 1;
			if (edges[mid] <= value) lo = mid;
			else hi = mid;
		}
		return { lo, hi, t: (value - edges[lo]) / (edges[hi] - edges[lo] || 1) };
	};
	const ix = find(xUm, x);
	const iy = find(yUm, y);
	const sample = (re) => {
		const v00 = re[iy.lo * nx + ix.lo];
		const v10 = re[iy.lo * nx + ix.hi];
		const v01 = re[iy.hi * nx + ix.lo];
		const v11 = re[iy.hi * nx + ix.hi];
		return (v00 * (1 - ix.t) + v10 * ix.t) * (1 - iy.t) + (v01 * (1 - ix.t) + v11 * ix.t) * iy.t;
	};
	return { re: sample(field.re), im: sample(field.im) };
}

/**
 * Normalized overlap of two scalar 2D modes on a common grid:
 * η = |⟨ψ_A|ψ_B⟩|² / (⟨ψ_A|ψ_A⟩⟨ψ_B|ψ_B⟩). Used by the intermediate
 * scalar-2D backend; the flux overlap above is the release instrument.
 */
export function overlapScalarModes(modeA, modeB) {
	const commonX = unionSortedArray(modeA.mesh.xUm, modeB.mesh.xUm);
	const commonY = unionSortedArray(modeA.mesh.yUm, modeB.mesh.yUm);
	const weights = trapezoid2D(commonX, commonY);
	let normA = 0;
	let normB = 0;
	let cRe = 0;
	let cIm = 0;
	for (let j = 0; j < commonY.length; j++) {
		for (let i = 0; i < commonX.length; i++) {
			const w = weights[j * commonX.length + i];
			const a = sampleBilinear(modeA.scalar, modeA.mesh, commonX[i], commonY[j]);
			const b = sampleBilinear(modeB.scalar, modeB.mesh, commonX[i], commonY[j]);
			normA += w * (a.re * a.re + a.im * a.im);
			normB += w * (b.re * b.re + b.im * b.im);
			cRe += w * (a.re * b.re + a.im * b.im);
			cIm += w * (a.im * b.re - a.re * b.im);
		}
	}
	const eta = normA > 0 && normB > 0 ? (cRe * cRe + cIm * cIm) / (normA * normB) : null;
	return { overlapVersion: OVERLAP_VERSION, eta, cAB: { re: cRe, im: cIm }, normA, normB, positivePower: normA > 0 && normB > 0 };
}

/**
 * Flux overlap of two 1D-slab mode bundles on a common grid.
 * Returns the raw coefficient, normalized eta and the integrand for display.
 */
export function overlapModes(modeA, modeB) {
	const common = unionSorted(modeA.mesh.xUm, modeB.mesh.xUm);
	const weights = trapezoidWeightsMeters(common);
	const viewFor = (mode) => ({ mesh: mode.mesh, fields: mode.fields });
	const A = viewFor(modeA);
	const B = viewFor(modeB);
	let powerA = 0;
	let powerB = 0;
	let cRe = 0;
	let cIm = 0;
	const integrandRe = new Float64Array(common.length);
	const integrandIm = new Float64Array(common.length);
	for (let i = 0; i < common.length; i++) {
		const x = common[i];
		const w = weights[i];
		const szA = sz(A, x);
		const szB = sz(B, x);
		powerA += w * szA.re;
		powerB += w * szB.re;
		const exA = sampleComplex(A, 'Ex', x);
		const eyA = sampleComplex(A, 'Ey', x);
		const hxA = sampleComplex(A, 'Hx', x);
		const hyA = sampleComplex(A, 'Hy', x);
		const exB = sampleComplex(B, 'Ex', x);
		const eyB = sampleComplex(B, 'Ey', x);
		const hxB = sampleComplex(B, 'Hx', x);
		const hyB = sampleComplex(B, 'Hy', x);
		const t1 = sub(mulConj(exA, hyB), mulConj(eyA, hxB));
		const t2 = sub(conjMul(exB, hyA), conjMul(eyB, hxA));
		const c = scale(add(t1, t2), 0.25);
		integrandRe[i] = c.re;
		integrandIm[i] = c.im;
		cRe += w * c.re;
		cIm += w * c.im;
	}
	powerA *= 0.5;
	powerB *= 0.5;
	const valid = powerA > 0 && powerB > 0;
	const eta = valid ? (cRe * cRe + cIm * cIm) / (powerA * powerB) : null;
	return {
		overlapVersion: OVERLAP_VERSION,
		eta,
		cAB: { re: cRe, im: cIm },
		powerA,
		powerB,
		positivePower: valid,
		commonGridUm: common,
		integrand: { re: integrandRe, im: integrandIm }
	};
}
