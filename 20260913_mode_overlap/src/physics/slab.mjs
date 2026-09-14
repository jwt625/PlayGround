/** 1D slab full-vector finite-element solver (C3.2 first benchmark gate).
 *
 * TE and TM decouple exactly for a slab, so this is a genuine full-vector
 * benchmark against closed-form 3-layer dispersion. Operators are assembled in
 * Hermitian weak form on boundary-aligned elements, reduced by Dirichlet walls,
 * and solved for the largest β² via a Sturm-sequence bisection plus inverse
 * iteration. No third-party linear algebra.
 *
 * Returns normalized forward-power mode bundles; overlap is computed separately.
 */
import { relativePermittivity, MATERIAL_DB_VERSION } from './materials.mjs';
import { MESH_VERSION, buildSlabMesh } from './mesh.mjs';

export const SLAB_SOLVER_VERSION = 'slab-fem-1d-1';

const C0 = 299792458;
const MU0 = 4 * Math.PI * 1e-7;
const EPS0 = 1 / (MU0 * C0 * C0);
const UM = 1e-6;

/** Closed-form 3-layer slab dispersion; the external benchmark for the solver. */
export function analyticSlabModes({
	coreIndex,
	substrateIndex,
	claddingIndex,
	thicknessUm,
	wavelengthUm,
	polarization = 'TE',
	modeCount = 4
}) {
	const k0 = (2 * Math.PI) / (wavelengthUm * UM);
	const d = thicknessUm * UM;
	const nMaxClad = Math.max(substrateIndex, claddingIndex);
	if (coreIndex <= nMaxClad) return [];
	const kappa = (beta) => Math.sqrt(Math.max(0, k0 * k0 * coreIndex * coreIndex - beta * beta));
	const gammaOf = (beta, n) => Math.sqrt(Math.max(0, beta * beta - k0 * k0 * n * n));
	const ratio = (n) => (polarization === 'TM' ? (coreIndex * coreIndex) / (n * n) : 1);
	const equation = (beta) =>
		kappa(beta) * d -
		Math.atan((ratio(substrateIndex) * gammaOf(beta, substrateIndex)) / kappa(beta)) -
		Math.atan((ratio(claddingIndex) * gammaOf(beta, claddingIndex)) / kappa(beta));

	const modes = [];
	for (let m = 0; m < modeCount && m < 10; m++) {
		const target = m * Math.PI;
		const betaLow = k0 * nMaxClad + 1e-6;
		const betaHigh = k0 * coreIndex - 1e-6;
		let lo = betaLow;
		let hi = betaHigh;
		let flo = equation(lo) - target;
		let fhi = equation(hi) - target;
		if (flo * fhi > 0) break;
		for (let iter = 0; iter < 300; iter++) {
			const mid = (lo + hi) / 2;
			const fm = equation(mid) - target;
			if (fm > 0) {
				lo = mid;
				flo = fm;
			} else {
				hi = mid;
				fhi = fm;
			}
			if (hi - lo < 1e-12 * k0) break;
		}
		const beta = (lo + hi) / 2;
		modes.push({
			modeNumber: m,
			beta,
			neff: beta / k0,
			kappa: kappa(beta),
			gammaSubstrate: gammaOf(beta, substrateIndex),
			gammaCladding: gammaOf(beta, claddingIndex)
		});
	}
	return modes;
}

function thomasSolve(diag, off, rhs) {
	const n = diag.length;
	const c = new Float64Array(n - 1);
	const d = new Float64Array(n);
	let denom = diag[0] === 0 ? 1e-300 : diag[0];
	c[0] = off[0] / denom;
	d[0] = rhs[0] / denom;
	for (let i = 1; i < n; i++) {
		const pivot = diag[i] - off[i - 1] * c[i - 1];
		denom = pivot === 0 ? 1e-300 : pivot;
		c[i] = i < n - 1 ? off[i] / denom : 0;
		d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / denom;
	}
	const x = new Float64Array(n);
	x[n - 1] = d[n - 1];
	for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
	return x;
}

/** Number of eigenvalues strictly below sigma (Sturm sequence). */
export function sturmCountBelow(diag, off, sigma) {
	let count = 0;
	let d = diag[0] - sigma;
	if (d < 0) count++;
	for (let i = 1; i < diag.length; i++) {
		if (d === 0) d = 1e-300;
		d = diag[i] - sigma - (off[i - 1] * off[i - 1]) / d;
		if (d < 0) count++;
	}
	return count;
}

function gershgorin(diag, off) {
	let lo = Infinity;
	let hi = -Infinity;
	for (let i = 0; i < diag.length; i++) {
		const radius = (i > 0 ? Math.abs(off[i - 1]) : 0) + (i < off.length ? Math.abs(off[i]) : 0);
		lo = Math.min(lo, diag[i] - radius);
		hi = Math.max(hi, diag[i] + radius);
	}
	return { lo, hi };
}

/** k-th largest (k = 1 is the fundamental) eigenpair of a symmetric tridiagonal. */
export function tridiagonalEigenpair(diag, off, k = 1) {
	const n = diag.length;
	if (k < 1 || k > n) throw new RangeError('rank out of range');
	const { lo: gLo, hi: gHi } = gershgorin(diag, off);
	const target = n - k + 1;
	let lo = gLo - 1;
	let hi = gHi + 1;
	if (sturmCountBelow(diag, off, lo) >= target) throw new Error('lower bracket failed');
	if (sturmCountBelow(diag, off, hi) < target) throw new Error('upper bracket failed');
	for (let iter = 0; iter < 200; iter++) {
		const mid = (lo + hi) / 2;
		if (sturmCountBelow(diag, off, mid) >= target) hi = mid;
		else lo = mid;
		if (hi - lo < 1e-13 * (1 + Math.abs(hi))) break;
	}
	const lambda = (lo + hi) / 2;
	const mu = lambda + 1e-10 * (1 + Math.abs(lambda));
	const shifted = new Float64Array(n);
	for (let i = 0; i < n; i++) shifted[i] = diag[i] - mu;
	let v = new Float64Array(n).fill(1 / Math.sqrt(n));
	for (let iter = 0; iter < 8; iter++) {
		const w = thomasSolve(shifted, off, v);
		let norm = 0;
		for (const value of w) norm += value * value;
		norm = Math.sqrt(norm);
		if (!Number.isFinite(norm) || norm === 0) break;
		for (let i = 0; i < n; i++) v[i] = w[i] / norm;
	}
	let num = 0;
	let den = 0;
	for (let i = 0; i < n; i++) {
		let tv = diag[i] * v[i];
		if (i > 0) tv += off[i - 1] * v[i - 1];
		if (i < n - 1) tv += off[i] * v[i + 1];
		num += v[i] * tv;
		den += v[i] * v[i];
	}
	return { value: num / den, vector: v };
}

/** Assemble interior G (stiffness/mass) and B (mass) for a slab polarization. */
function assembleSlab(eps, lengthM, k0, polarization) {
	const elementCount = eps.length;
	const diagNode = new Float64Array(elementCount + 1);
	const offNode = new Float64Array(elementCount);
	const massNode = new Float64Array(elementCount + 1);
	// Self-adjoint forms:
	//   TE: k0^2 ∫ eps φ v - ∫ φ' v' = β² ∫ φ v
	//   TM: k0^2 ∫ φ v - ∫ (1/eps) φ' v' = β² ∫ (1/eps) φ v
	for (let e = 0; e < elementCount; e++) {
		const l = lengthM[e];
		const stiffnessCoeff = polarization === 'TM' ? 1 / eps[e] : 1;
		const s = stiffnessCoeff / l;
		offNode[e] += s;
		diagNode[e] -= s;
		diagNode[e + 1] -= s;
		const massG = (k0 * k0 * l) / 2 * (polarization === 'TM' ? 1 : eps[e]);
		diagNode[e] += massG;
		diagNode[e + 1] += massG;
		const massB = (l / 2) * (polarization === 'TM' ? 1 / eps[e] : 1);
		massNode[e] += massB;
		massNode[e + 1] += massB;
	}
	const n = elementCount - 1;
	const diagG = new Float64Array(n);
	const offG = new Float64Array(n - 1);
	const diagB = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		diagG[i] = diagNode[i + 1];
		diagB[i] = massNode[i + 1];
		if (i < n - 1) offG[i] = offNode[i + 1];
	}
	return { diagG, offG, diagB };
}

function reconstructFields({ polarization, phi, xM, nodeEps, beta, omega }) {
	const n = phi.length;
	const deriv = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		if (i === 0) deriv[i] = (phi[1] - phi[0]) / (xM[1] - xM[0]);
		else if (i === n - 1) deriv[i] = (phi[n - 1] - phi[n - 2]) / (xM[n - 1] - xM[n - 2]);
		else deriv[i] = (phi[i + 1] - phi[i - 1]) / (xM[i + 1] - xM[i - 1]);
	}
	const complex = () => ({ re: new Float64Array(n), im: new Float64Array(n) });
	const fields = { Ex: complex(), Ey: complex(), Ez: complex(), Hx: complex(), Hy: complex(), Hz: complex() };
	if (polarization === 'TE') {
		fields.Ey.re.set(phi);
		for (let i = 0; i < n; i++) {
			fields.Hx.re[i] = -(beta / (omega * MU0)) * phi[i];
			fields.Hz.im[i] = deriv[i] / (omega * MU0);
		}
	} else {
		fields.Hy.re.set(phi);
		for (let i = 0; i < n; i++) {
			const denom = omega * EPS0 * nodeEps[i];
			fields.Ex.re[i] = (beta / denom) * phi[i];
			fields.Ez.im[i] = -deriv[i] / denom;
		}
	}
	return fields;
}

/** Real part of 0.5 ∫ (E x H*) · z dA using the supplied quadrature weights. */
export function forwardPower(fields, weights) {
	let sum = 0;
	for (let i = 0; i < weights.length; i++) {
		const sz =
			fields.Ex.re[i] * fields.Hy.re[i] +
			fields.Ex.im[i] * fields.Hy.im[i] -
			(fields.Ey.re[i] * fields.Hx.re[i] + fields.Ey.im[i] * fields.Hx.im[i]);
		sum += weights[i] * sz;
	}
	return 0.5 * sum;
}

function scaleFields(fields, factor) {
	for (const component of Object.values(fields)) {
		for (let i = 0; i < component.re.length; i++) {
			component.re[i] *= factor;
			component.im[i] *= factor;
		}
	}
}

function sumSquared(values, weights) {
	let sum = 0;
	for (let i = 0; i < values.length; i++) sum += weights[i] * values[i] * values[i];
	return sum;
}

function sumSquaredRange(phi, xM, weights, low, high) {
	let sum = 0;
	for (let i = 0; i < phi.length; i++) {
		if (xM[i] >= low - 1e-15 && xM[i] <= high + 1e-15) sum += weights[i] * phi[i] * phi[i];
	}
	return sum;
}

export function solveSlab({
	core,
	substrate,
	cladding,
	wavelengthUm,
	polarization = 'TE',
	modeCount = 2,
	minPerFeature = 12,
	paddingUm,
	maxStepUm
}) {
	if (polarization !== 'TE' && polarization !== 'TM') throw new RangeError("polarization must be 'TE' or 'TM'");
	const mesh = buildSlabMesh({ core, substrate, cladding, wavelengthUm, minPerFeature, paddingUm, maxStepUm });
	const nodeCount = mesh.x.length;
	const xM = new Float64Array(nodeCount);
	for (let i = 0; i < nodeCount; i++) xM[i] = mesh.x[i] * UM;
	const lengthM = new Float64Array(mesh.elementLength.length);
	for (let e = 0; e < lengthM.length; e++) lengthM[e] = mesh.elementLength[e] * UM;
	const k0 = (2 * Math.PI) / (wavelengthUm * UM);
	const omega = (2 * Math.PI * C0) / (wavelengthUm * UM);
	const { diagG, offG, diagB } = assembleSlab(mesh.eps, lengthM, k0, polarization);
	const n = diagG.length;
	// T = B^{-1/2} G B^{-1/2} with diagonal lumped B.
	const diag = new Float64Array(n);
	const off = new Float64Array(n - 1);
	for (let i = 0; i < n; i++) diag[i] = diagG[i] / diagB[i];
	for (let i = 0; i < n - 1; i++) off[i] = offG[i] / Math.sqrt(diagB[i] * diagB[i + 1]);

	const weights = new Float64Array(nodeCount);
	for (let e = 0; e < lengthM.length; e++) {
		weights[e] += lengthM[e] / 2;
		weights[e + 1] += lengthM[e] / 2;
	}
	const nodeEps = new Float64Array(nodeCount);
	for (let e = 0; e < lengthM.length; e++) {
		nodeEps[e] = mesh.eps[e];
		nodeEps[e + 1] = mesh.eps[e];
	}
	for (let i = 1; i < nodeCount - 1; i++) nodeEps[i] = (mesh.eps[i - 1] + mesh.eps[i]) / 2;

	const modes = [];
	for (let k = 1; k <= modeCount; k++) {
		const pair = tridiagonalEigenpair(diag, off, k);
		if (!(pair.value > 0)) break;
		const beta = Math.sqrt(pair.value);
		// Undo the B^{-1/2} similarity: y = D^{-1/2} z.
		const y = new Float64Array(n);
		for (let i = 0; i < n; i++) y[i] = pair.vector[i] / Math.sqrt(diagB[i]);
		const phi = new Float64Array(nodeCount);
		for (let i = 0; i < n; i++) phi[i + 1] = y[i];
		let maxAbs = 0;
		let index = 0;
		for (let i = 0; i < nodeCount; i++) {
			if (Math.abs(phi[i]) > maxAbs) {
				maxAbs = Math.abs(phi[i]);
				index = i;
			}
		}
		if (phi[index] < 0) for (let i = 0; i < nodeCount; i++) phi[i] = -phi[i];
		const fields = reconstructFields({ polarization, phi, xM, nodeEps, beta, omega });
		const rawPower = forwardPower(fields, weights);
		scaleFields(fields, (rawPower < 0 ? -1 : 1) / Math.sqrt(Math.abs(rawPower)));
		const power = forwardPower(fields, weights);

		let num = 0;
		let den = 0;
		for (let i = 0; i < n; i++) {
			let gv = diagG[i] * y[i];
			if (i > 0) gv += offG[i - 1] * y[i - 1];
			if (i < n - 1) gv += offG[i] * y[i + 1];
			const bv = diagB[i] * y[i];
			const r = gv - pair.value * bv;
			num += r * r;
			den += gv * gv + pair.value * pair.value * bv * bv;
		}
		const residual = den === 0 ? Infinity : Math.sqrt(num / den);

		const totalEnergy = sumSquared(phi, weights);
		const coreEnergy = sumSquaredRange(phi, xM, weights, mesh.coreStartUm * UM, mesh.coreEndUm * UM);
		const boundaryDepth = Math.min(0.05 * mesh.lengthUm * UM, (mesh.featureUm * UM) / 2);
		const boundaryEnergy =
			sumSquaredRange(phi, xM, weights, 0, boundaryDepth) +
			sumSquaredRange(phi, xM, weights, mesh.lengthUm * UM - boundaryDepth, mesh.lengthUm * UM);

		const xCenter = (mesh.coreStartUm + mesh.coreEndUm) / 2;
		const centeredX = new Float64Array(nodeCount);
		for (let i = 0; i < nodeCount; i++) centeredX[i] = mesh.x[i] - xCenter;
		modes.push({
			model: 'slab-fem-1d',
			solverVersion: SLAB_SOLVER_VERSION,
			materialDBVersion: MATERIAL_DB_VERSION,
			meshVersion: MESH_VERSION,
			polarization,
			wavelengthUm,
			neff: beta / k0,
			beta,
			k0,
			fields,
			power,
			residual,
			coreConfinement: totalEnergy === 0 ? 0 : coreEnergy / totalEnergy,
			boundaryEnergyFraction: totalEnergy === 0 ? 1 : boundaryEnergy / totalEnergy,
			modeIndex: k - 1,
			mesh: {
				// Centered on the core so guides of different thickness share an origin.
				xUm: centeredX,
				yUm: Float64Array.of(0),
				weights,
				eps: nodeEps,
				lengthUm: mesh.lengthUm,
				coreStartUm: mesh.coreStartUm - xCenter,
				coreEndUm: mesh.coreEndUm - xCenter,
				featureUm: mesh.featureUm
			}
		});
	}
	return {
		model: 'slab-fem-1d',
		solverVersion: SLAB_SOLVER_VERSION,
		wavelengthUm,
		polarization,
		paddingUm: mesh.coreStartUm,
		steps: lengthM.length,
		modes
	};
}
