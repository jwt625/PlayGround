/** 2D scalar finite-element mode solver on rectilinear grids (C3.2 infrastructure).
 *
 * This is the validated 2D assembly / sparse-Lanczos backbone used by the
 * waveguide pipeline. It solves ∇²ψ + k0² ε ψ = β² ψ with Q1 elements and
 * natural (zero-flux) or Dirichlet boundaries. In the y-invariant limit it must
 * reproduce the 1D slab TE benchmark; that equivalence is asserted in tests.
 *
 * Scope note: this is the scalar (quasi-TE-like) operator. The full-vector 2D
 * operator is a drop-in replacement for the assembly routine and is the next
 * C3.2 step; released answers stay gated on it.
 */
import { MESH_VERSION } from './mesh.mjs';
import { MATERIAL_DB_VERSION } from './materials.mjs';

export const FEM2D_VERSION = 'scalar-fem-2d-1';

// --- Sparse symmetric CSR ---------------------------------------------------

export function buildCSR(rows, n) {
	const rowPtr = new Int32Array(n + 1);
	const colIndex = [];
	const values = [];
	for (let i = 0; i < n; i++) {
		rowPtr[i] = colIndex.length;
		const row = rows[i];
		const keys = [...row.keys()].sort((a, b) => a - b);
		for (const key of keys) {
			colIndex.push(key);
			values.push(row.get(key));
		}
	}
	rowPtr[n] = colIndex.length;
	return { n, rowPtr, colIndex: Int32Array.from(colIndex), values: Float64Array.from(values) };
}

export function csrMatvec(csr, x) {
	const y = new Float64Array(csr.n);
	for (let i = 0; i < csr.n; i++) {
		let sum = 0;
		for (let k = csr.rowPtr[i]; k < csr.rowPtr[i + 1]; k++) sum += csr.values[k] * x[csr.colIndex[k]];
		y[i] = sum;
	}
	return y;
}

// --- Q1 assembly ------------------------------------------------------------

const GAUSS = [0.5 - 0.5 / Math.sqrt(3), 0.5 + 0.5 / Math.sqrt(3)];
const GAUSS_W = [0.5, 0.5];

function shape(xi, eta) {
	return [
		{ phi: (1 - xi) * (1 - eta), dxi: -(1 - eta), deta: -(1 - xi) },
		{ phi: xi * (1 - eta), dxi: 1 - eta, deta: -xi },
		{ phi: xi * eta, dxi: eta, deta: xi },
		{ phi: (1 - xi) * eta, dxi: -eta, deta: 1 - xi }
	];
}

/**
 * Assemble G = k0² M_eps - K and the lumped mass diagonal B for a scalar mode.
 * `nodeEps` is permittivity per node; each element uses the corner average.
 */
export function assembleScalar2D(grid, nodeEps, k0) {
	const { nx, ny, x, y } = grid;
	const rows = Array.from({ length: nx * ny }, () => new Map());
	const add = (a, b, v) => rows[a].set(b, (rows[a].get(b) ?? 0) + v);
	const massDiag = new Float64Array(nx * ny);
	for (let j = 0; j < ny - 1; j++) {
		for (let i = 0; i < nx - 1; i++) {
			const hx = x[i + 1] - x[i];
			const hy = y[j + 1] - y[j];
			const nodes = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i];
			const epsE = (nodeEps[nodes[0]] + nodeEps[nodes[1]] + nodeEps[nodes[2]] + nodeEps[nodes[3]]) / 4;
			const ke = [
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0]
			];
			const me = [
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0],
				[0, 0, 0, 0]
			];
			for (const xi of GAUSS) {
				for (const eta of GAUSS) {
					const w = GAUSS_W[GAUSS.indexOf(xi)] * GAUSS_W[GAUSS.indexOf(eta)] * hx * hy;
					const s = shape(xi, eta);
					for (let a = 0; a < 4; a++) {
						for (let b = 0; b < 4; b++) {
							const dot = (s[a].dxi * s[b].dxi) / (hx * hx) + (s[a].deta * s[b].deta) / (hy * hy);
							ke[a][b] += w * dot;
							me[a][b] += w * s[a].phi * s[b].phi;
						}
					}
				}
			}
			const area = hx * hy;
			for (let a = 0; a < 4; a++) {
				massDiag[nodes[a]] += area / 4;
				for (let b = 0; b < 4; b++) add(nodes[a], nodes[b], k0 * k0 * epsE * me[a][b] - ke[a][b]);
			}
		}
	}
	return { rows, massDiag };
}

function jacobiEigen(matrix, iterations = 100) {
	const n = matrix.length;
	const a = matrix.map((row) => row.slice());
	const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
	for (let sweep = 0; sweep < iterations; sweep++) {
		let off = 0;
		for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
		if (off < 1e-30) break;
		for (let p = 0; p < n; p++) {
			for (let q = p + 1; q < n; q++) {
				if (Math.abs(a[p][q]) < 1e-300) continue;
				const theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
				const c = Math.cos(theta);
				const s = Math.sin(theta);
				for (let k = 0; k < n; k++) {
					const akp = a[k][p];
					const akq = a[k][q];
					a[k][p] = c * akp - s * akq;
					a[k][q] = s * akp + c * akq;
				}
				for (let k = 0; k < n; k++) {
					const apk = a[p][k];
					const aqk = a[q][k];
					a[p][k] = c * apk - s * aqk;
					a[q][k] = s * apk + c * aqk;
				}
				for (let k = 0; k < n; k++) {
					const vkp = v[k][p];
					const vkq = v[k][q];
					v[k][p] = c * vkp - s * vkq;
					v[k][q] = s * vkp + c * vkq;
				}
			}
		}
	}
	const values = a.map((row, i) => row[i]);
	const order = values.map((value, i) => ({ value, i })).sort((p, q) => q.value - p.value);
	return { values: order.map((o) => o.value), vectors: order.map((o) => v.map((row) => row[o.i])) };
}

function dot(a, b) {
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
	return sum;
}

function norm(a) {
	return Math.sqrt(dot(a, a));
}

function orthonormalize(vectors) {
	const basis = [];
	for (const vector of vectors) {
		let v = vector;
		for (let pass = 0; pass < 2; pass++) {
			for (const q of basis) {
				const projection = dot(q, v);
				for (let i = 0; i < v.length; i++) v[i] -= projection * q[i];
			}
		}
		const length = norm(v);
		if (length > 1e-12) basis.push(v.map((value) => value / length));
	}
	return basis;
}

/**
 * Lanczos with full reorthogonalization for the largest eigenvalues of the
 * symmetric operator T = B^{-1/2} G B^{-1/2} (B diagonal).
 */
export function lanczosLargest(matvec, n, count, { steps = 100, seed = 12345, tolerance = 1e-10 } = {}) {
	const m = Math.min(steps, n);
	let state = seed >>> 0;
	const random = () => {
		state = (state * 1664525 + 1013904223) >>> 0;
		return state / 4294967296 - 0.5;
	};
	let q = Array.from({ length: n }, () => random());
	const qn = norm(q);
	q = q.map((value) => value / qn);
	const Q = [q];
	const alpha = [];
	const beta = [];
	let previous = new Float64Array(n);
	let betaPrev = 0;
	for (let j = 0; j < m; j++) {
		let z = matvec(Q[j]);
		if (j > 0) for (let i = 0; i < n; i++) z[i] -= betaPrev * Q[j - 1][i];
		const a = dot(Q[j], z);
		alpha.push(a);
		for (let i = 0; i < n; i++) z[i] -= a * Q[j][i];
		// Full reorthogonalization for numerical stability.
		for (let pass = 0; pass < 2; pass++) {
			for (const basis of Q) {
				const projection = dot(basis, z);
				for (let i = 0; i < n; i++) z[i] -= projection * basis[i];
			}
		}
		const b = norm(z);
		beta.push(b);
		if (b < 1e-14 || j === m - 1) break;
		betaPrev = b;
		Q.push(z.map((value) => value / b));
	}
	const dim = alpha.length;
	const tridiagonal = Array.from({ length: dim }, (_, i) =>
		Array.from({ length: dim }, (_, j) => (i === j ? alpha[i] : Math.abs(i - j) === 1 ? beta[Math.min(i, j)] : 0))
	);
	const { values, vectors } = jacobiEigen(tridiagonal);
	const results = [];
	for (let k = 0; k < Math.min(count, dim); k++) {
		const eigenvector = new Float64Array(n);
		for (let j = 0; j < dim; j++) {
			const coeff = vectors[k][j];
			if (coeff === 0) continue;
			for (let i = 0; i < n; i++) eigenvector[i] += coeff * Q[j][i];
		}
		const length = norm(eigenvector);
		if (length > 0) for (let i = 0; i < n; i++) eigenvector[i] /= length;
		results.push({ value: values[k], vector: eigenvector });
	}
	return results;
}

function trapezoidWeights1D(edges) {
	const n = edges.length;
	const w = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const left = i > 0 ? edges[i] - edges[i - 1] : edges[1] - edges[0];
		const right = i < n - 1 ? edges[i + 1] - edges[i] : edges[i] - edges[i - 1];
		w[i] = 0.5 * (left + right);
	}
	return w;
}

/**
 * Solve the scalar 2D mode problem. `dirichlet` selects which grid sides are
 * fixed walls; natural sides are zero-flux (used to emulate y-invariance).
 */
export function solveScalar2D({
	grid,
	nodeEps,
	wavelengthUm,
	modeCount = 1,
	dirichlet = { xMin: true, xMax: true, yMin: true, yMax: true },
	lanczosSteps = 120
}) {
	const k0 = (2 * Math.PI) / (wavelengthUm * 1e-6);
	const { nx, ny } = grid;
	// Assemble in SI metres so k0 (1/m) is consistent with the element sizes.
	const gridM = { nx, ny, x: Float64Array.from(grid.x, (v) => v * 1e-6), y: Float64Array.from(grid.y, (v) => v * 1e-6) };
	const { rows, massDiag } = assembleScalar2D(gridM, nodeEps, k0);
	const fixed = new Uint8Array(nx * ny);
	for (let j = 0; j < ny; j++) {
		for (let i = 0; i < nx; i++) {
			if (
				(i === 0 && dirichlet.xMin) ||
				(i === nx - 1 && dirichlet.xMax) ||
				(j === 0 && dirichlet.yMin) ||
				(j === ny - 1 && dirichlet.yMax)
			) {
				fixed[j * nx + i] = 1;
			}
		}
	}
	const index = new Int32Array(nx * ny).fill(-1);
	let free = 0;
	for (let i = 0; i < index.length; i++) if (!fixed[i]) index[i] = free++;
	const freeRows = Array.from({ length: free }, () => new Map());
	for (let i = 0; i < nx * ny; i++) {
		if (fixed[i]) continue;
		for (const [j, value] of rows[i]) {
			if (!fixed[j]) freeRows[index[i]].set(index[j], value);
		}
	}
	const csr = buildCSR(freeRows, free);
	const B = new Float64Array(free);
	for (let i = 0; i < nx * ny; i++) if (!fixed[i]) B[index[i]] = massDiag[i];
	const sqrtB = B.map((value) => Math.sqrt(Math.max(value, 1e-300)));
	const n = free;
	const matvec = (v) => {
		const u = new Float64Array(n);
		for (let i = 0; i < n; i++) u[i] = v[i] / sqrtB[i];
		const w = csrMatvec(csr, u);
		for (let i = 0; i < n; i++) w[i] /= sqrtB[i];
		return w;
	};
	// Shift so the largest algebraic eigenvalue is the dominant one.
	let lo = Infinity;
	for (let i = 0; i < n; i++) {
		let sum = 0;
		for (let k = csr.rowPtr[i]; k < csr.rowPtr[i + 1]; k++) sum += Math.abs(csr.values[k]);
		const diag = rowsToDiagonal(csr, i);
		lo = Math.min(lo, diag - (sum - Math.abs(diag)));
	}
	const shift = Math.max(0, -lo) + 1;
	const pairs = lanczosLargest((v) => {
		const w = matvec(v);
		for (let i = 0; i < n; i++) w[i] += shift * v[i];
		return w;
	}, n, modeCount + 2, { steps: lanczosSteps });
	const modes = [];
	const wx = trapezoidWeights1D(grid.x);
	const wy = trapezoidWeights1D(grid.y);
	for (const pair of pairs) {
		const betaSq = pair.value - shift;
		if (!(betaSq > 0)) continue;
		// Physical eigenvector is D^{-1/2} times the T-eigenvector.
		const y = new Float64Array(n);
		for (let i = 0; i < n; i++) y[i] = pair.vector[i] / sqrtB[i];
		const psiRe = new Float64Array(nx * ny);
		for (let i = 0; i < nx * ny; i++) if (!fixed[i]) psiRe[i] = y[index[i]];
		// Normalize to unit L2 norm with trapezoidal quadrature.
		let energy = 0;
		for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) energy += wx[i] * wy[j] * psiRe[j * nx + i] ** 2;
		const scale = energy > 0 ? 1 / Math.sqrt(energy) : 0;
		for (let i = 0; i < psiRe.length; i++) psiRe[i] *= scale;
		const beta = Math.sqrt(betaSq);
		// Residual on the free system.
		let num = 0;
		let den = 0;
		for (let i = 0; i < n; i++) {
			let gv = 0;
			for (let k = csr.rowPtr[i]; k < csr.rowPtr[i + 1]; k++) gv += csr.values[k] * y[csr.colIndex[k]];
			const bv = B[i] * y[i];
			const r = gv - betaSq * bv;
			num += r * r;
			den += gv * gv + betaSq * betaSq * bv * bv;
		}
		const residual = den === 0 ? Infinity : Math.sqrt(num / den);
		let total = 0;
		let boundary = 0;
		const depthX = 0.05 * (grid.x[nx - 1] - grid.x[0]);
		const depthY = 0.05 * (grid.y[ny - 1] - grid.y[0]);
		for (let j = 0; j < ny; j++) {
			for (let i = 0; i < nx; i++) {
				const contribution = wx[i] * wy[j] * psiRe[j * nx + i] ** 2;
				total += contribution;
				const nearBoundary =
					grid.x[i] < grid.x[0] + depthX ||
					grid.x[i] > grid.x[nx - 1] - depthX ||
					grid.y[j] < grid.y[0] + depthY ||
					grid.y[j] > grid.y[ny - 1] - depthY;
				if (nearBoundary) boundary += contribution;
			}
		}
		const weights = new Float64Array(nx * ny);
		for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) weights[j * nx + i] = wx[i] * wy[j];
		modes.push({
			model: FEM2D_VERSION,
			solverVersion: FEM2D_VERSION,
			materialDBVersion: MATERIAL_DB_VERSION,
			meshVersion: MESH_VERSION,
			wavelengthUm,
			polarization: 'scalar',
			neff: beta / k0,
			beta,
			scalar: { re: psiRe, im: new Float64Array(psiRe.length) },
			power: 1,
			residual,
			boundaryEnergyFraction: total === 0 ? 1 : boundary / total,
			modeIndex: modes.length,
			mesh: {
				xUm: grid.x,
				yUm: grid.y,
				nx,
				ny,
				weights,
				eps: nodeEps
			}
		});
		if (modes.length >= modeCount) break;
	}
	return {
		model: FEM2D_VERSION,
		solverVersion: FEM2D_VERSION,
		wavelengthUm,
		steps: 1,
		freeNodes: free,
		modes
	};
}

function rowsToDiagonal(csr, i) {
	for (let k = csr.rowPtr[i]; k < csr.rowPtr[i + 1]; k++) if (csr.colIndex[k] === i) return csr.values[k];
	return 0;
}
