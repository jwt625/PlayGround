/** 2D full-vector mode solver (C3.2/C3.3).
 *
 * Formulation: the self-adjoint magnetic curl-curl weak statement
 *   ∫ (1/ε)(∇×H)·(∇×V*) dA = k0² ∫ H·V* dA
 * on a rectilinear Q1 mesh. The z-component is encoded as H_z = i·g with g
 * real, so the transverse components and g are real and the assembled matrices
 * K0, K1, K2, M are real symmetric. Propagation then gives the symmetric
 * quadratic eigenproblem
 *   (K0 + β K1 − β² K2) x = 0,   H·V* mass M folded into K0 via −k0²M.
 *
 * It is solved with nonlinear Rayleigh iteration: solve S(σ)w = v with a dense
 * LU (S = K0 + σK1 + σ²K2, symmetric indefinite), then pick the β root of the
 * scalar quadratic wᵀS(β)w = 0 nearest σ and repeat. No interface-derivative
 * (∇ε) terms appear.
 *
 * Status: experimental. The operator is verified at the 1D level against closed
 * form TE/TM dispersion and its variational quotient converges to the analytic
 * slab value under refinement, but the 2D blind mode finder does not yet
 * reliably isolate the guided modes on a full 2D grid (spurious and irrelevant
 * modes ingress). It is NOT release-eligible and no recipe uses it.
 */
import { MESH_VERSION } from './mesh.mjs';
import { MATERIAL_DB_VERSION } from './materials.mjs';

export const FULLVECTOR2D_VERSION = 'fullvector-fem-2d-1';

const C0 = 299792458;
const MU0 = 4 * Math.PI * 1e-7;
const EPS0 = 1 / (MU0 * C0 * C0);

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

function rowsOf(n) {
	return Array.from({ length: n }, () => new Map());
}

function add(row, col, value) {
	if (value === 0) return;
	const row0 = row;
	row0.set(col, (row0.get(col) ?? 0) + value);
}

export function matvec(rows, v) {
	const out = new Float64Array(rows.length);
	for (let i = 0; i < rows.length; i++) {
		let sum = 0;
		for (const [j, value] of rows[i]) sum += value * v[j];
		out[i] = sum;
	}
	return out;
}

export function dot(a, b) {
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
	return sum;
}

export function norm(a) {
	return Math.sqrt(dot(a, a));
}

/** GMRES(m) with restart for a real (possibly indefinite) sparse system. */
export function gmres(rows, b, { restart = 60, maxIter = 400, tolerance = 1e-11 } = {}) {
	const n = rows.length;
	const x = new Float64Array(n);
	let residual = new Float64Array(b);
	let beta = norm(residual);
	if (beta < tolerance) return x;
	let iterations = 0;
	while (iterations < maxIter && beta > tolerance) {
		const m = Math.min(restart, maxIter - iterations);
		const Q = [new Float64Array(n)];
		for (let i = 0; i < n; i++) Q[0][i] = residual[i] / beta;
		const H = Array.from({ length: m + 1 }, () => new Float64Array(m));
		let k = 0;
		for (; k < m; k++) {
			const w = matvec(rows, Q[k]);
			for (let i = 0; i <= k; i++) {
				H[i][k] = dot(Q[i], w);
				for (let p = 0; p < n; p++) w[p] -= H[i][k] * Q[i][p];
			}
			H[k + 1][k] = norm(w);
			if (H[k + 1][k] > 1e-14) {
				const q = new Float64Array(n);
				for (let p = 0; p < n; p++) q[p] = w[p] / H[k + 1][k];
				Q.push(q);
			} else {
				k++;
				break;
			}
			iterations++;
		}
		const size = Math.min(k, m);
		// Least squares via Givens rotations.
		const g = new Float64Array(size + 1);
		g[0] = beta;
		const R = Array.from({ length: size + 1 }, () => new Float64Array(size));
		for (let i = 0; i <= size; i++) for (let j = 0; j < size; j++) R[i][j] = H[i][j];
		const cs = new Float64Array(size);
		const sn = new Float64Array(size);
		for (let i = 0; i < size; i++) {
			const a = R[i][i];
			const c = R[i + 1][i];
			const denom = Math.hypot(a, c) || 1;
			cs[i] = a / denom;
			sn[i] = c / denom;
			for (let j = i; j < size; j++) {
				const r0 = R[i][j];
				const r1 = R[i + 1][j];
				R[i][j] = cs[i] * r0 + sn[i] * r1;
				R[i + 1][j] = -sn[i] * r0 + cs[i] * r1;
			}
			const g0 = g[i];
			const g1 = g[i + 1];
			g[i] = cs[i] * g0 + sn[i] * g1;
			g[i + 1] = -sn[i] * g0 + cs[i] * g1;
		}
		const y = new Float64Array(size);
		for (let i = size - 1; i >= 0; i--) {
			let sum = g[i];
			for (let j = i + 1; j < size; j++) sum -= R[i][j] * y[j];
			y[i] = sum / (R[i][i] || 1);
		}
		for (let i = 0; i < size; i++) for (let p = 0; p < n; p++) x[p] += y[i] * Q[i][p];
		residual = new Float64Array(b);
		const ax = matvec(rows, x);
		for (let i = 0; i < n; i++) residual[i] -= ax[i];
		beta = norm(residual);
		if (size < m) break;
	}
	return x;
}

/**
 * Assemble K0, K1, K2 and the mass M (3 dof per node: hx, hy, g = imag Hz).
 * Coordinates must be in metres and k0 in 1/m.
 */
export function assembleFullVector2D(grid, nodeEps, k0, { penalty = 1 } = {}) {
	const { nx, ny, x, y } = grid;
	const nodes = nx * ny;
	const dofs = 3 * nodes;
	const K0 = rowsOf(dofs);
	const K1 = rowsOf(dofs);
	const K2 = rowsOf(dofs);
	const at = (node, field) => 3 * node + field;
	for (let j = 0; j < ny - 1; j++) {
		for (let i = 0; i < nx - 1; i++) {
			const hx = x[i + 1] - x[i];
			const hy = y[j + 1] - y[j];
			const element = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i];
			const epsE = (nodeEps[element[0]] + nodeEps[element[1]] + nodeEps[element[2]] + nodeEps[element[3]]) / 4;
			const wE = 1 / epsE;
			const M0 = [0, 1, 2, 3].map(() => new Float64Array(4));
			const Dx = [0, 1, 2, 3].map(() => new Float64Array(4));
			const Dy = [0, 1, 2, 3].map(() => new Float64Array(4));
			const Sxx = [0, 1, 2, 3].map(() => new Float64Array(4));
			const Syy = [0, 1, 2, 3].map(() => new Float64Array(4));
			const Sxy = [0, 1, 2, 3].map(() => new Float64Array(4));
			for (let gi = 0; gi < 2; gi++) {
				for (let gj = 0; gj < 2; gj++) {
					const w = GAUSS_W[gi] * GAUSS_W[gj] * hx * hy;
					const s = shape(GAUSS[gi], GAUSS[gj]);
					for (let a = 0; a < 4; a++) {
						const dax = s[a].dxi / hx;
						const day = s[a].deta / hy;
						for (let b = 0; b < 4; b++) {
							const bx = s[b].dxi / hx;
							const by = s[b].deta / hy;
							M0[a][b] += w * s[a].phi * s[b].phi;
							Dx[a][b] += w * dax * s[b].phi;
							Dy[a][b] += w * day * s[b].phi;
							Sxx[a][b] += w * dax * bx;
							Syy[a][b] += w * day * by;
							Sxy[a][b] += w * dax * by;
						}
					}
				}
			}
			for (let a = 0; a < 4; a++) {
				for (let b = 0; b < 4; b++) {
					const n0 = at(element[a], 0);
					const n1 = at(element[a], 1);
					const n2 = at(element[a], 2);
					const m0 = at(element[b], 0);
					const m1 = at(element[b], 1);
					const m2 = at(element[b], 2);
					const m = M0[a][b];
					// K2 = ∫(1/ε)(hx vx + hy vy)
					add(K2[n0], m0, wE * m);
					add(K2[n1], m1, wE * m);
					// K1 = β coefficient = -∫(1/ε) B1
					add(K1[n0], m2, -wE * Dx[b][a]);
					add(K1[n1], m2, -wE * Dy[b][a]);
					add(K1[n2], m0, -wE * Dx[a][b]);
					add(K1[n2], m1, -wE * Dy[a][b]);
					// K0 = B0 - k0²M
					add(K0[n0], m0, wE * Syy[a][b] - k0 * k0 * m);
					add(K0[n1], m1, wE * Sxx[a][b] - k0 * k0 * m);
					add(K0[n2], m2, wE * (Sxx[a][b] + Syy[a][b]) - k0 * k0 * m);
					add(K0[n0], m1, -wE * Sxy[b][a]);
					add(K0[n1], m0, -wE * Sxy[a][b]);
					if (penalty !== 0) {
						// p ∫ (∇·H)(∇·V*) suppresses irrotational spurious modes.
						add(K0[n0], m0, penalty * Sxx[a][b]);
						add(K0[n0], m1, penalty * Sxy[a][b]);
						add(K0[n1], m0, penalty * Sxy[b][a]);
						add(K0[n1], m1, penalty * Syy[a][b]);
						add(K1[n2], m0, -penalty * Dx[a][b]);
						add(K1[n2], m1, -penalty * Dy[a][b]);
						add(K1[n0], m2, -penalty * Dx[b][a]);
						add(K1[n1], m2, -penalty * Dy[b][a]);
						add(K2[n2], m2, penalty * m);
					}
				}
			}
		}
	}
	return { K0, K1, K2, dofs, nodes, nx, ny };
}

/** Dense partial-pivot LU solve of a sparse row map. Used for the near-singular
 * shift systems on validation/desktop-sized grids. */
export function denseLuSolve(rows, b) {
	const n = rows.length;
	const A = Array.from({ length: n }, (_, i) => {
		const row = new Float64Array(n);
		for (const [j, value] of rows[i]) row[j] = value;
		return row;
	});
	const x = Float64Array.from(b);
	for (let k = 0; k < n; k++) {
		let pivot = k;
		let best = Math.abs(A[k][k]);
		for (let i = k + 1; i < n; i++) {
			const value = Math.abs(A[i][k]);
			if (value > best) {
				best = value;
				pivot = i;
			}
		}
		if (pivot !== k) {
			const tmp = A[k];
			A[k] = A[pivot];
			A[pivot] = tmp;
			const tb = x[k];
			x[k] = x[pivot];
			x[pivot] = tb;
		}
		const diagonal = A[k][k] === 0 ? 1e-300 : A[k][k];
		for (let i = k + 1; i < n; i++) {
			const factor = A[i][k] / diagonal;
			if (factor === 0) continue;
			A[i][k] = factor;
			for (let j = k + 1; j < n; j++) A[i][j] -= factor * A[k][j];
			x[i] -= factor * x[k];
		}
	}
	for (let i = n - 1; i >= 0; i--) {
		let sum = x[i];
		for (let j = i + 1; j < n; j++) sum -= A[i][j] * x[j];
		x[i] = sum / (A[i][i] === 0 ? 1e-300 : A[i][i]);
	}
	return x;
}

export function combine(rowMaps, weights, dofs) {
	const rows = rowsOf(dofs);
	for (let s = 0; s < rowMaps.length; s++) {
		const w = weights[s];
		if (w === 0) continue;
		for (let i = 0; i < dofs; i++) {
			const target = rows[i];
			for (const [j, value] of rowMaps[s][i]) target.set(j, (target.get(j) ?? 0) + w * value);
		}
	}
	return rows;
}

export function applyDirichlet(rows, dofs, fixed) {
	for (let i = 0; i < dofs; i++) {
		if (!fixed[i]) continue;
		rows[i] = new Map();
		for (let r = 0; r < dofs; r++) rows[r].delete(i);
	}
}

export function quadraticRoots(a, b, c, target) {
	// a + b β + c β² = 0
	if (Math.abs(c) < 1e-30) return [Math.abs(b) < 1e-30 ? target : -a / b];
	const disc = b * b - 4 * a * c;
	if (disc < 0) {
		// No real root for this trial vector: step to the parabola vertex so the
		// iteration keeps moving instead of stalling on a mixed vector.
		return [-b / (2 * c)];
	}
	const root = Math.sqrt(disc);
	return [(-b + root) / (2 * c), (-b - root) / (2 * c)];
}

/**
 * Find one full-vector mode near `sigma` (β guess in 1/m) by nonlinear
 * Rayleigh iteration. `fixed` marks Dirichlet nodes.
 */
export function solveNear({ K0, K1, K2, dofs, fixed, sigma0, initial, iterations = 20, tolerance = 1e-9 }) {
	let sigma = sigma0;
	let v;
	if (initial) {
		v = Float64Array.from(initial);
		for (let i = 0; i < dofs; i++) if (fixed[i]) v[i] = 0;
	} else {
		v = new Float64Array(dofs);
		for (let i = 0; i < dofs; i++) v[i] = fixed[i] ? 0 : Math.sin(0.3 * i) + 1;
	}
	let vn = norm(v);
	for (let i = 0; i < dofs; i++) v[i] /= vn;
	// Companion second block starts at sigma*v so u = [v; sigma v].
	let companion = Float64Array.from(v, (value) => value * sigma);
	let beta = sigma;
	let vector = v;
	// Shift-invert inverse iteration on the companion pencil
	//   [[K0, K1], [0, I]] u = λ [[0, -K2], [I, 0]] u,   λ = β.
	for (let iter = 0; iter < iterations; iter++) {
		const k1u1 = matvec(K1, v);
		const k2u1 = matvec(K2, v);
		const k2u2 = matvec(K2, companion);
		const rhs = new Float64Array(dofs);
		for (let i = 0; i < dofs; i++) rhs[i] = -k2u2[i] - k1u1[i] - sigma * k2u1[i];
		for (let i = 0; i < dofs; i++) if (fixed[i]) rhs[i] = 0;
		const rows = combine([K0, K1, K2], [1, sigma, sigma * sigma], dofs);
		applyDirichlet(rows, dofs, fixed);
		const x = denseLuSolve(rows, rhs);
		for (let i = 0; i < dofs; i++) if (fixed[i]) x[i] = 0;
		const xNorm = norm(x);
		if (!Number.isFinite(xNorm) || xNorm < 1e-300) break;
		for (let i = 0; i < dofs; i++) x[i] /= xNorm;
		const a = dot(x, matvec(K0, x));
		const b = dot(x, matvec(K1, x));
		const c = dot(x, matvec(K2, x));
		const roots = quadraticRoots(a, b, c, sigma);
		const next = roots.reduce((best, candidate) =>
			Math.abs(candidate - sigma) < Math.abs(best - sigma) ? candidate : best
		);
		const delta = Math.abs(next - sigma);
		beta = next;
		sigma = next;
		vector = x;
		companion = Float64Array.from(x, (value) => value * next);
		if (delta < tolerance * (1 + Math.abs(next))) break;
	}
	// Residual for the quadratic at beta using the restricted (Dirichlet) system.
	const rows = combine([K0, K1, K2], [1, beta, beta * beta], dofs);
	for (let i = 0; i < dofs; i++) if (fixed[i]) rows[i] = new Map();
	const r = matvec(rows, vector);
	let num = norm(r);
	let den = 0;
	const terms = [matvec(K0, vector), matvec(K1, vector), matvec(K2, vector)];
	for (let i = 0; i < dofs; i++) {
		den += terms[0][i] ** 2 + (beta ** 2) * terms[1][i] ** 2 + (beta ** 4) * terms[2][i] ** 2;
	}
	return { beta, vector, residual: den === 0 ? Infinity : num / Math.sqrt(den) };
}

/**
 * Solve full-vector modes by scanning β guesses across the guided window and
 * refining each with Rayleigh iteration. Returns modes with reconstructed E/H.
 */
export function solveFullVector2D({
	grid,
	nodeEps,
	wavelengthUm,
	modeCount = 2,
	dirichlet = { xMin: true, xMax: true, yMin: true, yMax: true },
	neffGuesses,
	scanSamples = 60,
	coreRegion,
	confinementThreshold = 0.5,
	confinementFactor = 2
}) {
	const k0 = (2 * Math.PI) / (wavelengthUm * 1e-6);
	const { nx, ny } = grid;
	const gridM = {
		nx,
		ny,
		x: Float64Array.from(grid.x, (v) => v * 1e-6),
		y: Float64Array.from(grid.y, (v) => v * 1e-6)
	};
	const { K0, K1, K2, dofs, nodes } = assembleFullVector2D(gridM, nodeEps, k0);
	const fixed = new Uint8Array(dofs);
	for (let j = 0; j < ny; j++) {
		for (let i = 0; i < nx; i++) {
			const isFixed =
				(i === 0 && dirichlet.xMin) ||
				(i === nx - 1 && dirichlet.xMax) ||
				(j === 0 && dirichlet.yMin) ||
				(j === ny - 1 && dirichlet.yMax);
			if (isFixed) for (let f = 0; f < 3; f++) fixed[3 * (j * nx + i) + f] = 1;
		}
	}
	const epsMax = Math.max(...nodeEps);
	const epsMin = Math.min(...nodeEps);
	const guesses = neffGuesses
		? neffGuesses.map((n) => (k0 * n))
		: Array.from({ length: scanSamples }, (_, i) =>
				k0 * (Math.sqrt(epsMin) + ((Math.sqrt(epsMax) - Math.sqrt(epsMin)) * (i + 0.5)) / scanSamples)
			);
	const seeds = [];
	for (let field = 0; field < 3; field++) {
		const seed = new Float64Array(dofs);
		for (let n = 0; n < nodes; n++) seed[3 * n + field] = 1;
		seeds.push(seed);
	}
	const allSeed = new Float64Array(dofs).fill(1);
	seeds.push(allSeed);
	const found = [];
	for (const guess of guesses) {
		for (const seed of seeds) {
			const result = solveNear({ K0, K1, K2, dofs, fixed, sigma0: guess, initial: seed });
			const neff = result.beta / k0;
			if (!(neff > Math.sqrt(epsMin)) || neff > Math.sqrt(epsMax) + 1e-9) continue;
			if (result.residual > 1e-4 || !Number.isFinite(neff)) continue;
			if (found.some((mode) => Math.abs(mode.neff - neff) < 1e-4)) continue;
			found.push(result);
		}
	}
	// Spurious box/radiation modes are accepted by the scan; keep only modes
	// whose field energy is concentrated in a 2x-core window, then rank by neff
	// so the fundamental (highest index) comes first.
	const region = coreRegion ?? inferCoreRegion(grid, nodeEps);
	const centerX = (region.x0 + region.x1) / 2;
	const centerY = (region.y0 + region.y1) / 2;
	const halfX = ((region.x1 - region.x0) / 2) * confinementFactor;
	const halfY = ((region.y1 - region.y0) / 2) * confinementFactor;
	const inside = (x, y) => x >= centerX - halfX && x <= centerX + halfX && y >= centerY - halfY && y <= centerY + halfY;

	const wx = trapezoid1D(grid.x);
	const wy = trapezoid1D(grid.y);
	const evaluated = [];
	for (const result of found) {
		const hx = new Float64Array(nodes);
		const hy = new Float64Array(nodes);
		const gz = new Float64Array(nodes);
		for (let n = 0; n < nodes; n++) {
			hx[n] = result.vector[3 * n];
			hy[n] = result.vector[3 * n + 1];
			gz[n] = result.vector[3 * n + 2];
		}
		const { Ex, Ey, Ez } = reconstructE({ hx, hy, gz, grid, nodeEps, beta: result.beta, k0 });
		let total = 0;
		let core = 0;
		for (let j = 0; j < ny; j++) {
			for (let i = 0; i < nx; i++) {
				const n = j * nx + i;
				const energy = Ex[n] * Ex[n] + Ey[n] * Ey[n] + Ez[n] * Ez[n];
				const weighted = wx[i] * wy[j] * energy;
				total += weighted;
				if (inside(grid.x[i], grid.y[j])) core += weighted;
			}
		}
		const confinement = total > 0 ? core / total : 0;
		if (confinement < confinementThreshold) continue;
		evaluated.push({ result, hx, hy, gz, Ex, Ey, Ez, confinement });
	}
	evaluated.sort((a, b) => b.result.beta - a.result.beta);
	const modes = [];
	for (const item of evaluated.slice(0, modeCount)) {
		const { result, hx, hy, gz, Ex, Ey, Ez, confinement } = item;
		const fields = {
			Ex: { re: Float64Array.from(Ex), im: new Float64Array(nodes) },
			Ey: { re: Float64Array.from(Ey), im: new Float64Array(nodes) },
			Ez: { re: new Float64Array(nodes), im: Float64Array.from(Ez) },
			Hx: { re: Float64Array.from(hx), im: new Float64Array(nodes) },
			Hy: { re: Float64Array.from(hy), im: new Float64Array(nodes) },
			Hz: { re: new Float64Array(nodes), im: Float64Array.from(gz) }
		};
		let rawPower = 0;
		for (let j = 0; j < ny; j++)
			for (let i = 0; i < nx; i++) {
				const n = j * nx + i;
				rawPower += 0.5 * wx[i] * wy[j] * (Ex[n] * hy[n] - Ey[n] * hx[n]);
			}
		const scale = 1 / Math.sqrt(Math.abs(rawPower) || 1);
		for (const component of Object.values(fields)) {
			for (let i = 0; i < component.re.length; i++) {
				component.re[i] *= scale;
				component.im[i] *= scale;
			}
		}
		let powerNorm = rawPower * scale * scale;
		if (powerNorm < 0) {
			for (const name of ['Hx', 'Hy', 'Hz']) {
				for (let i = 0; i < fields[name].re.length; i++) {
					fields[name].re[i] *= -1;
					fields[name].im[i] *= -1;
				}
			}
			powerNorm = -powerNorm;
		}
		const weights = new Float64Array(nodes);
		for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) weights[j * nx + i] = wx[i] * wy[j];
		modes.push({
			model: FULLVECTOR2D_VERSION,
			solverVersion: FULLVECTOR2D_VERSION,
			materialDBVersion: MATERIAL_DB_VERSION,
			meshVersion: MESH_VERSION,
			wavelengthUm,
			polarization: 'vector',
			neff: result.beta / k0,
			beta: result.beta,
			k0,
			fields,
			power: powerNorm,
			residual: result.residual,
			confinement,
			modeIndex: modes.length,
			coreRegion: region,
			mesh: { xUm: grid.x, yUm: grid.y, nx, ny, weights, eps: nodeEps }
		});
	}
	return {
		model: FULLVECTOR2D_VERSION,
		solverVersion: FULLVECTOR2D_VERSION,
		wavelengthUm,
		steps: 1,
		dofs,
		modes
	};
}

/** Bounding box of nodes above the mid permittivity, used when no core is given. */
export function inferCoreRegion(grid, nodeEps) {
	let lo = Infinity;
	let hi = -Infinity;
	for (const value of nodeEps) {
		lo = Math.min(lo, value);
		hi = Math.max(hi, value);
	}
	const threshold = (lo + hi) / 2;
	let x0 = Infinity;
	let x1 = -Infinity;
	let y0 = Infinity;
	let y1 = -Infinity;
	for (let j = 0; j < grid.ny; j++) {
		for (let i = 0; i < grid.nx; i++) {
			if (nodeEps[j * grid.nx + i] > threshold) {
				x0 = Math.min(x0, grid.x[i]);
				x1 = Math.max(x1, grid.x[i]);
				y0 = Math.min(y0, grid.y[j]);
				y1 = Math.max(y1, grid.y[j]);
			}
		}
	}
	if (!Number.isFinite(x0)) return { x0: grid.x[0], x1: grid.x[grid.nx - 1], y0: grid.y[0], y1: grid.y[grid.ny - 1] };
	return { x0, x1, y0, y1 };
}

function reconstructE({ hx, hy, gz, grid, nodeEps, beta, k0 }) {
	const { nx, ny, x, y } = grid;
	const nodes = nx * ny;
	const omega = k0 * C0;
	const Ex = new Float64Array(nodes);
	const Ey = new Float64Array(nodes);
	const Ez = new Float64Array(nodes);
	const xM = Float64Array.from(x, (v) => v * 1e-6);
	const yM = Float64Array.from(y, (v) => v * 1e-6);
	const dgx = new Float64Array(nodes);
	const dgy = new Float64Array(nodes);
	const dhy = new Float64Array(nodes);
	const dhx = new Float64Array(nodes);
	for (let j = 0; j < ny; j++) {
		for (let i = 0; i < nx; i++) {
			const n = j * nx + i;
			const xl = i > 0 ? xM[i - 1] : xM[i];
			const xr = i < nx - 1 ? xM[i + 1] : xM[i];
			const yd = j > 0 ? yM[j - 1] : yM[j];
			const yu = j < ny - 1 ? yM[j + 1] : yM[j];
			dgx[n] = i > 0 && i < nx - 1 ? (gz[j * nx + i + 1] - gz[j * nx + i - 1]) / (xr - xl) : 0;
			dgy[n] = j > 0 && j < ny - 1 ? (gz[(j + 1) * nx + i] - gz[(j - 1) * nx + i]) / (yu - yd) : 0;
			dhy[n] = i > 0 && i < nx - 1 ? (hy[j * nx + i + 1] - hy[j * nx + i - 1]) / (xr - xl) : 0;
			dhx[n] = j > 0 && j < ny - 1 ? (hx[(j + 1) * nx + i] - hx[(j - 1) * nx + i]) / (yu - yd) : 0;
		}
	}
	for (let n = 0; n < nodes; n++) {
		const denom = omega * EPS0 * nodeEps[n];
		Ex[n] = (dgy[n] - beta * hy[n]) / denom;
		Ey[n] = (beta * hx[n] - dgx[n]) / denom;
		Ez[n] = -(dhy[n] - dhx[n]) / denom;
	}
	return { Ex, Ey, Ez };
}

function trapezoid1D(edges) {
	const n = edges.length;
	const w = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const left = i > 0 ? edges[i] - edges[i - 1] : edges[1] - edges[0];
		const right = i < n - 1 ? edges[i + 1] - edges[i] : edges[i] - edges[i - 1];
		w[i] = 0.5 * (left + right);
	}
	return w;
}
