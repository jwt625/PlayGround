/** `solveModes(problem)` contract dispatcher (C3).
 *
 * Implemented backend in this pass: 1D slab (`geometry.kind === 'slab'`),
 * the benchmark gate before any 2D release. The 2D full-vector backend is the
 * next C3.2 step and is intentionally absent rather than faked.
 */
import { solveSlab, SLAB_SOLVER_VERSION } from './slab.mjs';
import { MESH_VERSION, buildGrid2D, rasterizeRegions } from './mesh.mjs';
import { MATERIAL_DB_VERSION } from './materials.mjs';
import { solveScalar2D, FEM2D_VERSION } from './fem2d.mjs';

export const SOLVE_MODES_VERSION = 'solve-modes-1';

export function solveModes(problem) {
	if (!problem || typeof problem !== 'object') throw new TypeError('problem object required');
	const { geometry, wavelengthUm, polarization = 'TE', solver = {}, boundary = {} } = problem;
	if (!geometry) throw new TypeError('problem.geometry required');
	if (!Number.isFinite(wavelengthUm)) throw new TypeError('problem.wavelengthUm required');
	if (geometry.kind === 'slab') {
		const result = solveSlab({
			core: geometry.core,
			substrate: geometry.substrate,
			cladding: geometry.cladding,
			wavelengthUm,
			polarization,
			modeCount: solver.modeCount ?? 1,
			minPerFeature: solver.minPerFeature,
			paddingUm: solver.paddingUm,
			maxStepUm: solver.maxStepUm
		});
		return {
			schemaVersion: SOLVE_MODES_VERSION,
			solverVersion: SLAB_SOLVER_VERSION,
			materialDBVersion: MATERIAL_DB_VERSION,
			meshVersion: MESH_VERSION,
			geometry,
			boundary: { walls: 'dirichlet', ...boundary },
			forwardPowerConvention: 'positive-z with 0.5 Re(E x H*)',
			...result
		};
	}
	if (geometry.kind === 'scalar-2d') {
		const grid =
			geometry.grid ??
			buildGrid2D({ xEdges: geometry.xEdges, yEdges: geometry.yEdges });
		const nodeEps =
			geometry.nodeEps ?? rasterizeRegions(grid, geometry.regions ?? [], wavelengthUm, geometry.background ?? 'air');
		const result = solveScalar2D({
			grid,
			nodeEps,
			wavelengthUm,
			modeCount: solver.modeCount ?? 1,
			dirichlet: boundary.dirichlet,
			lanczosSteps: solver.lanczosSteps ?? 120
		});
		return {
			schemaVersion: SOLVE_MODES_VERSION,
			solverVersion: FEM2D_VERSION,
			materialDBVersion: MATERIAL_DB_VERSION,
			meshVersion: MESH_VERSION,
			geometry,
			releaseEligible: false,
			modelNote: 'scalar (quasi-TE-like) approximation; not release-eligible until the full-vector backend lands',
			boundary: { walls: 'scalar-natural-or-dirichlet', ...boundary },
			...result
		};
	}
	throw new Error(
		`solveModes: geometry.kind='${geometry.kind}' needs the 2D full-vector backend, which is not implemented in this pass`
	);
}

export { buildGrid2D };
