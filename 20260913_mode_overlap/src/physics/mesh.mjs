/** Boundary-aligned nonuniform meshes (C3.1).
 *
 * Segment boundaries become exact nodes, so every element is filled by a single
 * material. Element size starts at `smallest important feature / minPerFeature`
 * (007 initial heuristic); this is a starting mesh, never a correctness
 * certificate, so callers must also run refinement/domain checks.
 */
import { relativePermittivity, getMaterial } from './materials.mjs';

export const MESH_VERSION = 'mesh-1';

export function buildAlignedMesh1D(segments, { maxStepUm }) {
	if (!Array.isArray(segments) || segments.length === 0) throw new RangeError('segments required');
	if (!Number.isFinite(maxStepUm) || maxStepUm <= 0) throw new RangeError('maxStepUm must be positive');
	const x = [0];
	const elementMaterial = [];
	const elementLength = [];
	for (const segment of segments) {
		getMaterial(segment.material);
		const thickness = segment.thicknessUm;
		if (!Number.isFinite(thickness) || thickness <= 0) throw new RangeError('segment thickness must be positive');
		const steps = Math.max(1, Math.ceil(thickness / maxStepUm));
		const step = thickness / steps;
		for (let k = 0; k < steps; k++) {
			x.push(x[x.length - 1] + step);
			elementMaterial.push(segment.material);
			elementLength.push(step);
		}
	}
	return {
		x: Float64Array.from(x),
		elementMaterial,
		elementLength: Float64Array.from(elementLength),
		lengthUm: x[x.length - 1],
		maxStepUm
	};
}

/**
 * 1D slab: superstrate pad | core | substrate pad. Core boundaries are nodes.
 * `paddingUm` defaults to a conservative evanescent margin tied to the core.
 */
export function buildSlabMesh({
	core,
	substrate,
	cladding,
	wavelengthUm,
	minPerFeature = 12,
	paddingUm,
	maxStepUm
}) {
	getMaterial(core.material);
	getMaterial(substrate);
	getMaterial(cladding);
	const featureUm = core.thicknessUm;
	if (featureUm <= 0) throw new RangeError('core thickness must be positive');
	const padding = paddingUm ?? Math.max(2.5, 12 * featureUm);
	const step = maxStepUm ?? featureUm / minPerFeature;
	const mesh = buildAlignedMesh1D(
		[
			{ material: cladding, thicknessUm: padding },
			{ material: core.material, thicknessUm: featureUm },
			{ material: substrate, thicknessUm: padding }
		],
		{ maxStepUm: step }
	);
	const eps = new Float64Array(mesh.elementLength.length);
	for (let e = 0; e < eps.length; e++) eps[e] = relativePermittivity(mesh.elementMaterial[e], wavelengthUm);
	mesh.coreStartUm = padding;
	mesh.coreEndUm = padding + featureUm;
	mesh.featureUm = featureUm;
	mesh.eps = eps;
	mesh.wavelengthUm = wavelengthUm;
	return mesh;
}

/** Axis-aligned 2D node grid with separable trapezoidal quadrature weights. */
export function buildGrid2D({ xEdges, yEdges }) {
	if (xEdges.length < 2 || yEdges.length < 2) throw new RangeError('need at least two edges per axis');
	const nodeWeight = (edges, index) => {
		const left = index > 0 ? edges[index] - edges[index - 1] : 0;
		const right = index < edges.length - 1 ? edges[index + 1] - edges[index] : 0;
		return (left + right) / 2;
	};
	const nx = xEdges.length;
	const ny = yEdges.length;
	const x = Float64Array.from(xEdges);
	const y = Float64Array.from(yEdges);
	const weights = new Float64Array(nx * ny);
	for (let iy = 0; iy < ny; iy++) {
		for (let ix = 0; ix < nx; ix++) weights[iy * nx + ix] = nodeWeight(xEdges, ix) * nodeWeight(yEdges, iy);
	}
	return { x, y, nx, ny, weights, kind: 'grid-2d' };
}

/** Rasterize axis-aligned regions (highest priority wins) to node permittivity. */
export function rasterizeRegions(grid, regions, wavelengthUm, background = 'air') {
	const epsById = new Map();
	const lookup = (id) => {
		if (!epsById.has(id)) epsById.set(id, relativePermittivity(id, wavelengthUm));
		return epsById.get(id);
	};
	const backgroundEps = lookup(background);
	const eps = new Float64Array(grid.nx * grid.ny).fill(backgroundEps);
	const ordered = [...regions].sort((a, b) => (a.priority ?? 0) - (b.priority ?? 0));
	for (const region of ordered) {
		const regionEps = lookup(region.material);
		for (let iy = 0; iy < grid.ny; iy++) {
			const y = grid.y[iy];
			if (y < region.y0 || y > region.y1) continue;
			for (let ix = 0; ix < grid.nx; ix++) {
				const x = grid.x[ix];
				if (x < region.x0 || x > region.x1) continue;
				eps[iy * grid.nx + ix] = regionEps;
			}
		}
	}
	return eps;
}
