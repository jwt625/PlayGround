/** Build the nominal + variant waveguide-mode gallery (experimental scalar quasi-TE).
 *
 * Solves each rectangular guide at nominal and variant dimensions with the
 * validated scalar 2D solver, keeps the well-confined modes (fundamental plus
 * higher order), and writes a compact display bundle (index geometry + signed
 * mode field) for the gallery and on-the-fly overlap questions. These are NOT
 * full-vector and are not release-eligible; the bundle says so.
 */
import { readFile, mkdir, writeFile, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildGrid2D, rasterizeRegions, MESH_VERSION } from '../src/physics/mesh.mjs';
import { solveScalar2D, FEM2D_VERSION } from '../src/physics/fem2d.mjs';
import { MATERIAL_DB_VERSION, refractiveIndex } from '../src/physics/materials.mjs';
import { canonicalJSON } from '../src/storage/cache.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const MATERIAL_IDS = { Si: 'si', SiO2: 'sio2', Si3N4: 'sin' };
// How many confined modes to publish per preset (multimode guides get more).
const MULTIMODE_ORDER = { 'soi-wide': 3, 'soi-rib': 2, 'sin-thick': 3, 'sin-medium': 2 };
// Real geometric variants (independently solved) so on-the-fly pairing has
// many distinct guides and mode orders to compare.
const VARIANTS = {
	'soi-strip': [
		{ id: 'soi-strip-w035', width_um: 0.35 },
		{ id: 'soi-strip-w040', width_um: 0.4 },
		{ id: 'soi-strip-w050', width_um: 0.5 },
		{ id: 'soi-strip-w055', width_um: 0.55 },
		{ id: 'soi-strip-h019', height_um: 0.19 }
	],
	'soi-wide': [
		{ id: 'soi-wide-w070', width_um: 0.7, maxModes: 3 },
		{ id: 'soi-wide-w110', width_um: 1.1, maxModes: 4 },
		{ id: 'soi-wide-w140', width_um: 1.4, maxModes: 4 }
	],
	'soi-rib': [
		{ id: 'soi-rib-w050', width_um: 0.5, slab: 0.09 },
		{ id: 'soi-rib-w080', width_um: 0.8, slab: 0.12 }
	],
	'sin-thick': [
		{ id: 'sin-thick-w120', width_um: 1.2, maxModes: 3 },
		{ id: 'sin-thick-h065', height_um: 0.65, maxModes: 3 },
		{ id: 'sin-thick-h095', height_um: 0.95, maxModes: 4 }
	],
	'sin-medium': [
		{ id: 'sin-medium-w080', width_um: 0.8 },
		{ id: 'sin-medium-w100', width_um: 1.0 },
		{ id: 'sin-medium-w140', width_um: 1.4 },
		{ id: 'sin-medium-h030', height_um: 0.3 }
	],
	'sin-thin-wide': [
		{ id: 'sin-thin-wide-w200', width_um: 2.0 },
		{ id: 'sin-thin-wide-h020', height_um: 0.2 }
	],
	'sin-taper-endpoint': [
		{ id: 'sin-taper-w030', width_um: 0.3 },
		{ id: 'sin-taper-h025', height_um: 0.25 }
	]
};

function edges(center, half, nodes) {
	return Array.from({ length: nodes + 1 }, (_, i) => center - half + (2 * half * i) / nodes);
}

function resample(grid, values, nxOut, nyOut) {
	const out = new Array(nxOut * nyOut);
	for (let j = 0; j < nyOut; j++) {
		const fy = (j / (nyOut - 1)) * (grid.ny - 1);
		const j0 = Math.min(grid.ny - 1, Math.floor(fy));
		const j1 = Math.min(grid.ny - 1, j0 + 1);
		const ty = fy - j0;
		for (let i = 0; i < nxOut; i++) {
			const fx = (i / (nxOut - 1)) * (grid.nx - 1);
			const i0 = Math.min(grid.nx - 1, Math.floor(fx));
			const i1 = Math.min(grid.nx - 1, i0 + 1);
			const tx = fx - i0;
			const v00 = values[j0 * grid.nx + i0];
			const v10 = values[j0 * grid.nx + i1];
			const v01 = values[j1 * grid.nx + i0];
			const v11 = values[j1 * grid.nx + i1];
			out[j * nxOut + i] = Number(
				((v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty).toPrecision(4)
			);
		}
	}
	return out;
}

function solveGuide(descriptor) {
	const { width, height, slab, material, background, wavelengthUm, maxModes, modeLabel } = descriptor;
	const regions = [{ x0: -width / 2, x1: width / 2, y0: -height / 2, y1: height / 2, material, priority: 2 }];
	if (slab) regions.push({ x0: -1e3, x1: 1e3, y0: -height / 2, y1: -height / 2 + slab, material, priority: 1 });
	const pad = Math.max(1.0, 2.5 * Math.max(width, height));
	const halfX = pad + width / 2;
	const halfY = pad + height / 2;
	const step = Math.max(Math.min(width, height) / 12, (2 * Math.max(halfX, halfY)) / 130);
	const nx = Math.max(49, Math.ceil((2 * halfX) / step) + 1);
	const ny = Math.max(41, Math.ceil((2 * halfY) / step) + 1);
	const grid = buildGrid2D({ xEdges: edges(0, halfX, nx - 1), yEdges: edges(0, halfY, ny - 1) });
	const nodeEps = rasterizeRegions(grid, regions, wavelengthUm, background);
	const result = solveScalar2D({ grid, nodeEps, wavelengthUm, modeCount: 5, lanczosSteps: 200 });
	if (result.modes.length === 0) return null;
	const weights = grid.weights;
	const scored = result.modes.map((mode) => {
		const psi = mode.scalar.re;
		let total = 0;
		let core = 0;
		for (let j = 0; j < grid.ny; j++) {
			for (let i = 0; i < grid.nx; i++) {
				const n = j * grid.nx + i;
				const energy = weights[n] * psi[n] * psi[n];
				total += energy;
				if (Math.abs(grid.x[i]) <= width && Math.abs(grid.y[j]) <= height) core += energy;
			}
		}
		return { mode, confinement: total > 0 ? core / total : 0 };
	});
	scored.sort((a, b) => b.mode.neff - a.mode.neff);
	const qualified = scored.filter((item) => item.confinement >= 0.6 && item.mode.residual < 1e-4);
	const selected = qualified.length > 0 ? qualified.slice(0, maxModes) : [scored[0]];
	const displayX = 80;
	const displayY = 60;
	const toIndex = (id) => Number(refractiveIndex(id, wavelengthUm).toPrecision(4));
	const indexRegions = regions.map((region) => ({
		x0: region.x0,
		x1: region.x1,
		y0: region.y0,
		y1: region.y1,
		index: toIndex(region.material),
		material: region.material
	}));
	return selected.map((item, order) => ({
		modeLabel: order === 0 ? modeLabel : `higher-order ${order}`,
		neff: Number(item.mode.neff.toPrecision(5)),
		confinement: Number(item.confinement.toPrecision(4)),
		residual: Number(item.mode.residual.toExponential(2)),
		field: resample(grid, item.mode.scalar.re, displayX, displayY),
		axes: { x0: grid.x[0], x1: grid.x[grid.nx - 1], y0: grid.y[0], y1: grid.y[grid.ny - 1], nx: displayX, ny: displayY },
		backgroundIndex: toIndex(background),
		indexRegions
	}));
}

function pushDescriptor(entries, descriptor) {
	const solved = solveGuide(descriptor);
	if (!solved) return;
	solved.forEach((mode, order) => {
		entries.push({
			id: order === 0 ? descriptor.id : `${descriptor.id}-m${order}`,
			name: order === 0 ? descriptor.id : `${descriptor.id} (mode ${order})`,
			order,
			category: descriptor.category,
			modeLabel: mode.modeLabel,
			materialLabel: descriptor.materials.join(' / '),
			cladding: descriptor.cladding,
			wavelengthUm: descriptor.wavelengthUm,
			status: 'experimental-scalar-quasi-te',
			neff: mode.neff,
			confinement: mode.confinement,
			residual: mode.residual,
			dimensions: { widthUm: descriptor.width, heightUm: descriptor.height, slabUm: descriptor.slab ?? null },
			core: { x0: -descriptor.width / 2, x1: descriptor.width / 2, y0: -descriptor.height / 2, y1: descriptor.height / 2 },
			axes: mode.axes,
			backgroundIndex: mode.backgroundIndex,
			indexRegions: mode.indexRegions,
			field: mode.field
		});
	});
}

export async function buildModeGallery(output = join(root, 'static/cache/modes'), { ids } = {}) {
	const catalog = JSON.parse(await readFile(join(root, 'data/presets.json'), 'utf8'));
	const wanted = ids ?? ['soi-strip', 'soi-wide', 'soi-rib', 'sin-thick', 'sin-medium', 'sin-thin-wide', 'sin-taper-endpoint'];
	const entries = [];
	for (const preset of catalog.presets) {
		if (!wanted.includes(preset.id)) continue;
		const descriptor = {
			id: preset.id,
			category: preset.category,
			materials: preset.materials,
			cladding: preset.cladding ?? 'SiO2',
			wavelengthUm: preset.wavelength_um,
			width: preset.nominal.width_um,
			height: preset.nominal.height_um,
			slab: preset.nominal.slab_um ?? null,
			material: MATERIAL_IDS[preset.materials[0]],
			background: 'sio2',
			maxModes: MULTIMODE_ORDER[preset.id] ?? 1,
			modeLabel: preset.mode ?? 'fundamental'
		};
		pushDescriptor(entries, descriptor);
		for (const variant of VARIANTS[preset.id] ?? []) {
			pushDescriptor(entries, {
				...descriptor,
				id: variant.id,
				width: variant.width_um ?? descriptor.width,
				height: variant.height_um ?? descriptor.height,
				slab: variant.slab_um ?? descriptor.slab,
				maxModes: variant.maxModes ?? 1
			});
		}
	}
	const bundle = {
		schemaVersion: 'mode-gallery-1',
		provenance: {
			solverVersion: FEM2D_VERSION,
			materialDBVersion: MATERIAL_DB_VERSION,
			meshVersion: MESH_VERSION,
			source: 'data/presets.json'
		},
		note: 'Scalar quasi-TE finite-element modes at nominal and variant dimensions, fundamental plus confined higher-order modes. Experimental, not full-vector; SiN index is a process placeholder. Do not read as released neff.',
		entries
	};
	await mkdir(output, { recursive: true });
	const text = canonicalJSON(bundle) + '\n';
	const temporary = `${join(output, 'manifest.json')}.${process.pid}.tmp`;
	await writeFile(temporary, text);
	await rename(temporary, join(output, 'manifest.json'));
	return { entries: entries.length, output, bytes: text.length };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	console.log(JSON.stringify(await buildModeGallery(process.argv[2] ? process.argv[2] : undefined), null, 2));
}
