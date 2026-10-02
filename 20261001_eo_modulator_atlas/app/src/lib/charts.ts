import { materialGroup, GROUP_ORDER } from './colors';
import {
	bw3db,
	derivedMetric,
	ilOnchip,
	lengthMm,
	maxBaud,
	maxRate,
	mean,
	median,
	vpil,
	type Metric,
	countBy
} from './logic';
import { paretoFront } from './logic';
import type { Device, Paper, Qual } from './types';
import type { View } from './logic';

export interface Pt {
	id: string;
	x: number;
	y: number;
	group: string;
	sim: boolean;
	qx: Qual | null;
	qy: Qual | null;
	panel: number;
	/** Voltage convention and DC/RF context; null means comparison is unspecified. */
	comparison: string | null;
}

export interface Omitted {
	total: number;
	plotted: number;
	missingX: number;
	missingY: number;
	missingBoth: number;
	invalid: number;
	nonpositive: number;
	uncertain: number;
}

export type Getter = (d: Device, p: Paper) => Metric;

const MODEL_BASES = new Set(['simulated', 'predicted', 'design_target']);

/** Build plotted points; values that are not reported are omitted (never plotted at 0) and counted. */
export function buildPoints(devices: Device[], papers: Map<string, Paper>, gx: Getter, gy: Getter, panel = 0, axes: { xLog?: boolean; yLog?: boolean } = {}): { pts: Pt[]; omitted: Omitted } {
	const pts: Pt[] = [];
	const om: Omitted = { total: devices.length, plotted: 0, missingX: 0, missingY: 0, missingBoth: 0, invalid: 0, nonpositive: 0, uncertain: 0 };
	for (const d of devices) {
		const p = papers.get(d.paper_id);
		if (!p) { om.invalid++; continue; }
		const x = gx(d, p);
		const y = gy(d, p);
		if (x.v === null && y.v === null) om.missingBoth++;
		else if (x.v === null) om.missingX++;
		else if (y.v === null) om.missingY++;
		if (x.v === null || y.v === null) continue;
		if (!Number.isFinite(x.v) || !Number.isFinite(y.v)) { om.invalid++; continue; }
		if (x.boundUnresolved || y.boundUnresolved) { om.uncertain++; continue; }
		if ((axes.xLog && x.v <= 0) || (axes.yLog && y.v <= 0)) { om.nonpositive++; continue; }
		let voltage = [x.field, y.field].find(f => f?.startsWith('vpi'));
		if (voltage === 'vpi_il_vdb') voltage = d.derived.vpi_il_vdb?.inputs.find(f => f.startsWith('vpi'));
		const rf = voltage?.includes('_rf_');
		const comparison = voltage && d.vpi_convention && d.vpi_convention !== 'unspecified' && (!rf || typeof d.vpi_rf_freq_ghz === 'number')
			? `${d.vpi_convention} / ${rf ? `RF ${d.vpi_rf_freq_ghz} GHz` : 'DC'}` : null;
		pts.push({ id: d.device_id, x: x.v, y: y.v, group: materialGroup(d.eo_material), sim: !!x.modelled || !!y.modelled || MODEL_BASES.has(x.basis ?? '') || MODEL_BASES.has(y.basis ?? ''), qx: x.qual, qy: y.qual, panel, comparison });
		om.plotted++;
	}
	return { pts, omitted: om };
}

/** Nominal frontiers never use bound thresholds or mix voltage conventions. */
export function nominalFrontiers(pts: Pt[]): Pt[][] {
	const by = new Map<string, Pt[]>();
	for (const p of pts) {
		if (p.qx || p.qy || p.sim || !p.comparison) continue;
		by.set(p.comparison, [...(by.get(p.comparison) ?? []), p]);
	}
	return [...by.values()].map(group => paretoFront(group, true, true));
}

export const yearOf: Getter = (_d, p) => ({ v: p.year, qual: null, derived: false, basis: null, field: 'year' });
export const gVpil: Getter = (d) => vpil(d);
export const gBw: Getter = (d) => bw3db(d);
export const gIl: Getter = (d) => ilOnchip(d);
export const gLen: Getter = (d) => lengthMm(d);
export const gBaud: Getter = (d) => maxBaud(d);
export const gRate: Getter = (d) => maxRate(d);
export const gVpiIl: Getter = (d) => derivedMetric(d, 'vpi_il_vdb');

export interface GroupStat {
	group: string;
	index: number;
	n: number;
	total: number;
	min: number | null;
	max: number | null;
	median: number | null;
	mean: number | null;
}

/** Per-material range, median and mean of y; `index` is the category position (order of GROUP_ORDER present). */
export function groupStats(pts: Pt[]): GroupStat[] {
	const by = new Map<string, number[]>();
	for (const p of pts) {
		if (!by.has(p.group)) by.set(p.group, []);
		if (!p.qx && !p.qy && !p.sim) by.get(p.group)!.push(p.y);
	}
	const present = GROUP_ORDER.filter((g) => by.has(g));
	return present.map((g, i) => {
		const contexts = new Set(pts.filter(p => p.group === g && !p.qx && !p.qy && !p.sim).map(p => p.comparison));
		const ys = contexts.size === 1 && !contexts.has(null) ? by.get(g) as number[] : [];
		return { group: g, index: i, n: ys.length, total: pts.filter(p => p.group === g).length, min: ys.length ? Math.min(...ys) : null, max: ys.length ? Math.max(...ys) : null, median: median(ys), mean: mean(ys) };
	});
}

/** Deterministic jitter in [-1, 1] from an id string. */
export function jitter(id: string): number {
	let h = 2166136261;
	for (let i = 0; i < id.length; i++) {
		h ^= id.charCodeAt(i);
		h = Math.imul(h, 16777619);
	}
	return (((h >>> 0) % 2001) - 1000) / 1000;
}

export function papersPerYearByGroup(view: View): { years: number[]; groups: string[]; counts: Map<string, Map<number, number>> } {
	const counts = new Map<string, Map<number, number>>();
	const years = new Set<number>();
	for (const p of view.papers) {
		const rep = view.reps.get(p.paper_id);
		if (!rep) continue;
		const g = materialGroup(rep.eo_material);
		const m = counts.get(g) ?? new Map<number, number>();
		m.set(p.year, (m.get(p.year) ?? 0) + 1);
		counts.set(g, m);
		years.add(p.year);
	}
	const ys = [...years].sort((a, b) => a - b);
	const full: number[] = [];
	if (ys.length) for (let y = ys[0]; y <= ys[ys.length - 1]; y++) full.push(y);
	return { years: full, groups: GROUP_ORDER.filter((g) => counts.has(g)), counts };
}

export function topCounts(map: Map<string, number>, n: number | null): [string, number][] {
	const e = [...map.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
	return n === null ? e : e.slice(0, n);
}

export function orgCounts(papers: Paper[], kind: 'affil' | 'fab'): Map<string, number> {
	return countBy(papers, (p) => (kind === 'affil' ? p.orgs_affil : p.orgs_fab).map((o) => o.org_name));
}

export function countryCounts(papers: Paper[], countryName: (c: string) => string): Map<string, number> {
	return countBy(papers, (p) => p.countries_derived.map(countryName));
}

export const CORE_LABELS: Record<string, string> = {
	vpi: 'Vpi or Vpi*L',
	bw3db: '3 dB BW',
	il_onchip: 'On-chip IL',
	rf_loss: 'RF loss',
	z0: 'Z0',
	n_rf: 'n_RF',
	ng: 'n_g'
};

export function completenessMatrix(view: View, core: string[]): { rows: { paper: Paper; dev: Device }[]; z: number[][] } {
	const rows = view.papers
		.map((p) => ({ paper: p, dev: view.reps.get(p.paper_id) as Device }))
		.filter((r) => r.dev)
		.sort((a, b) => b.dev.derived.completeness.value - a.dev.derived.completeness.value || a.paper.label.localeCompare(b.paper.label));
	const z = rows.map((r) => core.map((c) => (r.dev.derived.completeness.reported.includes(c) ? 1 : 0)));
	return { rows, z };
}
