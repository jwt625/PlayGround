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
}

export interface Omitted {
	total: number;
	plotted: number;
	missingX: number;
	missingY: number;
	missingBoth: number;
}

export type Getter = (d: Device, p: Paper) => Metric;

const NO_FILTER = (v: number) => Number.isFinite(v);

/** Build plotted points; values that are not reported are omitted (never plotted at 0) and counted. */
export function buildPoints(devices: Device[], papers: Map<string, Paper>, gx: Getter, gy: Getter, panel = 0, positiveOnly = false): { pts: Pt[]; omitted: Omitted } {
	const pts: Pt[] = [];
	const om: Omitted = { total: devices.length, plotted: 0, missingX: 0, missingY: 0, missingBoth: 0 };
	for (const d of devices) {
		const p = papers.get(d.paper_id);
		if (!p) continue;
		const x = gx(d, p);
		const y = gy(d, p);
		if (x.v === null && y.v === null) om.missingBoth++;
		else if (x.v === null) om.missingX++;
		else if (y.v === null) om.missingY++;
		if (x.v === null || y.v === null || !NO_FILTER(x.v) || !NO_FILTER(y.v)) continue;
		if (positiveOnly && (x.v <= 0 || y.v <= 0)) continue;
		pts.push({ id: d.device_id, x: x.v, y: y.v, group: materialGroup(d.eo_material), sim: d.is_sim, qx: x.qual, qy: y.qual, panel });
		om.plotted++;
	}
	return { pts, omitted: om };
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
	min: number;
	max: number;
	median: number;
	mean: number;
}

/** Per-material range, median and mean of y; `index` is the category position (order of GROUP_ORDER present). */
export function groupStats(pts: Pt[]): GroupStat[] {
	const by = new Map<string, number[]>();
	for (const p of pts) by.set(p.group, [...(by.get(p.group) ?? []), p.y]);
	const present = GROUP_ORDER.filter((g) => by.has(g));
	return present.map((g, i) => {
		const ys = by.get(g) as number[];
		return { group: g, index: i, n: ys.length, min: Math.min(...ys), max: Math.max(...ys), median: median(ys) as number, mean: mean(ys) as number };
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
