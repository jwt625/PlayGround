import type { Atlas, Device, Filters, Paper, Qual, RepMode, SortKey } from './types';

export const EM_DASH = '—';

export function defaultFilters(): Filters {
	return {
		q: '',
		materials: [],
		classes: [],
		platforms: [],
		sourceTypes: [],
		regions: [],
		countries: [],
		yearMin: null,
		yearMax: null,
		measuredOnly: false,
		hasSim: false,
		rep: 'default',
		allDevices: false
	};
}

/** Three significant figures; null/NaN -> em dash. Never renders a missing value as 0. */
export function fmt(v: number | null | undefined, sig = 3): string {
	if (v === null || v === undefined || Number.isNaN(v)) return EM_DASH;
	if (v === 0) return '0';
	const a = Math.abs(v);
	if (a < 1e-3 || a >= 1e5) {
		const s = v.toExponential(sig - 1);
		const [m, e] = s.split('e');
		const mant = m.includes('.') ? m.replace(/0+$/, '').replace(/\.$/, '') : m;
		return `${mant}e${e.replace('+', '')}`;
	}
	return String(Number(v.toPrecision(sig)));
}

export function qualPrefix(q: Qual | null | undefined): string {
	return q === 'lt' ? '<' : q === 'gt' ? '>' : q === 'approx' ? '~' : '';
}

export function qualWord(q: Qual | null | undefined): string {
	return q === 'lt' ? 'upper bound (less than)' : q === 'gt' ? 'lower bound (greater than)' : q === 'approx' ? 'approximate' : '';
}

export const BASIS_MARK: Record<string, string> = {
	simulated: 's',
	predicted: 'p',
	extracted_from_figure: 'f',
	derived: 'd',
	author_estimate: 'e',
	design_target: 't'
};

export const BASIS_TIP: Record<string, string> = {
	measured: 'measured',
	simulated: 'simulated',
	predicted: 'predicted',
	extracted_from_figure: 'extracted from a figure',
	derived: 'derived',
	author_estimate: 'author estimate',
	design_target: 'design target'
};

export function median(xs: number[]): number | null {
	if (!xs.length) return null;
	const s = [...xs].sort((a, b) => a - b);
	const m = s.length >> 1;
	return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function mean(xs: number[]): number | null {
	return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

/** Pareto frontier. `maxX` true: larger x is better; `minY` true: smaller y is better. Returns points sorted by x. */
export function paretoFront<T extends { x: number; y: number }>(pts: T[], maxX = true, minY = true): T[] {
	const sorted = [...pts].sort((a, b) => (maxX ? b.x - a.x : a.x - b.x) || (minY ? a.y - b.y : b.y - a.y));
	const out: T[] = [];
	let best = minY ? Infinity : -Infinity;
	for (const p of sorted) {
		if (minY ? p.y < best : p.y > best) {
			out.push(p);
			best = p.y;
		}
	}
	return out.sort((a, b) => a.x - b.x);
}

export function countBy<T>(items: T[], key: (t: T) => string | string[]): Map<string, number> {
	const m = new Map<string, number>();
	for (const it of items) {
		const k = key(it);
		for (const kk of Array.isArray(k) ? k : [k]) m.set(kk, (m.get(kk) ?? 0) + 1);
	}
	return m;
}

// ---------- headline device metrics ----------

export interface Metric {
	v: number | null;
	qual: Qual | null;
	derived: boolean;
	basis: string | null;
	field: string | null;
	/** Conflicting or approximate input bounds have no defensible single direction. */
	boundUnresolved?: boolean;
	note?: string;
	modelled?: boolean;
}

const NONE: Metric = { v: null, qual: null, derived: false, basis: null, field: null };

function raw(d: Device, field: string, basisKey: string | null): Metric {
	const v = d[field];
	if (typeof v !== 'number' || !Number.isFinite(v)) return NONE;
	const ev = d.evidence[field];
	return {
		v,
		qual: d.qualifiers[field] ?? null,
		derived: !!ev?.derived,
		basis: ev?.basis ?? (basisKey ? d.headline_basis[basisKey] : null) ?? null,
		field
	};
}

export function vpil(d: Device): Metric {
	const b = d.vpil_best;
	if (!b) return NONE;
	const field = b.source === 'reported_dc' ? 'vpil_dc_vcm' : b.source === 'reported_rf' ? 'vpil_rf_vcm' : 'vpil_dc_vcm_derived';
	if (b.source === 'derived_dc') return derivedMetric(d, 'vpil_dc_vcm_derived');
	return { ...raw(d, field, 'vpil'), qual: b.qualifier ?? null };
}
export function vpi(d: Device): Metric {
	return typeof d.vpi_dc_v === 'number' ? raw(d, 'vpi_dc_v', 'vpi') : raw(d, 'vpi_rf_v', 'vpi');
}
export const bw3db = (d: Device): Metric => raw(d, 'bw3db_ghz', 'bw3db');
export const ilOnchip = (d: Device): Metric => raw(d, 'il_onchip_db', 'il_onchip');
export const ilF2f = (d: Device): Metric => raw(d, 'il_fiber_to_fiber_db', 'il_f2f');
export const rfLoss = (d: Device): Metric => raw(d, 'rf_loss_db_per_cm', 'rf_loss');
export const lengthMm = (d: Device): Metric => raw(d, 'length_mm', null);
export const maxBaud = (d: Device): Metric => raw(d, 'max_baud_gbd', 'rate');
export const maxRate = (d: Device): Metric => raw(d, 'max_line_rate_gbps', 'rate');

export function derivedMetric(d: Device, key: 'vpil_dc_vcm_derived' | 'vpi_il_vdb' | 'fom' | 'il_rf_total_db'): Metric {
	const x = d.derived[key];
	if (!x || !Number.isFinite(x.value)) return NONE;
	const inputs = x.inputs.map(field => ({ field, q: d.qualifiers[field] })).filter(i => i.q);
	const directions = inputs.filter(i => i.q !== 'approx').map(({ field, q }) =>
		key === 'fom' && field !== 'bw3db_ghz' ? (q === 'lt' ? 'gt' : 'lt') : q);
	const approx = inputs.some(i => i.q === 'approx');
	// Monotonic propagation here applies to the documented positive-input products
	// and FOM. A zero/negative factor or opposing directions cannot give one bound.
	const positive = x.inputs.every(field => typeof d[field] === 'number' && (d[field] as number) > 0);
	const boundUnresolved = directions.length > 0 && (!positive || approx || new Set(directions).size > 1);
	const qual = boundUnresolved ? null : (directions[0] ?? (approx ? 'approx' : null));
	const note = inputs.length ? `Input qualifiers: ${inputs.map(i => `${i.field}:${i.q}`).join(', ')}.${boundUnresolved ? ' Nominal calculation only; these inputs do not determine a single bound.' : ''}` : undefined;
	const modelled = x.inputs.some(field => {
		const rowBasis = field.startsWith('vpi') ? d.vpi_basis : field.startsWith('bw') ? d.bw_basis : field.startsWith('il_') ? d.il_basis : null;
		return ['simulated', 'predicted', 'design_target'].includes(String(d.evidence[field]?.basis ?? rowBasis ?? ''));
	});
	return { v: x.value, qual, derived: true, basis: 'derived', field: key, boundUnresolved, note, modelled };
}

export function metricValue(d: Device, key: string): number | null {
	switch (key) {
		case 'vpil': return vpil(d).v;
		case 'vpi': return vpi(d).v;
		case 'bw3db': return bw3db(d).v;
		case 'il_onchip': return ilOnchip(d).v;
		case 'il_f2f': return ilF2f(d).v;
		case 'rf_loss': return rfLoss(d).v;
		case 'length': return lengthMm(d).v;
		case 'baud': return maxBaud(d).v;
		case 'rate': return maxRate(d).v;
		case 'vpi_il': return derivedMetric(d, 'vpi_il_vdb').v;
		case 'fom': return derivedMetric(d, 'fom').v;
		default: return null;
	}
}

// ---------- representative device ----------

type RepKey = [number, number, number, string];

function repKey(d: Device): RepKey {
	const fom = d.derived.fom?.value;
	const vp = d.vpil_best?.value;
	return [-d.derived.completeness.value, -(fom ?? -Infinity), vp ?? Infinity, d.device_id];
}

function cmpKeys(a: RepKey, b: RepKey): number {
	for (let i = 0; i < 3; i++) {
		const x = a[i] as number;
		const y = b[i] as number;
		if (x !== y) return x < y ? -1 : 1;
	}
	return a[3] < b[3] ? -1 : a[3] > b[3] ? 1 : 0;
}

export function pickRep(devs: Device[], mode: RepMode): Device | null {
	if (!devs.length) return null;
	const def = [...devs].sort((a, b) => cmpKeys(repKey(a), repKey(b)))[0];
	if (mode === 'default') return def;
	const metric: Record<Exclude<RepMode, 'default'>, [(d: Device) => number | null, boolean]> = {
		lowest_vpil: [(d) => d.vpil_best?.value ?? null, false],
		highest_bw: [(d) => (typeof d.bw3db_ghz === 'number' ? d.bw3db_ghz : null), true],
		highest_fom: [(d) => d.derived.fom?.value ?? null, true],
		lowest_vpi_il: [(d) => d.derived.vpi_il_vdb?.value ?? null, false]
	};
	const [f, hi] = metric[mode];
	const cand = devs.filter((d) => f(d) !== null);
	if (!cand.length) return def;
	cand.sort((a, b) => {
		const x = f(a) as number;
		const y = f(b) as number;
		if (x !== y) return hi ? y - x : x - y;
		return cmpKeys(repKey(a), repKey(b));
	});
	return cand[0];
}

// ---------- filtering ----------

export interface Index {
	paperById: Map<string, Paper>;
	devById: Map<string, Device>;
	devsByPaper: Map<string, Device[]>;
	search: Map<string, string>;
}

export function buildIndex(a: Atlas): Index {
	const paperById = new Map(a.papers.map((p) => [p.paper_id, p]));
	const devById = new Map(a.devices.map((d) => [d.device_id, d]));
	const devsByPaper = new Map<string, Device[]>();
	for (const d of a.devices) {
		const l = devsByPaper.get(d.paper_id) ?? [];
		l.push(d);
		devsByPaper.set(d.paper_id, l);
	}
	const lbl = (en: string, v: string | null) => a.enums[en]?.find((e) => e.value === v)?.label ?? v ?? '';
	const search = new Map<string, string>();
	for (const p of a.papers) {
		const devs = devsByPaper.get(p.paper_id) ?? [];
		const parts = [
			p.paper_id,
			p.label,
			p.title,
			p.authors.join(' '),
			p.venue ?? '',
			p.doi ?? '',
			p.arxiv_id ?? '',
			p.orgs_affil.map((o) => o.org_name).join(' '),
			p.orgs_fab.map((o) => o.org_name).join(' '),
			p.research_groups.join(' '),
			p.countries_derived.map((c) => lbl('country', c)).join(' '),
			devs.map((d) => `${d.device_label} ${lbl('eo_material', d.eo_material)} ${lbl('device_class', d.device_class)} ${lbl('waveguide_platform', d.waveguide_platform)} ${d.tags.join(' ')}`).join(' ')
		];
		search.set(p.paper_id, parts.join(' ').toLowerCase());
	}
	return { paperById, devById, devsByPaper, search };
}

export function deviceMatches(d: Device, f: Filters): boolean {
	if (f.materials.length && !f.materials.includes(d.eo_material)) return false;
	if (f.classes.length && !f.classes.includes(d.device_class)) return false;
	if (f.platforms.length && !f.platforms.includes(d.waveguide_platform ?? 'unspecified')) return false;
	if (f.measuredOnly && !d.measured_only) return false;
	return true;
}

export function paperMatches(p: Paper, f: Filters, idx: Index): boolean {
	if (f.yearMin !== null && p.year < f.yearMin) return false;
	if (f.yearMax !== null && p.year > f.yearMax) return false;
	if (f.sourceTypes.length && !f.sourceTypes.includes(p.source_type)) return false;
	if (f.regions.length && !p.regions_derived.some((r) => f.regions.includes(r))) return false;
	if (f.countries.length && !p.countries_derived.some((c) => f.countries.includes(c))) return false;
	if (f.hasSim && !p.has_sim) return false;
	const q = f.q.trim().toLowerCase();
	if (q) {
		const hay = idx.search.get(p.paper_id) ?? '';
		for (const tok of q.split(/\s+/)) if (!hay.includes(tok)) return false;
	}
	return true;
}

export interface View {
	papers: Paper[];
	/** all devices passing filters (device-level) of passing papers */
	devices: Device[];
	/** one device per paper by the rep mode */
	reps: Map<string, Device>;
	matched: Set<string>;
}

export function applyFilters(a: Atlas, f: Filters, idx: Index): View {
	const papers: Paper[] = [];
	const devices: Device[] = [];
	const reps = new Map<string, Device>();
	const matched = new Set<string>();
	for (const p of a.papers) {
		if (!paperMatches(p, f, idx)) continue;
		const ds = (idx.devsByPaper.get(p.paper_id) ?? []).filter((d) => deviceMatches(d, f));
		if (!ds.length) continue;
		papers.push(p);
		for (const d of ds) {
			devices.push(d);
			matched.add(d.device_id);
		}
		const r = pickRep(ds, f.rep);
		if (r) reps.set(p.paper_id, r);
	}
	return { papers, devices, reps, matched };
}

/** Devices to draw in charts: all matching devices or only each paper's representative. */
export function chartDevices(v: View, allDevices: boolean): Device[] {
	return allDevices ? v.devices : [...v.reps.values()];
}

// ---------- sorting ----------

export type SortVal = number | string | null;

export function compareVals(a: SortVal, b: SortVal, dir: 'asc' | 'desc'): number {
	if (a === null && b === null) return 0;
	if (a === null) return 1; // empty always last
	if (b === null) return -1;
	let r: number;
	if (typeof a === 'number' && typeof b === 'number') r = a - b;
	else r = String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
	return dir === 'asc' ? r : -r;
}

export function sortRows<T>(rows: T[], keys: SortKey[], val: (row: T, key: string) => SortVal): T[] {
	if (!keys.length) return rows;
	return rows
		.map((r, i) => ({ r, i }))
		.sort((x, y) => {
			for (const k of keys) {
				const c = compareVals(val(x.r, k.key), val(y.r, k.key), k.dir);
				if (c) return c;
			}
			return x.i - y.i;
		})
		.map((x) => x.r);
}

/** none -> asc -> desc -> none. With `multi`, keeps other sort keys. */
export function cycleSort(keys: SortKey[], key: string, multi: boolean): SortKey[] {
	const cur = keys.find((k) => k.key === key);
	const others = multi ? keys.filter((k) => k.key !== key) : [];
	if (!cur) return [...others, { key, dir: 'asc' }];
	if (cur.dir === 'asc') return multi ? keys.map((k) => (k.key === key ? { key, dir: 'desc' as const } : k)) : [{ key, dir: 'desc' }];
	return others;
}

// ---------- URL hash ----------

export interface HashExtras {
	sort: SortKey[];
}

const list = (s: string | null): string[] => (s ? s.split(',').filter(Boolean) : []);

export function toHash(f: Filters, extras: HashExtras = { sort: [] }): string {
	const d = defaultFilters();
	const p = new URLSearchParams();
	if (f.q) p.set('q', f.q);
	if (f.materials.length) p.set('m', f.materials.join(','));
	if (f.classes.length) p.set('c', f.classes.join(','));
	if (f.platforms.length) p.set('p', f.platforms.join(','));
	if (f.sourceTypes.length) p.set('s', f.sourceTypes.join(','));
	if (f.regions.length) p.set('r', f.regions.join(','));
	if (f.countries.length) p.set('k', f.countries.join(','));
	if (f.yearMin !== null) p.set('y0', String(f.yearMin));
	if (f.yearMax !== null) p.set('y1', String(f.yearMax));
	if (f.measuredOnly) p.set('mo', '1');
	if (f.hasSim) p.set('sim', '1');
	if (f.rep !== d.rep) p.set('rep', f.rep);
	if (f.allDevices) p.set('all', '1');
	if (extras.sort.length) p.set('sort', extras.sort.map((k) => `${k.key}:${k.dir}`).join(','));
	return p.toString();
}

const REP_MODES: RepMode[] = ['default', 'lowest_vpil', 'highest_bw', 'highest_fom', 'lowest_vpi_il'];

export function fromHash(hash: string): { filters: Filters; extras: HashExtras } {
	const p = new URLSearchParams(hash.replace(/^#/, ''));
	const f = defaultFilters();
	f.q = p.get('q') ?? '';
	f.materials = list(p.get('m'));
	f.classes = list(p.get('c'));
	f.platforms = list(p.get('p'));
	f.sourceTypes = list(p.get('s'));
	f.regions = list(p.get('r'));
	f.countries = list(p.get('k'));
	const y0 = Number(p.get('y0'));
	const y1 = Number(p.get('y1'));
	f.yearMin = p.get('y0') && Number.isFinite(y0) ? y0 : null;
	f.yearMax = p.get('y1') && Number.isFinite(y1) ? y1 : null;
	f.measuredOnly = p.get('mo') === '1';
	f.hasSim = p.get('sim') === '1';
	const rep = p.get('rep') as RepMode | null;
	f.rep = rep && REP_MODES.includes(rep) ? rep : 'default';
	f.allDevices = p.get('all') === '1';
	const sort: SortKey[] = list(p.get('sort'))
		.map((s) => {
			const [key, dir] = s.split(':');
			return { key, dir: dir === 'desc' ? ('desc' as const) : ('asc' as const) };
		})
		.filter((k) => k.key);
	return { filters: f, extras: { sort } };
}

// ---------- CSV ----------

export function csvEscape(s: string): string {
	return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(header: string[], rows: string[][]): string {
	return [header, ...rows].map((r) => r.map(csvEscape).join(',')).join('\n') + '\n';
}

export function enumLabel(a: Atlas, en: string, v: string | null | undefined): string {
	if (v === null || v === undefined || v === '') return EM_DASH;
	return a.enums[en]?.find((e) => e.value === v)?.label ?? v;
}

export function firstAuthorLabel(p: Paper): string {
	return p.label;
}
