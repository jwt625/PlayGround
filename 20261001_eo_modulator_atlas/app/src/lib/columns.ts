import {
	EM_DASH,
	BASIS_TIP,
	bw3db,
	derivedMetric,
	enumLabel,
	fmt,
	ilF2f,
	ilOnchip,
	lengthMm,
	maxBaud,
	maxRate,
	qualPrefix,
	qualWord,
	rfLoss,
	vpi,
	vpil,
	type Metric,
	type SortVal
} from './logic';
import type { Atlas, Device, Paper } from './types';

export interface Cell {
	text: string;
	/** numeric value for CSV */
	num?: number | null;
	qual?: string;
	derived?: boolean;
	basis?: string | null;
	tip?: string;
	href?: string;
	muted?: boolean;
	multiline?: boolean;
}

export interface ColDef {
	id: string;
	label: string;
	unit?: string;
	tip: string;
	width: number;
	num?: boolean;
	level: 'paper' | 'device';
	defaultOn: boolean;
}

export const COLS: ColDef[] = [
	{ id: 'paper', label: 'Paper', tip: 'First author, year. Hover for the full title.', width: 150, level: 'paper', defaultOn: true },
	{ id: 'year', label: 'Year', tip: 'Publication year', width: 52, num: true, level: 'paper', defaultOn: true },
	{ id: 'material', label: 'EO material', tip: 'Electro-optic material', width: 150, level: 'device', defaultOn: true },
	{ id: 'class', label: 'Class', tip: 'Device class', width: 110, level: 'device', defaultOn: true },
	{ id: 'platform', label: 'Platform', tip: 'Waveguide platform', width: 130, level: 'device', defaultOn: true },
	{ id: 'vpil', label: 'Vpi*L', unit: 'V*cm', tip: 'Vpi times length (DC). Italic = derived from Vpi and length; the Vpi convention of the device is inherited.', width: 76, num: true, level: 'device', defaultOn: true },
	{ id: 'vpi', label: 'Vpi', unit: 'V', tip: 'Half-wave voltage as reported (DC; RF if DC not reported). Convention in the device drawer.', width: 60, num: true, level: 'device', defaultOn: true },
	{ id: 'bw3db', label: '3 dB BW', unit: 'GHz', tip: 'Electro-optic 3 dB bandwidth. Empty when no 3 dB crossing is reported.', width: 72, num: true, level: 'device', defaultOn: true },
	{ id: 'il_onchip', label: 'IL on-chip', unit: 'dB', tip: 'On-chip insertion loss as reported by the paper', width: 72, num: true, level: 'device', defaultOn: true },
	{ id: 'il_f2f', label: 'IL fiber-fiber', unit: 'dB', tip: 'Fiber-to-fiber insertion loss', width: 76, num: true, level: 'device', defaultOn: true },
	{ id: 'rf_loss', label: 'RF loss', unit: 'dB/cm', tip: 'RF electrode loss per length', width: 66, num: true, level: 'device', defaultOn: true },
	{ id: 'rate', label: 'Max rate', unit: 'Gb/s', tip: 'Highest demonstrated line rate', width: 68, num: true, level: 'device', defaultOn: true },
	{ id: 'orgs', label: 'Organizations', tip: 'Author affiliations (full names)', width: 260, level: 'paper', defaultOn: true },
	{ id: 'region', label: 'Country / region', tip: 'Derived from the affiliations', width: 130, level: 'paper', defaultOn: true },
	{ id: 'fab', label: 'Foundry / fab', tip: 'Who fabricated the device, as stated in the paper', width: 190, level: 'paper', defaultOn: true },
	{ id: 'grade', label: 'Repro', tip: 'Reproducibility grade: A geometry and materials fully disclosed; B needs figure digitization; C metrics only', width: 48, level: 'paper', defaultOn: true },
	{ id: 'sim', label: 'Sim', tip: 'Browser simulation config available', width: 44, level: 'paper', defaultOn: true },
	{ id: 'device', label: 'Device', tip: 'Device label', width: 180, level: 'device', defaultOn: false },
	{ id: 'length', label: 'Length', unit: 'mm', tip: 'Active electrode / phase-shifter length', width: 62, num: true, level: 'device', defaultOn: false },
	{ id: 'baud', label: 'Max baud', unit: 'GBd', tip: 'Highest demonstrated symbol rate', width: 64, num: true, level: 'device', defaultOn: false },
	{ id: 'vpi_il', label: 'Vpi*IL', unit: 'V*dB', tip: 'Vpi times on-chip insertion loss (derived)', width: 66, num: true, level: 'device', defaultOn: false },
	{ id: 'fom', label: 'FOM', unit: 'GHz/V', tip: 'f3dB / (Vpi_eff * 10^(IL_onchip/10)); see About', width: 64, num: true, level: 'device', defaultOn: false },
	{ id: 'complete', label: 'Complete', tip: 'Fraction of the 7 core fields reported', width: 62, num: true, level: 'device', defaultOn: false },
	{ id: 'venue', label: 'Venue', tip: 'Journal / conference', width: 180, level: 'paper', defaultOn: false },
	{ id: 'doi', label: 'DOI / arXiv', tip: 'Persistent identifier', width: 170, level: 'paper', defaultOn: false }
];

export const NUMERIC_METRIC: Record<string, (d: Device) => Metric> = {
	vpil,
	vpi,
	bw3db,
	il_onchip: ilOnchip,
	il_f2f: ilF2f,
	rf_loss: rfLoss,
	rate: maxRate,
	baud: maxBaud,
	length: lengthMm,
	vpi_il: (d) => derivedMetric(d, 'vpi_il_vdb'),
	fom: (d) => derivedMetric(d, 'fom')
};

function metricTip(a: Atlas, id: string, d: Device, m: Metric): string {
	const parts: string[] = [];
	if (m.qual) parts.push(qualWord(m.qual));
	if (m.basis) parts.push(`basis: ${BASIS_TIP[m.basis] ?? m.basis}`);
	const ev = m.field ? d.evidence[m.field] : undefined;
	if (ev?.locator && !ev.derived) parts.push(`source: ${ev.locator}`);
	if (m.derived) {
		const key = m.field === 'vpil_dc_vcm_derived' ? 'vpil_dc_vcm_derived' : m.field;
		const dv = key ? (d.derived as unknown as Record<string, { formula?: string; warning?: string }>)[key] : undefined;
		if (dv?.formula) parts.push(`derived: ${dv.formula}`);
		if (dv?.warning) parts.push(dv.warning);
		else if (ev?.formula) parts.push(`derived: ${ev.formula}`);
	}
	if (id === 'vpil' || id === 'vpi') {
		if (d.vpi_convention) parts.push(`convention: ${enumLabel(a, 'vpi_convention', d.vpi_convention)}`);
		if (id === 'vpi' && m.field === 'vpi_rf_v' && typeof d.vpi_rf_freq_ghz === 'number') parts.push(`at ${fmt(d.vpi_rf_freq_ghz)} GHz`);
	}
	if (id === 'bw3db') {
		if (typeof d.bw_measured_to_ghz === 'number') {
			const q = d.qualifiers['bw_measured_to_ghz'];
			parts.push(`measured to ${qualPrefix(q)}${fmt(d.bw_measured_to_ghz)} GHz`);
		}
		if (d.bw3db_reference) {
			const ref = String(d.bw3db_reference);
			const refTxt = ref === 'other' && typeof d.bw3db_reference_freq_ghz === 'number' ? `${fmt(d.bw3db_reference_freq_ghz)} GHz` : enumLabel(a, 'bw_reference', ref);
			parts.push(`reference: ${refTxt}`);
		}
	}
	if (id === 'il_onchip') {
		if (d.il_onchip_includes) parts.push(`includes: ${d.il_onchip_includes}`);
		if (d.il_onchip_excludes) parts.push(`excludes: ${d.il_onchip_excludes}`);
	}
	if (id === 'rf_loss' && typeof d.rf_loss_freq_ghz === 'number') parts.push(`at ${fmt(d.rf_loss_freq_ghz)} GHz`);
	if (id === 'rate') {
		if (d.modulation_format) parts.push(`format: ${d.modulation_format}`);
		if (typeof d.max_baud_gbd === 'number') parts.push(`${fmt(d.max_baud_gbd)} GBd`);
	}
	return parts.join('\n');
}

export function metricCell(a: Atlas, id: string, d: Device): Cell {
	const fn = NUMERIC_METRIC[id];
	const m = fn(d);
	if (m.v === null) {
		let tip = '';
		if (id === 'bw3db' && typeof d.bw_measured_to_ghz === 'number') {
			tip = `no 3 dB crossing reported; response measured to ${qualPrefix(d.qualifiers['bw_measured_to_ghz'])}${fmt(d.bw_measured_to_ghz)} GHz`;
		} else {
			tip = 'not reported';
		}
		return { text: EM_DASH, num: null, muted: true, tip };
	}
	const prefix = qualPrefix(m.qual);
	return {
		text: `${prefix}${fmt(m.v)}`,
		num: m.v,
		qual: m.qual ?? undefined,
		derived: m.derived,
		basis: m.basis,
		tip: metricTip(a, id, d, m)
	};
}

const dash = (tip = 'not reported'): Cell => ({ text: EM_DASH, muted: true, tip });

export function cellFor(a: Atlas, id: string, p: Paper, d: Device): Cell {
	switch (id) {
		case 'paper':
			return { text: p.label, tip: `${p.title}\n${p.authors.join('; ')}\n${p.venue ?? ''}`.trim() };
		case 'year':
			return { text: String(p.year), num: p.year };
		case 'material':
			return { text: enumLabel(a, 'eo_material', d.eo_material) };
		case 'class':
			return { text: enumLabel(a, 'device_class', d.device_class) };
		case 'platform':
			return d.waveguide_platform ? { text: enumLabel(a, 'waveguide_platform', d.waveguide_platform) } : dash();
		case 'orgs': {
			const t = p.orgs_affil.map((o) => o.org_name).join('; ');
			return t ? { text: t, tip: t, multiline: true } : dash();
		}
		case 'region': {
			const c = p.countries_derived.map((x) => enumLabel(a, 'country', x));
			const r = p.regions_derived.map((x) => enumLabel(a, 'region', x));
			if (!c.length) return dash();
			return { text: `${c.join(', ')} / ${r.join(', ')}`, tip: `${c.join(', ')} (${r.join(', ')})` };
		}
		case 'fab': {
			const t = p.orgs_fab.map((o) => o.org_name).join('; ');
			return t ? { text: t, tip: t, multiline: true } : dash();
		}
		case 'grade':
			return p.repro_grade ? { text: p.repro_grade, tip: `Reproducibility grade ${p.repro_grade}` } : dash();
		case 'sim':
			return p.has_sim
				? { text: 'sim', href: `sim?id=${encodeURIComponent(p.sim_ids[0] ?? p.paper_id)}`, tip: `Simulation config: ${p.sim_ids.join(', ')}` }
				: dash('no simulation config');
		case 'device':
			return { text: d.device_label, tip: d.device_id };
		case 'venue':
			return p.venue ? { text: p.venue, tip: p.venue } : dash();
		case 'doi': {
			const t = p.doi ?? p.arxiv_id;
			return t ? { text: t, tip: t } : dash();
		}
		case 'complete':
			return {
				text: fmt(d.derived.completeness.value),
				num: d.derived.completeness.value,
				tip: `reported: ${d.derived.completeness.reported.join(', ') || 'none'}\nnot reported: ${d.derived.completeness.missing.join(', ') || 'none'}`
			};
		default:
			return metricCell(a, id, d);
	}
}

export function sortValue(a: Atlas, id: string, p: Paper, d: Device): SortVal {
	const fn = NUMERIC_METRIC[id];
	if (fn) return fn(d).v;
	switch (id) {
		case 'paper': return p.label.toLowerCase();
		case 'year': return p.year;
		case 'complete': return d.derived.completeness.value;
		case 'sim': return p.has_sim ? 1 : 0;
		case 'grade': return p.repro_grade;
		case 'device': return d.device_label.toLowerCase();
		default: {
			const c = cellFor(a, id, p, d);
			return c.text === EM_DASH ? null : c.text.toLowerCase();
		}
	}
}

export function defaultColumns(): string[] {
	return COLS.filter((c) => c.defaultOn).map((c) => c.id);
}
