export type Qual = 'lt' | 'gt' | 'approx';

export interface EvidenceEntry {
	locator: string;
	basis: string | null;
	note: string;
	unit: string;
	derived: boolean;
	formula?: string;
	inputs?: string[];
}

export interface DerivedValue {
	value: number;
	unit: string;
	formula: string;
	inputs: string[];
	qualifier?: Qual | null;
	qualifiers?: string[];
	warning?: string;
	rf_corrected?: boolean;
}

export interface Completeness {
	value: number;
	reported: string[];
	missing: string[];
	unit: string;
}

export interface Derived {
	vpil_dc_vcm_derived?: DerivedValue;
	il_rf_total_db?: DerivedValue;
	vpi_il_vdb?: DerivedValue;
	fom?: DerivedValue;
	completeness: Completeness;
}

export interface Device {
	device_id: string;
	paper_id: string;
	device_label: string;
	device_class: string;
	tags: string[];
	eo_material: string;
	waveguide_platform: string | null;
	integration: string | null;
	electrode_type: string | null;
	drive: string | null;
	vpi_convention: string | null;
	temperature_class: string | null;
	band: string | null;
	[key: string]: unknown;
	qualifiers: Record<string, Qual>;
	derived: Derived;
	vpil_best: { value: number; source: 'reported_dc' | 'derived_dc' | 'reported_rf'; qualifier?: Qual | null } | null;
	headline_basis: Record<string, string | null>;
	is_sim: boolean;
	measured_only: boolean;
	evidence: Record<string, EvidenceEntry>;
	sim_available: boolean;
}

export interface Org {
	org_name: string;
	org_type: string;
	country: string;
	region: string;
	parent_org: string;
	notes: string;
}

export interface Paper {
	paper_id: string;
	label: string;
	title: string;
	authors: string[];
	year: number;
	published_on: string | null;
	venue: string | null;
	doi: string | null;
	arxiv_id: string | null;
	url: string;
	source_type: string;
	access: string;
	license: string | null;
	redistribution: string;
	research_groups: string[];
	universities: string[];
	companies: string[];
	foundry_or_fab: string[];
	repro_grade: string | null;
	sim_config: string | null;
	notes: string | null;
	device_ids: string[];
	rep: Record<string, string>;
	orgs_affil: Org[];
	orgs_fab: Org[];
	countries_derived: string[];
	regions_derived: string[];
	sim_ids: string[];
	has_sim: boolean;
	n_devices: number;
	[key: string]: unknown;
}

export interface Sim {
	paper_id: string;
	id: string;
	device_id: string | null;
	title: string | null;
	repro_grade: string | null;
	validation_status: string | null;
	chain: string[];
	n_targets: number;
	path: string;
}

export interface ColumnMeta {
	name: string;
	type: string;
	unit: string;
	desc: string;
	enum: string;
	evidence: boolean;
	label: string;
}

export interface EnumItem {
	value: string;
	label: string;
}

export interface Atlas {
	meta: {
		generated_on: string;
		schema_version: number;
		core_fields: { name: string; fields: string[] }[];
		fom_classes: string[];
		fom_definition: string;
		vpi_il_definition: string;
		completeness_definition: string;
		column_groups: { name: string; fields: string[] }[];
	};
	integrity: {
		counts: Record<string, number>;
		per_platform: Record<string, number>;
		per_material: Record<string, number>;
		per_device_class: Record<string, number>;
		per_basis: Record<string, number>;
		per_headline_basis: Record<string, Record<string, number>>;
		per_source_type: Record<string, number>;
		per_access: Record<string, number>;
		per_repro_grade: Record<string, number>;
		per_redistribution: Record<string, number>;
		evidence: { nonempty_evidence_fields: number; without_evidence_entry: number };
		completeness_mean: number | null;
	};
	enums: Record<string, EnumItem[]>;
	columns: ColumnMeta[];
	paper_columns: { name: string; type: string; desc: string; label: string }[];
	papers: Paper[];
	devices: Device[];
	organizations: Org[];
	sims: Sim[];
	warnings: string[];
}

export type RepMode = 'default' | 'lowest_vpil' | 'highest_bw' | 'highest_fom' | 'lowest_vpi_il';

export interface Filters {
	q: string;
	materials: string[];
	classes: string[];
	platforms: string[];
	sourceTypes: string[];
	regions: string[];
	countries: string[];
	yearMin: number | null;
	yearMax: number | null;
	measuredOnly: boolean;
	hasSim: boolean;
	rep: RepMode;
	allDevices: boolean;
}

export interface SortKey {
	key: string;
	dir: 'asc' | 'desc';
}
