export type {
	GaussianMode,
	GaussianMode as Mode,
	GaussianTransform,
	GaussianTransform as Transform,
	ResolvedTransform,
	MismatchLoss
} from '$foundation/physics/gaussian.mjs';

export type {
	AnalyticQuestion,
	SampledPreset,
	Preset
} from '$foundation/scenarios/generator.mjs';

export interface StoredFieldView {
	id: string;
	nx: number;
	ny: number;
	x0: number;
	x1: number;
	y0: number;
	y1: number;
	field: number[];
	core: { x0: number; x1: number; y0: number; y1: number };
	backgroundIndex: number;
	indexRegions: { x0: number; x1: number; y0: number; y1: number; index: number; material: string }[];
}

export interface WaveguideQuestionView {
	a: StoredFieldView;
	b: StoredFieldView;
	explanation: string;
	note?: string;
}

/** A shipped analytic question record, exactly as persisted by the builder. */
export interface CachedQuestion {
	id: string;
	title: string;
	seed: number;
	category: string;
	status: 'ready-analytic';
	model: string;
	generatorVersion: string;
	solverVersion: string;
	materialDBVersion: string;
	meshSettings: { kind: string };
	wavelengthUm: number;
	a: import('$foundation/physics/gaussian.mjs').GaussianMode;
	b: import('$foundation/physics/gaussian.mjs').GaussianMode;
	transform: import('$foundation/physics/gaussian.mjs').ResolvedTransform;
	answer: { eta: number; lossDB: number | null; lossIsInfinite: boolean };
	bucket: number;
	provenance: {
		cacheVersion: string;
		generatorVersion: string;
		presetsVersion: string;
		presetsHash: string;
		examplesHash: string;
	};
	/** Present on on-the-fly waveguide questions; rendered instead of Gaussian fields. */
	view?: WaveguideQuestionView;
}

export interface ManifestEntry {
	id: string;
	key: string;
	path: string;
	status: string;
	category: string;
	bucket: number;
}

export interface Manifest {
	schemaVersion: string;
	provenance: Record<string, string>;
	readyCount: number;
	pendingCount: number;
	histogram: number[];
	entries: ManifestEntry[];
	pendingPath: string;
}

export interface PendingRecipe {
	id: string;
	title: string;
	kind: string;
	status: string;
	[key: string]: unknown;
}

export interface NumericalRecipes {
	provenance: Record<string, string>;
	recipes: PendingRecipe[];
}

export type AnswerMode = 'continuous' | 'interval' | 'ranking';

export type FieldComponent = 'intensity' | 'amplitude' | 'real' | 'phase';

export type Colormap = 'viridis' | 'magma' | 'diverging' | 'cyclic' | 'gray';

export type ScaleKind = 'linear' | 'log';

export type ColorScale = 'shared' | 'individual';

export type ViewMode = 'flash' | 'paired';

export interface ModeGalleryEntry {
	id: string;
	name: string;
	order: number;
	category: string;
	modeLabel: string;
	materialLabel: string;
	cladding: string;
	wavelengthUm: number;
	status: string;
	neff: number;
	confinement: number;
	residual: number;
	dimensions: { widthUm: number; heightUm: number; slabUm: number | null };
	core: { x0: number; x1: number; y0: number; y1: number };
	axes: { x0: number; x1: number; y0: number; y1: number; nx: number; ny: number };
	backgroundIndex: number;
	indexRegions: { x0: number; x1: number; y0: number; y1: number; index: number; material: string }[];
	field: number[];
}

export interface ModeGalleryManifest {
	schemaVersion: string;
	provenance: Record<string, string>;
	note: string;
	entries: ModeGalleryEntry[];
}

export type SessionPhase = 'idle' | 'playing' | 'revealed' | 'results';

export interface FieldGrid {
	/** Physical extent: coordinates span [-extentX, extentX] x [-extentY, extentY]. */
	extentXUm: number;
	extentYUm: number;
	samples: number;
	/** Flat row-major complex values reconstructed from the analytic model. */
	re: Float64Array;
	im: Float64Array;
}

export interface SessionConfig {
	answerMode: AnswerMode;
	questionCount: number;
	category: string | null;
	seed: number;
	/** null = unlimited practice; number = fixed speedrun length. */
	limit: number | null;
	/** Optional per-category caps so one generated family cannot dominate. */
	maxPerCategory?: Record<string, number>;
}

export interface AnswerRecord {
	questionId: string;
	category: string;
	bucket: number;
	mode: AnswerMode;
	guessEta: number;
	trueEta: number;
	guessLossDB: number;
	trueLossDB: number;
	signedEtaError: number;
	absEtaError: number;
	signedDbError: number;
	absDbError: number;
	score: number;
	durationMs: number;
}

export interface SessionResults {
	count: number;
	score: number;
	mae: number;
	medianAbsEtaError: number;
	rmse: number;
	meanDbError: number;
	meanDurationMs: number;
	medianDurationMs: number;
	categoryBreakdown: { category: string; count: number; mae: number; score: number }[];
	biasByBucket: { bucket: number; count: number; signedBias: number }[];
	answers: AnswerRecord[];
}
