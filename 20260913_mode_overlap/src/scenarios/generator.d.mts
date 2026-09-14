import type {
	GaussianMode,
	GaussianTransform,
	ResolvedTransform,
	MismatchLoss
} from '../physics/gaussian.mjs';

export interface Preset {
	id: string;
	name: string;
	model: string;
	wavelength_um: number;
	nominal: Record<string, number>;
	ranges: Record<string, [number, number]>;
	[key: string]: unknown;
}

export interface SampledPreset {
	presetId: string;
	seed: number;
	generatorVersion: string;
	wavelengthUm: number;
	parameters: Record<string, number>;
	status: string;
}

export interface AnalyticQuestion {
	id: string;
	title: string;
	seed: number;
	category: string;
	status: string;
	model: string;
	generatorVersion: string;
	solverVersion: string;
	materialDBVersion: string;
	meshSettings: { kind: string };
	wavelengthUm: number;
	a: GaussianMode;
	b: GaussianMode;
	transform: ResolvedTransform;
	answer: { eta: number } & MismatchLoss;
	bucket: number;
}

export declare const GENERATOR_VERSION: string;
export declare const BUCKET_EDGES: number[];
export declare function bucketIndex(eta: number): number;
export declare function rng(seed: number): () => number;
export declare function triangular(random: () => number, low: number, mode: number, high: number): number;
export declare function samplePreset(preset: Preset, seed: number): SampledPreset;
export declare function analyticQuestion(question: {
	id: string;
	title: string;
	a: GaussianMode;
	b: GaussianMode;
	wavelengthUm?: number;
	transform?: GaussianTransform;
	seed?: number;
	category?: string;
}): AnalyticQuestion;
export declare function balancedAlignmentBank(seed?: number, perBucket?: number): AnalyticQuestion[];
export declare function balancedMixedBank(seed?: number, perBucket?: number): AnalyticQuestion[];
