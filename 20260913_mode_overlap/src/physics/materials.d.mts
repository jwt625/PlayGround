export type MaterialStatus =
	| 'reference'
	| 'placeholder-process-specific'
	| 'disabled-requires-tensor';

export interface Material {
	id: string;
	name: string;
	kind: string;
	status: MaterialStatus;
	validityUm: [number, number];
	source: string;
	n(wavelengthUm: number): number;
}

export declare const MATERIAL_DB_VERSION: string;
export declare const MATERIALS: Record<string, Material>;
export declare function getMaterial(id: string): Material;
export declare function refractiveIndex(id: string, wavelengthUm: number): number;
export declare function relativePermittivity(id: string, wavelengthUm: number): number;
export declare function isReleaseEligible(id: string): boolean;
