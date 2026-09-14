export interface GaussianMode {
	mfdXUm: number;
	mfdYUm: number;
}

export interface GaussianTransform {
	dxUm?: number;
	dyUm?: number;
	thetaXRad?: number;
	thetaYRad?: number;
	polarizationRad?: number;
	gapIndex?: number;
}

export interface ResolvedTransform {
	dxUm: number;
	dyUm: number;
	thetaXRad: number;
	thetaYRad: number;
	polarizationRad: number;
	gapIndex: number;
}

export interface ComplexValue {
	re: number;
	im: number;
}

export interface MismatchLoss {
	lossDB: number | null;
	lossIsInfinite: boolean;
}

export declare const GAUSSIAN_VERSION: string;
export declare function positive(value: number, name: string): number;
export declare function validateMode(mode: GaussianMode): GaussianMode;
export declare function transformDefaults(t?: GaussianTransform): ResolvedTransform;
export declare function gaussianOverlap(
	a: GaussianMode,
	b: GaussianMode,
	wavelengthUm?: number,
	transform?: GaussianTransform
): number;
export declare function mismatchLoss(eta: number): MismatchLoss;
export declare function gaussianField(
	mode: GaussianMode,
	xUm: number,
	yUm: number,
	wavelengthUm?: number,
	transform?: GaussianTransform
): ComplexValue;
