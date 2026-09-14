import { gaussianOverlap } from '../physics/gaussian.mjs';
import type { CachedQuestion } from './types';

export interface ExplanationCandidate {
	effect: string;
	eta: number;
	improvement: number;
}

export interface Explanation {
	baseline: number;
	dominant: string;
	sentence: string;
	candidates: ExplanationCandidate[];
}

/**
 * G008: explanations come from counterfactual overlaps, not from an additive
 * loss budget. We remove one effect at a time and report the largest single
 * recoverable improvement. Interacting effects are described as such.
 */
export function explainQuestion(question: CachedQuestion): Explanation {
	if (question.view) {
		return {
			baseline: question.answer.eta,
			dominant: 'variant',
			sentence: question.view.explanation,
			candidates: []
		};
	}
	const { a, b, transform: t, wavelengthUm } = question;
	const baseline = question.answer.eta;
	const candidates: ExplanationCandidate[] = [];
	const consider = (effect: string, eta: number) => {
		const improvement = eta - baseline;
		if (improvement > 1e-9) candidates.push({ effect, eta, improvement });
	};

	if (t.dxUm !== 0 || t.dyUm !== 0) {
		consider('lateral offset', gaussianOverlap(a, b, wavelengthUm, { ...t, dxUm: 0, dyUm: 0 }));
	}
	if (t.thetaXRad !== 0 || t.thetaYRad !== 0) {
		consider('angular tilt', gaussianOverlap(a, b, wavelengthUm, { ...t, thetaXRad: 0, thetaYRad: 0 }));
	}
	if (a.mfdXUm !== b.mfdXUm || a.mfdYUm !== b.mfdYUm) {
		consider('mode-size mismatch', gaussianOverlap(a, a, wavelengthUm, t));
	}
	if (t.polarizationRad !== 0) {
		consider('polarization rotation', gaussianOverlap(a, b, wavelengthUm, { ...t, polarizationRad: 0 }));
	}
	candidates.sort((x, y) => y.improvement - x.improvement);

	if (baseline >= 0.999999 || candidates.length === 0) {
		return {
			baseline,
			dominant: 'none',
			sentence:
				baseline >= 0.999999
					? 'Modes are effectively identical at this plane; no perturbation dominates.'
					: 'Loss is dominated by interacting effects; no single perturbation recovers most of η.',
			candidates
		};
	}

	const top = candidates[0];
	const interacting = candidates.length > 1 && candidates[1].improvement > top.improvement * 0.5;
	const sentence = interacting
		? `Largest single improvement: ${top.effect} (η ${baseline.toFixed(3)} → ${top.eta.toFixed(3)}); other effects interact.`
		: `Largest single improvement: ${top.effect} (η ${baseline.toFixed(3)} → ${top.eta.toFixed(3)}).`;
	return { baseline, dominant: top.effect, sentence, candidates };
}
