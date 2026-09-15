import { seededRandom } from "../optics/channels";

/**
 * Stochastic parallel gradient descent for piston phase control.
 *
 * Each iteration perturbs every channel by +/- perturbation with random signs,
 * measures the objective at both points, and updates using the gradient-like
 * estimate g_i = (J_plus - J_minus) * sign_i. This is the conventional CBC
 * baseline from DevLog/000 section 8 and docs/CODING_TASKS.md T03.
 *
 * Objective evaluations are counted fairly: both perturbation measurements per
 * iteration are reported.
 */
export interface SpgdOptions {
  iterations: number;
  /** Update gain. */
  gain: number;
  /** Piston perturbation amplitude, rad. */
  perturbation: number;
  seed: number;
}

export interface SpgdResult {
  pistons: number[];
  history: number[];
  evaluations: number;
}

export const DEFAULT_SPGD: SpgdOptions = {
  iterations: 400,
  gain: 1.0,
  perturbation: 0.3,
  seed: 1,
};

export function runSpgd(
  objective: (pistons: readonly number[]) => number,
  initialPistons: readonly number[],
  options: Partial<SpgdOptions> = {},
): SpgdResult {
  const opts = { ...DEFAULT_SPGD, ...options };
  const rng = seededRandom(opts.seed);
  const count = initialPistons.length;
  const u = [...initialPistons];
  const uPlus = new Array<number>(count);
  const uMinus = new Array<number>(count);
  const signs = new Int8Array(count);
  const history: number[] = [];
  let evaluations = 0;

  for (let iter = 0; iter < opts.iterations; iter++) {
    for (let i = 0; i < count; i++) {
      signs[i] = rng() < 0.5 ? -1 : 1;
    }
    for (let i = 0; i < count; i++) {
      uPlus[i] = u[i] + opts.perturbation * signs[i];
      uMinus[i] = u[i] - opts.perturbation * signs[i];
    }
    const jPlus = objective(uPlus);
    const jMinus = objective(uMinus);
    evaluations += 2;
    for (let i = 0; i < count; i++) {
      u[i] += opts.gain * (jPlus - jMinus) * signs[i];
    }
    history.push(0.5 * (jPlus + jMinus));
  }

  return { pistons: u, history, evaluations };
}
