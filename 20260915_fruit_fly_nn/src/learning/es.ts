/**
 * Evolution strategies (OpenAI-ES style, antithetic sampling) used to train the
 * small readout around the fixed connectome. Easy to parallelize and gives a
 * fun evolutionary visualization (DevLog/000 section 12, option B).
 */

export interface EsOptions {
  population: number;
  sigma: number;
  learningRate: number;
  seed: number;
}

export const DEFAULT_ES: EsOptions = {
  population: 16,
  sigma: 0.08,
  learningRate: 0.05,
  seed: 1,
};

export interface GenerationStats {
  generation: number;
  best: number;
  mean: number;
  std: number;
}

export interface EsResult {
  params: Float64Array;
  history: GenerationStats[];
  evaluations: number;
}

function gaussianPair(rand: () => number): [number, number] {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  const r = Math.sqrt(-2 * Math.log(u));
  return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
}

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function trainEs(
  initial: Float64Array,
  fitness: (params: Float64Array) => number,
  options: Partial<EsOptions> = {},
  generations = 50,
  onGeneration?: (stats: GenerationStats) => void,
): EsResult {
  const opts = { ...DEFAULT_ES, ...options };
  if (opts.population % 2 !== 0) throw new Error("ES population must be even");
  const rand = makeRng(opts.seed);
  const dim = initial.length;
  const theta = Float64Array.from(initial);
  const history: GenerationStats[] = [];
  let evaluations = 0;

  for (let gen = 0; gen < generations; gen++) {
    const grads = new Float64Array(dim);
    const fitnesses = new Float64Array(opts.population);
    const epsilons: Float64Array[] = [];

    for (let p = 0; p < opts.population / 2; p++) {
      const eps = new Float64Array(dim);
      const plus = new Float64Array(dim);
      const minus = new Float64Array(dim);
      for (let i = 0; i < dim; i += 2) {
        const [g1, g2] = gaussianPair(rand);
        eps[i] = g1;
        if (i + 1 < dim) eps[i + 1] = g2;
      }
      for (let i = 0; i < dim; i++) {
        plus[i] = theta[i] + opts.sigma * eps[i];
        minus[i] = theta[i] - opts.sigma * eps[i];
      }
      const fPlus = fitness(plus);
      const fMinus = fitness(minus);
      evaluations += 2;
      fitnesses[2 * p] = fPlus;
      fitnesses[2 * p + 1] = fMinus;
      epsilons.push(eps);
    }

    // Normalize rewards for a stable update.
    let mean = 0;
    for (const f of fitnesses) mean += f;
    mean /= opts.population;
    let variance = 0;
    for (const f of fitnesses) variance += (f - mean) ** 2;
    const std = Math.sqrt(variance / opts.population) + 1e-8;

    for (let p = 0; p < opts.population / 2; p++) {
      const eps = epsilons[p];
      const advPlus = (fitnesses[2 * p] - mean) / std;
      const advMinus = (fitnesses[2 * p + 1] - mean) / std;
      const coeff = (advPlus - advMinus) / (opts.population * opts.sigma);
      for (let i = 0; i < dim; i++) grads[i] += coeff * eps[i];
    }
    for (let i = 0; i < dim; i++) theta[i] += opts.learningRate * grads[i];

    let best = -Infinity;
    for (const f of fitnesses) best = Math.max(best, f);
    const stats: GenerationStats = { generation: gen, best, mean, std };
    history.push(stats);
    onGeneration?.(stats);
  }

  return { params: theta, history, evaluations };
}
