import { buildSparseMatrix, type ReservoirGraph, type SparseMatrix } from "./graph";
import { seededRandom } from "../optics/channels";

/**
 * Rate-based leaky reservoir: the interface is intentionally close to a LIF
 * implementation so the neuron model can be swapped (DevLog/000 section 2.1):
 *
 *   x[t+1] = (1 - alpha) x[t] + alpha * tanh(W x[t] + W_in I[t] + b) + noise
 *
 * `W` is fixed; only the downstream readout is trained. Output neurons are a
 * deterministic subset used as the feature vector. With a synthetic graph those
 * outputs are not biological descending neurons; that is stated in `note`.
 */
export interface ReservoirConfig {
  alpha: number;
  inputScale: number;
  noise: number;
  outputCount: number;
  seed: number;
}

export const DEFAULT_RESERVOIR: ReservoirConfig = {
  alpha: 0.3,
  inputScale: 1.0,
  noise: 0.01,
  outputCount: 64,
  seed: 12345,
};

export class RateReservoir {
  readonly graph: ReservoirGraph;
  readonly config: ReservoirConfig;
  readonly inputDim: number;
  readonly outputCount: number;
  readonly W: SparseMatrix;
  readonly win: Float64Array;
  readonly bias: Float64Array;
  readonly outputIndices: Int32Array;
  x: Float64Array;
  private rng: () => number;
  private scratch: Float64Array;
  private featureBuffer: Float64Array;

  constructor(graph: ReservoirGraph, inputDim: number, config: Partial<ReservoirConfig> = {}) {
    this.graph = graph;
    this.config = { ...DEFAULT_RESERVOIR, ...config };
    this.inputDim = inputDim;
    this.W = buildSparseMatrix(graph.n, graph.edges);
    const rng = seededRandom(this.config.seed);
    this.rng = rng;
    this.win = new Float64Array(graph.n * inputDim);
    const inputNorm = this.config.inputScale / Math.sqrt(Math.max(1, inputDim));
    for (let i = 0; i < this.win.length; i++) this.win[i] = (rng() * 2 - 1) * inputNorm;
    this.bias = new Float64Array(graph.n);
    for (let i = 0; i < graph.n; i++) this.bias[i] = (rng() * 2 - 1) * 0.1;
    this.outputCount = Math.min(this.config.outputCount, graph.n);
    this.outputIndices = new Int32Array(this.outputCount);
    for (let i = 0; i < this.outputCount; i++) {
      this.outputIndices[i] = Math.floor((i * graph.n) / this.outputCount);
    }
    this.x = new Float64Array(graph.n);
    this.scratch = new Float64Array(graph.n);
    this.featureBuffer = new Float64Array(this.outputCount);
  }

  reset(): void {
    this.x.fill(0);
  }

  step(input: Float64Array): void {
    if (input.length !== this.inputDim) throw new Error("RateReservoir.step: input dimension mismatch");
    const { alpha, noise } = this.config;
    const n = this.graph.n;
    this.W.multiply(this.x, this.scratch);
    for (let i = 0; i < n; i++) {
      let drive = this.scratch[i] + this.bias[i];
      const base = i * this.inputDim;
      for (let j = 0; j < this.inputDim; j++) {
        drive += this.win[base + j] * input[j];
      }
      const activation = Math.tanh(drive);
      this.x[i] = (1 - alpha) * this.x[i] + alpha * activation + noise * (this.rng() * 2 - 1);
    }
  }

  features(): Float64Array {
    for (let i = 0; i < this.outputCount; i++) this.featureBuffer[i] = this.x[this.outputIndices[i]];
    return this.featureBuffer;
  }

  /** Snapshot of features for logging/visualization (copy). */
  featuresCopy(): Float64Array {
    const out = new Float64Array(this.outputCount);
    for (let i = 0; i < this.outputCount; i++) out[i] = this.x[this.outputIndices[i]];
    return out;
  }

  /** Aggregate activity statistics for visualization. */
  activityStats(): { mean: number; max: number; fractionActive: number } {
    let sum = 0;
    let max = 0;
    let active = 0;
    for (let i = 0; i < this.x.length; i++) {
      sum += this.x[i];
      max = Math.max(max, Math.abs(this.x[i]));
      if (Math.abs(this.x[i]) > 0.2) active++;
    }
    return { mean: sum / this.x.length, max, fractionActive: active / this.x.length };
  }
}
