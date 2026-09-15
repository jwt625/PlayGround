/**
 * Trainable linear readout around the fixed connectome substrate.
 *
 *   u = tanh(W_out z + b_out)
 *
 * Only W_out and b_out are learned; the reservoir graph is fixed. Parameter
 * counts are reported so ablations can be matched (DevLog/000 section 25).
 */
export class LinearReadout {
  readonly nOut: number;
  readonly nFeat: number;
  w: Float64Array;
  b: Float64Array;
  private buffer: Float64Array;

  constructor(nOut: number, nFeat: number, seed = 1) {
    this.nOut = nOut;
    this.nFeat = nFeat;
    this.w = new Float64Array(nOut * nFeat);
    this.b = new Float64Array(nOut);
    this.buffer = new Float64Array(nOut);
    // Deterministic small initialization.
    let s = seed >>> 0;
    const rand = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const scale = 1 / Math.sqrt(Math.max(1, nFeat));
    for (let i = 0; i < this.w.length; i++) this.w[i] = (rand() * 2 - 1) * scale;
    for (let i = 0; i < this.b.length; i++) this.b[i] = (rand() * 2 - 1) * 0.05;
  }

  get parameterCount(): number {
    return this.nOut * this.nFeat + this.nOut;
  }

  predict(features: Float64Array): Float64Array {
    if (features.length !== this.nFeat) throw new Error("LinearReadout.predict: feature mismatch");
    for (let i = 0; i < this.nOut; i++) {
      let sum = this.b[i];
      const base = i * this.nFeat;
      for (let j = 0; j < this.nFeat; j++) sum += this.w[base + j] * features[j];
      this.buffer[i] = Math.tanh(sum);
    }
    return this.buffer;
  }

  getParams(): Float64Array {
    const out = new Float64Array(this.parameterCount);
    out.set(this.w, 0);
    out.set(this.b, this.w.length);
    return out;
  }

  setParams(params: Float64Array): void {
    if (params.length !== this.parameterCount) throw new Error("LinearReadout.setParams: size mismatch");
    this.w.set(params.subarray(0, this.w.length), 0);
    this.b.set(params.subarray(this.w.length), 0);
  }
}
