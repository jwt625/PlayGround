/**
 * Sparse directed connectome graph and reservoir substrates.
 *
 * The real MaleCNS graph is not bundled and its availability/version/license
 * still needs an independent audit (T07). `loadConnectome` accepts an explicit
 * file; when none is provided the code generates a clearly-labelled synthetic
 * graph so tests and training remain honest about provenance.
 */

export interface EdgeList {
  pre: Int32Array;
  post: Int32Array;
  weight: Float64Array;
}

export interface ReservoirGraph {
  readonly n: number;
  readonly source: "synthetic-random" | "synthetic-connectome" | "malecns";
  readonly note: string;
  readonly edges: EdgeList;
  /** Original source body IDs per graph index, when loaded from MaleCNS. */
  readonly bodyIds?: Int32Array;
}

export class SparseMatrix {
  readonly rows: number;
  readonly cols: number;
  readonly indptr: Int32Array;
  readonly indices: Int32Array;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, indptr: Int32Array, indices: Int32Array, data: Float64Array) {
    this.rows = rows;
    this.cols = cols;
    this.indptr = indptr;
    this.indices = indices;
    this.data = data;
  }

  /** out = W * x (overwrites out; out must be length rows). */
  multiply(x: Float64Array, out: Float64Array): void {
    for (let i = 0; i < this.rows; i++) {
      let sum = 0;
      const start = this.indptr[i];
      const end = this.indptr[i + 1];
      for (let k = start; k < end; k++) sum += this.data[k] * x[this.indices[k]];
      out[i] = sum;
    }
  }
}

export function buildSparseMatrix(n: number, edges: EdgeList): SparseMatrix {
  const counts = new Int32Array(n + 1);
  for (let k = 0; k < edges.post.length; k++) counts[edges.post[k]]++;
  const indptr = new Int32Array(n + 1);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    indptr[i] = acc;
    acc += counts[i];
  }
  indptr[n] = acc;
  const nnz = acc;
  const indices = new Int32Array(nnz);
  const data = new Float64Array(nnz);
  const cursor = indptr.slice(0, n);
  for (let k = 0; k < edges.pre.length; k++) {
    const i = edges.post[k];
    const slot = cursor[i]++;
    indices[slot] = edges.pre[k];
    data[slot] = edges.weight[k];
  }
  return new SparseMatrix(n, n, indptr, indices, data);
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Scale edge weights so the spectral radius is approximately `target`. */
export function scaleEdges(edges: EdgeList, target: number, n: number, seed: number): EdgeList {
  const rand = mulberry32(seed);
  const out = new Float64Array(edges.weight.length);
  for (let i = 0; i < out.length; i++) {
    let w = edges.weight[i];
    if (w === 0) w = rand() * 2 - 1;
    out[i] = w;
  }
  const degree = out.length / Math.max(1, n);
  const norm = target / Math.sqrt(Math.max(1, degree));
  for (let i = 0; i < out.length; i++) out[i] *= norm;
  return { pre: edges.pre, post: edges.post, weight: out };
}

export interface SyntheticGraphOptions {
  n: number;
  avgDegree: number;
  seed: number;
  /** Add distance-independent clustered structure (default random). */
  clusterSize?: number;
}

/**
 * Synthetic sparse graph. `clusterSize > 1` produces clustered connectivity so
 * `kind` can be used as a topology label in ablations; this is NOT a connectome.
 */
export function generateSyntheticGraph(options: SyntheticGraphOptions): ReservoirGraph {
  const { n, avgDegree, seed, clusterSize = 1 } = options;
  const rand = mulberry32(seed);
  const edgesPerNeuron = Math.max(1, Math.round(avgDegree));
  const pre: number[] = [];
  const post: number[] = [];
  const weight: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < edgesPerNeuron; k++) {
      let src: number;
      if (clusterSize > 1 && rand() < 0.6) {
        const cluster = Math.floor(i / clusterSize);
        const base = cluster * clusterSize;
        src = base + Math.floor(rand() * clusterSize);
        src = ((src % n) + n) % n;
      } else {
        src = Math.floor(rand() * n);
      }
      pre.push(src);
      post.push(i);
      weight.push(rand() * 2 - 1);
    }
  }
  const edges: EdgeList = {
    pre: Int32Array.from(pre),
    post: Int32Array.from(post),
    weight: Float64Array.from(weight),
  };
  return {
    n,
    source: "synthetic-random",
    note: clusterSize > 1 ? "clustered synthetic graph (not a connectome)" : "degree-matched random graph (not a connectome)",
    edges: scaleEdges(edges, 0.9, n, seed ^ 0x9e3779b9),
  };
}

/** Degree-preserving edge swap, for the edge-shuffled ablation. */
export function edgeShuffled(graph: ReservoirGraph, seed: number): ReservoirGraph {
  const rand = mulberry32(seed);
  const pre = graph.edges.pre.slice();
  const post = graph.edges.post.slice();
  const weight = graph.edges.weight.slice();
  const m = pre.length;
  const swaps = Math.min(4 * m, 200000);
  for (let s = 0; s < swaps; s++) {
    const a = Math.floor(rand() * m);
    const b = Math.floor(rand() * m);
    const pa = pre[a];
    const pb = pre[b];
    if (post[a] === post[b] || pa === pb) continue;
    // Swap sources if no duplicate edge is introduced (approximate check).
    pre[a] = pb;
    pre[b] = pa;
  }
  return { ...graph, source: "synthetic-random", note: `${graph.note} [edge-shuffled]`, edges: { pre, post, weight } };
}

export interface ConnectomeJson {
  source: string;
  note?: string;
  neurons: { id: number; type?: string }[];
  synapses: { pre: number; post: number; weight?: number }[];
}

/**
 * Load an explicitly provided connectome JSON. The caller is responsible for
 * provenance/license (see docs/RENDERING_REFERENCES.md and T07). We do not ship
 * MaleCNS data or claim it is present.
 */
export function connectomeFromJson(json: ConnectomeJson): ReservoirGraph {
  const index = new Map<number, number>();
  json.neurons.forEach((neu, i) => index.set(neu.id, i));
  const pre = new Int32Array(json.synapses.length);
  const post = new Int32Array(json.synapses.length);
  const weight = new Float64Array(json.synapses.length);
  json.synapses.forEach((syn, i) => {
    pre[i] = index.get(syn.pre) ?? 0;
    post[i] = index.get(syn.post) ?? 0;
    weight[i] = syn.weight ?? 1;
  });
  const edges: EdgeList = { pre, post, weight };
  return {
    n: json.neurons.length,
    source: "malecns",
    note: json.note ?? json.source,
    edges: scaleEdges(edges, 0.9, json.neurons.length, 1234),
  };
}
