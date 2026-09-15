import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { scaleEdges, type EdgeList, type ReservoirGraph } from "./graph";

/**
 * Typed-array loader for the cached MaleCNS v1.0 derived graph.
 *
 * Files (see data/README.md):
 *   derived/body-ids.npy  int64, ascending original body IDs
 *   derived/edges.u32     headerless little-endian uint32 triples
 *                         (pre_index, post_index, raw_synapse_count)
 *
 * Indices refer to ascending body IDs. We never load the graph as JSON objects.
 *
 * Modelling assumptions (explicit, pending reviewer sign-off):
 *  - This is a documented selection subset (`status == 'Traced'`), not the
 *    paper's neuron total.
 *  - Raw synapse counts are used as magnitudes. Transmitter-based excitatory /
 *    inhibitory signs are NOT applied; parsing `neurotransmitters.feather`
 *    requires an Arrow reader and remains TODO. The reservoir therefore uses a
 *    placeholder all-positive sign and renormalizes weights; this is labelled
 *    in `note`, not presented as a biological sign assignment.
 */

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

function alignedUint32(buf: Buffer, byteOffset: number, byteLength: number): Uint32Array {
  const arrayBuffer = buf.buffer.slice(buf.byteOffset + byteOffset, buf.byteOffset + byteOffset + byteLength);
  return new Uint32Array(arrayBuffer);
}

/** Read a C-order int64 .npy file into a number array (values must fit in 2^53). */
export function readNpyInt(path: string): Int32Array {
  const buf = readFileSync(path);
  for (let i = 0; i < 6; i++) {
    if (buf[i] !== NPY_MAGIC[i]) throw new Error(`${path}: not an npy file`);
  }
  const major = buf[6];
  let headerLen: number;
  let headerStart: number;
  if (major === 1) {
    headerLen = buf.readUInt16LE(8);
    headerStart = 10;
  } else if (major === 2) {
    headerLen = buf.readUInt32LE(8);
    headerStart = 12;
  } else if (major === 3) {
    headerLen = buf.readUInt32LE(8);
    headerStart = 16;
  } else {
    throw new Error(`${path}: unsupported npy version ${major}`);
  }
  const header = buf.toString("latin1", headerStart, headerStart + headerLen);
  if (!header.includes("'i8'") && !header.includes("'<i8'")) {
    throw new Error(`${path}: expected little-endian int64, header=${header.trim()}`);
  }
  const shapeMatch = header.match(/\((\d+),?\)/);
  if (!shapeMatch) throw new Error(`${path}: cannot parse shape`);
  const count = Number(shapeMatch[1]);
  const dataOffset = headerStart + headerLen;
  const dataLen = count * 8;
  if (buf.byteLength < dataOffset + dataLen) throw new Error(`${path}: truncated int64 data`);

  const copied = buf.buffer.slice(buf.byteOffset + dataOffset, buf.byteOffset + dataOffset + dataLen);
  const int64 = new BigInt64Array(copied);
  const out = new Int32Array(count);
  for (let i = 0; i < count; i++) {
    const v = int64[i];
    if (v < -2147483648n || v > 2147483647n) throw new Error(`${path}: id out of int32 range`);
    out[i] = Number(v);
  }
  return out;
}

export interface MalecnsCachePaths {
  bodyIds: string;
  edges: string;
}

export interface MalecnsSelectionOptions {
  /** Number of neurons to sample (subset). */
  maxNeurons: number;
  seed: number;
  /** Drop connections with fewer than this many raw synapses. */
  minSynapses?: number;
  /** Target reservoir spectral-radius scaling. */
  targetSpectralRadius?: number;
  /** Optional hard cap on retained edges, for memory/time budgets. */
  maxEdges?: number;
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

export interface MalecnsGraphResult {
  graph: ReservoirGraph;
  bodyIds: Int32Array;
  totalNodesAvailable: number;
  totalEdgesAvailable: number;
}

/**
 * Build a reservoir graph from the cached MaleCNS derived files. The subset is
 * sampled deterministically over ascending body-ID indices so replays match.
 */
export function buildMalecnsGraph(
  paths: MalecnsCachePaths,
  options: MalecnsSelectionOptions,
): MalecnsGraphResult {
  if (!existsSync(paths.bodyIds)) throw new Error(`MaleCNS body ids not found: ${paths.bodyIds}`);
  if (!existsSync(paths.edges)) throw new Error(`MaleCNS edges not found: ${paths.edges}`);

  const allIds = readNpyInt(paths.bodyIds);
  const nAll = allIds.length;
  const choose = Math.min(options.maxNeurons, nAll);
  const rand = mulberry32(options.seed);

  // Pick `choose` distinct indices, then order them ascending so remapping is stable.
  const picked = new Set<number>();
  while (picked.size < choose) picked.add(Math.floor(rand() * nAll));
  const sorted = Array.from(picked).sort((a, b) => allIds[a] - allIds[b]);
  const newIndex = new Int32Array(nAll).fill(-1);
  const bodyIds = new Int32Array(choose);
  for (let i = 0; i < choose; i++) {
    newIndex[sorted[i]] = i;
    bodyIds[i] = allIds[sorted[i]];
  }

  const edgeBuf = readFileSync(paths.edges);
  const triples = alignedUint32(edgeBuf, 0, edgeBuf.byteLength - (edgeBuf.byteLength % 12));
  const totalEdgesAvailable = triples.length / 3;
  const minSynapses = options.minSynapses ?? 1;
  const maxEdges = options.maxEdges ?? Number.POSITIVE_INFINITY;

  const pre: number[] = [];
  const post: number[] = [];
  const weight: number[] = [];
  for (let k = 0; k < triples.length; k += 3) {
    const w = triples[k + 2];
    if (w < minSynapses) continue;
    const a = newIndex[triples[k]];
    if (a < 0) continue;
    const b = newIndex[triples[k + 1]];
    if (b < 0) continue;
    pre.push(a);
    post.push(b);
    weight.push(w);
    if (pre.length >= maxEdges) break;
  }

  const edges: EdgeList = {
    pre: Int32Array.from(pre),
    post: Int32Array.from(post),
    weight: Float64Array.from(weight),
  };
  const scaled = scaleEdges(edges, options.targetSpectralRadius ?? 0.9, choose, options.seed ^ 0x85ebca6b);
  const graph: ReservoirGraph = {
    n: choose,
    source: "malecns",
    note: `MaleCNS v1.0 derived subset: ${choose}/${nAll} nodes, raw synapse magnitudes, all-positive placeholder signs (transmitter signs TODO)`,
    edges: scaled,
    bodyIds,
  };
  return { graph, bodyIds, totalNodesAvailable: nAll, totalEdgesAvailable };
}

export function defaultMalecnsPaths(cacheDir = "data/cache/malecns-v1.0"): MalecnsCachePaths {
  return {
    bodyIds: join(cacheDir, "derived", "body-ids.npy"),
    edges: join(cacheDir, "derived", "edges.u32"),
  };
}

export function malecnsCacheAvailable(cacheDir = "data/cache/malecns-v1.0"): boolean {
  const p = defaultMalecnsPaths(cacheDir);
  return existsSync(p.bodyIds) && existsSync(p.edges);
}
