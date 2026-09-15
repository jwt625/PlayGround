import { describe, expect, it } from "vitest";
import { buildMalecnsGraph, defaultMalecnsPaths, malecnsCacheAvailable } from "../src/connectome/malecns";

const available = malecnsCacheAvailable();

describe.skipIf(!available)("T07 MaleCNS v1.0 graph loader", () => {
  it("builds a deterministic subset with preserved ascending body IDs", () => {
    const a = buildMalecnsGraph(defaultMalecnsPaths(), { maxNeurons: 2000, seed: 7, maxEdges: 100000 });
    const b = buildMalecnsGraph(defaultMalecnsPaths(), { maxNeurons: 2000, seed: 7, maxEdges: 100000 });

    expect(a.graph.n).toBe(2000);
    expect(a.graph.source).toBe("malecns");
    expect(a.graph.bodyIds?.length).toBe(2000);
    expect(a.graph.edges.pre.length).toBeGreaterThan(0);
    expect(a.graph.edges.pre.length).toBeLessThanOrEqual(100000);

    const ids = a.graph.bodyIds!;
    let ascending = true;
    for (let i = 1; i < ids.length; i++) if (ids[i] <= ids[i - 1]) ascending = false;
    expect(ascending).toBe(true);

    const { pre, post } = a.graph.edges;
    let inRange = true;
    for (let i = 0; i < pre.length; i++) {
      if (pre[i] < 0 || pre[i] >= a.graph.n || post[i] < 0 || post[i] >= a.graph.n) inRange = false;
    }
    expect(inRange).toBe(true);

    // Deterministic replay.
    expect(Array.from(a.graph.edges.pre.slice(0, 50))).toEqual(Array.from(b.graph.edges.pre.slice(0, 50)));
    expect(a.totalNodesAvailable).toBeGreaterThan(150000);
    expect(a.totalEdgesAvailable).toBeGreaterThan(25000000);
  }, 120_000);
});
