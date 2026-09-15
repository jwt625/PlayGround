import { describe, expect, it } from "vitest";
import { replayRunSpec } from "../src/sim/runSpecReplay";
import { defaultRunSpec } from "../src/sim/runner";

describe("checkpoint configuration replay", () => {
  it("retains real graph selection and nondefault training/environment dimensions", () => {
    const spec = defaultRunSpec("tracking");
    spec.malecns = { maxNeurons: 5000, seed: 97, maxEdges: 100000 };
    spec.reservoir.outputCount = 32;
    spec.environment.episodeSteps = 73;
    spec.arrayOverrides.pitch_m = 0.0004;
    expect(replayRunSpec({ task: "phase_lock" }, JSON.parse(JSON.stringify(spec)))).toEqual(spec);
  });
  it("restores available fields from legacy manifests and refuses unrecoverable JSON graphs", () => {
    expect(replayRunSpec({task:"phase_lock",malecns:{maxNeurons:3000,seed:7},reservoir:{outputCount:16}}).malecns?.maxNeurons).toBe(3000);
    expect(()=>replayRunSpec({task:"phase_lock",connectomeSource:"provided-json"})).toThrow("lacks its graph");
  });
});
