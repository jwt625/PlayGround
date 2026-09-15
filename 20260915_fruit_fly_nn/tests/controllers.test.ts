import { describe, expect, it } from "vitest";
import { createArrayConfig } from "../src/optics/geometry";
import { randomHiddenErrors, resolveActual, uniformCommands } from "../src/optics/channels";
import { normalizedFarField, strehlAtDirection } from "../src/optics/metrics";
import { makeDirectionObjective } from "../src/controllers/objective";
import { runSpgd } from "../src/controllers/spgd";
import { randomPistons } from "../src/controllers/random";
import { analyticSteeringCommands } from "../src/controllers/analytic";

const config = createArrayConfig();
const onAxis = { sx: 0, sy: 0, sz: 1 };

function strehlForPistons(pistons: readonly number[], seed: number): number {
  const hidden = randomHiddenErrors(config.channelIds.length, seed, 1, 0);
  const cmds = uniformCommands(config);
  pistons.forEach((p, i) => (cmds[i].piston_rad = p));
  const actual = resolveActual(config, cmds, hidden);
  return strehlAtDirection(config, actual, onAxis);
}

describe("T03 controllers", () => {
  it("analytic reference phase-locks the array on-axis at unit Strehl", () => {
    const cmds = analyticSteeringCommands(config, onAxis);
    const actual = resolveActual(config, cmds, randomHiddenErrors(config.channelIds.length, 5, 0, 0));
    expect(normalizedFarField(config, actual, onAxis)).toBeCloseTo(1, 9);
  });

  it("random controller fails as expected", () => {
    let mean = 0;
    const trials = 40;
    for (let s = 0; s < trials; s++) {
      const pistons = randomPistons(config, 100 + s);
      mean += strehlForPistons(pistons, 1);
    }
    mean /= trials;
    expect(mean).toBeLessThan(0.2);
  });

  it("SPGD recovers static phase alignment across 10 fixed seeds", () => {
    const results: { seed: number; strehl: number; start: number; evals: number }[] = [];
    for (let seed = 0; seed < 10; seed++) {
      const hidden = randomHiddenErrors(config.channelIds.length, 1000 + seed, 1.5, 0);
      const objective = makeDirectionObjective(config, hidden, onAxis);
      const initial = randomPistons(config, 2000 + seed);
      const start = objective(initial);
      const result = runSpgd(objective, initial, {
        iterations: 500,
        gain: 1.0,
        perturbation: 0.3,
        seed: 3000 + seed,
      });
      const strehl = strehlForPistons(result.pistons, 1000 + seed); // same hidden set
      expect(result.evaluations).toBe(500 * 2);
      results.push({ seed, strehl, start, evals: result.evaluations });
    }

    const sorted = [...results].sort((a, b) => a.strehl - b.strehl);
    const successes = results.filter((r) => r.strehl > 0.8).length;
    // Report the distribution; require a strong majority, not perfection.
    console.log(
      "SPGD Strehl distribution:",
      results.map((r) => r.strehl.toFixed(3)).join(", "),
    );
    expect(successes).toBeGreaterThanOrEqual(8);
    expect(sorted[0].strehl).toBeGreaterThan(0.6);
    // Every run must improve substantially from its random start.
    for (const r of results) expect(r.strehl).toBeGreaterThan(r.start + 0.3);
  });

  it("SPGD is reproducible for a fixed seed", () => {
    const hidden = randomHiddenErrors(config.channelIds.length, 7, 1, 0);
    const objective = makeDirectionObjective(config, hidden, onAxis);
    const initial = randomPistons(config, 8);
    const a = runSpgd(objective, initial, { seed: 42, iterations: 50 });
    const b = runSpgd(objective, initial, { seed: 42, iterations: 50 });
    expect(a.pistons).toEqual(b.pistons);
    expect(a.history).toEqual(b.history);
  });
});
