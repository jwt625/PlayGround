import { describe, expect, it } from "vitest";
import { createArrayConfig } from "../src/optics/geometry";
import { generateSyntheticGraph, edgeShuffled } from "../src/connectome/graph";
import { RateReservoir } from "../src/connectome/reservoir";
import { LinearReadout } from "../src/learning/readout";
import { trainReadout } from "../src/learning/train";
import { Environment, OBSERVATION_SIZE, defaultEnvironmentConfig } from "../src/sim/environment";
import { ConnectomeController } from "../src/sim/connectomeController";
import { staticTarget } from "../src/sim/target";

const array = createArrayConfig();

function makeSetup(graph = generateSyntheticGraph({ n: 120, avgDegree: 8, seed: 1 })) {
  const reservoir = new RateReservoir(graph, OBSERVATION_SIZE, {
    outputCount: 32,
    seed: 2,
    noise: 0.02,
  });
  const readout = new LinearReadout(array.channelIds.length, reservoir.outputCount, 3);
  const env = new Environment(
    defaultEnvironmentConfig(array, {
      target: staticTarget(0, 0, 1),
      hiddenPistonScale_rad: 1.0,
      hiddenGainScale: 0,
      episodeSteps: 60,
      scanSamples: 11,
      seed: 1,
    }),
  );
  const controller = new ConnectomeController(array, reservoir, readout);
  return { reservoir, readout, env, controller };
}

describe("T07 connectome learning loop", () => {
  it("ES training improves the closed-loop reward", () => {
    const { env, controller } = makeSetup();
    const result = trainReadout(env, controller, {
      generations: 40,
      population: 10,
      sigma: 0.1,
      learningRate: 0.05,
      esSeed: 7,
      evalSeeds: [1],
    });
    console.log(
      `connectome readout fitness: initial=${result.initialFitness.toFixed(4)} final=${result.finalFitness.toFixed(4)} evals=${result.evaluations}`,
    );
    expect(result.finalFitness).toBeGreaterThan(result.initialFitness + 0.1);
    expect(result.history.length).toBe(40);
    // Best-in-generation should trend upward over the second half.
    const early = result.history.slice(0, 10).reduce((a, s) => a + s.best, 0) / 10;
    const late = result.history.slice(-10).reduce((a, s) => a + s.best, 0) / 10;
    expect(late).toBeGreaterThan(early);
  });

  it("matched ablations run and are reported (no overclaim)", () => {
    const real = makeSetup();
    const shuffled = makeSetup(edgeShuffled(real.reservoir.graph, 99));
    const config = {
      generations: 20,
      population: 8,
      sigma: 0.1,
      learningRate: 0.05,
      esSeed: 5,
      evalSeeds: [1],
    };
    const realResult = trainReadout(real.env, real.controller, config);
    const shuffledResult = trainReadout(shuffled.env, shuffled.controller, config);
    console.log(
      `ablation final fitness: real=${realResult.finalFitness.toFixed(4)} edgeShuffled=${shuffledResult.finalFitness.toFixed(4)}`,
    );
    expect(realResult.finalFitness).toBeGreaterThan(realResult.initialFitness);
    expect(shuffledResult.finalFitness).toBeGreaterThan(shuffledResult.initialFitness);
  });
});
