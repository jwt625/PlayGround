import type { Environment } from "../sim/environment";
import type { ConnectomeController, EpisodeTraces } from "../sim/connectomeController";
import { runEpisode } from "../sim/connectomeController";
import { trainEs, type EsOptions, type GenerationStats } from "./es";
import type { LinearReadout } from "./readout";

export interface TrainConfig {
  generations: number;
  population: number;
  sigma: number;
  learningRate: number;
  esSeed: number;
  /** Evaluation seeds (hidden-error realizations) averaged per candidate. */
  evalSeeds: number[];
}

export const DEFAULT_TRAIN: TrainConfig = {
  generations: 40,
  population: 16,
  sigma: 0.1,
  learningRate: 0.05,
  esSeed: 7,
  evalSeeds: [1],
};

export interface TrainResult {
  readout: LinearReadout;
  initialFitness: number;
  finalFitness: number;
  history: GenerationStats[];
  evaluations: number;
  traces: EpisodeTraces;
}

export function trainReadout(
  env: Environment,
  controller: ConnectomeController,
  config: Partial<TrainConfig> = {},
  onGeneration?: (stats: GenerationStats) => void,
): TrainResult {
  const cfg = { ...DEFAULT_TRAIN, ...config };
  const steps = env.config.episodeSteps;

  const fitness = (params: Float64Array): number => {
    controller.readout.setParams(params);
    let sum = 0;
    for (const seed of cfg.evalSeeds) {
      sum += runEpisode(env, controller, { seed, steps }).meanReward;
    }
    return sum / cfg.evalSeeds.length;
  };

  const initial = controller.readout.getParams();
  const initialFitness = fitness(initial);
  const esOptions: Partial<EsOptions> = {
    population: cfg.population,
    sigma: cfg.sigma,
    learningRate: cfg.learningRate,
    seed: cfg.esSeed,
  };
  const result = trainEs(initial, fitness, esOptions, cfg.generations, onGeneration);
  controller.readout.setParams(result.params);
  const traces = runEpisode(env, controller, { seed: cfg.evalSeeds[0], steps }).traces;
  return {
    readout: controller.readout,
    initialFitness,
    finalFitness: fitness(result.params),
    history: result.history,
    evaluations: result.evaluations,
    traces,
  };
}
