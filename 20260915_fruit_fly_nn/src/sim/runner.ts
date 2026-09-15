import { createArrayConfig, type ArrayConfigOverrides } from "../optics/geometry";
import {
  connectomeFromJson,
  generateSyntheticGraph,
  type ConnectomeJson,
  type ReservoirGraph,
} from "../connectome/graph";
import { buildMalecnsGraph, defaultMalecnsPaths } from "../connectome/malecns";
import { RateReservoir, type ReservoirConfig } from "../connectome/reservoir";
import { LinearReadout } from "../learning/readout";
import { trainReadout, type TrainConfig, type TrainResult } from "../learning/train";
import { Environment, OBSERVATION_SIZE, defaultEnvironmentConfig, type EnvironmentConfig } from "./environment";
import { ConnectomeController, runEpisode, type EpisodeResult } from "./connectomeController";
import { staticTarget, lissajousTarget, circleTarget, flyTarget, type TargetMotion } from "./target";
import { randomHiddenErrors, resolveActual, uniformCommands, type HiddenChannelErrors } from "../optics/channels";
import { strehlAtDirection } from "../optics/metrics";
import { randomPistons } from "../controllers/random";

export type TaskKind = "phase_lock" | "steering" | "tracking";

export interface RunSpec {
  task: TaskKind;
  arrayOverrides: ArrayConfigOverrides;
  reservoir: Partial<ReservoirConfig> & { n?: number; avgDegree?: number; clusterSize?: number };
  environment: {
    episodeSteps: number;
    hiddenPistonScale_rad: number;
    hiddenGainScale: number;
    scanSamples: number;
    dt_s: number;
  };
  train: Partial<TrainConfig>;
  connectomeJson?: ConnectomeJson;
  /** Use the cached MaleCNS v1.0 derived graph (subset selection). */
  malecns?: {
    cacheDir?: string;
    maxNeurons: number;
    seed: number;
    minSynapses?: number;
    maxEdges?: number;
  };
}

export function defaultRunSpec(task: TaskKind = "phase_lock"): RunSpec {
  return {
    task,
    arrayOverrides: {},
    reservoir: { n: 600, avgDegree: 10, outputCount: 64, seed: 12345, noise: 0.01 },
    environment: {
      episodeSteps: 120,
      hiddenPistonScale_rad: 1.0,
      hiddenGainScale: 0.1,
      scanSamples: 13,
      dt_s: 1 / 60,
    },
    train: {},
  };
}

export function targetForTask(task: TaskKind): TargetMotion {
  switch (task) {
    case "phase_lock":
      return staticTarget(0, 0, 1);
    case "steering":
      return staticTarget(0.015, -0.01, 1);
    case "tracking":
      return lissajousTarget({ z_m: 1, amplitudeX_m: 0.012, amplitudeY_m: 0.008, freqX_hz: 0.11, freqY_hz: 0.17 });
  }
}

export function flyTargetForTask(): TargetMotion {
  return flyTarget({ z_m: 1, extentX_m: 0.012, extentY_m: 0.008, speed_mps: 0.03, seed: 4 });
}

export interface BuiltRun {
  graph: ReservoirGraph;
  graphNote: string;
  reservoir: RateReservoir;
  readout: LinearReadout;
  env: Environment;
  controller: ConnectomeController;
  array: ReturnType<typeof createArrayConfig>;
}

export function buildRun(spec: RunSpec, target: TargetMotion = targetForTask(spec.task)): BuiltRun {
  const array = createArrayConfig(spec.arrayOverrides);
  let graph: ReservoirGraph;
  if (spec.malecns) {
    graph = buildMalecnsGraph(defaultMalecnsPaths(spec.malecns.cacheDir), {
      maxNeurons: spec.malecns.maxNeurons,
      seed: spec.malecns.seed,
      minSynapses: spec.malecns.minSynapses,
      maxEdges: spec.malecns.maxEdges,
    }).graph;
  } else if (spec.connectomeJson) {
    graph = connectomeFromJson(spec.connectomeJson);
  } else {
    graph = generateSyntheticGraph({
      n: spec.reservoir.n ?? 600,
      avgDegree: spec.reservoir.avgDegree ?? 10,
      seed: spec.reservoir.seed ?? 12345,
      clusterSize: spec.reservoir.clusterSize ?? 1,
    });
  }
  const reservoir = new RateReservoir(graph, OBSERVATION_SIZE, spec.reservoir);
  const readout = new LinearReadout(array.channelIds.length, reservoir.outputCount, spec.reservoir.seed ?? 1);
  const envConfig: EnvironmentConfig = defaultEnvironmentConfig(array, {
    target,
    episodeSteps: spec.environment.episodeSteps,
    hiddenPistonScale_rad: spec.environment.hiddenPistonScale_rad,
    hiddenGainScale: spec.environment.hiddenGainScale,
    scanSamples: spec.environment.scanSamples,
    dt_s: spec.environment.dt_s,
    seed: 1,
  });
  const env = new Environment(envConfig);
  const controller = new ConnectomeController(array, reservoir, readout);
  return { graph, graphNote: graph.note, reservoir, readout, env, controller, array };
}

export interface TrainingOutcome {
  result: TrainResult;
  graphInfo: { source: string; nodes: number; edges: number; note: string };
  spec: RunSpec;
  array: ReturnType<typeof createArrayConfig>;
}

export function runTraining(
  spec: RunSpec,
  onGeneration?: Parameters<typeof trainReadout>[3],
): TrainingOutcome {
  const built = buildRun(spec);
  const result = trainReadout(built.env, built.controller, spec.train, onGeneration);
  return { result, spec, array: built.array, graphInfo: { source: built.graph.source, nodes: built.graph.n, edges: built.graph.edges.pre.length, note: built.graph.note } };
}

export interface EvaluationOutcome {
  meanReward: number;
  meanStrehl: number;
  meanPointing_rad: number;
  result: EpisodeResult;
}

/** Ground-truth evaluation using the trained readout over fixed seeds. */
export function evaluate(
  spec: RunSpec,
  params: Float64Array,
  seeds: number[] = [1, 2, 3],
  target: TargetMotion = targetForTask(spec.task),
): EvaluationOutcome {
  const built = buildRun(spec, target);
  built.controller.readout.setParams(params);
  let reward = 0;
  let strehl = 0;
  let pointing = 0;
  let last: EpisodeResult | null = null;
  for (const seed of seeds) {
    const episode = runEpisode(built.env, built.controller, { seed });
    reward += episode.meanReward;
    const m = episode.metrics;
    strehl += m.reduce((a, x) => a + x.targetIntensity, 0) / m.length;
    pointing += m.reduce((a, x) => a + x.pointingAngle_rad, 0) / m.length;
    last = episode;
  }
  const n = seeds.length;
  return {
    meanReward: reward / n,
    meanStrehl: strehl / n,
    meanPointing_rad: pointing / n,
    result: last!,
  };
}

/** Baselines for the comparison table. */
export function evaluateRandom(spec: RunSpec, seeds: number[] = [1, 2, 3]): number {
  const array = createArrayConfig(spec.arrayOverrides);
  const hidden: HiddenChannelErrors[] = randomHiddenErrors(array.channelIds.length, 1, spec.environment.hiddenPistonScale_rad, spec.environment.hiddenGainScale);
  let sum = 0;
  for (const seed of seeds) {
    const pistons = randomPistons(array, seed);
    const cmds = uniformCommands(array);
    pistons.forEach((p, i) => (cmds[i].piston_rad = p));
    const actual = resolveActual(array, cmds, hidden);
    sum += strehlAtDirection(array, actual, { sx: 0, sy: 0, sz: 1 });
  }
  return sum / seeds.length;
}

export { circleTarget };
