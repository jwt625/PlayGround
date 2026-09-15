import type { ArrayConfig } from "../optics/geometry";
import { analyticSteeringCommands } from "../controllers/analytic";
import type { Direction } from "../optics/metrics";
import type { ChannelCommand } from "../optics/types";
import type { RateReservoir } from "../connectome/reservoir";
import type { LinearReadout } from "../learning/readout";
import {
  observationToVector,
  type Environment,
  type Observation,
  type StepMetrics,
} from "./environment";

/**
 * Connectome-driven controller: a fixed reservoir receives the sensory
 * observation, a small trained readout maps selected neuron activity to piston
 * corrections, and analytic steering supplies the common tip/tilt plus base
 * piston ramp. The connectome materially changes the actuator commands.
 */
export interface ConnectomeControllerOptions {
  actionMode: "absolute" | "incremental";
  maxPhaseStep_rad: number;
}

export const DEFAULT_CONTROLLER_OPTIONS: ConnectomeControllerOptions = {
  actionMode: "absolute",
  maxPhaseStep_rad: 0.5,
};

export class ConnectomeController {
  readonly array: ArrayConfig;
  readonly reservoir: RateReservoir;
  readonly readout: LinearReadout;
  readonly options: ConnectomeControllerOptions;
  private phi: Float64Array;

  constructor(
    array: ArrayConfig,
    reservoir: RateReservoir,
    readout: LinearReadout,
    options: Partial<ConnectomeControllerOptions> = {},
  ) {
    this.array = array;
    this.reservoir = reservoir;
    this.readout = readout;
    this.options = { ...DEFAULT_CONTROLLER_OPTIONS, ...options };
    this.phi = new Float64Array(array.channelIds.length);
  }

  reset(): void {
    this.reservoir.reset();
    this.phi.fill(0);
  }

  commands(observation: Observation, targetDir: Direction): ChannelCommand[] {
    const base = analyticSteeringCommands(this.array, targetDir);
    this.reservoir.step(observationToVector(observation));
    const features = this.reservoir.features();
    const u = this.readout.predict(features);
    const commands: ChannelCommand[] = new Array(base.length);
    for (let i = 0; i < base.length; i++) {
      let correction: number;
      if (this.options.actionMode === "incremental") {
        this.phi[i] += this.options.maxPhaseStep_rad * u[i];
        correction = this.phi[i];
      } else {
        correction = Math.PI * u[i];
      }
      commands[i] = { ...base[i], piston_rad: base[i].piston_rad + correction };
    }
    return commands;
  }
}

export interface EpisodeTraces {
  targetSx: number[];
  targetSy: number[];
  beamSx: number[];
  beamSy: number[];
  targetIntensity: number[];
  pib: number[];
  reward: number[];
}

export interface EpisodeResult {
  meanReward: number;
  finalReward: number;
  rewards: number[];
  metrics: StepMetrics[];
  traces: EpisodeTraces;
}

export interface EpisodeOptions {
  seed: number;
  steps?: number;
}

/** Run one closed-loop episode: reservoir -> readout -> commands -> optics. */
export function runEpisode(
  env: Environment,
  controller: ConnectomeController,
  options: EpisodeOptions,
): EpisodeResult {
  const steps = options.steps ?? env.config.episodeSteps;
  env.reset(options.seed);
  controller.reset();

  const base = analyticSteeringCommands(env.array, env.targetDirection());
  let observation = env.observeWithCommands(base);

  const rewards: number[] = [];
  const metrics: StepMetrics[] = [];
  const traces: EpisodeTraces = {
    targetSx: [],
    targetSy: [],
    beamSx: [],
    beamSy: [],
    targetIntensity: [],
    pib: [],
    reward: [],
  };

  for (let step = 0; step < steps; step++) {
    const targetDir = env.targetDirection();
    const commands = controller.commands(observation, targetDir);
    const result = env.step(commands);
    rewards.push(result.reward);
    metrics.push(result.metrics);
    traces.targetSx.push(result.metrics.targetSx);
    traces.targetSy.push(result.metrics.targetSy);
    traces.beamSx.push(result.metrics.beamSx);
    traces.beamSy.push(result.metrics.beamSy);
    traces.targetIntensity.push(result.metrics.targetIntensity);
    traces.pib.push(result.metrics.pib);
    traces.reward.push(result.reward);
    observation = result.observation;
  }

  const meanReward = rewards.reduce((a, b) => a + b, 0) / Math.max(1, rewards.length);
  return {
    meanReward,
    finalReward: rewards[rewards.length - 1] ?? 0,
    rewards,
    metrics,
    traces,
  };
}
