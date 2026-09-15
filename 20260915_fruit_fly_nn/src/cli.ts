/**
 * Headless CLI: train / evaluate / benchmark / make-video(placeholder).
 *
 * This file must not import renderer or DOM code so training stays headless.
 *
 * Examples:
 *   npx tsx src/cli.ts train --task phase_lock --generations 40 --out outputs/phase_lock
 *   npx tsx src/cli.ts evaluate --run outputs/phase_lock
 *   npx tsx src/cli.ts benchmark --out outputs/benchmark.json
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { replayRunSpec } from "./sim/runSpecReplay";
import { DEFAULT_TRAIN } from "./learning/train";
import {
  buildRun,
  defaultRunSpec,
  evaluate,
  evaluateRandom,
  flyTargetForTask,
  runTraining,
  targetForTask,
  type RunSpec,
  type TaskKind,
} from "./sim/runner";
import { linePlotSvg } from "./analysis/svgPlots";
import { runEpisode } from "./sim/connectomeController";
import { sampleFieldFastOnPlane } from "./optics/gaussian";
import { makeGrid } from "./optics/grid";
import { buildMalecnsGraph, defaultMalecnsPaths, malecnsCacheAvailable } from "./connectome/malecns";
import { RateReservoir } from "./connectome/reservoir";
import { OBSERVATION_SIZE } from "./sim/environment";

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        out[key] = next;
        i++;
      } else {
        out[key] = "true";
      }
    }
  }
  return out;
}

function ensureDir(filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
}

function writeJson(filePath: string, value: unknown): void {
  ensureDir(filePath);
  writeFileSync(filePath, JSON.stringify(value, null, 2));
}

function buildSpec(args: Record<string, string>): RunSpec {
  const task = (args.task as TaskKind) ?? "phase_lock";
  const spec = defaultRunSpec(task);
  if (args.reservoir) {
    const n = Number(args.reservoir);
    spec.reservoir.n = n;
  }
  if (args.outputCount) spec.reservoir.outputCount = Number(args.outputCount);
  if (args.degree) spec.reservoir.avgDegree = Number(args.degree);
  if (args.steps) spec.environment.episodeSteps = Number(args.steps);
  if (args.hidden !== undefined) spec.environment.hiddenPistonScale_rad = Number(args.hidden);
  if (args.connectome) {
    spec.connectomeJson = JSON.parse(readFileSync(args.connectome, "utf8"));
  }
  if (args.malecns) {
    spec.malecns = {
      maxNeurons: Number(args.malecns),
      seed: args.malecnsSeed ? Number(args.malecnsSeed) : 12345,
      minSynapses: args.minSynapses ? Number(args.minSynapses) : undefined,
      maxEdges: args.maxEdges ? Number(args.maxEdges) : undefined,
      cacheDir: args.cacheDir,
    };
    spec.reservoir.outputCount = spec.reservoir.outputCount ?? 64;
  }
  return spec;
}

function learningCurve(result: { history: { generation: number; best: number; mean: number }[] }): string {
  return linePlotSvg(
    [
      { label: "best", color: "#3fb950", points: result.history.map((h) => [h.generation, h.best]) },
      { label: "mean", color: "#58a6ff", points: result.history.map((h) => [h.generation, h.mean]) },
    ],
    { title: "Learning curve (ES readout)", xLabel: "generation", yLabel: "fitness" },
  );
}

function trackingPlot(traces: { targetSx: number[]; beamSx: number[]; targetSy: number[]; beamSy: number[] }): string {
  const n = traces.targetSx.length;
  const points = (series: number[]): [number, number][] =>
    series.map((v, i) => [i, v] as [number, number]);
  return linePlotSvg(
    [
      { label: "target x", color: "#f0883e", points: points(traces.targetSx) },
      { label: "beam x", color: "#3fb950", points: points(traces.beamSx) },
      { label: "target y", color: "#db6d28", points: points(traces.targetSy) },
      { label: "beam y", color: "#58a6ff", points: points(traces.beamSy) },
    ],
    { title: `Tracking (${n} steps)`, xLabel: "step", yLabel: "direction" },
  );
}

function commandTrain(args: Record<string, string>): void {
  const spec = buildSpec(args);
  const outDir = args.out ?? `outputs/${spec.task}`;
  if (args.generations) spec.train.generations = Number(args.generations);
  if (args.population) spec.train.population = Number(args.population);
  if (args.sigma) spec.train.sigma = Number(args.sigma);
  if (args.lr) spec.train.learningRate = Number(args.lr);

  spec.train = { ...DEFAULT_TRAIN, ...spec.train };
  writeJson(join(outDir, "run-spec.json"), spec);
  const progress: unknown[] = [];
  console.log(`[train] task=${spec.task} out=${outDir}`);
  console.log(`[train] reservoir n=${spec.reservoir.n} degree=${spec.reservoir.avgDegree}`);
  console.log(`[train] generations=${spec.train.generations ?? "default"}`);

  const outcome = runTraining(spec, (stats) => {
    progress.push(stats);
    writeJson(join(outDir, "progress.json"), { status: "running", history: progress });
    if (stats.generation % 10 === 0) {
      console.log(`  gen ${stats.generation}: best=${stats.best.toFixed(4)} mean=${stats.mean.toFixed(4)}`);
    }
  });

  const result = outcome.result;
  writeJson(join(outDir, "progress.json"), { status: "complete", history: result.history, initialFitness: result.initialFitness, finalFitness: result.finalFitness });
  console.log(`[train] initial=${result.initialFitness.toFixed(4)} final=${result.finalFitness.toFixed(4)} evals=${result.evaluations}`);

  writeJson(join(outDir, "learning-history.json"), result.history);
  writeFileSync(join(outDir, "learning-curve.svg"), learningCurve(result));
  writeFileSync(join(outDir, "tracking.svg"), trackingPlot(result.traces));

  const sameArray = evaluate(spec, result.readout.getParams(), [1]);
  const transfer = evaluate(spec, result.readout.getParams(), [2, 3, 4, 5]);
  const randomStrehl = evaluateRandom(spec, [1, 2, 3, 4, 5]);
  writeJson(join(outDir, "evaluation.json"), {
    sameArray: {
      meanReward: sameArray.meanReward,
      meanStrehl: sameArray.meanStrehl,
      meanPointing_rad: sameArray.meanPointing_rad,
    },
    transferToNewArrays: {
      meanReward: transfer.meanReward,
      meanStrehl: transfer.meanStrehl,
      meanPointing_rad: transfer.meanPointing_rad,
    },
    randomStrehlBaseline: randomStrehl,
    note: "sameArray is the training hidden-error realization; transfer uses unseen realizations. A large gap means per-array memorization, not a general policy.",
  });

  // Trained readout weights, for reproducibility and inspection.
  writeJson(join(outDir, "readout.json"), { params: Array.from(result.readout.getParams()) });
  writeFileSync(
    join(outDir, "manifest.json"),
    JSON.stringify(
      {
        task: spec.task,
        createdAt: new Date().toISOString(),
        node: process.version,
        array: { id: spec.arrayOverrides.id ?? "default-19ch" },
        reservoir: { ...spec.reservoir },
        environment: spec.environment,
        train: spec.train,
        connectomeSource: spec.malecns
          ? "malecns-v1.0-cache-subset"
          : spec.connectomeJson
            ? "provided-json"
            : "synthetic-random",
        malecns: spec.malecns ?? null,
        note: "Synthetic reservoir unless a connectome JSON or --malecns cache was supplied. MaleCNS subset uses all-positive placeholder signs; transmitter signs TODO.",
      },
      null,
      2,
    ),
  );
  console.log(`[train] wrote ${outDir}/manifest.json, learning-curve.svg, tracking.svg, evaluation.json`);
}

function commandEvaluate(args: Record<string, string>): void {
  const runDir = args.run ?? args.out;
  if (!runDir) throw new Error("evaluate: --run <dir> required");
  const manifest = JSON.parse(readFileSync(join(runDir, "manifest.json"), "utf8"));
  const readoutJson = JSON.parse(readFileSync(join(runDir, "readout.json"), "utf8"));
  const savedSpecPath = join(runDir, "run-spec.json");
  const spec = replayRunSpec(manifest, existsSync(savedSpecPath) ? JSON.parse(readFileSync(savedSpecPath, "utf8")) : undefined);
  const target = args.fly === "true" ? flyTargetForTask() : targetForTask(spec.task);
  const outcome = evaluate(spec, Float64Array.from(readoutJson.params as number[]), [1, 2, 3, 4, 5], target);
  const out = {
    meanReward: outcome.meanReward,
    meanStrehl: outcome.meanStrehl,
    meanPointing_rad: outcome.meanPointing_rad,
    randomStrehlBaseline: evaluateRandom(spec, [1, 2, 3, 4, 5]),
    note: "Averages over hidden-error realizations 1..5.",
  };
  console.log(`[evaluate] reward=${out.meanReward.toFixed(4)} strehl=${out.meanStrehl.toFixed(4)} pointing=${out.meanPointing_rad.toExponential(3)}`);
  console.log(`[evaluate] random baseline strehl=${out.randomStrehlBaseline.toFixed(4)}`);
  writeJson(join(runDir, args.fly === "true" ? "evaluation-fly.json" : "reevaluation.json"), out);
}

function commandBenchmark(args: Record<string, string>): void {
  const out = args.out ?? "outputs/benchmark.json";
  const built = buildRun(defaultRunSpec());
  const array = built.array;
  const grid = makeGrid(256, 20e-6);

  const baseCommands = built.controller.commands(
    {
      targetSx: 0,
      targetSy: 0,
      beamSx: 0,
      beamSy: 0,
      deltaSx: 0,
      deltaSy: 0,
      peak: 1,
      pib: 0,
    },
    { sx: 0, sy: 0, sz: 1 },
  );
  const actual = built.env.setCommands(baseCommands);

  const t0 = performance.now();
  const reps = 20;
  for (let r = 0; r < reps; r++) sampleFieldFastOnPlane(array, actual, 0, grid);
  const t1 = performance.now();
  const planeSamples = grid.n * grid.n * reps;

  const t2 = performance.now();
  const ep = runEpisode(built.env, built.controller, { seed: 1, steps: 200 });
  const t3 = performance.now();

  // Real MaleCNS graph loader/step benchmark (when the cache is present).
  let malecns: Record<string, number> | null = null;
  if (malecnsCacheAvailable()) {
    const t4 = performance.now();
    const { graph } = buildMalecnsGraph(defaultMalecnsPaths(), {
      maxNeurons: 5000,
      seed: 1,
      maxEdges: 300000,
    });
    const t5 = performance.now();
    const reservoir = new RateReservoir(graph, OBSERVATION_SIZE, { outputCount: 64, seed: 1 });
    const input = new Float64Array(OBSERVATION_SIZE);
    reservoir.reset();
    const t6 = performance.now();
    const steps = 200;
    for (let i = 0; i < steps; i++) {
      input[i % OBSERVATION_SIZE] = Math.sin(i * 0.1);
      reservoir.step(input);
    }
    const t7 = performance.now();
    malecns = {
      nodes: graph.n,
      edges: graph.edges.pre.length,
      loadMs: t5 - t4,
      reservoirStepsPerSecond: steps / ((t7 - t6) / 1000),
      rssMB: Math.round(process.memoryUsage().rss / 1048576),
    };
  }

  const report = {
    array: array.id,
    grid: `${grid.n}x${grid.n} dx=${grid.dx}`,
    fastFieldSamplesPerSecond: planeSamples / ((t1 - t0) / 1000),
    closedLoopStepsPerSecond: 200 / ((t3 - t2) / 1000),
    meanReward: ep.meanReward,
    reservoir: built.graph.source,
    malecns,
    node: process.version,
    hardware: `${process.platform} ${process.arch}`,
  };
  console.log(JSON.stringify(report, null, 2));
  writeJson(out, report);
}

function main(): void {
  const [command, ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);
  switch (command) {
    case "train":
      commandTrain(args);
      break;
    case "evaluate":
      commandEvaluate(args);
      break;
    case "benchmark":
      commandBenchmark(args);
      break;
    default:
      console.log("Usage: tsx src/cli.ts <train|evaluate|benchmark> [options]");
      process.exitCode = 1;
  }
}

main();
