import * as THREE from "three";
import { createArrayConfig } from "./optics/geometry";
import { wrapPiston } from "./optics/channels";
import { idealSteeringPistons, phaseRmsRad } from "./optics/metrics";
import { analyticSteeringCommands } from "./controllers/analytic";
import { runSpgd } from "./controllers/spgd";
import { makeDirectionObjective } from "./controllers/objective";
import { Environment, defaultEnvironmentConfig, type Observation } from "./sim/environment";
import { ConnectomeController } from "./sim/connectomeController";
import { staticTarget, lissajousTarget, flyTarget, circleTarget, type TargetMotion } from "./sim/target";
import { generateSyntheticGraph } from "./connectome/graph";
import { RateReservoir } from "./connectome/reservoir";
import { LinearReadout } from "./learning/readout";
import { trainReadout } from "./learning/train";
import { createScene, DISPLAY } from "./renderer/scene";
import { OpticalBench } from "./renderer/bench";
import { FarFieldDome } from "./renderer/dome";
import { MeasurementSection } from "./renderer/section";
import { Schematic } from "./renderer/schematic";
import { FlyActors } from "./renderer/flyActors";
import type { ChannelCommand } from "./optics/types";

type Mode = "analytic" | "manual" | "spgd" | "connectome";

interface ChannelOverride {
  piston: number;
  amplitude: number;
  tiltX: number;
  tiltY: number;
  curvature: number;
}

const array = createArrayConfig();

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
}

function targetMotionFor(kind: string): TargetMotion {
  switch (kind) {
    case "lissajous":
      return lissajousTarget({ z_m: 1, amplitudeX_m: 0.012, amplitudeY_m: 0.008, freqX_hz: 0.08, freqY_hz: 0.13 });
    case "circle":
      return circleTarget({ z_m: 1, radius_m: 0.012, freq_hz: 0.1 });
    case "fly":
      return flyTarget({ z_m: 1, extentX_m: 0.012, extentY_m: 0.008, speed_mps: 0.03, seed: 4 });
    default:
      return staticTarget(0.008, -0.005, 1);
  }
}

class App {
  private readonly scene = createScene($("view") as HTMLCanvasElement, (dt) => this.frame(dt));
  private bench = new OpticalBench(array);
  private dome = new FarFieldDome();
  private section = new MeasurementSection({ size_m: 3e-3, samples: 72, range_m: 1 });
  private schematic = new Schematic($("schematic") as HTMLCanvasElement, array);
  private flies = new FlyActors();
  private env: Environment;
  private mode: Mode = "analytic";
  private selected = -1;
  private overrides: ChannelOverride[] = array.channelIds.map(() => ({
    piston: 0,
    amplitude: 1,
    tiltX: 0,
    tiltY: 0,
    curvature: 0,
  }));

  private graph = generateSyntheticGraph({ n: 240, avgDegree: 10, seed: 12345 });
  private reservoir = new RateReservoir(this.graph, 8, { outputCount: 48, seed: 2, noise: 0.02 });
  private readout = new LinearReadout(array.channelIds.length, this.reservoir.outputCount, 3);
  private connectome = new ConnectomeController(array, this.reservoir, this.readout);

  private spgdPistons = new Array<number>(array.channelIds.length).fill(0);
  private observation: Observation;
  private lastReward = 0;
  private targetKind = "static";
  private note = "";
  private spgdSeed = 1;
  private paused = false;
  private showBeams = true;
  private sectionRange = 1;

  constructor() {
    this.scene.scene.add(this.bench.group);
    this.scene.scene.add(this.dome.group);
    this.scene.scene.add(this.section.group);
    this.scene.scene.add(this.flies.group);

    this.env = new Environment(defaultEnvironmentConfig(array, { target: targetMotionFor(this.targetKind), scanSamples: 15 }));
    this.observation = this.env.observeWithCommands(analyticSteeringCommands(array, this.env.targetDirection()));

    this.schematic.onSelect = (index) => this.select(index);
    this.buildControls();
    this.scene.start();
    this.buildChannelControls();
    this.buildInspectionControls();
    this.setMode("analytic");
    document.querySelector(`[data-target="static"]`)?.classList.add("active");

    // Expose a small hook for browser tests.
    (window as unknown as { __cbc?: unknown }).__cbc = {
      getMode: () => this.mode,
      getMetrics: () => ({
        mode: this.mode,
        reward: this.lastReward,
        pib: this.observation.pib,
        phaseRms: this.currentPhaseRms(),
        selected: this.selected,
        connectome: this.reservoir.activityStats(),
        paused: this.paused,
        step: this.env.stepIndex,
        sectionRange: this.sectionRange,
        graphSource: this.graph.source,
        flyLoaded: this.flies.loaded,
        flyError: this.flies.error,
      }),
      setMode: (m: Mode) => this.setMode(m),
      target: (kind: string) => this.setTarget(kind),
      train: () => this.trainConnectome(),
    };
  }

  private buildControls(): void {
    const modeHost = $("modes");
    const modes: Mode[] = ["analytic", "manual", "spgd", "connectome"];
    modes.forEach((m) => {
      const btn = document.createElement("button");
      btn.textContent = m;
      btn.dataset.mode = m;
      btn.addEventListener("click", () => this.setMode(m));
      modeHost.appendChild(btn);
    });

    const targetHost = $("targets");
    const kinds = ["static", "lissajous", "circle", "fly"];
    kinds.forEach((k) => {
      const btn = document.createElement("button");
      btn.textContent = k;
      btn.dataset.target = k;
      btn.addEventListener("click", () => this.setTarget(k));
      targetHost.appendChild(btn);
    });

    $("reset-btn").addEventListener("click", () => {
      this.env.reset(1);
      this.connectome.reset();
      this.spgdPistons.fill(0);
    });
    $("train-btn").addEventListener("click", () => this.trainConnectome());
  }

  private buildInspectionControls(): void {
    const pause = $("pause-btn");
    pause.addEventListener("click", () => { this.paused = !this.paused; pause.textContent = this.paused ? "Resume simulation" : "Pause simulation"; });
    const presets: Record<string, [number[], number[]]> = {
      overview: [[90, 60, 145], [0, 0, 40]],
      array: [[20, 16, 32], [0, 0, -5]],
      target: [[30, 20, 135], [0, 0, 95]],
      wiring: [[35, 24, -38], [0, 0, -10]],
    };
    Object.entries(presets).forEach(([name, [position, target]]) => {
      const button = document.createElement("button"); button.textContent = name;
      button.addEventListener("click", () => { this.scene.camera.position.set(position[0], position[1], position[2]); this.scene.controls.target.set(target[0], target[1], target[2]); this.scene.controls.update(); });
      $("cameras").append(button);
    });
    const channelSelect = $("channel-select") as HTMLSelectElement;
    channelSelect.add(new Option("Choose channel", "-1"));
    array.channelIds.forEach((id, i) => channelSelect.add(new Option(id, String(i))));
    channelSelect.addEventListener("change", () => { this.selected = Number(channelSelect.value); this.bench.setPathHighlight(this.selected); this.syncChannelControls(); });
    const toggle = (id: string, change: (checked: boolean) => void) => {
      const input = $(id) as HTMLInputElement; input.addEventListener("change", () => change(input.checked));
    };
    toggle("show-dome", value => this.dome.group.visible = value);
    toggle("show-section", value => this.section.group.visible = value);
    toggle("show-flies", value => this.flies.group.visible = value);
    toggle("show-beams", value => this.showBeams = value);
    const range = $("section-range") as HTMLInputElement;
    range.addEventListener("input", () => { this.sectionRange = Number(range.value); $("section-range-value").textContent = `${this.sectionRange.toFixed(2)} m`; });
    void fetch("/training/latest.json").then(r => { if (!r.ok) throw new Error("No published training result"); return r.json(); }).then(run => {
      $("training-summary").textContent = `${run.nodes.toLocaleString()} MaleCNS nodes · ${run.generations} generations · reward ${run.initial.toFixed(3)} → ${run.final.toFixed(3)}. Held-out mean normalized target intensity: ${run.transfer.toFixed(3)}. Provisional positive signs; analytic steering assistance.`;
      ($("training-curve") as HTMLImageElement).src = "/training/learning-curve.svg";
      $("training-curve").hidden = false;
    }).catch(() => { $("training-summary").textContent = "Headless training results will appear here after publication. This scene uses its own synthetic demo controller."; });
  }

  private buildChannelControls(): void {
    const host = $("channel");
    host.innerHTML = "";
    const specs: { key: keyof ChannelOverride; label: string; min: number; max: number; step: number }[] = [
      { key: "piston", label: "piston", min: -Math.PI, max: Math.PI, step: 0.01 },
      { key: "amplitude", label: "amplitude", min: 0, max: 2, step: 0.01 },
      { key: "tiltX", label: "tilt x", min: -array.steeringLimit, max: array.steeringLimit, step: 1e-4 },
      { key: "tiltY", label: "tilt y", min: -array.steeringLimit, max: array.steeringLimit, step: 1e-4 },
      { key: "curvature", label: "curvature", min: -array.curvatureLimit_per_m, max: array.curvatureLimit_per_m, step: 1 },
    ];
    for (const spec of specs) {
      const row = document.createElement("div");
      row.className = "row";
      const label = document.createElement("span");
      label.textContent = spec.label;
      const input = document.createElement("input");
      input.type = "range";
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
      input.dataset.key = spec.key;
      input.setAttribute("aria-label", spec.label);
      input.value = String(spec.key === "amplitude" ? 1 : 0);
      const readout = document.createElement("span");
      readout.dataset.readout = spec.key;
      readout.textContent = input.value;
      input.addEventListener("input", () => {
        if (this.selected < 0) return;
        this.overrides[this.selected][spec.key] = Number(input.value);
        this.setMode("manual");
        readout.textContent = Number(input.value).toPrecision(3);
      });
      row.append(label, input, readout);
      host.appendChild(row);
    }
  }

  private syncChannelControls(): void {
    const host = $("channel");
    const o = this.selected >= 0 ? this.overrides[this.selected] : { piston: 0, amplitude: 1, tiltX: 0, tiltY: 0, curvature: 0 };
    host.querySelectorAll<HTMLInputElement>("input[type=range]").forEach((input) => {
      const key = input.dataset.key as keyof ChannelOverride;
      input.disabled = this.selected < 0;
      input.value = String(o[key]);
      const readout = host.querySelector<HTMLElement>(`[data-readout="${key}"]`);
      if (readout) readout.textContent = Number(o[key]).toPrecision(3);
    });
  }

  private select(index: number): void {
    this.selected = this.selected === index ? -1 : index;
    this.bench.setPathHighlight(this.selected);
    ($( "channel-select") as HTMLSelectElement).value = String(this.selected);
    this.syncChannelControls();
  }

  private setMode(mode: Mode): void {
    this.mode = mode;
    document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === mode);
    });
  }

  private setTarget(kind: string): void {
    this.targetKind = kind;
    this.env = new Environment(defaultEnvironmentConfig(array, { target: targetMotionFor(kind), scanSamples: 15 }));
    this.observation = this.env.observeWithCommands(analyticSteeringCommands(array, this.env.targetDirection()));
    this.connectome.reset();
    document.querySelectorAll<HTMLButtonElement>("#targets button").forEach((b) => {
      b.classList.toggle("active", b.dataset.target === kind);
    });
  }

  private trainConnectome(): void {
    this.note = "training...";
    const env = this.env;
    env.config.episodeSteps = 50;
    try {
      const result = trainReadout(env, this.connectome, {
        generations: 15,
        population: 8,
        sigma: 0.12,
        learningRate: 0.06,
        esSeed: 3,
        evalSeeds: [1],
      });
      this.note = `connectome trained: ${result.initialFitness.toFixed(3)} -> ${result.finalFitness.toFixed(3)}`;
    } catch (err) {
      this.note = `training failed: ${String(err)}`;
    }
  }

  private currentCommands(): ChannelCommand[] {
    const targetDir = this.env.targetDirection();
    switch (this.mode) {
      case "manual": {
        const base = analyticSteeringCommands(array, targetDir);
        return base.map((cmd, i) => {
          const o = this.overrides[i];
          return { ...cmd, piston_rad: cmd.piston_rad + o.piston, amplitude: o.amplitude,
            tiltX: cmd.tiltX + o.tiltX, tiltY: cmd.tiltY + o.tiltY, curvature_per_m: o.curvature };
        });
      }
      case "spgd": {
        const objective = makeDirectionObjective(array, this.env.hiddenErrors(), targetDir);
        const result = runSpgd(objective, this.spgdPistons, {
          iterations: 2,
          gain: 1.0,
          perturbation: 0.3,
          seed: this.spgdSeed++,
        });
        this.spgdPistons = result.pistons;
        const base = analyticSteeringCommands(array, targetDir);
        return base.map((cmd, i) => ({ ...cmd, piston_rad: cmd.piston_rad + wrapPiston(this.spgdPistons[i]) }));
      }
      case "connectome":
        return this.connectome.commands(this.observation, targetDir);
      default:
        return analyticSteeringCommands(array, targetDir);
    }
  }

  private currentPhaseRms(): number {
    const dir = this.env.targetDirection();
    const ideal = idealSteeringPistons(array, dir);
    return phaseRmsRad(this.env.currentActual(), ideal);
  }

  private frame(dt: number): void {
    if (this.paused) return;
    const commands = this.currentCommands();
    const result = this.env.step(commands);
    this.observation = result.observation;
    this.lastReward = result.reward;
    this.flies.update(this.env.targetDirection(), Math.min(dt, 0.05));

    const scan = this.env.scanBeam();
    const centroid = new THREE.Vector3(scan.beamSx, scan.beamSy, Math.sqrt(Math.max(0, 1 - scan.beamSx ** 2 - scan.beamSy ** 2)));
    const peak = new THREE.Vector3(scan.peakSx, scan.peakSy, Math.sqrt(Math.max(0, 1 - scan.peakSx ** 2 - scan.peakSy ** 2)));
    const targetPos = this.env.targetPosition();

    this.bench.update({
      actual: result.actual,
      selectedIndex: this.selected,
      beamTarget: centroid.clone().multiplyScalar(DISPLAY.targetDistance),
      showBeams: this.showBeams,
    });
    this.dome.update(array, result.actual, { sx: centroid.x, sy: centroid.y, sz: centroid.z }, { sx: peak.x, sy: peak.y, sz: peak.z });
    this.section.update(array, result.actual, { sx: centroid.x, sy: centroid.y, sz: centroid.z }, this.sectionRange, targetPos);
    this.schematic.update(result.actual, this.selected, this.mode);
    this.drawNeural();
    this.updateHud(result.metrics, result.reward);
  }

  private drawNeural(): void {
    const canvas = $("neural") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.width !== canvas.clientWidth) canvas.width = canvas.clientWidth;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const x = this.reservoir.x;
    const barW = canvas.width / x.length;
    for (let i = 0; i < x.length; i++) {
      const v = Math.max(-1, Math.min(1, x[i]));
      const h = Math.abs(v) * canvas.height * 0.5;
      ctx.fillStyle = v >= 0 ? "#3fb950" : "#f0883e";
      ctx.fillRect(i * barW, canvas.height / 2 - (v >= 0 ? h : 0), Math.max(1, barW - 0.5), h);
    }
  }

  private updateHud(metrics: { pib: number; pointingAngle_rad: number; targetIntensity: number; targetSx: number; targetSy: number; beamSx: number; beamSy: number }, reward: number): void {
    const set = (id: string, value: string) => ($(id).textContent = value);
    set("m-step", `${this.env.stepIndex}`);
    set("m-mode", this.mode);
    set("m-reward", reward.toFixed(4));
    set("m-pib", metrics.pib.toFixed(4));
    set("m-peak", metrics.targetIntensity.toFixed(4));
    set("m-point", (metrics.pointingAngle_rad * 1e6).toFixed(1));
    set("m-phirms", this.currentPhaseRms().toFixed(4));
    const stats = this.reservoir.activityStats();
    set("m-neural", `${stats.mean.toFixed(3)} / max ${stats.max.toFixed(2)}`);
    set("m-tx", `${(metrics.targetSx * 1e3).toFixed(2)} / ${(metrics.beamSx * 1e3).toFixed(2)} mrad`);
    set("m-ty", `${(metrics.targetSy * 1e3).toFixed(2)} / ${(metrics.beamSy * 1e3).toFixed(2)} mrad`);
    set("m-note", this.note);
  }
}

new App();
