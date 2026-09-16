import * as THREE from "three";
import { createArrayConfig } from "./optics/geometry";
import { wrapPiston } from "./optics/channels";
import { idealSteeringPistons, phaseRmsRad } from "./optics/metrics";
import { analyticSteeringCommands } from "./controllers/analytic";
import { runSpgd } from "./controllers/spgd";
import { makeDirectionObjective } from "./controllers/objective";
import { Environment, defaultEnvironmentConfig, observationToVector, type Observation } from "./sim/environment";
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
import { NeuralActivityView, type NeuronLocation } from "./renderer/neuralActivity";
import { graphFromBrowserRun, type BrowserRun } from "./sim/browserRun";
import { buildHardwareRegistry } from "./renderer/hardware/registry";
import { HardwareScene } from "./renderer/hardware/hardwareScene";

type Mode = "analytic" | "manual" | "spgd" | "connectome";

interface ChannelOverride {
  piston: number;
  amplitude: number;
  tiltX: number;
  tiltY: number;
  curvature: number;
}

interface CameraKey {
  pos: [number, number, number];
  target: [number, number, number];
  fov?: number;
  dur: number;
}

interface CameraTrajectory {
  name: string;
  keys: CameraKey[];
  /** Cubic Catmull-Rom through the keys (C1) instead of eased segments. */
  cubic?: boolean;
  /** Total duration in seconds for a cubic trajectory. */
  duration?: number;
}

/**
 * Canned replayable camera trajectories. Keyframes use the same pose the HUD
 * reports (world position, orbit target, optional fov) so a copied pose can be
 * pasted straight in.
 */
const CAMERA_TRAJECTORIES: CameraTrajectory[] = [
  {
    name: "overview pan",
    keys: [
      { pos: [140, 70, 150], target: [0, 0, 40], dur: 3 },
      { pos: [0, 82, 165], target: [0, 0, 30], dur: 3 },
      { pos: [-140, 70, 150], target: [0, 0, 40], dur: 3 },
    ],
  },
  {
    name: "console orbit",
    keys: [
      { pos: [-30, 9, 18], target: [-56, -10, -14], dur: 2.5 },
      { pos: [-56, 17, 24], target: [-56, -10, -14], dur: 2.5 },
      { pos: [-82, 9, 18], target: [-56, -10, -14], dur: 2.5 },
      { pos: [-56, 2, 6], target: [-56, -10, -14], dur: 2.5 },
    ],
  },
  {
    name: "array dive",
    keys: [
      { pos: [40, 34, 80], target: [0, 0, 20], dur: 2.5 },
      { pos: [16, 12, 34], target: [0, 0, 6], dur: 2.5 },
      { pos: [4, 4, 8], target: [0, 0, -2], dur: 2.5 },
    ],
  },
  {
    name: "target sweep",
    keys: [
      { pos: [-40, 20, 120], target: [-10, 0, 95], dur: 3 },
      { pos: [0, 24, 130], target: [10, 0, 95], dur: 3 },
      { pos: [40, 20, 120], target: [0, 0, 95], dur: 3 },
    ],
  },
  {
    // Authored 5-key fly-in. Cubic Catmull-Rom through all five poses with a
    // global smootherstep so velocity is zero at the first and last frame.
    name: "authored 5-key fly-in",
    cubic: true,
    duration: 12,
    keys: [
      { pos: [-56.097, 17.169, -7.702], target: [-56.041, 0.896, -13.477], fov: 50, dur: 1 },
      { pos: [-60.603, 19.029, 37.084], target: [-58.909, 4.206, -15.11], fov: 50, dur: 1 },
      { pos: [-92.564, 47.045, 30.749], target: [-49.867, 10.774, -14.969], fov: 50, dur: 1 },
      { pos: [-117.041, 42.691, -27.19], target: [-52.986, 9.524, -32.265], fov: 50, dur: 1 },
      { pos: [-77.992, 21.694, -133.973], target: [-32.685, 6.824, -47.588], fov: 50, dur: 1 },
    ],
  },
  {
    // Second authored tour: same first two keys and last key as the 5-key
    // fly-in, with three added orbit keys in between.
    name: "authored 6-key orbit",
    cubic: true,
    duration: 16,
    keys: [
      { pos: [-56.097, 17.169, -7.702], target: [-56.041, 0.896, -13.477], fov: 50, dur: 1 },
      { pos: [-60.603, 19.029, 37.084], target: [-58.909, 4.206, -15.11], fov: 50, dur: 1 },
      { pos: [3.884, 5.947, 44.054], target: [-32.685, 6.824, -47.588], fov: 50, dur: 1 },
      { pos: [55.033, 3.563, -2.519], target: [-32.685, 6.824, -47.588], fov: 50, dur: 1 },
      { pos: [61.291, 24.214, -109.107], target: [-9.458, -4.377, -28.772], fov: 50, dur: 1 },
      { pos: [-77.992, 21.694, -133.973], target: [-32.685, 6.824, -47.588], fov: 50, dur: 1 },
    ],
  },
];

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
  private section = new MeasurementSection({ size_m: 30e-3, samples: 160, range_m: 1 });
  private schematic = new Schematic($("schematic") as HTMLCanvasElement, array);
  private flies = new FlyActors();
  private hardware = new HardwareScene(buildHardwareRegistry());
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
  private targetKind = "fly";
  private note = "";
  private spgdSeed = 1;
  private paused = false;
  private showBeams = true;
  private sectionRange = 1;
  private savedRun: BrowserRun | null = null;
  private neuralView: NeuralActivityView | null = null;
  private replayPolicy: "before" | "after" | null = null;
  private replayRewardSum = 0;
  private replayIntensitySum = 0;
  private replayFinished = false;
  private replayAccumulator = 0;
  private singleStep = false;
  private camPlay: {
    name: string;
    keys: CameraKey[];
    t: number;
    cubic: boolean;
    duration: number;
    posCurve?: THREE.CatmullRomCurve3;
    targetCurve?: THREE.CatmullRomCurve3;
  } | null = null;

  constructor() {
    this.scene.scene.add(this.bench.group);
    this.scene.scene.add(this.dome.group);
    this.scene.scene.add(this.section.group);
    this.scene.scene.add(this.flies.group);
    this.scene.scene.add(this.hardware.group);
    // Dome is an inspection layer and starts hidden; the checkbox re-enables it.
    this.dome.group.visible = false;
    this.flies.setOperatorMode(true);
    this.flies.setTrailsVisible(true);

    this.env = new Environment(defaultEnvironmentConfig(array, { target: targetMotionFor(this.targetKind), scanSamples: 15 }));
    this.observation = this.env.observeWithCommands(analyticSteeringCommands(array, this.env.targetDirection()));

    this.schematic.onSelect = (index) => this.select(index);
    this.buildControls();
    this.scene.start();
    this.buildChannelControls();
    this.buildInspectionControls();
    this.buildCameraControls();
    this.setMode("analytic");
    document.querySelector(`[data-target="fly"]`)?.classList.add("active");
    // The 3D neuron cloud is always on: load the saved MaleCNS run at startup
    // and drive the scene with its trained controller.
    void this.autoLoadRealRun();
    // Hardware bench is loaded and shown at startup; the checkbox can hide it.
    void this.enableHardware();

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
        replayPolicy: this.replayPolicy,
        replayFinished: this.replayFinished,
        replayMeanIntensity: this.env.stepIndex ? this.replayIntensitySum / this.env.stepIndex : 0,
        neural3dNodes: this.neuralView?.renderedNodes ?? 0,
        neural3dEdges: this.neuralView?.renderedEdges ?? 0,
        flyLoaded: this.flies.loaded,
        flyError: this.flies.error,
        flyForelegJoints: this.flies.forelegJointCount,
        flyGestureActive: this.flies.gestureActive,
        flyForelegMotion: this.flies.forelegMotion,
        activeKnobs: this.hardware.allKnobTargets().map((k) => k.channel),
        camera: this.currentCameraPose(),
        forelegDiagnostics: this.flies.forelegDiagnostics,
        targetDisplay: this.flies.targetDisplayPosition.toArray(),
        operatorDisplay: this.flies.operatorDisplayPosition.toArray(),
        hardwareLoaded: this.hardware.loaded,
        hardwareVisible: this.hardware.group.visible,
        hardwareNodes: this.hardware.nodeCount,
        hardwareCables: this.hardware.cableCount,
        hardwareAssetInstances: this.hardware.assetInstances,
        hardwareConnectorInstances: this.hardware.connectorInstances,
        hardwareComponentKinds: this.hardware.componentKindCount,
        hardwareKnobs: this.hardware.knobCount,
        hardwareSelectors: this.hardware.selectorCount,
        hardwareMinRadiusMm: this.hardware.routingAudit.minRadius_mm,
        hardwareClearanceViolations: this.hardware.routingAudit.violations,
        hardwareFailures: this.hardware.failures.length,
        hardwareFailureSample: this.hardware.failures[0] ?? null,
        hardwareError: this.hardware.error,
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
    $("load-real-run").addEventListener("click", () => { void this.loadSavedRun(); });
    $("replay-before").addEventListener("click", () => this.startReplay("before"));
    $("replay-after").addEventListener("click", () => this.startReplay("after"));
    $("replay-step").addEventListener("click", () => {this.paused=true;this.singleStep=true;$("pause-btn").textContent="Resume simulation";});
    $("neural-camera").addEventListener("click", () => { void this.inspectNeurons(); });
    $("view").addEventListener("pointerup", (event) => {
      if(!this.neuralView) return;
      const ray=new THREE.Raycaster(); const box=$("view").getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2((event.clientX-box.left)/box.width*2-1,-(event.clientY-box.top)/box.height*2+1),this.scene.camera);
      const n=this.neuralView.pick(ray); if(n) $("neuron-detail").textContent=`Body ${n.bodyId} · ${n.type ?? "untyped"} · ${n.superclass ?? "unclassified"}`;
    });
  }

  private async enableHardware(): Promise<void> {
    if (!this.hardware.loaded && !this.hardware.error) {
      this.note = "loading hardware bench…";
      await this.hardware.load();
    }
    this.hardware.setVisible(!this.hardware.error);
    this.hardware.setSelectedChannel(this.selected >= 0 ? array.channelIds[this.selected] : null);
    this.note = this.hardware.error
      ? `hardware bench failed: ${this.hardware.error}`
      : `hardware bench: ${this.hardware.assetInstances} asset modules, ${this.hardware.connectorInstances} mated connectors, ${this.hardware.cableCount} routed cables (display scale, not solver coords)`;
  }

  private async loadSavedRun(): Promise<void> {
    const button=$("load-real-run") as HTMLButtonElement;button.disabled=true;
    $("replay-status").textContent="Loading saved graph, policies, and soma annotations…";
    try {
      const get=async(url:string)=>{const r=await fetch(url);if(!r.ok)throw new Error(`${url}: ${r.status}`);return r.json();};
      const [run,locations]=await Promise.all([get("/training/scene-run.json"),get("/training/neuron-locations.json")]);
      this.savedRun=run as BrowserRun;this.graph=graphFromBrowserRun(this.savedRun);
      if(this.neuralView){this.scene.scene.remove(this.neuralView.group);this.neuralView=null;}
      this.neuralView=new NeuralActivityView(this.graph,locations.neurons as NeuronLocation[]);this.scene.scene.add(this.neuralView.group);
      ($("replay-before") as HTMLButtonElement).disabled=false;($("replay-after") as HTMLButtonElement).disabled=false;
      ($("train-btn") as HTMLButtonElement).disabled=true;
      $("graph-provenance").textContent=`Active graph: MaleCNS v1.0 · ${this.graph.n.toLocaleString()} neurons / ${this.graph.edges.pre.length.toLocaleString()} connections. Provisional positive signs; generic sensory/readout mapping. 3D shows ${this.neuralView.renderedNodes.toLocaleString()} mapped somata and ${this.neuralView.renderedEdges.toLocaleString()} sampled edges, not neuron skeletons.`;
      this.startReplay("before");
    } catch(error) {$("replay-status").textContent=`Run load failed: ${String(error)}`;button.disabled=false;}
  }

  /**
   * Load the saved MaleCNS run at startup so the 3D neuron cloud is always on.
   * Uses the trained ("after") policy live and does not start the auto-pausing
   * before/after replay; those buttons remain available for comparison.
   */
  private async autoLoadRealRun(): Promise<void> {
    try {
      const get = async (url: string) => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${url}: ${r.status}`);
        return r.json();
      };
      const [run, locations] = await Promise.all([
        get("/training/scene-run.json"),
        get("/training/neuron-locations.json"),
      ]);
      if (this.savedRun) return; // a manual load won the race
      this.savedRun = run as BrowserRun;
      this.graph = graphFromBrowserRun(this.savedRun);
      this.reservoir = new RateReservoir(this.graph, 8, run.reservoir);
      this.readout = new LinearReadout(array.channelIds.length, this.reservoir.outputCount, run.reservoir.seed);
      this.readout.setParams(Float64Array.from(run.readouts.after));
      this.connectome = new ConnectomeController(array, this.reservoir, this.readout);
      this.neuralView = new NeuralActivityView(this.graph, locations.neurons as NeuronLocation[]);
      this.scene.scene.add(this.neuralView.group);
      ($("replay-before") as HTMLButtonElement).disabled = false;
      ($("replay-after") as HTMLButtonElement).disabled = false;
      ($("train-btn") as HTMLButtonElement).disabled = true;
      $("graph-provenance").textContent = `Active graph: MaleCNS v1.0 · ${this.graph.n.toLocaleString()} neurons / ${this.graph.edges.pre.length.toLocaleString()} connections. 3D shows ${this.neuralView.renderedNodes.toLocaleString()} mapped somata.`;
      $("replay-status").textContent = "Live MaleCNS controller (after training). Use Before/After to replay the fixed evaluation episode.";
      this.setMode("connectome");
    } catch (error) {
      this.note = `saved MaleCNS run unavailable: ${String(error)}`;
    }
  }

  /**
   * Inspect 3D neurons works standalone: if the saved run has not been loaded
   * yet it loads it first, then focuses the camera on the mapped somata. This
   * avoids the previous behavior where the camera moved to empty space when no
   * run was loaded.
   */
  private async inspectNeurons(): Promise<void> {
    if (!this.neuralView) {
      await this.loadSavedRun();
    }
    if (!this.neuralView) return;
    this.neuralView.group.visible = true;
    const c = this.neuralView.center;
    this.scene.camera.position.set(c.x + 10, c.y + 14, c.z + 40);
    this.scene.controls.target.copy(c);
    this.scene.controls.update();
  }

  private startReplay(policy:"before"|"after"):void {
    if(!this.savedRun)return;
    const run=this.savedRun;this.replayPolicy=policy;
    // Recreate reservoir to reset the noise stream, not just neural activities.
    this.reservoir=new RateReservoir(this.graph,8,run.reservoir);
    this.readout=new LinearReadout(array.channelIds.length,this.reservoir.outputCount,run.reservoir.seed);
    this.readout.setParams(Float64Array.from(run.readouts[policy]));
    this.connectome=new ConnectomeController(array,this.reservoir,this.readout);
    this.env=new Environment(defaultEnvironmentConfig(array,{...run.spec.environment,target:staticTarget(0,0,1),seed:1}));
    this.observation=this.env.observeWithCommands(analyticSteeringCommands(array,this.env.targetDirection()));
    this.replayRewardSum=0;this.replayIntensitySum=0;this.replayFinished=false;this.paused=false;this.replayAccumulator=0;
    $("pause-btn").textContent="Pause simulation";this.setMode("connectome");
    $("replay-status").textContent=`${policy === "before" ? "Before training" : "After 40 generations"} · replaying the same 120-step phase-lock episode`;
    $("replay-before").classList.toggle("active",policy==="before");$("replay-after").classList.toggle("active",policy==="after");
    this.targetKind="static";
    document.querySelectorAll<HTMLButtonElement>("#targets button").forEach(b=>b.classList.toggle("active",b.dataset.target==="static"));
  }

  /** Camera pose readout, copy-to-clipboard and canned trajectory playback. */
  private buildCameraControls(): void {
    const select = $("cam-trajectory") as HTMLSelectElement;
    CAMERA_TRAJECTORIES.forEach((traj, i) => select.add(new Option(traj.name, String(i))));
    $("copy-camera").addEventListener("click", () => void this.copyCameraPose());
    $("play-camera").addEventListener("click", () => {
      const traj = CAMERA_TRAJECTORIES[Number(select.value)] ?? CAMERA_TRAJECTORIES[0];
      this.startCameraPlayback(traj);
    });
    $("stop-camera").addEventListener("click", () => this.stopCameraPlayback());
    this.buildCollapseToggles();
  }

  /** Collapse/expand the viewer panels toward their anchored edge. */
  private buildCollapseToggles(): void {
    document.querySelectorAll<HTMLButtonElement>(".collapse-toggle").forEach((btn) => {
      const panel = document.getElementById(btn.dataset.collapse ?? "");
      if (!panel) return;
      btn.addEventListener("click", () => {
        const collapsed = panel.classList.toggle("panel-collapsed");
        btn.textContent = collapsed ? "▸" : "▾";
      });
    });
  }

  private currentCameraPose(): {
    position: number[];
    target: number[];
    fov: number;
    yaw_deg: number;
    pitch_deg: number;
  } {
    const cam = this.scene.camera;
    const target = this.scene.controls.target;
    const dir = cam.getWorldDirection(new THREE.Vector3());
    const r3 = (v: number) => Number(v.toFixed(3));
    return {
      position: cam.position.toArray().map(r3),
      target: target.toArray().map(r3),
      fov: Number(cam.fov.toFixed(2)),
      yaw_deg: Number((Math.atan2(-dir.x, -dir.z) * (180 / Math.PI)).toFixed(2)),
      pitch_deg: Number((Math.asin(Math.max(-1, Math.min(1, dir.y))) * (180 / Math.PI)).toFixed(2)),
    };
  }

  private updateCameraPose(): void {
    const p = this.currentCameraPose();
    $("camera-pose").textContent =
      `pos    ${p.position[0]}, ${p.position[1]}, ${p.position[2]}\n` +
      `target ${p.target[0]}, ${p.target[1]}, ${p.target[2]}\n` +
      `fov ${p.fov}   yaw ${p.yaw_deg}°   pitch ${p.pitch_deg}°`;
  }

  private async copyCameraPose(): Promise<void> {
    const json = JSON.stringify(this.currentCameraPose(), null, 2);
    try {
      await navigator.clipboard.writeText(json);
      $("camera-note").textContent = "Camera pose copied to clipboard.";
    } catch {
      const area = document.createElement("textarea");
      area.value = json;
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
        $("camera-note").textContent = "Camera pose copied.";
      } catch {
        $("camera-note").textContent = json;
      }
      area.remove();
    }
  }

  private startCameraPlayback(traj: CameraTrajectory): void {
    const total = traj.keys.reduce((a, k) => a + k.dur, 0);
    const cubic = !!traj.cubic && traj.keys.length >= 2;
    const posCurve = cubic
      ? new THREE.CatmullRomCurve3(traj.keys.map((k) => new THREE.Vector3(...k.pos)), false, "centripetal")
      : undefined;
    const targetCurve = cubic
      ? new THREE.CatmullRomCurve3(traj.keys.map((k) => new THREE.Vector3(...k.target)), false, "centripetal")
      : undefined;
    this.camPlay = {
      name: traj.name,
      keys: traj.keys,
      t: 0,
      cubic,
      duration: traj.duration ?? total,
      posCurve,
      targetCurve,
    };
    this.scene.controls.enabled = false;
    $("camera-note").textContent = `Playing "${traj.name}"…`;
  }

  private stopCameraPlayback(): void {
    this.camPlay = null;
    this.scene.controls.enabled = true;
    $("camera-note").textContent = "Stopped. Read/copy the pose or play a preset.";
  }

  private advanceCamera(dt: number): void {
    const play = this.camPlay;
    if (!play) return;
    const cam = this.scene.camera;

    if (play.cubic && play.posCurve && play.targetCurve) {
      // Cubic path with a global smootherstep: zero velocity at both ends, C1
      // through every keyframe.
      play.t += dt;
      const u = Math.max(0, Math.min(1, play.t / Math.max(1e-3, play.duration)));
      if (u >= 1) {
        cam.position.copy(play.posCurve.getPointAt(1));
        this.scene.controls.target.copy(play.targetCurve.getPointAt(1));
        this.applyFov(play.keys, 1);
        cam.lookAt(this.scene.controls.target);
        this.stopCameraPlayback();
        return;
      }
      const e = u * u * u * (u * (u * 6 - 15) + 10); // smootherstep
      cam.position.copy(play.posCurve.getPointAt(e));
      this.scene.controls.target.copy(play.targetCurve.getPointAt(e));
      this.applyFov(play.keys, e);
      cam.lookAt(this.scene.controls.target);
      return;
    }

    play.t += dt;
    const total = play.keys.reduce((a, k) => a + k.dur, 0);
    if (play.t >= total) {
      this.stopCameraPlayback();
      return;
    }
    let acc = 0;
    let i = 0;
    for (; i < play.keys.length - 1; i++) {
      if (play.t < acc + play.keys[i].dur) break;
      acc += play.keys[i].dur;
    }
    const a = play.keys[i];
    const b = play.keys[Math.min(i + 1, play.keys.length - 1)];
    const local = Math.min(1, (play.t - acc) / Math.max(1e-3, a.dur));
    const e = local * local * (3 - 2 * local);
    const lerp = (x: number, y: number) => x + (y - x) * e;
    cam.position.set(lerp(a.pos[0], b.pos[0]), lerp(a.pos[1], b.pos[1]), lerp(a.pos[2], b.pos[2]));
    this.scene.controls.target.set(
      lerp(a.target[0], b.target[0]),
      lerp(a.target[1], b.target[1]),
      lerp(a.target[2], b.target[2]),
    );
    if (a.fov !== undefined || b.fov !== undefined) {
      const af = a.fov ?? cam.fov;
      const bf = b.fov ?? cam.fov;
      cam.fov = af + (bf - af) * e;
      cam.updateProjectionMatrix();
    }
    cam.lookAt(this.scene.controls.target);
  }

  /** Interpolate fov across a trajectory's keys at normalized parameter e. */
  private applyFov(keys: CameraKey[], e: number): void {
    const cam = this.scene.camera;
    if (keys.length === 0) return;
    const scaled = e * (keys.length - 1);
    const i = Math.min(keys.length - 2, Math.floor(scaled));
    const f = scaled - i;
    const a = keys[i]?.fov ?? keys[0].fov ?? cam.fov;
    const b = keys[i + 1]?.fov ?? a;
    cam.fov = a + (b - a) * f;
    cam.updateProjectionMatrix();
  }

  private buildInspectionControls(): void {
    const pause = $("pause-btn");
    pause.addEventListener("click", () => { this.paused = !this.paused; pause.textContent = this.paused ? "Resume simulation" : "Pause simulation"; });
    const presets: Record<string, [number[], number[]]> = {
      overview: [[90, 60, 145], [0, 0, 40]],
      array: [[20, 16, 32], [0, 0, -5]],
      target: [[30, 20, 135], [0, 0, 95]],
      wiring: [[35, 24, -38], [0, 0, -10]],
      hardware: [[0, 75, 95], [0, -4, -18]],
      console: [[-38, 13, 20], [-56, -11, -15]],
    };
    Object.entries(presets).forEach(([name, [position, target]]) => {
      const button = document.createElement("button"); button.textContent = name;
      button.addEventListener("click", () => {
        this.stopCameraPlayback();
        this.scene.camera.position.set(position[0], position[1], position[2]); this.scene.controls.target.set(target[0], target[1], target[2]); this.scene.controls.update();
        if (name === "array" || name === "wiring") {
          this.showBeams = false; this.dome.group.visible = false;
          ($("show-beams") as HTMLInputElement).checked = false;
          ($("show-dome") as HTMLInputElement).checked = false;
        }
      });
      $("cameras").append(button);
    });
    const channelSelect = $("channel-select") as HTMLSelectElement;
    channelSelect.add(new Option("Choose channel", "-1"));
    array.channelIds.forEach((id, i) => channelSelect.add(new Option(id, String(i))));
    channelSelect.addEventListener("change", () => { this.selected = Number(channelSelect.value); this.bench.setPathHighlight(this.selected); this.hardware.setSelectedChannel(this.selected >= 0 ? array.channelIds[this.selected] : null); if (this.selected >= 0) this.flies.startGesture(); this.syncChannelControls(); });
    const toggle = (id: string, change: (checked: boolean) => void) => {
      const input = $(id) as HTMLInputElement; input.addEventListener("change", () => change(input.checked));
    };
    toggle("show-dome", value => this.dome.group.visible = value);
    toggle("show-section", value => this.section.group.visible = value);
    toggle("show-flies", value => this.flies.group.visible = value);
    toggle("show-beams", value => this.showBeams = value);
    toggle("show-hardware", value => { if (value) { void this.enableHardware(); } else { this.hardware.setVisible(false); } });
    toggle("show-motionblur", value => this.flies.setTrailsVisible(value));
    this.syncChannelControls();
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
    this.hardware.setSelectedChannel(this.selected >= 0 ? array.channelIds[this.selected] : null);
    if (this.selected >= 0) this.flies.startGesture();
    this.syncChannelControls();
  }

  private setMode(mode: Mode): void {
    this.mode = mode;
    document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === mode);
    });
  }

  private setTarget(kind: string): void {
    this.replayPolicy=null;
    $("replay-status").textContent="Custom target exploration; use Before/After to return to the fixed evaluation episode.";
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
    // Camera playback and the pose readout run even while the simulation pauses.
    if (this.camPlay) this.advanceCamera(dt);
    this.updateCameraPose();
    if (this.paused && !this.singleStep) return;
    if(this.replayPolicy && !this.singleStep){
      const speed=Number(($("replay-speed") as HTMLSelectElement).value);
      this.replayAccumulator+=Math.min(dt,.1)*speed;
      if(this.replayAccumulator<this.env.config.dt_s)return;
      this.replayAccumulator-=this.env.config.dt_s;
    }
    this.singleStep=false;
    const commands = this.currentCommands();
    const result = this.env.step(commands);
    this.observation = result.observation;
    this.lastReward = result.reward;
    // Keep the always-on neuron cloud alive in every mode. In connectome mode
    // the controller already steps the reservoir with this observation.
    if (this.mode !== "connectome") {
      this.reservoir.step(observationToVector(this.observation));
    }
    this.neuralView?.update(this.reservoir.x);
    if(this.replayPolicy && this.mode === "connectome"){
      this.replayRewardSum+=result.reward;this.replayIntensitySum+=result.metrics.targetIntensity;
      const steps=this.env.config.episodeSteps;
      const mean=this.replayIntensitySum/this.env.stepIndex;
      $("replay-status").textContent=`${this.replayPolicy} · step ${this.env.stepIndex}/${steps} · mean target intensity ${mean.toFixed(3)}`;
      if(this.env.stepIndex>=steps){this.replayFinished=true;this.paused=true;$("pause-btn").textContent="Resume simulation";
        $(this.replayPolicy==="before"?"before-score":"after-score").textContent=`${mean.toFixed(3)} target intensity · mean reward ${(this.replayRewardSum/steps).toFixed(3)}`;
        $("replay-status").textContent+=" · complete — select the other policy to compare";
      }
    }
    this.hardware.updateMotion(result.actual);
    this.hardware.updateConsole(result.actual, this.selected);
    this.flies.setForelegTargets(this.hardware.allKnobTargets());
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
      beamRange_m: this.sectionRange,
    });
    this.dome.update(array, result.actual, { sx: centroid.x, sy: centroid.y, sz: centroid.z }, { sx: peak.x, sy: peak.y, sz: peak.z });
    // The measurement plane is fixed at the nominal boresight, not swept with
    // the beam centroid: the steering range fits within the 30 mm span.
    this.section.update(array, result.actual, { sx: 0, sy: 0, sz: 1 }, this.sectionRange, targetPos);
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
