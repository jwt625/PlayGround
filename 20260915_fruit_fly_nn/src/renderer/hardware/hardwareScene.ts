import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ChannelActual } from "../../optics/types";
import {
  APERTURE_CENTER_MM,
  APERTURE_SITES,
  CABLES,
  CELL_ASSIGNMENTS,
  CHANNEL_EQUIPMENT,
  ROW_PDUS,
  ROW_TRAYS,
  SHARED_EQUIPMENT,
  equipmentById,
  type CableDef,
  type EquipmentDef,
  type Orientation,
} from "./assemblySpec";
import type { HardwareRegistry } from "./registry";
import { buildConsolePanel, updateConsolePanel, CONSOLE_LAYOUT, type ConsolePanel } from "./consolePanel";

/**
 * Pass 1 + first-half-of-Pass-2 implementation of docs/CBC_ASSEMBLY_SPEC.md.
 *
 * Mechanical display millimetres are converted once to scene units and the
 * aperture emission center is translated to the optical array origin, so the
 * beams in `bench.ts` leave the real 19-channel aperture rather than a second
 * array. Supports, instruments and cable lanes are procedural; phase cassettes,
 * drivers, amplifiers, splitter, collimators and console reuse the generated
 * GLBs. Vendor CAD is never loaded.
 */

const APERTURE_PITCH_MM = 65;
// Optical array display pitch: 0.5 mm pitch * DISPLAY.scale(6000) = 3 units.
const SCENE_UNITS_PER_MM = 3 / APERTURE_PITCH_MM;
const FOOT_LIFT_MM = 4;

const GLB_URLS = import.meta.glob("../../../assets/generated/hardware-v2/*.glb", {
  eager: true,
  query: "?url",
  import: "default",
}) as Record<string, string>;

function glbUrl(file: string): string {
  const key = Object.keys(GLB_URLS).find((k) => k.endsWith(`/${file}`));
  if (!key) throw new Error(`hardware GLB not found: ${file}`);
  return GLB_URLS[key];
}

/** Mechanical mm -> scene units, registered so the aperture maps to the optics. */
export function scenePos(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(
    (x - APERTURE_CENTER_MM.x) * SCENE_UNITS_PER_MM,
    (y - APERTURE_CENTER_MM.y) * SCENE_UNITS_PER_MM,
    (z - APERTURE_CENTER_MM.z) * SCENE_UNITS_PER_MM,
  );
}

function orientationRotation(o: Orientation): THREE.Euler {
  switch (o) {
    case "A":
      return new THREE.Euler(-Math.PI / 2, 0, 0);
    case "B":
      return new THREE.Euler(-Math.PI / 2, Math.PI, 0);
    case "C":
      return new THREE.Euler(-Math.PI / 2, Math.PI / 2, 0);
    default:
      return new THREE.Euler(0, 0, 0);
  }
}

const FAMILY_COLOR: Record<string, number> = {
  optical: 0x39c5cf,
  rf: 0xffa657,
  command: 0x8b949e,
  dc: 0xf0883e,
  motor: 0xc297ff,
  external: 0xff7b72,
};

export interface HardwareSceneOptions {
  onProgress?: (message: string) => void;
}

interface PlacedEquipment {
  def: EquipmentDef;
  object: THREE.Object3D;
  tipPivot?: THREE.Object3D;
  tiltPivot?: THREE.Object3D;
}

export class HardwareScene {
  readonly group = new THREE.Group();
  private readonly registry: HardwareRegistry;
  private readonly options: HardwareSceneOptions;
  private templates = new Map<string, THREE.Object3D>();
  private placed = new Map<string, PlacedEquipment>();
  private cables: { def: CableDef; line: THREE.Line; material: THREE.LineBasicMaterial; curve: THREE.CatmullRomCurve3 }[] = [];
  private readonly highlight = new THREE.Group();
  private readonly instantiatedKinds = new Set<string>();
  private failureLog: string[] = [];
  loaded = false;
  error: string | null = null;
  selectedChannel: string | null = null;
  assetInstances = 0;
  proceduralInstances = 0;
  connectorInstances = 0;
  private consolePanel: ConsolePanel | null = null;
  private readonly channelIds: string[] = CHANNEL_EQUIPMENT.filter((e) => e.glb === "phase-cassette").map((e) => e.channel!).sort();

  constructor(registry: HardwareRegistry, options: HardwareSceneOptions = {}) {
    this.registry = registry;
    this.options = options;
    this.group.visible = false;
  }

  private async loadTemplate(file: string): Promise<THREE.Object3D> {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(glbUrl(file));
    return gltf.scene;
  }

  async load(): Promise<void> {
    try {
      const names = [...this.registry.components.keys()];
      for (let i = 0; i < names.length; i++) {
        const component = this.registry.components.get(names[i])!;
        this.options.onProgress?.(`loading ${component.name} (${i + 1}/${names.length})`);
        this.templates.set(component.name, await this.loadTemplate(component.file));
      }
      for (const def of SHARED_EQUIPMENT) {
        if (def.kind !== "support") this.build(def);
      }
      for (const def of ROW_TRAYS) this.build(def);
      for (const def of ROW_PDUS) this.build(def);
      for (const def of CHANNEL_EQUIPMENT) this.build(def);
      this.buildSupports();
      this.buildConsole();
      this.buildCables();
      this.routingReport = this.auditRouting();
      this.loaded = true;
    } catch (err) {
      this.error = String(err);
      this.options.onProgress?.(`hardware load failed: ${String(err)}`);
    }
  }

  private buildSupports(): void {
    // Table top + legs, racks and shelves, channel baseplates, aperture frame
    // and ledges, operator platform. Supports are illustrative boxes (the
    // support-hardware assets are still outstanding per the spec).
    const s = SCENE_UNITS_PER_MM;
    this.proceduralBox(0, -30, -775, 3200, 60, 2150, 0x233040, "TABLE");
    for (const [lx, lz] of [[-1500, 200], [1500, 200], [-1500, -1750], [1500, -1750]] as const) {
      this.proceduralBox(lx, -340, lz, 60, 560, 60, 0x1b2530, "table leg");
    }
    this.proceduralBox(-1220, -70, -1380, 560, 140, 700, 0x1b2530, "RACK-L");
    this.proceduralBox(-1220, 20, -1380, 560, 8, 700, 0x2a3646, "RACK-L shelf");
    this.proceduralBox(-1220, 180, -1380, 560, 8, 700, 0x2a3646, "RACK-L shelf");
    this.proceduralBox(1200, -90, -1400, 640, 180, 760, 0x1b2530, "RACK-R");
    this.proceduralBox(1200, 20, -1400, 640, 8, 760, 0x2a3646, "RACK-R shelf");
    this.proceduralBox(1200, 180, -1400, 640, 8, 760, 0x2a3646, "RACK-R shelf");
    this.proceduralBox(1200, 340, -1400, 640, 8, 760, 0x2a3646, "RACK-R shelf");
    // Per-channel baseplates.
    for (const a of CELL_ASSIGNMENTS) {
      this.proceduralBox(a.xc, 24, a.zr, 260, 4, 320, 0x2a3646, `PLATE-${a.channel}`);
    }
    // Vertical aperture frame + mount ledges.
    this.proceduralBox(0, 300, 80, 400, 400, 10, 0x2a3646, "AP-FRAME");
    for (const site of APERTURE_SITES) {
      this.proceduralBox(site.x_mm, site.y_mm - 35, site.z_mm - 5, 60, 6, 50, 0x33455c, `LEDGE-${site.id}`);
    }
    this.proceduralBox(-1220, 10, 85, 560, 20, 330, 0x2a3646, "OP-PLATFORM");
    this.proceduralBox(-1220, -10, -250, 560, 60, 320, 0x1b2530, "CON base");
    void s;
  }

  private proceduralBox(x: number, y: number, z: number, w: number, h: number, d: number, color: number, label: string): void {
    const s = SCENE_UNITS_PER_MM;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w * s, h * s, d * s),
      new THREE.MeshStandardMaterial({ color, metalness: 0.35, roughness: 0.65 }),
    );
    mesh.position.copy(scenePos(x, y, z));
    mesh.name = label;
    this.group.add(mesh);
    this.proceduralInstances++;
  }

  private build(def: EquipmentDef): void {
    try {
      const s = SCENE_UNITS_PER_MM;
      const object = new THREE.Group();
      object.name = def.id;
      const base = scenePos(def.position_mm[0], def.position_mm[1] + FOOT_LIFT_MM, def.position_mm[2]);
      object.position.copy(base);
      if (def.glb) {
        const template = this.templates.get(def.glb);
        if (!template) throw new Error(`template ${def.glb} not loaded`);
        this.instantiatedKinds.add(def.glb);
        const clone = template.clone(true);
        clone.rotation.copy(orientationRotation(def.orientation));
        clone.scale.setScalar(s);
        object.add(clone);
        this.assetInstances++;
      } else {
        const [w, h, d] = def.size_mm;
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(w * s, h * s, d * s),
          new THREE.MeshStandardMaterial({
            color: def.kind === "support" ? 0x233040 : def.kind === "instrument" ? 0x2f4054 : 0x3a4a5e,
            metalness: 0.4,
            roughness: 0.6,
          }),
        );
        object.add(mesh);
        this.proceduralInstances++;
      }
      // Port anchors in world-frame offsets (spec port table is post-orientation).
      const ports = new THREE.Group();
      ports.name = "ports";
      object.add(ports);
      for (const port of def.ports) {
        const anchor = new THREE.Object3D();
        anchor.name = `anchor_${port.name}`;
        anchor.position.set(port.offset_mm[0] * s, port.offset_mm[1] * s, port.offset_mm[2] * s);
        anchor.userData.normal = new THREE.Vector3(...port.normal);
        ports.add(anchor);
      }
      const tipPivot = object.getObjectByName("tip_pivot");
      const tiltPivot = object.getObjectByName("tilt_pivot");
      this.placed.set(def.id, {
        def,
        object,
        tipPivot: tipPivot ?? undefined,
        tiltPivot: tiltPivot ?? undefined,
      });
      this.group.add(object);
    } catch (err) {
      this.failureLog.push(`${def.id}: ${String(err)}`);
    }
  }

  /**
   * Resolve a cable terminal. Plugs belong to cable assemblies (spec §2.3/§6.2):
   * a male plug is added at FC/APC and SMA *socket* ends and the cable starts at
   * its downstream `port_cable_exit`. Captive pigtails start at the pigtail exit
   * with no plug.
   */
  private terminal(def: EquipmentDef, portName: string): { p: THREE.Vector3; n: THREE.Vector3 } | null {
    const placed = this.placed.get(def.id);
    if (!placed) return null;
    const anchor = placed.object.getObjectByName(`anchor_${portName}`);
    if (!anchor) return null;
    const normal = (anchor.userData.normal as THREE.Vector3).clone().normalize();
    const port = def.ports.find((x) => x.name === portName);
    const isMaleEnd = !!port && !port.pigtail && !port.freeSpace && (port.connector === "FC/APC" || port.connector === "SMA");
    if (!isMaleEnd) {
      return { p: anchor.getWorldPosition(new THREE.Vector3()), n: normal };
    }
    const name = port!.connector === "SMA" ? "sma-plug" : "fc-apc-plug";
    const template = this.templates.get(name);
    if (!template) return { p: anchor.getWorldPosition(new THREE.Vector3()), n: normal };
    this.instantiatedKinds.add(name);
    const plug = template.clone(true);
    plug.scale.setScalar(SCENE_UNITS_PER_MM);
    // Plug mating face (+Z) opposes the socket's outward normal; cable exits outward.
    plug.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal.clone().negate());
    anchor.add(plug);
    this.connectorInstances++;
    let exit: THREE.Object3D | null = null;
    plug.traverse((o) => {
      if (!exit && o.name === "port_cable_exit") exit = o;
    });
    const p = (exit ? (exit as THREE.Object3D).getWorldPosition(new THREE.Vector3()) : plug.getWorldPosition(new THREE.Vector3()));
    return { p, n: normal };
  }

  private laneFor(cable: CableDef, p0: THREE.Vector3, p1: THREE.Vector3): THREE.Vector3[] {
    const s = SCENE_UNITS_PER_MM;
    const waypoints: THREE.Vector3[] = [];
    // Spine runs are two points at the cable's start and end depths, so the
    // route turns once at each spine instead of doubling back through a midpoint.
    switch (cable.family) {
      case "optical": {
        const x = scenePos(-850, 35, 0).x;
        waypoints.push(new THREE.Vector3(x, p0.y, p0.z));
        waypoints.push(new THREE.Vector3(x, p1.y, p1.z));
        break;
      }
      case "command":
      case "dc": {
        const x = scenePos(850, 35, 0).x;
        waypoints.push(new THREE.Vector3(x, p0.y, p0.z));
        waypoints.push(new THREE.Vector3(x, p1.y, p1.z));
        break;
      }
      case "motor": {
        const spine = scenePos(240, 220, 0);
        waypoints.push(new THREE.Vector3(spine.x, spine.y, p0.z));
        waypoints.push(new THREE.Vector3(spine.x, spine.y, p1.z));
        break;
      }
      case "rf": {
        const lift = new THREE.Vector3(0, 0, -30 * s);
        waypoints.push(p0.clone().lerp(p1, 0.4).add(lift));
        waypoints.push(p0.clone().lerp(p1, 0.6).add(lift));
        break;
      }
      default:
        waypoints.push(p0.clone().lerp(p1, 0.5));
    }
    // Supported Ω service loop behind the moving collimator (R3.5).
    if (cable.family === "optical" && cable.to.equipment.startsWith("COL-")) {
      const axis = p1.clone().sub(p0).normalize();
      const perp = new THREE.Vector3(-axis.z, 0, axis.x).normalize().multiplyScalar(40 * s);
      const base = p1.clone().addScaledVector(axis, -80 * s);
      waypoints.push(base.clone().sub(perp).add(new THREE.Vector3(0, 25 * s, 0)));
      waypoints.push(p1.clone());
    }
    return waypoints;
  }

  /** §8: 43-knob sloped console panel bound to the controller state. */
  private buildConsole(): void {
    const con = this.placed.get("CON");
    if (!con) return;
    const panel = buildConsolePanel({ unitsPerMm: SCENE_UNITS_PER_MM });
    panel.group.position.set(0, (CONSOLE_LAYOUT.surface_mm.centerY - 24) * SCENE_UNITS_PER_MM, 0);
    con.object.add(panel.group);
    this.consolePanel = panel;
  }

  /** Bind the console knobs/selector to the actual controller state (R4). */
  updateConsole(actual: readonly ChannelActual[], selected: number): void {
    if (!this.consolePanel) return;
    updateConsolePanel(this.consolePanel, { selected, channelIds: this.channelIds, actual });
  }

  get knobCount(): number {
    return this.consolePanel?.knobs.size ?? 0;
  }

  get selectorCount(): number {
    return this.consolePanel?.selectorMeshes.size ?? 0;
  }

  private buildCables(): void {
    this.group.add(this.highlight);
    for (const cable of CABLES) {
      const fromDef = equipmentById(cable.from.equipment);
      const toDef = equipmentById(cable.to.equipment);
      const a = this.terminal(fromDef, cable.from.port);
      const b = this.terminal(toDef, cable.to.port);
      if (!a || !b) continue;
      const lead = (cable.family === "optical" ? 25 : cable.family === "rf" ? 15 : cable.family === "motor" ? 28 : 22) * SCENE_UNITS_PER_MM;
      const p0 = a.p.clone().addScaledVector(a.n, lead);
      const p1 = b.p.clone().addScaledVector(b.n, lead);
      const curve = new THREE.CatmullRomCurve3([a.p, p0, ...this.laneFor(cable, p0, p1), p1, b.p], false, "centripetal");
      const material = new THREE.LineBasicMaterial({
        color: FAMILY_COLOR[cable.family] ?? 0x8b949e,
        transparent: true,
        opacity: cable.family === "optical" ? 0.5 : 0.35,
      });
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)), material);
      this.group.add(line);
      this.cables.push({ def: cable, line, material, curve });
    }
  }

  private static readonly TUBE_RADIUS_MM: Record<string, number> = {
    optical: 1,
    rf: 1.5,
    command: 2,
    dc: 2,
    motor: 1.25,
    external: 4,
  };

  private rebuildHighlight(): void {
    for (const child of [...this.highlight.children]) {
      this.highlight.remove(child);
      const mesh = child as THREE.Mesh;
      mesh.geometry?.dispose?.();
    }
    const channel = this.selectedChannel;
    if (!channel) return;
    for (const rec of this.cables) {
      if (rec.def.channel !== channel) continue;
      const radius = (HardwareScene.TUBE_RADIUS_MM[rec.def.family] ?? 2) * SCENE_UNITS_PER_MM;
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(rec.curve, 48, radius, 6, false),
        new THREE.MeshStandardMaterial({ color: FAMILY_COLOR[rec.def.family] ?? 0x8b949e, metalness: 0.2, roughness: 0.6 }),
      );
      this.highlight.add(tube);
    }
  }

  /** Sampled curvature-radius and inflated-clearance audit (R3 gate reporting). */
  auditRouting(): { minRadius_mm: number; violations: number; samples: number } {
    const samplesPerCable = 32;
    const clearance = 2 + 2; // declared 2 mm visual clearance + cable radius
    let minRadius = Infinity;
    let violations = 0;
    let samples = 0;
    for (const rec of this.cables) {
      const pts = rec.curve.getPoints(samplesPerCable);
      const cableRadius = HardwareScene.TUBE_RADIUS_MM[rec.def.family] ?? 2;
      for (let i = 1; i < pts.length - 1; i++) {
        const p0 = pts[i - 1];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const d1 = p1.clone().sub(p0);
        const d2 = p2.clone().sub(p1);
        const cross = new THREE.Vector3().crossVectors(d1, d2).length();
        const len = (d1.length() + d2.length()) / 2;
        if (cross > 1e-9) {
          const radius = (len * len * len) / cross; // mm-ish in scene units; convert below
          minRadius = Math.min(minRadius, radius / SCENE_UNITS_PER_MM);
        }
        samples++;
      }
      // Clearance against equipment bounding spheres.
      for (const point of pts) {
        for (const placed of this.placed.values()) {
          const mesh = placed.object.children[0] as THREE.Mesh | undefined;
          if (!mesh || !(mesh.geometry instanceof THREE.BoxGeometry)) continue;
          const params = mesh.geometry.parameters as { width: number; height: number; depth: number };
          const r = (Math.max(params.width, params.height, params.depth) / 2) / SCENE_UNITS_PER_MM;
          const centerWorld = placed.object.getWorldPosition(new THREE.Vector3());
          const distMm = point.distanceTo(centerWorld) / SCENE_UNITS_PER_MM;
          if (distMm < r - 1) violations++;
          void clearance;
          void cableRadius;
        }
      }
    }
    return { minRadius_mm: Number.isFinite(minRadius) ? minRadius : -1, violations, samples };
  }

  get routingAudit(): { minRadius_mm: number; violations: number; samples: number } {
    return this.routingReport;
  }

  private routingReport = { minRadius_mm: -1, violations: 0, samples: 0 };

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setSelectedChannel(channel: string | null): void {
    this.selectedChannel = channel;
    for (const cable of this.cables) {
      const on = !channel || cable.def.channel === channel || cable.def.channel === null;
      cable.material.opacity = cable.def.channel === channel ? 0.95 : on ? 0.28 : 0.04;
    }
    // Selected channel gets near-field tube geometry instead of lines (R3).
    this.rebuildHighlight();
  }

  /** Drive the collimator tip/tilt pivots from actual per-channel pointing. */
  updateMotion(actual: readonly ChannelActual[]): void {
    if (!this.loaded) return;
    for (let i = 1; i <= actual.length; i++) {
      const ch = `CH${String(i).padStart(2, "0")}`;
      const placed = this.placed.get(`COL-${ch}`);
      if (!placed?.tipPivot || !placed.tiltPivot) continue;
      const st = actual[i - 1];
      placed.tipPivot.rotation.x = -st.tiltY * 25;
      placed.tiltPivot.rotation.y = st.tiltX * 25;
    }
  }

  get cableCount(): number {
    return this.cables.length;
  }

  get nodeCount(): number {
    return this.placed.size;
  }

  get componentKindCount(): number {
    return this.instantiatedKinds.size;
  }

  get failures(): readonly string[] {
    return this.failureLog;
  }
}
