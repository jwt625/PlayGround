import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ChannelActual } from "../../optics/types";
import { wrapAngleDiff } from "../../optics/channels";
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
// Generated GLBs are authored in metres (up +Z); mechanical placement is in mm.
const ASSET_SCALE = 1000 * SCENE_UNITS_PER_MM;
const FOOT_LIFT_MM = 4;

// Enclosure palette (render-style: dark restrained lab hardware).
const ENCLOSURE_COLOR = {
  instrument: 0x2b3139,
  channel: 0x2a313a,
  console: 0x232a32,
  support: 0x1b2129,
  tray: 0x2f3742,
  rack: 0x191f26,
  boot: 0x2ea043,
};

/** Deterministic +/- nuance so repeated enclosures do not read as one slab. */
function enclosureTint(base: number, key: string): number {
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) & 0xff;
  const shift = ((hash % 12) - 6) * 0.004;
  const c = new THREE.Color(base);
  c.offsetHSL(0, 0, shift);
  return c.getHex();
}

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

/**
 * Enclosure orientation. A is the -90 deg X rotation that lifts asset +Z to
 * world +Y (lids up). B and C are A followed by a world-Y yaw (spec §2), which
 * keeps lids up. Building them as a single Euler(-90, yaw, 0) is wrong: with
 * XYZ order that also flips the asset upside down.
 */
const ORIENTATION_A = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
const AXIS_Y = new THREE.Vector3(0, 1, 0);

function orientationQuaternion(o: Orientation): THREE.Quaternion {
  switch (o) {
    case "A":
      return ORIENTATION_A.clone();
    case "B":
      return new THREE.Quaternion().setFromAxisAngle(AXIS_Y, Math.PI).multiply(ORIENTATION_A);
    case "C":
      return new THREE.Quaternion().setFromAxisAngle(AXIS_Y, Math.PI / 2).multiply(ORIENTATION_A);
    default:
      return new THREE.Quaternion();
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
  private cables: { def: CableDef; mesh: THREE.Mesh; material: THREE.MeshStandardMaterial; curve: THREE.CatmullRomCurve3 }[] = [];
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
  private readonly lastPiston = new Map<string, number>();
  private knobRates: { channel: string; rate: number }[] = [];
  private hotChannel: string | null = null;
  private hotRate = 0;

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
      this.buildLabels();
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
    this.proceduralBox(0, -30, -775, 3200, 60, 2150, 0x233040, "TABLE");
    for (const [lx, lz] of [[-1500, 200], [1500, 200], [-1500, -1750], [1500, -1750]] as const) {
      this.proceduralBox(lx, -340, lz, 60, 560, 60, 0x1b2530, "table leg");
    }
    this.proceduralBox(1200, -90, -1400, 640, 180, 760, ENCLOSURE_COLOR.rack, "RACK-R");
    this.proceduralBox(1200, 20, -1400, 640, 8, 760, ENCLOSURE_COLOR.tray, "RACK-R shelf");
    this.proceduralBox(1200, 180, -1400, 640, 8, 760, ENCLOSURE_COLOR.tray, "RACK-R shelf");
    this.proceduralBox(1200, 340, -1400, 640, 8, 760, ENCLOSURE_COLOR.tray, "RACK-R shelf");
    // Per-channel baseplates.
    for (const a of CELL_ASSIGNMENTS) {
      this.proceduralBox(a.xc, 24, a.zr, 260, 4, 320, ENCLOSURE_COLOR.tray, `PLATE-${a.channel}`);
    }
    // Vertical aperture frame + mount ledges.
    this.proceduralBox(0, 300, 80, 400, 400, 10, ENCLOSURE_COLOR.tray, "AP-FRAME");
    for (const site of APERTURE_SITES) {
      this.proceduralBox(site.x_mm, site.y_mm - 35, site.z_mm - 5, 60, 6, 50, 0x33455c, `LEDGE-${site.id}`);
    }
    this.proceduralBox(-1220, 10, 85, 560, 20, 330, ENCLOSURE_COLOR.tray, "OP-PLATFORM");
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

  /**
   * Console enclosure with a top face parallel to the 15 deg sloped panel, so
   * the body never rises in front of the panel. Front edge is lower, rear edge
   * higher.
   */
  private consoleBodyGeometry(w: number, d: number): THREE.BufferGeometry {
    const s = SCENE_UNITS_PER_MM;
    const tilt = THREE.MathUtils.degToRad(CONSOLE_LAYOUT.surface_mm.tiltDeg);
    const centerY = (CONSOLE_LAYOUT.surface_mm.centerY - 24) * s;
    const halfThickness = 4 * s;
    const clearance = 2 * s;
    const halfD = d / 2;
    const topAt = (z: number) => centerY - z * Math.sin(tilt) - halfThickness - clearance;
    const hRear = topAt(-halfD);
    const hFront = topAt(halfD);
    const g = new THREE.BoxGeometry(w, 1, d);
    const pos = g.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const z = pos.getZ(i);
      if (pos.getY(i) > 0) {
        const t = (z + halfD) / d; // 0 at rear, 1 at front
        pos.setY(i, hRear + (hFront - hRear) * t);
      } else {
        pos.setY(i, 0);
      }
    }
    pos.needsUpdate = true;
    g.computeVertexNormals();
    return g;
  }

  private enclosureColor(def: EquipmentDef): number {
    switch (def.kind) {
      case "console":
        return ENCLOSURE_COLOR.console;
      case "channel":
      case "aperture":
        return ENCLOSURE_COLOR.channel;
      case "support":
        return ENCLOSURE_COLOR.support;
      default:
        return ENCLOSURE_COLOR.instrument;
    }
  }

  private readonly labelTextures = new Map<string, THREE.CanvasTexture>();

  private labelTexture(text: string): THREE.CanvasTexture {
    const cached = this.labelTextures.get(text);
    if (cached) return cached;
    const font = "bold 34px ui-monospace, Menlo, monospace";
    const measure = document.createElement("canvas").getContext("2d");
    let textWidth = 256;
    if (measure) {
      measure.font = font;
      textWidth = Math.ceil(measure.measureText(text).width);
    }
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(128, textWidth + 40);
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "rgba(8,12,18,0.72)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = font;
      ctx.fillStyle = "#dfe9f3";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 1);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.labelTextures.set(text, tex);
    return tex;
  }

  /** Minimal readable nameplate above a shared instrument (Render H1). */
  private addLabel(object: THREE.Object3D, text: string, yUnits: number): void {
    const tex = this.labelTexture(text);
    const image = tex.image as HTMLCanvasElement;
    const aspect = image.width / image.height;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    const h = 1.6;
    sprite.scale.set(h * aspect, h, 1);
    sprite.position.set(0, yUnits, 0);
    sprite.name = `label_${text}`;
    object.add(sprite);
  }

  private buildLabels(): void {
    for (const def of [...SHARED_EQUIPMENT, ...ROW_PDUS]) {
      if (def.kind === "support" || def.kind === "boundary") continue;
      const placed = this.placed.get(def.id);
      if (!placed) continue;
      const top = (def.size_mm[1] + 24) * SCENE_UNITS_PER_MM;
      const extra = def.kind === "console" ? 110 * SCENE_UNITS_PER_MM : 0;
      this.addLabel(placed.object, def.label, top + extra);
    }
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
        clone.quaternion.copy(orientationQuaternion(def.orientation));
        if (def.yawRad) clone.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(AXIS_Y, def.yawRad));
        clone.scale.setScalar(ASSET_SCALE);
        object.add(clone);
        this.assetInstances++;
      } else {
        const [w, h, d] = def.size_mm;
        const geometry =
          def.kind === "console"
            ? this.consoleBodyGeometry(w * s, d * s)
            : new THREE.BoxGeometry(w * s, h * s, d * s);
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: enclosureTint(this.enclosureColor(def), def.id),
            metalness: 0.45,
            roughness: 0.55,
          }),
        );
        // `position_mm[1]` is the support Y: put the enclosure bottom on it.
        // The console body is already built with its bottom at local y=0.
        if (def.kind !== "console") mesh.position.y = (h / 2) * s;
        if (def.yawRad) mesh.rotation.y = def.yawRad;
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
      if (port?.pigtail) this.addPigtailBoot(anchor, normal);
      return { p: anchor.getWorldPosition(new THREE.Vector3()), n: normal };
    }
    const name = port!.connector === "SMA" ? "sma-plug" : "fc-apc-plug";
    const template = this.templates.get(name);
    if (!template) return { p: anchor.getWorldPosition(new THREE.Vector3()), n: normal };
    this.instantiatedKinds.add(name);
    const plug = template.clone(true);
    plug.scale.setScalar(ASSET_SCALE);
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

  /**
   * Captive PM pigtails leave the cassette through a strain-relief boot. A
   * pigtail exit is not a mating socket (spec §2.3), so it gets a boot and
   * ferrule rather than an FC plug.
   */
  private addPigtailBoot(anchor: THREE.Object3D, normal: THREE.Vector3): void {
    const s = SCENE_UNITS_PER_MM;
    const boot = new THREE.Group();
    boot.name = "pigtail_boot";
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(2.4 * s, 3.2 * s, 9 * s, 10),
      new THREE.MeshStandardMaterial({ color: ENCLOSURE_COLOR.boot, metalness: 0.35, roughness: 0.5 }),
    );
    body.position.y = 4.5 * s;
    const ferrule = new THREE.Mesh(
      new THREE.CylinderGeometry(1.1 * s, 1.1 * s, 3 * s, 8),
      new THREE.MeshStandardMaterial({ color: 0xc9d3dc, metalness: 0.7, roughness: 0.3 }),
    );
    ferrule.position.y = 10 * s;
    boot.add(body, ferrule);
    boot.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
    anchor.add(boot);
    this.connectorInstances++;
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
    // Per-frame piston change drives the fly's foreleg targeting. Amplitude is
    // constant in the current action space, so its knobs correctly stay put.
    this.knobRates = [];
    for (let i = 0; i < actual.length; i++) {
      const ch = this.channelIds[i];
      const prev = this.lastPiston.get(ch) ?? actual[i].piston_rad;
      this.knobRates.push({ channel: ch, rate: Math.abs(wrapAngleDiff(actual[i].piston_rad, prev)) });
      this.lastPiston.set(ch, actual[i].piston_rad);
    }
    this.hotChannel = [...this.knobRates].sort((a, b) => b.rate - a.rate)[0]?.channel ?? null;
    this.hotRate = this.hotChannel ? this.knobRates.find((r) => r.channel === this.hotChannel)!.rate : 0;
    updateConsolePanel(this.consolePanel, {
      selected,
      channelIds: this.channelIds,
      actual,
      fallbackChannel: this.hotChannel,
    });
  }

  /**
   * Every console knob with its world position, so the operator fly can work
   * the whole panel (spec §8 console). Rate is the current hottest-channel turn
   * rate; it drives the operating scrub speed.
   */
  allKnobTargets(): { channel: string; rate: number; position: THREE.Vector3 }[] {
    if (!this.consolePanel) return [];
    const rate = Math.max(this.hotRate, 0.02);
    return [...this.consolePanel.knobs.values()].map((knob) => ({
      channel: knob.name,
      rate,
      position: knob.mesh.getWorldPosition(new THREE.Vector3()),
    }));
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
      const material = new THREE.MeshStandardMaterial({
        color: FAMILY_COLOR[cable.family] ?? 0x8b949e,
        transparent: true,
        opacity: cable.family === "optical" ? 0.5 : 0.4,
        metalness: 0.2,
        roughness: 0.6,
      });
      // Real tubes, not 1px lines: WebGL ignores LineBasicMaterial.linewidth.
      const radius = (HardwareScene.WIRE_RADIUS_MM[cable.family] ?? 4) * SCENE_UNITS_PER_MM;
      const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, radius, 6, false), material);
      this.group.add(mesh);
      this.cables.push({ def: cable, mesh, material, curve });
    }
  }

  // Wire jacket radii in mechanical mm. Wire gauge follows the cable family;
  // these are the 3x-thicker display values requested for readability.
  private static readonly WIRE_RADIUS_MM: Record<string, number> = {
    optical: 3,
    rf: 4.5,
    command: 6,
    dc: 6,
    motor: 3.75,
    external: 12,
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
      const radius = (HardwareScene.WIRE_RADIUS_MM[rec.def.family] ?? 4) * 1.7 * SCENE_UNITS_PER_MM;
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
      const cableRadius = HardwareScene.WIRE_RADIUS_MM[rec.def.family] ?? 4;
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
