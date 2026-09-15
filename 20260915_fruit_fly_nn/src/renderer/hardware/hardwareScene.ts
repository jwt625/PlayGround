import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ChannelActual } from "../../optics/types";
import type { HardwareRegistry, PortSpec } from "./registry";

/**
 * Hardware bench scene (docs/HARDWARE_SCENE_TASKS.md).
 *
 * Imports the generated hardware-v2 GLBs under an explicit display transform
 * (HARDWARE_SCALE), separate from the SI solver coordinates. Every registry node
 * is instantiated; connectors are mated as children of the GLB port nodes so
 * they follow tip/tilt motion; cables are routed from the wiring plan; the
 * collimator pivots are driven by the actual per-channel pointing.
 *
 * Vendor STEP reference files are never fetched.
 */

export const HARDWARE_SCALE = 60;
const MOTION_GAIN = 25;
const BASE_TRAY_Z = -22;
const TRAY_DZ = 24;

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

export interface HardwareSceneOptions {
  onProgress?: (message: string) => void;
}

interface PlacedNode {
  id: string;
  component: string;
  channel: string | null;
  object: THREE.Object3D;
  ports: Map<string, THREE.Object3D>;
  tipPivot?: THREE.Object3D;
  tiltPivot?: THREE.Object3D;
}

interface CableRecord {
  id: string;
  kind: string;
  channel: string | null;
  line: THREE.Line;
  source: { node: PlacedNode; port: PortSpec } | null;
  target: { node: PlacedNode; port: PortSpec } | null;
}

function findNode(root: THREE.Object3D, name: string): THREE.Object3D | null {
  let found: THREE.Object3D | null = null;
  root.traverse((o) => {
    if (!found && o.name === name) found = o;
  });
  return found;
}

function trayPosition(index: number, stage: number): THREE.Vector3 {
  const perRow = 7;
  const row = Math.floor(index / perRow);
  const col = index % perRow;
  const x = (col - 3) * 13;
  const y = 12 - row * 11;
  const z = BASE_TRAY_Z - stage * TRAY_DZ;
  return new THREE.Vector3(x, y, z);
}

/** Aperture collimator positions spread out for readability (display only). */
function aperturePosition(x_m: number, y_m: number, spread: number): THREE.Vector3 {
  return new THREE.Vector3(x_m * spread, y_m * spread + 8, -4);
}

export class HardwareScene {
  readonly group = new THREE.Group();
  private readonly registry: HardwareRegistry;
  private readonly options: HardwareSceneOptions;
  private templates = new Map<string, THREE.Object3D>();
  private placed: PlacedNode[] = [];
  private cables: CableRecord[] = [];
  private readonly labels = new THREE.Group();
  private readonly connectors = new THREE.Group();
  loaded = false;
  error: string | null = null;
  selectedChannel: string | null = null;
  assetInstances = 0;
  connectorInstances = 0;
  private readonly instantiatedKinds = new Set<string>();
  private failureLog: string[] = [];

  constructor(registry: HardwareRegistry, options: HardwareSceneOptions = {}) {
    this.registry = registry;
    this.options = options;
    this.group.add(this.labels, this.connectors);
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
      this.buildLayout();
      this.buildCables();
      this.loaded = true;
    } catch (err) {
      this.error = String(err);
      this.options.onProgress?.(`hardware load failed: ${String(err)}`);
    }
  }

  private instantiate(component: string, scale = HARDWARE_SCALE): THREE.Object3D {
    const template = this.templates.get(component);
    if (!template) throw new Error(`hardware template ${component} not loaded`);
    this.instantiatedKinds.add(component);
    const clone = template.clone(true);
    clone.scale.setScalar(scale);
    return clone;
  }

  private buildLayout(): void {
    // Shared nodes.
    this.placePlaceholder("seed", new THREE.Vector3(-95, 30, -46), 22);
    this.placePlaceholder("supply", new THREE.Vector3(-95, 14, -46), 22);
    this.placeAsset("splitter-19", "splitter", null, new THREE.Vector3(0, 34, -30));
    // Console enlarged so the display-scaled fly can reach it; labelled below.
    this.placeAsset("fly-console", "console", null, new THREE.Vector3(-70, -6, 4), 1.9);
    this.placePlaceholder("phase-command-junction", new THREE.Vector3(96, 18, -36), 16);
    this.placePlaceholder("motor-driver-junction", new THREE.Vector3(96, 2, -36), 16);

    for (let i = 1; i <= 19; i++) {
      const channel = `CH${String(i).padStart(2, "0")}`;
      const entry = this.registry.channels.get(channel);
      if (!entry) continue;
      // aperturePosition uses raw x scaled by spread; index gives array slots.
      const ii = i - 1;
      const qx = ((ii % 5) - 2) * 0.9e-3;
      const qy = (Math.floor(ii / 5) - 2) * 0.9e-3;
      const mount = this.placeAsset("tiptilt-collimator", entry.mount, channel, aperturePosition(qx, qy, 4200));
      if (mount) {
        mount.tipPivot = findNode(mount.object, "tip_pivot") ?? undefined;
        mount.tiltPivot = findNode(mount.object, "tilt_pivot") ?? undefined;
      }
      this.placeAsset("phase-cassette", entry.phase, channel, trayPosition(ii, 0));
      this.placeAsset("optical-amplifier", entry.amplifier, channel, trayPosition(ii, 1));
      this.placeAsset("phase-driver", entry.driver, channel, trayPosition(ii, 2));
      this.placePlaceholder(entry.focus, trayPosition(ii, 3), 6);
      this.addChannelLabel(channel, aperturePosition(qx, qy, 4200).add(new THREE.Vector3(0, 7, 0)));
    }
  }

  private placeAsset(component: string, nodeId: string, channel: string | null, position: THREE.Vector3, scale = HARDWARE_SCALE): PlacedNode | null {
    try {
      const object = this.instantiate(component, scale);
      object.position.copy(position);
      this.group.add(object);
      const node = this.registerNode(component, nodeId, channel, object);
      this.placed.push(node);
      this.assetInstances++;
      this.mateConnectors(node, component);
      return node;
    } catch (err) {
      this.failureLog.push(`${nodeId}: ${String(err)}`);
      return null;
    }
  }

  private placePlaceholder(id: string, position: THREE.Vector3, size: number): PlacedNode {
    const object = new THREE.Mesh(
      new THREE.BoxGeometry(size, size * 0.6, size),
      new THREE.MeshStandardMaterial({ color: 0x2a3646, metalness: 0.4, roughness: 0.6 }),
    );
    object.position.copy(position);
    this.group.add(object);
    const node: PlacedNode = { id, component: "placeholder", channel: id.startsWith("CH") ? id.slice(0, 4) : null, object, ports: new Map() };
    this.placed.push(node);
    return node;
  }

  private registerNode(component: string, id: string, channel: string | null, object: THREE.Object3D): PlacedNode {
    const spec = this.registry.components.get(component)!;
    const ports = new Map<string, THREE.Object3D>();
    for (const port of spec.ports) {
      const portNode = findNode(object, port.node);
      if (portNode) ports.set(port.name, portNode);
    }
    return { id, component, channel, object, ports };
  }

  /** Attach matching connector GLBs to each declared port as a child. */
  private mateConnectors(node: PlacedNode, component: string): void {
    const spec = this.registry.components.get(component);
    if (!spec) return;
    for (const port of spec.ports) {
      if (port.kind === "beam") continue;
      const templateName = port.connector === "SMA" ? "sma-plug" : port.connector === "FC/APC" || port.connector === "PM-pigtail" ? "fc-apc-plug" : null;
      if (!templateName) continue;
      const portNode = node.ports.get(port.name);
      if (!portNode) continue;

      // Splitter channel outputs get a panel bulkhead feedthrough, then the plug
      // mates to the bulkhead front (uses the fc-bulkhead asset as designed).
      if (component === "splitter-19" && /^CH\d/.test(port.name)) {
        const bulkhead = this.instantiate("fc-bulkhead", 1);
        portNode.add(bulkhead);
        this.connectorInstances++;
        const front = findNode(bulkhead, "port_front") ?? bulkhead;
        const plug = this.instantiate("fc-apc-plug", 1);
        plug.rotation.x = Math.PI;
        front.add(plug);
        this.connectorInstances++;
        continue;
      }

      const connector = this.instantiate(templateName, 1);
      connector.rotation.x = Math.PI; // plug body extends outward along port +Z
      portNode.add(connector);
      this.connectorInstances++;
    }
  }

  private addChannelLabel(channel: string, position: THREE.Vector3): void {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 48;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "rgba(10,16,25,0.85)";
    ctx.fillRect(0, 0, 128, 48);
    ctx.fillStyle = "#7ee787";
    ctx.font = "bold 28px monospace";
    ctx.fillText(channel, 12, 34);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
    sprite.position.copy(position);
    sprite.scale.set(10, 3.75, 1);
    this.labels.add(sprite);
  }

  private placedById(): Map<string, PlacedNode> {
    const map = new Map<string, PlacedNode>();
    for (const node of this.placed) map.set(node.id, node);
    return map;
  }

  private worldPortPosition(placed: PlacedNode, portName: string): THREE.Vector3 | null {
    const portNode = placed.ports.get(portName);
    if (!portNode) return null;
    return portNode.getWorldPosition(new THREE.Vector3());
  }

  private buildCables(): void {
    const byId = this.placedById();
    let drawn = 0;
    for (const cable of this.registry.cables) {
      const sourceNode = byId.get(cable.source.nodeId);
      const targetNode = byId.get(cable.target.nodeId);
      if (!sourceNode || !targetNode) continue;
      const sourcePort = this.registry.nodes.get(cable.source.nodeId)?.ports.get(cable.source.portName) ?? null;
      const targetPort = this.registry.nodes.get(cable.target.nodeId)?.ports.get(cable.target.portName) ?? null;
      const start = this.worldPortPosition(sourceNode, cable.source.portName) ?? sourceNode.object.position.clone();
      const end = this.worldPortPosition(targetNode, cable.target.portName) ?? targetNode.object.position.clone();
      const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 10, 0));
      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const color = cable.kind === "optical" ? 0x39c5cf : cable.kind === "rf" ? 0xffa657 : cable.kind === "power" ? 0xf0883e : 0x8b949e;
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(14)),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.32 }),
      );
      this.group.add(line);
      this.cables.push({ id: cable.id, kind: cable.kind, channel: cable.channel, line, source: sourcePort ? { node: sourceNode, port: sourcePort } : null, target: targetPort ? { node: targetNode, port: targetPort } : null });
      drawn++;
    }
    void drawn;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  setSelectedChannel(channel: string | null): void {
    this.selectedChannel = channel;
    for (const cable of this.cables) {
      const mat = cable.line.material as THREE.LineBasicMaterial;
      const on = !channel || cable.channel === channel;
      mat.opacity = on ? (cable.channel === channel ? 0.95 : 0.3) : 0.03;
    }
    for (const node of this.placed) {
      const selected = channel !== null && node.channel === channel;
      node.object.visible = true;
      const scale = selected ? HARDWARE_SCALE * 1.08 : HARDWARE_SCALE;
      if (node.component !== "placeholder") node.object.scale.setScalar(node.component === "fly-console" ? HARDWARE_SCALE * 1.9 : scale);
    }
  }

  /** Drive nested tip/tilt pivots from actual per-channel pointing angles. */
  updateMotion(actual: readonly ChannelActual[]): void {
    if (!this.loaded) return;
    const byId = this.placedById();
    for (let i = 1; i <= actual.length; i++) {
      const channel = `CH${String(i).padStart(2, "0")}`;
      const entry = this.registry.channels.get(channel);
      if (!entry) continue;
      const node = byId.get(entry.mount);
      if (!node?.tipPivot || !node.tiltPivot) continue;
      const s = actual[i - 1];
      // Emission is +Z. Rotation about X moves it toward +/-Y; about Y toward +/-X.
      node.tipPivot.rotation.x = -s.tiltY * MOTION_GAIN;
      node.tiltPivot.rotation.y = s.tiltX * MOTION_GAIN;
    }
  }

  refreshCables(): void {
    for (const cable of this.cables) {
      if (!cable.source || !cable.target) continue;
      const start = this.worldPortPosition(cable.source.node, cable.source.port.name) ?? cable.source.node.object.position.clone();
      const end = this.worldPortPosition(cable.target.node, cable.target.port.name) ?? cable.target.node.object.position.clone();
      const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 10, 0));
      cable.line.geometry.setFromPoints(new THREE.QuadraticBezierCurve3(start, mid, end).getPoints(14));
    }
  }

  get cableCount(): number {
    return this.cables.length;
  }

  get nodeCount(): number {
    return this.placed.length;
  }

  get componentKindCount(): number {
    return this.instantiatedKinds.size;
  }

  get failures(): readonly string[] {
    return this.failureLog;
  }
}
