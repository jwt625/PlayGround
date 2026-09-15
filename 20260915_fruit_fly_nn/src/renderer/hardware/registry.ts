import manifestJson from "../../../assets/generated/hardware-v2/manifest.json";
import wiringJson from "../../../assets/generated/hardware-v2/wiring-plan.json";

/**
 * H0 typed hardware registry (docs/HARDWARE_SCENE_TASKS.md, Phase H0).
 *
 * Loads the generated hardware-v2 manifest and wiring plan into a typed
 * component / port / channel / cable registry. This is display metadata only:
 * it never carries solver aperture coordinates, so importing it cannot change
 * any optics or training value. The 68 MB vendor STEP reference cache is NOT
 * importable from here (see assets/reference/hardware) and must never ship to
 * the browser.
 *
 * Port convention (manifest): port node local +Z points outward along the cable
 * tangent; mating ports have opposed normals; positions are parent-local.
 */

export type PortKind = "optical" | "rf" | "control" | "power" | "beam";

export interface PortSpec {
  name: string;
  node: string;
  position_m: readonly [number, number, number];
  outward_normal: readonly [number, number, number];
  kind: PortKind;
  connector: string;
  channel: string | null;
  parent: string;
  key_reference_local?: readonly [number, number, number];
}

export interface ControlSpec {
  node: string;
  axis: readonly [number, number, number];
  contact_node?: string;
}

export interface ComponentSpec {
  name: string;
  file: string;
  bytes: number;
  sha256: string;
  bounds_m: readonly [readonly [number, number, number], readonly [number, number, number]];
  triangles: number;
  ports: readonly PortSpec[];
  controls: readonly ControlSpec[];
  dimensionStatus: string;
}

export interface HardwareNode {
  id: string;
  asset: string | null;
  status: "asset-ready" | "asset-needed";
  /** Channel this node belongs to, e.g. CH07, or null for shared nodes. */
  channel: string | null;
  /** Logical port name -> spec, for asset-ready nodes. */
  ports: ReadonlyMap<string, PortSpec>;
  /** Declared placeholder port names, for asset-needed nodes. */
  placeholderPorts: readonly string[];
}

export interface PortRef {
  nodeId: string;
  portName: string;
}

export interface CableSpec {
  id: string;
  source: PortRef;
  target: PortRef;
  kind: PortKind;
  channel: string | null;
}

export interface ChannelHardware {
  channel: string;
  phase: string;
  driver: string;
  amplifier: string;
  mount: string;
  focus: string;
}

export interface HardwareRegistry {
  units: string;
  upAxis: string;
  portConvention: string;
  components: ReadonlyMap<string, ComponentSpec>;
  nodes: ReadonlyMap<string, HardwareNode>;
  cables: readonly CableSpec[];
  channels: ReadonlyMap<string, ChannelHardware>;
  provenance: string;
  limitations: readonly string[];
}

interface RawPort {
  name: string;
  node: string;
  position_m: number[];
  outward_normal: number[];
  kind: string;
  connector: string;
  channel: string | null;
  parent: string;
  key_reference_local?: number[];
}
interface RawComponent {
  name: string;
  file: string;
  bytes: number;
  sha256: string;
  bounds_m: number[][];
  triangles: number;
  ports: RawPort[];
  controls: { node: string; axis: number[]; contact_node?: string }[];
  dimension_status: string;
}
interface RawManifest {
  units: string;
  up_axis: string;
  port_convention: string;
  provenance: string;
  limitations: string[];
  components: RawComponent[];
}
interface RawNode {
  id: string;
  asset: string | null;
  placeholder_ports: string[];
  status: string;
}
interface RawEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  channel: string | null;
}
interface RawWiring {
  schema_version: number;
  channels: number;
  nodes: RawNode[];
  edges: RawEdge[];
}

const CHANNEL_ID_RE = /^CH\d{2}$/;

function parseRefs(ref: string): PortRef {
  const dot = ref.indexOf(".");
  if (dot <= 0 || dot === ref.length - 1) throw new Error(`invalid port reference: ${ref}`);
  return { nodeId: ref.slice(0, dot), portName: ref.slice(dot + 1) };
}

function channelOf(nodeId: string): string | null {
  const head = nodeId.split("-")[0];
  return CHANNEL_ID_RE.test(head) ? head : null;
}

/** Placeholder nodes have no asset; infer a port kind from the logical name. */
function placeholderKind(name: string): PortKind {
  if (/dc|power/i.test(name)) return "power";
  if (/optical/i.test(name)) return "optical";
  return "control";
}

export function buildHardwareRegistry(
  manifest: RawManifest = manifestJson as unknown as RawManifest,
  wiring: RawWiring = wiringJson as unknown as RawWiring,
): HardwareRegistry {
  const components = new Map<string, ComponentSpec>();
  for (const c of manifest.components) {
    if (components.has(c.name)) throw new Error(`duplicate component ${c.name}`);
    components.set(c.name, {
      name: c.name,
      file: c.file,
      bytes: c.bytes,
      sha256: c.sha256,
      bounds_m: c.bounds_m as unknown as ComponentSpec["bounds_m"],
      triangles: c.triangles,
      ports: c.ports.map((p) => ({
        name: p.name,
        node: p.node,
        position_m: p.position_m as unknown as PortSpec["position_m"],
        outward_normal: p.outward_normal as unknown as PortSpec["outward_normal"],
        kind: p.kind as PortKind,
        connector: p.connector,
        channel: p.channel,
        parent: p.parent,
        key_reference_local: p.key_reference_local as unknown as PortSpec["key_reference_local"],
      })),
      controls: c.controls.map((k) => ({
        node: k.node,
        axis: k.axis as unknown as ControlSpec["axis"],
        contact_node: k.contact_node,
      })),
      dimensionStatus: c.dimension_status,
    });
  }

  const nodes = new Map<string, HardwareNode>();
  for (const n of wiring.nodes) {
    if (nodes.has(n.id)) throw new Error(`duplicate node ${n.id}`);
    const component = n.asset ? components.get(n.asset) : undefined;
    if (n.asset && !component) throw new Error(`node ${n.id} references unknown asset ${n.asset}`);
    const ports = new Map<string, PortSpec>();
    for (const p of component?.ports ?? []) ports.set(p.name, p);
    nodes.set(n.id, {
      id: n.id,
      asset: n.asset,
      status: n.status === "asset-ready" ? "asset-ready" : "asset-needed",
      channel: channelOf(n.id),
      ports,
      placeholderPorts: n.placeholder_ports,
    });
  }

  const cables: CableSpec[] = wiring.edges.map((e) => {
    if (!e.id) throw new Error("cable without id");
    return { id: e.id, source: parseRefs(e.source), target: parseRefs(e.target), kind: e.kind as PortKind, channel: e.channel };
  });
  if (new Set(cables.map((c) => c.id)).size !== cables.length) throw new Error("duplicate cable id");

  const channels = new Map<string, ChannelHardware>();
  for (const [id, node] of nodes) {
    if (!node.channel) continue;
    const role = id.slice(node.channel.length + 1);
    const entry = channels.get(node.channel) ?? { channel: node.channel, phase: "", driver: "", amplifier: "", mount: "", focus: "" };
    if (role === "phase") entry.phase = id;
    else if (role === "driver") entry.driver = id;
    else if (role === "amplifier") entry.amplifier = id;
    else if (role === "mount") entry.mount = id;
    else if (role === "focus") entry.focus = id;
    channels.set(node.channel, entry);
  }

  return {
    units: manifest.units,
    upAxis: manifest.up_axis,
    portConvention: manifest.port_convention,
    provenance: manifest.provenance,
    limitations: manifest.limitations,
    components,
    nodes,
    cables,
    channels,
  };
}

export interface RegistryIssue {
  severity: "error" | "warning";
  cable?: string;
  message: string;
}

const KIND_COMPAT: Record<string, PortKind[]> = {
  optical: ["optical", "beam"],
  rf: ["rf"],
  control: ["control"],
  power: ["power"],
};

function connectorClass(connector: string): "mate" | "pigtail" | "cable" | "free" | "placeholder" {
  if (connector === "placeholder") return "placeholder";
  if (connector === "PM-pigtail") return "pigtail";
  if (connector === "jacket-2mm" || connector === "coax") return "cable";
  if (connector === "free-space") return "free";
  return "mate";
}

/**
 * Validates the registry. Errors are structural (unresolvable/duplicate/incompatible
 * endpoints); warnings mark connectors that need an explicit assembly or
 * distribution junction before they can be drawn as mated.
 */
export function validateRegistry(registry: HardwareRegistry): RegistryIssue[] {
  const issues: RegistryIssue[] = [];

  for (const channel of ["CH01", "CH02", "CH03", "CH04", "CH05", "CH06", "CH07", "CH08", "CH09", "CH10", "CH11", "CH12", "CH13", "CH14", "CH15", "CH16", "CH17", "CH18", "CH19"]) {
    const entry = registry.channels.get(channel);
    if (!entry) {
      issues.push({ severity: "error", message: `channel ${channel} has no hardware mapping` });
      continue;
    }
    for (const role of ["phase", "driver", "amplifier", "mount", "focus"] as const) {
      if (!entry[role]) issues.push({ severity: "error", message: `${channel} missing ${role}` });
    }
  }
  if (registry.channels.size !== 19) issues.push({ severity: "error", message: `expected 19 channels, got ${registry.channels.size}` });

  const resolve = (ref: PortRef): PortSpec | undefined => {
    const node = registry.nodes.get(ref.nodeId);
    if (!node) return undefined;
    const port = node.ports.get(ref.portName);
    if (port) return port;
    if (node.placeholderPorts.includes(ref.portName)) {
      return { name: ref.portName, node: ref.portName, position_m: [0, 0, 0], outward_normal: [0, 0, 1], kind: placeholderKind(ref.portName), connector: "placeholder", channel: node.channel, parent: "placeholder" };
    }
    return undefined;
  };

  for (const cable of registry.cables) {
    const source = resolve(cable.source);
    const target = resolve(cable.target);
    if (!source) issues.push({ severity: "error", cable: cable.id, message: `unresolved source ${cable.source.nodeId}.${cable.source.portName}` });
    if (!target) issues.push({ severity: "error", cable: cable.id, message: `unresolved target ${cable.target.nodeId}.${cable.target.portName}` });
    if (!source || !target) continue;

    const compatible = KIND_COMPAT[cable.kind] ?? [cable.kind];
    if (!compatible.includes(source.kind) || !compatible.includes(target.kind)) {
      issues.push({ severity: "error", cable: cable.id, message: `kind ${cable.kind} endpoints ${source.kind}/${target.kind}` });
    }
    const sourceClass = connectorClass(source.connector);
    const targetClass = connectorClass(target.connector);
    if (sourceClass === "mate" && targetClass === "mate" && source.connector !== target.connector) {
      issues.push({ severity: "error", cable: cable.id, message: `incompatible mates ${source.connector} -> ${target.connector}` });
    }
    if (sourceClass === "pigtail" || targetClass === "pigtail") {
      issues.push({ severity: "warning", cable: cable.id, message: "pigtail exit needs an expanded connector assembly" });
    }
    const sourceChannel = registry.nodes.get(cable.source.nodeId)?.channel ?? null;
    const targetChannel = registry.nodes.get(cable.target.nodeId)?.channel ?? null;
    if (cable.channel && ((sourceChannel && sourceChannel !== cable.channel) || (targetChannel && targetChannel !== cable.channel))) {
      issues.push({ severity: "error", cable: cable.id, message: `channel mismatch for ${cable.channel}` });
    }
  }

  return issues;
}
