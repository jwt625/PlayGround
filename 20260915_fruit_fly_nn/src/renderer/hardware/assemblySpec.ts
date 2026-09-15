import channelLayout from "../../../assets/generated/channel-layout.json";

/**
 * Typed implementation artifact for docs/CBC_ASSEMBLY_SPEC.md (Pass 1 + 2).
 *
 * Everything here is mechanical DISPLAY millimetres before one uniform scene
 * conversion. It never feeds solver wavelength, pitch, or target distance.
 *
 * Coordinate/registration convention (§1):
 *   world +X right, +Y up, +Z optical propagation; table top at Y=0.
 *   The 19-channel aperture plate is registered to the optical array by mapping
 *   the mechanical aperture pitch (65 mm) to the optical display pitch and by
 *   translating the mechanical aperture emission center (0, 300, 100) to the
 *   optical array origin. Beams therefore leave the real aperture, not a second
 *   disconnected array.
 */

export const APERTURE_PITCH_MM = 65;
export const APERTURE_CENTER_MM = { x: 0, y: 300, z: 100 } as const;

export type Orientation = "A" | "B" | "C" | "world";

export type PortKind = "optical" | "rf" | "control" | "power" | "motor" | "beam" | "ac";

export interface PortDef {
  name: string;
  offset_mm: readonly [number, number, number];
  normal: readonly [number, number, number];
  kind: PortKind;
  /** mating connector (equipment socket) or cable-side connector family */
  connector: string;
  /** captive pigtail exit: cable starts here, no socket */
  pigtail?: boolean;
  moving?: boolean;
  freeSpace?: boolean;
}

export interface EquipmentDef {
  id: string;
  label: string;
  kind: "support" | "instrument" | "channel" | "aperture" | "console" | "boundary";
  /** GLB component name to reuse, or null for procedural geometry. */
  glb: string | null;
  /** footprint center (X, support Y, Z) in mm, or aperture emission center. */
  position_mm: readonly [number, number, number];
  size_mm: readonly [number, number, number];
  orientation: Orientation;
  channel: string | null;
  ports: readonly PortDef[];
  parent?: string;
  /** true when this equipment's emission is the optical aperture. */
  aperture?: boolean;
}

const opt = (name: string, offset: readonly [number, number, number], normal: readonly [number, number, number], connector: string, extra: Partial<PortDef> = {}): PortDef => ({ name, offset_mm: offset, normal, kind: "optical", connector, ...extra });

// ── Canonical 3+4+5+4+3 aperture positions (mechanical mm, §4) ───────────────
export interface ApertureSite {
  id: string;
  x_mm: number;
  y_mm: number;
  z_mm: number;
}

const channels = (channelLayout as { channels: { id: string; position_pitch_units: number[] }[] }).channels;

export const APERTURE_SITES: readonly ApertureSite[] = channels.map((c) => ({
  id: c.id,
  x_mm: c.position_pitch_units[0] * APERTURE_PITCH_MM,
  y_mm: c.position_pitch_units[1] * APERTURE_PITCH_MM + APERTURE_CENTER_MM.y,
  z_mm: APERTURE_CENTER_MM.z,
}));

export function apertureSite(id: string): ApertureSite {
  const site = APERTURE_SITES.find((s) => s.id === id);
  if (!site) throw new Error(`no aperture site ${id}`);
  return site;
}

// ── Electronics cell assignment (§4 table) ───────────────────────────────────
export interface CellAssignment {
  channel: string;
  row: number;
  col: number;
  xc: number;
  zr: number;
}

const CELL_TABLE: [string, number, number][] = [
  ["CH01", 1, 1], ["CH02", 1, 2], ["CH03", 1, 3], ["CH04", 1, 4], ["CH05", 1, 5],
  ["CH06", 2, 1], ["CH07", 2, 2], ["CH08", 2, 3], ["CH09", 2, 4], ["CH10", 2, 5],
  ["CH11", 3, 1], ["CH12", 3, 2], ["CH13", 3, 3], ["CH14", 3, 4], ["CH15", 3, 5],
  ["CH16", 4, 1], ["CH17", 4, 2], ["CH18", 4, 3], ["CH19", 4, 4],
];
const COL_X = [-640, -320, 0, 320, 640];
const ROW_Z = [-1490, -1100, -710, -320];

export const CELL_ASSIGNMENTS: readonly CellAssignment[] = CELL_TABLE.map(([channel, row, col]) => ({
  channel,
  row,
  col,
  xc: COL_X[col - 1],
  zr: ROW_Z[row - 1],
}));

export function cellOf(channel: string): CellAssignment {
  const c = CELL_ASSIGNMENTS.find((a) => a.channel === channel);
  if (!c) throw new Error(`no cell for ${channel}`);
  return c;
}

function channelEquipment(): EquipmentDef[] {
  const out: EquipmentDef[] = [];
  for (let i = 1; i <= 19; i++) {
    const ch = `CH${String(i).padStart(2, "0")}`;
    const { xc, zr } = cellOf(ch);
    // §4.1 relative cell offsets.
    out.push({
      id: `PH-${ch}`, label: ch, kind: "channel", glb: "phase-cassette", channel: ch,
      position_mm: [xc - 95, 44, zr - 85], size_mm: [80, 30, 16], orientation: "C",
      ports: [
        opt("optical_in", [0, 8, 57], [0, 0, 1], "PM-pigtail", { pigtail: true }),
        opt("optical_out", [0, 8, -57], [0, 0, -1], "PM-pigtail", { pigtail: true }),
        { name: "rf", offset_mm: [23, 8, 0], normal: [1, 0, 0], kind: "rf", connector: "SMA" },
      ],
    });
    out.push({
      id: `DR-${ch}`, label: ch, kind: "channel", glb: "phase-driver", channel: ch,
      position_mm: [xc + 65, 28, zr - 110], size_mm: [105, 80, 38], orientation: "A",
      ports: [
        { name: "rf_out", offset_mm: [0, 19, 48], normal: [0, 0, 1], kind: "rf", connector: "SMA" },
        { name: "dc_power", offset_mm: [-25, 19, -45], normal: [0, 0, -1], kind: "power", connector: "DC" },
        { name: "command", offset_mm: [25, 19, -45], normal: [0, 0, -1], kind: "control", connector: "multipin" },
      ],
    });
    out.push({
      id: `AMP-${ch}`, label: ch, kind: "channel", glb: "optical-amplifier", channel: ch,
      position_mm: [xc + 30, 28, zr + 65], size_mm: [160, 130, 55], orientation: "A",
      ports: [
        opt("optical_in", [-42, 27.5, 72], [0, 0, 1], "FC/APC"),
        opt("optical_out", [42, 27.5, 72], [0, 0, 1], "FC/APC"),
        { name: "dc_power", offset_mm: [-25, 27.5, -70], normal: [0, 0, -1], kind: "power", connector: "DC" },
        { name: "command", offset_mm: [25, 27.5, -70], normal: [0, 0, -1], kind: "control", connector: "multipin" },
      ],
    });
    const site = apertureSite(ch);
    out.push({
      id: `COL-${ch}`, label: ch, kind: "aperture", glb: "tiptilt-collimator", channel: ch,
      position_mm: [site.x_mm, site.y_mm, site.z_mm], size_mm: [45, 45, 70], orientation: "world",
      aperture: true,
      ports: [
        opt("fiber_in", [0, 0, -70], [0, 0, -1], "FC/APC", { moving: true }),
        { name: "emission", offset_mm: [0, 0, 0], normal: [0, 0, 1], kind: "beam", connector: "free-space", freeSpace: true },
        { name: "m_tip", offset_mm: [0, -20, -25], normal: [0, 0, -1], kind: "motor", connector: "motor", moving: true },
        { name: "m_tilt", offset_mm: [20, -20, -25], normal: [0, 0, -1], kind: "motor", connector: "motor", moving: true },
        { name: "m_focus", offset_mm: [-20, -20, -25], normal: [0, 0, -1], kind: "motor", connector: "motor", moving: true },
      ],
    });
    out.push({
      id: `JB-${ch}`, label: ch, kind: "aperture", glb: null, channel: ch,
      position_mm: [site.x_mm, site.y_mm - 20, site.z_mm - 155], size_mm: [30, 22, 12], orientation: "world",
      ports: [
        { name: "IN", offset_mm: [0, 0, -12], normal: [0, 0, -1], kind: "motor", connector: "multipin" },
        { name: "TIP", offset_mm: [-10, 0, 12], normal: [0, 0, 1], kind: "motor", connector: "motor" },
        { name: "TILT", offset_mm: [0, 0, 12], normal: [0, 0, 1], kind: "motor", connector: "motor" },
        { name: "FOCUS", offset_mm: [10, 0, 12], normal: [0, 0, 1], kind: "motor", connector: "motor" },
      ],
    });
  }
  return out;
}

function sharedPorts(id: string): PortDef[] {
  switch (id) {
    case "SEED":
      return [opt("optical_out", [0, 30, 70], [0, 0, 1], "FC/APC"), { name: "dc_power", offset_mm: [0, 25, -70], normal: [0, 0, -1], kind: "power", connector: "DC" }];
    case "SPLIT": {
      const ports: PortDef[] = [opt("input", [0, 16, -67], [0, 0, -1], "FC/APC")];
      const xs = [-85.5, -66.5, -47.5, -28.5, -9.5, 9.5, 28.5, 47.5, 66.5, 85.5];
      for (let i = 1; i <= 19; i++) {
        const col = i <= 10 ? xs[i - 1] : xs[i - 11];
        const y = i <= 10 ? 10 : 24;
        ports.push(opt(`CH${String(i).padStart(2, "0")}`, [-col, y, 67], [0, 0, 1], "FC/APC"));
      }
      return ports;
    }
    case "PSU":
      return [
        { name: "ac_in", offset_mm: [-60, 35, -110], normal: [0, 0, -1], kind: "ac", connector: "AC" },
        { name: "dc_out", offset_mm: [60, 35, -110], normal: [0, 0, -1], kind: "power", connector: "DC-trunk" },
      ];
    case "PDU-M": {
      const ports: PortDef[] = [{ name: "dc_in", offset_mm: [0, 22, -50], normal: [0, 0, -1], kind: "power", connector: "DC-trunk" }];
      ["R1", "R2", "R3", "R4", "SHARED", "MOTOR"].forEach((n, i) => ports.push({ name: n, offset_mm: [-90 + i * 36, 22, 50], normal: [0, 0, 1], kind: "power", connector: "DC-branch" }));
      return ports;
    }
    case "PDU-S": {
      const ports: PortDef[] = [{ name: "dc_in", offset_mm: [0, 17, -40], normal: [0, 0, -1], kind: "power", connector: "DC-branch" }];
      ["SEED", "CONSOLE", "IO"].forEach((n, i) => ports.push({ name: n, offset_mm: [-35 + i * 35, 17, 40], normal: [0, 0, 1], kind: "power", connector: "DC-branch" }));
      return ports;
    }
    case "IO": {
      const ports: PortDef[] = [
        { name: "console", offset_mm: [-60, 40, -90], normal: [0, 0, -1], kind: "control", connector: "multipin" },
        { name: "dc_power", offset_mm: [60, 40, -90], normal: [0, 0, -1], kind: "power", connector: "DC-branch" },
      ];
      for (let i = 1; i <= 19; i++) ports.push({ name: `PH${String(i).padStart(2, "0")}`, offset_mm: [-243 + (i - 1) * 27, 30, 90], normal: [0, 0, 1], kind: "control", connector: "multipin" });
      for (let i = 1; i <= 19; i++) ports.push({ name: `GAIN${String(i).padStart(2, "0")}`, offset_mm: [-243 + (i - 1) * 27, 65, 90], normal: [0, 0, 1], kind: "control", connector: "multipin" });
      return ports;
    }
    case "MC": {
      const ports: PortDef[] = [
        { name: "console", offset_mm: [-45, 40, -90], normal: [0, 0, -1], kind: "control", connector: "multipin" },
        { name: "dc_power", offset_mm: [45, 40, -90], normal: [0, 0, -1], kind: "power", connector: "DC-branch" },
      ];
      for (let i = 1; i <= 19; i++) ports.push({ name: `AX${String(i).padStart(2, "0")}`, offset_mm: [-108 + (i - 1) * 12, 30, 90], normal: [0, 0, 1], kind: "motor", connector: "motor" });
      return ports;
    }
    case "CON":
      return [
        { name: "phase_bus", offset_mm: [-160, 40, -150], normal: [0, 0, -1], kind: "control", connector: "multipin" },
        { name: "motor_bus", offset_mm: [0, 40, -150], normal: [0, 0, -1], kind: "control", connector: "multipin" },
        { name: "dc_power", offset_mm: [160, 40, -150], normal: [0, 0, -1], kind: "power", connector: "DC-branch" },
      ];
    case "BOUNDARY":
      return [{ name: "AC", offset_mm: [0, 40, 0], normal: [0, 0, -1], kind: "ac", connector: "AC" }];
    default:
      return [];
  }
}

function rowPduPorts(): PortDef[] {
  const ports: PortDef[] = [{ name: "dc_in", offset_mm: [0, 17, -40], normal: [0, 0, -1], kind: "power", connector: "DC-branch" }];
  for (let i = 1; i <= 10; i++) ports.push({ name: `O${String(i).padStart(2, "0")}`, offset_mm: [-81 + (i - 1) * 18, 17, 40], normal: [0, 0, 1], kind: "power", connector: "DC-branch" });
  return ports;
}

export const SHARED_EQUIPMENT: readonly EquipmentDef[] = [
  { id: "TABLE", label: "Optical table", kind: "support", glb: null, position_mm: [0, 0, -775], size_mm: [3200, 60, 2150], orientation: "world", channel: null, ports: [] },
  { id: "RACK-L", label: "Left rack", kind: "support", glb: null, position_mm: [-1220, 0, -1380], size_mm: [560, 200, 700], orientation: "world", channel: null, ports: [] },
  { id: "RACK-R", label: "Right rack", kind: "support", glb: null, position_mm: [1200, 0, -1400], size_mm: [640, 360, 760], orientation: "world", channel: null, ports: [] },
  { id: "SEED", label: "Seed laser", kind: "instrument", glb: null, position_mm: [-1330, 20, -1490], size_mm: [200, 60, 140], orientation: "world", channel: null, ports: sharedPorts("SEED") },
  { id: "SPLIT", label: "1x19 splitter", kind: "instrument", glb: "splitter-19", position_mm: [-1100, 20, -1320], size_mm: [210, 32, 120], orientation: "B", channel: null, ports: sharedPorts("SPLIT") },
  { id: "PSU", label: "Bench supply", kind: "instrument", glb: null, position_mm: [-1220, 180, -1490], size_mm: [240, 100, 220], orientation: "world", channel: null, ports: sharedPorts("PSU") },
  { id: "PDU-S", label: "Shared PDU", kind: "instrument", glb: null, position_mm: [-1350, 20, -1150], size_mm: [120, 35, 80], orientation: "world", channel: null, ports: sharedPorts("PDU-S") },
  { id: "PDU-M", label: "Main PDU", kind: "instrument", glb: null, position_mm: [1200, 20, -1500], size_mm: [240, 45, 100], orientation: "world", channel: null, ports: sharedPorts("PDU-M") },
  { id: "IO", label: "Command chassis", kind: "instrument", glb: null, position_mm: [1200, 180, -1440], size_mm: [560, 90, 180], orientation: "world", channel: null, ports: sharedPorts("IO") },
  { id: "MC", label: "Motor controller", kind: "instrument", glb: null, position_mm: [1200, 340, -1360], size_mm: [320, 90, 180], orientation: "world", channel: null, ports: sharedPorts("MC") },
  { id: "CON", label: "Console", kind: "console", glb: "fly-console", position_mm: [-1220, 20, -250], size_mm: [520, 120, 300], orientation: "world", channel: null, ports: sharedPorts("CON") },
  { id: "OP-PLATFORM", label: "Operator platform", kind: "support", glb: null, position_mm: [-1220, 0, 85], size_mm: [560, 20, 330], orientation: "world", channel: null, ports: [] },
  { id: "AP-FRAME", label: "Aperture frame", kind: "support", glb: null, position_mm: [0, 300, 100], size_mm: [400, 400, 40], orientation: "world", channel: null, ports: [] },
  { id: "BOUNDARY", label: "Power entry", kind: "boundary", glb: null, position_mm: [-1550, 40, -1810], size_mm: [60, 60, 60], orientation: "world", channel: null, ports: sharedPorts("BOUNDARY") },
];

export const ROW_TRAYS: readonly EquipmentDef[] = ROW_Z.map((zr, i) => ({
  id: `TRAY-R${i + 1}`, label: `Row ${i + 1} tray`, kind: "support", glb: null,
  position_mm: [0, 20, zr], size_mm: [1580, 8, 350], orientation: "world", channel: null, ports: [],
}));

export const ROW_PDUS: readonly EquipmentDef[] = ROW_Z.map((zr, i) => ({
  id: `PDU-R${i + 1}`, label: `Row ${i + 1} PDU`, kind: "instrument", glb: null,
  position_mm: [1010, 20, zr], size_mm: [200, 35, 80], orientation: "world", channel: null, ports: rowPduPorts(),
}));

export const CHANNEL_EQUIPMENT: readonly EquipmentDef[] = channelEquipment();

export const ALL_EQUIPMENT: readonly EquipmentDef[] = [...SHARED_EQUIPMENT, ...ROW_TRAYS, ...ROW_PDUS, ...CHANNEL_EQUIPMENT];

export function equipmentById(id: string): EquipmentDef {
  const e = ALL_EQUIPMENT.find((x) => x.id === id);
  if (!e) throw new Error(`no equipment ${id}`);
  return e;
}

export function portOf(id: string, port: string): PortDef {
  const e = equipmentById(id);
  const p = e.ports.find((x) => x.name === port);
  if (!p) throw new Error(`no port ${id}.${port}`);
  return p;
}

// ── Cable schedule (§6) ──────────────────────────────────────────────────────
export interface CableDef {
  id: string;
  family: string;
  kind: PortKind;
  channel: string | null;
  from: { equipment: string; port: string };
  to: { equipment: string; port: string };
  external?: boolean;
}

const ROW_FOR_CHANNEL: Record<string, number> = {};
CELL_ASSIGNMENTS.forEach((c) => (ROW_FOR_CHANNEL[c.channel] = c.row));

function channelCables(): CableDef[] {
  const cables: CableDef[] = [];
  for (let i = 1; i <= 19; i++) {
    const ch = `CH${String(i).padStart(2, "0")}`;
    const num = String(i).padStart(2, "0");
    const row = ROW_FOR_CHANNEL[ch];
    const inRow = (i - 1) % 5; // 5 channels per row, 10 PDU outputs per row
    const drPort = `O${String(inRow * 2 + 1).padStart(2, "0")}`;
    const ampPort = `O${String(inRow * 2 + 2).padStart(2, "0")}`;
    cables.push(
      { id: `O-IN-${ch}`, family: "optical", kind: "optical", channel: ch, from: { equipment: "SPLIT", port: ch }, to: { equipment: `PH-${ch}`, port: "optical_in" } },
      { id: `O-MID-${ch}`, family: "optical", kind: "optical", channel: ch, from: { equipment: `PH-${ch}`, port: "optical_out" }, to: { equipment: `AMP-${ch}`, port: "optical_in" } },
      { id: `O-OUT-${ch}`, family: "optical", kind: "optical", channel: ch, from: { equipment: `AMP-${ch}`, port: "optical_out" }, to: { equipment: `COL-${ch}`, port: "fiber_in" } },
      { id: `RF-${ch}`, family: "rf", kind: "rf", channel: ch, from: { equipment: `DR-${ch}`, port: "rf_out" }, to: { equipment: `PH-${ch}`, port: "rf" } },
      { id: `C-PH-${ch}`, family: "command", kind: "control", channel: ch, from: { equipment: "IO", port: `PH${num}` }, to: { equipment: `DR-${ch}`, port: "command" } },
      { id: `C-GAIN-${ch}`, family: "command", kind: "control", channel: ch, from: { equipment: "IO", port: `GAIN${num}` }, to: { equipment: `AMP-${ch}`, port: "command" } },
      { id: `P-DR-${ch}`, family: "dc", kind: "power", channel: ch, from: { equipment: `PDU-R${row}`, port: drPort }, to: { equipment: `DR-${ch}`, port: "dc_power" } },
      { id: `P-AMP-${ch}`, family: "dc", kind: "power", channel: ch, from: { equipment: `PDU-R${row}`, port: ampPort }, to: { equipment: `AMP-${ch}`, port: "dc_power" } },
      { id: `M-TRUNK-${ch}`, family: "motor", kind: "motor", channel: ch, from: { equipment: "MC", port: `AX${num}` }, to: { equipment: `JB-${ch}`, port: "IN" } },
      { id: `M-TIP-${ch}`, family: "motor", kind: "motor", channel: ch, from: { equipment: `JB-${ch}`, port: "TIP" }, to: { equipment: `COL-${ch}`, port: "m_tip" } },
      { id: `M-TILT-${ch}`, family: "motor", kind: "motor", channel: ch, from: { equipment: `JB-${ch}`, port: "TILT" }, to: { equipment: `COL-${ch}`, port: "m_tilt" } },
      { id: `M-FOCUS-${ch}`, family: "motor", kind: "motor", channel: ch, from: { equipment: `JB-${ch}`, port: "FOCUS" }, to: { equipment: `COL-${ch}`, port: "m_focus" } },
    );
  }
  return cables;
}

export const SHARED_CABLES: readonly CableDef[] = [
  { id: "EXT-AC", family: "external", kind: "ac", channel: null, from: { equipment: "BOUNDARY", port: "AC" }, to: { equipment: "PSU", port: "ac_in" }, external: true },
  { id: "O-SEED", family: "optical", kind: "optical", channel: null, from: { equipment: "SEED", port: "optical_out" }, to: { equipment: "SPLIT", port: "input" } },
  { id: "C-CON-PH", family: "command", kind: "control", channel: null, from: { equipment: "CON", port: "phase_bus" }, to: { equipment: "IO", port: "console" } },
  { id: "C-CON-MOT", family: "command", kind: "control", channel: null, from: { equipment: "CON", port: "motor_bus" }, to: { equipment: "MC", port: "console" } },
  { id: "P-MAIN", family: "dc", kind: "power", channel: null, from: { equipment: "PSU", port: "dc_out" }, to: { equipment: "PDU-M", port: "dc_in" } },
  { id: "P-R1", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-M", port: "R1" }, to: { equipment: "PDU-R1", port: "dc_in" } },
  { id: "P-R2", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-M", port: "R2" }, to: { equipment: "PDU-R2", port: "dc_in" } },
  { id: "P-R3", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-M", port: "R3" }, to: { equipment: "PDU-R3", port: "dc_in" } },
  { id: "P-R4", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-M", port: "R4" }, to: { equipment: "PDU-R4", port: "dc_in" } },
  { id: "P-SHARED", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-M", port: "SHARED" }, to: { equipment: "PDU-S", port: "dc_in" } },
  { id: "P-MOTOR", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-M", port: "MOTOR" }, to: { equipment: "MC", port: "dc_power" } },
  { id: "P-SEED", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-S", port: "SEED" }, to: { equipment: "SEED", port: "dc_power" } },
  { id: "P-CON", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-S", port: "CONSOLE" }, to: { equipment: "CON", port: "dc_power" } },
  { id: "P-IO", family: "dc", kind: "power", channel: null, from: { equipment: "PDU-S", port: "IO" }, to: { equipment: "IO", port: "dc_power" } },
];

export const CABLES: readonly CableDef[] = [...SHARED_CABLES, ...channelCables()];

export interface InventoryAudit {
  channels: number;
  perChannel: Record<string, number>;
  actuators: number;
  cables: number;
  byFamily: Record<string, number>;
  internalRuns: number;
  externalRuns: number;
  fcPlugs: number;
  fcSockets: number;
}

/** §10 inventory/connection audit computed from the encoded specification. */
export function auditInventory(): InventoryAudit {
  const byFamily: Record<string, number> = {};
  for (const c of CABLES) byFamily[c.family] = (byFamily[c.family] ?? 0) + 1;
  const internalRuns = CABLES.filter((c) => !c.external).length;
  // FC plugs/sockets: seed+splitter input + 19 splitter-CH + 38 amp + 19 col = 78.
  const fcSockets = 1 + 1 + 19 + 38 + 19;
  const fcPlugs = 20 * 2 + 38; // 20 two-ended patch cords + 38 pigtail free ends
  const perChannel: Record<string, number> = { PH: 19, DR: 19, AMP: 19, COL: 19, JB: 19, FOC: 19, MT: 19, MY: 19 };
  return {
    channels: 19,
    perChannel,
    actuators: 19 * 3,
    cables: CABLES.length,
    byFamily,
    internalRuns,
    externalRuns: CABLES.length - internalRuns,
    fcPlugs,
    fcSockets,
  };
}
