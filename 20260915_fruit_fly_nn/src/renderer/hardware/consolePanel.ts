import * as THREE from "three";

/**
 * Console panel from docs/CBC_ASSEMBLY_SPEC.md §8 (Pass 3).
 *
 * 19 phase/amplitude knob pairs + five larger selected-channel knobs = 43 knobs,
 * a 19-button canonical hex selector, and a status display. Knob indicators are
 * bound to the actual controller state; they are not decorative. Panel surface
 * coordinates u (right) / v (rearward) are in mechanical mm.
 */

export interface ConsoleLayout {
  surface_mm: { uHalf: number; vHalf: number; tiltDeg: number; centerY: number };
}

export const CONSOLE_LAYOUT: ConsoleLayout = {
  surface_mm: { uHalf: 260, vHalf: 150, tiltDeg: 15, centerY: 75 },
};

export interface ConsoleKnob {
  name: string;
  mesh: THREE.Mesh;
  pointer: THREE.Mesh;
  channel: string | null;
  kind: "phase" | "amplitude" | "tip" | "tilt" | "focus" | "gain";
}

export interface ConsolePanel {
  group: THREE.Group;
  knobs: Map<string, ConsoleKnob>;
  selectorMeshes: Map<string, THREE.Mesh>;
  displayMaterial: THREE.MeshBasicMaterial;
  /** channelId -> local surface coords (u, v) for foreleg targets */
  stripCenters: Map<string, { u: number; v: number }>;
}

export interface ConsoleBuildOptions {
  unitsPerMm: number;
  panelWidthMm?: number;
  panelDepthMm?: number;
}

const KNOB_R = 6;
const KNOB_H = 8;
const BIG_KNOB_R = 9;

function knobMesh(radius: number, height: number, color: number): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, height, 14),
    new THREE.MeshStandardMaterial({ color, metalness: 0.6, roughness: 0.35 }),
  );
  return mesh;
}

function pointerMesh(radius: number, height: number, color: number): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(radius * 0.25, height * 0.2, radius),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.6 }),
  );
  return mesh;
}

export function buildConsolePanel(options: ConsoleBuildOptions): ConsolePanel {
  const s = options.unitsPerMm;
  const group = new THREE.Group();
  group.name = "console-panel";
  const knobs = new Map<string, ConsoleKnob>();
  const selectorMeshes = new Map<string, THREE.Mesh>();
  const stripCenters = new Map<string, { u: number; v: number }>();

  // Sloped surface: u -> local X, v -> local -Z (rearward), w -> local Y.
  const surface = new THREE.Group();
  surface.rotation.x = THREE.MathUtils.degToRad(CONSOLE_LAYOUT.surface_mm.tiltDeg);
  group.add(surface);

  const panel = new THREE.Mesh(
    new THREE.BoxGeometry((CONSOLE_LAYOUT.surface_mm.uHalf * 2 + 30) * s, 8 * s, (CONSOLE_LAYOUT.surface_mm.vHalf * 2 + 30) * s),
    new THREE.MeshStandardMaterial({ color: 0x22303f, metalness: 0.4, roughness: 0.55 }),
  );
  panel.position.y = -4 * s;
  surface.add(panel);

  const place = (u: number, v: number, y: number): THREE.Vector3 => new THREE.Vector3(u * s, y * s, -v * s);

  // Five large selected-channel knobs (near/front row, v = -110).
  const selected: [string, number][] = [
    ["selected_phase", -80],
    ["selected_amplitude", -40],
    ["selected_tip", 0],
    ["selected_tilt", 40],
    ["selected_focus", 80],
  ];
  for (const [name, u] of selected) {
    const body = knobMesh(BIG_KNOB_R, KNOB_H, 0x3fb0c8);
    body.position.copy(place(u, -110, 4));
    const pointer = pointerMesh(BIG_KNOB_R, KNOB_H, 0xe6edf3);
    pointer.position.set(0, KNOB_H * 0.55 * s, BIG_KNOB_R * 0.55 * s);
    body.add(pointer);
    surface.add(body);
    knobs.set(name, { name, mesh: body, pointer, channel: null, kind: name.replace("selected_", "") as ConsoleKnob["kind"] });
  }

  // 19 channel strips: phase/amplitude pairs (10 + 9 banks).
  for (let i = 1; i <= 19; i++) {
    const ch = `CH${String(i).padStart(2, "0")}`;
    const u = i <= 10 ? -225 + (i - 1) * 35 : -207.5 + (i - 11) * 35;
    const vPhase = i <= 10 ? 100 : 25;
    const vAmp = i <= 10 ? 70 : -5;
    for (const [kind, v] of [["phase", vPhase], ["amplitude", vAmp]] as const) {
      const body = knobMesh(KNOB_R, KNOB_H, kind === "phase" ? 0xffb44d : 0x79c0ff);
      body.position.copy(place(u, v, 4));
      const pointer = pointerMesh(KNOB_R, KNOB_H, 0x0b1016);
      pointer.position.set(0, KNOB_H * 0.55 * s, KNOB_R * 0.55 * s);
      body.add(pointer);
      surface.add(body);
      knobs.set(`knob_${ch}_${kind}`, { name: `knob_${ch}_${kind}`, mesh: body, pointer, channel: ch, kind });
    }
    stripCenters.set(ch, { u, v: vPhase });
  }

  // Canonical hex selector at (u=180, v=+15), 18 mm pitch.
  const hexPitch = 18;
  const selector: [string, number, number][] = [];
  const axial: [number, number][] = [
    [-2, 2], [-1, 2], [0, 2],
    [-2, 1], [-1, 1], [0, 1], [1, 1],
    [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
    [-1, -1], [0, -1], [1, -1], [2, -1],
    [0, -2], [1, -2], [2, -2],
  ];
  axial.forEach(([q, r], i) => selector.push([`CH${String(i + 1).padStart(2, "0")}`, q + r / 2, r]));
  for (const [ch, qx, ry] of selector) {
    const button = new THREE.Mesh(
      new THREE.CylinderGeometry(3.5 * s, 3.5 * s, 3 * s, 10),
      new THREE.MeshStandardMaterial({ color: 0x394b5f, emissive: 0x101820 }),
    );
    button.position.copy(place(180 + qx * hexPitch, 15 + ry * hexPitch * 0.866, 3));
    surface.add(button);
    selectorMeshes.set(ch, button);
  }

  // Status display.
  const displayMaterial = new THREE.MeshBasicMaterial({ color: 0x0a2228 });
  const display = new THREE.Mesh(new THREE.PlaneGeometry(100 * s, 35 * s), displayMaterial);
  display.position.copy(place(180, 105, 4.5));
  display.rotation.x = -Math.PI / 2;
  surface.add(display);

  return { group, knobs, selectorMeshes, displayMaterial, stripCenters };
}

export interface ConsoleBinding {
  /** selected channel index -1 when none. */
  selected: number;
  channelIds: readonly string[];
  /** actual per-channel command state. */
  actual: readonly { piston_rad: number; amplitude: number; tiltX: number; tiltY: number; curvature_per_m: number }[];
}

function setKnobAngle(knob: ConsoleKnob, angle: number): void {
  knob.mesh.rotation.y = angle;
  knob.pointer.rotation.y = 0;
}

/** Bind all knobs/selectors to the actual controller state (R4 gate). */
export function updateConsolePanel(panel: ConsolePanel, binding: ConsoleBinding): void {
  const ampToAngle = (a: number) => (Math.max(0, Math.min(2, a)) - 1) * (Math.PI / 3);
  const selectedCh = binding.selected >= 0 ? binding.channelIds[binding.selected] : null;
  for (const [ch, mesh] of panel.selectorMeshes) {
    (mesh.material as THREE.MeshStandardMaterial).emissive.setHex(ch === selectedCh ? 0x1f6feb : 0x101820);
  }
  for (const knob of panel.knobs.values()) {
    const ch = knob.channel ?? selectedCh;
    const idx = ch ? binding.channelIds.indexOf(ch) : -1;
    const state = idx >= 0 ? binding.actual[idx] : undefined;
    if (!state) continue;
    switch (knob.kind) {
      case "phase":
        setKnobAngle(knob, state.piston_rad);
        break;
      case "amplitude":
        setKnobAngle(knob, ampToAngle(state.amplitude));
        break;
      case "tip":
        setKnobAngle(knob, state.tiltX * 1000);
        break;
      case "tilt":
        setKnobAngle(knob, state.tiltY * 1000);
        break;
      case "focus":
        setKnobAngle(knob, state.curvature_per_m * 0.05);
        break;
      default:
        break;
    }
  }
}
