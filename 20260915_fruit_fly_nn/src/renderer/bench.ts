import * as THREE from "three";
import type { ArrayConfig } from "../optics/geometry";
import type { ChannelActual } from "../optics/types";
import { phaseColor } from "./color";
import { DISPLAY } from "./scene";

export interface BenchState {
  actual: readonly ChannelActual[];
  selectedIndex: number;
  /** Beam endpoint in render coordinates (section centroid). */
  beamTarget: THREE.Vector3;
  showBeams: boolean;
}

const UP = new THREE.Vector3(0, 1, 0);

function renderPosition(config: ArrayConfig, index: number): THREE.Vector3 {
  return new THREE.Vector3(
    config.x_m[index] * DISPLAY.scale,
    config.y_m[index] * DISPLAY.scale,
    config.z_m[index] * DISPLAY.scale,
  );
}

/**
 * Optical bench: coherent seed, splitter tree, numbered branches with
 * modulator boxes, electrical command wiring, 19 emitters with phase rings, and
 * translucent beam envelopes. Selecting a channel highlights the full path in
 * both this view and the schematic (see app.ts).
 */
export class OpticalBench {
  readonly group = new THREE.Group();
  private readonly config: ArrayConfig;
  private phaseRings: THREE.Mesh[] = [];
  private beamMeshes: THREE.Mesh[] = [];
  private fibers: THREE.Line[] = [];
  private electrical: THREE.Line[] = [];
  private emitters: THREE.Mesh[] = [];
  private selectedIndex = -1;

  constructor(config: ArrayConfig) {
    this.config = config;
    this.buildSeedAndSplitter();
    this.buildBranches();
    this.buildEmitters();
    this.buildBeams();
  }

  private buildSeedAndSplitter(): void {
    const seed = new THREE.Mesh(
      new THREE.BoxGeometry(6, 3, 3),
      new THREE.MeshStandardMaterial({ color: 0x1f6feb, metalness: 0.6, roughness: 0.35 }),
    );
    seed.position.set(0, 0, -18);
    this.group.add(seed);

    const splitter = new THREE.Mesh(
      new THREE.BoxGeometry(4, 4, 3),
      new THREE.MeshStandardMaterial({ color: 0x2ea043, metalness: 0.6, roughness: 0.35 }),
    );
    splitter.position.set(0, 0, -12);
    this.group.add(splitter);

    const link = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, -16.6), new THREE.Vector3(0, 0, -13.8)]),
      new THREE.LineBasicMaterial({ color: 0x39c5cf }),
    );
    this.group.add(link);
    this.splitterPosition = new THREE.Vector3(0, 0, -12);
  }

  private splitterPosition = new THREE.Vector3();

  private buildBranches(): void {
    const n = this.config.channelIds.length;
    for (let i = 0; i < n; i++) {
      const p = renderPosition(this.config, i);
      const fiberGeom = new THREE.BufferGeometry().setFromPoints([
        this.splitterPosition.clone(),
        new THREE.Vector3(p.x, p.y, -2.5),
      ]);
      const fiber = new THREE.Line(
        fiberGeom,
        new THREE.LineBasicMaterial({ color: 0x1b6b73, transparent: true, opacity: 0.7 }),
      );
      this.group.add(fiber);
      this.fibers.push(fiber);

      const modulator = new THREE.Mesh(
        new THREE.BoxGeometry(1.0, 1.0, 1.4),
        new THREE.MeshStandardMaterial({ color: 0x30363d, metalness: 0.7, roughness: 0.4 }),
      );
      modulator.position.lerpVectors(this.splitterPosition, p, 0.55);
      this.group.add(modulator);

      const consolePos = new THREE.Vector3(0, -12, -14);
      const elecGeom = new THREE.BufferGeometry().setFromPoints([consolePos, modulator.position.clone()]);
      const elec = new THREE.Line(
        elecGeom,
        new THREE.LineBasicMaterial({ color: 0xffa657, transparent: true, opacity: 0.35 }),
      );
      this.group.add(elec);
      this.electrical.push(elec);

      const connector = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p.x, p.y, -2.5), p.clone()]),
        new THREE.LineBasicMaterial({ color: 0x39c5cf }),
      );
      this.group.add(connector);
    }
  }

  private buildEmitters(): void {
    const n = this.config.channelIds.length;
    for (let i = 0; i < n; i++) {
      const p = renderPosition(this.config, i);
      const body = new THREE.Mesh(
        new THREE.CylinderGeometry(0.7, 0.8, 1.2, 16),
        new THREE.MeshStandardMaterial({ color: 0x8b949e, metalness: 0.85, roughness: 0.3 }),
      );
      body.position.copy(p);
      body.rotation.x = Math.PI / 2;
      this.group.add(body);
      this.emitters.push(body);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(1.1, 0.18, 8, 24),
        new THREE.MeshBasicMaterial({ color: 0xff8800 }),
      );
      ring.position.set(p.x, p.y, p.z + 0.9);
      this.group.add(ring);
      this.phaseRings.push(ring);
    }
  }

  private buildBeams(): void {
    const n = this.config.channelIds.length;
    const geometry = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true);
    for (let i = 0; i < n; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x39c5cf,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      this.group.add(mesh);
      this.beamMeshes.push(mesh);
    }
  }

  update(state: BenchState): void {
    this.selectedIndex = state.selectedIndex;
    const n = this.config.channelIds.length;
    for (let i = 0; i < n; i++) {
      const s = state.actual[i];
      const color = new THREE.Color(...phaseColor(s.piston_rad));
      (this.phaseRings[i].material as THREE.MeshBasicMaterial).color = color;
      const intensity = s.enabled ? Math.min(1, s.amplitude) : 0.05;
      (this.phaseRings[i].material as THREE.MeshBasicMaterial).opacity = 0.3 + 0.7 * intensity;
      this.phaseRings[i].visible = true;

      const selected = state.selectedIndex === i;
      this.emitters[i].scale.setScalar(selected ? 1.6 : 1);
      (this.emitters[i].material as THREE.MeshStandardMaterial).emissive = selected
        ? new THREE.Color(0x224466)
        : new THREE.Color(0x000000);
    }

    // Beams: from each emitter to the section/beam endpoint.
    const start = new THREE.Vector3();
    const dir = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      const mesh = this.beamMeshes[i];
      const s = state.actual[i];
      mesh.visible = state.showBeams && s.enabled && s.amplitude > 0;
      if (!mesh.visible) continue;
      start.copy(renderPosition(this.config, i));
      dir.copy(state.beamTarget).sub(start);
      const len = dir.length();
      mesh.position.copy(start).addScaledVector(dir, 0.5);
      mesh.scale.set(0.32, len, 0.32);
      mesh.quaternion.setFromUnitVectors(UP, dir.normalize());
      const color = new THREE.Color(...phaseColor(s.piston_rad));
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.color.copy(color).lerp(new THREE.Color(0xffffff), 0.35);
      mat.opacity = 0.1 + 0.08 * (state.selectedIndex === i ? 3 : 1);
    }
  }

  setPathHighlight(selected: number): void {
    this.fibers.forEach((f, i) => {
      const mat = f.material as THREE.LineBasicMaterial;
      const on = selected < 0 || selected === i;
      mat.opacity = on ? 0.85 : 0.12;
      mat.color.set(selected === i ? 0x7ee787 : 0x1b6b73);
    });
    this.electrical.forEach((e, i) => {
      const mat = e.material as THREE.LineBasicMaterial;
      mat.opacity = selected < 0 || selected === i ? 0.5 : 0.08;
    });
    this.selectedIndex = selected;
  }

  get highlightedIndex(): number {
    return this.selectedIndex;
  }
}
