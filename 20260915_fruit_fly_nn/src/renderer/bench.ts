import * as THREE from "three";
import type { ArrayConfig } from "../optics/geometry";
import type { ChannelActual } from "../optics/types";
import { launchQ } from "../optics/gaussian";
import { DISPLAY } from "./scene";

export interface BenchState {
  actual: readonly ChannelActual[];
  selectedIndex: number;
  /** Beam endpoint in render coordinates (section centroid). */
  beamTarget: THREE.Vector3;
  showBeams: boolean;
  beamRange_m?: number;
}

/**
 * Field-derived beam-envelope visualization for the 19-channel array.
 *
 * The physical seed/splitter/modulator/emitter enclosures, fiber and command
 * runs are drawn by `HardwareScene` from the CBC assembly specification; the
 * earlier code-native placeholder boxes were retired so the scene contains only
 * objects that correspond to the schematic. This layer keeps the translucent
 * Gaussian envelopes computed from the actual per-channel command state.
 */
export class OpticalBench {
  readonly group = new THREE.Group();
  private readonly config: ArrayConfig;
  private beamMeshes: THREE.Mesh[] = [];
  private beamTemplate!: Float32Array;
  private selectedIndex = -1;

  constructor(config: ArrayConfig) {
    this.config = config;
    this.buildBeams();
  }

  private buildBeams(): void {
    const n = this.config.channelIds.length;
    const geometry = new THREE.CylinderGeometry(1, 1, 1, 12, 32, true);
    this.beamTemplate = Float32Array.from(geometry.attributes.position.array);
    for (let i = 0; i < n; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x39c5cf,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry.clone(), mat);
      this.group.add(mesh);
      this.beamMeshes.push(mesh);
    }
  }

  update(state: BenchState): void {
    this.selectedIndex = state.selectedIndex;
    const n = this.config.channelIds.length;

    // Gaussian 1/e² envelopes from actual channel direction/curvature. Axial
    // distance is compressed for presentation; transverse scale follows DISPLAY.
    // Envelope transparency is illustrative, not a volume interference integral.
    const axis = new THREE.Vector3(), u = new THREE.Vector3(), v = new THREE.Vector3();
    const range = state.beamRange_m ?? 1;
    for (let i = 0; i < n; i++) {
      const mesh = this.beamMeshes[i], channel = state.actual[i];
      mesh.visible = state.showBeams && channel.enabled && channel.amplitude > 0;
      if (!mesh.visible) continue;
      axis.set(channel.tiltX, channel.tiltY, Math.sqrt(Math.max(0, 1 - channel.tiltX ** 2 - channel.tiltY ** 2)));
      u.crossVectors(new THREE.Vector3(1, 0, 0), axis).normalize();
      v.crossVectors(axis, u).normalize();
      const q = launchQ(this.config, channel.curvature_per_m);
      const positions = mesh.geometry.attributes.position;
      const tf = DISPLAY.beamTighten;
      for (let j = 0; j < positions.count; j++) {
        const t = (this.beamTemplate[j * 3 + 1] + 0.5) * range;
        const width = Math.sqrt(this.config.wavelength_m * ((q.re + t) ** 2 + q.im ** 2) / (Math.PI * q.im));
        const a = this.beamTemplate[j * 3] * width, b = this.beamTemplate[j * 3 + 2] * width;
        positions.setXYZ(j,
          this.config.x_m[i] * DISPLAY.scale + (t * axis.x + a * u.x + b * v.x) * DISPLAY.scale * tf,
          this.config.y_m[i] * DISPLAY.scale + (t * axis.y + a * u.y + b * v.y) * DISPLAY.scale * tf,
          this.config.z_m[i] * DISPLAY.targetDistance + t * axis.z * DISPLAY.targetDistance + (a * u.z + b * v.z) * DISPLAY.targetDistance * tf);
      }
      positions.needsUpdate = true; mesh.geometry.computeBoundingSphere();
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.color.set(0x45ded0);
      mat.opacity = state.selectedIndex === i ? 0.2 : state.selectedIndex < 0 ? 0.025 : 0.008;
    }
  }

  setPathHighlight(selected: number): void {
    this.selectedIndex = selected;
  }

  get highlightedIndex(): number {
    return this.selectedIndex;
  }
}
