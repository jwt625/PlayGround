import * as THREE from "three";
import type { ArrayConfig } from "../optics/geometry";
import type { ChannelActual } from "../optics/types";
import { totalFieldFast } from "../optics/gaussian";
import { intensityColor, logNormalize } from "./color";
import { DISPLAY } from "./scene";

export interface SectionOptions {
  size_m: number;
  samples: number;
  range_m: number;
}

/**
 * Movable transverse intensity section. Samples come directly from the complex
 * field evaluator at actual world coordinates (never a stretched dome texture),
 * so the section is a real physical measurement of the simulated field.
 */
export class MeasurementSection {
  readonly group = new THREE.Group();
  private readonly mesh: THREE.Mesh;
  private readonly texture: THREE.DataTexture;
  private readonly data: Uint8Array;
  private readonly targetMarker: THREE.Mesh;
  private readonly centroidMarker: THREE.Mesh;
  private readonly options: SectionOptions;
  private center = new THREE.Vector3();

  constructor(options: SectionOptions) {
    this.options = options;
    const n = options.samples;
    this.data = new Uint8Array(n * n * 4);
    this.texture = new THREE.DataTexture(this.data, n, n, THREE.RGBAFormat);
    this.texture.needsUpdate = true;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;

    const size = options.size_m * DISPLAY.scale;
    const geom = new THREE.PlaneGeometry(size, size);
    const mat = new THREE.MeshBasicMaterial({
      map: this.texture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(geom, mat);
    this.group.add(this.mesh);

    this.targetMarker = new THREE.Mesh(
      new THREE.RingGeometry(2.6, 3.4, 24),
      new THREE.MeshBasicMaterial({ color: 0xff7b72, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }),
    );
    this.group.add(this.targetMarker);
    this.centroidMarker = new THREE.Mesh(
      new THREE.RingGeometry(1.4, 1.9, 20),
      new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }),
    );
    this.group.add(this.centroidMarker);
  }

  update(
    config: ArrayConfig,
    actual: readonly ChannelActual[],
    dir: { sx: number; sy: number; sz: number },
    range_m: number,
    targetPos: { x: number; y: number; z: number },
  ): void {
    // Stable orthonormal basis around the section normal.
    const n = new THREE.Vector3(dir.sx, dir.sy, dir.sz).normalize();
    const ref = Math.abs(n.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(ref, n).normalize();
    const v = new THREE.Vector3().crossVectors(n, u).normalize();

    this.center.copy(n).multiplyScalar(range_m);
    this.mesh.position.copy(n).multiplyScalar(DISPLAY.targetDistance * range_m);
    const basisMatrix = new THREE.Matrix4().makeBasis(u, v, n);
    this.mesh.quaternion.setFromRotationMatrix(basisMatrix);
    this.targetMarker.quaternion.copy(this.mesh.quaternion);
    this.centroidMarker.quaternion.copy(this.mesh.quaternion);

    // Sample the physical field on the section plane.
    const m = this.options.samples;
    const half = this.options.size_m / 2;
    const intensities = new Float32Array(m * m);
    let max = 0;
    for (let j = 0; j < m; j++) {
      const b = -half + (this.options.size_m * j) / (m - 1);
      for (let i = 0; i < m; i++) {
        const a = -half + (this.options.size_m * i) / (m - 1);
        const x = this.center.x + u.x * a + v.x * b;
        const y = this.center.y + u.y * a + v.y * b;
        const z = this.center.z + u.z * a + v.z * b;
        const field = totalFieldFast(config, actual, x, y, z);
        const intensity = field.re * field.re + field.im * field.im;
        intensities[j * m + i] = intensity;
        max = Math.max(max, intensity);
      }
    }
    for (let p = 0; p < m * m; p++) {
      const [r, g, b] = intensityColor(logNormalize(intensities[p], max, 4));
      this.data[p * 4] = Math.round(r * 255);
      this.data[p * 4 + 1] = Math.round(g * 255);
      this.data[p * 4 + 2] = Math.round(b * 255);
      this.data[p * 4 + 3] = 255;
    }
    this.texture.needsUpdate = true;

    // Markers: target and beam-centroid projection onto the section plane.
    this.placeMarker(this.targetMarker, targetPos, u, v);
    this.placeMarker(this.centroidMarker, {
      x: n.x * range_m,
      y: n.y * range_m,
      z: n.z * range_m,
    }, u, v);
  }

  private placeMarker(
    marker: THREE.Mesh,
    world: { x: number; y: number; z: number },
    u: THREE.Vector3,
    v: THREE.Vector3,
  ): void {
    const offset = new THREE.Vector3(world.x, world.y, world.z).sub(this.center);
    const a = offset.dot(u);
    const b = offset.dot(v);
    marker.position.copy(this.mesh.position).addScaledVector(u, a * DISPLAY.scale).addScaledVector(v, b * DISPLAY.scale);
    const inPlane = Math.hypot(a, b) <= this.options.size_m * 1.5;
    marker.visible = inPlane;
  }
}
