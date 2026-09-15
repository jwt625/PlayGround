import * as THREE from "three";
import type { ArrayConfig } from "../optics/geometry";
import type { ChannelActual } from "../optics/types";
import { farFieldAngular } from "../optics/gaussian";
import { intensityColor, logNormalize } from "./color";
import { DISPLAY } from "./scene";

/**
 * Forward-hemisphere far-field dome. Colors encode simulated far-field angular
 * power density dP/dOmega (linear intensities, log-normalized for display).
 * A separate marker shows the power-weighted centroid; it can fall between
 * lobes and is never snapped to a peak.
 */
export class FarFieldDome {
  readonly group = new THREE.Group();
  private readonly mesh: THREE.Mesh;
  private readonly colors: Float32Array;
  private readonly centroid: THREE.Mesh;
  private readonly peak: THREE.Mesh;
  private readonly positions: THREE.Vector3[] = [];

  constructor(widthSegments = 64, heightSegments = 32) {
    const geom = new THREE.SphereGeometry(DISPLAY.domeRadius, widthSegments, heightSegments, 0, Math.PI * 2, 0, Math.PI / 2);
    geom.rotateX(Math.PI / 2);
    this.colors = new Float32Array(geom.attributes.position.count * 3);
    geom.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    const mat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(geom, mat);
    this.group.add(this.mesh);

    const pos = geom.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      this.positions.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)).normalize());
    }

    this.centroid = new THREE.Mesh(
      new THREE.SphereGeometry(1.1, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    this.group.add(this.centroid);
    this.peak = new THREE.Mesh(
      new THREE.RingGeometry(1.6, 2.1, 16),
      new THREE.MeshBasicMaterial({ color: 0xff7b72, side: THREE.DoubleSide }),
    );
    this.peak.lookAt(new THREE.Vector3());
    this.group.add(this.peak);
  }

  update(
    config: ArrayConfig,
    actual: readonly ChannelActual[],
    centroidDir: { sx: number; sy: number; sz: number },
    peakDir: { sx: number; sy: number; sz: number },
  ): void {
    let max = 0;
    const values = new Float32Array(this.positions.length);
    for (let i = 0; i < this.positions.length; i++) {
      const d = this.positions[i];
      const u = farFieldAngular(config, actual, d.x, d.y);
      const intensity = u.re * u.re + u.im * u.im;
      values[i] = intensity;
      max = Math.max(max, intensity);
    }
    for (let i = 0; i < this.positions.length; i++) {
      const [r, g, b] = intensityColor(logNormalize(values[i], max, 5));
      this.colors[i * 3] = r;
      this.colors[i * 3 + 1] = g;
      this.colors[i * 3 + 2] = b;
    }
    this.mesh.geometry.attributes.color.needsUpdate = true;

    this.centroid.position.set(centroidDir.sx, centroidDir.sy, centroidDir.sz).multiplyScalar(DISPLAY.domeRadius);
    this.peak.position.set(peakDir.sx, peakDir.sy, peakDir.sz).multiplyScalar(DISPLAY.domeRadius);
    this.peak.lookAt(new THREE.Vector3());
  }
}
