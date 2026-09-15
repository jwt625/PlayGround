import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { DISPLAY } from "./scene";

/**
 * Learner and target fly instances using the converted articulated FlyBody GLB
 * (assets/generated/flybody/flybody-articulated.glb, meters, Y-up).
 *
 * - The learner (neural controller) sits fixed beside the console in its rest
 *   pose and does NOT flap: it is not an animated locomotion demo.
 * - The target fly carries the illustrative wing clip and moves across the
 *   displayed target plane as the commanded target changes.
 *
 * The target plane is axially compressed for presentation while its transverse
 * extent uses DISPLAY.scale, matching the beam envelopes in bench.ts. Target
 * motion therefore stays where the beams converge.
 */
const FLY_URL = new URL("../../assets/generated/flybody/flybody-articulated.glb", import.meta.url).href;
const TARGET_RANGE_M = 1; // fixed-distance training plane
const FLY_DISPLAY_LENGTH = 16;
const LEARNER_POSITION = new THREE.Vector3(-30, -18, -6);

export class FlyActors {
  readonly group = new THREE.Group();
  private learner: THREE.Object3D;
  private target: THREE.Object3D;
  private targetMixer: THREE.AnimationMixer | null = null;
  private clips: THREE.AnimationClip[] = [];
  private previousTarget = new THREE.Vector3();
  loaded = false;
  error: string | null = null;

  constructor() {
    this.learner = makePlaceholder(0x79c0ff);
    this.target = makePlaceholder(0xffa657);
    this.learner.position.copy(LEARNER_POSITION);
    this.target.position.set(0, 0, DISPLAY.targetDistance);
    this.group.add(this.learner, this.target);
    void this.load();
  }

  private async load(): Promise<void> {
    try {
      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(FLY_URL);
      const root = gltf.scene;
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      box.getSize(size);
      const longest = Math.max(size.x, size.y, size.z) || 1;
      const scale = FLY_DISPLAY_LENGTH / longest;
      this.clips = gltf.animations;

      const build = (): THREE.Object3D => {
        const obj = skeletonClone(root);
        obj.scale.setScalar(scale);
        return obj;
      };

      this.group.remove(this.learner, this.target);
      this.learner = build();
      this.target = build();
      this.learner.position.copy(LEARNER_POSITION);
      this.learner.rotation.y = Math.PI / 2; // face the array/console
      this.target.position.set(0, 0, DISPLAY.targetDistance);
      this.previousTarget.copy(this.target.position);
      this.group.add(this.learner, this.target);

      // Only the target flaps. The learner keeps the GLB rest pose.
      if (this.clips.length > 0) {
        this.targetMixer = new THREE.AnimationMixer(this.target);
        this.clips.forEach((clip) => this.targetMixer!.clipAction(clip).play());
      }
      this.loaded = true;
    } catch (err) {
      this.error = String(err);
    }
  }

  update(targetDir: { sx: number; sy: number; sz: number }, dt_s: number): void {
    const tx = targetDir.sx * TARGET_RANGE_M * DISPLAY.scale;
    const ty = targetDir.sy * TARGET_RANGE_M * DISPLAY.scale;
    const tz = DISPLAY.targetDistance * TARGET_RANGE_M;
    this.target.position.set(tx, ty, tz);

    // Face the direction of travel (model forward treated as +Z).
    const dx = tx - this.previousTarget.x;
    const dz = tz - this.previousTarget.z;
    if (dx * dx + dz * dz > 1e-6) {
      this.target.rotation.y = Math.atan2(dx, dz);
    }
    this.previousTarget.set(tx, ty, tz);

    this.targetMixer?.update(dt_s);
  }

  /** Display-space position of the moving target fly (for tests/telemetry). */
  get targetDisplayPosition(): THREE.Vector3 {
    return this.target.position;
  }
}

function makePlaceholder(color: number): THREE.Object3D {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(1.6, 12, 10),
    new THREE.MeshStandardMaterial({ color, metalness: 0.3, roughness: 0.6 }),
  );
  body.scale.set(2.2, 1, 1);
  g.add(body);
  const wing = new THREE.Mesh(
    new THREE.CircleGeometry(1.4, 3),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5, side: THREE.DoubleSide }),
  );
  wing.position.set(0, 0.8, 0);
  g.add(wing);
  return g;
}
