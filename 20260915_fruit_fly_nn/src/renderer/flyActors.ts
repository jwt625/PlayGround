import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";

/**
 * Learner and target fly instances using the converted articulated FlyBody GLB
 * (assets/generated/flybody/flybody-articulated.glb, meters, Y-up). The wing
 * clip is an illustrative display loop, not measured flight kinematics.
 *
 * Loading is non-blocking; a procedural placeholder is shown until the GLB is
 * ready so the simulation never depends on asset availability.
 */
const FLY_URL = new URL("../../assets/generated/flybody/flybody-articulated.glb", import.meta.url).href;
const TARGET_DISTANCE = 70;

export class FlyActors {
  readonly group = new THREE.Group();
  private learner: THREE.Object3D;
  private target: THREE.Object3D;
  private mixers: THREE.AnimationMixer[] = [];
  private clips: THREE.AnimationClip[] = [];
  loaded = false;
  error: string | null = null;

  constructor() {
    this.learner = makePlaceholder(0x79c0ff);
    this.target = makePlaceholder(0xffa657);
    this.learner.position.set(-26, -16, -6);
    this.target.position.set(0, 0, TARGET_DISTANCE);
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
      const desired = 28; // display units
      const scale = desired / longest;
      this.clips = gltf.animations;

      const build = (): THREE.Object3D => {
        const obj = skeletonClone(root);
        obj.scale.setScalar(scale);
        return obj;
      };

      this.group.remove(this.learner, this.target);
      this.learner = build();
      this.target = build();
      this.learner.position.set(-26, -16, -6);
      this.target.position.set(0, 0, TARGET_DISTANCE);
      this.group.add(this.learner, this.target);

      for (const obj of [this.learner, this.target]) {
        if (this.clips.length > 0) {
          const mixer = new THREE.AnimationMixer(obj);
          this.clips.forEach((clip) => mixer.clipAction(clip).play());
          this.mixers.push(mixer);
        }
      }
      this.loaded = true;
    } catch (err) {
      this.error = String(err);
    }
  }

  update(targetDir: { sx: number; sy: number; sz: number }, dt_s: number): void {
    this.target.position
      .set(targetDir.sx, targetDir.sy, targetDir.sz)
      .multiplyScalar(TARGET_DISTANCE);
    // Gentle time-scaled wing motion; not tied to physics.
    for (let i = 0; i < this.mixers.length; i++) {
      this.mixers[i].update(dt_s * (i === 1 ? 1.35 : 1));
    }
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
