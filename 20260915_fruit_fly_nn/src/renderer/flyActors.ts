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
// Console-front operator stance (mechanical mm registered like hardwareScene).
const HW_S = 3 / 65;
const OPERATOR_POSITION = new THREE.Vector3(-1220 * HW_S, (20 - 300) * HW_S, -180 * HW_S);

export class FlyActors {
  readonly group = new THREE.Group();
  private learner: THREE.Object3D;
  private target: THREE.Object3D;
  private targetMixer: THREE.AnimationMixer | null = null;
  private clips: THREE.AnimationClip[] = [];
  private previousTarget = new THREE.Vector3();
  private forelegJoints: { object: THREE.Object3D; rest: THREE.Euler; role: "femur" | "tibia" | "tarsus"; side: "left" | "right" }[] = [];
  private operatorMode = false;
  private gesturePhase = -1;
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

      this.collectForelegJoints();
      if (this.operatorMode) this.applyOperatorPlacement();

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

  private collectForelegJoints(): void {
    this.forelegJoints = [];
    this.learner.traverse((o) => {
      const m = /^(femur|tibia|tarsus)_T1_(left|right)$/.exec(o.name);
      if (!m) return;
      this.forelegJoints.push({
        object: o,
        rest: o.rotation.clone(),
        role: m[1] as "femur" | "tibia" | "tarsus",
        side: m[2] as "left" | "right",
      });
    });
  }

  private applyOperatorPlacement(): void {
    this.learner.position.copy(OPERATOR_POSITION);
    this.learner.rotation.set(0, Math.PI / 2, 0); // native +X forward -> world -Z
  }

  /** Place the learner as the console operator (Pass 3 / R5). */
  setOperatorMode(on: boolean): void {
    this.operatorMode = on;
    if (this.loaded && on) this.applyOperatorPlacement();
  }

  /** Start a staged foreleg reach/contact/retract gesture. */
  startGesture(): void {
    this.gesturePhase = 0;
  }

  private applyGesture(phase: number): void {
    // Timeline (spec §8.3): lift, reach, contact, turn, retract.
    const seg = (a: number, b: number) => Math.max(0, Math.min(1, (phase - a) / (b - a)));
    const lift = seg(0, 0.12);
    const reach = seg(0.12, 0.36);
    const contact = seg(0.36, 0.45);
    const turn = seg(0.45, 0.75);
    const retract = seg(0.75, 1);
    const amount = Math.min(lift + reach, 1) * (contact > 0 ? 1 : 1) * (1 - retract);
    const turnArc = Math.sin(turn * Math.PI) * 0.25;
    for (const joint of this.forelegJoints) {
      const sign = joint.side === "left" ? 1 : -0.5; // left leg operates, right assists
      const factor = joint.role === "femur" ? 0.9 : joint.role === "tibia" ? 1.1 : 0.6;
      joint.object.rotation.set(
        joint.rest.x + sign * amount * factor * 0.9,
        joint.rest.y + sign * turnArc * factor,
        joint.rest.z + sign * amount * factor * 0.3,
      );
    }
  }

  private resetGesture(): void {
    for (const joint of this.forelegJoints) joint.object.rotation.copy(joint.rest);
  }

  get forelegJointCount(): number {
    return this.forelegJoints.length;
  }

  get gestureActive(): boolean {
    return this.gesturePhase >= 0;
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

    if (this.gesturePhase >= 0) {
      this.gesturePhase += dt_s / 1.65;
      if (this.gesturePhase >= 1) {
        this.gesturePhase = -1;
        this.resetGesture();
      } else {
        this.applyGesture(this.gesturePhase);
      }
    }
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
