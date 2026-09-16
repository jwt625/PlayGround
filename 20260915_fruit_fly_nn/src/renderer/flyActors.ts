import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { DISPLAY } from "./scene";
import { APERTURE_CENTER_MM } from "./hardware/assemblySpec";

/**
 * Learner and target fly instances using the converted articulated FlyBody GLB
 * (assets/generated/flybody/flybody-articulated.glb, meters, Y-up, +X forward).
 *
 * - The target fly carries the illustrative wing clip and moves across the
 *   displayed target plane as the commanded target changes. Its head points
 *   along the trajectory tangent and its dorsal axis stays as close to world
 *   up as the tangent allows, which fully constrains its orientation.
 * - The learner (neural controller) sits fixed in front of the console panel in
 *   its rest pose and does NOT flap: it is not an animated locomotion demo.
 *
 * The target plane is axially compressed for presentation while its transverse
 * extent uses DISPLAY.scale, matching the beam envelopes in bench.ts. Target
 * motion therefore stays where the beams converge.
 */
const FLY_URL = new URL("../../assets/generated/flybody/flybody-articulated.glb", import.meta.url).href;
const TARGET_RANGE_M = 1; // fixed-distance training plane
const FLY_TARGET_LENGTH = 16;
const FLY_OPERATOR_LENGTH = 8;
const LEARNER_POSITION = new THREE.Vector3(-30, -18, -6);
// Uniform lengthening of the foreleg chain so the operator can actually reach
// the near control row (the articulated model's legs are short at this display
// scale). This is the explicit "stretch the arms to reach" display treatment.
const FORELEG_STRETCH = 1.9;
// Mechanical mm -> scene units, the same registration hardwareScene uses.
const HW_S = 3 / 65;
const PLATFORM_TOP_MM_Y = 20;
// OP-PLATFORM supporting-foot centroid (assembly spec §8.3), moved toward the
// panel so the articulated forelegs can actually reach the near control row.
const OPERATOR_STANCE_MM = { x: -1220, z: -40 };
const KNOB_DWELL_S = 0.03;

export class FlyActors {
  readonly group = new THREE.Group();
  private learner: THREE.Object3D;
  private target: THREE.Object3D;
  private targetMixer: THREE.AnimationMixer | null = null;
  private clips: THREE.AnimationClip[] = [];
  private previousTarget = new THREE.Vector3();
  private forelegJoints: {
    object: THREE.Object3D;
    rest: THREE.Euler;
    restQuat: THREE.Quaternion;
    previousQuat: THREE.Quaternion;
    role: "femur" | "tibia" | "tarsus";
    side: "left" | "right";
  }[] = [];
  private operatorMode = false;
  private gesturePhase = -1;
  private operatorBase: THREE.Vector3 | null = null;
  private readonly knobTargets = new Map<string, { position: THREE.Vector3; amount: number; lastSeen: number }>();
  private knobOrder: string[] = [];
  private tapPhase = 0;
  private knobCursor = 0;
  private dwellTimer = 0;
  private trailsVisible = true;
  private readonly armTrails = new Map<"left" | "right", { line: THREE.Line; positions: Float32Array; history: THREE.Vector3[]; geometry: THREE.BufferGeometry; material: THREE.LineBasicMaterial }>();
  /** Largest foreleg joint rotation this frame, radians (telemetry). */
  forelegMotion = 0;
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
      this.clips = gltf.animations;

      const build = (length: number): THREE.Object3D => {
        const obj = skeletonClone(root);
        obj.scale.setScalar(length / longest);
        return obj;
      };

      this.group.remove(this.learner, this.target);
      this.learner = build(FLY_OPERATOR_LENGTH);
      this.target = build(FLY_TARGET_LENGTH);
      this.learner.rotation.y = Math.PI / 2; // native +X forward -> world -Z
      this.target.position.set(0, 0, DISPLAY.targetDistance);
      this.previousTarget.copy(this.target.position);
      this.group.add(this.learner, this.target);

      this.collectForelegJoints();
      if (this.operatorMode) this.applyOperatorPlacement();
      this.buildArmTrails();

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
      const m = /^(coxa|femur|tibia|tarsus\d*|claw|tarsal_claw)_T1_(left|right)$/.exec(o.name);
      if (!m) return;
      const segment = m[1];
      const role: "femur" | "tibia" | "tarsus" =
        segment === "coxa" || segment === "femur" ? "femur" : segment === "tibia" ? "tibia" : "tarsus";
      const restQuat = new THREE.Quaternion().setFromEuler(o.rotation);
      this.forelegJoints.push({
        object: o,
        rest: o.rotation.clone(),
        restQuat,
        previousQuat: restQuat.clone(),
        role,
        side: m[2] as "left" | "right",
      });
    });
    // Lengthen each foreleg chain at its root so the limbs can span the near row.
    for (const side of ["left", "right"] as const) {
      const root = this.forelegJoints.find((j) => j.side === side);
      if (root) root.object.scale.multiplyScalar(FORELEG_STRETCH);
    }
  }

  /** The terminal foreleg segment (claw) whose world position is the tip. */
  private tipJoint(side: "left" | "right"): THREE.Object3D | undefined {
    const onSide = this.forelegJoints.filter((j) => j.side === side);
    return (onSide.find((j) => j.object.name.includes("claw")) ?? onSide.at(-1))?.object;
  }

  /**
   * Put the learner in front of the console panel on the operator platform,
   * facing the controls. Placement is derived from the panel/platform stance
   * (OP-PLATFORM top Y=20, centerline X=-1220) and the fly's own bounding box
   * so its feet rest on the platform surface. The stance is pulled toward the
   * panel (Z=-40 instead of +30 mm) so the forelegs can reach the near row.
   */
  private applyOperatorPlacement(): void {
    this.learner.rotation.set(0, Math.PI / 2, 0); // native +X forward -> world -Z
    this.learner.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.learner);
    const platformTopY = (PLATFORM_TOP_MM_Y - APERTURE_CENTER_MM.y) * HW_S;
    this.learner.position.set(
      (OPERATOR_STANCE_MM.x - APERTURE_CENTER_MM.x) * HW_S,
      platformTopY - box.min.y,
      (OPERATOR_STANCE_MM.z - APERTURE_CENTER_MM.z) * HW_S,
    );
    this.operatorBase = this.learner.position.clone();
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

  /**
   * Point the two foreleg chains at the knobs that are actually changing. A
   * stable ordered list is kept so the legs travel from knob to knob instead of
   * jittering when the per-frame ranking fluctuates.
   */
  setForelegTargets(targets: readonly { channel: string; position: THREE.Vector3; rate: number }[], now_s = performance.now() / 1000): void {
    for (const t of targets) {
      this.knobTargets.set(t.channel, {
        position: t.position.clone(),
        amount: THREE.MathUtils.clamp(t.rate * 20, 0.3, 1),
        lastSeen: now_s,
      });
      if (!this.knobOrder.includes(t.channel)) this.knobOrder.push(t.channel);
    }
    // Drop knobs that have not been active for a while.
    this.knobOrder = this.knobOrder.filter((ch) => now_s - (this.knobTargets.get(ch)?.lastSeen ?? 0) < 2.5);
  }

  /**
   * Solve one foreleg chain to put its claw on a world-space target using cyclic
   * coordinate descent. CCD bends the limb naturally and extends it when the
   * target is far, instead of only rotating single joints.
   */
  private aimChain(side: "left" | "right", target: THREE.Vector3, _amount: number): void {
    const chain = this.forelegJoints.filter((j) => j.side === side);
    const tip = this.tipJoint(side);
    if (chain.length === 0 || !tip) return;
    const tipPos = new THREE.Vector3();
    const jointPos = new THREE.Vector3();
    const toTip = new THREE.Vector3();
    const toTarget = new THREE.Vector3();
    const qWorld = new THREE.Quaternion();
    const parentQ = new THREE.Quaternion();
    const worldQ = new THREE.Quaternion();
    for (let iter = 0; iter < 6; iter++) {
      for (let i = chain.length - 1; i >= 0; i--) {
        const joint = chain[i].object;
        const parent = joint.parent;
        if (!parent) continue;
        joint.updateWorldMatrix(true, false);
        joint.getWorldPosition(jointPos);
        tip.getWorldPosition(tipPos);
        toTip.copy(tipPos).sub(jointPos);
        toTarget.copy(target).sub(jointPos);
        if (toTip.lengthSq() < 1e-9 || toTarget.lengthSq() < 1e-9) continue;
        qWorld.setFromUnitVectors(toTip.normalize(), toTarget.normalize());
        parent.getWorldQuaternion(parentQ);
        joint.getWorldQuaternion(worldQ);
        worldQ.premultiply(qWorld);
        joint.quaternion.copy(parentQ.invert().multiply(worldQ));
        joint.updateWorldMatrix(false, true);
      }
    }
  }

  /** Measure foreleg joint speed for telemetry. */
  private finishForelegs(): void {
    let motion = 0;
    for (const joint of this.forelegJoints) {
      const change = joint.previousQuat.angleTo(joint.object.quaternion);
      joint.previousQuat.copy(joint.object.quaternion);
      if (change > motion) motion = change;
    }
    this.forelegMotion = motion;
  }

  /**
   * Orient an object so its native forward (+X) points along `velocity` and its
   * native dorsal axis (+Y) stays as close to world up as possible. The two
   * constraints determine the full rotation (no roll ambiguity).
   */
  private orientAlongTangent(object: THREE.Object3D, velocity: THREE.Vector3, dt: number): void {
    const desired = orientationFromTangent(velocity);
    if (!desired) return;
    if (dt > 0) object.quaternion.slerp(desired, Math.min(1, dt * 8));
    else object.quaternion.copy(desired);
  }

  get forelegJointCount(): number {
    return this.forelegJoints.length;
  }

  get gestureActive(): boolean {
    return this.gesturePhase >= 0;
  }

  update(targetDir: { sx: number; sy: number; sz: number }, dt_s: number): void {
    const tx = targetDir.sx * TARGET_RANGE_M * DISPLAY.scale * DISPLAY.beamTighten;
    const ty = targetDir.sy * TARGET_RANGE_M * DISPLAY.scale * DISPLAY.beamTighten;
    const tz = DISPLAY.targetDistance * TARGET_RANGE_M;
    this.target.position.set(tx, ty, tz);

    // Head follows the trajectory tangent; dorsal axis stays along world up.
    const velocity = new THREE.Vector3(tx - this.previousTarget.x, ty - this.previousTarget.y, tz - this.previousTarget.z);
    this.orientAlongTangent(this.target, velocity, dt_s);
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
    } else if (this.knobOrder.length > 0 && this.forelegJoints.length > 0) {
      // Both forelegs work the controls: they reach between the knobs that are
      // actually turning, dwelling on each, with a small operating scrub. The
      // chain is aimed joint-by-joint at the knob, which extends it toward the
      // target (maximum reach) rather than just shaking in place.
      const n = this.knobOrder.length;
      this.dwellTimer += dt_s;
      if (this.dwellTimer >= KNOB_DWELL_S) {
        this.dwellTimer -= KNOB_DWELL_S;
        this.knobCursor = (this.knobCursor + 1) % n;
      }
      const left = this.knobTargets.get(this.knobOrder[this.knobCursor % n])!;
      const right = this.knobTargets.get(this.knobOrder[(this.knobCursor + 1) % n])!;
      this.tapPhase += dt_s * (120 + 240 * Math.max(left.amount, right.amount));
      this.aimChain("left", this.scrubPosition(left, 0), left.amount);
      this.aimChain("right", this.scrubPosition(right, Math.PI), right.amount);
      this.applyOperatingStance(left, right);
    } else if (this.operatorMode && this.operatorBase) {
      // Idle: drift back to the home stance in front of the panel.
      this.learner.position.lerp(this.operatorBase, 0.08);
    }

    this.finishForelegs();
    this.updateArmTrails();
  }

  /** A small circular "operating" offset around the knob centre. */
  private scrubPosition(target: { position: THREE.Vector3; amount: number }, phaseOffset: number): THREE.Vector3 {
    const radius = 0.35 * target.amount;
    const angle = this.tapPhase + phaseOffset;
    return target.position.clone().add(new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0).multiplyScalar(radius));
  }

  /**
   * Hover the body just above/behind the current knob so the stretching
   * forelegs can work every control across the whole console, not only the
   * near row. The fly visibly crawls across the panel.
   */
  private applyOperatingStance(
    a: { position: THREE.Vector3 },
    b: { position: THREE.Vector3 },
  ): void {
    if (!this.operatorMode || !this.operatorBase) return;
    const mid = a.position.clone().add(b.position).multiplyScalar(0.5);
    const desired = new THREE.Vector3(mid.x, mid.y + 1.7, mid.z + 2.4);
    this.learner.position.lerp(desired, 0.2);
  }

  /** Build the two additive foreleg tip trails (localized motion streaks). */
  private buildArmTrails(): void {
    for (const side of ["left", "right"] as const) {
      if (this.armTrails.has(side)) continue;
      const count = 18;
      const positions = new Float32Array(count * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({
        color: side === "left" ? 0x8ff3ff : 0xffd479,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const line = new THREE.Line(geometry, material);
      line.frustumCulled = false;
      line.visible = this.trailsVisible;
      this.group.add(line);
      this.armTrails.set(side, { line, positions, history: [], geometry, material });
    }
  }

  /** Append the current tarsus tips to the short streaks (fly/arms only). */
  private updateArmTrails(): void {
    for (const side of ["left", "right"] as const) {
      const rec = this.armTrails.get(side);
      if (!rec) continue;
      const tipJoint = this.tipJoint(side);
      if (!tipJoint) continue;
      const tip = tipJoint.getWorldPosition(new THREE.Vector3()).sub(this.group.position);
      rec.history.push(tip);
      while (rec.history.length > 18) rec.history.shift();
      for (let i = 0; i < 18; i++) {
        const src = rec.history[Math.max(0, rec.history.length - 18 + i)] ?? tip;
        rec.positions[i * 3] = src.x;
        rec.positions[i * 3 + 1] = src.y;
        rec.positions[i * 3 + 2] = src.z;
      }
      rec.geometry.attributes.position.needsUpdate = true;
      rec.geometry.computeBoundingSphere();
      rec.line.visible = this.trailsVisible && rec.history.length > 3;
      rec.material.opacity = THREE.MathUtils.clamp(this.forelegMotion * 90, 0, 0.7);
    }
  }

  /** Toggle the localized foreleg streaks (whole-scene blur is never used). */
  setTrailsVisible(on: boolean): void {
    this.trailsVisible = on;
    for (const rec of this.armTrails.values()) rec.line.visible = on && rec.history.length > 3;
  }

  /** Display-space position of the moving target fly (for tests/telemetry). */
  get targetDisplayPosition(): THREE.Vector3 {
    return this.target.position;
  }

  /** Display-space position of the console operator fly (for tests/telemetry). */
  get operatorDisplayPosition(): THREE.Vector3 {
    return this.learner.position;
  }

  /** Guidance diagnostics: current foreleg tips and the target knobs. */
  get forelegDiagnostics(): { tips: number[][]; targets: number[][] } {
    const tips: number[][] = [];
    for (const side of ["left", "right"] as const) {
      const tip = this.tipJoint(side);
      if (tip) tips.push(tip.getWorldPosition(new THREE.Vector3()).toArray());
    }
    const targets = this.knobOrder.map((ch) => this.knobTargets.get(ch)!.position.toArray());
    return { tips, targets };
  }
}

/**
 * Full orientation from a trajectory tangent plus world up. Local +X (the fly
 * head) maps to the tangent and local +Y (the dorsal axis) is projected from
 * world up, which fixes the remaining roll. Returns null for a zero tangent.
 */
export function orientationFromTangent(velocity: THREE.Vector3, upRef = new THREE.Vector3(0, 1, 0)): THREE.Quaternion | null {
  if (velocity.lengthSq() < 1e-9) return null;
  const forward = velocity.clone().normalize();
  const up = upRef.clone();
  if (Math.abs(forward.dot(up)) > 0.999) up.set(0, 0, 1); // tangent is vertical
  up.addScaledVector(forward, -up.dot(forward)).normalize();
  const side = new THREE.Vector3().crossVectors(forward, up).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(forward, up, side));
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
