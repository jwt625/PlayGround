import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { orientationFromTangent } from "../src/renderer/flyActors";

function axes(q: THREE.Quaternion) {
  return {
    head: new THREE.Vector3(1, 0, 0).applyQuaternion(q),
    up: new THREE.Vector3(0, 1, 0).applyQuaternion(q),
  };
}

describe("target fly orientation from trajectory tangent", () => {
  it("returns null for a zero tangent so the previous heading is kept", () => {
    expect(orientationFromTangent(new THREE.Vector3(0, 0, 0))).toBeNull();
  });

  it("points the head (+X) along a horizontal tangent and keeps dorsal up", () => {
    const q = orientationFromTangent(new THREE.Vector3(1, 0, 0))!;
    const { head, up } = axes(q);
    expect(head.x).toBeCloseTo(1, 6);
    expect(head.y).toBeCloseTo(0, 6);
    expect(head.z).toBeCloseTo(0, 6);
    expect(up.y).toBeCloseTo(1, 6);
  });

  it("points the head along a diagonal tangent with positive dorsal-up component", () => {
    const tangent = new THREE.Vector3(0.6, 0.5, 0.623).normalize();
    const q = orientationFromTangent(tangent)!;
    const { head, up } = axes(q);
    expect(head.dot(tangent)).toBeCloseTo(1, 6);
    expect(up.dot(new THREE.Vector3(0, 1, 0))).toBeGreaterThan(0);
    expect(head.dot(up)).toBeCloseTo(0, 6);
  });

  it("stays finite for a fully vertical tangent", () => {
    const q = orientationFromTangent(new THREE.Vector3(0, 1, 0))!;
    const { head, up } = axes(q);
    expect(Number.isFinite(up.x + up.y + up.z)).toBe(true);
    expect(head.y).toBeCloseTo(1, 6);
    expect(head.dot(up)).toBeCloseTo(0, 6);
  });
});
