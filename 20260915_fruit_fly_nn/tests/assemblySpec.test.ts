import { describe, expect, it } from "vitest";
import {
  APERTURE_SITES,
  CABLES,
  CELL_ASSIGNMENTS,
  auditInventory,
  equipmentById,
  portOf,
} from "../src/renderer/hardware/assemblySpec";

describe("CBC assembly specification inventory (Pass 1/2)", () => {
  it("has 19 channels, 19 electronics cells, no CH20", () => {
    expect(CELL_ASSIGNMENTS).toHaveLength(19);
    expect(new Set(CELL_ASSIGNMENTS.map((c) => c.channel)).size).toBe(19);
    expect(CELL_ASSIGNMENTS.some((c) => c.channel === "CH20")).toBe(false);
  });

  it("uses the canonical 3+4+5+4+3 aperture with exact channel identities", () => {
    expect(APERTURE_SITES).toHaveLength(19);
    const byY = new Map<number, number>();
    for (const s of APERTURE_SITES) byY.set(s.y_mm, (byY.get(s.y_mm) ?? 0) + 1);
    const counts = [...byY.entries()].sort((a, b) => b[0] - a[0]).map((e) => e[1]);
    expect(counts).toEqual([3, 4, 5, 4, 3]);
    const ch10 = APERTURE_SITES.find((s) => s.id === "CH10")!;
    expect(ch10.x_mm).toBeCloseTo(0, 6);
    expect(ch10.y_mm).toBeCloseTo(300, 6);
    expect(ch10.z_mm).toBeCloseTo(100, 6);
  });

  it("matches the 241 internal runs and family counts in §10", () => {
    const audit = auditInventory();
    expect(audit.channels).toBe(19);
    expect(audit.internalRuns).toBe(241);
    expect(audit.externalRuns).toBe(1);
    expect(audit.byFamily).toEqual({ optical: 58, rf: 19, command: 40, dc: 48, motor: 76, external: 1 });
    expect(audit.actuators).toBe(57);
    expect(audit.fcSockets).toBe(78);
    expect(audit.fcPlugs).toBe(78);
  });

  it("resolves every cable endpoint to a real equipment port", () => {
    expect(new Set(CABLES.map((c) => c.id)).size).toBe(CABLES.length);
    for (const cable of CABLES) {
      expect(equipmentById(cable.from.equipment)).toBeTruthy();
      expect(equipmentById(cable.to.equipment)).toBeTruthy();
      expect(portOf(cable.from.equipment, cable.from.port)).toBeTruthy();
      expect(portOf(cable.to.equipment, cable.to.port)).toBeTruthy();
    }
  });

  it("keeps pigtails and free-space apertures typed separately from sockets", () => {
    const phOut = portOf("PH-CH10", "optical_out");
    expect(phOut.pigtail).toBe(true);
    const emission = portOf("COL-CH10", "emission");
    expect(emission.freeSpace).toBe(true);
    const ampIn = portOf("AMP-CH10", "optical_in");
    expect(ampIn.pigtail).toBeUndefined();
    expect(ampIn.connector).toBe("FC/APC");
  });
});
