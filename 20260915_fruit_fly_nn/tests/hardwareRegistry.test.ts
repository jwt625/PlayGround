import { describe, expect, it } from "vitest";
import {
  buildHardwareRegistry,
  validateRegistry,
  type HardwareRegistry,
} from "../src/renderer/hardware/registry";

const registry: HardwareRegistry = buildHardwareRegistry();

describe("H0 hardware registry", () => {
  it("loads all nine generated components", () => {
    expect(registry.components.size).toBe(9);
    for (const name of [
      "fc-apc-plug",
      "fc-bulkhead",
      "sma-plug",
      "splitter-19",
      "phase-cassette",
      "phase-driver",
      "optical-amplifier",
      "tiptilt-collimator",
      "fly-console",
    ]) {
      expect(registry.components.has(name)).toBe(true);
    }
  });

  it("maps CH01..CH19 to phase/driver/amplifier/mount/focus with no duplicates", () => {
    expect(registry.channels.size).toBe(19);
    const assigned = new Set<string>();
    for (let i = 1; i <= 19; i++) {
      const ch = `CH${String(i).padStart(2, "0")}`;
      const entry = registry.channels.get(ch);
      expect(entry, ch).toBeTruthy();
      for (const role of ["phase", "driver", "amplifier", "mount", "focus"] as const) {
        const id = entry![role];
        expect(id, `${ch} ${role}`).toBeTruthy();
        expect(assigned.has(id), `duplicate node ${id}`).toBe(false);
        assigned.add(id);
      }
    }
  });

  it("traces CH01 splitter -> phase -> amplifier -> mount", () => {
    const entry = registry.channels.get("CH01")!;
    const optical = registry.cables.filter((c) => c.channel === "CH01" && c.kind === "optical");
    const paths = optical.map((c) => `${c.source.nodeId}.${c.source.portName}->${c.target.nodeId}.${c.target.portName}`);
    expect(paths).toContain("splitter.CH01->CH01-phase.optical_in");
    expect(paths).toContain("CH01-phase.optical_out->CH01-amplifier.optical_in");
    expect(paths).toContain(`CH01-amplifier.optical_out->${entry.mount}.fiber_in`);
  });

  it("has no structural errors and only pigtail-assembly warnings", () => {
    const issues = validateRegistry(registry);
    const errors = issues.filter((i) => i.severity === "error");
    expect(errors, JSON.stringify(errors, null, 2)).toEqual([]);
    for (const warning of issues.filter((i) => i.severity === "warning")) {
      expect(warning.message).toContain("pigtail");
    }
  });

  it("keeps display metadata separate from solver coordinates", () => {
    expect(registry.units).toBe("meters");
    expect(registry.upAxis).toBe("+Z");
    // No wavelength / pitch / aperture solver fields leaked into the registry.
    const keys = Object.keys(registry).join(",");
    expect(keys).not.toContain("wavelength");
    expect(keys).not.toContain("pitch");
    // Registry provenance must not reference the vendor STEP cache.
    expect(registry.provenance.toLowerCase()).toContain("no vendor cad");
  });
});
