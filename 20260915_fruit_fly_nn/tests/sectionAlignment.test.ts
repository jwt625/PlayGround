import { describe, expect, it } from "vitest";
import { createArrayConfig } from "../src/optics/geometry";
import { commandsToActual, uniformCommands } from "../src/optics/channels";
import { MeasurementSection } from "../src/renderer/section";
import { DISPLAY } from "../src/renderer/scene";

/**
 * The measured intensity plane and the animated beam envelopes must share one
 * display mapping: transverse axes use DISPLAY.scale, the propagation axis uses
 * DISPLAY.targetDistance. A uniform scale (the earlier bug) shifted the plane
 * off the beam intersection by ~60x.
 */
describe("measurement section display alignment", () => {
  it("places the plane center where the beam envelopes end", () => {
    const config = createArrayConfig();
    const sx = 0.012;
    const sy = -0.004;
    const sz = Math.sqrt(1 - sx * sx - sy * sy);
    const range = 1;

    const commands = uniformCommands(config);
    for (const cmd of commands) {
      cmd.tiltX = sx;
      cmd.tiltY = sy;
    }
    const actual = commandsToActual(config, commands);

    const section = new MeasurementSection({ size_m: 30e-3, samples: 4, range_m: range });
    section.update(config, actual, { sx, sy, sz }, range, { x: sx * range, y: sy * range, z: sz * range });

    const center = section.displayCenter;
    expect(center.x).toBeCloseTo(sx * range * DISPLAY.scale, 6);
    expect(center.y).toBeCloseTo(sy * range * DISPLAY.scale, 6);
    expect(center.z).toBeCloseTo(sz * range * DISPLAY.targetDistance, 6);

    // Transverse:axial display ratio must equal scale:targetDistance, matching
    // bench.ts. This is the invariant the alignment fix enforces.
    const transverse = Math.hypot(center.x, center.y);
    const axial = center.z;
    const physicalTransverse = Math.hypot(sx * range, sy * range);
    const physicalAxial = sz * range;
    expect(transverse / physicalTransverse).toBeCloseTo(DISPLAY.scale, 6);
    expect(axial / physicalAxial).toBeCloseTo(DISPLAY.targetDistance, 6);
  });
});
