import { describe, expect, it } from "vitest";
import {
  DEFAULT_FIXTURE,
  channelId,
  createArrayConfig,
  hexSites,
  orderSites,
  rowCounts,
  sitePosition,
  wavenumber,
} from "../src/optics/geometry";

describe("T01 hex geometry", () => {
  it("yields 19 unique sites at order 2", () => {
    const sites = hexSites(2);
    expect(sites).toHaveLength(19);
    const keys = new Set(sites.map((s) => `${s.q},${s.r}`));
    expect(keys.size).toBe(19);
  });

  it("has row counts 3/4/5/4/3 from +Y to -Y", () => {
    expect(rowCounts(2)).toEqual([3, 4, 5, 4, 3]);
  });

  it("assigns unique stable ids CH01..CH19 in row order, ascending X", () => {
    const config = createArrayConfig();
    expect(config.channelIds).toHaveLength(19);
    expect(new Set(config.channelIds).size).toBe(19);
    expect(config.channelIds[0]).toBe("CH01");
    expect(config.channelIds[18]).toBe("CH19");

    // CH01..CH03 are the top row (r = 2), ascending x.
    expect(config.r.slice(0, 3)).toEqual([2, 2, 2]);
    expect(config.x_m[0]).toBeLessThan(config.x_m[1]);
    expect(config.x_m[1]).toBeLessThan(config.x_m[2]);
    // CH08..CH12 is the central row (r = 0), five channels.
    expect(config.r.slice(7, 12)).toEqual([0, 0, 0, 0, 0]);
  });

  it("places the array centroid at the origin", () => {
    const config = createArrayConfig();
    const meanX = config.x_m.reduce((a, b) => a + b, 0) / config.x_m.length;
    const meanY = config.y_m.reduce((a, b) => a + b, 0) / config.y_m.length;
    expect(meanX).toBeCloseTo(0, 12);
    expect(meanY).toBeCloseTo(0, 12);
    expect(config.z_m.every((z) => z === 0)).toBe(true);
  });

  it("has nearest-neighbor spacing equal to the pitch", () => {
    const config = createArrayConfig();
    const n = config.x_m.length;
    let min = Infinity;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = config.x_m[i] - config.x_m[j];
        const dy = config.y_m[i] - config.y_m[j];
        min = Math.min(min, Math.hypot(dx, dy));
      }
    }
    expect(min).toBeCloseTo(config.pitch_m, 12);
  });

  it("is invariant under 60-degree rotation (sixfold symmetry)", () => {
    const config = createArrayConfig();
    const c = Math.cos(Math.PI / 3);
    const s = Math.sin(Math.PI / 3);
    const snap = (v: number) => Math.round(v * 1e9) / 1e9 + 0;
    const key = (x: number, y: number) => `${snap(x)},${snap(y)}`;
    const original = new Set(config.x_m.map((x, i) => key(x, config.y_m[i])));
    for (let i = 0; i < config.x_m.length; i++) {
      const x = config.x_m[i];
      const y = config.y_m[i];
      const rx = c * x - s * y;
      const ry = s * x + c * y;
      expect(original.has(key(rx, ry))).toBe(true);
    }
  });

  it("is deterministic and independent of insertion order", () => {
    const a = orderSites(hexSites(2));
    const flipped = [...hexSites(2)].reverse();
    const b = orderSites(flipped);
    expect(a).toEqual(b);
  });

  it("exposes the provisional numerical fixture", () => {
    const config = createArrayConfig();
    expect(config.wavelength_m).toBe(1550e-9);
    expect(config.pitch_m).toBe(0.5e-3);
    expect(config.launchRadius_m).toBe(0.18e-3);
    expect(wavenumber(config.wavelength_m)).toBeCloseTo((2 * Math.PI) / 1550e-9, 6);
    expect(DEFAULT_FIXTURE.order).toBe(2);
  });

  it("builds configurable smaller fixtures", () => {
    expect(createArrayConfig({ order: 1 }).channelIds).toHaveLength(7);
    expect(rowCounts(1)).toEqual([2, 3, 2]);
    expect(channelId(0)).toBe("CH01");
    expect(channelId(18)).toBe("CH19");
  });

  it("computes axial site positions with the documented mapping", () => {
    const p = sitePosition({ q: 1, r: 0 }, 2e-3);
    expect(p.x_m).toBeCloseTo(2e-3, 12);
    expect(p.y_m).toBeCloseTo(0, 12);
  });
});
