/**
 * Geometry and unit conventions for the tiled coherent beam-combining array.
 *
 * Internal units are SI. The array lies in the XY plane; forward propagation is
 * +Z. Camera/display scaling must never change these optical coordinates.
 *
 * Channel identification is stable and shared by commands, wiring, geometry,
 * logs, and learned outputs: CH01..CHnn ordered by row from +Y to -Y and then
 * ascending X. See `docs/CODING_TASKS.md` T01 and DevLog/001.
 */

export interface AxialSite {
  q: number;
  r: number;
}

export interface ArrayConfig {
  readonly id: string;
  readonly order: number;
  readonly wavelength_m: number;
  readonly pitch_m: number;
  readonly launchRadius_m: number;
  /** Declared circular launch-aperture truncation radius, m. */
  readonly apertureRadius_m: number;
  readonly channelIds: readonly string[];
  readonly q: readonly number[];
  readonly r: readonly number[];
  readonly x_m: readonly number[];
  readonly y_m: readonly number[];
  readonly z_m: readonly number[];
  /** Training steering limit (normalized direction, dimensionless). */
  readonly steeringLimit: number;
  /** Maximum convergence magnitude, 1/m. */
  readonly curvatureLimit_per_m: number;
  /** Piston command range, rad (symmetric). */
  readonly pistonLimit_rad: number;
  readonly amplitudeMin: number;
  readonly amplitudeMax: number;
}

/**
 * Hexagonal (axial) site set of a given order: integer (q, r) with
 * max(|q|, |r|, |q + r|) <= order. Order 2 yields 19 sites (rows 3/4/5/4/3).
 */
export function hexSites(order: number): AxialSite[] {
  if (!Number.isInteger(order) || order < 0) {
    throw new Error(`hexSites: order must be a non-negative integer, got ${order}`);
  }
  const sites: AxialSite[] = [];
  for (let q = -order; q <= order; q++) {
    for (let r = -order; r <= order; r++) {
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) <= order) {
        sites.push({ q, r });
      }
    }
  }
  return sites;
}

/** Physical site position in the array plane. */
export function sitePosition(
  site: AxialSite,
  pitch_m: number,
): { x_m: number; y_m: number; z_m: number } {
  return {
    x_m: pitch_m * (site.q + site.r / 2),
    y_m: (pitch_m * Math.sqrt(3)) / 2 * site.r,
    z_m: 0,
  };
}

/**
 * Order sites deterministically: rows from +Y to -Y (descending r), then
 * ascending X (ascending q within a row).
 */
export function orderSites(sites: readonly AxialSite[]): AxialSite[] {
  return [...sites].sort((a, b) => {
    if (a.r !== b.r) return b.r - a.r;
    return a.q - b.q;
  });
}

export function channelId(index: number): string {
  if (!Number.isInteger(index) || index < 0) {
    throw new Error(`channelId: index must be a non-negative integer, got ${index}`);
  }
  return `CH${String(index + 1).padStart(2, "0")}`;
}

export interface ArrayConfigOverrides {
  id?: string;
  order?: number;
  wavelength_m?: number;
  pitch_m?: number;
  launchRadius_m?: number;
  apertureRadius_m?: number;
  steeringLimit?: number;
  curvatureLimit_per_m?: number;
  pistonLimit_rad?: number;
  amplitudeMin?: number;
  amplitudeMax?: number;
}

/** Declared simulation fixture defaults (provisional until T02 convergence checks). */
export const DEFAULT_FIXTURE: Omit<Required<ArrayConfigOverrides>, "id" | "apertureRadius_m"> & {
  id: string;
} = {
  id: "default-19ch",
  order: 2,
  wavelength_m: 1550e-9,
  pitch_m: 0.5e-3,
  launchRadius_m: 0.18e-3,
  steeringLimit: (2 * Math.PI) / 180,
  curvatureLimit_per_m: 100,
  pistonLimit_rad: Math.PI,
  amplitudeMin: 0,
  amplitudeMax: 2,
};

export function createArrayConfig(overrides: ArrayConfigOverrides = {}): ArrayConfig {
  const order = overrides.order ?? DEFAULT_FIXTURE.order;
  return createArrayConfigFromSites(hexSites(order), overrides);
}

/**
 * Build a config from an explicit ordered site list. Used by tests and fixtures
 * that need coincident or intentionally unusual emitters; production geometry
 * goes through `createArrayConfig`.
 */
export function createArrayConfigFromSites(
  sites: readonly AxialSite[],
  overrides: ArrayConfigOverrides = {},
): ArrayConfig {
  const merged = { ...DEFAULT_FIXTURE, ...overrides };
  if (merged.pitch_m <= 0) throw new Error("pitch_m must be positive");
  if (merged.wavelength_m <= 0) throw new Error("wavelength_m must be positive");
  if (merged.launchRadius_m <= 0) throw new Error("launchRadius_m must be positive");
  const apertureRadius_m = overrides.apertureRadius_m ?? 3 * merged.launchRadius_m;
  if (apertureRadius_m <= 0) throw new Error("apertureRadius_m must be positive");

  const ordered = orderSites(sites);
  const q: number[] = [];
  const r: number[] = [];
  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  const ids: string[] = [];
  ordered.forEach((site, index) => {
    const p = sitePosition(site, merged.pitch_m);
    q.push(site.q);
    r.push(site.r);
    x.push(p.x_m);
    y.push(p.y_m);
    z.push(p.z_m);
    ids.push(channelId(index));
  });

  return Object.freeze({
    id: merged.id,
    order: merged.order,
    wavelength_m: merged.wavelength_m,
    pitch_m: merged.pitch_m,
    launchRadius_m: merged.launchRadius_m,
    apertureRadius_m,
    channelIds: Object.freeze(ids),
    q: Object.freeze(q),
    r: Object.freeze(r),
    x_m: Object.freeze(x),
    y_m: Object.freeze(y),
    z_m: Object.freeze(z),
    steeringLimit: merged.steeringLimit,
    curvatureLimit_per_m: merged.curvatureLimit_per_m,
    pistonLimit_rad: merged.pistonLimit_rad,
    amplitudeMin: merged.amplitudeMin,
    amplitudeMax: merged.amplitudeMax,
  });
}

/** Vacuum wavenumber k = 2 pi / lambda. */
export function wavenumber(wavelength_m: number): number {
  return (2 * Math.PI) / wavelength_m;
}

/** Nearest-neighbor spacing of a hex order-n lattice, in metres. */
export function neighborSpacing_m(pitch_m: number): number {
  return pitch_m;
}

/** Row counts of an order-n hex lattice from +Y to -Y. */
export function rowCounts(order: number): number[] {
  const counts: number[] = [];
  for (let r = order; r >= -order; r--) {
    let n = 0;
    for (let q = -order; q <= order; q++) {
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) <= order) n++;
    }
    counts.push(n);
  }
  return counts;
}
