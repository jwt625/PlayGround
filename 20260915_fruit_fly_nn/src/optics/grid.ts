/** Square, origin-centered transverse grids used by the numerical reference. */

export interface Grid2D {
  readonly n: number;
  readonly dx: number;
}

export function makeGrid(n: number, dx: number): Grid2D {
  if (!Number.isInteger(n) || n <= 0) throw new Error(`makeGrid: bad n ${n}`);
  if ((n & (n - 1)) !== 0) throw new Error(`makeGrid: n must be a power of two, got ${n}`);
  if (dx <= 0) throw new Error("makeGrid: dx must be positive");
  return { n, dx };
}

/** Coordinate of grid index i, centered so the grid spans [-L/2, L/2). */
export function gridCoord(i: number, n: number, dx: number): number {
  return (i - (n - 1) / 2) * dx;
}

export function gridExtent(grid: Grid2D): number {
  return grid.n * grid.dx;
}

export function gridIndex(ix: number, iy: number, n: number): number {
  return iy * n + ix;
}

/** Peak axis-aligned angle supported by the grid sampling, rad. */
export function maxParaxialAngle(grid: Grid2D, wavelength_m: number): number {
  return wavelength_m / (2 * grid.dx);
}
