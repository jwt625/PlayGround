// Test helpers: build normalised geometries/materials and special functions (no mocks of physics).
import { CrossSection } from '../src/section.mjs';
import { resolveMaterial } from '../src/materials.mjs';

export const rect = (x0, x1, y0, y1) => [
  [x0, y0],
  [x1, y0],
  [x1, y1],
  [x0, y1],
];

export function geom({ domain, regions, electrodes = [], mesh = {}, symmetry = 'none', mirror_x_um = 0, boundary = {}, optical_window = null }) {
  return {
    symmetry,
    mirror_x_um,
    domain: { x: domain.x, y: domain.y },
    regions,
    electrodes: electrodes.map((e) => ({ role: 'signal', material: 'gold', ...e })),
    optical_window,
    mesh,
    boundary: { left: 'neumann', right: 'neumann', top: 'neumann', bottom: 'neumann', ...boundary },
  };
}

export function mats(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) out[k] = resolveMaterial(k, v);
  return out;
}

export const section = (g) => new CrossSection(g);

/** Complete elliptic integral of the first kind K(k) via AGM. */
export function ellipK(k) {
  let a = 1,
    b = Math.sqrt(1 - k * k);
  for (let i = 0; i < 40; i++) {
    const an = 0.5 * (a + b);
    b = Math.sqrt(a * b);
    a = an;
  }
  return Math.PI / (2 * a);
}
export const ellipKp = (k) => ellipK(Math.sqrt(1 - k * k));

export const EPS0 = 8.8541878128e-12;
export const relErr = (a, b) => Math.abs(a - b) / Math.abs(b);
