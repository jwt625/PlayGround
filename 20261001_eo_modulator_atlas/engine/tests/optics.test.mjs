import test from 'node:test';
import assert from 'node:assert/strict';
import { buildOpticalModel, solveModesAt, groupIndex } from '../src/optics.mjs';
import { geom, rect, mats, section, relErr } from './helpers.mjs';

// Symmetric dielectric slab, half-thickness a. Fundamental even mode:
// u tan(u) = q sqrt(V²-u²), q=1 (TE), q=n_core²/n_clad² (TM).
// MIT 6.974, Optical Waveguides and Integrated Optics, section 2.7.
function analytic(lambda, pol) {
  const nc = 2, ns = 1.45, a = 0.3, k0 = 2 * Math.PI / lambda;
  const V = k0 * a * Math.sqrt(nc * nc - ns * ns);
  const q = pol === 'TE' ? 1 : nc * nc / (ns * ns);
  let lo = 0, hi = Math.min(V, Math.PI / 2 - 1e-10);
  for (let i = 0; i < 80; i++) {
    const u = (lo + hi) / 2;
    if (u * Math.tan(u) > q * Math.sqrt(V * V - u * u)) hi = u; else lo = u;
  }
  return Math.sqrt(nc * nc - ((hi + lo) / (2 * k0 * a)) ** 2);
}

function slab(pol, electrodeInside = false) {
  const g = geom({ domain: { x: [0, 0.3], y: [-4, 4] },
    regions: [{ name: 'cladding', material: 'clad', poly: rect(0, 0.3, -4, 4) }, { name: 'core', material: 'core', poly: rect(0, 0.3, -0.3, 0.3) }],
    electrodes: [{ name: 'metal', material: 'gold', weight: 1, poly: electrodeInside ? rect(0, 0.3, 2, 2.1) : rect(1, 2, 2, 2.1) }],
    optical_window: { x: [0, 0.3], y: [-4, 4] },
  });
  return buildOpticalModel(section(g), mats({ core: { n: 2, eps_r: 4 }, clad: { n: 1.45, eps_r: 2 }, gold: { conductor: true } }), {
    polarization: pol, lambda0Um: 1.55, boundary: { left: 'neumann', right: 'neumann' }, hints: { core_edge_um: 0.025, max_edge_um: 0.12 },
  });
}

for (const pol of ['TE', 'TM']) test(`${pol} slab mode agrees with analytic dispersion within 0.1%`, () => {
  const result = solveModesAt(slab(pol), 1.55, { numModes: 1 });
  assert.ok(result.converged);
  assert.ok(relErr(result.neff[0], analytic(1.55, pol)) < 0.001);
});

test('slab group index includes waveguide dispersion', () => {
  const result = groupIndex(slab('TE'), 1.55, 0);
  const expected = analytic(1.55, 'TE') - 1.55 * (analytic(1.56, 'TE') - analytic(1.54, 'TE')) / 0.02;
  assert.ok(result.base.converged && result.lo.converged && result.hi.converged);
  assert.ok(relErr(result.ng, expected) < 0.002);
});

test('metal within the optical window is rejected without inventing an optical index', () => {
  assert.throws(() => solveModesAt(slab('TE', true), 1.55), /metal optical modes are unsupported/);
});

test('metal remains unsupported even if a user supplies a real refractive index', () => {
  const model = slab('TE', true);
  model.mats.gold.nCfg = { n_o: 2, n_e: 2, comp: 'n' };
  assert.throws(() => solveModesAt(model, 1.55), /metal optical modes are unsupported/);
});

test('optical meshing obeys the configured vertex budget', () => {
  const model = slab('TE');
  model.section.geom.mesh.max_vertices = 10;
  assert.throws(() => buildOpticalModel(model.section, model.mats, { polarization: 'TE', lambda0Um: 1.55 }), /vertex budget exceeded/);
});

test('optical solve refuses to extrapolate library dispersion beyond its validity range', () => {
  const model = slab('TE');
  model.mats.clad.dispersionKey = 'silicon_dioxide';
  assert.throws(() => solveModesAt(model, 0.1), /dispersion is out of range/);
});
