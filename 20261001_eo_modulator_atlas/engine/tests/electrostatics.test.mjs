import test from 'node:test';
import assert from 'node:assert/strict';
import { solveElectrostatics, normalizeWeights, C_LIGHT } from '../src/electrostatics.mjs';
import { geom, rect, mats, section, EPS0, relErr } from './helpers.mjs';

// Parallel plates with insulating top/bottom boundaries: C' = eps0 eps_r H / gap.
// MIT 8.02, Chapter 5 (Capacitance), sections 5.2 and 5.4.
function plates(weights, symmetry = 'none', material = { eps_r: 4 }) {
  const g = geom({
    domain: { x: [-1, 1], y: [-1, 1] }, symmetry,
    regions: [{ name: 'dielectric', material: 'dielectric', poly: rect(-1, 1, -1, 1) }],
    electrodes: [
      { name: 'left', weight: weights[0], poly: rect(-1, -0.9, -1, 1) },
      { name: 'right', weight: weights[1], poly: rect(0.9, 1, -1, 1) },
    ],
    mesh: { max_edge_um: 0.25, electrode_edge_um: 0.1, electrode_face_um: 0.25 },
  });
  return solveElectrostatics(section(g), { resolvedMaterials: mats({ dielectric: material }) });
}

test('parallel plates: analytic capacitance, charge/energy agreement, RF index and impedance', () => {
  const r = plates([0, 1]);
  const c0 = EPS0 * 2 / 1.8;
  assert.ok(relErr(r.cPul, 4 * c0) < 1e-8);
  assert.ok(relErr(r.capacitance.energy, r.capacitance.charge) < 1e-8);
  assert.ok(relErr(r.c0Pul, c0) < 1e-8);
  assert.ok(relErr(r.nRf, 2) < 1e-8);
  assert.ok(relErr(r.z0, 1 / (C_LIGHT * 2 * c0)) < 1e-8);
  assert.ok(Math.abs(r.charges[0] + r.charges[1]) / r.cPul < 1e-8);
});

test('differential voltage normalization and odd half-domain preserve full line capacitance', () => {
  const ref = plates([0, 1]);
  for (const [weights, symmetry] of [[[-1, 1], 'none'], [[-0.5, 0.5], 'none'], [[4, 6], 'none'], [[-1, 1], 'x_mirror_odd']]) {
    const r = plates(weights, symmetry);
    assert.ok(relErr(r.cPul, ref.cPul) < 1e-8, `${weights}, ${symmetry}`);
    assert.ok(relErr(r.capacitance.energy, r.capacitance.charge) < 1e-8);
  }
});

test('anisotropic crystal rotation uses the permittivity along the plate field', () => {
  const r = plates([0, 1], 'none', { eps_r: { perp: 4, par: 9 }, crystal: { cut: 'x', propagation: 'y' } });
  assert.ok(relErr(r.cPul, EPS0 * 9 * 2 / 1.8) < 1e-8);
  assert.ok(relErr(r.nRf, 3) < 1e-8);
});

test('zero line voltage is rejected', () => {
  assert.throws(() => normalizeWeights([{ weight: 1 }, { weight: 1 }], 'none'), /zero line voltage/);
});
