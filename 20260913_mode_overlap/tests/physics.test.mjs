import test from 'node:test';
import assert from 'node:assert/strict';
import {gaussianOverlap, gaussianField, mismatchLoss} from '../src/physics/gaussian.mjs';
const mode = (x, y = x) => ({mfdXUm: x, mfdYUm: y});
const close = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
test('known Gaussian limits and diameter convention', () => {
  close(gaussianOverlap(mode(4), mode(4)), 1);
  close(gaussianOverlap(mode(4), mode(8)), .64);
  close(gaussianOverlap(mode(4), mode(8, 4)), .8);
  close(gaussianOverlap(mode(4), mode(4), 1.55, {dxUm: 2}), Math.exp(-1));
  close(gaussianOverlap(mode(4), mode(4), 1.55, {polarizationRad: Math.PI / 3}), .25);
  assert.equal(gaussianOverlap(mode(4), mode(4), 1.55, {polarizationRad: Math.PI / 2}), 0);
  close(gaussianField(mode(4), 2, 0).re / gaussianField(mode(4), 0, 0).re, Math.exp(-1));
});
test('tilt uses specified gap index, not guide effective index', () => {
  const theta = .02, n = 1.45, w = 5.2;
  close(gaussianOverlap(mode(10.4), mode(10.4), 1.55, {thetaXRad: theta, gapIndex: n}), Math.exp(-((Math.PI * n * w * Math.sin(theta) / 1.55) ** 2)));
});
test('independent 2D complex midpoint quadrature, unequal elliptical widths, shift, tilt and polarization', () => {
  const a = mode(3, 4), b = mode(5, 2.5), t = {dxUm: .7, dyUm: -.4, thetaXRad: .03, thetaYRad: -.02, polarizationRad: .4, gapIndex: 1.2};
  const step = .06, half = 12, n = Math.round(2 * half / step);
  let re = 0, im = 0, pa = 0, pb = 0;
  // Construct fields independently of gaussianField / gaussianOverlap.
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = -half + (i + .5) * step, y = -half + (j + .5) * step;
    const fa = Math.exp(-((x / 1.5) ** 2 + (y / 2) ** 2));
    const fb = Math.exp(-(((x - .7) / 2.5) ** 2 + ((y + .4) / 1.25) ** 2));
    const phase = 2 * Math.PI * 1.2 / 1.55 * (Math.sin(.03) * x + Math.sin(-.02) * y);
    re += fa * fb * Math.cos(phase); im += fa * fb * Math.sin(phase);
    pa += fa * fa; pb += fb * fb;
  }
  close(gaussianOverlap(a, b, 1.55, t), (re * re + im * im) / (pa * pb) * Math.cos(.4) ** 2, 1e-8);
});
test('invalid inputs and zero-loss serialization', () => {
  assert.throws(() => gaussianOverlap(mode(0), mode(4)));
  assert.throws(() => gaussianOverlap(mode(4), mode(4), NaN));
  assert.throws(() => gaussianOverlap(mode(4), mode(4), 1.55, {thetaXRad: 1}));
  assert.throws(() => gaussianOverlap(mode(4), mode(4), 1.55, {dx: 1}));
  assert.deepEqual(mismatchLoss(0), {lossDB: null, lossIsInfinite: true});
  close(mismatchLoss(.95).lossDB, .22276394711152253);
});
