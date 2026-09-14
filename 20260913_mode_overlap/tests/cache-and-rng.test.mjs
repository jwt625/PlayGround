import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {canonicalJSON, contentKey, modeKey, pairKey} from '../src/storage/cache.mjs';
import {rng, samplePreset, balancedAlignmentBank, bucketIndex} from '../src/scenarios/generator.mjs';
import {buildCache} from '../scripts/build-cache.mjs';
const catalog = JSON.parse(await readFile(new URL('../data/presets.json', import.meta.url), 'utf8'));
test('canonical hash ignores property order; rejects lossy JSON values', async () => {
  assert.equal(await contentKey({x: 1, y: {a: 2}}), await contentKey({y: {a: 2}, x: 1}));
  for (const value of [NaN, Infinity, undefined, new Date(), {x: undefined}, new Array(2)]) assert.throws(() => canonicalJSON(value));
  assert.notEqual(await contentKey({x: .45}), await contentKey({x: .45000001}));
});
test('mode and pair cache invalidation includes all physics inputs', async () => {
  const p = {geometry: {width: .45}, materials: {version: '1', n: 3.48}, wavelengthUm: 1.55, requestedModes: ['TE0'], solver: {version: '1', residual: 1e-8}, mesh: {step: .02}, boundary: {type: 'closed', extent: 4}};
  const key = await modeKey(p);
  for (const k of Object.keys(p)) {
    const changed = structuredClone(p); changed[k] = {different: true};
    assert.notEqual(await modeKey(changed), key, k);
  }
  await assert.rejects(() => modeKey({geometry: {}}));
  const pair = {modeA: key, modeB: key, transform: {dxUm: 0}, overlapVersion: '1', integration: {step: .01}};
  assert.notEqual(await pairKey(pair), await pairKey({...pair, transform: {dxUm: .1}}));
  assert.notEqual(await pairKey(pair), await pairKey({...pair, overlapVersion: '2'}));
});
test('uint32 replay and every preset obeys bounds and geometry constraints', () => {
  const a = rng(42), b = rng(42);
  for (let i = 0; i < 100; i++) assert.equal(a(), b());
  assert.throws(() => rng(-1)); assert.throws(() => rng(2 ** 32));
  for (const preset of catalog.presets) for (let seed = 0; seed < 1000; seed++) {
    const q = samplePreset(preset, seed), p = q.parameters;
    for (const [key, [low, high]] of Object.entries(preset.ranges)) assert.ok(p[key] >= low && p[key] <= high, `${preset.id}:${key}`);
    if ('slab_um' in p) assert.ok(p.slab_um < p.height_um);
    if ('etch_um' in p) assert.ok(p.etch_um < p.film_um);
    assert.equal(q.wavelengthUm, preset.wavelength_um);
    if (seed === 0) assert.deepEqual(q, samplePreset(preset, seed));
  }
});
test('balanced bank is deterministic, bounded and has full bucket coverage', () => {
  const bank = balancedAlignmentBank(123, 20), counts = Array(7).fill(0);
  assert.deepEqual(bank, balancedAlignmentBank(123, 20));
  for (const q of bank) {
    counts[bucketIndex(q.answer.eta)]++;
    assert.ok(Math.hypot(q.transform.dxUm / (q.a.mfdXUm / 2), q.transform.dyUm / (q.a.mfdYUm / 2)) <= 2.5);
  }
  assert.deepEqual(counts, Array(7).fill(20));
  for (const [eta, bucket] of [[0,0],[.1,1],[.3,2],[.6,3],[.8,4],[.95,5],[.99,6],[1,6]]) assert.equal(bucketIndex(eta), bucket);
});
test('cold build, byte-identical warm hits, content verification, corruption repair and required 20 cases', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mode-overlap-test-'));
  try {
    const cold = await buildCache(dir);
    assert.equal(cold.ready, 146); assert.equal(cold.pending, 14); assert.equal(cold.hits, 0);
    const first = await readFile(join(dir, 'manifest.json'), 'utf8');
    const warm = await buildCache(dir);
    assert.equal(warm.writes, 0); assert.equal(warm.hits, 148);
    assert.equal(await readFile(join(dir, 'manifest.json'), 'utf8'), first);
    const manifest = JSON.parse(first);
    for (const entry of manifest.entries) assert.equal(await contentKey(JSON.parse(await readFile(join(dir, entry.path), 'utf8'))), entry.key);
    const pending = JSON.parse(await readFile(join(dir, 'numerical-recipes.json'), 'utf8')).recipes;
    assert.equal(new Set([...manifest.entries.filter(e => /^\d{2}$/.test(e.id)), ...pending].map(e => e.id)).size, 20);
    assert.ok(pending.every(e => !('answer' in e)));
    await writeFile(join(dir, manifest.entries[0].path), '{broken');
    const repaired = await buildCache(dir);
    assert.equal(repaired.writes, 1); assert.equal(repaired.hits, 147);
  } finally {await rm(dir, {recursive: true, force: true});}
});
