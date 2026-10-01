import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseConfig, normalizeConfig, STAGES, TARGET_METRICS, MAX_VERTICES } from '../src/config.mjs';
import { compareTargets, runCrossSection } from '../src/run.mjs';

const text = readFileSync(new URL('../../sims/chen2022/config.yaml', import.meta.url), 'utf8');
const raw = () => structuredClone(parseConfig(text).raw);
const fixture = readFileSync(new URL('./fixtures/parallel-plates.yaml', import.meta.url), 'utf8');

test('Chen config resolves regions, materials and both cross-sections', () => {
  const c = parseConfig(text);
  assert.deepEqual(Object.keys(c.geometries), ['geometry', 'unloaded']);
  assert.equal(c.geometries.geometry.electrodes.length, 7);
  assert.equal(c.raw.validation_status, 'unvalidated');
});

test('input boundary rejects unknown schema, duplicate YAML keys, missing physics and bad geometry', () => {
  assert.throws(() => parseConfig('schema: a\nschema: b'), /YAML/);
  const cases = [
    [c => c.schema = 'other', /schema/],
    [c => delete c.materials.air.eps_r, /RF permittivity/],
    [c => c.materials.air.eps_r = -1, /positive/],
    [c => c.geometry.domain.x_um = [1, -1], /max must exceed/],
    [c => c.geometry.regions[0].material = 'missing', /unknown material/],
    [c => c.geometry.electrodes[0].weight = NaN, /finite/],
    [c => c.geometry.mesh.max_edge_um = 0, /positive/],
    [c => c.targets[0].tol_rel = -1, /tolerance cannot/],
    [c => c.targets = {}, /expected an array/],
    [c => c.targets[0].metric = 'constructor', /expected one of/],
    [c => c.targets[0].at_ghz = -1, /positive/],
    [c => c.geometry.mesh.max_vertices = 80001, /4 to 80000/],
    [c => c.geometry.mesh.max_vertices = 100.5, /integer/],
    [c => c.geometry.mesh.typo_edge_um = 1, /unsupported field/],
    [c => c.geometry.boundary = { botom: 'dirichlet' }, /unsupported field/],
    [c => c.materials.gold.conductor = 'false', /boolean/],
    [c => delete c.materials.lithium_niobate.n_e, /provided together/],
    [c => c.line.loading.type = 'mystery', /expected one of/],
    [c => c.line.loading.loaded_length_um = 100, /exceed the period/],
    [c => c.line.loading.unloaded_cross_section = 'missing', /existing cross-section/],
    [c => c.chain = 'electrostatics', /expected an array/],
    [c => c.optics.mode_index = 99, /integer from 0 to 4/],
  ];
  for (const [change, message] of cases) { const c = raw(); change(c); assert.throws(() => normalizeConfig(c), message); }
});

test('unknown topology, frequency-specific, missing and configured metrics are never counted as predictions', () => {
  const target = { metric: 'n_rf', value: 2, tol_abs: 0.1 };
  assert.equal(compareTargets([target], { n_rf: 2 })[0].status, 'not_evaluated');
  assert.equal(compareTargets([{ ...target, at_ghz: 40 }], { n_rf: 2 }, { topology: 'none' })[0].status, 'not_evaluated');
  assert.equal(compareTargets([{ ...target, metric: 'ng_opt' }], { ng_opt: 2 }, { fixedGroupIndex: true })[0].status, 'not_evaluated');
  assert.equal(compareTargets([target], { n_rf: NaN }, { topology: 'none' })[0].actual, null);
});

test('runner computes analytic input and excludes fixed group index from target coverage', () => {
  const result = runCrossSection(fixture, { optical: true });
  assert.ok(Math.abs(result.metrics.n_rf - 2) < 1e-8);
  assert.deepEqual(result.stages, ['electrostatics', 'optical_mode']);
  assert.deepEqual(result.targetSummary, { total: 2, evaluated: 1, passed: 1, failed: 0 });
  assert.equal(result.targets[1].actual, null);
  assert.ok(result.warnings.includes('group_index_is_configured_not_predicted'));
  assert.throws(() => runCrossSection(fixture, { meshScale: 0 }), /meshScale/);
  assert.throws(() => runCrossSection(fixture, { section: 'missing' }), /Unknown cross-section/);
  const capped = parseConfig(fixture).raw;
  capped.geometry.mesh.max_vertices = 10;
  assert.throws(() => runCrossSection(JSON.stringify(capped)), /vertex budget exceeded/);
});

test('CLI exit status distinguishes execution, input failure and evaluated target failure', () => {
  const dir = mkdtempSync(join(tmpdir(), 'eo-atlas-cli-'));
  try {
    const path = join(dir, 'config.yaml');
    const cli = fileURLToPath(new URL('../cli.mjs', import.meta.url));
    const run = () => spawnSync(process.execPath, [cli, path], { encoding: 'utf8', timeout: 10000 });
    writeFileSync(path, fixture);
    let result = run();
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).targetSummary.evaluated, 1);
    const changed = parseConfig(fixture).raw;
    changed.targets[0].value = 20;
    writeFileSync(path, JSON.stringify(changed));
    result = run();
    assert.equal(result.status, 2, result.stderr);
    assert.equal(JSON.parse(result.stdout).targets[0].status, 'fail');
    writeFileSync(path, 'schema: wrong');
    result = run();
    assert.equal(result.status, 1);
    assert.match(result.stderr, /schema/);
    assert.equal(result.stdout, '');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('documented schema enums and resource limit match the runtime contract', () => {
  const schema = JSON.parse(readFileSync(new URL('../schema/sim.schema.json', import.meta.url), 'utf8'));
  assert.deepEqual(schema.properties.chain.items.enum, STAGES);
  assert.deepEqual(schema.$defs.target.properties.metric.enum, TARGET_METRICS);
  assert.equal(schema.$defs.mesh.properties.max_vertices.maximum, MAX_VERTICES);
});

test('periodic targets never pass based on a static section value', () => {
  const targets = [{ metric: 'n_rf', value: 2, tol_abs: 0.1 }, { metric: 'ng_opt', value: 2.2, tol_rel: 0.01 }, { metric: 'bw3db_ghz', value: 100, tol_abs: 10 }];
  const results = compareTargets(targets, { n_rf: 2, ng_opt: 2.2 }, { topology: 'periodic_t_rail' });
  assert.deepEqual(results.map(t => t.status), ['not_evaluated', 'pass', 'not_evaluated']);
  assert.equal(compareTargets(targets, { n_rf: 3 }, { topology: 'none' })[0].status, 'fail');
  assert.ok(compareTargets(targets, { n_rf: 2, ng_opt: 2.2 }, { section: 'unloaded' }).every(t => t.status === 'not_evaluated'));
});
