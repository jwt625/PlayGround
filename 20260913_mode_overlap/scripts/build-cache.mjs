import {readFile, mkdir, writeFile, rename} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {contentKey, canonicalJSON, CACHE_VERSION} from '../src/storage/cache.mjs';
import {analyticQuestion, balancedAlignmentBank, GENERATOR_VERSION} from '../src/scenarios/generator.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
export async function buildCache(output = join(root, 'static/cache')) {
  const catalog = JSON.parse(await readFile(join(root, 'data/presets.json'), 'utf8'));
  const examples = JSON.parse(await readFile(join(root, 'data/examples.json'), 'utf8'));
  const byId = new Map(catalog.presets.map(p => [p.id, p]));
  const questions = examples.examples.filter(e => e.kind === 'analytic').map(e => analyticQuestion(e));
  questions.push(...balancedAlignmentBank());
  const provenance = {cacheVersion: CACHE_VERSION, presetsVersion: catalog.schemaVersion, presetsHash: await contentKey(catalog), examplesHash: await contentKey(examples), generatorVersion: GENERATOR_VERSION};
  const pending = examples.examples.filter(e => e.kind !== 'analytic').map(e => {
    if (e.kind === 'complex-field-fixture') return {...e, status: 'pending-complex-field-fixture'};
    const instantiate = (id, overrides = {}) => {
      const p = byId.get(id);
      if (!p) throw new Error(`Unknown preset ${id}`);
      return {...p, nominal: {...p.nominal, ...overrides}};
    };
    return {...e, structureA: instantiate(e.presetA, e.overrideA), structureB: instantiate(e.presetB, e.overrideB), status: 'pending-vector-solver', solverVersion: null, materialDBVersion: null, meshSettings: null};
  });
  await mkdir(join(output, 'questions'), {recursive: true});
  let hits = 0, writes = 0;
  async function persist(path, value) {
    const text = canonicalJSON(value) + '\n';
    try {if (await readFile(path, 'utf8') === text) {hits++; return;}} catch (e) {if (e.code !== 'ENOENT') throw e;}
    const temporary = `${path}.${process.pid}.tmp`;
    await writeFile(temporary, text); await rename(temporary, path); writes++;
  }
  const entries = [];
  for (const question of questions) {
    const record = {...question, provenance};
    const key = await contentKey(record);
    const path = `questions/${key}.json`;
    await persist(join(output, path), record);
    entries.push({id: question.id, key, path, status: question.status, category: question.category, bucket: question.bucket});
  }
  const histogram = Array(7).fill(0);
  for (const q of questions) histogram[q.bucket]++;
  const manifest = {schemaVersion: CACHE_VERSION, provenance, readyCount: entries.length, pendingCount: pending.length, histogram, entries, pendingPath: 'numerical-recipes.json'};
  await persist(join(output, 'numerical-recipes.json'), {provenance, recipes: pending});
  await persist(join(output, 'manifest.json'), manifest);
  return {ready: entries.length, pending: pending.length, histogram, hits, writes};
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) console.log(JSON.stringify(await buildCache(process.argv[2] ? resolve(process.argv[2]) : undefined), null, 2));
