import {readFile} from 'node:fs/promises';
import {contentKey} from '../src/storage/cache.mjs';
import {gaussianOverlap} from '../src/physics/gaussian.mjs';
import {bucketIndex} from '../src/scenarios/generator.mjs';
const base = new URL('../static/cache/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', base), 'utf8'));
const histogram = Array(7).fill(0);
for (const entry of manifest.entries) {
  const record = JSON.parse(await readFile(new URL(entry.path, base), 'utf8'));
  if (await contentKey(record) !== entry.key) throw new Error(`Hash mismatch ${entry.id}`);
  const eta = gaussianOverlap(record.a, record.b, record.wavelengthUm, record.transform);
  if (Math.abs(eta - record.answer.eta) > 1e-14) throw new Error(`Answer mismatch ${entry.id}`);
  if (bucketIndex(eta) !== entry.bucket) throw new Error(`Bucket mismatch ${entry.id}`);
  histogram[entry.bucket]++;
}
const {recipes} = JSON.parse(await readFile(new URL(manifest.pendingPath, base), 'utf8'));
if (recipes.some(r => 'answer' in r)) throw new Error('Pending recipe has a released answer');
if (JSON.stringify(histogram) !== JSON.stringify(manifest.histogram)) throw new Error('Histogram mismatch');
console.log(JSON.stringify({verifiedReady: manifest.entries.length, pending: recipes.length, histogram}, null, 2));
