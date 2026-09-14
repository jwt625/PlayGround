/** Browser-compatible canonical JSON and SHA-256, shared with offline builder. */
export const CACHE_VERSION = 'cache-1';
export function canonicalJSON(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) {
    if (Object.keys(value).length !== value.length) throw new TypeError('Sparse/decorated arrays are unsupported');
    return `[${value.map(canonicalJSON).join(',')}]`;
  }
  if (value && Object.getPrototypeOf(value) === Object.prototype) {
    return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonicalJSON(value[k])}`).join(',')}}`;
  }
  throw new TypeError('Cache keys require finite, plain JSON values');
}
export async function contentKey(value) {
  const bytes = new TextEncoder().encode(canonicalJSON(value));
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
}
export async function modeKey(problem) {
  for (const key of ['geometry', 'materials', 'wavelengthUm', 'requestedModes', 'solver', 'mesh', 'boundary']) {
    if (problem[key] === undefined) throw new TypeError(`Missing mode key input: ${key}`);
  }
  return contentKey({schemaVersion: CACHE_VERSION, kind: 'mode', problem});
}
export async function pairKey({modeA, modeB, transform, overlapVersion, integration}) {
  return contentKey({schemaVersion: CACHE_VERSION, kind: 'pair', modeA, modeB, transform, overlapVersion, integration});
}
