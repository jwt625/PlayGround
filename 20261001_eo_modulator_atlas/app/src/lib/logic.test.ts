import { describe, expect, it } from 'vitest';
import atlasJson from '../../static/data/atlas.json';
import type { Atlas } from './types';
import { applyFilters, buildIndex, defaultFilters, fromHash, toHash, pickRep, sortRows, vpil, ilOnchip, ilF2f } from './logic';

const atlas = atlasJson as unknown as Atlas;
const index = buildIndex(atlas);

describe('integrated atlas data in the app', () => {
  it('selects a representative only from matching variants', () => {
    const f = { ...defaultFilters(), q: 'kohli2025', materials: ['barium_titanate'], classes: ['mzm'] };
    const view = applyFilters(atlas, f, index);
    expect(view.papers.map(p => p.paper_id)).toEqual(['kohli2025']);
    expect(view.reps.get('kohli2025')?.device_id).toBe('kohli2025-mzm');
  });
  it('keeps missing on-chip loss distinct from fiber-to-fiber loss', () => {
    const d = index.devById.get('kohli2025-mzm')!;
    expect(ilOnchip(d).v).toBeNull();
    expect(ilF2f(d).v).not.toBeNull();
    expect(vpil(d).derived).toBe(true);
  });
  it('uses the same default representative as the generated view', () => {
    for (const p of atlas.papers) expect(pickRep(index.devsByPaper.get(p.paper_id)!, 'default')?.device_id).toBe(p.rep.default);
  });
  it('round trips search, filters and multi-sort through the shareable hash', () => {
    const f = { ...defaultFilters(), q: 'Chen & TFLN', yearMin: 2020, hasSim: true, allDevices: true, materials: ['lithium_niobate'] };
    const extras = { sort: [{ key: 'year', dir: 'desc' as const }, { key: 'vpil', dir: 'asc' as const }] };
    expect(fromHash(toHash(f, extras))).toEqual({ filters: f, extras });
  });
  it('keeps unreported metrics last in both sort directions', () => {
    const rows = [{ value: null }, { value: 2 }, { value: 1 }];
    for (const dir of ['asc', 'desc'] as const) expect(sortRows(rows, [{ key: 'value', dir }], r => r.value).at(-1)?.value).toBeNull();
  });
});
