import { describe, expect, it } from 'vitest';
import atlasJson from '../../static/data/atlas.json';
import type { Atlas, Device } from './types';
import { derivedMetric, vpil, type Metric } from './logic';
import { buildPoints, gBw, gIl, gVpiIl, gVpil, groupStats, nominalFrontiers, type Pt } from './charts';
import { COLS, metricCell, tableCsv } from './columns';

const atlas = atlasJson as unknown as Atlas;
const papers = new Map(atlas.papers.map(p => [p.paper_id, p]));
const metric = (v: number | null): Metric => ({ v, qual: null, derived: false, basis: 'measured', field: null });
function device(): Device {
  const d = structuredClone(atlas.devices.find(d => d.device_id === 'chen2022-a')!);
  d.qualifiers = {};
  d.evidence = {};
  d.is_sim = false;
  d.vpi_dc_v = 2; d.length_mm = 10; d.il_onchip_db = 3; d.bw3db_ghz = 100;
  d.vpi_convention = 'mzm_push_pull';
  d.vpi_basis = d.il_basis = d.bw_basis = 'measured';
  d.headline_basis = { vpil: 'measured', bw3db: 'measured', il_onchip: 'measured' };
  d.vpil_dc_vcm = 2;
  d.vpil_best = { source: 'reported_dc', value: 2 };
  d.derived = {
    completeness: d.derived.completeness,
    vpi_il_vdb: { value: 6, unit: 'V*dB', formula: 'vpi_dc_v * il_onchip_db', inputs: ['vpi_dc_v', 'il_onchip_db'] },
    fom: { value: 25, unit: 'GHz/V', formula: 'bw3db_ghz / (vpi_dc_v * 10^(il_onchip_db/10))', inputs: ['bw3db_ghz', 'vpi_dc_v', 'il_onchip_db'] },
    vpil_dc_vcm_derived: { value: 2, unit: 'V*cm', formula: 'vpi_dc_v * length_mm / 10', inputs: ['vpi_dc_v', 'length_mm'] }
  };
  return d;
}

describe('qualified derived comparisons', () => {
  it('inverts denominator bounds and combines aligned directions for FOM', () => {
    const d = device();
    d.qualifiers = { vpi_dc_v: 'lt' };
    expect(derivedMetric(d, 'fom').qual).toBe('gt');
    d.qualifiers.bw3db_ghz = 'gt';
    expect(derivedMetric(d, 'fom').qual).toBe('gt');
    d.qualifiers = { il_onchip_db: 'gt' };
    expect(derivedMetric(d, 'fom').qual).toBe('lt');
  });
  it('does not invent a direction from opposing bounds or a bound and approximation', () => {
    const d = device();
    for (const qualifiers of [{ vpi_dc_v: 'lt', il_onchip_db: 'gt' }, { vpi_dc_v: 'gt', il_onchip_db: 'approx' }] as const) {
      d.qualifiers = { ...qualifiers };
      const m = derivedMetric(d, 'vpi_il_vdb');
      expect(m.boundUnresolved).toBe(true);
      expect(m.qual).toBeNull();
      expect(metricCell(atlas, 'vpi_il', d).text).toBe('6 (nominal)');
      expect(metricCell(atlas, 'vpi_il', d).qual).toBe('indeterminate');
      const chart = buildPoints([d], papers, () => metric(0), gVpiIl);
      expect(chart.pts).toEqual([]);
      expect(chart.omitted.uncertain).toBe(1);
    }
  });
  it('propagates bounds through derived VpiL instead of trusting the first generated qualifier', () => {
    const d = device();
    d.vpil_best = { value: 2, source: 'derived_dc', qualifier: 'lt' };
    d.qualifiers = { vpi_dc_v: 'lt', length_mm: 'gt' };
    expect(vpil(d).boundUnresolved).toBe(true);
    d.qualifiers = { vpi_dc_v: 'approx' };
    expect(vpil(d).qual).toBe('approx');
  });
  it('retains all audited Chen input qualifications in the FOM context', () => {
    const d = atlas.devices.find(d => d.device_id === 'chen2022-c')!;
    const c = metricCell(atlas, 'fom', d);
    expect(c.qual).toBe('indeterminate');
    expect(c.tip).toContain('bw3db_ghz:gt');
    expect(c.tip).toContain('il_onchip_db:approx');
    expect(c.tip).toContain('RF correction uses loss at 67 GHz');
  });
});

describe('chart validity and summary scope', () => {
  it('keeps the zero category coordinate and real zero on a linear axis', () => {
    const d = device();
    expect(buildPoints([d], papers, () => metric(0), gVpiIl, 0, { yLog: true }).pts).toHaveLength(1);
    d.il_onchip_db = 0;
    expect(buildPoints([d], papers, gVpil, gIl, 0, { xLog: true }).pts).toHaveLength(1);
    const log = buildPoints([d], papers, gVpil, gIl, 0, { xLog: true, yLog: true });
    expect(log.pts).toHaveLength(0);
    expect(log.omitted.nonpositive).toBe(1);
  });
  it('accounts separately for missing, invalid, and log-incompatible values', () => {
    const ds = Array.from({ length: 5 }, (_, i) => ({ ...device(), device_id: String(i) }));
    const values = [null, 0, NaN, 3, null];
    const result = buildPoints(ds, papers, d => metric(values[Number(d.device_id)]), d => metric(d.device_id === '4' ? null : 2), 0, { xLog: true });
    expect(result.omitted).toEqual({ total: 5, plotted: 1, missingX: 1, missingY: 0, missingBoth: 1, nonpositive: 1, invalid: 1, uncertain: 0 });
  });
  it('bases modelled markers on the plotted metrics and derived inputs', () => {
    const d = device(); d.is_sim = true; // an unrelated headline is modelled
    expect(buildPoints([d], papers, gBw, gVpil).pts[0].sim).toBe(false);
    d.bw_basis = 'simulated'; d.headline_basis.bw3db = 'simulated';
    expect(buildPoints([d], papers, gBw, gVpil).pts[0].sim).toBe(true);
    d.vpi_basis = 'simulated';
    expect(buildPoints([d], papers, () => metric(0), gVpiIl).pts[0].sim).toBe(true);
  });
  it('does not let bound thresholds dominate exact frontier points or join voltage contexts', () => {
    const pt = (id: string, x: number, y: number, extra: Partial<Pt> = {}): Pt => ({ id, x, y, group: 'lithium_niobate', sim: false, qx: null, qy: null, panel: 0, comparison: 'mzm_push_pull / DC', ...extra });
    const points = [pt('a', 50, 1), pt('b', 100, 2), pt('bound', 200, 0.1, { qx: 'gt' }), pt('approx', 300, 0.1, { qy: 'approx' }), pt('model', 400, 0.1, { sim: true }), pt('other', 80, 0.5, { comparison: 'per_arm_phase_shifter / DC' }), pt('unknown', 500, 0.1, { comparison: null })];
    expect(nominalFrontiers(points).map(f => f.map(p => p.id))).toEqual([['a', 'b'], ['other']]);
    const stats = groupStats(points);
    expect(stats[0].n).toBe(0); // different conventions and unknown context: no pooled summary
    const clean = groupStats(points.slice(0, 5));
    expect(clean[0]).toMatchObject({ n: 2, total: 5, min: 1, max: 2, median: 1.5 });
  });
  it('restores actual integrated material points and retains groups with bounds only', () => {
    const result = buildPoints(atlas.devices, papers, () => metric(0), gVpiIl, 0, { yLog: true });
    expect(result.pts.length).toBeGreaterThan(0);
    const onlyBound: Pt = { id: 'bound', x: 0, y: 2, group: 'lithium_niobate', qx: null, qy: 'lt', sim: false, panel: 0, comparison: 'mzm_push_pull / DC' };
    expect(groupStats([onlyBound])[0]).toMatchObject({ total: 1, n: 0, min: null, max: null, median: null });
  });
});

describe('CSV comparison context', () => {
  it('exports identity, filter status, voltage convention, bound, basis and measurement context', () => {
    const d = atlas.devices.find(d => d.device_id === 'chen2022-c')!;
    const paper = papers.get(d.paper_id)!;
    const csv = tableCsv(atlas, COLS.filter(c => ['bw3db', 'rf_loss', 'fom'].includes(c.id)), [{ kind: 'device', paper, dev: d, dim: true, isRep: true }]);
    expect(csv).toContain('paper_id,device_id,matches_filters,vpi_convention');
    expect(csv).toContain('chen2022,chen2022-c,false,mzm_push_pull');
    expect(csv).toContain('3 dB BW qualifier,3 dB BW basis,3 dB BW context');
    expect(csv).toContain(',67,gt,measured,');
    expect(csv).toContain('at 67 GHz');
    expect(csv).toContain('indeterminate,derived,');
    expect(csv).toContain('source:');
  });
});
