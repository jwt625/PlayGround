// Shared browser/CLI input boundary. Physical inputs are never silently filled in.
// Numerical meshing defaults are documented in sims/SPEC.md. GPL-3.0-or-later.
import { parseDocument } from 'yaml';
import { rectToPolygon } from './geometry.mjs';
import { resolveMaterial } from './materials.mjs';
import { normalizeWeights } from './electrostatics.mjs';

export const MAX_VERTICES = 80000;
export const STAGES = ['electrostatics', 'optical_mode', 'eo_overlap', 'rf_line', 'loaded_line', 'eo_response'];
export const TARGET_METRICS = ['vpi_l_dc_vcm', 'vpi_dc_v', 'n_eff', 'ng_opt', 'n_rf', 'z0_ohm', 'c_pul_pf_per_m', 'c0_pul_pf_per_m', 'l_pul_nh_per_m', 'rf_loss_db_per_cm', 'eo_rolloff_db', 'bw3db_ghz', 'bw6db_ghz', 'optical_confinement_in_region'];

const fail = (path, message) => { throw new Error(`${path}: ${message}`); };
const object = (v, p) => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) fail(p, 'expected an object');
  return v;
};
const number = (v, p, positive = false) => {
  if (typeof v !== 'number' || !Number.isFinite(v) || (positive && v <= 0)) fail(p, `expected a finite ${positive ? 'positive ' : ''}number`);
  return v;
};
const oneOf = (v, options, p) => {
  if (!options.includes(v)) fail(p, `expected one of ${options.join(', ')}`);
  return v;
};
const interval = (v, p) => {
  if (!Array.isArray(v) || v.length !== 2) fail(p, 'expected [min, max]');
  v.forEach((x) => number(x, p));
  if (v[1] <= v[0]) fail(p, 'max must exceed min');
  return [...v];
};
const keys = (v, allowed, p) => {
  object(v, p);
  for (const k of Object.keys(v)) if (!allowed.includes(k)) fail(`${p}.${k}`, 'unsupported field');
};
const array = (v, p) => {
  if (!Array.isArray(v)) fail(p, 'expected an array');
  return v;
};
const window = (v, p) => {
  keys(v, ['x_um', 'y_um'], p);
  return { x: interval(v.x_um, `${p}.x_um`), y: interval(v.y_um, `${p}.y_um`) };
};

function shape(v, p) {
  if (!!v.rect === !!v.polygon_um) fail(p, 'provide exactly one of rect or polygon_um');
  if (v.rect) {
    const r = window(v.rect, `${p}.rect`);
    return rectToPolygon(r.x, r.y);
  }
  if (!Array.isArray(v.polygon_um) || v.polygon_um.length < 3) fail(p, 'polygon requires at least three points');
  const poly = v.polygon_um.map((pt) => {
    if (!Array.isArray(pt) || pt.length !== 2) fail(p, 'polygon point must be [x, y]');
    return pt.map((x) => number(x, p));
  });
  const area2 = poly.reduce((a, q, i) => { const r = poly[(i + 1) % poly.length]; return a + q[0] * r[1] - r[0] * q[1]; }, 0);
  if (Math.abs(area2) < 1e-12) fail(p, 'polygon has zero area');
  return poly;
}

export function normalizeGeometry(raw, materials, path = 'geometry') {
  keys(raw, ['symmetry', 'mirror_x_um', 'domain', 'regions', 'electrodes', 'optical_window', 'mesh', 'boundary'], path);
  const domain = window(raw.domain, `${path}.domain`);
  const symmetry = oneOf(raw.symmetry ?? 'none', ['none', 'x_mirror_odd', 'x_mirror_even'], `${path}.symmetry`);
  const mirror = number(raw.mirror_x_um ?? 0, `${path}.mirror_x_um`);
  if (symmetry !== 'none' && (mirror < domain.x[0] || mirror > domain.x[1])) fail(path, 'mirror plane must meet the domain');
  const entities = (rows, kind) => {
    if (!Array.isArray(rows) || !rows.length) fail(`${path}.${kind}`, 'expected a non-empty array');
    const names = new Set();
    return rows.map((r, i) => {
      const p = `${path}.${kind}[${i}]`;
      object(r, p);
      if (typeof r.name !== 'string' || !r.name || names.has(r.name)) fail(p, 'name must be non-empty and unique');
      names.add(r.name);
      if (!Object.hasOwn(materials, r.material)) fail(p, `unknown material ${r.material}`);
      if (kind === 'regions' && materials[r.material].conductor) fail(p, 'conductors must be electrodes');
      if (kind === 'electrodes' && !materials[r.material].conductor) fail(p, 'electrode material must be a conductor');
      return { ...r, poly: shape(r, p), ...(kind === 'electrodes' ? { weight: number(r.weight, `${p}.weight`) } : {}) };
    });
  };
  const mesh = { ...object(raw.mesh ?? {}, `${path}.mesh`) };
  keys(mesh, ['max_edge_um', 'electrode_edge_um', 'electrode_face_um', 'window_edge_um', 'far_edge_um', 'min_edge_um', 'grade', 'grade_far', 'grade_corner', 'min_angle_bound', 'max_vertices', 'regions'], `${path}.mesh`);
  for (const [k, v] of Object.entries(mesh)) {
    if (k === 'regions') { Object.values(object(v, `${path}.mesh.regions`)).forEach((x) => number(x, `${path}.mesh.regions`, true)); }
    else number(v, `${path}.mesh.${k}`, true);
  }
  // Explicit resource ceiling for an interactive solve; never silently coarsen the mesh.
  mesh.max_vertices ??= MAX_VERTICES;
  if (!Number.isInteger(mesh.max_vertices) || mesh.max_vertices < 4 || mesh.max_vertices > MAX_VERTICES) fail(`${path}.mesh.max_vertices`, `expected an integer from 4 to ${MAX_VERTICES}`);
  if (mesh.min_angle_bound != null && mesh.min_angle_bound < 1) fail(`${path}.mesh.min_angle_bound`, 'radius/edge bound must be at least 1');
  const boundary = { left: 'neumann', right: 'neumann', top: 'neumann', bottom: 'neumann', ...object(raw.boundary ?? {}, `${path}.boundary`) };
  keys(boundary, ['left', 'right', 'top', 'bottom'], `${path}.boundary`);
  Object.entries(boundary).forEach(([k, v]) => oneOf(v, ['neumann', 'dirichlet'], `${path}.boundary.${k}`));
  const optical_window = raw.optical_window ? window(raw.optical_window, `${path}.optical_window`) : null;
  if (optical_window && ['x', 'y'].some((k) => optical_window[k][0] < domain[k][0] || optical_window[k][1] > domain[k][1])) fail(path, 'optical_window must lie within the domain');
  const regions = entities(raw.regions, 'regions');
  const electrodes = entities(raw.electrodes, 'electrodes');
  normalizeWeights(electrodes, symmetry);
  for (const name of Object.keys(mesh.regions ?? {})) if (!regions.some(r => r.name === name)) fail(`${path}.mesh.regions`, `unknown region ${name}`);
  return { domain, symmetry, mirror_x_um: mirror, regions, electrodes, optical_window, mesh, boundary };
}

export function parseConfig(text) {
  if (typeof text !== 'string' || text.length > 1000000) fail('config', 'expected YAML text of at most 1,000,000 characters');
  const doc = parseDocument(text, { uniqueKeys: true });
  if (doc.errors.length) throw new Error(`YAML: ${doc.errors.map((e) => e.message).join('; ')}`);
  return normalizeConfig(doc.toJS({ maxAliasCount: 100 }));
}

export function normalizeConfig(raw) {
  object(raw, 'config');
  if (raw.schema !== 'eo-atlas.sim/v1') fail('schema', 'expected eo-atlas.sim/v1');
  if (typeof raw.id !== 'string' || !raw.id) fail('id', 'expected a non-empty string');
  if (raw.chain != null) array(raw.chain, 'chain').forEach((s, i) => oneOf(s, STAGES, `chain[${i}]`));
  if (raw.repro_grade != null) oneOf(raw.repro_grade, ['A', 'B'], 'repro_grade');
  if (raw.validation_status != null) oneOf(raw.validation_status, ['unvalidated', 'analytic_gates_pass', 'literature_regression', 'cross_solver'], 'validation_status');
  if (raw.missing != null) array(raw.missing, 'missing');
  if (raw.limitations != null) array(raw.limitations, 'limitations').forEach(v => { if (typeof v !== 'string') fail('limitations', 'expected strings'); });
  if (raw.provenance != null) for (const [key, p] of Object.entries(object(raw.provenance, 'provenance'))) {
    object(p, `provenance.${key}`);
    oneOf(p.class, ['paper_exact', 'figure_digitized', 'project_inference', 'standard_reference', 'unknown'], `provenance.${key}.class`);
  }
  const materials = Object.create(null);
  for (const [name, m] of Object.entries(object(raw.materials, 'materials'))) {
    const p = `materials.${name}`;
    object(m, p);
    if (m.conductor != null && typeof m.conductor !== 'boolean') fail(`${p}.conductor`, 'expected a boolean');
    if (!m.conductor) {
      const e = m.eps_r;
      if (typeof e === 'number') number(e, `${p}.eps_r`, true);
      else if (e?.perp !== undefined && e?.par !== undefined) { keys(e, ['perp', 'par'], `${p}.eps_r`); number(e.perp, `${p}.eps_r.perp`, true); number(e.par, `${p}.eps_r.par`, true); }
      else if (e?.xx !== undefined && e?.yy !== undefined && e?.zz !== undefined) { keys(e, ['xx', 'yy', 'zz'], `${p}.eps_r`); Object.values(e).forEach((v) => number(v, `${p}.eps_r`, true)); }
      else fail(`${p}.eps_r`, 'RF permittivity is required; unknown is not vacuum');
    }
    for (const k of ['n', 'n_o', 'n_e', 'sigma_Sm']) if (m[k] != null) number(m[k], `${p}.${k}`, true);
    if ((m.n_o != null) !== (m.n_e != null)) fail(p, 'n_o and n_e must be provided together');
    if (m.n != null && m.n_o != null) fail(p, 'provide either n or n_o/n_e, not both');
    materials[name] = resolveMaterial(name, m);
  }
  const geometries = Object.create(null);
  geometries.geometry = normalizeGeometry(raw.geometry, materials);
  for (const [name, g] of Object.entries(object(raw.alt_geometries ?? {}, 'alt_geometries'))) {
    if (name === 'geometry' || name === '__proto__') fail('alt_geometries', 'reserved geometry name');
    geometries[name] = normalizeGeometry(g, materials, `alt_geometries.${name}`);
  }
  if (raw.optics != null) opticalOptions(raw);
  if (raw.line != null) {
    object(raw.line, 'line');
    if (raw.line.differential != null && typeof raw.line.differential !== 'boolean') fail('line.differential', 'expected a boolean');
    const loading = raw.line.loading;
    if (loading != null) {
      object(loading, 'line.loading');
      oneOf(loading.type, ['none', 'periodic_t_rail'], 'line.loading.type');
      if (loading.type === 'periodic_t_rail') {
        number(loading.period_um, 'line.loading.period_um', true);
        number(loading.loaded_length_um, 'line.loading.loaded_length_um', true);
        if (loading.loaded_length_um > loading.period_um) fail('line.loading.loaded_length_um', 'cannot exceed the period');
        for (const k of ['loaded_cross_section', 'unloaded_cross_section']) {
          const name = k === 'loaded_cross_section' ? loading[k] ?? 'geometry' : loading[k];
          if (typeof name !== 'string' || !Object.hasOwn(geometries, name)) fail(`line.loading.${k}`, 'must reference an existing cross-section');
        }
      }
    }
  }
  for (const [i, t] of array(raw.targets ?? [], 'targets').entries()) {
    const p = `targets[${i}]`;
    object(t, p);
    number(t.value, `${p}.value`);
    oneOf(t.metric, TARGET_METRICS, `${p}.metric`);
    if (t.at_ghz != null) number(t.at_ghz, `${p}.at_ghz`, true);
    if (t.tol_abs == null && t.tol_rel == null) fail(p, 'tolerance is required');
    for (const k of ['tol_abs', 'tol_rel']) if (t[k] != null && number(t[k], `${p}.${k}`) < 0) fail(p, 'tolerance cannot be negative');
    if (!Number.isFinite(Math.max(t.tol_abs ?? 0, Math.abs(t.value) * (t.tol_rel ?? 0)))) fail(p, 'tolerance must remain finite');
  }
  return { raw, materials, geometries };
}

export function opticalOptions(raw) {
  const o = object(raw.optics, 'optics');
  number(o.wavelength_nm, 'optics.wavelength_nm', true);
  const polarization = oneOf(o.polarization, ['TE', 'TM', 'quasi-TE', 'quasi-TM'], 'optics.polarization').replace('quasi-', '');
  const modeIndex = number(o.mode_index ?? 0, 'optics.mode_index');
  if (!Number.isInteger(modeIndex) || modeIndex < 0 || modeIndex > 4) fail('optics.mode_index', 'expected an integer from 0 to 4');
  oneOf(o.group_index_from, ['fixed', 'finite_difference_wavelength'], 'optics.group_index_from');
  if (o.group_index_from === 'fixed') number(o.ng_fixed, 'optics.ng_fixed', true);
  return { polarization, lambda0Um: o.wavelength_nm / 1000, modeIndex };
}
