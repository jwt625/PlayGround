// Cross-section runner shared by Node and the browser worker. No results are persisted.
import { parseConfig, opticalOptions } from './config.mjs';
import { CrossSection } from './section.mjs';
import { solveElectrostatics } from './electrostatics.mjs';
import { buildOpticalModel, solveModesAt, groupIndex } from './optics.mjs';

export function compareTargets(targets, metrics, { topology = 'unspecified', section = 'geometry', fixedGroupIndex = false } = {}) {
  return targets.map((target) => {
    const rf = ['n_rf', 'z0_ohm', 'c_pul_pf_per_m', 'c0_pul_pf_per_m', 'l_pul_nh_per_m'].includes(target.metric);
    let reason = null;
    if (section !== 'geometry') reason = 'Alternate cross-section; targets apply to the configured device.';
    else if (target.at_ghz != null) reason = 'Frequency-specific targets require the frequency-dependent line/response stages.';
    else if (rf && topology !== 'none') reason = topology === 'periodic_t_rail'
      ? 'Periodic-line target needs the loaded-line model; a single cross-section is not comparable.'
      : 'Line topology is unspecified; uniform-line target comparison requires line.loading.type: none.';
    else if (target.metric === 'ng_opt' && fixedGroupIndex) reason = 'Group index is a configured input, not an independent prediction.';
    else if (!Object.hasOwn(metrics, target.metric) || !Number.isFinite(metrics[target.metric])) reason = 'Metric was not computed by the requested cross-section stages.';
    const value = reason ? null : metrics[target.metric];
    const tolerance = Math.max(target.tol_abs ?? 0, Math.abs(target.value) * (target.tol_rel ?? 0));
    return { ...target, actual: value, tolerance, reason,
      status: value == null ? 'not_evaluated' : Math.abs(value - target.value) <= tolerance ? 'pass' : 'fail' };
  });
}

export function runCrossSection(text, { section: name = 'geometry', optical = false, meshScale = 1, onProgress = () => {} } = {}) {
  if (typeof optical !== 'boolean') throw new Error('optical must be a boolean');
  if (typeof meshScale !== 'number' || !Number.isFinite(meshScale) || meshScale < 0.5 || meshScale > 4) throw new Error('meshScale must be between 0.5 and 4');
  const config = parseConfig(text);
  if (!Object.hasOwn(config.geometries, name)) throw new Error(`Unknown cross-section: ${name}`);
  const section = new CrossSection(config.geometries[name], name);
  const opts = optical ? opticalOptions(config.raw) : null;
  const started = performance.now();
  onProgress('Meshing and solving electrostatics');
  const es = solveElectrostatics(section, { resolvedMaterials: config.materials, meshScale });
  const metrics = { c_pul_pf_per_m: es.cPul * 1e12, c0_pul_pf_per_m: es.c0Pul * 1e12, l_pul_nh_per_m: es.lPul * 1e9, n_rf: es.nRf, z0_ohm: es.z0 };
  const warnings = ['quasi_static_2d_rf', 'finite_domain_boundary', 'no_3d_launch_pad_effects'];
  const stages = ['electrostatics'];
  let opticalResult = null;
  if (optical) {
    onProgress('Meshing and solving the scalar optical mode');
    const model = buildOpticalModel(section, config.materials, { ...opts, meshScale });
    const modeIndex = opts.modeIndex;
    const lambda = opts.lambda0Um;
    let base;
    if (config.raw.optics.group_index_from === 'fixed') {
      base = solveModesAt(model, lambda, { numModes: modeIndex + 1 });
      metrics.ng_opt = config.raw.optics.ng_fixed;
      warnings.push('group_index_is_configured_not_predicted');
    } else {
      const g = groupIndex(model, lambda, modeIndex, { delta: Math.min(0.01, lambda / 100) });
      if (!g.lo.converged || !g.hi.converged) throw new Error('Optical group-index solves did not converge');
      base = g.base;
      metrics.ng_opt = g.ng;
    }
    if (!base.converged) throw new Error('Optical mode solve did not converge');
    metrics.n_eff = base.neff[modeIndex];
    opticalResult = { nNodes: model.mesh.nNodes, nTris: model.mesh.nTris, iterations: base.iterations, form: base.form };
    warnings.push('scalar_optical_not_full_vector');
    stages.push('optical_mode');
  }
  for (const [k, v] of Object.entries(metrics)) if (!Number.isFinite(v) || v <= 0) throw new Error(`Nonphysical result for ${k}: ${v}`);
  const topology = config.raw.line?.loading?.type ?? 'unspecified';
  const periodic = topology === 'periodic_t_rail';
  if (periodic) warnings.push('cross_section_only_not_periodic_line');
  const targets = compareTargets(config.raw.targets ?? [], metrics, { topology, section: name, fixedGroupIndex: config.raw.optics?.group_index_from === 'fixed' });
  const targetSummary = { total: targets.length, evaluated: targets.filter(t => t.status !== 'not_evaluated').length, passed: targets.filter(t => t.status === 'pass').length, failed: targets.filter(t => t.status === 'fail').length };
  return { id: config.raw.id, section: name, scope: 'cross_section', stages, metrics, targets,
    targetSummary,
    pendingStages: (config.raw.chain ?? []).filter((s) => !stages.includes(s)),
    warnings, meshScale, electrostatics: { ...es.stats, energyChargeRelativeError: Math.abs(es.capacitance.energy - es.capacitance.charge) / es.cPul },
    optical: opticalResult, elapsed_ms: performance.now() - started };
}
