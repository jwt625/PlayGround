// Material library access, dispersion, crystal-frame -> lab-frame rotation, Pockels tensor handling.
// Lab frame: x lateral, y film normal (up), z propagation.  Crystal axes 1,2,3 = x,y,z (c axis = 3).
// License: GPL-3.0-or-later.

import library from '../data/materials.json' with { type: 'json' };

export const MATERIAL_LIBRARY = library;

const AXIS = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
  '+x': [1, 0, 0],
  '+y': [0, 1, 0],
  '+z': [0, 0, 1],
  '-x': [-1, 0, 0],
  '-y': [0, -1, 0],
  '-z': [0, 0, -1],
};

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** 3x3 helpers (row-major arrays of 9). */
export const mat3 = {
  mul(A, B) {
    const C = new Array(9).fill(0);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[3 * i + j] += A[3 * i + k] * B[3 * k + j];
    return C;
  },
  T(A) {
    return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
  },
  diag(a, b, c) {
    return [a, 0, 0, 0, b, 0, 0, 0, c];
  },
  identity() {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  },
  apply(A, v) {
    return [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
  },
};

/**
 * Rotation A with v_lab = A v_crystal for a film cut (axis along lab y) and propagation (lab z).
 * lab_x = n x p so that (x,y,z) is right handed.
 * @param {string} cut crystal axis normal to the film (x|y|z, optional sign)
 * @param {string} propagation crystal axis of propagation
 * @param {number} [rotationDeg] additional rotation of the lab frame about z (propagation axis), CCW
 */
export function labFromCrystal(cut, propagation, rotationDeg = 0) {
  const n = AXIS[String(cut).toLowerCase()];
  const p = AXIS[String(propagation).toLowerCase()];
  if (!n || !p) throw new Error(`crystal cut/propagation must be one of x,y,z (optionally signed); got ${cut}, ${propagation}`);
  if (Math.abs(dot3(n, p)) > 1e-12) throw new Error('crystal cut and propagation axes must be orthogonal');
  const lx = cross(n, p);
  let A = [...lx, ...n, ...p];
  if (rotationDeg) {
    const th = (rotationDeg * Math.PI) / 180;
    const c = Math.cos(th),
      s = Math.sin(th);
    A = mat3.mul([c, -s, 0, s, c, 0, 0, 0, 1], A);
  }
  return A;
}

// ---- dispersion -------------------------------------------------------------------------------

function lagrange3(xs, ys, x) {
  const n = xs.length;
  let i = 0;
  while (i < n - 2 && xs[i + 1] < x) i++;
  if (i > n - 3) i = n - 3;
  if (i < 0) i = 0;
  const [x0, x1, x2] = [xs[i], xs[i + 1], xs[i + 2]];
  const [y0, y1, y2] = [ys[i], ys[i + 1], ys[i + 2]];
  return (
    (y0 * (x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2)) +
    (y1 * (x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2)) +
    (y2 * (x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
  );
}

/** Evaluate a library dispersion entry at wavelength (um). */
export function evalDispersion(entry, lambdaUm) {
  const [lo, hi] = entry.range_um;
  const inRange = lambdaUm >= lo && lambdaUm <= hi;
  let n;
  if (entry.model === 'sellmeier_c_squared' || entry.model === 'sellmeier_c') {
    const l2 = lambdaUm * lambdaUm;
    let s = 1;
    for (let i = 0; i < entry.B.length; i++) {
      const C2 = entry.model === 'sellmeier_c_squared' ? entry.C[i] * entry.C[i] : entry.C[i];
      s += (entry.B[i] * l2) / (l2 - C2);
    }
    n = Math.sqrt(s);
  } else if (entry.model === 'table') {
    n = lagrange3(entry.lambda_um, entry.n, lambdaUm);
  } else throw new Error(`unknown dispersion model ${entry.model}`);
  return { n, inRange };
}

/**
 * Library index n(lambda) for material key and component ('n' | 'n_o' | 'n_e').
 * Returns null when the library has no entry.
 */
export function libraryIndex(key, comp, lambdaUm) {
  const m = MATERIAL_LIBRARY.materials[key];
  const e = m?.optical?.[comp];
  if (!e) return null;
  return evalDispersion(e, lambdaUm);
}

// ---- Pockels ----------------------------------------------------------------------------------

const VOIGT = [
  [0, 0],
  [1, 1],
  [2, 2],
  [1, 2],
  [0, 2],
  [0, 1],
];

/**
 * Build 6x3 Voigt Pockels matrix (m/V) from a config entry. Accepts either
 *  - named independent coefficients r13, r22, r33, r51 (point groups 3m, 4mm, mm2 subset), pm/V, or
 *  - explicit `voigt` 6x3 array (pm/V).
 * Named-coefficient fill (3m with x,y,z = crystal axes, c = z):
 *  r11=r21 pattern: r12=-r22, r22, r32=r23? (see code): row(11)=[0,-r22,r13] row(22)=[0,r22,r13] row(33)=[0,0,r33]
 *  row(23)=[0,r51,0] row(13)=[r51,0,0] row(12)=[-r22,0,0].
 */
export function pockelsVoigt(spec) {
  if (!spec) return null;
  const pm = 1e-12;
  const r = new Float64Array(18);
  if (Array.isArray(spec.voigt)) {
    if (spec.voigt.length !== 6 || spec.voigt.some((row) => row.length !== 3)) throw new Error('r voigt must be 6x3');
    for (let i = 0; i < 6; i++) for (let k = 0; k < 3; k++) r[3 * i + k] = spec.voigt[i][k] * pm;
    return r;
  }
  const g = (k) => (spec[k] ?? 0) * pm;
  const r13 = g('r13'),
    r22 = g('r22'),
    r33 = g('r33'),
    r51 = g('r51');
  const set = (i, k, v) => (r[3 * i + k] = v);
  set(0, 1, -r22);
  set(0, 2, r13);
  set(1, 1, r22);
  set(1, 2, r13);
  set(2, 2, r33);
  set(3, 1, r51);
  set(4, 0, r51);
  set(5, 0, -r22);
  return r;
}

/** Delta(1/eps) 3x3 in the crystal frame for crystal-frame field Ec (V/m). */
export function deltaInvEpsCrystal(r, Ec) {
  const d = new Array(6).fill(0);
  for (let i = 0; i < 6; i++) d[i] = r[3 * i] * Ec[0] + r[3 * i + 1] * Ec[1] + r[3 * i + 2] * Ec[2];
  return [d[0], d[5], d[4], d[5], d[1], d[3], d[4], d[3], d[2]];
}

/**
 * First-order optical permittivity change in the lab frame for a lab-frame static field Elab (V/m):
 *   d(eps) = - eps * d(1/eps) * eps   (all in lab frame)
 * @param {Float64Array|number[]} r 6x3 Voigt (m/V), crystal frame
 * @param {number[]} A lab-from-crystal rotation
 * @param {number[]} epsOptLab 3x3 optical permittivity (n^2) in lab frame
 * @param {number[]} Elab
 */
export function deltaEpsLab(r, A, epsOptLab, Elab) {
  const Ec = mat3.apply(mat3.T(A), Elab);
  const dc = deltaInvEpsCrystal(r, Ec);
  const dl = mat3.mul(mat3.mul(A, dc), mat3.T(A));
  const t = mat3.mul(mat3.mul(epsOptLab, dl), epsOptLab);
  return t.map((v) => -v);
}

// ---- resolution of a config material -----------------------------------------------------------

/**
 * Resolve a config material into numerical tensors.
 * @param {string} name
 * @param {any} cfg material block from the sim config
 * @returns resolved material object
 */
export function resolveMaterial(name, cfg) {
  const out = {
    name,
    conductor: !!cfg.conductor,
    sigmaS: cfg.sigma_Sm ?? null,
    tanDelta: cfg.tan_delta_rf ?? 0,
    thicknessUm: cfg.thickness_um ?? null,
    A: mat3.identity(),
    hasCrystal: false,
    epsRfCrystal: null,
    nCfg: null,
    r: null,
    dispersionKey: cfg.dispersion ?? (MATERIAL_LIBRARY.materials[name] ? name : null),
    labels: [],
  };
  if (cfg.crystal) {
    out.A = labFromCrystal(cfg.crystal.cut, cfg.crystal.propagation, cfg.crystal.rotation_deg ?? 0);
    out.hasCrystal = true;
  }
  // RF permittivity (crystal frame diag)
  const e = cfg.eps_r;
  if (e === undefined || e === null) out.epsRfCrystal = out.conductor ? null : [1, 1, 1];
  else if (typeof e === 'number') out.epsRfCrystal = [e, e, e];
  else if (e.perp !== undefined && e.par !== undefined) out.epsRfCrystal = [e.perp, e.perp, e.par];
  else if (e.xx !== undefined) out.epsRfCrystal = [e.xx, e.yy, e.zz];
  else throw new Error(`material ${name}: eps_r must be a number or {perp, par}`);
  if (!out.hasCrystal && out.epsRfCrystal && !(out.epsRfCrystal[0] === out.epsRfCrystal[1] && out.epsRfCrystal[1] === out.epsRfCrystal[2])) {
    throw new Error(`material ${name}: anisotropic eps_r requires a crystal {cut, propagation}`);
  }
  // optical index (crystal frame diag)
  if (cfg.n !== undefined && cfg.n !== null) out.nCfg = { n_o: cfg.n, n_e: cfg.n, comp: 'n' };
  else if (cfg.n_o !== undefined && cfg.n_e !== undefined) out.nCfg = { n_o: cfg.n_o, n_e: cfg.n_e, comp: 'n_o_e' };
  else if (!out.conductor && cfg.eps_r !== undefined && cfg.n === undefined) out.nCfg = null;
  if (out.nCfg && out.nCfg.n_o !== out.nCfg.n_e && !out.hasCrystal) throw new Error(`material ${name}: n_o != n_e requires a crystal {cut, propagation}`);
  out.r = pockelsVoigt(cfg.r_pm_per_v);
  if (out.r && !out.hasCrystal) throw new Error(`material ${name}: Pockels coefficients require a crystal {cut, propagation}`);
  out.nComplex = cfg.n_complex ?? null;
  return out;
}

/** RF permittivity tensor in the lab frame (3x3) for a resolved material. */
export function epsRfLab(m) {
  if (!m.epsRfCrystal) return mat3.identity();
  const d = mat3.diag(...m.epsRfCrystal);
  return mat3.mul(mat3.mul(m.A, d), mat3.T(m.A));
}

/**
 * Optical index components (n_o, n_e) at wavelength lambdaUm, anchored at the configured value at lambda0
 * and following the library dispersion slope when available.
 * @returns {{n_o:number,n_e:number,dispersion:'library'|'library_anchored'|'none'}|null}
 */
export function opticalIndexAt(m, lambdaUm, lambda0Um) {
  if (m.conductor && !m.nCfg) return null;
  const lib = (comp) => (m.dispersionKey ? libraryIndex(m.dispersionKey, comp, lambdaUm) : null);
  const lib0 = (comp) => (m.dispersionKey ? libraryIndex(m.dispersionKey, comp, lambda0Um) : null);
  const comps = m.nCfg?.comp === 'n' ? ['n', 'n'] : ['n_o', 'n_e'];
  // library component names: isotropic library entries use `n`; crystals use n_o / n_e
  const get = (cfgVal, libComps) => {
    for (const lc of libComps) {
      const l = lib(lc);
      if (l) {
        const l0 = lib0(lc);
        if (cfgVal === null) return { n: l.n, mode: 'library', inRange: l.inRange };
        return { n: cfgVal + (l.n - l0.n), mode: 'library_anchored', inRange: l.inRange };
      }
    }
    return { n: cfgVal, mode: 'none', inRange: true };
  };
  if (!m.nCfg) {
    // fully from library
    const o = get(null, comps[0] === 'n' ? ['n'] : ['n_o', 'n']);
    const e = get(null, comps[0] === 'n' ? ['n'] : ['n_e', 'n']);
    if (o.n === null || e.n === null || Number.isNaN(o.n)) return null;
    return { n_o: o.n, n_e: e.n, dispersion: o.mode, inRange: o.inRange && e.inRange };
  }
  const o = get(m.nCfg.n_o, m.nCfg.comp === 'n' ? ['n'] : ['n_o', 'n']);
  const e = get(m.nCfg.n_e, m.nCfg.comp === 'n' ? ['n'] : ['n_e', 'n']);
  return { n_o: o.n, n_e: e.n, dispersion: o.mode, inRange: o.inRange && e.inRange };
}

/** Lab-frame optical permittivity (n^2) tensor. */
export function epsOptLab(m, lambdaUm, lambda0Um) {
  const idx = opticalIndexAt(m, lambdaUm, lambda0Um);
  if (!idx) return null;
  const d = mat3.diag(idx.n_o * idx.n_o, idx.n_o * idx.n_o, idx.n_e * idx.n_e);
  return mat3.mul(mat3.mul(m.A, d), mat3.T(m.A));
}
