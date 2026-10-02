// Voltage / current waveform model. Units: t in ms, V in volts, I in nA, charge in pC (pF * V = pC, pC / ms = nA).
const WF = (() => {
  const TMAX = 10, DT = 0.002, N = Math.round(TMAX / DT) + 1;
  // Parameters read from the supplied figure, panel (c).
  const P = { Q: 90, Vmax: 560, t0: 1.5, t1: 2.7, t2: 3.5, t3: 8.5, C: 0.1, Ip: 280, tPk: 2.43, sig: 0.06, smooth: 0.05 };
  const V = new Float64Array(N), Ic = new Float64Array(N), Isw = new Float64Array(N), I = new Float64Array(N);
  const F = new Float64Array(N);

  const raw = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const t = i * DT;
    raw[i] = t < P.t0 ? 0 : t < P.t1 ? P.Vmax * (t - P.t0) / (P.t1 - P.t0) : t < P.t2 ? P.Vmax
      : t < P.t3 ? P.Vmax * (P.t3 - t) / (P.t3 - P.t2) : 0;
  }
  const w = Math.round(P.smooth / DT);
  for (let i = 0; i < N; i++) {
    let s = 0, n = 0;
    for (let k = -w; k <= w; k++) { const j = i + k; if (j >= 0 && j < N) { s += raw[j]; n++; } }
    V[i] = s / n;
  }
  for (let i = 0; i < N; i++) {
    const a = V[Math.max(0, i - 1)], b = V[Math.min(N - 1, i + 1)];
    Ic[i] = P.C * (b - a) / (2 * DT); // V/ms * pF = nA
  }
  const tau = P.Q / P.Ip - Math.sqrt(Math.PI / 2) * P.sig;
  let area = 0;
  for (let i = 0; i < N; i++) {
    const t = i * DT;
    Isw[i] = t < P.tPk ? P.Ip * Math.exp(-((t - P.tPk) ** 2) / (2 * P.sig ** 2)) : P.Ip * Math.exp(-(t - P.tPk) / tau);
    area += Isw[i] * DT;
  }
  const k = P.Q / area;
  for (let i = 0; i < N; i++) { Isw[i] *= k; I[i] = Ic[i] + Isw[i]; }
  let acc = 0;
  for (let i = 0; i < N; i++) {
    if (i > 0) acc += 0.5 * (Isw[i] + Isw[i - 1]) * DT;
    F[i] = Math.min(1, acc / P.Q);
  }
  F[N - 1] = Math.max(F[N - 1], F[N - 2]);

  const at = (arr, t) => {
    const x = Math.min(Math.max(t, 0), TMAX) / DT, i = Math.min(N - 2, Math.floor(x)), u = x - i;
    return arr[i] * (1 - u) + arr[i + 1] * u;
  };
  // Smallest t where F(t) >= f (f in (0,1]); Infinity if never reached.
  const tOfF = (f) => {
    if (f <= 0) return 0;
    if (F[N - 1] < f) return Infinity;
    let lo = 0, hi = N - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (F[m] >= f) hi = m; else lo = m; }
    const d = F[hi] - F[lo];
    return (lo + (d > 0 ? (f - F[lo]) / d : 0)) * DT;
  };
  const BASE = 0.625, SLOW = 8; // sim ms per wall second outside the switching window; slow-down factor inside
  const rate = (t) => {
    const wgt = Math.min(1, at(Isw, t) / (0.04 * P.Ip));
    return BASE / (1 + (SLOW - 1) * wgt);
  };
  return { TMAX, P, V: (t) => at(V, t), I: (t) => at(I, t), Ic: (t) => at(Ic, t), Isw: (t) => at(Isw, t),
    F: (t) => at(F, t), Q: (t) => at(F, t) * P.Q, tOfF, rate, arrays: { V, I } };
})();
