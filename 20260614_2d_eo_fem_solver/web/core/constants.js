export const EPS0 = 8.8541878128e-12;

export function capacitanceUnits(cPerM) {
  return {
    F_per_m: cPerM,
    fF_per_mm: (cPerM * 1e12) / 1e3,
    pF_per_cm: (cPerM * 1e12) / 100.0,
  };
}
