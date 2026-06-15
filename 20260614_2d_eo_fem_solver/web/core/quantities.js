export const QUANTITIES = {
  phi: {
    label: "phi",
    description: "electrostatic potential",
    expression: "phi",
  },
  Ex: {
    label: "Ex",
    description: "x electric-field component",
    expression: "-d(phi)/dx",
  },
  Ey: {
    label: "Ey",
    description: "y electric-field component",
    expression: "-d(phi)/dy",
  },
  normE: {
    label: "|E|",
    description: "electric-field magnitude",
    expression: "sqrt(Ex^2 + Ey^2)",
  },
  mode: {
    label: "mode",
    description: "scalar optical modal field",
    expression: "psi",
  },
  mode_Ex: {
    label: "Ex",
    description: "scalar optical modal x component proxy",
    expression: "psi if mode_polarization == Ex else 0",
  },
  mode_Ey: {
    label: "Ey",
    description: "scalar optical modal y component proxy",
    expression: "psi if mode_polarization == Ey else 0",
  },
  mode_Ez: {
    label: "Ez",
    description: "scalar optical modal z component proxy",
    expression: "psi if mode_polarization == Ez else 0",
  },
  mode_normE: {
    label: "|E|",
    description: "scalar optical modal magnitude",
    expression: "abs(psi)",
  },
  mode_abs: {
    label: "|mode|",
    description: "scalar optical modal-field magnitude",
    expression: "abs(psi)",
  },
  mode_intensity: {
    label: "I",
    description: "scalar optical modal intensity",
    expression: "psi^2",
  },
  n: {
    label: "n",
    description: "selected optical refractive index",
    expression: "material.n_component || material.n || sqrt(eps_r)",
  },
  n_xx: {
    label: "n_xx",
    description: "optical refractive-index tensor xx component",
    expression: "material.n_xx || n",
  },
  n_yy: {
    label: "n_yy",
    description: "optical refractive-index tensor yy component",
    expression: "material.n_yy || n",
  },
  n_zz: {
    label: "n_zz",
    description: "optical refractive-index tensor zz component",
    expression: "material.n_zz || n",
  },
  eps_r: {
    label: "epsilon_r",
    description: "relative permittivity",
    expression: "material.eps_r",
  },
  eps_r_xx: {
    label: "epsilon_r_xx",
    description: "relative-permittivity tensor xx component",
    expression: "material.eps_r_xx || eps_r",
  },
  eps_r_yy: {
    label: "epsilon_r_yy",
    description: "relative-permittivity tensor yy component",
    expression: "material.eps_r_yy || eps_r",
  },
  eps_r_xy: {
    label: "epsilon_r_xy",
    description: "relative-permittivity tensor xy component",
    expression: "material.eps_r_xy || 0",
  },
  r13: {
    label: "r13",
    description: "EO tensor coefficient r13",
    expression: "material.r13 || 0",
  },
  r33: {
    label: "r33",
    description: "EO tensor coefficient r33",
    expression: "material.r33 || 0",
  },
  r22: {
    label: "r22",
    description: "EO tensor coefficient r22",
    expression: "material.r22 || 0",
  },
  r_eff: {
    label: "r_eff",
    description: "effective EO coefficient",
    expression: "material.r_eff || 0",
  },
};

export function quantityInfo(key) {
  return QUANTITIES[key] ?? { label: key, description: key, expression: key };
}
