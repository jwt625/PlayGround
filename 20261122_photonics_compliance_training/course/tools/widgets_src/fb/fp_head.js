/* Parasitic Fabry-Perot etalon (feedback course, Module 2).
   Model: two real interfaces with normal-incidence Fresnel reflectance R = ((n1 - n2) / (n1 + n2))^2,
   a gap of length L and group index n_g, and a single-pass power transmission eta in the gap.
   Lossy Airy transmission T = (1 - R1)(1 - R2) eta / (1 + rho^2 - 2 rho cos d), rho = sqrt(R1 R2) eta,
   d = 2 pi (nu - nu_peak) / FSR, FSR = c / (2 n_g L), ripple period in wavelength lambda^2 / (2 n_g L).
   Reflection seen from side 1: (R1 + R2 eta^2 - 2 sqrt(R1 R2) eta cos d) / (1 + rho^2 - 2 rho cos d).
   Finesse F = pi sqrt(rho) / (1 - rho) (the coefficient of finesse would be 4 rho / (1 - rho)^2).
   Indices at 1550 nm: fused silica 1.444 (Malitson 1965), silicon 3.48 (Li 1980), InP 3.165 (Pettit and
   Turner 1965 Sellmeier), congruent LiNbO3 extraordinary 2.138 (Zelmon, Small and Jundt 1997 Sellmeier),
   Si3N4 1.996 (Sellmeier of Luke et al., Opt. Lett. 40, 4823 (2015), 1.9963 at 1550 nm). Fiber group index 1.4682 at 1550 nm (Corning SMF-28e+ product information, 2014).
   Overrides the in-course widget of the same name. */
(function () {
  'use strict';
