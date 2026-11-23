/* Mode hops from external feedback (feedback course, Module 4).
   Model: steady state of the Lang-Kobayashi equations (Lang and Kobayashi, IEEE J. Quantum Electron. 16, 347, 1980):
   (w_s - w_th) tau = -C sin(w_s tau + atan(alpha)), C = kappa (tau / tau_in) sqrt(1 + alpha^2),
   kappa = (1 - R1) / sqrt(R1) * sqrt(f_ext) (Fabry-Perot facet coupling, as written in Rogister, PhD thesis,
   Faculte Polytechnique de Mons 2001, arXiv:0909.4449, eqs. 2.145 and 2.156; tau_in = 8 ps and alpha = 4 from its Table 2-I).
   Modes: 1 + C cos(w_s tau + atan alpha) > 0; anti-modes (saddles) otherwise (Rogister eq. 2.170).
   Regime levels: Tkach and Chraplyvy, J. Lightwave Technol. 4, 1655 (1986), as summarized by Rogister section 2.6.
   InP index 3.165 at 1550 nm (Sellmeier fit to Pettit and Turner 1965); fiber group index 1.4682 (Corning SMF-28e+, 2014).
   Overrides the in-course widget of the same name. */
(function () {
  'use strict';
