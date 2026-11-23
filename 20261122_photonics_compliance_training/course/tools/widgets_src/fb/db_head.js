/* Backscatter doublet in a high-Q ring (feedback course, Module 6).
   Model: temporal coupled-mode theory for the clockwise and counter-clockwise modes with intrinsic loss kappa_0,
   bus coupling kappa_e (optional second bus), and backscatter coupling beta (Kippenberg, Spillane and Vahala,
   Opt. Lett. 27, 1669, 2002; Weiss et al., Opt. Lett. 20, 1835, 1995):
   da_cw/dt = (i Delta - kappa/2) a_cw + i beta a_ccw + sqrt(kappa_e) s_in,  da_ccw/dt = (i Delta - kappa/2) a_ccw + i beta a_cw.
   Steady state, D = kappa/2 - i Delta: t = 1 - kappa_e D / (D^2 + beta^2), |r|^2 = kappa_e^2 beta^2 / |D^2 + beta^2|^2,
   drop = kappa_e kappa_e2 |D|^2 / |D^2 + beta^2|^2, add-port backscatter = kappa_e kappa_e2 beta^2 / |D^2 + beta^2|^2.
   Overrides the in-course widget of the same name. */
(function () {
  'use strict';
