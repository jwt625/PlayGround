# Cross-section fields for the "rods at twelve o'clock" keyframe

Generates `course/anims/fe_rods_fields_data.js` (about 117 KB), drawn by `course/anims/fe_rods_fields.js`
(`LMS_DRAW.fe_rodsFields(ctx, W, H, u, opts)`): RF field lines and |E_RF| of a G-S-G line on x-cut thin-film
LiNbO3, the TE0 and TM0 optical modes of the rib in one gap, and the overlap and V_pi numbers shown in the frame.

```
uv run --no-project --with numpy --with scipy python rods_fields.py all --cache <scratch>/cache
```

Stages `es`, `modes`, `eval`, `export` (or `all`); each result is cached as `.npz` / `.json` in `--cache` and skipped
if present (`--force` recomputes). Options: `--hes` electrostatic grid near the gap (default 0.02 um), `--h` mode grid
(default 0.025 um), `--win` mode window, `--lines` number of field lines (40). One process, two threads. Peak resident
memory: about 1.0 GB for the electrostatics at 0.02 um (3.2 GB at 0.01 um), 1.9 GB for the mode solve at 0.025 um.
Logs in `logs/` with a timestamp in the file name.

## Cross-section

Frame: x lateral (crystal z), y film normal (crystal x), z propagation (crystal y). Representative values in the range of
published x-cut thin-film LN traveling-wave modulators, not a specific device (for comparison: 600 nm film, 300 nm rib
height, 0.5 um gold, 6.5 um gap in Pan et al., Opt. Express 29, 17710 (2021); 4.7 um thermal oxide on silicon in
Li et al., Nat. Commun. 17, 1138 (2026) and Gao et al., Opt. Lett. 49, 5783 (2024)).

| | value |
|---|---|
| LN film / etch / slab | 600 nm / 300 nm / 300 nm, x-cut, waveguide along crystal y |
| rib | 1.0 um top width, 60 deg sidewalls (from the substrate plane), 1.35 um bottom width |
| buried oxide / substrate | 4.7 um SiO2 on 500 um Si; air above (no cladding) |
| electrodes | Au 0.8 um thick on the slab; signal 20 um, gaps 5 um, grounds 100 um; one rib centred in each gap |

## Materials

- RF permittivity, clamped (constant strain, high frequency): LN eps11 = eps22 = 44.3, eps33 = 27.9; Weis and Gaylord,
  Appl. Phys. A 37, 191 (1985), Table 1 (eps^S row of Warner, Onoe and Coquin 1967). Zhu et al., Adv. Opt. Photon. 13,
  242 (2021), Table 1 lists 44 / 27.9. SiO2 3.9 and Si 11.7 (Zhu et al. 2021, Table 1); air 1; gold a perfect conductor.
- Low-frequency set (unclamped, for the bench triangle sweep): eps11 85.2, eps33 28.7 (Weis and Gaylord, Table 1, eps^T,
  same row).
- Pockels coefficients (3m), 633 nm: clamped r33 30.8, r13 8.6, r22 3.4, r51 28 pm/V (Weis and Gaylord, Table 4, r^S
  row of Turner 1966); unclamped r33 32.2, r13 10.0, r22 6.81 pm/V and r51 32 pm/V (Table 4, r^T rows). No 1550 nm
  values in the source; dispersion of r is not applied.
- Optical: n_o 2.2111, n_e 2.1376 at 1550 nm (Zelmon, Small and Jundt, JOSA B 14, 3319 (1997)); SiO2 1.4440 (Malitson,
  JOSA 55, 1205 (1965)); air 1. Electrodes are left out of the optical problem (the rib centre is 2.5 um from the metal).

## Method

- RF: 2D quasi-static (electrostatic) finite-volume solve of div(eps grad V) = 0 on a non-uniform rectilinear grid
  (0.02 um within 6 to 19 um laterally and -2.5 to 3 um vertically, growing to 20 um), full line width
  -500..500 um by symmetry (mirror plane through the signal centre, so half is solved), Si bottom at -504.7 um, air to
  +500 um, zero normal flux on the outer boundary. Diagonal anisotropic eps per cell (4 x 4 sub-sampling, area
  weighted on the slanted sidewalls); signal 1 V, grounds 0 V. Field from the cell gradients. Capacitance from the
  stored energy; n_RF = sqrt(C / C0) and Z0 = 1 / (c sqrt(C C0)) with C0 from the same solve with all eps = 1.
- Optical: the course's full-vector finite-difference solver (`../modesolver/fdmode.py`: all three E components on a
  Yee grid, quadratic eigenproblem in beta) on an 8 x 5 um window, 0.025 um grid, PEC walls.
- Overlap (shown in the frame): Gamma = (g / V) * integral over LN of E_RF,z |E_opt,i|^2 dA / integral of
  |E_opt,i|^2 dA, with g = 5 um and i the dominant optical component (z for TE, x for TM; crystal axes).
- Index change per volt: first-order perturbation with the full 3m tensor and all three optical field components,
  dn = integral(E* . d_eps . E) / (4 P) with d_eps_ij = -eps_ii eps_jj d_eta_ij and P the Poynting power from H = curl E.
  Push-pull: the second arm is the mirror image with the opposite sign, V_pi L = lambda / (2 |dn_A - dn_B|).
- Field lines: RK4 along E, seeded on the signal electrode faces at equal increments of sqrt(eps) E_n (between E
  flux and D flux; D flux alone puts nearly all lines in the eps 44 slab under the metal, E flux alone puts most
  in the air above it). Decimated with Ramer-Douglas-Peucker (6 nm). Field lines leave the signal and end on the ground,
  normal to the metal faces.
- Export: |E_RF| on a 224 x 100 grid over the 11.2 x 5 um view, log10 from 0.01 to 1 V/um per volt, uint8 base64;
  optical |E| / max on the 0.025 um mode grid cropped to 5.2 x 3.2 um, uint8 base64.

## Results (es grid 0.02 um, mode grid 0.025 um)

| | RF (clamped) | low frequency (unclamped) |
|---|---|---|
| C, C0 (pF/m) | 146.71, 37.00 | 148.79, 37.00 |
| n_RF, Z0 (ohm) | 1.991, 45.3 | 2.005, 45.0 |
| n_eff TE0 / TM0 | 1.8936 (TE fraction 0.991) / 1.8606 (TE fraction 0.036) | same |
| Gamma TE / TM | 0.505 / 0.433 | 0.514 / 0.440 |
| V_pi L push-pull, TE / TM (V cm) | 2.28 / 6.39 | 2.16 / 5.65 |
| V_pi ratio TM / TE, full tensor | 2.80 | 2.62 |
| same, scalar estimate n_e^3 r33 Gamma_TE / (n_o^3 r13 Gamma_TM) | 3.78 | 3.40 |
| same, equal overlap n_e^3 r33 / (n_o^3 r13) | 3.24 | 2.91 |

The full-tensor ratio is lower than the scalar estimate because the TM mode is not purely x polarized: of its index
change, about 70 % comes from E_x (crystal) with r13, 16 % from its longitudinal component (crystal y, also r13),
10 % from its small crystal-z component with r33 and 5 % from r51 with the vertical RF field. The scalar estimate also
omits the n_material / n_eff power-normalisation factor, which differs for the two modes (2.138 / 1.894 vs
2.211 / 1.861). For TE, 97 % comes from r33.

## Checks

- Same geometry with the constants of an independent finite-element reference computation of this cross-section
  (P2 electrostatics and vector FEM modes; eps 44 / 28, r33 30.9, r13 9.6, r22 6.8, r51 32.6 pm/V), extrapolated
  reference values in brackets: C 146.81 pF/m (146.52), C0 37.00 (36.95), n_eff TE0 1.8936 (1.8942), TE fraction 0.991
  (0.991), Gamma TE 0.505 (0.501), dn per volt 1.714e-5 (1.701e-5), V_pi L 2.261 V cm (2.279).
- Electrostatic grid: C 147.02 / 146.81 / 146.73 pF/m at 0.05 / 0.02 / 0.01 um near the gap (check constants).
- Mode grid: TE0 1.89470 / 1.89386 / 1.89357 and TM0 1.85968 / 1.85961 / 1.86063 at 0.04 / 0.03 / 0.025 um; Gamma TE
  0.502 / 0.503 / 0.505, Gamma TM 0.435 / 0.430 / 0.433; full-tensor V_pi ratio (RF) 2.81 / 2.82 / 2.81.
- TM0 (1.861) is above the TE slab modes of the 300 nm slab (below 1.76 in this window), so it does not leak
  laterally into the slab.

## Caveats

- 2D quasi-static RF field: no skin effect, conductor or dielectric loss, or frequency dependence of the field
  pattern; Si treated as a lossless dielectric (no conductivity).
- Pockels coefficients at 633 nm used at 1550 nm. Whether a bonded thin film is clamped or unclamped at low frequency
  is not settled by the bulk tables; the two columns bracket it.
- Bulk Sellmeier indices for the film; designed (not fabricated) dimensions; metal absorption of the optical mode not
  included.
