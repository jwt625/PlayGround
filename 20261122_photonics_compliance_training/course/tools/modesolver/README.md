# x-cut rib mode sweep (data for the "Rotate the x-cut die" widget)

Generates `course/widgets/xcut_modes_data.js`: effective index, TE fraction and field maps of the two lowest modes of
a thin-film lithium niobate rib on x-cut, as a function of the in-plane propagation angle.

```
uv run --no-project --with numpy --with scipy python validate.py            # checks, prints a table (about 3 min; one process, two threads)
uv run --no-project --with numpy --with scipy python sweep.py --profile thick --cache <scratch>/cache
uv run --no-project --with numpy --with scipy python sweep.py --profile thin --step 2 --cache <scratch>/cache
uv run --no-project --with numpy --with scipy python converge_crossing.py   # grid convergence of the closest approach
uv run --no-project --with numpy --with scipy python export.py --cache <scratch>/cache --out ../../widgets/xcut_modes_data.js
```

The sweep is resumable (one cached `.npz` per angle, skipped if present) and solves one angle at a time (one
process, two threads). `converge_crossing.py` peaked at about 2.4 GB resident memory on its finest (15 nm) grid. Logs go to `logs/` with a timestamp in the file name.

## Physics and formulation

- Coordinates: x lateral, y vertical (film normal = crystal x), z propagation. theta is the in-plane angle between the
  propagation direction and the crystal y axis. The c axis is c = cos(theta) x_hat + sin(theta) z_hat, so the
  relative permittivity is
  eps_xx = n_o^2 + (n_e^2 - n_o^2) cos^2 theta, eps_yy = n_o^2, eps_zz = n_o^2 + (n_e^2 - n_o^2) sin^2 theta,
  eps_xz = (n_e^2 - n_o^2) sin theta cos theta.
- eps_xz couples the lateral and the longitudinal field. The usual transverse-field formulations (linear in beta^2)
  rely on the longitudinal field being in quadrature with the transverse field, which this term breaks, so the
  solver keeps all three electric field components on a Yee grid and solves the resulting quadratic eigenvalue
  problem in beta exactly (companion linearization, shift-invert Arnoldi; one sparse LU of a 3N matrix per solve).
  The eps_xz block is applied with a symmetric two-point average between the staggered Ex and Ez positions, so the
  permittivity operator stays real symmetric. Details in the header of `fdmode.py`.
- Interfaces: horizontal interfaces on grid lines; sub-cell averaging (8 x 8 samples per dual cell; harmonic along
  the component direction, arithmetic across it; off-diagonal and longitudinal terms arithmetic) for the slanted
  sidewalls. Metal walls on the window edges, about 1 um or more from the mode (moving them changes the indices by
  less than 1e-6).
- Symmetry: theta and 180 - theta flip the sign of eps_xz, which is the mirror image of a left-right symmetric rib,
  so only 0 to 90 degrees are solved; the widget mirrors the fields for 90 to 180 degrees.

## Materials (1550 nm)

- LiNbO3, congruent, undoped: Zelmon, Small and Jundt, J. Opt. Soc. Am. B 14, 3319 (1997); n_o 2.21111, n_e 2.13756.
  Bulk values; thin-film indices can differ slightly.
- SiO2: Malitson, J. Opt. Soc. Am. 55, 1205 (1965); 1.44402. Air 1.

## Profiles (representative modelling choices, not a specific published device)

| | film | etch | slab | top width | sidewall (from substrate) | below | above |
|---|---|---|---|---|---|---|---|
| main ("thick") | 800 nm | 400 nm | 400 nm | 1.2 um | 65 deg | oxide | air |
| comparison ("thin") | 600 nm | 300 nm | 300 nm | 1.2 um | 65 deg | oxide | air |

The 600 nm / 300 nm rib was tried first. In it the TE-like fundamental mode stays above the TM-like one at every
angle (even along y, where TE sees n_e), because the geometric birefringence of the thinner film exceeds the material
birefringence. That agrees with Wang et al., Nature 629, 784 (2024), who state that mode mixing in x-cut LiNbO3 bends
at 1.55 um sets in above a film thickness of around 700 nm. The 800 nm film with a half etch is the main profile.
Wang et al. report a TE00 / TM00 crossing at 25 degrees for a 600 nm thick, 2 um wide waveguide of their own; their
slab, sidewall and cladding are not given in that passage, and this solver does not reproduce a crossing for 600 nm
films in the variants tried (slab 0 to 300 nm, width 0.8 to 2 um, air or oxide cover). The comparison with their
number is therefore qualitative only.

## Validation (see `logs/`)

- V1 isotropic three-layer slab vs analytic dispersion relation: error 1.3e-4 (20 nm grid), 8e-6 (5 nm), second order.
- V2 anisotropic x-cut slab, full tensor, at 0 / 30 / 60 / 90 deg vs a 4 x 4 transfer-matrix solution: agreement
  within 1.1e-5 at 5 nm grid, including the hybridized slab pair at 30 deg.
- V3 rib at 0 and 90 deg (diagonal tensor) vs an independent transverse-field formulation on the same grid: first
  two modes identical to 1e-7.
- V4 grid refinement 20 -> 10 nm: indices change by at most 1.4e-4; window size: below 1e-6.
- V5 theta and 180 - theta: identical indices.
- Closest approach of the two modes (`converge_crossing.py`): 30.96 / 29.89 / 30.25 / 30.21 deg and minimum splitting
  0.00271 / 0.00272 / 0.00270 / 0.00271 at 40 / 30 / 20 / 15 nm grid. The angle wobbles by up to about 0.7 deg on
  coarse grids (sidewall sampling) and settles near 30.2 deg; the shipped sweep (20 nm) puts it at 30.3 deg with a
  splitting of 0.0027.

## Result (main profile)

TE-like and TM-like fundamental modes anticross at about 30 deg from crystal y (and at 150 deg), minimum splitting
0.0027 in effective index; mode 1 is TM-like below the crossing and TE-like above it, and both are about half TE at
the closest approach. Along y the TM-like mode is higher (1.9912 vs 1.9711); along z the TE-like mode is higher
(2.0406 vs 1.9831).

## Caveats

- Bulk Sellmeier indices for the film; real thin films, etch damage and sidewall redeposition shift the numbers.
- Metal walls: a mode below the slab TE index would leak laterally in reality; both modes here stay well confined
  (more than 99 % of the field energy within 1.5 um of the rib centre at every angle), but no leakage loss is computed.
- Straight waveguide at each angle; bend radiation and the bend-induced shift of the crossing are not included.
- 1550 nm only; no material dispersion in the angle sweep.
