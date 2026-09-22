# R6-P1: shared tessellation + isolated-port mode export

## Shared boundary tessellation (done)

`src/geometry.py` now samples oxide and substrate from one master vertex
sequence (`s_shared`); the oxide's 400 lower-boundary vertices are all present
in the substrate boundary (verified: 0 missing), so no overlap/air sliver can
form by construction.

## Isolated-port complex-mode export (D1, compact, no coupler run)

`scripts/mode_export_port.py` builds a short straight guide at the declared
53.5 deg orientation and mesh (`res 60/um`) for two stacks: the isolated
air/Si(262)/oxide reference and the port air/Si(262)/oxide(3um)/Si. The guide
band is launched and the complex field slice exported; the guide 1/e decay
length is ~0.14 um, so 3 um oxide isolates it.

| branch | isolated n_eff (band 1) | port n_eff | delta | profile overlap | analytic isolated |
|---|---|---|---|---|---|
| Hz | 2.2622 | 2.2622 | 0.0000 | 1.0000 | 2.2858 |
| Ez | 2.9230 | 2.9230 | 0.0000 | 1.0000 | 2.9598 |

- The port guide channel is identified (band 1, n_eff within 0.024 (Hz) and
  0.037 (Ez) of the analytic isolated value at res 60/um; the earlier matched
  straight-guide reference at res 100 was Hz 2.277, Ez 2.9624). The residual is
  finite-resolution, consistent with the known subpixel finding.
- Identical isolated/port indices confirm the port is effectively isolated at
  3 um clearance; no loading shift remains.
- Figures: `reports/figs/fig_port_mode_export.png` (Hz),
  `reports/figs/fig_port_mode_export_ez.png` (Ez).

Band identification here still uses index proximity as a first pass; field
localization remains the acceptance metric. The broad half-span-6 substrate
aperture is not used for this export.

## Still open (from the controlled-comparison handoff)

- The C0/C1 controlled comparison (nominal vs port in one common cell/grid,
  initialized dielectric maps, calibrated incident fields, broad s=28
  half-span-6 and narrow s=31 half-span-2.4 probes, complex fields and time
  traces) has not been run yet.
- The reason for the earlier net-flux difference (2.06 vs 104) is therefore
  still unresolved; it remains a net-aperture quantity, not forward guide
  power.
