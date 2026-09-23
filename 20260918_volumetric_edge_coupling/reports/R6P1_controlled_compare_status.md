# C0/C1 controlled comparison: infrastructure and status

> **Reviewer checkpoint (2026-09-22):** The physical pair remains unrun. [NEXT_SESSION.md](../NEXT_SESSION.md) is the restart handoff. Inspection found that the nominal material endpoint is not extended by the common-cell override, incident calibration is imported but unused, probe results are only printed, epsilon lacks its own saved coordinates, and the regional DFT map excludes the downstream port. Repair these before the pair. “Until after sources = 1” is not one source period; the existing run remains a non-acceptance smoke artifact.

## Implemented

- Common cell/grid override in `NumericsConfig`
  (`cell_sx_um`, `cell_sy_um`, `cell_center_x_um`, `cell_center_y_um`) applied
  in `src/geometry.py`.
- `scripts/controlled_compare.py`: runs nominal (`isolated_port: false`) and
  port (`isolated_port: true`) in the same cell/grid/source/time window, saves
  initialized dielectric, complex DFT fields over the interaction, signed flux
  and modal bands at broad s=28 (half-span 6), narrow s=31 (half-span 2.4),
  downstream s=50 (half-span 2.52), plus a Hz time trace at the impact point.
- `scripts/mode_export_port.py`: compact isolated/port mode export (validated,
  see `R6P1_mode_export_status.md`).

## Status: not yet interpreted

- A coarse sanity run of `controlled_compare.py` at res 8 / until 1 completed
  and wrote `results/controlled_compare/{nominal,port}.npz`, but this mesh and
  one-source-period window are not physical and must not be read as C0/C1.
- The dielectric `max|delta| = 11.11` over 221k cells in that run is
  **dominated by the downstream region**, where the two configurations
  legitimately differ (nominal wedge vs port continuation). It is not yet
  restricted to s<33 and is not evidence of an interaction-region defect.
- A separate ad-hoc epsilon probe was run without the common-cell override and
  is therefore invalid; disregard it.
- The C0/C1 runs at res 25 / until 60 have **not** been executed.

## Required before interpreting

1. Restrict the dielectric comparison to the interaction (s <= 33, and the
   full monitor/eigensolver volumes), on the common grid.
2. Run the res 25 / until 60 pair and compare incident field, s=28 and s=31
   signed flux, and modal bands; attribute any divergence to illumination,
   geometry, upstream field, feedback or extraction.
3. Keep the localization/full-field overlap at the actual port mesh alongside.
