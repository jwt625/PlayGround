# R6-P1 status: isolated-output-port-v1 built; port transfer unresolved

> **Review correction (2026-09-19):** [Geometry repair and compute plan](R6P1_geometry_repair_and_compute_plan.md) identifies defects that supersede the physical interpretation below. The four-vertex oxide polygon overrides upstream substrate, so the nominal interaction is not preserved in the composed geometry. The downstream substrate boundary is not parallel to the guide, the guide terminates before PML, and the s=44/half-span=5 aperture intersects the transition. Repair these before a single coarse multiprobe diagnosis; do not interpret 60→0.006 as splice loss. At the quoted dimensions, resolution 50 has ~13M cells; ~29M corresponds to 75. Full-domain convergence is deferred, not waived.

## Implemented (geometry)

`configs/isolated_port.yaml` + `src/geometry.py` implement the named
continuation:

- nominal interaction preserved through s = 33 um, thickness 262 nm and
  guide angle 53.5 deg unchanged;
- downstream of s = 33 um the substrate boundary `q_sub(s)` moves with a
  smoothstep over an 8 um transition to a line parallel to the guide at
  3 um oxide clearance (`oxide_clearance_um`);
- substrate is now a polygon (not a half-space block) when `isolated_port`;
- guide/oxide continue to s = 58 um (uniform section ~17 um long) for the
  launch/monitor placement.
- Figure: `reports/figs/fig_geometry_isolated_port.png`.

The original reconstruction is retained via `isolated_port: false`.

## Measured (forward, Hz, res 25)

Port monitor at s = 44 um, half-span 5 um:

| quantity | value |
|---|---|
| isolated Hz guide n_eff (analytic) | 2.2858 |
| port band-1 n_eff | 2.3241 (delta 0.038) |
| band-1 forward power | 0.006 |
| same-aperture bands 2-6 forward | < 0.03 |

Two problems:

1. **Tiny guided signal at the port.** The nominal forward run gave band-1
   forward ~60 (0.70 % of Pinc) at s = 31; after the 8 um splice the band-1
   power at s = 44 is ~0.006. Either the guided mode is largely radiated at
   the transition, or the channel identity is lost, or the coupled signal
   never propagates that far. The probe at s = 31 vs s = 44 cannot be
   compared until the transition is validated.
2. **Identity metric not met.** Port n_eff 2.3241 vs isolated 2.2858 gives
   |delta| = 0.038, above the proposed 0.005; the guide channel identity at
   the port is therefore not established. A 3 um blanket oxide clearance may
   still be insufficient, or the band-1 mode at the port is a different
   branch.

## Cost

One forward port run at res 25 (cell 75.6 x 68.7 um) takes ~575 s, versus
~40 s for the nominal cell. R6-P2/P3 (forward+reverse, sensitivity for
transition 8/16, clearance 3/4, meshes 50/75) is therefore a large compute
allocation on this laptop; the 2D ladder at res 50 on this domain is itself
~30M cells.

## Requested direction

1. Should the continuation be made gentler (longer transition and/or a
   graded clearance) before judging port transfer, or is a large splice loss
   itself a reportable result?
2. Is there a preferred cheaper diagnostic to locate where the guided power
   is lost (probe at s = 34, 38, 41, 44 in one run) before committing to the
   full portable pair?
3. Confirm the compute budget policy for the res 50/75 convergence runs
   before I launch them.
