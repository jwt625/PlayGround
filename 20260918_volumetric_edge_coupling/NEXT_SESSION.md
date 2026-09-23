# Coding-session restart: complete the coarse C0/C1 diagnostic

Updated 2026-09-22 after review of 41afae9 / 7ac0c14 / fb00885. This is the current execution handoff. The orchestration reviewer changed documentation only and did not run simulations.

## Objective and authorization

Determine where the nominal-versus-isolated-port difference first appears: sampled geometry, incident illumination, upstream fields, feedback, or extraction. Complete the already authorized Hz C0/C1 pair at 25/µm and 1550 nm, sequentially within a 12 GiB peak-RSS operating budget, after the harness corrections below. Add actual-port full-field identity alongside the comparison. Do not wait for a complete band selector to acquire calibrated fields and net flux.

Preserve 262 nm, 53.5°, the source convention and the coupling region through s=33. No efficiency optimization or full-domain 50/75 ladder is authorized by this handoff. T02 remains independent. No new policy decision is needed for the listed repairs and coarse runs.

## Truthful starting state

- Shared oxide/substrate tessellation and the domain override exist.
- Compact res-60 isolated/port scalar profiles agree: Hz 2.2622, Ez 2.9230, overlap modulus 1.0000. This supports the compact test; it is not full-field identity at the actual res-25 coupler port.
- **The physical C0/C1 pair has not run.** `results/controlled_compare/{nominal,port}.npz` contains a res-8, `until_after_sources=1` smoke run. Preserve it with an explicit smoke label; never overwrite or use it as physical acceptance.
- Its global max dielectric difference 11.11 is not an interaction-region comparison. No conclusion follows until coordinates and spatial masks are applied correctly.
- `until_after_sources=1` is a time interval after the source ends, not one source period or the entire simulation duration. Record actual source/end times without changing the smoke run's non-acceptance status.
- The historic 104-versus-2.06 quantity is net aperture flux, not forward guide power; its broad s=31 aperture spans changed downstream material.

## Mandatory harness repairs before the two runs

The following were found in the current `scripts/controlled_compare.py` and geometry implementation:

| Gap | Required repair and verification |
|---|---|
| **Nominal guide ends at s=45 inside the common enlarged cell.** Geometry sets nominal s_port_end=33 and adds the default 12 µm extension; the later domain override changes cell bounds but not material endpoints. | Extend nominal guide/oxide, and port output materials, through the actual common PML. Verify composed dielectric at the cell exit. Keep nominal support geometry nominal; extending its endpoint must not insert the isolated-port transition. |
| **No incident reference is run.** `run_incident_reference` is imported but never called. | Save a homogeneous-Si reference using the actual common source/domain/grid/timestep/frequency. A shared reference is valid only after those inputs are checked identical. Save calibrated complex incident E/H on the common source-free Si plane and incident power. |
| **Flux/modal results are printed, not persisted.** `probe_out` is returned and printed; the NPZ contains only epsilon, regional fields and the trace. | Save native flux, frequency, candidate forward/backward powers, complex coefficients, wavevectors, monitor/eigensolver geometry and normalization. Label unvalidated modal identities as candidates. Persist results before running the next variant. |
| **Dielectric and DFT coordinates are conflated.** `eps` covers the whole cell, while saved x/y describe only the 45×40 µm DFT region. | Save separate coordinate arrays and quadrature metadata for each dataset, in Meep and/or explicitly transformed paper coordinates. Build s≤33 and full-aperture masks from the epsilon coordinates, not the DFT-region arrays. |
| **Downstream complex fields are absent.** The saved DFT region covers paper y=-12…28 µm; the s=50 port is below it. | Add localized Ex/Ey/Hz DFT storage on the upstream comparison apertures, Si reference plane and actual downstream port. Save the port dielectric and actual eigenmode fields needed for identity. Avoid enlarging a full-cell six-component map. |
| **The trace cannot support the planned phase/timing diagnosis.** It stores only abs(Hz) at impact every 2 Meep time units. | Record signed fields (complex if applicable) at source/incident reference, interaction, upstream probe and downstream port. Use a cadence resolving the ~1.55-time-unit carrier, e.g. ≤0.25, or a correctly demodulated envelope. Record convergence of the relevant complex DFT observables as well. |
| **Outputs are not immutable run records.** Fixed filenames overwrite previous data; effective settings/source versions/runtime/RSS are missing. | Use unique records for C0, C1 and the reference, including all CLI overrides, cell origin, resolution, branch, duration, actual timestep, smoothing, solver versions, code revision/dirty state, source and monitor definitions. Save status and measured peak RSS/runtime. |

Initialize and compare geometry **before** expensive time stepping. Include exact splice endpoints in the shared boundary tessellation and compare sampled dielectric in the source/beam region, preserved interaction and complete common upstream apertures. Report interface-localized differences separately from changes inside materials; the intentionally different downstream domain is not an error mask.

## Execution sequence

1. Complete the repairs and initialization-only checks in the common grid. Verify source support and material at its center, monitor/PML clearance and output continuation for both variants.
2. Prepare the matched incident reference. Keep acquisition at one frequency and save only the active Hz branch's Ex/Ey/Hz where needed. Validate saved coordinate/array shapes and reloadability before the long pair.
3. Run C0 nominal and C1 isolated-port, one at a time, initially at **25/µm, 1550 nm, until_after_sources=60**. Use broad **s=28, vertical half-span 6**, narrow **s=31, half-span 2.4**, and the actual isolated-port **s=50, half-span 2.52**. If a downstream C0 probe is included, label its different local support/identity explicitly. The two common upstream probes are the primary comparison.
4. Save data and release simulation resources between runs. Extend the integration if traces/DFT accumulation show incomplete settling; equal preset durations are not proof of convergence. Report measured cost rather than treating the estimated ~20 min for the pair as a guaranteed total including calibration.
5. Compute incident-normalized signed flux and compare complex fields/time histories. Separate early illumination differences from later returning fields; do not equate a net-flux change with forward guide attenuation. Use direction separation before attributing power to a reflected channel.
6. Complete actual-port identity at **res 25 and the same subcell registration**: matched isolated-reference complex E/H, fixed-window strip/substrate participation, squared full-field similarity and aperture stability. Follow the existing identity thresholds; keep the scalar compact-test overlap separate from this metric. A guide-index proximity rule may seed the search but cannot accept the channel.

## Completion and next decision

Deliver `reports/C0_C1_results.md` with links to immutable records, initialized-material difference maps with masks, calibrated incident-field comparison, the two common upstream fluxes, downstream candidate/identified-channel quantities, timing/DFT convergence, peak RSS/runtime and the actual-port identity table. Separate measurements from inferred causes.

C0/C1 is complete when these data localize the discrepancy or identify the specific remaining numerical dependency. A physical splice-loss or low-coupling conclusion is not required. If identity remains open, the calibrated net-flux diagnosis can still be reported, while guided-power and R6 reciprocity acceptance remain blocked.

Keep 50/75 full-domain convergence deferred. Memory profiling is part of these coarse runs, not a new automatic mesh escalation. Continue T02 page-5 provenance, descending-wavelength and exact-endpoint repairs, then digitization as independent work.

## Context documents

- [Controlled comparison and mode export](reports/R6P1_controlled_comparison_and_mode_export.md): comparison logic and the 2026-09-21 sequencing decision.
- [Port/channel/overlap contract](reports/R6_port_and_overlap_decision.md): full-field identity, Gaussian reciprocity and acceptance criteria.
- [Geometry/compute review](reports/R6P1_geometry_repair_and_compute_plan.md): numerical-budget rationale and known analysis edge cases.

The next coding agent should implement and execute this handoff. The orchestration agent remains responsible for review/planning/visual validation and does not write implementation code.
