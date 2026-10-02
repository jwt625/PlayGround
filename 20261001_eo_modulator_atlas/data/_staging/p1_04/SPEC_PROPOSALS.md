# SPEC proposals from batch p1_04

## valdez2022, valdez2023 (inductive slot slow-wave electrodes)
- `line.loading.type` accepts only `none | periodic_t_rail`. The slow-wave electrodes are inductively loaded by periodic rectangular recesses of the signal and ground edges (slot width w = 5 um, depth 4 um, period 25 um; valdez2022 p.6, valdez2023 p.5 Fig. 2). The configs use `periodic_t_rail` with the slotted section as the loaded cross-section and the protrusion section as `unloaded_cross_section`, which is geometrically a two-section cascade but not a T-rail. Proposal: add `periodic_slot` (or a generic `periodic_two_section`) with the same fields.
- The schema requires `eps_r` for every non-conductor material. These papers do not state RF permittivities, so the configs omit them (listed under `missing`) and are not schema-valid until supplied; same situation as sims/deng2026.

## meng2023
- ITO is a finite-conductivity conductor (400 S/m in the paper's simulation, 477 S/m measured on films). The config uses `conductor: true` with `sigma_Sm`. Proposal: state in SPEC whether the quasi-static RF solve treats `sigma_Sm` conductors as equipotentials (the ITO strips are thin and resistive, so equipotential is an approximation at high frequency) or as lossy sheets.

## renaud2023
- The 3 dB bandwidth is referenced to 3 GHz, not DC or 1 GHz. Proposal: allow a `reference_ghz` field on `bw3db_ghz` targets.
- Series of devices differing only in gap or wavelength (renaud2023 has eight rows sharing one cross-section family) have no variants mechanism; only the headline device is configured.
