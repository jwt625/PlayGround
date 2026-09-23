# R6-P1: controlled comparison and mode export

## Execution update — 2026-09-21, after fb00885

**Proceed with the two coarse Hz C0/C1 runs after the common-domain initialization checks. Do not wait for a complete localization-based band selector.** C0/C1's primary diagnostic is calibrated fields and net aperture flux, which does not depend on accepting a guide-band identity. Complete the identity metric alongside this work, before labeling any modal coefficient as accepted receiving-guide power or starting reciprocal launch acceptance.

The shared master vertex sequence now addresses the earlier oxide/substrate tessellation mismatch. Still include the exact splice joins in that sequence and verify the initialized dielectric on the common grid; shared vertices remove the interface-sliver problem but do not by themselves certify every sampled geometry detail.

### Interpretation of the new compact mode figures

The inspected Hz and Ez plots show coincident localized profiles in the compact reference and port-stack tests. Accept this as **matched compact-test agreement**. It supports the proposed isolated-port construction, but does not finish identity validation in the full coupler:

- `mode_export_port.py` builds independent rotated blocks centered at the origin. Matching orientation and mesh between these two tests does not automatically match the actual coupler's subcell registration, sampled dielectric, aperture or port position.
- Its reported overlap is the modulus of a scalar complex inner product of propagated Hz or Ez DFT slices. It is not the prescribed squared full-E/H identity metric, a confinement measurement or a power overlap. Index proximity still chooses the port candidate and isolated band 1 is assumed as the reference.
- The narrow local eigensolver apertures see matching air/strip/oxide sections and exclude the distant substrate. Equal computed indices are expected for identical local dielectric inputs; by themselves they do not measure loading/leakage of the complete substrate-containing propagation problem. The finite-run profile agreement is useful additional evidence, not a convergence bound.
- The 60/µm residuals from the analytic indices are consistent with numerical error, but one resolution does not establish their cause or convergence. The next propagation pair uses 25/µm, so numerical reference identity must be checked at that resolution and registration too.

Do not require another full-domain run to resolve these qualifications. Use local actual-port eigenmode export and the fields saved by C0/C1.

### Immediate implementation and acceptance

1. Add the small explicit common-domain override, including common origin/grid registration and a recorded transform from paper coordinates. Apply it consistently to geometry, sources, monitors and homogeneous reference. Extend **both** variants' uniform output materials through the common PML; a larger empty cell around a short nominal guide would create a new facet.
2. Initialize both and save sampled-material comparisons at the source, beam path, preserved interaction and complete upstream apertures. Include exact join vertices in the shared tessellation. Confirm source frequency/branch, aperture coordinates and timestep from effective settings.
3. Run C0/C1 at 25/µm, Hz, 1550 nm, with broad s=28/half-span=6, narrow s=31/half-span=2.4, and the narrow downstream port. Keep the old broad s=31 aperture only as a separately labeled diagnostic if useful.
4. Save complex **Ex, Ey and Hz**, native signed flux, time traces, DFT convergence and actual sampled port dielectric. Preserve raw complex candidate modal coefficients, wavevectors and eigenmode fields where available. Use each run's calibrated incident reference, or one shared reference only when all source/domain/discretization settings match. Put all effective settings and overrides in immutable records.
5. In the actual uniform-port frame at 25/µm, add the fixed-window strip/substrate participation and full complex-field similarity to a matched isolated reference. Use the prior window/aperture checks and identity criteria; keep index proximity only as a search seed. This step can follow field acquisition, but must precede a guided-power or reciprocity claim.

**C0/C1 is done when:** the comparison identifies whether the discrepancy first appears in illumination, initialized geometry, upstream fields, returning fields, or extraction; incomplete temporal convergence is reported explicitly. A missing modal identity need not prevent finishing this net-flux diagnostic. Record modal quantities as candidates until the separate identity gate passes.

The bounded allocation remains two coarse runs plus the shared incident reference/local initialization and eigenmode checks, sequentially on the local host. No 50/75 full-domain ladder, dense polarization scan or geometry optimization is needed. T02 continues independently.

---

Review date: 2026-09-19. Inspected commits f2364bd/20f161d, `port_multiprobe.py`, current geometry/source/port implementations and the geometry-repair report. No implementation changes or simulations were made by this review.

**Direction:** pursue measurement/comparison isolation first, with initialized-dielectric verification as its first cheap check. Implement complex mode-field export now, initially at a genuinely isolated downstream port. Do not expand the blind band scan or change the transition to recover efficiency. Continue the existing coarse-compute policy and independent T02 work.

There is no defensible probability estimate for geometry versus normalization from the current scalar readings. There are, however, specific observable/aperture confounds that can be removed before another physical interpretation. The approved bounded work below does not require another policy stop.

## 1. The claimed 50x forward-power discrepancy is not yet that observable

`port_multiprobe.py` prints `mp.get_fluxes(f)[0]` as `native`. This is **signed net flux across a vertical aperture**, not forward receiving-guide power. The reported 2.057 and ~104 must retain that label. A decrease in net flux can result from reduced forward transport, increased backward transport, radiation redistribution, or their interference in the measurement region.

The monitor centered at s=31 has vertical half-span 6 µm. Since sidewall coordinate changes by -sin(alpha) times vertical displacement, it covers **s=26.1008–35.8992 µm**. Its lower portion therefore samples the modified region beyond s=33. Even a perfectly correct material-preservation check through s=33 does not establish identical geometry across this aperture or its eigenmode-solving volume.

Moreover, downstream loading/reflection can change upstream steady-state fields, even at a truly upstream port. The statement that a 50x drop “cannot come from the downstream continuation” is not justified. The magnitude alone establishes neither a physical explanation nor a bug. Comparing early-time fields before a possible return and later fields is a useful discriminator, with the return-time bound based on actual propagation paths.

Keep the broad aperture as a total-transport diagnostic. Add two controlled probes in both configurations:

- **s=28, vertical half-span 6 µm:** its maximum s is 32.8992, wholly before the intended splice. This preserves the broad collection geometry while moving it upstream.
- **s=31, vertical half-span 2.4 µm:** its maximum s is 32.9597, also before the splice. This is a narrower local transport diagnostic; it is not interchangeable with the broad reading and still contains substrate near the loaded guide.

Record the material map over every complete aperture and eigensolver volume, rather than checking only the center. A broad minus narrow flux difference is an aperture-dependent quantity, not an automatic measure of cladding loss.

## 2. Status of G1–G4 and cheap geometry verification

The former four-vertex oxide chord is gone. The corrected downstream q_sub expression has the guide's slope, and the domain/extension separation addresses the previous terminal-facet construction. These are meaningful repairs. They do not establish the sampled Meep materials or PML behavior without the requested maps/checks.

One remaining source-level discrepancy: the two materials use the **same analytic function but different sampled polylines**. Oxide uses 400 points from its start to s_absorb, while substrate uses 600 points from -80 to s_absorb+20. On the curved transition their piecewise-linear boundaries need not coincide; neither grid guarantees exact vertices at s=33 and the transition end. Use a shared master vertex sequence over the common boundary, with exact joins, so overlap/air slivers are excluded by construction. This is a concrete repair, not evidence that it explains the 50x flux difference.

The aperture guidance was recorded but was not applied to the reported half-span-6 eigenmode calculations. For example, s=44 with that aperture begins at s=39.1008 and still intersects the transition. G4 remains open for those measurements.

Before time stepping, initialize both models and compare the actual sampled dielectric/material tensors in the source, beam path, interaction and complete common upstream apertures. Use a common paper-to-Meep coordinate transform, cell, grid origin, resolution and subpixel policy. A point-in-polygon check on analytic vertices is complementary evidence, not a check of solver initialization. Include representative points strictly inside each material, not only interface points.

Check the material at the Gaussian source center and along its support. The current focus offset `focus - source_center` is consistent with Meep's documented **relative** `beam_x0` convention; do not change it merely because the enlarged domain recenters the computational coordinates. Meep also derives beam propagation from the material at the source center. [Gaussian beam source definition](https://meep.readthedocs.io/en/latest/Python_User_Interface/#gaussianbeam3dsource).

## 3. Minimal controlled comparison

Use **one common enlarged cell and mesh registration** for the following pair. The control must keep its straight nominal guide/materials extended into that cell's PML; reusing its old, shorter material endpoint would introduce a new facet.

| Run | Physical structure | What it isolates |
|---|---|---|
| C0 | Nominal straight-sidewall reconstruction, continued to PML in the common enlarged cell | Baseline with the same domain, source placement, grid and monitors as the port variant. |
| C1 | Corrected isolated-port continuation in that same cell | Effect of the downstream change after holding numerical domain and illumination fixed. |

First compare C0 with the old nominal result as a **domain/registration check**, without assuming they must have identical raw flux. If the difference already appears there, do not attribute it to the splice. A block-versus-planar-polygon representation check can be done at initialization first; perform another propagation control only if the sampled-material or field comparison warrants it.

For C0/C1:

1. Save actual branch, source frequency/pulse parameters, source-center material, source/monitor coordinates in both frames, timestep, cell dimensions, grid origin, smoothing, end time and all CLI overrides. Assert the actual monitor frequency equals 1/1.55 µm. Use each monitor's frequency list, not an assumed index convention.
2. Calibrate incident Gaussian power in homogeneous Si with the identical common domain/source/time discretization. One reference may serve the pair only when all those inputs are identical. Compare incoming field amplitude, phase, waist and power on a common homogeneous-Si plane before the interaction, separating incoming from returned fields.
3. Save normalized signed native flux and complex E/H at s=28 and the narrow s=31 probe, plus the downstream isolated port. Save forward/backward coefficients only with explicit candidate identity; do not rename their sum guide power.
4. Record selected time traces near the source, interaction, upstream probes and remote port, and the convergence of complex DFT amplitudes. Extend the same simulation's integration if the remote response is still evolving. Equal `until_after_sources=60` is not evidence of equal convergence.
5. If early fields differ in regions causally unaffected by the downstream change, investigate source/material/registration or extraction. If agreement holds until a return arrives, quantify reflection/feedback with propagation-direction separation. If physical fields agree but reported metrics do not, focus on monitor sampling and postprocessing.

**Completion:** a reproducible table distinguishes changes in Pinc, upstream field, broad/narrow native flux, backward return and downstream identified-channel power. Differences have an identified numerical or physical dependency. No splice-loss conclusion is required to finish this diagnostic.

Current multiprobe output is stdout only; save raw arrays and effective run/monitor metadata with the next pair. The inspected `results/` records do not contain the new comparison. A narrative table alone cannot support an independent replay.

## 4. Mode identity: export fields now, starting at s=50

A 6 µm vertical half-span reaches about **3.569 µm along the guide normal**. A strip-centered guide with 3 µm oxide clearance has its substrate boundary approximately 3.131 µm below the center, so this aperture includes substrate even in the uniform section. Finite-aperture substrate modes can occupy many band indices; failure to find the strip fundamental in bands 1–8 does not demonstrate that it is absent. Nor does a real neff alone establish that every candidate is a substrate mode.

Use **s=50, vertical half-span approximately 2.52 µm** as the initial port-mode reference. Its guide-normal half-width is about 1.50 µm, and its longitudinal footprint lies beyond the transition. Verify the actual eigenmode-solving volume has the same properties. Expand to the prior |q|≤2 µm aperture check only while remaining uniform and substrate-free.

Start with a compact initialized eigensolver calculation; a full Gaussian-driven coupler run is unnecessary to export eigenmode fields. Compare with a truly isolated strip at matched angle, resolution, subcell placement and material sampling. Export:

- dielectric and complex E/H profiles in a common guide frame;
- strip/oxide/air/substrate participation in the fixed window;
- normalized full-field similarity to the isolated fundamental;
- frequency, band, signed propagation vector, convergence settings and aperture/domain bounds.

Use the prior isolated-port criteria (squared field similarity ≥0.99, reference-relative participation, aperture stability) after resolving discretization. The continuum neff≈2.2858 is a convergence reference, not an index filter for the coarse calculation. An appropriate wavevector seed can speed the eigensolve but cannot substitute for field identity.

If the narrow, uniform port still fails to reproduce the matched isolated mode, debug its sampled dielectric/eigensolver setup locally before another full-domain propagation. Once it passes, use that channel for R6. Loaded-wedge fields at s=31 can then be exported as a secondary diagnostic with continuous branch/subspace tracking; finding a unique real loaded MPB band is not a prerequisite for measuring coupling to the validated asymptotic port.

## 5. Allocation and next deliverables

Proceed without another policy stop through shared-boundary cleanup, initialization comparisons, compact mode export, and the **two matched coarse runs** above. Start with Hz, one frequency and localized DFT storage; defer Ez repetition and full-domain 50/75 until the failure is localized. The earlier memory/convergence policy remains in force.

Return: initialized-material difference maps, the isolated-port mode profile/identity table, and a raw-data-backed C0/C1 field/flux comparison with temporal convergence. These determine whether the next action is a construction fix, extraction fix or a physical feedback study. No broad geometry optimization is authorized.

T02 continues independently with page-5 provenance, descending-wavelength and endpoint fixes, then curve digitization. Those corrections do not depend on resolving the coupler.
