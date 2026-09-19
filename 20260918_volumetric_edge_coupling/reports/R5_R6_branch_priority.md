# R5/R6 branch priority and mode-identification gates

> **Subsequent review:** [Figure validation and R6 handoff](V1_V4_validation_and_R6_plan.md) records that the two implementation defects identified below are addressed in current source. Loaded-mode identity and numerical acceptance remain open. The current execution decision is port validation followed by R6, with Hz primary and one Ez control; completion of a full complex-beta R5 solver is not required first. The remainder of this document preserves the earlier review.

**Decision: investigate Hz first, with matched Ez controls; do not give both branches equal production-sweep effort.** This is a provisional allocation of work, not a determination that the paper's TE label means Hz or that Hz phase matching has been established. The corrected isolated fundamental Hz mode remains closer to the incident tangential index, which is sufficient to justify investigating it first.

The R1–R4 report and current source were reviewed. The factor-of-two correction and official oblique-guide benchmark are useful progress. Two concrete remaining implementation issues invalidate the reported higher-order slab modes and the present loaded-index comparison. Correct these before selecting a primary physical interpretation or counting modal power as receiving-guide power. No implementation code was changed by this review and no FDTD run was performed.

## 1. R1: fundamental indices agree, but the extra modes are still spurious

`src/modes.py` evaluates `state[1]/state[0] + q_b` for Ez, or the corresponding dielectric-weighted expression for Hz. Division by `state[0]` creates poles. `_scan_roots` accepts any sign change and does not require a small final boundary-condition residual, so it can converge to a pole rather than an eigenmode. A transfer-matrix formulation does not by itself eliminate this problem.

For the stated isolated air / Si / oxide slab, **there is one bound mode per polarization at 262 nm and 1550 nm**. An independent cutoff check uses full thickness:

- V = k0 d sqrt(nSi² − nOx²) = **3.36277**.
- First higher-order TE cutoff: π + atan[sqrt((nOx² − nair²)/(nSi² − nOx²))] = **3.45944**.
- First higher-order TM cutoff: π + atan[(nSi²/nair²) sqrt((nOx² − nair²)/(nSi² − nOx²))] = **4.46648**.

Both higher-order cutoffs exceed the available V. Direct calculator substitution of N = 2.6484 and 2.0206 into the current TE/TM transfer matrices respectively gives a nearly zero `state[0]` (about −1.2e−4 and −7.2e−5 for the rounded indices), while the undivided lower-cladding boundary residual is far from zero. These are poles, not TE1/TM1.

Required correction: use an undivided, appropriately normalized boundary determinant/residual or the branch-resolved arctangent relation. Accept a root only if its residual and both boundary conditions are satisfied; reconstruct its field and verify node count and decay. Preserve the valid TE0 ≈2.9598 and TM0 ≈2.2858 values. The proposed acceptance criterion is exactly one bound mode in each polarization for this reference and a dimensionless normalized boundary residual below 1e−8. Test at a thicker slab with known extra modes so mode-count validation is not tailored to this one case.

## 2. R3: reported wavevectors mix frequencies with bands

In `src/ports.py`, `oblique_modal_powers` reads `res.kpoints[bi]` and divides by `fcen`, while the monitor has 51 frequencies. Meep v1.34.0 stores the wavevector and group velocity at flattened index **band_index × number_of_frequencies + frequency_index**. Its Python return reshapes `alpha`, but leaves `kpoints` and `vgrp` flattened. See the version-pinned [C++ implementation](https://github.com/NanoComp/meep/blob/v1.34.0/src/mpb.cpp) and [Python wrapper](https://github.com/NanoComp/meep/blob/v1.34.0/python/simulation.py).

Thus entries 0 and 1 normally represent **one band's first two frequencies**, not bands 1 and 2 at 1550 nm. Moreover, dividing the first-frequency wavevector by the center frequency is inconsistent. For this frequency grid, merely replacing that denominator would turn a reported 2.061 into approximately **2.144 at the first frequency**; this is an illustration of the bug, not a corrected 1550 nm eigenvalue. Extract the actual center-frequency result.

Required correction: save a band-by-frequency index and group-velocity table using each sample's actual frequency, with vector components, signed projection along the declared propagation axis, and mode identity. Compare a single-frequency 1550 nm calculation with the central sample of the broadband calculation for each mode. Proposed agreement: |Δneff| <1e−4 for identical eigensolver geometry/settings, or a documented tighter-solver rerun explaining any failure. This is an indexing regression check, separate from spatial convergence.

The alpha array is already indexed differently and this finding does not automatically invalidate every saved modal-power coefficient. It does invalidate the association of those coefficients with the claimed phase-matched indices until the mapping is corrected.

## 3. A real MPB band near 2.07 is not sufficient mode identification

The 0.20% official-example result validates the oblique API for that isolated guide. It does not validate the identity of modes of the full substrate-loaded cross section.

The coupler mode monitor spans the receiving strip and a substantial silicon-substrate region. A real-eigenvalue supercell calculation can include discretized substrate/radiation modes. Its first six bands must not automatically be summed as receiving-guide power. MPB's documented eigenproblem uses lossless periodic structures; it is not an outgoing-wave complex-β resonance solver. [MPB introduction](https://mpb.readthedocs.io/en/latest/Introduction/), [Meep mode decomposition](https://meep.readthedocs.io/en/latest/Mode_Decomposition/).

Before assigning a loaded mode:

1. Export its actual dielectric cross section and complex E/H profiles, with guide, gap, substrate, computational boundaries and mode-solving volume marked.
2. Quantify guide-region versus substrate participation using a consistent finite observation window. Do not impose a universal bound-mode energy normalization on outgoing leaky modes.
3. Vary transverse supercell/aperture size independently of spatial resolution. Track fields by overlap rather than band number; movement/reordering of substrate bands must not be called guide dispersion.
4. Follow the receiving-guide branch toward a sufficiently large gap. It should approach the independently validated isolated fundamental of the same polarization. Record branch interactions rather than jumping to whichever real index lies closest to 2.07.
5. For R5's loaded resonance, obtain complex β with outgoing substrate conditions and demonstrate convergence against exterior truncation/PML choices. A periodic real mode near the target is only a candidate.

For a convention E ∝ exp(iβs − iωt), a passive forward leaky solution has β'' >0 and power decay length Lp = 1/(2β''). Record conventions explicitly. Report detuning, leakage rate, film localization and phase/amplitude of the emitted beam together; matching the real part alone is insufficient.

Keep the intended primary observable as **forward power in an identified receiving-guide output channel**. Until mode identities are validated, label the current sum as a sum over selected MPB bands. The difference between perpendicular-line flux and modal power can include substrate/radiation transport, aperture differences and interpolation. It cannot be assigned entirely to evanescent contamination from the two numbers alone; an evanescent field can carry tangential real power.

## 4. Bounded execution plan

| Stage | Hz effort | Ez control | Completion criterion |
|---|---|---|---|
| Index/mode audit | Correct band-frequency mapping; inspect center-frequency modes | Same checks | Correct roots and frequency association; profiles identify what is being projected |
| Initial R5 screen | At 1550 nm and 262 nm thickness, evaluate local gaps 64, 150, 244 and 500 nm, plus a sufficiently large-gap reference | Same sparse gap set and reference | Outgoing-wave branch tracked; complex β and guide participation converge; no forced neff target |
| R5 refinement | Refine Hz gap dependence only if the screen identifies a relevant receiving-guide resonance | No dense scan unless its screen changes the ranking | Detuning/leakage relative to the actual beam angular spectrum and interaction length reported |
| Initial R6 | One nominal reciprocal run using a validated receiving-guide port | One matched nominal reciprocal run | Emitted angular spectrum, complex Gaussian overlap, direct/reverse efficiency and energy balance |
| Further R6 work | Prioritize Hz if it remains better supported | Retain as a documented alternative | Reciprocity agrees within max(1e−4 absolute, 5% relative); no geometry retuning to force 88% |

All initial comparisons keep the same reported thickness/angle, assumed gap geometry, source convention, physical output reference and numerical accuracy. Local parallel-stack approximations must declare their tangent/normal frame. Compare β with the incident wavevector projected into that same frame: approximately 2.009 for the sidewall tangent versus 2.070 for the guide tangent. Do not mix those two values when interpreting a small residual mismatch.

R6 must launch the receiving-guide mode from a well-defined, effectively isolated output port. Do not reverse-launch a substrate-dominated mode just because its real index matches the desired value. Confirm that any uniform port continuation leaves the actual coupling region unchanged within the existing termination-sensitivity criterion. Overlay the reciprocal emitted phase, amplitude and propagation direction with the target Gaussian on a common reference plane. No new mode-profile images supporting the loaded-mode claim were supplied in the inspected report, so that visual gate remains open.

If the corrected screen removes the apparent Hz advantage, reassess priority before a dense scan. If Hz remains supported, call it **the primary working polarization hypothesis**, with Ez retained as the unresolved paper-label alternative. An author/source-field definition is still needed to establish what the paper meant by TE.

## 5. Subpixel smoothing is an unresolved convergence issue

A closer answer with smoothing disabled at one resolution is not evidence that disabling it is the correct production policy. Independently converge FDTD geometry resolution, mode-solver resolution, transverse extent and subcell translation with smoothing on and off. Export the sampled dielectric tensor/interface and physical thickness used by the mode solver. Both consistent discretizations should approach the same continuum reference; disagreement at a single mesh can reflect accidental staircase thickness, interpolation or material sampling.

Keep the reported ~0.02 index difference as an observed finite-resolution discrepancy, not a physical correction. Production ±7 nm tolerance claims remain gated on the original geometry/thickness convergence tests. The 3.1% custom/native flux residual likewise remains open against the 1% acceptance criterion; the cause and convergence rate require data.

The main 262 nm/53.5° reconstruction is unchanged. Further production sweeps remain conditional on validated port identity and convergence. These documentation changes do not mark the numerical corrections complete.
