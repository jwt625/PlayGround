# R1–R4 diagnostics (following the T03 orchestration decision)

> Orchestration review: [R5/R6 branch priority and mode-identification gates](R5_R6_branch_priority.md). Hz is prioritized provisionally, with matched Ez controls. The TE1/TM1 entries below are poles of the current residual, and the loaded-index reporting mixes band and frequency indices. Preserve the observations as history; phase matching and receiving-guide mode identity are not yet established by these tables.

All results at 1550 nm, constant indices n_Si=3.48, n_Ox=1.444, n_air=1.

## R1 — slab modes reconciled

Robust transfer-matrix solver (`src/modes.py`, decaying-cladding conditions)
and Meep/MPB through an axis-aligned mode monitor on an isolated straight
guide.  The earlier `TE = 2.615` was a spurious higher-branch root of the
`tan` form.

| Mode (air / Si 262 nm / oxide) | Analytic TMM | MPB, no subpixel, res 200 | diff |
|---|---|---|---|
| TE0 / Ez band 1 | 2.9598 | 2.9624 | +0.0027 |
| TM0 / Hz band 1 | 2.2858 | 2.2900 | +0.0042 |
| TE1 / Ez band 2 | 2.6484 | — | — |
| TM1 / Hz band 2 | 2.0206 | — | — |

With Meep subpixel smoothing enabled at res 200 the Ez index shifts to 2.9439
(diff 0.016 from analytic).  Conclusion: analytic and MPB agree to the R1
tolerance once smoothing is removed; the smoothing shift of ~0.02 n_eff is a
numerical model consideration for the 262 nm high-contrast layer and must be
watched in the 7 nm thickness-tolerance sweeps.

## R2 — flux convention

`src/ports.py` multiplied the sampled Poynting product by 0.5 while Meep's
`dft_flux` does not; the 0.5 has been removed.  Custom/native agreement in
homogeneous silicon on the same vertical plane, res 40/um: ratio 0.969
(3.1 %), residual attributed to collocation/interpolation (shrinks with
resolution).  The factor-of-two is resolved and documented; the ~3 % residual
must be reduced at production resolution before a 1 % claim.

## R3 — oblique modal decomposition

The Meep v1.34.0 oblique route (`direction=mp.NO_DIRECTION`, `kpoint_func`)
is validated against the official example: net flux vs `|alpha|^2` ratio
1.0020 (0.20 %) for an isolated 1 um guide at 20 deg.

Applied to the receiving guide (res 20, until 40):

| branch | loaded mode n_eff (bands 1/2) | required (beam projection) |
|---|---|---|
| Ez (E out of plane) | 2.757 / 2.763 | ~2.07 |
| **Hz (E in plane)** | **2.061 / 2.068** | **~2.07** |

The Hz (TM) loaded modes are essentially phase-matched to the horizontal
beam; the Ez (TE) loaded modes are not.  This is a physics-based resolution of
the polarization ambiguity flagged in SIMULATION_SPEC s3: the paper's "TE"
label most plausibly maps to the Meep Hz branch.  Per the spec we do not
select a branch merely because it transmits more, but the phase-matching
diagnostic does select Hz.

## R4 — nominal reconstruction, absorbing termination

The guide/oxide now continue into the PML (`guide_absorb_um`), removing the
hard termination.  Coarse result (res 20, until 40, no gap optimisation):

| branch | eta_net (custom line) | sum forward modal | sum backward modal |
|---|---|---|---|
| Ez | 1.93 % | 1.51 % | 0.55 % |
| Hz | 1.57 % | 1.03 % | 0.07 % |

Caution: the custom perpendicular-line flux is contaminated near the
substrate interface.  A diagnostic scan of interaction length for Hz gave
net 7.4 % at L = 8 um but only 1.4 % forward modal power; the "gain" is the
residual evanescent/TIR field being sampled when the monitor sits close to
the interaction.  The custom line is therefore only a cross-check; the
modal/summed guided power is the primary observable.

## Interim conclusion

Phase matching is not the limiting factor for the Hz branch (loaded n_eff
2.06 vs required 2.07), yet guided coupling is ~1 %.  The open question is now
whether the loaded, wedged, finite geometry (gap, apodization, beam/guide
overlap, loaded leakage) can produce strong coupling at the declared 262 nm
and 53.5 deg, which is the R5/R6 diagnostic.  No thickness/angle has been
changed to recover 88 %.

## Corrections after orchestration review (post-R4)

Two bugs were fixed and one claim retracted.

1. **Spurious slab roots removed.** `src/modes.py` now uses the undivided
   lower-cladding residual (no division by `state[0]`, so no pole roots) and
   accepts a root only if its normalised boundary residual is <1e-8.  At
   262 nm the isolated slab now returns exactly **one** bound mode per
   polarization: Ez 2.9598, Hz 2.2858.  A thicker 800 nm test returns the
   expected multiple modes (Ez 3.384/3.082/2.525/1.598, Hz
   3.349/2.928/2.101), so mode counting is not tailored to one case.
   The earlier TE1/TM1 values (2.6484, 2.0206) were poles and are rejected.

2. **Kpoint band/frequency indexing fixed.** `src/ports.py` now reads
   `res.kpoints[band*nfreq + freq]` (Meep stores a flattened table) and uses
   each sample's own frequency for `n_eff = |k|/f`.  With the old bug the
   first two entries were adjacent frequencies of band 1, so the reported
   "Hz 2.061" was an artifact.  Corrected loaded band-1 indices at the
   1550 nm centre sample are **Ez 2.897, Hz 2.229** (isolated 2.960/2.286).
   Bands 2+ fall below the oxide index (radiation/substrate), so only band 1
   is a guide candidate.

3. **"Hz is phase-matched" is retracted.** On the corrected indices the
   required tangential index is 2.07 (guide projection) / 2.009 (sidewall
   tangent).  Hz is closer (mismatch ~0.16) than Ez (~0.83), but neither is
   matched.  Hz remains the priority working hypothesis for investigation,
   not an established paper-TE mapping.

## R5 status (parallel-layer loading screen)

`scripts/r5_loading.py` solves air / Si(262 nm) / oxide(g) / semi-infinite
Si by MPB for g = 64, 150, 244, 500, 2000 nm.  At large gap (2000 nm) the
guide-associated real band is near the isolated value (Hz 2.262, Ez 2.917).
As g shrinks toward the coupling range the nearest real band jumps upward
(Hz ~3.09-3.19, Ez ~3.12-3.21 at 64-500 nm): the guide mode is merging into
the discretized substrate continuum rather than forming a clean bound
branch.  This is precisely the mode-identity gate flagged in the review.
The real-eigenvalue MPB solver is not sufficient here; R5 needs complex-beta
outgoing-wave conditions (or an equivalent scattering/transfer-matrix
resonance solve) before any loaded index is assigned.  No forced 2.07 target.
