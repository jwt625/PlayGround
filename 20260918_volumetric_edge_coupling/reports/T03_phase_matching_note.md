# T03 model-decision note: phase matching of the receiving guide

> Orchestration review: see [T03_orchestration_decision.md](T03_orchestration_decision.md). The original observations below are retained as history. The isolated TE index and phase-matched thickness estimates need correction, custom/native flux normalization differs, and output termination/modal separation must be validated. The 0.86% result is provisional net flux, not a certified guided-mode efficiency or established phase-mismatch limit.

Status: **conditional / unresolved interpretation**. This note records a
first-principles diagnostic required by PAPER_REVIEW.md s6 and
SIMULATION_SPEC.md s4 before any spectrum is scored.

## What was computed

Constant indices `n_Si=3.48`, `n_Ox=1.444`, `n_air=1`; `lambda=1550 nm`.

A horizontal beam in silicon meets the 54.74 deg KOH sidewall. The incidence
angle from the sidewall normal is `90 - 54.74 = 35.26 deg`, so the tangential
(phase-matching) effective index is

```
n_t = n_Si * sin(35.26 deg) = n_Si * cos(54.74 deg) = 2.009
```

Projecting onto the 53.50 deg guide gives `n_Si * cos(53.50 deg) = 2.070`.
So an efficient, phase-matched transfer into the inclined guide requires a
guide mode with `n_eff ~= 2.0-2.1`.

## Actual receiving-guide modes (three-layer slab)

Solved with the standard TE/TM slab dispersion relations for
air / Si(262 nm) / oxide:

| Polarization branch | n_eff of fundamental |
|---|---|
| TE (Ez out of plane, Meep Ez) | **2.615** |
| TM (Hz out of plane, Meep Hz) | **2.285** |

The required `2.0-2.1` sits well below both. The mismatch is
`dk/k0 ~ 0.5` (TE) or `~0.2` (TM), i.e. `dk ~ 2.1` / `0.9 rad/um`. Over a
~16 um interaction the coupling beats many times and largely averages out.

To phase match at 35.26 deg incidence the Si core would have to be ~80-85 nm,
not 262 nm. Alternatively the beam would need ~13 deg (TE) or ~6 deg (TM)
extra tilt, far outside the reported +-0.8 deg tolerance.

## Simulation observation (Meep, 2D, paper frame)

`configs/nominal.yaml`, constant indices, resolution 50/um reported;
search runs used 20-25/um, until 40-60 after a Gaussian pulse.

- Fields show the expected topology: beam in silicon, TIR at the sidewall,
  evanescent region, and a guided wave leaving along the inclined Si strip
  (see `reports/nominal_fields_cw.png`).
- Measured coupling `eta(1550) = 0.86 %` (-20.7 dB); broad-band peak 1.7 %.
- Reducing the gap at the beam from ~244 nm to ~64 nm changed eta only from
  ~1.1 % to ~0.86 %; varying depth, stack start and wedge sign did not
  produce more than a few percent. This is consistent with a phase mismatch
  rather than a gap/overlap-limited result.

## Interpretation options (for planning/orchestration)

1. The paper's internal-coupling model relies on a **leaky / substrate-loaded
   quasi-mode** of the full air/Si/oxide/Si stack whose real `n_eff` is near
   2.0, rather than the isolated 262 nm strip mode. That would make the
   literal text self-consistent, but such a mode radiates into the Si
   substrate, which is hard to reconcile with 88 % over the interaction.
2. The 53.50 deg "waveguide angle" is referenced to a different axis, or the
   incident beam direction is not parallel to the wafer surface, so the
   quoted phase matching is not the one used. The reported 0-deg tilt
   optimum argues against a large beam tilt.
3. The paper is internally inconsistent / the reported 88 % is not
   reproducible from the stated geometry and parameters. PAPER_REVIEW.md s8
   explicitly permits a faithful, documented disagreement.

No parameter (gap, depth, wedge sign, stack window) has been, or should be,
silently retuned to move the peak to 1550 nm or the efficiency to 88 %.

## Requested decision

Which interpretation should the reproduction adopt as primary?
(a) documented mismatch with the literal geometry (current default),
(b) a named leaky-mode / substrate-loaded hypothesis,
(c) a named alternative-angle hypothesis, or
(d) request the Lumerical files from the authors before further tuning.
