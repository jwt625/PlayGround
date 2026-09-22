# Figure and artifact index + multimodal validation requests

> **Latest review (2026-09-19):** [R6 port/channel/overlap decision](R6_port_and_overlap_decision.md) accepts the repaired isolated-profile rendering and acknowledges the new same-aperture comparison. Loaded-mode identity and reciprocal power extraction remain open; the reported M≈0.08 is not yet a Gaussian-channel power fraction. Reverse figures qualitatively show broad emission from the current launch. Build the named isolated output port, repair complex E/H overlap and input normalization, then rerun both directions. T02 proceeds independently. The [earlier visual review](V1_V4_validation_and_R6_plan.md) records impact gap 63.77 nm and other remaining geometry gates. Historical claims below do not establish phase matching or a coupling ceiling.

All figures are generated from the current config and are under `reports/figs/`.
Regenerate static figures with:

```
PYTHONPATH=. python scripts/make_figures.py --poynting --res 25 --until 60
```

| Artifact | What it shows | Source |
|---|---|---|
| `fig_geometry.png` | full device frame + wedge zoom; source, beam envelope, TIR impact, guide (262 nm), oxide gap, sidewall, cell/PML | `fig_geometry` |
| `fig_gap_profile.png` | g(s) in nm vs sidewall coordinate, beam amplitude footprint, interaction window, impact location | `fig_gap_profile` |
| `fig_mode_profiles.png` | isolated receiving-guide Ez/Hz mode profiles with Si core shading and n_eff | `fig_mode_profiles` |
| `fig_nominal_poynting_hz.png` | time-averaged Poynting vector at 1550 nm (Hz branch), quiver flow, guide/oxide outlines, net output flux | `fig_poynting` |
| `fig_r6_reverse_hz.png` | R6 reverse Hz launch: Sx map with the Si reference plane; shows the broad emitted fan | `scripts/r6_reciprocal.py` |
| `provenance.md` | config hash / branch / mesh / runtime / solver per figure generation | `write_provenance` |
| `reports/nominal_fields_cw.png` | (older) instantaneous \|Ez\| CW snapshot | `scripts/field_maps.py` |
| `reports/T03_phase_matching_note.md`, `reports/R1_R4_diagnostics.md`, `reports/R5_R6_branch_priority.md` | analysis notes | — |

Paper-figure reproduction (Fig. 3/4/5 overlays) is **not** done yet; it is
waiting on T02 digitization and an accepted nominal observable.

## What the Poynting map shows (honest reading)

- Incident beam in silicon, TIR at the sidewall, reflected beam down-left.
- A narrow guided jet on the receiving guide is present, but weak relative to
  radiation into the cladding; net output-line flux / Pinc is ~2.3 % at
  res 25, and summed band-1 forward modal power is ~0.5-0.8 %.
- The perpendicular output line crosses both guide and cladding, so its flux
  includes radiation; the modal power is the primary observable.

## New evidence: substrate-loaded phase match (Hz)

Loaded band-1 n_eff vs monitor position / local gap (res 25):

| s (um) | gap (nm) | Hz n_eff | Ez n_eff |
|---|---|---|---|
| 11.0 | 63 | **2.049** | 2.788 |
| 13.0 | 107 | 2.125 | 2.806 |
| 16.0 | 172 | 2.195 | 2.830 |
| 20.0 | 258 | 2.211 | 2.939 |
| 26.0 | 388 | 2.316 | 2.937 |
| 31.0 | 496 | 2.298 | 2.963 |

Required guide-projected index is 2.070 (sidewall tangent 2.009). Hz is
phase-matched at the coupling gap and detunes as the gap opens; Ez is never
matched. Bounded calibration of Hz (sstart 6/9/12, g_ref 5/20/40/70 nm, no
change to thickness or angles) gives band-1 forward guided coupling of
0.2-0.8 %; the best case is ~0.8 %. So phase matching is not the limiter for
Hz; coupling strength/overlap in the declared reconstruction is.

## Requests to the multimodal validation agent

1. **Geometry gate (after T04).** On `fig_geometry.png` and
   `fig_gap_profile.png`: confirm the Fig. 2 topology (beam inside silicon,
   guide on the sidewall cavity side, oxide wedge opening in the correct
   direction), source axis at y = 11 um above the cavity base, guide measured
   262 nm normal to the guide, no guide/substrate overlap, no clipped source,
   no PML overlap. Report the numeric gap at the impact point.
2. **Field gate (after T07).** On `fig_nominal_poynting.png` and the complex
   field maps: confirm the reflected TIR direction obeys the vector
   reflection law (not the schematic vertical arrow), that the guided jet is
   on the intended guide, and that apparent guide power is not direct beam
   flux. Flag any field terminating on mesh/PML artifacts.
3. **Mode-identity gate.** Review exported MPB loaded-mode profiles (to be
   produced with dielectric cross-section, guide/gap/substrate/boundaries
   marked, guide-region vs substrate participation). Confirm whether the
   band-1 Hz mode is the receiving-guide mode or a substrate-continuum mode
   before any loaded index is accepted.
4. **Self-consistency gate.** Check that the perpendicular-line flux and the
   summed modal power are compared on the same aperture and that neither is
   labeled guided power while mode identity is open.
5. **Paper comparison gate (later).** Only after a nominal is accepted,
   inspect digitized Fig. 3/4/5 overlays for amplitude rescaling or
   wavelength shifting; the reflection ordinate must not be forced to
   `1 - eta`.
