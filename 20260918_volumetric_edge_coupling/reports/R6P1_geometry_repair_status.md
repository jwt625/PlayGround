# R6-P1 geometry repair status

> **Latest direction (2026-09-21, after fb00885):** Shared master tessellation is now present and the compact scalar mode-profile comparisons agree. [Proceed with C0/C1](R6P1_controlled_comparison_and_mode_export.md) after common-domain initialization checks; finish actual-port full-E/H identity alongside it. The compact res-60 reference does not yet establish the actual coupler's res-25 guide-channel identity. Historical net-flux and mode claims below remain provisional.

> **Review update (2026-09-19):** [Controlled comparison and mode export](R6P1_controlled_comparison_and_mode_export.md) supersedes the interpretation below. The 2.057/~104 readings are net signed flux, not forward guide power; the s=31/half-span-6 aperture spans s≈26.10–35.90 and samples changed downstream material. Downstream reflection can also change upstream fields. Export the narrow isolated-port modes at s=50 and compare both structures in a common cell/grid. Current oxide/substrate boundaries share an analytic function but still use different sampled vertex lists. The earlier large construction defects are addressed; complete sampled-material/port verification remains open.

G1-G4 repaired in `src/geometry.py` / `configs/isolated_port.yaml`.

| defect | repair |
|---|---|
| G1 oxide chord corrupted the interaction | oxide and substrate now share the same sampled boundary `B(s)`; no 4-vertex chord. D0 check: 0 real material mismatches through s=33 (only q=0 boundary-point ambiguities). |
| G2 downstream boundary parallel to sidewall | `q_sub(s) = g(s) - D/cos(beta)` downstream (same slope as the guide), C1 Hermite blend from q_sub=0 over the 8 um transition. Normal clearance is analytically constant D=3 um. |
| G3 guide ended before PML | materials now extend to s_absorb = s_port_end + guide_absorb into the PML; the domain is sized only to s_port_end. |
| G4 aperture straddled the transition | monitor aperture must use half-span ~2.52 um (|q|<=1.5 um) or be placed farther downstream. |

## Coarse multiprobe (D2), Hz, single 1550 nm

`scripts/port_multiprobe.py`, res 25, until 60, probes s=31/34/38/41/44/50,
half-span 4 um:

| s (um) | native | band1 neff | band1 fwd |
|---|---|---|---|
| 31 | 0.110 | 3.452 | 0.008 |
| 34 | 0.075 | 3.364 | 0.002 |
| 38 | 0.013 | 2.611 | 0.042 |
| 41 | 0.001 | 2.621 | 0.003 |
| 44 | 0.000 | 2.635 | 0.000 |
| 50 | 0.000 | 2.531 | 0.000 |

**The guide channel was not read.** With the 4 um aperture the solver's band
ordering puts a substrate mode at band 1 (neff 3.45) at s=31, so these are
not the receiving-guide coefficients. Band assignment moves with aperture and
domain, exactly the mode-identity gate. The native flux also mixed
frequencies (the same-aperture flux monitor stored 51 frequencies while the
mode monitor stored one), so `get_fluxes(...)[0]` was not 1550 nm.

## Next (no policy decision needed)

1. Fix the same-aperture flux to the same single frequency as the mode monitor.
2. Identify the guide band per probe by n_eff/localization over bands 1..8
   (and a matched isolated reference), not band number; then report the guide
   forward/backward and native flux.
3. Only then interpret any downstream power decrease as splice loss.
4. Defer 50/75 production; profile memory with the corrected single-frequency
   line monitors first.

## Second multiprobe (single-frequency flux fixed, bands 1-8, half-span 6)

| s | native | guide-like bands (neff>1.45) |
|---|---|---|
| 31 | 2.057 | no band near the guide (all 2.77-3.47 substrate) |
| 38 | 0.166 | b2 2.611 fwd 0.042 |
| 44 | 0.067 | b2 2.635 fwd 0.000 |
| 50 | 0.043 | b2 2.531 fwd 0.000 |

Two problems beyond band identity:

1. **The port variant's forward power at s=31 is ~2.06 versus ~104 for the
   nominal cell at the same settings** (res 25, until 60, 1550 nm, half-span 6,
   same interaction materials per D0). A ~50x drop cannot come from the
   downstream continuation and is not explained by the D0 sampling check.
   This must be resolved before any splice interpretation.
2. **No guide band appears at s=31** (all bands 1-8 are substrate, neff
   2.77-3.47), and downstream the closest band is neff ~2.53-2.64, not the
   isolated 2.286. Band identity/ordering remains unresolved even at the
   port.

The handoff's caution stands: the current numbers do not establish splice
loss. The immediate blocker is the unexplained forward-power change between
the nominal and port configurations plus the missing guide branch.
