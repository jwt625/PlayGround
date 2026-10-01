# Skill feedback: pilot on kohli2025 (plasmonic BTO-on-SiN, 3 devices)

Date: 2026-10-01. Time figures are rough wall-clock estimates (the session did not log per-step timers; the only measured point is that extraction + Crossref + validation tool calls took well under 1 min each).

## Time per step (approx)

| Step | Time | Note |
|---|---|---|
| Read SKILL, DevLog, schema, SPEC | 2 min | Clear overall |
| Crossref + license check | 1 min | Crossref returns no affiliations for this DOI; author affiliations come only from the PDF |
| extract_source.py | under 30 s | 11 pages, 31 PNGs, worked first time |
| Read text.md | 3 min | Text of an 11-page paper is cheap; figures were needed for pages 3-9 (7 PNG views) |
| Ledger + row-granularity decisions | 10-15 min | Dominant cost; see below |
| Writing staging files (script-generated) | 8 min | Generated CSV and evidence from one Python dict so values cannot diverge |
| Validation | under 30 s, passed first try | |

## Unclear in SKILL.md / schema / scripts

1. **No rule for choosing which Vpi convention fills the single `vpi_dc_v` / `vpi_rf_v` pair when the paper gives both per-phase-shifter and MZ push-pull numbers.** Rule 3 demands a convention but a row holds one. I chose per-phase-shifter (3.6 V DC, 6.4 V at 40 GHz) because the RF value is only given for the phase shifter, and left the headline 1.8 V push-pull in notes. This loses the paper's headline MZ figure from the numeric columns. Needed: either `vpi_dc_mzm_pushpull_v` (second column) or a rule "headline convention goes in columns, the other is converted only if the paper states it".
2. **Evidence validator rejects entries for non-evidence fields** (`bw3db_reference`, `drive`, `vpi_convention`, `il_onchip_excludes`, `driver`). So convention choices have no locator trail. Either mark the convention enums `evidence: true` or allow a `conventions:` block in the evidence yaml.
3. **Bounds ("less than 2 dB", "ER above 6 dB") have no representation.** The paper's headline for the racetrack is "2 dB device loss" but the text only gives an upper bound. I left `il_onchip_db` and `extinction_ratio_db` empty and put the bounds in notes. A `*_bound` qualifier column (`lt`, `gt`, `approx`) or a numeric-with-operator convention would avoid losing the number. Same for "approx" values (~200 nm, ~10 fJ/bit): no flag besides the evidence note.
4. **`bw3db_ghz` when the 3 dB point coincides with the measurement limit** (MZM: 110 GHz, measured to 110 GHz). Rule 3 says use `bw_measured_to_ghz` when no crossing was observed, but here the paper claims a crossing "around 110 GHz". I filled both and flagged in notes. Clarify whether a claimed crossing at the instrument limit should still be entered as `bw3db_ghz`.
5. **Reference frequency of 3 dB bandwidth**: enum has `dc, 1ghz, 10ghz`. Here the response also drops several dB between 10 MHz and 10 GHz (BTO Pockels dispersion), and the figure's 0 dB normalization is not stated. Consider adding a `low_freq_drop_db` field or an explicit note convention for plasmonic ferroelectrics, since a DC-referenced 3 dB bandwidth would be tens of GHz lower.
6. **`bw_basis` is per row**, but fields within a row can have different bases (RT: measurement limit read from a figure axis; MZM: stated in text). Same for `vpi_basis` (DC measured, RF extracted from frequency response). Per-field basis lives only in the evidence yaml; the CSV column is a lossy summary. Fine, but say which wins when they disagree.
7. **`discovered_via` has no value for "assigned by coordinator / local corpus"**. I used `web` which is not accurate. Add `assigned` or `local_corpus`.
8. **Org type for foundry-like companies** (Ligentec SA): `foundry` vs `company` both defensible; `companies` column does not care but UI filtering by type will. Add a rule. Also cleanroom facilities (Binnig and Rohrer Nanotechnology Center) fit none of the types (`other` used); country had to come from outside the paper because no address is given. Rule 6 says take country from the affiliation address, which does not exist for a facility named only in the acknowledgements.
9. **"Full unabbreviated" org names vs the institution's own short name**: "ETH Zurich" is the official English name but looks abbreviated. Added as in affiliation; state the policy.
10. **`wafer_supplier` / `foundry_or_fab` role split** (SiN foundry vs BTO grower vs device cleanroom) is not expressible in list columns without free text; I put roles in `wafer_supplier` text and kept the org list names clean. A `fab_roles` note convention would help.
11. **Evidence `derived:` list vs `entries:` with basis `derived`**: SKILL says derived items go to the `derived` list (formula) but the validator also accepts `basis: derived` in `entries`. I used `entries` for paper-computed values (RT Vpi from tuning efficiency, energy per bit from capacitance, where the paper gives the number and the method is only sketched) and the `derived` list for my own arithmetic (200 GBd x 1 bit/symbol = 200 Gb/s). State this split explicitly.
12. **Row-granularity rule 5** says one row per distinct device and operating point. I treated DC vs 40 GHz Vpi and multiple modulation formats as columns of one row (RF Vpi has its own column; `max_*` columns take the extreme with the format string listing the others). Rule is clear for devices but silent on whether several baud/format points deserve rows. Recommend stating: system-demo points stay in `modulation_format` text and notes.
13. **Evidence note limit (25 words) is tight** for conventions; I moved the longer remarks to `devices.csv notes`. Fine, but then notes carry unvalidated facts. Consider allowing a `remarks` field in evidence entries.
14. `extract_source.py`: page numbers in text.md match journal page numbers (PDF page = "Page n of 11") which is convenient; page-render PNGs exist only for pages with "Fig." captions (pages 1-2 not rendered, no figures there). Embedded raster extraction produced 23 small crops that are mostly unneeded for vector-plot papers. No bug found.
15. `merge_staging.py` dry run ran against the empty shared tables; with real data present, conflicts on `organizations.csv` for a differing `notes` string will be common (identical-row requirement). Suggest comparing only org_type/country/region and keeping the existing notes.
16. The `eo_film_thickness_nm` column for plasmonic devices: the paper gives the BTO thickness only for the directional-coupler waveguide (~200 nm), not the plasmonic section. Entered 200 with a note; a "where measured" convention would help.

## Missing columns / enums needed for plasmonic and resonant devices

- `device_class`: there is `plasmonic_mzm` but no `plasmonic_iq`; IQ was classed `iq_mzm` and plasmonic-ness is only visible via `electrode_type=plasmonic_lumped`. Racetrack classed `ring`. Consider a boolean or `plasmonic` tag independent of class.
- `waveguide_platform`: both `bto_on_sin` and `plasmonic_mim` exist and both apply; I used `bto_on_sin` (platform named by the paper). Define which wins.
- Resonator metrics: `q_loaded`, `fsr_nm`, `tuning_nm_per_v` (Q = 1931, FSR 1.79 nm, 0.3 nm/V are headline for the racetrack; kept in notes only). Also `slot_width_nm` (here mapped to `electrode_gap_um`, which is correct for MIM but not obvious).
- `capacitance_ff` (30 fF estimate behind the energy figure) and `drive_vpp_v` (1.13 Vpp for all demos; the single most important driverless-claim number).
- `vpi_dc_mzm_pushpull_v` or an equivalent (see item 1); `input_power_dbm` as an operating condition distinct from `optical_power_handling_dbm` (I left handling empty; the paper reports only the launch power used).
- `poling_required` / `bias_v` (BTO must be poled with 2-3 V bias; affects Vpi interpretation).
- Energy-per-bit basis: no `energy_basis` column, so the "paper estimate from capacitance" qualifier lives only in the evidence note.
- `plasmonic_section_length_um` vs total active footprint: the converter (15 um) and vertical directional coupler (80 um) add length not in `length_mm`; I entered only the plasmonic section length.

## Pitfalls encountered in this paper

- Fiber-to-fiber IL (20.3 / 23.9 / 9.4 dB) dwarfs the on-chip device loss; only the racetrack has an on-chip statement and that is a bound. Do not convert (paper gives component breakdowns that tempt subtraction).
- Abstract says "bandwidths up to 110 GHz" for the phase shifter; per-device values differ (MZM 110, IQ 70, RT no number). The abstract also quotes "2 dB" for the racetrack, which is an upper bound.
- Simulated ideal losses (8.1 dB MZM, approx 12 dB IQ) sit next to measured ones in the text; kept out of the columns.
- "Vpi of 6.4 V at 40 GHz" does not say phase shifter vs MZ explicitly; resolved from context only.
- Conference precursors (OFC 2024, CLEO 2024) exist with possibly different numbers; not consulted, flagged in notes.
- The racetrack Vpi (3 V) is a derived statement by the authors from wavelength tuning and does not follow the MZ definitions; `per_arm_phase_shifter` is the closest enum.

## v2 revision (2026-10-01)

Staging re-done against schema v2 (generated from one Python dict; dry-run merge: 0 conflicts, 0 validation errors; 71 evidence entries).

Applied:
- Headline push-pull MZ Vpi in `vpi_mzm_pushpull_dc_v` (MZM 1.8 V, IQ 2 V) with locators; `vpi_dc_v` stays per phase shifter (`per_arm_phase_shifter`). Racetrack `vpi_convention` is now `resonance_tuning_derived` (basis `derived`).
- Racetrack bounds as numbers plus qualifiers: `il_onchip_db`=2 (lt), `extinction_ratio_db`=6 (gt), `er_type`=static (read from the Fig. 4b transmission spectrum; my reading, noted in evidence).
- `drive_vpp_v`=1.13 and `optical_input_power_dbm` (20.3 / 21 / 13) flagged approx; `capacitance_ff`=30 and MZM energy 10 fJ/bit with basis `author_estimate`, flagged approx; `q_loaded`, `fsr_nm`, `tuning_nm_per_v` for the racetrack; `epitaxy_or_stack` text for the SiN/BTO stack; tags `plasmonic;poled` (+`racetrack`).
- Evidence entries added for convention fields (`vpi_convention`, `drive`, `bw3db_reference`, `electrode_type`, `waveguide_platform`, `er_type`). `discovered_via` = `local_corpus`. Binnig and Rohrer Nanotechnology Center typed `facility`, with a note that its host institution and country are not stated in the paper.
- Several notes shortened because the content now has columns.

Remaining gaps:
- `bias_for_vpi_v` left empty: the paper says only that the BTO was fully poled before the DC sweep and that a 2-3 V poling bias was used in data runs; no bias at which Vpi was measured is given.
- `il_onchip_includes` / `il_onchip_excludes` empty for the racetrack: the paper does not say what the "below 2 dB" covers (only "reference waveguides").
- `er_type` for the racetrack is my inference (static spectral dip), not stated by the paper.
- `eo_film_thickness_nm` 200 (approx) is stated for the coupler-section BTO waveguide only; applied to all three devices.
- The 6.4 V at 40 GHz is still phase-shifter-by-context; no MZ-level RF Vpi exists and no column pairs an RF value with `vpi_mzm_pushpull_dc_v`.
- Racetrack 3 dB bandwidth/RF response has no number; only the measurement span (70 GHz) is entered.
- `device_class` still has no plasmonic IQ / plasmonic ring value; the `plasmonic` tag covers it.
- Notes still carry unvalidated facts (Pockels coefficient ~180 pm/V at 40 GHz, linear-EQ-only baud limits, 400 m fiber results, simulated ideal losses): no columns for them.
